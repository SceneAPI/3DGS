"""Multi-provider radiance trainer engine.

One engine drives the four native splat trainers that previously shipped as
four near-identical repos (sfmapi_brush, sfmapi_lfs, sfmapi_spirulae,
sfmapi_fastergs); each of those repos already carried all four ``_train_*``
implementations and differed only in four module-level constants, which are
folded into the :data:`PROVIDERS` config table here. gsplat's in-process
CUDA/torch trainer is genuinely different and lives in
:mod:`sceneapi_3dgs.gsplat_trainer`.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExecuteRequest:
    """One ``/execute`` task as the ``sceneapi.plugin_service`` kit hands it to
    the executor (protocol envelope already validated and stripped)."""

    task_kind: str
    capability: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    spec: dict[str, Any] = field(default_factory=dict)
    tenant_id: str = ""
    job_id: str = ""
    task_id: str = ""
    provider: str = ""


@dataclass(frozen=True)
class ProviderConfig:
    """Per-provider engine constants.

    These four values were the only module-level constants that differed
    across the four repos this engine unifies.
    """

    provider: str
    native_root_env: str
    default_native_root: Path
    cuda_required: bool


PROVIDERS: dict[str, ProviderConfig] = {
    "brush": ProviderConfig(
        provider="brush",
        native_root_env="SFMAPI_BRUSH_ROOT",
        default_native_root=Path("/opt/brush"),
        cuda_required=False,  # brush renders via wgpu/Vulkan, not CUDA
    ),
    "lfs": ProviderConfig(
        provider="lfs",
        native_root_env="SFMAPI_LFS_ROOT",
        default_native_root=Path("/opt/LichtFeld-Studio"),
        cuda_required=True,  # LichtFeld-Studio native CUDA splat trainer
    ),
    "spirulae": ProviderConfig(
        provider="spirulae",
        native_root_env="SFMAPI_SPIRULAE_ROOT",
        default_native_root=Path("/opt/spirulae-splat"),
        cuda_required=True,  # torch + CUDA splat trainer
    ),
    "fastergs": ProviderConfig(
        provider="fastergs",
        native_root_env="SFMAPI_FASTERGS_ROOT",
        default_native_root=Path("/opt/fastergs"),
        cuda_required=True,  # torch + CUDA splat trainer
    ),
}


def _config(provider: str) -> ProviderConfig:
    config = PROVIDERS.get(provider)
    if config is None:
        raise ValueError(f"request.provider must be one of {', '.join(sorted(PROVIDERS))}")
    return config


def runtime_info(provider: str) -> dict[str, Any]:
    config = _config(provider)
    info: dict[str, Any] = {
        "provider": provider,
        "native_root": str(_native_root(provider)),
        "gpu_required": True,
        "cuda_required": config.cuda_required,
    }
    info.update(_gpu_runtime_info(provider))
    if provider == "brush":
        exe = _brush_executable()
        info["executable"] = str(exe)
        info["executable_exists"] = exe.is_file()
    elif provider == "lfs":
        exe = _lfs_executable()
        info["executable"] = str(exe)
        info["executable_exists"] = exe.is_file()
    else:
        try:
            import torch

            info["torch"] = torch.__version__
            info["torch_cuda"] = getattr(torch.version, "cuda", None)
            info["cuda_available"] = bool(torch.cuda.is_available())
            info["cuda_device"] = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            )
        except Exception as exc:
            info["torch_error"] = f"{type(exc).__name__}: {exc}"
            info["cuda_available"] = False
    return info


# Canonical radiance.train config-schema option names -> each engine's native
# backend_options keys. Back-fill only (an explicit native key always wins), so
# one canonical spec (num_gaussians, max_resolution, test_every) drives every
# splatting engine identically. See sceneapi radiance.train config-schema.
_CANONICAL_OPTION_ALIASES: dict[str, dict[str, str]] = {
    "brush": {"num_gaussians": "max_splats", "test_every": "eval_split_every"},
    "lfs": {"num_gaussians": "max_cap", "max_resolution": "max_width"},
    "spirulae": {"num_gaussians": "model.cap_max"},
    "fastergs": {"num_gaussians": "max_primitives"},
}


def _apply_canonical_options(provider: str, options: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(options)
    for canonical, native in _CANONICAL_OPTION_ALIASES.get(provider, {}).items():
        if canonical in resolved and native not in resolved:
            resolved[native] = resolved[canonical]
    return resolved


def train(request: ExecuteRequest) -> dict[str, Any]:
    provider = _config(request.provider).provider
    _require_gpu_runtime(provider)
    spec = request.spec
    inputs = request.inputs
    options = spec.get("backend_options") if isinstance(spec.get("backend_options"), dict) else {}
    options = _apply_canonical_options(provider, options)
    project_id = _required_str(inputs, "project_id")
    radiance_field_id = _required_str(inputs, "radiance_field_id")
    max_steps = _max_steps(spec, options)
    seq = int(options.get("snapshot_seq") or 1)
    snapshot_path = _snapshot_path(options, project_id, radiance_field_id, seq)
    work_dir = _work_path(options, project_id, radiance_field_id, request.job_id or request.task_id)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    eval_config = _eval_config(spec)
    started = time.perf_counter()
    result = _train_provider(provider, options, work_dir, max_steps, snapshot_path, eval_config)
    duration_s = round(time.perf_counter() - started, 3)
    source_ply = Path(result["ply"])
    if not source_ply.is_file():
        raise FileNotFoundError(f"{provider} did not produce a PLY artifact: {source_ply}")
    target_ply = snapshot_path / "point_cloud.ply"
    shutil.copy2(source_ply, target_ply)

    vertex_count = _ply_vertex_count(target_ply)
    summary = {
        "provider": provider,
        "method": str(spec.get("method") or f"{provider}.train.default"),
        "radiance_field_id": radiance_field_id,
        "dataset_id": inputs.get("dataset_id"),
        "max_steps": max_steps,
        "completed_steps": result.get("completed_steps", max_steps),
        "duration_s": duration_s,
        "vertex_count": vertex_count,
        "format": "ply",
        "native_artifact": str(source_ply),
        "work_dir": str(work_dir),
    }
    _write_json(snapshot_path / "summary.json", summary)
    _write_json(
        snapshot_path / "metadata.json",
        {"runtime": runtime_info(provider), "process": result.get("process"), **summary},
    )
    evaluation_id = inputs.get("evaluation_id")
    requested_metrics = _requested_metrics(eval_config)
    metrics = _normalize_metrics(
        result.get("metrics"),
        duration_s=duration_s,
        required_metrics=requested_metrics if eval_config.get("enabled") is True else None,
    )
    evaluations: list[dict[str, Any]] = []
    if eval_config.get("enabled") is True:
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise ValueError("inputs.evaluation_id is required when spec.eval.enabled=true")
        if metrics is None:
            raise RuntimeError(
                f"{provider} did not emit requested eval metrics: "
                f"{', '.join(sorted(requested_metrics))}"
            )
        _write_json(snapshot_path / "metrics.json", metrics)
        eval_artifacts = [
            {
                "kind": "radiance.evaluation.metrics",
                "name": "metrics.json",
                "uri": str(snapshot_path / "metrics.json"),
                "media_type": "application/json",
                "artifact_format": "sfmapi.radiance.metrics.v1",
                "summary": metrics,
            }
        ]
        evaluations.append(
            {
                "evaluation_id": evaluation_id,
                "radiance_field_id": radiance_field_id,
                "snapshot_seq": seq,
                "metrics": metrics,
                "artifacts": eval_artifacts,
            }
        )
    elif metrics is not None:
        _write_json(snapshot_path / "metrics.json", metrics)

    ply_uri = str(target_ply)
    return {
        "radiance_field_id": radiance_field_id,
        "snapshot_seq": seq,
        "snapshot_path": str(snapshot_path),
        "summary": summary,
        "evaluations": evaluations,
        "artifacts": [
            {
                "kind": "radiance.snapshot",
                "name": f"snapshot-{seq}",
                "uri": str(snapshot_path),
                "artifact_format": "sfmapi.radiance.snapshot.v1",
                "metadata": {"radiance_field_id": radiance_field_id, "snapshot_seq": seq},
                "summary": summary,
            },
            {
                "kind": "radiance.variant.ply",
                "name": "point_cloud.ply",
                "uri": ply_uri,
                "media_type": "application/octet-stream",
                "artifact_format": "sfmapi.radiance.variant.ply.v1",
                "metadata": {"radiance_field_id": radiance_field_id, "snapshot_seq": seq},
                "summary": {"vertex_count": vertex_count},
            },
        ],
        "variants": [
            {
                "format": "ply",
                "uri": ply_uri,
                "media_type": "application/octet-stream",
                "summary": {"vertex_count": vertex_count},
            }
        ],
    }


def evaluate(request: ExecuteRequest) -> dict[str, Any]:
    provider = _config(request.provider).provider
    _require_gpu_runtime(provider)
    spec = request.spec
    inputs = request.inputs
    options = spec.get("backend_options") if isinstance(spec.get("backend_options"), dict) else {}
    radiance_field_id = _required_str(inputs, "radiance_field_id")
    evaluation_id = _required_str(inputs, "evaluation_id")
    project_id = _required_str(inputs, "project_id")
    seq = int(
        inputs.get("snapshot_seq") or spec.get("snapshot_seq") or options.get("snapshot_seq") or 1
    )
    snapshot_path = _snapshot_path(options, project_id, radiance_field_id, seq)
    started = time.perf_counter()
    metrics_path = Path(str(options.get("metrics_path") or snapshot_path / "metrics.json"))
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"{provider} standalone evaluation requires an existing metrics file: {metrics_path}"
        )
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = _normalize_metrics(
        raw,
        duration_s=round(time.perf_counter() - started, 3),
        required_metrics=_requested_metrics(_eval_config(spec)),
    )
    if metrics is None:
        raise RuntimeError(f"{provider} metrics file does not contain requested eval metrics")
    artifacts = [
        {
            "kind": "radiance.evaluation.metrics",
            "name": "metrics.json",
            "uri": str(metrics_path),
            "media_type": "application/json",
            "artifact_format": "sfmapi.radiance.metrics.v1",
            "summary": metrics,
        }
    ]
    return {
        "radiance_field_id": radiance_field_id,
        "evaluation_id": evaluation_id,
        "snapshot_seq": seq,
        "metrics": metrics,
        "artifacts": artifacts,
    }


def _train_provider(
    provider: str,
    options: dict[str, Any],
    work_dir: Path,
    max_steps: int,
    snapshot_path: Path,
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    if provider == "brush":
        return _train_brush(options, work_dir, max_steps, eval_config)
    if provider == "lfs":
        return _train_lfs(options, work_dir, max_steps, eval_config)
    if provider == "spirulae":
        return _train_spirulae(options, work_dir, max_steps, eval_config)
    if provider == "fastergs":
        return _train_fastergs(options, work_dir, max_steps, snapshot_path, eval_config)
    raise RuntimeError(f"unsupported provider {provider}")


def _train_brush(
    options: dict[str, Any], work_dir: Path, max_steps: int, eval_config: dict[str, Any]
) -> dict[str, Any]:
    source = _dataset_path(options)
    exe = _brush_executable()
    if not exe.is_file():
        raise FileNotFoundError(f"Brush executable is missing: {exe}")
    export_name = str(options.get("export_name") or "export_{iter}.ply")
    argv = [
        str(exe),
        str(source),
        "--total-train-iters",
        str(max_steps),
        "--max-resolution",
        str(int(options.get("max_resolution") or 384)),
        "--eval-every",
        str(int(options.get("eval_every") or eval_config.get("every_steps") or max(max_steps, 1))),
        "--export-every",
        str(int(options.get("export_every") or max(max_steps, 1))),
        "--export-path",
        str(work_dir),
        "--export-name",
        export_name,
    ]
    if "max_splats" in options:
        argv.extend(["--max-splats", str(int(options["max_splats"]))])
    eval_split_every: int | None = None
    if eval_config.get("enabled") is True or options.get("eval_split_every"):
        eval_split_every = int(options.get("eval_split_every") or options.get("test_every") or 8)
        argv.extend(
            [
                "--eval-split-every",
                str(eval_split_every),
            ]
        )
    if eval_config.get("save_images") is True or options.get("eval_save_to_disk"):
        argv.append("--eval-save-to-disk")
    if options.get("with_viewer"):
        argv.append("--with-viewer")
    process = _run(
        "brush",
        argv,
        cwd=_native_root("brush"),
        env={"RUST_LOG": str(options.get("rust_log") or "info")},
        log_dir=work_dir / "logs",
    )
    ply = _latest_existing(work_dir, ["*.ply", "**/*.ply"])
    metrics = _parse_brush_metrics(process, duration_s=None)
    if metrics is not None and int(metrics.get("num_images") or 0) == 0 and eval_split_every:
        metrics["num_images"] = _count_eval_split_images(source, eval_split_every)
    return {"ply": str(ply), "completed_steps": max_steps, "process": process, "metrics": metrics}


def _train_lfs(
    options: dict[str, Any], work_dir: Path, max_steps: int, eval_config: dict[str, Any]
) -> dict[str, Any]:
    source = _dataset_path(options)
    exe = _lfs_executable()
    if not exe.is_file():
        raise FileNotFoundError(f"LichtFeld Studio executable is missing: {exe}")
    argv = [
        str(exe),
        "--headless",
        "--train",
        "--no-splash",
        "--data-path",
        str(source),
        "--output-path",
        str(work_dir),
        "--output-name",
        "point_cloud",
        "--max-width",
        str(int(options.get("max_width") or 384)),
        "--log-level",
        str(options.get("log_level") or "info"),
    ]
    if eval_config.get("enabled") is True:
        argv.extend(
            ["--steps-scaler", str(float(options.get("steps_scaler") or max_steps / 30000.0))]
        )
    else:
        argv.extend(["--iter", str(max_steps)])
        if "steps_scaler" in options:
            argv.extend(["--steps-scaler", str(float(options["steps_scaler"]))])
    if eval_config.get("enabled") is True or options.get("eval"):
        argv.append("--eval")
        test_every = int(options.get("test_every") or 8)
        argv.extend(["--test-every", str(test_every)])
        if eval_config.get("save_images") is True or options.get("save_eval_images"):
            argv.append("--save-eval-images")
    if options.get("images"):
        argv.extend(["--images", str(options["images"])])
    if options.get("strategy"):
        argv.extend(["--strategy", str(options["strategy"])])
    if options.get("gut"):
        argv.append("--gut")
    if options.get("max_cap"):
        argv.extend(["--max-cap", str(int(options["max_cap"]))])
    process = _run("lfs", argv, cwd=_native_root("lfs"), log_dir=work_dir / "logs")
    ply = _latest_existing(work_dir, ["*.ply", "**/*.ply"])
    metrics = _parse_lfs_metrics(work_dir)
    return {"ply": str(ply), "completed_steps": max_steps, "process": process, "metrics": metrics}


def _train_spirulae(
    options: dict[str, Any], work_dir: Path, max_steps: int, eval_config: dict[str, Any]
) -> dict[str, Any]:
    _require_torch_cuda("spirulae")
    source = _dataset_path(options)
    root = _native_root("spirulae")
    if not (root / "spirulae_splat" / "ss_trainer.py").is_file():
        raise FileNotFoundError(f"spirulae-splat checkout is missing trainer under: {root}")
    preset = str(options.get("preset") or "3dgs")
    exporter = str(options.get("exporter") or "3dgs")
    train_args = [
        preset,
        "--data",
        str(source),
        "--output_dir_prefix",
        str(work_dir.parent),
        "--output_dir_name",
        work_dir.name,
        "--steps_per_save",
        str(int(options.get("steps_per_save") or max_steps)),
        "--num_iterations",
        str(max_steps),
        "--viewer_port",
        str(int(options.get("viewer_port") or 7007)),
    ]
    for key in (
        "dataparser.data_format",
        "dataparser.colmap_recon_dir",
        "dataparser.image_dir",
        "dataparser.eval_mode",
        "dataparser.eval_interval",
        "model.primitive",
        "model.cap_max",
        "model.sh_degree",
        "model.no_randomize_background",
    ):
        if key in options and options[key] is not None:
            train_args.extend([f"--{key}", str(options[key])])
    if eval_config.get("enabled") is True:
        if "dataparser.eval_mode" not in options:
            train_args.extend(["--dataparser.eval_mode", "interval"])
        if "dataparser.eval_interval" not in options:
            train_args.extend(
                ["--dataparser.eval_interval", str(int(options.get("test_every") or 8))]
            )
    if not bool(options.get("save_only_latest_checkpoint", True)):
        train_args.extend(["--save_only_latest_checkpoint", "False"])
    argv = [
        sys.executable,
        "-c",
        _spirulae_train_and_export_script(exporter),
        str(root),
        str(work_dir),
        *train_args,
    ]
    process = _run(
        "spirulae",
        argv,
        cwd=work_dir,
        env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        log_dir=work_dir / "logs",
    )
    ply = work_dir / "splat.ply"
    if not ply.is_file():
        ply = _latest_existing(work_dir, ["splat.ply", "*.ply", "**/*.ply"])
    metrics = _parse_json_metrics(work_dir / "metrics.json")
    return {"ply": str(ply), "completed_steps": max_steps, "process": process, "metrics": metrics}


def _train_fastergs(
    options: dict[str, Any],
    work_dir: Path,
    max_steps: int,
    snapshot_path: Path,
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    _require_torch_cuda("fastergs")
    source = _dataset_path(options)
    provider_root = _native_root("fastergs")
    framework = Path(os.environ.get("SFMAPI_FASTERGS_FRAMEWORK_ROOT", "/opt/nerficg"))
    method_name = str(options.get("method_name") or "FasterGS")
    if not (framework / "scripts" / "train.py").is_file():
        raise FileNotFoundError(f"NeRFICG train.py is missing under: {framework}")
    if not (framework / "scripts" / "convert_to_ply.py").is_file():
        raise FileNotFoundError(f"NeRFICG convert_to_ply.py is missing under: {framework}")
    if not (framework / "src" / "Methods" / method_name).exists():
        raise FileNotFoundError(
            f"Faster-GS method is not installed: {framework / 'src' / 'Methods' / method_name}"
        )
    model_name = str(options.get("model_name") or f"{snapshot_path.parent.parent.name}_fastergs")
    template = Path(str(options.get("config_template") or provider_root / "fastergs_garden.yaml"))
    generated_config = work_dir / f"{model_name}.yaml"
    eval_enabled = eval_config.get("enabled") is True
    overrides = {
        "dataset_type": str(options.get("dataset_type") or "Colmap"),
        "model_name": model_name,
        "num_iterations": max_steps,
        "image_scale_factor": float(options.get("image_scale_factor") or 1.0),
        "gui": bool(options.get("gui", False)),
        "run_validation": bool(options.get("run_validation", False)),
        "render_testset": True if eval_enabled else bool(options.get("render_testset", False)),
        "preloading_level": int(options.get("preloading_level", 0)),
    }
    for key in (
        "use_mcmc",
        "max_primitives",
        "speedysplat_pruning",
        "filter_3d",
        "random_background",
        "densification_end_iteration",
        "morton_ordering_end_iteration",
        "random_initialization",
        "random_points",
    ):
        if key in options:
            overrides[key] = options[key]
    argv = [
        sys.executable,
        "-c",
        _fastergs_train_wrapper_script(),
        str(framework),
        str(provider_root),
        method_name,
        str(options.get("config") or ""),
        str(template),
        str(generated_config),
        str(source),
        json.dumps(overrides, separators=(",", ":")),
        str(bool(options.get("export_ply", True))).lower(),
        str(bool(options.get("ascii_ply", False))).lower(),
    ]
    process = _run(
        "fastergs",
        argv,
        cwd=framework,
        env=_pythonpath_env(framework / "src", extra=[framework / "scripts"]),
        log_dir=work_dir / "logs",
    )
    output_root = framework / "output" / method_name
    candidates = sorted(
        output_root.glob(f"{model_name}_*"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0,
    )
    if not candidates:
        raise FileNotFoundError(f"no Faster-GS output directory found under {output_root}")
    latest = candidates[-1]
    ply = latest / "final.ply"
    if not ply.is_file():
        ply = _latest_existing(latest, ["final.ply", "*.ply", "**/*.ply"])
    metrics = _parse_fastergs_metrics(latest, max_steps)
    return {
        "ply": str(ply),
        "completed_steps": max_steps,
        "process": process,
        "metrics": metrics,
    }


def _run(
    provider: str,
    argv: list[str],
    *,
    cwd: Path,
    log_dir: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    run_env.update(env or {})
    run_env.setdefault("PYTHONUTF8", "1")
    run_env.setdefault("PYTHONIOENCODING", "utf-8")
    timeout = int(os.environ.get("SFMAPI_PLUGIN_EXECUTE_TIMEOUT", "86400"))
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        env=run_env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    process = {
        "argv": argv,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    _write_json(log_dir / "process.json", process)
    (log_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{provider} command failed with exit {proc.returncode}; "
            f"stdout_tail={_tail(proc.stdout)!r}; stderr_tail={_tail(proc.stderr)!r}"
        )
    return process


def _native_root(provider: str) -> Path:
    config = _config(provider)
    return Path(os.environ.get(config.native_root_env, str(config.default_native_root))).resolve()


def _brush_executable() -> Path:
    override = os.environ.get("SFMAPI_BRUSH_EXECUTABLE")
    if override:
        return Path(override)
    root = _native_root("brush")
    return _first_existing(
        [root / "target" / "release" / "brush", root / "target" / "release" / "brush.exe"]
    )


def _lfs_executable() -> Path:
    override = os.environ.get("SFMAPI_LFS_EXECUTABLE")
    if override:
        return Path(override)
    root = _native_root("lfs")
    return _first_existing(
        [
            root / "dist" / "bin" / "LichtFeld-Studio",
            root / "dist" / "bin" / "run_lichtfeld.sh",
            root / "build" / "LichtFeld-Studio",
            root / "build" / "Release" / "LichtFeld-Studio",
            root / "build" / "Release" / "LichtFeld-Studio.exe",
        ]
    )


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _gpu_runtime_info(provider: str) -> dict[str, Any]:
    if provider == "brush":
        return _vulkan_gpu_runtime_info()
    return _nvidia_gpu_runtime_info()


def _nvidia_gpu_runtime_info() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except FileNotFoundError as exc:
        return {"gpu_runtime_available": False, "gpu_error": f"nvidia-smi not found: {exc}"}
    except subprocess.TimeoutExpired as exc:
        return {"gpu_runtime_available": False, "gpu_error": f"nvidia-smi timed out: {exc}"}
    devices = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    info: dict[str, Any] = {
        "gpu_runtime_available": proc.returncode == 0 and bool(devices),
        "gpu_devices": devices,
        "gpu_device": devices[0] if devices else None,
    }
    if proc.returncode != 0:
        info["gpu_error"] = (
            f"nvidia-smi exited {proc.returncode}; "
            f"stdout={_tail(proc.stdout)!r}; stderr={_tail(proc.stderr)!r}"
        )
    elif not devices:
        info["gpu_error"] = "nvidia-smi returned no GPUs"
    return info


def _vulkan_gpu_runtime_info() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except FileNotFoundError as exc:
        return {"gpu_runtime_available": False, "gpu_error": f"vulkaninfo not found: {exc}"}
    except subprocess.TimeoutExpired as exc:
        return {"gpu_runtime_available": False, "gpu_error": f"vulkaninfo timed out: {exc}"}
    devices = _parse_vulkan_devices(proc.stdout)
    hardware_devices = [
        device["name"]
        for device in devices
        if device["name"] and not _is_software_vulkan_device(device)
    ]
    device_names = [device["name"] for device in devices if device["name"]]
    available = proc.returncode == 0 and bool(hardware_devices)
    info: dict[str, Any] = {
        "gpu_runtime_available": available,
        "gpu_runtime_kind": "vulkan",
        "gpu_devices": hardware_devices,
        "gpu_device": hardware_devices[0] if hardware_devices else None,
        "vulkan_devices": device_names,
    }
    if proc.returncode != 0:
        info["gpu_error"] = (
            f"vulkaninfo exited {proc.returncode}; "
            f"stdout={_tail(proc.stdout)!r}; stderr={_tail(proc.stderr)!r}"
        )
    elif not device_names:
        info["gpu_error"] = "vulkaninfo returned no physical GPU devices"
    elif not hardware_devices:
        info["gpu_error"] = "vulkaninfo reported only software or CPU Vulkan devices"
    return info


def _parse_vulkan_devices(text: str) -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in text.splitlines():
        if re.match(r"^\s*GPU\d+:", line):
            if current is not None:
                devices.append(current)
            current = {"name": "", "type": "", "block": line.lower()}
            continue
        if current is None:
            continue
        current["block"] = f"{current['block']}\n{line.lower()}"
        match = re.match(r"^\s*(deviceName|deviceType)\s*=\s*(.+)$", line)
        if not match:
            continue
        key = "name" if match.group(1) == "deviceName" else "type"
        current[key] = match.group(2).strip()
    if current is not None:
        devices.append(current)
    if devices:
        return devices
    names = [
        match.strip()
        for match in re.findall(r"^\s*deviceName\s*=\s*(.+)$", text, flags=re.MULTILINE)
    ]
    return [{"name": name, "type": "", "block": name.lower()} for name in names]


def _is_software_vulkan_device(device: dict[str, str]) -> bool:
    text = f"{device.get('name', '')}\n{device.get('type', '')}\n{device.get('block', '')}".lower()
    software_markers = (
        "llvmpipe",
        "lavapipe",
        "software rasterizer",
        "physical_device_type_cpu",
    )
    return any(marker in text for marker in software_markers)


def _require_gpu_runtime(provider: str) -> None:
    info = _gpu_runtime_info(provider)
    if info.get("gpu_runtime_available") is True:
        return
    detail = info.get("gpu_error") or "GPU runtime is unavailable or returned no hardware GPUs"
    raise RuntimeError(f"GPU runtime is required for {provider}; {detail}")


def _require_torch_cuda(provider: str) -> None:
    _require_gpu_runtime(provider)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA torch runtime is required for {provider}; torch.cuda.is_available() is false"
        )


def _dataset_path(options: dict[str, Any]) -> Path:
    raw = options.get("dataset_path") or options.get("data_path") or options.get("data")
    if not isinstance(raw, str) or not raw:
        raise ValueError("backend_options.dataset_path is required")
    path = Path(raw)
    if not path.is_dir():
        raise FileNotFoundError(f"dataset_path does not exist: {path}")
    return path


def _snapshot_path(
    options: dict[str, Any],
    project_id: str,
    radiance_field_id: str,
    seq: int,
) -> Path:
    root = Path(
        str(
            options.get("output_path")
            or os.environ.get("SFMAPI_PLUGIN_OUTPUT_ROOT")
            or "/sfmapi/output"
        )
    )
    return root / project_id / radiance_field_id / "snapshots" / str(seq)


def _work_path(
    options: dict[str, Any], project_id: str, radiance_field_id: str, run_id: str | None
) -> Path:
    root = Path(
        str(options.get("work_path") or os.environ.get("SFMAPI_PLUGIN_WORK_ROOT") or "/sfmapi/work")
    )
    return root / project_id / radiance_field_id / (run_id or "run")


def _required_str(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"inputs.{key} is required")
    return value


def _max_steps(spec: dict[str, Any], options: dict[str, Any]) -> int:
    value = (
        spec.get("max_steps")
        or options.get("max_steps")
        or options.get("num_iterations")
        or options.get("iter")
        or options.get("total_train_iters")
        or 3000
    )
    steps = int(value)
    if steps < 1:
        raise ValueError("max_steps must be >= 1")
    return steps


def _eval_config(spec: dict[str, Any]) -> dict[str, Any]:
    raw = spec.get("eval")
    if not isinstance(raw, dict):
        return {"enabled": False}
    return raw


def _requested_metrics(eval_config: dict[str, Any]) -> set[str]:
    raw = eval_config.get("metrics")
    if isinstance(raw, list) and raw:
        return {str(item).lower() for item in raw}
    return {"psnr", "ssim", "lpips"}


def _parse_brush_metrics(
    process: dict[str, Any], *, duration_s: float | None
) -> dict[str, Any] | None:
    text = f"{process.get('stdout') or ''}\n{process.get('stderr') or ''}"
    matches = re.findall(
        r"Eval iter\s+(?P<iter>\d+):\s+PSNR\s+(?P<psnr>[-+0-9.eE]+),\s+ssim\s+(?P<ssim>[-+0-9.eE]+)",
        text,
    )
    if not matches:
        return None
    iteration, psnr, ssim = matches[-1]
    return {
        "iteration": int(iteration),
        "psnr_db": float(psnr),
        "ssim": float(ssim),
        "num_images": 0,
        "duration_s": duration_s,
        "render_time_s_total": None,
        "render_time_s_mean": None,
    }


def _parse_lfs_metrics(work_dir: Path) -> dict[str, Any] | None:
    metrics_path = work_dir / "metrics.csv"
    if not metrics_path.is_file():
        return None
    with metrics_path.open("r", encoding="utf-8", newline="") as fp:
        rows = [row for row in csv.DictReader(fp) if row]
    if not rows:
        return None
    row = rows[-1]
    num_images = _as_float(row.get("num_images") or row.get("image_count"))
    if num_images is None:
        num_images = _count_latest_eval_images(work_dir)
    return {
        "iteration": _as_float(row.get("iteration")),
        "psnr_db": _as_float(row.get("psnr")),
        "ssim": _as_float(row.get("ssim")),
        "num_images": num_images,
        "render_time_s_mean": _as_float(row.get("time_per_image")),
        "num_gaussians": _as_float(row.get("num_gaussians")),
    }


def _parse_json_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_fastergs_metrics(output_dir: Path, max_steps: int) -> dict[str, Any] | None:
    candidates = [
        output_dir / f"test_{max_steps}" / "metrics_8bit.txt",
        output_dir / "test" / "metrics_8bit.txt",
    ]
    candidates.extend(output_dir.glob("test_*/metrics_8bit.txt"))
    metrics_path = next((path for path in candidates if path.is_file()), None)
    if metrics_path is None:
        return None
    text = metrics_path.read_text(encoding="utf-8", errors="replace")
    parsed: dict[str, Any] = {"metrics_path": str(metrics_path)}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] in {"PSNR", "SSIM", "LPIPS"}:
            parsed[parts[0].lower()] = _as_float(parts[1])
    for key, value in re.findall(r"\b(PSNR|SSIM|LPIPS):([-+0-9.eE]+)", text):
        parsed[key.lower()] = _as_float(value)
    image_count = _count_fastergs_eval_images(metrics_path.parent)
    if image_count:
        parsed["num_images"] = image_count
    return parsed


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _find_metric(payload: Any, names: set[str]) -> float | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = key.lower().replace("-", "_")
            if normalized in names:
                parsed = _as_float(value)
                if parsed is not None:
                    return parsed
        for value in payload.values():
            found = _find_metric(value, names)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in reversed(payload):
            found = _find_metric(value, names)
            if found is not None:
                return found
    return None


def _metric_sample_count(payload: Any) -> int | None:
    metric_names = {
        "psnr",
        "psnr_db",
        "ssim",
        "ssim_pytorch_msssim",
        "ssim_torchmetrics",
        "lpips",
        "lpips_alex",
        "lpips_vgg",
    }
    if isinstance(payload, dict):
        counts: list[int] = []
        for key, value in payload.items():
            normalized = key.lower().replace("-", "_")
            if normalized in metric_names and isinstance(value, list):
                numeric_items = [item for item in value if _as_float(item) is not None]
                if numeric_items:
                    counts.append(len(numeric_items))
            nested = _metric_sample_count(value)
            if nested is not None:
                counts.append(nested)
        return max(counts) if counts else None
    if isinstance(payload, list):
        counts = [count for item in payload if (count := _metric_sample_count(item)) is not None]
        return max(counts) if counts else None
    return None


def _normalize_metrics(
    raw: Any, *, duration_s: float, required_metrics: set[str] | None = None
) -> dict[str, Any] | None:
    if raw is None:
        return None
    requested = required_metrics or set()
    psnr = _find_metric(raw, {"psnr", "psnr_db", "psnr_final", "avg_psnr"})
    ssim = _find_metric(
        raw,
        {
            "ssim",
            "ssim_final",
            "avg_ssim",
            "ssim_pytorch_msssim",
            "ssim_torchmetrics",
            "avg_ssim_pytorch_msssim",
            "avg_ssim_torchmetrics",
        },
    )
    lpips = _find_metric(
        raw,
        {
            "lpips",
            "lpips_final",
            "avg_lpips",
            "lpips_alex",
            "lpips_vgg",
            "avg_lpips_alex",
            "avg_lpips_vgg",
        },
    )
    missing = []
    if "psnr" in requested and psnr is None:
        missing.append("psnr")
    if "ssim" in requested and ssim is None:
        missing.append("ssim")
    if "lpips" in requested and lpips is None:
        missing.append("lpips")
    if missing:
        return None
    if psnr is None and ssim is None and lpips is None:
        return None
    num_images = _find_metric(raw, {"num_images", "image_count", "n_images"})
    if num_images is None:
        inferred_count = _metric_sample_count(raw)
        num_images = float(inferred_count) if inferred_count is not None else None
    render_total = _find_metric(raw, {"render_time_s_total", "render_time_total_s"})
    render_mean = _find_metric(
        raw, {"render_time_s_mean", "render_time_mean_s", "time_per_image", "elapsed_time"}
    )
    if render_total is None and render_mean is not None and num_images is not None:
        render_total = render_mean * int(num_images)
    return {
        "psnr_db": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "num_images": int(num_images or 0),
        "duration_s": duration_s,
        "render_time_s_total": render_total,
        "render_time_s_mean": render_mean,
    }


def _count_latest_eval_images(work_dir: Path) -> int | None:
    eval_dirs = [path for path in work_dir.glob("eval_step_*") if path.is_dir()]
    if not eval_dirs:
        return None
    latest = sorted(eval_dirs, key=lambda item: item.stat().st_mtime)[-1]
    count = _count_image_files(latest)
    return count or None


def _count_eval_split_images(dataset_path: Path, interval: int) -> int:
    interval = max(1, int(interval))
    image_count = _count_dataset_images(dataset_path)
    if image_count <= 0:
        return 0
    return (image_count + interval - 1) // interval


def _count_dataset_images(dataset_path: Path) -> int:
    preferred = [
        dataset_path / "images",
        dataset_path / "images_2",
        dataset_path / "images_4",
        dataset_path / "images_8",
    ]
    for path in preferred:
        count = _count_image_files(path)
        if count:
            return count
    return _count_image_files(dataset_path)


def _count_image_files(root: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    if not root.is_dir():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _count_fastergs_eval_images(root: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    if not root.is_dir():
        return 0
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]
    if not files:
        return 0
    prefixes = (
        "gt_",
        "ground_truth_",
        "truth_",
        "render_",
        "renders_",
        "pred_",
        "prediction_",
        "rgb_",
        "color_",
        "error_",
        "diff_",
    )
    view_ids: set[str] = set()
    for path in files:
        stem = path.stem.lower()
        for prefix in prefixes:
            if stem.startswith(prefix):
                stem = stem[len(prefix) :]
                break
        view_ids.add(stem)
    return len(view_ids) if 0 < len(view_ids) < len(files) else len(files)


def _latest_existing(root: Path, patterns: list[str]) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in root.glob(pattern) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"no artifact matched {patterns} under {root}")
    return sorted(matches, key=lambda item: item.stat().st_mtime)[-1]


def _ply_vertex_count(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                if line.startswith("element vertex "):
                    return int(line.split()[-1])
                if line.strip() == "end_header":
                    break
    except Exception:
        return None
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def _pythonpath_env(root: Path, *, extra: list[Path] | None = None) -> dict[str, str]:
    paths = [str(root), *[str(path) for path in extra or []]]
    if os.environ.get("PYTHONPATH"):
        paths.append(str(os.environ["PYTHONPATH"]))
    return {
        "PYTHONPATH": os.pathsep.join(paths),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }


def _spirulae_train_and_export_script(exporter: str) -> str:
    if exporter not in {"3dgs", "triangle"}:
        raise ValueError("spirulae exporter must be 3dgs or triangle")
    script_name = "export_ply_triangle.py" if exporter == "triangle" else "export_ply_3dgs.py"
    return (
        "from pathlib import Path\n"
        "import runpy, sys\n"
        "from spirulae_splat.ss_trainer import entrypoint\n"
        "root = Path(sys.argv[1])\n"
        "work_dir = Path(sys.argv[2])\n"
        "train_args = sys.argv[3:]\n"
        "sys.argv = ['spirulae-train', *train_args]\n"
        "entrypoint()\n"
        f"export_script = root / 'scripts' / {script_name!r}\n"
        "work_dir_arg = str(work_dir)\n"
        "if not work_dir_arg.endswith(('/', '\\\\')):\n"
        "    work_dir_arg += '/'\n"
        "sys.argv = [str(export_script), work_dir_arg]\n"
        "runpy.run_path(str(export_script), run_name='__main__')\n"
    )


def _fastergs_train_wrapper_script() -> str:
    return (
        "from pathlib import Path\n"
        "import json, runpy, sys\n"
        "import yaml\n"
        "framework = Path(sys.argv[1]).resolve()\n"
        "provider = Path(sys.argv[2]).resolve()\n"
        "method_name = sys.argv[3]\n"
        "config_arg = sys.argv[4]\n"
        "template = Path(sys.argv[5])\n"
        "generated_config = Path(sys.argv[6])\n"
        "dataset_path = sys.argv[7]\n"
        "overrides = json.loads(sys.argv[8])\n"
        "export_ply = sys.argv[9].lower() == 'true'\n"
        "ascii_ply = sys.argv[10].lower() == 'true'\n"
        "train_script = framework / 'scripts' / 'train.py'\n"
        "convert_script = framework / 'scripts' / 'convert_to_ply.py'\n"
        "if config_arg:\n"
        "    config_path = Path(config_arg)\n"
        "else:\n"
        "    cfg = yaml.safe_load(template.read_text())\n"
        "    cfg.setdefault('GLOBAL', {})['METHOD_TYPE'] = method_name\n"
        "    cfg['GLOBAL']['DATASET_TYPE'] = overrides.get('dataset_type', cfg['GLOBAL'].get('DATASET_TYPE'))\n"
        "    cfg.setdefault('TRAINING', {})['MODEL_NAME'] = overrides['model_name']\n"
        "    cfg['TRAINING']['NUM_ITERATIONS'] = overrides['num_iterations']\n"
        "    cfg['TRAINING']['RUN_VALIDATION'] = overrides.get('run_validation', False)\n"
        "    cfg['TRAINING'].setdefault('BACKUP', {})['RENDER_TESTSET'] = overrides.get('render_testset', False)\n"
        "    cfg['TRAINING'].setdefault('GUI', {})['ACTIVATE'] = overrides.get('gui', False)\n"
        "    cfg['TRAINING'].setdefault('DATA', {})['PRELOADING_LEVEL'] = overrides.get('preloading_level', cfg['TRAINING']['DATA'].get('PRELOADING_LEVEL', 2))\n"
        "    if 'use_mcmc' in overrides:\n"
        "        cfg['TRAINING']['USE_MCMC'] = overrides['use_mcmc']\n"
        "    if 'max_primitives' in overrides:\n"
        "        cfg['TRAINING']['MAX_PRIMITIVES'] = overrides['max_primitives']\n"
        "    if 'speedysplat_pruning' in overrides:\n"
        "        cfg['TRAINING'].setdefault('SPEEDYSPLAT_PRUNING', {})['USE'] = overrides['speedysplat_pruning']\n"
        "    if 'filter_3d' in overrides:\n"
        "        cfg['TRAINING'].setdefault('FILTER_3D', {})['USE'] = overrides['filter_3d']\n"
        "    if 'random_background' in overrides:\n"
        "        cfg['TRAINING']['USE_RANDOM_BACKGROUND_COLOR'] = overrides['random_background']\n"
        "    if 'densification_end_iteration' in overrides:\n"
        "        cfg['TRAINING']['DENSIFICATION_END_ITERATION'] = overrides['densification_end_iteration']\n"
        "    if 'morton_ordering_end_iteration' in overrides:\n"
        "        cfg['TRAINING']['MORTON_ORDERING_END_ITERATION'] = overrides['morton_ordering_end_iteration']\n"
        "    if 'random_initialization' in overrides:\n"
        "        cfg['TRAINING'].setdefault('RANDOM_INITIALIZATION', {})['FORCE'] = overrides['random_initialization']\n"
        "    if 'random_points' in overrides:\n"
        "        cfg['TRAINING'].setdefault('RANDOM_INITIALIZATION', {})['N_POINTS'] = overrides['random_points']\n"
        "    cfg.setdefault('DATASET', {})['PATH'] = dataset_path\n"
        "    cfg['DATASET']['IMAGE_SCALE_FACTOR'] = overrides.get('image_scale_factor', cfg['DATASET'].get('IMAGE_SCALE_FACTOR', 1.0))\n"
        "    generated_config.parent.mkdir(parents=True, exist_ok=True)\n"
        "    generated_config.write_text(yaml.safe_dump(cfg, sort_keys=False))\n"
        "    config_path = generated_config\n"
        "sys.path.insert(0, str(framework / 'src'))\n"
        "sys.path.insert(0, str(framework / 'scripts'))\n"
        "sys.argv = [str(train_script), '-c', str(config_path)]\n"
        "runpy.run_path(str(train_script), run_name='__main__')\n"
        "if export_ply:\n"
        "    output_root = framework / 'output' / method_name\n"
        "    model_name = overrides.get('model_name')\n"
        "    candidates = sorted(output_root.glob(f'{model_name}_*'), key=lambda p: p.stat().st_mtime)\n"
        "    if not candidates:\n"
        "        raise SystemExit(f'no Faster-GS output directory found under {output_root}')\n"
        "    latest = candidates[-1]\n"
        "    sys.argv = [str(convert_script), '-d', str(latest)]\n"
        "    if ascii_ply:\n"
        "        sys.argv.append('-t')\n"
        "    runpy.run_path(str(convert_script), run_name='__main__')\n"
    )
