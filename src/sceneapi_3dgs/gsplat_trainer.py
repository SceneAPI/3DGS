"""gsplat trainer engine -- the one radiance provider whose trainer is
genuinely different from the shared native-engine module
(:mod:`sceneapi_3dgs.trainer`): it trains in process with CUDA torch +
gsplat rasterization (plus lpips/pycolmap for evaluation) instead of
shelling out to a native engine build."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sceneapi_3dgs.trainer import ExecuteRequest

PROVIDER = "gsplat"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {"provider": "gsplat", "gpu_required": True, "cuda_required": True}
    info.update(_gpu_runtime_info())
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
        info["cuda_available"] = False
    try:
        import gsplat

        info["gsplat"] = getattr(gsplat, "__version__", "installed")
    except Exception as exc:
        info["gsplat_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _apply_canonical_options(options: dict[str, Any]) -> dict[str, Any]:
    # Canonical radiance.train config-schema names -> gsplat's native keys.
    # num_gaussians is already native; map max_resolution -> target_size.
    # Back-fill only: an explicit native key always wins.
    resolved = dict(options)
    if "max_resolution" in resolved and "target_size" not in resolved:
        resolved["target_size"] = resolved["max_resolution"]
    return resolved


def train(request: ExecuteRequest) -> dict[str, Any]:
    started = time.perf_counter()
    torch, rasterization = _require_cuda_gsplat()
    spec = request.spec
    inputs = request.inputs
    options = spec.get("backend_options") if isinstance(spec.get("backend_options"), dict) else {}
    options = _apply_canonical_options(options)
    radiance_field_id = _required_str(inputs, "radiance_field_id")
    project_id = _required_str(inputs, "project_id")
    max_steps = int(spec.get("max_steps") or 3000)
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    if _colmap_sparse_path(options) is not None and options.get("single_image_smoke") is not True:
        return _train_colmap_dataset(
            request=request,
            torch=torch,
            rasterization=rasterization,
            started=started,
            options=options,
            project_id=project_id,
            radiance_field_id=radiance_field_id,
            max_steps=max_steps,
        )

    target = _target_tensor(torch, options)
    height, width = int(target.shape[0]), int(target.shape[1])
    num_gaussians = int(options.get("num_gaussians") or 2048)
    lr = float(options.get("learning_rate") or 0.03)
    log_interval = max(1, int(options.get("log_interval") or max_steps // 10 or 1))

    device = torch.device("cuda")
    means = torch.nn.Parameter(torch.randn(num_gaussians, 3, device=device) * 0.45)
    with torch.no_grad():
        means[:, 2].add_(2.0)
    raw_scales = torch.nn.Parameter(torch.full((num_gaussians, 3), -4.0, device=device))
    raw_quats = torch.nn.Parameter(torch.zeros(num_gaussians, 4, device=device))
    with torch.no_grad():
        raw_quats[:, 0] = 1.0
    raw_opacity = torch.nn.Parameter(torch.zeros(num_gaussians, device=device))
    raw_colors = torch.nn.Parameter(torch.rand(num_gaussians, 3, device=device))
    optimizer = torch.optim.Adam(
        [means, raw_scales, raw_quats, raw_opacity, raw_colors],
        lr=lr,
    )

    viewmats = torch.eye(4, device=device, dtype=torch.float32)[None]
    focal = float(options.get("focal") or max(width, height) * 0.9)
    Ks = torch.tensor(
        [[[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )

    samples: list[dict[str, float | int]] = []
    first_loss: float | None = None
    rgb = None
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        quats = torch.nn.functional.normalize(raw_quats, dim=-1)
        scales = torch.nn.functional.softplus(raw_scales) + 1e-4
        opacities = torch.sigmoid(raw_opacity)
        colors = torch.sigmoid(raw_colors)
        rendered, _alphas, _meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            packed=False,
        )
        rgb = rendered[0, ..., :3].clamp(0, 1)
        loss = torch.mean((rgb - target) ** 2)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = loss_value
        if step == 1 or step % log_interval == 0 or step == max_steps:
            psnr = -10.0 * np.log10(max(loss_value, 1e-12))
            samples.append({"step": step, "loss": loss_value, "psnr": float(psnr)})

    torch.cuda.synchronize()
    duration_s = round(time.perf_counter() - started, 3)
    eval_config = spec.get("eval") if isinstance(spec.get("eval"), dict) else {}
    evaluation_id = inputs.get("evaluation_id")
    eval_metrics: dict[str, Any] | None = None
    evaluations: list[dict[str, Any]] = []
    if eval_config.get("enabled") is True:
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise ValueError("inputs.evaluation_id is required when spec.eval.enabled=true")
        render_started = time.perf_counter()
        with torch.no_grad():
            quats = torch.nn.functional.normalize(raw_quats, dim=-1)
            scales = torch.nn.functional.softplus(raw_scales) + 1e-4
            opacities = torch.sigmoid(raw_opacity)
            colors = torch.sigmoid(raw_colors)
            rendered, _alphas, _meta = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmats,
                Ks=Ks,
                width=width,
                height=height,
                packed=False,
            )
            rgb = rendered[0, ..., :3].clamp(0, 1)
        torch.cuda.synchronize()
        render_time_s = round(time.perf_counter() - render_started, 6)
        requested = set(eval_config.get("metrics") or ["psnr", "ssim", "lpips"])
        eval_metrics = _compute_eval_metrics(
            torch,
            rgb,
            target,
            requested=requested,
            lpips_net=str(eval_config.get("lpips_net") or "alex"),
            duration_s=duration_s,
            render_time_s=render_time_s,
        )
    seq = int(options.get("snapshot_seq") or 1)
    snapshot_path = _snapshot_path(options, project_id, radiance_field_id, seq)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    means_np = means.detach().cpu().numpy()
    colors_np = (torch.sigmoid(raw_colors).detach().cpu().numpy() * 255).clip(0, 255)
    _write_ply(snapshot_path / "point_cloud.ply", means_np, colors_np)
    # Persist the full (raw) gaussian state so standalone `:evaluate` can
    # reload and re-render the dataset rather than only re-reading metrics.
    # The .ply carries means+colors only, which is not re-renderable.
    _save_gaussian_checkpoint(
        snapshot_path, torch, means, raw_quats, raw_scales, raw_opacity, raw_colors
    )
    summary = {
        "provider": "gsplat",
        "method": str(spec.get("method") or "gsplat.train.default"),
        "radiance_field_id": radiance_field_id,
        "dataset_id": inputs.get("dataset_id"),
        "max_steps": max_steps,
        "completed_steps": max_steps,
        "loss_initial": first_loss,
        "loss_final": samples[-1]["loss"],
        "psnr_final": samples[-1]["psnr"],
        "duration_s": duration_s,
        "vertex_count": int(num_gaussians),
        "format": "ply",
        "target_size": [width, height],
    }
    _write_json(snapshot_path / "summary.json", summary)
    metrics_payload: dict[str, Any] = {"samples": samples, "max_steps": max_steps}
    if eval_metrics is not None:
        metrics_payload.update(eval_metrics)
    _write_json(snapshot_path / "metrics.json", metrics_payload)
    _write_json(snapshot_path / "metadata.json", {"runtime": runtime_info(), **summary})
    if eval_metrics is not None and isinstance(evaluation_id, str):
        eval_artifacts = [
            {
                "kind": "radiance.evaluation.metrics",
                "name": "metrics.json",
                "uri": str(snapshot_path / "metrics.json"),
                "media_type": "application/json",
                "artifact_format": "sfmapi.radiance.metrics.v1",
                "summary": eval_metrics,
            }
        ]
        evaluations.append(
            {
                "evaluation_id": evaluation_id,
                "radiance_field_id": radiance_field_id,
                "snapshot_seq": seq,
                "metrics": eval_metrics,
                "artifacts": eval_artifacts,
            }
        )
    ply_uri = str(snapshot_path / "point_cloud.ply")
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
                "summary": {"vertex_count": int(num_gaussians)},
            },
        ],
        "variants": [
            {
                "format": "ply",
                "uri": ply_uri,
                "media_type": "application/octet-stream",
                "summary": {"vertex_count": int(num_gaussians)},
            }
        ],
    }


def evaluate(request: ExecuteRequest) -> dict[str, Any]:
    torch, rasterization = _require_cuda_gsplat()
    spec = request.spec
    inputs = request.inputs
    options = spec.get("backend_options") if isinstance(spec.get("backend_options"), dict) else {}
    radiance_field_id = _required_str(inputs, "radiance_field_id")
    evaluation_id = _required_str(inputs, "evaluation_id")
    project_id = _required_str(inputs, "project_id")
    seq = int(inputs.get("snapshot_seq") or spec.get("snapshot_seq") or options.get("snapshot_seq") or 1)
    snapshot_path = _snapshot_path(options, project_id, radiance_field_id, seq)
    started = time.perf_counter()
    eval_config = spec.get("eval") if isinstance(spec.get("eval"), dict) else {}
    checkpoint_path = _gaussian_checkpoint_path(snapshot_path)
    # True standalone evaluation: when a re-renderable checkpoint AND the
    # dataset are available, reload the gaussians and re-render the eval
    # split for fresh metrics. Otherwise fall back to the recorded metrics
    # (e.g. smoke-trained snapshots or callers that don't pass the dataset),
    # preserving prior behavior.
    if checkpoint_path.is_file() and _colmap_sparse_path(options) is not None:
        params = _load_gaussian_checkpoint(snapshot_path, torch)
        _reconstruction, frames = _load_colmap_frames(torch, options)
        test_every = max(1, int(options.get("test_every") or 8))
        eval_frames = [frame for idx, frame in enumerate(frames) if idx % test_every == 0]
        if not eval_frames:
            eval_frames = frames[: min(len(frames), 1)]
        requested = {str(metric).lower() for metric in (eval_config.get("metrics") or ["psnr", "ssim", "lpips"])}
        metrics = _compute_colmap_eval_metrics(
            torch,
            rasterization,
            eval_frames,
            params["means"],
            params["raw_quats"],
            params["raw_scales"],
            params["raw_opacity"],
            params["raw_colors"],
            requested=requested,
            lpips_net=str(eval_config.get("lpips_net") or "alex"),
            duration_s=round(time.perf_counter() - started, 3),
        )
        metrics_path = snapshot_path / "metrics.json"
        merged: dict[str, Any] = {}
        if metrics_path.is_file():
            try:
                merged = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                merged = {}
        merged.update(metrics)
        _write_json(metrics_path, merged)
    else:
        metrics_path = Path(str(options.get("metrics_path") or snapshot_path / "metrics.json"))
        if not metrics_path.is_file():
            raise FileNotFoundError(
                "gsplat standalone evaluation requires a gaussian checkpoint + dataset to "
                f"re-render, or an existing metrics file: {metrics_path}"
            )
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = _normalize_metrics_payload(raw, duration_s=round(time.perf_counter() - started, 3))
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


def _train_colmap_dataset(
    *,
    request: ExecuteRequest,
    torch: Any,
    rasterization: Any,
    started: float,
    options: dict[str, Any],
    project_id: str,
    radiance_field_id: str,
    max_steps: int,
) -> dict[str, Any]:
    spec = request.spec
    inputs = request.inputs
    reconstruction, frames = _load_colmap_frames(torch, options)
    test_every = max(1, int(options.get("test_every") or 8))
    train_frames = [frame for idx, frame in enumerate(frames) if idx % test_every != 0]
    eval_frames = [frame for idx, frame in enumerate(frames) if idx % test_every == 0]
    if not train_frames:
        train_frames = frames
    if not eval_frames:
        eval_frames = frames[: min(len(frames), 1)]

    device = torch.device("cuda")
    max_points = int(options.get("num_gaussians") or options.get("max_splats") or 100_000)
    points, point_colors = _colmap_points(reconstruction, max_points=max_points)
    means = torch.nn.Parameter(torch.as_tensor(points, device=device, dtype=torch.float32))
    colors_init = torch.as_tensor(point_colors, device=device, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    raw_colors = torch.nn.Parameter(torch.logit(colors_init))
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    init_scale = float(options.get("init_scale") or max(extent / 1200.0, 1e-3))
    raw_scales = torch.nn.Parameter(
        torch.full(
            (len(points), 3),
            float(np.log(np.expm1(init_scale))),
            device=device,
            dtype=torch.float32,
        )
    )
    raw_quats = torch.nn.Parameter(torch.zeros(len(points), 4, device=device, dtype=torch.float32))
    raw_opacity = torch.nn.Parameter(torch.full((len(points),), -2.1972246, device=device))
    with torch.no_grad():
        raw_quats[:, 0] = 1.0

    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": float(options.get("means_lr") or 1.6e-4)},
            {"params": [raw_scales], "lr": float(options.get("scales_lr") or 5e-3)},
            {"params": [raw_quats], "lr": float(options.get("quats_lr") or 1e-3)},
            {"params": [raw_opacity], "lr": float(options.get("opacities_lr") or 5e-2)},
            {"params": [raw_colors], "lr": float(options.get("colors_lr") or 2.5e-3)},
        ]
    )
    log_interval = max(1, int(options.get("log_interval") or max_steps // 10 or 1))
    samples: list[dict[str, float | int]] = []
    first_loss: float | None = None
    for step in range(1, max_steps + 1):
        frame = train_frames[(step - 1) % len(train_frames)]
        optimizer.zero_grad(set_to_none=True)
        rgb = _render_colmap_frame(
            torch,
            rasterization,
            frame,
            means,
            raw_quats,
            raw_scales,
            raw_opacity,
            raw_colors,
        )
        target = frame["image"].to(device, non_blocking=True)
        loss = torch.mean((rgb - target) ** 2)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = loss_value
        if step == 1 or step % log_interval == 0 or step == max_steps:
            samples.append(
                {
                    "step": step,
                    "loss": loss_value,
                    "psnr": float(-10.0 * np.log10(max(loss_value, 1e-12))),
                }
            )

    torch.cuda.synchronize()
    duration_s = round(time.perf_counter() - started, 3)
    eval_config = spec.get("eval") if isinstance(spec.get("eval"), dict) else {}
    evaluation_id = inputs.get("evaluation_id")
    eval_metrics: dict[str, Any] | None = None
    evaluations: list[dict[str, Any]] = []
    if eval_config.get("enabled") is True:
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise ValueError("inputs.evaluation_id is required when spec.eval.enabled=true")
        requested = set(eval_config.get("metrics") or ["psnr", "ssim", "lpips"])
        eval_metrics = _compute_colmap_eval_metrics(
            torch,
            rasterization,
            eval_frames,
            means,
            raw_quats,
            raw_scales,
            raw_opacity,
            raw_colors,
            requested=requested,
            lpips_net=str(eval_config.get("lpips_net") or "alex"),
            duration_s=duration_s,
        )

    seq = int(options.get("snapshot_seq") or 1)
    snapshot_path = _snapshot_path(options, project_id, radiance_field_id, seq)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    means_np = means.detach().cpu().numpy()
    colors_np = (torch.sigmoid(raw_colors).detach().cpu().numpy() * 255).clip(0, 255)
    _write_ply(snapshot_path / "point_cloud.ply", means_np, colors_np)
    # Persist the full (raw) gaussian state so standalone `:evaluate` can
    # reload and re-render the dataset rather than only re-reading metrics.
    # The .ply carries means+colors only, which is not re-renderable.
    _save_gaussian_checkpoint(
        snapshot_path, torch, means, raw_quats, raw_scales, raw_opacity, raw_colors
    )
    summary = {
        "provider": "gsplat",
        "method": str(spec.get("method") or "gsplat.train.default"),
        "radiance_field_id": radiance_field_id,
        "dataset_id": inputs.get("dataset_id"),
        "max_steps": max_steps,
        "completed_steps": max_steps,
        "loss_initial": first_loss,
        "loss_final": samples[-1]["loss"],
        "psnr_final": samples[-1]["psnr"],
        "duration_s": duration_s,
        "vertex_count": len(points),
        "format": "ply",
        "target_size": [frames[0]["width"], frames[0]["height"]],
        "eval_protocol": "colmap_interval",
    }
    _write_json(snapshot_path / "summary.json", summary)
    metrics_payload: dict[str, Any] = {"samples": samples, "max_steps": max_steps}
    if eval_metrics is not None:
        metrics_payload.update(eval_metrics)
    _write_json(snapshot_path / "metrics.json", metrics_payload)
    _write_json(snapshot_path / "metadata.json", {"runtime": runtime_info(), **summary})
    if eval_metrics is not None and isinstance(evaluation_id, str):
        eval_artifacts = [
            {
                "kind": "radiance.evaluation.metrics",
                "name": "metrics.json",
                "uri": str(snapshot_path / "metrics.json"),
                "media_type": "application/json",
                "artifact_format": "sfmapi.radiance.metrics.v1",
                "summary": eval_metrics,
            }
        ]
        evaluations.append(
            {
                "evaluation_id": evaluation_id,
                "radiance_field_id": radiance_field_id,
                "snapshot_seq": seq,
                "metrics": eval_metrics,
                "artifacts": eval_artifacts,
            }
        )
    ply_uri = str(snapshot_path / "point_cloud.ply")
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
                "summary": {"vertex_count": len(points)},
            },
        ],
        "variants": [
            {
                "format": "ply",
                "uri": ply_uri,
                "media_type": "application/octet-stream",
                "summary": {"vertex_count": len(points)},
            }
        ],
    }


def _render_colmap_frame(
    torch: Any,
    rasterization: Any,
    frame: dict[str, Any],
    means: Any,
    raw_quats: Any,
    raw_scales: Any,
    raw_opacity: Any,
    raw_colors: Any,
) -> Any:
    device = means.device
    quats = torch.nn.functional.normalize(raw_quats, dim=-1)
    scales = torch.nn.functional.softplus(raw_scales) + 1e-4
    opacities = torch.sigmoid(raw_opacity)
    colors = torch.sigmoid(raw_colors)
    rendered, _alphas, _meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=frame["viewmat"].to(device)[None],
        Ks=frame["K"].to(device)[None],
        width=int(frame["width"]),
        height=int(frame["height"]),
        packed=False,
    )
    return rendered[0, ..., :3].clamp(0, 1)


def _gaussian_checkpoint_path(snapshot_path: Path) -> Path:
    return snapshot_path / "gaussians.npz"


def _save_gaussian_checkpoint(
    snapshot_path: Path,
    torch: Any,
    means: Any,
    raw_quats: Any,
    raw_scales: Any,
    raw_opacity: Any,
    raw_colors: Any,
) -> None:
    # Raw (pre-activation) params: _render_colmap_frame applies
    # normalize/softplus/sigmoid, so these are exactly what re-render needs.
    np.savez(
        _gaussian_checkpoint_path(snapshot_path),
        means=means.detach().cpu().numpy(),
        raw_quats=raw_quats.detach().cpu().numpy(),
        raw_scales=raw_scales.detach().cpu().numpy(),
        raw_opacity=raw_opacity.detach().cpu().numpy(),
        raw_colors=raw_colors.detach().cpu().numpy(),
    )


def _load_gaussian_checkpoint(snapshot_path: Path, torch: Any) -> dict[str, Any]:
    data = np.load(_gaussian_checkpoint_path(snapshot_path))
    device = torch.device("cuda")
    return {
        key: torch.as_tensor(data[key], device=device, dtype=torch.float32)
        for key in ("means", "raw_quats", "raw_scales", "raw_opacity", "raw_colors")
    }


def _compute_colmap_eval_metrics(
    torch: Any,
    rasterization: Any,
    frames: list[dict[str, Any]],
    means: Any,
    raw_quats: Any,
    raw_scales: Any,
    raw_opacity: Any,
    raw_colors: Any,
    *,
    requested: set[str],
    lpips_net: str,
    duration_s: float,
) -> dict[str, Any]:
    if not frames:
        raise RuntimeError("gsplat COLMAP evaluation has no validation images")
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    lpips_model = _lpips_model(raw_colors.device, lpips_net) if "lpips" in requested else None
    render_time_s_total = 0.0
    with torch.no_grad():
        for frame in frames:
            torch.cuda.synchronize()
            render_started = time.perf_counter()
            rgb = _render_colmap_frame(
                torch,
                rasterization,
                frame,
                means,
                raw_quats,
                raw_scales,
                raw_opacity,
                raw_colors,
            )
            torch.cuda.synchronize()
            render_time_s_total += time.perf_counter() - render_started
            target = frame["image"].to(raw_colors.device, non_blocking=True)
            mse = torch.mean((rgb - target) ** 2).detach()
            if "psnr" in requested:
                psnr_values.append(float((-10.0 * torch.log10(torch.clamp(mse, min=1e-12))).cpu()))
            if "ssim" in requested:
                ssim_values.append(_ssim_value(torch, rgb, target))
            if "lpips" in requested:
                lpips_values.append(_lpips_value_with_model(torch, rgb, target, lpips_model))
    metrics: dict[str, Any] = {
        "psnr_db": _mean(psnr_values) if "psnr" in requested else None,
        "ssim": _mean(ssim_values) if "ssim" in requested else None,
        "lpips": _mean(lpips_values) if "lpips" in requested else None,
        "num_images": len(frames),
        "eval_protocol": "colmap_interval",
        "duration_s": duration_s,
        "render_time_s_total": round(render_time_s_total, 6),
        "render_time_s_mean": round(render_time_s_total / len(frames), 6),
    }
    missing = [name for name in ("psnr", "ssim", "lpips") if name in requested and metrics[_metric_key(name)] is None]
    if missing:
        raise RuntimeError(f"failed to compute requested eval metrics: {', '.join(missing)}")
    return metrics


def _compute_eval_metrics(
    torch: Any,
    rgb: Any,
    target: Any,
    *,
    requested: set[str],
    lpips_net: str,
    duration_s: float,
    render_time_s: float,
) -> dict[str, Any]:
    mse = torch.mean((rgb - target) ** 2).detach()
    metrics: dict[str, Any] = {
        "psnr_db": None,
        "ssim": None,
        "lpips": None,
        "num_images": 1,
        "eval_protocol": "single_image_smoke",
        "duration_s": duration_s,
        "render_time_s_total": render_time_s,
        "render_time_s_mean": render_time_s,
    }
    if "psnr" in requested:
        metrics["psnr_db"] = float((-10.0 * torch.log10(torch.clamp(mse, min=1e-12))).cpu())
    if "ssim" in requested:
        metrics["ssim"] = _ssim_value(torch, rgb, target)
    if "lpips" in requested:
        metrics["lpips"] = _lpips_value(torch, rgb, target, lpips_net)
    missing = [name for name in ("psnr", "ssim", "lpips") if name in requested and metrics[_metric_key(name)] is None]
    if missing:
        raise RuntimeError(f"failed to compute requested eval metrics: {', '.join(missing)}")
    return metrics


def _metric_key(name: str) -> str:
    return "psnr_db" if name == "psnr" else name


def _ssim_value(torch: Any, rgb: Any, target: Any) -> float:
    # Windowed SSIM (Wang et al. 2004): local statistics under an 11x11
    # Gaussian window (sigma=1.5), averaged over the image — not a single
    # global mean/variance over the whole flattened image (which reads
    # artificially high and is not comparable to standard/upstream SSIM).
    # Inputs are HxWx3 in [0, 1]; L (dynamic range) = 1 so C1/C2 use K1=0.01,
    # K2=0.03 directly.
    x = rgb.permute(2, 0, 1).unsqueeze(0)
    y = target.to(device=x.device, dtype=x.dtype).permute(2, 0, 1).unsqueeze(0)
    channels = int(x.shape[1])
    c1 = 0.01**2
    c2 = 0.03**2
    win = min(11, int(x.shape[-1]), int(x.shape[-2]))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        # Image smaller than a usable window: fall back to global statistics.
        xf, yf = x.reshape(-1), y.reshape(-1)
        mu_x, mu_y = xf.mean(), yf.mean()
        var_x = ((xf - mu_x) ** 2).mean()
        var_y = ((yf - mu_y) ** 2).mean()
        cov_xy = ((xf - mu_x) * (yf - mu_y)).mean()
        score = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
            (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        )
        return float(score.clamp(0, 1).detach().cpu())
    sigma = 1.5
    coords = torch.arange(win, dtype=x.dtype, device=x.device) - (win - 1) / 2.0
    gauss = torch.exp(-(coords**2) / (2.0 * sigma**2))
    gauss = gauss / gauss.sum()
    window = (gauss[:, None] * gauss[None, :]).expand(channels, 1, win, win).contiguous()
    conv2d = torch.nn.functional.conv2d  # 'valid' convolution, grouped per channel
    mu_x = conv2d(x, window, groups=channels)
    mu_y = conv2d(y, window, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = conv2d(x * x, window, groups=channels) - mu_x2
    sigma_y2 = conv2d(y * y, window, groups=channels) - mu_y2
    sigma_xy = conv2d(x * y, window, groups=channels) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return float(ssim_map.mean().clamp(0, 1).detach().cpu())


def _lpips_value(torch: Any, rgb: Any, target: Any, net: str) -> float:
    model = _lpips_model(rgb.device, net)
    return _lpips_value_with_model(torch, rgb, target, model)


def _lpips_model(device: Any, net: str) -> Any:
    try:
        import lpips
    except Exception as exc:
        raise RuntimeError("Python package 'lpips' is required for LPIPS evaluation") from exc
    return lpips.LPIPS(net=net).to(device).eval()


def _lpips_value_with_model(torch: Any, rgb: Any, target: Any, model: Any) -> float:
    pred = rgb.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    ref = target.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    with torch.no_grad():
        value = model(pred, ref)
    return float(value.detach().cpu().reshape(-1)[0])


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _load_colmap_frames(torch: Any, options: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    try:
        import pycolmap
    except Exception as exc:
        raise RuntimeError("Python package 'pycolmap' is required for gsplat COLMAP training") from exc
    sparse_path = _colmap_sparse_path(options)
    if sparse_path is None:
        raise FileNotFoundError("gsplat COLMAP training requires dataset_path/sparse or sparse_path")
    reconstruction = pycolmap.Reconstruction(str(sparse_path))
    dataset_path = Path(str(options.get("dataset_path") or sparse_path.parents[1]))
    image_root = _image_root(options, dataset_path)
    max_size = int(options.get("target_size") or options.get("max_resolution") or 384)
    frames: list[dict[str, Any]] = []
    for image in sorted(reconstruction.images.values(), key=lambda item: item.name):
        path = image_root / image.name
        if not path.is_file():
            continue
        pil_image = Image.open(path).convert("RGB")
        source_width, source_height = pil_image.size
        scale = min(1.0, max_size / max(source_width, source_height)) if max_size > 0 else 1.0
        if scale < 1.0:
            width = max(1, round(source_width * scale))
            height = max(1, round(source_height * scale))
            pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
        width, height = pil_image.size
        arr = np.asarray(pil_image, dtype=np.float32) / 255.0
        camera = reconstruction.cameras[image.camera_id]
        K = np.asarray(camera.calibration_matrix(), dtype=np.float32).copy()
        K[0, :] *= width / float(camera.width)
        K[1, :] *= height / float(camera.height)
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :4] = np.asarray(image.cam_from_world().matrix(), dtype=np.float32)
        frames.append(
            {
                "name": image.name,
                "image": torch.from_numpy(arr).pin_memory(),
                "K": torch.from_numpy(K),
                "viewmat": torch.from_numpy(viewmat),
                "width": width,
                "height": height,
            }
        )
    if len(frames) < 2:
        raise RuntimeError(f"gsplat COLMAP training found fewer than two images under {image_root}")
    return reconstruction, frames


def _colmap_points(reconstruction: Any, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    points = list(reconstruction.points3D.values())
    if not points:
        raise RuntimeError("gsplat COLMAP training requires sparse points")
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = [points[int(index)] for index in indices]
    xyz = np.asarray([point.xyz for point in points], dtype=np.float32)
    colors = np.asarray([point.color for point in points], dtype=np.float32) / 255.0
    return xyz, colors


def _colmap_sparse_path(options: dict[str, Any]) -> Path | None:
    raw_sparse = options.get("sparse_path") or options.get("colmap_sparse_path")
    candidates: list[Path] = []
    if isinstance(raw_sparse, str) and raw_sparse:
        candidates.append(Path(raw_sparse))
    raw_dataset = options.get("dataset_path")
    if isinstance(raw_dataset, str) and raw_dataset:
        dataset = Path(raw_dataset)
        candidates.extend([dataset / "sparse" / "0", dataset / "sparse"])
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if any((candidate / name).is_file() for name in ("cameras.bin", "cameras.txt")):
            return candidate
    return None


def _image_root(options: dict[str, Any], dataset_path: Path) -> Path:
    raw = options.get("image_root") or options.get("images")
    candidates = [Path(raw)] if isinstance(raw, str) and raw else []
    candidates.extend([dataset_path / "images_2", dataset_path / "images"])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"gsplat COLMAP training could not find an image directory under {dataset_path}")


def _normalize_metrics_payload(raw: Any, *, duration_s: float) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("metrics payload must be a JSON object")
    psnr = _number(raw.get("psnr_db", raw.get("psnr_final", raw.get("psnr"))))
    ssim = _number(raw.get("ssim"))
    lpips_value = _number(raw.get("lpips"))
    if psnr is None or ssim is None or lpips_value is None:
        raise ValueError("metrics payload must include psnr_db, ssim, and lpips")
    return {
        "psnr_db": psnr,
        "ssim": ssim,
        "lpips": lpips_value,
        "num_images": int(_number(raw.get("num_images")) or 1),
        "eval_protocol": str(raw.get("eval_protocol") or "single_image_smoke"),
        "duration_s": _number(raw.get("duration_s")) or duration_s,
        "render_time_s_total": _number(raw.get("render_time_s_total")),
        "render_time_s_mean": _number(raw.get("render_time_s_mean")),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _gpu_runtime_info() -> dict[str, Any]:
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


def _require_gpu_runtime() -> None:
    info = _gpu_runtime_info()
    if info.get("gpu_runtime_available") is True:
        return
    detail = info.get("gpu_error") or "nvidia-smi is unavailable or returned no GPUs"
    raise RuntimeError(f"GPU runtime is required for sfmapi-gsplat; {detail}")


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def _require_cuda_gsplat() -> tuple[Any, Any]:
    _require_gpu_runtime()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA torch runtime is required for sfmapi-gsplat; torch.cuda.is_available() is false")
    try:
        from gsplat import rasterization
    except Exception as exc:
        raise RuntimeError("Python package 'gsplat' is required for training") from exc
    return torch, rasterization


def _target_tensor(torch: Any, options: dict[str, Any]) -> Any:
    path = _target_image_path(options)
    if path is None:
        if options.get("allow_synthetic_target") is True:
            size = int(options.get("target_size") or 128)
            arr = np.zeros((size, size, 3), dtype=np.float32)
            yy, xx = np.mgrid[0:size, 0:size]
            arr[..., 0] = xx / max(size - 1, 1)
            arr[..., 1] = yy / max(size - 1, 1)
            arr[..., 2] = 0.25
            return torch.from_numpy(arr).to("cuda")
        raise RuntimeError(
            "gsplat training requires backend_options.image_path or dataset_path; "
            "set allow_synthetic_target=true only for explicit CUDA smoke tests"
        )
    size = int(options.get("target_size") or 256)
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    image = image.resize((image.width, image.height), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to("cuda")


def _target_image_path(options: dict[str, Any]) -> Path | None:
    for key in ("image_path", "target_image"):
        raw = options.get(key)
        if isinstance(raw, str) and raw:
            path = Path(raw)
            if not path.is_file():
                raise FileNotFoundError(f"{key} does not exist: {path}")
            return path
    for key in ("dataset_path", "image_root"):
        raw = options.get(key)
        if not isinstance(raw, str) or not raw:
            continue
        root = Path(raw)
        if not root.is_dir():
            raise FileNotFoundError(f"{key} does not exist: {root}")
        for item in sorted(root.rglob("*")):
            if item.suffix.lower() in IMAGE_EXTENSIONS and item.is_file():
                return item
        raise FileNotFoundError(f"{key} contains no supported images: {root}")
    return None


def _snapshot_path(
    options: dict[str, Any],
    project_id: str,
    radiance_field_id: str,
    seq: int,
) -> Path:
    root = Path(
        str(
            options.get("output_path")
            # Unified cross-plugin convention; SFMAPI_GSPLAT_OUTPUT_ROOT kept
            # for back-compat so this plugin honors the same env as its siblings.
            or os.environ.get("SFMAPI_PLUGIN_OUTPUT_ROOT")
            or os.environ.get("SFMAPI_GSPLAT_OUTPUT_ROOT")
            or "/sfmapi/output"
        )
    )
    return root / project_id / radiance_field_id / "snapshots" / str(seq)


def _required_str(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"inputs.{key} is required")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ply(path: Path, means: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as fp:
        fp.write("ply\nformat ascii 1.0\n")
        fp.write(f"element vertex {len(means)}\n")
        fp.write("property float x\nproperty float y\nproperty float z\n")
        fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fp.write("end_header\n")
        for xyz, rgb in zip(means, colors, strict=True):
            fp.write(
                f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )
