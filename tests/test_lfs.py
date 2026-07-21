"""LichtFeld Studio-specific engine behavior, ported from
sfmapi_lfs/tests/test_protocol.py."""

from __future__ import annotations

from gs3.trainer import _parse_lfs_metrics


def test_parse_lfs_metrics_csv(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text(
        "iteration,psnr,ssim,time_per_image,num_gaussians\n3,19.5,0.64,0.012,123\n",
        encoding="utf-8",
    )
    eval_dir = tmp_path / "eval_step_3"
    eval_dir.mkdir()
    for idx in range(4):
        (eval_dir / f"{idx}.png").write_bytes(b"fake")

    metrics = _parse_lfs_metrics(tmp_path)

    assert metrics is not None
    assert metrics["psnr_db"] == 19.5
    assert metrics["ssim"] == 0.64
    assert metrics["num_images"] == 4
    assert metrics["render_time_s_mean"] == 0.012
