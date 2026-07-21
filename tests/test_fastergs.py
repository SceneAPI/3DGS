"""Faster-GS-specific engine behavior, ported from
sfmapi_fastergs/tests/test_protocol.py (including the view-pair image
counting case that repo alone carried)."""

from __future__ import annotations

from gs3.trainer import _parse_fastergs_metrics


def test_parse_fastergs_metrics_file(tmp_path) -> None:
    metrics_dir = tmp_path / "test_5"
    metrics_dir.mkdir()
    (metrics_dir / "metrics_8bit.txt").write_text(
        "Metric\tMean\tMedian\tPixelMean\n"
        "PSNR\t22.50\t22.10\t22.20\n"
        "SSIM\t0.820\t0.810\t0.815\n"
        "LPIPS\t0.120\t0.110\t0.115\n",
        encoding="utf-8",
    )
    for idx in range(2):
        (metrics_dir / f"{idx}.png").write_bytes(b"fake")

    metrics = _parse_fastergs_metrics(tmp_path, 5)

    assert metrics is not None
    assert metrics["psnr"] == 22.5
    assert metrics["ssim"] == 0.82
    assert metrics["lpips"] == 0.12
    assert metrics["num_images"] == 2


def test_parse_fastergs_metrics_counts_view_pairs_once(tmp_path) -> None:
    metrics_dir = tmp_path / "test_5"
    metrics_dir.mkdir()
    (metrics_dir / "metrics_8bit.txt").write_text(
        "Metric\tMean\tMedian\tPixelMean\n"
        "PSNR\t22.50\t22.10\t22.20\n"
        "SSIM\t0.820\t0.810\t0.815\n"
        "LPIPS\t0.120\t0.110\t0.115\n",
        encoding="utf-8",
    )
    for idx in range(2):
        (metrics_dir / f"gt_{idx:04d}.png").write_bytes(b"fake")
        (metrics_dir / f"render_{idx:04d}.png").write_bytes(b"fake")
        (metrics_dir / f"error_{idx:04d}.png").write_bytes(b"fake")

    metrics = _parse_fastergs_metrics(tmp_path, 5)

    assert metrics is not None
    assert metrics["num_images"] == 2
