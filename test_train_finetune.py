"""Unit tests for train_finetune.py's data-prep logic (bbox conversion,
class-index stability). Does NOT exercise the actual training call --
that needs ultralytics + a CUDA GPU and only runs on Carmody-PC.
"""

import pytest

from train_finetune import bbox_to_yolo, build_dataset


def test_bbox_to_yolo_center_and_size():
    # A 100x50 box centered at (400, 300) inside a 800x600 image.
    xc, yc, w, h = bbox_to_yolo([350, 275, 450, 325], img_w=800, img_h=600)
    assert xc == pytest.approx(400 / 800)
    assert yc == pytest.approx(300 / 600)
    assert w == pytest.approx(100 / 800)
    assert h == pytest.approx(50 / 600)


def test_bbox_to_yolo_full_frame():
    xc, yc, w, h = bbox_to_yolo([0, 0, 800, 600], img_w=800, img_h=600)
    assert xc == pytest.approx(0.5)
    assert yc == pytest.approx(0.5)
    assert w == pytest.approx(1.0)
    assert h == pytest.approx(1.0)


def test_bbox_to_yolo_clamps_out_of_bounds():
    # A bbox that overruns the image edges shouldn't produce values outside [0, 1].
    xc, yc, w, h = bbox_to_yolo([-50, -20, 900, 700], img_w=800, img_h=600)
    assert 0.0 <= xc <= 1.0
    assert 0.0 <= yc <= 1.0
    assert 0.0 <= w <= 1.0
    assert 0.0 <= h <= 1.0


def test_bbox_to_yolo_handles_swapped_corners():
    # bbox stored as [x2, y2, x1, y1] should still convert correctly.
    xc, yc, w, h = bbox_to_yolo([450, 325, 350, 275], img_w=800, img_h=600)
    assert xc == pytest.approx(400 / 800)
    assert yc == pytest.approx(300 / 600)


def test_build_dataset_skips_entries_missing_required_fields(tmp_path):
    manifest = [
        {"label": "bear", "bbox": None, "image": "/snapshots/x.jpg"},  # missing bbox
        {"label": "", "bbox": [1, 2, 3, 4], "image": "/snapshots/y.jpg"},  # missing label
        {"label": "bear", "bbox": [1, 2, 3, 4], "image": None},  # missing image
    ]
    stats = build_dataset(manifest, api_url="http://unused.invalid", dataset_dir=tmp_path / "ds")
    assert stats["count"] == 0
    assert stats["skipped"] == 3


def test_build_dataset_empty_manifest(tmp_path):
    stats = build_dataset([], api_url="http://unused.invalid", dataset_dir=tmp_path / "ds")
    assert stats["count"] == 0
    assert stats["classes"] == []
