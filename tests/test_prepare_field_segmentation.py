"""Tests for field-level train/validation splitting."""

from pathlib import Path

from scripts.prepare_field_segmentation import deterministic_split


def test_tiles_from_same_field_stay_in_same_split() -> None:
    first = Path("FIELD123_0-0-512-512.jpg")
    second = Path("FIELD123_512-0-1024-512.jpg")

    assert deterministic_split(first, 0.2) == deterministic_split(second, 0.2)
