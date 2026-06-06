"""Tests for field-boundary overlay rendering."""

import numpy as np

from app import create_field_overlay


def test_create_field_overlay_dims_outside_field() -> None:
    image = np.full((20, 20, 3), 200, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True

    overlay = create_field_overlay(image, mask)

    assert overlay[0, 0].tolist() == [70, 70, 70]
    assert overlay[10, 10].tolist() == [200, 200, 200]
