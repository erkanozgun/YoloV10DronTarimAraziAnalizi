"""Tests for field-boundary filtering in the Streamlit app."""

from __future__ import annotations

import numpy as np
import torch
from ultralytics.engine.results import Boxes, Results

from app import calculate_report, filter_boxes_by_field


def create_result() -> Results:
    result = Results(
        np.zeros((100, 100, 3), dtype=np.uint8), path="test.jpg", names={0: "weed_cluster"}
    )
    result.boxes = Boxes(
        torch.tensor(
            [
                [0, 0, 40, 40, 0.9, 0],
                [50, 50, 90, 90, 0.8, 0],
            ]
        ),
        orig_shape=(100, 100),
    )
    return result


def test_filter_boxes_by_field() -> None:
    field_mask = np.zeros((100, 100), dtype=bool)
    field_mask[50:100, 50:100] = True

    result = filter_boxes_by_field(create_result(), field_mask)

    assert len(result.boxes) == 1
    report = calculate_report(result, (100, 100), field_mask)
    assert report == [
        {
            "Sorun": "Duzensiz bitki ortusu (yabanci ot suphesi)",
            "Tespit sayisi": 1,
            "Yaklasik alan orani": "%64.00",
        }
    ]
