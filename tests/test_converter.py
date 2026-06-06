"""Small end-to-end test for the Agriculture-Vision converter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.convert_agriculture_vision import DEFAULT_CLASSES, convert_split, write_dataset_yaml


ROOT = Path(__file__).resolve().parents[1]


def create_split(source: Path, split: str) -> None:
    image_dir = source / split / "images" / "rgb"
    mask_dir = source / split / "labels" / "planter_skip"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[20:60, 50:150] = 255
    cv2.imencode(".png", image)[1].tofile(image_dir / "field.png")
    cv2.imencode(".png", mask)[1].tofile(mask_dir / "field.png")


def test_converter(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "processed"
    create_split(source, "train")
    create_split(source, "val")

    classes = list(DEFAULT_CLASSES)
    for split in ("train", "val"):
        convert_split(source, output, split, classes, min_area=64, limit=None)
    write_dataset_yaml(output, classes)

    label = (output / "labels" / "train" / "field.txt").read_text(encoding="utf-8")
    assert label == "0 0.500000 0.400000 0.500000 0.400000\n"
    config = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert config["names"][0] == "planter_skip"
