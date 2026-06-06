"""Convert Agriculture-Vision segmentation masks into YOLO detection labels."""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_CLASSES = ("planter_skip", "double_plant", "weed_cluster", "water")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Agriculture-Vision root")
    parser.add_argument("--output", type=Path, required=True, help="YOLO dataset output root")
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--min-area", type=int, default=64, help="Minimum component area in pixels")
    parser.add_argument("--limit", type=int, help="Optional image limit per split for quick tests")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when sampling with --limit")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio for the official miniscale field_images/field_labels layout",
    )
    return parser.parse_args()


def find_images(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_mask(labels_dir: Path, class_name: str, image: Path) -> Path | None:
    class_dir = labels_dir / class_name
    for extension in (".png", image.suffix):
        candidate = class_dir / f"{image.stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def deterministic_split(image: Path, val_ratio: float) -> str:
    digest = hashlib.sha256(image.stem.encode("utf-8")).digest()
    score = int.from_bytes(digest[:8], byteorder="big") / (2**64 - 1)
    return "val" if score < val_ratio else "train"


def mask_to_yolo_lines(mask_path: Path, class_id: int, min_area: int) -> list[str]:
    mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Maske okunamadi: {mask_path}")

    binary = (mask > 0).astype("uint8")
    height, width = binary.shape
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    lines: list[str] = []

    for component_id in range(1, component_count):
        x, y, box_width, box_height, area = stats[component_id]
        if area < min_area:
            continue
        center_x = (x + box_width / 2) / width
        center_y = (y + box_height / 2) / height
        normalized_width = box_width / width
        normalized_height = box_height / height
        lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{normalized_width:.6f} {normalized_height:.6f}"
        )
    return lines


def convert_split(
    source: Path,
    output: Path,
    split: str,
    classes: list[str],
    min_area: int,
    limit: int | None,
) -> tuple[int, int]:
    image_dir = source / split / "images" / "rgb"
    labels_dir = source / split / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Goruntu klasoru bulunamadi: {image_dir}")

    output_images = output / "images" / split
    output_labels = output / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    images = find_images(image_dir)
    if limit is not None:
        images = images[:limit]

    box_count = 0
    for image in images:
        lines: list[str] = []
        for class_id, class_name in enumerate(classes):
            mask = find_mask(labels_dir, class_name, image)
            if mask is not None:
                lines.extend(mask_to_yolo_lines(mask, class_id, min_area))

        shutil.copy2(image, output_images / image.name)
        (output_labels / f"{image.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        box_count += len(lines)

    return len(images), box_count


def convert_miniscale(
    source: Path,
    output: Path,
    classes: list[str],
    min_area: int,
    limit: int | None,
    val_ratio: float,
    seed: int,
) -> dict[str, tuple[int, int]]:
    image_dir = source / "field_images" / "rgb"
    labels_dir = source / "field_labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Goruntu klasoru bulunamadi: {image_dir}")
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio 0 ile 1 arasinda olmali")

    images = find_images(image_dir)
    if limit is not None:
        random.Random(seed).shuffle(images)
        images = images[:limit]
    totals = {"train": [0, 0], "val": [0, 0]}

    for image in images:
        split = deterministic_split(image, val_ratio)
        output_images = output / "images" / split
        output_labels = output / "labels" / split
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for class_id, class_name in enumerate(classes):
            mask = find_mask(labels_dir, class_name, image)
            if mask is not None:
                lines.extend(mask_to_yolo_lines(mask, class_id, min_area))

        shutil.copy2(image, output_images / image.name)
        (output_labels / f"{image.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        totals[split][0] += 1
        totals[split][1] += len(lines)

    return {split: (counts[0], counts[1]) for split, counts in totals.items()}


def write_dataset_yaml(output: Path, classes: list[str]) -> None:
    config = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(classes)},
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    classes = list(args.classes)
    args.output.mkdir(parents=True, exist_ok=True)

    if (args.source / "field_images" / "rgb").exists():
        totals = convert_miniscale(
            args.source, args.output, classes, args.min_area, args.limit, args.val_ratio, args.seed
        )
        for split, (image_count, box_count) in totals.items():
            print(f"{split}: {image_count} goruntu, {box_count} kutu")
    else:
        for split in args.splits:
            image_count, box_count = convert_split(
                args.source, args.output, split, classes, args.min_area, args.limit
            )
            print(f"{split}: {image_count} goruntu, {box_count} kutu")

    write_dataset_yaml(args.output, classes)
    print(f"Tamamlandi: {args.output / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
