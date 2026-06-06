"""Prepare RGB images and field-boundary masks for binary segmentation training."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Official miniscale dataset root")
    parser.add_argument("--output", type=Path, required=True, help="Segmentation dataset output root")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser.parse_args()


def field_id(image: Path) -> str:
    return image.stem.split("_", maxsplit=1)[0]


def deterministic_split(image: Path, val_ratio: float) -> str:
    digest = hashlib.sha256(field_id(image).encode("utf-8")).digest()
    score = int.from_bytes(digest[:8], byteorder="big") / (2**64 - 1)
    return "val" if score < val_ratio else "train"


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio 0 ile 1 arasinda olmali")

    images_dir = args.source / "field_images" / "rgb"
    masks_dir = args.source / "field_bounds"
    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    totals = {"train": 0, "val": 0}

    for image in images:
        mask = masks_dir / f"{image.stem}.png"
        if not mask.exists():
            raise FileNotFoundError(f"Sinir maskesi bulunamadi: {mask}")

        split = deterministic_split(image, args.val_ratio)
        output_images = args.output / "images" / split
        output_masks = args.output / "masks" / split
        output_images.mkdir(parents=True, exist_ok=True)
        output_masks.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, output_images / image.name)
        shutil.copy2(mask, output_masks / mask.name)
        totals[split] += 1

    print(f"train: {totals['train']} goruntu")
    print(f"val: {totals['val']} goruntu")
    print(f"toplam: {sum(totals.values())} goruntu")


if __name__ == "__main__":
    main()
