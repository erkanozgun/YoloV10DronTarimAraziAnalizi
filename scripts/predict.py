"""Run agricultural problem detection on an image or image directory."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    parser.add_argument("--source", required=True, help="Image, directory, or video")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        project="runs/agriculture_analysis",
        name="predict",
    )


if __name__ == "__main__":
    main()
