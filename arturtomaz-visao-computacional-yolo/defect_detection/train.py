#!/usr/bin/env python3
"""Train a YOLO model for metallic label defect detection."""

from __future__ import annotations

import argparse
from pathlib import Path


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run: pip install -r requirements"
        ) from exc
    return YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO for ranhura/amassado/mancha detection."
    )
    parser.add_argument(
        "--data",
        default="datasets/falhas.yaml",
        help="Path to the YOLO dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Base YOLO weights or YAML. Examples: yolo26n.pt, yolo11n.pt, yolov8n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", default=8, help="Batch size, or -1 for auto.")
    parser.add_argument("--device", default=None, help="Example: 0, cpu, cuda:0.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/defects")
    parser.add_argument("--name", default="train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze first N layers for small datasets; omit to train normally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    batch: int | float | str
    try:
        batch = int(args.batch)
    except ValueError:
        batch = args.batch

    YOLO = _load_yolo()
    model = YOLO(args.model)

    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": batch,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
        "plots": True,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze

    results = model.train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    print(f"Training finished. Results: {save_dir}")
    print(f"Best weights should be at: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

