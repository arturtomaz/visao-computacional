#!/usr/bin/env python3
"""Run YOLO defect detection on original or rectified metallic label images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ratio_calculator import _estimate_warp_size, detect_label_in_photo, warp_label


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
        description="Detect ranhura/amassado/mancha with a trained YOLO model."
    )
    parser.add_argument("--weights", required=True, help="Path to trained best.pt.")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--output", default="runs/defects/predict")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--rectify",
        action="store_true",
        help="Detect the metallic label first and run YOLO on the rectified crop.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Also save YOLO txt predictions.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save label-corner debug images when --rectify is used.",
    )
    return parser.parse_args()


def iter_images(source: Path) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            yield source
        return

    for path in sorted(source.rglob("*")):
        if "debug" in {part.lower() for part in path.parts}:
            continue
        if "template" in path.name.lower():
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def safe_stem(path: Path) -> str:
    parts = [p for p in path.with_suffix("").parts if p not in (".", "..")]
    return "_".join(parts[-3:]).replace(" ", "_").replace(":", "")


def rectify_sources(source: Path, output_dir: Path, debug: bool) -> Path:
    rectified_dir = output_dir / "_rectified_input"
    debug_dir = output_dir / "_rectify_debug" if debug else None
    rectified_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    failed: list[str] = []

    for image_path in iter_images(source):
        corners, (h_img, w_img), _ = detect_label_in_photo(
            str(image_path), str(debug_dir) if debug_dir else None
        )
        if corners is None:
            failed.append(str(image_path))
            continue

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            failed.append(str(image_path))
            continue

        out_w, out_h = _estimate_warp_size(corners, 640, 256)
        rectified = warp_label(img, corners.astype(np.float32), out_w, out_h)
        out_path = rectified_dir / f"{safe_stem(image_path)}.jpg"
        cv2.imwrite(str(out_path), rectified)
        saved += 1

    if failed:
        fail_path = output_dir / "rectify_failed.txt"
        fail_path.write_text("\n".join(failed), encoding="utf-8")
        print(f"Could not rectify {len(failed)} image(s). See: {fail_path}")

    if saved == 0:
        raise SystemExit("No image could be rectified.")

    print(f"Rectified {saved} image(s) into: {rectified_dir}")
    return rectified_dir


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def result_to_record(result) -> dict:
    detections = []
    names = getattr(result, "names", {})
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for box, score, class_id in zip(xyxy, conf, cls):
            detections.append(
                {
                    "class_id": int(class_id),
                    "class_name": class_name(names, int(class_id)),
                    "confidence": float(score),
                    "xyxy": [float(v) for v in box],
                }
            )

    return {
        "image": str(getattr(result, "path", "")),
        "detections": detections,
    }


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    predict_source = rectify_sources(source, output_dir, args.debug) if args.rectify else source

    YOLO = _load_yolo()
    model = YOLO(args.weights)

    predict_kwargs = {
        "source": str(predict_source),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "save": True,
        "save_txt": args.save_txt,
        "project": str(output_dir),
        "name": "predictions",
        "exist_ok": True,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)
    records = [result_to_record(result) for result in results]

    json_path = output_dir / "detections.json"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(len(record["detections"]) for record in records)
    print(f"Detected {total} defect candidate(s).")
    print(f"Images: {output_dir / 'predictions'}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

