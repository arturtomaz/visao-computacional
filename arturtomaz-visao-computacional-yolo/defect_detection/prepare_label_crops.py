#!/usr/bin/env python3
"""Create rectified label crops to annotate a YOLO defect dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ratio_calculator import _estimate_warp_size, detect_label_in_photo, warp_label


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rectify metallic labels before manual YOLO annotation."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="Image files or directories with original photos.",
    )
    parser.add_argument(
        "--output",
        default="datasets/falhas/images_to_annotate",
        help="Folder where rectified images will be written.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--copy-failed",
        action="store_true",
        help="Copy unreadable/undetected images as-is for later inspection.",
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "_debug" if args.debug else None
    failed_dir = output_dir / "_failed" if args.copy_failed else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
    if failed_dir:
        failed_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    failed: list[str] = []

    for source_arg in args.sources:
        source = Path(source_arg)
        if not source.exists():
            failed.append(str(source))
            continue

        for image_path in iter_images(source):
            corners, _, _ = detect_label_in_photo(
                str(image_path), str(debug_dir) if debug_dir else None
            )
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if corners is None or img is None:
                failed.append(str(image_path))
                if failed_dir and img is not None:
                    cv2.imwrite(str(failed_dir / f"{safe_stem(image_path)}.jpg"), img)
                continue

            out_w, out_h = _estimate_warp_size(corners, 640, 256)
            rectified = warp_label(img, corners.astype(np.float32), out_w, out_h)
            out_path = output_dir / f"{safe_stem(image_path)}.jpg"
            cv2.imwrite(str(out_path), rectified)
            saved += 1

    if failed:
        fail_path = output_dir / "failed.txt"
        fail_path.write_text("\n".join(failed), encoding="utf-8")
        print(f"Failed images: {len(failed)}. See: {fail_path}")

    print(f"Saved rectified images: {saved}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

