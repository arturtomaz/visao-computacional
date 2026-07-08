#!/usr/bin/env python3
"""Rebalance a YOLO detection dataset exported by Roboflow or LabelImg.

The script merges all existing train/valid/test images, optionally limits
background-only images, redistributes samples by class, and writes a clean
YOLO detection dataset with box-only labels.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "val": "valid",
    "valid": "valid",
    "test": "test",
}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path | None
    classes: frozenset[int]
    label_lines: tuple[str, ...]

    @property
    def is_background(self) -> bool:
        return not self.classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced YOLO detection dataset from an existing export."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Dataset root or data.yaml path from Roboflow/YOLO.",
    )
    parser.add_argument(
        "--output",
        default="datasets/falhas_rebalanced",
        help="Output dataset directory.",
    )
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument(
        "--max-background-ratio",
        type=float,
        default=0.30,
        help="Maximum background-only fraction in the output dataset.",
    )
    parser.add_argument(
        "--keep-all-backgrounds",
        action="store_true",
        help="Do not downsample background-only images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    return parser.parse_args()


def read_yaml_light(path: Path) -> dict:
    """Read the small subset of YAML commonly used by YOLO data files."""
    data: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            continue
        if key in {"names"}:
            try:
                data[key] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                data[key] = value
        elif key in {"nc"}:
            try:
                data[key] = int(value)
            except ValueError:
                data[key] = value
        else:
            data[key] = value.strip("'\"")
    return data


def normalize_names(raw_names: object, labels: list[str] | None = None) -> list[str]:
    if isinstance(raw_names, dict):
        return [str(raw_names[i]) for i in sorted(raw_names)]
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if labels:
        return labels
    return ["amassado", "mancha", "ranhura"]


def resolve_source(source_arg: str) -> tuple[Path, list[str]]:
    source = Path(source_arg)
    yaml_data: dict = {}
    yaml_path: Path | None = None

    if source.is_file():
        yaml_path = source
        yaml_data = read_yaml_light(source)
        root_value = yaml_data.get("path")
        if isinstance(root_value, str):
            root = Path(root_value)
            if not root.is_absolute():
                root = (source.parent / root).resolve()
        else:
            root = source.parent.resolve()
    else:
        root = source.resolve()
        candidate = root / "data.yaml"
        if candidate.exists():
            yaml_path = candidate
            yaml_data = read_yaml_light(candidate)

    names = normalize_names(yaml_data.get("names"))
    return root, names


def convert_line_to_box(line: str) -> str | None:
    """Return a clean YOLO box label line, converting polygons when needed."""
    parts = line.strip().split()
    if not parts:
        return None

    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None

    if len(coords) == 4:
        x_center, y_center, width, height = coords
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))
        y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
    else:
        return None

    x_center = min(1.0, max(0.0, x_center))
    y_center = min(1.0, max(0.0, y_center))
    width = min(1.0, max(0.0, width))
    height = min(1.0, max(0.0, height))
    if width <= 0.0 or height <= 0.0:
        return None

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def read_label(label_path: Path | None) -> tuple[frozenset[int], tuple[str, ...]]:
    if label_path is None or not label_path.exists():
        return frozenset(), tuple()

    class_ids: set[int] = set()
    lines: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        converted = convert_line_to_box(raw_line)
        if converted is None:
            continue
        class_ids.add(int(converted.split()[0]))
        lines.append(converted)
    return frozenset(class_ids), tuple(lines)


def split_candidates(root: Path) -> list[tuple[str, Path, Path]]:
    candidates: list[tuple[str, Path, Path]] = []

    for alias, split in SPLIT_ALIASES.items():
        image_dir = root / alias / "images"
        label_dir = root / alias / "labels"
        if image_dir.exists():
            candidates.append((split, image_dir, label_dir))

    for alias, split in SPLIT_ALIASES.items():
        image_dir = root / "images" / alias
        label_dir = root / "labels" / alias
        if image_dir.exists():
            candidates.append((split, image_dir, label_dir))

    return candidates


def collect_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen_images: set[Path] = set()

    for _split, image_dir, label_dir in split_candidates(root):
        for image_path in sorted(image_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved = image_path.resolve()
            if resolved in seen_images:
                continue
            seen_images.add(resolved)

            label_path = label_dir / f"{image_path.stem}.txt"
            classes, label_lines = read_label(label_path)
            samples.append(
                Sample(
                    image_path=image_path,
                    label_path=label_path if label_path.exists() else None,
                    classes=classes,
                    label_lines=label_lines,
                )
            )

    if not samples:
        raise SystemExit(f"No images found under YOLO dataset root: {root}")
    return samples


def split_targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: total * ratio for split, ratio in ratios.items()}
    targets = {split: int(raw[split]) for split in ratios}
    remaining = total - sum(targets.values())
    for split, _fraction in sorted(
        ((split, raw[split] - targets[split]) for split in ratios),
        key=lambda item: item[1],
        reverse=True,
    ):
        if remaining <= 0:
            break
        targets[split] += 1
        remaining -= 1
    return targets


def choose_split_for_positive(
    sample: Sample,
    split_names: list[str],
    split_items: dict[str, list[Sample]],
    split_class_counts: dict[str, Counter],
    target_counts: dict[str, int],
    target_class_counts: dict[str, Counter],
) -> str:
    best_split = split_names[0]
    best_score: tuple[float, float] | None = None

    for split in split_names:
        class_deficit = sum(
            max(0, target_class_counts[split][class_id] - split_class_counts[split][class_id])
            for class_id in sample.classes
        )
        size_deficit = max(0, target_counts[split] - len(split_items[split]))
        score = (class_deficit + 0.05 * size_deficit, -len(split_items[split]))
        if best_score is None or score > best_score:
            best_score = score
            best_split = split
    return best_split


def rebalance_samples(
    samples: list[Sample],
    ratios: dict[str, float],
    max_background_ratio: float,
    keep_all_backgrounds: bool,
    seed: int,
) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    positives = [sample for sample in samples if not sample.is_background]
    backgrounds = [sample for sample in samples if sample.is_background]

    if not keep_all_backgrounds and positives:
        max_backgrounds = int((len(positives) * max_background_ratio) / (1.0 - max_background_ratio))
        if len(backgrounds) > max_backgrounds:
            rng.shuffle(backgrounds)
            backgrounds = backgrounds[:max_backgrounds]

    selected = positives + backgrounds
    target_counts = split_targets(len(selected), ratios)

    global_class_counts = Counter()
    for sample in positives:
        global_class_counts.update(sample.classes)

    target_class_counts = {
        split: Counter(
            {
                class_id: max(1 if count >= 3 and split != "train" else 0, round(count * ratios[split]))
                for class_id, count in global_class_counts.items()
            }
        )
        for split in ratios
    }

    split_names = list(ratios)
    split_items: dict[str, list[Sample]] = {split: [] for split in split_names}
    split_class_counts: dict[str, Counter] = {split: Counter() for split in split_names}

    positives_sorted = positives[:]
    rng.shuffle(positives_sorted)
    positives_sorted.sort(key=lambda sample: min(global_class_counts[c] for c in sample.classes))

    for sample in positives_sorted:
        split = choose_split_for_positive(
            sample,
            split_names,
            split_items,
            split_class_counts,
            target_counts,
            target_class_counts,
        )
        split_items[split].append(sample)
        split_class_counts[split].update(sample.classes)

    rng.shuffle(backgrounds)
    for sample in backgrounds:
        split = max(split_names, key=lambda item: target_counts[item] - len(split_items[item]))
        split_items[split].append(sample)

    for split in split_names:
        rng.shuffle(split_items[split])
    return split_items


def unique_output_name(dest_dir: Path, image_path: Path) -> str:
    candidate = image_path.name
    if not (dest_dir / candidate).exists():
        return candidate

    index = 2
    while True:
        candidate = f"{image_path.stem}_{index}{image_path.suffix.lower()}"
        if not (dest_dir / candidate).exists():
            return candidate
        index += 1


def write_dataset(output_root: Path, split_items: dict[str, list[Sample]], names: list[str]) -> dict:
    report = {
        "output": str(output_root),
        "splits": {},
        "names": names,
    }

    for split, samples in split_items.items():
        image_dir = output_root / split / "images"
        label_dir = output_root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        split_counter = Counter()
        backgrounds = 0
        for sample in samples:
            out_name = unique_output_name(image_dir, sample.image_path)
            shutil.copy2(sample.image_path, image_dir / out_name)
            label_out = label_dir / f"{Path(out_name).stem}.txt"
            label_out.write_text("\n".join(sample.label_lines) + ("\n" if sample.label_lines else ""), encoding="utf-8")

            if sample.is_background:
                backgrounds += 1
            else:
                for line in sample.label_lines:
                    split_counter[int(line.split()[0])] += 1

        report["splits"][split] = {
            "images": len(samples),
            "background_images": backgrounds,
            "instances_by_class": {
                names[class_id] if class_id < len(names) else str(class_id): count
                for class_id, count in sorted(split_counter.items())
            },
        }

    yaml_text = "\n".join(
        [
            f"path: {output_root.resolve()}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "",
            f"nc: {len(names)}",
            f"names: {names!r}",
            "",
        ]
    )
    (output_root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (output_root / "balance_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def validate_ratios(args: argparse.Namespace) -> dict[str, float]:
    ratios = {"train": args.train, "valid": args.val, "test": args.test}
    total = sum(ratios.values())
    if total <= 0:
        raise SystemExit("Split ratios must sum to a positive number.")
    return {split: value / total for split, value in ratios.items()}


def main() -> None:
    args = parse_args()
    ratios = validate_ratios(args)
    root, names = resolve_source(args.source)
    output_root = Path(args.output)

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output already exists: {output_root}. Use --overwrite or choose another output."
            )
        shutil.rmtree(output_root)

    samples = collect_samples(root)
    split_items = rebalance_samples(
        samples,
        ratios,
        args.max_background_ratio,
        args.keep_all_backgrounds,
        args.seed,
    )
    report = write_dataset(output_root, split_items, names)

    print(f"Source: {root}")
    print(f"Output: {output_root.resolve()}")
    print(f"Classes: {names}")
    print(json.dumps(report["splits"], indent=2, ensure_ascii=False))
    print(f"Train with: --data \"{output_root / 'data.yaml'}\"")


if __name__ == "__main__":
    main()

