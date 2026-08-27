"""Dataset inspection and visualization helpers for YOLO instance segmentation."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
from PIL import Image
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_COLORS = ["#ef4444", "#22c55e", "#f59e0b", "#3b82f6", "#a855f7"]


def _read_config(dataset_dir: Path) -> tuple[dict[str, Any], list[str]]:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        yaml_path = dataset_dir / "data.yml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No data.yaml or data.yml found in {dataset_dir}")

    with yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    raw_names = config.get("names", [])
    if isinstance(raw_names, dict):
        names = [str(raw_names[key]) for key in sorted(raw_names, key=lambda value: int(value))]
    else:
        names = [str(name) for name in raw_names]
    if not names:
        names = [f"class_{index}" for index in range(int(config.get("nc", 0)))]
    return config, names


def _image_files(images_dir: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def _source_id(stem: str) -> str:
    """Strip Roboflow's augmentation hash to expose likely common source images."""
    return stem.split(".rf.", maxsplit=1)[0]


def audit_dataset(
    dataset_dir: str | Path,
    *,
    deep_coordinate_check: bool = False,
    max_recorded_issues: int = 200,
) -> dict[str, Any]:
    """Audit split structure, class balance, polygons, and likely source leakage.

    The deep coordinate check parses every polygon coordinate. It is intentionally
    optional because dense grain labels can contain millions of coordinates.
    """

    dataset_dir = Path(dataset_dir).expanduser().resolve()
    config, class_names = _read_config(dataset_dir)
    split_dirs = {"train": "train", "valid": "valid", "test": "test"}
    class_totals = {split: [0] * len(class_names) for split in split_dirs}
    issues: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    source_sets: dict[str, set[str]] = {}

    def record_issue(split: str, file: str, issue: str) -> None:
        if len(issues) < max_recorded_issues:
            issues.append({"split": split, "file": file, "issue": issue})

    for split, folder in split_dirs.items():
        images_dir = dataset_dir / folder / "images"
        labels_dir = dataset_dir / folder / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            record_issue(split, str(dataset_dir / folder), "missing images/ or labels/ directory")
            source_sets[split] = set()
            continue

        images = _image_files(images_dir)
        labels = {path.stem: path for path in labels_dir.glob("*.txt")}
        source_sets[split] = {_source_id(stem) for stem in images}

        for stem in sorted(images.keys() - labels.keys()):
            record_issue(split, images[stem].name, "image has no label file")
        for stem in sorted(labels.keys() - images.keys()):
            record_issue(split, labels[stem].name, "label has no matching image")

        for stem, image_path in images.items():
            label_path = labels.get(stem)
            instance_counts = [0] * len(class_names)
            invalid_instances = 0

            if label_path is not None:
                with label_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        tokens = line.split()
                        if not tokens:
                            continue
                        if len(tokens) < 7 or (len(tokens) - 1) % 2:
                            invalid_instances += 1
                            record_issue(split, label_path.name, f"line {line_number}: invalid polygon token count")
                            continue
                        try:
                            class_id = int(tokens[0])
                        except ValueError:
                            invalid_instances += 1
                            record_issue(split, label_path.name, f"line {line_number}: invalid class id")
                            continue
                        if not 0 <= class_id < len(class_names):
                            invalid_instances += 1
                            record_issue(split, label_path.name, f"line {line_number}: class id {class_id} out of range")
                            continue
                        if deep_coordinate_check:
                            try:
                                coordinates = [float(value) for value in tokens[1:]]
                            except ValueError:
                                invalid_instances += 1
                                record_issue(split, label_path.name, f"line {line_number}: non-numeric coordinate")
                                continue
                            if any(value < 0.0 or value > 1.0 for value in coordinates):
                                invalid_instances += 1
                                record_issue(split, label_path.name, f"line {line_number}: coordinate outside [0, 1]")
                                continue
                        instance_counts[class_id] += 1
                        class_totals[split][class_id] += 1

            per_image_rows.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "source_id": _source_id(stem),
                    "instances": sum(instance_counts),
                    "invalid_instances": invalid_instances,
                    **{class_names[index]: count for index, count in enumerate(instance_counts)},
                }
            )

    per_image = pd.DataFrame(per_image_rows)
    summary_rows: list[dict[str, Any]] = []
    for split in split_dirs:
        split_frame = per_image[per_image["split"] == split] if not per_image.empty else pd.DataFrame()
        counts = split_frame["instances"] if not split_frame.empty else pd.Series(dtype=float)
        summary_rows.append(
            {
                "split": split,
                "images": int(len(split_frame)),
                "instances": int(counts.sum()) if not counts.empty else 0,
                "instances/image mean": round(float(counts.mean()), 1) if not counts.empty else 0.0,
                "instances/image p95": int(counts.quantile(0.95)) if not counts.empty else 0,
                "instances/image max": int(counts.max()) if not counts.empty else 0,
                "images > 100 instances": int((counts > 100).sum()) if not counts.empty else 0,
            }
        )

    class_rows: list[dict[str, Any]] = []
    for split, totals in class_totals.items():
        split_total = sum(totals)
        for class_id, count in enumerate(totals):
            class_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "instances": count,
                    "percentage": round(100.0 * count / split_total, 3) if split_total else 0.0,
                }
            )

    overlap_rows: list[dict[str, Any]] = []
    for left, right in combinations(split_dirs, 2):
        overlap = source_sets.get(left, set()) & source_sets.get(right, set())
        overlap_rows.append(
            {
                "split_pair": f"{left}/{right}",
                "likely_shared_sources": len(overlap),
                "examples": ", ".join(sorted(overlap)[:8]),
            }
        )

    return {
        "dataset_dir": dataset_dir,
        "config": config,
        "class_names": class_names,
        "summary": pd.DataFrame(summary_rows),
        "class_counts": pd.DataFrame(class_rows),
        "source_overlaps": pd.DataFrame(overlap_rows),
        "issues": pd.DataFrame(issues, columns=["split", "file", "issue"]),
        "per_image": per_image,
    }


def plot_class_balance(class_counts: pd.DataFrame, *, log_scale: bool = True) -> None:
    """Plot instance counts per class and split."""

    pivot = class_counts.pivot(index="class_name", columns="split", values="instances")
    axis = pivot.plot(kind="bar", figsize=(10, 5), color=["#2563eb", "#10b981", "#f59e0b"])
    axis.set_title("Instance counts by class and split")
    axis.set_xlabel("")
    axis.set_ylabel("Instances" + (" (log scale)" if log_scale else ""))
    if log_scale:
        axis.set_yscale("log")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def _read_polygons(label_path: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    polygons: list[tuple[int, list[tuple[float, float]]]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.split()
            if len(tokens) < 7 or (len(tokens) - 1) % 2:
                continue
            class_id = int(tokens[0])
            values = [float(value) for value in tokens[1:]]
            polygons.append((class_id, list(zip(values[0::2], values[1::2]))))
    return polygons


def plot_yolo_samples(
    dataset_dir: str | Path,
    *,
    split: str = "train",
    count: int = 4,
    seed: int = 7,
    class_id: int | None = None,
    columns: int = 2,
) -> None:
    """Show YOLO polygon masks directly over their source images."""

    dataset_dir = Path(dataset_dir).expanduser().resolve()
    _, class_names = _read_config(dataset_dir)
    folder = "valid" if split in {"val", "valid"} else split
    images = _image_files(dataset_dir / folder / "images")
    labels_dir = dataset_dir / folder / "labels"

    candidates: list[tuple[Path, Path]] = []
    for stem, image_path in images.items():
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            continue
        if class_id is not None:
            prefix = f"{class_id} "
            if not any(line.startswith(prefix) for line in label_path.read_text(encoding="utf-8").splitlines()):
                continue
        candidates.append((image_path, label_path))

    if not candidates:
        raise ValueError(f"No matching samples found for split={split!r}, class_id={class_id!r}")
    rng = random.Random(seed)
    selected = rng.sample(candidates, min(count, len(candidates)))
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(7 * columns, 6 * rows), squeeze=False)

    for axis, (image_path, label_path) in zip(axes.flat, selected):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        axis.imshow(image)
        polygons = _read_polygons(label_path)
        for polygon_class, normalized_points in polygons:
            points = [(x * width, y * height) for x, y in normalized_points]
            color = DEFAULT_COLORS[polygon_class % len(DEFAULT_COLORS)]
            axis.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor=color, alpha=0.30, linewidth=1.0))
        counts = pd.Series([item[0] for item in polygons]).value_counts().to_dict()
        labels = ", ".join(
            f"{class_names[index]}={counts.get(index, 0)}" for index in range(len(class_names))
        )
        axis.set_title(f"{image_path.name}\n{labels}", fontsize=9)
        axis.axis("off")

    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    plt.tight_layout()
    plt.show()

