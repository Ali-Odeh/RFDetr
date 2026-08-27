from __future__ import annotations

import base64
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


CLASS_NAMES = ("bad seed", "healthy seed", "impurity")
# Approach 1 checkpoint semantics are reversed for bad/healthy. The raw model
# class ID is preserved on every GrainDetection for auditability, while all
# user-facing labels, colors, counts, exports, and evaluation use this mapping.
APPROACH_ONE_EVALUATION_CLASS_MAP = {
    "bad seed": "healthy seed",
    "healthy seed": "bad seed",
    "impurity": "impurity",
}
APPROACH_ONE_EVALUATION_CLASS_ID_MAP = {0: 1, 1: 0, 2: 2}
MODEL_VARIANT = "RF-DETR Segmentation Large V2"
MODEL_QUERY_CAPACITY = 200
DEFAULT_SATURATION_THRESHOLD = 180
CLASS_COLORS_BGR = {
    0: (66, 76, 239),    # red
    1: (236, 247, 240),  # pale near-white mint
    2: (38, 187, 245),   # amber
}


def get_approach_one_evaluation_class(raw_class_name: str) -> str:
    return APPROACH_ONE_EVALUATION_CLASS_MAP.get(raw_class_name, raw_class_name)


def get_approach_one_evaluation_class_id(raw_class_id: int) -> int:
    return APPROACH_ONE_EVALUATION_CLASS_ID_MAP.get(raw_class_id, raw_class_id)


@dataclass(slots=True)
class Tile:
    x1: int
    y1: int
    x2: int
    y2: int
    depth: int = 0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(slots=True)
class GrainDetection:
    bbox: np.ndarray
    class_id: int
    confidence: float
    mask_crop: np.ndarray | None
    mask_origin: tuple[int, int]
    source_tile: int = 0
    _mask_area: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._mask_area = int(self.mask_crop.sum()) if self.mask_crop is not None else 0

    @property
    def mask_area(self) -> int:
        return self._mask_area

    @property
    def display_class_id(self) -> int:
        """Return the corrected semantic class exposed to users."""
        return get_approach_one_evaluation_class_id(self.class_id)

    @property
    def display_class_name(self) -> str:
        return CLASS_NAMES[self.display_class_id]

    def as_dict(self, detection_id: int) -> dict[str, Any]:
        x1, y1, x2, y2 = (float(value) for value in self.bbox)
        return {
            "id": detection_id,
            "class_id": self.display_class_id,
            "class_name": self.display_class_name,
            "raw_class_id": self.class_id,
            "raw_class_name": CLASS_NAMES[self.class_id],
            "confidence": round(self.confidence, 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
        }


@dataclass(slots=True)
class AnalysisResult:
    detections: list[GrainDetection]
    annotated_image: np.ndarray
    tiles_processed: int
    saturated_tiles: int
    raw_predictions: int
    candidate_predictions: int
    elapsed_ms: float


def _axis_starts(length: int, tile_size: int, overlap_ratio: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1.0 - overlap_ratio)))
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def generate_tiles(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap_ratio: float,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    depth: int = 0,
) -> list[Tile]:
    tile_width = min(tile_size, image_width)
    tile_height = min(tile_size, image_height)
    x_starts = _axis_starts(image_width, tile_width, overlap_ratio)
    y_starts = _axis_starts(image_height, tile_height, overlap_ratio)
    return [
        Tile(
            x1=offset_x + x,
            y1=offset_y + y,
            x2=offset_x + min(x + tile_width, image_width),
            y2=offset_y + min(y + tile_height, image_height),
            depth=depth,
        )
        for y in y_starts
        for x in x_starts
    ]


def subdivide_tile(tile: Tile, overlap_ratio: float, min_tile_size: int) -> list[Tile]:
    child_size = max(
        min_tile_size,
        math.ceil(max(tile.width, tile.height) * (0.5 + overlap_ratio / 2.0)),
    )
    if child_size >= max(tile.width, tile.height):
        return []
    return generate_tiles(
        tile.width,
        tile.height,
        child_size,
        overlap_ratio,
        offset_x=tile.x1,
        offset_y=tile.y1,
        depth=tile.depth + 1,
    )


def _extract_detections(
    raw_detections: Any,
    tile: Tile,
    source_tile: int,
) -> list[GrainDetection]:
    masks = getattr(raw_detections, "mask", None)
    extracted: list[GrainDetection] = []

    for index in range(len(raw_detections)):
        local_box = np.asarray(raw_detections.xyxy[index], dtype=np.float32)
        local_box[[0, 2]] = np.clip(local_box[[0, 2]], 0, tile.width)
        local_box[[1, 3]] = np.clip(local_box[[1, 3]], 0, tile.height)

        lx1 = max(0, int(math.floor(local_box[0])))
        ly1 = max(0, int(math.floor(local_box[1])))
        lx2 = min(tile.width, int(math.ceil(local_box[2])))
        ly2 = min(tile.height, int(math.ceil(local_box[3])))
        if lx2 <= lx1 or ly2 <= ly1:
            continue

        global_box = local_box.copy()
        global_box[[0, 2]] += tile.x1
        global_box[[1, 3]] += tile.y1

        mask_crop: np.ndarray | None = None
        if masks is not None:
            mask = np.asarray(masks[index], dtype=np.uint8)
            if mask.shape != (tile.height, tile.width):
                mask = cv2.resize(
                    mask,
                    (tile.width, tile.height),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask_crop = mask[ly1:ly2, lx1:lx2].astype(bool, copy=False)

        extracted.append(
            GrainDetection(
                bbox=global_box,
                class_id=int(raw_detections.class_id[index]),
                confidence=float(raw_detections.confidence[index]),
                mask_crop=mask_crop,
                mask_origin=(tile.x1 + lx1, tile.y1 + ly1),
                source_tile=source_tile,
            )
        )
    return extracted


def _bbox_intersection(first: GrainDetection, second: GrainDetection) -> float:
    x1 = max(float(first.bbox[0]), float(second.bbox[0]))
    y1 = max(float(first.bbox[1]), float(second.bbox[1]))
    x2 = min(float(first.bbox[2]), float(second.bbox[2]))
    y2 = min(float(first.bbox[3]), float(second.bbox[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_overlap(first: GrainDetection, second: GrainDetection) -> tuple[float, float]:
    intersection = _bbox_intersection(first, second)
    if intersection <= 0:
        return 0.0, 0.0
    first_area = max(1.0, float(first.bbox[2] - first.bbox[0])) * max(
        1.0, float(first.bbox[3] - first.bbox[1])
    )
    second_area = max(1.0, float(second.bbox[2] - second.bbox[0])) * max(
        1.0, float(second.bbox[3] - second.bbox[1])
    )
    union = first_area + second_area - intersection
    return intersection / max(union, 1.0), intersection / max(min(first_area, second_area), 1.0)


def _mask_overlap(first: GrainDetection, second: GrainDetection) -> tuple[float, float]:
    if first.mask_crop is None or second.mask_crop is None:
        return _bbox_overlap(first, second)

    ax, ay = first.mask_origin
    bx, by = second.mask_origin
    ax2, ay2 = ax + first.mask_crop.shape[1], ay + first.mask_crop.shape[0]
    bx2, by2 = bx + second.mask_crop.shape[1], by + second.mask_crop.shape[0]

    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0

    first_slice = first.mask_crop[y1 - ay : y2 - ay, x1 - ax : x2 - ax]
    second_slice = second.mask_crop[y1 - by : y2 - by, x1 - bx : x2 - bx]
    intersection = int(np.logical_and(first_slice, second_slice).sum())
    if intersection == 0:
        return 0.0, 0.0

    first_area = max(first.mask_area, 1)
    second_area = max(second.mask_area, 1)
    union = first_area + second_area - intersection
    return intersection / max(union, 1), intersection / max(min(first_area, second_area), 1)


def merge_duplicate_detections(
    detections: Iterable[GrainDetection],
    *,
    mask_iou_threshold: float = 0.50,
    containment_threshold: float = 0.78,
    box_iou_fallback: float = 0.65,
) -> list[GrainDetection]:
    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: list[GrainDetection] = []
    spatial_index: dict[tuple[int, int], list[int]] = {}
    spatial_cell_size = 128

    def occupied_cells(item: GrainDetection) -> list[tuple[int, int]]:
        x1, y1, x2, y2 = (int(value) for value in item.bbox)
        return [
            (grid_x, grid_y)
            for grid_y in range(y1 // spatial_cell_size, max(y1, y2 - 1) // spatial_cell_size + 1)
            for grid_x in range(x1 // spatial_cell_size, max(x1, x2 - 1) // spatial_cell_size + 1)
        ]

    for candidate in ordered:
        duplicate = False
        cells = occupied_cells(candidate)
        nearby_indices = {
            accepted_index
            for cell in cells
            for accepted_index in spatial_index.get(cell, [])
        }
        for accepted_index in nearby_indices:
            accepted = kept[accepted_index]
            # Predictions emitted by one DETR tile are separate instances. Only
            # overlapping tiles can create duplicate views of one grain.
            if candidate.source_tile == accepted.source_tile:
                continue
            if _bbox_intersection(candidate, accepted) <= 0:
                continue
            mask_iou, containment = _mask_overlap(candidate, accepted)
            box_iou, _ = _bbox_overlap(candidate, accepted)
            if (
                mask_iou >= mask_iou_threshold
                or containment >= containment_threshold
                or box_iou >= box_iou_fallback
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
            kept_index = len(kept) - 1
            for cell in cells:
                spatial_index.setdefault(cell, []).append(kept_index)

    return kept


def annotate_image(image_bgr: np.ndarray, detections: list[GrainDetection]) -> np.ndarray:
    annotated = image_bgr.copy()
    draw_labels = len(detections) <= 200

    for detection in detections:
        display_class_id = detection.display_class_id
        color = CLASS_COLORS_BGR.get(display_class_id, (255, 255, 255))
        if detection.mask_crop is not None and detection.mask_crop.any():
            x, y = detection.mask_origin
            mask_h, mask_w = detection.mask_crop.shape
            region = annotated[y : y + mask_h, x : x + mask_w]
            if region.shape[:2] == detection.mask_crop.shape:
                color_layer = np.empty_like(region)
                color_layer[:] = color
                blended = cv2.addWeighted(region, 0.58, color_layer, 0.42, 0)
                region[detection.mask_crop] = blended[detection.mask_crop]

        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        if draw_labels:
            label = f"{detection.display_class_name} {detection.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
    return annotated


def encode_jpeg_data_url(image_bgr: np.ndarray, quality: int = 90) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode the annotated image.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


class TiledGrainAnalyzer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        optimize_fp16: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        import torch
        from rfdetr import RFDETRSegLarge

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_variant = MODEL_VARIANT
        # This checkpoint was produced by the source-safe Dataset V2 Large run.
        self.model = RFDETRSegLarge.from_checkpoint(
            str(self.checkpoint_path),
            trust_checkpoint=True,
        )
        self.optimized = False
        if optimize_fp16 and self.device == "cuda":
            try:
                self.model.inference(compile=False, dtype="float16")
                self.optimized = True
            except Exception:
                # Prediction remains available through the original FP32 model.
                self.optimized = False
        self._lock = threading.Lock()

    def analyze(
        self,
        image_bgr: np.ndarray,
        *,
        threshold: float = 0.35,
        tile_size: int = 640,
        overlap_ratio: float = 0.20,
        saturation_threshold: int = DEFAULT_SATURATION_THRESHOLD,
        max_depth: int = 2,
        min_tile_size: int = 256,
    ) -> AnalysisResult:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR color image.")
        height, width = image_bgr.shape[:2]
        if height < 32 or width < 32:
            raise ValueError("Image is too small for analysis.")

        started = time.perf_counter()
        queue = generate_tiles(width, height, tile_size, overlap_ratio)
        collected: list[GrainDetection] = []
        tiles_processed = 0
        saturated_tiles = 0
        raw_predictions = 0

        with self._lock:
            while queue:
                tile = queue.pop(0)
                crop = image_bgr[tile.y1 : tile.y2, tile.x1 : tile.x2]
                # RF-DETR requires NumPy inputs in RGB order; OpenCV images are BGR.
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                raw = self.model.predict(crop_rgb, threshold=threshold)
                prediction_count = len(raw)
                tiles_processed += 1
                raw_predictions += prediction_count

                can_split = (
                    tile.depth < max_depth
                    and min(tile.width, tile.height) > min_tile_size
                )
                if prediction_count >= saturation_threshold and can_split:
                    children = subdivide_tile(tile, overlap_ratio, min_tile_size)
                    if children:
                        saturated_tiles += 1
                        queue[0:0] = children
                        continue

                collected.extend(_extract_detections(raw, tile, tiles_processed))

        merged = merge_duplicate_detections(collected)
        annotated = annotate_image(image_bgr, merged)
        return AnalysisResult(
            detections=merged,
            annotated_image=annotated,
            tiles_processed=tiles_processed,
            saturated_tiles=saturated_tiles,
            raw_predictions=raw_predictions,
            candidate_predictions=len(collected),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


def summarize_result(result: AnalysisResult) -> dict[str, Any]:
    class_counts = Counter(
        get_approach_one_evaluation_class(CLASS_NAMES[item.class_id])
        for item in result.detections
    )
    total = len(result.detections)
    counts = {
        class_name: int(class_counts.get(class_name, 0))
        for class_name in CLASS_NAMES
    }
    percentages = {
        class_name: round((count / total * 100.0) if total else 0.0, 2)
        for class_name, count in counts.items()
    }

    bad_percentage = percentages["bad seed"]
    impurity_percentage = percentages["impurity"]
    if bad_percentage <= 5.0 and impurity_percentage <= 2.0:
        quality_key, quality_label = "good", "Good"
    elif bad_percentage <= 15.0 and impurity_percentage <= 5.0:
        quality_key, quality_label = "acceptable", "Acceptable"
    else:
        quality_key, quality_label = "poor", "Poor"

    return {
        "total": total,
        "counts": counts,
        "percentages": percentages,
        "quality": {
            "key": quality_key,
            "label": quality_label,
            "is_demo_rule": True,
            "rule": "Good: bad <= 5% and impurity <= 2%; Acceptable: bad <= 15% and impurity <= 5%.",
        },
        "evaluation_class_mapping": {
            "name": "Approach 1",
            "applied_to": "all user-facing outputs and Ground Truth evaluation",
            "raw_to_evaluation": dict(APPROACH_ONE_EVALUATION_CLASS_MAP),
            "raw_fields_preserved_in_exports": True,
            "visualization_uses_corrected_classes": True,
        },
    }
