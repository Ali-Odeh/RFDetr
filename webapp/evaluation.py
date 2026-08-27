from __future__ import annotations

import math
from collections import Counter
from typing import Any

import cv2
import numpy as np

from webapp.tiling import (
    CLASS_NAMES,
    GrainDetection,
    _bbox_intersection,
    _mask_overlap,
    get_approach_one_evaluation_class_id,
)


def parse_yolo_segmentation_labels(
    label_text: str,
    image_width: int,
    image_height: int,
) -> list[GrainDetection]:
    """Parse normalized YOLO polygon labels into cropped instance masks."""
    ground_truth: list[GrainDetection] = []
    for line_number, raw_line in enumerate(label_text.lstrip("\ufeff").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        tokens = line.split()
        if len(tokens) < 7 or (len(tokens) - 1) % 2:
            raise ValueError(
                f"Label line {line_number} is not YOLO Segmentation format. "
                "Expected: class_id x1 y1 x2 y2 x3 y3 ..."
            )
        try:
            class_value = float(tokens[0])
            coordinates = np.asarray([float(value) for value in tokens[1:]], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"Label line {line_number} contains a non-numeric value.") from exc

        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(
                f"Label line {line_number} has invalid class {tokens[0]}; "
                f"expected 0-{len(CLASS_NAMES) - 1}."
            )
        if not np.isfinite(coordinates).all():
            raise ValueError(f"Label line {line_number} contains NaN or infinity.")
        if np.any(coordinates < -1e-6) or np.any(coordinates > 1.0 + 1e-6):
            raise ValueError(
                f"Label line {line_number} must use normalized coordinates between 0 and 1."
            )

        points = np.clip(coordinates.reshape(-1, 2), 0.0, 1.0)
        points[:, 0] *= max(image_width - 1, 1)
        points[:, 1] *= max(image_height - 1, 1)
        polygon = np.rint(points).astype(np.int32)

        x1 = max(0, int(polygon[:, 0].min()))
        y1 = max(0, int(polygon[:, 1].min()))
        x2 = min(image_width, int(polygon[:, 0].max()) + 1)
        y2 = min(image_height, int(polygon[:, 1].max()) + 1)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Label line {line_number} produces an empty polygon.")

        local_polygon = polygon - np.asarray([x1, y1], dtype=np.int32)
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(mask, [local_polygon], 1)
        if not mask.any():
            raise ValueError(f"Label line {line_number} produces an empty mask.")

        ground_truth.append(
            GrainDetection(
                bbox=np.asarray([x1, y1, x2, y2], dtype=np.float32),
                class_id=class_id,
                confidence=1.0,
                mask_crop=mask.astype(bool),
                mask_origin=(x1, y1),
                source_tile=-1,
            )
        )
    return ground_truth


def _occupied_cells(item: GrainDetection, cell_size: int = 128) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = (int(value) for value in item.bbox)
    return [
        (grid_x, grid_y)
        for grid_y in range(y1 // cell_size, max(y1, y2 - 1) // cell_size + 1)
        for grid_x in range(x1 // cell_size, max(x1, x2 - 1) // cell_size + 1)
    ]


def evaluate_instance_segmentation(
    predictions: list[GrainDetection],
    ground_truth: list[GrainDetection],
    *,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Greedily match masks by IoU, then compute class-aware instance metrics."""
    gt_spatial_index: dict[tuple[int, int], list[int]] = {}
    for gt_index, gt_item in enumerate(ground_truth):
        for cell in _occupied_cells(gt_item):
            gt_spatial_index.setdefault(cell, []).append(gt_index)

    candidate_pairs: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        nearby_gt = {
            gt_index
            for cell in _occupied_cells(prediction)
            for gt_index in gt_spatial_index.get(cell, [])
        }
        for gt_index in nearby_gt:
            target = ground_truth[gt_index]
            if _bbox_intersection(prediction, target) <= 0:
                continue
            mask_iou, _ = _mask_overlap(prediction, target)
            if mask_iou >= iou_threshold:
                candidate_pairs.append((float(mask_iou), pred_index, gt_index))

    matched_predictions: set[int] = set()
    matched_ground_truth: set[int] = set()
    matches: list[tuple[float, int, int]] = []
    for mask_iou, pred_index, gt_index in sorted(candidate_pairs, reverse=True):
        if pred_index in matched_predictions or gt_index in matched_ground_truth:
            continue
        matched_predictions.add(pred_index)
        matched_ground_truth.add(gt_index)
        matches.append((mask_iou, pred_index, gt_index))

    correct_matches = [
        match
        for match in matches
        if get_approach_one_evaluation_class_id(predictions[match[1]].class_id)
        == ground_truth[match[2]].class_id
    ]
    true_positive = len(correct_matches)
    false_positive = len(predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    precision = true_positive / (true_positive + false_positive) if predictions else 0.0
    recall = true_positive / (true_positive + false_negative) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    class_accuracy = true_positive / len(matches) if matches else 0.0

    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for _, pred_index, gt_index in matches:
        gt_class = ground_truth[gt_index].class_id
        pred_class = get_approach_one_evaluation_class_id(
            predictions[pred_index].class_id
        )
        confusion[gt_class][pred_class] += 1

    per_class: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_tp_matches = [
            match
            for match in correct_matches
            if get_approach_one_evaluation_class_id(
                predictions[match[1]].class_id
            )
            == class_id
        ]
        class_tp = len(class_tp_matches)
        predicted_count = sum(
            get_approach_one_evaluation_class_id(item.class_id) == class_id
            for item in predictions
        )
        gt_count = sum(item.class_id == class_id for item in ground_truth)
        class_precision = class_tp / predicted_count if predicted_count else 0.0
        class_recall = class_tp / gt_count if gt_count else 0.0
        class_f1 = (
            2 * class_precision * class_recall / (class_precision + class_recall)
            if class_precision + class_recall
            else 0.0
        )
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "ground_truth": gt_count,
                "predictions": predicted_count,
                "true_positive": class_tp,
                "precision": round(class_precision, 6),
                "recall": round(class_recall, 6),
                "f1": round(class_f1, 6),
                "mean_iou": round(
                    float(np.mean([match[0] for match in class_tp_matches])), 6
                )
                if class_tp_matches
                else 0.0,
            }
        )

    gt_counts = Counter(item.class_id for item in ground_truth)
    return {
        "iou_threshold": round(iou_threshold, 2),
        "ground_truth_total": len(ground_truth),
        "ground_truth_counts": {
            CLASS_NAMES[class_id]: int(gt_counts.get(class_id, 0))
            for class_id in range(len(CLASS_NAMES))
        },
        "predictions_total": len(predictions),
        "spatial_matches": len(matches),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round(class_accuracy, 6),
        "matched_class_accuracy": round(class_accuracy, 6),
        "mean_mask_iou": round(float(np.mean([match[0] for match in matches])), 6)
        if matches
        else 0.0,
        "mean_correct_mask_iou": round(
            float(np.mean([match[0] for match in correct_matches])), 6
        )
        if correct_matches
        else 0.0,
        "per_class": per_class,
        "evaluation_class_mapping": {
            "name": "Approach 1",
            "raw_prediction_id_to_evaluation_id": {"0": 1, "1": 0, "2": 2},
            "ground_truth_order_unchanged": True,
            "visualization_uses_corrected_classes": True,
        },
        "confusion_matrix": {
            "labels": list(CLASS_NAMES),
            "rows_are_ground_truth": True,
            "values": confusion,
        },
        "note": (
            "Accuracy is classification accuracy among spatially matched instances; "
            "object detection/segmentation quality is represented by IoU, precision, recall, and F1."
        ),
    }
