import unittest

import numpy as np

from webapp.evaluation import evaluate_instance_segmentation
from webapp.tiling import (
    AnalysisResult,
    CLASS_COLORS_BGR,
    GrainDetection,
    annotate_image,
    get_approach_one_evaluation_class,
    summarize_result,
)


def detection(class_id: int, index: int) -> GrainDetection:
    x = index * 12
    mask = np.ones((8, 8), dtype=bool)
    return GrainDetection(
        bbox=np.asarray([x, 0, x + 8, 8], dtype=np.float32),
        class_id=class_id,
        confidence=1.0,
        mask_crop=mask,
        mask_origin=(x, 0),
        source_tile=index,
    )


class ApproachOneMappingTests(unittest.TestCase):
    def test_name_mapping_is_fixed(self) -> None:
        self.assertEqual(get_approach_one_evaluation_class("bad seed"), "healthy seed")
        self.assertEqual(get_approach_one_evaluation_class("healthy seed"), "bad seed")
        self.assertEqual(get_approach_one_evaluation_class("impurity"), "impurity")

    def test_all_user_facing_classes_are_mapped_and_raw_fields_are_preserved(self) -> None:
        detections = [detection(0, 0), detection(0, 1), detection(1, 2), detection(2, 3)]
        result = AnalysisResult(
            detections=detections,
            annotated_image=np.zeros((8, 44, 3), dtype=np.uint8),
            tiles_processed=1,
            saturated_tiles=0,
            raw_predictions=4,
            candidate_predictions=4,
            elapsed_ms=1.0,
        )

        summary = summarize_result(result)

        self.assertEqual(summary["counts"], {"bad seed": 1, "healthy seed": 2, "impurity": 1})
        serialized = detections[0].as_dict(1)
        self.assertEqual(serialized["class_id"], 1)
        self.assertEqual(serialized["class_name"], "healthy seed")
        self.assertEqual(serialized["raw_class_id"], 0)
        self.assertEqual(serialized["raw_class_name"], "bad seed")

    def test_visualization_uses_mapped_color(self) -> None:
        raw_bad = detection(0, 0)
        annotated = annotate_image(np.zeros((12, 12, 3), dtype=np.uint8), [raw_bad])

        self.assertTrue(np.array_equal(annotated[0, 0], CLASS_COLORS_BGR[1]))

    def test_accuracy_uses_mapped_predictions_and_original_ground_truth(self) -> None:
        raw_predictions = [detection(0, 0), detection(1, 1), detection(2, 2)]
        ground_truth = [detection(1, 0), detection(0, 1), detection(2, 2)]

        evaluation = evaluate_instance_segmentation(raw_predictions, ground_truth)

        self.assertEqual(evaluation["accuracy"], 1.0)
        self.assertEqual(evaluation["true_positive"], 3)
        self.assertEqual(
            [item["predictions"] for item in evaluation["per_class"]],
            [1, 1, 1],
        )


if __name__ == "__main__":
    unittest.main()
