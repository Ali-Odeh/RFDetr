import unittest

import numpy as np

from webapp.tiling import GrainDetection, generate_tiles, merge_duplicate_detections


def detection(box, confidence=0.9, class_id=0, mask=None, source_tile=0):
    x1, y1, x2, y2 = box
    if mask is None:
        mask = np.ones((y2 - y1, x2 - x1), dtype=bool)
    return GrainDetection(
        bbox=np.asarray(box, dtype=np.float32),
        class_id=class_id,
        confidence=confidence,
        mask_crop=mask,
        mask_origin=(x1, y1),
        source_tile=source_tile,
    )


class TileGenerationTests(unittest.TestCase):
    def test_grid_covers_full_image(self):
        tiles = generate_tiles(1920, 1080, 640, 0.20)
        self.assertGreater(len(tiles), 1)
        coverage = np.zeros((1080, 1920), dtype=bool)
        for tile in tiles:
            coverage[tile.y1 : tile.y2, tile.x1 : tile.x2] = True
        self.assertTrue(coverage.all())

    def test_small_image_uses_one_tile(self):
        tiles = generate_tiles(320, 240, 640, 0.20)
        self.assertEqual(len(tiles), 1)
        self.assertEqual((tiles[0].x2, tiles[0].y2), (320, 240))


class DuplicateMergingTests(unittest.TestCase):
    def test_duplicate_masks_keep_highest_confidence(self):
        first = detection((100, 100, 150, 150), confidence=0.95, source_tile=1)
        duplicate = detection((102, 101, 152, 151), confidence=0.70, source_tile=2)
        merged = merge_duplicate_detections([duplicate, first])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].confidence, 0.95)

    def test_neighboring_grains_are_kept(self):
        first = detection((100, 100, 150, 150), confidence=0.95)
        neighbor = detection((145, 100, 195, 150), confidence=0.90)
        merged = merge_duplicate_detections([first, neighbor])
        self.assertEqual(len(merged), 2)

    def test_cross_class_duplicate_is_one_physical_instance(self):
        first = detection((20, 20, 60, 60), confidence=0.91, class_id=1, source_tile=1)
        duplicate = detection((20, 20, 60, 60), confidence=0.80, class_id=0, source_tile=2)
        merged = merge_duplicate_detections([duplicate, first])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].class_id, 1)

    def test_touching_instances_from_same_tile_are_never_merged(self):
        first = detection((20, 20, 60, 60), confidence=0.91, source_tile=7)
        second = detection((20, 20, 60, 60), confidence=0.80, source_tile=7)
        merged = merge_duplicate_detections([first, second])
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
