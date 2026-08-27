import unittest

import numpy as np

from webapp.tiling import (
    EdgeFragment,
    GrainDetection,
    build_seam_recaptures,
    filter_image_border_detections,
    filter_tile_detections,
    generate_tiles,
    merge_duplicate_detections,
    select_seam_recoveries,
)


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

    def test_ownership_regions_cover_each_pixel_exactly_once(self):
        width, height = 192, 108
        tiles = generate_tiles(width, height, 64, 0.20)
        for y in range(height):
            for x in range(width):
                owners = sum(tile.owns(x + 0.5, y + 0.5) for tile in tiles)
                self.assertEqual(owners, 1)

    def test_overlap_prediction_is_kept_by_one_owner_tile(self):
        tiles = generate_tiles(1152, 640, 640, 0.20)
        first_view = detection((580, 100, 620, 140), source_tile=1)
        second_view = detection((580, 100, 620, 140), source_tile=2)

        first_kept, first_edge, first_non_owner = filter_tile_detections(
            [first_view], tiles[0], image_width=1152, image_height=640
        )
        second_kept, second_edge, second_non_owner = filter_tile_detections(
            [second_view], tiles[1], image_width=1152, image_height=640
        )

        self.assertEqual(len(first_kept) + len(second_kept), 1)
        self.assertEqual(first_edge + second_edge, 0)
        self.assertEqual(first_non_owner + second_non_owner, 1)

    def test_internal_tile_edge_fragment_is_rejected(self):
        first_tile = generate_tiles(1152, 640, 640, 0.20)[0]
        fragment = detection((630, 100, 640, 140), source_tile=1)

        kept, edge_filtered, ownership_filtered = filter_tile_detections(
            [fragment], first_tile, image_width=1152, image_height=640
        )

        self.assertEqual(kept, [])
        self.assertEqual(edge_filtered, 1)
        self.assertEqual(ownership_filtered, 0)

    def test_image_border_margin_ignores_partial_grains(self):
        border_grain = detection((5, 100, 30, 130), source_tile=1)
        inner_grain = detection((20, 100, 45, 130), source_tile=1)

        kept, ignored = filter_image_border_detections(
            [border_grain, inner_grain],
            image_width=200,
            image_height=200,
            margin=10,
        )

        self.assertEqual(len(kept), 1)
        self.assertIs(kept[0], inner_grain)
        self.assertEqual(ignored, 1)


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


class SeamRecoveryTests(unittest.TestCase):
    def test_recapture_crop_is_centered_across_the_cut_seam(self):
        fragment = EdgeFragment(
            detection=detection((85, 90, 100, 115), source_tile=1),
            seam_x=100,
        )

        recaptures = build_seam_recaptures(
            [fragment], image_width=200, image_height=200, crop_size=100
        )

        self.assertEqual(len(recaptures), 1)
        tile = recaptures[0].tile
        self.assertLess(tile.x1, 100)
        self.assertGreater(tile.x2, 100)
        self.assertIs(recaptures[0].fragments[0], fragment)

    def test_complete_second_pass_mask_replaces_partial_fragment(self):
        fragment = EdgeFragment(
            detection=detection((85, 90, 100, 115), source_tile=1),
            seam_x=100,
        )
        recapture = build_seam_recaptures(
            [fragment], image_width=200, image_height=200, crop_size=100
        )[0]
        complete = detection((85, 90, 115, 115), confidence=0.95, source_tile=2)
        unrelated = detection((130, 130, 150, 150), source_tile=2)

        recovered = select_seam_recoveries([complete, unrelated], recapture)

        self.assertEqual(len(recovered), 1)
        self.assertIs(recovered[0], complete)

if __name__ == "__main__":
    unittest.main()
