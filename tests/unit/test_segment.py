import numpy as np

from captcha_crack.core.segment import (
    CharacterBox,
    find_contour_boxes,
    find_valleys,
    merge_fragments,
    normalize_character,
    segment,
    split_wide_box,
    vertical_projection,
)


def make_box(x: int, width: int, height: int = 20) -> CharacterBox:
    return CharacterBox(
        x=x,
        y=0,
        width=width,
        height=height,
        image=np.full((height, width), 255, dtype=np.uint8),
    )


class TestCharacterBox:
    def test_area(self):
        box = make_box(0, 10, 20)
        assert box.area == 200

    def test_aspect_ratio(self):
        box = make_box(0, 10, 20)
        assert box.aspect_ratio == 0.5

    def test_aspect_ratio_guards_zero_height(self):
        box = CharacterBox(x=0, y=0, width=10, height=0, image=np.zeros((0, 10), dtype=np.uint8))
        assert box.aspect_ratio == 0.0

    def test_center_x(self):
        box = make_box(10, 20)
        assert box.center_x == 20


class TestContourBoxes:
    def test_finds_all_blobs(self, binary_chars):
        boxes = find_contour_boxes(binary_chars)
        assert len(boxes) == 4

    def test_sorted_left_to_right(self, binary_chars):
        boxes = find_contour_boxes(binary_chars)
        positions = [b.x for b in boxes]
        assert positions == sorted(positions)

    def test_filters_by_area(self, binary_chars):
        boxes = find_contour_boxes(binary_chars, min_area=100_000)
        assert boxes == []

    def test_empty_image_yields_nothing(self):
        boxes = find_contour_boxes(np.zeros((40, 100), dtype=np.uint8))
        assert boxes == []


class TestProjection:
    def test_counts_pixels_per_column(self):
        image = np.zeros((10, 5), dtype=np.uint8)
        image[:, 2] = 255
        projection = vertical_projection(image)
        assert projection[2] == 10
        assert projection[0] == 0

    def test_finds_valley_between_blobs(self):
        projection = np.array([10, 10, 0, 0, 10, 10])
        valleys = find_valleys(projection)
        assert len(valleys) == 1

    def test_no_valleys_in_flat_projection(self):
        valleys = find_valleys(np.array([10, 10, 10, 10]))
        assert valleys == []

    def test_handles_all_zero_projection(self):
        valleys = find_valleys(np.zeros(10))
        assert valleys == []


class TestSplitWideBox:
    def test_leaves_normal_box_intact(self):
        box = make_box(0, 20)
        assert split_wide_box(box, expected_width=20) == [box]

    def test_splits_double_width_box(self):
        box = make_box(0, 44)
        pieces = split_wide_box(box, expected_width=20)
        assert len(pieces) == 2

    def test_preserves_absolute_position(self):
        box = make_box(100, 44)
        pieces = split_wide_box(box, expected_width=20)
        assert pieces[0].x == 100

    def test_splits_triple_width_box(self):
        box = make_box(0, 66)
        pieces = split_wide_box(box, expected_width=20)
        assert len(pieces) == 3


class TestMergeFragments:
    def test_empty_input(self):
        assert merge_fragments([]) == []

    def test_single_box_passthrough(self):
        boxes = [make_box(0, 20)]
        assert len(merge_fragments(boxes)) == 1

    def test_merges_adjacent_thin_boxes(self):
        boxes = [make_box(0, 4), make_box(5, 4)]
        merged = merge_fragments(boxes, max_gap=2)
        assert len(merged) == 1

    def test_keeps_separated_boxes(self):
        boxes = [make_box(0, 4), make_box(30, 4)]
        merged = merge_fragments(boxes, max_gap=2)
        assert len(merged) == 2

    def test_keeps_wide_adjacent_boxes(self):
        boxes = [make_box(0, 20), make_box(21, 20)]
        merged = merge_fragments(boxes, max_gap=2)
        assert len(merged) == 2


class TestSegment:
    def test_finds_expected_count(self, binary_chars):
        boxes = segment(binary_chars, expected_count=4)
        assert len(boxes) == 4

    def test_works_without_expected_count(self, binary_chars):
        boxes = segment(binary_chars)
        assert len(boxes) == 4

    def test_trims_excess_boxes(self, binary_chars):
        boxes = segment(binary_chars, expected_count=2)
        assert len(boxes) == 2

    def test_empty_image_yields_nothing(self):
        assert segment(np.zeros((40, 100), dtype=np.uint8)) == []

    def test_output_ordered_by_position(self, binary_chars):
        boxes = segment(binary_chars, expected_count=4)
        positions = [b.x for b in boxes]
        assert positions == sorted(positions)


class TestNormalize:
    def test_produces_square_output(self):
        box = make_box(0, 10, 30)
        result = normalize_character(box, size=32)
        assert result.shape == (32, 32)

    def test_handles_empty_image(self):
        box = CharacterBox(x=0, y=0, width=0, height=0, image=np.zeros((0, 0), dtype=np.uint8))
        result = normalize_character(box, size=32)
        assert result.shape == (32, 32)
        assert result.sum() == 0

    def test_centers_content(self):
        box = make_box(0, 8, 8)
        result = normalize_character(box, size=32)
        assert result[16, 16] > 0

    def test_respects_custom_size(self):
        box = make_box(0, 10, 10)
        result = normalize_character(box, size=48)
        assert result.shape == (48, 48)
