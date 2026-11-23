import numpy as np

from captcha_crack.core.preprocess import (
    clean_speckles,
    denoise,
    deskew,
    find_skew_angle,
    preprocess,
    remove_lines,
    threshold,
    to_grayscale,
)


class TestGrayscale:
    def test_converts_color_to_single_channel(self, text_image):
        result = to_grayscale(text_image)
        assert result.ndim == 2

    def test_passes_through_existing_grayscale(self, binary_chars):
        result = to_grayscale(binary_chars)
        assert result.shape == binary_chars.shape


class TestDenoise:
    def test_preserves_shape_on_color(self, noisy_image):
        result = denoise(noisy_image)
        assert result.shape == noisy_image.shape

    def test_reduces_variance(self, noisy_image):
        result = denoise(noisy_image)
        assert result.std() <= noisy_image.std()

    def test_handles_grayscale(self, binary_chars):
        result = denoise(binary_chars)
        assert result.shape == binary_chars.shape


class TestThreshold:
    def test_adaptive_produces_binary(self, text_image):
        gray = to_grayscale(text_image)
        result = threshold(gray, adaptive=True)
        assert set(np.unique(result)).issubset({0, 255})

    def test_otsu_produces_binary(self, text_image):
        gray = to_grayscale(text_image)
        result = threshold(gray, adaptive=False)
        assert set(np.unique(result)).issubset({0, 255})


class TestDeskew:
    def test_returns_zero_angle_for_empty(self):
        empty = np.zeros((40, 100), dtype=np.uint8)
        assert find_skew_angle(empty) == 0.0

    def test_skips_rotation_below_threshold(self, binary_chars):
        result, angle = deskew(binary_chars, angle=0.2)
        assert angle == 0.0
        assert np.array_equal(result, binary_chars)

    def test_rotates_above_threshold(self, binary_chars):
        result, angle = deskew(binary_chars, angle=10.0)
        assert angle == 10.0
        assert result.shape == binary_chars.shape


class TestRemoveLines:
    def test_removes_horizontal_line(self):
        image = np.zeros((40, 100), dtype=np.uint8)
        image[20, :] = 255
        result = remove_lines(image, min_length=20)
        assert result.sum() < image.sum()

    def test_preserves_blobs(self, binary_chars):
        result = remove_lines(binary_chars, min_length=40)
        assert result.sum() > 0


class TestCleanSpeckles:
    def test_removes_small_components(self):
        image = np.zeros((40, 100), dtype=np.uint8)
        image[5, 5] = 255
        image[10:30, 40:60] = 255
        result = clean_speckles(image, min_area=8)
        assert result[5, 5] == 0
        assert result[20, 50] == 255

    def test_keeps_all_above_threshold(self, binary_chars):
        result = clean_speckles(binary_chars, min_area=8)
        assert result.sum() == binary_chars.sum()


class TestPipeline:
    def test_returns_binary_image(self, text_image):
        result = preprocess(text_image)
        assert set(np.unique(result.image)).issubset({0, 255})

    def test_stages_empty_by_default(self, text_image):
        result = preprocess(text_image)
        assert result.stages == {}

    def test_stages_captured_on_request(self, text_image):
        result = preprocess(text_image, keep_stages=True)
        assert "grayscale" in result.stages
        assert "threshold" in result.stages
        assert "deskewed" in result.stages

    def test_handles_blank_image(self, blank_image):
        result = preprocess(blank_image)
        assert result.image.shape[:2] == blank_image.shape[:2]

    def test_line_removal_toggleable(self, lined_image):
        with_lines = preprocess(lined_image, strip_lines=False)
        without_lines = preprocess(lined_image, strip_lines=True)
        assert without_lines.image.sum() <= with_lines.image.sum()

    def test_detects_rotation(self, rotated_image):
        result = preprocess(rotated_image)
        assert isinstance(result.deskew_angle, float)
