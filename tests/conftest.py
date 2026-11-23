import cv2
import numpy as np
import pytest

from captcha_crack.config import Settings


@pytest.fixture
def blank_image() -> np.ndarray:
    return np.full((60, 200, 3), 255, dtype=np.uint8)


@pytest.fixture
def text_image() -> np.ndarray:
    image = np.full((60, 200, 3), 255, dtype=np.uint8)
    cv2.putText(image, "AB12", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    return image


@pytest.fixture
def noisy_image(text_image: np.ndarray) -> np.ndarray:
    image = text_image.copy()
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 60, image.shape, dtype=np.uint8)
    return cv2.subtract(image, noise)


@pytest.fixture
def lined_image(text_image: np.ndarray) -> np.ndarray:
    image = text_image.copy()
    cv2.line(image, (0, 30), (200, 30), (0, 0, 0), 2)
    cv2.line(image, (0, 15), (200, 50), (0, 0, 0), 1)
    return image


@pytest.fixture
def rotated_image(text_image: np.ndarray) -> np.ndarray:
    height, width = text_image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), 8, 1.0)
    return cv2.warpAffine(text_image, matrix, (width, height), borderValue=(255, 255, 255))


@pytest.fixture
def binary_chars() -> np.ndarray:
    image = np.zeros((40, 120), dtype=np.uint8)
    for index in range(4):
        x = 10 + index * 26
        cv2.rectangle(image, (x, 8), (x + 14, 32), 255, -1)
    return image


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(backend="template", template_dir=tmp_path / "templates", max_candidates=5)
