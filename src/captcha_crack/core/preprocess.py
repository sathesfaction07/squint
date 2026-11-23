from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    image: np.ndarray
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    deskew_angle: float = 0.0


def denoise(image: np.ndarray, strength: int = 9) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.bilateralFilter(image, strength, 75, 75)
    return cv2.medianBlur(image, 3)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def threshold(image: np.ndarray, adaptive: bool = True) -> np.ndarray:
    if adaptive:
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def find_skew_angle(binary: np.ndarray) -> float:
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return 0.0

    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[2]

    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    return float(angle)


def deskew(binary: np.ndarray, angle: float | None = None) -> tuple[np.ndarray, float]:
    if angle is None:
        angle = find_skew_angle(binary)

    if abs(angle) < 0.5:
        return binary, 0.0

    height, width = binary.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def remove_lines(binary: np.ndarray, min_length: int = 20) -> np.ndarray:
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    lines = cv2.bitwise_or(horizontal, vertical)
    return cv2.subtract(binary, lines)


def clean_speckles(binary: np.ndarray, min_area: int = 8) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)

    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def preprocess(
    image: np.ndarray,
    keep_stages: bool = False,
    adaptive_threshold: bool = True,
    strip_lines: bool = True,
) -> PreprocessResult:
    stages: dict[str, np.ndarray] = {}

    denoised = denoise(image)
    if keep_stages:
        stages["denoised"] = denoised.copy()

    gray = to_grayscale(denoised)
    if keep_stages:
        stages["grayscale"] = gray.copy()

    binary = threshold(gray, adaptive=adaptive_threshold)
    if keep_stages:
        stages["threshold"] = binary.copy()

    if strip_lines:
        binary = remove_lines(binary)
        if keep_stages:
            stages["lines_removed"] = binary.copy()

    binary = clean_speckles(binary)
    if keep_stages:
        stages["despeckled"] = binary.copy()

    deskewed, angle = deskew(binary)
    if keep_stages:
        stages["deskewed"] = deskewed.copy()

    return PreprocessResult(image=deskewed, stages=stages, deskew_angle=angle)
