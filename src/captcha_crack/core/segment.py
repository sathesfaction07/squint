from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CharacterBox:
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2


def find_contour_boxes(binary: np.ndarray, min_area: int = 20) -> list[CharacterBox]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        boxes.append(CharacterBox(x=x, y=y, width=w, height=h, image=binary[y : y + h, x : x + w]))

    return sorted(boxes, key=lambda b: b.x)


def vertical_projection(binary: np.ndarray) -> np.ndarray:
    return np.sum(binary > 0, axis=0)


def find_valleys(projection: np.ndarray, threshold_ratio: float = 0.15) -> list[int]:
    if projection.max() == 0:
        return []

    threshold = projection.max() * threshold_ratio
    valleys = []
    in_valley = False
    valley_start = 0

    for i, value in enumerate(projection):
        if value <= threshold and not in_valley:
            in_valley = True
            valley_start = i
        elif value > threshold and in_valley:
            in_valley = False
            valleys.append((valley_start + i) // 2)

    return valleys


def split_wide_box(box: CharacterBox, expected_width: int) -> list[CharacterBox]:
    if box.width <= expected_width * 1.4:
        return [box]

    estimated_count = max(2, round(box.width / expected_width))
    projection = vertical_projection(box.image)
    valleys = find_valleys(projection)

    if len(valleys) >= estimated_count - 1:
        cuts = sorted(valleys)[: estimated_count - 1]
    else:
        step = box.width // estimated_count
        cuts = [step * (i + 1) for i in range(estimated_count - 1)]

    pieces = []
    boundaries = [0] + cuts + [box.width]

    for start, end in zip(boundaries[:-1], boundaries[1:], strict=False):
        if end - start < 3:
            continue
        pieces.append(
            CharacterBox(
                x=box.x + start,
                y=box.y,
                width=end - start,
                height=box.height,
                image=box.image[:, start:end],
            )
        )

    return pieces if pieces else [box]


def merge_fragments(boxes: list[CharacterBox], max_gap: int = 2) -> list[CharacterBox]:
    if not boxes:
        return []

    merged = [boxes[0]]

    for box in boxes[1:]:
        previous = merged[-1]
        gap = box.x - (previous.x + previous.width)

        overlapping = gap < max_gap
        both_thin = previous.width < 6 and box.width < 6

        if overlapping and both_thin:
            x_start = min(previous.x, box.x)
            y_start = min(previous.y, box.y)
            x_end = max(previous.x + previous.width, box.x + box.width)
            y_end = max(previous.y + previous.height, box.y + box.height)
            merged[-1] = CharacterBox(
                x=x_start,
                y=y_start,
                width=x_end - x_start,
                height=y_end - y_start,
                image=np.zeros((y_end - y_start, x_end - x_start), dtype=np.uint8),
            )
        else:
            merged.append(box)

    return merged


def segment(
    binary: np.ndarray,
    expected_count: int | None = None,
    min_area: int = 20,
) -> list[CharacterBox]:
    boxes = find_contour_boxes(binary, min_area=min_area)

    if not boxes:
        return []

    boxes = merge_fragments(boxes)

    if expected_count and len(boxes) < expected_count:
        median_width = int(np.median([b.width for b in boxes]))
        expanded: list[CharacterBox] = []
        for box in boxes:
            expanded.extend(split_wide_box(box, median_width))
        boxes = sorted(expanded, key=lambda b: b.x)

    if expected_count and len(boxes) > expected_count:
        boxes = sorted(boxes, key=lambda b: b.area, reverse=True)[:expected_count]
        boxes = sorted(boxes, key=lambda b: b.x)

    return boxes


def normalize_character(box: CharacterBox, size: int = 32) -> np.ndarray:
    if box.image.size == 0:
        return np.zeros((size, size), dtype=np.uint8)

    height, width = box.image.shape[:2]
    scale = (size - 4) / max(height, width)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(box.image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - new_height) // 2
    x_offset = (size - new_width) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized

    return canvas
