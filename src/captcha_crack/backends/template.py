import json
from pathlib import Path

import cv2
import numpy as np

from ..core.candidates import Candidate, CharacterPrediction, build_candidates
from ..core.segment import normalize_character, segment
from .base import RecognitionBackend

TEMPLATE_SIZE = 32


class TemplateBackend(RecognitionBackend):
    name = "template"
    requires_segmentation = True

    def __init__(self, template_dir: Path | str | None = None) -> None:
        self.template_dir = Path(template_dir) if template_dir else None
        self._templates: dict[str, list[np.ndarray]] = {}
        if self.template_dir and self.template_dir.exists():
            self._load_templates()

    def _load_templates(self) -> None:
        manifest = self.template_dir / "manifest.json"
        if manifest.exists():
            labels = json.loads(manifest.read_text())
        else:
            labels = {p.stem: p.stem[0] for p in self.template_dir.glob("*.png")}

        for filename, label in labels.items():
            path = self.template_dir / f"{filename}.png"
            if not path.exists():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            resized = cv2.resize(
                image, (TEMPLATE_SIZE, TEMPLATE_SIZE), interpolation=cv2.INTER_AREA
            )
            self._templates.setdefault(label, []).append(resized)

    def is_available(self) -> bool:
        return bool(self._templates)

    def _match_character(self, glyph: np.ndarray, top_n: int = 3) -> CharacterPrediction:
        scores: list[tuple[str, float]] = []

        for label, templates in self._templates.items():
            best = 0.0
            for template in templates:
                result = cv2.matchTemplate(glyph, template, cv2.TM_CCOEFF_NORMED)
                score = float(result.max())
                best = max(best, score)
            scores.append((label, best))

        scores.sort(key=lambda x: x[1], reverse=True)

        if not scores:
            return CharacterPrediction(char="?", confidence=0.0)

        top_char, top_score = scores[0]
        alternatives = scores[1 : 1 + top_n]

        return CharacterPrediction(
            char=top_char,
            confidence=max(0.0, min(1.0, top_score)),
            alternatives=alternatives,
        )

    def recognize(
        self,
        image: np.ndarray,
        expected_length: int | None = None,
        max_candidates: int = 5,
    ) -> list[Candidate]:
        if not self._templates:
            return []

        boxes = segment(image, expected_count=expected_length)
        if not boxes:
            return []

        predictions = [
            self._match_character(normalize_character(box, TEMPLATE_SIZE)) for box in boxes
        ]

        return build_candidates(predictions, max_results=max_candidates)
