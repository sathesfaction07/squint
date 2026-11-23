from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..backends import get_backend
from ..config import Settings
from .candidates import Candidate, merge_candidate_lists
from .preprocess import preprocess


@dataclass
class SolveResult:
    source: str
    candidates: list[Candidate] = field(default_factory=list)
    backend_used: str = ""
    backends_attempted: list[str] = field(default_factory=list)
    deskew_angle: float = 0.0
    error: str | None = None

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def ok(self) -> bool:
        return bool(self.candidates) and self.error is None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "best": self.best.text if self.best else None,
            "confidence": round(self.best.confidence, 4) if self.best else 0.0,
            "candidates": [
                {"text": c.text, "confidence": round(c.confidence, 4)} for c in self.candidates
            ],
            "backend_used": self.backend_used,
            "backends_attempted": self.backends_attempted,
            "deskew_angle": round(self.deskew_angle, 2),
            "error": self.error,
        }


def load_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


class Pipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._backend_cache: dict[str, object] = {}

    def _backend(self, name: str):
        if name not in self._backend_cache:
            kwargs = self.settings.backend_kwargs(name)
            self._backend_cache[name] = get_backend(name, **kwargs)
        return self._backend_cache[name]

    def _chain(self) -> list[str]:
        chain = [self.settings.backend]
        chain.extend(b for b in self.settings.fallback_list if b not in chain)
        return chain

    def solve_image(
        self,
        image: np.ndarray,
        source: str = "<memory>",
        expected_length: int | None = None,
    ) -> SolveResult:
        length = expected_length or self.settings.expected_length
        result = SolveResult(source=source)

        processed = preprocess(
            image,
            adaptive_threshold=self.settings.adaptive_threshold,
            strip_lines=self.settings.strip_lines,
        )
        result.deskew_angle = processed.deskew_angle

        if self.settings.hybrid:
            return self._solve_hybrid(image, processed.image, result, length)

        for name in self._chain():
            result.backends_attempted.append(name)
            try:
                backend = self._backend(name)
            except ValueError as exc:
                result.error = str(exc)
                return result

            if not backend.is_available():
                continue

            source_image = processed.image if backend.requires_segmentation else image
            candidates = backend.recognize(
                source_image,
                expected_length=length,
                max_candidates=self.settings.max_candidates,
            )

            if candidates and candidates[0].confidence >= self.settings.confidence_floor:
                result.candidates = candidates
                result.backend_used = name
                return result

            if candidates and not result.candidates:
                result.candidates = candidates
                result.backend_used = name

        if not result.candidates:
            result.error = "No backend produced a reading"

        return result

    def _solve_hybrid(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        result: SolveResult,
        expected_length: int | None,
    ) -> SolveResult:
        collected: list[list[Candidate]] = []
        weights: list[float] = []

        for name in self._chain():
            result.backends_attempted.append(name)
            try:
                backend = self._backend(name)
            except ValueError:
                continue

            if not backend.is_available():
                continue

            source_image = processed if backend.requires_segmentation else original
            candidates = backend.recognize(
                source_image,
                expected_length=expected_length,
                max_candidates=self.settings.max_candidates,
            )

            if candidates:
                collected.append(candidates)
                weights.append(2.0 if name in ("claude", "gemini") else 1.0)

        if not collected:
            result.error = "No backend produced a reading"
            return result

        result.candidates = merge_candidate_lists(
            collected, weights=weights, max_results=self.settings.max_candidates
        )
        result.backend_used = "hybrid(" + "+".join(result.backends_attempted) + ")"
        return result

    def solve_file(self, path: Path, expected_length: int | None = None) -> SolveResult:
        image = load_image(path)
        if image is None:
            return SolveResult(source=str(path), error=f"Cannot read image: {path}")
        return self.solve_image(image, source=str(path), expected_length=expected_length)

    def solve_batch(
        self,
        paths: list[Path],
        expected_length: int | None = None,
    ) -> list[SolveResult]:
        results = []
        for path in paths:
            try:
                results.append(self.solve_file(path, expected_length=expected_length))
            except Exception as exc:
                results.append(SolveResult(source=str(path), error=f"{type(exc).__name__}: {exc}"))
        return results
