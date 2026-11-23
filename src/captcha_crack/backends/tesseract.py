import numpy as np

from ..core.candidates import Candidate, CharacterPrediction, build_candidates
from .base import RecognitionBackend

DEFAULT_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


class TesseractBackend(RecognitionBackend):
    name = "tesseract"

    def __init__(self, whitelist: str = DEFAULT_WHITELIST, psm: int = 8) -> None:
        self.whitelist = whitelist
        self.psm = psm
        self._tesseract = None

    def _get_tesseract(self):
        if self._tesseract is None:
            import pytesseract

            self._tesseract = pytesseract
        return self._tesseract

    def is_available(self) -> bool:
        try:
            tesseract = self._get_tesseract()
            tesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def recognize(
        self,
        image: np.ndarray,
        expected_length: int | None = None,
        max_candidates: int = 5,
    ) -> list[Candidate]:
        tesseract = self._get_tesseract()

        config = f"--psm {self.psm} -c tessedit_char_whitelist={self.whitelist}"

        try:
            data = tesseract.image_to_data(
                image,
                config=config,
                output_type=tesseract.Output.DICT,
            )
        except Exception:
            return []

        predictions: list[CharacterPrediction] = []
        for text, conf in zip(data.get("text", []), data.get("conf", []), strict=False):
            cleaned = text.strip()
            if not cleaned:
                continue
            confidence = max(0.0, float(conf) / 100.0)
            for char in cleaned:
                predictions.append(CharacterPrediction(char=char, confidence=confidence))

        if not predictions:
            return []

        if expected_length and len(predictions) != expected_length:
            text = "".join(p.char for p in predictions)
            confidence = sum(p.confidence for p in predictions) / len(predictions)
            penalty = 0.75
            return [Candidate(text=text, confidence=confidence * penalty)]

        return build_candidates(predictions, max_results=max_candidates)
