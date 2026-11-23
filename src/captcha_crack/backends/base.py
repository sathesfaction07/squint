from abc import ABC, abstractmethod

import numpy as np

from ..core.candidates import Candidate


class RecognitionBackend(ABC):
    name: str = "base"
    requires_segmentation: bool = False

    @abstractmethod
    def recognize(
        self,
        image: np.ndarray,
        expected_length: int | None = None,
        max_candidates: int = 5,
    ) -> list[Candidate]:
        ...

    def is_available(self) -> bool:
        return True
