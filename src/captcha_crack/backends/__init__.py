from .base import RecognitionBackend
from .template import TemplateBackend
from .tesseract import TesseractBackend
from .vlm import ClaudeVisionBackend, GeminiVisionBackend, RekognitionBackend

BACKENDS: dict[str, type[RecognitionBackend]] = {
    "template": TemplateBackend,
    "tesseract": TesseractBackend,
    "claude": ClaudeVisionBackend,
    "gemini": GeminiVisionBackend,
    "rekognition": RekognitionBackend,
}

LOCAL_BACKENDS = ("template", "tesseract")
CLOUD_BACKENDS = ("claude", "gemini", "rekognition")


def get_backend(name: str, **kwargs) -> RecognitionBackend:
    cls = BACKENDS.get(name)
    if not cls:
        available = ", ".join(BACKENDS.keys())
        raise ValueError(f"Unknown backend: {name}. Available: {available}")
    return cls(**kwargs)


__all__ = [
    "BACKENDS",
    "CLOUD_BACKENDS",
    "LOCAL_BACKENDS",
    "ClaudeVisionBackend",
    "GeminiVisionBackend",
    "RecognitionBackend",
    "RekognitionBackend",
    "TemplateBackend",
    "TesseractBackend",
    "get_backend",
]
