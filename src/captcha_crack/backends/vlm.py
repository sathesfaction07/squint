import base64
import json
import re

import cv2
import numpy as np

from ..core.candidates import Candidate
from .base import RecognitionBackend

PROMPT = """You are reading a CAPTCHA image. Return ONLY valid JSON, no other text.

Rules:
1. Report exactly what characters you see, character by character.
2. If a character is genuinely ambiguous (could be O or 0, l or 1, 5 or S), list alternatives.
3. Never guess wildly. If you cannot read a character, use "?" for it.
4. Confidence is 0.0-1.0 reflecting how certain you are of the whole string.
5. Provide up to {max_candidates} candidate readings, best first.
{length_hint}

Output format:
{{"candidates": [{{"text": "ABC123", "confidence": 0.95}},
{{"text": "ABC1Z3", "confidence": 0.61}}]}}"""


def _encode_png(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image as PNG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _parse_response(raw: str, max_candidates: int) -> list[Candidate]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    candidates = []
    for entry in payload.get("candidates", []):
        text = entry.get("text", "").strip()
        if not text:
            continue
        confidence = float(entry.get("confidence", 0.5))
        candidates.append(Candidate(text=text, confidence=max(0.0, min(1.0, confidence))))

    return sorted(candidates, key=lambda c: c.confidence, reverse=True)[:max_candidates]


class ClaudeVisionBackend(RecognitionBackend):
    name = "claude"

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
    ) -> None:
        self.model_id = model_id
        self.region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def is_available(self) -> bool:
        try:
            self._get_client()
            return True
        except Exception:
            return False

    def recognize(
        self,
        image: np.ndarray,
        expected_length: int | None = None,
        max_candidates: int = 5,
    ) -> list[Candidate]:
        client = self._get_client()

        length_hint = (
            f"6. The CAPTCHA has exactly {expected_length} characters."
            if expected_length
            else "6. Character count is unknown."
        )
        prompt = PROMPT.format(max_candidates=max_candidates, length_hint=length_hint)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _encode_png(image),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        try:
            response = client.invoke_model(modelId=self.model_id, body=json.dumps(body))
            payload = json.loads(response["body"].read())
            raw_text = payload["content"][0]["text"]
        except Exception:
            return []

        return _parse_response(raw_text, max_candidates)


class GeminiVisionBackend(RecognitionBackend):
    name = "gemini"

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        project: str | None = None,
        location: str = "us-central1",
    ) -> None:
        self.model_name = model_name
        self.project = project
        self.location = location
        self._model = None

    def _get_model(self):
        if self._model is None:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=self.project, location=self.location)
            self._model = GenerativeModel(self.model_name)
        return self._model

    def is_available(self) -> bool:
        try:
            self._get_model()
            return True
        except Exception:
            return False

    def recognize(
        self,
        image: np.ndarray,
        expected_length: int | None = None,
        max_candidates: int = 5,
    ) -> list[Candidate]:
        from vertexai.generative_models import Part

        model = self._get_model()

        length_hint = (
            f"6. The CAPTCHA has exactly {expected_length} characters."
            if expected_length
            else "6. Character count is unknown."
        )
        prompt = PROMPT.format(max_candidates=max_candidates, length_hint=length_hint)

        success, buffer = cv2.imencode(".png", image)
        if not success:
            return []

        try:
            response = model.generate_content(
                [
                    Part.from_data(data=buffer.tobytes(), mime_type="image/png"),
                    prompt,
                ],
                generation_config={"temperature": 0.0, "max_output_tokens": 512},
            )
            raw_text = response.text
        except Exception:
            return []

        return _parse_response(raw_text, max_candidates)


class RekognitionBackend(RecognitionBackend):
    name = "rekognition"

    def __init__(self, region: str = "us-east-1") -> None:
        self.region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("rekognition", region_name=self.region)
        return self._client

    def is_available(self) -> bool:
        try:
            self._get_client()
            return True
        except Exception:
            return False

    def recognize(
        self,
        image: np.ndarray,
        expected_length: int | None = None,
        max_candidates: int = 5,
    ) -> list[Candidate]:
        client = self._get_client()

        success, buffer = cv2.imencode(".png", image)
        if not success:
            return []

        try:
            response = client.detect_text(Image={"Bytes": buffer.tobytes()})
        except Exception:
            return []

        words = [
            (d["DetectedText"], d["Confidence"] / 100.0)
            for d in response.get("TextDetections", [])
            if d.get("Type") == "WORD"
        ]

        if not words:
            return []

        text = "".join(word for word, _ in words)
        confidence = sum(conf for _, conf in words) / len(words)

        if expected_length and len(text) != expected_length:
            confidence *= 0.7

        return [Candidate(text=text, confidence=confidence)]
