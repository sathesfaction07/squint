
from captcha_crack.backends.base import RecognitionBackend
from captcha_crack.config import Settings
from captcha_crack.core.candidates import Candidate
from captcha_crack.core.pipeline import Pipeline, SolveResult, load_image


class StubBackend(RecognitionBackend):
    name = "stub"

    def __init__(self, text: str = "ABCD", confidence: float = 0.9, available: bool = True) -> None:
        self.text = text
        self.confidence = confidence
        self.available = available
        self.calls = 0

    def is_available(self) -> bool:
        return self.available

    def recognize(self, image, expected_length=None, max_candidates=5):
        self.calls += 1
        if not self.text:
            return []
        return [Candidate(text=self.text, confidence=self.confidence)]


class TestSolveResult:
    def test_best_returns_first_candidate(self):
        result = SolveResult(
            source="x",
            candidates=[Candidate("AB", 0.9), Candidate("A8", 0.5)],
        )
        assert result.best.text == "AB"

    def test_best_none_when_empty(self):
        assert SolveResult(source="x").best is None

    def test_ok_requires_candidates(self):
        assert SolveResult(source="x").ok is False

    def test_ok_false_when_error_present(self):
        result = SolveResult(source="x", candidates=[Candidate("AB", 0.9)], error="boom")
        assert result.ok is False

    def test_to_dict_shape(self):
        result = SolveResult(
            source="img.png",
            candidates=[Candidate("AB", 0.912345)],
            backend_used="stub",
        )
        payload = result.to_dict()
        assert payload["best"] == "AB"
        assert payload["confidence"] == 0.9123
        assert payload["backend_used"] == "stub"

    def test_to_dict_handles_empty(self):
        payload = SolveResult(source="x", error="nope").to_dict()
        assert payload["best"] is None
        assert payload["confidence"] == 0.0


class TestLoadImage:
    def test_returns_none_for_missing_file(self, tmp_path):
        assert load_image(tmp_path / "nope.png") is None

    def test_reads_valid_image(self, tmp_path, text_image):
        import cv2

        path = tmp_path / "captcha.png"
        cv2.imwrite(str(path), text_image)
        assert load_image(path) is not None


class TestPipelineChain:
    def test_primary_only_without_fallback(self):
        pipeline = Pipeline(Settings(backend="template"))
        assert pipeline._chain() == ["template"]

    def test_includes_fallbacks_in_order(self):
        settings = Settings(backend="template", fallback_backends="tesseract,claude")
        pipeline = Pipeline(settings)
        assert pipeline._chain() == ["template", "tesseract", "claude"]

    def test_deduplicates_primary_in_fallbacks(self):
        settings = Settings(backend="claude", fallback_backends="claude,tesseract")
        pipeline = Pipeline(settings)
        assert pipeline._chain() == ["claude", "tesseract"]


class TestPipelineSolve:
    def test_uses_first_confident_backend(self, text_image):
        pipeline = Pipeline(Settings(backend="stub", confidence_floor=0.5))
        primary = StubBackend(text="AB12", confidence=0.9)
        pipeline._backend_cache["stub"] = primary

        result = pipeline.solve_image(text_image)
        assert result.best.text == "AB12"
        assert result.backend_used == "stub"

    def test_falls_through_on_low_confidence(self, text_image):
        settings = Settings(backend="weak", fallback_backends="strong", confidence_floor=0.8)
        pipeline = Pipeline(settings)
        weak = StubBackend(text="WEAK", confidence=0.3)
        strong = StubBackend(text="STRONG", confidence=0.95)
        pipeline._backend_cache["weak"] = weak
        pipeline._backend_cache["strong"] = strong

        result = pipeline.solve_image(text_image)
        assert result.best.text == "STRONG"
        assert weak.calls == 1
        assert strong.calls == 1

    def test_skips_unavailable_backend(self, text_image):
        settings = Settings(backend="offline", fallback_backends="online")
        pipeline = Pipeline(settings)
        offline = StubBackend(available=False)
        online = StubBackend(text="OK", confidence=0.9)
        pipeline._backend_cache["offline"] = offline
        pipeline._backend_cache["online"] = online

        result = pipeline.solve_image(text_image)
        assert result.best.text == "OK"
        assert offline.calls == 0

    def test_error_when_no_backend_produces_result(self, text_image):
        pipeline = Pipeline(Settings(backend="empty"))
        pipeline._backend_cache["empty"] = StubBackend(text="")

        result = pipeline.solve_image(text_image)
        assert result.ok is False
        assert result.error is not None

    def test_records_attempted_backends(self, text_image):
        settings = Settings(backend="a", fallback_backends="b", confidence_floor=0.99)
        pipeline = Pipeline(settings)
        pipeline._backend_cache["a"] = StubBackend(text="AAA", confidence=0.5)
        pipeline._backend_cache["b"] = StubBackend(text="BBB", confidence=0.6)

        result = pipeline.solve_image(text_image)
        assert result.backends_attempted == ["a", "b"]

    def test_unknown_backend_reports_error(self, text_image):
        pipeline = Pipeline(Settings(backend="does-not-exist"))
        result = pipeline.solve_image(text_image)
        assert result.error is not None
        assert "Unknown backend" in result.error

    def test_captures_deskew_angle(self, rotated_image):
        pipeline = Pipeline(Settings(backend="stub"))
        pipeline._backend_cache["stub"] = StubBackend()
        result = pipeline.solve_image(rotated_image)
        assert isinstance(result.deskew_angle, float)


class TestPipelineHybrid:
    def test_merges_agreeing_backends(self, text_image):
        settings = Settings(backend="a", fallback_backends="b", hybrid=True)
        pipeline = Pipeline(settings)
        pipeline._backend_cache["a"] = StubBackend(text="SAME", confidence=0.7)
        pipeline._backend_cache["b"] = StubBackend(text="SAME", confidence=0.8)

        result = pipeline.solve_image(text_image)
        assert result.best.text == "SAME"
        assert "hybrid" in result.backend_used

    def test_calls_every_available_backend(self, text_image):
        settings = Settings(backend="a", fallback_backends="b", hybrid=True)
        pipeline = Pipeline(settings)
        first = StubBackend(text="AAA", confidence=0.9)
        second = StubBackend(text="BBB", confidence=0.9)
        pipeline._backend_cache["a"] = first
        pipeline._backend_cache["b"] = second

        pipeline.solve_image(text_image)
        assert first.calls == 1
        assert second.calls == 1

    def test_error_when_all_backends_silent(self, text_image):
        settings = Settings(backend="a", fallback_backends="b", hybrid=True)
        pipeline = Pipeline(settings)
        pipeline._backend_cache["a"] = StubBackend(text="")
        pipeline._backend_cache["b"] = StubBackend(text="")

        result = pipeline.solve_image(text_image)
        assert result.ok is False


class TestPipelineBatch:
    def test_processes_every_file(self, tmp_path, text_image):
        import cv2

        paths = []
        for index in range(3):
            path = tmp_path / f"captcha{index}.png"
            cv2.imwrite(str(path), text_image)
            paths.append(path)

        pipeline = Pipeline(Settings(backend="stub"))
        pipeline._backend_cache["stub"] = StubBackend()

        results = pipeline.solve_batch(paths)
        assert len(results) == 3

    def test_continues_after_unreadable_file(self, tmp_path, text_image):
        import cv2

        good = tmp_path / "good.png"
        cv2.imwrite(str(good), text_image)
        bad = tmp_path / "bad.png"
        bad.write_text("not an image")

        pipeline = Pipeline(Settings(backend="stub"))
        pipeline._backend_cache["stub"] = StubBackend()

        results = pipeline.solve_batch([bad, good])
        assert len(results) == 2
        assert results[0].ok is False
        assert results[1].ok is True

    def test_empty_batch(self):
        pipeline = Pipeline(Settings(backend="stub"))
        assert pipeline.solve_batch([]) == []


class TestSettings:
    def test_fallback_list_empty_by_default(self):
        assert Settings().fallback_list == []

    def test_fallback_list_parses_csv(self):
        settings = Settings(fallback_backends="a, b ,c")
        assert settings.fallback_list == ["a", "b", "c"]

    def test_backend_kwargs_for_template(self, tmp_path):
        settings = Settings(template_dir=tmp_path)
        assert settings.backend_kwargs("template") == {"template_dir": tmp_path}

    def test_backend_kwargs_for_claude(self):
        settings = Settings(aws_region="eu-west-1")
        kwargs = settings.backend_kwargs("claude")
        assert kwargs["region"] == "eu-west-1"

    def test_backend_kwargs_empty_for_tesseract(self):
        assert Settings().backend_kwargs("tesseract") == {}
