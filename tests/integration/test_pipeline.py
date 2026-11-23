import json

import cv2
import numpy as np
import pytest

from captcha_crack.backends.template import TemplateBackend
from captcha_crack.config import Settings
from captcha_crack.core.pipeline import Pipeline
from captcha_crack.export import collect_images, export


@pytest.fixture
def template_dir(tmp_path):
    directory = tmp_path / "templates"
    directory.mkdir()

    manifest = {}
    for char in "AB12":
        glyph = np.zeros((32, 32), dtype=np.uint8)
        cv2.putText(glyph, char, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
        cv2.imwrite(str(directory / f"{char}.png"), glyph)
        manifest[char] = char

    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture
def captcha_files(tmp_path, text_image):
    directory = tmp_path / "captchas"
    directory.mkdir()
    for index in range(3):
        cv2.imwrite(str(directory / f"sample{index}.png"), text_image)
    (directory / "notes.txt").write_text("not an image")
    return directory


class TestTemplateBackendIntegration:
    def test_reports_unavailable_without_templates(self, tmp_path):
        backend = TemplateBackend(template_dir=tmp_path / "missing")
        assert backend.is_available() is False

    def test_loads_templates_from_manifest(self, template_dir):
        backend = TemplateBackend(template_dir=template_dir)
        assert backend.is_available() is True

    def test_returns_empty_without_templates(self, tmp_path, binary_chars):
        backend = TemplateBackend(template_dir=tmp_path / "missing")
        assert backend.recognize(binary_chars) == []

    def test_produces_candidates(self, template_dir, binary_chars):
        backend = TemplateBackend(template_dir=template_dir)
        candidates = backend.recognize(binary_chars, expected_length=4)
        assert isinstance(candidates, list)

    def test_handles_blank_input(self, template_dir):
        backend = TemplateBackend(template_dir=template_dir)
        assert backend.recognize(np.zeros((40, 100), dtype=np.uint8)) == []


class TestEndToEndPipeline:
    def test_full_solve_with_templates(self, template_dir, tmp_path, text_image):
        path = tmp_path / "captcha.png"
        cv2.imwrite(str(path), text_image)

        settings = Settings(backend="template", template_dir=template_dir, expected_length=4)
        pipeline = Pipeline(settings)
        result = pipeline.solve_file(path)

        assert result.source == str(path)
        assert result.backends_attempted == ["template"]

    def test_batch_solve_directory(self, template_dir, captcha_files):
        settings = Settings(backend="template", template_dir=template_dir, expected_length=4)
        pipeline = Pipeline(settings)

        images = collect_images(captcha_files)
        results = pipeline.solve_batch(images)

        assert len(images) == 3
        assert len(results) == 3

    def test_unavailable_backend_surfaces_error(self, tmp_path, text_image):
        path = tmp_path / "captcha.png"
        cv2.imwrite(str(path), text_image)

        settings = Settings(backend="template", template_dir=tmp_path / "no-templates")
        pipeline = Pipeline(settings)
        result = pipeline.solve_file(path)

        assert result.ok is False
        assert result.error is not None


class TestCollectImages:
    def test_single_file(self, tmp_path, text_image):
        path = tmp_path / "one.png"
        cv2.imwrite(str(path), text_image)
        assert collect_images(path) == [path]

    def test_directory_filters_non_images(self, captcha_files):
        images = collect_images(captcha_files)
        assert len(images) == 3
        assert all(p.suffix == ".png" for p in images)

    def test_missing_path(self, tmp_path):
        assert collect_images(tmp_path / "nope") == []

    def test_results_sorted(self, captcha_files):
        images = collect_images(captcha_files)
        assert images == sorted(images)


class TestExport:
    def test_json_roundtrip(self, template_dir, captcha_files, tmp_path):
        settings = Settings(backend="template", template_dir=template_dir)
        pipeline = Pipeline(settings)
        results = pipeline.solve_batch(collect_images(captcha_files))

        output = tmp_path / "out.json"
        export(results, "json", output)

        payload = json.loads(output.read_text())
        assert len(payload) == 3
        assert "source" in payload[0]

    def test_csv_has_header_and_rows(self, template_dir, captcha_files, tmp_path):
        settings = Settings(backend="template", template_dir=template_dir)
        pipeline = Pipeline(settings)
        results = pipeline.solve_batch(collect_images(captcha_files))

        output = tmp_path / "out.csv"
        export(results, "csv", output)

        lines = output.read_text().strip().split("\n")
        assert lines[0].startswith("source,best,confidence")
        assert len(lines) == 4

    def test_text_format(self, template_dir, captcha_files, tmp_path):
        settings = Settings(backend="template", template_dir=template_dir)
        pipeline = Pipeline(settings)
        results = pipeline.solve_batch(collect_images(captcha_files))

        output = tmp_path / "out.txt"
        export(results, "text", output)
        assert output.exists()

    def test_rejects_unknown_format(self):
        with pytest.raises(ValueError, match="Unknown format"):
            export([], "yaml")
