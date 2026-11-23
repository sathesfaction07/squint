import sys
from pathlib import Path

import click
import cv2

from .backends import BACKENDS
from .config import Settings
from .core.pipeline import Pipeline, load_image
from .core.preprocess import preprocess
from .core.segment import segment
from .export import collect_images, export


@click.group()
@click.version_option()
def main() -> None:
    """Read CAPTCHAs with OpenCV preprocessing and optional VLM backends."""


@main.command()
@click.argument("target", type=click.Path(exists=True, path_type=Path))
@click.option("--backend", type=click.Choice(list(BACKENDS)), help="Primary recognition backend.")
@click.option("--fallback", help="Comma-separated fallback backends.")
@click.option("--hybrid", is_flag=True, help="Run all backends and merge results by weight.")
@click.option("--length", type=int, help="Expected character count.")
@click.option(
    "--candidates", type=int, default=5, show_default=True, help="Max candidates to return."
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "csv", "text"]),
    default="text",
    show_default=True,
)
@click.option("--output", type=click.Path(path_type=Path), help="Write results to file.")
@click.option("--templates", type=click.Path(path_type=Path), help="Character template directory.")
def solve(
    target: Path,
    backend: str | None,
    fallback: str | None,
    hybrid: bool,
    length: int | None,
    candidates: int,
    fmt: str,
    output: Path | None,
    templates: Path | None,
) -> None:
    """Solve a single image or every image in a directory."""
    settings = Settings()
    if backend:
        settings.backend = backend
    if fallback:
        settings.fallback_backends = fallback
    if hybrid:
        settings.hybrid = True
    if length:
        settings.expected_length = length
    if templates:
        settings.template_dir = templates
    settings.max_candidates = candidates

    images = collect_images(target)
    if not images:
        click.echo(f"No images found at {target}", err=True)
        sys.exit(1)

    pipeline = Pipeline(settings)
    results = pipeline.solve_batch(images)

    payload = export(results, fmt, output)
    if payload:
        click.echo(payload)

    failures = sum(1 for r in results if not r.ok)
    if failures:
        click.echo(f"\n{failures}/{len(results)} failed", err=True)


@main.command()
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir", type=click.Path(path_type=Path), default=Path("debug"), show_default=True
)
def stages(image: Path, output_dir: Path) -> None:
    """Write every preprocessing stage to disk for inspection."""
    loaded = load_image(image)
    if loaded is None:
        click.echo(f"Cannot read {image}", err=True)
        sys.exit(1)

    result = preprocess(loaded, keep_stages=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / "00-original.png"), loaded)
    for index, (name, stage) in enumerate(result.stages.items(), start=1):
        cv2.imwrite(str(output_dir / f"{index:02d}-{name}.png"), stage)

    boxes = segment(result.image)
    for index, box in enumerate(boxes):
        cv2.imwrite(str(output_dir / f"char-{index:02d}.png"), box.image)

    click.echo(f"Wrote {len(result.stages) + 1} stages and {len(boxes)} characters to {output_dir}")
    click.echo(f"Deskew angle: {result.deskew_angle:.2f} degrees")


@main.command()
def backends() -> None:
    """Show which backends are usable in this environment."""
    settings = Settings()
    for name in BACKENDS:
        try:
            instance = BACKENDS[name](**settings.backend_kwargs(name))
            status = "ready" if instance.is_available() else "unavailable"
        except Exception as exc:
            status = f"error: {type(exc).__name__}"
        click.echo(f"{name:14} {status}")


if __name__ == "__main__":
    main()
