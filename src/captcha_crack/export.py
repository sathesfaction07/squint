import csv
import json
import sys
from pathlib import Path

from .core.pipeline import SolveResult

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}


def collect_images(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return []


def write_json(results: list[SolveResult], output: Path | None) -> str:
    payload = json.dumps([r.to_dict() for r in results], indent=2)
    if output:
        output.write_text(payload, encoding="utf-8")
    return payload


def write_csv(results: list[SolveResult], output: Path | None) -> str:
    rows = []
    for result in results:
        rows.append(
            {
                "source": result.source,
                "best": result.best.text if result.best else "",
                "confidence": f"{result.best.confidence:.4f}" if result.best else "0",
                "alternatives": " | ".join(c.text for c in result.candidates[1:]),
                "backend": result.backend_used,
                "deskew_angle": f"{result.deskew_angle:.2f}",
                "error": result.error or "",
            }
        )

    fieldnames = [
        "source",
        "best",
        "confidence",
        "alternatives",
        "backend",
        "deskew_angle",
        "error",
    ]

    if output:
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return f"Wrote {len(rows)} rows to {output}"

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return ""


def write_text(results: list[SolveResult], output: Path | None) -> str:
    lines = []
    for result in results:
        if result.error:
            lines.append(f"{result.source}\tERROR\t{result.error}")
            continue
        alternatives = ",".join(c.text for c in result.candidates[1:])
        lines.append(
            f"{result.source}\t{result.best.text}\t{result.best.confidence:.3f}\t{alternatives}"
        )

    payload = "\n".join(lines)
    if output:
        output.write_text(payload + "\n", encoding="utf-8")
    return payload


WRITERS = {
    "json": write_json,
    "csv": write_csv,
    "text": write_text,
}


def export(results: list[SolveResult], fmt: str, output: Path | None = None) -> str:
    writer = WRITERS.get(fmt)
    if not writer:
        available = ", ".join(WRITERS.keys())
        raise ValueError(f"Unknown format: {fmt}. Available: {available}")
    return writer(results, output)
