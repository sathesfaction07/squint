# CaptchaCrack — Technical Architecture

## Overview

CaptchaCrack is a distorted-text recognition tool built for CAPTCHA images. It runs a five-stage OpenCV preprocessing pipeline (denoise → grayscale → threshold → line removal → deskew) to clean intentionally mangled input, then routes the image to whichever recognition backend is available — local template matching, Tesseract OCR, or cloud vision models (AWS Bedrock Claude, AWS Rekognition, Google Vertex Gemini). Returns ranked candidate readings rather than a single confident guess, enabling callers to retry alternatives without re-fetching.

**Design principle:** Ranked candidates with honest confidence, not one confident answer. When a character is ambiguous (second-best within 0.12 of best), the system expands confusable alternatives and returns them in confidence order.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLI (captcha-crack)                              │
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────────┐    │
│  │   solve      │   │   stages     │   │     backends            │    │
│  │ - single img │   │ - debug each │   │ - list available        │    │
│  │ - directory  │   │   preprocess │   │   in current env        │    │
│  │ - fallback   │   │   stage      │   └─────────────────────────┘    │
│  │ - hybrid     │   └──────────────┘                                    │
│  └──────────────┘                                                       │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE (5 stages)                       │
│                     src/captcha_crack/core/preprocess.py                  │
│                                                                         │
│  1. Denoise       → Bilateral filter (color) / median blur (gray)        │
│  2. Grayscale     → Convert to single channel                            │
│  3. Threshold     → Adaptive Gaussian or Otsu                            │
│  4. Remove Lines  → Morphological opening (H+V kernels)                  │
│  5. Clean+Deskew  → Remove speckles, minAreaRect rotation               │
│                                                                         │
│  Output: PreprocessResult(image, stages, deskew_angle)                   │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SEGMENTATION (conditional)                           │
│                      src/captcha_crack/core/segment.py                    │
│                                                                         │
│  Run ONLY if backend.requires_segmentation == True                       │
│                                                                         │
│  - Find contours (cv2.RETR_EXTERNAL)                                     │
│  - Merge thin fragments (gap < 2px, width < 6px)                         │
│  - Split wide boxes by vertical projection valleys                       │
│  - Normalize to 32x32 per character                                      │
│                                                                         │
│  Output: list[CharacterBox]                                              │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
          ┌───────────┴────────────┐
          │                        │
   requires_segmentation?    reads whole image
          │                        │
          ▼                        ▼
┌──────────────────────┐   ┌──────────────────────────────────────────┐
│  TEMPLATE BACKEND    │   │  TESSERACT / VLM BACKENDS                  │
│  (local)             │   │  (local / cloud)                           │
│                      │   │                                            │
│  Template matching   │   │  Tesseract: pytesseract wrapper            │
│  - Load char library │   │  Claude: Bedrock invoke_model              │
│  - TM_CCOEFF_NORMED  │   │  Gemini: Vertex GenerativeModel            │
│  - Per-char scoring  │   │  Rekognition: detect_text                  │
│  - Top-3 alts        │   │                                            │
│                      │   │  All read original or preprocessed image,  │
│                      │   │  not segmented glyphs                      │
└──────────┬───────────┘   └──────────┬─────────────────────────────────┘
           │                          │
           └───────────┬──────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 CANDIDATE RANKING & MERGE                                 │
│                 src/captcha_crack/core/candidates.py                      │
│                                                                         │
│  CharacterPrediction.is_ambiguous → abs(best - 2nd) < 0.12              │
│  Ambiguity ratio → count ambiguous / total positions                     │
│  If ratio < 0.4 → return single best reading                             │
│  Else → expand combinations:                                             │
│    - Only ambiguous positions expand                                     │
│    - Max 2 alternatives per position                                     │
│    - Add confusables from AMBIGUOUS_GROUPS                               │
│    - Score by average confidence                                         │
│    - Dedupe and truncate to max_candidates                               │
│                                                                         │
│  Hybrid mode: merge_candidate_lists with per-backend weights             │
│    (VLMs weighted 2.0, local 1.0)                                        │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 ORCHESTRATION & FALLBACK                                  │
│                 src/captcha_crack/core/pipeline.py                        │
│                                                                         │
│  Fallback chain: try backends sequentially, stop at first reading       │
│                  above confidence_floor (default 0.55)                   │
│                                                                         │
│  Hybrid mode: run all available backends, merge by weighted average     │
│                                                                         │
│  Output: SolveResult(best, candidates, backend_used, deskew_angle)       │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          EXPORT                                           │
│                          src/captcha_crack/export.py                      │
│                                                                         │
│  Formats: JSON / CSV / text                                              │
│  Batch graceful failure: one unreadable image does not stop the run      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. CLI (`src/captcha_crack/cli.py`)

Three commands exposed via Click:

| Command | Purpose |
|:--|:--|
| `solve` | Read one image or all images in a directory. Options: `--backend`, `--fallback`, `--hybrid`, `--length`, `--format` (json/csv/text), `--output`. |
| `stages` | Write every preprocessing stage to disk for debugging. Outputs `00-original.png` through `06-deskewed.png` plus one file per segmented character. |
| `backends` | List which backends are usable in the current environment. Checks `is_available()` for each registered backend. |

Entry point: `captcha-crack` console script defined in `pyproject.toml`.

### 2. Preprocessing (`src/captcha_crack/core/preprocess.py`)

Five-stage pipeline. Each stage is a standalone function; `preprocess()` orchestrates.

1. **denoise** — Bilateral filter for color images (preserves edges), median blur for grayscale. Strength parameter default 9.
2. **to_grayscale** — `cv2.cvtColor(COLOR_BGR2GRAY)` if 3-channel.
3. **threshold** — Adaptive Gaussian thresholding (block size 11, C=2) by default. Otsu available via `adaptive_threshold=False`. Returns binary inverted (text white, background black).
4. **remove_lines** — Morphological opening with horizontal kernel (20×1) and vertical kernel (1×20). Finds long lines, subtracts them. Does not remove character strokes because characters are rarely 20px wide in one direction.
5. **deskew** — `minAreaRect` over character pixels to find dominant angle. Rotate by inverse using `warpAffine(INTER_CUBIC)`. Skipped if angle < 0.5° to avoid interpolation blur.

Also: **clean_speckles** — `connectedComponentsWithStats` removes components smaller than 8px area. Runs after line removal, before deskew.

Returns: `PreprocessResult` dataclass with final binary image, optional stages dict (if `keep_stages=True`), and deskew angle in degrees.

### 3. Segmentation (`src/captcha_crack/core/segment.py`)

Runs only for backends with `requires_segmentation=True` (currently only `template`).

**segment()** — Main entry point
- `find_contour_boxes` — `cv2.findContours(RETR_EXTERNAL)` with bounding rectangles, sorted left-to-right.
- `merge_fragments` — Joins thin boxes (width < 6px) separated by < 2px gap. Catches character strokes that broke during preprocessing.
- `split_wide_box` — If box width > 1.4× median, assumes overlapping characters. Computes vertical projection (sum of dark pixels per column). Finds valleys (columns below 15% of max). Cuts at valley midpoints. Fallback: equal-width slicing if valleys don't match expected count.
- `normalize_character` — Resizes to 32×32 canvas with padding, maintains aspect ratio.

Output: `list[CharacterBox]` with x, y, width, height, and normalized image for each glyph.

Segmentation quality affects confidence: if the split count doesn't match `expected_length`, the pipeline treats it as low-quality and penalizes confidence (in template backend) or moves to the next backend in a fallback chain.

### 4. Recognition Backends (`src/captcha_crack/backends/`)

All inherit `RecognitionBackend` ABC. Interface:
- `recognize(image, expected_length, max_candidates) -> list[Candidate]`
- `is_available() -> bool`
- `requires_segmentation: bool` class attribute

| Backend | Accuracy | Needs Segmentation | Cost | Offline |
|:--|:--|:--|:--|:--|
| `template` | 70–80% | Yes | Free | Yes |
| `tesseract` | 75–85% | No | Free | Yes |
| `rekognition` | 85–90% | No | Per image | No |
| `claude` | 95–98% | No | Per image + tokens | No |
| `gemini` | 95–98% | No | Per image + tokens | No |

**TemplateBackend** — Loads character templates from `template_dir/manifest.json`. For each segmented glyph, runs `cv2.matchTemplate(TM_CCOEFF_NORMED)` against all templates, keeps best score per label. Returns `CharacterPrediction` with top match and top-3 alternatives. Calls `build_candidates()` to expand ambiguous positions.

**TesseractBackend** — Wraps `pytesseract.image_to_data()` with PSM 8 (single word) and a character whitelist. Returns per-character confidence. Lazy-loads pytesseract so the tool doesn't require it unless used. If character count != expected, applies 0.75 penalty.

**ClaudeVisionBackend** — `boto3` client for `bedrock-runtime`. Encodes image as base64 PNG. Prompt instructs Claude to return JSON with ranked candidates and confidence. Temperature 0 for determinism. Parses response with regex fallback for partial JSON. Max tokens 512.

**GeminiVisionBackend** — Vertex AI `GenerativeModel` client. Same prompt structure as Claude. Encodes image as PNG bytes, passes as `Part.from_data()`. Temperature 0, max tokens 512.

**RekognitionBackend** — `boto3` client for AWS Rekognition. Calls `detect_text()`, filters `Type == "WORD"`. Concatenates detected words, averages confidence. Does not return multiple candidates. If length != expected, applies 0.7 penalty.

All backends catch exceptions and return empty list rather than raising. This enables batch processing to continue when one image or one backend fails.

### 5. Candidate Ranking (`src/captcha_crack/core/candidates.py`)

Core data structures:
- `CharacterPrediction` — one character, its confidence, and a list of `(char, confidence)` alternatives.
- `Candidate` — one complete reading (string) plus overall confidence.

**Ambiguity detection** — `CharacterPrediction.is_ambiguous` checks `abs(confidence - alternatives[0][1]) < 0.12`. A character is ambiguous when the second-best match is close to the best. A low absolute score (e.g., 0.3) without a close runner-up is not ambiguous — it is a weak match, which is a different signal.

**AMBIGUOUS_GROUPS** — Hardcoded confusable sets: `{O,0,D,Q}`, `{l,1,I,|,i}`, `{9,g,q}`, `{5,S,s}`, `{2,Z,z}`, etc. When a character is ambiguous, `swap_candidates()` adds these confusables at 0.7× confidence as fallback alternatives.

**build_candidates()** — Main ranking function
1. Count how many positions are ambiguous. If fewer than 40% (`ambiguity_threshold`), return single best reading.
2. For ambiguous positions, expand to top 2 alternatives plus confusables.
3. For non-ambiguous positions, keep only the best.
4. Generate combinations via `itertools.product`. Break early after `max_results * 20` combinations to bound runtime.
5. Deduplicate by text (keep highest confidence).
6. Sort by confidence descending, truncate to `max_results`.

**merge_candidate_lists()** — Hybrid mode merger. Weighted sum of confidences across backends. VLMs default to weight 2.0, local to 1.0. Normalizes by total weight. Deduplicates and ranks.

### 6. Orchestration (`src/captcha_crack/core/pipeline.py`)

**Pipeline** class — Main entry point for solving.

- `solve_image(image, source, expected_length)` — Core solve logic. Calls `preprocess()`, then routes to `_solve_hybrid()` or fallback chain.
- `solve_file(path, expected_length)` — Loads image with `cv2.imread()`, delegates to `solve_image()`.
- `solve_batch(paths, expected_length)` — Solves all images, catches per-image exceptions, returns list of `SolveResult`. One failure does not stop the batch.

**Fallback chain** — `_chain()` builds ordered list: primary backend + `fallback_backends` (deduplicated). Tries each in order. Stops at first reading with `confidence >= confidence_floor`. If no backend clears the floor, returns the best reading from any backend that produced one.

**Hybrid mode** — `_solve_hybrid()` runs all available backends in the chain, collects their candidate lists, merges with per-backend weights. VLMs (Claude, Gemini) weighted 2.0, local (template, Tesseract) weighted 1.0. Returns merged candidate list with `backend_used = "hybrid(...)"`.

**Backend routing** — Each backend declares `requires_segmentation`. Template gets `processed.image` (binary, segmented). All others get `original` (color, whole image). VLMs perform better with the original because they use spatial context.

Output: `SolveResult` with `best: Candidate | None`, full `candidates` list, `backend_used`, `backends_attempted`, `deskew_angle`, `error: str | None`.

### 7. Export (`src/captcha_crack/export.py`)

Three output formats:

**JSON** — `[{source, best, confidence, candidates: [{text, confidence}], backend_used, deskew_angle, error}]`

**CSV** — Columns: `source`, `best`, `confidence`, `alternatives` (pipe-delimited), `backend`, `deskew_angle`, `error`.

**Text** — Tab-delimited: `source \t best \t confidence \t alternatives`. One line per image.

`collect_images(target)` — If target is file, return `[target]`. If directory, glob `**/*` for image suffixes `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.gif`, `.webp`.

Batch processing writes all results regardless of individual failures. Failed images have `error` field populated.

---

## Data Flow

```
1. User runs: captcha-crack solve captcha.png --backend claude --fallback tesseract --length 6
2. CLI parses args, builds Settings(backend="claude", fallback_backends="tesseract", expected_length=6)
3. Pipeline instantiates. collect_images(target) returns [captcha.png]
4. Pipeline.solve_file() loads image with cv2.imread()
5. preprocess() runs 5 stages: denoise → grayscale → adaptive threshold → remove_lines → deskew
6. preprocess() returns PreprocessResult with binary image and deskew_angle=-2.3°
7. Pipeline checks settings.hybrid=False, so enters fallback chain mode
8. First backend: claude. ClaudeVisionBackend.is_available() checks boto3 bedrock-runtime client → True
9. claude.requires_segmentation=False, so passes original color image (not binary)
10. ClaudeVisionBackend.recognize() encodes as PNG base64, calls bedrock-runtime.invoke_model()
11. Claude returns JSON: {"candidates": [{"text":"A3B7K9","confidence":0.97}, {"text":"A3B7KG","confidence":0.61}]}
12. _parse_response() extracts, sorts, returns [Candidate("A3B7K9", 0.97), Candidate("A3B7KG", 0.61)]
13. Best confidence 0.97 >= confidence_floor 0.55, so chain stops here
14. Pipeline returns SolveResult(best=Candidate("A3B7K9", 0.97), candidates=[...], backend_used="claude")
15. Export to text format: "captcha.png\tA3B7K9\t0.970\tA3B7KG"
16. CLI prints output to stdout
```

---

## Ambiguity and Candidate Ranking Design

This section covers the project's defining architectural choice: returning ranked candidates with honest confidence rather than one confident answer. (See ADR-001 for the original rationale.)

### Why ranked candidates

The first version returned a single string. Template matching on distorted CAPTCHAs lands in the 70-80% band (see ADR-002 — that is a general characteristic of the technique, not a number this repo measures). Even taken at the optimistic end, a single string is the wrong output shape, because the caller has no way to distinguish the 75% from the 25%. Every submission is a gamble with no information about the odds. When one fails, the only recourse is to re-fetch the CAPTCHA and start over — throwing away a reading that may have been off by exactly one character.

The failure mode is not uniform. It concentrates on character pairs that are genuinely hard to separate at low resolution: `O`/`0`, `l`/`1`/`I`, `5`/`S`, `2`/`Z`, `9`/`g`. A reading of `A3B7KO` that should have been `A3B7K0` is one substitution away from correct, and the matcher already knows which position was doubtful. Returning both candidates lets the caller try both without another network round trip.

Ranked candidates also made hybrid mode possible. When three backends disagree, merging their ranked lists is well-defined (weighted sum of confidences). Merging three single-string answers is not.

### Relative confidence, not absolute

The critical detail: ambiguity is judged by **relative confidence** (second-best within 0.12 of best), not absolute score.

My first attempt flagged a character as ambiguous when its score fell below a threshold. That was wrong. A character can score low simply because the template set is poor, without there being any competing interpretation. What actually matters is whether the second-best match is close to the best. If `O` scores 0.61 and `0` scores 0.58, that is a real ambiguity. If `O` scores 0.61 and the runner-up is 0.20, it is just a weak match — the system is fairly sure, just not confident in the template library.

This distinction is implemented in `CharacterPrediction.is_ambiguous`:

```python
@property
def is_ambiguous(self) -> bool:
    if not self.alternatives:
        return False
    return abs(self.confidence - self.alternatives[0][1]) < 0.12
```

A low absolute score is a different problem. It signals "try a better backend" rather than "expand alternatives". The fallback chain handles this: if the best reading from backend A scores below `confidence_floor`, try backend B.

### Bounding combinatorial explosion

Expanding every ambiguous character into all its alternatives causes combinatorial explosion. A 6-character CAPTCHA with 3 ambiguous positions, 2 alternatives each, yields 2³ = 8 combinations. Three bounds keep this tractable:

1. **Ambiguity ratio threshold** — If fewer than 40% of positions are ambiguous, return the single best reading without expansion. (`ambiguity_threshold=0.4` in `build_candidates()`.)
2. **Max alternatives per position** — At most 2 alternatives per ambiguous position. (`max_alternatives=2` in `CharacterPrediction.swap_candidates()`.)
3. **Early truncation** — Stop generating combinations after `max_results * 20` to bound runtime. Then deduplicate and truncate to `max_results`.

Without bound 1, a clean reading with one slightly doubtful character would still expand unnecessarily. Without bound 2, deeply ambiguous positions would balloon the result set. Without bound 3, a heavily noised image could stall on millions of combinations.

### Confusable character groups

When a character is ambiguous, `swap_candidates()` adds confusables from `AMBIGUOUS_GROUPS`:

```python
AMBIGUOUS_GROUPS = [
    {"O", "0", "D", "Q"},
    {"l", "1", "I", "|", "i"},
    {"9", "g", "q"},
    {"5", "S", "s"},
    {"2", "Z", "z"},
    {"8", "B"},
    # ...9 more groups (15 total)
]
```

These are not inferred from data. They are hardcoded based on common CAPTCHA confusions. When the top match for a position is `O` and the runner-up is `D`, the system also offers `0` and `Q` as fallback candidates, weighted at 0.7× the original confidence. This catches cases where template matching ranked them lower than they deserve.

### Fallback chain and confidence floor

A fallback chain tries backends sequentially. The chain stops at the first reading with `confidence >= confidence_floor` (default 0.55). If no backend clears the floor, the pipeline returns the best reading from any backend that produced one.

The floor exists because template matching can return a candidate at 0.25 confidence that is almost certainly garbage. Returning it to the caller is misleading. Better to try Tesseract or Claude first.

Hybrid mode ignores the floor and merges everything. The assumption is that if you enabled hybrid mode, you want the system to use all available information rather than short-circuit.

### Hybrid mode weighting

Hybrid mode runs all available backends and merges their candidate lists. The merge is a weighted sum of confidences:

```python
weights.append(2.0 if name in ("claude", "gemini") else 1.0)
```

Vision language models are weighted 2x local backends. This is a judgement call, not a measurement — it encodes the accuracy gap asserted in ADR-002 (VLMs 95-98%, template 70-80%) rather than anything this repo benchmarks. The purpose is to stop two weak local backends from outvoting one strong cloud one, which is what a simple average would allow. If your local templates are excellent, lower the VLM weight.

### Why the local CV path is kept

Claude Vision reads these images at 95–98% accuracy out of the box. Template matching manages 70–80% on a good day. By accuracy alone, the local path has no reason to exist. It stays for three reasons (ADR-002):

1. **Zero cost** — A batch of 10,000 images costs nothing locally and real money through a VLM.
2. **Debuggability** — `captcha-crack stages` writes every intermediate image. When a new CAPTCHA style fails, you see which stage is destroying the characters. A VLM gives you a wrong answer and no explanation.
3. **Offline operation** — Works in air-gapped environments and inside networks with no outbound access.

The local CV path is the default. Cloud backends are opt-in via `--backend`, `--fallback`, or `--hybrid`.

### Honest limitations

A concrete observed run: a synthetic 6-character CAPTCHA (`K7M3P9`) rendered with per-character rotation, subtractive noise and two strike-through lines was solved with the `template` backend against a generated 36-glyph library. Segmentation over-split it into 14 boxes rather than 6. The top reading came back `HHKQ6O` at 0.337 confidence with four alternatives clustered between 0.333 and 0.334 — wrong, and nowhere near the 0.55 floor.

That is the design working, not failing. Three things went right even though the answer was wrong: the confidence was honestly low rather than inflated, the near-identical alternative scores correctly signalled that the matcher was guessing between equally poor options, and the sub-floor score is exactly the condition that makes a fallback chain escalate to a vision model. A single-string API would have returned `HHKQ6O` with no indication anything was wrong.

It also exposes the real weak link: segmentation, not recognition. No template library can recover from a 14-way split of a 6-character image. This is why `captcha-crack stages` exists — it writes each intermediate image so you can see the over-split rather than guessing at it.

When segmentation splits a character count that does not match `expected_length`, the confidence is accurate — the system knows the split is questionable. The alternative would be to return a reading at 0.85 confidence that is almost certainly malformed.

Confusable groups are incomplete. `rn` versus `m` is a known failure mode for projection-based splitting and is not in the groups. Adding it would cause too many false positives in normal text.

---

## Configuration

All configuration via environment variables with Pydantic Settings:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CAPTCHA_CRACK_",
        env_file=".env",
        extra="ignore",
    )

    backend: str = "template"
    fallback_backends: str = ""
    hybrid: bool = False

    expected_length: int | None = None
    max_candidates: int = 5

    template_dir: Path = Path("data/templates")

    adaptive_threshold: bool = True
    strip_lines: bool = True
    min_character_area: int = 20

    aws_region: str = "us-east-1"
    bedrock_model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    gcp_project: str | None = None
    gcp_location: str = "us-central1"
    gemini_model: str = "gemini-1.5-flash"

    confidence_floor: float = 0.55
```

Example `.env`:

```
CAPTCHA_CRACK_BACKEND=claude
CAPTCHA_CRACK_FALLBACK_BACKENDS=tesseract,template
CAPTCHA_CRACK_EXPECTED_LENGTH=6
CAPTCHA_CRACK_MAX_CANDIDATES=5
CAPTCHA_CRACK_CONFIDENCE_FLOOR=0.55
CAPTCHA_CRACK_AWS_REGION=us-east-1
```

---

## Running

```bash
# Install
bash setup.sh
source .venv/bin/activate

# Optional backends
bash setup.sh --tesseract  # Local OCR
bash setup.sh --aws        # Bedrock + Rekognition
bash setup.sh --google     # Vertex Gemini
bash setup.sh --all        # Everything

# Solve one image
captcha-crack solve captcha.png --length 6

# Solve directory, write CSV
captcha-crack solve ./captchas/ --length 6 --format csv --output results.csv

# Fallback chain: template first, Claude if unsure
captcha-crack solve captcha.png --backend template --fallback claude --length 6

# Hybrid: run all available, merge by weight
captcha-crack solve captcha.png --hybrid --fallback tesseract,claude,gemini --length 6

# Debug preprocessing stages
captcha-crack stages captcha.png --output-dir debug/
# Writes 00-original.png through 06-deskewed.png plus char-00.png...char-N.png

# Check available backends
captcha-crack backends
# Output example:
# template       unavailable  (no templates loaded)
# tesseract      ready
# claude         ready
# gemini         error: DefaultCredentialsError
# rekognition    ready
```

---

## Testing

```bash
# All tests (117 total)
pytest

# Unit tests only (no external dependencies)
pytest tests/unit -v

# Integration tests (requires template files if testing template backend)
pytest tests/integration -v

# Coverage
pytest --cov=captcha_crack --cov-report=term-missing
```

Test fixtures in `conftest.py` generate synthetic CAPTCHAs with OpenCV — clean text, noisy text, text with strike-through lines, rotated text — so nothing external is needed to run the suite. Real test count: **117 tests**.
