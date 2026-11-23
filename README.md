# Squint

Reads distorted text from CAPTCHA images. Runs an OpenCV preprocessing pipeline locally, then hands the cleaned image to whichever recognition backend you pick — template matching, Tesseract, or a vision language model. Returns a ranked list of candidate readings rather than one confident guess.

## The Problem

An internal compliance team had to validate food safety license numbers against a government registry. The registry gated every lookup behind a CAPTCHA. Validating a few thousand licenses meant a person sitting there typing CAPTCHAs for hours, and no automation could run overnight because it would stall on the first challenge.

Commercial CAPTCHA solving services were not an option — sending regulatory data to a third party was a non-starter. So the tool had to run locally, or at worst inside our own cloud account.

The hard part was not reading clean text. It was reading text that had been deliberately mangled: characters overlapping each other, random lines drawn through them, coloured noise in the background, and every glyph rotated a few degrees in a different direction. A single wrong character means the whole submission fails and you have to try again.

## How It Works

Five stages, each of which can be inspected on its own.

**1. Denoise** — Bilateral filter for colour images, median blur for grayscale. Bilateral is slower than a plain Gaussian but it preserves edges, which matters when the characters are already thin.

**2. Threshold** — Adaptive Gaussian thresholding by default. CAPTCHAs often have a gradient background, and a single global threshold either eats the light characters or keeps the dark background. Otsu is available for images with even lighting.

**3. Strip lines** — Morphological opening with a long horizontal kernel finds horizontal lines, the same with a vertical kernel finds vertical ones. Subtract both from the image. This removes the strike-through lines without removing the characters, because a character is never 20 pixels wide in a single row.

**4. Deskew** — `minAreaRect` over the character pixels gives the dominant angle. Rotate by the inverse. Skipped entirely if the angle is under half a degree, since rotating introduces interpolation blur for no benefit.

**5. Recognize** — Whichever backend you selected. Two of them work on individual characters (template matching), the rest read the whole image at once.

Segmentation sits between stages 4 and 5 for the backends that need it. Contour detection finds candidate character boxes; boxes wider than 1.4× the median get split at the lowest point in their vertical pixel projection, which is where two overlapping characters touch.

### Recognition backends

| Backend | Where it runs | Typical accuracy | Cost |
|:--|:--|:--|:--|
| `template` | Local | 70–80% | Free |
| `tesseract` | Local | 75–85% | Free |
| `rekognition` | AWS | 85–90% | Per image |
| `claude` | AWS Bedrock | 95–98% | Per image |
| `gemini` | Google Vertex | 95–98% | Per image |
| `--hybrid` | Both | 97–99% | Sum of the above |

The accuracy figures are from testing against a set of synthetic CAPTCHAs with progressively worse distortion. Your numbers will differ depending on the CAPTCHA style.

## Architecture

```
                        ┌──────────────┐
   image file ────────► │  load_image  │
                        └──────┬───────┘
                               │
                    ┌──────────▼──────────┐
                    │    PREPROCESSING    │
                    │  denoise            │
                    │  grayscale          │
                    │  threshold          │
                    │  remove lines       │
                    │  clean speckles     │
                    │  deskew             │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
    needs segmentation?                  reads whole image
              │                                 │
    ┌─────────▼─────────┐                      │
    │   SEGMENTATION    │                      │
    │  find contours    │                      │
    │  merge fragments  │                      │
    │  split wide boxes │                      │
    │  normalize 32x32  │                      │
    └─────────┬─────────┘                      │
              │                                 │
    ┌─────────▼─────────┐        ┌─────────────▼──────────────┐
    │  template match   │        │  tesseract / claude /      │
    │  per character    │        │  gemini / rekognition      │
    └─────────┬─────────┘        └─────────────┬──────────────┘
              │                                 │
              └────────────────┬────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  CANDIDATE RANKING  │
                    │  detect ambiguity   │
                    │  expand confusables │
                    │  score combinations │
                    │  dedupe and rank    │
                    └──────────┬──────────┘
                               │
                        ┌──────▼───────┐
                        │ SolveResult  │
                        │  best        │
                        │  candidates  │
                        │  backend     │
                        └──────────────┘
```

Fallback chains and hybrid mode both live in `Pipeline`. A chain tries backends in order and stops at the first one that clears the confidence floor. Hybrid runs every available backend and merges their candidate lists with per-backend weights.

## Project Structure

```
squint/
├── src/captcha_crack/
│   ├── cli.py                 solve / stages / backends commands
│   ├── config.py              Pydantic settings, env-driven
│   ├── export.py              json / csv / text writers
│   ├── core/
│   │   ├── preprocess.py      the five-stage CV pipeline
│   │   ├── segment.py         contour detection, projection splitting
│   │   ├── candidates.py      ambiguity handling and ranking
│   │   └── pipeline.py        orchestration, fallback, hybrid merge
│   └── backends/
│       ├── base.py            RecognitionBackend ABC
│       ├── template.py        OpenCV template matching
│       ├── tesseract.py       local OCR
│       └── vlm.py             Claude, Gemini, Rekognition
├── tests/
│   ├── conftest.py            synthetic CAPTCHA fixtures
│   ├── unit/                  preprocess, segment, candidates, pipeline
│   └── integration/           end-to-end with real template files
├── data/
│   ├── templates/             character templates for the template backend
│   └── sample_captchas/       drop test images here
├── docs/adr/                  architecture decision records
└── setup.sh
```

## Running

```bash
bash setup.sh
source .venv/bin/activate
```

Optional backends:

```bash
bash setup.sh --tesseract    # local OCR
bash setup.sh --aws          # Bedrock Claude Vision + Rekognition
bash setup.sh --google       # Vertex AI Gemini
bash setup.sh --all
```

## Usage

Solve one image:

```bash
squint solve captcha.png --length 6
```

Solve a directory, write CSV:

```bash
squint solve ./captchas/ --length 6 --format csv --output results.csv
```

Local first, cloud only if local is unsure:

```bash
squint solve captcha.png --backend template --fallback claude
```

Run everything and merge:

```bash
squint solve captcha.png --hybrid --fallback tesseract,claude,gemini
```

Inspect every preprocessing stage:

```bash
squint stages captcha.png --output-dir debug/
```

Writes `00-original.png` through `06-deskewed.png` plus one file per segmented character. This is the command you actually use when a CAPTCHA style is failing and you need to see which stage is eating the characters.

Check what is usable in the current environment:

```bash
squint backends
```

### Python API

```python
from captcha_crack import Pipeline, Settings

pipeline = Pipeline(Settings(backend="claude", expected_length=6))
result = pipeline.solve_file(Path("captcha.png"))

if result.ok:
    print(result.best.text, result.best.confidence)
    for candidate in result.candidates[1:]:
        print("  alternative:", candidate.text)
```

Batch with graceful failure:

```python
results = pipeline.solve_batch(list(Path("captchas").glob("*.png")))
solved = [r for r in results if r.ok]
failed = [r for r in results if not r.ok]
```

One unreadable file does not stop the batch.

## Testing

```bash
pytest
pytest --cov=captcha_crack --cov-report=term-missing
pytest tests/unit -v
```

117 tests. The fixtures in `conftest.py` generate synthetic CAPTCHAs with OpenCV — clean text, noisy text, text with strike-through lines, rotated text — so nothing external is needed to run the suite.

## Configuration

Every setting is an env var prefixed `CAPTCHA_CRACK_`, or set them in `.env`.

| Variable | Default | Purpose |
|:--|:--|:--|
| `BACKEND` | `template` | Primary backend |
| `FALLBACK_BACKENDS` | empty | Comma-separated chain |
| `HYBRID` | `false` | Run all, merge results |
| `EXPECTED_LENGTH` | unset | Character count hint |
| `MAX_CANDIDATES` | `5` | Candidates returned |
| `CONFIDENCE_FLOOR` | `0.55` | Below this, try the next backend |
| `TEMPLATE_DIR` | `data/templates` | Character templates |
| `ADAPTIVE_THRESHOLD` | `true` | Adaptive vs Otsu |
| `STRIP_LINES` | `true` | Remove strike-through lines |
| `AWS_REGION` | `us-east-1` | Bedrock and Rekognition region |
| `BEDROCK_MODEL` | Claude 3.5 Sonnet | Model ID |
| `GCP_PROJECT` | unset | Required for Gemini |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Model name |

## Tradeoffs and Decisions I Made Along the Way

**Returning several candidates instead of one answer.** The first version returned a single string. It was right about three quarters of the time, which sounds acceptable until you realise the caller has no way to know which quarter they are in. Now the tool says "probably `A3B7K9`, but `A3B7KG` is close" and the caller can try both. A CAPTCHA is cheap to retry. Being confidently wrong is expensive.

**Ambiguity detection had to be about relative confidence, not absolute.** My first attempt flagged a character as ambiguous when its score fell below a threshold. That was wrong — a character can score low simply because the template set is poor, without there being any competing interpretation. What actually matters is whether the second-best match is close to the best. If `O` scores 0.61 and `0` scores 0.58, that is a real ambiguity. If `O` scores 0.61 and the runner-up is 0.20, it is just a weak match.

**I kept the OpenCV path even though the VLMs are far better.** Claude Vision reads these images at 95%+ out of the box. Template matching manages 70–80% on a good day. The local path stays for three reasons: it costs nothing, it works with no network, and it makes the tool debuggable. When a new CAPTCHA style breaks everything, `squint stages` shows you exactly which preprocessing step is destroying the characters. You cannot debug a VLM that way.

**Line removal is the stage I got wrong most often.** The morphological approach removes anything longer than N pixels in a single direction. Set N too low and you delete the horizontal stroke of an `E`. Too high and the strike-through survives. I settled on 20 pixels because CAPTCHA characters are rarely wider than that, but it is exposed as a parameter because that assumption will not hold everywhere.

**Splitting overlapping characters by pixel projection is a heuristic, and it shows.** When two characters touch, the column where they touch usually has fewer dark pixels than the columns through the character bodies. Cut there. It works often enough to be useful and fails on `rn` versus `m` in a way I have not solved. When the split count does not match the expected length, the confidence is penalised so the caller knows to be suspicious.

**Backends fail silently and return an empty list.** Every backend catches its own exceptions and returns `[]` rather than raising. In a batch of two thousand images you do not want the run to die because one API call timed out. The pipeline treats an empty list as "this backend had nothing to say" and moves to the next one. The tradeoff is that a misconfigured backend looks identical to a backend that simply could not read the image — which is why `squint backends` exists as a separate command.

**Hybrid mode weights the VLMs at 2× the local backends.** This is a judgement call, not a measurement. When Claude and template matching disagree, Claude is almost always right, so I did not want a simple average to let two weak local backends outvote one strong cloud one. If your local templates are excellent, lower the weight.

**Segmentation only runs for the backends that need it.** Template matching needs individual characters. Tesseract, Claude, Gemini, and Rekognition all read the whole image, and passing them a segmented image actively hurts — they lose the spatial context that helps them disambiguate. So `requires_segmentation` is a flag on the backend class and the pipeline routes accordingly: preprocessed-and-segmented for template, original image for everything else.

**On the ethics of this.** This exists to automate a compliance check against a public government registry that we were authorised to query — the CAPTCHA was an obstacle to a legitimate lookup, not a security boundary protecting anything. I would not build this to defeat a CAPTCHA that is doing its actual job of stopping abuse. The distinction matters, and if you are reusing this, it is worth being honest with yourself about which side of it you are on.

## Stack

Python 3.11+ · OpenCV · NumPy · Pydantic Settings · Click · Pytest

Optional: Tesseract · AWS Bedrock (Claude Vision) · AWS Rekognition · Google Vertex AI (Gemini)
