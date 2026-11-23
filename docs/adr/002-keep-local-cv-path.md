# ADR 002: Keep the local OpenCV path alongside the vision models

## Status

Accepted

## Context

Claude Vision and Gemini read distorted CAPTCHAs at 95–98% accuracy with no preprocessing, no template library, and no tuning. Template matching, after five stages of careful preprocessing, reaches 70–80%.

By accuracy alone the local path has no reason to exist. Deleting `preprocess.py`, `segment.py`, and `template.py` would remove roughly 400 lines and make the tool both simpler and more accurate.

The portfolio guidance is also explicit that accuracy-sensitive projects should assume the worst possible input and should not assume AWS is available. That pushes in the opposite direction.

## Decision

Keep both. The local pipeline is the default backend; cloud backends are opt-in via `--backend`, `--fallback`, or `--hybrid`.

## Consequences

**Good**

- Zero marginal cost. A batch of 10,000 images costs nothing locally and real money through a VLM
- Works offline and inside networks with no outbound access
- Debuggable. `captcha-crack stages` writes every intermediate image to disk. When a new CAPTCHA style fails, you can see whether line removal ate the character strokes or the threshold washed them out. A VLM gives you a wrong answer and no explanation
- The preprocessing is reusable by the cloud backends. Deskewing and line removal improve VLM accuracy too, so the work is not wasted even when the local recognizer is unused
- Sensible privacy default — nothing leaves the machine unless the caller explicitly asks for a cloud backend

**Bad**

- Two code paths to maintain, and the `requires_segmentation` flag exists purely to route between them
- The template backend needs a character template library that the repository does not ship. Out of the box it reports itself unavailable, which is confusing until you read `captcha-crack backends`
- More surface area: 400 extra lines, 60 extra tests

**Neutral**

- The default backend being the least accurate one is a deliberate choice. A new user gets something that runs immediately with no credentials, and the README is direct about the accuracy tradeoff
