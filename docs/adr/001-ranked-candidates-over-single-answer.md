# ADR 001: Return ranked candidates instead of a single answer

## Status

Accepted

## Context

The first version of the solver returned one string. Template matching on distorted CAPTCHAs sits in the 70-80% band as a general property of the technique (see ADR-002); this repo ships no labelled test set, so that range is a design assumption rather than a measurement taken here.

Even at the optimistic end of that range, a single string is the wrong output shape. The caller has no way to tell a correct reading from an incorrect one — every submission is a coin flip with no information about the odds. When a submission fails, the only recourse is to re-fetch the CAPTCHA and start over, throwing away a reading that may have been off by exactly one character.

The failure mode is also not uniform. It concentrates on character pairs that are genuinely hard to separate at low resolution: `O`/`0`, `l`/`1`/`I`, `5`/`S`, `2`/`Z`, `9`/`g`. A reading of `A3B7KO` that should have been `A3B7K0` is one substitution away from correct, and the matcher already knows which position was doubtful.

## Decision

Return a ranked list of candidate readings. Each character prediction carries its own confidence and its runner-up alternatives. When enough characters are ambiguous, expand the combinations and rank them by average confidence.

Ambiguity is judged relatively, not absolutely: a character is ambiguous when the second-best match is within 0.12 of the best. A low absolute score means "weak match", which is a different problem and does not warrant expansion.

Combinatorial explosion is bounded three ways — only ambiguous positions expand, at most two alternatives per position, and the candidate list is truncated to `max_candidates` after deduplication.

## Consequences

**Good**

- A caller can try candidate 2 when candidate 1 fails, without another network round trip
- Confidence is now actionable — the caller can route low-confidence results to a human
- Hybrid mode becomes possible, because merging ranked lists from several backends is well defined
- The failure mode changed from "wrong answer" to "answer plus alternatives", which is recoverable

**Bad**

- The API is more complex. `result.best.text` instead of `result.text`
- Callers who want one answer have to reach for `.best` and ignore the rest
- Ranking adds work on every solve, though it is negligible next to preprocessing

**Neutral**

- The 0.12 ambiguity threshold and the 0.4 expansion ratio are judgement calls, not values fitted to an evaluation set. Both are constants in `candidates.py` and will need adjusting for a very different CAPTCHA style
