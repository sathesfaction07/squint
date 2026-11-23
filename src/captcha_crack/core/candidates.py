from dataclasses import dataclass, field
from itertools import product

AMBIGUOUS_GROUPS: list[set[str]] = [
    {"O", "0", "D", "Q"},
    {"l", "1", "I", "|", "i"},
    {"9", "g", "q"},
    {"5", "S", "s"},
    {"2", "Z", "z"},
    {"8", "B"},
    {"6", "b", "G"},
    {"n", "h"},
    {"u", "v"},
    {"m", "rn"},
    {"c", "C", "e"},
    {"k", "K"},
    {"w", "W"},
    {"x", "X"},
    {"7", "T"},
]

_AMBIGUITY_MAP: dict[str, set[str]] = {}
for group in AMBIGUOUS_GROUPS:
    for char in group:
        _AMBIGUITY_MAP.setdefault(char, set()).update(group - {char})


@dataclass
class CharacterPrediction:
    char: str
    confidence: float
    alternatives: list[tuple[str, float]] = field(default_factory=list)

    @property
    def is_ambiguous(self) -> bool:
        if not self.alternatives:
            return False
        return abs(self.confidence - self.alternatives[0][1]) < 0.12

    def swap_candidates(self, max_alternatives: int = 2) -> list[tuple[str, float]]:
        options = [(self.char, self.confidence)]

        for alt_char, alt_conf in self.alternatives[:max_alternatives]:
            options.append((alt_char, alt_conf))

        for confusable in _AMBIGUITY_MAP.get(self.char, set()):
            if confusable not in [c for c, _ in options]:
                options.append((confusable, self.confidence * 0.7))

        return sorted(options, key=lambda x: x[1], reverse=True)


@dataclass
class Candidate:
    text: str
    confidence: float

    def __repr__(self) -> str:
        return f"Candidate({self.text!r}, {self.confidence:.3f})"


def build_candidates(
    predictions: list[CharacterPrediction],
    max_results: int = 5,
    ambiguity_threshold: float = 0.4,
) -> list[Candidate]:
    if not predictions:
        return []

    ambiguous_count = sum(1 for p in predictions if p.is_ambiguous)
    ambiguity_ratio = ambiguous_count / len(predictions)

    if ambiguity_ratio < ambiguity_threshold:
        text = "".join(p.char for p in predictions)
        confidence = sum(p.confidence for p in predictions) / len(predictions)
        return [Candidate(text=text, confidence=confidence)]

    option_sets = []
    for prediction in predictions:
        if prediction.is_ambiguous:
            option_sets.append(prediction.swap_candidates(max_alternatives=2))
        else:
            option_sets.append([(prediction.char, prediction.confidence)])

    combinations = []
    for combo in product(*option_sets):
        text = "".join(char for char, _ in combo)
        confidence = sum(conf for _, conf in combo) / len(combo)
        combinations.append(Candidate(text=text, confidence=confidence))

        if len(combinations) >= max_results * 20:
            break

    unique: dict[str, Candidate] = {}
    for candidate in combinations:
        existing = unique.get(candidate.text)
        if existing is None or candidate.confidence > existing.confidence:
            unique[candidate.text] = candidate

    ranked = sorted(unique.values(), key=lambda c: c.confidence, reverse=True)
    return ranked[:max_results]


def merge_candidate_lists(
    lists: list[list[Candidate]],
    weights: list[float] | None = None,
    max_results: int = 5,
) -> list[Candidate]:
    if not lists:
        return []

    if weights is None:
        weights = [1.0] * len(lists)

    scores: dict[str, float] = {}
    for candidates, weight in zip(lists, weights, strict=False):
        for candidate in candidates:
            scores[candidate.text] = scores.get(candidate.text, 0.0) + candidate.confidence * weight

    total_weight = sum(weights)
    merged = [
        Candidate(text=text, confidence=score / total_weight) for text, score in scores.items()
    ]

    return sorted(merged, key=lambda c: c.confidence, reverse=True)[:max_results]
