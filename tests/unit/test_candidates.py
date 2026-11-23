from captcha_crack.core.candidates import (
    AMBIGUOUS_GROUPS,
    Candidate,
    CharacterPrediction,
    build_candidates,
    merge_candidate_lists,
)


class TestAmbiguityGroups:
    def test_zero_and_letter_o_grouped(self):
        assert any({"O", "0"}.issubset(group) for group in AMBIGUOUS_GROUPS)

    def test_one_and_letter_l_grouped(self):
        assert any({"1", "l"}.issubset(group) for group in AMBIGUOUS_GROUPS)

    def test_five_and_s_grouped(self):
        assert any({"5", "S"}.issubset(group) for group in AMBIGUOUS_GROUPS)


class TestCharacterPrediction:
    def test_not_ambiguous_without_alternatives(self):
        prediction = CharacterPrediction(char="A", confidence=0.9)
        assert prediction.is_ambiguous is False

    def test_ambiguous_when_scores_close(self):
        prediction = CharacterPrediction(char="O", confidence=0.60, alternatives=[("0", 0.55)])
        assert prediction.is_ambiguous is True

    def test_not_ambiguous_when_scores_far(self):
        prediction = CharacterPrediction(char="A", confidence=0.95, alternatives=[("B", 0.40)])
        assert prediction.is_ambiguous is False

    def test_swap_candidates_includes_original(self):
        prediction = CharacterPrediction(char="O", confidence=0.7, alternatives=[("0", 0.65)])
        options = prediction.swap_candidates()
        assert "O" in [char for char, _ in options]

    def test_swap_candidates_adds_confusables(self):
        prediction = CharacterPrediction(char="O", confidence=0.7)
        options = prediction.swap_candidates()
        chars = [char for char, _ in options]
        assert "0" in chars

    def test_swap_candidates_sorted_by_confidence(self):
        prediction = CharacterPrediction(char="5", confidence=0.8, alternatives=[("S", 0.75)])
        options = prediction.swap_candidates()
        scores = [score for _, score in options]
        assert scores == sorted(scores, reverse=True)


class TestBuildCandidates:
    def test_empty_predictions(self):
        assert build_candidates([]) == []

    def test_single_candidate_when_confident(self):
        predictions = [
            CharacterPrediction(char="A", confidence=0.95),
            CharacterPrediction(char="B", confidence=0.93),
        ]
        results = build_candidates(predictions)
        assert len(results) == 1
        assert results[0].text == "AB"

    def test_multiple_candidates_when_ambiguous(self):
        predictions = [
            CharacterPrediction(char="O", confidence=0.6, alternatives=[("0", 0.58)]),
            CharacterPrediction(char="1", confidence=0.6, alternatives=[("l", 0.57)]),
        ]
        results = build_candidates(predictions, max_results=5)
        assert len(results) > 1

    def test_respects_max_results(self):
        predictions = [
            CharacterPrediction(char="O", confidence=0.6, alternatives=[("0", 0.59)]),
            CharacterPrediction(char="5", confidence=0.6, alternatives=[("S", 0.59)]),
            CharacterPrediction(char="1", confidence=0.6, alternatives=[("l", 0.59)]),
        ]
        results = build_candidates(predictions, max_results=3)
        assert len(results) <= 3

    def test_ranked_by_confidence(self):
        predictions = [
            CharacterPrediction(char="O", confidence=0.7, alternatives=[("0", 0.65)]),
            CharacterPrediction(char="5", confidence=0.7, alternatives=[("S", 0.65)]),
        ]
        results = build_candidates(predictions)
        confidences = [c.confidence for c in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_deduplicates_identical_text(self):
        predictions = [
            CharacterPrediction(char="A", confidence=0.6, alternatives=[("A", 0.59)]),
        ]
        results = build_candidates(predictions)
        texts = [c.text for c in results]
        assert len(texts) == len(set(texts))

    def test_averages_confidence_across_characters(self):
        predictions = [
            CharacterPrediction(char="A", confidence=1.0),
            CharacterPrediction(char="B", confidence=0.0),
        ]
        results = build_candidates(predictions)
        assert results[0].confidence == 0.5


class TestMergeCandidateLists:
    def test_empty_input(self):
        assert merge_candidate_lists([]) == []

    def test_single_list_passthrough(self):
        candidates = [Candidate(text="ABC", confidence=0.9)]
        merged = merge_candidate_lists([candidates])
        assert merged[0].text == "ABC"

    def test_agreement_boosts_confidence(self):
        list_a = [Candidate(text="ABC", confidence=0.7)]
        list_b = [Candidate(text="ABC", confidence=0.8)]
        merged = merge_candidate_lists([list_a, list_b])
        assert merged[0].text == "ABC"
        assert merged[0].confidence > 0.7

    def test_weights_favor_trusted_backend(self):
        weak = [Candidate(text="WRONG", confidence=0.9)]
        strong = [Candidate(text="RIGHT", confidence=0.8)]
        merged = merge_candidate_lists([weak, strong], weights=[0.5, 3.0])
        assert merged[0].text == "RIGHT"

    def test_respects_max_results(self):
        lists = [
            [Candidate(text=f"TEXT{i}", confidence=0.5) for i in range(10)],
        ]
        merged = merge_candidate_lists(lists, max_results=3)
        assert len(merged) == 3

    def test_disagreement_keeps_both(self):
        list_a = [Candidate(text="AAA", confidence=0.8)]
        list_b = [Candidate(text="BBB", confidence=0.8)]
        merged = merge_candidate_lists([list_a, list_b])
        assert len(merged) == 2
