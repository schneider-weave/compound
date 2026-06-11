import math

from molsearch.scorer import BoltzScorer


def test_mock_mode_is_deterministic() -> None:
    scorer = BoltzScorer(
        mode="mock",
        command_template="",
        timeout_seconds=1,
        mock_score=True,
        target={},
    )
    s1 = scorer.score("rxn:2:1:2", "CCO")
    s2 = scorer.score("rxn:2:1:2", "CCO")
    assert s1 == s2


def test_flatten_target_values_nested() -> None:
    flat = BoltzScorer._flatten_target_values(
        {"name": "p53", "meta": {"chain-id": "A", "site": 42}}
    )
    assert flat["target_name"] == "p53"
    assert flat["target_meta_chain_id"] == "A"
    assert flat["target_meta_site"] == 42


def test_command_mode_without_target_returns_nan() -> None:
    scorer = BoltzScorer(
        mode="command",
        command_template="python -c \"print('score: 0.2')\"",
        timeout_seconds=1,
        mock_score=False,
        target={},
    )
    assert math.isnan(scorer.score("rxn:2:1:2", "CCO"))


def test_extract_score_from_noisy_output_prefers_labeled_score() -> None:
    output = """
    [14:09:28] DEPRECATION WARNING: please use MorganGenerator
    some progress text 223.79it/s
    score: 0.1234
    """
    assert BoltzScorer._extract_score(output) == 0.1234


def test_extract_score_from_nested_json_affinity_key() -> None:
    output = '{"result": {"affinity_pred_value": 1.2345, "other": 0}}'
    assert BoltzScorer._extract_score(output) == 1.2345
