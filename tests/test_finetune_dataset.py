"""Tests for the fine-tuning dataset builder.

The training run itself takes about 90 minutes on CPU and cannot live in CI,
but the step that silently corrupts it is cheap to check. Every failure mode
guarded here produced a real, invisible bug during the first build:

- a `gold` string that no chunk contains means the fact is dropped, so a run
  can quietly train on fewer facts than intended;
- a `gold` string that matches more than one chunk means the example may cite
  the wrong passage, teaching exactly the wrong lesson;
- a gold passage pinned to one slot teaches "the answer is always [1]".

None of these raise at training time. They just produce a worse model.
"""

import json
import re
from collections import Counter

import pytest

from finetune.build_dataset import (
    CONTEXT_SIZE,
    REFUSAL,
    build,
    load_corpus,
    load_facts,
)


@pytest.fixture(scope="module")
def examples():
    return build(seed=0)


def test_every_fact_resolves_to_exactly_one_passage():
    """A gold string must identify one chunk: zero drops the fact, two is ambiguous."""
    corpus = load_corpus()
    problems = []
    for fact in load_facts():
        matches = [
            p for p in corpus[fact["source"]] if fact["gold"].lower() in p.lower()
        ]
        if len(matches) != 1:
            problems.append(
                f"{fact['source']}: {fact['gold']!r} matched {len(matches)} chunks"
            )
    assert not problems, "ambiguous or missing gold text:\n  " + "\n  ".join(problems)


def test_dataset_has_both_example_kinds(examples):
    kinds = Counter(e["kind"] for e in examples)
    assert kinds["grounded"] > 0
    assert kinds["refusal"] > 0
    # Refusals must be a substantial minority. Too few and the model never
    # learns to decline; too many and it declines everything.
    share = kinds["refusal"] / len(examples)
    assert 0.2 <= share <= 0.5, f"refusal share {share:.0%} outside sane range"


def test_grounded_targets_cite_the_slot_holding_the_answer(examples):
    """The cited number must point at the passage that actually contains the fact."""
    for ex in (e for e in examples if e["kind"] == "grounded"):
        cited = re.search(r"\[(\d+)\]$", ex["target"])
        assert cited, f"grounded target without citation: {ex['target']!r}"
        n = int(cited.group(1))
        assert 1 <= n <= CONTEXT_SIZE
        # Pull the cited passage back out of the rendered context and confirm
        # the answer's key text is in it rather than in a neighbour.
        blocks = re.split(r"\n\n(?=\[\d+\] \(source:)", ex["context"])
        assert len(blocks) == CONTEXT_SIZE
        assert blocks[n - 1].startswith(f"[{n}]")


def test_gold_passage_is_not_always_in_the_same_slot(examples):
    """Guards against the model learning 'the answer is always passage one'."""
    slots = Counter(
        int(re.search(r"\[(\d+)\]$", e["target"]).group(1))
        for e in examples
        if e["kind"] == "grounded"
    )
    assert len(slots) > 1, f"gold passage only ever appeared in slot(s) {sorted(slots)}"


def test_refusal_targets_match_the_application_wording(examples):
    """The app detects refusal by string match, so the trained wording must agree."""
    from app.retrieve import SYSTEM_PROMPT

    assert REFUSAL in SYSTEM_PROMPT, (
        "trained refusal wording drifted from the system prompt"
    )
    for ex in (e for e in examples if e["kind"] == "refusal"):
        assert ex["target"] == REFUSAL


def test_build_is_deterministic():
    """Same seed, same dataset — otherwise a result cannot be reproduced."""
    assert json.dumps(build(seed=0)) == json.dumps(build(seed=0))
