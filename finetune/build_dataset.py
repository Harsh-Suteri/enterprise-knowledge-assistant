"""Turn the training policy documents into grounded-QA training examples.

The thing being taught here is a *behaviour*, not a set of facts:

    answer only from the numbered passages, cite the passage you used, and
    when the passages do not contain the answer, refuse in exactly the words
    the application expects.

That distinction drives the whole design. If the model were trained on the
same document it is later evaluated on, a good score would prove only that it
memorised the document. So the training corpus here (finetune/corpus) is three
policy documents from three fictional companies that the model never sees at
eval time, and the evaluation runs against data/sample_hr_policy.md, which is
deliberately absent from training. A gain on that held-out document can only
come from the behaviour generalising.

Two example types are generated:

grounded  the gold passage is present among the distractors. Target is a short
          answer plus the citation of whichever slot the gold passage landed in.
refusal   the gold passage is removed and only distractors remain. Target is
          the exact refusal sentence from app.retrieve.SYSTEM_PROMPT.

Both matter, and the second matters more. A model that answers well but cannot
say "I don't know" is worse than useless on a confidential corpus, because the
failure is silent.

The gold passage is placed in a different slot each time. Without that the
model learns "the answer is always [1]" — which scores well on a careless eval
and is wrong the moment retrieval reorders.

Usage
-----
    py finetune/build_dataset.py              # writes finetune/data/train.jsonl
    py finetune/build_dataset.py --stats      # also print a breakdown
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import split
from app.retrieve import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
FACTS = ROOT / "facts"
OUT = ROOT / "data" / "train.jsonl"

REFUSAL = "I don't have enough information to answer that."

# How many distractor passages accompany the gold one. Matches settings.top_k
# so the training prompts look like the prompts the app actually builds.
CONTEXT_SIZE = 5

# Each fact becomes this many grounded examples, each with the gold passage in
# a different slot and a different set of distractors.
GROUNDED_PER_FACT = 4
# ...and this many refusal examples, where the gold passage is withheld.
REFUSAL_PER_FACT = 2


def load_corpus() -> dict[str, list[str]]:
    """Chunk every training document with the application's own splitter.

    Reusing app.ingest.split rather than re-implementing chunking means the
    passages the model trains on have the same shape and boundaries as the
    passages it will be handed in production.
    """
    corpus: dict[str, list[str]] = {}
    for path in sorted(CORPUS.glob("*.md")):
        corpus[path.name] = [
            c.text for c in split(path.read_text(encoding="utf-8"), path.name)
        ]
    return corpus


def load_facts() -> list[dict]:
    facts = []
    for path in sorted(FACTS.glob("*.jsonl")):
        source = path.name.replace(".jsonl", ".md")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    item = json.loads(line)
                    item["source"] = source
                    facts.append(item)
    return facts


def format_context(passages: list[tuple[str, str]]) -> str:
    """Render passages exactly as app.retrieve._format_context does."""
    return "\n\n".join(
        f"[{i + 1}] (source: {source})\n{text}"
        for i, (source, text) in enumerate(passages)
    )


def build(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    corpus = load_corpus()
    facts = load_facts()

    # Flat pool of (source, text) used to draw distractors.
    pool = [(src, text) for src, texts in corpus.items() for text in texts]

    examples: list[dict] = []
    unmatched: list[str] = []

    for fact in facts:
        passages = corpus[fact["source"]]
        gold = next((p for p in passages if fact["gold"].lower() in p.lower()), None)
        if gold is None:
            # A fact whose gold string is not in the chunked document is a bug
            # in the fact file, not a training example. Collect and report
            # rather than silently training on a broken target.
            unmatched.append(f"{fact['source']}: {fact['gold']!r}")
            continue
        gold_entry = (fact["source"], gold)
        others = [p for p in pool if p[1] != gold]

        for n in range(GROUNDED_PER_FACT):
            distractors = rng.sample(others, CONTEXT_SIZE - 1)
            slot = n % CONTEXT_SIZE  # rotate the gold passage through every slot
            context = distractors[:slot] + [gold_entry] + distractors[slot:]
            examples.append(
                {
                    "kind": "grounded",
                    "source": fact["source"],
                    "question": fact["question"],
                    "context": format_context(context),
                    "target": f"{fact['answer']} [{slot + 1}]",
                }
            )

        for _ in range(REFUSAL_PER_FACT):
            # Same question, gold passage withheld. This is the pair that
            # teaches the model the answer depends on the context in front of
            # it and not on what it happens to remember.
            distractors = rng.sample(others, CONTEXT_SIZE)
            examples.append(
                {
                    "kind": "refusal",
                    "source": fact["source"],
                    "question": fact["question"],
                    "context": format_context(distractors),
                    "target": REFUSAL,
                }
            )

    if unmatched:
        print("WARNING: gold text not found in chunked document:", file=sys.stderr)
        for u in unmatched:
            print(f"  - {u}", file=sys.stderr)

    rng.shuffle(examples)
    return examples


def to_chat(example: dict) -> dict:
    """Wrap an example in the same message structure the app sends at runtime."""
    return {
        "kind": example["kind"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{example['context']}\n\nQuestion: {example['question']}",
            },
            {"role": "assistant", "content": example["target"]},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    examples = build(seed=args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(to_chat(ex), ensure_ascii=False) + "\n")

    grounded = sum(1 for e in examples if e["kind"] == "grounded")
    refusal = len(examples) - grounded
    print(f"wrote {len(examples)} examples to {OUT.relative_to(ROOT.parent)}")
    print(f"  grounded: {grounded}   refusal: {refusal}")

    if args.stats:
        by_source: dict[str, int] = {}
        for e in examples:
            by_source[e["source"]] = by_source.get(e["source"], 0) + 1
        for src, n in sorted(by_source.items()):
            print(f"  {src:<28} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
