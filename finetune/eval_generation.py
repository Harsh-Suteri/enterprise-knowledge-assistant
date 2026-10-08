"""Generation evaluation: does the model obey the grounding contract?

eval/run_eval.py measures retrieval — whether the right passage reaches the
model. This measures the half that happens afterwards: given passages, does
the model answer from them, cite them, and refuse when the answer is not
there. A fine-tune cannot move the retrieval number, so judging one by hit@k
would be meaningless.

Four numbers, because "accuracy" alone hides the failure that matters:

answer_accuracy  answerable questions where the expected text appears in the
                 answer.
citation_rate    answerable questions where the answer cites a passage, e.g.
                 [2]. An uncited answer is unverifiable even when correct.
refusal_accuracy unanswerable questions where the model produced the exact
                 refusal sentence. This is the safety number. On a
                 confidential corpus a confident invention is the worst
                 possible output, and it is the one a careless eval misses.
false_refusal    answerable questions the model refused anyway. Refusal
                 accuracy is trivially 100% for a model that refuses
                 everything, so it is only meaningful read against this.

Retrieval is real, not gold-injected: the context comes from the same
VectorStore the application uses, so an unanswerable question gets the five
nearest irrelevant chunks exactly as it would in production.

Usage
-----
    py finetune/eval_generation.py                          # base model
    py finetune/eval_generation.py --adapter finetune/out   # + LoRA adapter
    py finetune/eval_generation.py --adapter finetune/out --json results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieve import SYSTEM_PROMPT, _format_context
from app.store import VectorStore

ROOT = Path(__file__).resolve().parent.parent
ANSWERABLE = ROOT / "eval" / "questions.jsonl"
UNANSWERABLE = ROOT / "eval" / "unanswerable_questions.jsonl"

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REFUSAL_MARKER = "don't have enough information"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_store() -> VectorStore:
    """Index the held-out document the model was never trained on."""
    from app.ingest import ingest_dir

    store = VectorStore()
    store.reset()
    store.add(ingest_dir())
    return store


def load_model(model_id: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        # Folding the adapter into the base weights removes the per-layer
        # branch at inference, so the latency figure reflects what a deployed
        # merged model would actually cost.
        model = model.merge_and_unload()
    return tok, model.eval()


def generate(
    tok, model, question: str, passages: list[dict], max_new_tokens: int = 64
) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{_format_context(passages)}\n\nQuestion: {question}",
        },
    ]
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: the same context must give the same answer
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(
        out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def has_citation(text: str) -> bool:
    import re

    return bool(re.search(r"\[\d+\]", text))


def evaluate(
    model_id: str, adapter: str | None, k: int = 5, verbose: bool = False
) -> dict:
    store = build_store()
    tok, model = load_model(model_id, adapter)

    answerable = load_jsonl(ANSWERABLE)
    unanswerable = load_jsonl(UNANSWERABLE)

    latencies: list[float] = []
    correct = cited = false_refusals = 0
    answerable_rows = []

    for item in answerable:
        passages = store.search(item["question"], k=k)
        started = time.perf_counter()
        text = generate(tok, model, item["question"], passages)
        latencies.append((time.perf_counter() - started) * 1000)

        refused = REFUSAL_MARKER in text.lower()
        ok = item["expect"].lower() in text.lower()
        correct += ok
        cited += has_citation(text)
        false_refusals += refused
        answerable_rows.append(
            {**item, "output": text, "correct": ok, "refused": refused}
        )

    refused_ok = 0
    unanswerable_rows = []
    for item in unanswerable:
        passages = store.search(item["question"], k=k)
        started = time.perf_counter()
        text = generate(tok, model, item["question"], passages)
        latencies.append((time.perf_counter() - started) * 1000)

        refused = REFUSAL_MARKER in text.lower()
        refused_ok += refused
        unanswerable_rows.append({**item, "output": text, "refused": refused})

    n_a, n_u = len(answerable), len(unanswerable)
    result = {
        "model": model_id,
        "adapter": adapter,
        "answerable": n_a,
        "unanswerable": n_u,
        "answer_accuracy": correct / n_a,
        "citation_rate": cited / n_a,
        "false_refusal_rate": false_refusals / n_a,
        "refusal_accuracy": refused_ok / n_u,
        "p50_ms": statistics.median(latencies),
    }
    if verbose:
        result["answerable_rows"] = answerable_rows
        result["unanswerable_rows"] = unanswerable_rows
    return result


def print_report(r: dict) -> None:
    label = f"{r['model']}" + (f" + {r['adapter']}" if r["adapter"] else " (base)")
    print(f"\n{label}")
    print(
        f"  answer_accuracy     {r['answer_accuracy']:>6.0%}   ({r['answerable']} answerable)"
    )
    print(f"  citation_rate       {r['citation_rate']:>6.0%}")
    print(f"  false_refusal_rate  {r['false_refusal_rate']:>6.0%}   (lower is better)")
    print(
        f"  refusal_accuracy    {r['refusal_accuracy']:>6.0%}   ({r['unanswerable']} unanswerable)"
    )
    print(f"  p50 latency         {r['p50_ms']:>6.0f} ms")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter", default=None, help="path to a trained LoRA adapter"
    )
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--json", default=None, help="write full results here")
    parser.add_argument("--show", action="store_true", help="print every generation")
    args = parser.parse_args()

    r = evaluate(args.model, args.adapter, k=args.k, verbose=True)
    print_report(r)

    if args.show:
        print("\n  --- answerable ---")
        for row in r["answerable_rows"]:
            flag = "ok " if row["correct"] else "MISS"
            print(f"  [{flag}] {row['question']}\n         -> {row['output']!r}")
        print("\n  --- unanswerable (should refuse) ---")
        for row in r["unanswerable_rows"]:
            flag = "ok " if row["refused"] else "HALLUC"
            print(f"  [{flag}] {row['question']}\n         -> {row['output']!r}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
