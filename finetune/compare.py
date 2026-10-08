"""Print a before/after table from two eval_generation result files.

Kept separate from the eval so the numbers being compared are always two
recorded runs on disk, not one recorded run and one produced in memory. If a
comparison cannot be reproduced from two files that already exist, it is not
evidence.

Usage
-----
    py finetune/compare.py finetune/out/baseline.json finetune/out/tuned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# metric key, label, whether higher is better
METRICS = [
    ("answer_accuracy", "answer accuracy", True),
    ("citation_rate", "citation rate", True),
    ("refusal_accuracy", "refusal accuracy", True),
    ("false_refusal_rate", "false refusal rate", False),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()

    a = json.loads(Path(args.before).read_text(encoding="utf-8"))
    b = json.loads(Path(args.after).read_text(encoding="utf-8"))

    print(
        f"\nbefore: {a['model']}{' + ' + a['adapter'] if a['adapter'] else ' (base)'}"
    )
    print(f"after:  {b['model']}{' + ' + b['adapter'] if b['adapter'] else ' (base)'}")
    print(f"\n{'metric':<22}{'before':>9}{'after':>9}{'delta':>10}")
    print("-" * 50)
    for key, label, higher_better in METRICS:
        before, after = a[key], b[key]
        delta = after - before
        arrow = (
            ""
            if abs(delta) < 1e-9
            else (" +" if (delta > 0) == higher_better else " -")
        )
        print(f"{label:<22}{before:>8.0%}{after:>9.0%}{delta:>+9.0%}{arrow}")
    print(
        f"{'p50 latency (ms)':<22}{a['p50_ms']:>8.0f}{b['p50_ms']:>9.0f}"
        f"{b['p50_ms'] - a['p50_ms']:>+9.0f}"
    )
    print(
        f"\nheld-out document: data/sample_hr_policy.md "
        f"({a['answerable']} answerable, {a['unanswerable']} unanswerable)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
