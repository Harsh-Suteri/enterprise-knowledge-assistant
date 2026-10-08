"""LoRA fine-tune of a small instruct model, on CPU.

Why LoRA and not a full fine-tune: the base model's weights stay frozen and
only two small low-rank matrices per attention projection are trained. For
SmolLM2-135M that is roughly a percent of the parameters, which is the
difference between a run that fits on a laptop and one that does not. The
output is a few megabytes of adapter rather than a copy of the model, so it
can be versioned next to the code and swapped per corpus.

Why not QLoRA: 4-bit quantised training needs bitsandbytes, which needs CUDA.
This machine has no NVIDIA GPU, so the honest choice is fp32 LoRA on a model
small enough to train without one. The method is the same; only the
quantisation step is missing.

Implementation notes that matter:

prompt masking  loss is computed on the assistant's reply only. Labels for the
                system and user tokens are set to -100. Training on the prompt
                as well teaches the model to generate context passages, which
                is not the task and wastes most of the gradient budget.
gradient accum  true batch size 1 keeps peak memory low; gradients are
                accumulated to reach the effective batch size, so the update
                is as stable as a real batch without the memory cost.
length filter   over-long examples are dropped rather than truncated, because
                a truncated example can cut the answer off the end and teach
                the model to stop mid-sentence.

Usage
-----
    py finetune/train_lora.py                    # train with defaults
    py finetune/train_lora.py --max-steps 5      # smoke test the loop
    py finetune/train_lora.py --epochs 2 --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "data" / "train.jsonl"
OUT = ROOT / "out" / "adapter"

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_examples(tok, max_len: int) -> list[dict]:
    """Tokenise each chat example into input_ids plus prompt-masked labels."""
    rows = []
    dropped = 0
    with TRAIN.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            prompt_msgs, answer = messages[:-1], messages[-1]["content"]

            # Render then tokenise in two explicit steps. apply_chat_template
            # returns a formatted string by default in transformers 5.x and
            # token ids in 4.x; going via text makes the result the same
            # either way.
            prompt_text = tok.apply_chat_template(
                prompt_msgs, add_generation_prompt=True, tokenize=False
            )
            prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = (
                list(prompt_ids) + tok(answer, add_special_tokens=False)["input_ids"]
            )
            full_ids.append(tok.eos_token_id)

            if len(full_ids) > max_len:
                dropped += 1
                continue

            # -100 is the ignore index: no loss on the prompt, all of it on
            # the reply the model is supposed to learn to produce.
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            rows.append({"input_ids": full_ids, "labels": labels})

    if dropped:
        print(f"  dropped {dropped} examples longer than {max_len} tokens")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument(
        "--accum", type=int, default=8, help="gradient accumulation steps"
    )
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument(
        "--max-steps", type=int, default=0, help="stop early (smoke test)"
    )
    parser.add_argument("--threads", type=int, default=0, help="0 = torch default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"loading {args.model} (threads={torch.get_num_threads()})")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.config.use_cache = False  # incompatible with gradient flow; saves memory

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Attention projections plus the MLP. Adapting the MLP as well as
        # attention matters here: the behaviour being taught is partly a
        # formatting and decision change ("refuse" vs "answer"), which lives
        # as much in the feed-forward blocks as in attention.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"  trainable {trainable / 1e6:.2f}M / {total / 1e6:.1f}M  ({trainable / total:.2%})"
    )

    print("tokenising")
    rows = load_examples(tok, args.max_len)
    print(
        f"  {len(rows)} examples, median length "
        f"{sorted(len(r['input_ids']) for r in rows)[len(rows) // 2]} tokens"
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    steps_per_epoch = math.ceil(len(rows) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(0.05 * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    print(
        f"training {args.epochs} epoch(s): {total_steps} optimiser steps "
        f"(accum={args.accum}, lr={args.lr})"
    )
    model.train()
    started = time.time()
    step = 0
    running: list[float] = []
    done = False

    for epoch in range(args.epochs):
        if done:
            break
        random.shuffle(rows)
        optim.zero_grad()
        for i, row in enumerate(rows):
            input_ids = torch.tensor([row["input_ids"]])
            labels = torch.tensor([row["labels"]])
            loss = model(input_ids=input_ids, labels=labels).loss
            (loss / args.accum).backward()
            running.append(loss.item())

            if (i + 1) % args.accum == 0 or i == len(rows) - 1:
                for g in optim.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optim.step()
                optim.zero_grad()
                step += 1

                if step % 5 == 0 or step == 1:
                    elapsed = time.time() - started
                    rate = elapsed / step
                    print(
                        f"  step {step:>4}/{total_steps}  loss {sum(running) / len(running):.4f}"
                        f"  {rate:.1f}s/step  eta {(total_steps - step) * rate / 60:.0f}m"
                    )
                    running = []
                if args.max_steps and step >= args.max_steps:
                    done = True
                    break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(
        f"\nsaved adapter to {out} ({size:.1f} MB) in {(time.time() - started) / 60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
