# Teaching a 135M model the grounding contract, with LoRA

## The question

The application's system prompt asks a model to do three things: answer only
from the numbered passages it is given, cite the passage it used, and when the
passages do not contain the answer, say exactly

> I don't have enough information to answer that.

Hosted models do this reliably. `llama3.2:3b` on Ollama does it too. Anything
much smaller does not — a 135M model will happily invent an answer rather than
refuse, which on a confidential corpus is the worst failure mode available,
because it is silent.

So: **can that behaviour be trained into a model small enough to run anywhere,
rather than prompted into a model large enough to already have it?**

That is worth answering because the behaviour, not the knowledge, is what RAG
actually needs from the generator. The facts come from retrieval.

## Why LoRA, and why not QLoRA

LoRA freezes the base weights and trains a pair of low-rank matrices beside
each target projection. Here that is **4.88M trainable parameters out of
139.4M — 3.5%**. The artefact is a small adapter rather than a second copy of
the model, so a corpus-specific behaviour can be versioned next to the code
and swapped at load time.

QLoRA would be the obvious next step, and it is what most job descriptions
actually ask for. It needs `bitsandbytes`, which needs CUDA. The machine this
was trained on is an **Intel i5-10310U with no NVIDIA GPU**, so 4-bit
quantised training was not available. The honest version of the experiment on
this hardware is fp32 LoRA on a model small enough to train without a GPU —
the method and the failure modes are the same, the quantisation step is the
part that is missing.

## The design decision that makes the result mean anything

The obvious way to run this experiment is to train on the corpus and then
evaluate on it. That measures memorisation, and it would score well for the
wrong reason.

Instead:

| | documents | used for |
|---|---|---|
| **train** | `finetune/corpus/` — travel, procurement and system-access policies for three fictional companies | 312 generated examples |
| **test** | `data/sample_hr_policy.md` — the existing HR handbook | 15 answerable + 12 unanswerable questions |

The model never sees the HR policy during training. It cannot have memorised
a single fact in the test set. Any improvement has to come from the
*behaviour* generalising to an unseen document, which is the only thing worth
claiming.

## The training data

`build_dataset.py` chunks the training documents with the application's own
splitter (`app.ingest.split`), so training passages have the same shape and
boundaries as the ones the app produces at runtime. From 52 hand-written facts
it generates two kinds of example:

- **grounded** (208) — the gold passage sits among four distractors. Target is
  a short answer plus a citation.
- **refusal** (104) — the *same question*, with the gold passage withheld.
  Target is the exact refusal sentence.

Those two together are the whole point. The matched pair teaches that the
answer depends on the context in front of the model, not on what it happens to
remember.

Two details that would quietly ruin the run if skipped:

- **The gold passage rotates through every slot.** Pin it at `[1]` and the
  model learns "the answer is always passage one" — which scores well against
  a careless eval and breaks the moment retrieval reorders.
- **Loss is masked to the assistant's reply.** Labels for the system and user
  tokens are `-100`. Train on the prompt too and most of the gradient budget
  goes into learning to generate context passages, which is not the task.

`build_dataset.py` also fails loudly when a fact's `gold` string is not found
in the chunked document — four were caught that way on the first run
(markdown bold markers and line breaks splitting the match), and one more that
matched two different chunks. Those would have become silently mistargeted
training examples.

## Measuring it

`eval/run_eval.py` measures retrieval and cannot move here: a fine-tune does
not change which passage is fetched. `finetune/eval_generation.py` measures
the half that happens afterwards, with four numbers rather than one:

- **answer_accuracy** — expected text appears in the answer.
- **citation_rate** — the answer cites a passage. An uncited answer is
  unverifiable even when it is right.
- **refusal_accuracy** — unanswerable questions that produced the refusal
  sentence. This is the safety number.
- **false_refusal_rate** — answerable questions refused anyway. Refusal
  accuracy is trivially 100% for a model that refuses everything, so it only
  means something read against this.

Retrieval is real, not gold-injected: context comes from the same
`VectorStore` the app uses, so an unanswerable question gets the five nearest
irrelevant chunks exactly as it would in production. Generation is greedy
(`do_sample=False`) so the numbers are reproducible.

## Results

Two epochs, 78 optimiser steps, **76 minutes on 4 CPU cores**. Training loss
fell 1.76 → 0.20 and converged smoothly. The adapter is 19.6 MB.

All numbers below are on `data/sample_hr_policy.md`, which is **not in the
training set**.

| metric | base | + LoRA | delta |
|---|---|---|---|
| answer accuracy | 47% | 47% | — |
| citation rate | 7% | **67%** | **+60 pts** |
| refusal accuracy | 0% | **100%** | **+100 pts** |
| false refusal rate | 0% | **33%** | **+33 pts** (worse) |
| p50 latency | 4724 ms | **1889 ms** | **2.5× faster** |

### What actually happened

**The refusal behaviour transferred completely.** 0% → 100% on a document the
model never trained on. Before, asked who the CEO of Acme Technologies is —
a person the handbook never names — it answered:

> The current Chief Executive Officer of Acme Technologies is **John Smith**.

A fabricated name, stated confidently. After, every one of the twelve
unanswerable questions returns the exact refusal sentence. That is the result
worth having: the facts were always going to come from retrieval, and what
the generator has to contribute is knowing when it has nothing.

**The output format collapsed into something usable.** Base answers rambled
for 64 tokens, re-stated the question, and drifted into invented Q&A
scaffolding. Tuned answers are `26 weeks. [1]` and
`The Chief Financial Officer. [3]`. That is where the 2.5× latency drop comes
from — not a faster model, a model that stops when it is done.

**Flat accuracy hides the real move.** 47% → 47% looks like nothing happened.
It is not:

|  | base | + LoRA |
|---|---|---|
| questions attempted | 15 / 15 | 10 / 15 |
| correct | 7 | 7 |
| **precision when it answers** | **47%** | **70%** |

The tuned model answers two-thirds as often and gets a far higher share right.
It traded recall for precision, which on a confidential corpus is the correct
direction — but it was not a free trade, and the next section is the cost.

### The regression, stated plainly

**False refusal went 0% → 33%.** Five answerable questions now get "I don't
have enough information" when the answer was sitting in the retrieved context:

- How many days do I have to request a refund? (14 days, present in context)
- How many days per week can I work remotely? (2 days)
- How long does it take to process an approved refund? (7 working days)
- Is a receipt required for small expense claims?
- Can I store company data on my personal laptop?

This is the classic failure mode of training refusal behaviour, and the reason
`false_refusal_rate` exists in the eval at all: a model that refuses
everything scores 100% refusal accuracy. With one third of the dataset being
refusal examples, the model learned "refuse" as a slightly-too-cheap default.
The fix is a dataset change, not more training — raise the grounded:refusal
ratio and add hard negatives where the answer is present but paraphrased, so
"the wording doesn't match" stops being evidence of absence.

### Where the metric is unfair to the model

One of the three remaining wrong answers is not wrong:

| question | expected | produced |
|---|---|---|
| How many unused annual leave days can be carried forward? | `10 unused days` | `10 days. [2]` |

That is the right answer, scored as a miss because `eval_generation.py` does
substring matching — the same limitation already noted for the retrieval
harness. Counting it, precision when answering is **80%, not 70%**. The two
genuine errors are the notice period (gave the below-manager figure, 30 days,
instead of 60) and the remote-work core hours (`12 months`, nonsense).

The honest summary: **substring matching understates this model, and the
headline accuracy number is the least informative one in the table.**

### What this does and does not show

It shows that a grounding and refusal contract can be taught to a 135M model
with 3.5% of its parameters trainable, on a laptop with no GPU, and that the
behaviour generalises to a document outside the training set.

It does not show that this model is ready to serve answers. 47% accuracy is
not production quality, 33% false refusal is worse than the baseline on that
axis, and one document with 27 questions is a small test. The honest claim is
about the *method* working and being measurable, not about the artefact being
good.

The obvious next runs, in order of expected value: fix the refusal ratio and
retrain; repeat on a 0.5B model where the headroom is larger; and swap
substring scoring for something that can tell a correct paraphrase from a
miss.

## Reproducing it

```bash
py finetune/build_dataset.py --stats
py finetune/eval_generation.py --json finetune/out/baseline.json
py finetune/train_lora.py --epochs 2 --accum 8 --threads 8
py finetune/eval_generation.py --adapter finetune/out/adapter --json finetune/out/tuned.json
py finetune/compare.py finetune/out/baseline.json finetune/out/tuned.json
```

The adapter is not committed — it is reproducible from one command, and the
eval result JSONs are the part that constitutes evidence.
