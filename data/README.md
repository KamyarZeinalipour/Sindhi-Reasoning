# Sindhi-Reasoning-Instruct data

| File | Items | Passages |
|---|---:|---:|
| `sindhi_reasoning_train.jsonl.gz` | 2,089 | 965 |
| `sindhi_reasoning_test.jsonl.gz` | 100 | 96 |

Each item asks a question about a Sindhi passage that must be answered with one of four
reasoning types:

| Type | What the question asks | Train | Test |
|---|---|---:|---:|
| `deductive` | apply a general rule to a specific case | 567 | 30 |
| `inductive` | derive a general rule from specific cases | 573 | 24 |
| `causal` | identify a cause and its effect | 474 | 24 |
| `commonsense` | infer something from everyday knowledge | 475 | 22 |

This is the split used in the paper. The same data is on the Hugging Face Hub as
[`Kamyar-zeinalipour/sindi_reasoning`](https://huggingface.co/datasets/Kamyar-zeinalipour/sindi_reasoning).

## Loading

```python
import pandas as pd

test = pd.read_json("data/sindhi_reasoning_test.jsonl.gz", lines=True)
row = test.iloc[0]
print(row.reasoning_type, "|", row.question)
print(row.answer)
```

## Fields

| Field | Description |
|---|---|
| `id`, `split` | Row id and split |
| `reasoning_type` | `deductive`, `inductive`, `causal` or `commonsense` |
| `passage` | The Sindhi source passage (median 194 words) |
| `question` | The reasoning question (instruction) in Sindhi |
| `context` | The fact or excerpt from the passage the question builds on |
| `answer` | Reference answer: a short explanation followed by the answer (median 54 words) |
| `messages` | The chat used for fine-tuning and evaluation: system prompt, user prompt (passage + reasoning type + question) and the assistant reply as JSON `{"final_answer": ...}` |

## How the data was made

Passages come from Sindhi Wikipedia, English Wikipedia articles on science, history and
sociology (machine-translated, then post-edited by native Sindhi linguists), and the Sindhi
newspaper Daily Kawish (about 20%). GPT-4 generated instruction–context–output triplets from
them with two prompt templates, one for inductive/deductive and one for causal/commonsense
reasoning ([`assets/generation_prompt.png`](../assets/generation_prompt.png)). Two Sindhi
linguists then checked every item in two passes, logical validity first (about 12% discarded or
rewritten) and then orthography and grammar.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Part of the source text comes
from a newspaper, so the data is for non-commercial research use.
