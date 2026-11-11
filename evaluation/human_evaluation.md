# Human evaluation protocol

For each of the six models, 100 test items were shown with two answers: the base
(off-the-shelf) model and the fine-tuned model, in random order (the order was swapped for
about half of the items). Native Sindhi annotators

1. rated each answer on a five-point scale, and
2. chose the preferred answer.

| Rating | Meaning |
|---|---|
| A (Excellent) | Comprehensive understanding; precise, cohesive and contextually relevant, with no grammatical errors |
| B (Good) | Accurate and comprehensive; minor errors in flow or clarity that do not affect the main idea |
| C (Average) | Partially accurate or slightly incomplete; may contain grammatical errors or ambiguous phrasing |
| D (Poor) | Mostly incorrect or hard to understand; significant logic or grammar problems |
| E (Unacceptable) | Illogical, hallucinated or unrelated |

For averages the ratings map to A = 5 … E = 1. The paper reports inter-annotator agreement of κ = 0.78 for ratings and κ = 0.82 for
preferences.

Aggregated results (mean ratings, relative gain, preference rate, rating counts; overall and per
reasoning type) are in [`paper_results/human_eval.csv`](../paper_results/human_eval.csv).
