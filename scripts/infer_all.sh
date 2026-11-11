#!/usr/bin/env bash
# Answer the test split with each model, before and after fine-tuning.
# Uses the released adapters from the Hugging Face Hub; set LOCAL=1 to use outputs/<model>.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Llama-3.1-8B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
  mistralai/Mistral-Nemo-Instruct-2407
  mistralai/Mistral-Small-24B-Instruct-2501
)

for model in "${MODELS[@]}"; do
  name=$(basename "$model")
  adapter="Kamyar-zeinalipour/Sindhi-Reasoning-${name}-LoRA"
  [[ "${LOCAL:-0}" == "1" ]] && adapter="outputs/$name"
  python inference/generate.py --model "$model" --output "results/raw/${name}_base.csv"
  python inference/generate.py --model "$model" --adapter "$adapter" --output "results/raw/${name}_finetuned.csv"
done
