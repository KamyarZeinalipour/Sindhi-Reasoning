"""Automatic metrics for Sindhi-Reasoning (Table 4 of the paper).

    python evaluation/metrics.py results/raw -o results/metrics

Compares each model reply (`generation`) with the gold answer (the assistant message, a JSON
object with `final_answer`) for every CSV written by inference/generate.py. Both texts are
truncated to 2,000 characters. Metrics:
  ROUGE-1/2/L (hazm tokenizer), BLEU (smoothing method 1), TF-IDF cosine,
  E5 cosine (intfloat/multilingual-e5-large), mean-pooled XLM-R cosine,
  BERTScore P/R/F1 (xlm-roberta-large, no baseline rescaling).
Writes per-row scores and `summary_metrics.csv` (one row per file).
"""
import argparse
from pathlib import Path

import hazm
import numpy as np
import pandas as pd
import torch
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHARS = 2000
XLMR = "FacebookAI/xlm-roberta-large"

_tok = hazm.WordTokenizer()


class _HazmTokenizer:
    def tokenize(self, text):
        return _tok.tokenize(text)


_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=_HazmTokenizer(), use_stemmer=False)
_smooth = SmoothingFunction().method1
_e5 = SentenceTransformer("intfloat/multilingual-e5-large").to(DEVICE)
_xlmr_tok = AutoTokenizer.from_pretrained(XLMR)
_xlmr = AutoModel.from_pretrained(XLMR).to(DEVICE).eval()


def tfidf_sim(ref, hyp):
    mat = TfidfVectorizer().fit_transform([" ".join(_tok.tokenize(ref)), " ".join(_tok.tokenize(hyp))])
    return float(cosine_similarity(mat[0:1], mat[1:2])[0][0])


def e5_sim(ref, hyp):
    emb = _e5.encode([ref, hyp], convert_to_numpy=True)
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])


@torch.no_grad()
def xlmr_sim(ref, hyp):
    def emb(text):
        inp = _xlmr_tok(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(DEVICE)
        return _xlmr(**inp).last_hidden_state.mean(dim=1).cpu().numpy()

    return float(cosine_similarity(emb(ref), emb(hyp))[0][0])


def score_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    refs = [str(m)[:MAX_CHARS] for m in df["target"]]
    hyps = [str(h)[:MAX_CHARS] for h in df["generation"].fillna("")]
    rows = []
    for ref, hyp in tqdm(zip(refs, hyps), total=len(refs), desc=path.name):
        r = _rouge.score(ref, hyp)
        rows.append({
            **{f"{k}_{a}": getattr(r[k], f) for k in ("rouge1", "rouge2", "rougeL")
               for a, f in (("p", "precision"), ("r", "recall"), ("f", "fmeasure"))},
            "bleu": sentence_bleu([_tok.tokenize(ref)], _tok.tokenize(hyp), smoothing_function=_smooth),
            "tfidf_sim": tfidf_sim(ref, hyp), "e5_sim": e5_sim(ref, hyp), "bert_sim": xlmr_sim(ref, hyp),
        })
    p, r, f = bert_score(refs, hyps, lang="sd", model_type="xlm-roberta-large", device=DEVICE, rescale_with_baseline=False)
    out = pd.DataFrame(rows)
    out["bertscore_p"], out["bertscore_r"], out["bertscore_f"] = p.tolist(), r.tolist(), f.tolist()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Folder of generation CSVs (or a single CSV).")
    ap.add_argument("-o", "--output", default="results/metrics")
    args = ap.parse_args()

    src, dst = Path(args.input), Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in sorted(src.glob("*.csv")) if src.is_dir() else [src]:
        scores = score_file(path)
        scores.to_csv(dst / f"{path.stem}_scores.csv", index=False)
        summary.append({"file": path.stem, **scores.mean(numeric_only=True).to_dict()})
    summary = pd.DataFrame(summary)
    summary.to_csv(dst / "summary_metrics.csv", index=False)
    print(summary[["file", "rougeL_f", "bleu", "tfidf_sim", "e5_sim", "bertscore_f"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
