import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

# Only utils you need
from entity_matching_utils import (
    build_structured_text,
    hybrid_similarity,
)


# ============================================================
# MODEL CONFIG
# ============================================================

@dataclass
class ModelConfig:
    name: str
    provider: str
    model_id: str
    api_key: str
    max_tokens: int = 0


def load_models_config(path="models_config2.json"):
    with open(path, "r") as f:
        cfg = json.load(f)

    models = []
    for m in cfg["models"]:
        if m["provider"] != "huggingface":
            print(f"Skipping non-HF model: {m['name']}")
            continue
        models.append(
            ModelConfig(
                name=m["name"],
                provider="huggingface",
                model_id=m["model_id"],
                api_key=os.getenv(m.get("api_key_env", ""), "")
            )
        )
    return models


# ============================================================
# EMBEDDING SIMILARITY
# ============================================================

def embed_pair(model, row):
    text_a = build_structured_text(row, prefix="_a")
    text_b = build_structured_text(row, prefix="_b")
    emb = model.encode([text_a, text_b], convert_to_tensor=True)
    return float((emb[0] @ emb[1]) / (emb[0].norm() * emb[1].norm()))


# ============================================================
# EVALUATION (MANUAL WEIGHTS ONLY)
# ============================================================

def sweep_thresholds(df, scorer, label_col="label"):
    thresholds = np.arange(0.20, 0.95, 0.01)

    sims = np.array([scorer(row) for _, row in df.iterrows()])
    labels = df[label_col].to_numpy()

    best = {"threshold": None, "f1": -1}

    for th in thresholds:
        preds = (sims >= th).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)

        if f1 > best["f1"]:
            best["threshold"] = th
            best["f1"] = f1

    return best


def evaluate(config, df, threshold=0.65, max_samples=None):
    print(f"\nEvaluating model {config.name} @ threshold {threshold:.3f}")

    if max_samples:
        df = df.sample(n=max_samples, random_state=42)

    model = SentenceTransformer(config.model_id)

    preds = []
    labels = df.label.to_numpy()
    sims = []

    for _, row in df.iterrows():

        # embedding similarity
        emb_sim = embed_pair(model, row)

        # FINAL SCORE = hybrid_similarity(embedding, fields)
        final_score = hybrid_similarity(emb_sim, row)

        sims.append(final_score)
        preds.append(1 if final_score >= threshold else 0)

    from sklearn.metrics import (
        precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
    )

    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    acc       = accuracy_score(labels, preds)
    f1        = f1_score(labels, preds, zero_division=0)
    cm        = confusion_matrix(labels, preds).tolist()

    print(f"Acc={acc:.3f}, Prec={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    print("Confusion Matrix:", cm)

    return {
        "model": config.name,
        "threshold": threshold,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }

def make_manual_scorer(model):
    """Return a scorer(row) function using hybrid_similarity()."""
    def score(row):
        emb_sim = embed_pair(model, row)
        return hybrid_similarity(emb_sim, row)
    return score


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("\n🔥 ENTITY MATCHING — MANUAL HYBRID SCORING ONLY 🔥")

    df = pd.read_parquet("data/data.parquet")

    # Renames for consistent prefix
    df = df.rename(columns={
        "names": "name_a",
        "base_names": "name_b",
        "categories": "cat_a",
        "base_categories": "cat_b",
        "addresses": "addr_a",
        "base_addresses": "addr_b",
        "phones": "phone_a",
        "base_phones": "phone_b",
        "websites": "web_a",
        "base_websites": "web_b",
        "brand": "brand_a",
        "base_brand": "brand_b",
    })

    # Only keep relevant fields
    KEEP_COLS = [
        "label",
        "name_a", "name_b",
        "cat_a", "cat_b",
        "addr_a", "addr_b",
        "phone_a", "phone_b",
        "web_a", "web_b",
        "brand_a", "brand_b",
    ]

    df = df[KEEP_COLS].copy()

    configs = load_models_config()
    results = []

    

    for cfg in configs:
        print(f"\n=== Evaluating {cfg.name} ===")

        # load embedding model once
        model = SentenceTransformer(cfg.model_id)

        # build real scorer
        scorer = make_manual_scorer(model)

        # find optimal threshold
        best = sweep_thresholds(df, scorer)
        print(f"Best threshold for {cfg.name}: {best}")

        # evaluate with that threshold
        res = evaluate(
            cfg,
            df,
            threshold=best["threshold"],
            max_samples=300
        )

        results.append(res)


    with open("hybrid_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved → hybrid_results2.json")


if __name__ == "__main__":
    main()
