# ============================================================
# entity_weight_optimizer.py
# Simple, interpretable similarity feature weight estimator
# ============================================================

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from entity_matching_utils import (
    name_similarity,
    address_similarity,
    category_similarity,
    phone_similarity,
    website_similarity,
    build_structured_text,
)


# ------------------------------------------------------------
# Compute embedding similarity for a row
# ------------------------------------------------------------

def compute_embedding_similarity(model, row):
    text_a = build_structured_text(row, prefix="_a")
    text_b = build_structured_text(row, prefix="_b")
    emb = model.encode([text_a, text_b], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]))


# ------------------------------------------------------------
# Compute all similarity features
# ------------------------------------------------------------

def compute_feature_vector(model, row):
    return {
        "embedding": compute_embedding_similarity(model, row),
        "name":      name_similarity(row["name_a"], row["name_b"]),
        "address":   address_similarity(row["addr_a"], row["addr_b"]),
        "category":  category_similarity(row["cat_a"], row["cat_b"]),
        "phone":     phone_similarity(row["phone_a"], row["phone_b"]),
        "website":   website_similarity(row["web_a"], row["web_b"]),
    }


# ------------------------------------------------------------
# SIMPLE, INTERPRETABLE FEATURE WEIGHT LEARNING
# ------------------------------------------------------------

def learn_feature_weights(df, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Compute weights based on how well each feature separates
    positive matches vs negative matches.

    weight(feature) = mean(feature | match) - mean(feature | nonmatch)
    """

    print("\n🔍 Learning simple discriminative feature weights...")

    model = SentenceTransformer(model_id)

    all_feats = []
    all_labels = []

    for _, row in df.iterrows():
        feats = compute_feature_vector(model, row)
        all_feats.append(feats)
        all_labels.append(int(row["label"]))

    feat_df = pd.DataFrame(all_feats)
    labels = np.array(all_labels)

    # Compute discriminative signal for each feature
    weights = {}
    for col in feat_df.columns:
        pos_mean = feat_df[col][labels == 1].mean()
        neg_mean = feat_df[col][labels == 0].mean()
        diff = max(pos_mean - neg_mean, 0)  # ensure non-negative weight
        weights[col] = diff

    # Normalize weights to sum to 1
    total = sum(weights.values())
    if total == 0:
        total = 1  # avoid divide-by-zero
    weights = {k: v / total for k, v in weights.items()}

    print("\n📊 Learned weights (normalized):")
    for k, v in weights.items():
        print(f"  {k:10s} → {v:.3f}")

    return weights
