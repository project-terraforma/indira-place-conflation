"""
Simplified Entity Matching Evaluation Framework
Optimized for small embedding models (MiniLM) instead of LLMs.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
)

from sentence_transformers import SentenceTransformer, util


# ============================================================================
# MODEL CONFIG
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for embedding-based models only."""
    name: str
    provider: str        # only "huggingface" supported now
    model_id: str
    api_key: str         # unused for local MiniLM
    max_tokens: int = 0  # not used but kept for compatibility


def load_models_config(config_path: str = "models_config.json") -> List[ModelConfig]:
    """Load HuggingFace models only."""
    with open(config_path, "r") as f:
        config = json.load(f)

    models = []
    for m in config["models"]:
        if m["provider"] != "huggingface":
            print(f"⚠️ Skipping model {m['name']} (provider {m['provider']} not supported).")
            continue

        models.append(ModelConfig(
            name=m["name"],
            provider=m["provider"],
            model_id=m["model_id"],
            api_key=os.getenv(m.get("api_key_env", ""), ""),
        ))

    return models


# ============================================================================
# EMBEDDING-BASED MATCHING
# ============================================================================

def embed_pair(model, row) -> float:
    """Compute cosine similarity between two place descriptions."""
    
    text_a = f"{row['name_a']} {row['addr_a']} {row['cat_a']}"
    text_b = f"{row['name_b']} {row['addr_b']} {row['cat_b']}"
    
    emb = model.encode([text_a, text_b], convert_to_tensor=True)
    similarity = util.cos_sim(emb[0], emb[1]).item()
    return similarity


def predict_from_similarity(similarity: float, threshold: float = 0.60) -> int:
    """Convert cosine similarity to binary MATCH/NO_MATCH."""
    return 1 if similarity >= threshold else 0


# ============================================================================
# EVALUATION
# ============================================================================

@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_similarity: float
    threshold: float
    n_samples: int
    confusion_matrix: List[List[int]]

    def to_dict(self):
        return asdict(self)


def evaluate_embedding_model(
    config: ModelConfig,
    test_data: pd.DataFrame,
    threshold: float = 0.60,
    max_samples: Optional[int] = None
):
    print(f"\n====================================================")
    print(f"Evaluating model: {config.name}")
    print("====================================================")

    if max_samples:
        test_data = test_data.sample(n=max_samples, random_state=42)
        print(f"Using {max_samples} samples.")

    # Load embedding model
    print("Loading model:", config.model_id)
    model = SentenceTransformer(config.model_id)

    similarities = []
    predictions = []
    ground_truth = []

    start_time = time.time()

    for _, row in test_data.iterrows():
        sim = embed_pair(model, row)
        similarities.append(sim)

        pred = predict_from_similarity(sim, threshold)
        predictions.append(pred)

        ground_truth.append(row["label"])

    elapsed_ms = (time.time() - start_time) * 1000 / len(test_data)

    # Metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    cm = confusion_matrix(ground_truth, predictions).tolist()

    result = EvaluationResult(
        model_name=config.name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        avg_similarity=float(np.mean(similarities)),
        threshold=threshold,
        n_samples=len(test_data),
        confusion_matrix=cm
    )

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Avg similarity: {result.avg_similarity:.3f}")
    print(f"Avg time per sample: {elapsed_ms:.1f} ms")

    print("Confusion Matrix:", cm)

    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class EntityMatchingEvaluator:
    def __init__(self, config_path):
        self.config_path = config_path
        self.models = []
        self.results = []

    def load_models(self):
        self.models = load_models_config(self.config_path)
        print("\nModels loaded:")
        for m in self.models:
            print(" •", m.name)
        return self.models

    def run_evaluation(
        self,
        test_data: pd.DataFrame,
        max_samples: Optional[int] = None
    ):
        self.load_models()

        self.results = []
        for config in self.models:
            res = evaluate_embedding_model(config, test_data, max_samples=max_samples)
            self.results.append(res)

        # Save
        with open("embedding_results.json", "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

        print("\nSaved results to embedding_results.json")
        return self.results


# ============================================================================
# SCRIPT ENTRYPOINT
# ============================================================================

def main():
    print("\nENTITY MATCHING WITH MINI-LM\n")

    evaluator = EntityMatchingEvaluator("models_config.json")
    
    df = pd.read_parquet("data/data.parquet")

    # Choose correct columns
    df = df.rename(columns={
        "names": "name_a",
        "base_names": "name_b",
        "base_addresses": "addr_b",
        "categories": "cat_a",
        "base_categories": "cat_b",
    })

    df["addr_a"] = ""  # No address_a in your dataset

    test_df = df[[
        "name_a", "name_b", "addr_a", "addr_b", "cat_a", "cat_b", "label"
    ]]

    evaluator.run_evaluation(test_df, max_samples=200)


if __name__ == "__main__":
    main()
