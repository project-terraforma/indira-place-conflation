# POI Conflation via Hybrid Similarity Scoring and LLM Reasoning

[View the full presentation (PDF)](./presentation.pdf)

Indira Mariya - Places Conflation Model Evaluation.pdf

This project determines whether two Points of Interest (POI) refer to the same physical place. It combines deterministic similarity scoring with LLM-based structured reasoning and evaluates multiple models for accuracy and consistency.

## Overview

POI conflation requires deciding whether two independently sourced business/location records represent the same real-world address.
This system implements a binary classifier with two main components:

1. **Feature-Based Similarity Scoring**
   - Hybrid scoring across name, address, phone, website, and category
   - Uses RapidFuzz, Jaro–Winkler, TF-IDF, and custom address parsing
   - Produces:
     - `match_score` (0–1)
     - `pred_label` (1 = same place, 0 = different)

2. **LLM-Based Verification Layer**
   - LLM receives:
     - `sim` (match_score)
     - `pred` (baseline prediction)
     - normalized address fields
   - Confirms or overrides the baseline prediction
   - Outputs:
     - `same_place`
     - `confidence`
     - one-sentence explanation referencing specific fields

This hybrid method allows deterministic scores to handle easy cases efficiently while the LLM resolves ambiguous cases.

## Pipeline

### 1. Data Cleaning
- Normalize names, categories, phone numbers, websites
- Expand street suffixes (street, avenue, road)
- Produce normalized full address strings
- Saves cleaned results to `output.csv`

### 2. Similarity Scoring (`fuzzmatch.py`)
- Name similarity from RapidFuzz + Jaro–Winkler + TF-IDF
- Address similarity with custom parsing (street number, street name, unit handling)
- Exact match comparison for phone, website domain, and category
- Produces:
  - `match_score`
  - `pred_label`

### 3. LLM Evaluation (`main.py`, `test.py`)
- LLM receives:
  - similarity score
  - baseline prediction
  - normalized address fields
- LLM applies rules:
  - prioritize sim + pred
  - use address as confirmation
  - override if addresses disagree
- Produces:
  - `llm_same_place`
  - `llm_confidence`
  - `llm_reason`
- Output saved to `llm_predictions.csv`

### 4. Evaluation (`check.py`, `benchmark.py`)
- Compute accuracy comparing LLM predictions to ground truth
- Generate confusion matrix
- Export incorrect rows for error analysis:
  - `incorrect_predictions.csv`
  - `incorrect_ml_predictions.csv`
  - `fails.csv`

## LLM Models Tested

All models were tested via on-demand API access:

- `llama-3.3-70b-versatile`
- `moonshotai/kimi-k2-instruct-0905`
- `moonshotai/kimi-k2-instruct`
- `meta-llama/llama-4-scout-17b-16e-instruct`
- `llama-3.1-8b-instant`

Each model was evaluated for JSON reliability, address consistency, handling of similarity scores, and overall accuracy.

## Repository Structure
`pipeline/`

     check.py
     dataclean.py
     fuzzmatch.py
     llm.py
     main.py
     output.csv
     test.py


## Summary

This project provides a reproducible POI conflation workflow combining:

- Deterministic similarity scoring  
- LLM-based structured decision making  
- Multi-model evaluation and error analysis  

This approach improves robustness and interpretability when matching business/location records across heterogeneous datasets.
