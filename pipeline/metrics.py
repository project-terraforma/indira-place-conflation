import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(file):
    df = pd.read_csv(file)

    y_true = df["label"].astype(int)
    y_pred = df["llm_same_place"].astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    return acc, prec, rec, f1

files = [
    "results_llama-3.1-8b-instant.csv",
    "results_llama-3.3-70b-versatile.csv",
    "results_meta-llama_llama-4-scout-17b-16e-instruct.csv",
    "results_moonshotai_kimi-k2-instruct-0905.csv",
    "results_moonshotai_kimi-k2-instruct.csv",
]

for f in files:
    acc, prec, rec, f1 = evaluate(f)
    print("\n=== Results for", f, "===")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))
