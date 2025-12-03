import pandas as pd

def evaluate_llm_predictions(
    csv_file="llm_predictions.csv",
    incorrect_csv="incorrect_predictions.csv"
):
    # Load data
    df = pd.read_csv(csv_file)

    # Required columns
    required_cols = {"label", "llm_same_place", "llm_confidence", "llm_reason"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required_cols}")

    # Extract true and predicted values
    y_true = df["label"].astype(int)
    y_pred = df["llm_same_place"].astype(int)

    # Identify incorrect predictions
    incorrect_mask = y_true != y_pred
    incorrect = df.loc[incorrect_mask, ["llm_confidence", "llm_reason"]].copy()
    incorrect["label"] = y_true[incorrect_mask].values

    # Save incorrect predictions
    incorrect.to_csv(incorrect_csv, index=False)
    print(f"Saved {len(incorrect)} incorrect predictions → {incorrect_csv}")

    # Compute metrics
    accuracy = (y_true == y_pred).mean()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    # Print metrics
    print("\n=======================================")
    print("        LLM EVALUATION METRICS         ")
    print("=======================================")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("---------------------------------------")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("=======================================\n")

    # Show first 5 incorrect rows
    print("=========== FIRST 5 INCORRECT ==========")
    if incorrect.empty:
        print("🎉 Perfect match — no incorrect predictions!")
    else:
        print(incorrect.head(5))
    print("========================================\n")

    return accuracy, precision, recall, f1, incorrect

if __name__ == "__main__":
    evaluate_llm_predictions()
