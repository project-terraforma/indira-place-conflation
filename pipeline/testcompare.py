import os
import json
import pandas as pd
from groq import Groq

# ============================================================
# MODELS TO TEST
# ============================================================
MODELS = [
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]

ROW_LIMIT = 200
client = Groq()

# ============================================================
# PROMPT TEMPLATE
# ============================================================
PROMPT_TEMPLATE = """
You are deciding whether two POI records refer to the same physical place.

You will receive:
- {sim}: similarity score between 0 and 1
- {pred}: baseline match prediction (1 = same place, 0 = different)
- fully normalized address fields (lowercase, expanded suffixes)

Decision rules:
1. Prioritize similarity and pred.
2. Use address as confirmation: street number, street name, city, postcode, country.
3. If all address components match → same_place = 1.
4. If cities or street numbers differ → same_place = 0.
5. Phone number equality strongly supports same_place = 1.

Output JSON with:
same_place (1/0)
confidence (0–1)
explanation (short sentence)

============================================================
RECORD A:
Address: {address_freeform}
City:    {address_locality}
Region:  {address_region}
Postcode:{address_postcode}
Country: {address_country}
Phone:   {phone}

============================================================
RECORD B:
Address: {base_address_freeform}
City:    {base_address_locality}
Region:  {base_address_region}
Postcode:{base_address_postcode}
Country: {base_address_country}
Phone:   {base_phone}
============================================================

Return ONLY JSON.
"""


# ============================================================
# CALL LLM
# ============================================================
def call_llm(prompt, model):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        text = resp.choices[0].message.content.strip()

        if "{" in text and "}" in text:
            json_str = text[text.index("{") : text.rindex("}") + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return {"same_place": 0, "confidence": 0.0, "explanation": "Invalid JSON"}
        else:
            return {"same_place": 0, "confidence": 0.0, "explanation": "No JSON returned"}

    except Exception as e:
        return {"same_place": 0, "confidence": 0.0, "explanation": f"LLM error: {e}"}


# ============================================================
# RUN EVALUATION FOR ONE MODEL
# ============================================================
def evaluate_model(model_name, df):
    df = df.copy()

    df["llm_same_place"] = 0
    df["llm_confidence"] = 0.0
    df["llm_reason"] = ""
    df["llm_correct"] = 0
    df["baseline_correct"] = (df["pred_label"] == df["label"]).astype(int)

    for i, row in df.iterrows():
        prompt = PROMPT_TEMPLATE.format(
            address_freeform=row.get("address_freeform", ""),
            address_locality=row.get("address_locality", ""),
            address_region=row.get("address_region", ""),
            address_postcode=row.get("address_postcode", ""),
            address_country=row.get("address_country", ""),
            phone=row.get("phone", ""),

            base_address_freeform=row.get("base_address_freeform", ""),
            base_address_locality=row.get("base_address_locality", ""),
            base_address_region=row.get("base_address_region", ""),
            base_address_postcode=row.get("base_address_postcode", ""),
            base_address_country=row.get("base_address_country", ""),
            base_phone=row.get("base_phone", ""),

            sim=row.get("match_score", 0),
            pred=row.get("pred_label", 0),
        )

        res = call_llm(prompt, model_name)

        df.at[i, "llm_same_place"] = res.get("same_place", 0)
        df.at[i, "llm_confidence"] = res.get("confidence", 0.0)
        df.at[i, "llm_reason"] = res.get("explanation", "")

        # correctness
        df.at[i, "llm_correct"] = int(res.get("same_place", 0) == row["label"])

        print(f"[{model_name}] Row {i+1}/{len(df)} → LLM={res.get('same_place')} Truth={row['label']}")

    # accuracy
    llm_acc = df["llm_correct"].mean()

    output_path = f"results_{model_name.replace('/', '_')}.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved results for {model_name} → {output_path}")
    print(f"Accuracy for {model_name}: {llm_acc:.4f}\n")

    return llm_acc


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("Loading data…")
    df = pd.read_csv("output.csv").iloc[:ROW_LIMIT].copy()
    print(f"Loaded {len(df)} rows for evaluation.\n")

    summary = {}

    for model in MODELS:
        print(f"\n==============================")
        print(f"Evaluating model: {model}")
        print(f"==============================\n")

        acc = evaluate_model(model, df)
        summary[model] = acc

    print("\n============== FINAL SUMMARY ==============")
    for m, acc in summary.items():
        print(f"{m}: {acc:.4f}")
