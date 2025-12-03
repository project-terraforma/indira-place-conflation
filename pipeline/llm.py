import pandas as pd
import json
import ollama

MODEL = "phi3:mini"   # Small, fast, ~1.6GB

PROMPT_TEMPLATE = """
Determine if these two business records refer to the same physical location.

RULES:
- If the confidence is over 0.5 there is a good chance they are the same place.
- Focus on street number + street name.
- Same ZIP code = same general area.
- Suite/unit differences = SAME place.
- Ignore category, brand, phone, website differences.
- Use 0 for different places, 1 for same place.


Record A:
Name: {name}
Address: {address_full}
City: {locality}, {region} {postcode}
Country: {country}

Record B:
Name: {base_name}
Address: {base_address_full}
City: {base_locality}, {base_region} {base_postcode}
Country: {base_country}

Return ONLY JSON:
{{"same_place": 1, "confidence": 0.95, "explanation": "reason"}}
"""

def call_llm(prompt: str):
    """Call Ollama locally and extract JSON output."""
    resp = ollama.generate(
        model=MODEL,
        prompt=prompt,
        options={"temperature": 0}  # deterministic output
    )["response"]

    # Extract JSON (minimal, safe)
    if "{" in resp and "}" in resp:
        text = resp[resp.index("{"):resp.rindex("}") + 1]
        try:
            return json.loads(text)
        except:
            return {"same_place": 0, "confidence": 0.0, "explanation": "Bad JSON"}

    return {"same_place": 0, "confidence": 0.0, "explanation": "No JSON returned"}

def process_csv(input_csv="output.csv", output_csv="llm_predictions.csv"):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")

    df["llm_same_place"] = 0
    df["llm_confidence"] = 0.0
    df["llm_reason"] = ""

    for i, row in df.iterrows():
        prompt = PROMPT_TEMPLATE.format(
            name=row.get("name_primary", ""),
            address_full=row.get("address_full", ""),
            locality=row.get("address_locality", ""),
            region=row.get("address_region", ""),
            postcode=row.get("address_postcode", ""),
            country=row.get("address_country", ""),
            base_name=row.get("base_name_primary", ""),
            base_address_full=row.get("base_address_full", ""),
            base_locality=row.get("base_address_locality", ""),
            base_region=row.get("base_address_region", ""),
            base_postcode=row.get("base_address_postcode", ""),
            base_country=row.get("base_address_country", "")
        )

        res = call_llm(prompt)

        df.at[i, "llm_same_place"] = res.get("same_place", 0)
        df.at[i, "llm_confidence"] = res.get("confidence", 0.0)
        df.at[i, "llm_reason"] = res.get("explanation", "")

        if i % 10 == 0:
            print(f"[{i}/{len(df)}] → {res}")
            df.to_csv(output_csv, index=False)

    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    if "label" in df.columns:
        acc = (df["label"] == df["llm_same_place"]).mean()
        print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    process_csv()
