import os
import json
import pandas as pd
from groq import Groq

# ============================================================
# GROQ CLIENT (auto-loads GROQ_API_KEY from environment)
# ============================================================
client = Groq()
MODEL = "llama-3.1-8b-instant"

# ============================================================
# PROMPT TEMPLATE (freeform-only matching)
# ============================================================
PROMPT_TEMPLATE = """
You are an expert at determining whether two business records refer to the same
physical location. Analyze the freeform addresses carefully and return a JSON object.

OUTPUT REQUIREMENTS:
- Return ONLY a JSON object.
- "same_place" must be 1 or 0.
- "confidence" must be a number between 0.0 and 1.0.
- "explanation" must be 1–2 sentences and MUST describe the specific fields
  that led to the decision.
- Do NOT reuse generic phrases across answers.


============================================================
RECORD A:
Address (freeform): {address_freeform}
Locality / City:    {address_locality}
Region / State:     {address_region}
Postcode:           {address_postcode}
Country:            {address_country}
Phone:              {phone}

============================================================
RECORD B:
Address (freeform): {base_address_freeform}
Locality / City:    {base_address_locality}
Region / State:     {base_address_region}
Postcode:           {base_address_postcode}
Country:            {base_address_country}
Phone:              {base_phone}

============================================================
MATCHING RULES YOU MUST FOLLOW:
1. Street number + street name are the strongest indicators.
2. Suite / unit / floor differences DO NOT matter (still same place).
3. ZIP+4 vs ZIP standard is the SAME.
4. City and region should match or be extremely close.
5. Phone numbers CAN suggest same place but cannot override address mismatches.
6. Different street numbers OR different street names → almost always different place.
7. Explanations MUST reference the exact fields that matched or mismatched.

Return ONLY JSON.
"""

# ============================================================
# CALL GROQ LLM
# ============================================================
def call_llm(prompt: str):
    """Call Groq model and return extracted JSON."""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        resp = completion.choices[0].message.content.strip()

        # Extract JSON block
        if "{" in resp and "}" in resp:
            text = resp[resp.index("{"):resp.rindex("}") + 1]
            try:
                return json.loads(text)
            except Exception:
                return {
                    "same_place": 0,
                    "confidence": 0.0,
                    "explanation": "Invalid JSON returned"
                }

        return {"same_place": 0, "confidence": 0.0, "explanation": "No JSON extracted"}

    except Exception as e:
        return {
            "same_place": 0,
            "confidence": 0.0,
            "explanation": f"LLM error: {e}"
        }


# ============================================================
# PROCESS CSV
# ============================================================
def process_csv(input_csv="output.csv", output_csv="llm_predictions.csv"):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")

    # Ensure prediction columns exist
    df["llm_same_place"] = 0
    df["llm_confidence"] = 0.0
    df["llm_reason"] = ""

    for i, row in df.iterrows():
        # prompt = PROMPT_TEMPLATE.format(
        #     address=row.get("address_freeform", ""),
        #     base_address=row.get("base_address_freeform", "")
        # )
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
        )



        res = call_llm(prompt)

        df.at[i, "llm_same_place"] = res.get("same_place", 0)
        df.at[i, "llm_confidence"] = res.get("confidence", 0.0)
        df.at[i, "llm_reason"] = res.get("explanation", "")

        # 🔥 Write to CSV ON EVERY SINGLE ROW
        df.iloc[: i+1].to_csv(output_csv, index=False)

        print(f"[{i}/{len(df)}] → {res}")

    # Final save
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    # ========================================================
    # Accuracy Evaluation
    # ========================================================
    if "label" in df.columns:
        acc = (df["label"].astype(int) == df["llm_same_place"].astype(int)).mean()
        print(f"\n=== LLM ACCURACY ===")
        print(f"Accuracy: {acc:.4f}")

    return df


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    process_csv()



# IMPORTANT DECISION RULES:
# 1. Street number AND street name must both match (or clearly refer to the same
#    location) for a strong "same place" decision.
# 2. unit, floor, room, department DO NOT matter.
# 3. Suite/unit differences DO matter.
# 3. Brand, business name, phone, category, and website DO NOT matter.
# 4. If street numbers or street names differ significantly, default to "different place."
# 5. Explanations MUST reference the specific matches or mismatches found.