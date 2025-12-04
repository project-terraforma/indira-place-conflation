import os
import json
import pandas as pd
from groq import Groq

# ============================================================
# GROQ CLIENT (auto-loads GROQ_API_KEY from environment)
# ============================================================
client = Groq()
MODEL = "llama-3.3-70b-versatile"

# ============================================================
# PROMPT TEMPLATE (freeform-only matching)
# ============================================================
PROMPT_TEMPLATE = """
You are deciding whether two POI records refer to the same physical place.

You will receive:
- {sim}: a similarity score between 0.0 and 1.0
- {pred}: a baseline predicted label (1 = same place, 0 = different)
- normalized address fields (all lowercase, expanded suffixes like "street", "avenue", "road")

Use this decision process:

1. PRIORITIZE "sim" AND "pred":
   - If pred = 1 and sim is high (>= 0.7), assume same_place = 1 unless the address clearly disagrees.
   - If pred = 0 and sim is low (<= 0.3), assume same_place = 0 unless the address clearly shows they match.

2. USE THE ADDRESS AS A CONFIRMATION CHECK (SECONDARY):
   - Address fields are already normalized.
   - Check:
       • street number
       • street name
       • city/locality
       • postcode
       • country
   - If the baseline decision from sim + pred conflicts with the address
     (e.g., different street numbers or different cities), flip the decision.

3. GENERAL RULE:
   - Similar address → supports same_place = 1
   - Very different address → supports same_place = 0
   - When uncertain, rely on sim + pred unless the address strongly contradicts it.
   - If all address components match return same_place=1 regardless of sim and pred.
   - if phone number is the exact same there is a very high chance of match


OUTPUT REQUIREMENTS:
- Return ONLY a JSON object.
- "same_place" must be 1 or 0.
- "confidence" must be a number between 0.0 and 1.0.
- "explanation" must be 1 sentence and MUST describe the specific fields
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

            sim = row.get("match_score", ""),
            pred = row.get("pred_label", ""),


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