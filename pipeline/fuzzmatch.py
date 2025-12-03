import re
import numpy as np
from rapidfuzz import fuzz
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================
#  TF-IDF Setup
# ============================
tfidf = TfidfVectorizer(analyzer="word")


def tfidf_cosine(a: str, b: str) -> float:
    """Compute cosine similarity using TF-IDF."""
    if not a or not b:
        return 0.0
    try:
        tfidf_vec = tfidf.fit([a, b]).transform([a, b])
        v1, v2 = tfidf_vec[0].toarray()[0], tfidf_vec[1].toarray()[0]
        num = np.dot(v1, v2)
        den = np.linalg.norm(v1) * np.linalg.norm(v2)
        return (num / den) if den != 0 else 0.0
    except:
        return 0.0


# ============================
#  Best-in-class similarity functions
# ============================

def best_name_similarity(a: str, b: str) -> float:
    """Hybrid name similarity using several state-of-the-art metrics."""
    if not a or not b:
        return 0.0

    token_sort = fuzz.token_sort_ratio(a, b) / 100
    partial = fuzz.partial_ratio(a, b) / 100
    token_set = fuzz.token_set_ratio(a, b) / 100
    jw = jellyfish.jaro_winkler_similarity(a, b)
    tfidf_sim = tfidf_cosine(a, b)

    return (
        0.30 * token_sort +
        0.20 * partial +
        0.20 * token_set +
        0.15 * jw +
        0.15 * tfidf_sim
    )


def extract_number(s: str) -> str:
    """Extract first numeric component (street number)."""
    if not s:
        return ""
    m = re.search(r"\b\d+\b", str(s))
    return m.group(0) if m else ""


# def best_address_similarity(a: str, b: str) -> float:
#     """Hybrid address similarity: number match + fuzzy + JW."""
#     if not a or not b:
#         return 0.0

#     n1, n2 = extract_number(a), extract_number(b)
#     number_match = 1.0 if n1 == n2 and n1 else 0.0

#     full_sim = fuzz.token_sort_ratio(a, b) / 100
#     jw = jellyfish.jaro_winkler_similarity(a, b)

#     return (
#         0.40 * full_sim +
#         0.30 * jw +
#         0.30 * number_match
#     )

def parse_address(a):
    if not a:
        return {"num": "", "street": "", "unit": ""}

    # street number
    num = re.search(r"\b\d+\b", a)
    num = num.group(0) if num else ""

    # extract unit (#, apt, unit, ste)
    unit = re.search(r"(unit|ste|suite|apt|#)\s?([A-Za-z0-9]+)", a)
    unit = unit.group(0) if unit else ""

    # remove number + unit
    street = re.sub(r"\b\d+\b", "", a)
    street = re.sub(r"(unit|ste|suite|apt|#)\s?([A-Za-z0-9]+)", "", street)
    street = street.strip()

    return {"num": num, "street": street, "unit": unit}


def best_address_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    pa, pb = parse_address(a), parse_address(b)

    # Street name fuzzy match
    street_sim = fuzz.token_set_ratio(pa["street"], pb["street"]) / 100

    # House number exact match
    num_match = 1.0 if pa["num"] == pb["num"] and pa["num"] else 0.8 if pa["num"][:1] == pb["num"][:1] else 0.0

    # Unit differences should NOT penalize much
    unit_match = 1.0 if pa["unit"] == pb["unit"] else 0.95

    return (
        0.60 * street_sim +
        0.30 * num_match +
        0.10 * unit_match
    )



def best_phone_similarity(a: str, b: str) -> float:
    """Phones match exactly after cleaning."""
    if not a or not b:
        return 0.0
    return 1.0 if a == b else 0.0


def best_website_similarity(a: str, b: str) -> float:
    """Domain-level website matching."""
    if not a or not b:
        return 0.0
    a_dom = str(a).split("/")[0]
    b_dom = str(b).split("/")[0]
    return 1.0 if a_dom == b_dom else 0.0


def best_category_similarity(a: str, b: str) -> float:
    """Exact category match."""
    if not a or not b:
        return 0.0
    return 1.0 if a == b else 0.0


# ============================
#  Scoring & labeling
# ============================

def score_row(row) -> float:
    """Compute the match score for a single cleaned row."""
    # Updated column names - no _clean suffix
    return (
        0.45 * best_name_similarity(row["name_primary"], row["base_name_primary"]) +
        0.35 * best_address_similarity(row["address_full"], row["base_address_full"]) +
        0.10 * best_phone_similarity(row["phone"], row["base_phone"]) +
        0.05 * best_website_similarity(row["website"], row["base_website"]) +
        0.05 * best_category_similarity(row["category_primary"], row["base_category_primary"])
    )


def predict_label(score: float, threshold: float = 0.50) -> int:
    """Convert score into a binary label."""
    return 1 if score >= threshold else 0


# ============================
#  DataFrame scoring interface
# ============================

def score_dataframe(df_clean):
    """
    Given a cleaned POI DataFrame (output of your cleaner),
    compute match_score + predicted label.
    
    Expected columns from cleaning:
    - name_primary, base_name_primary (cleaned names)
    - address_full, base_address_full (cleaned full addresses)
    - phone, base_phone (cleaned phone numbers)
    - website, base_website (cleaned websites)
    - category_primary, base_category_primary (cleaned categories)
    """
    df_clean = df_clean.copy()
    df_clean["match_score"] = df_clean.apply(score_row, axis=1)
    df_clean["pred_label"] = df_clean["match_score"].apply(predict_label)
    return df_clean