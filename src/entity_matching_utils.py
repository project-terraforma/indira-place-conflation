# ============================================================
# entity_matching_utils.py
# Cleaned / simplified hybrid similarity utilities
# Matches KEEP_COLS from main file
# ============================================================

import json as json_lib
from urllib.parse import urlparse
import re
import rapidfuzz
from rapidfuzz import fuzz



# ============================================================
# JSON + Parsing Helpers
# ============================================================

def safe_json_load(x):
    """Safely loads JSON or returns None."""
    if x is None:
        return None
    try:
        return json_lib.loads(x)
    except Exception:
        return None


def extract_primary(item):
    """Extract 'primary' field from JSON-like string."""
    d = safe_json_load(item)
    if isinstance(d, dict):
        return d.get("primary")
    return None



# ============================================================
# Normalization Helpers
# ============================================================

def extract_domain(urls):
    """Extract first domain from website list."""
    try:
        arr = safe_json_load(urls)
        if isinstance(arr, list) and len(arr) > 0:
            domain = urlparse(arr[0]).netloc
            return domain.replace("www.", "").lower()
    except:
        return None
    return None


def normalize_phone(phone):
    """Extract digits from phone number."""
    try:
        arr = safe_json_load(phone)
        if isinstance(arr, list):
            phone = arr[0]
        return re.sub(r"\D", "", phone)
    except:
        return None


def address_to_text(addr_json):
    """Convert structured address JSON into a readable text string."""
    try:
        arr = safe_json_load(addr_json)
        if isinstance(arr, list) and len(arr) > 0:
            a = arr[0]
            parts = [
                a.get("freeform", ""),
                a.get("locality", ""),
                a.get("postcode", ""),
                a.get("region", ""),
                a.get("country", "")
            ]
            return ", ".join([p for p in parts if p])
    except:
        pass
    return ""



# ============================================================
# Fuzzy Similarity Functions (name / address / category / phone / website)
# ============================================================

def name_similarity(n1, n2):
    n1 = extract_primary(n1) or ""
    n2 = extract_primary(n2) or ""
    return fuzz.token_set_ratio(n1, n2) / 100


def address_similarity(a1, a2):
    t1 = address_to_text(a1)
    t2 = address_to_text(a2)
    return rapidfuzz.fuzz.partial_ratio(t1, t2) / 100


def category_similarity(c1, c2):
    c1 = extract_primary(c1) or ""
    c2 = extract_primary(c2) or ""
    return fuzz.ratio(c1, c2) / 100


def phone_similarity(p1, p2):
    p1 = normalize_phone(p1)
    p2 = normalize_phone(p2)
    if not p1 or not p2:
        return 0
    return 1 if p1 == p2 else 0


def website_similarity(w1, w2):
    d1 = extract_domain(w1)
    d2 = extract_domain(w2)
    if not d1 or not d2:
        return 0
    return 1 if d1 == d2 else 0

def postcode_similarity(a1, a2):
    return 1.0 if a1 == a2 and a1 != "" else 0.0

def locality_similarity(a1, a2):
    return 1.0 if a1.lower() == a2.lower() and a1 != "" else 0.0

def country_similarity(a1, a2):
    return 1.0 if a1.lower() == a2.lower() and a1 != "" else 0.0

def region_similarity(a1, a2):
    return 1.0 if a1.lower() == a2.lower() and a1 != "" else 0.0

def street_similarity(s1, s2):
    return fuzz.partial_ratio(s1, s2) / 100




# ============================================================
# Structured Text for Embedding Model
# ============================================================

def build_structured_text(row, prefix="_a"):
    """Constructs a clean natural-language document for embeddings."""
    name = extract_primary(row["name" + prefix])
    category = extract_primary(row["cat" + prefix])
    address = address_to_text(row["addr" + prefix])
    phone = normalize_phone(row["phone" + prefix])
    website = extract_domain(row["web" + prefix])

    parts = []
    if name: parts.append(f"Name: {name}")
    if address: parts.append(f"Address: {address}")
    if category: parts.append(f"Category: {category}")
    if phone: parts.append(f"Phone: {phone}")
    if website: parts.append(f"Website Domain: {website}")

    return "\n".join(parts)



# ============================================================
# Hybrid Similarity (MANUAL WEIGHTS ONLY)
# ============================================================

MANUAL_WEIGHTS = {
    "embedding": 0.55,
    "name":      0.20,
    "address":   0.10,
    "category":  0.10,
    "website":   0.08,
    "phone":     0.07,
    "bias":     -0.15
}


def hybrid_similarity(emb_sim, row):
    """Return weighted similarity with *manual weights only*."""
    w = MANUAL_WEIGHTS

    n = name_similarity(row["name_a"], row["name_b"])
    a = address_similarity(row["addr_a"], row["addr_b"])
    c = category_similarity(row["cat_a"], row["cat_b"])
    p = phone_similarity(row["phone_a"], row["phone_b"])
    wv = website_similarity(row["web_a"], row["web_b"])

    return (
        w["embedding"] * emb_sim +
        w["name"]      * n +
        w["address"]   * a +
        w["category"]  * c +
        w["website"]   * wv +
        w["phone"]     * p +
        w["bias"]
    )

def parse_address(addr_json):
    """Return dictionary with detailed address components."""
    arr = safe_json_load(addr_json)
    if isinstance(arr, list) and len(arr) > 0:
        a = arr[0]
        return {
            "freeform": a.get("freeform", "") or "",
            "locality": a.get("locality", "") or "",
            "region": a.get("region", "") or "",
            "postcode": a.get("postcode", "") or "",
            "country": a.get("country", "") or ""
        }
    return {"freeform": "", "locality": "", "region": "", "postcode": "", "country": ""}

