import pandas as pd
import json
import re
from unidecode import unidecode
from typing import Any, Dict, List, Optional


# -------------------------------------------------------
# Abbreviation maps
# -------------------------------------------------------
STREET_ABBR = {
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bavda\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\brd\b": "road",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bpl\b": "place",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bctr\b": "center",
    r"\bcir\b": "circle",
    r"\bct\b": "court",
    r"\bsq\b": "square",
    r"\bter\b": "terrace"
}

DIRECTION_MAP = {
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
    r"\bne\b": "northeast",
    r"\bnw\b": "northwest",
    r"\bse\b": "southeast",
    r"\bsw\b": "southwest"
}

UNIT_ABBR = {
    r"\bapt\b": "apartment",
    r"\bsuite\b": "suite",
    r"\bste\b": "suite",
    r"\bunit\b": "unit",
    r"\bfl\b": "floor",
    r"\brm\b": "room",
    r"#": "unit "
}

# Common business type abbreviations
BUSINESS_ABBR = {
    r"\bco\b": "company",
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bllc\b": "limited liability company",
    r"\bltd\b": "limited",
    r"\b&\b": "and"
}


# -------------------------------------------------------
# Safe JSON parsing
# -------------------------------------------------------
def safe_json_parse(value: Any) -> Any:
    """Safely parse JSON string or return original value."""
    if value is None or pd.isna(value):
        return None
    
    if isinstance(value, str):
        try:
            # Handle string representation of lists/dicts
            if value.startswith('[') or value.startswith('{'):
                return json.loads(value.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            pass
    
    return value


# -------------------------------------------------------
# Extract nested fields
# -------------------------------------------------------
def extract_primary_name(names_json: Any) -> str:
    """Extract primary name from names JSON."""
    parsed = safe_json_parse(names_json)
    if isinstance(parsed, dict):
        return str(parsed.get('primary', ''))
    return ''


def extract_primary_category(categories_json: Any) -> str:
    """Extract primary category from categories JSON."""
    parsed = safe_json_parse(categories_json)
    if isinstance(parsed, dict):
        return str(parsed.get('primary', ''))
    return ''


def extract_all_categories(categories_json: Any) -> List[str]:
    """Extract all categories (primary + alternates)."""
    parsed = safe_json_parse(categories_json)
    if isinstance(parsed, dict):
        cats = [parsed.get('primary', '')]
        alts = parsed.get('alternate', [])
        if alts:
            cats.extend(alts)
        return [c for c in cats if c]
    return []


def extract_first_address(addresses_json: Any) -> Dict[str, str]:
    """Extract first address with all components."""
    parsed = safe_json_parse(addresses_json)
    if isinstance(parsed, list) and len(parsed) > 0:
        addr = parsed[0]
        return {
            'freeform': addr.get('freeform', ''),
            'locality': addr.get('locality', ''),
            'region': addr.get('region', ''),
            'postcode': addr.get('postcode', ''),
            'country': addr.get('country', '')
        }
    return {'freeform': '', 'locality': '', 'region': '', 'postcode': '', 'country': ''}


def extract_first_phone(phones_json: Any) -> str:
    """Extract first phone number."""
    parsed = safe_json_parse(phones_json)
    if isinstance(parsed, list) and len(parsed) > 0:
        return str(parsed[0])
    return ''


def extract_first_website(websites_json: Any) -> str:
    """Extract first website."""
    parsed = safe_json_parse(websites_json)
    if isinstance(parsed, list) and len(parsed) > 0:
        return str(parsed[0])
    return ''


def extract_brand_name(brand_json: Any) -> str:
    """Extract brand primary name."""
    parsed = safe_json_parse(brand_json)
    if isinstance(parsed, dict):
        names = parsed.get('names', {})
        if isinstance(names, dict):
            return str(names.get('primary', ''))
    return ''


# -------------------------------------------------------
# Text normalization
# -------------------------------------------------------
def normalize_text(text: str, expand_business_terms: bool = False) -> str:
    """Normalize any text field for comparison."""
    if text is None or pd.isna(text) or text == '':
        return ""

    text = str(text).strip().lower()
    text = unidecode(text)  # remove accents
    
    # Remove common business suffixes that don't affect identity
    text = re.sub(r'\b(llc|inc|corp|ltd|co|company)\b\.?', '', text)
    
    # Remove punctuation but keep alphanumeric
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Expand abbreviations
    for pattern, repl in STREET_ABBR.items():
        text = re.sub(pattern, repl, text)
    
    for pattern, repl in DIRECTION_MAP.items():
        text = re.sub(pattern, repl, text)
    
    for pattern, repl in UNIT_ABBR.items():
        text = re.sub(pattern, repl, text)
    
    if expand_business_terms:
        for pattern, repl in BUSINESS_ABBR.items():
            text = re.sub(pattern, repl, text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_phone(phone: str) -> str:
    """Normalize phone number to digits only."""
    if phone is None or pd.isna(phone) or phone == '':
        return ""
    
    # Keep only digits
    phone = re.sub(r'\D', '', str(phone))
    
    # Remove leading country codes (1 for US, common patterns)
    if len(phone) > 10 and phone.startswith('1'):
        phone = phone[1:]
    
    return phone


def normalize_postcode(postcode: str) -> str:
    """Normalize postal code."""
    if postcode is None or pd.isna(postcode) or postcode == '':
        return ""
    
    postcode = str(postcode).strip().upper()
    # Remove spaces and dashes
    postcode = re.sub(r'[\s\-]', '', postcode)
    return postcode


def normalize_url(url: str) -> str:
    """Normalize URL for comparison."""
    if url is None or pd.isna(url) or url == '':
        return ""
    
    url = str(url).lower().strip()
    # Remove protocol
    url = re.sub(r'^https?://', '', url)
    # Remove www
    url = re.sub(r'^www\.', '', url)
    # Remove trailing slash
    url = url.rstrip('/')
    
    return url


# -------------------------------------------------------
# Main cleaning function
# -------------------------------------------------------
def clean_poi_data(data: List[Dict]) -> pd.DataFrame:
    """
    Clean POI data from list of dictionaries.
    
    Parameters:
        data: List of POI dictionaries
        
    Returns:
        DataFrame with cleaned columns (original columns are replaced with cleaned values)
    """
    df = pd.DataFrame(data)
    
    # ===== Extract PRIMARY/DATA fields =====
    df['name_primary'] = df['names'].apply(extract_primary_name)
    df['category_primary'] = df['categories'].apply(extract_primary_category)
    df['categories_all'] = df['categories'].apply(extract_all_categories)
    df['brand_name'] = df['brand'].apply(extract_brand_name)
    
    # Extract address components
    address_dict = df['addresses'].apply(extract_first_address)
    df['address_freeform'] = address_dict.apply(lambda x: x['freeform'])
    df['address_locality'] = address_dict.apply(lambda x: x['locality'])
    df['address_region'] = address_dict.apply(lambda x: x['region'])
    df['address_postcode'] = address_dict.apply(lambda x: x['postcode'])
    df['address_country'] = address_dict.apply(lambda x: x['country'])
    
    # Extract contact info
    df['phone'] = df['phones'].apply(extract_first_phone)
    df['website'] = df['websites'].apply(extract_first_website)
    
    # ===== Extract BASE fields =====
    df['base_name_primary'] = df['base_names'].apply(extract_primary_name)
    df['base_category_primary'] = df['base_categories'].apply(extract_primary_category)
    df['base_categories_all'] = df['base_categories'].apply(extract_all_categories)
    df['base_brand_name'] = df['base_brand'].apply(extract_brand_name)
    
    # Extract base address components
    base_address_dict = df['base_addresses'].apply(extract_first_address)
    df['base_address_freeform'] = base_address_dict.apply(lambda x: x['freeform'])
    df['base_address_locality'] = base_address_dict.apply(lambda x: x['locality'])
    df['base_address_region'] = base_address_dict.apply(lambda x: x['region'])
    df['base_address_postcode'] = base_address_dict.apply(lambda x: x['postcode'])
    df['base_address_country'] = base_address_dict.apply(lambda x: x['country'])
    
    # Extract base contact info
    df['base_phone'] = df['base_phones'].apply(extract_first_phone)
    df['base_website'] = df['base_websites'].apply(extract_first_website)
    
    # ===== Clean and REPLACE PRIMARY/DATA fields =====
    df['name_primary'] = df['name_primary'].apply(lambda x: normalize_text(x, expand_business_terms=True))
    df['brand_name'] = df['brand_name'].apply(lambda x: normalize_text(x, expand_business_terms=True))
    df['category_primary'] = df['category_primary'].apply(normalize_text)
    
    # Clean address components (replace originals)
    df['address_freeform'] = df['address_freeform'].apply(normalize_text)
    df['address_locality'] = df['address_locality'].apply(normalize_text)
    df['address_region'] = df['address_region'].apply(normalize_text)
    df['address_postcode'] = df['address_postcode'].apply(normalize_postcode)
    df['address_country'] = df['address_country'].apply(lambda x: str(x).upper() if x else '')
    
    # Create full address for comparison
    df['address_full'] = df.apply(
        lambda row: ' '.join(filter(None, [
            row['address_freeform'],
            row['address_locality'],
            row['address_region'],
            row['address_postcode']
        ])),
        axis=1
    )
    
    # Clean contact info (replace originals)
    df['phone'] = df['phone'].apply(normalize_phone)
    df['website'] = df['website'].apply(normalize_url)
    
    # ===== Clean and REPLACE BASE fields =====
    df['base_name_primary'] = df['base_name_primary'].apply(lambda x: normalize_text(x, expand_business_terms=True))
    df['base_brand_name'] = df['base_brand_name'].apply(lambda x: normalize_text(x, expand_business_terms=True))
    df['base_category_primary'] = df['base_category_primary'].apply(normalize_text)
    
    # Clean base address components (replace originals)
    df['base_address_freeform'] = df['base_address_freeform'].apply(normalize_text)
    df['base_address_locality'] = df['base_address_locality'].apply(normalize_text)
    df['base_address_region'] = df['base_address_region'].apply(normalize_text)
    df['base_address_postcode'] = df['base_address_postcode'].apply(normalize_postcode)
    df['base_address_country'] = df['base_address_country'].apply(lambda x: str(x).upper() if x else '')
    
    # Create full base address for comparison
    df['base_address_full'] = df.apply(
        lambda row: ' '.join(filter(None, [
            row['base_address_freeform'],
            row['base_address_locality'],
            row['base_address_region'],
            row['base_address_postcode']
        ])),
        axis=1
    )
    
    # Clean base contact info (replace originals)
    df['base_phone'] = df['base_phone'].apply(normalize_phone)
    df['base_website'] = df['base_website'].apply(normalize_url)
    
    # Convert confidence to float
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    df['base_confidence'] = pd.to_numeric(df['base_confidence'], errors='coerce')
    
    return df


# -------------------------------------------------------
# Main testing function
# -------------------------------------------------------
def main():
    """Test cleaning on data.parquet and show before/after for 3 rows."""
    
    try:
        # Load the parquet file
        print("Loading data.parquet...")
        df_raw = pd.read_parquet('../src/data/data.parquet')
        print(f"✓ Loaded {len(df_raw)} records\n")
        
        # Convert to list of dicts for cleaning
        data_list = df_raw.to_dict('records')
        
        # Clean the data
        print("Cleaning data...")
        df_clean = clean_poi_data(data_list)
        print(f"✓ Cleaned {len(df_clean)} records\n")
        
        # Show before/after for first 3 rows
        print("="*100)
        print("BEFORE & AFTER COMPARISON - FIRST 3 ROWS")
        print("="*100)
        
        for idx in range(min(3, len(df_clean))):
            print(f"\n{'='*100}")
            print(f"ROW {idx + 1}")
            print('='*100)
            
            row = df_clean.iloc[idx]
            
            # Name
            print(f"\n📍 NAME:")
            print(f"  Raw:     {row.get('name_primary', 'N/A')}")
            print(f"  Cleaned: {row.get('name_clean', 'N/A')}")
            
            # Brand
            if row.get('brand_name'):
                print(f"\n🏷️  BRAND:")
                print(f"  Raw:     {row.get('brand_name', 'N/A')}")
                print(f"  Cleaned: {row.get('brand_clean', 'N/A')}")
            
            # Category
            print(f"\n📂 CATEGORY:")
            print(f"  Raw:     {row.get('category_primary', 'N/A')}")
            print(f"  Cleaned: {row.get('category_clean', 'N/A')}")
            
            # Address
            print(f"\n🏠 ADDRESS:")
            print(f"  Raw Street:   {row.get('address_freeform', 'N/A')}")
            print(f"  Clean Street: {row.get('address_freeform_clean', 'N/A')}")
            print(f"  Raw City:     {row.get('address_locality', 'N/A')}")
            print(f"  Clean City:   {row.get('address_locality_clean', 'N/A')}")
            print(f"  Raw Postcode: {row.get('address_postcode', 'N/A')}")
            print(f"  Clean Postcode: {row.get('address_postcode_clean', 'N/A')}")
            print(f"  Country:      {row.get('address_country_clean', 'N/A')}")
            print(f"  Full Clean:   {row.get('address_full_clean', 'N/A')}")
            
            # Phone
            if row.get('phone'):
                print(f"\n📞 PHONE:")
                print(f"  Raw:     {row.get('phone', 'N/A')}")
                print(f"  Cleaned: {row.get('phone_clean', 'N/A')}")
            
            # Website
            if row.get('website'):
                print(f"\n🌐 WEBSITE:")
                print(f"  Raw:     {row.get('website', 'N/A')}")
                print(f"  Cleaned: {row.get('website_clean', 'N/A')}")
            
            # Metadata
            print(f"\n📊 METADATA:")
            print(f"  ID:         {row.get('id', 'N/A')}")
            print(f"  Label:      {row.get('label', 'N/A')}")
            print(f"  Confidence: {row.get('confidence', 'N/A')}")
        
        print(f"\n{'='*100}")
        print(f"\n✓ Cleaning completed successfully!")
        print(f"  Total records processed: {len(df_clean)}")
        print(f"  Total columns: {len(df_clean.columns)}")
        
        # Show column summary
        print(f"\n📋 CLEANED COLUMNS:")
        clean_cols = [col for col in df_clean.columns if '_clean' in col]
        for col in sorted(clean_cols):
            non_empty = df_clean[col].astype(bool).sum()
            print(f"  • {col:30s} ({non_empty}/{len(df_clean)} non-empty)")
        
        return df_clean
        
    except FileNotFoundError:
        print("❌ Error: data.parquet not found!")
        print("   Please make sure data.parquet is in the current directory.")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    df = main()