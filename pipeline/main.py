from dataclean import clean_poi_data
from fuzzmatch import score_dataframe
import pandas as pd

# Load parquet file
df_raw = pd.read_parquet("../src/data/data.parquet")
df_raw.to_csv("raw.csv", index=False)
# Clean POI data
print("\nCleaning data...")
# Clean
data_list = df_raw.to_dict(orient="records")
df_clean = clean_poi_data(data_list)

# Score + label
df_scored = score_dataframe(df_clean)
print(df_scored[['address_freeform', 'base_address_freeform', 'match_score', 'label', 'pred_label']].head())

accuracy = (df_scored['label'] == df_scored['pred_label']).mean()
print("Accuracy:", accuracy)

# df_scored.to_csv("output.csv", index=False)

df = df_scored
# df = df.drop(["id", "base_id", "sources", "base_sources"], axis=1)
df = df.drop([
    "id",
    "base_id",
    "sources",
    "names",
    "categories",
    "confidence",
    "websites",
    "socials",
    "emails",
    "phones",
    "brand",
    "addresses",
    "base_sources",
    "base_names",
    "base_categories",
    "base_confidence",
    "base_websites",
    "base_socials",
    "base_emails",
    "base_phones",
    "base_brand",
    "base_addresses",
    "name_primary",
    "category_primary",
    "categories_all",
    "brand_name",
    "website",
    "base_name_primary",
    "base_category_primary",
    "base_categories_all",
    "base_brand_name",
    "base_website",
    "address_full",
    "base_address_full",
], axis=1)
df.to_csv("output.csv", index=False)


