from pathlib import Path
import pandas as pd
import numpy as np


def clean_fragrantica(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Trim whitespace in all object columns
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(
        lambda s: s.astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
    )

    # Convert rating_value to numeric (comma → dot)
    df["rating_value"] = (
        df["rating_value"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df["rating_value"] = pd.to_numeric(df["rating_value"], errors="coerce")

    # Ensure rating_count is numeric and valid
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce").astype("Int64")

    # Clean Year Column (nullable integer)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df.loc[(df["year"] < 1700) | (df["year"] > 2026), "year"] = pd.NA
    df["year"] = df["year"].astype("Int64")

    # Standardize "unknown" in perfumers
    for col in ["perfumer1", "perfumer2"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"unknown": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA, "": pd.NA})
        )

    # Clean Gender Column
    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    valid_genders = {"men", "women", "unisex"}
    df.loc[~df["gender"].isin(valid_genders), "gender"] = pd.NA
    df["gender"] = df["gender"].astype("category")

    # Standardize country + brand text
    df["brand"] = df["brand"].str.lower()
    df["country"] = df["country"].str.strip()

    # Handle Perfume+Brand duplicates (keep highest rating_count)
    before = df.shape[0]
    df = df.sort_values(["brand", "perfume", "rating_count"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["brand", "perfume"], keep="first")
    after = df.shape[0]
    removed_dupes = before - after

    # Reorder columns
    ordered_cols = [
        "url", "perfume", "brand", "country", "gender",
        "rating_value", "rating_count", "year",
        "top", "middle", "base",
        "perfumer1", "perfumer2",
        "mainaccord1", "mainaccord2", "mainaccord3", "mainaccord4", "mainaccord5"
    ]
    df = df[ordered_cols]

    return df, removed_dupes


def main():
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "raw" / "fragrantica_raw.csv"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    clean_path = processed_dir / "fragrantica_clean.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found at: {raw_path}")

    df_raw = pd.read_csv(raw_path, sep=";", encoding="ISO-8859-1")
    df_clean, removed_dupes = clean_fragrantica(df_raw)

    df_clean.to_csv(clean_path, index=False, encoding="utf-8")

    # Print quick summary
    print("Cleaning completed.")
    print(f"Raw shape: {df_raw.shape}")
    print(f"Clean shape: {df_clean.shape}")
    print(f"Duplicate (brand, perfume) removed: {removed_dupes}")
    print(f"Saved to: {clean_path}")

    # Missingness top 8
    missing_pct = (df_clean.isna().mean() * 100).sort_values(ascending=False).round(2).head(8)
    print("\nTop missing columns (%):")
    print(missing_pct)


if __name__ == "__main__":
    main()