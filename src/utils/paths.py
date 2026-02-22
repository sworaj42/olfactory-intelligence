import pandas as pd
from src.utils.paths import RAW_DIR


def main():
    file_path = RAW_DIR / "fra_cleaned.csv"

    print("Looking for file at:")
    print(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    print("\n===== BASIC INFO =====")
    print("Shape:", df.shape)

    print("\n===== COLUMNS =====")
    print(df.columns.tolist())

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== MISSING VALUES (top 20) =====")
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(20)
    if len(missing) == 0:
        print("No missing values found.")
    else:
        print(missing.to_string())

    print("\n===== DUPLICATES =====")
    if "url" in df.columns:
        print("Duplicate URLs:", df["url"].duplicated().sum())
    else:
        print("No 'url' column found.")

    print("\n===== SAMPLE ROWS =====")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()