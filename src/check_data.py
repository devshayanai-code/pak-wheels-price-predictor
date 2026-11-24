"""Quick first look at the raw dataset.

Run from the project root:  python -m src.check_data
"""
import sys

import pandas as pd

from src.config import RAW_DATA_PATH


def main() -> None:
    if not RAW_DATA_PATH.exists():
        sys.exit(f"Dataset not found at {RAW_DATA_PATH}. Download it first.")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}\n")
    print("First 5 rows:")
    print(df.head().to_string(), "\n")
    print("Column types:")
    df.info()
    print("\nMissing values per column:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
