"""
prepare.py
-----------
Reads the raw Kaggle CSV, cleans it up, and splits it into a training set
and a test set.

Input : data/raw/data.csv
Output: data/processed/train.csv
        data/processed/test.csv
"""

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def main():
    # Read settings from params.yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["prepare"]

    test_size = params["test_size"]
    random_state = params["random_state"]

    df = pd.read_csv("data/raw/data.csv")


    cols_to_drop = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    df["target"] = df["diagnosis"].map({"M": 0, "B": 1})
    df = df.drop(columns=["diagnosis"])

    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    after = len(df)
    print(f"Cleaning: removed {before - after} rows (duplicates/missing values)")


    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"],
    )

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print(f"Train set: {train_df.shape}, Test set: {test_df.shape}")

if __name__ == "__main__":
    main()