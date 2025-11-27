
import pandas as pd
import numpy as np
from src.path import DATASET_CLEAN_PARQUET

def investigate_data():
    if not DATASET_CLEAN_PARQUET.exists():
        print("Dataset not found.")
        return

    print("Loading dataset...")
    df = pd.read_parquet(DATASET_CLEAN_PARQUET, columns=["price", "amount"])
    
    n_ticks = len(df)
    print(f"Total ticks: {n_ticks}")
    
    df["dollar_value"] = df["price"] * df["amount"]
    mean_dv = df["dollar_value"].mean()
    median_dv = df["dollar_value"].median()
    max_dv = df["dollar_value"].max()
    min_dv = df["dollar_value"].min()
    
    print(f"Mean Dollar Value per tick: {mean_dv:.2f}")
    print(f"Median Dollar Value per tick: {median_dv:.2f}")
    print(f"Max Dollar Value: {max_dv:.2f}")
    
    target_ticks = 50
    theoretical_threshold = mean_dv * target_ticks
    print(f"Theoretical Threshold (fixed) for {target_ticks} ticks/bar: {theoretical_threshold:.2f}")
    
    expected_bars = df["dollar_value"].sum() / theoretical_threshold
    print(f"Expected bars with fixed threshold: {int(expected_bars)}")

if __name__ == "__main__":
    investigate_data()
