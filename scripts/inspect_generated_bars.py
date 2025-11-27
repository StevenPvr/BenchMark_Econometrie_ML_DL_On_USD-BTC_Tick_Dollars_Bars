
import pandas as pd
import numpy as np
from src.path import DOLLAR_BARS_PARQUET

def inspect_bars():
    if not DOLLAR_BARS_PARQUET.exists():
        print("Dollar bars file not found.")
        return

    print(f"Loading {DOLLAR_BARS_PARQUET}...")
    df = pd.read_parquet(DOLLAR_BARS_PARQUET)
    
    print(f"Total Bars: {len(df)}")
    
    if "n_ticks" in df.columns:
        print("\n--- Ticks per Bar Stats ---")
        print(df["n_ticks"].describe())
        
    if "threshold_used" in df.columns:
        print("\n--- Threshold Stats ---")
        print(df["threshold_used"].describe())
        
    if "cum_dollar_value" in df.columns:
        print("\n--- Dollar Value per Bar Stats ---")
        print(df["cum_dollar_value"].describe())

if __name__ == "__main__":
    inspect_bars()
