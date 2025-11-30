import pandas as pd
import numpy as np

def repair_and_summarize():
    csv_file = "stage3_verification_full.csv"
    
    try:
        # Read all as string first to avoid type errors during load
        df = pd.read_csv(csv_file, dtype=str)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    print(f"Loaded {len(df)} rows.")
    
    # The column causing issue is 'Conn_Cor_MyoAo'
    # It likely contains concatenated strings like '1.01.01.0...'
    
    # Function to clean the column
    def clean_numeric(x):
        try:
            return float(x)
        except:
            return np.nan

    # Convert all columns to numeric, coercing errors to NaN
    # We want to preserve case_id as is (though it's numeric usually)
    # But for mean calculation, we only care about metrics.
    
    # Identify metric columns
    metric_cols = [c for c in df.columns if c not in ['case_id', 'missing_orig', 'shape_mismatch']]
    
    for col in metric_cols:
        # Check for bad values
        # If a value is longer than say 5 chars and contains multiple dots, it's suspicious
        # But simple to_numeric with coerce is easiest
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop rows where critical metrics became NaN?
    # Or just ignore them in mean calculation (mean ignores NaN by default)
    
    # Check how many NaNs we have in the problematic column
    nans = df['Conn_Cor_MyoAo'].isna().sum()
    print(f"Found {nans} corrupted entries in 'Conn_Cor_MyoAo'.")
    
    # Calculate means
    print("\n--- Verification Summary (Repaired) ---")
    
    print("\nMean Dice (vs Original):")
    dice_cols = [c for c in df.columns if c.startswith('Dice_')]
    for col in dice_cols:
        name = col.replace('Dice_', '')
        print(f"{name:<15}: {df[col].mean():.4f}")
            
    print("\nMean CC Count (Stage 3):")
    cc_cols = [c for c in df.columns if c.startswith('CC_')]
    for col in cc_cols:
        name = col.replace('CC_', '')
        print(f"{name:<15}: {df[col].mean():.4f}")
            
    print("\nConnectivity (Ratio of connected components):")
    conn_cols = [c for c in df.columns if c.startswith('Conn_')]
    for col in conn_cols:
        name = col.replace('Conn_', '')
        print(f"{name:<15}: {df[col].mean():.4f}")

if __name__ == "__main__":
    repair_and_summarize()
