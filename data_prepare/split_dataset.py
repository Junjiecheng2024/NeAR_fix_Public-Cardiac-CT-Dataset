#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset train/validation splitting script."""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def split_dataset(info_csv_path, train_ratio=0.8, seed=12, output_dir=None):
    """Split dataset into training and validation sets.
    
    Args:
        info_csv_path: Path to info.csv file containing sample IDs
        train_ratio: Training set ratio (0-1), default 0.8 (80% train, 20% val)
        seed: Random seed for reproducibility, default 42
        output_dir: Output directory (None = same as info.csv)
    
    Returns:
        train_df: Training set DataFrame
        val_df: Validation set DataFrame
    """
    df = pd.read_csv(info_csv_path)
    print(f"Total samples: {len(df)}")
    
    np.random.seed(seed)
    
    indices = np.random.permutation(len(df))
    
    n_train = int(len(df) * train_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    
    if output_dir is None:
        output_dir = Path(info_csv_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "info_train.csv"
    val_path = output_dir / "info_val.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\nFiles saved:")
    print(f"  Training: {train_path}")
    print(f"  Validation: {val_path}")
    
    return train_df, val_df

def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Split dataset into training and validation sets"
    )
    
    parser.add_argument(
        "--info_csv", 
        type=str, 
        required=True,
        help="Path to info.csv file"
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as info.csv)"
    )
    
    args = parser.parse_args()
    
    info_csv_path = Path(args.info_csv)
    
    if not info_csv_path.exists():
        print(f"Error: File not found: {info_csv_path}")
        return
    
    print("=" * 80)
    print("Dataset Splitting")
    print("=" * 80)
    print(f"Input file: {info_csv_path}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Random seed: {args.seed}")
    print()
    
    split_dataset(
        info_csv_path=str(info_csv_path),
        train_ratio=args.train_ratio,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print("\n" + "=" * 80)
    print("Splitting complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
