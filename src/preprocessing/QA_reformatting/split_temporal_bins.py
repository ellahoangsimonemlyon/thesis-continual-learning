#!/usr/bin/env python3
"""
SCOTUS Dataset Temporal Splitting Script
Splits the cleaned dataset into 10 temporal bins based on landmark cases.
Each bin has 90% training, 10% validation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tiktoken
import json
from datetime import datetime

def count_tokens(text, encoding_name="cl100k_base"):
    """Count tokens in text using tiktoken."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: character-based estimation (~4 chars per token)
        return len(text) // 4

def get_bin_definitions():
    """
    Define the 10 bins based on landmark Supreme Court cases.
    Returns list of bin definitions with temporal boundaries.
    """
    bins = [
        {
            'step': 1,
            'name': 'Foundations & Reconstruction',
            'start_year': 1791,
            'end_year': 1896,
            'cutoff_case': 'Plessy v. Ferguson (1896)',
            'description': 'Early period through end of Reconstruction'
        },
        {
            'step': 2,
            'name': 'Lochner-Early Speech',
            'start_year': 1897,
            'end_year': 1919,
            'cutoff_case': 'Schenck v. United States (1919)',
            'description': 'Liberty-of-contract era and first modern First Amendment cases'
        },
        {
            'step': 3,
            'name': 'Incorporation, Pre-New Deal',
            'start_year': 1920,
            'end_year': 1936,
            'cutoff_case': 'West Coast Hotel v. Parrish (1937)',
            'description': 'Bill of Rights incorporation begins'
        },
        {
            'step': 4,
            'name': 'New Deal & War Powers',
            'start_year': 1937,
            'end_year': 1953,
            'cutoff_case': 'Brown v. Board of Education (1954)',
            'description': 'Court accepts federal power and WWII/Korean War era'
        },
        {
            'step': 5,
            'name': 'Warren Court Rights Revolution',
            'start_year': 1954,
            'end_year': 1969,
            'cutoff_case': 'Brandenburg v. Ohio (1969)',
            'description': 'Civil rights and criminal-procedure revolution'
        },
        {
            'step': 6,
            'name': 'Burger Court Recalibration',
            'start_year': 1970,
            'end_year': 1984,
            'cutoff_case': 'Chevron U.S.A. v. NRDC (1984)',
            'description': 'Civil liberties recalibrated, Roe v. Wade, Chevron'
        },
        {
            'step': 7,
            'name': 'Rehnquist Court Begins',
            'start_year': 1985,
            'end_year': 1994,
            'cutoff_case': 'U.S. v. Lopez (1995)',
            'description': 'Conservative shift begins'
        },
        {
            'step': 8,
            'name': 'Federalism Revival',
            'start_year': 1995,
            'end_year': 2004,
            'cutoff_case': 'Hamdi v. Rumsfeld (2004)',
            'description': 'Federalism limits on federal power'
        },
        {
            'step': 9,
            'name': 'Roberts Court Part I',
            'start_year': 2005,
            'end_year': 2014,
            'cutoff_case': 'Obergefell v. Hodges (2015)',
            'description': 'War on Terror and early Roberts Court'
        },
        {
            'step': 10,
            'name': 'Roberts Court Part II',
            'start_year': 2015,
            'end_year': 2023,
            'cutoff_case': 'End of dataset',
            'description': 'Modern era through recent decisions'
        }
    ]
    return bins

def create_temporal_bins(df, bin_definitions):
    """
    Split dataset into bins based on year ranges.
    Returns list of dataframes, one per bin.
    """
    bins = []
    
    print("Creating temporal bins...")
    print()
    
    for bin_def in bin_definitions:
        step = bin_def['step']
        start_year = bin_def['start_year']
        end_year = bin_def['end_year']
        
        # Filter by year
        mask = (df['date_filed'].dt.year >= start_year) & (df['date_filed'].dt.year <= end_year)
        bin_df = df[mask].copy()
        
        bins.append(bin_df)
        
        # Get actual date range
        if len(bin_df) > 0:
            actual_start = bin_df['date_filed'].min()
            actual_end = bin_df['date_filed'].max()
            date_range = f"{actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}"
        else:
            date_range = "No cases"
        
        print(f"Step {step}: {bin_def['name']}")
        print(f"  Years: {start_year}-{end_year}")
        print(f"  Cases: {len(bin_df):,}")
        print(f"  Actual dates: {date_range}")
        print()
    
    return bins

def split_train_validation(df, val_ratio=0.1, random_state=42):
    """
    Split dataframe into train and validation sets.
    Randomly samples for validation while maintaining chronological order within each set.
    """
    if len(df) == 0:
        return df.copy(), df.copy()
    
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split point
    val_size = int(len(df_shuffled) * val_ratio)
    
    # Ensure at least 1 case in validation if possible
    if val_size == 0 and len(df_shuffled) > 0:
        val_size = 1
    
    # Split
    val_df = df_shuffled.iloc[:val_size].copy()
    train_df = df_shuffled.iloc[val_size:].copy()
    
    # Sort both by date to maintain chronological order
    train_df = train_df.sort_values('date_filed').reset_index(drop=True)
    val_df = val_df.sort_values('date_filed').reset_index(drop=True)
    
    return train_df, val_df

def save_bin_data(bin_df, bin_def, output_dir, val_ratio=0.1):
    """
    Save training and validation data for a single bin.
    """
    step = bin_def['step']
    
    # Create directory for this bin
    bin_dir = output_dir / f"step_{step}"
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    # Split into train/val
    train_df, val_df = split_train_validation(bin_df, val_ratio=val_ratio)
    
    # Save CSVs
    train_path = bin_dir / "train.csv"
    val_path = bin_dir / "validation.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    # Save metadata
    metadata = {
        'step': step,
        'name': bin_def['name'],
        'years': f"{bin_def['start_year']}-{bin_def['end_year']}",
        'cutoff_case': bin_def['cutoff_case'],
        'description': bin_def['description'],
        'num_train': len(train_df),
        'num_val': len(val_df),
        'num_total': len(bin_df),
        'val_ratio': val_ratio,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = bin_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved step_{step}: {len(train_df):,} train, {len(val_df):,} val")
    
    return len(train_df), len(val_df)

def calculate_bin_statistics(bins, bin_definitions):
    """
    Calculate detailed statistics for each bin including token counts.
    """
    print("\nCalculating statistics...")
    detailed_stats = []
    
    for bin_df, bin_def in zip(bins, bin_definitions):
        step = bin_def['step']
        
        if len(bin_df) == 0:
            stats = {
                'step': step,
                'name': bin_def['name'],
                'years': f"{bin_def['start_year']}-{bin_def['end_year']}",
                'cutoff_case': bin_def['cutoff_case'],
                'num_cases_total': 0,
                'num_cases_train': 0,
                'num_cases_val': 0,
                'tokens_total': 0,
                'tokens_train': 0,
                'tokens_val': 0,
                'avg_tokens_per_case': 0
            }
            detailed_stats.append(stats)
            continue
        
        # Calculate tokens
        bin_df['token_count'] = bin_df['opinion_text'].apply(count_tokens)
        
        # Split for train/val stats
        train_df, val_df = split_train_validation(bin_df, val_ratio=0.1)
        
        train_tokens = train_df['token_count'].sum()
        val_tokens = val_df['token_count'].sum()
        total_tokens = bin_df['token_count'].sum()
        
        stats = {
            'step': step,
            'name': bin_def['name'],
            'years': f"{bin_def['start_year']}-{bin_def['end_year']}",
            'cutoff_case': bin_def['cutoff_case'],
            'num_cases_total': len(bin_df),
            'num_cases_train': len(train_df),
            'num_cases_val': len(val_df),
            'tokens_total': int(total_tokens),
            'tokens_train': int(train_tokens),
            'tokens_val': int(val_tokens),
            'avg_tokens_per_case': int(total_tokens / len(bin_df)) if len(bin_df) > 0 else 0
        }
        detailed_stats.append(stats)
        
        print(f"  Step {step}: {len(bin_df):,} cases, {total_tokens:,} tokens "
              f"(avg: {stats['avg_tokens_per_case']:,} tokens/case)")
    
    return detailed_stats

def save_statistics_json(stats, output_dir):
    """Save detailed statistics to JSON file."""
    stats_path = output_dir / "bin_statistics.json"
    
    # Add summary
    summary = {
        'total_bins': len(stats),
        'total_cases': sum(s['num_cases_total'] for s in stats),
        'total_train_cases': sum(s['num_cases_train'] for s in stats),
        'total_val_cases': sum(s['num_cases_val'] for s in stats),
        'total_tokens': sum(s['tokens_total'] for s in stats),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    output = {
        'summary': summary,
        'bins': stats
    }
    
    with open(stats_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nStatistics saved to: {stats_path}")
    return stats_path

def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: python split_temporal_bins.py <path_to_cleaned_csv>")
        print("Example: python split_temporal_bins.py data/scotus_final_clean.csv")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Output directory in data/ folder
    output_dir = input_path.parent

    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    df = pd.read_csv(input_path)
    df['date_filed'] = pd.to_datetime(df['date_filed'], errors='coerce')
    print(f"Loaded {len(df):,} cases")
    print(f"Date range: {df['date_filed'].min().strftime('%Y-%m-%d')} to "
          f"{df['date_filed'].max().strftime('%Y-%m-%d')}")
    
    # Get bin definitions
    bin_definitions = get_bin_definitions()
    
    # Create bins
    bins = create_temporal_bins(df, bin_definitions)
    
    # Save each bin
    for bin_df, bin_def in zip(bins, bin_definitions):
        save_bin_data(bin_df, bin_def, output_dir, val_ratio=0.1)
    
    # Calculate detailed statistics
    detailed_stats = calculate_bin_statistics(bins, bin_definitions)
    
    # Save statistics JSON
    stats_path = save_statistics_json(detailed_stats, output_dir)

if __name__ == "__main__":
    main()