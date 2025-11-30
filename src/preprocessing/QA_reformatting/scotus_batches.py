#!/usr/bin/env python3
"""
Prepare OpenAI batch job files for question-answer generation from SCOTUS cases.

Creates JSONL batch files under data/step_<step>/prepared_batches/ and a manifest.csv
that tracks metadata for each batch.

Usage:
    python scotus_prepare_batches.py --step 1
    python scotus_prepare_batches.py --step 1 --max-tokens-per-batch 1500000
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
import pandas as pd

# Configuration
MODEL_NAME = "gpt-4o-mini"
DEFAULT_MAX_TOKENS_PER_BATCH = 1_600_000  # Keep below OpenAI's 2M queue limit
PER_REQUEST_OVERHEAD_TOKENS = 450  # Overhead for system prompt, formatting, etc.
USE_TIKTOKEN = True

# Global tiktoken encoder
_TK = None

def maybe_load_tiktoken():
    """Attempt to load tiktoken for accurate token counting."""
    global _TK
    if not USE_TIKTOKEN:
        return False
    try:
        import tiktoken
        _TK = tiktoken.get_encoding("cl100k_base")
        return True
    except Exception:
        _TK = None
        return False

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken or character-based estimation."""
    if not isinstance(text, str) or not text:
        return 0
    if _TK is not None:
        return len(_TK.encode(text))
    # Fallback: ~4 characters per token
    return math.ceil(len(text) / 4)

# PROMPT TEMPLATES
def create_system_prompt():
    """System prompt for question-answer generation."""
    return (
        "You are a legal expert specializing in U.S. Supreme Court cases. "
        "Create ONE high-quality question-answer pair from the opinion text.\n\n"
        "Question:\n"
        "- Specific, answerable from the text\n"
        "- Focus on holding, reasoning, constitutional issue, or key facts\n"
        "- Clear and professional\n\n"
        "Answer:\n"
        "- Accurate, 2-4 sentences, uses appropriate legal terms\n"
        "- Captures the essential legal reasoning/holding\n\n"
        "Return ONLY JSON:\n"
        "{\n"
        '  "question": "...",\n'
        '  "answer": "..."\n'
        "}"
    )

def create_user_prompt(row, max_chars=50_000):
    """Create user prompt from a case row."""
    case_name = row.get("case_name", "Unknown Case")
    date_filed = row.get("date_filed", "Unknown Date")
    opinion_text = row.get("opinion_text", "")
    
    # Handle None, NaN, and non-string values
    if opinion_text is None or (isinstance(opinion_text, float) and pd.isna(opinion_text)):
        opinion_text = ""
    elif not isinstance(opinion_text, str):
        opinion_text = str(opinion_text)
    
    # Truncate if too long
    if len(opinion_text) > max_chars:
        opinion_text = opinion_text[:max_chars] + "\n\n[Opinion truncated for length]"
    
    prompt = (
        f"Case: {case_name}\n"
        f"Date Filed: {date_filed}\n\n"
        f"Opinion Text:\n{opinion_text}\n\n"
        "Generate ONE question-answer pair valuable for training a legal AI model. "
        "Return JSON with 'question' and 'answer'."
    )
    return prompt

# BATCHING LOGIC
def estimate_case_tokens(row) -> int:
    """Estimate total input tokens contributed by this case within a batch."""
    opinion_text = row.get("opinion_text", "")
    
    # Handle None, NaN, non-string values
    if opinion_text is None or (isinstance(opinion_text, float) and pd.isna(opinion_text)):
        opinion_text = ""
    elif not isinstance(opinion_text, str):
        opinion_text = str(opinion_text)
    
    opinion_tokens = count_tokens(opinion_text)
    return opinion_tokens + PER_REQUEST_OVERHEAD_TOKENS

def split_into_batches(df: pd.DataFrame, max_tokens_per_batch: int) -> list:
    """
    Split dataframe into batches that don't exceed max_tokens_per_batch.
    Returns list of dataframes.
    """
    batches = []
    current_batch = []
    current_tokens = 0
    
    for idx, row in df.iterrows():
        case_tokens = estimate_case_tokens(row)
        
        # If adding this case would exceed limit, start new batch
        if current_batch and current_tokens + case_tokens > max_tokens_per_batch:
            batches.append(pd.DataFrame(current_batch))
            current_batch = []
            current_tokens = 0
        
        current_batch.append(row)
        current_tokens += case_tokens
    
    # Add final batch if not empty
    if current_batch:
        batches.append(pd.DataFrame(current_batch))
    
    return batches

def make_request_record(idx_int: int, system_prompt: str, user_prompt: str):
    """Create a single request record for the batch API."""
    return {
        "custom_id": f"request-{idx_int}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
            "response_format": {"type": "json_object"}
        }
    }

# BATCH PREPARATION
def prepare_batches_to_disk(step_dir: Path, train_csv: Path, val_csv: Path,
                            max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH) -> Path:
    
    # Create output directory
    prepared_dir = step_dir / "prepared_batches"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CSVs
    print(f"Loading training data from: {train_csv}")
    train_df = pd.read_csv(train_csv)
    print(f"  Loaded {len(train_df):,} training cases")
    
    print(f"Loading validation data from: {val_csv}")
    val_df = pd.read_csv(val_csv)
    print(f"  Loaded {len(val_df):,} validation cases")
    
    # Filter out rows with missing opinion_text
    initial_train = len(train_df)
    initial_val = len(val_df)
    
    train_df = train_df[train_df['opinion_text'].notna() & (train_df['opinion_text'] != '')]
    val_df = val_df[val_df['opinion_text'].notna() & (val_df['opinion_text'] != '')]
    
    if len(train_df) < initial_train:
        filtered = initial_train - len(train_df)
        print(f"Filtered out {filtered} training cases with missing opinion_text")
    if len(val_df) < initial_val:
        filtered = initial_val - len(val_df)
        print(f"Filtered out {filtered} validation cases with missing opinion_text")
    
    print(f"\nAfter filtering:")
    print(f"  Training cases: {len(train_df):,}")
    print(f"  Validation cases: {len(val_df):,}")
    
    # Split into batches
    print("Splitting into batches...")
    train_batches = split_into_batches(train_df, max_tokens_per_batch)
    val_batches = split_into_batches(val_df, max_tokens_per_batch)
    
    print(f"  Training batches: {len(train_batches)}")
    print(f"  Validation batches: {len(val_batches)}")
    print(f"  Total batches: {len(train_batches) + len(val_batches)}")
    
    # Prepare system prompt
    system_prompt = create_system_prompt()
    manifest_rows = []
    
    def write_batches(kind: str, batches: list):
        """Write JSONL files for a set of batches (train or val)."""
        print(f"Writing {kind.upper()} batches...")
        
        for i, batch_df in enumerate(batches, start=1):
            # Output file path
            output_path = prepared_dir / f"{kind}_batch_{i:03d}.jsonl"
            
            # Write JSONL
            total_batch_tokens = 0
            with open(output_path, "w", encoding="utf-8") as f:
                for row_idx, row in batch_df.iterrows():
                    user_prompt = create_user_prompt(row.to_dict())
                    request_record = make_request_record(int(row_idx), system_prompt, user_prompt)
                    f.write(json.dumps(request_record) + "\n")
                    total_batch_tokens += estimate_case_tokens(row)
            
            # Add to manifest
            manifest_rows.append({
                "kind": kind,
                "batch_num": i,
                "file_path": str(output_path),
                "file_name": output_path.name,
                "num_cases": len(batch_df),
                "est_input_tokens": total_batch_tokens,
                "status": "READY",
            })
            
            print(f"{output_path.name}: {len(batch_df):,} cases, ~{total_batch_tokens:,} tokens")
    
    # Write train and val batches
    write_batches("train", train_batches)
    write_batches("val", val_batches)

    # Create and save manifest
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = prepared_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return prepared_dir

# main
def main():
    parser = argparse.ArgumentParser(
        description="Prepare OpenAI batch files for SCOTUS question-answer generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scotus_prepare_batches.py --step 1
  python scotus_prepare_batches.py --step 1 --max-tokens-per-batch 1500000
  python scotus_prepare_batches.py --step 2 --train-csv custom_train.csv --val-csv custom_val.csv
        """
    )
    
    parser.add_argument(
        "--step", 
        required=True, 
        help="Step number (e.g., 1). Will read/write under data/step_<step>/"
    )
    parser.add_argument(
        "--max-tokens-per-batch", 
        type=int, 
        default=DEFAULT_MAX_TOKENS_PER_BATCH,
        help=f"Maximum tokens per batch (default: {DEFAULT_MAX_TOKENS_PER_BATCH:,})"
    )
    parser.add_argument(
        "--train-csv", 
        default=None, 
        help="Path to training CSV (default: data/step_<step>/train.csv)"
    )
    parser.add_argument(
        "--val-csv", 
        default=None, 
        help="Path to validation CSV (default: data/step_<step>/validation.csv)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    step_dir = Path(f"data/step_{args.step}")
    
    if args.train_csv:
        train_csv = Path(args.train_csv)
    else:
        train_csv = step_dir / "train.csv"
    
    if args.val_csv:
        val_csv = Path(args.val_csv)
    else:
        val_csv = step_dir / "validation.csv"
    
    # Check files exist
    if not train_csv.exists():
        print(f"Error: Training CSV not found: {train_csv}")
        sys.exit(1)
    
    if not val_csv.exists():
        print(f"Error: Validation CSV not found: {val_csv}")
        sys.exit(1)
    
    # Load tiktoken if available
    if maybe_load_tiktoken():
        print("Using tiktoken for accurate token counting")
    else:
        print("tiktoken not available, using character-based estimation")
    
    # Prepare batches
    try:
        prepared_dir = prepare_batches_to_disk(
            step_dir, 
            train_csv, 
            val_csv,
            max_tokens_per_batch=args.max_tokens_per_batch
        )
    except Exception as e:
        print(f"\nError during batch preparation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()