#!/usr/bin/env python3
"""
Combine and prepare batch Q&A results for fine-tuning
"""

import json
import pandas as pd
from pathlib import Path
import sys
from collections import defaultdict
import hashlib

def parse_batch_result(result):
    """Parse a single batch result and extract Q&A pair"""
    try:
        # Check if request succeeded
        if result['response']['status_code'] != 200:
            return None, f"API error: {result['response']['status_code']}"
        
        # Get the content
        content = result['response']['body']['choices'][0]['message']['content']
        
        # Parse the JSON content
        qa_pair = json.loads(content)
        
        # Validate structure
        if 'question' not in qa_pair or 'answer' not in qa_pair:
            return None, "Missing question or answer field"
        
        # Extract custom_id to get original case index
        custom_id = result['custom_id']
        case_index = int(custom_id.split('-')[1])
        
        return {
            'question': qa_pair['question'].strip(),
            'answer': qa_pair['answer'].strip(),
            'case_index': case_index,
            'custom_id': custom_id
        }, None
        
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    except Exception as e:
        return None, f"Parse error: {e}"

def load_batch_results(batch_file_paths):
    """Load and parse all batch result files"""
    all_qa_pairs = []
    errors = []
    
    for batch_path in batch_file_paths:
        print(f"  Loading {batch_path.name}...")
        
        with open(batch_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    result = json.loads(line)
                    qa_pair, error = parse_batch_result(result)
                    
                    if qa_pair:
                        all_qa_pairs.append(qa_pair)
                    else:
                        errors.append({
                            'file': batch_path.name,
                            'line': line_num,
                            'custom_id': result.get('custom_id', 'unknown'),
                            'error': error
                        })
    
    return all_qa_pairs, errors

def check_duplicates(qa_pairs):
    """Check for duplicate Q&A pairs"""
    seen = {}
    duplicates = []
    
    for qa in qa_pairs:
        # Create hash of question + answer
        content = qa['question'] + qa['answer']
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in seen:
            duplicates.append({
                'current': qa,
                'duplicate_of': seen[content_hash]
            })
        else:
            seen[content_hash] = qa
    
    return duplicates

def quality_checks(qa_pairs):
    """Run quality checks on Q&A pairs"""
    issues = defaultdict(list)
    
    for qa in qa_pairs:
        # Check minimum lengths
        if len(qa['question']) < 10:
            issues['question_too_short'].append(qa)
        
        if len(qa['answer']) < 20:
            issues['answer_too_short'].append(qa)
        
        # Check maximum lengths
        if len(qa['question']) > 500:
            issues['question_too_long'].append(qa)
        
        if len(qa['answer']) > 2000:
            issues['answer_too_long'].append(qa)
        
        # Check for empty strings
        if not qa['question'].strip():
            issues['empty_question'].append(qa)
        
        if not qa['answer'].strip():
            issues['empty_answer'].append(qa)
        
        # Check for placeholder text
        if 'your question here' in qa['question'].lower():
            issues['placeholder_question'].append(qa)
        
        if 'your answer here' in qa['answer'].lower():
            issues['placeholder_answer'].append(qa)
    
    return dict(issues)

def create_finetuning_format(qa_pairs, original_df):
    """
    Convert Q&A pairs to SIMPLE Q&A FORMAT (no instructions, no system messages)
    
    Output format:
    {
        "question": "What was the ruling in Brown v. Board of Education?",
        "answer": "The Supreme Court ruled..."
    }
    
    This is better for continual learning research as it:
    - Tests pure knowledge retention (no instruction confounds)
    - Provides cleaner forgetting measurements
    - Matches standard continual learning methodology
    """
    
    # Sort by case_index to match with original df
    qa_pairs_sorted = sorted(qa_pairs, key=lambda x: x['case_index'])
    
    finetuning_data = []
    
    for qa in qa_pairs_sorted:
        case_index = qa['case_index']
        
        # Simple Q&A format - NO instruction field, NO system message
        entry = {
            "question": qa['question'],
            "answer": qa['answer']
        }
        
        finetuning_data.append(entry)
    
    return finetuning_data

def save_report(output_dir, stats):
    """Save quality report"""
    report_path = output_dir / "quality_report.json"
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)
    return report_path

def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Usage: python prepare_finetuning_data.py <step_number>")
        print("Example: python prepare_finetuning_data.py 1")
        sys.exit(1)
    
    step = sys.argv[1]
    
    # Paths
    base_dir = Path(f"data/step_{step}")
    
    # Try different possible batch directories
    possible_dirs = [
        base_dir / "prepared_batches" / "outputs",
        base_dir / "batch_processing_sequential",
        base_dir / "batch_processing_multi"
    ]
    
    batch_dir = None
    for dir_path in possible_dirs:
        if dir_path.exists():
            batch_dir = dir_path
            break
    
    if not batch_dir:
        print(f"Error: No batch output directory found in {base_dir}")
        print("Tried:")
        for d in possible_dirs:
            print(f"  {d}")
        sys.exit(1)
    
    # Create output directory
    output_dir = base_dir / "finetuning_ready"
    output_dir.mkdir(exist_ok=True)

    print(f"Preparing Fine-Tuning Data - Step {step}")
    print(f"Batch directory: {batch_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load original data
    print("Loading original data")
    train_df = pd.read_csv(base_dir / "train.csv")
    val_df = pd.read_csv(base_dir / "validation.csv")
    print(f"  Train: {len(train_df):,} cases")
    print(f"  Val: {len(val_df):,} cases")
    
    # Find all batch output files - try different naming patterns
    train_batch_files = sorted(batch_dir.glob("train_batch_*_output.jsonl"))
    if not train_batch_files:
        train_batch_files = sorted(batch_dir.glob("train_batch_*_output_raw.jsonl"))
    if not train_batch_files:
        train_batch_files = sorted(batch_dir.glob("train_batch_*.jsonl"))
    
    val_batch_files = sorted(batch_dir.glob("val_batch_*_output.jsonl"))
    if not val_batch_files:
        val_batch_files = sorted(batch_dir.glob("val_batch_*_output_raw.jsonl"))
    if not val_batch_files:
        val_batch_files = sorted(batch_dir.glob("val_batch_*.jsonl"))
    
    if not train_batch_files:
        print("Error: No train batch output files found")
        print(f"Looking for: {batch_dir}/train_batch_*_output_raw.jsonl")
        sys.exit(1)
    
    print(f"Found {len(train_batch_files)} train batch files")
    print(f"Found {len(val_batch_files)} val batch files")
    print()
    
    # Process train batches
    train_qa_pairs, train_errors = load_batch_results(train_batch_files)
    print(f"Loaded {len(train_qa_pairs):,} Q&A pairs")
    print(f"  Errors: {len(train_errors)}")

    # Process validation batches
    val_qa_pairs, val_errors = load_batch_results(val_batch_files)
    print(f"Loaded {len(val_qa_pairs):,} Q&A pairs")
    print(f"Errors: {len(val_errors)}")
    print()
    
    # Check for duplicates
    train_dupes = check_duplicates(train_qa_pairs)
    val_dupes = check_duplicates(val_qa_pairs)
    print(f"Train duplicates: {len(train_dupes)}")
    print(f"Val duplicates: {len(val_dupes)}")
    
    if train_dupes:
        print(f"Warning: {len(train_dupes)} duplicate train Q&A pairs found")
        # Remove duplicates
        seen = set()
        train_qa_pairs_unique = []
        for qa in train_qa_pairs:
            content_hash = hashlib.md5((qa['question'] + qa['answer']).encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                train_qa_pairs_unique.append(qa)
        train_qa_pairs = train_qa_pairs_unique
        print(f"  ✓ Removed duplicates: {len(train_qa_pairs):,} unique pairs remain")
    
    if val_dupes:
        print(f"Warning: {len(val_dupes)} duplicate val Q&A pairs found")
        # Remove duplicates
        seen = set()
        val_qa_pairs_unique = []
        for qa in val_qa_pairs:
            content_hash = hashlib.md5((qa['question'] + qa['answer']).encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                val_qa_pairs_unique.append(qa)
        val_qa_pairs = val_qa_pairs_unique
        print(f"Removed duplicates: {len(val_qa_pairs):,} unique pairs remain")
    
    # Quality checks
    train_issues = quality_checks(train_qa_pairs)
    val_issues = quality_checks(val_qa_pairs)
    
    print("Train issues:")
    if train_issues:
        for issue_type, items in train_issues.items():
            print(f"  {issue_type}: {len(items)}")
    else:
        print("No issues found!")
    
    print("\nVal issues:")
    if val_issues:
        for issue_type, items in val_issues.items():
            print(f"  {issue_type}: {len(items)}")
    else:
        print("No issues found!")
    print()
    
    # Convert to fine-tuning format
    train_finetuning = create_finetuning_format(train_qa_pairs, train_df)
    val_finetuning = create_finetuning_format(val_qa_pairs, val_df)
    print(f"Train: {len(train_finetuning):,} examples")
    print(f"Val: {len(val_finetuning):,} examples")
    print()
    
    # Save fine-tuning files
    print("Saving Fine-Tuning Files")
    train_output = output_dir / "train.jsonl"
    val_output = output_dir / "val.jsonl"
    
    with open(train_output, 'w') as f:
        for entry in train_finetuning:
            f.write(json.dumps(entry) + '\n')
    
    with open(val_output, 'w') as f:
        for entry in val_finetuning:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Train: {train_output}")
    print(f"Val: {val_output}")
    print()
    
    # Save quality report
    stats = {
        'step': step,
        'format': 'simple_qa',  # Note the format
        'train': {
            'original_cases': len(train_df),
            'qa_pairs_generated': len(train_qa_pairs),
            'qa_pairs_final': len(train_finetuning),
            'success_rate': f"{len(train_finetuning)/len(train_df)*100:.1f}%",
            'errors': len(train_errors),
            'duplicates_removed': len(train_dupes),
            'issues': {k: len(v) for k, v in train_issues.items()} if train_issues else {}
        },
        'val': {
            'original_cases': len(val_df),
            'qa_pairs_generated': len(val_qa_pairs),
            'qa_pairs_final': len(val_finetuning),
            'success_rate': f"{len(val_finetuning)/len(val_df)*100:.1f}%",
            'errors': len(val_errors),
            'duplicates_removed': len(val_dupes),
            'issues': {k: len(v) for k, v in val_issues.items()} if val_issues else {}
        }
    }
    
    report_path = save_report(output_dir, stats)
    print(f"Quality report: {report_path}")

if __name__ == "__main__":
    main()
