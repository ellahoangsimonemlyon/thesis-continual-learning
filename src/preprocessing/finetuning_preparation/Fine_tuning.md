# Fine-Tuning Data Preparation

***Overview***

After generating Q&A pairs using OpenAI's Batch API, use this script to combine, validate, and prepare the data for fine-tuning. The script performs comprehensive quality checks and formats data according to OpenAI's fine-tuning requirements.

Script: prepare_finetuning_data.py

Purpose
Combines multiple batch output files, validates quality, removes duplicates, and formats data for OpenAI fine-tuning.
Key Features

Batch Combination: Merges all train and validation batch outputs into single files
Q&A Extraction: Parses question-answer pairs from GPT responses
Duplicate Detection: Identifies and removes duplicate Q&A pairs using content hashing
Quality Validation: Checks for:

Empty or placeholder content
Questions too short (<10 characters) or too long (>500 characters)
Answers too short (<20 characters) or too long (>2000 characters)
Malformed JSON responses


Format Conversion: Converts to OpenAI's fine-tuning format with system/user/assistant messages
Metadata Preservation: Maintains case names, dates, and citation counts
Quality Report: Generates detailed statistics and success rates

Usage
bashpython prepare_finetuning_data.py <step_number>
Example:
bashpython prepare_finetuning_data.py 1
Input Structure
The script expects batch output files in this structure:

data/step_1/
├── train.csv                          # Original training data
├── validation.csv                     # Original validation data
└── prepared_batches/outputs/          # Batch API results
    ├── train_batch_001_output.jsonl
    ├── train_batch_002_output.jsonl
    ...
    ├── train_batch_026_output.jsonl
    ├── val_batch_001_output.jsonl
    ├── val_batch_002_output.jsonl
    └── val_batch_003_output.jsonl

Output Files
Generated in: data/step_<N>/finetuning_ready/

train.jsonl - Training data in OpenAI fine-tuning format
val.jsonl - Validation data in OpenAI fine-tuning format
quality_report.json - Detailed statistics and quality metrics

Output Format

entry = {
    "question": qa['question'],
    "answer": qa['answer']
}

Sample

{"question": "What was the ruling in Brown v. Board of Education?", "answer": "The Supreme Court ruled..."}
{"question": "What is the significance of Marbury v. Madison?", "answer": "This case established..."}



Quality Checks Performed
CheckDescriptionActionDuplicatesIdentical question-answer pairsAutomatically removedEmpty ContentBlank questions or answersFlagged in reportLength ValidationQuestions <10 chars, Answers <20 charsFlagged in reportPlaceholder Text"your question here", "your answer here"Flagged in reportJSON ValidityMalformed responses from GPTCounted as errorsAPI ErrorsFailed batch requestsCounted as errors
Verification Commands
Before running the script:
bash# Count train batches
ls data/step_1/prepared_batches/outputs/train_batch_*_output.jsonl | wc -l

# Count validation batches
ls data/step_1/prepared_batches/outputs/val_batch_*_output.jsonl | wc -l

# Check file content
head -1 data/step_1/prepared_batches/outputs/train_batch_001_output.jsonl | python -m json.tool
After running the script:
bash# Count training examples
wc -l data/step_1/finetuning_ready/train.jsonl

# Count validation examples
wc -l data/step_1/finetuning_ready/val.jsonl

# View sample Q&A
head -1 data/step_1/finetuning_ready/train.jsonl | python -m json.tool

# Check quality report
cat data/step_1/finetuning_ready/quality_report.json | python -m json.tool
Example Output
================================================================================
Preparing Fine-Tuning Data - Step 1
================================================================================
Batch directory: data/step_1/prepared_batches/outputs
Output directory: data/step_1/finetuning_ready

Loading original data...
  Train: 10,477 cases
  Val: 1,164 cases

Found 26 train batch files
Found 3 val batch files

================================================================================
STEP 1: Loading and Parsing TRAIN Batches
================================================================================
  Loading train_batch_001_output.jsonl...
  Loading train_batch_002_output.jsonl...
  [...]
✓ Loaded 10,450 Q&A pairs
  Errors: 27

================================================================================
STEP 2: Loading and Parsing VALIDATION Batches
================================================================================
  Loading val_batch_001_output.jsonl...
  [...]
✓ Loaded 1,161 Q&A pairs
  Errors: 3

================================================================================
STEP 3: Checking for Duplicates
================================================================================
Train duplicates: 2
  ⚠ Warning: 2 duplicate train Q&A pairs found
  ✓ Removed duplicates: 10,448 unique pairs remain
Val duplicates: 0

================================================================================
STEP 4: Quality Checks
================================================================================
Train issues:
  ✓ No issues found!

Val issues:
  answer_too_short: 1

================================================================================
STEP 5: Converting to Fine-Tuning Format
================================================================================
✓ Train: 10,448 examples
✓ Val: 1,161 examples

================================================================================
STEP 6: Saving Fine-Tuning Files
================================================================================
✓ Train: data/step_1/finetuning_ready/train.jsonl
✓ Val: data/step_1/finetuning_ready/val.jsonl
✓ Quality report: data/step_1/finetuning_ready/quality_report.json

================================================================================
SUMMARY
================================================================================

Step 1 fine-tuning data ready!

Train:
  Original: 10,477 cases
  Generated: 10,450 Q&A pairs
  Final: 10,448 examples (99.7%)

Validation:
  Original: 1,164 cases
  Generated: 1,161 Q&A pairs
  Final: 1,161 examples (99.7%)

Files ready for fine-tuning:
  📄 data/step_1/finetuning_ready/train.jsonl
  📄 data/step_1/finetuning_ready/val.jsonl

Next step: Upload to OpenAI and start fine-tuning!


## Troubleshooting

Issue: "No batch output directory found"

Ensure batch files are in data/step_<N>/prepared_batches/outputs/
Check file naming: train_batch_###_output.jsonl format

Issue: "No train batch output files found"

Verify files exist with: ls data/step_1/prepared_batches/outputs/train_batch_*.jsonl
Check file extensions are .jsonl not .json

Issue: High error rate

Review quality_report.json for error details
Check a few batch output files manually for malformed JSON
Verify batch API requests completed successfully


### Related Scripts

scotus_batches.py - Generates batches within a certain token limit
send_one_batch.py - Generates Q&A pairs using OpenAI Batch API for **one** batch of a given step 
split_temporal_bins.py - Creates temporal training steps
download_batch_outputs.py - Downloads output files from OpenAI