# SCOTUS Question-Answer Generation Pipeline

## 1. Split Dataset into Temporal Bins

Split the dataset into 10 temporal bins based on constitutional eras:
```bash
python split_temporal_bins.py data/scotus_final_clean.csv
```

This uses the `date_filed` column to split data into steps 1-10, creating:
- `data/step_1/` through `data/step_10/`
- Each contains `train.csv` and `validation.csv`
- Also generates `data/bin_statistics.json` with metadata

## 2. Set OpenAI API Key
```bash
export OPENAI_API_KEY='your-key-here'
```

## 3. Prepare Batches

Prepare batch files for OpenAI's Batch API (creates JSONL files and manifest):
```bash
# Basic usage - prepares batches for step 2
python scotus_prepare_batches.py --step 2
```

This reads `data/step_2/train.csv` and `data/step_2/validation.csv` and creates:
- `data/step_2/prepared_batches/train_batch_001.jsonl`, `train_batch_002.jsonl`, etc.
- `data/step_2/prepared_batches/val_batch_001.jsonl`, `val_batch_002.jsonl`, etc.
- `data/step_2/prepared_batches/manifest.csv` (tracks all batches)

### Optional: Custom Token Cap

Override the default 1.6M tokens per batch:
```bash
python scotus_prepare_batches.py --step 2 --max-tokens-per-batch 1500000
```

### Optional: Custom CSV Paths

If your CSVs are in a different location:
```bash
python scotus_prepare_batches.py --step 2 \
  --train-csv /path/to/my_train.csv \
  --val-csv /path/to/my_val.csv
```

## 4. Submit Batches

Use your separate submission script to submit batches one at a time:
```bash
# Submit a single batch
python submit_one.py data/step_2/prepared_batches/train_batch_001.jsonl \
  --desc "Step 2 train batch 1"

# Submit validation batch
python submit_one.py data/step_2/prepared_batches/val_batch_001.jsonl \
  --desc "Step 2 val batch 1"
```

**Note:** The `manifest.csv` tracks all prepared batches. You can update it with batch IDs and status as you submit.

## 5. Download Batch Outputs

Once batches complete, download the outputs:
```bash
# Download all completed batch outputs for step 2
python download_batch_outputs.py --step 2

# Download and parse (extract clean Q/A JSON)
python download_batch_outputs.py --step 2 --parse
```

## 6. Parse and Validate Outputs

Parse, validate, deduplicate, and combine into training format (messages format JSONL):
```bash
python parse_validate_outputs.py --step 2
```

This creates the final training-ready files from all batch outputs.

---

## Complete Workflow Example
```bash
# 1. Split dataset
python split_temporal_bins.py data/scotus_final_clean.csv

# 2. Set API key
export OPENAI_API_KEY='your-key-here'

# 3. Prepare batches for step 1
python scotus_prepare_batches.py --step 1

# 4. Submit batches (using your submission script)
python submit_one.py data/step_1/prepared_batches/train_batch_001.jsonl --desc "Step 1 train batch 1"
# ... repeat for all batches

# 5. Download outputs when complete
python download_batch_outputs.py --step 1 --parse

# 6. Parse and create training files
python parse_validate_outputs.py --step 1

# Repeat for steps 2-10
```

---

## Output Structure
```
data/
├── scotus_final_clean.csv
├── bin_statistics.json
├── step_1/
│   ├── train.csv
│   ├── validation.csv
│   └── prepared_batches/
│       ├── train_batch_001.jsonl
│       ├── train_batch_002.jsonl
│       ├── val_batch_001.jsonl
│       └── manifest.csv
├── step_2/
│   ├── train.csv
│   ├── validation.csv
│   └── prepared_batches/
│       └── ...
└── ...
```

---

## Tips

- **Check manifest.csv** before submitting to see how many batches were created
- **Submit batches gradually** to one at a time, other wise will hit usage error
- **Track batch IDs** in the manifest as you submit them
- The default batch size (1.6M tokens) keeps you safely under OpenAI's 2M enqueued token limit