**Continual Learning Replay Buffer Preprocessor**
This repository contains the replay_buffer_preprocessor.pyscript, which is used to generate augmented training datasets for a Fixed-Size Experience Replay experiment in the context of Continual Learning (CL) and Temporal Distribution Shift (TDS).

The script processes sequential training data (Supreme Court cases split into 10 temporal steps,$T_1$through$T_{10}$) and outputs two distinct sets of training files required for the thesis's two experimental groups: the Baseline Group (Original Data Only) and the Replay Group (Original Data + Fixed-Size Replay Buffer).

## 1. Scientific Methodology Implemented

The core function implements the Fixed Memory Budget paradigm (Rebuffi et al., 2017):

- **Incremental Buffer Growth ($B_{step}$)**: After each step's training, 200 boxes are randomly sampled and added to the central buffer.
- **Fixed Replay Size ($R_{fixed}$)**: At every subsequent step ($T_2$to$T_{10}$), exactly 5,000 cases are sampled from the buffer for rehearsal.
- **Primacy Bias / Upsampling**: When the total buffer size is less than 5,000 cases (during early steps), the script uses sampling with replacement ( random.choices). This implements a controlled "Primacy Bias," prioritizing repeated exposure for the oldest, most critical legal knowledge at high risk of catastrophic forgetting.
- **Output Separation**: The script ensures that the training data for the Baseline Group (sequential training) is never contaminated by the replay samples.

## 2. Prerequisites and Data Structure

The script expects the original, unprocessed data to be organized in a specific directory structure under a common data/ root:
```
data/
└── Step_1/
    └── finetuning_ready/
        ├── train.jsonl
        └── val.jsonl
└── Step_2/
    └── finetuning_ready/
        ├── train.jsonl
        └── val.jsonl
...
└── Step_10/
    └── finetuning_ready/
        ├── train.jsonl
        └── val.jsonl
```

## 3. Configuration

Key parameters are set within the `main()` function. These values must match the parameters defined in the thesis methodology (Section 4.3).

| Constant | Default Value | Description |
| :---: | :---: | :--- |
| `DATA_ROOT` | `"data"` | Root folder containing all `Step_n` directories. |
| `BUFFER_SIZE_PER_STEP` | `200` | $B_{step}$. Number of unique samples stored from each step. |
| `FIXED_REPLAY_SIZE` | `5000` | $R_{fixed}$. Total number of replay samples rehearsed in every augmented training batch. |
| `RANDOM_SEED` | `42` | Ensures reproducibility of the random sampling process. |

---

## 4. Usage

To run the full preprocessing pipeline:

```
python create_buffer.py
```

The script will iterate through $T_1$ to $T_{10}$, calculating the buffer composition, and saving the final, ready-to-train files into the same /finetuning_ready/ directory.

---

## 5. Output Files

For each Step ($T_i$), the script generates two new training files that are used directly in the fine-tuning code:

| File Name | Experimental Group | Content |
| :--- | :--- | :--- |
| `train_original.jsonl` | **Baseline** (Sequential LoRA) | Contains **only** the original $T_i$ training data. |
| `train_augmented_replay.jsonl` | **Replay** (LoRA + Buffer) | Contains the original $T_i$ data **mixed with** $R_{fixed}$ replay samples, fully shuffled. |
| `val_original.jsonl` | **Validation** (Both Groups) | Contains the original $T_i$ validation data. |

---

## 6. Key Functions in `ReplayBufferPreprocessor`

| Function | Purpose |
| :--- | :--- |
| `process_all_steps()` | Main pipeline control. Manages the loop from $T_1$ to $T_{10}$ and calls core functions sequentially. |
| `add_to_buffer()` | Implements the $B_{step}$ policy: samples 200 cases from the current step and adds them to the global buffer. |
| `create_replay_dataset()` | Implements the $R_{fixed}$ policy: draws 5,000 cases (using upsampling if needed) from the current buffer, mixes them with the current training data, and shuffles the output. |
| `verify_output()` | Confirms all output files exist and checks that the file sizes reflect the correct number of replay samples added. |

