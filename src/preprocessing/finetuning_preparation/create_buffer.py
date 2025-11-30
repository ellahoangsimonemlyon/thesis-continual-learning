import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any

class ReplayBufferPreprocessor:
    def __init__(self, data_root: str = "data", buffer_size_per_step: int = 200, fixed_replay_size: int = 5000):
        """
        Preprocess Supreme Court data for Continual Learning (CL) with 
        fixed-size Experience Replay.
        
        Args:
            data_root: Root directory containing Step_1, Step_2, etc folders.
            buffer_size_per_step: Number of samples to store from each step in buffer (B_step).
            fixed_replay_size: Fixed number of replay samples to use at each step (R_fixed).
        """
        self.data_root = Path(data_root)
        self.buffer_size_per_step = buffer_size_per_step
        self.fixed_replay_size = fixed_replay_size
        self.buffer: List[Dict[str, Any]] = []  # Stores samples for replay
        
    def load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file into list of dicts"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        return data
    
    def save_jsonl(self, data: List[Dict[str, Any]], file_path: Path):
        """Save list of dicts to JSONL file"""
        # Ensure the directory exists before saving
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                # Ensure buffer metadata is handled if present
                item_to_save = {k: v for k, v in item.items() if k != 'buffer_step'}
                f.write(json.dumps(item) + '\n')
    
    def get_step_paths(self, step_num: int):
        """Get standard paths for a given step."""
        step_dir = self.data_root / f"Step_{step_num}" / "finetuning_ready"
        return {
            'train_source': step_dir / "train.jsonl",
            'val_source': step_dir / "val.jsonl",
            'train_baseline': step_dir / "train_original.jsonl",
            'train_replay': step_dir / "train_augmented_replay.jsonl",
            'val_baseline': step_dir / "val_original.jsonl",  # This is the key verify_output needs
        }
    
    def add_to_buffer(self, data: List[Dict[str, Any]], step_num: int):
        """
        Randomly sample cases from current step and add them to the buffer. 
        This is the B_step mechanism.
        """
        total_size = len(data)
        
        # Random sample B_step cases
        sample_size = min(self.buffer_size_per_step, total_size)
        buffer_samples = random.sample(data, sample_size)
        
        # Add metadata to track the temporal origin of the sample
        for item in buffer_samples:
            # Note: A deep copy may be required if data structure is mutable, 
            # but is omitted here for JSONL efficiency.
            item['buffer_step'] = step_num
        
        self.buffer.extend(buffer_samples)
        
        print(f"  --> Added {len(buffer_samples):,} cases from Step {step_num} (B_step={self.buffer_size_per_step})")
        print(f"  --> Buffer now contains {len(self.buffer):,} total cases (B_total)")
    
    def create_replay_dataset(self, current_data: List[Dict[str, Any]], step_num: int) -> List[Dict[str, Any]]:
        """
        Combine current step data with FIXED SIZE replay buffer (R_fixed)
        """
        current_size = len(current_data)
        
        # Step 1: Baseline group and Replay group are identical 
        if step_num == 1 or len(self.buffer) == 0:
            print(f"  Step {step_num}: No replay augmentation needed.")
            return current_data
        
        # Use FIXED replay size (R_fixed)
        replay_size_needed = self.fixed_replay_size
        buffer_size = len(self.buffer)

        print(f"     Replay Composition for T{step_num}:")
        
        # Sampling Logic
        if buffer_size >= replay_size_needed:
            # Fixed-size sampling without replacement
            replay_samples = random.sample(self.buffer, replay_size_needed)
            print(f"     Sampled {replay_size_needed:,} cases from full buffer (No upsampling).")
        else:
            # Fixed-size sampling WITH replacement (Upsampling )
            replay_samples = random.choices(self.buffer, k=replay_size_needed)
            repetition_factor = replay_size_needed / buffer_size
            # This implements the Primacy Bias mechanism discussed in the methodology
            print(f"Upsampled buffer {repetition_factor:.1f}x to reach {replay_size_needed:,} samples (R_fixed).")
        
        # --- Combine and Shuffle ---
        combined_data = current_data + replay_samples
        random.shuffle(combined_data)
        
        # --- Reporting ---
        new_ratio = current_size / len(combined_data)
        replay_ratio = len(replay_samples) / len(combined_data)
        
        print(f"Final dataset size: {len(combined_data):,} cases")
        print(f"Current Data Ratio: {new_ratio:.1%} | Replay Data Ratio: {replay_ratio:.1%}")
        
        return combined_data
    
    def process_all_steps(self, seed: int = 42):
        """
        Main execution: Process all 10 steps and save augmented data for the Replay Group.
        """
        random.seed(seed)
        
        # Process each step
        for step in range(1, 11):
            print(f"PROCESSING STEP T{step} (Year T{step})")
     
            paths = self.get_step_paths(step)
            
            # Load Data 
            if not paths['train_source'].exists():
                print(f" {paths['train_source']} not found. Skipping step.")
                continue
            
            print(f"Loading data from {paths['train_source'].parent}...")
            train_data = self.load_jsonl(paths['train_source'])
            val_data = self.load_jsonl(paths['val_source']) if paths['val_source'].exists() else []
            
            # Create Replay-Augmented Training Set 
            train_with_replay = self.create_replay_dataset(
                current_data=train_data,
                step_num=step
            )
            
            # Save Datasets for BOTH Experimental Groups and original validation set
            self.save_jsonl(train_data, paths['train_baseline'])
            self.save_jsonl(train_with_replay, paths['train_replay'])
            self.save_jsonl(val_data, paths['val_baseline'])
            
            # Add Current Step to Buffer (For NEXT step)
            if step < 10:
                print(f"\nUpdating replay buffer for T{step+1}...")
                self.add_to_buffer(train_data, step)
            
            # Report Buffer Composition 
            if len(self.buffer) > 0 and step < 10:
                print(f"\n  Current Buffer Composition (B_total = {len(self.buffer):,}):")
                buffer_steps = {}
                for item in self.buffer:
                    step_id = item.get('buffer_step', 'unknown')
                    buffer_steps[step_id] = buffer_steps.get(step_id, 0) + 1
                
                # Report contribution from each source step
                for s in sorted(buffer_steps.keys()):
                    print(f"     Source T{s}: {buffer_steps[s]:,} cases")


        print("ALL STEPS PROCESSED SUCCESSFULLY")
        print(f"Final buffer contains {len(self.buffer):,} cases across steps 1-9")

    def save_buffer_state(self, output_path="buffer_metadata.json"):
        """Save buffer state for reproducibility"""
        buffer_metadata = {
            'buffer_size': len(self.buffer),
            'buffer_size_per_step': self.buffer_size_per_step,
            'fixed_replay_size': self.fixed_replay_size,
            'composition': {}
        }
        
        for item in self.buffer:
            step = item.get('buffer_step', 'unknown')
            buffer_metadata['composition'][step] = buffer_metadata['composition'].get(step, 0) + 1
        
        with open(output_path, 'w') as f:
            json.dump(buffer_metadata, f, indent=2)
        
        print(f"\nBuffer metadata saved to {output_path}")

def main():
    """Main execution function"""
    
    # EXPERIMENT CONFIGS
    DATA_ROOT = "data"  
    BUFFER_SIZE_PER_STEP = 200  # B_step: Store 200 cases from each step
    FIXED_REPLAY_SIZE = 5000    # R_fixed: Always use 5,000 replay cases
    RANDOM_SEED = 42
    
    # Initialize preprocessor
    preprocessor = ReplayBufferPreprocessor(
        data_root=DATA_ROOT,
        buffer_size_per_step=BUFFER_SIZE_PER_STEP,
        fixed_replay_size=FIXED_REPLAY_SIZE
    )
    
    # Execute full process
    preprocessor.process_all_steps(seed=RANDOM_SEED)
    preprocessor.save_buffer_state("buffer_metadata.json")
    
    print("\n Preprocessing complete! Files ready for fine-tuning.")
    print(f" - Baseline Group uses 'train_original.jsonl'")
    print(f" - Replay Group uses 'train_augmented_replay.jsonl'")
    

if __name__ == "__main__":
    main()
    
    