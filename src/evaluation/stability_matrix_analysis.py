#!/usr/bin/env python3
"""
Stability Matrix Analysis for Continual Learning
Analyzes perplexity-based task performance matrices for baseline and replay conditions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality plot parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

BASELINE_CSV = "/Users/ellahoang-simon/Documents/GitHub/thesis-continual-learning/data/perplexity/stability_perplexity_baseline.csv"
REPLAY_CSV = "/Users/ellahoang-simon/Documents/GitHub/thesis-continual-learning/data/perplexity/stability_perplexity_matrix_replay.csv"
OUTPUT_DIR = "stability_analysis_outputs"
FIGURES_DIR = "stability_analysis_outputs/figures"

# Create output directories
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_and_clean_data(filepath):
    """Load CSV and clean Training_Step column."""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"  Columns found: {df.columns.tolist()}")
    print(f"  Shape: {df.shape}")
    print(f"  Raw model names: {df['Model_Family'].unique()}")
    
    # Standardize model family names
    model_name_mapping = {
        'pythia2.8': 'pythia_2.8B',
        'pythia_2.8': 'pythia_2.8B',
        'pythia-2.8': 'pythia_2.8B',
        'pythia_2.8b': 'pythia_2.8B',
        'neo1.3': 'neo_1.3B',
        'neo_1.3': 'neo_1.3B',
        'neo-1.3': 'neo_1.3B',
        'neo_1.3b': 'neo_1.3B',
        'phi1.5': 'phi_1.5',
        'phi-1.5': 'phi_1.5',
        'phi_1.5': 'phi_1.5',
        't5-base': 't5_base',
        't5_base': 't5_base',
        't5base': 't5_base'
    }
    
    # Apply mapping (case-insensitive)
    df['Model_Family'] = df['Model_Family'].str.lower().map(
        lambda x: model_name_mapping.get(x, x)
    )
    
    # Handle Training_Step - can be "step_X" or just "X"
    if df['Training_Step'].dtype == 'object':
        # If it's a string, remove "step_" prefix if present
        df['Training_Step'] = df['Training_Step'].astype(str).str.replace('step_', '', regex=False)
    
    # Convert to integer
    df['Training_Step'] = df['Training_Step'].astype(int)
    
    # Ensure Data_Step is integer
    df['Data_Step'] = df['Data_Step'].astype(int)
    
    print(f"  ✓ Cleaned data: Training steps {df['Training_Step'].min()}-{df['Training_Step'].max()}")
    print(f"  ✓ Standardized models: {', '.join(df['Model_Family'].unique())}")
    
    return df

def create_stability_matrix(df, model_family):
    """
    Create a 10x10 stability matrix for a specific model family.
    Rows = Training Step, Columns = Data Step
    """
    model_df = df[df['Model_Family'] == model_family].copy()
    
    # Pivot to create matrix
    matrix = model_df.pivot_table(
        values='Perplexity',
        index='Training_Step',
        columns='Data_Step',
        aggfunc='mean'
    )
    
    return matrix

# ============================================================================
# METRIC CALCULATIONS
# ============================================================================

def calculate_forgetting(matrix):
    """
    Calculate catastrophic forgetting for each training step.
    F_i = average increase in PPL on past tasks compared to their historical best.
    """
    forgetting_scores = {}
    
    for train_step in range(2, 11):  # Steps 2-10 (need history)
        step_forgetting = []
        
        for data_step in range(1, train_step):  # Only past tasks
            # Check if this combination exists in the matrix
            if train_step not in matrix.index or data_step not in matrix.columns:
                continue
                
            # Current PPL on this old task
            current_ppl = matrix.at[train_step, data_step]
            
            if pd.isna(current_ppl):
                continue
            
            # Best (minimum) PPL ever achieved on this task
            historical_ppls = []
            for step in range(data_step, train_step):
                if step in matrix.index and data_step in matrix.columns:
                    val = matrix.at[step, data_step]
                    if not pd.isna(val):
                        historical_ppls.append(val)
            
            if not historical_ppls:
                continue
                
            min_ppl = min(historical_ppls)
            
            # Forgetting: positive = worse performance
            forgetting = current_ppl - min_ppl
            step_forgetting.append(forgetting)
        
        # Average forgetting across all past tasks
        if step_forgetting:
            forgetting_scores[train_step] = np.mean(step_forgetting)
    
    return forgetting_scores

def calculate_stability_ratio(matrix):
    """
    Calculate stability ratio (retention) for each training step.
    S_i = average ratio of best historical PPL to current PPL.
    1.0 = perfect retention, <1.0 = forgetting
    """
    stability_ratios = {}
    
    for train_step in range(2, 11):  # Steps 2-10
        step_ratios = []
        
        for data_step in range(1, train_step):  # Only past tasks
            # Check if this combination exists
            if train_step not in matrix.index or data_step not in matrix.columns:
                continue
                
            # Current PPL on this old task
            current_ppl = matrix.at[train_step, data_step]
            
            if pd.isna(current_ppl):
                continue
            
            # Best (minimum) PPL ever achieved on this task
            historical_ppls = []
            for step in range(data_step, train_step):
                if step in matrix.index and data_step in matrix.columns:
                    val = matrix.at[step, data_step]
                    if not pd.isna(val):
                        historical_ppls.append(val)
            
            if not historical_ppls:
                continue
                
            min_ppl = min(historical_ppls)
            
            # Stability ratio: closer to 1.0 = better retention
            if current_ppl > 0:  # Avoid division by zero
                stability = min_ppl / current_ppl
                step_ratios.append(stability)
        
        # Average stability across all past tasks
        if step_ratios:
            stability_ratios[train_step] = np.mean(step_ratios)
    
    return stability_ratios

def calculate_backward_transfer(matrix):
    """
    Calculate Backward Transfer (BWT): average change in performance on past tasks.
    """
    bwt_scores = []
    
    for data_step in range(1, 10):  # Tasks 1-9 (exclude last task)
        # Check if values exist
        if 10 not in matrix.index or data_step not in matrix.columns:
            continue
        if data_step not in matrix.index or data_step not in matrix.columns:
            continue
            
        # PPL on this task after training on all 10 tasks
        final_ppl = matrix.at[10, data_step]
        
        # PPL on this task immediately after training on it
        initial_ppl = matrix.at[data_step, data_step]
        
        if pd.isna(final_ppl) or pd.isna(initial_ppl):
            continue
        
        # BWT: positive = forgetting (performance degraded)
        bwt = final_ppl - initial_ppl
        bwt_scores.append(bwt)
    
    return np.mean(bwt_scores) if bwt_scores else np.nan

def calculate_forward_transfer(matrix):
    """
    Calculate Forward Transfer (FWT): performance on future tasks before training.
    """
    fwt_scores = []
    
    for data_step in range(2, 11):  # Tasks 2-10
        # Check if values exist
        if (data_step - 1) not in matrix.index or data_step not in matrix.columns:
            continue
        if data_step not in matrix.index or data_step not in matrix.columns:
            continue
            
        # PPL on this task before training on it (using model from previous step)
        before_ppl = matrix.at[data_step - 1, data_step]
        
        # PPL on this task after training on it
        after_ppl = matrix.at[data_step, data_step]
        
        if pd.isna(before_ppl) or pd.isna(after_ppl):
            continue
        
        # FWT: positive = helped by prior training
        fwt = before_ppl - after_ppl
        fwt_scores.append(fwt)
    
    return np.mean(fwt_scores) if fwt_scores else np.nan

def calculate_all_metrics(matrix, model_family, condition):
    """Calculate all metrics for a model."""
    forgetting = calculate_forgetting(matrix)
    stability = calculate_stability_ratio(matrix)
    bwt = calculate_backward_transfer(matrix)
    fwt = calculate_forward_transfer(matrix)
    
    # Final task performance (diagonal) - check if exists
    if 10 in matrix.index and 10 in matrix.columns:
        final_performance = matrix.at[10, 10]
    else:
        final_performance = np.nan
    
    # Average performance on all tasks at step 10 - check if row exists
    if 10 in matrix.index:
        avg_performance = matrix.loc[10, :].mean()
    else:
        avg_performance = np.nan
    
    return {
        'Model_Family': model_family,
        'Condition': condition,
        'Final_Task_PPL': final_performance,
        'Avg_PPL_Step10': avg_performance,
        'Backward_Transfer': bwt,
        'Forward_Transfer': fwt,
        'Avg_Forgetting_Step10': forgetting.get(10, np.nan),
        'Avg_Stability_Step10': stability.get(10, np.nan),
        'Forgetting_by_Step': forgetting,
        'Stability_by_Step': stability
    }

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_stability_heatmap(matrix, model_family, condition, output_path):
    """Create heatmap of stability matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Custom colormap: lower PPL = better = darker green
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    # Create heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        cbar_kws={'label': 'Perplexity (lower is better)'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=matrix.min().min(),
        vmax=matrix.max().max()
    )
    
    ax.set_xlabel('Data Step (Validation Set)', fontweight='bold')
    ax.set_ylabel('Training Step (Model Checkpoint)', fontweight='bold')
    ax.set_title(f'Stability Matrix: {model_family} - {condition}\n(Rows=Model Step, Cols=Data Step)', 
                 fontweight='bold', pad=15)
    
    # Highlight diagonal (current task performance)
    for i in range(min(matrix.shape)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                   edgecolor='blue', lw=3))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_forgetting_trajectories(baseline_metrics, replay_metrics, output_path):
    """Plot forgetting over training steps for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    model_families = ['pythia_2.8B', 'neo_1.3B', 't5_base', 'phi_1.5']
    
    for idx, model_family in enumerate(model_families):
        ax = axes[idx]
        
        # Get data for this model
        baseline = [m for m in baseline_metrics if m['Model_Family'] == model_family][0]
        replay = [m for m in replay_metrics if m['Model_Family'] == model_family][0]
        
        # Extract forgetting trajectories
        baseline_steps = sorted(baseline['Forgetting_by_Step'].keys())
        baseline_values = [baseline['Forgetting_by_Step'][s] for s in baseline_steps]
        
        replay_steps = sorted(replay['Forgetting_by_Step'].keys())
        replay_values = [replay['Forgetting_by_Step'][s] for s in replay_steps]
        
        # Plot
        ax.plot(baseline_steps, baseline_values, marker='o', linewidth=2.5, 
                markersize=7, label='Baseline', color='#E63946', alpha=0.9)
        ax.plot(replay_steps, replay_values, marker='s', linewidth=2.5, 
                markersize=7, label='Replay', color='#06A77D', alpha=0.9)
        
        ax.set_xlabel('Training Step', fontweight='bold')
        ax.set_ylabel('Average Forgetting (PPL)', fontweight='bold')
        ax.set_title(f'{model_family}', fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xticks(range(2, 11))
    
    fig.suptitle('Catastrophic Forgetting Trajectories: Baseline vs Replay', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_stability_trajectories(baseline_metrics, replay_metrics, output_path):
    """Plot stability ratio over training steps for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    model_families = ['pythia_2.8B', 'neo_1.3B', 't5_base', 'phi_1.5']
    
    for idx, model_family in enumerate(model_families):
        ax = axes[idx]
        
        # Get data for this model
        baseline = [m for m in baseline_metrics if m['Model_Family'] == model_family][0]
        replay = [m for m in replay_metrics if m['Model_Family'] == model_family][0]
        
        # Extract stability trajectories
        baseline_steps = sorted(baseline['Stability_by_Step'].keys())
        baseline_values = [baseline['Stability_by_Step'][s] * 100 for s in baseline_steps]  # Convert to %
        
        replay_steps = sorted(replay['Stability_by_Step'].keys())
        replay_values = [replay['Stability_by_Step'][s] * 100 for s in replay_steps]
        
        # Plot
        ax.plot(baseline_steps, baseline_values, marker='o', linewidth=2.5, 
                markersize=7, label='Baseline', color='#E63946', alpha=0.9)
        ax.plot(replay_steps, replay_values, marker='s', linewidth=2.5, 
                markersize=7, label='Replay', color='#06A77D', alpha=0.9)
        
        ax.set_xlabel('Training Step', fontweight='bold')
        ax.set_ylabel('Stability Ratio (%)', fontweight='bold')
        ax.set_title(f'{model_family}', fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xticks(range(2, 11))
        ax.set_ylim(0, 110)
    
    fig.suptitle('Knowledge Stability Trajectories: Baseline vs Replay', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_comparison_bar_chart(baseline_metrics, replay_metrics, output_path):
    """Bar chart comparing baseline vs replay on key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    model_families = ['pythia_2.8B', 'neo_1.3B', 't5_base', 'phi_1.5']
    
    # Extract metrics with clean labels
    metrics_to_plot = [
        ('Final_Task_PPL', 'Final Task Perplexity', False),
        ('Avg_PPL_Step10', 'Average Perplexity', False),
        ('Avg_Forgetting_Step10', 'Average Forgetting', False),
        ('Avg_Stability_Step10', 'Average Stability Ratio (%)', True)
    ]
    
    for idx, (metric_name, y_label, multiply_100) in enumerate(metrics_to_plot):
        ax = axes.flatten()[idx]
        
        baseline_values = []
        replay_values = []
        
        for model in model_families:
            b_val = [m[metric_name] for m in baseline_metrics if m['Model_Family'] == model][0]
            r_val = [m[metric_name] for m in replay_metrics if m['Model_Family'] == model][0]
            
            if multiply_100:
                b_val *= 100
                r_val *= 100
            
            baseline_values.append(b_val)
            replay_values.append(r_val)
        
        x = np.arange(len(model_families))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', 
                      color='#E63946', edgecolor='black', linewidth=1, alpha=0.85)
        bars2 = ax.bar(x + width/2, replay_values, width, label='Replay', 
                      color='#06A77D', edgecolor='black', linewidth=1, alpha=0.85)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel(y_label, fontweight='bold')
        ax.set_title(y_label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_families, rotation=0)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("="*80)
    print("STABILITY MATRIX ANALYSIS")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    baseline_df = load_and_clean_data(BASELINE_CSV)
    replay_df = load_and_clean_data(REPLAY_CSV)
    print(f"✓ Baseline: {len(baseline_df)} rows")
    print(f"✓ Replay: {len(replay_df)} rows")
    print()
    
    # Get unique model families
    model_families = baseline_df['Model_Family'].unique()
    print(f"Model families: {', '.join(model_families)}")
    print()
    
    # Process each model family
    baseline_metrics = []
    replay_metrics = []
    
    print("Creating stability matrices and calculating metrics...")
    print()
    
    for model_family in model_families:
        print(f"Processing {model_family}...")
        
        # Create matrices
        baseline_matrix = create_stability_matrix(baseline_df, model_family)
        replay_matrix = create_stability_matrix(replay_df, model_family)
        
        # Save matrices to CSV
        baseline_matrix.to_csv(f"{OUTPUT_DIR}/matrix_{model_family}_baseline.csv")
        replay_matrix.to_csv(f"{OUTPUT_DIR}/matrix_{model_family}_replay.csv")
        print(f"  ✓ Saved matrices to CSV")
        
        # Calculate metrics
        baseline_m = calculate_all_metrics(baseline_matrix, model_family, 'Baseline')
        replay_m = calculate_all_metrics(replay_matrix, model_family, 'Replay')
        
        baseline_metrics.append(baseline_m)
        replay_metrics.append(replay_m)
        
        # Create heatmaps
        plot_stability_heatmap(
            baseline_matrix, model_family, 'Baseline',
            f"{FIGURES_DIR}/heatmap_{model_family}_baseline.png"
        )
        plot_stability_heatmap(
            replay_matrix, model_family, 'Replay',
            f"{FIGURES_DIR}/heatmap_{model_family}_replay.png"
        )
        print(f"  ✓ Generated heatmaps")
        print()
    
    # Create summary tables
    print("Creating summary tables...")
    
    # Baseline summary
    baseline_summary = pd.DataFrame([
        {
            'Model_Family': m['Model_Family'],
            'Final_Task_PPL': m['Final_Task_PPL'],
            'Avg_PPL_Step10': m['Avg_PPL_Step10'],
            'Backward_Transfer': m['Backward_Transfer'],
            'Forward_Transfer': m['Forward_Transfer'],
            'Avg_Forgetting_Step10': m['Avg_Forgetting_Step10'],
            'Avg_Stability_Step10': m['Avg_Stability_Step10']
        }
        for m in baseline_metrics
    ])
    baseline_summary.to_csv(f"{OUTPUT_DIR}/summary_baseline.csv", index=False)
    
    # Replay summary
    replay_summary = pd.DataFrame([
        {
            'Model_Family': m['Model_Family'],
            'Final_Task_PPL': m['Final_Task_PPL'],
            'Avg_PPL_Step10': m['Avg_PPL_Step10'],
            'Backward_Transfer': m['Backward_Transfer'],
            'Forward_Transfer': m['Forward_Transfer'],
            'Avg_Forgetting_Step10': m['Avg_Forgetting_Step10'],
            'Avg_Stability_Step10': m['Avg_Stability_Step10']
        }
        for m in replay_metrics
    ])
    replay_summary.to_csv(f"{OUTPUT_DIR}/summary_replay.csv", index=False)
    print(f"  ✓ Saved summary tables")
    print()
    
    # Create comparison visualizations
    print("Creating comparison visualizations...")
    plot_forgetting_trajectories(
        baseline_metrics, replay_metrics,
        f"{FIGURES_DIR}/forgetting_trajectories.png"
    )
    print(f"  ✓ Forgetting trajectories")
    
    plot_stability_trajectories(
        baseline_metrics, replay_metrics,
        f"{FIGURES_DIR}/stability_trajectories.png"
    )
    print(f"  ✓ Stability trajectories")
    
    plot_comparison_bar_chart(
        baseline_metrics, replay_metrics,
        f"{FIGURES_DIR}/comparison_bar_chart.png"
    )
    print(f"  ✓ Comparison bar chart")
    print()
    
    # Print summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print()
    
    print("BASELINE:")
    print(baseline_summary.to_string(index=False))
    print()
    
    print("REPLAY:")
    print(replay_summary.to_string(index=False))
    print()
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"Figures directory: {FIGURES_DIR}/")
    print()
    print("Files generated:")
    print("  - matrix_[model]_[condition].csv (8 files)")
    print("  - summary_baseline.csv")
    print("  - summary_replay.csv")
    print("  - heatmap_[model]_[condition].png (8 files)")
    print("  - forgetting_trajectories.png")
    print("  - stability_trajectories.png")
    print("  - comparison_bar_chart.png")
    print()

if __name__ == "__main__":
    main()