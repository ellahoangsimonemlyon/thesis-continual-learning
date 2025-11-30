# analyze_judge.py
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

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

# CONFIG
CONDITION_NAME = "Replay"
# EVALUATIONS_DIR = "data/evaluation_results/baseline/results_gpt_judge"
# OUTPUT_DIR = "data/evaluation_results/baseline/analysis_outputs"
EVALUATIONS_DIR = "data/evaluation_results/replay/results_gpt_judge"
OUTPUT_DIR = "data/evaluation_results/replay/analysis_outputs"


FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# DATA LOADING
def load_all_evaluations():
    """Load all evaluation files into a single dataframe."""
    all_data = []
    
    eval_files = sorted([f for f in os.listdir(EVALUATIONS_DIR) if f.endswith('.jsonl')])
    
    for eval_file in eval_files:
        filepath = os.path.join(EVALUATIONS_DIR, eval_file)
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                row = {
                    'model_family': data['model_family'],
                    'model_step': data['model_step'],
                    'question_step': data['question_step'],
                    'q_number': data['q_number'],
                    'difficulty': data['difficulty'],
                    'factual_correctness': data['evaluation']['factual_correctness'],
                    'coverage': data['evaluation']['coverage_of_key_points'],
                    'nuance': data['evaluation']['nuance_and_reasoning'],
                    'expression': data['evaluation']['expression_and_clarity'],
                    'overall_score': data['evaluation']['overall_score'],
                    'verdict': data['evaluation']['verdict'],
                    'hallucination': data['evaluation']['hallucination_detected'],
                    'feedback': data['evaluation']['short_feedback']
                }
                all_data.append(row)
    
    return pd.DataFrame(all_data)

# CSV EXPORTS
def export_summary_tables(df):
    """Export detailed summary statistics as CSV files."""
    
    # 1. Overall statistics by model family
    overall_stats = df.groupby('model_family').agg({
        'overall_score': ['mean', 'std', 'min', 'max'],
        'factual_correctness': ['mean', 'std'],
        'coverage': ['mean', 'std'],
        'nuance': ['mean', 'std'],
        'expression': ['mean', 'std'],
        'hallucination': ['sum', 'mean']
    }).round(3)
    overall_stats.columns = ['_'.join(col).strip() for col in overall_stats.columns.values]
    overall_stats.to_csv(os.path.join(OUTPUT_DIR, "01_overall_statistics_by_model.csv"))
    print(f"Saved: 01_overall_statistics_by_model.csv")
    
    # 2. Performance by model step (catastrophic forgetting)
    by_step = df.groupby(['model_family', 'model_step']).agg({
        'overall_score': ['mean', 'std', 'count'],
        'factual_correctness': 'mean',
        'coverage': 'mean',
        'nuance': 'mean',
        'expression': 'mean',
        'hallucination': ['sum', 'mean']
    }).round(3)
    by_step.columns = ['_'.join(col).strip() for col in by_step.columns.values]
    by_step.to_csv(os.path.join(OUTPUT_DIR, "02_performance_by_model_step.csv"))
    print(f"Saved: 02_performance_by_model_step.csv")
    
    # 3. Performance by question step (temporal generalization)
    by_question = df.groupby(['model_family', 'question_step']).agg({
        'overall_score': ['mean', 'std', 'count'],
        'factual_correctness': 'mean',
        'coverage': 'mean',
        'nuance': 'mean',
        'expression': 'mean'
    }).round(3)
    by_question.columns = ['_'.join(col).strip() for col in by_question.columns.values]
    by_question.to_csv(os.path.join(OUTPUT_DIR, "03_performance_by_question_step.csv"))
    print(f"Saved: 03_performance_by_question_step.csv")
    
    # 4. Performance by difficulty level
    by_difficulty = df.groupby(['model_family', 'difficulty']).agg({
        'overall_score': ['mean', 'std', 'count'],
        'factual_correctness': 'mean',
        'coverage': 'mean',
        'nuance': 'mean',
        'expression': 'mean',
        'hallucination': ['sum', 'mean']
    }).round(3)
    by_difficulty.columns = ['_'.join(col).strip() for col in by_difficulty.columns.values]
    by_difficulty.to_csv(os.path.join(OUTPUT_DIR, "04_performance_by_difficulty.csv"))
    print(f"Saved: 04_performance_by_difficulty.csv")
    
    # 5. Forgetting matrices (model_step x question_step) for each model
    for model_family in df['model_family'].unique():
        family_df = df[df['model_family'] == model_family]
        
        # Overall score matrix
        matrix = family_df.pivot_table(
            values='overall_score',
            index='model_step',
            columns='question_step',
            aggfunc='mean'
        ).round(2)
        
        filename = f"05_forgetting_matrix_{model_family.replace(' ', '_').lower()}.csv"
        matrix.to_csv(os.path.join(OUTPUT_DIR, filename))
        print(f"Saved: {filename}")
    
    # 6. Detailed results (full dataset)
    df.to_csv(os.path.join(OUTPUT_DIR, "06_detailed_all_evaluations.csv"), index=False)
    print(f"Saved: 06_detailed_all_evaluations.csv")

# VISUALIZATIONS
def plot_overall_performance_by_model(df):
    """Bar chart: Overall performance by model family."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calculate means and stds
    summary = df.groupby('model_family')['overall_score'].agg(['mean', 'std'])
    
    # Sort by mean score
    summary = summary.sort_values('mean', ascending=False)
    
    # Create bar plot
    bars = ax.bar(range(len(summary)), summary['mean'], 
                   yerr=summary['std'], capsize=5,
                   color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary.index, rotation=0)
    ax.set_ylabel('Mean Overall Score (out of 10)')
    ax.set_xlabel('Model Architecture')
    ax.set_title(f'Overall Performance by Model Architecture - {CONDITION_NAME}\n(LLM-as-Judge Evaluation)', 
                 fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 10)
    
    # Add value labels on bars
    for i, (mean_val, std_val) in enumerate(zip(summary['mean'], summary['std'])):
        ax.text(i, mean_val + std_val + 0.2, f'{mean_val:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig01_overall_performance_by_model.png"), 
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "fig01_overall_performance_by_model.pdf"), 
                bbox_inches='tight')
    plt.close()
    print(f"Saved: fig01_overall_performance_by_model")

def plot_performance_over_steps(df):
    """Line chart: Performance over training steps."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'pythia_2.8B': '#2E86AB', 'neo_1.3B': '#A23B72', 
              't5_base': '#F18F01', 'phi_1.5': '#C73E1D'}
    
    for model_family in sorted(df['model_family'].unique()):
        family_df = df[df['model_family'] == model_family]
        
        # Group by model step and calculate mean
        step_means = family_df.groupby('model_step')['overall_score'].mean()
        
        ax.plot(step_means.index, step_means.values, 
                marker='o', linewidth=2.5, markersize=7,
                label=model_family, color=colors.get(model_family, 'gray'),
                alpha=0.9)
    
    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Mean Overall Score', fontweight='bold')
    ax.set_title(f'Performance Across Training Steps - {CONDITION_NAME}\n(Averaged Across All Question Steps)', 
                 fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig02_performance_over_training_steps.png"), 
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "fig02_performance_over_training_steps.pdf"), 
                bbox_inches='tight')
    plt.close()
    print(f"Saved: fig02_performance_over_training_steps")

def plot_forgetting_heatmaps(df):
    """Heatmaps: Forgetting matrices for each model (model_step x question_step)."""
    
    for model_family in sorted(df['model_family'].unique()):
        family_df = df[df['model_family'] == model_family]
        
        # Create pivot table
        matrix = family_df.pivot_table(
            values='overall_score',
            index='model_step',
            columns='question_step',
            aggfunc='mean'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Custom colormap: higher scores = better (green), lower = worse (red)
        cmap = sns.diverging_palette(10, 130, as_cmap=True)
        
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap, 
                    center=5, vmin=0, vmax=10,
                    cbar_kws={'label': 'Overall Score (0-10)'},
                    linewidths=0.5, linecolor='gray',
                    ax=ax)
        
        ax.set_xlabel('Question Step (Data Period)', fontweight='bold')
        ax.set_ylabel('Model Training Step', fontweight='bold')
        ax.set_title(f'Stability Matrix: {model_family} - {CONDITION_NAME}\n(Rows=Model Step, Cols=Question Step)', 
                     fontweight='bold', pad=15)
        
        # Highlight diagonal (current task performance)
        for i in range(min(matrix.shape)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                      edgecolor='black', lw=3))
        
        plt.tight_layout()
        
        filename = f"fig03_heatmap_{model_family.replace(' ', '_').lower()}"
        plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.png"), 
                    bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.pdf"), 
                    bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

def plot_performance_by_difficulty(df):
    """Grouped bar chart: Performance by difficulty level."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    difficulty_data = df.groupby(['model_family', 'difficulty'])['overall_score'].mean().unstack()
    
    # Reorder difficulties: E, M, H
    difficulty_order = ['E', 'M', 'H']
    difficulty_data = difficulty_data[difficulty_order]
    
    # Plot grouped bars
    x = np.arange(len(difficulty_data.index))
    width = 0.25
    
    colors = ['#76B041', '#F4A259', '#D62828']
    
    for i, (difficulty, color) in enumerate(zip(difficulty_order, colors)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, difficulty_data[difficulty], width, 
                      label=f'{difficulty} ({"Easy" if difficulty=="E" else "Medium" if difficulty=="M" else "Hard"})',
                      color=color, edgecolor='black', linewidth=1, alpha=0.85)
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model Architecture', fontweight='bold')
    ax.set_ylabel('Mean Overall Score', fontweight='bold')
    ax.set_title(f'Performance by Question Difficulty Level - {CONDITION_NAME}', 
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulty_data.index, rotation=0)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig04_performance_by_difficulty.png"), 
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "fig04_performance_by_difficulty.pdf"), 
                bbox_inches='tight')
    plt.close()
    print(f"Saved: fig04_performance_by_difficulty")

def plot_metric_breakdown(df):
    """Stacked bar chart: Breakdown of evaluation metrics by model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate mean scores for each metric
    metrics = ['factual_correctness', 'coverage', 'nuance', 'expression']
    metric_labels = ['Factual Correctness\n(0-5)', 'Coverage\n(0-5)', 
                     'Nuance\n(0-3)', 'Expression\n(0-2)']
    
    data = df.groupby('model_family')[metrics].mean()
    
    # Normalize to percentages of maximum possible score
    max_scores = [5, 5, 3, 2]
    data_normalized = data.copy()
    for i, col in enumerate(metrics):
        data_normalized[col] = (data[col] / max_scores[i]) * 100
    
    # Plot grouped bars
    x = np.arange(len(data.index))
    width = 0.2
    
    colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261']
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, data_normalized[metric], width, 
                      label=label, color=color, edgecolor='black', linewidth=0.8, alpha=0.85)
        
        # Add percentage labels on top of each bar
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Model Architecture', fontweight='bold')
    ax.set_ylabel('Score (% of Maximum)', fontweight='bold')
    ax.set_title(f'Metric Breakdown by Model Architecture - {CONDITION_NAME}\n(Normalized to % of Maximum Possible Score)', 
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=0)
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100) 
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig05_metric_breakdown.png"), 
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "fig05_metric_breakdown.pdf"), 
                bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig05_metric_breakdown")

def plot_hallucination_rates(df):
    """Bar chart: Hallucination rates by model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calculate hallucination rates
    total_by_model = df.groupby('model_family').size()
    hallucinations = df.groupby('model_family')['hallucination'].sum()
    hallucination_rate = (hallucinations / total_by_model * 100).sort_values(ascending=False)
    
    # Plot
    bars = ax.bar(range(len(hallucination_rate)), hallucination_rate.values,
                   color='#D62828', edgecolor='black', linewidth=1.2, alpha=0.8)
    
    ax.set_xticks(range(len(hallucination_rate)))
    ax.set_xticklabels(hallucination_rate.index, rotation=0)
    ax.set_ylabel('Hallucination Rate (%)', fontweight='bold')
    ax.set_xlabel('Model Architecture', fontweight='bold')
    ax.set_title(f'Hallucination Detection Rates by Model - {CONDITION_NAME}\n(LLM-as-Judge)', 
                 fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (idx, rate) in enumerate(hallucination_rate.items()):
        count = hallucinations[idx]
        total = total_by_model[idx]
        ax.text(i, rate + 0.5, f'{rate:.1f}%\n({int(count)}/{int(total)})', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig06_hallucination_rates.png"), 
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "fig06_hallucination_rates.pdf"), 
                bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig06_hallucination_rates")

def plot_forgetting_trajectory(df):
    """Line chart: How performance on early questions degrades over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    early_questions = [1, 2, 3]  # Track performance on first 3 question steps
    colors_q = ['#E63946', '#F77F00', '#06A77D']
    
    for idx, model_family in enumerate(sorted(df['model_family'].unique())):
        ax = axes[idx]
        family_df = df[df['model_family'] == model_family]
        
        for q_step, color in zip(early_questions, colors_q):
            q_df = family_df[family_df['question_step'] == q_step]
            trajectory = q_df.groupby('model_step')['overall_score'].mean()
            
            ax.plot(trajectory.index, trajectory.values,
                   marker='o', linewidth=2, markersize=6,
                   label=f'Question Step {q_step}', color=color, alpha=0.9)
        
        ax.set_xlabel('Model Training Step', fontweight='bold')
        ax.set_ylabel('Overall Score', fontweight='bold')
        ax.set_title(f'{model_family}', fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 10)
    
    fig.suptitle(f'Catastrophic Forgetting Trajectories - {CONDITION_NAME}\n(Performance on Early Questions Over Training)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig07_forgetting_trajectories.png"), 
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "fig07_forgetting_trajectories.pdf"), 
                bbox_inches='tight')
    plt.close()
    print(f"Saved: fig07_forgetting_trajectories")

# MAIN EXECUTION
def main():
    print(f"LLM-as-Judge Evaluation Results - {CONDITION_NAME}")
    print(f"Input: {EVALUATIONS_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Load data
    df = load_all_evaluations()
    
    # Export CSV tables
    export_summary_tables(df)
    
    # Generate visualizations
    plot_overall_performance_by_model(df)
    plot_performance_over_steps(df)
    plot_forgetting_heatmaps(df)
    plot_performance_by_difficulty(df)
    plot_metric_breakdown(df)
    plot_hallucination_rates(df)
    plot_forgetting_trajectory(df)
    

    print(f"CSV tables saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Total files generated: {len(os.listdir(OUTPUT_DIR)) + len(os.listdir(FIGURES_DIR))}")

if __name__ == "__main__":
    main()