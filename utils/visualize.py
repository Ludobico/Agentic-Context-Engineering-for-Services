import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Union, Optional
from utils import highlight_print
from config.getenv import GetEnv

env = GetEnv()
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path : Union[os.PathLike, str]):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file does not exist, check your path : {file_path}")
    return pd.read_csv(file_path)


def calculate_metrics(df : pd.DataFrame):
    df['cumulative_success'] = df['is_success'].expanding().mean() * 100

    window_size = min(20, len(df))
    df['rolling_success'] = df['is_success'].rolling(window=window_size, min_periods=1).mean() * 100

    df['hit_rate'] = df.apply(
        lambda x: (x['helpful_count_in_retrieved'] / x['retrieved_count']) * 100 if x['retrieved_count'] > 0 else 0,
        axis=1
    )
    df['rolling_hit_rate'] = df['hit_rate'].rolling(window=window_size, min_periods=1).mean()

    return df

def plot_dashboard(df : pd.DataFrame, dataset_name : str, output_path : Union[os.PathLike, str]):
    fig, axes = plt.subplots(3, 1, figsize=(12,15), sharex=True)

    sns.lineplot(data=df, x=df.index, y='cumulative_success', ax=axes[0], color="#1f77b4", label="Cumulative Accuracy", linewidth=2)
    sns.lineplot(data=df, x=df.index, y='rolling_success', ax=axes[0], color='#ff7f0e', linestyle='--', label='Moving Avg (Trend)', alpha=0.8)
    axes[0].set_ylabel('Success Rate (%)', fontsize=12)
    axes[0].set_title(f'[{dataset_name}] Learning Curve: Self-Improvement', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    sns.lineplot(data=df, x=df.index, y='playbook_size', ax=axes[1], color='#2ca02c', label='Total Knowledge Entries', linewidth=2)

    max_size_detected = df['playbook_size'].max()
    axes[1].axhline(y=max_size_detected, color='red', linestyle=':', alpha=0.5, label=f'Max Size Observed ({max_size_detected})')
    
    axes[1].set_ylabel('Playbook Size (Count)', fontsize=12)
    axes[1].set_title('Knowledge Evolution: Accumulation & Pruning', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.bar(df.index, df['retrieved_count'], color='lightgray', label='Retrieved Contexts', alpha=0.7)
    ax3.set_ylabel('Count', fontsize=12)

    ax3_right = ax3.twinx()
    sns.lineplot(data=df, x=df.index, y='rolling_hit_rate', ax=ax3_right, color='#d62728', label='Hit Rate Trend (%)', linewidth=2)
    ax3_right.set_ylabel('Hit Rate (%)', color='#d62728', fontsize=12)
    ax3_right.tick_params(axis='y', labelcolor='#d62728')
    ax3_right.set_ylim(0, 100)

    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_right.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper left')
    
    axes[2].set_title('Retrieval Dynamics: Usage & Utility', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Task Progress (Task ID)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    highlight_print(f"Graph Saved: {output_path}", 'green')

def plot_internal_impact(csv_path : Union[os.PathLike, str], dataset_name : str, output_dir : Union[os.PathLike, str]):
    df = pd.read_csv(csv_path)

    group_helpful = df[df['helpful_count_in_retrieved'] > 0]
    group_neutral = df[df['helpful_count_in_retrieved'] == 0]

    success_helpful = group_helpful['is_success'].mean() * 100
    success_neutral = group_neutral['is_success'].mean() * 100

    plt.figure(figsize=(8, 6))
    categories = ['ACE (Low Utility)\n(Helpful=0)', 'ACE (High Utility)\n(Helpful>0)']
    values = [success_neutral, success_helpful]
    colors = ['#95a5a6', '#3498db']

    bars = plt.bar(categories, values, color=colors, width=0.5, edgecolor='black', linewidth=1)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=15, fontweight='bold')
        
    plt.text(0, success_neutral/2, f"N={len(group_neutral)}", ha='center', color='white', fontweight='bold')
    plt.text(1, success_helpful/2, f"N={len(group_helpful)}", ha='center', color='white', fontweight='bold')

    gap = success_helpful - success_neutral
    plt.annotate(f"+{gap:.1f}%p Boost", 
                 xy=(0.5, (success_helpful + success_neutral)/2), 
                 ha='center', fontsize=12, color='red', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Impact of Retrieved Knowledge Quality', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    output_path = os.path.join(output_dir, dataset_name)
    plt.savefig(output_path, dpi=300)
    highlight_print(f"Graph Saved: {output_path}", 'green')

def main(csv_path : Union[os.PathLike, str],
         dataset_name : Optional[str] = None,
         output_dir : Optional[Union[os.PathLike, str]] = None):
    df = load_data(csv_path)

    df = calculate_metrics(df)

    if dataset_name is None:
        dataset_name = os.path.basename(csv_path).split('.')[0] + '.png'
        dataset_name_impact = os.path.basename(csv_path).split('.')[0] + '_impact' + '.png'
    if output_dir is None:
        output_dir = env.get_figures_dir
    
    output_path = os.path.join(output_dir, dataset_name)
    
    plot_dashboard(df, dataset_name, output_path)
    plot_internal_impact(csv_path, dataset_name_impact, output_dir)

if __name__ == "__main__":
    target = os.path.join(env.get_log_dir, 'human_eval_metrics.csv')
    main(target)

