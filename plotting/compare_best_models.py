#!/usr/bin/env python3
"""
Compare best models across different strategies and runs.
Plots ROC curves and balanced accuracy by run (no means).
Individual runs colored by strategy, differentiated by marker.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import csv

# Professional styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#2E86AB', '#A23B72', '#F18F01']  # Professional palette: blue, purple, orange
MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x', '|']


def find_best_model_runs(root: Path) -> List[Path]:
    return list(root.rglob("**/best_models/**/run_*.json"))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open('r') as f:
        return json.load(f)


def collect_data(root: Path, strategies: List[str]) -> List[Dict[str, Any]]:
    """Collect best model metadata and thresholds from best_models directories."""
    runs = []
    for run_file in find_best_model_runs(root):
        try:
            data = load_json(run_file)
        except Exception:
            continue
        strategy = data.get('strategy')
        if strategy not in strategies:
            continue
        entry = {
            'strategy': strategy,
            'bundle': data.get('bundle_name'),
            'run': data.get('run'),
            'round': data.get('round'),
            'threshold': data.get('threshold'),
            'recall': data.get('recall'),
            'spec': data.get('spec'),
            'auc': data.get('auc'),
            'auprc': data.get('auprc'),
            'balanced_accuracy': data.get('balanced_accuracy'),
            'alerts_per_1000': data.get('alerts_per_1000'),
            'source_round_json': data.get('source_round_json'),
            'path': run_file,
        }
        # load reconstructed thresholds if available
        src = entry['source_round_json']
        if src:
            src_path = Path(src)
            if not src_path.is_absolute():
                src_path = root / src
            if src_path.exists():
                try:
                    src_data = load_json(src_path)
                    thresholds = src_data.get('metrics', {}).get('all_thresholds', [])
                    entry['thresholds'] = thresholds
                except Exception:
                    entry['thresholds'] = []
            else:
                entry['thresholds'] = []
        else:
            entry['thresholds'] = []
        runs.append(entry)
    return runs


def plot_roc_curves(runs: List[Dict[str, Any]], outdir: Path) -> Path:
    """Plot ROC curves per run, color by strategy, marker by run."""
    outdir.mkdir(parents=True, exist_ok=True)
    strategies = sorted({r['strategy'] for r in runs})
    
    fig, ax = plt.subplots(figsize=(11, 9))
    
    # Track added legend items to avoid duplicates
    added_legend = set()
    
    for strat_idx, strat in enumerate(strategies):
        subset = [r for r in runs if r['strategy'] == strat and r.get('thresholds')]
        runs_ids = sorted({r['run'] for r in subset})
        color = COLORS[strat_idx % len(COLORS)]
        
        for run_idx, runid in enumerate(runs_ids):
            run_subset = [r for r in subset if r['run'] == runid]
            marker = MARKERS[run_idx % len(MARKERS)]
            
            for r in run_subset:
                th = r['thresholds']
                fprs = np.array([t.get('fpr', 0.0) for t in th])
                tprs = np.array([t.get('tpr', 0.0) for t in th])
                if len(fprs) < 2:
                    continue
                order = np.argsort(fprs)
                fprs_sorted = fprs[order]
                tprs_sorted = tprs[order]
                
                # Add to legend only once per (strategy, run)
                legend_key = (strat, runid)
                label = f"{strat} run{runid}" if legend_key not in added_legend else None
                if legend_key not in added_legend:
                    added_legend.add(legend_key)
                
                ax.plot(fprs_sorted, tprs_sorted, color=color, marker=marker, markersize=6, 
                       alpha=0.75, linewidth=2.5, label=label)
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='#666666', linewidth=1.5, alpha=0.6)
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curves by Strategy and Run', fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    # Custom legend with smaller font, better placement
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=9, loc='lower right', framealpha=0.95, edgecolor='black')
    
    outpath = outdir / 'roc_curves.png'
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def plot_balanced_accuracy(runs: List[Dict[str, Any]], outdir: Path) -> Path:
    """Plot balanced accuracy vs recall scatter, color by strategy, shape by run."""
    outdir.mkdir(parents=True, exist_ok=True)
    strategies = sorted({r['strategy'] for r in runs})
    
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Track added legend items to avoid duplicates
    added_legend = set()
    
    for strat_idx, strat in enumerate(strategies):
        subset = [r for r in runs if r['strategy'] == strat]
        runs_ids = sorted({r['run'] for r in subset})
        color = COLORS[strat_idx % len(COLORS)]
        
        for run_idx, runid in enumerate(runs_ids):
            run_subset = [r for r in subset if r['run'] == runid]
            marker = MARKERS[run_idx % len(MARKERS)]
            
            for r in run_subset:
                recall = r.get('recall')
                ba = r.get('balanced_accuracy')
                if recall is not None and ba is not None:
                    # Add to legend only once per (strategy, run)
                    legend_key = (strat, runid)
                    label = f"{strat} run{runid}" if legend_key not in added_legend else None
                    if legend_key not in added_legend:
                        added_legend.add(legend_key)
                    
                    ax.scatter(recall, ba, color=color, marker=marker, s=120, alpha=0.8, 
                              label=label, edgecolor='#333333', linewidth=1.5)
    
    ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax.set_ylabel('Balanced Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Balanced Accuracy vs Recall by Strategy and Run', fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    # Custom legend with smaller font
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=9, loc='best', framealpha=0.95, edgecolor='black')
    
    outpath = outdir / 'balanced_accuracy_scatter.png'
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def save_runs_csv(runs: List[Dict[str, Any]], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / 'collected_best_models.csv'
    keys = ['strategy','bundle','run','round','threshold','recall','spec','auc','auprc','alerts_per_1000','path']
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in runs:
            row = {k: r.get(k) for k in keys}
            writer.writerow(row)
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description='Compare best models across strategies and runs (no means, individual runs only).'
    )
    parser.add_argument('--results-root', type=str, default='result', help='Root folder with result runs')
    parser.add_argument('--strategies', type=str, nargs='+', default=['FedAdam', 'FedProx', 'Scaffold'])
    parser.add_argument('--outdir', type=str, default='result/plots/best_models_comparison')
    parser.add_argument('--bundle', type=str, nargs='*', default=None, help='Filter to specific bundle names (e.g. best_ROC_model)')
    args = parser.parse_args()

    root = Path(args.results_root)
    outdir = Path(args.outdir)

    runs = collect_data(root, args.strategies)
    
    # apply bundle filter if specified
    if args.bundle:
        runs = [r for r in runs if r.get('bundle') in args.bundle]
    
    if not runs:
        print(f'No best_model run JSONs found for strategies {args.strategies} under {root}')
        return

    print(f'Found {len(runs)} best-model entries for strategies {args.strategies}')
    
    # save collected data
    csvp = save_runs_csv(runs, outdir)
    saved = [csvp]
    
    # generate plots
    saved.append(plot_roc_curves(runs, outdir))
    saved.append(plot_balanced_accuracy(runs, outdir))
    
    print('\nSaved outputs:')
    for p in saved:
        print(f'  - {p}')


if __name__ == '__main__':
    main()
