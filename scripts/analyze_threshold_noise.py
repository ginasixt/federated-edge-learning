#!/usr/bin/env python3
import json
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_threshold_series(rounds, recs, specs, counts, neigh_lower, neigh_upper, lower_series, upper_series, family_dir, t, outdir: Path, classification):
    outdir.mkdir(parents=True, exist_ok=True)
    rounds = np.array(rounds)
    recs = np.array(recs, dtype=float)
    specs = np.array(specs, dtype=float)
    counts = np.array(counts, dtype=int)

    fig, ax1 = plt.subplots(figsize=(9,4))
    ax1.plot(rounds, recs, marker='o', label=f'Recall thr={t}', color='C0')
    ax1.plot(rounds, specs, marker='s', linestyle='--', label=f'Spec thr={t}', color='C1')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Metric')
    ax1.set_ylim(-0.02,1.02)
    ax1.grid(alpha=0.2)

    # neighbor series
    if lower_series is not None:
        ax1.plot(rounds, lower_series, marker='x', linestyle=':', label=f'Neighbor lower', color='C2')
    if upper_series is not None:
        ax1.plot(rounds, upper_series, marker='x', linestyle=':', label=f'Neighbor upper', color='C3')

    # mark big jumps in recall
    for i in range(1, len(rounds)):
        if np.isfinite(recs[i-1]) and np.isfinite(recs[i]) and abs(recs[i]-recs[i-1]) > 0.08:
            ax1.annotate('', xy=(rounds[i], recs[i]), xytext=(rounds[i-1], recs[i-1]), arrowprops=dict(arrowstyle='->', color='red'))

    # twin axis for nearby counts
    ax2 = ax1.twinx()
    ax2.bar(rounds, counts, alpha=0.12, color='gray', label='samples near thr')
    ax2.set_ylabel('Samples near threshold')

    title = f"{family_dir}: thr={t} — {classification}"
    ax1.set_title(title)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='best', fontsize=8)

    outpath = outdir / f"{family_dir.replace('/','_')}_thr_{str(t).replace('.','p')}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    print(f"[INFO] Saved plot {outpath}")


def load_round_files(folder: Path):
    files = sorted(folder.glob("round_*_run_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    rounds = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        rounds.append((int(data.get("round", -1)), data))
    rounds.sort()
    return rounds


def find_threshold_row(thr_list, target):
    # thr_list: list of dicts with 'threshold' key
    if not thr_list:
        return None
    thr_vals = [float(x.get("threshold", np.nan)) for x in thr_list]
    diffs = [abs(v - target) for v in thr_vals]
    idx = int(np.argmin(diffs))
    return thr_list[idx]


def analyze_family(root: Path, family_dir: str, targets):
    folder = root / family_dir
    if not folder.exists():
        print(f"[WARN] {folder} not found")
        return

    rounds = load_round_files(folder)
    if not rounds:
        print(f"[WARN] no rounds for {family_dir}")
        return

    report = {}
    for t in targets:
        recs = []
        specs = []
        nearby_counts = []
        round_nums = []
        for rnum, data in rounds:
            thr_list = data.get("metrics", {}).get("all_thresholds", [])
            row = find_threshold_row(thr_list, t)
            round_nums.append(rnum)
            if row is None:
                recs.append(np.nan)
                specs.append(np.nan)
            else:
                recs.append(float(row.get("recall", np.nan)))
                specs.append(float(row.get("spec", np.nan)))

            # estimate nearby sample count from risk_distribution (counts around thr)
            rd = data.get("metrics", {}).get("risk_distribution", [])
            # find bin that contains threshold
            count_near = 0
            for b in rd:
                low = float(b.get("bin_edge_lower", 0.0))
                high = float(b.get("bin_edge_upper", 1.0))
                if low <= t <= high:
                    count_near = int(b.get("count_y0", 0)) + int(b.get("count_y1", 0))
                    break
            nearby_counts.append(count_near)

        recs = np.array(recs, dtype=float)
        specs = np.array(specs, dtype=float)
        counts = np.array(nearby_counts, dtype=int)

        # metrics
        def stats(arr):
            valid = np.isfinite(arr)
            if not valid.any():
                return {"mean": np.nan, "std": np.nan, "max_jump": np.nan}
            a = arr[valid]
            diffs = np.abs(np.diff(a))
            return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)), "max_jump": float(np.nanmax(diffs) if diffs.size>0 else 0.0)}

        r_stats = stats(recs)
        s_stats = stats(specs)

        # neighbor behavior: check nearest available thresholds averaged across rounds
        # we'll compute for +/- one step from available threshold list in first round
        first_thr_list = rounds[0][1].get("metrics", {}).get("all_thresholds", [])
        thr_vals = [float(x.get("threshold", np.nan)) for x in first_thr_list]
        thr_vals = sorted(thr_vals)
        # find neighbors
        neigh_lower = None
        neigh_upper = None
        if thr_vals:
            diffs = [abs(v - t) for v in thr_vals]
            idx = int(np.argmin(diffs))
            if idx-1 >= 0:
                neigh_lower = thr_vals[idx-1]
            if idx+1 < len(thr_vals):
                neigh_upper = thr_vals[idx+1]

        def neighbor_series(neigh):
            if neigh is None:
                return None
            arr = []
            for rnum, data in rounds:
                row = find_threshold_row(data.get("metrics", {}).get("all_thresholds", []), neigh)
                if row is None:
                    arr.append(np.nan)
                else:
                    arr.append(float(row.get("recall", np.nan)))
            return np.array(arr, dtype=float)

        lower_series = neighbor_series(neigh_lower)
        upper_series = neighbor_series(neigh_upper)

        def max_jump_of(arr):
            valid = np.isfinite(arr)
            if valid.sum() < 2:
                return 0.0
            return float(np.nanmax(np.abs(np.diff(arr[valid]))))

        lower_jump = max_jump_of(lower_series) if lower_series is not None else 0.0
        upper_jump = max_jump_of(upper_series) if upper_series is not None else 0.0

        max_neigh = max(lower_jump, upper_jump, 1e-9)

        classification = "uncertain"
        # heuristics
        if r_stats["max_jump"] > 2.0 * max_neigh and np.nanmedian(counts) < 50:
            classification = "threshold-noise (few samples near threshold)"
        elif r_stats["max_jump"] > 0.15 and (lower_jump > 0.12 or upper_jump > 0.12):
            classification = "likely-model-instability (neighbors also jump)"
        else:
            classification = "threshold-sensitive or stable (mixed)"

        report[t] = {
            "rounds": round_nums,
            "recall": recs.tolist(),
            "spec": specs.tolist(),
            "nearby_counts": nearby_counts,
            "rec_stats": r_stats,
            "spec_stats": s_stats,
            "neighbor_lower": neigh_lower,
            "neighbor_upper": neigh_upper,
            "neighbor_jumps": {"lower": lower_jump, "upper": upper_jump},
            "classification": classification,
        }

        # create plot for this threshold
        outdir = root / f"plots_analysis" / family_dir
        plot_threshold_series(round_nums, recs, specs, nearby_counts, neigh_lower, neigh_upper, lower_series, upper_series, family_dir, t, outdir, classification)

    # print summary
    print(f"\n=== Analysis for {family_dir} ===")
    for t, info in report.items():
        print(f"\nThreshold {t}: {info['classification']}")
        print(f"  Recall mean={info['rec_stats']['mean']:.3f}, std={info['rec_stats']['std']:.3f}, max_round_jump={info['rec_stats']['max_jump']:.3f}")
        print(f"  Spec  mean={info['spec_stats']['mean']:.3f}, std={info['spec_stats']['std']:.3f}, max_round_jump={info['spec_stats']['max_jump']:.3f}")
        print(f"  Neighbor thresholds: lower={info['neighbor_lower']}, upper={info['neighbor_upper']}")
        print(f"  Neighbor max jumps: lower={info['neighbor_jumps']['lower']:.3f}, upper={info['neighbor_jumps']['upper']:.3f}")
        med_count = int(np.nanmedian(info['nearby_counts'])) if info['nearby_counts'] else 0
        print(f"  Median samples in risk-bin around thr: {med_count}")
        # show large jumps
        rec = np.array(info['recall'], dtype=float)
        rounds = info['rounds']
        for i in range(1, len(rec)):
            if not np.isfinite(rec[i-1]) or not np.isfinite(rec[i]):
                continue
            if abs(rec[i] - rec[i-1]) > 0.08:
                print(f"    Big recall jump at round {rounds[i-1]}->{rounds[i]}: {rec[i-1]:.3f}->{rec[i]:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="result/splits_iid_scaling/splits_iid_16384_clients.json", help="root result folder")
    args = parser.parse_args()

    root = Path(args.root)
    # mapping: folder names under root
    analyze_family(root, "all_rounds_FedAdam_3", [0.25, 0.55])
    analyze_family(root, "all_rounds_FedAdam_1", [0.5, 0.65])


if __name__ == "__main__":
    main()
