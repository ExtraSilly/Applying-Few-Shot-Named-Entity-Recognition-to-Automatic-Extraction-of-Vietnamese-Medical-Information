"""
Chay few-shot experiments voi nhieu seeds va tong hop ket qua mean +- std

Usage:
    python run_multiseed.py                    # chay tat ca modes va seeds
    python run_multiseed.py --modes k5 k10     # chi chay k5 va k10
    python run_multiseed.py --seeds 1 42       # chi chay 2 seeds
    python run_multiseed.py --summarize-only   # chi tong hop, khong train
"""

import argparse
import json
import subprocess
import sys
import numpy as np
from pathlib import Path
from collections import Counter

BASE     = Path(__file__).parent
PYTHON   = sys.executable           # dung chinh Python dang chay script nay
ENTITIES = [
    "AGE", "DATE", "GENDER", "JOB", "LOCATION", "NAME",
    "ORGANIZATION", "PATIENT_ID", "SYMPTOM_AND_DISEASE", "TRANSPORTATION",
]


def run_experiment(mode, seed):
    result_path = BASE / f"output/baseline_{mode}_seed{seed}/results.json"
    if result_path.exists():
        print(f"  [SKIP] {mode} seed={seed} — ket qua da ton tai")
        return
    print(f"  [RUN]  {mode} seed={seed} ...")
    subprocess.run(
        [PYTHON, "train_baseline.py", "--mode", mode, "--seed", str(seed)],
        cwd=BASE, check=True,
    )


def load_result(mode, seed):
    path = BASE / f"output/baseline_{mode}_seed{seed}/results.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_full_f1():
    """Doc Micro F1 cua full training tu file ket qua thuc te."""
    for seed in [42, 1, 100]:
        path = BASE / f"output/baseline_full_seed{seed}/results.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
            return r["report"]["micro avg"]["f1-score"] * 100, seed
    return None, None


def aggregate(mode, seeds):
    records = [load_result(mode, s) for s in seeds]
    records = [r for r in records if r is not None]
    if not records:
        return None

    micro_f1s = [r["report"]["micro avg"]["f1-score"] * 100 for r in records]
    per_entity = {}
    for e in ENTITIES:
        vals = [r["report"].get(e, {}).get("f1-score", 0.0) * 100 for r in records]
        per_entity[e] = {
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals)),
            "values": vals,
        }

    return {
        "mode":   mode,
        "seeds":  seeds,
        "n":      len(records),
        "micro_f1": {
            "mean":   float(np.mean(micro_f1s)),
            "std":    float(np.std(micro_f1s)),
            "values": micro_f1s,
        },
        "per_entity": per_entity,
    }


def print_summary(results, modes, full_f1=None, full_seed=None):
    seeds = next(iter(results.values()))["seeds"] if results else []
    print(f"\n{'='*65}")
    print(f"FEW-SHOT NER RESULTS — seeds={seeds}")
    print(f"{'='*65}")

    mode_labels = {"k5": "5-shot", "k10": "10-shot", "k20": "20-shot", "k1": "1-shot"}
    print(f"\n{'Setting':<12} {'Train':<8} {'Micro F1':>12}  {'Std':>8}  {'vs Full':>10}")
    print("-" * 55)

    train_sizes = {"k1": 10, "k5": 50, "k10": 100, "k20": 200}
    for mode in modes:
        r = results.get(mode)
        if r:
            m, s = r["micro_f1"]["mean"], r["micro_f1"]["std"]
            diff = f"{m - full_f1:+.2f}pp" if full_f1 else "—"
            label = mode_labels.get(mode, mode)
            print(f"{label:<12} {train_sizes.get(mode, '?'):<8} {m:>10.2f}%  {s:>6.2f}%  {diff:>10}")

    if full_f1:
        print(f"{'Full':<12} {'5126':<8} {full_f1:>10.2f}%  {'—':>6}  {'—':>10}  (seed={full_seed})")

    print(f"\n{'Entity':<25}", end="")
    for mode in modes:
        print(f"  {mode_labels.get(mode, mode):>14}", end="")
    print()
    print("-" * (25 + 16 * len(modes)))

    for e in ENTITIES:
        print(f"{e:<25}", end="")
        for mode in modes:
            r = results.get(mode)
            if r:
                m = r["per_entity"][e]["mean"]
                s = r["per_entity"][e]["std"]
                print(f"  {m:>5.1f}+-{s:<4.1f}", end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes",  nargs="+", default=["k5", "k10", "k20"],
                        choices=["k1", "k5", "k10", "k20"])
    parser.add_argument("--seeds",  nargs="+", type=int, default=[1, 42, 100])
    parser.add_argument("--summarize-only", action="store_true",
                        help="Chi tong hop ket qua, khong chay training moi")
    args = parser.parse_args()

    # 1. Chay experiments
    if not args.summarize_only:
        print(f"Chay experiments: modes={args.modes}, seeds={args.seeds}")
        for mode in args.modes:
            for seed in args.seeds:
                run_experiment(mode, seed)

    # 2. Tong hop ket qua
    print("\nTong hop ket qua...")
    results = {}
    for mode in args.modes:
        agg = aggregate(mode, args.seeds)
        if agg:
            results[mode] = agg
        else:
            print(f"  [WARN] Khong co ket qua cho mode={mode}")

    # 3. Doc full F1 tu file thuc te
    full_f1, full_seed = load_full_f1()
    if full_f1 is None:
        print("  [WARN] Chua co ket qua full training. Chay: python train_baseline.py --mode full")

    # 4. In ket qua
    print_summary(results, args.modes, full_f1, full_seed)

    # 5. Luu
    if not results:
        print("Khong co ket qua de luu.")
        return

    out = {}
    for mode, agg in results.items():
        out[mode] = {
            "micro_f1_mean": round(agg["micro_f1"]["mean"], 4),
            "micro_f1_std":  round(agg["micro_f1"]["std"],  4),
            "seeds":         agg["seeds"],
            "per_entity": {
                e: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)}
                for e, v in agg["per_entity"].items()
            },
        }

    save_path = BASE / "output/multiseed_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nLuu tai: {save_path}")


if __name__ == "__main__":
    main()
