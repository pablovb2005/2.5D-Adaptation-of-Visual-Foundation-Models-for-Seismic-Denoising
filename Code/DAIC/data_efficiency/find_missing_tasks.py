#!/usr/bin/env python3
"""Audit and print missing channel-window data-efficiency array tasks.

This helper is intended to run on DAIC. It checks the staff-bulk run folders for
the 5-variant channel-window study and reports the array IDs that still need a
resubmission for each matrix:

* large: train sizes 20,35,50,75,100, array ids 0-224
* small: train sizes 5,10,15, array ids 0-134
"""

from __future__ import print_function

import argparse
import csv
import os
from collections import Counter, defaultdict


DEFAULT_STUDY = (
    "/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal/"
    "experiments/runs/data_efficiency_100train_channel_window_v2"
)

DATA_SEEDS = [101, 202, 303]
TRAINING_SEEDS = [42, 43, 44]
VARIANTS = ["2d", "3ch", "5ch", "7ch", "9ch"]
RUN_SUFFIX = "100train_channel_window_v2"
TARGET_EPOCHS = 50

RUN_BASES = {
    "2d": "impeccable_repeated_stride5_lora_r16",
    "3ch": "impeccable_neighbors3_stride5_lora_r16",
    "5ch": "impeccable_neighbors5_stride5_patch_emb_lora_r16",
    "7ch": "impeccable_neighbors7_stride5_patch_emb_lora_r16",
    "9ch": "impeccable_neighbors9_stride5_patch_emb_lora_r16",
}

RUN_ID_BY_SEED = {
    42: "seed42_run01",
    43: "seed43_run02",
    44: "seed44_run03",
}

MATRICES = {
    "large": [20, 35, 50, 75, 100],
    "small": [5, 10, 15],
}


def task_id(ds_idx, ts_idx, variant_idx, size_idx, train_sizes):
    return (
        ds_idx * (len(TRAINING_SEEDS) * len(VARIANTS) * len(train_sizes))
        + ts_idx * (len(VARIANTS) * len(train_sizes))
        + variant_idx * len(train_sizes)
        + size_idx
    )


def run_path(study_root, variant, train_n, data_seed, training_seed):
    run_base = "%s_n%svols_%s" % (RUN_BASES[variant], train_n, RUN_SUFFIX)
    return os.path.join(
        study_root,
        variant,
        run_base,
        "data_seed%s" % data_seed,
        RUN_ID_BY_SEED[training_seed],
    )


def last_epoch(history_path):
    if not os.path.exists(history_path):
        return None
    try:
        with open(history_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            epochs = []
            for row in reader:
                raw = row.get("epoch")
                if raw is None:
                    continue
                try:
                    epochs.append(int(float(raw)))
                except ValueError:
                    pass
            return max(epochs) if epochs else None
    except Exception:
        return None


def nonempty_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def classify(path):
    history_path = os.path.join(path, "history.csv")
    eval_path = os.path.join(path, "eval_results", "results.csv")
    best_path = os.path.join(path, "best.pt")
    last_path = os.path.join(path, "last.pt")

    epoch = last_epoch(history_path)
    has_history = epoch is not None
    has_eval = nonempty_file(eval_path)
    has_best = nonempty_file(best_path)
    has_last = nonempty_file(last_path)

    if has_history and epoch >= TARGET_EPOCHS and has_eval:
        status = "done"
    elif has_history and epoch >= TARGET_EPOCHS:
        status = "train_complete_eval_missing"
    elif has_history:
        status = "partial"
    else:
        status = "missing"

    return {
        "status": status,
        "epoch": epoch,
        "has_eval": has_eval,
        "has_best": has_best,
        "has_last": has_last,
    }


def compress_ids(ids):
    ids = sorted(set(ids))
    if not ids:
        return ""
    ranges = []
    start = prev = ids[0]
    for value in ids[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(str(start))
        else:
            parts.append("%s-%s" % (start, end))
    return ",".join(parts)


def audit_matrix(study_root, matrix_name, train_sizes):
    rows = []
    counts = Counter()
    by_variant_size = defaultdict(Counter)

    for ds_idx, data_seed in enumerate(DATA_SEEDS):
        for ts_idx, training_seed in enumerate(TRAINING_SEEDS):
            for variant_idx, variant in enumerate(VARIANTS):
                for size_idx, train_n in enumerate(train_sizes):
                    tid = task_id(ds_idx, ts_idx, variant_idx, size_idx, train_sizes)
                    path = run_path(study_root, variant, train_n, data_seed, training_seed)
                    info = classify(path)
                    info.update(
                        {
                            "task_id": tid,
                            "variant": variant,
                            "train_n": train_n,
                            "data_seed": data_seed,
                            "training_seed": training_seed,
                            "path": path,
                        }
                    )
                    rows.append(info)
                    counts[info["status"]] += 1
                    by_variant_size[(variant, train_n)][info["status"]] += 1

    needs_resubmit = [
        row["task_id"] for row in rows if row["status"] != "done"
    ]
    needs_eval_only = [
        row["task_id"] for row in rows if row["status"] == "train_complete_eval_missing"
    ]
    needs_training = [
        row["task_id"]
        for row in rows
        if row["status"] in ("missing", "partial")
    ]

    print("=== %s matrix ===" % matrix_name)
    print("train sizes: %s" % ",".join(str(x) for x in train_sizes))
    print("total expected: %s" % len(rows))
    for key in ["done", "train_complete_eval_missing", "partial", "missing"]:
        print("%s: %s" % (key, counts.get(key, 0)))
    print("needs resubmit count: %s" % len(needs_resubmit))
    print("needs resubmit array: %s" % (compress_ids(needs_resubmit) or "(none)"))
    print("needs training array: %s" % (compress_ids(needs_training) or "(none)"))
    print("eval-only candidates: %s" % (compress_ids(needs_eval_only) or "(none)"))
    print("")

    print("By variant/n, incomplete counts:")
    any_incomplete = False
    for variant in VARIANTS:
        for train_n in train_sizes:
            counter = by_variant_size[(variant, train_n)]
            incomplete = sum(
                counter.get(status, 0)
                for status in ("train_complete_eval_missing", "partial", "missing")
            )
            if incomplete:
                any_incomplete = True
                print(
                    "  %s n=%s: %s"
                    % (
                        variant,
                        train_n,
                        ", ".join(
                            "%s=%s" % (status, counter.get(status, 0))
                            for status in ("train_complete_eval_missing", "partial", "missing")
                            if counter.get(status, 0)
                        ),
                    )
                )
    if not any_incomplete:
        print("  (none)")
    print("")

    print("Incomplete rows:")
    for row in rows:
        if row["status"] == "done":
            continue
        print(
            "  task={task_id:03d} status={status} variant={variant} n={train_n} "
            "data_seed={data_seed} training_seed={training_seed} epoch={epoch} "
            "best={has_best} last={has_last} eval={has_eval}".format(**row)
        )
    print("")

    return {
        "rows": rows,
        "needs_resubmit": needs_resubmit,
        "needs_training": needs_training,
        "needs_eval_only": needs_eval_only,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", default=DEFAULT_STUDY)
    parser.add_argument(
        "--matrix",
        choices=["all", "large", "small"],
        default="all",
        help="Which task matrix to audit.",
    )
    args = parser.parse_args()

    matrix_names = ["large", "small"] if args.matrix == "all" else [args.matrix]
    summaries = {}
    for matrix_name in matrix_names:
        summaries[matrix_name] = audit_matrix(
            args.study_root,
            matrix_name,
            MATRICES[matrix_name],
        )

    print("=== Resubmit commands ===")
    script = "$HOME/RP/Code/DAIC/data_efficiency/submit_100train_data_efficiency.sh"
    for matrix_name in matrix_names:
        ids = summaries[matrix_name]["needs_resubmit"]
        if not ids:
            continue
        sizes = ",".join(str(x) for x in MATRICES[matrix_name])
        array_spec = compress_ids(ids)
        print(
            "DATA_SEEDS_CSV=101,202,303 TRAINING_SEEDS_CSV=42,43,44 "
            "TRAIN_SIZES_CSV=%s VARIANTS_CSV=2d,3ch,5ch,7ch,9ch "
            "STUDY_NAME=data_efficiency_100train_channel_window_v2 "
            "RUN_SUFFIX=100train_channel_window_v2 "
            "sbatch --export=ALL --array=%s %s"
            % (sizes, array_spec, script)
        )


if __name__ == "__main__":
    main()
