#!/usr/bin/env python3
"""Build the static Countdown JSONL files used by DTM.

Training data is generated from the Hugging Face dataset
Jiayi-Pan/Countdown-Tasks-3to4 by keeping only examples with exactly three
numbers. The eval file is a fixed 256-example held-out snapshot bundled with
this repo; pass --test-source to normalize a replacement held-out JSONL.
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, load_dataset

COUNTDOWN_HF_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"


def _parse_record(rec, require_three=True):
    if "nums" in rec and "target" in rec:
        numbers = [int(n) for n in rec["nums"]]
        target = int(rec["target"])
    elif "input" in rec and "output" in rec:
        numbers = [int(n.strip()) for n in str(rec["input"]).split(",") if n.strip()]
        target = int(rec["output"])
    else:
        raise KeyError(f"Unsupported Countdown record keys: {sorted(rec.keys())}")
    if require_three and len(numbers) != 3:
        raise ValueError(f"Expected 3 numbers, got {numbers!r}")
    return numbers, target


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ": ")) + "\n")
    os.replace(tmp_path, path)


def _build_train_rows(dataset_name, split, arrow_file=None):
    if arrow_file:
        ds = Dataset.from_file(arrow_file)
    else:
        ds = load_dataset(dataset_name, split=split)
    rows = []
    for rec in ds:
        numbers, target = _parse_record(rec, require_three=False)
        if len(numbers) == 3:
            rows.append({"nums": numbers, "target": target})
    return rows


def _build_test_rows(test_source):
    rows = []
    with open(test_source) as f:
        for line in f:
            if not line.strip():
                continue
            numbers, target = _parse_record(json.loads(line))
            rows.append({"input": ",".join(str(n) for n in numbers), "output": str(target)})
    return rows


def _check_count(name, rows, expected):
    if expected >= 0 and len(rows) != expected:
        raise ValueError(f"{name} row count mismatch: expected {expected}, got {len(rows)}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=COUNTDOWN_HF_DATASET, help="HF dataset used for the training snapshot.")
    p.add_argument("--split", default="train", help="HF split used for the training snapshot.")
    p.add_argument(
        "--arrow-file",
        default=None,
        help="Optional cached Arrow file to read instead of calling datasets.load_dataset.",
    )
    p.add_argument("--output-dir", default="data", help="Directory to write countdown_cd3_{train,test}.jsonl.")
    p.add_argument(
        "--test-source",
        default="data/countdown_cd3_test.jsonl",
        help="Held-out eval JSONL to normalize. The HF dataset only provides the training source used here.",
    )
    p.add_argument("--expected-train-size", type=int, default=240632)
    p.add_argument("--expected-test-size", type=int, default=256)
    p.add_argument("--skip-train", action="store_true", help="Only normalize/validate the held-out eval JSONL.")
    p.add_argument("--skip-test", action="store_true", help="Only build the training JSONL.")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    if not args.skip_train:
        train_rows = _build_train_rows(args.dataset, args.split, arrow_file=args.arrow_file)
        _check_count("train", train_rows, args.expected_train_size)
        train_path = output_dir / "countdown_cd3_train.jsonl"
        _write_jsonl(train_path, train_rows)
        print(f"wrote {len(train_rows)} rows to {train_path}")

    if not args.skip_test:
        test_rows = _build_test_rows(args.test_source)
        _check_count("test", test_rows, args.expected_test_size)
        test_path = output_dir / "countdown_cd3_test.jsonl"
        _write_jsonl(test_path, test_rows)
        print(f"wrote {len(test_rows)} rows to {test_path}")


if __name__ == "__main__":
    main()
