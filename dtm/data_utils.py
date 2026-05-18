"""Dataset loading and prompts for Countdown and Sudoku tasks."""

import json
import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

COUNTDOWN_HF_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""

SUDOKU_FEW_SHOT_EXAMPLES = [
    "Question:\nSolve the following Sudoku puzzle: 3014002020004130\nAnswer:\n<reasoning>\nInterpret puzzle as 4 rows of 4:\nR1: 3 0 1 4\nR2: 0 0 2 0\nR3: 2 0 0 0\nR4: 4 1 3 0\n\nFill easy singles:\nR1 missing 2 → R1C2=2.\nR4 missing 2 → R4C4=2.\nBox D (R3-4,C3-4) then needs {1,4}; column4 can only accept 1 → R3C4=1, R3C3=4.\nR3 now missing 3 → R3C2=3.\nColumn1 missing 1 → R2C1=1.\nColumn2 missing 4 → R2C2=4.\nLast cell R2C4=3.\n\nFinal grid:\nR1: 3 2 1 4\nR2: 1 4 2 3\nR3: 2 3 4 1\nR4: 4 1 3 2\n</reasoning>\n<answer>\n3214142323414132\n</answer>",
    "Question:\nSolve the following Sudoku puzzle: 0000100420013142\nAnswer:\n<reasoning>\nInterpret puzzle as 4 rows of 4:\nR1: 0 0 0 0\nR2: 1 0 0 4\nR3: 2 0 0 1\nR4: 3 1 4 2\n\nFill easy singles:\nCol1 missing 4 → R1C1=4.\nCol4 missing 3 → R1C4=3.\nBox A (R1-2,C1-2) missing {2,3} and R1 now needs {1,2} → R1C2=2, R2C2=3.\nR1C3=1.\nR2 now missing 2 → R2C3=2.\nCol2 missing 4 → R3C2=4, then R3C3=3.\n\nFinal grid:\nR1: 4 2 1 3\nR2: 1 3 2 4\nR3: 2 4 3 1\nR4: 3 1 4 2\n</reasoning>\n<answer>\n4213132424313142\n</answer>",
    "Question:\nSolve the following Sudoku puzzle: 2001403002001420\nAnswer:\n<reasoning>\nInterpret puzzle as 4 rows of 4:\nR1: 2 0 0 1\nR2: 4 0 3 0\nR3: 0 2 0 0\nR4: 1 4 2 0\n\nFill easy singles:\nR1 missing {3,4}; Col2 can't be 1 so R1C2=3 → R1C3=4.\nR4 missing 3 → R4C4=3.\nCol4 missing {2,4}; R2 must take 2 → R2C4=2 → R2C2=1.\nCol1 missing 3 → R3C1=3.\nCol3 missing 1 → R3C3=1 → R3C4=4.\n\nFinal grid:\nR1: 2 3 4 1\nR2: 4 1 3 2\nR3: 3 2 1 4\nR4: 1 4 2 3\n</reasoning>\n<answer>\n2341413232141423\n</answer>",
]


def _parse_countdown_record(rec):
    """Return (numbers, target) for either bundled Countdown JSONL schema."""
    if "nums" in rec and "target" in rec:
        numbers = [int(n) for n in rec["nums"]]
        target = int(rec["target"])
    elif "input" in rec and "output" in rec:
        numbers = [int(n.strip()) for n in str(rec["input"]).split(",") if n.strip()]
        target = int(rec["output"])
    else:
        raise KeyError("Countdown records must have either nums/target or input/output fields")
    if len(numbers) != 3:
        raise ValueError(f"Expected 3 Countdown numbers, got {numbers!r}")
    return numbers, target


def _countdown_user_prompt(numbers, target):
    return (
        f"{SYSTEM_PROMPT}\nUsing only the numbers {numbers}, create an arithmetic expression "
        f"that evaluates to exactly {target}. You must use all numbers from the list, and each "
        f"number must be used exactly once. You may use the operations +, -, *, and / as needed. "
        f"After reasoning, provide only your final expression inside <answer></answer> tags "
        f"without including an equals sign or the target number. For example, if the numbers are "
        f"[2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>"
    )


def get_countdown_questions(split: str = "train") -> Dataset:
    """Load Countdown training prompts from a static local jsonl.

    The original HF dataset (Jiayi-Pan/Countdown-Tasks-3to4) is filtered to 3-number
    targets and snapshotted in ../data/countdown_cd3_train.jsonl. We read that file
    directly to avoid the HF datasets cache + NFS flock contention across DDP ranks.
    """
    if split != "train":
        raise ValueError("Only the train split is bundled; use eval/datasets.py for the test split.")
    cur_path = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(os.path.join(cur_path, "..", "data", "countdown_cd3_train.jsonl"))
    rows = []
    with open(src, "r") as f:
        for line in f:
            rec = json.loads(line)
            numbers, target = _parse_countdown_record(rec)
            rows.append({
                "prompt": [{"role": "user", "content": _countdown_user_prompt(numbers, target)}],
                "target": target,
                "numbers": numbers,
            })
    return Dataset.from_list(rows)


def get_sudoku_questions(few_shot: int = 0) -> Dataset:
    if few_shot > len(SUDOKU_FEW_SHOT_EXAMPLES):
        raise ValueError(f"few_shot must be <= {len(SUDOKU_FEW_SHOT_EXAMPLES)}")
    cur_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file = os.path.normpath(os.path.join(cur_path, "..", "data", "train_sudoku_split_new.csv"))
    df = pd.read_csv(sudoku_file, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    few_shot_block = "\n\n".join(SUDOKU_FEW_SHOT_EXAMPLES[:few_shot])
    system_prompt = f"{SUDOKU_SYSTEM_PROMPT}\n\n{few_shot_block}" if few_shot_block else SUDOKU_SYSTEM_PROMPT

    return data.map(
        lambda x: {
            "prompt": [{
                "role": "user",
                "content": f"{system_prompt}\n\nQuestion: Solve the following Sudoku puzzle: {x['Puzzle']}\nAnswer:\n",
            }],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )
