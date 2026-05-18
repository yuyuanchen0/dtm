"""Aggregate accuracy from generation JSONs produced by eval/eval.py.

Run from the directory holding `*_generations.json`:
    python -m eval.parser_json
"""

import glob
import json
import os
import re
from collections import defaultdict


def _count_effective_tokens(text):
    if not text:
        return 0
    text = text.replace("<|endoftext|>", "")
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text.split())


def parse_countdown_answers(json_path):
    with open(json_path) as f:
        data = json.load(f)

    correct = processed = total_tokens = 0
    items = []

    def _validate(equation_str, available_numbers):
        try:
            in_eq = sorted(int(n) for n in re.findall(r"\d+", equation_str))
            return in_eq == sorted(available_numbers)
        except Exception:
            return False

    def _safe_eval(equation_str):
        try:
            if not re.match(r"^[\d+\-*/().\s]+$", equation_str):
                return float("inf")
            return eval(equation_str.strip(), {"__builtins__": None}, {})
        except Exception:
            return float("inf")

    for item in data.get("generations", []):
        processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", [])
        gen = item.get("generations", "")
        total_tokens += _count_effective_tokens(gen)

        if isinstance(ground_truth, list) and len(ground_truth) == 2:
            numbers, target = ground_truth[0], ground_truth[1]
        else:
            numbers, target = [], None
            m = re.search(r"Numbers: \[([\d, ]+)\]", question)
            if m:
                numbers = [int(n.strip()) for n in m.group(1).split(",")]
            m = re.search(r"Target: (\d+)", question)
            if m:
                target = int(m.group(1))

        equation = ""
        m = re.search(r"<answer>(.*?)</answer>", gen, re.DOTALL)
        if m:
            equation = m.group(1).strip()
        else:
            equation = gen
        equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")
        m = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
        if m:
            equation = m.group(1).strip()

        is_correct = False
        result = None
        if _validate(equation, numbers):
            result = _safe_eval(equation)
            if target is not None and abs(result - target) < 1e-5:
                is_correct = True
                correct += 1
        items.append({
            "question": question, "extracted_answer": equation,
            "evaluation_result": result, "ground_truth": ground_truth,
            "is_correct": is_correct,
        })
    return correct, processed, items, total_tokens


def parse_sudoku_answers(json_path):
    with open(json_path) as f:
        data = json.load(f)

    total_correct_cells = total_empty_cells = total_processed = total_tokens = 0
    items = []
    patterns = [
        r"<answer>.*?```\s*([\d\s]+)```",
        r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
        r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
        r".*?(\d{16})\s*</answer>",
        r"\b(\d{16})\b",
    ]

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        gen = item.get("generations", "")
        total_tokens += _count_effective_tokens(gen)

        puzzle = ""
        if len(question) >= 16 and all(c.isdigit() or c == "0" for c in question[:16]):
            puzzle = question[:16]
        else:
            m = re.search(r"Sudoku puzzle: ([0-9]{16})", question)
            if m:
                puzzle = m.group(1)
        if len(puzzle) != 16:
            continue
        empty = [i for i in range(16) if puzzle[i] == "0"]

        solution = ""
        for pat in patterns:
            if solution:
                break
            m = re.search(pat, gen, re.DOTALL)
            if m and m.group(1).strip():
                solution = re.sub(r"\s", "", m.group(1).strip())
        if solution:
            if len(solution) < 16:
                solution = solution + "0" * (16 - len(solution))
            elif len(solution) > 16:
                solution = solution[:16]
            correct = sum(1 for i in empty if solution[i] == ground_truth[i])
        else:
            correct = 0

        total_correct_cells += correct
        total_empty_cells += len(empty)
        items.append({
            "question": question, "extracted_answer": solution,
            "ground_truth": ground_truth,
            "empty_cells": len(empty), "correct_cells": correct,
            "accuracy": correct / len(empty) if empty else 0.0,
        })
    return total_correct_cells, total_empty_cells, items, total_tokens


def _setup_name(filename):
    m = re.match(r"(.+)_\d+_generations\.json$", filename)
    return m.group(1) if m else None


def aggregate_results(directory=".", save_detailed=True):
    setups = defaultdict(lambda: {"correct": 0, "processed": 0, "total_tokens": 0, "questions": []})
    for json_file in glob.glob(os.path.join(directory, "*_generations.json")):
        name = _setup_name(os.path.basename(json_file))
        if not name:
            continue
        if "countdown" in name:
            correct, processed, items, tokens = parse_countdown_answers(json_file)
        elif "sudoku" in name:
            correct, processed, items, tokens = parse_sudoku_answers(json_file)
        else:
            continue
        setups[name]["correct"] += correct
        setups[name]["processed"] += processed
        setups[name]["total_tokens"] += tokens
        setups[name]["questions"].extend(items)

    print("\n===== AGGREGATED RESULTS =====")
    for setup, r in sorted(setups.items()):
        r["accuracy"] = (r["correct"] / r["processed"] * 100) if r["processed"] else 0.0
        r["avg_effective_tokens"] = r["total_tokens"] / max(1, len(r["questions"]))
        print(
            f"{setup}: {r['correct']}/{r['processed']} ({r['accuracy']:.2f}%), "
            f"avg_tokens={r['avg_effective_tokens']:.1f}"
        )
        if save_detailed:
            with open(f"{setup}_aggregated_results.json", "w") as f:
                json.dump(r, f, indent=2)


if __name__ == "__main__":
    aggregate_results()
