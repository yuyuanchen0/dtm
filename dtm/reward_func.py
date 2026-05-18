"""Reward functions for Countdown and Sudoku tasks."""

import re


# ---------------- Countdown ----------------

def _extract_answer(solution_str: str):
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def _validate_countdown_equation(equation_str: str, available_numbers) -> bool:
    """Check that the equation uses each available number exactly once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except Exception:
        return False


def _evaluate_countdown_equation(equation_str: str):
    """Safely evaluate the arithmetic expression. Returns None on failure."""
    try:
        if not re.match(r"^[\d+\-*/().\s]+$", equation_str):
            return None
        return eval(equation_str, {"__builtins__": None}, {})
    except Exception:
        return None


def _countdown_score(solution_str, target, numbers, format_score: float = 0.1, score: float = 1.0) -> float:
    """1.0 for correct equation, 0.1 for valid format, 0.0 otherwise."""
    equation = _extract_answer(solution_str)
    if equation is None:
        return 0.0
    if not _validate_countdown_equation(equation, numbers):
        return format_score
    result = _evaluate_countdown_equation(equation)
    if result is None:
        return format_score
    if abs(result - target) < 1e-5:
        return score
    return format_score


def countdown_reward_func(prompts, completions, **kwargs):
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [c[0]["content"] for c in completions]
    else:
        responses = completions
    targets = kwargs["target"]
    numbers = kwargs["numbers"]
    return [_countdown_score(r, targets[i], numbers[i]) for i, r in enumerate(responses)]


# ---------------- Sudoku ----------------

def _extract_sudoku_answer(solution_str: str):
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if not matches:
        return None
    return "".join(c for c in matches[-1].strip() if c.isdigit())


def _validate_sudoku_solution(solution_str, ground_truth, puzzle) -> float:
    """Returns a reward computed as 2.0 × (correct cells / empty cells)."""
    if not solution_str:
        return 0.0
    if len(solution_str) < 16:
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        solution_str = solution_str[:16]
    empty_indices = [i for i in range(16) if puzzle[i] == "0"]
    if not empty_indices:
        return 0.0
    correct = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
    return 2.0 * correct / len(empty_indices)


def sudoku_reward_func(prompts, completions, **kwargs):
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [c[0]["content"] for c in completions]
    else:
        responses = completions
    puzzles = kwargs["puzzle"]
    solutions = kwargs["solution"]
    scores = []
    for i, r in enumerate(responses):
        extracted = _extract_sudoku_answer(r)
        if extracted is None:
            scores.append(0.0)
        else:
            scores.append(_validate_sudoku_solution(extracted, solutions[i], puzzles[i]))
    return scores
