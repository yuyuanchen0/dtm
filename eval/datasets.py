"""Evaluation datasets for Countdown and Sudoku."""

import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset

from dtm.data_utils import (
    SUDOKU_FEW_SHOT_EXAMPLES,
    SUDOKU_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    _countdown_user_prompt,
    _parse_countdown_record,
)


class _BaseEvalDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, system_prompt, add_reasoning, subsample):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.add_reasoning = add_reasoning
        self._load()
        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )

    def __len__(self):
        return len(self.subsample)

    def _load(self):
        raise NotImplementedError

    def _wrap_prompt(self, content):
        messages = [{"role": "user", "content": content}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.add_reasoning:
            prompt += "<reasoning>"
        return prompt

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts}


class CountdownDataset(_BaseEvalDataset):
    def __init__(self, tokenizer, num_examples=0, add_reasoning=True, subsample=256, **_):
        super().__init__(tokenizer, system_prompt=SYSTEM_PROMPT, add_reasoning=add_reasoning, subsample=subsample)

    def _load(self):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(cur_path, "..", "data", "countdown_cd3_test.jsonl"))
        with open(path) as f:
            self.dataset = [json.loads(line) for line in f]

    def __getitem__(self, idx):
        row = self.dataset[self.subsample[idx].item()]
        numbers, target = _parse_countdown_record(row)
        question = f"Numbers: {numbers}\nTarget: {target}"
        prompt = self._wrap_prompt(_countdown_user_prompt(numbers, target))
        return prompt, question, (numbers, target)


class SudokuDataset(_BaseEvalDataset):
    def __init__(self, tokenizer, num_examples=3, add_reasoning=True, subsample=256, **_):
        self.num_examples = max(0, min(num_examples, len(SUDOKU_FEW_SHOT_EXAMPLES)))
        few_shot_block = "\n\n".join(SUDOKU_FEW_SHOT_EXAMPLES[: self.num_examples])
        system_prompt = (
            f"{SUDOKU_SYSTEM_PROMPT}\n\n{few_shot_block}" if few_shot_block else SUDOKU_SYSTEM_PROMPT
        )
        super().__init__(tokenizer, system_prompt=system_prompt, add_reasoning=add_reasoning, subsample=subsample)

    def _load(self):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(cur_path, "..", "data", "test_sudoku_split_new.csv"))
        df = pd.read_csv(path, dtype={"Puzzle": str, "Solution": str})
        self.dataset = HFDataset.from_pandas(df)

    def __getitem__(self, idx):
        row = self.dataset[self.subsample[idx].item()]
        puzzle = row["Puzzle"]
        solution = row["Solution"]
        question = f"Solve the following Sudoku puzzle: {puzzle}\n"
        content = f"{self.system_prompt}\n\nQuestion: {question}Answer:\n"
        prompt = self._wrap_prompt(content)
        return prompt, question, solution
