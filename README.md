<div align="center">

# Discrete Tilt Matching

**Likelihood-free reward fine-tuning for masked diffusion language models**

*ICML 2026*

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2604.18739-blue)](https://arxiv.org/pdf/2604.18739)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](environment.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

[Yuyuan Chen*](https://yuyuanchen0.github.io/), [Shiyi Wang*](https://www.math.harvard.edu/people/wang-shiyi-franklin/), [Peter Potaptchik](https://peterpotaptchik.github.io/), [Jaeyeon Kim](https://jaeyeonkim01.github.io/), [Michael S. Albergo](https://malbergo.me/)

*Equal contribution, alphabetical order

[Overview](#overview) · [Method](#method-highlights) · [Data](#data) · [Training](#training) · [Evaluation](#evaluation) · [Citation](#citation)

</div>

## Overview

Masked diffusion language models (dLLMs) generate by progressively unmasking tokens. This gives them flexible and parallel decoding, but it also makes standard RL-style sequence-level objectives awkward: the marginal likelihood of a completed sequence requires summing over intractably many possible unmasking orders.

**Discrete Tilt Matching (DTM)** avoids this likelihood bottleneck. Instead of optimizing a sequence likelihood surrogate, DTM views fine-tuning as **progressive reward tilting** and matches **state-level local unmasking posteriors**. The resulting *c*-DTM objective is a weighted cross-entropy loss with an explicit minimizer and a control variate for improved stability.

This repository contains the cleaned training and evaluation pipeline used for the LLaDA-8B-Instruct fine-tuning experiments on:

| Task | Setting | Released config |
| --- | --- | --- |
| Countdown | 3-number arithmetic, zero-shot | `configs/countdown.yaml`|
| 4x4 Sudoku | 3-shot structured solving | `configs/sudoku.yaml`|

## Headline Results

The paper reports strong gains for DTM on structured planning tasks when fine-tuning LLaDA-8B-Instruct with SAR decoding.

| Benchmark | Length 256 acc. (%) | Length 512 acc. (%) |
| --- | ---: | ---: |
| Sudoku | 99.2 | 99.4 |
| Countdown | 81.3 | 78.9 |

## Method Highlights

- **Likelihood-free fine-tuning:** DTM trains local unmasking posteriors directly and bypasses the intractable sequence marginal likelihood of masked diffusion models.
- **Progressive reward tilting:** training proceeds through incremental updates from `rho_{1,a}` to `rho_{1,a+h}` instead of targeting a heavily tilted distribution in one step which risks mode collapse.
- **Weighted cross-entropy with explicit minimizer:** the *c*-DTM loss is minimized by the reward-tilted local posterior and admits a control variate that preserves the minimizer while reducing gradient variance.
- **SAR-aligned training:** masked training states are aligned with semi-autoregressive block decoding, matching the inference-time state distribution more closely.
- **Replay buffer:** rollouts are cached and periodically refreshed to amortize expensive diffusion sampling.

## Repository Layout

```text
DTM/
├── configs/
│   ├── countdown.yaml
│   └── sudoku.yaml
├── data/
│   ├── countdown_cd3_train.jsonl    # Countdown train snapshot
│   ├── countdown_cd3_test.jsonl     # Countdown eval snapshot
│   ├── train_sudoku_split_new.csv
│   └── test_sudoku_split_new.csv
├── dtm/
│   ├── lightning_module.py          # DTM trainer
│   ├── train.py                     # Hydra/Lightning training entrypoint
│   ├── data_utils.py
│   ├── reward_func.py
│   └── configuration_llada.py
├── eval/
│   ├── eval.py                      # Distributed generation runner
│   ├── generate.py                  # Diffusion/SAR generation helper
│   ├── datasets.py                  # Eval dataset wrappers
│   └── parser_json.py               # Accuracy aggregation
├── scripts/
│   ├── build_countdown_snapshot.py  # Rebuild/validate Countdown JSONL snapshots
│   ├── train_countdown.sbatch
│   ├── train_sudoku.sbatch
│   └── eval.sbatch
├── environment.yml
└── requirements.txt
```

Countdown JSONL snapshots need to be [downloaded or rebuilt](#data) before training/evaluation.

## Installation

Tested with Python 3.10, PyTorch 2.6.0, CUDA 12.4, and H100 GPUs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively:

```bash
micromamba env create -f environment.yml
conda activate dtm
```

Download the base model and expose it through `LLADA_MODEL_PATH`:

```bash
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct \
  --local-dir /path/to/LLaDA-8B-Instruct

export LLADA_MODEL_PATH=/path/to/LLaDA-8B-Instruct
```

The configs use:

```yaml
base_model_path: ${oc.env:LLADA_MODEL_PATH,/path/to/LLaDA-8B-Instruct}
checkpoint_dir: ${oc.env:DTM_CHECKPOINT_DIR,runs/checkpoints/...}
```

The Slurm scripts additionally accept `MODEL_PATH`, `REPO`, `CONDA_ENV`, `OUTPUT_ROOT`, and `HF_HOME` overrides.

## Data

Countdown is bundled as static JSONL snapshots to avoid Hugging Face dataset cache locks during distributed training:

| File | Contents |
| --- | --- |
| `data/countdown_cd3_train.jsonl` | 240,632 train examples from `Jiayi-Pan/Countdown-Tasks-3to4`, filtered to 3-number instances |
| `data/countdown_cd3_test.jsonl` | Fixed 256-example held-out eval snapshot |

Rebuild or validate the Countdown snapshot:

```bash
PYTHONPATH=. python scripts/build_countdown_snapshot.py
```

If you already have the HF Arrow file cached:

```bash
PYTHONPATH=. python scripts/build_countdown_snapshot.py \
  --arrow-file /path/to/countdown-tasks-3to4-train.arrow
```

Sudoku uses the bundled CSV splits:

| File | Contents |
| --- | --- |
| `data/train_sudoku_split_new.csv` | 4x4 Sudoku training split |
| `data/test_sudoku_split_new.csv` | 4x4 Sudoku eval split for standalone evaluation |

The Countdown and Sudoku data/evaluation plumbing is adapted from the [SPG codebase](https://github.com/facebookresearch/SPG/tree/main). We thank the SPG authors for releasing their clean codebase!

## Training

The provided Slurm scripts are Kempner/H100 examples. On other clusters, edit the `#SBATCH` account/partition lines and networking interface names.

```bash
export LLADA_MODEL_PATH=/path/to/LLaDA-8B-Instruct

# Countdown
sbatch scripts/train_countdown.sbatch

# Sudoku
sbatch scripts/train_sudoku.sbatch
```

## Checkpointing

DTM writes two kinds of checkpoints:

| Checkpoint | Contents | Intended use |
| --- | --- | --- |
| Mid-phase `checkpoint-*.ckpt` | Student LoRA adapter only | Evaluation only |
| Phase-boundary `phase-boundary-*.ckpt` | EMA adapter in both student and teacher slots, prompt-loader state, scheduler metadata | Resume training |

Resume training only from phase-boundary checkpoints. Both checkpoint types omit optimizer state; mid-phase checkpoints are rejected by the trainer for resume.

Countdown online validation runs at phase boundaries on `data/countdown_cd3_test.jsonl`. Sudoku online validation is disabled in training; use the standalone eval path below.

## Evaluation

Run generation:

```bash
sbatch scripts/eval.sbatch countdown /path/to/checkpoint.ckpt /path/to/output_dir
sbatch scripts/eval.sbatch sudoku /path/to/checkpoint.ckpt /path/to/output_dir
```

Aggregate accuracy:

```bash
cd /path/to/output_dir
PYTHONPATH=/path/to/DTM python -m eval.parser_json
```

## Citation

If you find DTM useful in your research, please cite:

```bibtex
@inproceedings{chen2026dtm,
  title={Discrete Tilt Matching},
  author={Chen, Yuyuan and Wang, Shiyi and Potaptchik, Peter and Kim, Jaeyeon and Albergo, Michael S.},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
