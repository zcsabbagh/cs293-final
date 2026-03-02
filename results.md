# CS293 Fine-Tuning Results

## Setup
- Dataset: MathFish (allenai/mathfish) from HuggingFace
- Task: Predict CCSS "Addressing" standards from math problem text
- Train: 4,960 problems | Eval: 200 random test problems (seed=42)
- Fine-tuning: QLoRA via Unsloth, LoRA r=16, alpha=16, 3 epochs, lr=2e-4
- Hardware: NVIDIA A100-SXM4-40GB (Colab Pro)

## Experiment 1: Llama 3.2 1B (baseline vs fine-tuned, 3 epochs, r=16)

| Model | Level | Precision | Recall | F1 | Exact Match |
|-------|-------|-----------|--------|-----|-------------|
| 1B base (zero-shot) | standard | 0.006 | 0.013 | 0.008 | 0.000 |
| 1B fine-tuned | standard | 0.067 | 0.063 | 0.059 | 0.030 |
| 1B base (zero-shot) | cluster | 0.012 | 0.015 | 0.013 | 0.010 |
| 1B fine-tuned | cluster | 0.119 | 0.118 | 0.114 | 0.085 |
| 1B base (zero-shot) | domain | 0.027 | 0.030 | 0.028 | 0.025 |
| 1B fine-tuned | domain | 0.233 | 0.222 | 0.224 | 0.195 |

F1 improvement: 7-8x across all levels.

## Experiment 2: 1B with 8 epochs (r=16)

| Model | Level | Precision | Recall | F1 | Exact Match |
|-------|-------|-----------|--------|-----|-------------|
| 1B fine-tuned (8 ep) | standard | 0.120 | 0.134 | 0.119 | 0.060 |
| 1B fine-tuned (8 ep) | cluster | 0.225 | 0.228 | 0.218 | 0.160 |
| 1B fine-tuned (8 ep) | domain | 0.316 | 0.309 | 0.306 | 0.245 |

F1 improvement over 3 epochs: 2x at standard, 1.9x at cluster, 1.4x at domain. Model hadn't converged at 3 epochs.

## Experiment 3: Llama 3.2 3B (3 epochs, r=16)

| Model | Level | Precision | Recall | F1 | Exact Match |
|-------|-------|-----------|--------|-----|-------------|
| 3B fine-tuned (3 ep) | standard | 0.162 | 0.157 | 0.154 | 0.110 |
| 3B fine-tuned (3 ep) | cluster | 0.268 | 0.254 | 0.256 | 0.215 |
| 3B fine-tuned (3 ep) | domain | 0.360 | 0.334 | 0.342 | 0.300 |

3B at 3 epochs beats 1B at 8 epochs. Model scale matters more than training duration.

## Experiment 4: Llama 3.1 8B (in progress)

## Experiment 5: Building On / Prerequisites (in progress)
- 1B model, 8 epochs, trained on "Building On" labels instead of "Addressing"

## Summary: Addressing Task F1 by Model

| Model | Standard | Cluster | Domain |
|-------|----------|---------|--------|
| 1B base (zero-shot) | 0.008 | 0.013 | 0.028 |
| 1B fine-tuned (3 ep) | 0.059 | 0.114 | 0.224 |
| 1B fine-tuned (8 ep) | 0.119 | 0.218 | 0.306 |
| 3B fine-tuned (3 ep) | 0.154 | 0.256 | 0.342 |

## A2 Zero-Shot Baselines (for comparison)

| Model | Level | Precision | Recall | F1 | Exact Match |
|-------|-------|-----------|--------|-----|-------------|
| GPT-5.2 | standard | 0.446 | 0.568 | 0.500 | 0.350 |
| Claude Sonnet 4.6 | standard | 0.400 | 0.591 | 0.477 | 0.375 |
| Gemini | standard | 0.682 | 0.341 | 0.455 | 0.400 |
| GPT-5.2 | domain | 0.587 | 0.711 | 0.643 | 0.500 |
| Claude Sonnet 4.6 | domain | 0.521 | 0.658 | 0.581 | 0.525 |
| Gemini | domain | 0.714 | 0.395 | 0.508 | 0.450 |

Note: A2 baselines were on a different 40-problem validation set, not directly comparable but directionally useful.
