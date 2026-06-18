# PALS: Propagation-Aware Layerwise Sparsity for Post-Training LLM Pruning

This repository implements **PALS** (**Propagation-Aware Layerwise Sparsity**), a post-training pruning framework for large language models. The project is adapted from the RIA pruning codebase and extends it with a propagation-aware non-uniform layerwise sparsity allocation strategy.

PALS is designed for high-sparsity LLM pruning. Instead of assigning the same sparsity ratio to every Transformer layer, PALS estimates how pruning errors from each layer propagate through downstream layers, then protects layers with stronger propagated influence and prunes less sensitive layers more aggressively.

## Highlights

- **Post-training pruning:** no full retraining is required.
- **Layerwise non-uniform sparsity:** different Transformer layers receive different sparsity ratios.
- **Propagation-aware allocation:** layer importance is estimated by single-layer probing and downstream hidden-state deviation.
- **Compatible with existing base pruners:** PALS can be combined with pruning criteria such as Magnitude, SparseGPT, and Wanda.
- **High-sparsity oriented:** mainly evaluated under 60%, 70%, and 80% unstructured sparsity.
- **Efficiency support:** sparse models can obtain CPU inference speedup with sparse inference engines such as DeepSparse.

## Method Overview

Most post-training pruning methods focus on selecting unimportant weights inside each layer, while simply applying a uniform sparsity ratio across all layers. This assumption is weak under high sparsity because different Transformer layers have different sensitivity and redundancy.

PALS addresses this issue by explicitly measuring **pruning-error propagation**.

The overall pipeline is:

1. **Dense reference tracing**

   The dense LLM is first evaluated on a small calibration set. Hidden states and final logits are cached as reference outputs.

2. **Single-layer probing**

   For each Transformer layer, PALS temporarily prunes only that layer while keeping all other layers dense. This creates an isolated pruning perturbation for the current layer.

3. **Propagation measurement**

   The hidden-state deviation between the dense model and the probed model is measured across downstream Transformer layers. This forms an upper-triangular propagation matrix.

4. **Propagation score aggregation**

   For each layer, PALS aggregates downstream hidden-state deviations into a propagation-aware score. A larger score indicates that pruning this layer causes stronger downstream influence.

5. **Rank-logistic sparsity allocation**

   Layers are ranked according to propagation scores. High-propagation layers are assigned lower sparsity, while low-propagation layers are assigned higher sparsity.

6. **Budget-preserving projection**

   The generated layerwise sparsity schedule is projected to strictly satisfy the target global sparsity constraint.

7. **Final pruning**

   The selected base pruner, such as Wanda or SparseGPT, is applied layer by layer using the PALS-generated sparsity schedule.

## Installation

Create a conda environment:

```bash
conda create -n pals python=3.10
conda activate pals
```

Install dependencies:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cu121
```

If zero-shot evaluation is needed, install `lm-evaluation-harness`:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

## Usage

### Run PALS pruning

```bash
python main.py \
  --model YOUR_MODEL_NAME_OR_PATH \
  --prune_method pals \
  --sparsity_ratio 0.7 \
  --sparsity_type unstructured \
  --save
```

### Run PALS with Wanda as the base pruner

```bash
python main.py \
  --model YOUR_MODEL_NAME_OR_PATH \
  --prune_method pals \
  --base_pruner wanda \
  --sparsity_ratio 0.7 \
  --sparsity_type unstructured \
  --save
```

### Run PALS with SparseGPT as the base pruner

```bash
python main.py \
  --model YOUR_MODEL_NAME_OR_PATH \
  --prune_method pals \
  --base_pruner sparsegpt \
  --sparsity_ratio 0.7 \
  --sparsity_type unstructured \
  --save
```

### Evaluate perplexity

```bash
python main.py \
  --model YOUR_MODEL_NAME_OR_PATH \
  --prune_method pals \
  --base_pruner wanda \
  --sparsity_ratio 0.7 \
  --sparsity_type unstructured \
  --eval_ppl
```

### Evaluate zero-shot tasks

```bash
python main.py \
  --model YOUR_MODEL_NAME_OR_PATH \
  --prune_method pals \
  --base_pruner wanda \
  --sparsity_ratio 0.7 \
  --sparsity_type unstructured \
  --eval_zero_shot
```

> Note: The exact argument names may depend on your local implementation. If your code registers the method under a different name, such as `pals_wanda` or `pals_sparsegpt`, replace the command-line arguments accordingly.

## Supported Models

The method is mainly designed for decoder-only Transformer-based LLMs. In the experiments, PALS is evaluated on:

- LLaMA-7B / LLaMA-13B
- LLaMA2-7B / LLaMA2-13B
- LLaMA3-8B
- LLaMA3.1-8B
- Vicuna-7B
- Mistral-7B

## Evaluation

PALS is evaluated from three perspectives:

### Language Modeling

Perplexity is measured on **WikiText-2**. Lower perplexity indicates better language modeling performance after pruning.

### Zero-shot Tasks

Zero-shot accuracy is evaluated with EleutherAI LM Evaluation Harness on common reasoning benchmarks, including:

- BoolQ
- RTE
- HellaSwag
- WinoGrande
- ARC-Easy
- ARC-Challenge
- OpenBookQA

### Inference Efficiency

Sparse inference efficiency can be tested on CPU sparse inference engines such as **DeepSparse**. In the paper, the PALS-pruned LLaMA2-7B model achieves clear latency reduction as sparsity increases.

## Main Results

### WikiText-2 Perplexity at 70% Sparsity

Lower is better.

| Base Pruner | Layerwise Sparsity | LLaMA1-7B | LLaMA1-13B | LLaMA2-7B | LLaMA2-13B |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense | - | 5.85 | 5.21 | 5.88 | 4.92 |
| Magnitude | Uniform | 5.2e4 | 8.7e4 | 5.1e4 | 2.1e2 |
| Magnitude | OWL | 2.5e4 | 2.1e4 | 1.8e4 | 68.94 |
| Magnitude | ALP | 310.57 | 2483.12 | 1.2e4 | 35.14 |
| Magnitude | PALS | 1254.24 | 813.81 | 1374.17 | 69.48 |
| SparseGPT | Uniform | 24.22 | 19.54 | 27.14 | 20.45 |
| SparseGPT | OWL | 20.78 | 15.81 | 20.54 | 17.62 |
| SparseGPT | ALP | 19.72 | 14.88 | 19.98 | 15.91 |
| SparseGPT | PALS | 18.51 | 13.84 | 18.54 | 13.48 |
| Wanda | Uniform | 89.61 | 58.15 | 77.27 | 47.99 |
| Wanda | OWL | 25.84 | 17.75 | 30.81 | 21.64 |
| Wanda | ALP | 24.24 | 15.31 | 29.87 | 19.37 |
| Wanda | PALS | 20.84 | 15.72 | 22.84 | 17.21 |

### Recent LLMs on WikiText-2

Lower is better.

| Model | Method | 60% | 70% | 80% |
| --- | --- | ---: | ---: | ---: |
| LLaMA3-8B | Uniform | 24.24 | 173.98 | 729.99 |
| LLaMA3-8B | PALS | 20.12 | 96.98 | 705.27 |
| LLaMA3.1-8B | Uniform | 22.47 | 119.14 | 1254.44 |
| LLaMA3.1-8B | PALS | 19.40 | 85.43 | 1054.78 |
| Vicuna-7B | Uniform | 13.93 | 71.36 | 1614.08 |
| Vicuna-7B | PALS | 12.07 | 29.98 | 482.81 |
| Mistral-7B | Uniform | 12.15 | 67.26 | 382.20 |
| Mistral-7B | PALS | 11.01 | 40.64 | 202.88 |

### Average Zero-shot Accuracy at 70% Sparsity

Higher is better.

| Base Pruner | Layerwise Sparsity | LLaMA1-7B | LLaMA1-13B | LLaMA2-7B | LLaMA2-13B |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense | - | 63.62 | 66.56 | 64.04 | 66.25 |
| Magnitude | Uniform | 34.06 | 36.18 | 35.11 | 36.01 |
| Magnitude | OWL | 36.03 | 38.85 | 36.08 | 39.95 |
| Magnitude | PALS | 37.57 | 39.73 | 39.99 | 43.81 |
| SparseGPT | Uniform | 44.48 | 47.58 | 44.40 | 47.02 |
| SparseGPT | OWL | 47.37 | 50.51 | 47.14 | 51.02 |
| SparseGPT | PALS | 47.47 | 52.28 | 48.67 | 52.97 |
| Wanda | Uniform | 39.27 | 40.76 | 36.15 | 39.78 |
| Wanda | OWL | 45.56 | 49.35 | 43.12 | 47.85 |
| Wanda | PALS | 48.23 | 51.75 | 45.54 | 50.62 |

### CPU Inference Speedup on DeepSparse

The following results are reported on the PALS-pruned LLaMA2-7B model.

| Sparsity | Latency (ms) | Throughput (tokens/s) | Speedup |
| --- | ---: | ---: | ---: |
| Dense | 504.89 | 1.98 | 1.0x |
| 10% | 503.43 | 1.99 | 1.0x |
| 20% | 483.39 | 2.07 | 1.0x |
| 30% | 462.13 | 2.16 | 1.1x |
| 40% | 390.61 | 2.56 | 1.3x |
| 50% | 285.96 | 3.49 | 1.8x |
| 60% | 234.73 | 4.26 | 2.2x |
| 70% | 178.23 | 5.61 | 2.8x |
| 80% | 143.29 | 6.98 | 3.5x |
| 90% | 138.37 | 7.22 | 3.6x |

## Project Structure

```text
.
├── main.py                  # Main entry for pruning and evaluation
├── lib/                     # Pruning methods, model utilities, and evaluation code
├── requirements.txt          # Python dependencies
├── scripts/                  # Optional running scripts
├── README.md                 # Project documentation
└── results/                  # Saved pruning and evaluation results
```

The exact structure may vary depending on your local code organization.

## Notes

- PALS mainly focuses on **unstructured pruning**.
- The embedding layer and language modeling head are usually kept dense.
- The final pruning stage uses the same base pruning criterion as the probing stage.
- Calibration data quality may affect the estimated propagation scores.
- PALS introduces extra probing overhead compared with static allocation methods, but the sparsity schedule only needs to be computed once.

## Acknowledgement

This project is adapted from the RIA codebase and builds on the post-training pruning pipelines of SparseGPT and Wanda. We thank the authors of RIA, SparseGPT, Wanda, OWL, and AlphaPruning for their open-source contributions and related research.

## Citation

If you use this repository, please cite the related pruning methods and the PALS paper after publication.

```bibtex
@misc{pals2026,
  title={PALS: Propagation-Aware Layerwise Sparsity for Post-Training LLM Pruning},
  author={Anonymous},
  year={2026},
  note={Under review}
}
```
