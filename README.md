# DART-LLM: Dynamic Adaptive Rank Tuning for Tensor-Train Decomposed Large Language Models

**DART-LLM** is a research project that aims to dynamically adapt the TT (Tensor-Train) ranks of LLM (Large Language Model) components during fine-tuning. By integrating differentiable regularization with a hybrid continuous-to-discrete relaxation scheme, DART-LLM automatically prunes redundant dimensions in the TT cores, leading to reduced parameter counts, faster inference, and efficient deployment on resource-constrained devices—all while maintaining competitive performance on language tasks.

## Overview
Large language models have achieved state-of-the-art results in NLP, but their enormous parameter counts hinder deployment—especially on edge devices. Tensor-Train decomposition has been used to compress these models (e.g., in approaches like LoRETTA and TT-LoRA) by factorizing weight matrices into a series of smaller cores with fixed ranks.

**DART-LLM** introduces a dynamic approach to TT rank selection that enables the model to learn its own optimal TT rank configuration during fine-tuning. Our method has two main stages:

- **Differentiable Regularization**: A nuclear norm (or similar surrogate) is applied to the unfolded TT cores to encourage low-rank representations.
- **Hybrid Continuous-to-Discrete Relaxation**: Once the regularization phase has converged, a thresholding or Gumbel-Softmax based discrete selection is used to fix the effective TT ranks, followed by fine-tuning.
This adaptive mechanism is designed to maintain performance (e.g., measured by perplexity for language modeling) while significantly reducing model size and improving inference efficiency.

## Motivation
- **Model Efficiency**: Reduce redundant parameters in TT-decomposed LLMs.
- **Dynamic Adaptation**: Automatically adapt TT ranks during fine-tuning based on data complexity.
- **Resource-Constrained Deployment**: Lower memory footprint and latency for edge applications.
- **Competitive Performance**: Maintain or even improve task performance (perplexity, BLEU scores) despite aggressive compression.

## Key Features
Dynamic TT Rank Selection: Incorporates differentiable regularization to drive redundant singular values toward zero.
Hybrid Relaxation Scheme: Transitions from a continuous adaptation phase to a discrete, pruned TT rank configuration.
Integration with Fine-Tuning: Designed to be used during the fine-tuning of large language models, similar in spirit to LoRA but with adaptive capacity.
Extensive Experimental Protocol: Evaluation on standard language modeling benchmarks (WikiText-103, Penn Treebank) and optional sequence-to-sequence tasks.

## Project Structure

```bash
DART-LLM/
├── main.tex              # LaTeX project source for the paper proposal
├── references.bib        # Bibliography file
├── README.md             # This file
├── src/
│   ├── tt_core.py        # Module for TT core and adaptive regularization
│   ├── discrete_rank.py  # Module for discrete rank selection (thresholding/Gumbel-Softmax)
│   ├── model.py          # Integration into a transformer/LLM architecture
│   └── train.py          # Training pipeline with adaptive rank selection
├── experiments/
│   ├── PTB/              # Scripts and notebooks for Penn Treebank experiments
│   ├── WikiText103/      # Scripts and notebooks for WikiText-103 experiments
│   └── NMT/              # (Optional) Scripts for machine translation experiments
└── docs/
    └── literature_review.md  # Detailed literature review and background notes
```

## Installation

### Requirements:

- Python 3.8+
- PyTorch 1.9+
- NumPy
- TensorLy (or an alternative TT library)
- Additional libraries: matplotlib, tqdm, etc.

### Setup:

1. Clone the repository:

```bash
git clone https://github.com/yourusername/DART-LLM.git
cd DART-LLM
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Training with Adaptive TT Rank Selection
To run the training pipeline with adaptive TT rank selection on a language modeling task:

```bash
python src/train.py --dataset WikiText103 --epochs 50 --lambda_reg 0.01 --threshold 0.1
```

The training script will:

- Load your baseline LLM with TT-decomposed embedding layers.
- Train with a combined loss: application loss (e.g., cross-entropy) plus the differentiable regularization loss.
- Periodically log the distribution of surrogate singular values.
- At a designated epoch, perform discrete rank selection (either thresholding or Gumbel-Softmax) to update TT ranks.
- Continue fine-tuning with the updated (compressed) TT cores.

### Inference
After training, use the provided inference scripts to evaluate the model on validation/test sets:

```bash
python src/infer.py --dataset PTB --model_path saved_models/DART-LLM.pth
```

## Experiments and Evaluation
Our experimental protocol includes:

- **Datasets**: WikiText-103, Penn Treebank for language modeling; optionally, WMT English–German for translation.

- **Metrics**:

    - Performance: Perplexity (LM), BLEU (if applicable).
    - Compression: Effective TT ranks, total parameter count reduction, and compression ratio.
    - Efficiency: Training convergence, inference latency (measured on edge devices like Raspberry Pi).

- Ablation Studies:

    - Compare continuous-phase only versus the full hybrid approach.
    - Sensitivity analysis on the regularization coefficient $\lambda$ and threshold $\delta$.

All experimental results, graphs, and tables are saved in the experiments/ directory.

## Roadmap and Future Work

- Short Term:
    - Complete literature review and baseline replication.
    - Implement and test the differentiable regularization module.

- Mid Term:
    - Implement the discrete relaxation module.
    - Conduct comprehensive experiments on standard benchmarks.
    - Refine hyperparameters based on ablation studies.


## Citation
If you use or reference this project in your work, please cite our foundational papers:

```
@article{oseledets2011tensor,
  title={Tensor-Train Decomposition},
  author={Oseledets, Ivan V.},
  journal={SIAM Journal on Scientific Computing},
  volume={33},
  number={5},
  pages={2295--2317},
  year={2011}
}

@inproceedings{yangetal2024loretta,
  title={LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models},
  author={Yang, Yifan and Zhou, Jiajun and Wong, Ngai and Zhang, Zheng},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={3161--3176},
  year={2024}
}

@misc{anjum2024ttlora,
  title={Tensor Train Low-Rank Approximation (TT-LoRA): Democratizing AI with Accelerated LLMs},
  author={Anjum, Afia and Eren, Maksim E and Boureima, Ismael and Alexandrov, Boian and Bhattarai, Manish},
  howpublished={arXiv preprint arXiv:2408.01008},
  year={2024}
}
```

## License
This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

