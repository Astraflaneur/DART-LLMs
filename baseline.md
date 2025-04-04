# LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models

**Authors**: Yifan Yang, Jiajun Zhou, Ngai Wong, Zheng Zhang  
**Affiliations**: University of California, Santa Barbara; The University of Hong Kong  
**Code**: [GitHub Repository](https://github.com/yifanycc/loretta)  
**Paper**: [2024.NAACL-Long.174](https://arxiv.org/abs/XXXX.XXXXX)  

---

## Abstract
LoRETTA is a parameter-efficient fine-tuning (PEFT) framework for large language models (LLMs) that leverages tensor-train (TT) decomposition to drastically reduce trainable parameters. It introduces two variants:  
- **LoRETTAₐdₚ**: Uses tensorized adapters for lightweight fine-tuning.  
- **LoRETTAᵣₑₚ**: Reparameterizes weights via tensor factors for ultra-low parameter updates.  

Key results:  
- Achieves **100× fewer parameters** than LoRA/Adapters on LLaMA-2 models.  
- Matches or outperforms full fine-tuning and existing PEFT methods across GLUE, SuperGLUE, and generation tasks.  
- Demonstrates **anti-overfitting** capabilities and enhanced **multi-task learning** efficiency.  

---

## Method
### Tensor-Train (TT) Decomposition
- Reshapes weight matrices into high-dimensional tensors, decomposed into small tensor factors.  
- Reduces parameters from `M×N` to `∑rᵢ₋₁kᵢrᵢ` (controlled by TT ranks).  

### LoRETTA Variants
1. **LoRETTAₐdₚ**  
   - Injects tensorized adapters after attention and feed-forward layers.  
   - Compresses parameters via TT layers (e.g., 1.2K vs. 98K parameters for Adapters).  
2. **LoRETTAᵣₑₚ**  
   - Reparameterizes weight updates using TT factors (e.g., 1K vs. 12K parameters for LoRA).  
   - Initializes tensor factors via noise reduction to avoid optimization issues.  

---

## Experiments
### Datasets & Models
- **BERT-family**: DeBERTa-base, RoBERTa-base (GLUE benchmark).  
- **LLaMA-2**: 7B, 13B, 70B models (SuperGLUE, SQuAD, DROP).  
- **Low-data setup**: 1,000 training examples for LLaMA-2 tasks.  

### Baselines
- Full fine-tuning (FT), LoRA, Adapters, Prefix/Prompt Tuning, BitFit, IA3.  

### Key Hyperparameters
- **Learning rate**: `1e-4` to `5e-4` (AdamW optimizer).  
- **Batch size**: 16–32 for BERT-family; 1–2 for LLaMA-2.  
- **TT ranks**: 2–32 (adaptively adjusted).  

---

## Results
### Performance Highlights
| Model          | Method           | Trainable Params | SST-2 (Acc) | SQuAD (F1) | Avg. GLUE Score |
|----------------|------------------|------------------|-------------|------------|-----------------|
| DeBERTa-Base   | LoRETTAₐdₚ       | 0.10M            | 95.30       | -          | 84.96           |
| LLaMA-2-7B     | LoRETTAₐdₚ       | 0.88M            | -           | 90.17      | 87.0 (BoolQ)    |
| LLaMA-2-70B    | LoRETTAₐdₚ       | 4.79M            | -           | **94.33**  | 74.50 (DROP)    |

- **Parameter Efficiency**:  
  - LoRETTAᵣₑₚ uses **1MB storage** (vs. 3.5MB for LoRA on DeBERTa).  
  - **57× fewer parameters** than Adapters on LLaMA-2-70B.  

### Anti-Overfitting & Multi-Task Learning
- **Stable training curves** (Fig. 4) with lower evaluation loss variance.  
- **Multi-task retention**: LoRETTA achieves **65.45% avg accuracy** vs. 55.70% for LoRA (Table 4).  

### Memory & Computation Efficiency
- **57.4× less memory** vs. LoRA on LLaMA-2-7B.  
- **Reduced FLOPs**: `6.14E+15` for LoRETTAₐdₚ vs. `6.18E+15` for Adapters.  

Here's a structured presentation of the results from the tables and a guide to recreate **Figure 4** using Python:

---

### **Table 1: GLUE Benchmark Results (BERT-family Models)**
| Model & Method          | Train. Params (M) | MNLI  | SST-2 | MRPC  | CoLA  | QNLI  | QQP   | RTE   | STS-B | Avg.  |
|-------------------------|-------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| DeBERTa-Base (FT)       | 139.19           | 88.67 | 94.61 | 91.98 | 59.32 | 93.04 | 91.42 | 68.23 | 91.10 | 84.79 |
| DeBERTa-Base (LoRETTAₐdₚ) | 0.10             | 85.93 | 95.30 | 93.53 | 60.84 | 92.99 | 84.08 | 75.50 | 91.32 | **84.96** |
| DeBERTa-Base (LoRETTAᵣₑₚ) | 0.05             | 86.80 | 95.53 | 88.73 | 59.69 | 93.25 | 89.20 | 75.81 | 90.66 | 84.95 |
| RoBERTa-Base (LoRETTAₐdₚ) | 0.10             | 85.61 | 94.38 | 91.08 | 62.70 | 92.12 | 87.22 | 78.70 | 90.26 | **85.26** |

---

### **Table 2: LLaMA-2-7B Performance (Low-Data Setting)**
| Model & Method          | Train. Params (M) | CB    | BoolQ | WSC   | COPA  | ReCoRD | SQuAD | DROP  |
|-------------------------|-------------------|-------|-------|-------|-------|--------|-------|-------|
| LLaMA2-7B (FT)          | 6738.42          | 66.07 | 84.6  | 63.46 | 86    | 81.1   | 90.71 | 51.38 |
| LLaMA2-7B (LoRETTAₐdₚ)  | 0.88             | 66.07 | **87.0** | **63.46** | **87** | 80.0   | 90.17 | **51.60** |

---

### **Table 3: LLaMA-2-13B/70B Performance**
| Model & Method          | Train. Params (M) | COPA | ReCoRD | SQuAD | DROP  |
|-------------------------|-------------------|------|--------|-------|-------|
| LLaMA2-70B (LoRETTAₐdₚ) | 4.79              | -    | -      | **94.33** | **74.50** |

---

### **Table 4: Multi-Task Learning Anti-Forgetting**
| Model & Method          | SST-2 | MRPC  | QNLI  | Avg.  |
|-------------------------|-------|-------|-------|-------|
| DeBERTa-Base (LoRETTAₐdₚ) | 52.29 | 39.22 | 91.52 | 61.01 |
| DeBERTa-Base (LoRETTAᵣₑₚ) | 51.26 | **52.94** | **92.15** | **65.45** |

---

### **Table 5: Memory & FLOPs Efficiency**
| Model & Method          | Memory (µs) | FLOPs (Reduction) |
|-------------------------|-------------|--------------------|
| LLaMA2-7B (LoRETTAₐdₚ)  | **9879**    | **6.14E+15**       |

---

### **Table 6: Tensor Rank Analysis**
| LoRETTAₐdₚ (Rank) | SST-2 | QNLI  |
|--------------------|-------|-------|
| r=2               | 95.41 | 92.04 |
| r=5               | 95.30 | 92.99 |

---

### **Table 7: Tensor Shape Configuration**
| Tensor Shape       | Params (M) | SST-2 | MRPC  | QNLI  |
|--------------------|------------|-------|-------|-------|
| [8,8,12,8,8]      | 0.10       | 95.30 | 93.53 | 93.25 |

