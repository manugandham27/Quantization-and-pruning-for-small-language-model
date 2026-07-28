# EdgeTune Experiment Logs & Hyperparameter Configurations

## Run Log Overview

| Run ID | Variant Name | Target Sparsity | Quant Bits | ROUGE-L | Disk (MB) | Peak Mem (MB) | TTFT (ms) | Tokens/sec |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `run-001` | Baseline (FP16) | 0% | 16 | 0.00% | 30.8 MB | 30.8 MB | 23.8 ms | 103.6 tok/s |
| `run-002` | LoRA Fine-tuned | 0% | 16 | 0.00% | 34.2 MB | 62.0 MB | 24.5 ms | 101.8 tok/s |
| `run-003` | QLoRA Fine-tuned | 0% | 4 | 0.00% | 10.8 MB | 180.7 MB | 21.1 ms | 72.6 tok/s |
| `run-004` | LoRA + Pruned (50%) | 50% | 16 | 0.00% | 30.8 MB | 118.7 MB | 13.4 ms | 120.0 tok/s |
| `run-005` | LoRA + Quantized (4-bit)| 0% | 4 | 0.87% | 26.5 MB | 152.4 MB | 15.3 ms | 90.5 tok/s |
| `run-006` | Combined Stack | 50% | 4 | 0.00% | 26.5 MB | 221.2 MB | 26.6 ms | 46.0 tok/s |

---

## Detailed Experiment Configurations

### Experiment 1: Baseline FP16 Model
- **Config**: `configs/base_model.yaml`
- **Precision**: `float16`
- **Device**: Apple Silicon MPS / CUDA

### Experiment 2: LoRA Fine-Tuning
- **Config**: `configs/lora.yaml`
- **Rank ($r$)**: 16
- **Alpha ($\alpha$)**: 32
- **Learning Rate**: $3 \times 10^{-4}$
- **Target Modules**: `["c_attn", "c_proj", "c_fc", "q_proj", "v_proj", "k_proj", "o_proj"]`

### Experiment 3: QLoRA Fine-Tuning
- **Config**: `configs/qlora.yaml`
- **Base Quantization**: 4-bit NormalFloat (NF4)
- **Double Quantization**: Enabled
- **Compute Dtype**: `float16`

### Experiment 4: Pruning (50% Sparsity)
- **Config**: `configs/pruning.yaml`
- **Method**: L1 Unstructured Magnitude Pruning
- **Sparsity**: 50.0%

### Experiment 5: 4-Bit Groupwise Quantization
- **Config**: `configs/quantization_gptq.yaml`
- **Method**: Uniform Groupwise Block Packing
- **Group Size**: 64

### Experiment 6: Combined Stacking Pipeline
- **Execution Order**: Fine-tuning $\rightarrow$ Magnitude Pruning (50%) $\rightarrow$ Groupwise 4-Bit Quantization ($g=64$)
