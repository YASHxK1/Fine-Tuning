# PRD: Fine-Tuning DeepSeek-R1-Distill-Qwen-1.5B with OpenMathReasoning

## 1. Project Overview

Project name: MathReason-1.5B

Objective: Fine-tune `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` using NVIDIA's OpenMathReasoning dataset to improve mathematical reasoning, step-by-step problem solving, and answer accuracy.

Base model: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

Dataset: [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning)

Training approach: Supervised fine-tuning (SFT) with LoRA or QLoRA.

## 2. Goals and Success Criteria

- Improve accuracy on mathematical reasoning benchmarks.
- Preserve the base model's general language capabilities as much as possible.
- Produce reproducible training runs and evaluation results.
- Support inference through Hugging Face Transformers.
- Make the fine-tuned adapter and training configuration reusable.

### Success metrics

| Metric                  | Target                          |
| ----------------------- | ------------------------------- |
| GSM8K accuracy          | Improve over baseline           |
| MATH benchmark accuracy | Improve over baseline           |
| Training stability      | No NaN or divergent loss        |
| Reproducibility         | Fixed seed and versioned config |
| Evaluation integrity    | No training/test contamination  |

Targets should be finalized after baseline evaluation.

## 3. Dataset Requirements

Use the OpenMathReasoning dataset, which includes mathematical reasoning traces and different solution formats.

### Data pipeline

1. Load the dataset using Hugging Face Datasets.
2. Inspect available configurations, splits, and columns.
3. Select examples with valid mathematical questions and solutions.
4. Remove empty, malformed, and excessively long examples.
5. Deduplicate examples and check for benchmark contamination.
6. Convert examples into the base model's chat format.
7. Split training and validation data by problem, avoiding duplicate leakage.

Important: OpenMathReasoning is a large dataset. Start with a small, representative subset before downloading or processing the entire dataset.

Dataset fields must be verified against the actual repository schema before implementation.

## 4. Model and Training Architecture

### Model configuration

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Framework: PyTorch, Transformers, TRL, PEFT
- Fine-tuning: LoRA initially; QLoRA if GPU memory is constrained
- Precision: BF16 when supported, otherwise FP16 or suitable quantized training
- Checkpointing: Save adapter checkpoints and training state

### Initial hyperparameters

| Parameter            | Starting value |
| -------------------- | -------------- |
| LoRA rank            | 16             |
| LoRA alpha           | 32             |
| LoRA dropout         | 0.05           |
| Learning rate        | 2e-5           |
| Epochs               | 1              |
| Sequence length      | 2,048 tokens   |
| Effective batch size | 16             |
| Warmup ratio         | 0.03           |
| Weight decay         | 0.01           |
| Random seed          | 42             |

These are starting values, not guaranteed optimal settings. Adjust sequence length and batch size to available VRAM. Use gradient accumulation to achieve the target effective batch size.

## 5. Training Pipeline

The system must implement these stages:

1. Environment setup: Install compatible dependencies and record library versions.
2. Baseline evaluation: Evaluate the original model before training.
3. Dataset preparation: Filter, normalize, tokenize, and split the data.
4. Training: Run SFT with LoRA/QLoRA.
5. Checkpointing: Save checkpoints at defined intervals.
6. Validation: Monitor loss and evaluate a held-out validation subset.
7. Model export: Save the adapter and tokenizer configuration.
8. Inference testing: Verify that the saved adapter loads and produces valid answers.

### Training objective

Use causal language-model next-token prediction over the target solution sequence. Mask prompt tokens from the loss where appropriate so the model learns to generate solutions rather than reproduce the question.

## 6. Evaluation Plan

Evaluate the base and fine-tuned models under identical decoding and scoring conditions.

- GSM8K: Arithmetic and grade-school mathematical reasoning.
- MATH: Competition-level mathematical problem solving.
- AIME-style problems: More difficult reasoning, subject to appropriate dataset licensing and evaluation setup.
- Custom tests: Algebra, geometry, number theory, calculus, and multi-step word problems.

Use exact-match or normalized numerical answer extraction where applicable. Report accuracy, invalid-output rate, and inference latency. Keep benchmark test examples out of training.

## 7. Functional Requirements

- Configurable dataset subset size and sequence length.
- Support for resuming interrupted training.
- Structured logs for training and validation loss.
- Automatic saving of model configuration and experiment metadata.
- Reproducible evaluation scripts.
- CLI commands for training, evaluation, and inference.
- Clear error handling for malformed samples, out-of-memory errors, and failed checkpoints.

## 8. Suggested Repository Structure

```
mathreason-1.5b/
├── configs/
│   └── train.yaml
├── data/
│   └── README.md
├── src/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── tests/
│   ├── test_data.py
│   └── test_inference.py
├── outputs/
├── requirements.txt
├── README.md
└── LICENSE
```

## 9. Deliverables

- Reproducible fine-tuning code and configuration.
- Dataset preparation and validation pipeline.
- Fine-tuned LoRA adapter.
- Before-and-after benchmark report.
- Inference script and setup documentation.
- Model card documenting training data, limitations, and intended use.

## 10. Risks and Constraints

- GPU memory: A 1.5B model is relatively small, but sequence length and batch size can still cause out-of-memory errors.
- Dataset scale: Processing the complete dataset may require substantial storage and compute.
- Reasoning quality: Training on long reasoning traces does not guarantee better mathematical accuracy.
- Data licensing: Verify the model and dataset licenses and applicable redistribution requirements.
- Catastrophic forgetting: Monitor general capabilities and reduce training intensity if they degrade.

## 11. Definition of Done

The project is complete when the fine-tuned adapter loads successfully, training can be reproduced from documented configuration, benchmark evaluation runs end-to-end, and results are compared against the original model using the same evaluation protocol.

Recommended first milestone: Fine-tune on 10,000–50,000 curated examples, establish benchmark improvements, and scale up only if results justify the additional compute.