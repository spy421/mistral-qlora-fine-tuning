Mistral-7B QLoRA Fine-Tuning
Fine-tuning Mistral-7B-Instruct-v0.3 with QLoRA on a custom instruction-response dataset.
The project demonstrates how to fine-tune a 7B language model while training only a small fraction of its parameters.
Overview
The project uses:
Mistral-7B-Instruct-v0.3
4-bit quantization
LoRA / QLoRA
Hugging Face Transformers
PEFT
PyTorch
Google Colab with NVIDIA Tesla T4
The main goal was to adapt the model to a custom instruction-response dataset without updating the full 7B parameter model.
Dataset
The final dataset contains 500 examples.
Split	Examples
Training	450
Validation	50
Total	500
Each example contains:
instruction
response
The dataset was processed by removing duplicates and filtering low-quality examples before training.
QLoRA Configuration
The base model was loaded in 4-bit precision.
LoRA configuration:
LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
Trainable Parameters
The full model contains:
7,251,431,424 parameters
Only:
3,407,872 parameters
were trainable.
That is:
0.0470% of the model.
The original model weights remained frozen.
Training
Training configuration:
Setting	Value
Epochs	3
Train batch size	2
Evaluation batch size	2
Gradient accumulation	4
Learning rate	2e-4
Max sequence length	512
GPU	NVIDIA Tesla T4
Results
Loss
Epoch	Train Loss	Validation Loss
1	1.8579	1.9127
2	1.8018	1.9038
3	1.7881	1.9183
The training loss decreased during all three epochs.
Validation loss improved slightly by epoch 2 and increased slightly during epoch 3.

Base vs Fine-Tuned
The original Mistral model was compared with the fine-tuned model using the same evaluation questions.
The comparison results are stored in:
results/base_vs_finetuned.json
The fine-tuned model uses the trained LoRA adapter together with the original base model.
Project Structure
project/
│
├── adapter/
│   └── mistral-qlora-adapter/
│
├── data/
│   └── train.jsonl
│
├── notebooks/
│   └── mistral_qlora.ipynb
│
├── results/
│   ├── loss_curve.png
│   ├── training_metrics.json
│   └── base_vs_finetuned.json
│
├── src/
│
└── README.md
Technologies
Python · PyTorch · Transformers · PEFT · BitsAndBytes · Datasets · CUDA
Adapter
Only the trained LoRA adapter is stored in this repository.
The full Mistral-7B base model is not included.
The adapter can be loaded together with the original base model using PEFT.
Notebook
The complete experiment is available in:
notebooks/mistral_qlora.ipynb
The notebook contains the complete workflow:
dataset → preprocessing → tokenization → QLoRA → training → evaluation
Hardware
Training was performed using a single:
NVIDIA Tesla T4 — 15 GB VRAM
Google Colab was used as the training environment.