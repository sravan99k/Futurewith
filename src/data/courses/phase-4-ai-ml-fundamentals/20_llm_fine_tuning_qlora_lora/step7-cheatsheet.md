# LLM Fine-tuning Cheatsheet

## Quick Reference Guide for LoRA & QLoRA

### Table of Contents

1. [Installation Commands](#installation)
2. [LoRA Configuration](#lora-config)
3. [QLoRA Setup](#qlora-setup)
4. [Training Commands](#training)
5. [Common Code Snippets](#code-snippets)
6. [Hyperparameters](#hyperparameters)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#optimization)

## Installation Commands {#installation}

### Essential Packages

```bash
# Core dependencies
pip install transformers
pip install peft
pip install bitsandbytes
pip install accelerate
pip install datasets
pip install evaluate
pip install wandb

# For development
pip install jupyter
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy

# Memory optimization
pip install xformers  # Optional but recommended
```

### GPU Setup Verification

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

## LoRA Configuration {#lora-config}

### Basic LoRA Setup

```python
from peft import LoraConfig, get_peft_model

# Standard LoRA configuration
lora_config = LoraConfig(
    r=16,                    # Rank (4-32 recommended)
    lora_alpha=16,           # Alpha scaling (usually 2*r)
    lora_dropout=0.1,        # Dropout rate (0.05-0.2)
    bias="none",             # Bias adaptation
    task_type="CAUSAL_LM",   # Task type
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Target layers
)

# Apply LoRA to model
model = get_peft_model(base_model, lora_config)
```

### Advanced LoRA Configurations

```python
# For different model sizes
configurations = {
    "7B_model": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    },
    "13B_model": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"]
    },
    "creative_writing": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.2,
        "target_modules": ["q_proj", "v_proj"]
    },
    "technical_tasks": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "mlp"]
    }
}
```

## QLoRA Setup {#qlora-setup}

### QLoRA Configuration

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,    # Double quantization
    bnb_4bit_quant_type="nf4",         # Quantization type
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Add LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

### Quantization Types

```python
quantization_types = {
    "nf4": {     # Optimal for normal distribution
        "description": "4-bit NormalFloat",
        "use_case": "General purpose"
    },
    "fp4": {     # 4-bit floating point
        "description": "4-bit Floating Point",
        "use_case": "When maintaining FP characteristics"
    },
    "int4": {    # 4-bit integer
        "description": "4-bit Integer",
        "use_case": "Simple quantization"
    }
}
```

## Training Commands {#training}

### Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # Output
    output_dir="./fine-tuned-model",
    overwrite_output_dir=True,

    # Training
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,      # Effective batch size

    # Optimization
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=500,
    lr_scheduler_type="cosine",

    # Mixed precision
    fp16=True,                          # Enable FP16

    # Evaluation
    evaluation_strategy="epoch",
    eval_steps=500,

    # Saving
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,

    # Logging
    logging_steps=100,
    logging_dir="./logs",

    # Other
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to="wandb"                   # or None
)
```

### Training Loop

```python
from transformers import Trainer, DataCollatorForLanguageModeling

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # For causal LM
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Save
trainer.save_model()
tokenizer.save_pretrained("./fine-tuned-model")
```

## Common Code Snippets {#code-snippets}

### Data Preparation

```python
def prepare_data(dataset, tokenizer, max_length=512):
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

# Usage
tokenized_dataset = prepare_data(dataset, tokenizer)
```

### Text Generation

```python
def generate_text(model, tokenizer, prompt, **kwargs):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
result = generate_text(
    model, tokenizer,
    "Explain the benefits of renewable energy:",
    temperature=0.8,
    max_new_tokens=150
)
```

### Memory Optimization

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use 8-bit Adam optimizer
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(model.parameters(), lr=2e-4)

# Mixed precision training
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="fp16")

# Memory monitoring
def check_memory():
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
```

### Multi-GPU Training

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=4
)

# Prepare for distributed training
model, optimizer, training_loader = accelerator.prepare(
    model, optimizer, training_loader
)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(training_loader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()
```

### Model Merging

```python
# Merge LoRA weights with base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")

# Load for inference
merged_model = AutoModelForCausalLM.from_pretrained("./merged-model")
```

## Hyperparameters {#hyperparameters}

### Recommended Hyperparameter Ranges

```python
hyperparameter_ranges = {
    "learning_rate": {
        "range": [1e-5, 5e-4],
        "typical": [2e-4, 1e-4],
        "note": "Lower for larger models"
    },
    "batch_size": {
        "range": [1, 16],
        "typical": [2, 8],
        "note": "Increase with gradient accumulation"
    },
    "lora_rank": {
        "range": [4, 64],
        "typical": [8, 32],
        "note": "Higher for more complex tasks"
    },
    "lora_alpha": {
        "range": [8, 128],
        "typical": [16, 64],
        "note": "Usually 2x the rank"
    },
    "lora_dropout": {
        "range": [0.0, 0.3],
        "typical": [0.05, 0.2],
        "note": "Higher for small datasets"
    },
    "epochs": {
        "range": [1, 10],
        "typical": [2, 5],
        "note": "Less needed with good data"
    },
    "max_length": {
        "range": [256, 2048],
        "typical": [512, 1024],
        "note": "Match your task requirements"
    }
}
```

### Quick Configuration Templates

```python
configs = {
    "quick_test": {
        "learning_rate": 1e-4,
        "batch_size": 2,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "epochs": 1,
        "max_length": 512
    },
    "balanced": {
        "learning_rate": 2e-4,
        "batch_size": 4,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "epochs": 3,
        "max_length": 1024
    },
    "high_quality": {
        "learning_rate": 5e-5,
        "batch_size": 2,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "epochs": 5,
        "max_length": 2048
    },
    "memory_limited": {
        "learning_rate": 1e-4,
        "batch_size": 1,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "epochs": 3,
        "max_length": 512,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 8
    }
}
```

## Troubleshooting {#troubleshooting}

### Common Issues and Solutions

#### Out of Memory Errors

```python
# Solutions for OOM errors

# 1. Reduce batch size
per_device_train_batch_size = 1

# 2. Increase gradient accumulation
gradient_accumulation_steps = 8

# 3. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 4. Use QLoRA instead of LoRA
# (4-bit quantization instead of 16-bit)

# 5. Clear GPU cache
import torch
torch.cuda.empty_cache()

# 6. Use mixed precision
fp16 = True

# 7. Reduce max length
max_length = 512
```

#### Slow Training

```python
# Solutions for slow training

# 1. Use smaller models initially
base_model = "microsoft/DialoGPT-medium"

# 2. Optimize data loading
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,  # Parallel data loading
    pin_memory=True
)

# 3. Enable compilation (PyTorch 2.0+)
model = torch.compile(model)

# 4. Use efficient attention kernels
# Install: pip install xformers

# 5. Optimize data types
torch.bfloat16  # More efficient than float32
```

#### Poor Performance

```python
# Solutions for poor performance

# 1. Check data quality
print(f"Dataset size: {len(dataset)}")
print(f"Sample text length: {len(dataset[0]['text'])}")

# 2. Verify tokenization
tokenizer.decode(tokenizer.encode(sample_text))

# 3. Adjust learning rate
learning_rate = 1e-4  # Try different values

# 4. Increase LoRA rank
lora_rank = 32  # Increase from 16

# 5. Add more target modules
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "mlp"]

# 6. Check for overfitting
# Monitor validation loss
if val_loss > train_loss * 1.5:
    # Reduce epochs or increase dropout
    lora_dropout = 0.2
```

### Debug Code Snippets

```python
# Check model parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_parameters(model)
print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Trainable percentage: {trainable/total*100:.2f}%")

# Check model size
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

model_size = get_model_size(model)
print(f"Model size: {model_size:.2f} MB")

# Monitor training
class TrainingMonitor(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step}: Loss = {logs.get('loss', 'N/A')}")
```

## Performance Optimization {#optimization}

### Memory Optimization Checklist

```python
optimization_checklist = {
    "data": [
        "Use efficient data types (bfloat16)",
        "Implement gradient checkpointing",
        "Clear unnecessary variables",
        "Use weak references for large objects"
    ],
    "model": [
        "Use QLoRA for large models",
        "Enable gradient checkpointing",
        "Use efficient attention mechanisms",
        "Implement model compilation (PyTorch 2.0+)"
    ],
    "training": [
        "Use gradient accumulation",
        "Implement mixed precision",
        "Optimize batch size",
        "Use efficient optimizers (8-bit Adam)"
    ],
    "hardware": [
        "Use compatible GPU memory",
        "Enable memory-efficient parallelism",
        "Monitor GPU utilization",
        "Use memory mapping for large datasets"
    ]
}
```

### Speed Optimization

```python
# For faster training
speed_optimizations = {
    "data_preprocessing": {
        "use_multiprocessing": True,
        "batch_tokenization": True,
        "cache_encodings": True
    },
    "model_optimization": {
        "compile_model": True,
        "use_flash_attention": True,
        "optimize_memory_layout": True
    },
    "training_loop": {
        "prefetch_data": True,
        "overlap_compute_communication": True,
        "use_efficient_dataloaders": True
    }
}

# Example implementation
from accelerate import Accelerator
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=4,
    dataloader_prefetch_factor=2
)

# Compile model if using PyTorch 2.0+
try:
    model = torch.compile(model)
    print("Model compiled successfully")
except:
    print("Compilation not available, using regular model")
```

### Performance Monitoring

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.metrics = []

    def start(self):
        self.start_time = time.time()

    def log_step(self, step, loss=None, lr=None):
        metrics = {
            'step': step,
            'time': time.time() - self.start_time,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }

        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            metrics.update({
                'gpu_util': gpu.load * 100,
                'gpu_memory': gpu.memoryUsed,
                'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
            })

        if loss is not None:
            metrics['loss'] = loss
        if lr is not None:
            metrics['learning_rate'] = lr

        self.metrics.append(metrics)
        return metrics

    def get_summary(self):
        if not self.metrics:
            return "No metrics available"

        latest = self.metrics[-1]
        return f"""
        Performance Summary:
        - Current step: {latest['step']}
        - Total time: {latest['time']:.2f}s
        - CPU usage: {latest['cpu_percent']:.1f}%
        - Memory usage: {latest['memory_percent']:.1f}%
        - Current loss: {latest.get('loss', 'N/A')}
        - Learning rate: {latest.get('learning_rate', 'N/A')}
        """
```

### Model Selection Guide

```python
model_selection_guide = {
    "small_models": {
        "examples": ["DialoGPT-medium", "GPT-2-medium"],
        "parameters": "100M - 500M",
        "use_case": "Quick prototyping, limited compute",
        "memory": "2-4 GB GPU memory",
        "training_time": "Hours to days"
    },
    "medium_models": {
        "examples": ["Llama-2-7B", "Mistral-7B"],
        "parameters": "7B - 13B",
        "use_case": "General fine-tuning, good performance",
        "memory": "8-16 GB GPU memory (with LoRA)",
        "training_time": "Days to weeks"
    },
    "large_models": {
        "examples": ["Llama-2-70B", "Claude-70B"],
        "parameters": "70B+",
        "use_case": "High-quality fine-tuning, complex tasks",
        "memory": "80+ GB GPU memory (with QLoRA)",
        "training_time": "Weeks to months"
    }
}

def recommend_model(requirements):
    """
    requirements = {
        'max_memory_gb': 8,
        'max_training_time_hours': 24,
        'quality_requirement': 'medium',  # low, medium, high
        'task_complexity': 'simple'  # simple, medium, complex
    }
    """
    if requirements['max_memory_gb'] < 4:
        return "DialoGPT-medium"
    elif requirements['max_memory_gb'] < 16:
        return "Llama-2-7B with LoRA"
    else:
        return "Llama-2-13B with LoRA"
```

## Quick Commands Reference

### Essential Commands

```bash
# Quick model loading
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained('model_name')
config = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'])
model = get_peft_model(model, config)
print(f'Model loaded: {model.num_parameters(only_trainable=True):,} trainable params')
"

# Quick training
python -c "
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load and prepare data
dataset = load_dataset('your_dataset')
tokenized = dataset.map(tokenize_function, batched=True)

# Training arguments
args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    fp16=True,
    evaluation_strategy='epoch'
)

# Train
trainer = Trainer(model=model, args=args, train_dataset=tokenized['train'])
trainer.train()
"
```

### Environment Check

```bash
# Check GPU availability
nvidia-smi

# Check memory
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')"

# Test installation
python -c "
import transformers
import peft
import bitsandbytes
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
print(f'bitsandbytes: {bitsandbytes.__version__}')
"
```

### Model Conversion

```python
# Convert model formats
def convert_and_save_model(model, tokenizer, output_dir):
    # Save LoRA adapter
    model.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # For merged model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{output_dir}/merged")

    print(f"Model saved to {output_dir}")

# Load converted model
def load_model(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained("base_model")
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained("base_model")

    return model, tokenizer
```

This cheatsheet provides quick access to essential LLM fine-tuning commands, configurations, and solutions. Bookmark this page for rapid reference during your fine-tuning projects!
