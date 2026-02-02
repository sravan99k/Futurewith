# LLM Fine-tuning Theory

# LLM Fine-tuning Fundamentals with QLoRA and LoRA

## Table of Contents

1. [Introduction to LLM Fine-tuning](#introduction)
2. [Traditional Fine-tuning vs Parameter-Efficient Methods](#comparison)
3. [LoRA (Low-Rank Adaptation)](#lora)
4. [QLoRA (Quantized Low-Rank Adaptation)](#qlora)
5. [Advanced Fine-tuning Techniques](#advanced-techniques)
6. [Implementation Considerations](#implementation)
7. [Evaluation and Optimization](#evaluation)
8. [Production Deployment](#deployment)

## Introduction to LLM Fine-tuning {#introduction}

### What is Fine-tuning?

Fine-tuning is the process of adapting a pre-trained large language model (LLM) to specific tasks or domains by continuing training on task-specific data. Instead of training from scratch, we leverage the pre-trained knowledge and add specialized capabilities.

### Why Fine-tune?

- **Task Specialization**: Adapt general models for specific domains (legal, medical, coding)
- **Cost Efficiency**: Leverage pre-trained knowledge rather than training from scratch
- **Domain Adaptation**: Incorporate specialized terminology and patterns
- **Performance Improvement**: Better accuracy on target tasks
- **Customization**: Adapt model behavior, style, and output format

### Types of Fine-tuning

1. **Full Fine-tuning**: Update all parameters
2. **Partial Fine-tuning**: Update only certain layers
3. **Parameter-Efficient Fine-tuning (PEFT)**: Add small trainable modules

## Traditional Fine-tuning vs Parameter-Efficient Methods {#comparison}

### Full Fine-tuning

```
Base Model (7B parameters) + Training Data → Updated Model (7B parameters)
```

**Pros:**

- Maximum flexibility and performance
- Can adapt all model capabilities

**Cons:**

- Memory intensive (7B+ parameters × 2 copies)
- High computational cost
- Risk of catastrophic forgetting
- Difficulty in managing multiple adapters

### Parameter-Efficient Fine-tuning (PEFT)

```
Base Model (7B parameters) + Small Adapter (<100M parameters) → Specialized Model
```

**Pros:**

- Memory efficient
- Fast training and inference
- Easy to switch between adapters
- Reduced risk of forgetting

**Cons:**

- May not achieve full fine-tuning performance
- Additional complexity in deployment

## LoRA (Low-Rank Adaptation) {#lora}

### Core Concept

LoRA introduces low-rank matrices to capture important weight updates without modifying the original weights.

### Mathematical Foundation

For a weight matrix W ∈ ℝ^(d×k), instead of updating W directly:

```
W = W₀ + ΔW
ΔW = BA
```

Where:

- W₀: Pre-trained weight matrix
- B ∈ ℝ^(d×r), A ∈ ℝ^(r×k): Low-rank matrices
- r << min(d,k): Rank of the adaptation

### LoRA Implementation

```python
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # Original weight (frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # LoRA matrices (trainable)
        self.A = nn.Parameter(torch.randn(in_features, rank))
        self.B = nn.Parameter(torch.zeros(out_features, rank))

        # Scaling factor
        self.scaling = self.alpha / self.rank

    def forward(self, x):
        # Original forward pass
        original = torch.nn.functional.linear(x, self.weight)

        # LoRA adaptation
        lora = torch.nn.functional.linear(x, self.A)
        lora = torch.nn.functional.linear(lora, self.B)

        return original + self.scaling * lora
```

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

# LoraConfig parameters
lora_config = LoraConfig(
    r=16,                          # Rank
    lora_alpha=16,                 # Alpha parameter
    lora_dropout=0.1,              # Dropout rate
    bias="none",                   # Bias adaptation
    task_type="CAUSAL_LM",         # Task type
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Target layers
)

# Apply LoRA to model
model = get_peft_model(base_model, lora_config)
```

### LoRA Advantages

1. **Memory Efficiency**: Only train ~1% of parameters
2. **No Inference Overhead**: Can be merged or applied dynamically
3. **Modular**: Easy to switch between tasks
4. **Proven Effectiveness**: Competitive with full fine-tuning

## QLoRA (Quantized Low-Rank Adaptation) {#qlora}

### What is QLoRA?

QLoRA combines quantization with LoRA to further reduce memory requirements while maintaining performance.

### Quantization Overview

```python
# 4-bit quantization scheme
# Original: 16-bit (4 bytes per parameter)
# QLoRA: 4-bit (0.5 bytes per parameter)
# Memory reduction: 16x
```

### QLoRA Implementation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training
from bitsandbytes import BitsAndBytesConfig

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4-bit quantization
    bnb_4bit_use_double_quant=True,       # Double quantization
    bnb_4bit_quant_type="nf4",           # Quantization type
    bnb_4bit_compute_dtype=torch.bfloat16 # Compute dtype
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
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

### QLoRA Benefits

1. **Ultra-low Memory**: Train 7B models on single GPU
2. **Maintained Quality**: 4-bit quantization with minimal loss
3. **Fast Training**: Optimized quantization kernels
4. **Easy Deployment**: Single unified model

### Quantization Types

- **NF4**: Optimized for normal distribution
- **FP4**: Floating-point 4-bit
- **Int4**: Integer 4-bit
- **Double Quant**: Further compression

## Advanced Fine-tuning Techniques {#advanced-techniques}

### 1. Prefix Tuning

```python
class PrefixTuning(nn.Module):
    def __init__(self, num_virtual_tokens=20, hidden_size=768):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        # Learnable prefix tokens
        self.prefix_tokens = nn.Parameter(
            torch.randn(num_virtual_tokens, hidden_size)
        )

    def forward(self, input_ids):
        batch_size = input_ids.size(0)
        # Repeat prefix for each sequence in batch
        prefix = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return prefix
```

### 2. P-Tuning v2

```python
# Continuous prompt tuning
config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Classify if this text is about:",
    num_virtual_tokens=20,
    encoder_hidden_size=768
)
```

### 3. Adapter Layers

```python
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_size)
        self.up = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.up(x)
        x = self.dropout(x)
        return x + residual
```

### 4. (IA)³ - Few-Shot Parameter-Efficient Tuning

```python
class IA3Layer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Infusion adapters
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Element-wise multiplication
        return x * self.scale
```

### 5. Multi-Task Learning

```python
# Multiple LoRA adapters for different tasks
task_adapters = {
    "summarization": load_adapter("path/to/summarization"),
    "translation": load_adapter("path/to/translation"),
    "qa": load_adapter("path/to/qa")
}

# Dynamic adapter switching
def apply_task_adapter(model, task_name):
    if current_task != task_name:
        model.set_adapter(task_name)
        current_task = task_name
```

## Implementation Considerations {#implementation}

### 1. Data Preparation

```python
def prepare_fine_tuning_data(dataset, tokenizer, max_length=512):
    def tokenize_function(examples):
        # Tokenize with proper formatting
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        # Add labels for causal LM
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset
```

### 2. Training Configuration

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./llm-fine-tuned",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    fp16=True,  # Mixed precision
    gradient_accumulation_steps=4,  # Effective batch size
    gradient_checkpointing=True,    # Save memory
    dataloader_pin_memory=False,
    report_to="wandb"
)
```

### 3. Memory Optimization

```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Mixed precision
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="fp16")

# Optimizer with 8-bit Adam
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(model.parameters(), lr=2e-5)

# Memory monitoring
def check_memory():
    import psutil
    import torch
    print(f"CPU Memory: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
```

### 4. Distributed Training

```python
# Multi-GPU setup
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=4
)

# Prepare for distributed training
model, optimizer, training_loader = accelerator.prepare(
    model, optimizer, training_loader
)

# Distributed training loop
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

## Evaluation and Optimization {#evaluation}

### 1. Evaluation Metrics

```python
import evaluate

# For text generation
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# For classification
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def evaluate_model(model, tokenizer, eval_dataset):
    model.eval()
    predictions = []
    references = []

    for batch in eval_dataset:
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )

        predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        references.extend(tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))

    # Calculate metrics
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)

    return bleu_score, rouge_score
```

### 2. Hyperparameter Tuning

```python
# Optuna for hyperparameter optimization
import optuna
from transformers import HfArgumentParser

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lora_alpha = trial.suggest_int("lora_alpha", 8, 64)
    rank = trial.suggest_categorical("rank", [8, 16, 32, 64])

    # Train with suggested parameters
    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, config)

    # Training and evaluation
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train()

    # Return validation loss
    eval_results = trainer.evaluate()
    return eval_results["eval_loss"]
```

### 3. Model Merging and Unloading

```python
# Merge LoRA weights with base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")

# Load merged model for inference
merged_model = AutoModelForCausalLM.from_pretrained("./merged-model")
```

### 4. Quantization-Aware Training

```python
# Prepare for QAT
model = prepare_model_for_kbit_training(model)
model = prepare_model_for_quantized_training(model)

# QAT configuration
qat_config = QuantizationConfig(
    is_qat=True,
    weight_only_quant_config=quantization_config
)
```

## Production Deployment {#deployment}

### 1. Model Optimization

```python
# Model compilation for inference
model = model.to_bettertransformer()  # Faster inference

# TensorRT optimization (NVIDIA GPUs)
from tensorrt_llm import LLM, BuildConfig

with trt.Builder() as builder:
    with builder.create_network() as network:
        # Build optimized engine
        pass

    # Save optimized model
    builder.save_engine(engine, "optimized_model.engine")
```

### 2. FastAPI Deployment

```python
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

app = FastAPI()

@app.on_event("startup")
async def load_model():
    global model, tokenizer

    # Load LoRA adapter
    model = AutoModelForCausalLM.from_pretrained("base-model")
    model = PeftModel.from_pretrained(model, "path/to/lora-adapter")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("base-model")
    tokenizer.pad_token = tokenizer.eos_token

@app.post("/generate")
async def generate_text(request: dict):
    prompt = request["prompt"]
    max_length = request.get("max_length", 100)
    temperature = request.get("temperature", 0.7)

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}
```

### 3. Container Deployment

```dockerfile
FROM pytorch/pytorch:latest

# Install dependencies
RUN pip install transformers peft accelerate bitsandbytes fastapi uvicorn

# Copy model and code
COPY ./model /app/model
COPY ./app.py /app/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4. Scaling Considerations

```python
# Load balancing for multiple model instances
from fastapi import BackgroundTasks

class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_weights = [0.3, 0.3, 0.4]  # Load distribution
        self.current_index = 0

    def get_next_model(self):
        # Round-robin load balancing
        model = self.models[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.models)
        return model
```

### 5. Monitoring and Logging

```python
import wandb
import time

class FineTuneMonitor:
    def __init__(self):
        wandb.init(project="llm-fine-tuning")

    def log_training_step(self, step, loss, learning_rate, grad_norm):
        wandb.log({
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm
        })

    def log_eval_results(self, metrics):
        wandb.log(metrics)
```

### 6. A/B Testing

```python
# A/B test different fine-tuned models
def ab_test_request(request):
    variant = hash(request.user_id) % 100  # Consistent user assignment

    if variant < 50:
        model_name = "model-a"  # Original fine-tuned model
    else:
        model_name = "model-b"  # Experimental model

    return select_model(model_name)
```

## Best Practices

### 1. Data Quality

- Use high-quality, diverse training data
- Balance dataset to avoid bias
- Clean and preprocess text properly
- Validate data integrity

### 2. Training Strategy

- Start with small models for experimentation
- Use appropriate learning rates (typically 1e-5 to 1e-4)
- Monitor gradient norms
- Implement early stopping

### 3. Evaluation

- Use task-specific metrics
- Implement comprehensive evaluation datasets
- Monitor for overfitting
- Test on out-of-domain data

### 4. Production

- Optimize for inference speed
- Implement proper error handling
- Monitor model drift
- Plan for model updates

## Common Pitfalls

### 1. Overfitting

**Symptoms**: Training loss decreasing but validation loss increasing
**Solutions**:

- Add regularization
- Reduce learning rate
- Early stopping
- More training data

### 2. Catastrophic Forgetting

**Symptoms**: Model loses pre-trained abilities
**Solutions**:

- Mix pre-training data with fine-tuning data
- Use lower learning rates
- Implement elastic weight consolidation

### 3. Memory Issues

**Solutions**:

- Use gradient checkpointing
- Implement gradient accumulation
- Use mixed precision training
- Consider model parallelism

### 4. Poor Generalization

**Solutions**:

- Increase dataset diversity
- Implement data augmentation
- Use ensembling
- Fine-tune on larger models

## Conclusion

LLM fine-tuning, especially with parameter-efficient methods like LoRA and QLoRA, enables powerful customization of large models for specific tasks. The key is understanding the trade-offs between full fine-tuning and PEFT methods, choosing the right configuration for your use case, and implementing robust training and deployment pipelines.

The field continues to evolve with new techniques like DoRA (Weight-Decomposed Low-Rank Adaptation), AdaLoRA, and other advances that further improve the efficiency and effectiveness of fine-tuning approaches.
