# LLM Fine-tuning Techniques: QLoRA and Advanced Methods

## Table of Contents

1. [Introduction to LLM Fine-tuning](#introduction-to-llm-fine-tuning)
2. [Traditional Fine-tuning Approaches](#traditional-fine-tuning-approaches)
3. [Parameter-Efficient Fine-tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
4. [QLoRA: Quantized Low-Rank Adaptation](#qlora-quantized-low-rank-adaptation)
5. [Implementation Guide](#implementation-guide)
6. [Advanced Fine-tuning Techniques](#advanced-fine-tuning-techniques)
7. [Evaluation and Optimization](#evaluation-and-optimization)
8. [Production Considerations](#production-considerations)
9. [Best Practices and Common Pitfalls](#best-practices-and-common-pitfalls)
10. [Future Directions](#future-directions)

## Introduction to LLM Fine-tuning

Fine-tuning large language models has become essential for adapting pre-trained models to specific tasks, domains, or applications. However, traditional fine-tuning approaches face significant challenges with modern large language models due to their massive parameter counts and computational requirements.

### The Fine-tuning Challenge

Modern LLMs like GPT-3, LLaMA, and PaLM contain billions to trillions of parameters, making full fine-tuning:

- **Computationally Expensive**: Requiring substantial GPU memory and compute resources
- **Costly**: With training costs potentially reaching millions of dollars
- **Storage Intensive**: Requiring separate model weights for each fine-tuned version
- **Time-consuming**: Taking days to weeks for complete fine-tuning cycles

### Why Fine-tuning Matters

Despite these challenges, fine-tuning remains crucial because:

1. **Domain Adaptation**: Tailoring models to specific domains (legal, medical, financial)
2. **Task Specialization**: Optimizing for specific tasks (summarization, translation, coding)
3. **Style Alignment**: Matching desired output styles and formats
4. **Knowledge Updates**: Incorporating new information or changing model behavior
5. **Efficiency Gains**: Achieving better performance with less inference compute

## Traditional Fine-tuning Approaches

### 1. Full Fine-tuning

Full fine-tuning involves updating all model parameters during training:

```python
# Traditional full fine-tuning approach
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Enable all parameters for training
for param in model.parameters():
    param.requires_grad = True

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = tokenizer(batch['text'], return_tensors="pt")
        outputs = model(**inputs, labels=inputs["input_ids"])

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Advantages:**

- Maximum flexibility and performance potential
- Can learn complex domain-specific patterns
- No additional inference overhead

**Disadvantages:**

- Requires massive compute resources
- High memory requirements (model + optimizer states)
- Risk of catastrophic forgetting
- Expensive to maintain multiple fine-tuned versions

### 2. Feature-based Fine-tuning

Using the pre-trained model as a feature extractor:

```python
# Feature-based approach
class FeatureExtractor:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_features(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            features = self.model(**inputs).last_hidden_state
        return features.mean(dim=1)  # Pool to get sentence embeddings

# Train a lightweight classifier on extracted features
class TaskClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

feature_extractor = FeatureExtractor("meta-llama/Llama-2-7b-hf")
classifier = TaskClassifier(4096, num_classes)

# Only train the classifier
optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
```

**Advantages:**

- Very efficient computationally
- Lower memory requirements
- Stable training process

**Disadvantages:**

- Limited to tasks compatible with feature extraction
- Cannot capture complex generative patterns
- May underperform on tasks requiring deep model understanding

## Parameter-Efficient Fine-tuning (PEFT)

Parameter-Efficient Fine-tuning techniques aim to achieve comparable performance to full fine-tuning while training only a small subset of parameters.

### 1. Additive Methods

#### Adapter Layers

```python
# Adapter layer implementation
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()

        # Initialize with small random values
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return x + residual

# Integration with transformer layers
class TransformerBlockWithAdapters(nn.Module):
    def __init__(self, hidden_size, num_heads, adapter_size=64):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Add adapter layers
        self.adapter1 = AdapterLayer(hidden_size, adapter_size)
        self.adapter2 = AdapterLayer(hidden_size, adapter_size)

    def forward(self, x):
        # Self-attention with adapter
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.adapter1(x)

        # Feed-forward with adapter
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        x = self.adapter2(x)

        return x
```

#### LoRA (Low-Rank Adaptation)

```python
# LoRA implementation
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # Original weight (frozen)
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # LoRA parameters (trainable)
        self.lora_A = nn.Parameter(torch.empty((rank, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_features, rank)))

        # Scaling factor
        self.scaling = alpha / rank

        # Freeze original weight
        self.weight.requires_grad = False

        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Compute LoRA adaptation
        lora_out = (self.lora_A @ x.T).T  # (batch_size, rank) @ (rank, in_features) -> (batch_size, in_features)
        lora_out = self.lora_B @ lora_out.T  # (out_features, rank) @ (rank, batch_size) -> (out_features, batch_size)
        lora_out = lora_out.T * self.scaling  # (batch_size, out_features)

        # Original output
        original_out = F.linear(x, self.weight)

        return original_out + lora_out

# Integration with linear layers
class LinearWithLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=16, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora = LoRALayer(in_features, out_features, rank, alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x) + self.lora(x)
```

### 2. Selective Methods

#### BitFit (Bias Fine-tuning)

```python
# BitFit implementation - only fine-tune bias terms
class BitFitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Freeze all parameters except bias
        for name, param in self.linear.named_parameters():
            if name != 'bias':
                param.requires_grad = False

    def forward(self, x):
        return self.linear(x)

# Apply to entire model
def apply_bitfit(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            bitfit_module = BitFitLinear(
                module.in_features,
                module.out_features,
                module.bias is not None
            )
            bitfit_module.linear.weight.data = module.weight.data
            if module.bias is not None:
                bitfit_module.linear.bias.data = module.bias.data
            yield bitfit_module
```

## QLoRA: Quantized Low-Rank Adaptation

QLoRA combines quantization with LoRA to enable fine-tuning of 4-bit quantized models with minimal memory usage and performance degradation.

### 1. 4-Bit Quantization Theory

QLoRA uses a novel quantization scheme that preserves the information needed for gradient-based optimization while dramatically reducing memory usage:

```python
# 4-bit quantization implementation
class NF4Quantizer:
    """NF4 (Non-negative 4-bit) quantization for LLM weights"""

    def __init__(self):
        # NF4 quantization levels (symmetric quantization)
        self.quant_levels = torch.tensor([
            -1.0, -0.6961928006362911, -0.5250730514526367,
            -0.3949174800402749, -0.2844413816923499, -0.18477343072891235,
            -0.09105001825162574, 0.0, 0.07958029955755234,
            0.1609302014110025, 0.2581529417037961, 0.3706073951721191,
            0.47999063164711, 0.5964397938089673, 0.7229568967819214,
            0.8600694537162774, 1.0
        ])

    def quantize(self, weight):
        """Quantize weight tensor to 4-bit NF4"""
        # Normalize weight to [-1, 1] range
        weight_norm = weight / torch.max(torch.abs(weight))

        # Quantize to nearest NF4 level
        quantized = torch.zeros_like(weight)
        for i, level in enumerate(self.quant_levels):
            mask = (weight_norm >= level).float()
            if i > 0:
                mask *= (weight_norm < self.quant_levels[i-1]).float()
            quantized += mask * level

        # Store quantized values and scales for dequantization
        max_val = torch.max(torch.abs(weight))
        scales = max_val / self.quant_levels[-1]  # Scale factor

        return quantized, scales

    def dequantize(self, quantized, scales):
        """Dequantize back to full precision"""
        return quantized * scales

class QLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=16, quantize=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.quantize = quantize

        # Base weight (4-bit quantized)
        self.base_weight = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        # LoRA parameters (full precision)
        self.lora_A = nn.Parameter(torch.empty((rank, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_features, rank)))

        # Quantization parameters
        if quantize:
            self.quantizer = NF4Quantizer()
            self.quantized_weight, self.scales = self.quantizer.quantize(self.base_weight.data)

        self.scaling = alpha / rank

        # Freeze base weight
        if quantize:
            self.base_weight.requires_grad = False

        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Dequantize weight for computation
        if self.quantize:
            weight = self.quantizer.dequantize(self.quantized_weight, self.scales)
        else:
            weight = self.base_weight

        # Compute LoRA adaptation
        lora_out = F.linear(x, self.lora_A.T)
        lora_out = F.linear(lora_out, self.lora_B.T) * self.scaling

        # Original output
        original_out = F.linear(x, weight)

        return original_out + lora_out
```

### 2. Double Quantization

QLoRA employs a sophisticated double quantization scheme to further reduce memory usage:

```python
# Double quantization implementation
class DoubleQuantizer:
    def __init__(self, nbits=4, group_size=64):
        self.nbits = nbits
        self.group_size = group_size

        # Quantization codebook for the specified bit width
        if nbits == 4:
            self.codebook = torch.tensor([
                -1.0, -0.6961928006362911, -0.5250730514526367,
                -0.3949174800402749, -0.2844413816923499, -0.18477343072891235,
                -0.09105001825162574, 0.0, 0.07958029955755234,
                0.1609302014110025, 0.2581529417037961, 0.3706073951721191,
                0.47999063164711, 0.5964397938089673, 0.7229568967819214,
                0.8600694537162774, 1.0
            ])

    def quantize_weight(self, weight):
        """Quantize weight with double quantization"""
        batch_size, in_features = weight.shape

        # Reshape for group quantization
        weight_groups = weight.view(-1, self.group_size)

        quantized_weights = []
        scales = []

        for group in weight_groups:
            # First level quantization
            abs_max = torch.max(torch.abs(group))
            group_normalized = group / abs_max

            # Second level quantization to codebook
            distances = torch.abs(group_normalized.unsqueeze(1) - self.codebook.unsqueeze(0))
            quantized_indices = torch.argmin(distances, dim=1)
            quantized_values = self.codebook[quantized_indices]

            quantized_weights.append(quantized_values)
            scales.append(abs_max)

        return torch.stack(quantized_weights), torch.stack(scales)

    def dequantize_weight(self, quantized_weights, scales):
        """Dequantize weight back to full precision"""
        return quantized_weights * scales.unsqueeze(1)
```

### 3. Memory Optimization Strategy

```python
# Complete QLoRA implementation with memory optimization
class QLoRAModel:
    def __init__(self, base_model, rank=16, alpha=16, dropout=0.1):
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Replace linear layers with QLoRA layers
        self._replace_layers()

        # Enable gradient computation for LoRA parameters only
        self._setup_training_parameters()

    def _replace_layers(self):
        """Replace linear layers with QLoRA layers"""
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear) and 'lm_head' not in name:
                # Create QLoRA replacement
                qlora_layer = QLoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=self.rank,
                    alpha=self.alpha,
                    quantize=True
                )

                # Copy weight data
                qlora_layer.base_weight.data = module.weight.data.clone()

                # Replace in model
                parent = self._get_parent_module(name)
                child_name = name.split('.')[-1]
                setattr(parent, child_name, qlora_layer)

    def _get_parent_module(self, module_path):
        """Get parent module for replacement"""
        path_parts = module_path.split('.')
        parent = self.base_model

        for part in path_parts[:-1]:
            parent = getattr(parent, part)

        return parent

    def _setup_training_parameters(self):
        """Setup parameters for training"""
        # Only LoRA parameters require gradients
        for name, param in self.base_model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_memory_footprint(self):
        """Calculate memory footprint of the model"""
        total_params = 0
        trainable_params = 0

        for name, param in self.base_model.named_parameters():
            param_size = param.numel() * param.element_size()
            total_params += param_size

            if param.requires_grad:
                trainable_params += param_size

        print(f"Total model size: {total_params / (1024**3):.2f} GB")
        print(f"Trainable parameters: {trainable_params / (1024**3):.2f} GB")
        print(f"Memory savings: {(total_params - trainable_params) / total_params * 100:.1f}%")

        return total_params, trainable_params

# Usage example
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
qlora_model = QLoRAModel(base_model, rank=16, alpha=16)

# Check memory usage
total_size, trainable_size = qlora_model.get_memory_footprint()
```

## Implementation Guide

### 1. Environment Setup

```bash
# Install required packages
pip install torch torchvision torchaudio
pip install transformers datasets accelerate bitsandbytes
pip install peft wandb tqdm

# For GPU training
pip install flash-attn --no-build-isolation
```

### 2. Data Preparation

```python
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

def prepare_dataset(dataset_path, tokenizer, max_length=512):
    """Prepare dataset for fine-tuning"""

    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Set labels for causal language modeling
        tokenized['labels'] = tokenized['input_ids'].clone()

        return tokenized

    # Load and process dataset
    dataset = load_dataset('json', data_files=dataset_path)

    # Split into train/validation
    split_dataset = dataset['train'].train_test_split(test_size=0.1)

    # Tokenize
    tokenized_train = split_dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    tokenized_val = split_dataset['test'].map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    return tokenized_train, tokenized_val

# Example usage
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

train_dataset, val_dataset = prepare_dataset("your_dataset.json", tokenizer)
```

### 3. QLoRA Configuration

```python
# QLoRA configuration
qlora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # Rank
    lora_alpha=16,  # LoRA scaling parameter
    lora_dropout=0.1,  # Dropout rate
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)

# Load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Add LoRA adapters
model = get_peft_model(model, qlora_config)
model.print_trainable_parameters()
```

### 4. Training Setup

```python
# Training arguments optimized for QLoRA
training_args = TrainingArguments(
    output_dir="./qlora-llama2",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=1000,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    evaluation_strategy="steps",
    eval_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",
    run_name="qlora-llama2-experiment",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    group_by_length=True,
    length_column_name="length"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
```

### 5. Model Saving and Loading

```python
# Save the fine-tuned model
trainer.save_model("./qlora-llama2-finetuned")

# Save tokenizer
tokenizer.save_pretrained("./qlora-llama2-finetuned")

# Load for inference
from peft import PeftModel, PeftConfig

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./qlora-llama2-finetuned")

# Merge LoRA weights with base model for faster inference
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./qlora-llama2-merged")
```

## Advanced Fine-tuning Techniques

### 1. Instruction Fine-tuning

```python
def format_instruction_dataset(examples, tokenizer):
    """Format dataset for instruction following"""
    formatted_examples = []

    for example in examples:
        conversation = [
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": example['response']}
        ]

        # Format using chat template
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )

        formatted_examples.append(formatted_text)

    return formatted_examples

# Apply instruction formatting
train_formatted = format_instruction_dataset(train_dataset, tokenizer)
val_formatted = format_instruction_dataset(val_dataset, tokenizer)

# Create new datasets
from datasets import Dataset

train_instruct = Dataset.from_dict({"text": train_formatted})
val_instruct = Dataset.from_dict({"text": val_formatted})

# Tokenize instruction data
def tokenize_instruction_data(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )

train_tokenized = train_instruct.map(tokenize_instruction_data, batched=True)
val_tokenized = val_instruct.map(tokenize_instruction_data, batched=True)
```

### 2. Multi-task Fine-tuning

```python
class MultiTaskDataset:
    def __init__(self, datasets_dict, tokenizer, max_length=512):
        self.datasets = datasets_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_names = list(datasets_dict.keys())

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets.values())

    def __getitem__(self, idx):
        # Determine which dataset this index belongs to
        cumulative_sizes = []
        total_size = 0

        for task_name, dataset in self.datasets.items():
            cumulative_sizes.append((total_size, total_size + len(dataset), task_name))
            total_size += len(dataset)

        # Find appropriate dataset
        for start, end, task_name in cumulative_sizes:
            if start <= idx < end:
                example_idx = idx - start
                example = self.datasets[task_name][example_idx]

                # Add task-specific formatting
                if task_name == 'summarization':
                    text = f"Summarize the following text: {example['text']}"
                elif task_name == 'translation':
                    text = f"Translate to English: {example['source']}"
                elif task_name == 'question_answering':
                    text = f"Question: {example['question']} Context: {example['context']}"

                # Tokenize
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )

                inputs['task'] = task_name
                return inputs

        raise IndexError("Index out of range")

# Create multi-task dataset
multitask_data = MultiTaskDataset({
    'summarization': summarization_dataset,
    'translation': translation_dataset,
    'question_answering': qa_dataset
}, tokenizer)
```

### 3. Gradient Accumulation and Mixed Precision

```python
# Advanced training configuration
class AdvancedTrainingConfig:
    def __init__(self):
        # Mixed precision training
        self.fp16 = True
        self.bf16 = False

        # Gradient accumulation
        self.gradient_accumulation_steps = 4

        # Gradient clipping
        self.max_grad_norm = 1.0

        # Learning rate scheduling
        self.warmup_steps = 500
        self.warmup_ratio = 0.1

        # Optimizer settings
        self.optim = "adamw_bnb_8bit"
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.weight_decay = 0.01

        # Memory optimization
        self.dataloader_pin_memory = False
        self.dataloader_num_workers = 4

        # Checkpointing
        self.save_strategy = "steps"
        self.save_steps = 500
        self.save_total_limit = 5

        # Evaluation
        self.evaluation_strategy = "steps"
        self.eval_steps = 500

# Custom trainer with advanced features
class AdvancedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_scaler = torch.cuda.amp.GradScaler()

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
            loss = outputs.loss

        # Scale loss for gradient accumulation
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # Backward pass with gradient scaling
        self.grad_scaler.scale(loss).backward()

        return loss

    def on_train_epoch_end(self):
        # Custom learning rate scheduling
        current_lr = self.optimizer.param_groups[0]['lr']
        self.log({'learning_rate': current_lr})

        # Memory cleanup
        torch.cuda.empty_cache()
```

### 4. Distributed Training

```python
# Distributed training setup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def setup_distributed_training():
    """Setup distributed training environment"""

    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank

def create_distributed_model(model, local_rank):
    """Create distributed model"""
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

    # Wrap with DistributedDataParallel
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

    return model

# Distributed training script
def main():
    local_rank = setup_distributed_training()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Apply QLoRA
    model = get_peft_model(model, qlora_config)
    model = create_distributed_model(model, local_rank)

    # Create distributed sampler for data loading
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        shuffle=True,
        seed=42
    )

    # Create dataloader with distributed sampler
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=False
    )

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(local_rank) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
```

## Evaluation and Optimization

### 1. Evaluation Metrics

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu

def evaluate_model(model, tokenizer, test_dataset):
    """Comprehensive evaluation of fine-tuned model"""
    model.eval()
    results = {}

    # Generate predictions
    predictions = []
    references = []

    for example in test_dataset:
        inputs = tokenizer(
            example['input'],
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode predictions
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(example['target'])

    # Compute metrics based on task type
    if test_dataset.task_type == 'classification':
        results['accuracy'] = accuracy_score(references, predictions)
        results['f1'] = f1_score(references, predictions, average='weighted')

    elif test_dataset.task_type == 'summarization':
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        rouge_scores = scorer.score(references, predictions)
        results['rouge1'] = rouge_scores['rouge1'].fmeasure
        results['rouge2'] = rouge_scores['rouge2'].fmeasure
        results['rougeL'] = rouge_scores['rougeL'].fmeasure

    elif test_dataset.task_type == 'translation':
        bleu_scores = []
        for ref, pred in zip(references, predictions):
            ref_tokens = ref.split()
            pred_tokens = pred.split()
            bleu = sentence_bleu([ref_tokens], pred_tokens)
            bleu_scores.append(bleu)

        results['bleu'] = np.mean(bleu_scores)

    # Perplexity calculation
    total_loss = 0
    total_tokens = 0

    for example in test_dataset:
        inputs = tokenizer(
            example['input'],
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            total_loss += outputs.loss.item() * inputs['input_ids'].numel()
            total_tokens += inputs['input_ids'].numel()

    results['perplexity'] = np.exp(total_loss / total_tokens)

    return results

# Automated evaluation pipeline
def run_evaluation_pipeline(model_path, test_dataset):
    """Complete evaluation pipeline"""

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Run evaluation
    results = evaluate_model(model, tokenizer, test_dataset)

    # Log results
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # Save results
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results
```

### 2. Hyperparameter Optimization

```python
import optuna
from optuna.samplers import TPESampler

def objective(trial):
    """Objective function for hyperparameter optimization"""

    # Sample hyperparameters
    rank = trial.suggest_categorical('rank', [8, 16, 32, 64])
    alpha = trial.suggest_categorical('alpha', [8, 16, 32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])

    # Create model with trial hyperparameters
    qlora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = get_peft_model(model, qlora_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./trial_{trial.number}",
        num_train_epochs=2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None  # Disable wandb for hyperparameter search
    )

    # Train model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()

    # Return validation loss as objective to minimize
    return eval_results['eval_loss']

# Run hyperparameter optimization
def run_hyperparameter_optimization(n_trials=20):
    """Run hyperparameter optimization study"""

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print(f"Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    return study.best_params
```

### 3. Model Compression

```python
# Post-training quantization
from transformers import BitsAndBytesConfig
import torch

def quantize_model_for_inference(model_path, quantization_config):
    """Apply quantization to model for efficient inference"""

    # Define quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization_config['quant_type'],  # nf4 or fp4
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    return model

# Knowledge distillation
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=3.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

        # Freeze teacher model
        if teacher_model:
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard loss (student predictions vs ground truth)
        outputs = model(**inputs)
        student_loss = outputs.loss

        # Distillation loss (student vs teacher predictions)
        if self.teacher_model:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)

            # Soft targets distillation
            student_logits = outputs.logits / self.temperature
            teacher_logits = teacher_outputs.logits / self.temperature

            # KL divergence loss
            distillation_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2)

            # Combine losses
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            outputs.loss = total_loss

        return (outputs, outputs) if return_outputs else outputs.loss

# Pruning for model size reduction
def apply_structured_pruning(model, sparsity_ratio=0.3):
    """Apply structured pruning to reduce model size"""

    import torch.nn.utils.prune as prune

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Structured pruning (remove entire neurons)
            prune.l1_unstructured(module, name='weight', amount=sparsity_ratio)

            # Make pruning permanent
            prune.remove(module, 'weight')

    return model

# Model optimization pipeline
def optimize_model_for_deployment(model_path):
    """Complete model optimization pipeline"""

    # 1. Load model
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # 2. Apply quantization
    quantized_model = quantize_model_for_inference(
        model_path,
        {'quant_type': 'nf4'}
    )

    # 3. Apply pruning
    pruned_model = apply_structured_pruning(quantized_model, sparsity_ratio=0.2)

    # 4. Save optimized model
    pruned_model.save_pretrained("./optimized_model")

    # Calculate size reduction
    original_size = os.path.getsize(f"{model_path}/pytorch_model.bin") / (1024**2)
    optimized_size = os.path.getsize("./optimized_model/pytorch_model.bin") / (1024**2)

    print(f"Original model size: {original_size:.2f} MB")
    print(f"Optimized model size: {optimized_size:.2f} MB")
    print(f"Size reduction: {(1 - optimized_size/original_size)*100:.1f}%")

    return pruned_model
```

## Production Considerations

### 1. Model Serving Architecture

```python
# FastAPI model serving with QLoRA
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import uvicorn

app = FastAPI(title="QLoRA Model Server")

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    num_return_sequences: int = 1

class GenerationResponse(BaseModel):
    generated_text: str
    input_tokens: int
    output_tokens: int
    generation_time: float

# Global model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, tokenizer

    print("Loading model...")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, "./qlora-llama2-finetuned")
    tokenizer = AutoTokenizer.from_pretrained("./qlora-llama2-finetuned")

    print("Model loaded successfully!")

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text based on prompt"""

    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import time
        start_time = time.time()

        # Tokenize input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                num_return_sequences=request.num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        generation_time = time.time() - start_time

        return GenerationResponse(
            generated_text=generated_text,
            input_tokens=len(inputs['input_ids'][0]),
            output_tokens=len(outputs[0]) - len(inputs['input_ids'][0]),
            generation_time=generation_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "unhealthy", "model_loaded": False}

    return {"status": "healthy", "model_loaded": True}

@app.get("/model_info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_type": "QLoRA",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "memory_efficiency": f"{(1 - trainable_params/total_params)*100:.1f}% reduction"
    }

if __name__ == "__main__":
    uvicorn.run(
        "qlora_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for GPU model
        reload=False
    )
```

### 2. Batch Processing and Caching

```python
# Advanced batching and caching system
import asyncio
import hashlib
from functools import lru_cache
from typing import Dict, List, Tuple
import redis
import json

class BatchProcessor:
    def __init__(self, model, tokenizer, max_batch_size=8, max_wait_time=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.processing = False

        # Redis for caching (optional)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    @lru_cache(maxsize=1000)
    def get_cache_key(self, prompt: str, max_length: int, temperature: float) -> str:
        """Generate cache key for request"""
        content = f"{prompt}_{max_length}_{temperature}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available"""
        try:
            cached = self.redis_client.get(f"response:{cache_key}")
            return cached.decode() if cached else None
        except:
            return None

    def cache_response(self, cache_key: str, response: str):
        """Cache response with TTL"""
        try:
            self.redis_client.setex(
                f"response:{cache_key}",
                3600,  # 1 hour TTL
                response
            )
        except:
            pass  # Fail silently if caching fails

    async def add_request(self, request: GenerationRequest) -> str:
        """Add request to batch queue"""
        cache_key = self.get_cache_key(
            request.prompt,
            request.max_length,
            request.temperature
        )

        # Check cache first
        cached_response = self.get_cached_response(cache_key)
        if cached_response:
            return cached_response

        # Add to pending requests
        future = asyncio.Future()
        self.pending_requests.append({
            'request': request,
            'future': future,
            'cache_key': cache_key
        })

        # Start processing if not already processing
        if not self.processing:
            asyncio.create_task(self.process_batch())

        # Wait for result
        result = await future
        return result

    async def process_batch(self):
        """Process batch of requests"""
        self.processing = True

        try:
            while self.pending_requests:
                # Wait for batch to fill or timeout
                batch = []
                start_time = asyncio.get_event_loop().time()

                while (len(batch) < self.max_batch_size and
                       asyncio.get_event_loop().time() - start_time < self.max_wait_time and
                       self.pending_requests):
                    if self.pending_requests:
                        batch.append(self.pending_requests.pop(0))

                if not batch:
                    break

                # Process batch
                await self._process_batch(batch)

        finally:
            self.processing = False

    async def _process_batch(self, batch: List[Dict]):
        """Process a single batch"""

        # Collect all inputs
        prompts = [item['request'].prompt for item in batch]
        max_lengths = [item['request'].max_length for item in batch]
        temperatures = [item['request'].temperature for item in batch]

        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )

        # Generate for batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max(max_lengths),
                temperature=max(temperatures),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode outputs
        for i, item in enumerate(batch):
            generated_text = self.tokenizer.decode(
                outputs[i],
                skip_special_tokens=True
            )

            # Cache response
            self.cache_response(item['cache_key'], generated_text)

            # Set future result
            item['future'].set_result(generated_text)

# Usage example
batch_processor = BatchProcessor(model, tokenizer)

# Async endpoint
@app.post("/generate_batch")
async def generate_text_batch(request: GenerationRequest):
    result = await batch_processor.add_request(request)
    return {"generated_text": result}
```

### 3. Monitoring and Logging

```python
import logging
import time
from dataclasses import dataclass
from typing import Dict, List
import psutil
import GPUtil

@dataclass
class GenerationMetrics:
    timestamp: float
    prompt_length: int
    output_length: int
    generation_time: float
    tokens_per_second: float
    memory_usage: float
    gpu_utilization: float

class ModelMonitor:
    def __init__(self, log_file="model_metrics.log"):
        self.metrics = []
        self.logger = logging.getLogger("model_monitor")
        self.logger.setLevel(logging.INFO)

        # File handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def record_generation(self, prompt: str, output: str, generation_time: float):
        """Record generation metrics"""

        # Calculate metrics
        prompt_tokens = len(prompt.split())
        output_tokens = len(output.split())
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0

        # System metrics
        memory_usage = psutil.virtual_memory().percent

        try:
            gpu = GPUtil.getGPUs()[0]
            gpu_utilization = gpu.load * 100
        except:
            gpu_utilization = 0

        # Create metrics object
        metrics = GenerationMetrics(
            timestamp=time.time(),
            prompt_length=prompt_tokens,
            output_length=output_tokens,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            memory_usage=memory_usage,
            gpu_utilization=gpu_utilization
        )

        self.metrics.append(metrics)

        # Log metrics
        self.logger.info(f"Generation metrics: {metrics}")

        # Alert on performance issues
        if tokens_per_second < 10:  # Threshold for concern
            self.logger.warning(f"Low generation speed: {tokens_per_second:.2f} tokens/sec")

        if memory_usage > 90:  # High memory usage
            self.logger.warning(f"High memory usage: {memory_usage:.1f}%")

    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if not self.metrics:
            return {"error": "No metrics available"}

        recent_metrics = self.metrics[-100:]  # Last 100 generations

        return {
            "avg_generation_time": sum(m.generation_time for m in recent_metrics) / len(recent_metrics),
            "avg_tokens_per_second": sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics),
            "avg_memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "avg_gpu_utilization": sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics),
            "total_generations": len(self.metrics),
            "recent_generations": len(recent_metrics)
        }

    def export_metrics(self, filename="metrics_export.json"):
        """Export metrics to file"""
        import json

        metrics_data = [
            {
                'timestamp': m.timestamp,
                'prompt_length': m.prompt_length,
                'output_length': m.output_length,
                'generation_time': m.generation_time,
                'tokens_per_second': m.tokens_per_second,
                'memory_usage': m.memory_usage,
                'gpu_utilization': m.gpu_utilization
            }
            for m in self.metrics
        ]

        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)

# Integration with FastAPI
monitor = ModelMonitor()

@app.post("/generate_with_monitoring")
async def generate_with_monitoring(request: GenerationRequest):
    start_time = time.time()

    # Generate text
    inputs = tokenizer(request.prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=request.max_length)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_time = time.time() - start_time

    # Record metrics
    monitor.record_generation(request.prompt, generated_text, generation_time)

    return {
        "generated_text": generated_text,
        "generation_time": generation_time,
        "metrics_recorded": True
    }

@app.get("/performance_summary")
async def get_performance_summary():
    return monitor.get_performance_summary()
```

## Best Practices and Common Pitfalls

### 1. Data Quality and Preparation

```python
# Data quality checks
def validate_training_data(dataset):
    """Comprehensive data validation for fine-tuning"""

    issues = []

    # Check for empty or very short texts
    short_texts = [i for i, example in enumerate(dataset)
                   if len(example['text'].strip()) < 10]
    if short_texts:
        issues.append(f"Found {len(short_texts)} very short texts")

    # Check for extremely long texts
    long_texts = [i for i, example in enumerate(dataset)
                  if len(example['text']) > 32000]  # LLaMA context length
    if long_texts:
        issues.append(f"Found {len(long_texts)} very long texts")

    # Check for duplicate entries
    text_hashes = {}
    for i, example in enumerate(dataset):
        text_hash = hash(example['text'])
        if text_hash in text_hashes:
            issues.append(f"Duplicate text found at indices {text_hashes[text_hash]} and {i}")
        else:
            text_hashes[text_hash] = i

    # Check for encoding issues
    encoding_issues = []
    for i, example in enumerate(dataset):
        try:
            example['text'].encode('utf-8')
        except UnicodeEncodeError:
            encoding_issues.append(i)

    if encoding_issues:
        issues.append(f"Found {len(encoding_issues)} encoding issues")

    # Check label distribution (for classification tasks)
    if 'labels' in dataset.column_names:
        label_counts = {}
        for example in dataset:
            label = example['labels']
            label_counts[label] = label_counts.get(label, 0) + 1

        # Check for class imbalance
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / min_count

        if imbalance_ratio > 10:
            issues.append(f"Severe class imbalance: ratio {imbalance_ratio:.1f}")

    return issues

# Data cleaning pipeline
def clean_training_data(dataset):
    """Clean and preprocess training data"""

    def clean_text(text):
        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove very short responses
        if len(text.strip()) < 3:
            return None

        # Truncate very long texts
        if len(text) > 32000:
            text = text[:32000]

        return text

    # Apply cleaning
    def filter_and_clean(example):
        cleaned_text = clean_text(example['text'])
        if cleaned_text is None:
            return False

        example['text'] = cleaned_text
        return True

    # Filter out invalid examples
    cleaned_dataset = dataset.filter(filter_and_clean)

    # Remove duplicates
    seen_texts = set()
    unique_examples = []

    for example in cleaned_dataset:
        text_hash = hash(example['text'])
        if text_hash not in seen_texts:
            seen_texts.add(text_hash)
            unique_examples.append(example)

    return unique_examples

# Data augmentation techniques
def augment_text_data(text, augmentation_type='paraphrase'):
    """Apply data augmentation techniques"""

    if augmentation_type == 'paraphrase':
        # Simple paraphrase by replacing synonyms
        synonyms = {
            'good': ['excellent', 'great', 'positive'],
            'bad': ['poor', 'negative', 'unfortunate'],
            'big': ['large', 'huge', 'massive'],
            'small': ['tiny', 'little', 'miniature']
        }

        words = text.split()
        augmented_words = []

        for word in words:
            if word.lower() in synonyms and random.random() < 0.3:
                # Replace with synonym
                synonym = random.choice(synonyms[word.lower()])
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)

        return ' '.join(augmented_words)

    elif augmentation_type == 'back_translation':
        # Implement back-translation augmentation
        # (requires external translation services)
        pass

    elif augmentation_type == 'noise_injection':
        # Add small amount of noise
        words = text.split()
        noisy_words = []

        for word in words:
            if random.random() < 0.1:  # 10% chance to modify word
                # Simple character replacement
                chars = list(word)
                if len(chars) > 2:
                    idx = random.randint(1, len(chars) - 2)
                    chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
                noisy_words.append(''.join(chars))
            else:
                noisy_words.append(word)

        return ' '.join(noisy_words)

    return text
```

### 2. Training Stability

```python
# Gradient clipping and monitoring
class GradientMonitor:
    def __init__(self, model, max_norm=1.0):
        self.model = model
        self.max_norm = max_norm
        self.gradient_norms = []

    def check_gradients(self):
        """Monitor gradient norms for training stability"""

        total_norm = 0
        param_count = 0

        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        total_norm = total_norm ** (1. / 2)

        self.gradient_norms.append(total_norm)

        # Check for exploding gradients
        if total_norm > self.max_norm:
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            return True  # Indicates clipping was applied

        return False

    def get_gradient_stats(self):
        """Get gradient statistics"""
        if not self.gradient_norms:
            return {}

        return {
            'mean_gradient_norm': np.mean(self.gradient_norms),
            'max_gradient_norm': np.max(self.gradient_norms),
            'min_gradient_norm': np.min(self.gradient_norms),
            'std_gradient_norm': np.std(self.gradient_norms),
            'clipped_count': sum(1 for norm in self.gradient_norms if norm > self.max_norm)
        }

# Learning rate scheduling
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0

        # Initialize learning rates
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

    def step(self):
        """Update learning rate"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.optimizer.param_groups[0]['initial_lr'] - self.min_lr) * (1 + math.cos(math.pi * progress))

        # Update learning rates
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

# Training stability monitoring
class TrainingStabilityMonitor:
    def __init__(self, window_size=100):
        self.losses = []
        self.window_size = window_size
        self.stability_issues = []

    def update(self, loss):
        """Update loss history"""
        self.losses.append(loss)

        # Keep only recent losses
        if len(self.losses) > self.window_size:
            self.losses = self.losses[-self.window_size:]

    def check_stability(self):
        """Check for training instability"""
        if len(self.losses) < 10:
            return {"status": "insufficient_data"}

        recent_losses = self.losses[-10:]

        # Check for NaN losses
        if any(math.isnan(loss) for loss in recent_losses):
            self.stability_issues.append("NaN loss detected")
            return {"status": "unstable", "issue": "NaN loss"}

        # Check for loss explosion
        if recent_losses[-1] > recent_losses[0] * 10:
            self.stability_issues.append("Loss explosion detected")
            return {"status": "unstable", "issue": "loss explosion"}

        # Check for loss stagnation
        if len(recent_losses) >= 50:
            variance = np.var(recent_losses[-20:])
            if variance < 1e-6:
                self.stability_issues.append("Loss stagnation detected")
                return {"status": "warning", "issue": "loss stagnation"}

        return {"status": "stable"}

    def get_stability_report(self):
        """Get comprehensive stability report"""
        if not self.losses:
            return {"status": "no_data"}

        recent_losses = self.losses[-50:]

        return {
            "current_loss": self.losses[-1],
            "loss_trend": "decreasing" if recent_losses[-1] < recent_losses[0] else "increasing",
            "loss_variance": np.var(recent_losses),
            "loss_mean": np.mean(recent_losses),
            "stability_issues": self.stability_issues[-10:],  # Recent issues
            "recommendations": self._get_recommendations()
        }

    def _get_recommendations(self):
        """Get recommendations for improving training stability"""
        recommendations = []

        if len(self.losses) >= 50:
            variance = np.var(self.losses[-50:])
            if variance > 10:
                recommendations.append("Consider reducing learning rate")

            if variance < 1e-6:
                recommendations.append("Consider increasing learning rate or adding noise")

        if any("explosion" in issue for issue in self.stability_issues):
            recommendations.append("Apply gradient clipping")

        if any("stagnation" in issue for issue in self.stability_issues):
            recommendations.append("Check for dead neurons or poor data quality")

        return recommendations
```

### 3. Common Pitfalls and Solutions

```python
# Common pitfalls and solutions
class FineTuningPitfalls:
    @staticmethod
    def prevent_catastrophic_forgetting(model, tokenizer, original_dataset):
        """Strategies to prevent catastrophic forgetting"""

        # 1. Regular validation on original task
        def validate_original_performance():
            # Sample from original dataset
            original_samples = original_dataset.shuffle().select(range(100))

            correct_predictions = 0
            total_predictions = len(original_samples)

            for sample in original_samples:
                inputs = tokenizer(sample['input'], return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50)

                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Simple check for correct behavior
                if FineTuningPitfalls._check_original_behavior(sample, generated):
                    correct_predictions += 1

            return correct_predictions / total_predictions

        return validate_original_performance

    @staticmethod
    def _check_original_behavior(sample, generated):
        """Check if model maintains original behavior"""
        # Implement domain-specific validation logic
        return True  # Placeholder

    @staticmethod
    def handle_imbalanced_data(dataset):
        """Handle class imbalance in fine-tuning data"""

        from collections import Counter

        # Count class distribution
        if 'labels' not in dataset.column_names:
            return dataset

        label_counts = Counter(example['labels'] for example in dataset)

        # Identify minority and majority classes
        min_class = min(label_counts, key=label_counts.get)
        max_class = max(label_counts, key=label_counts.get)

        imbalance_ratio = label_counts[max_class] / label_counts[min_class]

        if imbalance_ratio > 5:  # Significant imbalance
            print(f"Detected class imbalance: {imbalance_ratio:.1f}:1")

            # Apply oversampling for minority class
            minority_examples = [ex for ex in dataset if ex['labels'] == min_class]
            majority_examples = [ex for ex in dataset if ex['labels'] == max_class]

            # Oversample minority class
            oversampled_minority = minority_examples * (imbalance_ratio // 2)

            # Combine datasets
            balanced_dataset = oversampled_minority + majority_examples

            return balanced_dataset

        return dataset

    @staticmethod
    def optimize_memory_usage(model, batch_size):
        """Optimize memory usage for large models"""

        # Enable gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        # Use mixed precision training
        use_fp16 = torch.cuda.is_available()

        # Adjust batch size based on available memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)

            # Adjust batch size for memory efficiency
            if gpu_memory_gb < 16:  # Less than 16GB
                batch_size = min(batch_size, 2)
            elif gpu_memory_gb < 24:  # Less than 24GB
                batch_size = min(batch_size, 4)

        # Use gradient accumulation to maintain effective batch size
        effective_batch_size = batch_size
        gradient_accumulation_steps = max(1, 8 // batch_size)

        return {
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'use_fp16': use_fp16
        }

    @staticmethod
    def validate_model_outputs(model, tokenizer, validation_dataset):
        """Comprehensive model output validation"""

        validation_results = {
            'coherence_scores': [],
            'length_stats': [],
            'vocabulary_diversity': [],
            'safety_scores': []
        }

        for example in validation_dataset:
            inputs = tokenizer(example['input'], return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Validate output quality
            coherence = FineTuningPitfalls._calculate_coherence(generated_text)
            validation_results['coherence_scores'].append(coherence)

            # Length validation
            output_length = len(generated_text.split())
            validation_results['length_stats'].append(output_length)

            # Vocabulary diversity
            diversity = FineTuningPitfalls._calculate_vocabulary_diversity(generated_text)
            validation_results['vocabulary_diversity'].append(diversity)

            # Safety validation
            safety = FineTuningPitfalls._check_safety(generated_text)
            validation_results['safety_scores'].append(safety)

        return validation_results

    @staticmethod
    def _calculate_coherence(text):
        """Calculate text coherence score"""
        # Simple coherence measure - can be replaced with more sophisticated methods
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.5

        # Check for logical flow (simplified)
        coherence_score = 0.8  # Placeholder
        return coherence_score

    @staticmethod
    def _calculate_vocabulary_diversity(text):
        """Calculate vocabulary diversity using TTR"""
        words = text.lower().split()
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0

    @staticmethod
    def _check_safety(text):
        """Basic safety check for generated content"""
        # Implement safety filtering
        unsafe_keywords = ['harmful', 'dangerous', 'illegal']
        return not any(keyword in text.lower() for keyword in unsafe_keywords)
```

## Future Directions

### 1. Emerging Techniques

The field of parameter-efficient fine-tuning is rapidly evolving. Key trends include:

#### Continual Learning

```python
# Continual learning with QLoRA
class ContinualQLoRA:
    def __init__(self, base_model, tokenizer, memory_size=1000):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.memory_size = memory_size
        self.task_memory = {}
        self.current_task = 0

    def add_task_memory(self, examples):
        """Add examples to episodic memory"""
        if self.current_task not in self.task_memory:
            self.task_memory[self.current_task] = []

        # Add new examples
        self.task_memory[self.current_task].extend(examples)

        # Maintain memory size
        if len(self.task_memory[self.current_task]) > self.memory_size:
            self.task_memory[self.current_task] = self.task_memory[self.current_task][-self.memory_size:]

    def train_on_task(self, task_examples, task_id=None):
        """Train on new task with memory replay"""
        if task_id is not None:
            self.current_task = task_id

        # Add current task to memory
        self.add_task_memory(task_examples)

        # Prepare replay set
        replay_examples = []
        for prev_task_id, examples in self.task_memory.items():
            if prev_task_id != self.current_task:
                # Sample from previous tasks
                replay_examples.extend(random.sample(examples, min(100, len(examples))))

        # Combine current task with replay examples
        training_examples = task_examples + replay_examples

        # Fine-tune with QLoRA
        self._fine_tune(training_examples)

    def _fine_tune(self, examples):
        """Fine-tune model with QLoRA"""
        # Implementation of QLoRA fine-tuning with replay
        pass
```

#### Meta-Learning Integration

```python
# MAML (Model-Agnostic Meta-Learning) with QLoRA
class MAMLQLoRA:
    def __init__(self, base_model, meta_lr=0.01, adaptation_lr=0.001):
        self.base_model = base_model
        self.meta_lr = meta_lr
        self.adaptation_lr = adaptation_lr
        self.meta_parameters = self._initialize_meta_parameters()

    def _initialize_meta_parameters(self):
        """Initialize meta-parameters for each layer"""
        meta_params = {}
        for name, param in self.base_model.named_parameters():
            if 'weight' in name:
                # Meta-parameter for weight adaptation
                meta_params[name] = torch.zeros_like(param)
        return meta_params

    def adapt_to_task(self, support_set, query_set):
        """Adapt model to a new task using support set"""

        # Create task-specific model
        adapted_model = self._clone_model()

        # Perform inner loop adaptation
        for step in range(5):  # Inner loop steps
            # Sample batch from support set
            batch = self._sample_batch(support_set)

            # Compute loss
            loss = self._compute_loss(adapted_model, batch)

            # Compute gradients
            gradients = torch.autograd.grad(loss, adapted_model.parameters())

            # Update model parameters
            for param, grad in zip(adapted_model.parameters(), gradients):
                param.data -= self.adaptation_lr * grad

        # Evaluate on query set
        query_loss = self._evaluate_loss(adapted_model, query_set)

        return adapted_model, query_loss

    def meta_update(self, tasks):
        """Perform meta-update across multiple tasks"""

        meta_losses = []

        for task in tasks:
            support_set, query_set = task

            # Adapt to task
            adapted_model, query_loss = self.adapt_to_task(support_set, query_set)
            meta_losses.append(query_loss)

        # Compute meta-gradient
        meta_loss = torch.stack(meta_losses).mean()

        # Update meta-parameters
        meta_gradients = torch.autograd.grad(
            meta_loss,
            self.meta_parameters.values(),
            create_graph=True
        )

        for param, grad in zip(self.meta_parameters.values(), meta_gradients):
            param.data -= self.meta_lr * grad

        return meta_loss.item()
```

### 2. Hardware-Specific Optimizations

#### Quantization-Aware Training

```python
# Quantization-aware training implementation
class QATAwareQLoRA:
    def __init__(self, model, quantization_config):
        self.model = model
        self.quantization_config = quantization_config
        self.quantized_layers = []
        self._setup_quantization()

    def _setup_quantization(self):
        """Setup quantization for specific layers"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                quantized_layer = QuantizedLinear(
                    module.in_features,
                    module.out_features,
                    self.quantization_config
                )
                self.quantized_layers.append((name, quantized_layer))

    def forward(self, x):
        """Forward pass with quantization awareness"""

        # Apply quantization during training
        for name, module in self.model.named_modules():
            if name in [layer_name for layer_name, _ in self.quantized_layers]:
                # Apply quantization
                quantized_weight = self._simulate_quantization(module.weight)
                x = F.linear(x, quantized_weight)
            else:
                x = module(x)

        return x

    def _simulate_quantization(self, weight):
        """Simulate quantization during training"""
        # Quantization simulation for QAT
        quantized_weight = torch.round(weight / self.quantization_config['scale']) * self.quantization_config['scale']
        return quantized_weight
```

### 3. Automated Fine-tuning

#### Neural Architecture Search for LoRA

```python
# Automated LoRA architecture search
class AutoLoRA:
    def __init__(self, base_model, search_space):
        self.base_model = base_model
        self.search_space = search_space
        self.search_history = []

    def search_architecture(self, validation_dataset, num_trials=20):
        """Search for optimal LoRA architecture"""

        best_config = None
        best_score = float('-inf')

        for trial in range(num_trials):
            # Sample configuration from search space
            config = self._sample_configuration()

            # Create model with this configuration
            model = self._create_model_with_config(config)

            # Evaluate model
            score = self._evaluate_model(model, validation_dataset)

            # Update best configuration
            if score > best_score:
                best_score = score
                best_config = config

            # Log trial
            self.search_history.append({
                'trial': trial,
                'config': config,
                'score': score
            })

            print(f"Trial {trial}: Score = {score:.4f}, Config = {config}")

        return best_config, best_score

    def _sample_configuration(self):
        """Sample configuration from search space"""
        config = {}
        for param, values in self.search_space.items():
            config[param] = random.choice(values)
        return config

    def _create_model_with_config(self, config):
        """Create model with specific LoRA configuration"""

        # Apply LoRA with searched configuration
        qlora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['rank'],
            lora_alpha=config['alpha'],
            lora_dropout=config['dropout'],
            target_modules=config['target_modules']
        )

        model = get_peft_model(self.base_model, qlora_config)
        return model

    def _evaluate_model(self, model, validation_dataset):
        """Evaluate model on validation set"""
        # Implementation of evaluation metric
        return 0.85  # Placeholder score
```

This comprehensive guide covers the essential aspects of LLM fine-tuning with a focus on QLoRA and advanced techniques. The field continues to evolve rapidly, with new methods and optimizations being developed regularly. The key to successful fine-tuning lies in understanding the trade-offs between model performance, computational efficiency, and implementation complexity, then choosing the appropriate techniques for your specific use case and constraints.
