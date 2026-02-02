# LLM Engineering - Quick Reference Cheatsheet

## Table of Contents

1. [Model Selection & Architecture](#model-selection)
2. [Prompt Engineering](#prompt-engineering)
3. [Fine-tuning Techniques](#fine-tuning)
4. [Inference Optimization](#inference)
5. [Evaluation Metrics](#evaluation)
6. [Production Deployment](#deployment)
7. [Cost Optimization](#cost-optimization)
8. [Safety & Guardrails](#safety)

## Model Selection & Architecture {#model-selection}

### Choosing the Right Model

```python
# Model comparison framework
class ModelSelector:
    def __init__(self):
        self.models = {
            'gpt-3.5-turbo': {
                'max_tokens': 4096,
                'cost_per_1k_tokens': {'input': 0.0015, 'output': 0.002},
                'capabilities': ['general', 'code', 'reasoning'],
                'latency': 'low',
                'quality': 'good'
            },
            'gpt-4': {
                'max_tokens': 8192,
                'cost_per_1k_tokens': {'input': 0.03, 'output': 0.06},
                'capabilities': ['general', 'code', 'reasoning', 'analysis'],
                'latency': 'medium',
                'quality': 'excellent'
            },
            'claude-3': {
                'max_tokens': 200000,
                'cost_per_1k_tokens': {'input': 0.008, 'output': 0.024},
                'capabilities': ['general', 'analysis', 'writing', 'coding'],
                'latency': 'medium',
                'quality': 'excellent'
            },
            'llama-2-70b': {
                'max_tokens': 4096,
                'cost_per_1k_tokens': {'input': 0.0007, 'output': 0.0009},
                'capabilities': ['general', 'code'],
                'latency': 'medium',
                'quality': 'good'
            }
        }

    def select_model(self, requirements):
        """Select model based on requirements"""
        suitable_models = []

        for model_name, model_info in self.models.items():
            # Check requirements
            if (requirements.get('max_tokens', 4096) <= model_info['max_tokens'] and
                requirements.get('quality', 'good') in ['good', 'excellent'] and
                all(cap in model_info['capabilities'] for cap in requirements.get('capabilities', []))):

                # Calculate cost estimate
                estimated_cost = self.calculate_cost_estimate(
                    model_info, requirements.get('input_tokens', 1000),
                    requirements.get('output_tokens', 500)
                )

                suitable_models.append({
                    'model': model_name,
                    'info': model_info,
                    'estimated_cost': estimated_cost,
                    'score': self.calculate_model_score(model_info, requirements)
                })

        # Sort by score
        suitable_models.sort(key=lambda x: x['score'], reverse=True)
        return suitable_models

    def calculate_cost_estimate(self, model_info, input_tokens, output_tokens):
        """Calculate estimated cost for usage"""
        input_cost = (input_tokens / 1000) * model_info['cost_per_1k_tokens']['input']
        output_cost = (output_tokens / 1000) * model_info['cost_per_1k_tokens']['output']
        return input_cost + output_cost

    def calculate_model_score(self, model_info, requirements):
        """Calculate model suitability score"""
        score = 0

        # Quality score
        quality_scores = {'good': 7, 'excellent': 10}
        score += quality_scores.get(model_info['quality'], 5)

        # Latency score (inverse - lower latency is better)
        latency_scores = {'low': 10, 'medium': 7, 'high': 4}
        score += latency_scores.get(model_info['latency'], 5)

        # Cost score (inverse - lower cost is better)
        avg_cost = (model_info['cost_per_1k_tokens']['input'] +
                   model_info['cost_per_1k_tokens']['output']) / 2
        score += max(0, 10 - avg_cost * 100)  # Penalize expensive models

        # Capability match score
        required_caps = requirements.get('capabilities', [])
        matched_caps = len([cap for cap in required_caps if cap in model_info['capabilities']])
        score += (matched_caps / max(len(required_caps), 1)) * 5

        return score
```

### Architecture Patterns

```python
# Simple API wrapper
class LLMAPI:
    def __init__(self, model_name, api_key, base_url=None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"

    def generate(self, prompt, max_tokens=150, temperature=0.7, **kwargs):
        """Generate text from prompt"""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        # API call implementation would go here
        return self._make_api_call(payload)

    def chat(self, messages, max_tokens=150, temperature=0.7, **kwargs):
        """Chat completion"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        return self._make_api_call(payload)

    def _make_api_call(self, payload):
        """Make actual API call (implementation depends on provider)"""
        # This would implement the actual HTTP request
        pass

# Multi-model wrapper
class MultiModelWrapper:
    def __init__(self, models_config):
        self.models = {}
        for model_name, config in models_config.items():
            self.models[model_name] = LLMAPI(**config)

    def generate_with_fallback(self, prompt, primary_model, fallback_models, **kwargs):
        """Generate with fallback to other models if primary fails"""
        try:
            return self.models[primary_model].generate(prompt, **kwargs)
        except Exception as e:
            print(f"Primary model {primary_model} failed: {e}")

            for fallback_model in fallback_models:
                try:
                    return self.models[fallback_model].generate(prompt, **kwargs)
                except Exception as fallback_e:
                    print(f"Fallback model {fallback_model} failed: {fallback_e}")
                    continue

            raise Exception("All models failed")

    def route_request(self, request_type, prompt, model_routing):
        """Route request to appropriate model based on content"""
        # Simple routing logic
        if request_type == "code":
            return self.models[model_routing.get('code', 'gpt-4')].generate(prompt)
        elif request_type == "creative":
            return self.models[model_routing.get('creative', 'claude-3')].generate(prompt)
        else:
            return self.models[model_routing.get('general', 'gpt-3.5-turbo')].generate(prompt)
```

## Prompt Engineering {#prompt-engineering}

### Prompt Templates

```python
# Standard prompt templates
class PromptTemplates:
    @staticmethod
    def question_answering(context, question):
        return f"""Context: {context}

Question: {question}

Answer:"""

    @staticmethod
    def code_generation(task_description, language="python"):
        return f"""Task: {task_description}
Language: {language}

Please generate code that accomplishes the task. Include comments explaining the implementation."""

    @staticmethod
    def summarization(text, max_length=150):
        return f"""Text to summarize: {text}

Please provide a concise summary (max {max_length} words):"""

    @staticmethod
    def classification(text, categories):
        categories_str = ", ".join(categories)
        return f"""Text: {text}

Categories: {categories_str}

Classify the text into one of the above categories. Respond with just the category name."""

    @staticmethod
    def extraction(text, fields):
        fields_str = "\\n".join([f"- {field}" for field in fields])
        return f"""Text: {text}

Extract the following information:
{fields_str}

Provide the extracted information in a structured format."""

    @staticmethod
    def conversation_system(system_prompt, user_prompt, history=None):
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_prompt})

        return messages

    @staticmethod
    def chain_of_thought(problem):
        return f"""Problem: {problem}

Let's think step by step:

Step 1:
Step 2:
Step 3:
Final Answer:"""

# Advanced prompt engineering
class AdvancedPrompts:
    @staticmethod
    def few_shot_learning(examples, new_input, labels):
        """Few-shot learning prompt"""
        prompt = "Here are some examples:\\n\\n"

        for example_input, example_label in zip(examples, labels):
            prompt += f"Input: {example_input}\\nOutput: {example_label}\\n\\n"

        prompt += f"Input}\\nOutput:"

        return prompt

    @staticmethod_consistency(problem, num: {new_input
    def self_samples):
        """Self"""
        return f"""Problem-consistency prompting=3: {problem}

Please solve this problem {num_samples} different ways and provide your reasoning for each approach.

Approach 1:
Approach 2:
Approach 3:

Which approach do you think is most reliable and why?"""

    @staticmethod
    def prompt_chaining(task_1, task_2):
        """Chain of prompts"""
        prompt_1 = f"{task_1}\\n\\n"
        prompt_2 = f"Based on the above, {task_2}"

        return [prompt_1, prompt_2]

    @staticmethod
    def structured_output(schema_description, task):
        """Structured output prompting"""
        return f"""Task: {task}

Please provide your response in the following JSON schema:
{schema_description}

Respond with only valid JSON."""

# Prompt optimization
class PromptOptimizer:
    def __init__(self, llm_api):
        self.llm_api = llm_api

    def optimize_prompt(self, task_description, evaluation_criteria):
        """Automatically optimize prompt for a task"""
        optimization_prompt = f"""Task: {task_description}

Evaluation criteria: {evaluation_criteria}

Please provide an optimized prompt that will help an LLM perform this task effectively.
The prompt should be clear, specific, and include relevant instructions.

Optimized prompt:"""

        response = self.llm_api.generate(optimization_prompt, max_tokens=300)
        return response.strip()

    def evaluate_prompt(self, prompt, test_cases, expected_outputs):
        """Evaluate prompt performance on test cases"""
        results = []

        for test_input, expected in zip(test_cases, expected_outputs):
            actual_output = self.llm_api.generate(f"{prompt}\\n\\nInput: {test_input}")

            # Simple evaluation (in practice, you'd use more sophisticated metrics)
            match = self.evaluate_output_quality(actual_output, expected)
            results.append({
                'input': test_input,
                'expected': expected,
                'actual': actual_output,
                'match': match
            })

        accuracy = sum(r['match'] for r in results) / len(results)
        return {
            'accuracy': accuracy,
            'results': results
        }

    def evaluate_output_quality(self, actual, expected):
        """Simple quality evaluation"""
        # This is a simplified evaluation - in practice, you'd use
        # more sophisticated metrics based on the task type
        actual_lower = actual.lower().strip()
        expected_lower = expected.lower().strip()

        # Check for exact match or key terms
        if actual_lower == expected_lower:
            return True

        # Check for key terms
        expected_words = set(expected_lower.split())
        actual_words = set(actual_lower.split())

        if len(expected_words) > 0:
            overlap = len(expected_words.intersection(actual_words))
            return overlap / len(expected_words) > 0.7

        return False
```

## Fine-tuning Techniques {#fine-tuning}

### Data Preparation

```python
import json
import pandas as pd
from typing import List, Dict

class FineTuningDataPrep:
    def __init__(self):
        self.supported_formats = ['openai', 'alpaca', 'chatml', 'custom']

    def prepare_openai_format(self, conversations: List[List[Dict]]):
        """Prepare data in OpenAI fine-tuning format"""
        formatted_data = []

        for conversation in conversations:
            messages = []
            for turn in conversation:
                messages.append({
                    "role": turn['role'],  # 'system', 'user', 'assistant'
                    "content": turn['content']
                })

            formatted_data.append({"messages": messages})

        return formatted_data

    def prepare_alpaca_format(self, instructions: List[str], inputs: List[str], outputs: List[str]):
        """Prepare data in Alpaca format"""
        formatted_data = []

        for instruction, input_text, output in zip(instructions, inputs, outputs):
            if input_text.strip():
                formatted_data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output
                })
            else:
                formatted_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output
                })

        return formatted_data

    def prepare_chatml_format(self, conversations: List[List[Dict]]):
        """Prepare data in ChatML format"""
        formatted_data = []

        for conversation in conversations:
            chatml_conversation = ""

            for turn in conversation:
                role = turn['role']
                content = turn['content']

                if role == 'system':
                    chatml_conversation += f"<|im_start|>system\\n{content}<|im_end|>\\n"
                elif role == 'user':
                    chatml_conversation += f"<|im_start|>user\\n{content}<|im_end|>\\n"
                elif role == 'assistant':
                    chatml_conversation += f"<|im_start|>assistant\\n{content}<|im_end|>\\n"

            formatted_data.append({"text": chatml_conversation})

        return formatted_data

    def augment_data(self, base_data: List[Dict], augmentation_strategies: List[str]):
        """Augment training data"""
        augmented_data = base_data.copy()

        for strategy in augmentation_strategies:
            if strategy == 'paraphrase':
                augmented_data.extend(self.paraphrase_augmentation(base_data))
            elif strategy == 'back_translation':
                augmented_data.extend(self.back_translation_augmentation(base_data))
            elif strategy == 'prompt_variation':
                augmented_data.extend(self.prompt_variation_augmentation(base_data))
            elif strategy == 'noise_addition':
                augmented_data.extend(self.noise_augmentation(base_data))

        return augmented_data

    def paraphrase_augmentation(self, data: List[Dict]):
        """Paraphrase augmentation (simplified)"""
        # In practice, you'd use a paraphrasing model or API
        augmented = []
        for item in data:
            # This is a placeholder - implement actual paraphrasing
            if 'instruction' in item:
                augmented_item = item.copy()
                augmented_item['instruction'] = f"Please help with: {item['instruction']}"
                augmented.append(augmented_item)
        return augmented

    def quality_filter(self, data: List[Dict], criteria: Dict):
        """Filter data based on quality criteria"""
        filtered_data = []

        for item in data:
            if self.meets_quality_criteria(item, criteria):
                filtered_data.append(item)

        return filtered_data

    def meets_quality_criteria(self, item: Dict, criteria: Dict):
        """Check if item meets quality criteria"""
        # Length requirements
        if 'min_length' in criteria:
            content = str(item.get('output', item.get('content', '')))
            if len(content) < criteria['min_length']:
                return False

        if 'max_length' in criteria:
            content = str(item.get('output', item.get('content', '')))
            if len(content) > criteria['max_length']:
                return False

        # Language requirements
        if 'language' in criteria:
            # This would require language detection
            pass

        # Other quality checks...
        return True

    def split_data(self, data: List[Dict], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Split data into train/validation/test sets"""
        import random
        random.shuffle(data)

        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }

    def save_data(self, data: List[Dict], filepath: str, format_type='openai'):
        """Save prepared data to file"""
        if format_type == 'jsonl':
            with open(filepath, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\\n')
        elif format_type == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        print(f"Saved {len(data)} examples to {filepath}")
```

### Fine-tuning Implementation

```python
# OpenAI fine-tuning
class OpenAIFineTuner:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"

    def create_fine_tune(self, training_file, model="gpt-3.5-turbo",
                        hyperparameters=None):
        """Create a fine-tuning job"""
        payload = {
            "training_file": training_file,
            "model": model
        }

        if hyperparameters:
            payload["hyperparameters"] = hyperparameters

        # API call to create fine-tune job
        # response = requests.post(f"{self.base_url}/fine-tunes", ...)

        return payload  # Return job ID in real implementation

    def monitor_fine_tune(self, job_id):
        """Monitor fine-tuning progress"""
        # API call to get job status
        # response = requests.get(f"{self.base_url}/fine-tunes/{job_id}")
        pass

    def list_fine_tunes(self):
        """List all fine-tuning jobs"""
        # API call to list fine-tunes
        pass

    def delete_fine_tune(self, model_id):
        """Delete a fine-tuned model"""
        # API call to delete model
        pass

# Hugging Face fine-tuning
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

class HuggingFaceFineTuner:
    def __init__(self, model_name, tokenizer_name=None):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_dataset(self, data, max_length=512):
        """Prepare dataset for training"""
        def tokenize_function(examples):
            # Combine instruction and input for training
            if 'instruction' in examples:
                texts = [f"### Instruction:\\n{ex.get('instruction', '')}\\n\\n### Response:\\n{ex.get('output', '')}"
                        for ex in examples]
            else:
                texts = examples

            return self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )

        # Convert to dataset format
        from datasets import Dataset
        dataset = Dataset.from_pandas(pd.DataFrame(data))

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def train(self, train_dataset, eval_dataset=None, output_dir="./fine-tuned-model"):
        """Fine-tune the model"""
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            push_to_hub=False,
            report_to=None  # Disable wandb/tensorboard
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        # Train
        trainer.train()

        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        return trainer

    def generate(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """Generate text with fine-tuned model"""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Parameter-Efficient Fine-tuning

```python
# LoRA (Low-Rank Adaptation)
class LoRAFineTuner:
    def __init__(self, base_model_name, lora_config):
        from peft import LoraConfig, get_peft_model, TaskType

        self.base_model_name = base_model_name
        self.lora_config = lora_config

        # Load base model
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.get('rank', 8),
            lora_alpha=lora_config.get('alpha', 32),
            lora_dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules', ['q_proj', 'v_proj'])
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def train(self, train_dataset, eval_dataset=None, **training_args):
        """Train LoRA-adapted model"""
        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(
            output_dir="./lora-finetuned",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            **training_args
        )

        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args
        )

        trainer.train()
        return trainer

    def save_adapters(self, output_dir):
        """Save LoRA adapters"""
        self.model.save_pretrained(output_dir)
        return output_dir

    def load_adapters(self, model_path):
        """Load LoRA adapters for inference"""
        from peft import PeftModel, PeftConfig

        config = PeftConfig.from_pretrained(model_path)
        self.model = PeftModel.from_pretrained(self.model, model_path)

    def generate(self, prompt, **generation_kwargs):
        """Generate with LoRA-adapted model"""
        inputs = self.model.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **generation_kwargs)
        return self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)

# QLoRA (Quantized LoRA)
class QLoRAFineTuner:
    def __init__(self, model_name, load_in_4bit=True, load_in_8bit=False):
        from transformers import BitsAndBytesConfig

        # Quantization configuration
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None

        # Load quantized model
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup_lora(self, lora_config):
        """Setup LoRA on quantized model"""
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.get('rank', 64),
            lora_alpha=lora_config.get('alpha', 16),
            lora_dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ])
        )

        self.model = get_peft_model(self.model, peft_config)
        return self.model

    def train(self, train_dataset, **training_args):
        """Train QLoRA model"""
        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(
            output_dir="./qlora-finetuned",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=20,
            max_steps=1000,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            evaluation_strategy="no",
            ddp_find_unused_parameters=False,
            report_to=None,
            **training_args
        )

        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=self._data_collator
        )

        trainer.train()
        return trainer

    def _data_collator(self, features):
        """Data collator for QLoRA"""
        batch = self.tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        return batch
```

## Inference Optimization {#inference}

### Model Quantization

```python
# Post-training quantization
class ModelQuantizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def quantize_to_int8(self, calibration_dataset=None):
        """Quantize model to INT8"""
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from optimum.onnxruntime import ORTModelForCausalLM
        from optimum.onnxruntime.configuration import QuantizationConfig

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="float16",
            load_in_8bit=True
        )

        return self.model

    def quantize_to_int4(self):
        """Quantize model to INT4"""
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        # Load model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        return self.model

    def benchmark_model(self, prompts, batch_sizes=[1, 2, 4, 8]):
        """Benchmark model performance"""
        import time

        results = {}

        for batch_size in batch_sizes:
            batch_prompts = prompts[:batch_size]

            start_time = time.time()

            # Generate for batch
            for prompt in batch_prompts:
                # Your generation code here
                pass

            end_time = time.time()
            total_time = end_time - start_time

            results[batch_size] = {
                'total_time': total_time,
                'time_per_prompt': total_time / batch_size,
                'tokens_per_second': self.calculate_tokens_per_second(total_time)
            }

        return results

    def calculate_tokens_per_second(self, time_taken):
        """Calculate tokens per second (simplified)"""
        # This would depend on your specific model and generation parameters
        estimated_tokens = 100  # Placeholder
        return estimated_tokens / time_taken

# ONNX Runtime optimization
class ONNXOptimizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.onnx_model = None

    def convert_to_onnx(self, output_path):
        """Convert model to ONNX format"""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Export to ONNX
        dummy_input = tokenizer("Hello, world!", return_tensors="pt")

        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size", 1: "sequence"}
            }
        )

        return output_path

    def optimize_onnx(self, onnx_path, optimized_path):
        """Optimize ONNX model"""
        from onnxruntime.tools.optimizer import optimize_model

        # Optimize model
        optimized_model = optimize_model(onnx_path, opt_level=99)

        # Save optimized model
        optimized_model.save(optimized_path)

        return optimized_path

    def create_inference_session(self, model_path, providers=['CPUExecutionProvider']):
        """Create optimized inference session"""
        import onnxruntime as ort

        session = ort.InferenceSession(model_path, providers=providers)
        return session

    def run_inference(self, session, input_ids, attention_mask):
        """Run inference with ONNX model"""
        outputs = session.run(None, {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })

        return outputs

# TensorRT optimization for NVIDIA GPUs
class TensorRTOptimizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.trt_engine = None

    def convert_to_tensorrt(self, output_path, max_seq_length=512, max_batch_size=8):
        """Convert model to TensorRT engine"""
        try:
            import tensorrt as trt
            import torch_tensorrt

            # Enable TensorRT compilation
            compiled_model = torch_tensorrt.compile(
                self.model_path,
                inputs=([
                    torch_tensorrt.Input(
                        shape=[max_batch_size, max_seq_length],
                        dtype=torch.int32
                    )
                ]),
                enabled_precisions={torch.float, torch.half},
                workspace_size=1 << 22,
                max_timing_cache_size=100,
                use_fp16=True
            )

            # Save compiled model
            torch.jit.save(compiled_model, output_path)

            return compiled_model

        except ImportError:
            print("TensorRT not available. Install with: pip install tensorrt")
            return None

    def run_tensorrt_inference(self, compiled_model, input_ids):
        """Run inference with TensorRT model"""
        with torch.no_grad():
            outputs = compiled_model(input_ids)
        return outputs
```

### Caching and Optimization

```python
import hashlib
import pickle
from functools import lru_cache
from typing import Any, Dict, Optional

class InferenceCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
        self.hits = 0
        self.misses = 0

    def _hash_prompt(self, prompt: str, params: Dict) -> str:
        """Create hash for prompt and parameters"""
        content = f"{prompt}:{sorted(params.items())}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, prompt: str, params: Dict) -> Optional[Any]:
        """Get cached result"""
        cache_key = self._hash_prompt(prompt, params)

        if cache_key in self.cache:
            self.hits += 1
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            return self.cache[cache_key]
        else:
            self.misses += 1
            return None

    def put(self, prompt: str, params: Dict, result: Any):
        """Cache result"""
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        cache_key = self._hash_prompt(prompt, params)
        self.cache[cache_key] = result
        self.access_count[cache_key] = 0

    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_count:
            return

        lru_key = min(self.access_count, key=self.access_count.get)
        del self.cache[lru_key]
        del self.access_count[lru_key]

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }

class OptimizedLLMClient:
    def __init__(self, model_config: Dict, optimization_config: Dict):
        self.model_config = model_config
        self.optimization_config = optimization_config

        # Initialize optimization features
        self.cache = InferenceCache(max_size=optimization_config.get('cache_size', 1000))
        self.batch_processor = BatchProcessor(
            max_batch_size=optimization_config.get('max_batch_size', 8),
            batch_timeout=optimization_config.get('batch_timeout', 0.1)
        )

        # Initialize model
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize optimized model"""
        # This would initialize your chosen model with optimizations
        # based on the model_config and optimization_config
        pass

    def generate(self, prompt: str, **params) -> str:
        """Generate with optimizations"""
        # Check cache first
        cached_result = self.cache.get(prompt, params)
        if cached_result is not None:
            return cached_result

        # Use batch processing for efficiency
        result = self.batch_processor.add_request(prompt, params)

        # Cache the result
        self.cache.put(prompt, params, result)

        return result

    def batch_generate(self, prompts: List[str], **params) -> List[str]:
        """Batch generation for better throughput"""
        return self.batch_processor.add_batch(prompts, params)

    def get_optimization_stats(self) -> Dict:
        """Get optimization statistics"""
        return {
            'cache_stats': self.cache.get_stats(),
            'batch_stats': self.batch_processor.get_stats(),
            'model_config': self.model_config,
            'optimization_config': self.optimization_config
        }

class BatchProcessor:
    def __init__(self, max_batch_size=8, batch_timeout=0.1):
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests = []
        self.processed_requests = 0

    def add_request(self, prompt: str, params: Dict) -> str:
        """Add single request (may be processed in batch)"""
        # For simplicity, process immediately
        # In practice, you'd implement batching logic
        return self._process_single_request(prompt, params)

    def add_batch(self, prompts: List[str], params: Dict) -> List[str]:
        """Add batch of requests"""
        results = []

        for prompt in prompts:
            result = self._process_single_request(prompt, params)
            results.append(result)

        self.processed_requests += len(prompts)
        return results

    def _process_single_request(self, prompt: str, params: Dict) -> str:
        """Process single request"""
        # This would call your LLM API
        # For now, return a placeholder
        return f"Generated response for: {prompt[:50]}..."

    def get_stats(self) -> Dict:
        """Get batch processing statistics"""
        return {
            'processed_requests': self.processed_requests,
            'pending_requests': len(self.pending_requests),
            'max_batch_size': self.max_batch_size,
            'batch_timeout': self.batch_timeout
        }
```

This comprehensive LLM engineering cheatsheet covers all essential aspects of working with large language models, from model selection to production deployment. Each section provides practical code examples and implementation details that you can use directly in your projects.
