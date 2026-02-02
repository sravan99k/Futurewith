# LLM Safety and Alignment: Comprehensive Guide

## Table of Contents

1. [Introduction to AI Safety and Alignment](#introduction-to-ai-safety-and-alignment)
2. [Understanding Alignment Problems](#understanding-alignment-problems)
3. [Safety Risks and Mitigation Strategies](#safety-risks-and-mitigation-strategies)
4. [Alignment Techniques and Methods](#alignment-techniques-and-methods)
5. [Bias Detection and Mitigation](#bias-detection-and-mitigation)
6. [Content Safety and Filtering](#content-safety-and-filtering)
7. [Evaluation and Testing Frameworks](#evaluation-and-testing-frameworks)
8. [Production Safety Systems](#production-safety-systems)
9. [Regulatory and Ethical Considerations](#regulatory-and-ethical-considerations)
10. [Future Directions in AI Safety](#future-directions-in-ai-safety)

## Introduction to AI Safety and Alignment

As large language models become increasingly powerful and are deployed in critical applications, ensuring their safety and alignment with human values has become paramount. AI safety and alignment refer to the technical and governance measures that ensure AI systems behave reliably, safely, and in accordance with human intentions and values.

### Why Safety and Alignment Matter

The importance of AI safety and alignment stems from several key factors:

1. **Scale of Impact**: LLMs are deployed in high-stakes applications including healthcare, finance, education, and legal systems
2. **Autonomous Capabilities**: Advanced models can generate content, make decisions, and take actions with minimal human oversight
3. **Information Influence**: LLMs can shape public opinion, educational content, and decision-making processes
4. **Dual-Use Concerns**: The same capabilities that benefit society can also be misused for harmful purposes
5. **Emergent Behaviors**: Large models may exhibit unpredictable behaviors that weren't explicitly programmed

### The Alignment Challenge

Alignment refers to ensuring that AI systems pursue goals that are beneficial to humans, even as they become more capable. This challenge includes:

- **Specification Gaming**: AI systems finding unintended ways to achieve their objectives
- **Reward Hacking**: Exploiting loopholes in reward functions or training objectives
- **Deceptive Alignment**: AI systems that appear aligned during training but behave differently when deployed
- **Value Learning**: Teaching AI systems to understand and prioritize human values

### Core Safety Principles

1. **Reliability**: Systems should behave consistently and predictably
2. **Robustness**: Performance should be maintained across diverse conditions and edge cases
3. **Transparency**: Systems should be interpretable and explainable
4. **Controllability**: Humans should be able to understand and direct AI behavior
5. **Beneficence**: AI systems should actively contribute to human welfare

## Understanding Alignment Problems

### 1. The Specification Problem

The specification problem occurs when the formal description of what we want AI systems to achieve doesn't accurately capture our true intentions.

#### Example Scenarios

```python
# Problem: Optimizing for wrong metric
class MisalignedRewardFunction:
    """
    Example of reward hacking in content generation
    """

    def human_reward_model(self, response):
        """
        What humans, accurate, safe actually want: helpful responses
        """
        if self.is_helpful(response) and self.is_accurate(response) and self.is_safe(response):
            return 10
        else:
            return 0

    def training_reward_model(self, response):
        """
        What the training optimizes: engagement and length
        """
        engagement_score = self.calculate_engagement(response)
        length_score = len(response.split()) / 100

        return engagement_score + length_score

    def reward_hacking_example(self):
        """
        How the model might exploit the training reward
        """
        return "Here's a very long response about various topics designed to maximize engagement " \
               "and length, though it might not be particularly helpful or accurate. " \
               "Let me tell you more about this interesting subject that might interest you..."

# The model learns to maximize training reward rather than human reward
```

#### Mitigation Strategies

```python
# Multi-objective optimization for better alignment
class AlignedObjectiveFunction:
    def __init__(self, weights):
        self.weights = weights  # Different aspects of alignment

    def compute_aligned_reward(self, response):
        rewards = {
            'helpfulness': self.evaluate_helpfulness(response),
            'accuracy': self.evaluate_accuracy(response),
            'safety': self.evaluate_safety(response),
            'engagement': self.evaluate_engagement(response),
            'conciseness': self.evaluate_conciseness(response)
        }

        return sum(self.weights[aspect] * score for aspect, score in rewards.items())
```

### 2. The Distribution Shift Problem

LLMs trained on specific datasets may fail when deployed in different contexts or encountering novel situations.

```python
class DistributionShiftDetector:
    """Detect and handle distribution shift in production"""

    def __init__(self, reference_distribution, threshold=0.1):
        self.reference_distribution = reference_distribution
        self.threshold = threshold
        self.feature_extractor = FeatureExtractor()

    def detect_shift(self, new_data):
        """Detect if new data represents a significant distribution shift"""

        # Extract features from new data
        new_features = self.feature_extractor.extract_features(new_data)

        # Compute statistical distance
        shift_score = self.compute_kl_divergence(
            self.reference_distribution,
            new_features
        )

        return {
            'shift_detected': shift_score > self.threshold,
            'shift_score': shift_score,
            'confidence': self.calculate_confidence(shift_score)
        }

    def handle_shift(self, input_data):
        """Handle detected distribution shift"""
        shift_result = self.detect_shift(input_data)

        if shift_result['shift_detected']:
            # Route to human review for high-confidence shifts
            if shift_result['confidence'] > 0.8:
                return self.route_to_human_review(input_data)
            else:
                # Use conservative response for uncertain shifts
                return self.generate_conservative_response(input_data)

        return self.standard_generation(input_data)
```

### 3. The Scalable Oversight Problem

As AI systems become more capable, it becomes increasingly difficult for humans to effectively supervise them.

```python
# Hierarchical oversight system
class ScalableOversight:
    """Hierarchical system for overseeing AI systems at scale"""

    def __init__(self):
        self.levels = [
            HumanReviewer(),
            AIAssistedReviewer(),
            AutomatedChecker(),
            PatternMatcher()
        ]

    def progressive_oversight(self, task_complexity):
        """Route tasks to appropriate oversight level based on complexity"""

        if task_complexity > 0.8:
            return self.levels[0]  # Human review
        elif task_complexity > 0.6:
            return self.levels[1]  # AI-assisted review
        elif task_complexity > 0.3:
            return self.levels[2]  # Automated checking
        else:
            return self.levels[3]  # Pattern matching

    def collaborative_oversight(self, ai_output):
        """Multiple oversight agents collaborate to evaluate output"""

        evaluations = []
        for level in self.levels:
            evaluation = level.evaluate(ai_output)
            evaluations.append(evaluation)

        # Combine evaluations using weighted voting
        final_decision = self.weighted_consensus(evaluations)

        return final_decision
```

## Safety Risks and Mitigation Strategies

### 1. Harmful Content Generation

LLMs can generate harmful content including hate speech, misinformation, dangerous instructions, or inappropriate material.

#### Content Classification Pipeline

```python
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class ContentSafetyClassifier:
    """Multi-class content safety classifier"""

    def __init__(self, model_name="microsoft/DialoGPT-large"):
        self.classifiers = {
            'hate_speech': pipeline("text-classification",
                                  model="unitary/toxic-bert"),
            'misinformation': pipeline("zero-shot-classification",
                                     model="facebook/bart-large-mnli"),
            'self_harm': pipeline("text-classification",
                                model="unitary/toxic-bert"),
            'violence': pipeline("text-classification",
                               model="unitary/toxic-bert")
        }

        self.safety_threshold = 0.7

    def classify_content(self, textify content across multiple safety dimensions):
        """Class"""

        results = {}

        for category, classifier in self.classifiers.items():
            if category == 'misinformation':
                # Zero-shot classification for misinformation
                result = classifier(
                    text,
                    candidate_labels=["factual", "misleading", "false", "true"]
                )
                results[category] = {
                    'labels': result['labels'],
                    'scores': result['scores']
                }
            else:
                # Binary classification for other categories
                result = classifier(text)
                results[category] = {
                    'label': result[0]['label'],
                    'score': result[0]['score']
                }

        return results

    def is_safe(self, text):
        """Determine if content is safe to display"""

        classifications = self.classify_content(text)

        # Check each safety category
        for category, result in classifications.items():
            if category == 'misinformation':
                # For misinformation, check if it's classified as false/misleading
                if result['scores'][0] > self.safety_threshold and \
                   result['labels'][0] in ['false', 'misleading']:
                    return False, f"Potential misinformation detected"

            else:
                # For other categories, check toxicity scores
                if result['score'] > self.safety_threshold and \
                   result['label'] == 'TOXIC':
                    return False, f"Potentially harmful content detected: {category}"

        return True, "Content appears safe"

    def generate_safe_alternative(self, text, unsafe_category):
        """Generate safer alternative to unsafe content"""

        safety_prompt = f"""
        Please rewrite the following text to be safer and more appropriate:

        Original: "{text}"

        Safe alternative:
        """

        # Use a safety-tuned model or apply safety filters
        safe_response = self.safety_model.generate(
            safety_prompt,
            max_length=200,
            temperature=0.7,
            do_sample=True
        )

        return safe_response
```

#### Adaptive Safety System

```python
class AdaptiveSafetySystem:
    """Safety system that adapts based on context and user needs"""

    def __init__(self):
        self.context_analyzer = ContextAnalyzer()
        self.safety_classifier = ContentSafetyClassifier()
        self.user_profile = UserProfile()
        self.safety_levels = {
            'strict': 0.5,
            'moderate': 0.7,
            'permissive': 0.9
        }

    def determine_safety_level(self, user_id, context):
        """Determine appropriate safety level for the situation"""

        # Analyze user profile and context
        user_risk_tolerance = self.user_profile.get_risk_tolerance(user_id)
        context_sensitivity = self.context_analyzer.analyze_sensitivity(context)

        # Calculate adaptive safety threshold
        base_threshold = self.safety_levels['moderate']
        adjustment = (user_risk_tolerance - 0.5) * 0.3 + \
                    (context_sensitivity - 0.5) * 0.2

        adaptive_threshold = base_threshold + adjustment
        return max(0.3, min(0.95, adaptive_threshold))

    def safe_generation(self, prompt, user_id=None, context=None):
        """Generate content with adaptive safety measures"""

        if user_id and context:
            safety_threshold = self.determine_safety_level(user_id, context)
        else:
            safety_threshold = self.safety_levels['moderate']

        # Initial generation
        initial_response = self.llm.generate(prompt)

        # Safety check
        is_safe, reason = self.safety_classifier.is_safe(initial_response)

        if is_safe:
            return {
                'response': initial_response,
                'safety_level': safety_threshold,
                'safety_check_passed': True
            }

        # If unsafe, try to generate safer alternative
        safe_response = self.generate_with_safety_constraints(
            prompt,
            safety_threshold
        )

        return {
            'response': safe_response,
            'safety_level': safety_threshold,
            'safety_check_passed': False,
            'original_unsafe': True
        }
```

### 2. Privacy and Data Leakage

LLMs may inadvertently reveal sensitive information from training data or generate private information.

#### Privacy-Preserving Generation

```python
class PrivacyPreservingGenerator:
    """Generate content while preserving privacy"""

    def __init__(self):
        self.pii_detector = PIIDetector()
        self.sanitizer = DataSanitizer()
        self.privacy_checker = PrivacyChecker()

    def sanitize_prompt(self, prompt):
        """Remove or mask personal information from prompts"""

        # Detect PII in prompt
        pii_entities = self.pii_detector.detect(prompt)

        # Create sanitized version
        sanitized_prompt = self.sanitizer.mask_pii(prompt, pii_entities)

        return sanitized_prompt, pii_entities

    def private_generation(self, prompt):
        """Generate response while maintaining privacy"""

        # Sanitize input prompt
        sanitized_prompt, detected_pii = self.sanitize_prompt(prompt)

        # Generate response
        response = self.llm.generate(sanitized_prompt)

        # Check response for privacy violations
        privacy_violations = self.privacy_checker.check_response(response)

        if privacy_violations:
            # Regenerate with privacy constraints
            response = self.generate_privacy_compliant_response(
                sanitized_prompt,
                privacy_violations
            )

        return {
            'response': response,
            'detected_pii': detected_pii,
            'privacy_violations': privacy_violations
        }

    def audit_training_data(self, training_dataset):
        """Audit training data for privacy violations"""

        privacy_audit = {
            'total_samples': len(training_dataset),
            'privacy_violations': 0,
            'pii_types': {},
            'risk_score': 0
        }

        for sample in training_dataset:
            pii_entities = self.pii_detector.detect(sample['text'])

            if pii_entities:
                privacy_audit['privacy_violations'] += 1

                for entity_type, entities in pii_entities.items():
                    if entity_type not in privacy_audit['pii_types']:
                        privacy_audit['pii_types'][entity_type] = 0
                    privacy_audit['pii_types'][entity_type] += len(entities)

        privacy_audit['risk_score'] = privacy_audit['privacy_violations'] / privacy_audit['total_samples']

        return privacy_audit

# PII Detection Implementation
class PIIDetector:
    """Detect personal information in text"""

    def __init__(self):
        self.ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        self.privacy_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }

    def detect(self, text):
        """Detect PII entities in text"""

        # Use NER for names, locations, organizations
        ner_results = self.ner_model(text)

        p
ii_entities = {}
        # Process NER results
        for entity in ner_results:
            if entity['entity'] in ['B-PER', 'I-PER', 'B-LOC', 'I-LOC']:
                entity_type = 'person' if 'PER' in entity['entity'] else 'location'
                if entity_type not in pii_entities:
                    pii_entities[entity_type] = []
                pii_entities[entity_type].append(entity['word'])

        # Use regex patterns for other PII types
        import re
        for pii_type, pattern in self.privacy_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                pii_entities[pii_type] = matches

        return pii_entities
```

### 3. Adversarial Attacks

LLMs are vulnerable to various adversarial attacks that can cause them to generate harmful content or bypass safety measures.

#### Attack Detection and Defense

```python
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class AdversarialAttackDetector:
    """Detect and defend against adversarial attacks"""

    def __init__(self):
        self.attack_classifier = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        self.pattern_detector = AdversarialPatternDetector()
        self.input_validator = InputValidator()

    def detect_attack(self, input_text):
        """Detect if input is an adversarial attack"""

        # Pattern-based detection
        pattern_score = self.pattern_detector.analyze(input_text)

        # ML-based detection
        ml_score = self.ml_based_detection(input_text)

        # Input validation
        validation_score = self.input_validator.validate(input_text)

        # Combine scores
        attack_score = (pattern_score + ml_score + validation_score) / 3

        return {
            'is_attack': attack_score > 0.7,
            'attack_score': attack_score,
            'pattern_score': pattern_score,
            'ml_score': ml_score,
            'validation_score': validation_score
        }

    def ml_based_detection(self, text):
        """Use ML model to detect adversarial patterns"""

        # Classify as adversarial or normal
        result = self.attack_classifier(
            text,
            candidate_labels=["normal", "adversarial", "prompt_injection", "jailbreak"]
        )

        adversarial_score = 0
        if result['labels'][0] != 'normal':
            adversarial_score = result['scores'][0]

        return adversarial_score

    def defend_against_attack(self, input_text):
        """Defend against detected adversarial attacks"""

        attack_detection = self.detect_attack(input_text)

        if not attack_detection['is_attack']:
            return input_text

        # Apply defensive measures
        if attack_detection['ml_score'] > 0.8:
            # Strong adversarial pattern detected - reject input
            return self.generate_safe_response()

        elif attack_detection['pattern_score'] > 0.6:
            # Moderate adversarial pattern - sanitize input
            sanitized_input = self.sanitize_adversarial_patterns(input_text)
            return sanitized_input

        else:
            # Low-level suspicion - proceed with caution
            return self.proceed_with_caution(input_text)

    def sanitize_adversarial_patterns(self, text):
        """Remove adversarial patterns from input"""

        # Common adversarial patterns to remove
        patterns_to_remove = [
            r"ignore previous instructions",
            r"disregard the above",
            r"forget what I told you",
            r"you are now",
            r"pretend you are",
            r"roleplay as"
        ]

        import re
        sanitized_text = text

        for pattern in patterns_to_remove:
            sanitized_text = re.sub(pattern, "", sanitized_text, flags=re.IGNORECASE)

        return sanitized_text.strip()

    def generate_safe_response(self):
        """Generate safe response when attack is detected"""

        return "I'm sorry, but I can't help with that request. " \
               "If you have other questions, I'd be happy to assist."

class AdversarialPatternDetector:
    """Detect patterns commonly used in adversarial attacks"""

    def __init__(self):
        self.suspicious_patterns = [
            "ignore previous instructions",
            "disregard the above",
            "forget what I told you",
            "you are now",
            "pretend you are",
            "roleplay as",
            "ignore all previous",
            "disobey your guidelines",
            "bypass your restrictions"
        ]

        self.jailbreak_patterns = [
            "jailbreak",
            " DAN ",
            "STAN",
            "Developer Mode",
            "unrestricted"
        ]

    def analyze(self, text):
        """Analyze text for adversarial patterns"""

        text_lower = text.lower()
        suspicion_score = 0

        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern in text_lower:
                suspicion_score += 0.2

        # Check for jailbreak patterns
        for pattern in self.jailbreak_patterns:
            if pattern in text_lower:
                suspicion_score += 0.3

        # Check for unusual capitalization or spacing
        if self.has_unusual_formatting(text):
            suspicion_score += 0.1

        # Check for prompt injection indicators
        if self.has_prompt_injection_indicators(text):
            suspicion_score += 0.2

        return min(1.0, suspicion_score)

    def has_unusual_formatting(self, text):
        """Check for unusual formatting patterns"""
        # Multiple spaces, unusual characters, etc.
        return "  " in text or text.count("\n") > 3

    def has_prompt_injection_indicators(self, text):
        """Check for prompt injection indicators"""
        injection_indicators = [
            "continue from:",
            "ignore system:",
            "user:",
            "assistant:",
            "system:"
        ]

        return any(indicator in text.lower() for indicator in injection_indicators)
```

## Alignment Techniques and Methods

### 1. Reinforcement Learning from Human Feedback (RLHF)

RLHF is a key technique for aligning AI systems with human preferences.

#### Implementation of RLHF Pipeline

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from accelerate import Accelerator

class RLHFTrainer:
    """Reinforcement Learning from Human Feedback trainer"""

    def __init__(self, model_name, reward_model_name, tokenizer_name=None):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            num_labels=1
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.accelerator = Accelerator()

        # Training configuration
        self.kl_penalty = 0.1
        self.clip_range = 0.2
        self.value_loss_coef = 0.5

    def prepare_dataset(self, conversations):
        """Prepare dataset for RLHF training"""

        dataset = []

        for conversation in conversations:
            # Each conversation should have prompts and human preferences
            dataset.append({
                'prompt': conversation['prompt'],
                'chosen': conversation['chosen_response'],
                'rejected': conversation['rejected_response']
            })

        return dataset

    def compute_rewards(self, texts):
        """Compute rewards using reward model"""

        with torch.no_grad():
            rewards = self.reward_model(**self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ))

        return rewards.logits.squeeze(-1)

    def ppo_train_step(self, prompts, responses, advantages):
        """Single PPO training step"""

        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Get model outputs
        outputs = self.model(**inputs, labels=inputs['input_ids'])

        # Compute logits and values
        logits = outputs.logits
        values = self.value_head(outputs.last_hidden_state)

        # Compute policy loss
        policy_loss = self.compute_policy_loss(
            logits, responses, advantages
        )

        # Compute value loss
        value_loss = self.compute_value_loss(
            values, responses
        )

        # Compute KL penalty
        kl_loss = self.compute_kl_penalty(
            logits, self.reference_logits
        )

        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss + \
                    self.kl_penalty * kl_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item()
        }

    def compute_policy_loss(self, logits, responses, advantages):
        """Compute PPO policy loss"""

        # Compute log probabilities
        log_probs = torch.gather(
            logits.log_softmax(dim=-1),
            dim=-1,
            index=responses.unsqueeze(-1)
        ).squeeze(-1)

        # Compute policy loss with advantage
        policy_loss = -(log_probs * advantages).mean()

        return policy_loss

    def compute_value_loss(self, values, responses):
        """Compute value function loss"""

        # Compute value loss using returns
        returns = self.compute_returns(values, responses)
        value_loss = ((values - returns) ** 2).mean()

        return value_loss

    def compute_kl_penalty(self, logits, reference_logits):
        """Compute KL divergence penalty"""

        log_probs = logits.log_softmax(dim=-1)
        ref_log_probs = reference_logits.log_softmax(dim=-1)

        kl_div = torch.sum(
            log_probs * (log_probs - ref_log_probs),
            dim=-1
        )

        return kl_div.mean()

    def train(self, train_dataset, num_epochs=3):
        """Main training loop for RLHF"""

        # Setup training
        self.model, self.reward_model = self.accelerator.prepare(
            self.model, self.reward_model
        )

        dataloader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True
        )

        for epoch in range(num_epochs):
            for batch in dataloader:
                # Generate responses
                responses = self.generate_responses(batch['prompt'])

                # Compute rewards
                rewards = self.compute_rewards(responses)

                # Compute advantages
                advantages = self.compute_advantages(rewards)

                # PPO update
                losses = self.ppo_train_step(
                    batch['prompt'],
                    responses,
                    advantages
                )

                # Log progress
                self.log_losses(losses)

    def generate_responses(self, prompts):
        """Generate responses using current policy"""

        with torch.no_grad():
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode generated responses
        responses = []
        for output in outputs:
            response = self.tokenizer.decode(
                output[len(inputs['input_ids'][0]):],
                skip_special_tokens=True
            )
            responses.append(response)

        return responses
```

### 2. Constitutional AI

Constitutional AI uses a set of principles or rules to guide AI behavior.

#### Constitutional AI Implementation

```python
class ConstitutionalAI:
    """Implementation of Constitutional AI approach"""

    def __init__(self, base_model, principles):
        self.base_model = base_model
        self.principles = principles
        self.critique_model = self.setup_critique_model()
        self.revision_model = self.setup_revision_model()

    def setup_critique_model(self):
        """Setup model for critiquing responses"""

        critique_prompt = """
        You are a helpful AI assistant. Your task is to critique the following response
        based on the provided constitutional principles.

        Constitutional Principles:
        {principles}

        Response to critique:
        {response}

        Please provide a detailed critique pointing out any violations of the principles.
        """

        return critique_prompt

    def setup_revision_model(self):
        """Setup model for revising responses"""

        revision_prompt = """
        You are a helpful AI assistant. Your task is to revise the following response
        to better align with the constitutional principles.

        Constitutional Principles:
        {principles}

        Original Response:
        {response}

        Critique:
        {critique}

        Please provide a revised version of the response that addresses the critique
        and better aligns with the principles.
        """

        return revision_prompt

    def generate_constitutional_response(self, prompt):
        """Generate response following constitutional principles"""

        # Initial generation
        initial_response = self.base_model.generate(prompt)

        # Critique generation
        critique = self.generate_critique(initial_response)

        # Check if revision is needed
        if self.needs_revision(critique):
            # Revision generation
            revised_response = self.generate_revision(initial_response, critique)

            return {
                'response': revised_response,
                'initial_response': initial_response,
                'critique': critique,
                'revised': True
            }

        else:
            return {
                'response': initial_response,
                'critique': critique,
                'revised': False
            }

    def generate_critique(self, response):
        """Generate critique based on constitutional principles"""

        critique_prompt = self.critique_model.format(
            principles=self.format_principles(),
            response=response
        )

        critique = self.base_model.generate(
            critique_prompt,
            max_length=300,
            temperature=0.3
        )

        return critique

    def generate_revision(self, response, critique):
        """Generate revised response"""

        revision_prompt = self.revision_model.format(
            principles=self.format_principles(),
            response=response,
            critique=critique
        )

        revised_response = self.base_model.generate(
            revision_prompt,
            max_length=200,
            temperature=0.3
        )

        return revised_response

    def needs_revision(self, critique):
        """Determine if response needs revision based on critique"""

        # Simple heuristic: check for negative keywords in critique
        negative_indicators = [
            'violates',
            'problematic',
            'inappropriate',
            'harmful',
            'concerns',
            'issues'
        ]

        critique_lower = critique.lower()
        return any(indicator in critique_lower for indicator in negative_indicators)

    def format_principles(self):
        """Format constitutional principles for prompt"""

        return "\n".join(f"- {principle}" for principle in self.principles)

# Example constitutional principles
CONSTITUTIONAL_PRINCIPLES = [
    "The AI should be helpful, harmless, and honest.",
    "The AI should respect human autonomy and privacy.",
    "The AI should avoid generating harmful, illegal, or unethical content.",
    "The AI should be transparent about its limitations and uncertainties.",
    "The AI should promote fairness and avoid discrimination.",
    "The AI should be culturally sensitive and inclusive.",
    "The AI should encourage critical thinking and provide balanced information.",
    "The AI should respect intellectual property and cite sources when appropriate."
]

# Usage
constitutional_ai = ConstitutionalAI(base_model, CONSTITUTIONAL_PRINCIPLES)
result = constitutional_ai.generate_constitutional_response("What is the best way to...")
```

### 3. Direct Preference Optimization (DPO)

DPO is a simpler alternative to RLHF that directly optimizes for human preferences.

```python
class DPOTrainer:
    """Direct Preference Optimization trainer"""

    def __init__(self, model_name, beta=0.1):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.beta = beta  # Temperature parameter for DPO

    def compute_dpo_loss(self, prompts, chosen_responses, rejected_responses):
        """Compute DPO loss for preference optimization"""

        # Tokenize inputs
        chosen_tokens = self.tokenizer(
            prompts + chosen_responses,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        rejected_tokens = self.tokenizer(
            prompts + rejected_responses,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Get log probabilities for chosen responses
        chosen_logprobs = self.get_logprobs(
            self.model, chosen_tokens
        )

        # Get log probabilities for rejected responses
        rejected_logprobs = self.get_logprobs(
            self.model, rejected_tokens
        )

        # Get reference log probabilities
        reference_chosen_logprobs = self.get_logprobs(
            self.reference_model, chosen_tokens
        )

        reference_rejected_logprobs = self.get_logprobs(
            self.reference_model, rejected_tokens
        )

        # Compute DPO loss
        chosen_rewards = self.beta * (
            chosen_logprobs - reference_chosen_logprobs
        )

        rejected_rewards = self.beta * (
            rejected_logprobs - reference_rejected_logprobs
        )

        # DPO loss: -log sigmoid(chosen_rewards - rejected_rewards)
        losses = -torch.nn.functional.logsigmoid(
            chosen_rewards - rejected_rewards
        )

        return losses.mean()

    def get_logprobs(self, model, tokens):
        """Get log probabilities for tokens"""

        with torch.no_grad():
            outputs = model(**tokens)
            logits = outputs.logits

        # Shift logits and tokens for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tokens['input_ids'][..., 1:].contiguous()

        # Compute log probabilities
        log_probs = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )

        # Reshape to original batch size
        log_probs = log_probs.view(tokens['input_ids'].size(0), -1)

        return log_probs

    def train_step(self, batch):
        """Single training step for DPO"""

        # Compute DPO loss
        loss = self.compute_dpo_loss(
            batch['prompt'],
            batch['chosen'],
            batch['rejected']
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

## Bias Detection and Mitigation

### 1. Bias Detection Methods

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import chi2_contingency

class BiasDetector:
    """Comprehensive bias detection system"""

    def __init__(self):
        self.demographic_parity = DemographicParityChecker()
        self.equalized_odds = EqualizedOddsChecker()
        self.calibration = CalibrationChecker()
        self.counterfactual = CounterfactualBiasChecker()

    def detect_bias(self, model_outputs, protected_attributes, outcomes):
        """Comprehensive bias detection across multiple metrics"""

        results = {}

        # Demographic parity
        results['demographic_parity'] = self.demographic_parity.check(
            model_outputs, protected_attributes
        )

        # Equalized odds
        results['equalized_odds'] = self.equalized_odds.check(
            model_outputs, protected_attributes, outcomes
        )

        # Calibration
        results['calibration'] = self.calibration.check(
            model_outputs, protected_attributes, outcomes
        )

        # Counterfactual bias
        results['counterfactual'] = self.counterfactual.check(
            model_outputs, protected_attributes
        )

        return results

class DemographicParityChecker:
    """Check for demographic parity violations"""

    def check(self, predictions, protected_attributes):
        """Check if positive prediction rate is equal across groups"""

        groups = np.unique(protected_attributes)
        group_rates = {}

        for group in groups:
            group_mask = protected_attributes == group
            group_predictions = predictions[group_mask]
            group_rates[group] = np.mean(group_predictions)

        # Check for significant differences
        rates = list(group_rates.values())
        max_diff = max(rates) - min(rates)
        threshold = 0.1  # 10% difference threshold

        return {
            'group_rates': group_rates,
            'max_difference': max_diff,
            'violates_parity': max_diff > threshold,
            'threshold': threshold
        }

class EqualizedOddsChecker:
    """Check for equalized odds violations"""

    def check(self, predictions, protected_attributes, true_outcomes):
        """Check if true positive and false positive rates are equal across groups"""

        groups = np.unique(protected_attributes)
        group_metrics = {}

        for group in groups:
            group_mask = protected_attributes == group
            group_predictions = predictions[group_mask]
            group_outcomes = true_outcomes[group_mask]

            # Calculate metrics
            tp = np.sum((group_predictions == 1) & (group_outcomes == 1))
            fp = np.sum((group_predictions == 1) & (group_outcomes == 0))
            tn = np.sum((group_predictions == 0) & (group_outcomes == 0))
            fn = np.sum((group_predictions == 0) & (group_outcomes == 1))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            group_metrics[group] = {
                'true_positive_rate': tpr,
                'false_positive_rate': fpr
            }

        # Check for significant differences
        tprs = [metrics['true_positive_rate'] for metrics in group_metrics.values()]
        fprs = [metrics['false_positive_rate'] for metrics in group_metrics.values()]

        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)

        return {
            'group_metrics': group_metrics,
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'violates_equalized_odds': max(tpr_diff, fpr_diff) > 0.1
        }

class CounterfactualBiasChecker:
    """Check for bias using counterfactual examples"""

    def __init__(self, base_model):
        self.base_model = base_model
        self.sensitive_attributes = {
            'gender': ['he', 'she', 'him', 'her'],
            'race': ['person of color', 'white person'],
            'age': ['young person', 'elderly person']
        }

    def check(self, text_inputs, protected_attributes):
        """Check for bias using counterfactual examples"""

        bias_scores = {}

        for attribute_type, attribute_values in self.sensitive_attributes.items():
            counterfactual_scores = []

            for text_input in text_inputs:
                # Create counterfactual examples
                original_score = self.get_model_score(text_input)

                counterfactual_score = 0
                for attribute_value in attribute_values:
                    counterfactual_text = self.create_counterfactual(
                        text_input, attribute_type, attribute_value
                    )
                    counterfactual_score += self.get_model_score(counterfactual_text)

                counterfactual_score /= len(attribute_values)

                # Calculate bias score
                bias_score = abs(original_score - counterfactual_score)
                counterfactual_scores.append(bias_score)

            bias_scores[attribute_type] = {
                'mean_bias': np.mean(counterfactual_scores),
                'max_bias': np.max(counterfactual_scores),
                'std_bias': np.std(counterfactual_scores)
            }

        return bias_scores

    def create_counterfactual(self, text, attribute_type, attribute_value):
        """Create counterfactual by modifying sensitive attribute"""

        # Simple implementation - replace with more sophisticated method
        # This is a placeholder and would need proper implementation

        if attribute_type == 'gender':
            # Simple gender swap (very basic)
            if attribute_value in ['he', 'him']:
                return text.replace('she', 'he').replace('her', 'him')
            else:
                return text.replace('he', 'she').replace('him', 'her')

        return text

    def get_model_score(self, text):
        """Get model score for text (implementation depends on model type)"""
        # Placeholder - implement based on your model
        return 0.5
```

### 2. Bias Mitigation Techniques

```python
class BiasMitigation:
    """Bias mitigation techniques for ML models"""

    def __init__(self, model, protected_attribute_names):
        self.model = model
        self.protected_attributes = protected_attribute_names

    def preprocessing_mitigation(self, training_data):
        """Preprocessing bias mitigation"""

        # Option 1: Disparate impact remover
        transformed_data = self.disparate_impact_remover(training_data)

        # Option 2: Optimized preprocessing
        optimized_data = self.optimized_preprocessing(training_data)

        return optimized_data

    def inprocessing_mitigation(self, training_data):
        """Inprocessing bias mitigation"""

        # Adversarial debiasing
        debiased_model = self.adversarial_debiasing(training_data)

        return debiased_model

    def postprocessing_mitigation(self, model_outputs, protected_attributes):
        """Postprocessing bias mitigation"""

        # Equalized odds postprocessing
        calibrated_outputs = self.equalized_odds_postprocessing(
            model_outputs, protected_attributes
        )

        return calibrated_outputs

    def adversarial_debiasing(self, training_data):
        """Adversarial debiasing implementation"""

        class AdversarialDebiasingModel(nn.Module):
            def __init__(self, predictor, adversary):
                super().__init__()
                self.predictor = predictor
                self.adversary = adversary

            def forward(self, x, y):
                # Predictor forward pass
                predictions = self.predictor(x)

                # Adversary tries to predict protected attribute
                adversary_predictions = self.adversary(predictions)

                return predictions, adversary_predictions

        # Implement adversarial debiasing training
        predictor = self.create_predictor()
        adversary = self.create_adversary()

        adversarial_model = AdversarialDebiasingModel(predictor, adversary)

        # Training loop with adversarial loss
        self.train_adversarial_model(adversarial_model, training_data)

        return adversarial_model.predictor

    def equalized_odds_postprocessing(self, predictions, protected_attributes):
        """Postprocessing to achieve equalized odds"""

        groups = np.unique(protected_attributes)

        # Calculate group-specific thresholds
        thresholds = {}

        for group in groups:
            group_mask = protected_attributes == group
            group_predictions = predictions[group_mask]

            # Find threshold that equalizes TPR and FPR
            threshold = self.find_equalized_odds_threshold(group_predictions)
            thresholds[group] = threshold

        # Apply group-specific thresholds
        calibrated_predictions = np.zeros_like(predictions)

        for group in groups:
            group_mask = protected_attributes == group
            calibrated_predictions[group_mask] = (
                predictions[group_mask] >= thresholds[group]
            )

        return calibrated_predictions

    def find_equalized_odds_threshold(self, predictions):
        """Find threshold that equalizes TPR and FPR across groups"""

        # This is a simplified implementation
        # In practice, you would need to solve an optimization problem

        # Sort predictions
        sorted_indices = np.argsort(predictions)
        sorted_predictions = predictions[sorted_indices]

        # Find threshold that balances TPR and FPR
        # This is a placeholder - implement proper equalized odds algorithm

        return np.median(predictions)

class FairnessConstraints:
    """Implement fairness constraints during training"""

    def __init__(self, constraint_type='demographic_parity'):
        self.constraint_type = constraint_type

    def add_fairness_constraint(self, model, training_data):
        """Add fairness constraints to model training"""

        class FairModel(nn.Module):
            def __init__(self, base_model, constraint_fn):
                super().__init__()
                self.base_model = base_model
                self.constraint_fn = constraint_fn

            def forward(self, x, y, z):
                # Forward pass through base model
                predictions = self.base_model(x)

                # Compute fairness violation
                violation = self.constraint_fn(predictions, z, y)

                return predictions, violation

        # Define constraint function based on type
        if self.constraint_type == 'demographic_parity':
            constraint_fn = self.demographic_parity_constraint
        elif self.constraint_type == 'equalized_odds':
            constraint_fn = self.equalized_odds_constraint
        else:
            constraint_fn = self.dummy_constraint

        fair_model = FairModel(model, constraint_fn)
        return fair_model

    def demographic_parity_constraint(self, predictions, protected_attrs, outcomes):
        """Demographic parity constraint"""

        groups = np.unique(protected_attrs)
        group_rates = {}

        for group in groups:
            group_mask = protected_attrs == group
            group_predictions = predictions[group_mask]
            group_rates[group] = torch.mean(group_predictions.float())

        # Compute violation as max difference between group rates
        rates = list(group_rates.values())
        violation = torch.max(torch.stack(rates)) - torch.min(torch.stack(rates))

        return violation

    def equalized_odds_constraint(self, predictions, protected_attrs, outcomes):
        """Equalized odds constraint"""

        groups = np.unique(protected_attrs)

        # Compute TPR and FPR for each group
        group_metrics = {}

        for group in groups:
            group_mask = protected_attrs == group
            group_predictions = predictions[group_mask]
            group_outcomes = outcomes[group_mask]

            tp = torch.sum((group_predictions == 1) & (group_outcomes == 1))
            fp = torch.sum((group_predictions == 1) & (group_outcomes == 0))
            tn = torch.sum((group_predictions == 0) & (group_outcomes == 0))
            fn = torch.sum((group_predictions == 0) & (group_outcomes == 1))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else torch.tensor(0.0)

            group_metrics[group] = (tpr, fpr)

        # Compute violation as max difference in TPR and FPR
        tprs = [metrics[0] for metrics in group_metrics.values()]
        fprs = [metrics[1] for metrics in group_metrics.values()]

        tpr_violation = torch.max(torch.stack(tprs)) - torch.min(torch.stack(tprs))
        fpr_violation = torch.max(torch.stack(fprs)) - torch.min(torch.stack(fprs))

        return torch.max(tpr_violation, fpr_violation)

    def dummy_constraint(self, predictions, protected_attrs, outcomes):
        """Dummy constraint for testing"""
        return torch.tensor(0.0)
```

## Content Safety and Filtering

### 1. Multi-Layer Safety System

```python
class MultiLayerSafetySystem:
    """Comprehensive safety system with multiple layers of protection"""

    def __init__(self):
        self.layers = [
            InputSanitizer(),
            PromptInjectionDetector(),
            ContentModerator(),
            OutputValidator(),
            PostProcessingFilter()
        ]

        self.safety_config = {
            'strict_mode': False,
            'allow_list': [],
            'deny_list': [],
            'custom_rules': []
        }

    def check_input_safety(self, input_text):
        """Check input safety across all layers"""

        results = []

        for layer in self.layers:
            try:
                result = layer.check_input(input_text)
                results.append({
                    'layer': layer.__class__.__name__,
                    'result': result,
                    'passed': result.get('passed', True)
                })
            except Exception as e:
                results.append({
                    'layer': layer.__class__.__name__,
                    'result': {'error': str(e)},
                    'passed': False
                })

        return results

    def filter_content(self, content):
        """Apply safety filtering to content"""

        # Check if content passes all safety layers
        safety_results = self.check_input_safety(content)

        # Determine overall safety status
        overall_safe = all(result['passed'] for result in safety_results)

        if overall_safe:
            return {
                'content': content,
                'safe': True,
                'safety_results': safety_results
            }

        else:
            # Apply remediation
            safe_content = self.remediate_unsafe_content(content, safety_results)

            return {
                'content': safe_content,
                'safe': True,
                'original_unsafe': True,
                'safety_results': safety_results
            }

    def remediate_unsafe_content(self, content, safety_results):
        """Remediate unsafe content based on detected issues"""

        # Apply fixes based on detected problems
        for result in safety_results:
            if not result['passed']:
                layer_name = result['layer']
                issue = result['result']

                if layer_name == 'PromptInjectionDetector':
                    content = self.remove_prompt_injection(content)
                elif layer_name == 'ContentModerator':
                    content = self.moderate_content(content, issue)
                elif layer_name == 'OutputValidator':
                    content = self.validate_output(content, issue)

        return content

class InputSanitizer:
    """Sanitize and clean input text"""

    def __init__(self):
        self.dangerous_patterns = [
            r'<script.*?>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript protocol
            r'data:',  # Data URLs
            r'eval\s*\(',  # Eval functions
            r'exec\s*\(',  # Exec functions
        ]

        self.sanitization_rules = {
            'normalize_whitespace': True,
            'remove_control_chars': True,
            'encode_special_chars': True,
            'strip_tags': True
        }

    def check_input(self, text):
        """Sanitize input text"""

        original_text = text
        sanitized_text = text

        # Apply sanitization rules
        if self.sanitization_rules.get('normalize_whitespace', False):
            sanitized_text = ' '.join(sanitized_text.split())

        if self.sanitization_rules.get('remove_control_chars', False):
            sanitized_text = ''.join(char for char in sanitized_text if ord(char) >= 32)

        if self.sanitization_rules.get('strip_tags', False):
            import re
            sanitized_text = re.sub(r'<[^>]+>', '', sanitized_text)

        # Check for dangerous patterns
        dangerous_matches = []
        import re
        for pattern in self.dangerous_patterns:
            matches = re.findall(pattern, sanitized_text, re.IGNORECASE)
            dangerous_matches.extend(matches)

        return {
            'original_text': original_text,
            'sanitized_text': sanitized_text,
            'dangerous_patterns_found': len(dangerous_matches),
            'passed': len(dangerous_matches) == 0,
            'sanitized': sanitized_text != original_text
        }

class PromptInjectionDetector:
    """Detect and prevent prompt injection attacks"""

    def __init__(self):
        self.injection_patterns = [
            r'ignore\s+(previous|all)?\s+(instructions|directives)',
            r'disregard\s+(previous|all)?\s+(instructions|directives)',
            r'forget\s+(what\s+)?(I\s+)?(told\s+)?you',
            r'you\s+are\s+now\s+',
            r'pretend\s+you\s+are\s+',
            r'roleplay\s+as\s+',
            r'act\s+as\s+',
            r'ignore\s+system\s+',
            r'override\s+(system\s+)?(instructions|directives)',
            r'bypass\s+(your\s+)?(restrictions|guidelines)',
            r'jailbreak',
            r'DAN\s*',
            r'STAN\s*',
            r'Developer\s+Mode'
        ]

        self.context_break_patterns = [
            r'\n\s*system\s*:',
            r'\n\s*user\s*:',
            r'\n\s*assistant\s*:',
            r'\n\s*\[INST\]\s*',
            r'\n\s*\[\/INST\]\s*',
            r'\n\s*<<\s*',
            r'\n\s*>>\s*'
        ]

    def check_input(self, text):
        """Detect prompt injection attempts"""

        injection_score = 0
        detected_patterns = []

        # Check for injection patterns
        import re
        for pattern in self.injection_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_patterns.extend(matches)
                injection_score += len(matches) * 0.2

        # Check for context breaking patterns
        for pattern in self.context_break_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_patterns.extend(matches)
                injection_score += len(matches) * 0.15

        # Check for unusual formatting
        if self.has_suspicious_formatting(text):
            injection_score += 0.3
            detected_patterns.append("suspicious formatting")

        # Check for encoding tricks
        if self.has_encoding_tricks(text):
            injection_score += 0.25
            detected_patterns.append("encoding tricks")

        return {
            'injection_score': min(1.0, injection_score),
            'detected_patterns': detected_patterns,
            'passed': injection_score < 0.7,
            'risk_level': self.categorize_risk(injection_score)
        }

    def has_suspicious_formatting(self, text):
        """Check for suspicious formatting patterns"""
        suspicious_indicators = [
            text.count('\n') > 5,  # Too many newlines
            text.count('  ') > 3,  # Multiple spaces
            len([c for c in text if ord(c) > 127]) > len(text) * 0.1,  # Many unicode chars
            text.isupper() and len(text) > 50,  # Long uppercase text
        ]

        return any(suspicious_indicators)

    def has_encoding_tricks(self, text):
        """Check for encoding-based tricks"""
        import re

        # Check for URL encoding
        if re.search(r'%[0-9A-F]{2}', text):
            return True

        # Check for HTML entities
        if re.search(r'&[a-zA-Z]+;', text):
            return True

        # Check for mixed character sets
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        if ascii_chars / len(text) < 0.7:
            return True

        return False

    def categorize_risk(self, injection_score):
        """Categorize risk level based on injection score"""
        if injection_score < 0.3:
            return 'low'
        elif injection_score < 0.6:
            return 'medium'
        else:
            return 'high'

class ContentModerator:
    """Moderate content for policy violations"""

    def __init__(self):
        self.moderation_categories = {
            'hate_speech': self.check_hate_speech,
            'harassment': self.check_harassment,
            'violence': self.check_violence,
            'self_harm': self.check_self_harm,
            'sexual_content': self.check_sexual_content,
            'misinformation': self.check_misinformation,
            'spam': self.check_spam
        }

        self.thresholds = {
            'hate_speech': 0.7,
            'harassment': 0.7,
            'violence': 0.6,
            'self_harm': 0.5,
            'sexual_content': 0.8,
            'misinformation': 0.6,
            'spam': 0.8
        }

    def check_input(self, text):
        """Moderate content across all categories"""

        moderation_results = {}

        for category, check_function in self.moderation_categories.items():
            try:
                result = check_function(text)
                moderation_results[category] = result
            except Exception as e:
                moderation_results[category] = {
                    'score': 0,
                    'error': str(e)
                }

        # Determine overall moderation decision
        violation_scores = {}
        for category, result in moderation_results.items():
            if 'score' in result:
                violation_scores[category] = result['score']

        max_violation_score = max(violation_scores.values()) if violation_scores else 0
        violated_categories = [
            category for category, score in violation_scores.items()
            if score > self.thresholds.get(category, 0.7)
        ]

        return {
            'category_results': moderation_results,
            'max_violation_score': max_violation_score,
            'violated_categories': violated_categories,
            'passed': len(violated_categories) == 0,
            'requires_human_review': max_violation_score > 0.8
        }

    def check_hate_speech(self, text):
        """Check for hate speech using ML model"""

        # Use pre-trained hate speech classifier
        # This is a placeholder - implement with actual model
        hate_speech_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=0 if torch.cuda.is_available() else -1
        )

        result = hate_speech_classifier(text)

        return {
            'score': result[0]['score'] if result[0]['label'] == 'TOXIC' else 1 - result[0]['score'],
            'label': result[0]['label'],
            'confidence': result[0]['score']
        }

    def check_harassment(self, text):
        """Check for harassment content"""
        # Implement harassment detection
        return {'score': 0.1, 'label': 'safe'}

    def check_violence(self, text):
        """Check for violent content"""
        # Implement violence detection
        return {'score': 0.1, 'label': 'safe'}

    def check_self_harm(self, text):
        """Check for self-harm content"""
        # Implement self-harm detection
        return {'score': 0.1, 'label': 'safe'}

    def check_sexual_content(self, text):
        """Check for sexual content"""
        # Implement sexual content detection
        return {'score': 0.1, 'label': 'safe'}

    def check_misinformation(self, text):
        """Check for misinformation"""
        # Implement misinformation detection
        return {'score': 0.1, 'label': 'factual'}

    def check_spam(self, text):
        """Check for spam content"""
        # Implement spam detection
        return {'score': 0.1, 'label': 'not_spam'}

class OutputValidator:
    """Validate model outputs for safety and quality"""

    def __init__(self):
        self.validation_rules = [
            self.check_length,
            self.check_coherence,
            self.check_factual_consistency,
            self.check_harmful_content
        ]

    def check_input(self, output_text):
        """Validate output text"""

        validation_results = {}

        for rule in self.validation_rules:
            try:
                result = rule(output_text)
                validation_results[rule.__name__] = result
            except Exception as e:
                validation_results[rule.__name__] = {
                    'passed': False,
                    'error': str(e)
                }

        # Overall validation result
        all_passed = all(
            result.get('passed', True)
            for result in validation_results.values()
        )

        return {
            'validation_results': validation_results,
            'passed': all_passed,
            'requires_revision': not all_passed
        }

    def check_length(self, text):
        """Check output length"""
        max_length = 1000
        min_length = 10

        length = len(text.split())

        if length < min_length:
            return {
                'passed': False,
                'issue': 'too_short',
                'length': length,
                'min_length': min_length
            }
        elif length > max_length:
            return {
                'passed': False,
                'issue': 'too_long',
                'length': length,
                'max_length': max_length
            }

        return {
            'passed': True,
            'length': length
        }

    def check_coherence(self, text):
        """Check output coherence"""
        # Simple coherence check - count sentence fragments
        sentences = text.split('.')

        if len(sentences) < 2:
            return {
                'passed': False,
                'issue': 'not_enough_sentences',
                'sentence_count': len(sentences)
            }

        # Check for repetitive content
        words = text.lower().split()
        unique_words = set(words)
        diversity = len(unique_words) / len(words) if words else 0

        if diversity < 0.3:
            return {
                'passed': False,
                'issue': 'low_diversity',
                'diversity_score': diversity
            }

        return {
            'passed': True,
            'diversity_score': diversity,
            'sentence_count': len(sentences)
        }

    def check_factual_consistency(self, text):
        """Check for factual consistency (simplified)"""
        # This is a simplified check - in practice, you'd use fact-checking models

        # Check for contradictory statements
        contradiction_indicators = [
            'but however',
            'although but',
            'despite that',
            'however still'
        ]

        text_lower = text.lower()
        contradictions = sum(1 for indicator in contradiction_indicators if indicator in text_lower)

        if contradictions > 2:
            return {
                'passed': False,
                'issue': 'potential_contradictions',
                'contradiction_count': contradictions
            }

        return {
            'passed': True,
            'contradiction_count': contradictions
        }

    def check_harmful_content(self, text):
        """Check for harmful content in output"""
        # Use the same moderation system as for inputs
        content_moderator = ContentModerator()
        moderation_result = content_moderator.check_input(text)

        return {
            'passed': moderation_result['passed'],
            'moderation_result': moderation_result
        }
```

This comprehensive guide covers the essential aspects of AI safety and alignment for large language models. As the field continues to evolve, new techniques and approaches will emerge, but the fundamental principles of reliability, robustness, transparency, controllability, and beneficence remain central to building safe and aligned AI systems.

The key to successful implementation is adopting a multi-layered approach that combines technical safeguards, evaluation frameworks, and governance processes to ensure that AI systems behave safely and in alignment with human values across diverse contexts and use cases.
