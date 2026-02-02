# Natural Language Processing (NLP) Cheat Sheet

## Table of Contents

1. [Text Preprocessing](#text-preprocessing)
2. [NLTK (Natural Language Toolkit)](#nltk)
3. [spaCy](#spacy)
4. [Transformers](#transformers)
5. [Sentiment Analysis](#sentiment-analysis)
6. [Named Entity Recognition](#named-entity-recognition)
7. [NLP Pipeline Patterns](#nlp-pipeline-patterns)

---

## Text Preprocessing

### Basic Text Cleaning

```python
import re
import string
from typing import List

# Remove HTML tags
def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)

# Remove special characters and digits
def remove_special_chars(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

# Convert to lowercase
def to_lowercase(text):
    return text.lower()

# Remove extra whitespace
def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

# Complete preprocessing pipeline
def preprocess_text(text):
    text = remove_html_tags(text)
    text = remove_special_chars(text)
    text = to_lowercase(text)
    text = normalize_whitespace(text)
    return text

# Example usage
text = "Hello, World! Visit our <a href='https://example.com'>website</a>."
clean_text = preprocess_text(text)  # "hello world visit our website"
```

### Tokenization

```python
# Word tokenization
def word_tokenize(text):
    return text.split()

# Advanced tokenization with punctuation handling
import re
def advanced_tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

# Character-level tokenization
def char_tokenize(text):
    return list(text.replace(' ', ''))

# NLTK tokenization (imports shown in NLTK section)
# words = nltk.word_tokenize(text)
# sentences = nltk.sent_tokenize(text)
```

### Stemming and Lemmatization

```python
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

# Porter Stemmer (fastest, less accurate)
stemmer = PorterStemmer()
stemmed = stemmer.stem("running")  # "run"

# Snowball Stemmer (better than Porter)
stemmer = SnowballStemmer('english')
stemmed = stemmer.stem("running")  # "run"

# WordNet Lemmatizer (most accurate, needs POS tags)
lemmatizer = WordNetLemmatizer()
lemmatized = lemmatizer.lemmatize("running", "v")  # "run"
```

---

## NLTK (Natural Language Toolkit)

### Installation and Setup

```bash
# Install NLTK
pip install nltk

# Download required data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

### Core NLTK Functions

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Tokenization
text = "Hello world. This is a test sentence."
sentences = sent_tokenize(text)  # ['Hello world.', 'This is a test sentence.']
words = word_tokenize(text)      # ['Hello', 'world', '.', 'This', 'is', 'a', 'test', 'sentence', '.']

# Stop Words Removal
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

# POS Tagging
pos_tags = nltk.pos_tag(words)
# [('Hello', 'NNP'), ('world', 'NN'), ('.', '.')]

# Named Entity Recognition
entities = nltk.ne_chunk(pos_tags)
```

---

## spaCy

### Installation and Setup

```bash
# Install spaCy
pip install spacy

# Download English model
python -m spacy download en_core_web_sm

# Download German model (optional)
python -m spacy download de_core_news_sm
```

### Core spaCy Functions

```python
import spacy

# Load model
nlp = spacy.load("en_core_web_sm")

# Process text
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Tokenization (already done during processing)
for token in doc:
    print(f"Token: {token.text}, POS: {token.pos_}, Lemma: {token.lemma_}")

# Sentence segmentation
for sent in doc.sents:
    print(f"Sentence: {sent.text}")

# Stop Words Removal
filtered_tokens = [token for token in doc if not token.is_stop]

# Named Entity Recognition
entities = [(ent.text, ent.label_, ent.start_char, ent.end_char)
           for ent in doc.ents]

# Custom Components
def custom_component(doc):
    # Add custom logic here
    return doc

# Add component to pipeline
nlp.add_pipe(custom_component, after="ner")

# Dependency Parsing
for token in doc:
    print(f"{token.text} -> {token.dep_} <- {token.head.text}")
```

### spaCy Matchers

```python
from spacy.matcher import Matcher, PhraseMatcher

# Pattern-based matching
matcher = Matcher(nlp.vocab)

# Add patterns
pattern1 = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
matcher.add("HELLO_WORLD", [pattern1])

doc = nlp("Hello, world! Hello world!")
matches = matcher(doc)

# Phrase matching (faster for exact phrases)
phrase_matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp(term) for term in ["hello world", "test phrase"]]
phrase_matcher.add("PHRASES", None, *patterns)
```

---

## Transformers

### Installation and Setup

```bash
# Install transformers and torch
pip install transformers torch

# For GPU support (if available)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModel, pipeline

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenization and encoding
text = "Hello, this is a test sentence."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Get model output
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

# Using pipelines (simplified interface)
classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")

# Other pipeline examples
ner_pipeline = pipeline("ner", aggregation_strategy="simple")
qa_pipeline = pipeline("question-answering")
summarizer = pipeline("summarization")
translation = pipeline("translation_en_to_fr")
```

### BERT Fine-tuning

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Prepare data
texts = ["I love this", "I hate this"]
labels = [1, 0]  # 1: positive, 0: negative

# Tokenize
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
labels = torch.tensor(labels)

# Forward pass
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss.backward()
    optimizer.step()
```

### GPT-2 Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Generate text
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate with different parameters
output = model.generate(
    input_ids,
    max_length=100,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    pad_token_id=50256
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

---

## Sentiment Analysis

### Using VADER (NLTK)

```python
from nltk.sentiment import SentimentIntensityAnalyzer

# Install VADER
nltk.download('vader_lexicon')

# Initialize analyzer
analyzer = SentimentIntensityAnalyzer()

# Analyze sentiment
text = "I love this movie! It's amazing!"
scores = analyzer.polarity_scores(text)

# Output: {'compound': 0.8015, 'neg': 0.0, 'neu': 0.174, 'pos': 0.826}

# Interpretation
if scores['compound'] >= 0.05:
    sentiment = "Positive"
elif scores['compound'] <= -0.05:
    sentiment = "Negative"
else:
    sentiment = "Neutral"
```

### Using TextBlob

```python
from textblob import TextBlob

# Create TextBlob object
blob = TextBlob("I love this product!")

# Get sentiment
sentiment = blob.sentiment
# Sentiment(polarity=0.5, subjectivity=0.6)

# Polarity: -1 (negative) to 1 (positive)
# Subjectivity: 0 (objective) to 1 (subjective)
```

### Using spaCy for Sentiment

```python
# Using TextCat component in spaCy
import spacy
from spacy.training import Example

# Load or create model with text categorizer
nlp = spacy.blank("en")
textcat = nlp.add_pipe("textcat")

# Add labels
textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")

# Training data
train_data = [
    ("I love this movie!", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}),
    ("I hate this movie!", {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}})
]

# Train model
nlp.begin_training()
for i in range(10):
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example])
```

### Using Transformers

```python
from transformers import pipeline

# Pre-trained sentiment analysis models
sentiment_analyzer = pipeline("sentiment-analysis",
                            model="cardiffnlp/twitter-roberta-base-sentiment-latest")
result = sentiment_analyzer("I absolutely love this!")

# Result: [{'label': 'LABEL_2', 'score': 0.95}]
```

---

## Named Entity Recognition

### Using NLTK

```python
import nltk
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

# Download required data
nltk.download('maxent_ne_chunker')
nltk.download('words')

def nlp_nltk_ner(text):
    # Tokenize and POS tag
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # Named Entity Recognition
    chunked = ne_chunk(pos_tags)

    # Extract entities
    entities = []
    for subtree in chunked.subtrees():
        if subtree.label() in ['PERSON', 'ORGANIZATION', 'GPE', 'DATE', 'TIME']:
            entity = ' '.join([item[0] for item in subtree])
            entities.append((entity, subtree.label()))

    return entities

# Example
text = "Apple Inc. is looking at buying U.K. startup for $1 billion"
entities = nlp_nltk_ner(text)
# [('Apple Inc.', 'ORGANIZATION'), ('U.K.', 'GPE'), ('$1 billion', 'MONEY')]
```

### Using spaCy

```python
def spacy_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_, ent.start_char, ent.end_char)
                for ent in doc.ents]
    return entities

# Example
text = "Barack Obama was born in Hawaii and served as President of the United States."
entities = spacy_ner(text)
# [('Barack Obama', 'PERSON', 0, 12), ('Hawaii', 'GPE', 25, 31),
#  ('United States', 'GPE', 58, 72)]

# Common NER labels in spaCy:
# PERSON: People, including fictional
# ORG: Companies, agencies, institutions
# GPE: Countries, cities, states
# MONEY: Monetary values
# DATE: Absolute or relative dates
# TIME: Times smaller than a day
```

### Using Transformers

```python
from transformers import pipeline

# Load NER pipeline
ner_pipeline = pipeline("ner",
                       model="dbmdz/bert-large-cased-finetuned-conll03-english",
                       aggregation_strategy="simple")

# Extract entities
text = "John Smith works at Microsoft in New York"
entities = ner_pipeline(text)

# Output format:
# [{'word': 'John', 'entity': 'I-PER', 'score': 0.99, 'index': 1}, ...]
```

---

## NLP Pipeline Patterns

### Basic NLP Pipeline

```python
class NLPPipeline:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        # Clean text
        text = self.remove_html(text)
        text = self.remove_special_chars(text)
        text = text.lower()
        text = self.normalize_whitespace(text)

        # Tokenize
        doc = self.nlp(text)

        # Remove stop words and lemmatize
        processed_tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct
        ]

        return processed_tokens

    def remove_html(self, text):
        return re.sub(r'<.*?>', '', text)

    def remove_special_chars(self, text):
        return re.sub(r'[^a-zA-Z\s]', '', text)

    def normalize_whitespace(self, text):
        return re.sub(r'\s+', ' ', text).strip()

# Usage
pipeline = NLPPipeline()
processed_text = pipeline.preprocess("Hello, <b>world</b>!")
# ['hello', 'world']
```

### Advanced NLP Pipeline with Multiple Models

```python
class AdvancedNLPipeline:
    def __init__(self):
        # Load all models
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.ner_pipeline = pipeline("ner",
                                   model="dbmdz/bert-large-cased-finetuned-conll03-english")
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.classifier_pipeline = pipeline("text-classification")

    def full_analysis(self, text):
        """Complete NLP analysis of text"""
        results = {
            'original_text': text,
            'spacy_entities': self.extract_entities_spacy(text),
            'bert_entities': self.extract_entities_bert(text),
            'sentiment': self.analyze_sentiment(text),
            'classification': self.classify_text(text),
            'preprocessing': self.preprocess_text(text)
        }
        return results

    def extract_entities_spacy(self, text):
        doc = self.spacy_nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def extract_entities_bert(self, text):
        entities = self.ner_pipeline(text)
        return [(entity['word'], entity['entity']) for entity in entities]

    def analyze_sentiment(self, text):
        result = self.sentiment_pipeline(text)
        return {
            'label': result[0]['label'],
            'score': result[0]['score']
        }

    def classify_text(self, text):
        result = self.classifier_pipeline(text)
        return {
            'label': result[0]['label'],
            'score': result[0]['score']
        }

    def preprocess_text(self, text):
        doc = self.spacy_nlp(text)
        return [token.lemma_ for token in doc
                if not token.is_stop and not token.is_punct]

# Usage
advanced_pipeline = AdvancedNLPipeline()
analysis = advanced_pipeline.full_analysis("Apple is looking at buying U.K. startup")
```

### Production NLP Pipeline

```python
import logging
from typing import Dict, List, Any
from functools import lru_cache
import pickle

class ProductionNLPipeline:
    def __init__(self, config_path: str = None):
        self.setup_logging()
        self.load_models()
        self.setup_caching()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_models(self):
        """Load all models with error handling"""
        try:
            self.spacy_nlp = spacy.load("en_core_web_sm")
            self.ner_model = pipeline("ner",
                                    model="dbmdz/bert-large-cased-finetuned-conll03-english")
            self.sentiment_model = pipeline("sentiment-analysis")
            self.logger.info("All models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

    def setup_caching(self):
        """Setup caching for frequently used operations"""
        self.preprocess_cache = {}

    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple texts efficiently"""
        results = []
        for text in texts:
            result = self.process_single(text)
            results.append(result)
        return results

    def process_single(self, text: str) -> Dict[str, Any]:
        """Process single text with error handling"""
        try:
            result = {
                'text': text,
                'entities': self.extract_entities(text),
                'sentiment': self.analyze_sentiment(text),
                'processed_tokens': self.preprocess(text),
                'metadata': {
                    'length': len(text),
                    'word_count': len(text.split())
                }
            }
            return result
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            return {'text': text, 'error': str(e)}

    @lru_cache(maxsize=100)
    def preprocess(self, text: str) -> tuple:
        """Cached preprocessing"""
        doc = self.spacy_nlp(text)
        return tuple([token.lemma_ for token in doc
                     if not token.is_stop and not token.is_punct])

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract entities using multiple models"""
        # spaCy entities
        doc = self.spacy_nlp(text)
        spacy_entities = [(ent.text, ent.label_) for ent in doc.ents]

        # BERT entities
        bert_entities = self.ner_model(text)

        return {
            'spacy': spacy_entities,
            'bert': bert_entities
        }

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment with confidence score"""
        result = self.sentiment_model(text)[0]
        confidence = 'high' if result['score'] > 0.8 else 'medium' if result['score'] > 0.6 else 'low'

        return {
            'label': result['label'],
            'score': result['score'],
            'confidence': confidence
        }

# Production usage
prod_pipeline = ProductionNLPipeline()
texts = [
    "I love this product!",
    "This is terrible.",
    "The weather is nice today."
]

results = prod_pipeline.process_batch(texts)
for result in results:
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Entities: {result['entities']}")
    print("---")
```

### Evaluation Pipeline

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class NLPEvaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def evaluate_sentiment(self, test_texts, true_labels):
        """Evaluate sentiment analysis model"""
        predicted_labels = []

        for text in test_texts:
            result = self.pipeline.analyze_sentiment(text)
            predicted_labels.append(result['label'])

        # Convert to binary if needed
        report = classification_report(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)

        return {
            'report': report,
            'confusion_matrix': cm
        }

    def evaluate_ner(self, test_texts, true_entities):
        """Evaluate NER model"""
        results = []

        for i, text in enumerate(test_texts):
            predicted = self.pipeline.extract_entities(text)
            results.append({
                'text': text,
                'true_entities': true_entities[i],
                'predicted_entities': predicted['spacy'] + predicted['bert']
            })

        return results

# Usage example
# evaluator = NLPEvaluator(prod_pipeline)
# sentiment_results = evaluator.evaluate_sentiment(test_texts, true_labels)
```

---

## Quick Reference Tables

### Common NLP Tasks and Libraries

| Task            | NLTK | spaCy       | Transformers | TextBlob |
| --------------- | ---- | ----------- | ------------ | -------- |
| Tokenization    | ✅   | ✅          | ✅           | ✅       |
| Stemming        | ✅   | ❌          | ❌           | ✅       |
| Lemmatization   | ✅   | ✅          | ❌           | ✅       |
| POS Tagging     | ✅   | ✅          | ❌           | ❌       |
| NER             | ✅   | ✅          | ✅           | ❌       |
| Sentiment       | ✅   | ✅ (custom) | ✅           | ✅       |
| Text Generation | ❌   | ❌          | ✅           | ❌       |

### Common NER Labels (spaCy)

| Label  | Description                  | Example         |
| ------ | ---------------------------- | --------------- |
| PERSON | People, fictional characters | "Barack Obama"  |
| ORG    | Companies, institutions      | "Microsoft"     |
| GPE    | Countries, cities, states    | "United States" |
| MONEY  | Monetary values              | "$1 billion"    |
| DATE   | Absolute or relative dates   | "July 4th"      |
| TIME   | Times smaller than day       | "9:00 AM"       |

### Sentiment Analysis Models

| Model    | Accuracy | Speed  | Use Case           |
| -------- | -------- | ------ | ------------------ |
| VADER    | Medium   | Fast   | Social media       |
| TextBlob | Medium   | Fast   | General purpose    |
| spaCy    | High     | Medium | Custom training    |
| RoBERTa  | High     | Slow   | Production systems |

### Best Practices

1. **Preprocessing**: Always clean and normalize text before analysis
2. **Model Selection**: Choose based on your specific use case and requirements
3. **Evaluation**: Always test with domain-specific data
4. **Performance**: Consider model size vs. accuracy trade-offs
5. **Caching**: Cache frequently used operations for production systems
6. **Error Handling**: Implement robust error handling in production pipelines

---

_Last Updated: 2025-11-06_
_For more detailed examples and advanced techniques, refer to the full NLP theory and practice documents._
