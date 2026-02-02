# RAG Systems - Practice Exercises

## Table of Contents

1. [Basic RAG Implementation](#basic-rag)
2. [Vector Database Integration](#vector-db)
3. [Document Processing](#document-processing)
4. [Retrieval Strategies](#retrieval-strategies)
5. [RAG Evaluation](#rag-evaluation)
6. [Advanced RAG Patterns](#advanced-rag)
7. [Production RAG Systems](#production-rag)

## Basic RAG Implementation {#basic-rag}

### Exercise 1: Simple RAG Pipeline

```python
import numpy as np
from typing import List, Dict, Any
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRAG:
    """Basic RAG implementation for document retrieval"""

    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.indexed = False

    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the knowledge base"""
        if metadata is None:
            metadata = [{}] * len(documents)

        # Store documents and metadata
        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            self.documents.append({
                'id': len(self.documents),
                'content': doc,
                'metadata': meta
            })

        self.indexed = False

    def index_documents(self):
        """Create embeddings for all documents"""
        if not self.documents:
            print("No documents to index")
            return

        print(f"Indexing {len(self.documents)} documents...")

        # Extract document contents
        contents = [doc['content'] for doc in self.documents]

        # Generate embeddings
        embeddings = self.embedding_model.encode(contents)

        # Store embeddings
        self.embeddings = embeddings
        self.indexed = True

        print(f"Indexed {len(self.documents)} documents")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        if not self.indexed:
            raise ValueError("Documents not indexed. Call index_documents() first.")

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                'content': doc['content'],
                'metadata': doc['metadata'],
                'similarity': similarities[idx],
                'id': doc['id']
            })

        return results

    def generate_response(self, query: str, llm_function, top_k: int = 5) -> str:
        """Generate response using RAG"""
        # Retrieve relevant documents
        relevant_docs = self.retrieve(query, top_k)

        # Prepare context
        context = "\\n\\n".join([doc['content'] for doc in relevant_docs])

        # Generate response with context
        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""

        response = llm_function(prompt)

        return {
            'response': response,
            'retrieved_documents': relevant_docs,
            'context_used': context
        }

    def save_index(self, filepath: str):
        """Save indexed documents to file"""
        if not self.indexed:
            raise ValueError("No indexed documents to save")

        data = {
            'documents': self.documents,
            'embeddings': self.embeddings
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved index to {filepath}")

    def load_index(self, filepath: str):
        """Load indexed documents from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.indexed = True

        print(f"Loaded {len(self.documents)} documents from {filepath}")

# Practice Exercise 1.1: Build a Simple RAG System
def exercise_simple_rag():
    """Practice exercise: Build a simple RAG system"""

    # Sample documents about Python
    documents = [
        "Python is a high-level programming language created by Guido van Rossum. It emphasizes code readability and allows programmers to express concepts in fewer lines of code.",
        "Variables in Python are used to store data values. You can use variables to store numbers, strings, lists, dictionaries, and other data types.",
        "Functions in Python are defined using the def keyword. Functions help organize code into reusable blocks and can take parameters and return values.",
        "Lists in Python are ordered collections of items that can be changed. Lists can contain different data types and are created using square brackets.",
        "Dictionaries in Python are unordered collections of key-value pairs. They are useful for storing and retrieving data based on unique keys."
    ]

    # Create RAG system
    rag = SimpleRAG()

    # Add documents
    rag.add_documents(documents)

    # Index documents
    rag.index_documents()

    # Test retrieval
    queries = [
        "What is Python?",
        "How do you create variables in Python?",
        "What are lists used for?",
        "How do you define functions?"
    ]

    print("=== Simple RAG System Demo ===\\n")

    for query in queries:
        print(f"Query: {query}")
        results = rag.retrieve(query, top_k=3)

        print("Retrieved documents:")
        for i, result in enumerate(results, 1):
            print(f"{i}. [Similarity: {result['similarity']:.3f}] {result['content'][:100]}...")
        print()

    return rag

# Run the exercise
# rag_system = exercise_simple_rag()
```

### Exercise 2: Enhanced RAG with Metadata

```python
class EnhancedRAG(SimpleRAG):
    """Enhanced RAG with metadata filtering and hybrid search"""

    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        super().__init__(embedding_model)
        self.metadata_index = {}

    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents with enhanced metadata support"""
        if metadata is None:
            metadata = [{}] * len(documents)

        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            doc_id = len(self.documents)

            # Store document
            self.documents.append({
                'id': doc_id,
                'content': doc,
                'metadata': meta
            })

            # Update metadata index
            for key, value in meta.items():
                if key not in self.metadata_index:
                    self.metadata_index[key] = {}
                if value not in self.metadata_index[key]:
                    self.metadata_index[key][value] = []
                self.metadata_index[key][value].append(doc_id)

        self.indexed = False

    def retrieve_with_filter(self, query: str, filters: Dict = None, top_k: int = 5) -> List[Dict]:
        """Retrieve documents with metadata filtering"""
        if not self.indexed:
            raise ValueError("Documents not indexed. Call index_documents() first.")

        # Get initial candidates
        if filters:
            candidate_ids = self.get_candidates_by_filter(filters)
        else:
            candidate_ids = list(range(len(self.documents)))

        if not candidate_ids:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])

        # Calculate similarities for candidates
        candidate_embeddings = self.embeddings[candidate_ids]
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc_id = candidate_ids[idx]
            doc = self.documents[doc_id]
            results.append({
                'content': doc['content'],
                'metadata': doc['metadata'],
                'similarity': similarities[idx],
                'id': doc_id
            })

        return results

    def get_candidates_by_filter(self, filters: Dict) -> List[int]:
        """Get document IDs that match metadata filters"""
        filter_results = []

        for filter_key, filter_value in filters.items():
            if filter_key in self.metadata_index:
                if filter_value in self.metadata_index[filter_key]:
                    filter_results.append(self.metadata_index[filter_key][filter_value])

        # Intersection of all filters
        if filter_results:
            candidate_ids = set(filter_results[0])
            for result_set in filter_results[1:]:
                candidate_ids = candidate_ids.intersection(set(result_set))
            return list(candidate_ids)
        else:
            return []

    def hybrid_search(self, query: str, semantic_weight: float = 0.7,
                     keyword_weight: float = 0.3, top_k: int = 5) -> List[Dict]:
        """Hybrid search combining semantic and keyword matching"""

        # Semantic search
        semantic_results = self.retrieve(query, top_k=top_k*2)

        # Keyword search (simple implementation)
        keyword_results = self.keyword_search(query, top_k=top_k*2)

        # Combine results
        combined_scores = {}

        # Add semantic scores
        for result in semantic_results:
            doc_id = result['id']
            combined_scores[doc_id] = semantic_weight * result['similarity']

        # Add keyword scores
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined_scores:
                combined_scores[doc_id] += keyword_weight * result['similarity']
            else:
                combined_scores[doc_id] = keyword_weight * result['similarity']

        # Sort by combined score
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Get top-k results
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = self.documents[doc_id]
            results.append({
                'content': doc['content'],
                'metadata': doc['metadata'],
                'similarity': score,
                'id': doc_id
            })

        return results

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simple keyword-based search"""
        query_terms = query.lower().split()

        results = []
        for doc in self.documents:
            content_lower = doc['content'].lower()
            score = sum(1 for term in query_terms if term in content_lower)

            if score > 0:
                results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'similarity': score / len(query_terms),
                    'id': doc['id']
                })

        # Sort by score
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

# Practice Exercise 2.1: Enhanced RAG with Metadata
def exercise_enhanced_rag():
    """Practice exercise: Enhanced RAG with metadata"""

    # Documents with metadata
    documents = [
        "Python variables are containers for storing data values.",
        "JavaScript variables can be declared using var, let, or const.",
        "Python functions are defined using the def keyword.",
        "JavaScript functions can be declared using function keyword or arrow syntax.",
        "Python lists are ordered, mutable collections.",
        "JavaScript arrays are ordered collections that can contain different data types.",
        "Python dictionaries store key-value pairs.",
        "JavaScript objects are collections of properties and methods."
    ]

    metadata = [
        {'language': 'python', 'topic': 'variables'},
        {'language': 'javascript', 'topic': 'variables'},
        {'language': 'python', 'topic': 'functions'},
        {'language': 'javascript', 'topic': 'functions'},
        {'language': 'python', 'topic': 'lists'},
        {'language': 'javascript', 'topic': 'arrays'},
        {'language': 'python', 'topic': 'dictionaries'},
        {'language': 'javascript', 'topic': 'objects'}
    ]

    # Create enhanced RAG
    rag = EnhancedRAG()

    # Add documents with metadata
    rag.add_documents(documents, metadata)

    # Index documents
    rag.index_documents()

    print("=== Enhanced RAG System Demo ===\\n")

    # Test basic retrieval
    print("1. Basic retrieval:")
    query = "How do you create variables?"
    results = rag.retrieve(query, top_k=3)

    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['metadata']}] {result['content']}")
    print()

    # Test metadata filtering
    print("2. Filtered retrieval (Python only):")
    python_results = rag.retrieve_with_filter(query, {'language': 'python'}, top_k=3)

    for i, result in enumerate(python_results, 1):
        print(f"{i}. [{result['metadata']}] {result['content']}")
    print()

    # Test hybrid search
    print("3. Hybrid search:")
    hybrid_results = rag.hybrid_search(query, top_k=3)

    for i, result in enumerate(hybrid_results, 1):
        print(f"{i}. [{result['metadata']}] {result['content']}")
    print()

    return rag

# Run the exercise
# enhanced_rag = exercise_enhanced_rag()
```

## Vector Database Integration {#vector-db}

### Exercise 3: Vector Database Implementation

```python
import sqlite3
import json
from typing import Tuple, List
import numpy as np

class VectorDatabase:
    """Simple vector database implementation using SQLite"""

    def __init__(self, db_path: str = "vector_db.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                document_id INTEGER,
                vector BLOB,
                model_name TEXT,
                dimension INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')

        # Create index for faster similarity search
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_document_id
            ON embeddings (document_id)
        ''')

        conn.commit()
        conn.close()

    def add_document(self, content: str, metadata: Dict = None,
                    embedding: np.ndarray = None, model_name: str = 'unknown') -> int:
        """Add document to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert document
        metadata_str = json.dumps(metadata) if metadata else None
        cursor.execute('''
            INSERT INTO documents (content, metadata)
            VALUES (?, ?)
        ''', (content, metadata_str))

        doc_id = cursor.lastrowid

        # Insert embedding if provided
        if embedding is not None:
            embedding_blob = embedding.tobytes()
            cursor.execute('''
                INSERT INTO embeddings (document_id, vector, model_name, dimension)
                VALUES (?, ?, ?, ?)
            ''', (doc_id, embedding_blob, model_name, len(embedding)))

        conn.commit()
        conn.close()

        return doc_id

    def get_document(self, doc_id: int) -> Tuple[str, Dict, np.ndarray]:
        """Get document by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get document
        cursor.execute('''
            SELECT d.content, d.metadata, e.vector, e.model_name
            FROM documents d
            LEFT JOIN embeddings e ON d.id = e.document_id
            WHERE d.id = ?
        ''', (doc_id,))

        result = cursor.fetchone()
        conn.close()

        if not result:
            return None, None, None

        content, metadata_str, vector_blob, model_name = result

        metadata = json.loads(metadata_str) if metadata_str else {}

        # Reconstruct embedding
        embedding = None
        if vector_blob:
            embedding = np.frombuffer(vector_blob, dtype=np.float32)

        return content, metadata, embedding

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5,
                      model_name: str = None) -> List[Tuple[int, str, Dict, float]]:
        """Search for similar documents using cosine similarity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        query = '''
            SELECT d.id, d.content, d.metadata, e.vector
            FROM documents d
            JOIN embeddings e ON d.id = e.document_id
        '''

        params = []
        if model_name:
            query += ' WHERE e.model_name = ?'
            params.append(model_name)

        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        # Calculate similarities
        similarities = []
        for doc_id, content, metadata_str, vector_blob in results:
            if vector_blob:
                doc_embedding = np.frombuffer(vector_blob, dtype=np.float32)
                similarity = self.cosine_similarity(query_embedding, doc_embedding)

                metadata = json.loads(metadata_str) if metadata_str else {}
                similarities.append((doc_id, content, metadata, similarity))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[3], reverse=True)
        return similarities[:top_k]

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Count documents
        cursor.execute('SELECT COUNT(*) FROM documents')
        doc_count = cursor.fetchone()[0]

        # Count embeddings
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        embedding_count = cursor.fetchone()[0]

        conn.close()

        return {
            'documents': doc_count,
            'embeddings': embedding_count,
            'database_size': f"{doc_count +"
        }

# embedding_count} records Practice Exercise 3.1: Vector Database Usage
def exercise_vector_database():
    """Practice exercise: Using vector database"""

    # Initialize vector database
    vector_db = VectorDatabase('practice_db.db')

    # Sample data
    documents = [
        "The capital of France is Paris, known for its art, culture, and cuisine.",
        "Python is a versatile programming language used for web development, data science, and automation.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Natural language processing helps computers understand and generate human language.",
        "Deep learning uses neural networks with multiple layers to model complex patterns in data."
    ]

    metadata = [
        {'category': 'geography', 'country': 'france'},
        {'category': 'technology', 'topic': 'programming'},
        {'category': 'technology', 'topic': 'ai'},
        {'category': 'technology', 'topic': 'nlp'},
        {'category': 'technology', 'topic': 'deep_learning'}
    ]

    print("=== Vector Database Demo ===\\n")

    # Add documents with embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    for i, (content, meta) in enumerate(zip(documents, metadata)):
        embedding = model.encode([content])[0]
        doc_id = vector_db.add_document(content, meta, embedding, 'all-MiniLM-L6-v2')
        print(f"Added document {doc_id}: {content[:50]}...")

    print(f"\\nDatabase stats: {vector_db.get_stats()}\\n")

    # Search for similar documents
    queries = [
        "What programming languages are popular?",
        "Tell me about artificial intelligence",
        "What is the capital of France?"
    ]

    for query in queries:
        print(f"Query: {query}")
        query_embedding = model.encode([query])[0]
        results = vector_db.search_similar(query_embedding, top_k=2)

        for doc_id, content, metadata, similarity in results:
            print(f"  [{similarity:.3f}] {content}")
        print()

    return vector_db

# Run the exercise
# vector_db = exercise_vector_database()
```

## Document Processing {#document-processing}

### Exercise 4: Document Chunking and Processing

```python
import re
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

class DocumentProcessor:
    """Advanced document processing for RAG systems"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.nlp = None

        # Initialize NLP pipeline
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")

    def chunk_document(self, text: str, method: str = 'fixed') -> List[Dict]:
        """Chunk document into smaller pieces"""

        if method == 'fixed':
            return self.fixed_size_chunking(text)
        elif method == 'sentence':
            return self.sentence_based_chunking(text)
        elif method == 'semantic':
            return self.semantic_chunking(text)
        elif method == 'paragraph':
            return self.paragraph_chunking(text)
        else:
            raise ValueError(f"Unknown chunking method: {method}")

    def fixed_size_chunking(self, text: str) -> List[Dict]:
        """Chunk text into fixed-size pieces"""
        chunks = []
        words = text.split()

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunks.append({
                'id': len(chunks),
                'content': chunk_text,
                'word_count': len(chunk_words),
                'start_pos': i,
                'end_pos': i + len(chunk_words)
            })

        return chunks

    def sentence_based_chunking(self, text: str) -> List[Dict]:
        """Chunk text based on sentences"""
        chunks = []
        sentences = sent_tokenize(text)
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # Check if adding this sentence would exceed chunk size
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                # Create chunk with overlap
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'id': len(chunks),
                    'content': chunk_text,
                    'word_count': current_word_count,
                    'sentence_count': len(current_chunk)
                })

                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self.chunk_overlap//10:] if len(current_chunk) > 1 else []
                current_chunk = overlap_sentences + [sentence]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'id': len(chunks),
                'content': chunk_text,
                'word_count': current_word_count,
                'sentence_count': len(current_chunk)
            })

        return chunks

    def semantic_chunking(self, text: str) -> List[Dict]:
        """Chunk text using semantic boundaries"""
        if not self.nlp:
            return self.sentence_based_chunking(text)

        chunks = []
        doc = self.nlp(text)

        current_chunk = []
        current_word_count = 0

        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_words = len(sent_text.split())

            # Check for semantic boundaries
            semantic_break = self.is_semantic_boundary(sent)

            if (current_word_count + sent_words > self.chunk_size or semantic_break) and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'id': len(chunks),
                    'content': chunk_text,
                    'word_count': current_word_count,
                    'semantic_boundary': semantic_break
                })

                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self.chunk_overlap//10:] if len(current_chunk) > 1 else []
                current_chunk = overlap_sentences + [sent_text]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sent_text)
                current_word_count += sent_words

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'id': len(chunks),
                'content': chunk_text,
                'word_count': current_word_count,
                'semantic_boundary': False
            })

        return chunks

    def is_semantic_boundary(self, sentence) -> bool:
        """Determine if sentence is a semantic boundary"""
        # Check for strong semantic indicators
        boundary_indicators = [
            'however', 'therefore', 'moreover', 'furthermore', 'in conclusion',
            'on the other hand', 'meanwhile', 'subsequently', 'as a result',
            'in summary', 'to summarize', 'in closing'
        ]

        sentence_text = sentence.text.lower()
        return any(indicator in sentence_text for indicator in boundary_indicators)

    def paragraph_chunking(self, text: str) -> List[Dict]:
        """Chunk text by paragraphs"""
        chunks = []
        paragraphs = text.split('\\n\\n')

        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                chunks.append({
                    'id': len(chunks),
                    'content': paragraph.strip(),
                    'word_count': len(paragraph.split()),
                    'paragraph_number': i + 1
                })

        return chunks

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text"""
        if not self.nlp:
            # Simple keyword extraction without spaCy
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text.lower())
            keywords = [word for word in words if word.isalpha() and word not in stop_words]

            from collections import Counter
            word_freq = Counter(keywords)
            return [word for word, freq in word_freq.most_common(top_k)]

        # Use spaCy for better keyword extraction
        doc = self.nlp(text)
        keywords = []

        # Extract named entities
        for ent in doc.ents:
            keywords.append(ent.text)

        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Short noun phrases
                keywords.append(chunk.text)

        # Extract important words
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and
                not token.is_stop and
                len(token.text) > 3):
                keywords.append(token.lemma_)

        # Remove duplicates and return top-k
        unique_keywords = list(set(keywords))
        return unique_keywords[:top_k]

    def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate simple extractive summary"""
        sentences = sent_tokenize(text)

        if len(sentences) <= max_sentences:
            return text

        # Score sentences based on word frequency
        word_freq = {}
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            for word in words:
                if word.isalpha() and word not in stopwords.words('english'):
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Score each sentence
        sentence_scores = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            score = sum(word_freq.get(word, 0) for word in words)
            sentence_scores.append((sentence, score))

        # Select top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent for sent, score in sentence_scores[:max_sentences]]

        # Maintain original order
        ordered_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                ordered_sentences.append(sentence)

        return ' '.join(ordered_sentences)

# Practice Exercise 4.1: Document Processing
def exercise_document_processing():
    """Practice exercise: Document processing and chunking"""

    # Sample document
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior. Machine learning is a subset of AI that focuses on algorithms that can learn from data.

    Deep learning is a specialized form of machine learning that uses neural networks with multiple layers. These networks can automatically discover patterns in large datasets without explicit programming.

    Natural language processing (NLP) is another important area of AI that deals with the interaction between computers and human language. NLP enables machines to understand, interpret, and generate human language.

    Computer vision is the field of AI that enables computers to derive meaningful information from digital images, videos, and other visual inputs. This technology has applications in autonomous vehicles, medical imaging, and facial recognition.

    However, AI also raises important ethical considerations. Issues such as bias in algorithms, privacy concerns, and the impact on employment need to be carefully addressed as AI systems become more prevalent.
    """

    processor = DocumentProcessor(chunk_size=200, chunk_overlap=30)

    print("=== Document Processing Demo ===\\n")

    # Test different chunking methods
    methods = ['fixed', 'sentence', 'paragraph']

    for method in methods:
        print(f"{method.capitalize()} chunking:")
        chunks = processor.chunk_document(sample_text, method)

        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"  Chunk {i+1}: {chunk['content'][:100]}...")
            print(f"    Words: {chunk.get('word_count', 'N/A')}")
        print()

    # Extract keywords
    keywords = processor.extract_keywords(sample_text, top_k=10)
    print(f"Keywords: {', '.join(keywords)}")
    print()

    # Generate summary
    summary = processor.generate_summary(sample_text, max_sentences=3)
    print(f"Summary: {summary}")
    print()

    return processor

# Run the exercise
# processor = exercise_document_processing()
```

## Retrieval Strategies {#retrieval-strategies}

### Exercise 5: Advanced Retrieval Strategies

```python
import numpy as np
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    document_id: int
    content: str
    metadata: Dict
    score: float
    strategy: str

class AdvancedRetrieval:
    """Advanced retrieval strategies for RAG systems"""

    def __init__(self, documents: List[Dict], embeddings: np.ndarray):
        self.documents = documents
        self.embeddings = embeddings
        self.document_vectors = embeddings

    def bm25_retrieval(self, query: str, top_k: int = 5, k1: float = 1.5, b: float = 0.75) -> List[RetrievalResult]:
        """BM25 retrieval algorithm"""

        # Tokenize query
        query_terms = query.lower().split()

        # Calculate document frequencies
        doc_freq = {}
        for doc in self.documents:
            doc_text = doc['content'].lower()
            unique_terms = set(doc_text.split())
            for term in query_terms:
                if term in unique_terms:
                    doc_freq[term] = doc_freq.get(term, 0) + 1

        # Calculate average document length
        avg_doc_length = np.mean([len(doc['content'].split()) for doc in self.documents])

        # Calculate BM25 scores
        scores = []
        for i, doc in enumerate(self.documents):
            doc_text = doc['content'].lower()
            doc_length = len(doc_text.split())

            score = 0
            for term in query_terms:
                if term in doc_text:
                    # Term frequency in document
                    tf = doc_text.count(term)

                    # Document frequency
                    df = doc_freq.get(term, 0)

                    # BM25 formula
                    idf = np.log((len(self.documents) - df + 0.5) / (df + 0.5))
                    k1_term = k1 * (1 - b + b * (doc_length / avg_doc_length))
                    bm25_score = idf * ((tf * (k1 + 1)) / (tf + k1_term))

                    score += bm25_score

            scores.append(score)

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                document_id=self.documents[idx]['id'],
                content=self.documents[idx]['content'],
                metadata=self.documents[idx]['metadata'],
                score=scores[idx],
                strategy='bm25'
            ))

        return results

    def multistage_retrieval(self, query: str, stages: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """Multi-stage retrieval combining different strategies"""

        current_candidates = list(range(len(self.documents)))
        current_results = []

        for stage in stages:
            if stage == 'bm25':
                stage_results = self.bm25_retrieval(query, top_k=len(current_candidates))
            elif stage == 'semantic':
                stage_results = self.semantic_retrieval(query, top_k=len(current_candidates))
            elif stage == 'keyword':
                stage_results = self.keyword_retrieval(query, top_k=len(current_candidates))
            else:
                raise ValueError(f"Unknown retrieval stage: {stage}")

            # Filter candidates based on stage results
            stage_doc_ids = [r.document_id for r in stage_results]
            current_candidates = [i for i in current_candidates
                                if self.documents[i]['id'] in stage_doc_ids]

            current_results = stage_results

            # Early termination if we have enough candidates
            if len(current_candidates) <= top_k:
                break

        # Return final results
        return current_results[:top_k]

    def semantic_retrieval(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Semantic retrieval using embeddings"""
        from sentence_transformers import SentenceTransformer

        # Generate query embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])

        # Calculate similarities
        similarities = np.dot(self.document_vectors, query_embedding.T).flatten()

        # Normalize
        doc_norms = np.linalg.norm(self.document_vectors, axis=1)
        query_norm = np.linalg.norm(query_embedding)

        if doc_norms.any() and query_norm > 0:
            similarities = similarities / (doc_norms * query_norm)

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                document_id=self.documents[idx]['id'],
                content=self.documents[idx]['content'],
                metadata=self.documents[idx]['metadata'],
                score=similarities[idx],
                strategy='semantic'
            ))

        return results

    def keyword_retrieval(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Keyword-based retrieval"""
        query_terms = set(query.lower().split())

        scores = []
        for doc in self.documents:
            doc_text = set(doc['content'].lower().split())
            overlap = len(query_terms.intersection(doc_text))
            score = overlap / len(query_terms) if query_terms else 0
            scores.append(score)

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                document_id=self.documents[idx]['id'],
                content=self.documents[idx]['content'],
                metadata=self.documents[idx]['metadata'],
                score=scores[idx],
                strategy='keyword'
            ))

        return results

    def learned_retrieval(self, query: str, query_encoder: Callable,
                         document_encoder: Callable, top_k: int = 5) -> List[RetrievalResult]:
        """Learned retrieval using custom encoders"""

        # Encode query and documents
        query_embedding = query_encoder(query)
        document_embeddings = document_encoder([doc['content'] for doc in self.documents])

        # Calculate similarities
        similarities = np.dot(document_embeddings, query_embedding.T).flatten()

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                document_id=self.documents[idx]['id'],
                content=self.documents[idx]['content'],
                metadata=self.documents[idx]['metadata'],
                score=similarities[idx],
                strategy='learned'
            ))

        return results

    def diversity_aware_retrieval(self, query: str, top_k: int = 5,
                                 diversity_threshold: float = 0.8) -> List[RetrievalResult]:
        """Diversity-aware retrieval to avoid redundant results"""

        # Get initial candidates using semantic retrieval
        candidates = self.semantic_retrieval(query, top_k=top_k*3)

        results = []
        selected_embeddings = []

        for candidate in candidates:
            # Check diversity with already selected results
            if not selected_embeddings:
                # First result - always include
                results.append(candidate)
                selected_embeddings.append(candidate.metadata.get('embedding', np.zeros(384)))
            else:
                # Check diversity
                candidate_embedding = candidate.metadata.get('embedding', np.zeros(384))

                max_similarity = max([
                    np.dot(candidate_embedding, selected_emb) /
                    (np.linalg.norm(candidate_embedding) * np.linalg.norm(selected_emb))
                    for selected_emb in selected_embeddings
                ])

                # Include if diverse enough
                if max_similarity < diversity_threshold:
                    results.append(candidate)
                    selected_embeddings.append(candidate_embedding)

            # Stop if we have enough results
            if len(results) >= top_k:
                break

        return results

    def query_expansion_retrieval(self, query: str, expansion_terms: List[str] = None,
                                top_k: int = 5) -> List[RetrievalResult]:
        """Retrieval with query expansion"""

        # Expand query
        if expansion_terms is None:
            # Simple expansion using synonyms or related terms
            expansion_terms = self.expand_query_simple(query)

        expanded_query = query + ' ' + ' '.join(expansion_terms)

        # Use semantic retrieval on expanded query
        return self.semantic_retrieval(expanded_query, top_k)

    def expand_query_simple(self, query: str) -> List[str]:
        """Simple query expansion"""
        # This is a simplified expansion - in practice, you'd use more sophisticated methods
        expansions = {
            'ai': ['artificial intelligence', 'machine learning', 'neural networks'],
            'computer': ['computing', 'technology', 'software'],
            'data': ['information', 'dataset', 'database'],
            'learning': ['training', 'education', 'study']
        }

        expanded_terms = []
        query_words = query.lower().split()

        for word in query_words:
            if word in expansions:
                expanded_terms.extend(expansions[word])

        return expanded_terms

# Practice Exercise 5.1: Advanced Retrieval Strategies
def exercise_advanced_retrieval():
    """Practice exercise: Advanced retrieval strategies"""

    # Sample documents
    documents = [
        {
            'id': 1,
            'content': 'Artificial intelligence is the simulation of human intelligence in machines.',
            'metadata': {'topic': 'ai', 'category': 'technology'}
        },
        {
            'id': 2,
            'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
            'metadata': {'topic': 'ml', 'category': 'technology'}
        },
        {
            'id': 3,
            'content': 'Deep learning uses neural networks with multiple layers to process data.',
            'metadata': {'topic': 'dl', 'category': 'technology'}
        },
        {
            'id': 4,
            'content': 'Natural language processing helps computers understand human language.',
            'metadata': {'topic': 'nlp', 'category': 'technology'}
        },
        {
            'id': 5,
            'content': 'Computer vision enables machines to interpret and understand visual information.',
            'metadata': {'topic': 'cv', 'category': 'technology'}
        }
    ]

    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([doc['content'] for doc in documents])

    # Add embeddings to metadata
    for i, doc in enumerate(documents):
        doc['metadata']['embedding'] = embeddings[i]

    # Initialize advanced retrieval
    retriever = AdvancedRetrieval(documents, embeddings)

    query = "What is artificial intelligence?"

    print("=== Advanced Retrieval Strategies Demo ===\\n")

    # Test different retrieval methods
    methods = [
        ('semantic', 'Semantic Retrieval'),
        ('bm25', 'BM25 Retrieval'),
        ('keyword', 'Keyword Retrieval'),
        ('multistage', 'Multi-stage Retrieval'),
        ('diversity', 'Diversity-aware Retrieval'),
        ('expansion', 'Query Expansion Retrieval')
    ]

    for method, name in methods:
        print(f"{name}:")

        if method == 'semantic':
            results = retriever.semantic_retrieval(query, top_k=3)
        elif method == 'bm25':
            results = retriever.bm25_retrieval(query, top_k=3)
        elif method == 'keyword':
            results = retriever.keyword_retrieval(query, top_k=3)
        elif method == 'multistage':
            results = retriever.multistage_retrieval(query, ['bm25', 'semantic'], top_k=3)
        elif method == 'diversity':
            results = retriever.diversity_aware_retrieval(query, top_k=3)
        elif method == 'expansion':
            results = retriever.query_expansion_retrieval(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result.score:.3f}] {result.content}")
        print()

    return retriever

# Run the exercise
# retriever = exercise_advanced_retrieval()
```

This comprehensive RAG practice guide provides hands-on exercises covering all essential aspects of RAG systems. Each exercise builds upon the previous ones, providing a complete learning path from basic concepts to advanced implementations.
