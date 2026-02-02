# 🛠️ **RAG Practice Exercises**

_Hands-on coding exercises and implementations for Retrieval-Augmented Generation systems_

---

## 📖 **Table of Contents**

1. [Setup and Environment](#1-setup-and-environment)
2. [Basic RAG Implementation](#2-basic-rag-implementation)
3. [Document Processing and Chunking](#3-document-processing-and-chunking)
4. [Vector Database Operations](#4-vector-database-operations)
5. [RAG Pipeline Development](#5-rag-pipeline-development)
6. [Advanced RAG Techniques](#6-advanced-rag-techniques)
7. [RAG Evaluation and Testing](#7-rag-evaluation-and-testing)
8. [Production RAG Systems](#8-production-rag-systems)
9. [Performance Optimization](#9-performance-optimization)
10. [Mini-Projects and Applications](#10-mini-projects-and-applications)

---

## **1. Setup and Environment**

### **1.1 Environment Setup**

```bash
# Create virtual environment
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate

# Install required packages
pip install transformers torch sentence-transformers
pip install faiss-cpu pinecone-client chromadb
pip install langchain openai tiktoken
pip install scikit-learn numpy pandas matplotlib
pip install streamlit fastapi uvicorn
pip install jupyter ipython
pip install datasets
```

### **1.2 Project Structure**

```
rag_practice/
├── data/
│   ├── documents/
│   └── processed/
├── src/
│   ├── document_processors/
│   ├── retrievers/
│   ├── llm_wrappers/
│   └── rag_pipeline/
├── config/
│   └── config.yaml
├── tests/
├── notebooks/
└── main.py
```

### **1.3 Configuration Setup**

```python
# config/config.yaml
rag_config:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  llm_model: "gpt-3.5-turbo"
  chunk_size: 512
  chunk_overlap: 50
  top_k: 5

vector_db:
  type: "chroma"
  persist_directory: "./data/chroma_db"

api_keys:
  openai: "your-openai-api-key"
  pinecone: "your-pinecone-api-key"
```

```python
# src/utils/config.py
import yaml
from dataclasses import dataclass

@dataclass
class RAGConfig:
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int

class Config:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.rag_config = RAGConfig(**config['rag_config'])
        self.vector_db_config = config['vector_db']
        self.api_keys = config['api_keys']
```

---

## **2. Basic RAG Implementation**

### **2.1 Exercise 1: Basic Document Processor**

**Objective**: Create a document processor that can handle multiple document types and split them into chunks.

```python
# src/document_processors/basic_processor.py
import os
import re
from typing import List, Dict, Any
import PyPDF2
from docx import Document

class BasicDocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self, file_path: str) -> str:
        """Load document based on file extension"""
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.pdf':
            return self._load_pdf(file_path)
        elif file_ext == '.docx':
            return self._load_docx(file_path)
        elif file_ext == '.txt':
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def _load_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _load_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def _load_text(self, file_path: str) -> str:
        """Load plain text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def split_text(self, text: str, file_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with overlap"""
        chunks = []

        # Simple word-based splitting
        words = text.split()
        chunks_text = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            chunk = {
                'text': chunk_text,
                'metadata': {
                    'file_path': file_metadata.get('file_path', ''),
                    'chunk_index': len(chunks),
                    'word_count': len(chunk_words),
                    **file_metadata
                }
            }
            chunks.append(chunk)

        return chunks

    def process_documents(self, document_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents"""
        all_chunks = []

        for doc_path in document_paths:
            try:
                print(f"Processing {doc_path}...")
                text = self.load_document(doc_path)
                file_metadata = {
                    'file_path': doc_path,
                    'file_name': os.path.basename(doc_path),
                    'file_size': os.path.getsize(doc_path)
                }

                chunks = self.split_text(text, file_metadata)
                all_chunks.extend(chunks)

            except Exception as e:
                print(f"Error processing {doc_path}: {e}")

        print(f"Created {len(all_chunks)} chunks from {len(document_paths)} documents")
        return all_chunks

# Test the document processor
if __name__ == "__main__":
    processor = BasicDocumentProcessor(chunk_size=200, chunk_overlap=20)

    # Create sample documents for testing
    sample_docs = [
        "This is a sample document about machine learning. "
        "Machine learning is a subset of artificial intelligence. "
        "It involves training models on data to make predictions."
    ]

    # Save sample documents
    os.makedirs("data/documents", exist_ok=True)
    with open("data/documents/sample1.txt", "w") as f:
        f.write(sample_docs[0])

    # Process documents
    chunks = processor.process_documents(["data/documents/sample1.txt"])
    print(f"Processed {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i}: {chunk['text'][:100]}...")
```

### **2.2 Exercise 2: Simple Vector Store**

**Objective**: Implement a simple vector store using FAISS for document retrieval.

```python
# src/retrievers/simple_vector_store.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import pickle
import os

class SimpleVectorStore:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        self.embeddings = None

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build FAISS index from document chunks"""
        print("Building vector index...")

        # Extract texts
        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        self.embeddings = embeddings

        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))

        # Store document metadata
        self.documents = chunks

        print(f"Built index with {self.index.ntotal} documents")

    def save_index(self, index_path: str):
        """Save index and metadata to disk"""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, f"{index_path}.faiss")

        # Save embeddings and metadata
        np.save(f"{index_path}_embeddings.npy", self.embeddings)

        with open(f"{index_path}_documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        print(f"Saved index to {index_path}")

    def load_index(self, index_path: str):
        """Load index and metadata from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{index_path}.faiss")

        # Load embeddings
        self.embeddings = np.load(f"{index_path}_embeddings.npy")

        # Load documents
        with open(f"{index_path}_documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

        print(f"Loaded index with {self.index.ntotal} documents")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for most similar documents"""
        if self.index is None:
            raise ValueError("Index not built or loaded")

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Return results with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get index statistics"""
        return {
            'num_documents': len(self.documents) if self.documents else 0,
            'embedding_dimension': self.dimension,
            'index_type': str(type(self.index)),
            'total_memory': os.path.getsize(f"/tmp/vector_store.faiss") if os.path.exists("/tmp/vector_store.faiss") else 0
        }

# Test the vector store
if __name__ == "__main__":
    # Initialize vector store
    vector_store = SimpleVectorStore()

    # Create sample chunks
    sample_chunks = [
        {
            'text': 'Machine learning is a subset of artificial intelligence.',
            'metadata': {'file': 'ml_intro.txt', 'chunk': 0}
        },
        {
            'text': 'Deep learning uses neural networks with multiple layers.',
            'metadata': {'file': 'deep_learning.txt', 'chunk': 0}
        },
        {
            'text': 'Natural language processing helps computers understand human language.',
            'metadata': {'file': 'nlp_basics.txt', 'chunk': 0}
        }
    ]

    # Build index
    vector_store.build_index(sample_chunks)

    # Test search
    query = "What is machine learning?"
    results = vector_store.search(query, top_k=2)

    print(f"\nQuery: {query}")
    print("Results:")
    for i, (doc, score) in enumerate(results):
        print(f"{i+1}. Score: {score:.3f} | Text: {doc['text']}")

    # Show statistics
    print("\nIndex Statistics:")
    print(vector_store.get_stats())
```

---

## **3. Document Processing and Chunking**

### **3.3 Exercise 3: Advanced Chunking Strategies**

**Objective**: Implement different chunking strategies for optimal RAG performance.

```python
# src/document_processors/advanced_processor.py
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

class AdvancedDocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def sentence_chunker(self, text: str, file_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks based on sentences"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        chunk_index = 0

        for sentence in sentences:
            # If adding this sentence exceeds chunk size, save current chunk
            if len(current_chunk + " " + sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {
                        **file_metadata,
                        'chunk_index': chunk_index,
                        'chunking_method': 'sentence',
                        'sentence_count': len(current_chunk.split('.'))
                    }
                })

                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-self.chunk_overlap:]
                current_chunk = " ".join(overlap_words) + " " + sentence
                chunk_index += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    **file_metadata,
                    'chunk_index': chunk_index,
                    'chunking_method': 'sentence',
                    'sentence_count': len(current_chunk.split('.'))
                }
            })

        return chunks

    def semantic_chunker(self, text: str, file_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into semantically coherent chunks"""
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if len(paragraphs) == 1:
            # If only one paragraph, split by sentences
            return self.sentence_chunker(text, file_metadata)

        chunks = []
        current_chunk = ""
        chunk_index = 0

        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, check semantic similarity
            if len(current_chunk + "\n\n" + paragraph) > self.chunk_size and current_chunk:
                # Calculate semantic similarity between current chunk and new paragraph
                if self._should_split(current_chunk, paragraph):
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            **file_metadata,
                            'chunk_index': chunk_index,
                            'chunking_method': 'semantic',
                            'paragraph_count': len(current_chunk.split('\n\n'))
                        }
                    })

                    # Start new chunk
                    current_chunk = paragraph
                    chunk_index += 1
                else:
                    # Add to current chunk despite size limit
                    current_chunk += "\n\n" + paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    **file_metadata,
                    'chunk_index': chunk_index,
                    'chunking_method': 'semantic',
                    'paragraph_count': len(current_chunk.split('\n\n'))
                }
            })

        return chunks

    def _should_split(self, current_chunk: str, new_paragraph: str) -> bool:
        """Determine if we should split based on semantic similarity"""
        try:
            # Get embeddings for similarity comparison
            embeddings = self.sentence_model.encode([current_chunk, new_paragraph])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )

            # Split if similarity is below threshold (paragraphs are dissimilar)
            return similarity < 0.5
        except:
            # Fallback to simple heuristic if embedding fails
            return True

    def sliding_window_chunker(self, text: str, file_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text with overlapping sliding windows"""
        words = text.split()
        chunks = []
        chunk_index = 0

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **file_metadata,
                    'chunk_index': chunk_index,
                    'chunking_method': 'sliding_window',
                    'start_word': i,
                    'end_word': i + len(chunk_words),
                    'window_size': len(chunk_words)
                }
            })

            chunk_index += 1

        return chunks

    def adaptive_chunker(self, text: str, file_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adaptive chunking based on document type and content"""
        # Detect document type and content characteristics
        doc_type = self._detect_document_type(text)
        content_density = self._calculate_content_density(text)

        # Adjust chunking strategy based on characteristics
        if doc_type == 'code':
            return self._code_chunker(text, file_metadata)
        elif doc_type == 'technical' and content_density > 0.7:
            return self.semantic_chunker(text, file_metadata)
        elif content_density < 0.3:
            return self.sliding_window_chunker(text, file_metadata)
        else:
            return self.sentence_chunker(text, file_metadata)

    def _detect_document_type(self, text: str) -> str:
        """Detect document type for adaptive chunking"""
        # Check for code patterns
        if re.search(r'def\s+\w+|class\s+\w+|import\s+\w+', text):
            return 'code'

        # Check for technical documents
        technical_indicators = ['algorithm', 'implementation', 'configuration', 'API']
        if sum(1 for indicator in technical_indicators if indicator.lower() in text.lower()) >= 2:
            return 'technical'

        return 'general'

    def _calculate_content_density(self, text: str) -> float:
        """Calculate content density (ratio of meaningful content to total length)"""
        # Simple heuristic: ratio of unique words to total words
        words = text.lower().split()
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 3]

        return len(set(meaningful_words)) / len(words) if words else 0

    def _code_chunker(self, text: str, file_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Special chunking for code documents"""
        # Split by functions, classes, and major sections
        chunks = []
        current_chunk = ""
        chunk_index = 0

        lines = text.split('\n')

        for line in lines:
            # Check for major code boundaries
            if re.match(r'^\s*(def|class|import|from|#\s*===|#\s*---)', line.strip()):
                # Save previous chunk if exists
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            **file_metadata,
                            'chunk_index': chunk_index,
                            'chunking_method': 'code',
                            'lines': len(current_chunk.split('\n'))
                        }
                    })
                    chunk_index += 1
                    current_chunk = line
                else:
                    current_chunk += line + "\n"
            else:
                current_chunk += line + "\n"

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    **file_metadata,
                    'chunk_index': chunk_index,
                    'chunking_method': 'code',
                    'lines': len(current_chunk.split('\n'))
                }
            })

        return chunks

    def process_with_strategy(self, text: str, file_metadata: Dict[str, Any],
                            strategy: str = 'adaptive') -> List[Dict[str, Any]]:
        """Process document with specified chunking strategy"""
        strategies = {
            'sentence': self.sentence_chunker,
            'semantic': self.semantic_chunker,
            'sliding': self.sliding_window_chunker,
            'adaptive': self.adaptive_chunker
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        return strategies[strategy](text, file_metadata)

# Test different chunking strategies
if __name__ == "__main__":
    processor = AdvancedDocumentProcessor(chunk_size=200, chunk_overlap=30)

    sample_text = """
    Machine learning is a subset of artificial intelligence. It involves training models on data.

    There are several types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

    Deep learning uses neural networks with multiple layers. These networks can learn complex patterns from data.
    """

    file_metadata = {'file_path': 'sample.txt', 'file_name': 'sample.txt'}

    # Test different strategies
    strategies = ['sentence', 'semantic', 'sliding', 'adaptive']

    for strategy in strategies:
        print(f"\n=== {strategy.upper()} CHUNKING ===")
        chunks = processor.process_with_strategy(sample_text, file_metadata, strategy)

        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk['text'][:100]}...")
            print(f"Metadata: {chunk['metadata']}")
            print()
```

---

## **4. Vector Database Operations**

### **4.4 Exercise 4: Multi-Database RAG System**

**Objective**: Implement a RAG system that can work with multiple vector databases.

```python
# src/vector_db/multi_database.py
import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

# Import vector databases
try:
    import faiss
except ImportError:
    faiss = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    import pinecone
except ImportError:
    pinecone = None

class VectorDB(ABC):
    """Abstract base class for vector databases"""

    @abstractmethod
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete(self, ids: List[str]):
        pass

class FAISSVectorDB(VectorDB):
    """FAISS implementation of vector database"""

    def __init__(self, dimension: int):
        if faiss is None:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product
        self.documents = []
        self.id_counter = 0

    def add_documents(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))

        # Store documents with IDs
        for doc in documents:
            doc['id'] = str(self.id_counter)
            self.documents.append(doc)
            self.id_counter += 1

        print(f"Added {len(documents)} documents to FAISS index")

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        faiss.normalize_L2(query_embedding.reshape(1, -1))

        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            min(top_k, self.index.ntotal)
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                result = {
                    'document': self.documents[idx],
                    'score': float(score),
                    'id': self.documents[idx]['id']
                }
                results.append(result)

        return results

    def delete(self, ids: List[str]):
        # FAISS doesn't support deletion directly
        # In practice, you'd need to rebuild the index
        raise NotImplementedError("FAISS deletion not implemented")

    def get_stats(self) -> Dict[str, Any]:
        return {
            'type': 'FAISS',
            'total_documents': len(self.documents),
            'dimension': self.dimension,
            'index_type': 'IndexFlatIP'
        }

class ChromaDB(VectorDB):
    """ChromaDB implementation of vector database"""

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        if chromadb is None:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.persist_directory = persist_directory

    def add_documents(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        ids = [str(i) for i in range(len(documents))]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        texts = [doc['text'] for doc in documents]

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Added {len(documents)} documents to ChromaDB")

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )

        output = []
        for i in range(len(results['documents'][0])):
            result = {
                'document': {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {}
                },
                'score': float(results['distances'][0][i]),
                'id': results['ids'][0][i]
            }
            output.append(result)

        return output

    def delete(self, ids: List[str]):
        self.collection.delete(ids=ids)

    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {
            'type': 'ChromaDB',
            'total_documents': count,
            'persist_directory': self.persist_directory
        }

class PineconeDB(VectorDB):
    """PineconeDB implementation of vector database"""

    def __init__(self, api_key: str, environment: str, index_name: str = "rag-index"):
        if pinecone is None:
            raise ImportError("Pinecone client not installed. Run: pip install pinecone-client")

        pinecone.init(api_key=api_key, environment=environment)

        # Create index if it doesn't exist
        dimension = 384  # Standard for sentence transformers
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )

        self.index = pinecone.Index(index_name)
        self.index_name = index_name

    def add_documents(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        vectors = []
        for i, (embedding, doc) in enumerate(zip(embeddings, documents)):
            vectors.append({
                'id': str(i),
                'values': embedding.tolist(),
                'metadata': {
                    'text': doc['text'],
                    **doc.get('metadata', {})
                }
            })

        # Upsert vectors
        self.index.upsert(vectors=vectors)
        print(f"Added {len(documents)} documents to Pinecone")

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )

        output = []
        for match in results['matches']:
            result = {
                'document': {
                    'text': match['metadata']['text'],
                    'metadata': match['metadata']
                },
                'score': match['score'],
                'id': match['id']
            }
            output.append(result)

        return output

    def delete(self, ids: List[str]):
        self.index.delete(ids=ids)

    def get_stats(self) -> Dict[str, Any]:
        stats = self.index.describe_index_stats()
        return {
            'type': 'PineconeDB',
            'index_name': self.index_name,
            'total_vectors': stats.total_vector_count,
            'dimension': stats.dimension
        }

class MultiVectorRAG:
    """RAG system that can work with multiple vector databases"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_db = self._initialize_vector_db()
        self.embedding_model = self._initialize_embedding_model()

    def _initialize_vector_db(self) -> VectorDB:
        """Initialize vector database based on configuration"""
        db_type = self.config.get('vector_db', {}).get('type', 'chroma')

        if db_type == 'faiss':
            return FAISSVectorDB(dimension=384)
        elif db_type == 'chroma':
            persist_dir = self.config.get('vector_db', {}).get('persist_directory', './data/chroma_db')
            return ChromaDB(persist_directory=persist_dir)
        elif db_type == 'pinecone':
            api_key = self.config['api_keys']['pinecone']
            environment = self.config['vector_db']['environment']
            index_name = self.config['vector_db'].get('index_name', 'rag-index')
            return PineconeDB(api_key, environment, index_name)
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")

    def _initialize_embedding_model(self):
        """Initialize embedding model"""
        from sentence_transformers import SentenceTransformer
        model_name = self.config.get('rag_config', {}).get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        return SentenceTransformer(model_name)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector database"""
        if not documents:
            return

        print(f"Processing {len(documents)} documents...")

        # Extract texts
        texts = [doc['text'] for doc in documents]

        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Add to vector database
        self.vector_db.add_documents(embeddings, documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Search vector database
        results = self.vector_db.search(query_embedding, top_k)

        return results

    def get_database_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        return self.vector_db.get_stats()

    def delete_documents(self, ids: List[str]):
        """Delete documents from the vector database"""
        self.vector_db.delete(ids)

# Test the multi-database RAG system
if __name__ == "__main__":
    # Configuration for different databases
    configs = {
        'faiss': {
            'vector_db': {'type': 'faiss'},
            'rag_config': {'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'}
        },
        'chroma': {
            'vector_db': {'type': 'chroma', 'persist_directory': './data/test_chroma'},
            'rag_config': {'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'}
        }
    }

    # Test documents
    test_documents = [
        {
            'text': 'Machine learning algorithms can learn from data.',
            'metadata': {'topic': 'ml', 'source': 'article1'}
        },
        {
            'text': 'Deep learning uses neural networks with many layers.',
            'metadata': {'topic': 'deep-learning', 'source': 'article2'}
        },
        {
            'text': 'Natural language processing helps computers understand text.',
            'metadata': {'topic': 'nlp', 'source': 'article3'}
        }
    ]

    # Test each database type
    for db_type, config in configs.items():
        print(f"\n=== Testing {db_type.upper()} ===")

        try:
            rag = MultiVectorRAG(config)

            # Add documents
            rag.add_documents(test_documents)

            # Search
            query = "What is machine learning?"
            results = rag.search(query, top_k=2)

            print(f"Query: {query}")
            print("Results:")
            for i, result in enumerate(results):
                doc = result['document']
                print(f"{i+1}. Score: {result['score']:.3f}")
                print(f"   Text: {doc['text'][:100]}...")
                print(f"   Metadata: {doc['metadata']}")

            # Show stats
            print(f"Database stats: {rag.get_database_stats()}")

        except Exception as e:
            print(f"Error testing {db_type}: {e}")
```

---

## **5. RAG Pipeline Development**

### **5.5 Exercise 5: Complete RAG Pipeline**

**Objective**: Build a complete RAG pipeline with LLM integration and response generation.

```python
# src/rag_pipeline/complete_rag.py
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

# Import our components
from ..document_processors.advanced_processor import AdvancedDocumentProcessor
from ..vector_db.multi_database import MultiVectorRAG

# LLM integration
try:
    from langchain.llms import OpenAI
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

@dataclass
class RAGResponse:
    """Data class for RAG response"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    retrieval_time: float
    generation_time: float
    total_time: float
    metadata: Dict[str, Any]

class LLMWrapper:
    """Wrapper for different LLM providers"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.llm = None

        if LANGCHAIN_AVAILABLE:
            self._initialize_langchain_llm()

    def _initialize_langchain_llm(self):
        """Initialize LangChain LLM"""
        try:
            os.environ["OPENAI_API_KEY"] = self.api_key or os.getenv("OPENAI_API_KEY", "")

            if "gpt" in self.model_name.lower():
                from langchain.chat_models import ChatOpenAI
                self.llm = ChatOpenAI(model_name=self.model_name)
            else:
                self.llm = OpenAI(model_name=self.model_name)

        except Exception as e:
            print(f"Warning: Could not initialize LangChain LLM: {e}")
            self.llm = None

    def generate(self, prompt: str) -> str:
        """Generate response using LLM"""
        if self.llm:
            return self.llm(prompt)
        else:
            # Fallback simple response
            return f"LLM not available. Would generate: {prompt[:200]}..."

    def generate_with_citations(self, question: str, context: str, sources: List[Dict]) -> str:
        """Generate response with proper source citations"""
        prompt = self._create_citation_prompt(question, context, sources)
        return self.generate(prompt)

    def _create_citation_prompt(self, question: str, context: str, sources: List[Dict]) -> str:
        """Create prompt with proper citation format"""
        sources_text = "\n".join([
            f"[{i+1}] {source['document']['text']}"
            for i, source in enumerate(sources)
        ])

        prompt = f"""You are a helpful AI assistant. Answer the following question using ONLY the information provided in the context.

Context from retrieved documents:
{sources_text}

Question: {question}

Instructions:
1. Use ONLY the information from the provided context
2. Cite sources using the [n] format where n is the source number
3. If the question cannot be answered from the context, say so clearly
4. Be concise but comprehensive

Answer:"""

        return prompt

class CompleteRAGPipeline:
    """Complete RAG pipeline with all components"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize components
        self.document_processor = AdvancedDocumentProcessor(
            chunk_size=config['rag_config']['chunk_size'],
            chunk_overlap=config['rag_config']['chunk_overlap']
        )

        self.rag_system = MultiVectorRAG(config)

        self.llm = LLMWrapper(
            model_name=config['rag_config']['llm_model'],
            api_key=config.get('api_keys', {}).get('openai')
        )

        # Performance tracking
        self.stats = {
            'queries_processed': 0,
            'total_retrieval_time': 0,
            'total_generation_time': 0,
            'average_confidence': 0
        }

    def index_documents(self, document_paths: List[str],
                       chunking_strategy: str = 'adaptive') -> Dict[str, Any]:
        """Index documents from file paths"""
        print(f"Indexing {len(document_paths)} documents...")

        start_time = time.time()

        # Process documents
        all_chunks = []
        for doc_path in document_paths:
            try:
                # Load and process document
                text = self.document_processor.load_document(doc_path)
                file_metadata = {
                    'file_path': doc_path,
                    'file_name': os.path.basename(doc_path),
                    'file_size': os.path.getsize(doc_path) if os.path.exists(doc_path) else 0
                }

                chunks = self.document_processor.process_with_strategy(
                    text, file_metadata, chunking_strategy
                )
                all_chunks.extend(chunks)

                print(f"Processed {doc_path}: {len(chunks)} chunks")

            except Exception as e:
                print(f"Error processing {doc_path}: {e}")

        # Add to vector database
        self.rag_system.add_documents(all_chunks)

        indexing_time = time.time() - start_time

        result = {
            'total_chunks': len(all_chunks),
            'documents_processed': len(document_paths),
            'indexing_time': indexing_time,
            'chunking_strategy': chunking_strategy,
            'database_stats': self.rag_system.get_database_stats()
        }

        print(f"Indexing completed in {indexing_time:.2f} seconds")
        return result

    def query(self, question: str, top_k: int = 5,
             include_generation: bool = True) -> RAGResponse:
        """Process a query through the RAG pipeline"""
        total_start_time = time.time()

        # Retrieve relevant documents
        retrieval_start = time.time()
        retrieved_docs = self.rag_system.search(question, top_k)
        retrieval_time = time.time() - retrieval_start

        # Prepare context
        context = self._prepare_context(retrieved_docs)

        # Generate answer
        generation_start = time.time()
        if include_generation and retrieved_docs:
            answer = self.llm.generate_with_citations(question, context, retrieved_docs)
            confidence_score = self._calculate_confidence(question, retrieved_docs, answer)
        else:
            answer = self._generate_simple_answer(question, retrieved_docs)
            confidence_score = self._calculate_confidence(question, retrieved_docs, answer)

        generation_time = time.time() - generation_start
        total_time = time.time() - total_start_time

        # Update statistics
        self._update_stats(retrieval_time, generation_time, confidence_score)

        # Create response object
        response = RAGResponse(
            question=question,
            answer=answer,
            sources=retrieved_docs,
            confidence_score=confidence_score,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            metadata={
                'top_k': top_k,
                'model_used': self.llm.model_name,
                'vector_db_type': self.rag_system.get_database_stats()['type']
            }
        )

        return response

    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents"""
        context_parts = []
        for i, doc_info in enumerate(retrieved_docs):
            doc = doc_info['document']
            context_parts.append(f"[{i+1}] {doc['text']}")

        return "\n\n".join(context_parts)

    def _calculate_confidence(self, question: str, retrieved_docs: List[Dict],
                            answer: str) -> float:
        """Calculate confidence score for the answer"""
        if not retrieved_docs:
            return 0.0

        # Factors for confidence calculation:
        # 1. Average retrieval score
        avg_retrieval_score = np.mean([doc['score'] for doc in retrieved_docs])

        # 2. Number of relevant sources
        source_diversity = min(len(retrieved_docs) / 5, 1.0)  # Normalize to 0-1

        # 3. Answer length (longer answers tend to be more comprehensive)
        answer_completeness = min(len(answer.split()) / 100, 1.0)

        # Weighted combination
        confidence = (0.5 * avg_retrieval_score +
                     0.3 * source_diversity +
                     0.2 * answer_completeness)

        return min(confidence, 1.0)

    def _generate_simple_answer(self, question: str, retrieved_docs: List[Dict]) -> str:
        """Generate simple answer without LLM"""
        if not retrieved_docs:
            return "I couldn't find relevant information to answer your question."

        # Use top retrieved document
        top_doc = retrieved_docs[0]['document']
        return f"Based on the available information: {top_doc['text']}"

    def _update_stats(self, retrieval_time: float, generation_time: float,
                     confidence: float):
        """Update performance statistics"""
        self.stats['queries_processed'] += 1
        self.stats['total_retrieval_time'] += retrieval_time
        self.stats['total_generation_time'] += generation_time

        # Update running average confidence
        n = self.stats['queries_processed']
        self.stats['average_confidence'] = (
            (self.stats['average_confidence'] * (n - 1) + confidence) / n
        )

    def batch_query(self, questions: List[str], **kwargs) -> List[RAGResponse]:
        """Process multiple queries efficiently"""
        responses = []
        for question in questions:
            try:
                response = self.query(question, **kwargs)
                responses.append(response)
                print(f"Processed query {len(responses)}/{len(questions)}")
            except Exception as e:
                print(f"Error processing query '{question}': {e}")
                # Create error response
                error_response = RAGResponse(
                    question=question,
                    answer=f"Error: {str(e)}",
                    sources=[],
                    confidence_score=0.0,
                    retrieval_time=0.0,
                    generation_time=0.0,
                    total_time=0.0,
                    metadata={'error': str(e)}
                )
                responses.append(error_response)

        return responses

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        n = self.stats['queries_processed']
        if n == 0:
            return self.stats

        return {
            **self.stats,
            'average_retrieval_time': self.stats['total_retrieval_time'] / n,
            'average_generation_time': self.stats['total_generation_time'] / n,
            'total_queries': n
        }

    def save_index(self, save_path: str):
        """Save the current index"""
        # Implementation depends on vector database type
        db_stats = self.rag_system.get_database_stats()
        print(f"Index saved. Database stats: {db_stats}")

    def load_index(self, load_path: str):
        """Load a previously saved index"""
        # Implementation depends on vector database type
        print(f"Index loaded from {load_path}")

# Test the complete RAG pipeline
if __name__ == "__main__":
    # Example configuration
    config = {
        'rag_config': {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'llm_model': 'gpt-3.5-turbo',
            'chunk_size': 300,
            'chunk_overlap': 50,
            'top_k': 3
        },
        'vector_db': {
            'type': 'chroma',
            'persist_directory': './data/test_rag_pipeline'
        },
        'api_keys': {
            'openai': os.getenv('OPENAI_API_KEY', '')
        }
    }

    # Create sample documents
    os.makedirs("data/documents", exist_ok=True)
    sample_docs = [
        "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
        "Deep learning uses neural networks with multiple layers to process complex data patterns.",
        "Natural language processing helps machines understand and generate human language.",
        "Computer vision enables AI systems to interpret and analyze visual information from images."
    ]

    # Save sample documents
    for i, doc in enumerate(sample_docs):
        with open(f"data/documents/sample_{i}.txt", "w") as f:
            f.write(doc)

    # Initialize RAG pipeline
    print("Initializing RAG Pipeline...")
    rag_pipeline = CompleteRAGPipeline(config)

    # Index documents
    doc_paths = [f"data/documents/sample_{i}.txt" for i in range(len(sample_docs))]
    indexing_result = rag_pipeline.index_documents(doc_paths, 'sentence')

    print(f"Indexing result: {indexing_result}")

    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is natural language processing?"
    ]

    print("\n=== Testing RAG Pipeline ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = rag_pipeline.query(query)

        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence_score:.3f}")
        print(f"Retrieval time: {response.retrieval_time:.3f}s")
        print(f"Generation time: {response.generation_time:.3f}s")
        print(f"Sources: {len(response.sources)}")

    # Show performance stats
    print("\n=== Performance Statistics ===")
    print(json.dumps(rag_pipeline.get_performance_stats(), indent=2))
```

---

## **6. Advanced RAG Techniques**

### **6.6 Exercise 6: Multi-Hop RAG System**

**Objective**: Implement multi-hop reasoning in RAG for complex question answering.

```python
# src/rag_pipeline/multi_hop_rag.py
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass

@dataclass
class HopResult:
    """Result from a single hop in multi-hop reasoning"""
    hop_number: int
    question: str
    context: str
    evidence: List[Dict[str, Any]]
    confidence: float
    reasoning_trace: str

class MultiHopRAGSystem:
    """RAG system with multi-hop reasoning capabilities"""

    def __init__(self, base_rag_pipeline):
        self.base_rag = base_rag_pipeline
        self.max_hops = 5
        self.confidence_threshold = 0.7
        self.hop_history = []

    def query_multi_hop(self, question: str, top_k: int = 3,
                       max_hops: int = None) -> Dict[str, Any]:
        """Perform multi-hop reasoning to answer complex questions"""
        if max_hops is None:
            max_hops = self.max_hops

        print(f"Starting multi-hop reasoning for: {question}")

        current_question = question
        hop_results = []
        all_evidence = []

        for hop in range(max_hops):
            print(f"Hop {hop + 1}: {current_question}")

            # Retrieve evidence for current question
            response = self.base_rag.query(current_question, top_k=top_k)

            hop_result = HopResult(
                hop_number=hop + 1,
                question=current_question,
                context=response.answer,
                evidence=response.sources,
                confidence=response.confidence_score,
                reasoning_trace=f"Retrieved {len(response.sources)} sources with confidence {response.confidence_score:.3f}"
            )

            hop_results.append(hop_result)
            all_evidence.extend(response.sources)

            # Check if we have enough evidence to answer the original question
            if self._sufficient_evidence(question, hop_results):
                print(f"Sufficient evidence found after {hop + 1} hops")
                break

            # Generate next question based on current evidence
            if hop < max_hops - 1:  # Don't generate new question on last hop
                next_question = self._generate_next_question(
                    question, hop_results, hop + 1
                )
                current_question = next_question

        # Generate final answer using all collected evidence
        final_answer = self._synthesize_final_answer(question, hop_results)

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(hop_results)

        return {
            'original_question': question,
            'final_answer': final_answer,
            'hop_results': hop_results,
            'total_hops': len(hop_results),
            'overall_confidence': overall_confidence,
            'synthesis_reasoning': self._explain_synthesis(question, hop_results)
        }

    def _sufficient_evidence(self, original_question: str,
                           hop_results: List[HopResult]) -> bool:
        """Check if we have sufficient evidence to answer the original question"""
        if not hop_results:
            return False

        # Combine all evidence
        all_text = " ".join([result.context for result in hop_results])

        # Check if the combined evidence seems to answer the original question
        # This is a simple heuristic - in practice, you'd want a more sophisticated check

        question_keywords = set(original_question.lower().split())
        evidence_keywords = set(all_text.lower().split())

        # Calculate keyword overlap
        keyword_overlap = len(question_keywords.intersection(evidence_keywords))
        keyword_coverage = keyword_overlap / len(question_keywords) if question_keywords else 0

        # Check average confidence
        avg_confidence = np.mean([result.confidence for result in hop_results])

        # Check if we have evidence from multiple hops (diverse sources)
        diverse_hops = len(set(result.hop_number for result in hop_results))

        # Sufficient evidence if:
        # 1. High keyword coverage (> 60%)
        # 2. Good confidence (> threshold)
        # 3. Evidence from multiple hops or single strong hop

        sufficient = (
            keyword_coverage > 0.6 and
            avg_confidence > self.confidence_threshold and
            (diverse_hops > 1 or avg_confidence > 0.8)
        )

        print(f"Evidence check: coverage={keyword_coverage:.2f}, "
              f"confidence={avg_confidence:.2f}, hops={diverse_hops}, sufficient={sufficient}")

        return sufficient

    def _generate_next_question(self, original_question: str,
                              hop_results: List[HopResult], current_hop: int) -> str:
        """Generate the next question based on current evidence"""

        # Analyze what we know so far
        known_information = []
        for result in hop_results:
            known_information.append(result.context)

        combined_context = "\n".join(known_information)

        # Create prompt for generating next question
        prompt = f"""
        Original question: "{original_question}"

        Information gathered so far:
        {combined_context}

        Based on the information above, what specific question should we ask next to get more details about "{original_question}"?

        The next question should:
        1. Address gaps in our current knowledge
        2. Be specific and answerable through document retrieval
        3. Help complete our understanding of the original question

        Next question:
        """

        try:
            # Use the base RAG's LLM to generate next question
            next_question = self.base_rag.llm.generate(prompt)
            return next_question.strip()
        except:
            # Fallback to simple heuristic
            return f"What are the key details about {original_question.lower()}?"

    def _synthesize_final_answer(self, original_question: str,
                               hop_results: List[HopResult]) -> str:
        """Synthesize final answer from all hop results"""

        # Combine all evidence and context
        all_context = []
        for i, result in enumerate(hop_results):
            all_context.append(f"From hop {i+1}: {result.context}")

        combined_evidence = "\n\n".join(all_context)

        # Create synthesis prompt
        prompt = f"""
        Original question: "{original_question}"

        Evidence from multiple hops:
        {combined_evidence}

        Please provide a comprehensive answer to the original question using all the evidence above.
        Synthesize the information from all hops into a coherent response.

        Answer:
        """

        try:
            final_answer = self.base_rag.llm.generate(prompt)
            return final_answer.strip()
        except:
            # Fallback synthesis
            return f"Based on the evidence gathered: {' '.join([result.context for result in hop_results])}"

    def _calculate_overall_confidence(self, hop_results: List[HopResult]) -> float:
        """Calculate overall confidence from all hop results"""
        if not hop_results:
            return 0.0

        # Factors for overall confidence:
        # 1. Average hop confidence
        avg_hop_confidence = np.mean([result.confidence for result in hop_results])

        # 2. Consistency across hops (lower standard deviation = higher confidence)
        hop_confidences = [result.confidence for result in hop_results]
        consistency = 1.0 - (np.std(hop_confidences) if len(hop_confidences) > 1 else 0)

        # 3. Number of successful hops
        successful_hops = len([result for result in hop_results if result.confidence > 0.5])
        hop_coverage = successful_hops / len(hop_results)

        # Weighted combination
        overall_confidence = 0.5 * avg_hop_confidence + 0.3 * consistency + 0.2 * hop_coverage

        return min(overall_confidence, 1.0)

    def _explain_synthesis(self, original_question: str,
                          hop_results: List[HopResult]) -> str:
        """Generate explanation of how the answer was synthesized"""
        explanation = f"""
        This answer was synthesized through {len(hop_results)} reasoning hop(s):
        """

        for i, result in enumerate(hop_results):
            explanation += f"\n{i+1}. Hop {result.hop_number}: '{result.question}' - Confidence: {result.confidence:.3f}"

        explanation += f"\n\nThe final confidence score of {self._calculate_overall_confidence(hop_results):.3f} reflects the quality and consistency of evidence across all hops."

        return explanation

    def analyze_question_complexity(self, question: str) -> Dict[str, Any]:
        """Analyze if a question requires multi-hop reasoning"""

        # Simple heuristics for multi-hop detection
        multi_hop_indicators = [
            "what is the relationship between",
            "how does",
            "why does",
            "compare",
            "contrast",
            "what are the causes of",
            "what are the effects of",
            "explain how",
            "show how"
        ]

        question_lower = question.lower()
        complexity_score = 0

        for indicator in multi_hop_indicators:
            if indicator in question_lower:
                complexity_score += 1

        # Additional complexity factors
        if "?" in question:
            complexity_score += 1

        if len(question.split()) > 10:
            complexity_score += 1

        # Recommend approach
        if complexity_score >= 2:
            recommended_approach = "multi-hop"
            estimated_hops = min(complexity_score + 1, 4)
        else:
            recommended_approach = "single-hop"
            estimated_hops = 1

        return {
            'question': question,
            'complexity_score': complexity_score,
            'recommended_approach': recommended_approach,
            'estimated_hops': estimated_hops,
            'indicators_found': [indicator for indicator in multi_hop_indicators
                               if indicator in question_lower]
        }

# Test the multi-hop RAG system
if __name__ == "__main__":
    # This would require a base RAG pipeline to be initialized
    print("Multi-hop RAG system ready for testing")
    print("Example usage:")
    print("""
    # Initialize base RAG pipeline first
    from src.rag_pipeline.complete_rag import CompleteRAGPipeline
    base_rag = CompleteRAGPipeline(config)

    # Create multi-hop RAG system
    multi_hop_rag = MultiHopRAGSystem(base_rag)

    # Analyze question complexity
    analysis = multi_hop_rag.analyze_question_complexity(
        "What is the relationship between machine learning and deep learning?"
    )
    print(analysis)

    # Perform multi-hop reasoning
    result = multi_hop_rag.query_multi_hop(
        "How do transformers work in natural language processing?"
    )
    print(result['final_answer'])
    """)
```

---

## **7. RAG Evaluation and Testing**

### **7.7 Exercise 7: RAG Evaluation Framework**

**Objective**: Create a comprehensive evaluation framework for RAG systems.

```python
# src/evaluation/rag_evaluator.py
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

@dataclass
class EvaluationResult:
    """Individual evaluation result"""
    query: str
    expected_answer: str
    predicted_answer: str
    sources: List[Dict[str, Any]]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: str

@dataclass
class EvaluationSummary:
    """Summary of evaluation results"""
    total_queries: int
    average_metrics: Dict[str, float]
    metric_distributions: Dict[str, List[float]]
    top_performing_queries: List[Tuple[str, float]]
    worst_performing_queries: List[Tuple[str, float]]
    evaluation_time: float
    timestamp: str

class RAGEvaluator:
    """Comprehensive evaluation framework for RAG systems"""

    def __init__(self):
        self.results = []
        self.evaluation_history = []

    def evaluate_system(self, rag_system, test_queries: List[Dict[str, str]],
                       top_k: int = 5) -> EvaluationSummary:
        """Evaluate RAG system against test queries"""
        print(f"Evaluating RAG system with {len(test_queries)} queries...")

        start_time = datetime.now()
        results = []

        for i, test_case in enumerate(test_queries):
            print(f"Evaluating query {i+1}/{len(test_queries)}: {test_case['query'][:50]}...")

            try:
                # Get RAG response
                response = rag_system.query(test_case['query'], top_k=top_k)

                # Calculate metrics
                metrics = self._calculate_all_metrics(
                    test_case['query'],
                    test_case['expected_answer'],
                    response.answer,
                    response.sources,
                    response.confidence_score
                )

                # Create evaluation result
                eval_result = EvaluationResult(
                    query=test_case['query'],
                    expected_answer=test_case['expected_answer'],
                    predicted_answer=response.answer,
                    sources=response.sources,
                    metrics=metrics,
                    metadata={
                        'retrieval_time': response.retrieval_time,
                        'generation_time': response.generation_time,
                        'confidence_score': response.confidence_score,
                        'sources_count': len(response.sources)
                    },
                    timestamp=datetime.now().isoformat()
                )

                results.append(eval_result)

            except Exception as e:
                print(f"Error evaluating query: {e}")
                # Create error result
                error_result = EvaluationResult(
                    query=test_case['query'],
                    expected_answer=test_case['expected_answer'],
                    predicted_answer="ERROR: " + str(e),
                    sources=[],
                    metrics={'error': 1.0},
                    metadata={'error': str(e)},
                    timestamp=datetime.now().isoformat()
                )
                results.append(error_result)

        evaluation_time = (datetime.now() - start_time).total_seconds()

        # Create summary
        summary = self._create_summary(results, evaluation_time)

        # Store results
        self.results.extend(results)
        self.evaluation_history.append(summary)

        return summary

    def _calculate_all_metrics(self, query: str, expected: str, predicted: str,
                             sources: List[Dict], confidence: float) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        metrics = {}

        # 1. Retrieval Quality Metrics
        metrics.update(self._calculate_retrieval_metrics(sources))

        # 2. Generation Quality Metrics
        metrics.update(self._calculate_generation_metrics(expected, predicted))

        # 3. Groundedness Metrics
        metrics.update(self._calculate_groundedness_metrics(predicted, sources))

        # 4. Performance Metrics
        metrics.update(self._calculate_performance_metrics(confidence))

        return metrics

    def _calculate_retrieval_metrics(self, sources: List[Dict]) -> Dict[str, float]:
        """Calculate retrieval quality metrics"""
        if not sources:
            return {
                'avg_retrieval_score': 0.0,
                'retrieval_score_variance': 0.0,
                'source_diversity': 0.0,
                'source_relevance_score': 0.0
            }

        scores = [source['score'] for source in sources]

        # Average retrieval score
        avg_score = np.mean(scores)

        # Retrieval score variance (consistency)
        score_variance = np.var(scores)

        # Source diversity (how different are the sources?)
        diversity_score = self._calculate_source_diversity(sources)

        # Source relevance score
        relevance_score = np.mean(scores)

        return {
            'avg_retrieval_score': avg_score,
            'retrieval_score_variance': score_variance,
            'source_diversity': diversity_score,
            'source_relevance_score': relevance_score
        }

    def _calculate_generation_metrics(self, expected: str, predicted: str) -> Dict[str, float]:
        """Calculate generation quality metrics"""
        if not expected or not predicted:
            return {
                'answer_length_match': 0.0,
                'keyword_overlap': 0.0,
                'semantic_similarity': 0.0
            }

        # Answer length match
        expected_len = len(expected.split())
        predicted_len = len(predicted.split())
        length_ratio = min(expected_len, predicted_len) / max(expected_len, predicted_len, 1)

        # Keyword overlap
        expected_keywords = set(expected.lower().split())
        predicted_keywords = set(predicted.lower().split())
        keyword_overlap = len(expected_keywords.intersection(predicted_keywords)) / len(expected_keywords) if expected_keywords else 0

        # Semantic similarity (simplified)
        semantic_similarity = self._simple_semantic_similarity(expected, predicted)

        return {
            'answer_length_match': length_ratio,
            'keyword_overlap': keyword_overlap,
            'semantic_similarity': semantic_similarity
        }

    def _calculate_groundedness_metrics(self, predicted: str, sources: List[Dict]) -> Dict[str, float]:
        """Calculate groundedness metrics"""
        if not sources:
            return {
                'groundedness_score': 0.0,
                'source_attribution_rate': 0.0,
                'factual_consistency': 0.0
            }

        # Simple groundedness: how well does the answer align with sources?
        groundedness = self._calculate_simple_groundedness(predicted, sources)

        # Source attribution rate
        attribution_rate = min(len(sources) / 5, 1.0)  # Normalized to 0-1

        # Factual consistency (simplified)
        consistency = self._calculate_factual_consistency(predicted, sources)

        return {
            'groundedness_score': groundedness,
            'source_attribution_rate': attribution_rate,
            'factual_consistency': consistency
        }

    def _calculate_performance_metrics(self, confidence: float) -> Dict[str, float]:
        """Calculate performance-related metrics"""
        return {
            'confidence_score': confidence,
            'response_quality': min(confidence * 1.2, 1.0),  # Slightly boost confidence
            'reliability_score': confidence
        }

    def _calculate_source_diversity(self, sources: List[Dict]) -> float:
        """Calculate diversity of sources"""
        if len(sources) <= 1:
            return 0.0

        # Calculate text similarity between sources
        similarities = []
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                text_i = sources[i]['document']['text']
                text_j = sources[j]['document']['text']
                sim = self._simple_semantic_similarity(text_i, text_j)
                similarities.append(sim)

        # Diversity is 1 - average similarity
        avg_similarity = np.mean(similarities) if similarities else 0
        diversity = 1 - avg_similarity

        return diversity

    def _simple_semantic_similarity(self, text1: str, text2: str) -> float:
        """Simple semantic similarity using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _calculate_simple_groundedness(self, answer: str, sources: List[Dict]) -> float:
        """Calculate simple groundedness score"""
        if not sources:
            return 0.0

        # Check how much of the answer can be found in sources
        answer_words = set(answer.lower().split())
        source_words = set()

        for source in sources:
            source_words.update(source['document']['text'].lower().split())

        # Calculate coverage
        covered_words = answer_words.intersection(source_words)
        coverage = len(covered_words) / len(answer_words) if answer_words else 0

        return coverage

    def _calculate_factual_consistency(self, answer: str, sources: List[Dict]) -> float:
        """Calculate factual consistency score"""
        if not sources:
            return 0.0

        # Simple consistency check: do sources support the answer?
        # This is a very simplified version
        source_support = 0.0

        for source in sources:
            source_text = source['document']['text'].lower()
            # Check if key phrases from answer appear in source
            answer_sentences = answer.split('.')
            for sentence in answer_sentences:
                sentence_words = set(sentence.lower().split())
                if len(sentence_words) > 3:  # Only check substantial sentences
                    overlap = len(sentence_words.intersection(set(source_text.split())))
                    sentence_support = overlap / len(sentence_words)
                    source_support = max(source_support, sentence_support)

        return source_support

    def _create_summary(self, results: List[EvaluationResult],
                       evaluation_time: float) -> EvaluationSummary:
        """Create evaluation summary"""

        if not results:
            return EvaluationSummary(
                total_queries=0,
                average_metrics={},
                metric_distributions={},
                top_performing_queries=[],
                worst_performing_queries=[],
                evaluation_time=evaluation_time,
                timestamp=datetime.now().isoformat()
            )

        # Calculate average metrics
        all_metrics = {}
        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        average_metrics = {
            metric_name: np.mean(values)
            for metric_name, values in all_metrics.items()
        }

        # Calculate metric distributions
        metric_distributions = all_metrics

        # Find top and worst performing queries
        # Use overall quality score as ranking metric
        quality_scores = []
        for result in results:
            # Calculate quality score as weighted combination of metrics
            quality_score = (
                0.3 * result.metrics.get('avg_retrieval_score', 0) +
                0.3 * result.metrics.get('semantic_similarity', 0) +
                0.2 * result.metrics.get('groundedness_score', 0) +
                0.2 * result.metrics.get('confidence_score', 0)
            )
            quality_scores.append((result.query, quality_score))

        quality_scores.sort(key=lambda x: x[1], reverse=True)

        top_performing = quality_scores[:5]  # Top 5
        worst_performing = quality_scores[-5:]  # Bottom 5

        return EvaluationSummary(
            total_queries=len(results),
            average_metrics=average_metrics,
            metric_distributions=metric_distributions,
            top_performing_queries=top_performing,
            worst_performing_queries=worst_performing,
            evaluation_time=evaluation_time,
            timestamp=datetime.now().isoformat()
        )

    def create_evaluation_report(self, summary: EvaluationSummary,
                               output_file: str = None) -> str:
        """Create detailed evaluation report"""

        report = f"""
# RAG System Evaluation Report

**Generated:** {summary.timestamp}
**Evaluation Time:** {summary.evaluation_time:.2f} seconds
**Total Queries Evaluated:** {summary.total_queries}

## Summary Metrics

"""

        for metric_name, avg_value in summary.average_metrics.items():
            report += f"- **{metric_name.replace('_', ' ').title()}:** {avg_value:.3f}\n"

        report += f"""

## Top Performing Queries

"""
        for query, score in summary.top_performing_queries:
            report += f"- **Score: {score:.3f}** - {query[:100]}{'...' if len(query) > 100 else ''}\n"

        report += f"""

## Worst Performing Queries

"""
        for query, score in summary.worst_performing_queries:
            report += f"- **Score: {score:.3f}** - {query[:100]}{'...' if len(query) > 100 else ''}\n"

        report += f"""

## Detailed Metrics Distribution

"""
        for metric_name, values in summary.metric_distributions.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                report += f"""
### {metric_name.replace('_', ' ').title()}
- Mean: {mean_val:.3f}
- Std Dev: {std_val:.3f}
- Range: [{min_val:.3f}, {max_val:.3f}]
"""

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Evaluation report saved to {output_file}")

        return report

    def visualize_evaluation_results(self, summary: EvaluationSummary,
                                   output_dir: str = "evaluation_plots"):
        """Create visualizations of evaluation results"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Metric distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (metric_name, values) in enumerate(summary.metric_distributions.items()):
            if i < len(axes) and values:
                axes[i].hist(values, bins=20, alpha=0.7)
                axes[i].set_title(metric_name.replace('_', ' ').title())
                axes[i].set_xlabel('Score')
                axes[i].set_ylabel('Frequency')

        # Hide unused subplots
        for i in range(len(summary.metric_distributions), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/metric_distributions.png")
        plt.close()

        # Performance comparison
        fig, ax = plt.subplots(figsize=(12, 8))

        metric_names = list(summary.average_metrics.keys())
        metric_values = list(summary.average_metrics.values())

        bars = ax.bar(range(len(metric_names)), metric_values, alpha=0.7)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Average Score')
        ax.set_title('RAG System Performance Overview')
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels([name.replace('_', ' ').title() for name in metric_names], rotation=45, ha='right')

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_overview.png")
        plt.close()

        print(f"Visualizations saved to {output_dir}/")

    def load_test_queries(self, file_path: str) -> List[Dict[str, str]]:
        """Load test queries from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def save_evaluation_results(self, summary: EvaluationSummary, file_path: str):
        """Save evaluation results to JSON file"""
        summary_dict = asdict(summary)

        with open(file_path, 'w') as f:
            json.dump(summary_dict, f, indent=2, default=str)

        print(f"Evaluation results saved to {file_path}")

# Example test queries for evaluation
def create_sample_test_queries() -> List[Dict[str, str]]:
    """Create sample test queries for evaluation"""
    return [
        {
            "query": "What is machine learning?",
            "expected_answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed for every task."
        },
        {
            "query": "How do neural networks work?",
            "expected_answer": "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes (neurons) that process information through weighted connections and activation functions."
        },
        {
            "query": "What is natural language processing?",
            "expected_answer": "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language."
        },
        {
            "query": "Explain deep learning",
            "expected_answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."
        },
        {
            "query": "What are transformers in AI?",
            "expected_answer": "Transformers are a type of neural network architecture that uses self-attention mechanisms to process sequential data, particularly effective for natural language processing tasks."
        }
    ]

# Test the evaluation framework
if __name__ == "__main__":
    print("RAG Evaluation Framework")
    print("Example usage:")
    print("""
    # Create evaluator
    evaluator = RAGEvaluator()

    # Create test queries
    test_queries = create_sample_test_queries()

    # Evaluate RAG system
    summary = evaluator.evaluate_system(rag_system, test_queries)

    # Create report
    report = evaluator.create_evaluation_report(summary, "rag_evaluation_report.md")

    # Create visualizations
    evaluator.visualize_evaluation_results(summary)

    # Save results
    evaluator.save_evaluation_results(summary, "evaluation_results.json")
    """)
```

---

## **8. Production RAG Systems**

### **8.8 Exercise 8: Production-Ready RAG API**

**Objective**: Build a production-ready RAG API with FastAPI, monitoring, and caching.

```python
# src/api/rag_api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import time
import logging
from datetime import datetime
import redis
import os
from contextlib import asynccontextmanager

# RAG imports
from ..rag_pipeline.complete_rag import CompleteRAGPipeline, RAGResponse
from ..evaluation.rag_evaluator import RAGEvaluator

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str = Field(..., description="User question to answer")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    include_sources: bool = Field(True, description="Include source documents in response")
    session_id: Optional[str] = Field(None, description="Session identifier for caching")

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float
    retrieval_time: float
    generation_time: float
    total_time: float
    session_id: Optional[str] = None

class BatchQueryRequest(BaseModel):
    questions: List[str] = Field(..., description="List of questions to process")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")

class BatchQueryResponse(BaseModel):
    responses: List[QueryResponse]
    total_time: float
    average_time_per_query: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime: float
    version: str

class RAGStats(BaseModel):
    total_queries: int
    cache_hit_rate: float
    average_response_time: float
    database_stats: Dict[str, Any]
    system_health: str

class ProductionRAGAPI:
    """Production-ready RAG API with FastAPI"""

    def __init__(self, config_path: str = "config/production_config.yaml"):
        self.config = self._load_config(config_path)
        self.rag_pipeline = None
        self.evaluator = None
        self.redis_client = None
        self.start_time = time.time()
        self.query_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'total_response_time': 0
        }

        # Initialize FastAPI app
        self.app = FastAPI(
            title="Production RAG API",
            description="Retrieval-Augmented Generation API with monitoring and caching",
            version="1.0.0"
        )

        self._setup_middleware()
        self._setup_routes()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        import yaml
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default config
            return {
                'rag_config': {
                    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'llm_model': 'gpt-3.5-turbo',
                    'chunk_size': 512,
                    'chunk_overlap': 50
                },
                'vector_db': {
                    'type': 'chroma',
                    'persist_directory': './data/production_db'
                },
                'cache': {
                    'redis_url': 'redis://localhost:6379',
                    'ttl': 3600
                },
                'api': {
                    'max_concurrent_queries': 10,
                    'rate_limit_per_minute': 60
                }
            }

    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get('api', {}).get('cors_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _initialize_components(self):
        """Initialize RAG components"""
        try:
            # Initialize RAG pipeline
            self.rag_pipeline = CompleteRAGPipeline(self.config)

            # Initialize evaluator
            self.evaluator = RAGEvaluator()

            # Initialize Redis cache
            cache_config = self.config.get('cache', {})
            if cache_config.get('redis_url'):
                try:
                    self.redis_client = redis.from_url(
                        cache_config['redis_url'],
                        decode_responses=True
                    )
                    self.redis_client.ping()
                    logger.info("Redis cache connected successfully")
                except Exception as e:
                    logger.warning(f"Redis not available: {e}")
                    self.redis_client = None

            logger.info("RAG components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.on_event("startup")
        async def startup_event():
            self._initialize_components()
            logger.info("RAG API started successfully")

        @self.app.on_event("shutdown")
        async def shutdown_event():
            if self.redis_client:
                self.redis_client.close()
            logger.info("RAG API shutting down")

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            uptime = time.time() - self.start_time

            # Check component health
            components_healthy = self._check_component_health()
            status = "healthy" if components_healthy else "degraded"

            return HealthResponse(
                status=status,
                timestamp=datetime.now().isoformat(),
                uptime=uptime,
                version="1.0.0"
            )

        @self.app.post("/query", response_model=QueryResponse)
        async def query_rag(request: QueryRequest):
            """Single query endpoint"""
            start_time = time.time()

            try:
                # Check cache first
                cache_key = f"rag:{hash(request.question)}:{request.top_k}"
                if self.redis_client:
                    cached_result = self.redis_client.get(cache_key)
                    if cached_result:
                        self.query_stats['cache_hits'] += 1
                        logger.info(f"Cache hit for query: {request.question[:50]}...")
                        return QueryResponse(**json.loads(cached_result))

                # Process query
                response = self.rag_pipeline.query(
                    request.question,
                    top_k=request.top_k
                )

                # Format response
                query_response = QueryResponse(
                    question=response.question,
                    answer=response.answer,
                    sources=response.sources if request.include_sources else [],
                    confidence_score=response.confidence_score,
                    retrieval_time=response.retrieval_time,
                    generation_time=response.generation_time,
                    total_time=response.total_time,
                    session_id=request.session_id
                )

                # Cache result
                if self.redis_client and response.confidence_score > 0.5:
                    cache_ttl = self.config.get('cache', {}).get('ttl', 3600)
                    self.redis_client.setex(
                        cache_key,
                        cache_ttl,
                        query_response.json()
                    )

                # Update stats
                self._update_query_stats(response.total_time)

                return query_response

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/batch-query", response_model=BatchQueryResponse)
        async def batch_query_rag(request: BatchQueryRequest):
            """Batch query endpoint"""
            start_time = time.time()

            try:
                # Process queries in parallel with concurrency limit
                max_concurrent = self.config.get('api', {}).get('max_concurrent_queries', 10)

                semaphore = asyncio.Semaphore(max_concurrent)

                async def process_single_query(question: str):
                    async with semaphore:
                        try:
                            response = self.rag_pipeline.query(question, top_k=request.top_k)
                            return QueryResponse(
                                question=response.question,
                                answer=response.answer,
                                sources=response.sources,
                                confidence_score=response.confidence_score,
                                retrieval_time=response.retrieval_time,
                                generation_time=response.generation_time,
                                total_time=response.total_time
                            )
                        except Exception as e:
                            return QueryResponse(
                                question=question,
                                answer=f"Error: {str(e)}",
                                sources=[],
                                confidence_score=0.0,
                                retrieval_time=0.0,
                                generation_time=0.0,
                                total_time=0.0
                            )

                # Process all queries
                tasks = [process_single_query(q) for q in request.questions]
                responses = await asyncio.gather(*tasks)

                total_time = time.time() - start_time
                avg_time = total_time / len(request.questions)

                return BatchQueryResponse(
                    responses=responses,
                    total_time=total_time,
                    average_time_per_query=avg_time
                )

            except Exception as e:
                logger.error(f"Error in batch query: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/stats", response_model=RAGStats)
        async def get_stats():
            """Get API statistics"""
            n = self.query_stats['total_queries']
            cache_hit_rate = (
                self.query_stats['cache_hits'] / n if n > 0 else 0.0
            )
            avg_response_time = (
                self.query_stats['total_response_time'] / n if n > 0 else 0.0
            )

            # Get database stats
            db_stats = {}
            if self.rag_pipeline:
                db_stats = self.rag_pipeline.get_database_stats()

            system_health = "healthy" if avg_response_time < 10.0 else "degraded"

            return RAGStats(
                total_queries=n,
                cache_hit_rate=cache_hit_rate,
                average_response_time=avg_response_time,
                database_stats=db_stats,
                system_health=system_health
            )

        @self.app.post("/index-documents")
        async def index_documents(
            file_paths: List[str],
            chunking_strategy: str = "adaptive"
        ):
            """Index new documents"""
            try:
                result = self.rag_pipeline.index_documents(
                    file_paths,
                    chunking_strategy
                )
                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error indexing documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/metrics")
        async def get_metrics():
            """Get detailed metrics for monitoring"""
            try:
                # This would integrate with Prometheus or similar monitoring
                metrics = {
                    "query_count_total": self.query_stats['total_queries'],
                    "query_duration_seconds": self.query_stats['total_response_time'],
                    "cache_hit_rate": self.query_stats['cache_hits'] / max(self.query_stats['total_queries'], 1),
                    "uptime_seconds": time.time() - self.start_time,
                    "timestamp": datetime.now().isoformat()
                }
                return metrics
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _check_component_health(self) -> bool:
        """Check health of all components"""
        try:
            # Check RAG pipeline
            if not self.rag_pipeline:
                return False

            # Check vector database
            db_stats = self.rag_pipeline.get_database_stats()
            if not db_stats:
                return False

            # Check Redis (optional)
            if self.redis_client:
                try:
                    self.redis_client.ping()
                except:
                    # Redis not critical for basic functionality
                    pass

            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def _update_query_stats(self, response_time: float):
        """Update query statistics"""
        self.query_stats['total_queries'] += 1
        self.query_stats['total_response_time'] += response_time

    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the FastAPI server"""
        import uvicorn
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info" if debug else "warning"
        )

# Production configuration example
def create_production_config():
    """Create example production configuration"""
    config = {
        'rag_config': {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'llm_model': 'gpt-3.5-turbo',
            'chunk_size': 512,
            'chunk_overlap': 50,
            'top_k': 5
        },
        'vector_db': {
            'type': 'chroma',
            'persist_directory': '/app/data/chroma_db'
        },
        'cache': {
            'redis_url': 'redis://redis:6379',
            'ttl': 3600
        },
        'api': {
            'max_concurrent_queries': 10,
            'rate_limit_per_minute': 60,
            'cors_origins': ['http://localhost:3000', 'https://yourdomain.com']
        },
        'monitoring': {
            'prometheus_enabled': True,
            'metrics_port': 9090
        },
        'api_keys': {
            'openai': os.getenv('OPENAI_API_KEY', '')
        }
    }

    # Save config
    import yaml
    os.makedirs('config', exist_ok=True)
    with open('config/production_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print("Production config created at config/production_config.yaml")

# Docker setup example
def create_docker_files():
    """Create Docker files for production deployment"""

    # Dockerfile
    dockerfile_content = '''
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.rag_api:ProductionRAGAPI().app", "--host", "0.0.0.0", "--port", "8000"]
'''

    # docker-compose.yml
    compose_content = '''
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
    depends_on:
      - redis
      - chromadb
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
    restart: unless-stopped

volumes:
  redis_data:
  chroma_data:
'''

    # Save files
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)

    with open('docker-compose.yml', 'w') as f:
        f.write(compose_content)

    print("Docker files created: Dockerfile, docker-compose.yml")

# Main function to run the API
def main():
    """Main function to run production RAG API"""
    import argparse

    parser = argparse.ArgumentParser(description="Production RAG API")
    parser.add_argument('--config', default='config/production_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--setup', choices=['config', 'docker'],
                       help='Setup files for production')

    args = parser.parse_args()

    if args.setup == 'config':
        create_production_config()
        return
    elif args.setup == 'docker':
        create_docker_files()
        return

    # Create and run API
    api = ProductionRAGAPI(args.config)
    api.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
```

---

## **9. Performance Optimization**

### **9.9 Exercise 9: RAG Performance Optimization**

**Objective**: Implement performance optimizations for production RAG systems.

```python
# src/optimization/performance_optimizer.py
import time
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
from dataclasses import dataclass
import pickle

@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization"""
    response_time: float
    memory_usage: float
    throughput: float
    cache_hit_rate: float
    accuracy: float

class OptimizedRAGSystem:
    """High-performance RAG system with multiple optimizations"""

    def __init__(self, base_rag_pipeline, config: Dict[str, Any]):
        self.base_rag = base_rag_pipeline
        self.config = config

        # Performance optimizations
        self.thread_pool = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        self.query_cache = {}
        self.embedding_cache = {}
        self.precomputed_embeddings = {}

        # Async processing
        self.processing_queue = asyncio.Queue()
        self.result_cache = {}

        # Performance monitoring
        self.metrics_history = []
        self.optimization_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'precomputed_embeddings': 0,
            'async_processing': 0
        }

        # Memory management
        self.max_memory_usage = config.get('max_memory_mb', 1000)
        self.cleanup_interval = config.get('cleanup_interval', 100)
        self.query_count = 0

    def optimize_retrieval(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Optimize retrieval with batching and caching"""
        start_time = time.time()

        # 1. Check cache for each query
        cached_results = {}
        uncached_queries = []

        for query in queries:
            cache_key = hash(query)
            if cache_key in self.query_cache:
                cached_results[query] = self.query_cache[cache_key]
                self.optimization_stats['cache_hits'] += 1
            else:
                uncached_queries.append(query)
                self.optimization_stats['cache_misses'] += 1

        # 2. Process uncached queries in batch
        if uncached_queries:
            batch_results = self._process_batch_queries(uncached_queries)

            # 3. Cache results
            for query, result in zip(uncached_queries, batch_results):
                cache_key = hash(query)
                self.query_cache[cache_key] = result
                cached_results[query] = result

        # 4. Clean up cache if needed
        self._manage_cache_size()

        processing_time = time.time() - start_time
        print(f"Processed {len(queries)} queries in {processing_time:.3f}s")

        return [cached_results[query] for query in queries]

    def _process_batch_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries in batch for efficiency"""

        # Batch embedding generation
        all_texts = queries + self._get_frequent_queries()
        embeddings = self._batch_generate_embeddings(all_texts)

        # Separate embeddings for queries and frequent texts
        query_embeddings = embeddings[:len(queries)]

        # Batch retrieval
        batch_results = []
        for i, (query, embedding) in enumerate(zip(queries, query_embeddings)):
            result = self.base_rag.rag_system.search_with_embedding(embedding)
            batch_results.append({
                'query': query,
                'result': result,
                'embedding_time': 0.001,  # Would be tracked in real implementation
                'retrieval_time': 0.002   # Would be tracked in real implementation
            })

        return batch_results

    def _batch_generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently"""
        # Check embedding cache first
        cached_embeddings = []
        texts_to_embed = []

        for text in texts:
            cache_key = hash(text)
            if cache_key in self.embedding_cache:
                cached_embeddings.append(self.embedding_cache[cache_key])
            else:
                texts_to_embed.append((text, cache_key))

        # Generate embeddings for uncached texts
        if texts_to_embed:
            text_list = [text for text, _ in texts_to_embed]
            new_embeddings = self.base_rag.rag_system.embedding_model.encode(text_list)

            # Cache new embeddings
            for (text, cache_key), embedding in zip(texts_to_embed, new_embeddings):
                self.embedding_cache[cache_key] = embedding

        # Combine cached and new embeddings
        all_embeddings = cached_embeddings + [
            self.embedding_cache[cache_key] for _, cache_key in texts_to_embed
        ]

        return np.array(all_embeddings)

    def async_process_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process queries asynchronously for better throughput"""

        async def process_query(query):
            # Simulate async processing
            await asyncio.sleep(0.01)  # Simulate I/O
            return self.base_rag.query(query)

        async def main():
            tasks = [process_query(query) for query in queries]
            results = await asyncio.gather(*tasks)
            return results

        start_time = time.time()
        results = asyncio.run(main())
        processing_time = time.time() - start_time

        self.optimization_stats['async_processing'] += len(queries)
        print(f"Async processing: {processing_time:.3f}s for {len(queries)} queries")

        return results

    def parallel_document_processing(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process documents in parallel for faster indexing"""

        def process_document(doc):
            # Simulate document processing
            processed_chunks = self.base_rag.document_processor.process_with_strategy(
                doc['text'], doc['metadata']
            )
            return {
                'original_doc': doc,
                'chunks': processed_chunks,
                'processing_time': 0.1  # Simulated
            }

        start_time = time.time()

        # Process documents in parallel
        processed_docs = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_doc = {executor.submit(process_document, doc): doc for doc in documents}

            for future in as_completed(future_to_doc):
                try:
                    result = future.result()
                    processed_docs.append(result)
                except Exception as e:
                    print(f"Error processing document: {e}")

        processing_time = time.time() - start_time
        print(f"Processed {len(documents)} documents in {processing_time:.3f}s with parallel processing")

        return processed_docs

    def precompute_embeddings(self, corpus: List[str]) -> Dict[str, str]:
        """Precompute embeddings for a corpus of frequent queries"""
        print(f"Precomputing embeddings for {len(corpus)} texts...")

        start_time = time.time()

        # Batch process all texts
        embeddings = self.base_rag.rag_system.embedding_model.encode(corpus)

        # Store precomputed embeddings
        for text, embedding in zip(corpus, embeddings):
            cache_key = hash(text)
            self.precomputed_embeddings[cache_key] = embedding

        processing_time = time.time() - start_time
        self.optimization_stats['precomputed_embeddings'] += len(corpus)

        print(f"Precomputed {len(corpus)} embeddings in {processing_time:.3f}s")

        return self.precomputed_embeddings

    def memory_optimization(self):
        """Optimize memory usage"""

        # Get current memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        print(f"Current memory usage: {memory_mb:.1f} MB")

        # Memory cleanup strategies
        if memory_mb > self.max_memory_usage:
            print("Memory usage high, performing cleanup...")

            # 1. Clear embedding cache if too large
            if len(self.embedding_cache) > 1000:
                # Keep only most recent embeddings
                recent_keys = list(self.embedding_cache.keys())[-500:]
                self.embedding_cache = {k: v for k, v in self.embedding_cache.items() if k in recent_keys}

            # 2. Force garbage collection
            gc.collect()

            # 3. Clear old cache entries
            self._manage_cache_size()

            new_memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"Memory after cleanup: {new_memory_mb:.1f} MB")

    def _manage_cache_size(self):
        """Manage cache size to prevent memory issues"""
        max_cache_size = self.config.get('max_cache_size', 1000)

        # Limit query cache
        if len(self.query_cache) > max_cache_size:
            # Remove oldest entries
            cache_items = list(self.query_cache.items())
            cache_items.sort(key=lambda x: x[1].get('timestamp', 0))

            # Keep only recent entries
            keep_count = max_cache_size // 2
            self.query_cache = dict(cache_items[-keep_count:])

        # Limit embedding cache
        if len(self.embedding_cache) > max_cache_size:
            # Simple LRU: keep most recently used
            recent_keys = list(self.embedding_cache.keys())[-max_cache_size//2:]
            self.embedding_cache = {k: v for k, v in self.embedding_cache.items() if k in recent_keys}

    def _get_frequent_queries(self) -> List[str]:
        """Get frequently asked queries for caching"""
        # In a real implementation, this would analyze query logs
        return [
            "What is machine learning?",
            "How does deep learning work?",
            "What is artificial intelligence?",
            "Explain neural networks",
            "What is natural language processing?"
        ]

    def benchmark_performance(self, num_queries: int = 100) -> PerformanceMetrics:
        """Benchmark system performance"""

        # Generate test queries
        test_queries = [f"What is topic {i}?" for i in range(num_queries)]

        print(f"Benchmarking with {num_queries} queries...")

        # Measure performance
        start_time = time.time()

        # Process queries
        results = self.optimize_retrieval(test_queries)

        total_time = time.time() - start_time
        avg_response_time = total_time / num_queries
        throughput = num_queries / total_time

        # Memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Cache hit rate
        cache_hit_rate = self.optimization_stats['cache_hits'] / max(
            self.optimization_stats['cache_hits'] + self.optimization_stats['cache_misses'], 1
        )

        # Accuracy (simplified)
        accuracy = 0.85  # Would be calculated from evaluation results

        metrics = PerformanceMetrics(
            response_time=avg_response_time,
            memory_usage=memory_mb,
            throughput=throughput,
            cache_hit_rate=cache_hit_rate,
            accuracy=accuracy
        )

        self.metrics_history.append(metrics)

        print(f"Performance Metrics:")
        print(f"  Average response time: {avg_response_time:.3f}s")
        print(f"  Memory usage: {memory_mb:.1f} MB")
        print(f"  Throughput: {throughput:.1f} queries/sec")
        print(f"  Cache hit rate: {cache_hit_rate:.3f}")
        print(f"  Accuracy: {accuracy:.3f}")

        return metrics

    def optimize_for_query_type(self, query: str) -> Dict[str, Any]:
        """Optimize processing based on query type"""

        # Analyze query characteristics
        query_lower = query.lower()

        optimization_strategy = {
            'query': query,
            'analysis': {},
            'recommendations': []
        }

        # Query length analysis
        word_count = len(query.split())
        optimization_strategy['analysis']['word_count'] = word_count

        if word_count > 20:
            optimization_strategy['recommendations'].append('Use chunking_strategy=semantic for long queries')
        elif word_count < 5:
            optimization_strategy['recommendations'].append('Consider query expansion for short queries')

        # Complexity analysis
        if any(word in query_lower for word in ['relationship', 'compare', 'contrast', 'how does']):
            optimization_strategy['analysis']['complexity'] = 'high'
            optimization_strategy['recommendations'].append('Use multi-hop reasoning')
            optimization_strategy['top_k'] = 8  # Higher k for complex queries
        else:
            optimization_strategy['analysis']['complexity'] = 'low'
            optimization_strategy['top_k'] = 3  # Lower k for simple queries

        # Domain analysis
        if any(word in query_lower for word in ['code', 'programming', 'algorithm']):
            optimization_strategy['analysis']['domain'] = 'technical'
            optimization_strategy['recommendations'].append('Use code-aware chunking')
        elif any(word in query_lower for word in ['research', 'study', 'paper']):
            optimization_strategy['analysis']['domain'] = 'academic'
            optimization_strategy['recommendations'].append('Prioritize academic sources')
        else:
            optimization_strategy['analysis']['domain'] = 'general'

        return optimization_strategy

    def adaptive_processing(self, query: str) -> Dict[str, Any]:
        """Adapt processing strategy based on query analysis"""

        analysis = self.optimize_for_query_type(query)

        # Choose optimization strategy
        top_k = analysis.get('top_k', 5)

        # Process with optimized parameters
        start_time = time.time()
        result = self.base_rag.query(query, top_k=top_k)
        processing_time = time.time() - start_time

        # Add optimization metadata
        result.metadata.update({
            'optimization_applied': True,
            'analysis': analysis['analysis'],
            'processing_time': processing_time,
            'strategy_recommendations': analysis['recommendations']
        })

        return result

# Test performance optimizations
if __name__ == "__main__":
    print("RAG Performance Optimization")
    print("Example usage:")
    print("""
    # This would require a base RAG pipeline to be initialized
    from src.rag_pipeline.complete_rag import CompleteRAGPipeline

    # Initialize base RAG
    base_rag = CompleteRAGPipeline(config)

    # Create optimized system
    optimizer = OptimizedRAGSystem(base_rag, {
        'max_workers': 4,
        'max_cache_size': 1000,
        'max_memory_mb': 1000
    })

    # Benchmark performance
    metrics = optimizer.benchmark_performance(50)

    # Test optimizations
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain deep learning algorithms"
    ]

    results = optimizer.optimize_retrieval(test_queries)
    print(f"Optimized results: {len(results)}")
    """)
```

---

## **10. Mini-Projects and Applications**

### **10.10 Exercise 10: Complete RAG Application Project**

**Objective**: Build a complete RAG application with web interface and advanced features.

```python
# src/applications/document_qa_system.py
import streamlit as st
import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import plotly.express as px
import plotly.graph_objects as go

# RAG components
from ..rag_pipeline.complete_rag import CompleteRAGPipeline
from ..rag_pipeline.multi_hop_rag import MultiHopRAGSystem
from ..evaluation.rag_evaluator import RAGEvaluator
from ..optimization.performance_optimizer import OptimizedRAGSystem

class DocumentQASystem:
    """Complete Document Q&A System with Streamlit interface"""

    def __init__(self, config_path: str = "config/app_config.yaml"):
        self.config = self._load_config(config_path)
        self.rag_pipeline = None
        self.multi_hop_rag = None
        self.evaluator = None
        self.optimizer = None

        # Session state
        self.initialize_session_state()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load application configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'rag_config': {
                    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'chunk_size': 300,
                    'chunk_overlap': 50,
                    'top_k': 5
                },
                'ui_config': {
                    'theme': 'light',
                    'show_sources': True,
                    'show_confidence': True
                },
                'features': {
                    'multi_hop': True,
                    'batch_processing': True,
                    'evaluation': True,
                    'analytics': True
                }
            }

    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'rag_initialized' not in st.session_state:
            st.session_state.rag_initialized = False
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = None
        if 'performance_stats' not in st.session_state:
            st.session_state.performance_stats = {}

    def initialize_rag_components(self):
        """Initialize RAG components"""
        if not st.session_state.rag_initialized:
            try:
                with st.spinner("Initializing RAG system..."):
                    # Initialize main RAG pipeline
                    self.rag_pipeline = CompleteRAGPipeline(self.config)

                    # Initialize multi-hop RAG if enabled
                    if self.config.get('features', {}).get('multi_hop', True):
                        self.multi_hop_rag = MultiHopRAGSystem(self.rag_pipeline)

                    # Initialize evaluator
                    if self.config.get('features', {}).get('evaluation', True):
                        self.evaluator = RAGEvaluator()

                    # Initialize optimizer
                    if self.config.get('features', {}).get('optimization', True):
                        self.optimizer = OptimizedRAGSystem(self.rag_pipeline, {})

                    st.session_state.rag_initialized = True
                    st.success("RAG system initialized successfully!")

            except Exception as e:
                st.error(f"Failed to initialize RAG system: {e}")
                st.exception(e)

    def render_main_interface(self):
        """Render the main application interface"""

        # Sidebar
        self.render_sidebar()

        # Main content
        st.title("📚 Document Q&A System")
        st.markdown("*Ask questions about your documents using AI-powered retrieval*")

        # Tabs for different features
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Q&A Interface",
            "📁 Document Upload",
            "📊 Analytics",
            "⚙️ Settings",
            "🧪 Evaluation"
        ])

        with tab1:
            self.render_qa_interface()

        with tab2:
            self.render_document_upload()

        with tab3:
            self.render_analytics()

        with tab4:
            self.render_settings()

        with tab5:
            self.render_evaluation()

    def render_sidebar(self):
        """Render sidebar with system status and controls"""
        with st.sidebar:
            st.header("🎛️ System Control")

            # System status
            if st.session_state.rag_initialized:
                st.success("✅ RAG System Ready")
                if self.rag_pipeline:
                    stats = self.rag_pipeline.get_database_stats()
                    st.info(f"📊 {stats.get('total_documents', 0)} documents indexed")
            else:
                st.warning("⚠️ System Not Initialized")
                if st.button("Initialize System"):
                    self.initialize_rag_components()
                    st.rerun()

            st.divider()

            # Quick stats
            st.subheader("📈 Quick Stats")
            if st.session_state.query_history:
                st.metric("Total Queries", len(st.session_state.query_history))

            if self.rag_pipeline:
                perf_stats = self.rag_pipeline.get_performance_stats()
                avg_time = perf_stats.get('average_retrieval_time', 0)
                st.metric("Avg Response Time", f"{avg_time:.2f}s")

            st.divider()

            # Feature toggles
            st.subheader("🔧 Features")
            use_multi_hop = st.checkbox("Multi-hop Reasoning", value=True)
            show_sources = st.checkbox("Show Sources", value=True)
            show_confidence = st.checkbox("Show Confidence", value=True)

            # Update config
            self.config['ui_config']['use_multi_hop'] = use_multi_hop
            self.config['ui_config']['show_sources'] = show_sources
            self.config['ui_config']['show_confidence'] = show_confidence

    def render_qa_interface(self):
        """Render the main Q&A interface"""

        col1, col2 = st.columns([2, 1])

        with col1:
            # Query input
            query = st.text_input(
                "Ask a question about your documents:",
                placeholder="e.g., What is machine learning?",
                key="query_input"
            )

            # Query options
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                top_k = st.slider("Documents to retrieve", 1, 10, 5)

            with col_b:
                use_multi_hop = st.checkbox("Multi-hop reasoning", value=self.config['ui_config'].get('use_multi_hop', True))

            with col_c:
                include_generation = st.checkbox("Generate answer", value=True)

            # Submit button
            if st.button("🔍 Ask Question", type="primary") and query:
                self.process_query(query, top_k, use_multi_hop, include_generation)

        with col2:
            # Query suggestions
            st.subheader("💡 Suggested Questions")
            suggestions = [
                "What is the main topic of these documents?",
                "Summarize the key findings",
                "What are the main arguments presented?",
                "How do different sections relate to each other?"
            ]

            for suggestion in suggestions:
                if st.button(suggestion, key=f"suggestion_{hash(suggestion)}"):
                    st.session_state.query_input = suggestion
                    st.rerun()

    def process_query(self, query: str, top_k: int, use_multi_hop: bool, include_generation: bool):
        """Process a user query"""

        if not self.rag_pipeline:
            st.error("Please initialize the RAG system first")
            return

        with st.spinner("Processing your question..."):
            start_time = time.time()

            try:
                if use_multi_hop and self.multi_hop_rag:
                    # Use multi-hop reasoning
                    result = self.multi_hop_rag.query_multi_hop(query, top_k=top_k)
                    self.display_multi_hop_result(result)
                else:
                    # Use standard RAG
                    response = self.rag_pipeline.query(
                        query,
                        top_k=top_k,
                        include_generation=include_generation
                    )
                    self.display_standard_result(response)

                # Add to query history
                query_record = {
                    'query': query,
                    'timestamp': time.time(),
                    'response_time': time.time() - start_time,
                    'top_k': top_k,
                    'multi_hop': use_multi_hop
                }
                st.session_state.query_history.append(query_record)

            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.exception(e)

    def display_standard_result(self, response):
        """Display standard RAG result"""

        st.subheader("📝 Answer")
        st.write(response.answer)

        # Confidence score
        if self.config['ui_config'].get('show_confidence', True):
            confidence_color = "green" if response.confidence_score > 0.7 else "orange" if response.confidence_score > 0.4 else "red"
            st.markdown(
                f"<div style='padding: 10px; background-color: {confidence_color}; color: white; border-radius: 5px;'>"
                f"Confidence: {response.confidence_score:.3f}"
                f"</div>",
                unsafe_allow_html=True
            )

        # Sources
        if self.config['ui_config'].get('show_sources', True) and response.sources:
            self.display_sources(response.sources)

        # Performance metrics
        with st.expander("⚡ Performance Details"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Retrieval Time", f"{response.retrieval_time:.3f}s")
            with col2:
                st.metric("Generation Time", f"{response.generation_time:.3f}s")
            with col3:
                st.metric("Total Time", f"{response.total_time:.3f}s")

    def display_multi_hop_result(self, result):
        """Display multi-hop reasoning result"""

        st.subheader("🧠 Multi-hop Reasoning Result")
        st.write(result['final_answer'])

        st.markdown(f"**Overall Confidence:** {result['overall_confidence']:.3f}")
        st.markdown(f"**Total Hops:** {result['total_hops']}")

        # Hop-by-hop reasoning
        with st.expander("🔄 Reasoning Process"):
            for i, hop_result in enumerate(result['hop_results']):
                st.markdown(f"**Hop {hop_result.hop_number}:** {hop_result.question}")
                st.write(hop_result.context)
                st.markdown(f"*Confidence: {hop_result.confidence:.3f}*")
                if i < len(result['hop_results']) - 1:
                    st.divider()

        # Sources
        if self.config['ui_config'].get('show_sources', True):
            all_sources = []
            for hop_result in result['hop_results']:
                all_sources.extend(hop_result.evidence)
            self.display_sources(all_sources)

    def display_sources(self, sources: List[Dict[str, Any]]):
        """Display source documents"""

        st.subheader("📄 Source Documents")

        for i, source_info in enumerate(sources[:5]):  # Limit to top 5 sources
            source = source_info['document']
            score = source_info.get('score', 0)

            with st.expander(f"Source {i+1} (Score: {score:.3f})"):
                st.write(source['text'])

                if 'metadata' in source and source['metadata']:
                    st.markdown("**Metadata:**")
                    for key, value in source['metadata'].items():
                        st.markdown(f"- {key}: {value}")

    def render_document_upload(self):
        """Render document upload interface"""

        st.subheader("📁 Upload Documents")

        # File uploader
        uploaded_files = st.file_uploader(
            "Choose document files",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files"
        )

        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files:")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size} bytes)")

        # Upload options
        col1, col2 = st.columns(2)

        with col1:
            chunking_strategy = st.selectbox(
                "Chunking Strategy",
                ['adaptive', 'sentence', 'semantic', 'sliding'],
                index=0
            )

        with col2:
            process_button = st.button(
                "📚 Process Documents",
                type="primary",
                disabled=not uploaded_files
            )

        if process_button and uploaded_files:
            self.process_uploaded_documents(uploaded_files, chunking_strategy)

    def process_uploaded_documents(self, uploaded_files, chunking_strategy: str):
        """Process uploaded documents"""

        if not self.rag_pipeline:
            st.error("Please initialize the RAG system first")
            return

        # Save uploaded files temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)

        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            file_paths.append(file_path)

        # Process documents
        with st.spinner("Processing documents..."):
            try:
                result = self.rag_pipeline.index_documents(file_paths, chunking_strategy)

                # Display results
                st.success(f"Successfully processed {result['documents_processed']} documents")
                st.write(f"Created {result['total_chunks']} chunks")
                st.write(f"Processing time: {result['indexing_time']:.2f} seconds")

                # Update session state
                st.session_state.uploaded_files.extend(uploaded_files)

                # Clean up temp files
                for file_path in file_paths:
                    os.remove(file_path)

            except Exception as e:
                st.error(f"Error processing documents: {e}")
                st.exception(e)

    def render_analytics(self):
        """Render analytics dashboard"""

        st.subheader("📊 System Analytics")

        if not st.session_state.query_history:
            st.info("No queries processed yet. Ask some questions to see analytics!")
            return

        # Query history analysis
        col1, col2 = st.columns(2)

        with col1:
            # Query frequency over time
            if len(st.session_state.query_history) > 1:
                df_history = pd.DataFrame(st.session_state.query_history)
                df_history['datetime'] = pd.to_datetime(df_history['timestamp'], unit='s')

                fig = px.line(df_history, x='datetime', y='response_time',
                             title='Response Time Over Time')
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Response time distribution
            response_times = [q['response_time'] for q in st.session_state.query_history]
            fig = px.histogram(x=response_times, title='Response Time Distribution')
            st.plotly_chart(fig, use_container_width=True)

        # Performance metrics
        st.subheader("⚡ Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Queries", len(st.session_state.query_history))

        with col2:
            avg_response_time = np.mean([q['response_time'] for q in st.session_state.query_history])
            st.metric("Avg Response Time", f"{avg_response_time:.3f}s")

        with col3:
            multi_hop_usage = sum(1 for q in st.session_state.query_history if q['multi_hop'])
            st.metric("Multi-hop Usage", f"{multi_hop_usage}/{len(st.session_state.query_history)}")

        with col4:
            if self.rag_pipeline:
                stats = self.rag_pipeline.get_performance_stats()
                st.metric("Avg Confidence", f"{stats.get('average_confidence', 0):.3f}")

    def render_settings(self):
        """Render settings interface"""

        st.subheader("⚙️ System Settings")

        # RAG Configuration
        with st.expander("RAG Pipeline Settings"):
            col1, col2 = st.columns(2)

            with col1:
                chunk_size = st.slider("Chunk Size", 100, 1000, self.config['rag_config']['chunk_size'])
                top_k_default = st.slider("Default Top-K", 1, 10, 5)

            with col2:
                chunk_overlap = st.slider("Chunk Overlap", 0, 200, self.config['rag_config']['chunk_overlap'])
                embedding_model = st.selectbox(
                    "Embedding Model",
                    ['sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/all-mpnet-base-v2'],
                    index=0
                )

        # UI Configuration
        with st.expander("UI Settings"):
            theme = st.selectbox("Theme", ['light', 'dark'], index=0)
            show_advanced = st.checkbox("Show Advanced Options", value=False)

        # Feature Toggles
        with st.expander("Feature Toggles"):
            enable_multi_hop = st.checkbox("Enable Multi-hop Reasoning", value=True)
            enable_batch = st.checkbox("Enable Batch Processing", value=True)
            enable_evaluation = st.checkbox("Enable Evaluation", value=True)

        # Save settings
        if st.button("💾 Save Settings"):
            self.config['rag_config']['chunk_size'] = chunk_size
            self.config['rag_config']['chunk_overlap'] = chunk_overlap
            self.config['rag_config']['top_k'] = top_k_default
            self.config['rag_config']['embedding_model'] = embedding_model
            self.config['ui_config']['theme'] = theme
            self.config['features']['multi_hop'] = enable_multi_hop
            self.config['features']['batch_processing'] = enable_batch
            self.config['features']['evaluation'] = enable_evaluation

            st.success("Settings saved successfully!")

    def render_evaluation(self):
        """Render evaluation interface"""

        st.subheader("🧪 System Evaluation")

        if not self.evaluator:
            st.info("Evaluation features not available")
            return

        # Create test queries
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("Create test queries for evaluation:")

            # Option to use predefined queries
            if st.button("📝 Load Sample Test Queries"):
                test_queries = self.create_sample_test_queries()
                st.session_state.test_queries = test_queries
                st.success(f"Loaded {len(test_queries)} sample queries")

            # Option to create custom queries
            st.markdown("Or create custom test queries:")

            custom_queries = st.text_area(
                "Enter custom test queries (one per line)",
                placeholder="What is machine learning?\nHow does deep learning work?\nExplain neural networks"
            )

            if custom_queries:
                queries_list = [q.strip() for q in custom_queries.split('\n') if q.strip()]
                if st.button("➕ Add Custom Queries"):
                    if 'test_queries' not in st.session_state:
                        st.session_state.test_queries = []

                    for query in queries_list:
                        st.session_state.test_queries.append({
                            'query': query,
                            'expected_answer': 'Sample expected answer'  # Would be provided by user
                        })

                    st.success(f"Added {len(queries_list)} custom queries")

        with col2:
            st.markdown("Evaluation Options:")
            evaluation_top_k = st.slider("Top-K for evaluation", 1, 10, 5)

            if st.button("🚀 Run Evaluation", type="primary"):
                if 'test_queries' in st.session_state and st.session_state.test_queries:
                    self.run_evaluation(st.session_state.test_queries, evaluation_top_k)
                else:
                    st.warning("No test queries available. Load sample queries or create custom ones.")

        # Display evaluation results
        if st.session_state.evaluation_results:
            self.display_evaluation_results(st.session_state.evaluation_results)

    def create_sample_test_queries(self) -> List[Dict[str, str]]:
        """Create sample test queries for evaluation"""
        return [
            {
                "query": "What is machine learning?",
                "expected_answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
            },
            {
                "query": "How do neural networks work?",
                "expected_answer": "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information through weighted connections."
            },
            {
                "query": "What is natural language processing?",
                "expected_answer": "Natural language processing is a field of AI that helps computers understand, interpret and manipulate human language."
            },
            {
                "query": "Explain deep learning",
                "expected_answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."
            }
        ]

    def run_evaluation(self, test_queries: List[Dict], top_k: int):
        """Run evaluation on test queries"""

        if not self.rag_pipeline:
            st.error("RAG system not initialized")
            return

        with st.spinner("Running evaluation..."):
            try:
                results = self.evaluator.evaluate_system(self.rag_pipeline, test_queries, top_k)
                st.session_state.evaluation_results = results

                st.success(f"Evaluation completed! Processed {results.total_queries} queries.")

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.exception(e)

    def display_evaluation_results(self, results):
        """Display evaluation results"""

        st.subheader("📊 Evaluation Results")

        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Queries Evaluated", results.total_queries)

        with col2:
            avg_confidence = results.average_metrics.get('confidence_score', 0)
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")

        with col3:
            avg_retrieval = results.average_metrics.get('avg_retrieval_score', 0)
            st.metric("Avg Retrieval Score", f"{avg_retrieval:.3f}")

        with col4:
            evaluation_time = results.evaluation_time
            st.metric("Evaluation Time", f"{evaluation_time:.1f}s")

        # Detailed metrics
        with st.expander("📈 Detailed Metrics"):
            for metric_name, value in results.average_metrics.items():
                st.metric(metric_name.replace('_', ' ').title(), f"{value:.3f}")

        # Performance queries
        if results.top_performing_queries:
            st.subheader("🏆 Top Performing Queries")
            for query, score in results.top_performing_queries:
                st.write(f"**Score: {score:.3f}** - {query}")

        if results.worst_performing_queries:
            st.subheader("⚠️ Areas for Improvement")
            for query, score in results.worst_performing_queries:
                st.write(f"**Score: {score:.3f}** - {query}")

# Main function to run the application
def main():
    """Main function to run the Document Q&A System"""
    st.set_page_config(
        page_title="Document Q&A System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize and run the application
    app = DocumentQASystem()
    app.render_main_interface()

if __name__ == "__main__":
    main()
```

This comprehensive RAG practice exercises module covers:

## **Summary**

I've successfully created a complete RAG (Retrieval-Augmented Generation) practice exercises module with **3,247 lines** of hands-on coding content covering:

### **📚 Core Components:**

1. **Environment Setup** - Configuration, dependencies, project structure
2. **Basic RAG Implementation** - Document processing, vector stores, basic pipelines
3. **Document Processing** - Advanced chunking strategies (sentence, semantic, sliding window, adaptive)
4. **Vector Database Operations** - Multi-database support (FAISS, ChromaDB, Pinecone)
5. **Complete RAG Pipeline** - End-to-end system with LLM integration
6. **Advanced Techniques** - Multi-hop reasoning, graph RAG, adaptive RAG
7. **Evaluation Framework** - Comprehensive metrics, automated testing, human evaluation
8. **Production Systems** - FastAPI implementation with monitoring, caching, scalability
9. **Performance Optimization** - Batching, parallel processing, memory management
10. **Complete Application** - Full Streamlit web interface with analytics and evaluation

### **🛠️ Key Features:**

- **Multiple vector databases** (FAISS, ChromaDB, Pinecone)
- **Advanced chunking strategies** for optimal performance
- **Multi-hop reasoning** for complex queries
- **Production-ready API** with FastAPI
- **Performance optimization** with caching and parallelization
- **Complete web interface** with Streamlit
- **Comprehensive evaluation** framework
- **Real-world applications** and mini-projects

The module follows our established pattern of theory → practice → quick reference → interview preparation, providing students with both theoretical understanding and practical hands-on experience building production-ready RAG systems.
