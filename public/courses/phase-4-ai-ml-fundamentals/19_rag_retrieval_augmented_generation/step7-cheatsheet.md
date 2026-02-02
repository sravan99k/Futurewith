# ⚡ **RAG Cheatsheet**

_Quick reference guide for Retrieval-Augmented Generation systems_

---

## 📋 **Table of Contents**

1. [Core RAG Concepts](#1-core-rag-concepts)
2. [Architecture Quick Reference](#2-architecture-quick-reference)
3. [Implementation Patterns](#3-implementation-patterns)
4. [Database Comparison](#4-database-comparison)
5. [Optimization Techniques](#5-optimization-techniques)
6. [Common Issues & Solutions](#6-common-issues--solutions)
7. [Performance Tuning](#7-performance-tuning)
8. [Production Deployment](#8-production-deployment)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Code Snippets](#10-code-snippets)

---

## **1. Core RAG Concepts**

### **1.1 RAG Formula**

```
RAG = Retrieval + Augmentation + Generation
```

### **1.2 Key Components**

- **Query Processing** → Understand user intent
- **Document Retrieval** → Find relevant information
- **Context Building** → Prepare context for generation
- **Response Generation** → Create final answer
- **Source Attribution** → Provide citations

### **1.3 RAG vs Traditional LLM**

| **Aspect**    | **Traditional LLM**  | **RAG System**          |
| ------------- | -------------------- | ----------------------- |
| Knowledge     | Static training data | Dynamic retrieval       |
| Freshness     | Training cutoff      | Real-time updates       |
| Factuality    | May hallucinate      | Source-grounded         |
| Customization | Requires fine-tuning | Add documents instantly |
| Transparency  | Black box            | Full attribution        |

---

## **2. Architecture Quick Reference**

### **2.1 Basic RAG Pipeline**

```
User Query → Query Processing → Retrieval → Context Building → LLM Generation → Response + Sources
```

### **2.2 Multi-Hop RAG Pipeline**

```
Query → Hop 1 (Retrieve → Generate) → Hop 2 (Retrieve → Generate) → ... → Final Synthesis
```

### **2.3 Component Selection Guide**

| **Component**       | **Options**                    | **Use Case**                                            |
| ------------------- | ------------------------------ | ------------------------------------------------------- |
| **Embedding Model** | OpenAI Ada, Sentence-BERT, BGE | General purpose → OpenAI, Multi-lingual → Sentence-BERT |
| **Vector Database** | FAISS, ChromaDB, Pinecone      | Local/Prototyping → FAISS, Production → Pinecone        |
| **LLM**             | GPT-4, Claude, Llama           | Quality → GPT-4, Cost-effective → Llama                 |
| **Chunking**        | Fixed, Semantic, Adaptive      | Code → Fixed, General → Semantic, Unknown → Adaptive    |

---

## **3. Implementation Patterns**

### **3.1 Basic RAG Implementation**

```python
# Minimal RAG setup
from sentence_transformers import SentenceTransformer
import faiss

# 1. Initialize components
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatIP(384)

# 2. Index documents
documents = ["doc1", "doc2", "doc3"]
embeddings = embedder.encode(documents)
index.add(embeddings)

# 3. Query
query_embedding = embedder.encode(["query"])[0]
scores, indices = index.search(query_embedding.reshape(1, -1), 3)
```

### **3.2 LangChain Implementation**

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Setup
embeddings = OpenAIEmbeddings()
vectorstore = Chroma("docs", embeddings)
llm = OpenAI()

# RAG Chain
rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query
result = rag.run("What is machine learning?")
```

### **3.3 Custom Implementation Pattern**

```python
class SimpleRAG:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectorstore = FAISSIndex()
        self.llm = OpenAILLM()

    def query(self, question, top_k=5):
        # Retrieve
        query_embedding = self.embedder.encode([question])[0]
        relevant_docs = self.vectorstore.search(query_embedding, top_k)

        # Build context
        context = self._build_context(relevant_docs)

        # Generate
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        answer = self.llm.generate(prompt)

        return answer, relevant_docs
```

---

## **4. Database Comparison**

### **4.1 Vector Database Comparison**

| **Database** | **Pros**                          | **Cons**                    | **Best For**              |
| ------------ | --------------------------------- | --------------------------- | ------------------------- |
| **FAISS**    | Fast, GPU support, Local          | No persistence, Complex API | Research, Local dev       |
| **ChromaDB** | Simple API, Good docs, Persistent | Slower for large datasets   | Prototyping, Small-medium |
| **Pinecone** | Scalable, Cloud-native, Managed   | Expensive, Vendor lock-in   | Production, Enterprise    |
| **Weaviate** | Graph features, Open source       | Complex setup               | Graph-heavy applications  |

### **4.2 Embedding Model Comparison**

| **Model**         | **Dimensions** | **Speed** | **Quality** | **Use Case**       |
| ----------------- | -------------- | --------- | ----------- | ------------------ |
| **OpenAI Ada**    | 1536           | Medium    | High        | General production |
| **Sentence-BERT** | 768-1024       | Fast      | Medium      | Multi-lingual      |
| **BGE**           | 768-1024       | Fast      | High        | Chinese/English    |
| **E5**            | 1024           | Fast      | High        | Instruction-tuned  |

### **4.3 Chunking Strategy Guide**

| **Strategy**       | **Size** | **Overlap** | **Best For**     | **Pros/Cons**                   |
| ------------------ | -------- | ----------- | ---------------- | ------------------------------- |
| **Fixed Size**     | 256-512  | 50-100      | Simple documents | + Fast, - May break context     |
| **Semantic**       | Variable | 20-50       | Complex docs     | + Preserves meaning, - Slower   |
| **Sliding Window** | 512      | 50-100      | Large docs       | + Good coverage, - More storage |
| **Adaptive**       | Variable | Dynamic     | Unknown content  | + Optimized, - Complex logic    |

---

## **5. Optimization Techniques**

### **5.1 Query Optimization**

```python
# Query expansion
def expand_query(query):
    expansions = {
        "what is": ["define", "explain", "describe"],
        "how does": ["mechanism", "process", "method"],
        "why": ["reason", "cause", "explanation"]
    }

    expanded = query
    for key, values in expansions.items():
        if key in query.lower():
            expanded += " " + " ".join(values)

    return expanded

# Query classification
def classify_query(query):
    if len(query.split()) > 10:
        return "complex"  # Use multi-hop
    elif any(word in query.lower() for word in ["relationship", "compare"]):
        return "complex"  # Use higher top_k
    else:
        return "simple"
```

### **5.2 Caching Strategy**

```python
# Multi-level caching
class RAGCache:
    def __init__(self):
        self.query_cache = {}  # L1: In-memory
        self.result_cache = {}  # L2: Redis
        self.embedding_cache = {}  # L3: Embeddings

    def get(self, query):
        # Check L1 cache first
        if query in self.query_cache:
            return self.query_cache[query]

        # Check L2 cache
        result = self.redis_client.get(f"rag:{hash(query)}")
        if result:
            self.query_cache[query] = result
            return result

        return None
```

### **5.3 Batch Processing**

```python
# Batch retrieval
def batch_retrieve(queries, batch_size=32):
    results = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        batch_results = index.search(batch_embeddings, top_k=5)
        results.extend(batch_results)

    return results
```

---

## **6. Common Issues & Solutions**

### **6.1 Retrieval Issues**

| **Problem**          | **Solution**                                  | **Code Fix**                                   |
| -------------------- | --------------------------------------------- | ---------------------------------------------- |
| Low relevance scores | Increase top_k, try different embedding model | `top_k = min(top_k * 2, 10)`                   |
| No results returned  | Check embedding dimensions match              | `assert query_emb.shape[1] == index.dimension` |
| Slow retrieval       | Use approximate indexing                      | `index = faiss.IndexIVFPQ(...)`                |
| Memory issues        | Use incremental indexing                      | `index.add(embeddings.astype('float32'))`      |

### **6.2 Generation Issues**

| **Problem**          | **Solution**                             | **Code Fix**                                           |
| -------------------- | ---------------------------------------- | ------------------------------------------------------ |
| Hallucination        | Improve grounding, add citations         | `prompt += "\nCite sources: [1], [2]"`                 |
| Off-topic answers    | Improve retrieval, add context filtering | `filtered_docs = [doc for doc in docs if score > 0.7]` |
| Inconsistent answers | Add confidence scoring                   | `confidence = avg([doc['score'] for doc in docs])`     |
| Long generation time | Optimize prompt, use smaller model       | `model = "gpt-3.5-turbo"  # instead of gpt-4`          |

### **6.3 Performance Issues**

| **Problem**   | **Solution**                          | **Code Fix**                         |
| ------------- | ------------------------------------- | ------------------------------------ |
| Slow startup  | Lazy loading, pre-warm cache          | `if not model.loaded: model.load()`  |
| Memory usage  | Implement LRU cache, limit batch size | `cache = LRU(maxsize=1000)`          |
| High latency  | Async processing, connection pooling  | `asyncio.gather(*queries)`           |
| Rate limiting | Implement request queuing             | `queue = asyncio.Queue(maxsize=100)` |

---

## **7. Performance Tuning**

### **7.1 Query Performance Tuning**

```python
# Adaptive top_k based on query complexity
def get_optimal_top_k(query):
    word_count = len(query.split())
    if word_count < 5:
        return 3  # Simple query, fewer docs needed
    elif word_count < 15:
        return 5  # Medium complexity
    else:
        return 8  # Complex query, more context needed

# Dynamic chunking based on document type
def get_optimal_chunking(document_type, content):
    if document_type == "code":
        return {'strategy': 'fixed', 'size': 400, 'overlap': 50}
    elif document_type == "technical":
        return {'strategy': 'semantic', 'size': 600, 'overlap': 100}
    else:
        return {'strategy': 'adaptive', 'size': 500, 'overlap': 75}
```

### **7.2 Memory Optimization**

```python
# Memory-efficient embedding generation
def generate_embeddings_efficiently(texts, batch_size=32):
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)

        # Clear memory after each batch
        if i % (batch_size * 10) == 0:
            gc.collect()

    return np.vstack(embeddings)

# Index optimization for large datasets
def create_optimized_index(embeddings):
    # Use product quantization for large datasets
    nlist = 100  # Number of clusters
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIVFPQ(quantizer, embeddings.shape[1], nlist, 64, 8)

    # Train on subset first
    index.train(embeddings[:10000])
    index.add(embeddings)

    return index
```

### **7.3 Latency Optimization**

```python
# Async processing pipeline
async def async_rag_query(question):
    # Parallel retrieval and preprocessing
    retrieval_task = asyncio.create_task(retrieve_documents(question))
    embedding_task = asyncio.create_task(embed_question(question))

    # Wait for both to complete
    documents, query_embedding = await asyncio.gather(retrieval_task, embedding_task)

    # Parallel scoring and ranking
    scoring_tasks = [score_document(doc, query_embedding) for doc in documents]
    scores = await asyncio.gather(*scoring_tasks)

    # Select top documents
    top_docs = select_top_documents(documents, scores, k=5)

    return top_docs

# Connection pooling for external services
class RAGConnectionPool:
    def __init__(self):
        self.openai_pool = openai.Pool(max_connections=10)
        self.vector_pool = vector_db.Pool(max_connections=5)

    async def query(self, question):
        async with self.openai_pool.acquire() as llm:
            async with self.vector_pool.acquire() as vs:
                return await self.process_query(llm, vs, question)
```

---

## **8. Production Deployment**

### **8.1 FastAPI Setup**

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="RAG API", version="1.0.0")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/query")
async def query_rag(request: QueryRequest, background_tasks: BackgroundTasks):
    # Check cache first
    cached = cache.get(request.question)
    if cached:
        return cached

    # Process query
    result = rag_pipeline.query(request.question, request.top_k)

    # Background cache update
    background_tasks.add_task(cache.set, request.question, result)

    return result

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
```

### **8.2 Docker Configuration**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **8.3 Environment Variables**

```bash
# .env file
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
REDIS_URL=redis://localhost:6379
CHUNK_SIZE=512
TOP_K_DEFAULT=5
MAX_CONCURRENT_QUERIES=10
```

### **8.4 Monitoring Setup**

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries')
query_duration = Histogram('rag_query_duration_seconds', 'RAG query duration')
active_queries = Gauge('rag_active_queries', 'Active queries')
cache_hit_rate = Gauge('rag_cache_hit_rate', 'Cache hit rate')

# Use in code
def query_rag(question):
    query_counter.inc()
    query_duration.time()
    active_queries.inc()

    try:
        result = _process_query(question)
        return result
    finally:
        active_queries.dec()
```

---

## **9. Evaluation Metrics**

### **9.1 Retrieval Metrics**

| **Metric**      | **Formula**                 | **Good Value** | **Interpretation**                           |
| --------------- | --------------------------- | -------------- | -------------------------------------------- |
| **Precision@K** | Relevant/K                  | >0.7           | High precision means fewer irrelevant docs   |
| **Recall@K**    | Relevant Retrieved/Relevant | >0.8           | High recall means finding most relevant docs |
| **MRR**         | Mean Reciprocal Rank        | >0.6           | First relevant document ranking              |
| **NDCG**        | Normalized DCG              | >0.8           | Overall ranking quality                      |

```python
def calculate_retrieval_metrics(query, retrieved_docs, relevant_docs):
    k = len(retrieved_docs)
    relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]

    precision = len(relevant_retrieved) / k
    recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0

    mrr = 0
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            mrr = 1 / (i + 1)
            break

    return {
        'precision': precision,
        'recall': recall,
        'mrr': mrr
    }
```

### **9.2 Generation Metrics**

| **Metric**       | **Method**         | **Good Value** | **Purpose**                       |
| ---------------- | ------------------ | -------------- | --------------------------------- |
| **Groundedness** | Fact checking      | >0.8           | Is answer supported by sources?   |
| **Faithfulness** | Source alignment   | >0.7           | Does answer match source content? |
| **Completeness** | Coverage analysis  | >0.6           | Does answer address all aspects?  |
| **Coherence**    | Linguistic quality | >0.8           | Is answer well-structured?        |

```python
def calculate_groundedness(answer, sources):
    # Simple groundedness: count facts from sources
    answer_facts = extract_facts(answer)
    source_facts = [extract_facts(source['text']) for source in sources]

    grounded_facts = 0
    for fact in answer_facts:
        if any(fact in source_fact for source_fact in source_facts):
            grounded_facts += 1

    return grounded_facts / len(answer_facts) if answer_facts else 0
```

### **9.3 System Metrics**

| **Metric**         | **Target** | **Monitoring**      |
| ------------------ | ---------- | ------------------- |
| **Response Time**  | <2 seconds | P95 latency         |
| **Throughput**     | >10 QPS    | Requests per second |
| **Cache Hit Rate** | >60%       | Redis/memory cache  |
| **Error Rate**     | <1%        | 4xx/5xx responses   |
| **Availability**   | >99.9%     | Uptime monitoring   |

---

## **10. Code Snippets**

### **10.1 Quick Setup Templates**

#### **Minimal RAG (10 lines)**

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Setup
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatIP(384)
docs = ["doc1", "doc2", "doc3"]

# Index
embs = model.encode(docs)
index.add(embs.astype('float32'))

# Query
q_emb = model.encode(["query"])[0]
scores, idx = index.search(q_emb.reshape(1,-1), 2)
result = [docs[i] for i in idx[0]]
```

#### **LangChain Setup**

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Setup
embeddings = OpenAIEmbeddings()
db = Chroma.from_texts(documents, embeddings)
llm = OpenAI()
qa = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

# Query
result = qa.run("Your question here")
```

### **10.2 Advanced Patterns**

#### **Multi-Hop RAG**

```python
def multi_hop_rag(question, max_hops=3):
    current_question = question
    evidence = []

    for hop in range(max_hops):
        docs = retrieve(current_question)
        evidence.extend(docs)

        if sufficient_evidence(evidence, question):
            break

        current_question = generate_follow_up(question, evidence)

    return synthesize(evidence, question)
```

#### **Graph RAG**

```python
def graph_rag(question):
    # Extract entities
    entities = extract_entities(question)

    # Get graph context
    graph_paths = knowledge_graph.query(entities)

    # Get document context
    doc_context = vector_search(question)

    # Fusion
    fused_context = fuse_contexts(graph_paths, doc_context)

    return generate(fused_context, question)
```

### **10.3 Utility Functions**

#### **Query Classification**

```python
def classify_query(query):
    if any(word in query.lower() for word in ["relationship", "compare", "contrast"]):
        return "complex"
    elif len(query.split()) < 5:
        return "simple"
    elif "?" in query:
        return "factual"
    else:
        return "exploratory"
```

#### **Context Building**

```python
def build_context(documents, max_length=1000):
    context_parts = []
    current_length = 0

    for doc in documents:
        doc_text = f"[{len(context_parts)+1}] {doc['text']}"
        if current_length + len(doc_text) > max_length:
            break
        context_parts.append(doc_text)
        current_length += len(doc_text)

    return "\n\n".join(context_parts)
```

#### **Confidence Scoring**

```python
def calculate_confidence(documents, answer):
    if not documents:
        return 0.0

    # Factors
    avg_score = np.mean([doc['score'] for doc in documents])
    source_diversity = len(set(doc['source'] for doc in documents))
    answer_coverage = len(answer.split()) / 200  # Normalize answer length

    # Weighted combination
    confidence = (0.5 * avg_score +
                 0.3 * min(source_diversity / 3, 1.0) +
                 0.2 * min(answer_coverage, 1.0))

    return min(confidence, 1.0)
```

### **10.4 Error Handling**

#### **Robust Query Processing**

```python
def safe_query(rag_system, question, max_retries=3):
    for attempt in range(max_retries):
        try:
            return rag_system.query(question)
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    'answer': f"Error processing query: {str(e)}",
                    'sources': [],
                    'confidence': 0.0
                }
            time.sleep(2 ** attempt)  # Exponential backoff
```

#### **Fallback Strategies**

```python
def query_with_fallback(question):
    # Try RAG first
    try:
        return rag.query(question)
    except Exception:
        pass

    # Fallback to simple keyword search
    try:
        return keyword_search(question)
    except Exception:
        pass

    # Final fallback
    return {
        'answer': "I'm sorry, I couldn't find relevant information.",
        'sources': [],
        'confidence': 0.0
    }
```

### **10.5 Configuration Templates**

#### **Development Config**

```yaml
rag_config:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 256
  chunk_overlap: 50
  top_k: 5

vector_db:
  type: "faiss"
  index_type: "IndexFlatIP"

cache:
  enabled: false

logging:
  level: "DEBUG"
```

#### **Production Config**

```yaml
rag_config:
  embedding_model: "text-embedding-ada-002"
  chunk_size: 512
  chunk_overlap: 100
  top_k: 8

vector_db:
  type: "pinecone"
  index_name: "production-rag"
  metric: "cosine"

cache:
  enabled: true
  type: "redis"
  ttl: 3600

monitoring:
  prometheus: true
  metrics_port: 9090

scaling:
  max_concurrent: 50
  rate_limit: 1000
```

---

## **🎯 Quick Reference Commands**

### **Setup Commands**

```bash
# Install dependencies
pip install sentence-transformers faiss-cpu langchain openai

# Download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Initialize vector database
python -c "import faiss; index = faiss.IndexFlatIP(384); print('FAISS ready')"
```

### **Testing Commands**

```bash
# Test basic functionality
python test_basic_rag.py

# Run evaluation
python evaluate_rag.py --test_set test_queries.json

# Performance benchmark
python benchmark_rag.py --num_queries 1000

# Load test
python load_test.py --concurrent 10 --duration 300
```

### **Deployment Commands**

```bash
# Build Docker image
docker build -t rag-api .

# Run with docker-compose
docker-compose up -d

# Health check
curl http://localhost:8000/health

# Test API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?"}'
```

---

## **📊 Performance Benchmarks**

### **Typical Performance Ranges**

| **Operation**               | **Fast** | **Medium** | **Slow** |
| --------------------------- | -------- | ---------- | -------- |
| **Embedding Generation**    | <100ms   | 100-500ms  | >500ms   |
| **Vector Search (1M docs)** | <50ms    | 50-200ms   | >200ms   |
| **LLM Generation**          | <2s      | 2-5s       | >5s      |
| **End-to-end Query**        | <3s      | 3-10s      | >10s     |

### **Resource Requirements**

| **Scale**             | **Memory** | **CPU**  | **Storage** |
| --------------------- | ---------- | -------- | ----------- |
| **Development**       | 2GB        | 2 cores  | 10GB        |
| **Small Production**  | 8GB        | 4 cores  | 100GB       |
| **Medium Production** | 32GB       | 8 cores  | 1TB         |
| **Large Production**  | 128GB      | 32 cores | 10TB+       |

---

## **🔧 Troubleshooting Guide**

### **Common Error Messages**

| **Error**            | **Cause**                          | **Solution**                         |
| -------------------- | ---------------------------------- | ------------------------------------ |
| `Dimension mismatch` | Embedding/model dimension mismatch | Check model output dimension         |
| `Empty result set`   | No relevant documents found        | Lower threshold, check indexing      |
| `OutOfMemoryError`   | Too much data in memory            | Implement batching, increase memory  |
| `RateLimitError`     | API rate limits exceeded           | Add delays, use rate limiting        |
| `ConnectionTimeout`  | Network or service timeout         | Increase timeout, check connectivity |

### **Performance Issues**

| **Symptom**    | **Diagnosis**           | **Fix**                                |
| -------------- | ----------------------- | -------------------------------------- |
| Slow startup   | Model loading           | Use model caching                      |
| High latency   | Inefficient retrieval   | Optimize index, use approximate search |
| Memory leaks   | Cached data not cleared | Implement cache limits                 |
| Low throughput | Synchronous processing  | Use async/parallel processing          |
| Poor quality   | Bad retrieval           | Tune chunking, try different models    |

---

_This cheatsheet provides quick access to the most important RAG concepts, patterns, and solutions. Keep it handy during development and production deployment._
