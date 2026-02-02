# 📋 Knowledge Graphs Cheatsheet

_Quick reference guide for knowledge graph development and operations_

---

## 🏗️ **Quick Setup**

### **Essential Libraries**

```bash
# Install packages
pip install networkx rdflib py2neo sentence-transformers transformers spacy scikit-learn torch torch-geometric

# Download spaCy model
python -m spacy download en_core_web_sm
```

### **Basic Imports**

```python
import networkx as nx
import torch
import torch.nn as nn
from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef
from sentence_transformers import SentenceTransformer
import spacy
```

---

## 📊 **Graph Construction**

### **Simple Knowledge Graph**

```python
# Create basic KG
kg = SimpleKnowledgeGraph()

# Add entities
kg.add_entity("steve_jobs", "Person", {"name": "Steve Jobs", "birth_year": 1955})

# Add relationships
kg.add_relationship("steve_jobs", "co_founded", "apple_inc", {"year": 1976})

# Query
kg.get_relationships("steve_jobs")
kg.find_path("steve_jobs", "bill_gates")
```

### **RDF Knowledge Graph**

```python
# Create RDF KG
rdf_kg = RDFKnowledgeGraph("http://example.org/kg/")

# Add RDF data
rdf_kg.add_entity("steve_jobs", "Person", {"name": "Steve Jobs"})
rdf_kg.add_relationship("steve_jobs", "co_founded", "apple_inc")

# SPARQL query
query = """
SELECT ?person ?company WHERE {
    ?person kg:co_founded ?company .
}
"""
results = rdf_kg.sparql_query(query)
```

---

## 🔍 **Entity Operations**

### **Entity Extraction**

```python
# Extract entities from text
extractor = EntityExtractor()
entities = extractor.extract_entities("Steve Jobs founded Apple in 1976.")

# Extract relationships
relationships = extractor.extract_relationships(text, entities)
```

### **Entity Linking**

```python
# Link to knowledge graph
linker = EntityLinker(kg)
linked_entities = linker.batch_link_entities(entities)

# Find best match
candidates = linker.find_candidates("Steve Jobs")
```

---

## 🤖 **Embedding Models**

### **TransE Model**

```python
# Create model
model = TransE(num_entities=100, num_relations=10, embedding_dim=50)

# Training
trainer = TransETrainer(model)
dataset = TransETrainingDataset(triples, entity_to_id, relation_to_id)
dataloader = DataLoader(dataset, batch_size=32)

for epoch in range(100):
    loss = trainer.train_epoch(dataloader)
```

### **Similarity Analysis**

```python
# Calculate similarity
similarity_analyzer = EmbeddingSimilarity(model, entity_to_id)
similar_entities = similarity_analyzer.find_similar_entities("steve_jobs", top_k=5)

# Get embedding
embedding = similarity_analyzer.get_entity_embedding("steve_jobs")
```

---

## 🧠 **Graph Neural Networks**

### **Simple GCN**

```python
# Convert to PyTorch Geometric
kg_to_pyg = KnowledgeGraphGCN(kg)
pyg_data = kg_to_pyg.to_pytorch_geometric()

# Create model
model = SimpleGCN(num_features=64, num_classes=5, hidden_dim=32)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = F.nll_loss(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
```

---

## 🔗 **RAG Enhancement**

### **Graph-Enhanced RAG**

```python
# Create RAG system
rag = KnowledgeGraphRAG(kg)

# Query processing
result = rag.query("Who founded Apple?")

# Components
entities = rag.extract_entities_from_text(question)
matched = rag.find_entities_in_kg(entities)
context = rag.retrieve_graph_context(matched)
combined = rag.combine_contexts(doc_context, graph_context)
```

### **Question Answering**

```python
# QA system
qa = KnowledgeGraphQA(kg, rag)

# Answer question
answer = qa.answer_question("Who is Steve Jobs?")
```

---

## 📈 **Analytics & Metrics**

### **Centrality Measures**

```python
# Calculate centrality
analytics = GraphAnalytics(kg)
centrality = analytics.calculate_centrality_measures()

# Key measures
degree_centrality = nx.degree_centrality(graph)
betweenness_centrality = nx.betweenness_centrality(graph)
closeness_centrality = nx.closeness_centrality(graph)
pagerank = nx.pagerank(graph)
```

### **Community Detection**

```python
# Find communities
import networkx.algorithms.community as nx_comm
communities = nx_comm.greedy_modularity_communities(graph)
modularity = nx_comm.modularity(graph, communities)
```

### **Graph Statistics**

```python
# Basic metrics
stats = {
    'nodes': graph.number_of_nodes(),
    'edges': graph.number_of_edges(),
    'density': nx.density(graph),
    'connected': nx.is_connected(graph),
    'diameter': nx.diameter(graph),
    'avg_clustering': nx.average_clustering(graph)
}
```

---

## 🎯 **Advanced Operations**

### **Knowledge Graph Completion**

```python
# Complete missing links
completer = KnowledgeGraphCompleter(kg, model)
predictions = completer.complete_entity_neighborhood("steve_jobs", max_predictions=10)

# Validate predictions
validated = completer.validate_predictions(predictions, validation_method='confidence')
```

### **Recommendation System**

```python
# Build recommender
recommender = KnowledgeGraphRecommender(kg)

# Create user profile
profile = recommender.build_user_profile("user1", [
    {'entity': 'steve_jobs', 'type': 'like', 'score': 1.0}
])

# Get recommendations
recommendations = recommender.recommend_entities("user1", top_k=5)
```

### **Bias Detection**

```python
# Detect bias
bias_detector = KnowledgeGraphBiasDetector(kg)
analysis = bias_detector.analyze_representation()
stereotypes = bias_detector.detect_stereotypes()
```

---

## 💾 **Database Operations**

### **Neo4j Integration**

```python
# Connect to Neo4j (requires running instance)
driver = GraphDatabase.driver(uri, auth=(user, password))

# Cypher queries
query = """
MATCH (p:Person)-[r]-(connected)
WHERE p.name CONTAINS 'Steve'
RETURN p.name, type(r), connected.name
"""

with driver.session() as session:
    result = session.run(query)
    for record in result:
        print(record['p.name'], record['type(r)'], record['connected.name'])
```

### **SPARQL Queries**

```python
# Common SPARQL patterns
queries = {
    "get_entities_by_type": """
        SELECT ?entity WHERE {
            ?entity rdf:type kg:Person .
        }
    """,

    "find_relationships": """
        SELECT ?subject ?predicate ?object WHERE {
            ?subject ?predicate ?object .
            FILTER(?subject = kg:steve_jobs)
        }
    """,

    "count_relationships": """
        SELECT ?entity (COUNT(?rel) as ?degree) WHERE {
            ?entity ?rel ?neighbor .
        }
        GROUP BY ?entity
        ORDER BY DESC(?degree)
    """
}
```

---

## 🔧 **Utility Functions**

### **Text Similarity**

```python
def calculate_similarity(text1, text2):
    """Calculate cosine similarity between texts"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode([text1, text2])
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return similarity
```

### **Entity Resolution**

```python
def resolve_entities(entities, similarity_threshold=0.8):
    """Group similar entities together"""
    clusters = []
    used = set()

    for i, entity1 in enumerate(entities):
        if i in used:
            continue

        cluster = [entity1]
        used.add(i)

        for j, entity2 in enumerate(entities):
            if j in used or i == j:
                continue

            if calculate_similarity(entity1['text'], entity2['text']) > similarity_threshold:
                cluster.append(entity2)
                used.add(j)

        clusters.append(cluster)

    return clusters
```

### **Path Finding**

```python
def find_shortest_paths(graph, source, target, max_length=5):
    """Find multiple shortest paths"""
    try:
        paths = list(nx.all_shortest_paths(graph, source, target, weight='weight'))
        return paths[:max_length]
    except nx.NetworkXNoPath:
        return []

def find_important_paths(graph, entity1, entity2):
    """Find semantically important paths"""
    shortest_path = nx.shortest_path(graph, entity1, entity2)

    # Analyze path importance
    path_score = 0
    for i in range(len(shortest_path) - 1):
        edge_data = graph[shortest_path[i]][shortest_path[i+1]]
        confidence = edge_data.get('confidence', 1.0)
        path_score += confidence

    return {
        'path': shortest_path,
        'length': len(shortest_path),
        'score': path_score / len(shortest_path)
    }
```

---

## 📊 **Visualization Data**

### **Network Graph Data**

```python
def create_network_data(kg):
    """Create data for network visualization"""
    nodes = []
    edges = []

    for i, node in enumerate(kg.graph.nodes):
        entity_info = kg.get_entity(node)
        nodes.append({
            'id': i,
            'label': entity_info['properties'].get('name', node) if entity_info else node,
            'type': entity_info['type'] if entity_info else 'Unknown'
        })

    for edge in kg.graph.edges(data=True):
        subject, obj, data = edge
        edges.append({
            'from': list(kg.graph.nodes).index(subject),
            'to': list(kg.graph.nodes).index(obj),
            'label': data.get('relationship', 'RELATED_TO')
        })

    return {'nodes': nodes, 'edges': edges}
```

### **Chart Data**

```python
def create_centrality_chart_data(kg, top_k=10):
    """Create centrality chart data"""
    analytics = GraphAnalytics(kg)
    centrality = analytics.calculate_centrality_measures()

    chart_data = {}
    for measure, scores in centrality.items():
        top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        chart_data[measure] = [
            {
                'entity': node,
                'score': score,
                'name': kg.get_entity(node)['properties'].get('name', node) if kg.get_entity(node) else node
            }
            for node, score in top_nodes
        ]

    return chart_data
```

---

## 🚀 **Performance Optimization**

### **Efficient Graph Operations**

```python
# Pre-compute adjacency lists
def create_adjacency_cache(graph):
    cache = {
        'successors': {node: list(graph.successors(node)) for node in graph.nodes},
        'predecessors': {node: list(graph.predecessors(node)) for node in graph.nodes},
        'degree': dict(graph.degree())
    }
    return cache

# Use cached lookups
def get_neighbors_fast(node, cache):
    return cache['successors'].get(node, [])
```

### **Batch Processing**

```python
def batch_entity_linking(entities, batch_size=100):
    """Process entities in batches for efficiency"""
    results = []

    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        batch_results = batch_link_entities(batch)
        results.extend(batch_results)

    return results
```

---

## 🐛 **Common Issues & Solutions**

### **Memory Issues**

```python
# For large graphs, use sparse representations
from scipy.sparse import csr_matrix

def create_sparse_adjacency(graph):
    nodes = list(graph.nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Create adjacency matrix
    adj_matrix = np.zeros((len(nodes), len(nodes)))
    for edge in graph.edges():
        i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
        adj_matrix[i, j] = 1

    return csr_matrix(adj_matrix)
```

### **Performance Issues**

```python
# Use graph algorithms efficiently
# Bad: Frequent graph modifications
for entity in many_entities:
    kg.add_relationship(entity, "related_to", target_entity)

# Good: Batch operations
new_edges = [(entity, "related_to", target_entity) for entity in many_entities]
for subject, predicate, obj in new_edges:
    kg.add_relationship(subject, predicate, obj)
```

### **Accuracy Issues**

```python
# Use multiple similarity metrics
def multi_metric_similarity(entity1, entity2):
    metrics = {
        'string_similarity': string_similarity(entity1, entity2),
        'embedding_similarity': embedding_similarity(entity1, entity2),
        'context_similarity': context_similarity(entity1, entity2)
    }

    # Weighted combination
    weights = {'string_similarity': 0.3, 'embedding_similarity': 0.5, 'context_similarity': 0.2}
    combined_score = sum(metrics[metric] * weights[metric] for metric in metrics)

    return combined_score
```

---

## 📚 **Best Practices**

### **Design Principles**

1. **Use typed edges** - Always specify relationship types
2. **Include confidence scores** - Store confidence for each fact
3. **Normalize entity names** - Use consistent naming conventions
4. **Version knowledge** - Track when facts were added/updated
5. **Validate consistency** - Check for contradictory information

### **Performance Tips**

1. **Cache frequently accessed data** - Use dictionaries for lookups
2. **Use appropriate data structures** - NetworkX for small, Graph databases for large
3. **Batch operations** - Process multiple items together
4. **Parallel processing** - Use multiprocessing for independent tasks
5. **Memory optimization** - Use sparse matrices for large graphs

### **Quality Assurance**

1. **Validate entities** - Check entity types and properties
2. **Test relationship validity** - Ensure logical consistency
3. **Monitor bias** - Regularly check for demographic bias
4. **Update regularly** - Keep knowledge current
5. **User feedback** - Incorporate feedback for improvements

---

## 🔗 **Useful Resources**

### **Libraries & Frameworks**

- **NetworkX** - Python graph library
- **PyTorch Geometric** - Graph neural networks
- **Neo4j** - Graph database
- **RDFlib** - RDF processing
- **spaCy** - NLP processing
- **Sentence-Transformers** - Embeddings

### **Knowledge Bases**

- **DBpedia** - Structured data from Wikipedia
- **Wikidata** - Collaborative knowledge base
- **Freebase** - Knowledge graph (deprecated)
- **ConceptNet** - Common sense knowledge

### **Datasets**

- **FB15K** - FreeBase subset for link prediction
- **WN18RR** - WordNet for relation extraction
- **Countries** - Geographic knowledge graph
- **nell** - Never-Ending Language Learner

---

## ⚡ **Quick Commands Reference**

### **Graph Creation**

```python
kg = SimpleKnowledgeGraph()                    # Create KG
kg.add_entity(id, type, properties)            # Add entity
kg.add_relationship(subj, pred, obj)           # Add relationship
kg.get_entity(id)                              # Get entity
kg.get_relationships(id)                       # Get relationships
kg.find_path(start, end)                       # Find path
```

### **Analytics**

```python
analytics = GraphAnalytics(kg)                 # Create analytics
analytics.calculate_centrality_measures()      # Calculate centrality
analytics.find_communities()                   # Find communities
analytics.detect_anomalies()                   # Detect anomalies
```

### **Embeddings**

```python
model = TransE(num_entities, num_relations)    # Create model
trainer = TransETrainer(model)                 # Create trainer
trainer.train_epoch(dataloader)                # Train model
similarity_analyzer = EmbeddingSimilarity(model, mapping)  # Similarity
```

### **GCN Operations**

```python
pyg_data = KnowledgeGraphGCN(kg).to_pyg()      # Convert to PyG
model = SimpleGCN(features, classes)           # Create GCN
model(x, edge_index)                           # Forward pass
```

This cheatsheet provides quick reference for all major knowledge graph operations, from basic construction to advanced applications. Use it as a companion to the theory and practice materials for rapid development and troubleshooting.
