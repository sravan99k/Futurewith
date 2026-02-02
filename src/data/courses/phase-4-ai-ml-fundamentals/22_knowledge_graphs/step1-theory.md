# 📊 Knowledge Graphs Theory

_Comprehensive theoretical foundation for knowledge graph systems_

---

## 📖 **Table of Contents**

1. [Introduction to Knowledge Graphs](#1-introduction-to-knowledge-graphs)
2. [Core Concepts & Architecture](#2-core-concepts--architecture)
3. [Knowledge Graph Construction](#3-knowledge-graph-construction)
4. [Querying & Retrieval](#4-querying--retrieval)
5. [Knowledge Graph Embeddings](#5-knowledge-graph-embeddings)
6. [Graph Neural Networks for KGs](#6-graph-neural-networks-for-kgs)
7. [RAG Enhancement with Knowledge Graphs](#7-rag-enhancement-with-knowledge-graphs)
8. [Advanced Applications](#8-advanced-applications)
9. [Ethics & Fairness in Knowledge Graphs](#9-ethics--fairness-in-knowledge-graphs)
10. [Future Trends](#10-future-trends)

---

## **1. Introduction to Knowledge Graphs**

### **1.1 What are Knowledge Graphs?**

**Knowledge Graphs** are structured representations of real-world entities, their properties, and the relationships between them. They organize information in a graph format where:

- **Nodes (Entities)**: Represent real-world objects, concepts, or things
- **Edges (Relationships)**: Represent connections or associations between entities
- **Properties (Attributes)**: Describe characteristics of entities and relationships

**Simple Example:**

```
Entity: "Barack Obama"
Properties: {type: Person, born: 1961, occupation: President}
Relationships: [married_to] → Michelle Obama, [served_as] → President of USA
```

### **1.2 Why Knowledge Graphs Matter**

**Traditional Data Storage vs Knowledge Graphs:**

| **Aspect**        | **Traditional Database** | **Knowledge Graph**    |
| ----------------- | ------------------------ | ---------------------- |
| **Structure**     | Tabular (rows/columns)   | Graph (nodes/edges)    |
| **Relationships** | Implicit (foreign keys)  | Explicit (typed edges) |
| **Flexibility**   | Schema-fixed             | Schema-flexible        |
| **Querying**      | SQL joins                | Graph traversals       |
| **Evolution**     | Complex migrations       | Additive updates       |
| **Intuitiveness** | Computer-centric         | Human-centric          |

### **1.3 Core Characteristics**

**1. Typed Entities and Relationships**

- Every node and edge has a specific type
- Enables precise querying and reasoning
- Example: (Person)-[:LIVES_IN]->(City)

**2. Identifiers and URIs**

- Unique global identifiers for entities
- Enables linking across different knowledge sources
- Example: http://dbpedia.org/resource/Barack_Obama

**3. Machine Readable**

- Structured in formats like RDF, JSON-LD
- Enables automated processing and reasoning
- Supports SPARQL and other graph query languages

**4. Evolving and Collaborative**

- Can be continuously updated with new knowledge
- Supports crowd-sourcing and expert curation
- Enables incremental learning and discovery

---

## **2. Core Concepts & Architecture**

### **2.1 Fundamental Components**

**Entities (Nodes)**

```json
{
  "id": "http://dbpedia.org/resource/Barack_Obama",
  "type": "Person",
  "labels": {
    "en": "Barack Obama",
    "fr": "Barack Obama"
  },
  "description": "44th President of the United States",
  "properties": {
    "birthDate": "1961-08-04",
    "birthPlace": "Honolulu",
    "occupation": "President"
  }
}
```

**Relationships (Edges)**

```json
{
  "subject": "http://dbpedia.org/resource/Barack_Obama",
  "predicate": "http://dbpedia.org/ontology/spouse",
  "object": "http://dbpedia.org/resource/Michelle_Obama",
  "confidence": 0.95,
  "source": "Wikipedia"
}
```

**Properties (Attributes)**

- **Literal Properties**: String, number, date values
- **Object Properties**: References to other entities
- **Datatype Properties**: Typed values with validation

### **2.2 Graph Representation Models**

**Resource Description Framework (RDF)**

```
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Person rdfs:subClassOf ex:Agent .
ex:Barack_Obama rdf:type ex:Person ;
    ex:hasSpouse ex:Michelle_Obama ;
    ex:wasBornOn ex:BirthDate .
```

**Property Graph Model**

```json
{
  "nodes": [
    {
      "id": "person:barack_obama",
      "label": "Person",
      "properties": {
        "name": "Barack Obama",
        "birth_year": 1961,
        "occupation": "President"
      }
    }
  ],
  "edges": [
    {
      "id": "married_to_1",
      "label": "MARRIED_TO",
      "from": "person:barack_obama",
      "to": "person:michelle_obama",
      "properties": {
        "since": 1992,
        "confidence": 1.0
      }
    }
  ]
}
```

### **2.3 Knowledge Graph Architecture Layers**

**1. Data Layer**

- Raw data storage (databases, files, APIs)
- Data preprocessing and cleaning
- Entity and relationship extraction

**2. Storage Layer**

- Graph database (Neo4j, Amazon Neptune, ArangoDB)
- Distributed storage systems
- Indexing for fast queries

**3. Processing Layer**

- ETL pipelines for knowledge extraction
- Entity resolution and deduplication
- Consistency checking and validation

**4. Reasoning Layer**

- Rule-based inference
- Statistical relational learning
- Neural-symbolic reasoning

**5. Application Layer**

- Query interfaces (SPARQL, Cypher)
- API endpoints
- User applications and dashboards

---

## **3. Knowledge Graph Construction**

### **3.1 Knowledge Extraction Pipeline**

**1. Text Processing**

```python
def extract_entities_from_text(text):
    """Extract named entities from unstructured text"""
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "confidence": ent._.confidence if hasattr(ent, '_') else 0.8
        })

    return entities

# Usage
text = "Barack Obama was born in Hawaii and served as the 44th President."
entities = extract_entities_from_text(text)
print(entities)
# Output: [{'text': 'Barack Obama', 'label': 'PERSON', ...}, ...]
```

**2. Relationship Extraction**

```python
def extract_relationships(text, entities):
    """Extract relationships between entities"""
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    relationships = []
    for sent in doc.sents:
        # Find entities in sentence
        sent_entities = [e for e in entities if e['start'] >= sent.start_char and e['end'] <= sent.end_char]

        if len(sent_entities) >= 2:
            # Extract dependency patterns
            for token in sent:
                if token.dep_ in ['nsubj', 'dobj']:
                    subj = token.text
                    obj = token.head.text

                    relationships.append({
                        'subject': subj,
                        'predicate': 'related_to',  # Simplification
                        'object': obj,
                        'confidence': 0.7
                    })

    return relationships
```

**3. Knowledge Graph Construction**

```python
class KnowledgeGraphBuilder:
    def __init__(self):
        self.entities = {}
        self.relationships = []

    def add_entity(self, entity_id, entity_type, properties):
        """Add or update an entity"""
        if entity_id not in self.entities:
            self.entities[entity_id] = {
                'type': entity_type,
                'properties': properties,
                'aliases': set()
            }
        else:
            self.entities[entity_id]['properties'].update(properties)

    def add_relationship(self, subject_id, predicate, object_id, confidence=1.0):
        """Add a relationship between entities"""
        self.relationships.append({
            'subject': subject_id,
            'predicate': predicate,
            'object': object_id,
            'confidence': confidence
        })

    def build_kg(self, text_data):
        """Build knowledge graph from text data"""
        for text in text_data:
            # Extract entities
            entities = extract_entities_from_text(text)
            relationships = extract_relationships(text, entities)

            # Add to knowledge graph
            for entity in entities:
                entity_id = f"entity:{entity['text'].lower().replace(' ', '_')}"
                self.add_entity(entity_id, entity['label'], {
                    'text': entity['text'],
                    'type': entity['label'],
                    'confidence': entity['confidence']
                })

            for rel in relationships:
                # Simplified relationship mapping
                subj_id = f"entity:{rel['subject'].lower().replace(' ', '_')}"
                obj_id = f"entity:{rel['object'].lower().replace(' ', '_')}"
                self.add_relationship(subj_id, rel['predicate'], obj_id, rel['confidence'])
```

### **3.2 Entity Resolution and Deduplication**

**Problem**: Multiple mentions of the same real-world entity
**Solution**: Entity linking and deduplication techniques

```python
class EntityResolver:
    def __init__(self, similarity_threshold=0.8):
        self.threshold = similarity_threshold

    def calculate_similarity(self, entity1, entity2):
        """Calculate similarity between two entities"""
        # Text similarity
        text_sim = self._text_similarity(entity1['text'], entity2['text'])

        # Property similarity
        prop_sim = self._property_similarity(entity1['properties'], entity2['properties'])

        return 0.6 * text_sim + 0.4 * prop_sim

    def _text_similarity(self, text1, text2):
        """Calculate text similarity using various methods"""
        from difflib import SequenceMatcher

        # String similarity
        str_sim = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

        # Word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))

        return str_sim if total > 0 else 0

    def resolve_entities(self, entities):
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

                if self.calculate_similarity(entity1, entity2) > self.threshold:
                    cluster.append(entity2)
                    used.add(j)

            clusters.append(cluster)

        return clusters
```

### **3.3 Knowledge Graph Validation**

**Quality Metrics**:

- **Completeness**: Coverage of domain knowledge
- **Consistency**: No contradictory information
- **Accuracy**: Correctness of facts
- **Freshness**: Currency of information

```python
class KnowledgeGraphValidator:
    def __init__(self, kg):
        self.kg = kg

    def validate_completeness(self, required_properties):
        """Check if entities have required properties"""
        missing_properties = []

        for entity_id, entity in self.kg.entities.items():
            for prop in required_properties:
                if prop not in entity['properties']:
                    missing_properties.append({
                        'entity': entity_id,
                        'property': prop
                    })

        return missing_properties

    def validate_consistency(self):
        """Check for contradictory relationships"""
        contradictions = []

        # Example: Check for conflicting nationality claims
        nationality_rels = [r for r in self.kg.relationships if r['predicate'] == 'hasNationality']

        for rel in nationality_rels:
            # Look for conflicting nationalities
            conflicting = [r for r in nationality_rels
                          if r['subject'] == rel['subject'] and r['object'] != rel['object']]

            if conflicting:
                contradictions.append({
                    'entity': rel['subject'],
                    'nationalities': [rel['object']] + [c['object'] for c in conflicting]
                })

        return contradictions

    def validate_accuracy(self, external_sources):
        """Validate facts against external sources"""
        validation_results = []

        for relationship in self.kg.relationships:
            for source in external_sources:
                if self._verify_fact(relationship, source):
                    relationship['verified'] = True
                    validation_results.append({
                        'relationship': relationship,
                        'source': source,
                        'status': 'verified'
                    })
                    break
            else:
                validation_results.append({
                    'relationship': relationship,
                    'status': 'unverified'
                })

        return validation_results
```

---

## **4. Querying & Retrieval**

### **4.1 SPARQL Query Language**

**SPARQL** is the standard query language for RDF knowledge graphs.

**Basic Query Structure**:

```sparql
PREFIX ex: <http://example.org/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?person ?birthplace
WHERE {
    ?person rdf:type ex:Person ;
            ex:bornIn ?birthplace .
    ?birthplace ex:country "United States" .
}
```

**Complex Queries**:

```sparql
# Find all presidents and their spouses
PREFIX ex: <http://example.org/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?president ?spouse ?spouseName
WHERE {
    ?president rdf:type ex:Person ;
               ex:hasPosition ex:President ;
               ex:hasSpouse ?spouse .

    ?spouse ex:name ?spouseName .
    ?spouse rdf:type ex:Person .
}

ORDER BY ?president
```

**Aggregation Queries**:

```sparql
# Count relationships per entity
PREFIX ex: <http://example.org/>

SELECT ?entity (COUNT(?relationship) as ?degree)
WHERE {
    ?entity ?relationship ?object .
}
GROUP BY ?entity
ORDER BY DESC(?degree)
```

### **4.2 Cypher Query Language**

**Cypher** is used for property graphs (Neo4j, etc.).

**Basic Pattern Matching**:

```cypher
// Find all people and their relationships
MATCH (p:Person)
RETURN p.name, p.birthYear;

// Find relationships between entities
MATCH (p1:Person)-[r]->(p2:Person)
RETURN p1.name, type(r), p2.name;
```

**Complex Patterns**:

```cypher
// Find 2-hop paths between people
MATCH path = (p1:Person)-[*1..2]->(p2:Person)
WHERE p1.name = "Barack Obama"
RETURN path;

// Find patterns with constraints
MATCH (p:Person)-[r:LIVES_IN]->(c:City)
WHERE c.population > 1000000
RETURN p.name, c.name, c.population
ORDER BY c.population DESC;
```

**Aggregation and Analysis**:

```cypher
// Calculate degree centrality
MATCH (p:Person)-[r]-()
RETURN p.name, COUNT(r) as degree
ORDER BY degree DESC;

// Find most connected cities
MATCH (p:Person)-[r:LIVES_IN]->(c:City)
RETURN c.name, COUNT(p) as resident_count
ORDER BY resident_count DESC
LIMIT 10;
```

### **4.3 Graph Query Processing**

```python
class KnowledgeGraphQueryEngine:
    def __init__(self, kg_backend):
        self.kg = kg_backend

    def sparql_query(self, sparql_query):
        """Execute SPARQL query on RDF knowledge graph"""
        try:
            from rdflib import Graph
            g = Graph()
            g.parse(data=self.kg.serialize_rdf())
            results = g.query(sparql_query)
            return self._format_sparql_results(results)
        except Exception as e:
            return {"error": str(e)}

    def cypher_query(self, cypher_query):
        """Execute Cypher query on property graph"""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(self.kg.connection_uri,
                                        auth=(self.kg.username, self.kg.password))

            with driver.session() as session:
                result = session.run(cypher_query)
                return [record.data() for record in result]
        except Exception as e:
            return {"error": str(e)}

    def natural_language_query(self, nl_query):
        """Convert natural language to graph query"""
        # Simplified NLP-based query parsing
        query_parser = NLPQueryParser()
        parsed_query = query_parser.parse(nl_query)

        if parsed_query['type'] == 'sparql':
            return self.sparql_query(parsed_query['sparql'])
        elif parsed_query['type'] == 'cypher':
            return self.cypher_query(parsed_query['cypher'])
        else:
            return {"error": "Unsupported query type"}

    def semantic_search(self, query_text, top_k=10):
        """Perform semantic search on knowledge graph"""
        # Embed query and entities
        query_embedding = self.embed_query(query_text)

        # Calculate similarities
        similarities = []
        for entity_id, entity in self.kg.entities.items():
            entity_embedding = self.get_entity_embedding(entity_id)
            similarity = cosine_similarity(query_embedding, entity_embedding)
            similarities.append((entity_id, similarity))

        # Return top-k most similar entities
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
```

---

## **5. Knowledge Graph Embeddings**

### **5.1 Graph Embedding Techniques**

**1. TransE (Translation-based Embeddings)**

```python
class TransE:
    def __init__(self, num_entities, num_relationships, embedding_dim=100):
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relationships, embedding_dim)

    def forward(self, head, relation, tail):
        """Score function for TransE"""
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        # Translation: h + r ≈ t
        return -torch.norm(h + r - t, p=1, dim=1)

    def loss(self, positive_scores, negative_scores):
        """Margin-based ranking loss"""
        margin = 1.0
        return torch.relu(positive_scores - negative_scores + margin)
```

**2. DistMult (Bilinear Scoring)**

```python
class DistMult:
    def __init__(self, num_entities, num_relationships, embedding_dim=100):
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relationships, embedding_dim)

    def forward(self, head, relation, tail):
        """Score function for DistMult"""
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        # Bilinear form: h ⊙ r ⊙ t (element-wise multiplication)
        return torch.sum(h * r * t, dim=1)
```

**3. ComplEx (Complex-valued Embeddings)**

```python
class ComplEx:
    def __init__(self, num_entities, num_relationships, embedding_dim=100):
        self.real_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.imag_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_real = nn.Embedding(num_relationships, embedding_dim)
        self.relation_imag = nn.Embedding(num_relationships, embedding_dim)

    def forward(self, head, relation, tail):
        """Score function for ComplEx"""
        h_real = self.real_embeddings(head)
        h_imag = self.imag_embeddings(head)
        r_real = self.relation_real(relation)
        r_imag = self.relation_imag(relation)
        t_real = self.real_embeddings(tail)
        t_imag = self.imag_embeddings(tail)

        # Complex multiplication: (h_real + ih_imag) ⊙ (r_real + ir_imag)
        real_part = h_real * r_real - h_imag * r_imag
        imag_part = h_real * r_imag + h_imag * r_real

        return torch.sum(real_part * t_real + imag_part * t_imag, dim=1)
```

### **5.2 Training Knowledge Graph Embeddings**

```python
class KnowledgeGraphEmbeddingTrainer:
    def __init__(self, kg, embedding_model, negative_sampling_size=10):
        self.kg = kg
        self.model = embedding_model
        self.negative_sampling_size = negative_sampling_size

    def generate_negative_samples(self, positive_triples):
        """Generate negative samples by random corruption"""
        negative_triples = []

        for head, relation, tail in positive_triples:
            # Randomly corrupt head or tail (but not both)
            if random.random() < 0.5:
                # Corrupt head
                new_head = random.choice(self.kg.entities.keys())
                negative_triples.append((new_head, relation, tail))
            else:
                # Corrupt tail
                new_tail = random.choice(self.kg.entities.keys())
                negative_triples.append((head, relation, new_tail))

        return negative_triples

    def train_epoch(self, optimizer, batch_size=100):
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0

        # Generate all positive triples
        positive_triples = [(r['subject'], r['predicate'], r['object'])
                           for r in self.kg.relationships]

        # Shuffle and batch
        random.shuffle(positive_triples)

        for i in range(0, len(positive_triples), batch_size):
            batch = positive_triples[i:i + batch_size]

            # Generate negative samples
            negative_triples = self.generate_negative_samples(batch)

            # Create training examples
            pos_scores = self.model(*zip(*batch))
            neg_scores = self.model(*zip(*negative_triples))

            # Calculate loss
            loss = self.model.loss(pos_scores, neg_scores)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def train(self, num_epochs=100, learning_rate=0.01):
        """Train the embedding model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(optimizer)
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

            # Evaluate on validation set (implement validation logic)
            if epoch % 10 == 0:
                self.evaluate()

    def evaluate(self):
        """Evaluate the trained embeddings"""
        # Implement evaluation metrics (MRR, Hits@K, etc.)
        pass
```

### **5.3 Using Embeddings for Tasks**

```python
class KnowledgeGraphEmbeddingApplications:
    def __init__(self, trained_model, entity_mapping, relation_mapping):
        self.model = trained_model
        self.entity_mapping = entity_mapping
        self.relation_mapping = relation_mapping

    def link_prediction(self, head, relation, top_k=10):
        """Predict missing tails for (head, relation, ?)"""
        head_idx = self.entity_mapping[head]
        relation_idx = self.relation_mapping[relation]

        # Calculate scores for all possible tails
        scores = []
        for entity, entity_idx in self.entity_mapping.items():
            tail_idx = entity_idx
            score = self.model(head_idx, relation_idx, tail_idx).item()
            scores.append((entity, score))

        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def entity_similarity(self, entity1, entity2):
        """Calculate similarity between two entities using embeddings"""
        emb1 = self.model.entity_embeddings[self.entity_mapping[entity1]]
        emb2 = self.model.entity_embeddings[self.entity_mapping[entity2]]

        # Cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
        return cosine_sim.item()

    def find_similar_entities(self, target_entity, top_k=10):
        """Find entities similar to target entity"""
        target_embedding = self.model.entity_embeddings[
            self.entity_mapping[target_entity]
        ]

        similarities = []
        for entity, entity_idx in self.entity_mapping.items():
            if entity != target_entity:
                entity_embedding = self.model.entity_embeddings[entity_idx]
                similarity = torch.nn.functional.cosine_similarity(
                    target_embedding, entity_embedding, dim=0
                ).item()
                similarities.append((entity, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
```

---

## **6. Graph Neural Networks for Knowledge Graphs**

### **6.1 Graph Convolutional Networks (GCNs) for KGs**

**GCN Applied to Knowledge Graphs:**

```python
class KnowledgeGraphGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations):
        super().__init__()
        self.num_relations = num_relations

        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, input_dim)

        # Relation-specific transformations
        self.relation_weights = nn.Parameter(torch.randn(num_relations, input_dim, hidden_dim))
        self.relation_bias = nn.Parameter(torch.zeros(num_relations, hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, entity_features, adjacency_matrices):
        """
        Args:
            entity_features: [batch_size, num_entities, input_dim]
            adjacency_matrices: List of [num_entities, num_entities] matrices for each relation
        """
        h = entity_features

        # Aggregate information from neighbors
        for relation_idx in range(self.num_relations):
            # Get relation-specific weight
            W = self.relation_weights[relation_idx]  # [input_dim, hidden_dim]
            b = self.relation_bias[relation_idx]     # [hidden_dim]

            # Apply relation-specific transformation
            transformed = torch.matmul(h, W) + b  # [batch_size, num_entities, hidden_dim]

            # Aggregate neighbor information
            neighbor_sum = torch.matmul(adjacency_matrices[relation_idx], transformed)

            # Add to current representations
            h = h + neighbor_sum

        # Apply activation and dropout
        h = torch.relu(h)
        h = self.dropout(h)

        # Output layer
        output = self.output_layer(h)
        return output
```

### **6.2 Relational Graph Convolutional Networks (R-GCNs)**

```python
class RelationalGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_relations, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations

        # Weight matrices for each relation type
        self.weight_matrices = nn.Parameter(torch.randn(num_relations, in_features, out_features))

        # Bias terms
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adjacency_matrices):
        """
        Args:
            input: [batch_size, num_nodes, in_features]
            adjacency_matrices: List of [num_nodes, num_nodes] sparse matrices
        """
        batch_size, num_nodes, _ = input.shape
        output = torch.zeros(batch_size, num_nodes, self.out_features)

        for r in range(self.num_relations):
            # Apply relation-specific weight matrix
            transformed = torch.matmul(input, self.weight_matrices[r])

            # Aggregate neighbor information
            neighbor_agg = torch.matmul(adjacency_matrices[r], transformed)

            # Add to output
            output += neighbor_agg

        # Add bias
        if self.bias is not None:
            output += self.bias

        return output

class RGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim=64, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_nodes, hidden_dim)

        # R-GCN layers
        self.rgcn_layers = nn.ModuleList([
            RelationalGraphConvolution(hidden_dim, hidden_dim, num_relations)
            for _ in range(num_layers)
        ])

        # Final classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, entity_ids, adjacency_matrices):
        # Get initial embeddings
        h = self.entity_embeddings(entity_ids).unsqueeze(0)  # [1, num_nodes, hidden_dim]

        # Apply R-GCN layers
        for layer in self.rgcn_layers:
            h = layer(h, adjacency_matrices)
            h = torch.relu(h)

        # Classification
        logits = self.classifier(h)
        return logits
```

### **6.3 Graph Attention Networks (GATs) for KGs**

```python
class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features

        # Attention parameters
        self.W = nn.Linear(in_features, num_heads * out_features)
        self.attn = nn.Linear(out_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.elu()

    def forward(self, input, adjacency_matrix):
        batch_size, num_nodes, in_features = input.shape

        # Linear transformation
        h = self.W(input)  # [batch_size, num_nodes, num_heads * out_features]
        h = h.view(batch_size, num_nodes, self.num_heads, self.out_features)
        h = h.transpose(1, 2)  # [batch_size, num_heads, num_nodes, out_features]

        # Calculate attention scores
        h_reshaped = h.reshape(batch_size * self.num_heads, num_nodes, self.out_features)

        # Self-attention mechanism
        attention_scores = torch.matmul(h_reshaped, h_reshaped.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.out_features)

        # Apply mask (if provided)
        if adjacency_matrix is not None:
            mask = adjacency_matrix.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        output = torch.matmul(attention_weights, h_reshaped)
        output = output.view(batch_size, self.num_heads, num_nodes, self.out_features)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, num_nodes, self.num_heads * self.out_features)

        return output

class KnowledgeGraphGAT(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim=64, num_heads=8):
        super().__init__()
        self.num_relations = num_relations

        # Multi-head attention for each relation
        self.attention_layers = nn.ModuleList([
            MultiHeadGraphAttention(hidden_dim, hidden_dim // num_heads, num_heads)
            for _ in range(num_relations)
        ])

        # Final prediction layer
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, entity_embeddings, adjacency_matrices):
        h = entity_embeddings

        # Apply attention for each relation type
        relation_outputs = []
        for relation_idx in range(self.num_relations):
            attention_out = self.attention_layers[relation_idx](h, adjacency_matrices[relation_idx])
            relation_outputs.append(attention_out)

        # Combine relation-specific representations
        combined = torch.stack(relation_outputs, dim=0)
        h_combined = torch.mean(combined, dim=0)  # Average pooling over relations

        # Final prediction
        scores = self.predictor(h_combined)
        return scores
```

---

## **7. RAG Enhancement with Knowledge Graphs**

### **7.1 Graph-Enhanced RAG Architecture**

**Traditional RAG vs Graph-Enhanced RAG:**

| **Component**              | **Traditional RAG** | **Graph-Enhanced RAG**          |
| -------------------------- | ------------------- | ------------------------------- |
| **Knowledge Source**       | Flat documents      | Structured knowledge graphs     |
| **Retrieval Strategy**     | Similarity search   | Graph traversal + similarity    |
| **Context Structure**      | Raw text chunks     | Entity-relationship paths       |
| **Reasoning Capabilities** | Limited             | Path-based reasoning            |
| **Explanation Quality**    | Basic               | Relationship-based explanations |

### **7.2 Knowledge Graph RAG Implementation**

```python
class KnowledgeGraphRAG:
    def __init__(self, knowledge_graph, llm, embeddings_model):
        self.kg = knowledge_graph
        self.llm = llm
        self.embeddings = embeddings_model
        self.entity_linker = EntityLinker()
        self.query_expander = GraphQueryExpander()

    def retrieve_with_graph_enhancement(self, question, k=5):
        """Enhanced retrieval using knowledge graph"""

        # Step 1: Extract entities from question
        entities = self.entity_linker.extract_entities(question)

        # Step 2: Expand query using knowledge graph
        expanded_concepts = self.query_expander.expand_with_graph(entities, self.kg)

        # Step 3: Traditional document retrieval
        doc_context = self.traditional_retrieval(question, k=k)

        # Step 4: Graph-based retrieval
        graph_context = self.graph_retrieval(entities, expanded_concepts)

        # Step 5: Combine contexts
        combined_context = self.combine_contexts(doc_context, graph_context)

        return combined_context

    def graph_retrieval(self, entities, expanded_concepts):
        """Retrieve relevant information from knowledge graph"""
        graph_paths = []

        for entity in entities:
            # Find related entities and relationships
            related_entities = self.kg.get_neighbors(entity, max_depth=2)
            graph_paths.extend(related_entities)

        # Convert graph paths to text context
        context_parts = []
        for path in graph_paths:
            path_text = self.path_to_text(path)
            context_parts.append(path_text)

        return "\n".join(context_parts)

    def path_to_text(self, graph_path):
        """Convert graph path to readable text"""
        if len(graph_path) == 2:  # Direct relationship
            return f"{graph_path[0]} is related to {graph_path[1]}"
        elif len(graph_path) == 3:  # Two-hop path
            return f"{graph_path[0]} is connected to {graph_path[1]} through {graph_path[2]}"
        else:
            return " → ".join(str(node) for node in graph_path)

    def generate_with_graph_context(self, question):
        """Generate response using graph-enhanced context"""

        # Retrieve enhanced context
        context = self.retrieve_with_graph_enhancement(question)

        # Construct prompt with graph context
        prompt = f"""
        Based on the following context, answer the question:

        Context: {context}

        Question: {question}

        Please provide a comprehensive answer using the provided context.
        """

        # Generate response
        response = self.llm.generate(prompt)

        return response
```

### **7.3 Entity Linking and Disambiguation**

```python
class EntityLinker:
    def __init__(self, nlp_model=None):
        self.nlp = nlp_model or spacy.load("en_core_web_sm")
        self.entity_candidates = {}

    def extract_entities(self, text):
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'context': text[max(0, ent.start_char-50):ent.end_char+50]
            })

        return entities

    def disambiguate_entities(self, entities, knowledge_graph):
        """Disambiguate entities using knowledge graph"""
        linked_entities = []

        for entity in entities:
            # Find candidate entities in knowledge graph
            candidates = self.find_candidate_entities(entity['text'], knowledge_graph)

            # Score candidates based on context similarity
            best_candidate = self.score_candidates(entity, candidates, knowledge_graph)

            if best_candidate:
                linked_entities.append({
                    'original': entity,
                    'kg_entity': best_candidate,
                    'confidence': best_candidate['confidence']
                })

        return linked_entities

    def find_candidate_entities(self, entity_text, knowledge_graph):
        """Find candidate entities in knowledge graph"""
        candidates = []

        # Search by exact match
        if entity_text in knowledge_graph.entities:
            candidates.append(knowledge_graph.entities[entity_text])

        # Search by fuzzy match
        for entity_id, entity in knowledge_graph.entities.items():
            if self.fuzzy_match(entity_text, entity['properties'].get('name', '')):
                candidates.append(entity)

        return candidates

    def fuzzy_match(self, text1, text2):
        """Simple fuzzy matching using string similarity"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() > 0.8

    def score_candidates(self, entity, candidates, knowledge_graph):
        """Score candidate entities based on context"""
        if not candidates:
            return None

        best_candidate = None
        best_score = 0

        for candidate in candidates:
            score = self.calculate_context_similarity(entity, candidate, knowledge_graph)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_score > 0.5:  # Threshold
            best_candidate['confidence'] = best_score
            return best_candidate

        return None
```

### **7.4 Query Expansion with Knowledge Graphs**

```python
class GraphQueryExpander:
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def expand_with_graph(self, entities, max_expansion=5):
        """Expand query using knowledge graph"""
        expanded_concepts = set()

        for entity in entities:
            # Add the original entity
            expanded_concepts.add(entity['text'])

            # Add related concepts
            related = self.find_related_concepts(entity['kg_entity'], max_expansion)
            expanded_concepts.update(related)

        return list(expanded_concepts)

    def find_related_concepts(self, entity, max_count):
        """Find concepts related to the given entity"""
        related_concepts = []

        # Get direct neighbors
        neighbors = self.kg.get_neighbors(entity['id'])

        for neighbor in neighbors:
            if len(related_concepts) >= max_count:
                break

            # Add neighbor entity
            related_concepts.append(neighbor['properties'].get('name', ''))

            # Add relationship type
            relationship_type = neighbor['relationship_type']
            related_concepts.append(relationship_type)

        return related_concepts[:max_count]

    def find_concept_paths(self, entities, max_path_length=3):
        """Find paths between entities in the query"""
        paths = []

        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i >= j:
                    continue

                # Find shortest path between entities
                path = self.kg.find_shortest_path(
                    entity1['kg_entity']['id'],
                    entity2['kg_entity']['id'],
                    max_length=max_path_length
                )

                if path:
                    paths.append(path)

        return paths
```

---

## **8. Advanced Applications**

### **8.1 Question Answering with Knowledge Graphs**

```python
class KnowledgeGraphQA:
    def __init__(self, knowledge_graph, llm):
        self.kg = knowledge_graph
        self.llm = llm
        self.sparql_generator = SPARQLGenerator()

    def answer_question(self, question):
        """Answer question using knowledge graph"""

        # Parse question to identify intent and entities
        parsed_question = self.parse_question(question)

        if parsed_question['type'] == 'factoid':
            return self.answer_factoid_question(parsed_question)
        elif parsed_question['type'] == 'complex':
            return self.answer_complex_question(parsed_question)
        else:
            return "I couldn't understand the question type."

    def answer_factoid_question(self, parsed):
        """Answer simple factoid questions"""
        # Generate SPARQL query
        sparql_query = self.sparql_generator.generate_query(parsed)

        # Execute query on knowledge graph
        results = self.kg.execute_sparql(sparql_query)

        if results:
            return self.format_factoid_answer(results, parsed)
        else:
            return "I couldn't find the answer in my knowledge base."

    def answer_complex_question(self, parsed):
        """Answer complex reasoning questions"""
        # Multi-hop reasoning
        intermediate_results = self.multi_hop_reasoning(parsed)

        # Combine with LLM reasoning
        combined_context = self.create_reasoning_context(intermediate_results)

        # Generate answer using LLM
        answer_prompt = f"""
        Question: {parsed['question']}
        Context: {combined_context}

        Please provide a comprehensive answer based on the context.
        """

        answer = self.llm.generate(answer_prompt)
        return answer

    def multi_hop_reasoning(self, parsed_question):
        """Perform multi-hop reasoning on knowledge graph"""
        hops = []

        # First hop: direct relationships
        direct_facts = self.kg.find_direct_facts(parsed_question['entities'])
        hops.append(direct_facts)

        # Subsequent hops: indirect relationships
        current_entities = [fact['object'] for fact in direct_facts]

        for hop in range(1, parsed_question['max_hops']):
            next_facts = []
            for entity in current_entities:
                facts = self.kg.find_direct_facts([entity])
                next_facts.extend(facts)

            hops.append(next_facts)
            current_entities = [fact['object'] for fact in next_facts]

        return hops
```

### **8.2 Recommendation Systems with Knowledge Graphs**

```python
class KnowledgeGraphRecommender:
    def __init__(self, knowledge_graph, embeddings_model):
        self.kg = knowledge_graph
        self.embeddings = embeddings_model
        self.user_profiler = UserProfiler()

    def recommend_items(self, user_id, item_type, top_k=10):
        """Recommend items using knowledge graph"""

        # Get user profile from knowledge graph
        user_profile = self.user_profiler.get_user_profile(user_id, self.kg)

        # Extract user preferences
        preferences = self.extract_user_preferences(user_profile)

        # Find items matching preferences
        candidate_items = self.find_candidate_items(item_type, preferences)

        # Score and rank items
        ranked_items = self.rank_items(user_id, candidate_items, preferences)

        return ranked_items[:top_k]

    def extract_user_preferences(self, user_profile):
        """Extract user preferences from profile"""
        preferences = {
            'categories': set(),
            'entities': set(),
            'relationships': set()
        }

        for fact in user_profile:
            if fact['predicate'] == 'likes':
                preferences['entities'].add(fact['object'])
            elif fact['predicate'] == 'interested_in':
                preferences['categories'].add(fact['object'])
            elif fact['predicate'] == 'related_to':
                preferences['relationships'].add(fact['object'])

        return preferences

    def find_candidate_items(self, item_type, preferences):
        """Find candidate items based on user preferences"""
        candidates = []

        # Direct entity matching
        for entity in preferences['entities']:
            related_items = self.kg.find_related_entities(entity, item_type)
            candidates.extend(related_items)

        # Category matching
        for category in preferences['categories']:
            items_in_category = self.kg.get_entities_by_type(category)
            candidates.extend(items_in_category)

        # Relationship-based matching
        for relationship in preferences['relationships']:
            related_items = self.kg.find_by_relationship(relationship, item_type)
            candidates.extend(related_items)

        return list(set(candidates))  # Remove duplicates

    def rank_items(self, user_id, candidate_items, preferences):
        """Rank items based on user preferences and knowledge graph structure"""
        scored_items = []

        for item in candidate_items:
            score = self.calculate_relevance_score(user_id, item, preferences)
            scored_items.append((item, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items

    def calculate_relevance_score(self, user_id, item, preferences):
        """Calculate relevance score for an item"""
        score = 0

        # Entity similarity
        item_embedding = self.embeddings.get_entity_embedding(item['id'])
        for preference_entity in preferences['entities']:
            pref_embedding = self.embeddings.get_entity_embedding(preference_entity)
            similarity = torch.nn.functional.cosine_similarity(
                item_embedding, pref_embedding, dim=0
            ).item()
            score += similarity

        # Path-based features
        path_score = self.calculate_path_based_score(user_id, item, preferences)
        score += path_score

        # Popularity and quality features
        quality_score = self.get_item_quality_score(item)
        score += quality_score * 0.2

        return score
```

### **8.3 Knowledge Graph Completion**

```python
class KnowledgeGraphCompleter:
    def __init__(self, knowledge_graph, embedding_model):
        self.kg = knowledge_graph
        self.embeddings = embedding_model
        self.completion_model = CompletionModel(embedding_model)

    def predict_missing_links(self, entity, relation, top_k=10):
        """Predict missing links for a given entity-relation pair"""
        # Get embedding for entity and relation
        entity_embedding = self.embeddings.get_entity_embedding(entity)
        relation_embedding = self.embeddings.get_relation_embedding(relation)

        # Score all possible tail entities
        scores = []
        for candidate_entity in self.kg.entities:
            if candidate_entity != entity:
                candidate_embedding = self.embeddings.get_entity_embedding(candidate_entity)
                score = self.completion_model.score_triple(
                    entity_embedding, relation_embedding, candidate_embedding
                )
                scores.append((candidate_entity, score))

        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def complete_subgraph(self, seed_entities, max_expansion=100):
        """Complete subgraph around seed entities"""
        completed_edges = []
        processed_pairs = set()

        for seed_entity in seed_entities:
            # Get existing connections
            existing_neighbors = self.kg.get_neighbors(seed_entity)

            # Predict missing relationships
            for relation in self.kg.relation_types:
                # Check if edge already exists
                existing_objects = [n['object'] for n in existing_neighbors
                                  if n['relation'] == relation]

                # Predict new connections
                predictions = self.predict_missing_links(seed_entity, relation, top_k=20)

                for predicted_entity, confidence in predictions:
                    # Check if this prediction is already known
                    pair_key = (seed_entity, relation, predicted_entity)
                    if pair_key not in processed_pairs:
                        completed_edges.append({
                            'subject': seed_entity,
                            'relation': relation,
                            'object': predicted_entity,
                            'confidence': confidence,
                            'type': 'predicted'
                        })
                        processed_pairs.add(pair_key)

        # Filter by confidence threshold
        high_confidence_edges = [
            edge for edge in completed_edges
            if edge['confidence'] > 0.7
        ]

        return high_confidence_edges[:max_expansion]

    def validate_predictions(self, predictions, validation_sources):
        """Validate predictions using external sources"""
        validated_predictions = []

        for prediction in predictions:
            is_valid = False
            for source in validation_sources:
                if self.verify_fact_with_source(prediction, source):
                    is_valid = True
                    break

            prediction['validated'] = is_valid
            if is_valid:
                validated_predictions.append(prediction)

        return validated_predictions
```

---

## **9. Ethics & Fairness in Knowledge Graphs**

### **9.1 Bias Detection in Knowledge Graphs**

```python
class KnowledgeGraphBiasDetector:
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.demographic_categories = [
            'gender', 'race', 'nationality', 'religion',
            'age_group', 'disability_status', 'socioeconomic_status'
        ]

    def detect_demographic_bias(self, target_concept):
        """Detect demographic bias in knowledge graph"""
        bias_analysis = {
            'demographic_distribution': {},
            'representation_ratios': {},
            'bias_metrics': {},
            'recommendations': []
        }

        # Extract demographic information
        demographic_entities = self.extract_demographic_entities(target_concept)

        # Calculate distribution
        for demo_category in self.demographic_categories:
            entities = demographic_entities.get(demo_category, [])
            bias_analysis['demographic_distribution'][demo_category] = len(entities)

        # Calculate representation ratios
        total_entities = sum(bias_analysis['demographic_distribution'].values())
        for demo_category, count in bias_analysis['demographic_distribution'].items():
            ratio = count / total_entities if total_entities > 0 else 0
            bias_analysis['representation_ratios'][demo_category] = ratio

        # Detect bias patterns
        bias_analysis['bias_metrics'] = self.calculate_bias_metrics(demographic_entities)
        bias_analysis['recommendations'] = self.generate_bias_recommendations(bias_analysis)

        return bias_analysis

    def extract_demographic_entities(self, target_concept):
        """Extract entities with demographic attributes"""
        demographic_entities = {}

        # Find entities of target concept
        target_entities = self.kg.find_entities_by_type(target_concept)

        for entity in target_entities:
            # Extract demographic attributes
            demographics = self.extract_demographics(entity)

            for demo_category, value in demographics.items():
                if demo_category not in demographic_entities:
                    demographic_entities[demo_category] = []
                demographic_entities[demo_category].append({
                    'entity': entity,
                    'value': value
                })

        return demographic_entities

    def calculate_bias_metrics(self, demographic_entities):
        """Calculate various bias metrics"""
        metrics = {}

        # Shannon diversity index
        for demo_category, entities in demographic_entities.items():
            distribution = [e['value'] for e in entities]
            diversity = self.calculate_shannon_diversity(distribution)
            metrics[f'{demo_category}_diversity'] = diversity

        # Representation ratio disparities
        representation_ratios = {}
        total_entities = sum(len(entities) for entities in demographic_entities.values())

        for demo_category, entities in demographic_entities.items():
            ratio = len(entities) / total_entities if total_entities > 0 else 0
            representation_ratios[demo_category] = ratio

        # Calculate disparity (difference from equal representation)
        equal_representation = 1.0 / len(representation_ratios) if representation_ratios else 0
        disparities = {cat: abs(ratio - equal_representation)
                      for cat, ratio in representation_ratios.items()}

        metrics['representation_disparities'] = disparities

        return metrics

    def generate_bias_recommendations(self, bias_analysis):
        """Generate recommendations to address bias"""
        recommendations = []

        # Check for underrepresented groups
        representation_ratios = bias_analysis['representation_ratios']
        total_categories = len(representation_ratios)
        equal_representation = 1.0 / total_categories if total_categories > 0 else 0

        for category, ratio in representation_ratios.items():
            if ratio < equal_representation * 0.5:  # Less than half of equal representation
                recommendations.append({
                    'type': 'underrepresentation',
                    'category': category,
                    'severity': 'high' if ratio < equal_representation * 0.1 else 'medium',
                    'suggestion': f'Increase representation of {category} entities by {(equal_representation - ratio) / ratio * 100:.1f}%'
                })

        # Check for diversity issues
        for metric_name, metric_value in bias_analysis['bias_metrics'].items():
            if 'diversity' in metric_name and metric_value < 0.7:  # Low diversity threshold
                category = metric_name.replace('_diversity', '')
                recommendations.append({
                    'type': 'low_diversity',
                    'category': category,
                    'severity': 'medium',
                    'suggestion': f'Improve diversity in {category} representation'
                })

        return recommendations
```

### **9.2 Fairness-Aware Knowledge Graph Completion**

```python
class FairnessAwareKGCompletion:
    def __init__(self, knowledge_graph, embedding_model):
        self.kg = knowledge_graph
        self.embeddings = embedding_model
        self.fairness_constraints = self.load_fairness_constraints()

    def fair_link_prediction(self, subject, relation, protected_attributes):
        """Predict links with fairness constraints"""

        # Get base predictions
        base_predictions = self.get_base_predictions(subject, relation)

        # Apply fairness constraints
        fair_predictions = self.apply_fairness_constraints(
            base_predictions, protected_attributes
        )

        return fair_predictions

    def apply_fairness_constraints(self, predictions, protected_attributes):
        """Apply fairness constraints to predictions"""

        # Demographic parity constraint
        fair_predictions = self.demographic_parity_constraint(predictions, protected_attributes)

        # Equalized odds constraint (if applicable)
        fair_predictions = self.equalized_odds_constraint(fair_predictions, protected_attributes)

        return fair_predictions

    def demographic_parity_constraint(self, predictions, protected_attributes):
        """Ensure demographic parity in predictions"""

        # Group predictions by protected attribute
        grouped_predictions = {}
        for pred in predictions:
            protected_value = self.extract_protected_value(pred['object'], protected_attributes)

            if protected_value not in grouped_predictions:
                grouped_predictions[protected_value] = []
            grouped_predictions[protected_value].append(pred)

        # Calculate target probability for each group
        total_predictions = len(predictions)
        num_groups = len(grouped_predictions)
        target_probability = 1.0 / num_groups if num_groups > 0 else 0

        # Adjust predictions to achieve demographic parity
        fair_predictions = []
        for group, group_preds in grouped_predictions.items():
            # Calculate current probability for this group
            current_probability = len(group_preds) / total_predictions

            # Adjust if significantly different from target
            if abs(current_probability - target_probability) > 0.1:
                # Reweight predictions
                adjusted_preds = self.reweight_predictions(
                    group_preds, target_probability, current_probability
                )
                fair_predictions.extend(adjusted_preds)
            else:
                fair_predictions.extend(group_preds)

        return fair_predictions

    def reweight_predictions(self, predictions, target_prob, current_prob):
        """Reweight predictions to achieve fairness"""

        if current_prob == 0:
            return predictions

        # Calculate adjustment factor
        adjustment_factor = target_prob / current_prob

        # Adjust number of predictions
        num_to_keep = int(len(predictions) * adjustment_factor)

        # Keep top predictions based on confidence
        sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        return sorted_predictions[:num_to_keep]

    def fairness_audit(self, target_concept, protected_attributes):
        """Perform comprehensive fairness audit"""

        audit_results = {
            'representation_analysis': {},
            'prediction_bias': {},
            'fairness_metrics': {},
            'recommendations': []
        }

        # Analyze representation fairness
        audit_results['representation_analysis'] = self.analyze_representation_fairness(
            target_concept, protected_attributes
        )

        # Analyze prediction bias
        audit_results['prediction_bias'] = self.analyze_prediction_bias(
            target_concept, protected_attributes
        )

        # Calculate fairness metrics
        audit_results['fairness_metrics'] = self.calculate_fairness_metrics(
            audit_results['representation_analysis'],
            audit_results['prediction_bias']
        )

        # Generate recommendations
        audit_results['recommendations'] = self.generate_fairness_recommendations(
            audit_results['fairness_metrics']
        )

        return audit_results
```

### **9.3 Privacy-Preserving Knowledge Graphs**

```python
class PrivacyPreservingKG:
    def __init__(self, knowledge_graph, encryption_key):
        self.kg = knowledge_graph
        self.encryption_key = encryption_key
        self.privacy_levels = ['public', 'internal', 'confidential', 'restricted']

    def encrypt_sensitive_entities(self, entity_type, privacy_level):
        """Encrypt entities based on privacy level"""

        sensitive_entities = self.kg.find_entities_by_type(entity_type)
        encrypted_entities = []

        for entity in sensitive_entities:
            # Encrypt sensitive properties
            encrypted_properties = {}
            for prop_name, prop_value in entity['properties'].items():
                if self.is_sensitive_property(prop_name, privacy_level):
                    encrypted_properties[prop_name] = self.encrypt_value(prop_value)
                else:
                    encrypted_properties[prop_name] = prop_value

            # Create encrypted entity
            encrypted_entity = {
                'id': entity['id'],
                'type': entity['type'],
                'properties': encrypted_properties,
                'privacy_level': privacy_level
            }

            encrypted_entities.append(encrypted_entity)

        return encrypted_entities

    def differential_privacy_query(self, sparql_query, epsilon=1.0):
        """Execute SPARQL query with differential privacy"""

        # Execute original query
        original_results = self.kg.execute_sparql(sparql_query)

        # Add calibrated noise for differential privacy
        noisy_results = self.add_dp_noise(original_results, epsilon)

        return noisy_results

    def add_dp_noise(self, results, epsilon):
        """Add differential privacy noise to results"""

        if not results:
            return results

        # Calculate sensitivity
        sensitivity = self.calculate_query_sensitivity(results)

        # Generate Laplace noise
        noise_scale = sensitivity / epsilon
        noisy_results = []

        for result in results:
            noisy_result = {}
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    noise = np.random.laplace(0, noise_scale)
                    noisy_result[key] = value + noise
                else:
                    # For categorical values, add noise through randomization
                    if np.random.random() < noise_scale / (noise_scale + 1):
                        # Flip category with some probability
                        noisy_result[key] = self.randomize_categorical_value(value)
                    else:
                        noisy_result[key] = value

            noisy_results.append(noisy_result)

        return noisy_results

    def federated_kg_learning(self, participating_entities):
        """Enable federated learning on knowledge graphs"""

        # Local training on each entity
        local_models = {}
        for entity in participating_entities:
            local_data = self.kg.get_local_subgraph(entity)
            local_model = self.train_local_model(local_data)
            local_models[entity] = local_model

        # Aggregate models (Federated Averaging)
        global_model = self.federated_averaging(local_models)

        return global_model

    def federated_averaging(self, local_models):
        """Perform federated averaging of local models"""

        # Weighted averaging based on data size
        total_samples = sum(model['data_size'] for model in local_models.values())

        global_parameters = {}
        for param_name in local_models[list(local_models.keys())[0]]['parameters']:
            weighted_sum = 0
            for entity, model in local_models.items():
                weight = model['data_size'] / total_samples
                weighted_sum += weight * model['parameters'][param_name]
            global_parameters[param_name] = weighted_sum

        return {
            'parameters': global_parameters,
            'aggregation_method': 'federated_averaging',
            'participating_entities': list(local_models.keys())
        }
```

---

## **10. Future Trends**

### **10.1 Multimodal Knowledge Graphs**

**Integration of Vision, Audio, and Text:**

```python
class MultimodalKnowledgeGraph:
    def __init__(self):
        self.text_kg = KnowledgeGraph()
        self.image_kg = KnowledgeGraph()
        self.audio_kg = KnowledgeGraph()
        self.cross_modal_alignments = []

    def add_multimodal_entity(self, entity_data):
        """Add entity with multiple modalities"""

        entity_id = entity_data['id']

        # Add to text KG
        if 'text' in entity_data:
            self.text_kg.add_entity(entity_id, 'MultimodalEntity', entity_data['text'])

        # Add to image KG
        if 'image' in entity_data:
            self.image_kg.add_entity(entity_id, 'ImageEntity', entity_data['image'])

        # Add to audio KG
        if 'audio' in entity_data:
            self.audio_kg.add_entity(entity_id, 'AudioEntity', entity_data['audio'])

        # Create cross-modal alignment
        alignment = {
            'entity_id': entity_id,
            'modalities': list(entity_data.keys()),
            'alignment_score': 1.0
        }
        self.cross_modal_alignments.append(alignment)

    def cross_modal_query(self, query_modality, query_text, target_modality):
        """Query across different modalities"""

        # Find entities in query modality
        query_entities = self.find_entities_by_modality(query_modality, query_text)

        # Find corresponding entities in target modality
        target_entities = []
        for entity in query_entities:
            corresponding = self.find_corresponding_entity(entity, target_modality)
            if corresponding:
                target_entities.append(corresponding)

        return target_entities
```

### **10.2 Dynamic and Temporal Knowledge Graphs**

```python
class TemporalKnowledgeGraph:
    def __init__(self):
        self.kg_snapshots = {}  # {timestamp: KnowledgeGraph}
        self.temporal_patterns = []

    def add_temporal_fact(self, subject, relation, object, start_time, end_time=None):
        """Add fact valid during specific time period"""

        temporal_fact = {
            'subject': subject,
            'relation': relation,
            'object': object,
            'start_time': start_time,
            'end_time': end_time,
            'validity': 'always' if end_time is None else 'bounded'
        }

        # Store in appropriate time snapshot
        snapshot_key = self.get_time_snapshot(start_time)
        if snapshot_key not in self.kg_snapshots:
            self.kg_snapshots[snapshot_key] = KnowledgeGraph()

        self.kg_snapshots[snapshot_key].add_relationship(
            temporal_fact['subject'],
            temporal_fact['relation'],
            temporal_fact['object']
        )

    def temporal_reasoning(self, query, time_constraint):
        """Reason about facts at specific times"""

        # Find relevant time snapshots
        relevant_snapshots = self.find_relevant_snapshots(time_constraint)

        # Query each snapshot
        results = []
        for snapshot in relevant_snapshots:
            snapshot_results = snapshot.execute_sparql(query)
            results.extend(snapshot_results)

        # Merge and deduplicate results
        merged_results = self.merge_temporal_results(results)
        return merged_results

    def predict_future_facts(self, current_time, prediction_horizon):
        """Predict future facts using temporal patterns"""

        # Extract temporal patterns
        patterns = self.extract_temporal_patterns()

        # Make predictions
        predictions = []
        for pattern in patterns:
            if pattern['validity_period'] <= prediction_horizon:
                future_fact = {
                    'subject': pattern['subject'],
                    'relation': pattern['relation'],
                    'object': pattern['object'],
                    'predicted_time': current_time + pattern['average_interval'],
                    'confidence': pattern['confidence']
                }
                predictions.append(future_fact)

        return predictions
```

### **10.3 Continual Learning for Knowledge Graphs**

```python
class ContinualLearningKG:
    def __init__(self, initial_kg, learning_rate=0.01):
        self.kg = initial_kg
        self.learning_rate = learning_rate
        self.learned_patterns = []
        self.forgetting_curve = {}

    def incremental_learning(self, new_facts, task_id):
        """Incrementally learn new facts without forgetting"""

        # Detect novelty in new facts
        novel_facts = self.detect_novel_facts(new_facts)

        # Update knowledge graph
        for fact in novel_facts:
            self.kg.add_fact(fact)
            self.update_patterns(fact)

        # Apply elastic weight consolidation to prevent catastrophic forgetting
        self.apply_elastic_weight_consolidation(task_id)

        # Update forgetting curve
        self.update_forgetting_curve(task_id)

    def detect_novel_facts(self, new_facts):
        """Detect genuinely novel facts vs variations of existing facts"""

        novel_facts = []

        for fact in new_facts:
            # Check similarity to existing facts
            similar_facts = self.find_similar_facts(fact)

            # If no similar facts found, it's novel
            if not similar_facts or self.calculate_novelty_score(fact, similar_facts) > 0.8:
                novel_facts.append(fact)

        return novel_facts

    def apply_elastic_weight_consolidation(self, task_id):
        """Apply EWC to protect important weights from forgetting"""

        # Calculate Fisher Information Matrix for important parameters
        fisher_matrix = self.calculate_fisher_information()

        # Store EWC penalty terms
        ewc_penalty = {
            'task_id': task_id,
            'fisher_matrix': fisher_matrix,
            'optimal_parameters': self.get_current_parameters()
        }

        self.ewc_memory.append(ewc_penalty)

    def meta_learning_for_kg(self, support_facts, query_facts):
        """Use meta-learning to quickly adapt to new KG tasks"""

        # Learn initialization for rapid adaptation
        meta_init = self.meta_initialize(support_facts)

        # Adapt quickly to new facts
        adapted_model = self.fine_tune(meta_init, query_facts)

        return adapted_model
```

### **10.4 Neuro-Symbolic Reasoning**

```python
class NeuroSymbolicKG:
    def __init__(self, knowledge_graph, neural_model):
        self.kg = knowledge_graph
        self.neural_model = neural_model
        self.symbolic_rules = []
        self.neural_symbolic_bridge = NeuralSymbolicBridge()

    def hybrid_reasoning(self, query, reasoning_type='abductive'):
        """Combine neural and symbolic reasoning"""

        if reasoning_type == 'deductive':
            return self.symbolic_reasoning(query)
        elif reasoning_type == 'abductive':
            return self.neural_abductive_reasoning(query)
        elif reasoning_type == 'inductive':
            return self.neural_inductive_reasoning(query)
        else:
            return self.hybrid_reasoning_strategy(query)

    def neural_abductive_reasoning(self, query):
        """Use neural networks for abductive reasoning"""

        # Use neural model to generate explanations
        candidate_explanations = self.neural_model.generate_explanations(query)

        # Verify explanations using symbolic reasoning
        verified_explanations = []
        for explanation in candidate_explanations:
            if self.symbolic_verification(explanation, query):
                verified_explanations.append(explanation)

        return verified_explanations

    def symbolic_verification(self, explanation, query):
        """Verify neural explanation using symbolic reasoning"""

        # Convert explanation to symbolic form
        symbolic_explanation = self.neural_symbolic_bridge.neural_to_symbolic(explanation)

        # Check if explanation logically entails query
        return self.check_logical_entailment(symbolic_explanation, query)

    def learn_neural_symbolic_rules(self, training_data):
        """Learn rules that bridge neural and symbolic representations"""

        for example in training_data:
            # Extract symbolic structure
            symbolic_structure = self.extract_symbolic_structure(example)

            # Learn neural representation
            neural_representation = self.neural_model.learn_representation(symbolic_structure)

            # Create bridge rule
            bridge_rule = {
                'symbolic_pattern': symbolic_structure,
                'neural_representation': neural_representation,
                'confidence': example['confidence']
            }

            self.neural_symbolic_bridge.add_rule(bridge_rule)
```

---

## **Summary**

Knowledge Graphs represent a fundamental shift from traditional data storage to structured, interconnected representations of knowledge. Key takeaways:

1. **Structured Knowledge**: Enable explicit modeling of entities and relationships
2. **Query Flexibility**: Support both structured (SPARQL, Cypher) and natural language queries
3. **Integration with AI**: Enhance RAG systems and enable neural-symbolic reasoning
4. **Applications**: Power question answering, recommendation systems, and knowledge completion
5. **Ethical Considerations**: Require bias detection, fairness constraints, and privacy preservation
6. **Future Directions**: Moving toward multimodal, temporal, and continual learning systems

This theoretical foundation provides the comprehensive understanding needed to implement and deploy knowledge graph systems in modern AI applications.
