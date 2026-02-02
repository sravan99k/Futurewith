# 🔧 Knowledge Graphs Practice

_Hands-on exercises and coding implementations for knowledge graph systems_

---

## 📋 **Table of Contents**

1. [Environment Setup](#1-environment-setup)
2. [Basic Knowledge Graph Construction](#2-basic-knowledge-graph-construction)
3. [Entity Extraction and Linking](#3-entity-extraction-and-linking)
4. [Graph Database Operations](#4-graph-database-operations)
5. [Knowledge Graph Embeddings](#5-knowledge-graph-embeddings)
6. [Graph Neural Networks](#6-graph-neural-networks)
7. [RAG Enhancement with Knowledge Graphs](#7-rag-enhancement-with-knowledge-graphs)
8. [Question Answering Systems](#8-question-answering-systems)
9. [Bias Detection and Fairness](#9-bias-detection-and-fairness)
10. [Advanced Applications](#10-advanced-applications)

---

## **1. Environment Setup**

### **1.1 Required Dependencies**

```python
# Install required packages
import subprocess
import sys

def install_packages():
    """Install required packages for knowledge graph development"""
    packages = [
        'networkx',
        'rdflib',
        'py2neo',  # Neo4j driver
        'sentence-transformers',
        'transformers',
        'spacy',
        'scikit-learn',
        'numpy',
        'pandas',
        'torch',
        'torch-geometric',
        'openai',  # For LLM integration
    ]

    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

# Run installation
install_packages()

# Download spaCy model
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy English model loaded")
except OSError:
    print("📥 Downloading spaCy English model...")
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy English model downloaded and loaded")
```

### **1.2 Import Libraries and Basic Setup**

```python
import networkx as nx
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize spaCy
import spacy
nlp = spacy.load("en_core_web_sm")

print("🚀 Knowledge Graphs Practice Environment Ready!")
```

---

## **2. Basic Knowledge Graph Construction**

### **2.1 Simple Knowledge Graph Implementation**

```python
class SimpleKnowledgeGraph:
    """Basic knowledge graph implementation using NetworkX"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_properties = {}
        self.relationship_types = set()

    def add_entity(self, entity_id: str, entity_type: str, properties: Dict = None):
        """Add an entity to the knowledge graph"""
        self.graph.add_node(entity_id, node_type='entity', entity_type=entity_type)
        self.entity_properties[entity_id] = properties or {}
        logger.info(f"Added entity: {entity_id} (type: {entity_type})")

    def add_relationship(self, subject_id: str, predicate: str, object_id: str,
                        properties: Dict = None, confidence: float = 1.0):
        """Add a relationship between entities"""
        self.graph.add_edge(
            subject_id, object_id,
            relationship=predicate,
            properties=properties or {},
            confidence=confidence
        )
        self.relationship_types.add(predicate)
        logger.info(f"Added relationship: {subject_id} --{predicate}--> {object_id}")

    def get_entity(self, entity_id: str) -> Dict:
        """Get entity information"""
        if entity_id not in self.graph.nodes:
            return None

        return {
            'id': entity_id,
            'type': self.graph.nodes[entity_id].get('entity_type'),
            'properties': self.entity_properties.get(entity_id, {})
        }

    def get_relationships(self, entity_id: str, direction: str = 'out') -> List[Dict]:
        """Get relationships for an entity"""
        if entity_id not in self.graph.nodes:
            return []

        relationships = []

        if direction in ['out', 'both']:
            for target in self.graph.successors(entity_id):
                edge_data = self.graph[entity_id][target]
                relationships.append({
                    'subject': entity_id,
                    'predicate': edge_data['relationship'],
                    'object': target,
                    'confidence': edge_data['confidence'],
                    'properties': edge_data['properties']
                })

        if direction in ['in', 'both']:
            for source in self.graph.predecessors(entity_id):
                edge_data = self.graph[source][entity_id]
                relationships.append({
                    'subject': source,
                    'predicate': edge_data['relationship'],
                    'object': entity_id,
                    'confidence': edge_data['confidence'],
                    'properties': edge_data['properties']
                })

        return relationships

    def find_path(self, start_entity: str, end_entity: str, max_length: int = 3) -> List[str]:
        """Find path between two entities"""
        try:
            path = nx.shortest_path(self.graph, start_entity, end_entity, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return []

    def get_graph_statistics(self) -> Dict:
        """Get basic statistics about the knowledge graph"""
        return {
            'num_entities': len(self.graph.nodes),
            'num_relationships': len(self.graph.edges),
            'num_relationship_types': len(self.relationship_types),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]),
            'density': nx.density(self.graph)
        }

# Exercise 2.1: Create a simple knowledge graph
def exercise_basic_kg():
    """Exercise: Build a simple knowledge graph about companies and founders"""

    # Create knowledge graph
    kg = SimpleKnowledgeGraph()

    # Add entities
    entities_data = [
        ("steve_jobs", "Person", {"name": "Steve Jobs", "birth_year": 1955, "occupation": "CEO"}),
        ("steve_wozniak", "Person", {"name": "Steve Wozniak", "birth_year": 1950, "occupation": "Engineer"}),
        ("apple_inc", "Company", {"name": "Apple Inc.", "founded": 1976, "industry": "Technology"}),
        ("microsoft", "Company", {"name": "Microsoft", "founded": 1975, "industry": "Technology"}),
        ("bill_gates", "Person", {"name": "Bill Gates", "birth_year": 1955, "occupation": "CEO"})
    ]

    for entity_id, entity_type, properties in entities_data:
        kg.add_entity(entity_id, entity_type, properties)

    # Add relationships
    relationships_data = [
        ("steve_jobs", "co_founded", "apple_inc", {"year": 1976}),
        ("steve_wozniak", "co_founded", "apple_inc", {"year": 1976}),
        ("steve_jobs", "worked_at", "apple_inc", {"position": "CEO", "years": 1976-2011}),
        ("bill_gates", "co_founded", "microsoft", {"year": 1975}),
        ("bill_gates", "worked_at", "microsoft", {"position": "CEO", "years": 1975-2000})
    ]

    for subject, predicate, obj, props in relationships_data:
        kg.add_relationship(subject, predicate, obj, props)

    # Print statistics
    stats = kg.get_graph_statistics()
    print("📊 Knowledge Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Query examples
    print("\n🔍 Query Examples:")

    # Get relationships for Steve Jobs
    steve_jobs_rels = kg.get_relationships("steve_jobs")
    print(f"Steve Jobs relationships: {len(steve_jobs_rels)} found")
    for rel in steve_jobs_rels[:3]:  # Show first 3
        print(f"  {rel['subject']} --{rel['predicate']}--> {rel['object']}")

    # Find path between Steve Jobs and Bill Gates
    path = kg.find_path("steve_jobs", "bill_gates")
    if path:
        print(f"Path from Steve Jobs to Bill Gates: {' -> '.join(path)}")
    else:
        print("No path found between Steve Jobs and Bill Gates")

    return kg

# Run exercise
kg = exercise_basic_kg()
```

### **2.2 RDF Knowledge Graph Implementation**

```python
from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef

class RDFKnowledgeGraph:
    """RDF-based knowledge graph implementation"""

    def __init__(self, base_namespace="http://example.org/kg/"):
        self.graph = Graph()
        self.base_ns = Namespace(base_namespace)

        # Bind prefixes
        self.graph.bind("kg", self.base_ns)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)

    def add_entity(self, entity_id: str, entity_type: str, properties: Dict):
        """Add entity with RDF representation"""
        entity_uri = self.base_ns[entity_id]

        # Add type
        type_uri = self.base_ns[entity_type]
        self.graph.add((entity_uri, RDF.type, type_uri))

        # Add properties
        for prop_name, prop_value in properties.items():
            prop_uri = self.base_ns[prop_name]
            if isinstance(prop_value, str):
                self.graph.add((entity_uri, prop_uri, Literal(prop_value)))
            elif isinstance(prop_value, (int, float)):
                self.graph.add((entity_uri, prop_uri, Literal(prop_value)))
            elif isinstance(prop_value, bool):
                self.graph.add((entity_uri, prop_uri, Literal(prop_value)))

        logger.info(f"Added RDF entity: {entity_id}")

    def add_relationship(self, subject_id: str, predicate: str, object_id: str, properties: Dict = None):
        """Add relationship with RDF representation"""
        subject_uri = self.base_ns[subject_id]
        predicate_uri = self.base_ns[predicate]
        object_uri = self.base_ns[object_id]

        # Add main relationship
        self.graph.add((subject_uri, predicate_uri, object_uri))

        # Add relationship properties if provided
        if properties:
            for prop_name, prop_value in properties.items():
                prop_uri = self.base_ns[f"{predicate}_{prop_name}"]
                if isinstance(prop_value, str):
                    self.graph.add((subject_uri, prop_uri, Literal(prop_value)))
                elif isinstance(prop_value, (int, float)):
                    self.graph.add((subject_uri, prop_uri, Literal(prop_value)))

        logger.info(f"Added RDF relationship: {subject_id} --{predicate}--> {object_id}")

    def sparql_query(self, query: str) -> List[Dict]:
        """Execute SPARQL query"""
        results = []
        try:
            query_result = self.graph.query(query)
            for row in query_result:
                result_dict = {}
                for i, var in enumerate(query_result.vars):
                    result_dict[str(var)] = str(row[i])
                results.append(result_dict)
        except Exception as e:
            logger.error(f"SPARQL query error: {e}")

        return results

    def get_entity_by_type(self, entity_type: str) -> List[str]:
        """Get all entities of a specific type"""
        query = f"""
        SELECT ?entity WHERE {{
            ?entity rdf:type kg:{entity_type} .
        }}
        """
        results = self.sparql_query(query)
        return [result['entity'].replace(str(self.base_ns), '') for result in results]

    def get_relationships(self, entity_id: str, direction: str = 'out') -> List[Dict]:
        """Get relationships for an entity"""
        entity_uri = self.base_ns[entity_id]
        relationships = []

        if direction in ['out', 'both']:
            # Outgoing relationships
            query = f"""
            SELECT ?predicate ?object WHERE {{
                <{entity_uri}> ?predicate ?object .
            }}
            """
            results = self.sparql_query(query)
            for result in results:
                relationships.append({
                    'subject': entity_id,
                    'predicate': result['predicate'].replace(str(self.base_ns), ''),
                    'object': result['object'].replace(str(self.base_ns), '')
                })

        if direction in ['in', 'both']:
            # Incoming relationships
            query = f"""
            SELECT ?subject ?predicate WHERE {{
                ?subject ?predicate <{entity_uri}> .
            }}
            """
            results = self.sparql_query(query)
            for result in results:
                relationships.append({
                    'subject': result['subject'].replace(str(self.base_ns), ''),
                    'predicate': result['predicate'].replace(str(self.base_ns), ''),
                    'object': entity_id
                })

        return relationships

    def export_to_file(self, filename: str, format: str = 'turtle'):
        """Export knowledge graph to file"""
        self.graph.serialize(destination=filename, format=format)
        logger.info(f"Exported knowledge graph to {filename}")

# Exercise 2.2: Create RDF knowledge graph
def exercise_rdf_kg():
    """Exercise: Create RDF knowledge graph for academic publications"""

    # Create RDF knowledge graph
    rdf_kg = RDFKnowledgeGraph("http://example.org/academic/")

    # Add authors
    authors = [
        ("alice_smith", {"name": "Alice Smith", "affiliation": "MIT", "orcid": "0000-0001-2345-6789"}),
        ("bob_jones", {"name": "Bob Jones", "affiliation": "Stanford", "orcid": "0000-0002-3456-7890"}),
        ("carol_brown", {"name": "Carol Brown", "affiliation": "Cambridge", "orcid": "0000-0003-4567-8901"})
    ]

    for author_id, props in authors:
        rdf_kg.add_entity(author_id, "Author", props)

    # Add papers
    papers = [
        ("paper1", {"title": "Neural Networks for Natural Language Processing", "year": 2023, "venue": "NeurIPS"}),
        ("paper2", {"title": "Graph Convolutional Networks", "year": 2022, "venue": "ICML"}),
        ("paper3", {"title": "Transformer Architecture", "year": 2021, "venue": "ACL"})
    ]

    for paper_id, props in papers:
        rdf_kg.add_entity(paper_id, "Paper", props)

    # Add relationships (authorship)
    rdf_kg.add_relationship("alice_smith", "authored", "paper1")
    rdf_kg.add_relationship("alice_smith", "authored", "paper3")
    rdf_kg.add_relationship("bob_jones", "authored", "paper1")
    rdf_kg.add_relationship("bob_jones", "authored", "paper2")
    rdf_kg.add_relationship("carol_brown", "authored", "paper2")

    # SPARQL query examples
    print("🔍 SPARQL Query Examples:")

    # Query 1: Find all authors
    authors_query = """
    SELECT ?author ?name WHERE {
        ?author rdf:type kg:Author ;
                kg:name ?name .
    }
    """
    authors_results = rdf_kg.sparql_query(authors_query)
    print(f"All authors: {[result['name'] for result in authors_results]}")

    # Query 2: Find papers by Alice Smith
    alice_papers_query = f"""
    SELECT ?paper ?title WHERE {{
        kg:alice_smith kg:authored ?paper .
        ?paper kg:title ?title .
    }}
    """
    papers_results = rdf_kg.sparql_query(alice_papers_query)
    print(f"Papers by Alice Smith: {[result['title'] for result in papers_results]}")

    # Export to file
    rdf_kg.export_to_file("academic_kg.ttl")
    print("📁 Exported RDF knowledge graph to academic_kg.ttl")

    return rdf_kg

# Run RDF exercise
rdf_kg = exercise_rdf_kg()
```

---

## **3. Entity Extraction and Linking**

### **3.1 Named Entity Recognition**

```python
import re
from collections import defaultdict

class EntityExtractor:
    """Extract entities from text using spaCy and custom rules"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.custom_patterns = []

    def add_custom_pattern(self, pattern: str, entity_type: str):
        """Add custom regex pattern for entity extraction"""
        self.custom_patterns.append((re.compile(pattern, re.IGNORECASE), entity_type))

    def extract_entities(self, text: str) -> List[Dict]:
        """Extract all entities from text"""
        entities = []

        # Use spaCy for NER
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 0.8,  # spaCy doesn't provide confidence by default
                'source': 'spacy'
            })

        # Apply custom patterns
        entities.extend(self.extract_with_patterns(text))

        return entities

    def extract_with_patterns(self, text: str) -> List[Dict]:
        """Extract entities using custom patterns"""
        entities = []

        for pattern, entity_type in self.custom_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9,
                    'source': 'custom_pattern'
                })

        return entities

    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relationships between entities"""
        relationships = []

        # Simple relationship extraction using dependency parsing
        doc = self.nlp(text)

        for sent in doc.sents:
            sent_entities = [e for e in entities
                           if e['start'] >= sent.start_char and e['end'] <= sent.end_char]

            if len(sent_entities) >= 2:
                # Extract dependency-based relationships
                for token in sent:
                    if token.dep_ in ['nsubj', 'dobj', 'compound']:
                        subject = self.find_closest_entity(token.text, sent_entities)
                        obj = self.find_closest_entity(token.head.text, sent_entities)

                        if subject and obj:
                            relationships.append({
                                'subject': subject['text'],
                                'predicate': token.dep_,
                                'object': obj['text'],
                                'confidence': 0.6,
                                'source': 'dependency_parse'
                            })

        return relationships

    def find_closest_entity(self, word: str, entities: List[Dict]) -> Dict:
        """Find the entity closest to a given word"""
        word_pos = word.lower()

        for entity in entities:
            if word_pos in entity['text'].lower():
                return entity

        return None

# Exercise 3.1: Entity extraction
def exercise_entity_extraction():
    """Exercise: Extract entities from news articles"""

    extractor = EntityExtractor()

    # Add custom patterns
    extractor.add_custom_pattern(r'\b[A-Z][a-z]+ \d{4}\b', 'DATE')  # Years like "2023"
    extractor.add_custom_pattern(r'\$\d+(?:\.\d{2})?', 'MONEY')    # Money amounts
    extractor.add_custom_pattern(r'\b[A-Z]{2,}\b', 'ORGANIZATION')  # Acronyms

    # Sample news text
    news_text = """
    Apple Inc. announced on September 12, 2023, that Tim Cook will step down as CEO in 2024.
    The company reported revenue of $394.3 billion in fiscal year 2023.
    Cook has been CEO since 2011, succeeding Steve Jobs.
    Analysts at Goldman Sachs expect the transition to be smooth.
    """

    # Extract entities
    entities = extractor.extract_entities(news_text)
    print("🔍 Extracted Entities:")
    for entity in entities:
        print(f"  {entity['text']} ({entity['label']}) - {entity['source']}")

    # Extract relationships
    relationships = extractor.extract_relationships(news_text, entities)
    print("\n🔗 Extracted Relationships:")
    for rel in relationships:
        print(f"  {rel['subject']} --{rel['predicate']}--> {rel['object']}")

    return entities, relationships

# Run entity extraction exercise
entities, relationships = exercise_entity_extraction()
```

### **3.2 Entity Linking and Disambiguation**

```python
from difflib import SequenceMatcher
import difflib

class EntityLinker:
    """Link extracted entities to knowledge graph entities"""

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.candidates = {}
        self.similarity_threshold = 0.8

    def build_candidate_set(self, entity_type: str = None):
        """Build candidate set for entity linking"""
        candidates = {}

        for entity_id in self.kg.graph.nodes:
            entity_info = self.kg.get_entity(entity_id)
            if entity_type is None or entity_info['type'] == entity_type:
                # Add name variants
                names = [entity_info['properties'].get('name', '')]
                names.extend(entity_info['properties'].get('aliases', []))

                for name in names:
                    if name:
                        candidates[name.lower()] = entity_id

        self.candidates = candidates
        logger.info(f"Built candidate set with {len(candidates)} entities")

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate string similarity between two texts"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def find_candidates(self, entity_text: str) -> List[Tuple[str, float]]:
        """Find candidate entities for a given text"""
        candidates = []

        # Exact match
        if entity_text.lower() in self.candidates:
            candidates.append((self.candidates[entity_text.lower()], 1.0))

        # Fuzzy match
        text_lower = entity_text.lower()
        for candidate_text, entity_id in self.candidates.items():
            similarity = self.calculate_similarity(text_lower, candidate_text)
            if similarity > self.similarity_threshold:
                candidates.append((entity_id, similarity))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def link_entity(self, entity: Dict) -> Dict:
        """Link a single entity to knowledge graph"""
        candidates = self.find_candidates(entity['text'])

        if candidates:
            best_candidate, similarity = candidates[0]
            return {
                'original_entity': entity,
                'kg_entity': best_candidate,
                'confidence': similarity,
                'candidates': candidates[:3]  # Top 3 candidates
            }
        else:
            return {
                'original_entity': entity,
                'kg_entity': None,
                'confidence': 0.0,
                'candidates': []
            }

    def batch_link_entities(self, entities: List[Dict]) -> List[Dict]:
        """Link multiple entities to knowledge graph"""
        if not self.candidates:
            self.build_candidate_set()

        linked_entities = []
        for entity in entities:
            linked_entity = self.link_entity(entity)
            linked_entities.append(linked_entity)

        return linked_entities

# Exercise 3.2: Entity linking
def exercise_entity_linking():
    """Exercise: Link extracted entities to knowledge graph"""

    # Use the knowledge graph from earlier
    linker = EntityLinker(kg)

    # Link previously extracted entities
    linked_entities = linker.batch_link_entities(entities)

    print("🔗 Entity Linking Results:")
    for linked in linked_entities:
        original = linked['original_entity']
        if linked['kg_entity']:
            kg_entity = kg.get_entity(linked['kg_entity'])
            print(f"  {original['text']} ({original['label']}) → {kg_entity['properties'].get('name', linked['kg_entity'])} "
                  f"(confidence: {linked['confidence']:.2f})")
        else:
            print(f"  {original['text']} ({original['label']}) → No match found")

    return linked_entities

# Run entity linking exercise
linked_entities = exercise_entity_linking()
```

---

## **4. Graph Database Operations**

### **4.1 Neo4j Integration**

```python
# Note: This requires a running Neo4j instance
# For demonstration purposes, we'll show the structure without actual connection

class Neo4jKnowledgeGraph:
    """Knowledge graph implementation using Neo4j"""

    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        # In a real implementation, you would connect here:
        # self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.warning("Neo4j driver not initialized - showing structure only")

    def create_constraints(self):
        """Create database constraints and indexes"""
        constraints_queries = [
            "CREATE CONSTRAINT unique_person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT unique_company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)"
        ]

        logger.info("Neo4j constraints defined (not executed - requires running instance)")
        return constraints_queries

    def create_entity_node(self, entity_id: str, entity_type: str, properties: Dict):
        """Create entity node in Neo4j"""
        query = f"""
        CREATE (n:{entity_type} {{
            id: $entity_id,
            name: $name,
            created_at: datetime()
        }})
        SET n += $properties
        RETURN n
        """

        # In real implementation:
        # with self.driver.session() as session:
        #     result = session.run(query, entity_id=entity_id, name=properties.get('name', ''), properties=properties)
        #     return result.single()

        logger.info(f"Neo4j query: CREATE entity {entity_id} of type {entity_type}")

    def create_relationship(self, subject_id: str, predicate: str, object_id: str, properties: Dict = None):
        """Create relationship between entities"""
        query = f"""
        MATCH (a {{id: $subject_id}}), (b {{id: $object_id}})
        CREATE (a)-[r:{predicate.upper()} $relationship_props]->(b)
        RETURN r
        """

        # In real implementation:
        # with self.driver.session() as session:
        #     result = session.run(query, subject_id=subject_id, object_id=object_id, relationship_props=properties or {})
        #     return result.single()

        logger.info(f"Neo4j query: CREATE relationship {subject_id} --{predicate}--> {object_id}")

    def find_shortest_path(self, start_id: str, end_id: str):
        """Find shortest path between two entities"""
        query = f"""
        MATCH (start {{id: $start_id}}), (end {{id: $end_id}})
        MATCH path = shortestPath((start)-[*]-(end))
        RETURN path
        """

        logger.info(f"Neo4j query: shortest path from {start_id} to {end_id}")
        return "Query structure shown - requires running Neo4j instance"

    def find_entities_by_type(self, entity_type: str, limit: int = 100):
        """Find entities by type"""
        query = f"""
        MATCH (n:{entity_type})
        RETURN n
        LIMIT $limit
        """

        logger.info(f"Neo4j query: find {limit} {entity_type} entities")
        return "Query structure shown - requires running Neo4j instance"

    def cypher_examples(self):
        """Show example Cypher queries"""
        examples = {
            "Find all relationships for a person": """
                MATCH (p:Person {id: 'steve_jobs'})-[r]-(connected)
                RETURN p.name, type(r), connected.name, connected.type
            """,

            "Find all companies founded by a person": """
                MATCH (p:Person)-[:CO_FOUNDED]->(c:Company)
                WHERE p.name CONTAINS 'Steve'
                RETURN p.name, c.name, c.founded
            """,

            "Find paths between two people": """
                MATCH path = shortestPath(
                    (p1:Person {id: 'steve_jobs'})-[*]-(p2:Person {id: 'bill_gates'})
                )
                RETURN path
            """,

            "Aggregate query - count relationships": """
                MATCH (p:Person)-[r]-(connected)
                RETURN p.name, COUNT(r) as relationship_count
                ORDER BY relationship_count DESC
            """
        }

        print("💬 Example Cypher Queries:")
        for description, query in examples.items():
            print(f"\n{description}:")
            print(query)

        return examples

# Exercise 4.1: Neo4j operations
def exercise_neo4j():
    """Exercise: Neo4j operations structure"""

    # Create Neo4j KG instance (without actual connection)
    neo4j_kg = Neo4jKnowledgeGraph("bolt://localhost:7687", "neo4j", "password")

    # Show constraint creation
    print("🔧 Creating Neo4j Constraints:")
    constraints = neo4j_kg.create_constraints()
    for constraint in constraints:
        print(f"  {constraint}")

    # Show Cypher examples
    cypher_queries = neo4j_kg.cypher_examples()

    return neo4j_kg

# Run Neo4j exercise
neo4j_kg = exercise_neo4j()
```

### **4.2 Graph Analytics and Mining**

```python
class GraphAnalytics:
    """Graph analytics and mining operations"""

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.graph = knowledge_graph.graph

    def calculate_centrality_measures(self):
        """Calculate various centrality measures"""
        centrality_measures = {}

        # Degree centrality
        degree_centrality = nx.degree_centrality(self.graph)
        centrality_measures['degree_centrality'] = degree_centrality

        # Betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        centrality_measures['betweenness_centrality'] = betweenness_centrality

        # Closeness centrality
        closeness_centrality = nx.closeness_centrality(self.graph)
        centrality_measures['closeness_centrality'] = closeness_centrality

        # PageRank (if graph is directed)
        try:
            pagerank = nx.pagerank(self.graph)
            centrality_measures['pagerank'] = pagerank
        except:
            logger.warning("PageRank calculation failed - graph may not be suitable")

        return centrality_measures

    def find_communities(self):
        """Find communities in the graph"""
        try:
            # Use Louvain method for community detection
            import networkx.algorithms.community as nx_comm

            communities = nx_comm.greedy_modularity_communities(self.graph)

            community_mapping = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_mapping[node] = i

            return {
                'communities': communities,
                'mapping': community_mapping,
                'modularity': nx_comm.modularity(self.graph, communities)
            }
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return None

    def analyze_graph_structure(self):
        """Analyze overall graph structure"""
        structure_analysis = {}

        # Basic metrics
        structure_analysis['num_nodes'] = self.graph.number_of_nodes()
        structure_analysis['num_edges'] = self.graph.number_of_edges()
        structure_analysis['density'] = nx.density(self.graph)

        # Connectivity
        structure_analysis['is_connected'] = nx.is_connected(self.graph)
        if not structure_analysis['is_connected']:
            structure_analysis['num_components'] = nx.number_connected_components(self.graph)
            structure_analysis['largest_component_size'] = len(max(nx.connected_components(self.graph), key=len))

        # Diameter and average shortest path
        if structure_analysis['is_connected']:
            structure_analysis['diameter'] = nx.diameter(self.graph)
            structure_analysis['avg_shortest_path'] = nx.average_shortest_path_length(self.graph)

        # Clustering
        structure_analysis['avg_clustering'] = nx.average_clustering(self.graph)

        return structure_analysis

    def find_important_entities(self, top_k=10):
        """Find most important entities based on various measures"""
        importance_scores = {}

        # Centrality measures
        centrality_measures = self.calculate_centrality_measures()

        for measure_name, scores in centrality_measures.items():
            top_entities = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            importance_scores[measure_name] = top_entities

        return importance_scores

    def detect_anomalies(self):
        """Detect anomalous patterns in the graph"""
        anomalies = {}

        # Detect isolated nodes
        isolated_nodes = list(nx.isolates(self.graph))
        if isolated_nodes:
            anomalies['isolated_nodes'] = isolated_nodes

        # Detect nodes with very high degree (potential hubs)
        degree_sequence = [d for n, d in self.graph.degree()]
        threshold = np.mean(degree_sequence) + 2 * np.std(degree_sequence)
        high_degree_nodes = [n for n, d in self.graph.degree() if d > threshold]
        if high_degree_nodes:
            anomalies['high_degree_nodes'] = high_degree_nodes

        # Detect potential duplicate entities (very similar names)
        potential_duplicates = self.find_potential_duplicates()
        if potential_duplicates:
            anomalies['potential_duplicates'] = potential_duplicates

        return anomalies

    def find_potential_duplicates(self, similarity_threshold=0.8):
        """Find potential duplicate entities based on name similarity"""
        potential_duplicates = []
        entities = list(self.graph.nodes())

        for i, entity1 in enumerate(entities):
            entity1_info = self.kg.get_entity(entity1)
            if not entity1_info:
                continue

            name1 = entity1_info['properties'].get('name', entity1)

            for entity2 in entities[i+1:]:
                entity2_info = self.kg.get_entity(entity2)
                if not entity2_info:
                    continue

                name2 = entity2_info['properties'].get('name', entity2)

                similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
                if similarity > similarity_threshold:
                    potential_duplicates.append({
                        'entity1': entity1,
                        'name1': name1,
                        'entity2': entity2,
                        'name2': name2,
                        'similarity': similarity
                    })

        return potential_duplicates

# Exercise 4.2: Graph analytics
def exercise_graph_analytics():
    """Exercise: Analyze the knowledge graph"""

    analytics = GraphAnalytics(kg)

    print("📊 Graph Structure Analysis:")
    structure = analytics.analyze_graph_structure()
    for key, value in structure.items():
        print(f"  {key}: {value}")

    print("\n🎯 Important Entities (Top 5):")
    important = analytics.find_important_entities(top_k=5)

    for measure, entities in important.items():
        print(f"\n{measure.replace('_', ' ').title()}:")
        for entity, score in entities:
            entity_info = kg.get_entity(entity)
            name = entity_info['properties'].get('name', entity) if entity_info else entity
            print(f"  {name}: {score:.3f}")

    print("\n🔍 Anomaly Detection:")
    anomalies = analytics.detect_anomalies()
    for anomaly_type, anomaly_data in anomalies.items():
        print(f"  {anomaly_type}: {len(anomaly_data)} found")
        if isinstance(anomaly_data, list) and len(anomaly_data) <= 5:
            for item in anomaly_data:
                print(f"    {item}")

    print("\n🤝 Community Detection:")
    communities = analytics.find_communities()
    if communities:
        print(f"  Found {len(communities['communities'])} communities")
        print(f"  Modularity score: {communities['modularity']:.3f}")

        # Show community members for first few communities
        for i, community in enumerate(communities['communities'][:3]):
            members = list(community)[:5]  # Show first 5 members
            member_names = []
            for member in members:
                entity_info = kg.get_entity(member)
                name = entity_info['properties'].get('name', member) if entity_info else member
                member_names.append(name)
            print(f"    Community {i}: {', '.join(member_names)}")

    return analytics

# Run graph analytics exercise
analytics = exercise_graph_analytics()
```

---

## **5. Knowledge Graph Embeddings**

### **5.1 TransE Implementation**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TransE(nn.Module):
    """TransE embedding model implementation"""

    def __init__(self, num_entities, num_relations, embedding_dim=100, margin=1.0):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin

        # Embedding layers
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings"""
        # Entity embeddings: uniform initialization in [-6/√d, 6/√d]
        bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)

        # Relation embeddings: uniform initialization in [-6/√d, 6/√d]
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)

    def forward(self, head, relation, tail):
        """Forward pass - calculate scores for triples"""
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        # TransE scoring: h + r ≈ t
        # Score is negative L1 distance (higher score = better)
        scores = -torch.norm(h + r - t, p=1, dim=1)
        return scores

    def predict_tail(self, head, relation, candidate_tails):
        """Predict most likely tail entity"""
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)

        scores = []
        for tail_idx in candidate_tails:
            t = self.entity_embeddings(tail_idx)
            score = -torch.norm(h + r - t, p=1).item()
            scores.append(score)

        # Return tail with highest score
        max_score_idx = np.argmax(scores)
        return candidate_tails[max_score_idx], scores[max_score_idx]

class TransETrainingDataset(Dataset):
    """Dataset for TransE training"""

    def __init__(self, triples, entity_to_id, relation_to_id, negative_sampling_size=10):
        self.triples = triples
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.negative_sampling_size = negative_sampling_size

        # Convert triples to tensor format
        self.tensor_triples = []
        for head, relation, tail in triples:
            self.tensor_triples.append((
                torch.tensor(entity_to_id[head]),
                torch.tensor(relation_to_id[relation]),
                torch.tensor(entity_to_id[tail])
            ))

    def __len__(self):
        return len(self.tensor_triples)

    def __getitem__(self, idx):
        head, relation, tail = self.tensor_triples[idx]

        # Generate negative samples by corrupting head or tail
        head_neg = head.clone()
        tail_neg = tail.clone()

        # Corrupt head (but not tail)
        if np.random.random() < 0.5:
            head_neg = torch.randint(0, len(self.entity_to_id), (1,)).item()

        # Corrupt tail (but not head)
        if np.random.random() < 0.5:
            tail_neg = torch.randint(0, len(self.entity_to_id), (1,)).item()

        return {
            'head': head,
            'relation': relation,
            'tail': tail,
            'head_neg': head_neg,
            'tail_neg': tail_neg
        }

class TransETrainer:
    """Trainer for TransE model"""

    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MarginRankingLoss(margin=1.0)

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            head = batch['head']
            relation = batch['relation']
            tail = batch['tail']
            head_neg = batch['head_neg']
            tail_neg = batch['tail_neg']

            # Forward pass
            pos_scores = self.model(head, relation, tail)
            neg_scores_head = self.model(head_neg, relation, tail)
            neg_scores_tail = self.model(head, relation, tail_neg)

            # Create labels (positive should have higher score than negative)
            y = torch.ones_like(pos_scores)

            # Compute losses
            loss1 = self.criterion(pos_scores, neg_scores_head, y)
            loss2 = self.criterion(pos_scores, neg_scores_tail, y)
            loss = loss1 + loss2

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, test_triples, entity_to_id, relation_to_id):
        """Evaluate the model"""
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for head_name, relation_name, tail_name in test_triples:
                if (head_name in entity_to_id and
                    tail_name in entity_to_id and
                    relation_name in relation_to_id):

                    head_id = entity_to_id[head_name]
                    relation_id = relation_to_id[relation_name]
                    tail_id = entity_to_id[tail_name]

                    # Get candidate tails
                    candidate_tails = list(entity_to_id.values())

                    # Predict tail
                    predicted_tail, score = self.model.predict_tail(
                        torch.tensor(head_id),
                        torch.tensor(relation_id),
                        candidate_tails
                    )

                    if predicted_tail == tail_id:
                        correct_predictions += 1
                    total_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy, correct_predictions, total_predictions

# Exercise 5.1: TransE training
def exercise_transE():
    """Exercise: Train TransE embeddings on knowledge graph"""

    # Prepare data
    entities = set()
    relations = set()
    triples = []

    for node in kg.graph.nodes():
        entities.add(node)

    for edge in kg.graph.edges(data=True):
        subject, obj, data = edge
        relation = data.get('relationship', 'RELATED_TO')
        relations.add(relation)
        triples.append((subject, relation, obj))

    # Create mappings
    entity_to_id = {entity: i for i, entity in enumerate(sorted(entities))}
    relation_to_id = {relation: i for i, relation in enumerate(sorted(relations))}

    print(f"📊 Dataset Statistics:")
    print(f"  Entities: {len(entities)}")
    print(f"  Relations: {len(relations)}")
    print(f"  Triples: {len(triples)}")

    # Split data
    train_size = int(0.8 * len(triples))
    test_size = len(triples) - train_size

    train_triples = triples[:train_size]
    test_triples = triples[train_size:]

    print(f"  Training triples: {len(train_triples)}")
    print(f"  Test triples: {len(test_triples)}")

    # Create model
    model = TransE(
        num_entities=len(entities),
        num_relations=len(relations),
        embedding_dim=50  # Smaller for quick training
    )

    # Create dataset and dataloader
    dataset = TransETrainingDataset(train_triples, entity_to_id, relation_to_id)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train model
    trainer = TransETrainer(model, learning_rate=0.01)

    print("\n🚀 Training TransE Model:")
    num_epochs = 10

    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch(dataloader)
        print(f"  Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Evaluate model
    print("\n📈 Evaluation:")
    accuracy, correct, total = trainer.evaluate(test_triples, entity_to_id, relation_to_id)
    print(f"  Accuracy: {accuracy:.3f} ({correct}/{total} correct)")

    # Show some predictions
    print("\n🔮 Sample Predictions:")
    with torch.no_grad():
        for i, (head, relation, tail) in enumerate(test_triples[:3]):
            if head in entity_to_id and relation in relation_to_id:
                head_id = entity_to_id[head]
                relation_id = relation_to_id[relation]
                candidate_tails = list(entity_to_id.values())

                predicted_tail, score = model.predict_tail(
                    torch.tensor(head_id),
                    torch.tensor(relation_id),
                    candidate_tails
                )

                # Convert back to entity name
                predicted_entity = [k for k, v in entity_to_id.items() if v == predicted_tail][0]
                actual_tail = tail

                print(f"  Query: {head} --{relation}--> ?")
                print(f"  Predicted: {predicted_entity} (score: {score:.3f})")
                print(f"  Actual: {actual_tail}")
                print()

    return model, trainer, entity_to_id, relation_to_id

# Run TransE exercise
transE_model, trainer, entity_to_id, relation_to_id = exercise_transE()
```

### **5.2 Entity Similarity with Embeddings**

```python
class EmbeddingSimilarity:
    """Calculate entity similarity using embeddings"""

    def __init__(self, model, entity_to_id):
        self.model = model
        self.entity_to_id = entity_to_id
        self.id_to_entity = {v: k for k, v in entity_to_id.items()}

    def get_entity_embedding(self, entity_name):
        """Get embedding for an entity"""
        if entity_name not in self.entity_to_id:
            return None

        entity_id = self.entity_to_id[entity_name]
        with torch.no_grad():
            embedding = self.model.entity_embeddings(torch.tensor(entity_id))
        return embedding.numpy()

    def calculate_similarity(self, entity1, entity2, metric='cosine'):
        """Calculate similarity between two entities"""
        emb1 = self.get_entity_embedding(entity1)
        emb2 = self.get_entity_embedding(entity2)

        if emb1 is None or emb2 is None:
            return 0.0

        if metric == 'cosine':
            # Cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            return dot_product / (norm1 * norm2)
        elif metric == 'euclidean':
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(emb1 - emb2)
            return 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def find_similar_entities(self, target_entity, top_k=5, metric='cosine'):
        """Find most similar entities to target entity"""
        similarities = []

        for entity_name in self.entity_to_id.keys():
            if entity_name != target_entity:
                similarity = self.calculate_similarity(target_entity, entity_name, metric)
                similarities.append((entity_name, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def analyze_embedding_space(self):
        """Analyze the embedding space"""
        embeddings = {}

        for entity_name in self.entity_to_id.keys():
            embedding = self.get_entity_embedding(entity_name)
            if embedding is not None:
                embeddings[entity_name] = embedding

        # Calculate clustering tendencies
        embedding_matrix = np.array(list(embeddings.values()))

        # Cosine similarity matrix
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i, entity1 in enumerate(embeddings.keys()):
            for j, entity2 in enumerate(embeddings.keys()):
                if i != j:
                    similarity = self.calculate_similarity(entity1, entity2)
                    similarity_matrix[i, j] = similarity

        return {
            'embeddings': embeddings,
            'similarity_matrix': similarity_matrix,
            'avg_similarity': np.mean(similarity_matrix[similarity_matrix > 0]),
            'max_similarity': np.max(similarity_matrix),
            'min_similarity': np.min(similarity_matrix[similarity_matrix > 0])
        }

# Exercise 5.2: Embedding similarity
def exercise_embedding_similarity():
    """Exercise: Analyze entity similarities using embeddings"""

    similarity_analyzer = EmbeddingSimilarity(transE_model, entity_to_id)

    print("🔍 Entity Similarity Analysis:")

    # Analyze embedding space
    analysis = similarity_analyzer.analyze_embedding_space()
    print(f"  Average similarity: {analysis['avg_similarity']:.3f}")
    print(f"  Max similarity: {analysis['max_similarity']:.3f}")
    print(f"  Min similarity: {analysis['min_similarity']:.3f}")

    # Find similar entities for each entity
    print("\n🎯 Most Similar Entities:")

    for entity in list(entity_to_id.keys())[:5]:  # Show first 5 entities
        similar_entities = similarity_analyzer.find_similar_entities(entity, top_k=3)

        entity_info = kg.get_entity(entity)
        entity_name = entity_info['properties'].get('name', entity) if entity_info else entity
        print(f"\n  Similar to {entity_name}:")

        for similar_entity, similarity in similar_entities:
            similar_info = kg.get_entity(similar_entity)
            similar_name = similar_info['properties'].get('name', similar_entity) if similar_info else similar_entity
            print(f"    {similar_name}: {similarity:.3f}")

    # Show some specific relationships
    print("\n🔗 Relationship-Based Similarity:")
    relationships = kg.get_relationships("steve_jobs")
    if relationships:
        print(f"  Steve Jobs relationships:")
        for rel in relationships[:3]:
            print(f"    --{rel['predicate']}--> {rel['object']}")

    return similarity_analyzer

# Run embedding similarity exercise
similarity_analyzer = exercise_embedding_similarity()
```

---

## **6. Graph Neural Networks**

### **6.1 Simple GCN Implementation**

```python
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class SimpleGCN(torch.nn.Module):
    """Simple Graph Convolutional Network for node classification"""

    def __init__(self, num_features, num_classes, hidden_dim=64):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class RelationalGCN(torch.nn.Module):
    """Relational GCN for knowledge graphs"""

    def __init__(self, num_nodes, num_relations, hidden_dim=64, num_layers=2):
        super(RelationalGCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim

        # Node embeddings
        self.node_embeddings = torch.nn.Embedding(num_nodes, hidden_dim)

        # Relation-specific transformations
        self.relation_weights = torch.nn.Parameter(
            torch.randn(num_relations, hidden_dim, hidden_dim)
        )
        self.relation_bias = torch.nn.Parameter(torch.zeros(num_relations, hidden_dim))

        # Output layer
        self.output_layer = torch.nn.Linear(hidden_dim, 32)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, node_features, edge_index, edge_type):
        h = self.node_embeddings.weight

        # Apply relation-specific transformations
        for relation_type in range(self.num_relations):
            # Get edges of this relation type
            relation_mask = edge_type == relation_type
            relation_edges = edge_index[:, relation_mask]

            if relation_edges.size(1) > 0:  # If there are edges of this type
                # Apply transformation
                W = self.relation_weights[relation_type]
                b = self.relation_bias[relation_type]

                # Aggregate neighbor information
                neighbor_agg = self.aggregate_neighbors(h, relation_edges)

                # Apply relation-specific transformation
                transformed = torch.matmul(neighbor_agg, W) + b

                # Update node representations
                h = h + transformed

        # Final transformation
        h = torch.relu(h)
        h = self.dropout(h)
        output = self.output_layer(h)

        return output

    def aggregate_neighbors(self, node_features, edges):
        """Aggregate information from neighbors"""
        # Simple sum aggregation
        aggregated = torch.zeros_like(node_features)

        source_nodes = edges[0]
        target_nodes = edges[1]

        for i in range(edges.size(1)):
            src = source_nodes[i]
            tgt = target_nodes[i]
            aggregated[tgt] += node_features[src]

        return aggregated

class KnowledgeGraphGCN:
    """Helper class to convert knowledge graph to PyTorch Geometric format"""

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def to_pytorch_geometric(self):
        """Convert knowledge graph to PyTorch Geometric format"""
        import torch

        # Get node mapping
        nodes = list(self.kg.graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # Create edge list with types
        edge_list = []
        edge_types = []

        for edge in self.kg.graph.edges(data=True):
            source, target, data = edge
            edge_list.append([node_to_idx[source], node_to_idx[target]])
            edge_types.append(0)  # Default edge type

        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

        # Create node features (one-hot for simplicity)
        num_nodes = len(nodes)
        num_features = min(64, num_nodes)  # Feature dimension
        x = torch.eye(num_nodes)[:, :num_features]  # One-hot features

        # Create masks for train/val/test split
        num_train = int(0.6 * num_nodes)
        num_val = int(0.2 * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[:num_train] = True
        val_mask[num_train:num_train + num_val] = True
        test_mask[num_train + num_val:] = True

        return {
            'x': x,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'num_nodes': num_nodes,
            'num_features': num_features,
            'node_mapping': node_to_idx
        }

# Exercise 6.1: GCN training
def exercise_gcn():
    """Exercise: Train GCN on knowledge graph"""

    # Convert KG to PyTorch Geometric format
    kg_to_pyg = KnowledgeGraphGCN(kg)
    pyg_data = kg_to_pyg.to_pytorch_geometric()

    print(f"📊 PyTorch Geometric Data:")
    print(f"  Nodes: {pyg_data['num_nodes']}")
    print(f"  Features: {pyg_data['num_features']}")
    print(f"  Edges: {pyg_data['edge_index'].size(1)}")

    # Create node labels (for demonstration, use node degree as pseudo-label)
    degrees = dict(kg.graph.degree())
    unique_degrees = sorted(set(degrees.values()))
    label_mapping = {degree: i for i, degree in enumerate(unique_degrees)}

    # Create node labels tensor
    node_labels = torch.zeros(pyg_data['num_nodes'], dtype=torch.long)
    for i, node in enumerate(pyg_data['node_mapping'].keys()):
        degree = degrees[node]
        node_labels[i] = label_mapping[degree]

    num_classes = len(unique_degrees)

    print(f"  Node labels: {num_classes} classes based on degree")
    print(f"  Class distribution: {dict(zip(unique_degrees, [sum(1 for d in degrees.values() if d == deg) for deg in unique_degrees]))}")

    # Create model
    model = SimpleGCN(
        num_features=pyg_data['num_features'],
        num_classes=num_classes,
        hidden_dim=32
    )

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    print("\n🚀 Training GCN:")

    # Training loop
    model.train()
    num_epochs = 50

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        out = model(pyg_data['x'], pyg_data['edge_index'])

        # Calculate loss on training nodes
        loss = criterion(out[pyg_data['train_mask']], node_labels[pyg_data['train_mask']])

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print progress
        if epoch % 10 == 0:
            # Calculate accuracy
            model.eval()
            with torch.no_grad():
                out = model(pyg_data['x'], pyg_data['edge_index'])
                pred = out.argmax(dim=1)

                train_acc = (pred[pyg_data['train_mask']] == node_labels[pyg_data['train_mask']]).float().mean()
                val_acc = (pred[pyg_data['val_mask']] == node_labels[pyg_data['val_mask']]).float().mean()

                print(f"  Epoch {epoch:03d}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            model.train()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        out = model(pyg_data['x'], pyg_data['edge_index'])
        pred = out.argmax(dim=1)

        train_acc = (pred[pyg_data['train_mask']] == node_labels[pyg_data['train_mask']]).float().mean()
        val_acc = (pred[pyg_data['val_mask']] == node_labels[pyg_data['val_mask']]).float().mean()
        test_acc = (pred[pyg_data['test_mask']] == node_labels[pyg_data['test_mask']]).float().mean()

        print(f"\n📈 Final Results:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")

    return model, pyg_data, node_labels

# Run GCN exercise
gcn_model, pyg_data, node_labels = exercise_gcn()
```

### **6.2 Node Classification and Link Prediction**

```python
class GCNNodeClassifier:
    """Node classification using GCN"""

    def __init__(self, model, pyg_data):
        self.model = model
        self.data = pyg_data
        self.node_mapping = pyg_data['node_mapping']
        self.idx_to_node = {v: k for k, v in self.node_mapping.items()}

    def predict_node_class(self, node_name):
        """Predict class for a specific node"""
        if node_name not in self.node_mapping:
            return None, None

        node_idx = self.node_mapping[node_name]

        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data['x'], self.data['edge_index'])
            probabilities = torch.softmax(out[node_idx], dim=0)
            predicted_class = out[node_idx].argmax().item()
            confidence = probabilities[predicted_class].item()

        return predicted_class, confidence

    def get_node_embedding(self, node_name):
        """Get embedding for a specific node"""
        if node_name not in self.node_mapping:
            return None

        node_idx = self.node_mapping[node_name]

        self.model.eval()
        with torch.no_grad():
            # Get node representation before final classification layer
            x = self.data['x']
            edge_index = self.data['edge_index']

            # Forward pass through GCN layers
            h = self.model.conv1(x, edge_index)
            h = torch.relu(h)
            embedding = h[node_idx].numpy()

        return embedding

    def classify_nodes_by_type(self):
        """Classify nodes and analyze by entity type"""
        results = {}

        for node_name, node_idx in self.node_mapping.items():
            entity_info = kg.get_entity(node_name)
            if entity_info:
                entity_type = entity_info['type']
                predicted_class, confidence = self.predict_node_class(node_name)

                if entity_type not in results:
                    results[entity_type] = []

                results[entity_type].append({
                    'name': node_name,
                    'properties': entity_info['properties'],
                    'predicted_class': predicted_class,
                    'confidence': confidence
                })

        return results

class GCNLinkPredictor:
    """Link prediction using GCN embeddings"""

    def __init__(self, model, pyg_data):
        self.model = model
        self.data = pyg_data
        self.node_mapping = pyg_data['node_mapping']
        self.idx_to_node = {v: k for k, v in self.node_mapping.items()}

    def get_link_prediction_score(self, node1, node2):
        """Predict if a link exists between two nodes"""
        if node1 not in self.node_mapping or node2 not in self.node_mapping:
            return 0.0

        node1_idx = self.node_mapping[node1]
        node2_idx = self.node_mapping[node2]

        # Get node embeddings
        emb1 = self.get_node_embedding(node1)
        emb2 = self.get_node_embedding(node2)

        if emb1 is None or emb2 is None:
            return 0.0

        # Calculate similarity (cosine)
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity

    def predict_missing_links(self, top_k=5):
        """Predict most likely missing links"""
        predictions = []

        # Get existing edges
        existing_edges = set()
        for edge in self.data['edge_index'].t().tolist():
            existing_edges.add((edge[0], edge[1]))

        # Check all possible non-edges
        for i in range(self.data['num_nodes']):
            for j in range(self.data['num_nodes']):
                if i != j and (i, j) not in existing_edges:
                    node1 = self.idx_to_node[i]
                    node2 = self.idx_to_node[j]

                    score = self.get_link_prediction_score(node1, node2)

                    predictions.append({
                        'node1': node1,
                        'node2': node2,
                        'score': score
                    })

        # Sort by score and return top-k
        predictions.sort(key=lambda x: x['score'], reverse=True)
        return predictions[:top_k]

    def get_node_embedding(self, node_name):
        """Get embedding for a specific node"""
        if node_name not in self.node_mapping:
            return None

        node_idx = self.node_mapping[node_name]

        self.model.eval()
        with torch.no_grad():
            # Forward pass to get embeddings
            h = self.model.conv1(self.data['x'], self.data['edge_index'])
            h = torch.relu(h)
            embedding = h[node_idx].numpy()

        return embedding

# Exercise 6.2: Node classification and link prediction
def exercise_node_classification_link_prediction():
    """Exercise: Node classification and link prediction with GCN"""

    # Node classification
    print("🎯 Node Classification:")
    classifier = GCNNodeClassifier(gcn_model, pyg_data)
    classification_results = classifier.classify_nodes_by_type()

    for entity_type, nodes in classification_results.items():
        print(f"\n  {entity_type} nodes ({len(nodes)} total):")
        for node_data in nodes[:3]:  # Show first 3
            name = node_data['properties'].get('name', node_data['name'])
            print(f"    {name}: class {node_data['predicted_class']} "
                  f"(confidence: {node_data['confidence']:.3f})")

    # Link prediction
    print("\n🔗 Link Prediction:")
    link_predictor = GCNLinkPredictor(gcn_model, pyg_data)

    # Predict missing links
    missing_links = link_predictor.predict_missing_links(top_k=5)

    print("  Top predicted missing links:")
    for link in missing_links:
        node1_info = kg.get_entity(link['node1'])
        node2_info = kg.get_entity(link['node2'])

        name1 = node1_info['properties'].get('name', link['node1']) if node1_info else link['node1']
        name2 = node2_info['properties'].get('name', link['node2']) if node2_info else link['node2']

        print(f"    {name1} ↔ {name2} (score: {link['score']:.3f})")

    # Test existing links
    print("\n✅ Existing link validation:")
    existing_edges = list(kg.graph.edges())

    for i, (node1, node2) in enumerate(existing_edges[:3]):
        score = link_predictor.get_link_prediction_score(node1, node2)
        node1_info = kg.get_entity(node1)
        node2_info = kg.get_entity(node2)

        name1 = node1_info['properties'].get('name', node1) if node1_info else node1
        name2 = node2_info['properties'].get('name', node2) if node2_info else node2

        print(f"    {name1} ↔ {name2}: score {score:.3f}")

    return classifier, link_predictor

# Run node classification and link prediction exercise
classifier, link_predictor = exercise_node_classification_link_prediction()
```

---

## **7. RAG Enhancement with Knowledge Graphs**

### **7.1 Graph-Enhanced RAG Implementation**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class KnowledgeGraphRAG:
    """Graph-Enhanced Retrieval-Augmented Generation"""

    def __init__(self, knowledge_graph, embedding_model_name='all-MiniLM-L6-v2'):
        self.kg = knowledge_graph
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.entity_cache = {}
        self.similarity_threshold = 0.7

    def extract_entities_from_text(self, text):
        """Extract entities from query text"""
        extractor = EntityExtractor()
        entities = extractor.extract_entities(text)

        # Filter for relevant entity types
        relevant_types = {'PERSON', 'ORG', 'GPE', 'WORK_OF_ART', 'EVENT'}
        filtered_entities = [e for e in entities if e['label'] in relevant_types]

        return filtered_entities

    def find_entities_in_kg(self, text_entities):
        """Find matching entities in knowledge graph"""
        matched_entities = []

        for text_entity in text_entities:
            entity_text = text_entity['text']

            # Find best match in KG
            best_match = None
            best_score = 0

            for kg_entity in self.kg.graph.nodes():
                kg_entity_info = self.kg.get_entity(kg_entity)
                if kg_entity_info:
                    kg_name = kg_entity_info['properties'].get('name', '')

                    # Calculate similarity
                    similarity = self.calculate_text_similarity(entity_text, kg_name)

                    if similarity > best_score and similarity > self.similarity_threshold:
                        best_score = similarity
                        best_match = kg_entity

            if best_match:
                matched_entities.append({
                    'text_entity': text_entity,
                    'kg_entity': best_match,
                    'confidence': best_score
                })

        return matched_entities

    def calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0

        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return similarity
        except:
            return 0.0

    def retrieve_graph_context(self, matched_entities, max_hops=2):
        """Retrieve context from knowledge graph"""
        context_parts = []

        for match in matched_entities:
            entity = match['kg_entity']

            # Get entity information
            entity_info = self.kg.get_entity(entity)
            if entity_info:
                entity_text = f"{entity_info['properties'].get('name', entity)} (type: {entity_info['type']})"
                context_parts.append(entity_text)

            # Get relationships (multi-hop)
            relationships = self.kg.get_relationships(entity)

            for hop in range(1, max_hops + 1):
                current_entities = [entity]
                hop_relationships = []

                for _ in range(hop):
                    next_entities = []
                    for curr_entity in current_entities:
                        rels = self.kg.get_relationships(curr_entity)
                        hop_relationships.extend(rels)
                        for rel in rels:
                            if rel['object'] not in [e for e in current_entities]:
                                next_entities.append(rel['object'])
                    current_entities = next_entities

                # Format hop relationships
                if hop_relationships:
                    hop_text = f"Related to {entity_info['properties'].get('name', entity)}:"
                    for rel in hop_relationships[:5]:  # Limit to top 5
                        rel_entity = self.kg.get_entity(rel['object'])
                        if rel_entity:
                            obj_name = rel_entity['properties'].get('name', rel['object'])
                            hop_text += f" {rel['predicate']} {obj_name};"

                    context_parts.append(hop_text)

        return "\n".join(context_parts)

    def retrieve_documents(self, query, k=5):
        """Traditional document retrieval (simplified)"""
        # This is a simplified version - in practice, you'd use vector databases
        # For demonstration, we'll return some placeholder text

        return [
            "Steve Jobs was the co-founder and former CEO of Apple Inc.",
            "Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976.",
            "Steve Jobs served as CEO of Apple from 1977 to 2011.",
            "Bill Gates co-founded Microsoft Corporation in 1975.",
            "Microsoft and Apple were major competitors in the personal computer market."
        ][:k]

    def combine_contexts(self, doc_context, graph_context):
        """Combine document and graph contexts"""
        combined = f"Document Context:\n{doc_context}\n\nGraph Context:\n{graph_context}"
        return combined

    def query(self, question, k=5):
        """Process query with graph-enhanced RAG"""

        # Step 1: Extract entities from question
        text_entities = self.extract_entities_from_text(question)

        # Step 2: Match entities in knowledge graph
        matched_entities = self.find_entities_in_kg(text_entities)

        # Step 3: Retrieve graph context
        graph_context = self.retrieve_graph_context(matched_entities)

        # Step 4: Traditional document retrieval
        doc_context = self.retrieve_documents(question, k=k)

        # Step 5: Combine contexts
        combined_context = self.combine_contexts("\n".join(doc_context), graph_context)

        return {
            'question': question,
            'extracted_entities': text_entities,
            'matched_entities': matched_entities,
            'document_context': doc_context,
            'graph_context': graph_context,
            'combined_context': combined_context
        }

# Exercise 7.1: Graph-enhanced RAG
def exercise_graph_enhanced_rag():
    """Exercise: Implement graph-enhanced RAG"""

    # Create RAG system
    rag_system = KnowledgeGraphRAG(kg)

    # Test queries
    test_queries = [
        "Who co-founded Apple Inc?",
        "Tell me about Steve Jobs and his companies",
        "What is the relationship between Steve Jobs and Steve Wozniak?",
        "Who founded Microsoft?",
        "What companies did technology leaders start?"
    ]

    print("🤖 Graph-Enhanced RAG System:")

    for query in test_queries:
        print(f"\n❓ Query: {query}")

        result = rag_system.query(query)

        print(f"  📝 Extracted entities: {len(result['extracted_entities'])}")
        for entity in result['extracted_entities']:
            print(f"    {entity['text']} ({entity['label']})")

        print(f"  🎯 Matched KG entities: {len(result['matched_entities'])}")
        for match in result['matched_entities']:
            kg_entity = kg.get_entity(match['kg_entity'])
            kg_name = kg_entity['properties'].get('name', match['kg_entity']) if kg_entity else match['kg_entity']
            print(f"    {match['text_entity']['text']} → {kg_name} (confidence: {match['confidence']:.2f})")

        print(f"  📖 Combined context preview:")
        print(f"    {result['combined_context'][:200]}...")

    return rag_system

# Run graph-enhanced RAG exercise
rag_system = exercise_graph_enhanced_rag()
```

### **7.2 Question Answering with Knowledge Graphs**

```python
class KnowledgeGraphQA:
    """Question Answering using Knowledge Graphs"""

    def __init__(self, knowledge_graph, rag_system=None):
        self.kg = knowledge_graph
        self.rag_system = rag_system

        # Question patterns for different types
        self.question_patterns = {
            'factoid': [
                (r'who is (\w+)', 'entity_info'),
                (r'what is (\w+)', 'entity_info'),
                (r'who founded (\w+)', 'founder_query'),
                (r'what company did (\w+) found', 'founded_company_query')
            ],
            'relationship': [
                (r'what is the relationship between (.+) and (.+)', 'relationship_query'),
                (r'how are (.+) and (.+) connected', 'path_query'),
                (r'who is related to (.+)', 'related_entities_query')
            ],
            'count': [
                (r'how many (\w+)', 'count_query'),
                (r'how many relationships does (\w+) have', 'degree_query')
            ]
        }

    def classify_question(self, question):
        """Classify question type"""
        import re

        for category, patterns in self.question_patterns.items():
            for pattern, query_type in patterns:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    return category, query_type, match.groups()

        return 'unknown', 'unknown', ()

    def answer_factoid_question(self, entity_name, query_type):
        """Answer factoid questions"""
        # Find entity in KG
        entity_id = self.find_entity_in_kg(entity_name)

        if not entity_id:
            return f"I couldn't find {entity_name} in the knowledge graph."

        entity_info = self.kg.get_entity(entity_id)

        if query_type == 'entity_info':
            return self.format_entity_info(entity_info)
        elif query_type == 'founder_query':
            return self.find_founders(entity_id)
        elif query_type == 'founded_company_query':
            return self.find_founded_companies(entity_id)
        else:
            return "I'm not sure how to answer this type of question."

    def answer_relationship_question(self, entity1, entity2, query_type):
        """Answer relationship questions"""
        # Find entities in KG
        entity1_id = self.find_entity_in_kg(entity1)
        entity2_id = self.find_entity_in_kg(entity2)

        if not entity1_id or not entity2_id:
            return f"I couldn't find both entities in the knowledge graph."

        if query_type == 'relationship_query':
            return self.find_direct_relationship(entity1_id, entity2_id)
        elif query_type == 'path_query':
            return self.find_path_between_entities(entity1_id, entity2_id)
        elif query_type == 'related_entities_query':
            return self.find_related_entities(entity1_id)
        else:
            return "I'm not sure how to answer this relationship question."

    def answer_count_question(self, entity_name, query_type):
        """Answer count questions"""
        entity_id = self.find_entity_in_kg(entity_name)

        if not entity_id:
            return f"I couldn't find {entity_name} in the knowledge graph."

        if query_type == 'count_query':
            return f"Found {len(self.kg.graph.nodes)} entities total."
        elif query_type == 'degree_query':
            degree = self.kg.graph.degree(entity_id)
            return f"{entity_name} has {degree} relationships."
        else:
            return "I'm not sure how to answer this count question."

    def find_entity_in_kg(self, entity_name):
        """Find entity ID by name"""
        # Simple exact match
        for entity_id in self.kg.graph.nodes:
            entity_info = self.kg.get_entity(entity_id)
            if entity_info:
                name = entity_info['properties'].get('name', '')
                if name.lower() == entity_name.lower():
                    return entity_id

        # Fuzzy match
        best_match = None
        best_score = 0

        for entity_id in self.kg.graph.nodes:
            entity_info = self.kg.get_entity(entity_id)
            if entity_info:
                name = entity_info['properties'].get('name', '')
                # Simple similarity (you could use embeddings for better results)
                score = self.simple_similarity(entity_name, name)
                if score > best_score and score > 0.8:
                    best_score = score
                    best_match = entity_id

        return best_match

    def simple_similarity(self, text1, text2):
        """Simple string similarity"""
        if not text1 or not text2:
            return 0.0

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        if text1_lower == text2_lower:
            return 1.0

        # Check if one is substring of other
        if text1_lower in text2_lower or text2_lower in text1_lower:
            return 0.8

        return 0.0

    def format_entity_info(self, entity_info):
        """Format entity information for display"""
        if not entity_info:
            return "Entity not found."

        result = f"{entity_info['properties'].get('name', entity_info['id'])} "
        result += f"(type: {entity_info['type']})\n"

        # Add properties
        for prop_name, prop_value in entity_info['properties'].items():
            if prop_name != 'name':
                result += f"  {prop_name}: {prop_value}\n"

        return result

    def find_direct_relationship(self, entity1_id, entity2_id):
        """Find direct relationship between two entities"""
        relationships = self.kg.get_relationships(entity1_id)

        for rel in relationships:
            if rel['object'] == entity2_id:
                return f"They are connected through: {entity1_id} --{rel['predicate']}--> {entity2_id}"

        # Check reverse direction
        relationships = self.kg.get_relationships(entity2_id)
        for rel in relationships:
            if rel['object'] == entity1_id:
                return f"They are connected through: {entity2_id} --{rel['predicate']}--> {entity1_id}"

        return "No direct relationship found."

    def find_path_between_entities(self, entity1_id, entity2_id):
        """Find path between two entities"""
        path = self.kg.find_path(entity1_id, entity2_id)

        if path:
            path_description = " → ".join(path)
            return f"Connection path: {path_description}"
        else:
            return "No connection path found."

    def find_related_entities(self, entity_id):
        """Find entities related to given entity"""
        relationships = self.kg.get_relationships(entity_id)

        if not relationships:
            return f"No related entities found for {entity_id}."

        related = []
        for rel in relationships:
            related.append(f"{rel['predicate']} {rel['object']}")

        return f"Related entities: {', '.join(related)}"

    def find_founders(self, company_id):
        """Find founders of a company"""
        relationships = self.kg.get_relationships(company_id)

        founders = []
        for rel in relationships:
            if rel['predicate'] in ['co_founded', 'founded']:
                founders.append(rel['object'])

        if founders:
            return f"Founded by: {', '.join(founders)}"
        else:
            return "No founder information found."

    def find_founded_companies(self, person_id):
        """Find companies founded by a person"""
        relationships = self.kg.get_relationships(person_id)

        companies = []
        for rel in relationships:
            if rel['predicate'] in ['co_founded', 'founded']:
                companies.append(rel['object'])

        if companies:
            return f"Founded companies: {', '.join(companies)}"
        else:
            return "No founded companies found."

    def answer_question(self, question):
        """Main question answering function"""
        question_type, query_type, entities = self.classify_question(question)

        if question_type == 'factoid':
            if entities:
                return self.answer_factoid_question(entities[0], query_type)
        elif question_type == 'relationship':
            if len(entities) >= 2:
                return self.answer_relationship_question(entities[0], entities[1], query_type)
        elif question_type == 'count':
            if entities:
                return self.answer_count_question(entities[0], query_type)
        elif question_type == 'unknown':
            # Try to use RAG system if available
            if self.rag_system:
                rag_result = self.rag_system.query(question)
                return f"Based on available information:\n{rag_result['combined_context']}"
            else:
                return "I couldn't understand this question. Could you rephrase it?"

        return "I'm not sure how to answer this question."

# Exercise 7.2: Question answering
def exercise_knowledge_graph_qa():
    """Exercise: Question answering with knowledge graphs"""

    # Create QA system
    qa_system = KnowledgeGraphQA(kg, rag_system)

    # Test questions
    test_questions = [
        "Who is Steve Jobs?",
        "Who founded Apple Inc?",
        "What is the relationship between Steve Jobs and Steve Wozniak?",
        "How many relationships does Apple Inc have?",
        "What companies did tech leaders start?",
        "Tell me about Bill Gates"
    ]

    print("❓ Knowledge Graph Question Answering:")

    for question in test_questions:
        print(f"\n🔍 Question: {question}")
        answer = qa_system.answer_question(question)
        print(f"💬 Answer: {answer}")

    return qa_system

# Run QA exercise
qa_system = exercise_knowledge_graph_qa()
```

---

## **8. Bias Detection and Fairness**

### **8.1 Bias Detection in Knowledge Graphs**

```python
class KnowledgeGraphBiasDetector:
    """Detect bias in knowledge graphs"""

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.demographic_attributes = {
            'gender': ['male', 'female', 'man', 'woman', 'he', 'she'],
            'race': ['white', 'black', 'asian', 'hispanic', 'latino', 'caucasian'],
            'nationality': ['american', 'british', 'canadian', 'chinese', 'indian'],
            'religion': ['christian', 'muslim', 'jewish', 'hindu', 'buddhist'],
            'occupation': ['ceo', 'engineer', 'doctor', 'teacher', 'scientist']
        }

    def extract_demographic_info(self, entity_id):
        """Extract demographic information from entity"""
        entity_info = self.kg.get_entity(entity_id)
        if not entity_info:
            return {}

        properties = entity_info['properties']
        demographics = {}

        # Check various properties for demographic information
        for attr_type, keywords in self.demographic_attributes.items():
            for prop_name, prop_value in properties.items():
                prop_str = str(prop_value).lower()
                for keyword in keywords:
                    if keyword in prop_str:
                        if attr_type not in demographics:
                            demographics[attr_type] = []
                        demographics[attr_type].append({
                            'property': prop_name,
                            'value': prop_value,
                            'keyword': keyword
                        })

        return demographics

    def analyze_representation(self):
        """Analyze demographic representation in knowledge graph"""
        analysis = {
            'total_entities': len(self.kg.graph.nodes),
            'demographic_distribution': {},
            'bias_metrics': {},
            'recommendations': []
        }

        # Extract demographic info for all entities
        demographic_entities = {}
        for entity_id in self.kg.graph.nodes:
            demo_info = self.extract_demographic_info(entity_id)
            if demo_info:
                for attr_type, attrs in demo_info.items():
                    if attr_type not in demographic_entities:
                        demographic_entities[attr_type] = []
                    demographic_entities[attr_type].append({
                        'entity': entity_id,
                        'demographics': attrs
                    })

        # Calculate distribution
        for attr_type, entities in demographic_entities.items():
            distribution = {}
            for entity_data in entities:
                for attr in entity_data['demographics']:
                    value = attr['value'].lower()
                    if value not in distribution:
                        distribution[value] = 0
                    distribution[value] += 1

            analysis['demographic_distribution'][attr_type] = distribution

            # Calculate bias metrics
            total_count = sum(distribution.values())
            if total_count > 0:
                max_count = max(distribution.values())
                min_count = min(distribution.values())
                diversity_score = len(distribution) / total_count if total_count > 0 else 0

                analysis['bias_metrics'][attr_type] = {
                    'total_entities': total_count,
                    'unique_values': len(distribution),
                    'max_representation': max_count,
                    'min_representation': min_count,
                    'representation_ratio': max_count / total_count,
                    'diversity_score': diversity_score
                }

        return analysis

    def detect_stereotypes(self):
        """Detect stereotypical patterns in relationships"""
        stereotypes = []

        # Analyze relationship patterns by entity type
        entity_types = {}
        for node in self.kg.graph.nodes:
            entity_info = self.kg.get_entity(node)
            if entity_info:
                entity_type = entity_info['type']
                if entity_type not in entity_types:
                    entity_types[entity_type] = []
                entity_types[entity_type].append(node)

        # Check for potential stereotypes
        for entity_type, entities in entity_types.items():
            relationships = {}
            for entity in entities:
                entity_rels = self.kg.get_relationships(entity)
                for rel in entity_rels:
                    pred = rel['predicate']
                    if pred not in relationships:
                        relationships[pred] = []
                    relationships[pred].append(rel['object'])

            # Look for skewed relationship distributions
            for pred, objects in relationships.items():
                if len(objects) > 10:  # Only check if significant data
                    obj_types = {}
                    for obj in objects:
                        obj_info = self.kg.get_entity(obj)
                        if obj_info:
                            obj_type = obj_info['type']
                            obj_types[obj_type] = obj_types.get(obj_type, 0) + 1

                    # Check if one type dominates (potential stereotype)
                    total = sum(obj_types.values())
                    max_type = max(obj_types, key=obj_types.get)
                    max_ratio = obj_types[max_type] / total

                    if max_ratio > 0.8:  # 80% threshold
                        stereotypes.append({
                            'subject_type': entity_type,
                            'relationship': pred,
                            'dominant_object_type': max_type,
                            'dominance_ratio': max_ratio,
                            'description': f"{entity_type} entities are predominantly connected to {max_type} entities via {pred}"
                        })

        return stereotypes

    def generate_bias_report(self):
        """Generate comprehensive bias report"""
        print("📊 Knowledge Graph Bias Analysis Report")
        print("=" * 50)

        # Representation analysis
        representation = self.analyze_representation()

        print(f"\n📈 Representation Statistics:")
        print(f"Total entities analyzed: {representation['total_entities']}")

        for attr_type, metrics in representation['bias_metrics'].items():
            print(f"\n{attr_type.title()} Representation:")
            print(f"  Entities with {attr_type} info: {metrics['total_entities']}")
            print(f"  Unique {attr_type} values: {metrics['unique_values']}")
            print(f"  Diversity score: {metrics['diversity_score']:.3f}")
            print(f"  Most represented: {metrics['max_representation'] / metrics['total_entities']:.1%}")
            print(f"  Least represented: {metrics['min_representation'] / metrics['total_entities']:.1%}")

        # Stereotype detection
        stereotypes = self.detect_stereotypes()

        print(f"\n🎭 Stereotype Detection:")
        if stereotypes:
            for i, stereotype in enumerate(stereotypes):
                print(f"  {i+1}. {stereotype['description']}")
                print(f"     Dominance ratio: {stereotype['dominance_ratio']:.1%}")
        else:
            print("  No obvious stereotypes detected.")

        # Recommendations
        print(f"\n💡 Recommendations:")

        if representation['bias_metrics']:
            low_diversity_attrs = [
                attr for attr, metrics in representation['bias_metrics'].items()
                if metrics['diversity_score'] < 0.5
            ]

            if low_diversity_attrs:
                print(f"  • Improve diversity in: {', '.join(low_diversity_attrs)}")

            high_dominance_attrs = [
                attr for attr, metrics in representation['bias_metrics'].items()
                if metrics['representation_ratio'] > 0.7
            ]

            if high_dominance_attrs:
                print(f"  • Address representation imbalance in: {', '.join(high_dominance_attrs)}")

        if stereotypes:
            print(f"  • Review relationship patterns to reduce stereotypical associations")

        print(f"  • Consider adding diverse entity types and relationships")
        print(f"  • Implement bias detection in knowledge graph updates")

        return {
            'representation': representation,
            'stereotypes': stereotypes
        }

# Exercise 8.1: Bias detection
def exercise_bias_detection():
    """Exercise: Detect bias in knowledge graph"""

    # Add some entities with demographic info to demonstrate bias detection
    demo_entities = [
        ("grace_hopper", "Person", {"name": "Grace Hopper", "gender": "female", "occupation": "Computer Scientist", "nationality": "american"}),
        ("ada_lovelace", "Person", {"name": "Ada Lovelace", "gender": "female", "occupation": "Mathematician", "nationality": "british"}),
        ("alan_turing", "Person", {"name": "Alan Turing", "gender": "male", "occupation": "Computer Scientist", "nationality": "british"})
    ]

    # Add to knowledge graph
    for entity_id, entity_type, properties in demo_entities:
        kg.add_entity(entity_id, entity_type, properties)

    # Add relationships
    demo_relationships = [
        ("grace_hopper", "worked_in", "computer_science", {"field": "programming"}),
        ("ada_lovelace", "worked_in", "mathematics", {"field": "algorithmic computing"}),
        ("alan_turing", "worked_in", "computer_science", {"field": "artificial intelligence"}),
        ("grace_hopper", "was_a", "pioneer", {"contribution": "COBOL language"}),
        ("alan_turing", "was_a", "pioneer", {"contribution": "AI theory"})
    ]

    for subject, predicate, obj, props in demo_relationships:
        kg.add_relationship(subject, predicate, obj, props)

    # Run bias detection
    bias_detector = KnowledgeGraphBiasDetector(kg)
    bias_report = bias_detector.generate_bias_report()

    return bias_detector, bias_report

# Run bias detection exercise
bias_detector, bias_report = exercise_bias_detection()
```

---

## **9. Advanced Applications**

### **9.1 Knowledge Graph Completion**

```python
class KnowledgeGraphCompleter:
    """Complete missing links in knowledge graph"""

    def __init__(self, knowledge_graph, embedding_model):
        self.kg = knowledge_graph
        self.embedding_model = embedding_model
        self.completion_threshold = 0.7

    def predict_missing_links(self, subject, relation, top_k=10):
        """Predict missing links for subject-relation pair"""
        # This is a simplified implementation
        # In practice, you would use trained embedding models

        # Get all entities
        all_entities = list(self.kg.graph.nodes)

        # Remove subject itself
        candidate_objects = [e for e in all_entities if e != subject]

        # Score each candidate
        scored_objects = []
        for obj in candidate_objects:
            # Check if relationship already exists
            existing_rels = self.kg.get_relationships(subject)
            already_connected = any(rel['object'] == obj and rel['predicate'] == relation
                                  for rel in existing_rels)

            if not already_connected:
                # Simple scoring based on entity type compatibility
                score = self.score_entity_compatibility(subject, relation, obj)
                scored_objects.append((obj, score))

        # Sort by score
        scored_objects.sort(key=lambda x: x[1], reverse=True)
        return scored_objects[:top_k]

    def score_entity_compatibility(self, subject, relation, obj):
        """Simple scoring for entity compatibility"""
        # This is a basic implementation - use embeddings for better results

        subject_info = self.kg.get_entity(subject)
        obj_info = self.kg.get_entity(obj)

        if not subject_info or not obj_info:
            return 0.0

        score = 0.0

        # Type-based scoring
        if relation == 'co_founded':
            if subject_info['type'] == 'Person' and obj_info['type'] == 'Person':
                score += 0.8
            elif subject_info['type'] == 'Person' and obj_info['type'] == 'Company':
                score += 0.6

        elif relation == 'worked_at':
            if subject_info['type'] == 'Person' and obj_info['type'] == 'Company':
                score += 0.9
            elif subject_info['type'] == 'Person' and obj_info['type'] == 'Organization':
                score += 0.8

        elif relation == 'related_to':
            # General relationship - lower score
            score += 0.3

        # Add small random component for diversity
        score += np.random.normal(0, 0.1)
        score = max(0, score)  # Ensure non-negative

        return score

    def complete_entity_neighborhood(self, entity_id, max_predictions=20):
        """Complete neighborhood for a specific entity"""
        predictions = []

        # Get existing relationships
        existing_rels = self.kg.get_relationships(entity_id)
        existing_objects = {rel['object'] for rel in existing_rels}

        # Define possible relationship types
        possible_relations = ['related_to', 'worked_with', 'collaborated_with', 'mentored', 'influenced']

        for relation in possible_relations:
            pred_links = self.predict_missing_links(entity_id, relation, top_k=5)

            for obj, score in pred_links:
                if score > self.completion_threshold:
                    predictions.append({
                        'subject': entity_id,
                        'relation': relation,
                        'object': obj,
                        'confidence': score
                    })

        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions[:max_predictions]

    def validate_predictions(self, predictions, validation_method='consistency'):
        """Validate predictions using various methods"""
        validated_predictions = []

        for pred in predictions:
            is_valid = False
            validation_score = 0.0

            if validation_method == 'consistency':
                # Check logical consistency
                is_valid = self.check_logical_consistency(pred)
                validation_score = 1.0 if is_valid else 0.0

            elif validation_method == 'confidence':
                # Use confidence threshold
                is_valid = pred['confidence'] > self.completion_threshold
                validation_score = pred['confidence']

            elif validation_method == 'graph_metrics':
                # Check graph-level metrics
                is_valid, metrics = self.check_graph_metrics(pred)
                validation_score = metrics['consistency_score']

            pred['validated'] = is_valid
            pred['validation_score'] = validation_score

            if is_valid:
                validated_predictions.append(pred)

        return validated_predictions

    def check_logical_consistency(self, prediction):
        """Check if prediction is logically consistent"""
        subject_info = self.kg.get_entity(prediction['subject'])
        obj_info = self.kg.get_entity(prediction['object'])

        if not subject_info or not obj_info:
            return False

        # Simple consistency checks
        relation = prediction['relation']

        if relation == 'co_founded':
            return (subject_info['type'] == 'Person' and obj_info['type'] == 'Person')

        elif relation == 'worked_at':
            return subject_info['type'] == 'Person' and obj_info['type'] in ['Company', 'Organization']

        elif relation == 'founded':
            return subject_info['type'] == 'Person' and obj_info['type'] == 'Company'

        # Default: allow the relationship
        return True

    def check_graph_metrics(self, prediction):
        """Check graph-level metrics for prediction validity"""
        # This is a simplified metric calculation
        metrics = {
            'local_consistency': 1.0,
            'global_consistency': 0.8,
            'consistency_score': 0.9
        }

        is_valid = metrics['consistency_score'] > 0.5
        return is_valid, metrics

# Exercise 9.1: Knowledge graph completion
def exercise_kg_completion():
    """Exercise: Complete missing links in knowledge graph"""

    # Create completion system
    completer = KnowledgeGraphCompleter(kg, transE_model)

    print("🔮 Knowledge Graph Completion:")

    # Complete neighborhood for specific entities
    target_entities = ['steve_jobs', 'apple_inc']

    for entity in target_entities:
        print(f"\n🎯 Completing neighborhood for {entity}:")

        predictions = completer.complete_entity_neighborhood(entity, max_predictions=10)

        if predictions:
            print(f"  Found {len(predictions)} potential links:")
            for i, pred in enumerate(predictions[:5]):
                obj_info = kg.get_entity(pred['object'])
                obj_name = obj_info['properties'].get('name', pred['object']) if obj_info else pred['object']
                print(f"    {i+1}. {pred['subject']} --{pred['relation']}--> {obj_name} "
                      f"(confidence: {pred['confidence']:.2f})")
        else:
            print(f"  No missing links predicted for {entity}")

    # Validate predictions
    print(f"\n✅ Validating predictions:")
    all_predictions = []
    for entity in target_entities:
        predictions = completer.complete_entity_neighborhood(entity, max_predictions=5)
        all_predictions.extend(predictions)

    if all_predictions:
        validated = completer.validate_predictions(all_predictions, validation_method='confidence')

        print(f"  Total predictions: {len(all_predictions)}")
        print(f"  Validated predictions: {len(validated)}")
        print(f"  Validation rate: {len(validated) / len(all_predictions):.1%}")

        if validated:
            print(f"  Top validated predictions:")
            for i, pred in enumerate(validated[:3]):
                obj_info = kg.get_entity(pred['object'])
                obj_name = obj_info['properties'].get('name', pred['object']) if obj_info else pred['object']
                print(f"    {pred['subject']} --{pred['relation']}--> {obj_name} "
                      f"(confidence: {pred['confidence']:.2f})")

    return completer

# Run KG completion exercise
completer = exercise_kg_completion()
```

### **9.2 Recommendation System with Knowledge Graphs**

```python
class KnowledgeGraphRecommender:
    """Recommendation system using knowledge graphs"""

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.user_profiles = {}
        self.similarity_cache = {}

    def build_user_profile(self, user_id, user_interactions):
        """Build user profile from interactions"""
        profile = {
            'liked_entities': [],
            'disliked_entities': [],
            'interaction_types': {},
            'preference_scores': {}
        }

        for interaction in user_interactions:
            entity = interaction['entity']
            interaction_type = interaction['type']  # 'like', 'dislike', 'view', 'purchase'
            score = interaction.get('score', 1.0)

            if interaction_type == 'like':
                profile['liked_entities'].append(entity)
                profile['preference_scores'][entity] = profile['preference_scores'].get(entity, 0) + score

            elif interaction_type == 'dislike':
                profile['disliked_entities'].append(entity)
                profile['preference_scores'][entity] = profile['preference_scores'].get(entity, 0) - score

            profile['interaction_types'][entity] = interaction_type

        self.user_profiles[user_id] = profile
        return profile

    def get_entity_features(self, entity_id):
        """Get features for an entity"""
        entity_info = self.kg.get_entity(entity_id)
        if not entity_info:
            return {}

        features = {
            'type': entity_info['type'],
            'properties': entity_info['properties']
        }

        # Get relationship features
        relationships = self.kg.get_relationships(entity_id)
        features['relationships'] = relationships

        return features

    def calculate_user_entity_similarity(self, user_id, entity_id):
        """Calculate similarity between user and entity"""
        if user_id not in self.user_profiles:
            return 0.0

        # Check cache
        cache_key = f"{user_id}_{entity_id}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        user_profile = self.user_profiles[user_id]
        entity_features = self.get_entity_features(entity_id)

        if not entity_features:
            return 0.0

        similarity_score = 0.0

        # Content-based similarity
        for liked_entity in user_profile['liked_entities']:
            liked_features = self.get_entity_features(liked_entity)
            content_sim = self.calculate_content_similarity(entity_features, liked_features)
            similarity_score += content_sim

        # Relationship-based similarity
        entity_rels = entity_features.get('relationships', [])
        for rel in entity_rels:
            if rel['object'] in user_profile['liked_entities']:
                relationship_bonus = 0.2  # Bonus for shared relationships
                similarity_score += relationship_bonus

        # Normalize by number of liked entities
        if user_profile['liked_entities']:
            similarity_score /= len(user_profile['liked_entities'])

        # Cache result
        self.similarity_cache[cache_key] = similarity_score

        return similarity_score

    def calculate_content_similarity(self, features1, features2):
        """Calculate content similarity between two entities"""
        similarity = 0.0

        # Type similarity
        if features1.get('type') == features2.get('type'):
            similarity += 0.3

        # Property similarity
        props1 = features1.get('properties', {})
        props2 = features2.get('properties', {})

        for prop_name, prop_value1 in props1.items():
            if prop_name in props2:
                prop_value2 = props2[prop_name]
                if prop_value1 == prop_value2:
                    similarity += 0.1

        return similarity

    def recommend_entities(self, user_id, entity_type=None, top_k=10):
        """Recommend entities to user"""
        if user_id not in self.user_profiles:
            return []

        user_profile = self.user_profiles[user_id]
        candidate_entities = []

        # Get all entities
        for entity_id in self.kg.graph.nodes:
            # Skip if already interacted with
            if entity_id in user_profile['preference_scores']:
                continue

            # Filter by type if specified
            if entity_type:
                entity_info = self.kg.get_entity(entity_id)
                if not entity_info or entity_info['type'] != entity_type:
                    continue

            # Calculate similarity score
            similarity_score = self.calculate_user_entity_similarity(user_id, entity_id)

            if similarity_score > 0:  # Only consider entities with positive similarity
                candidate_entities.append((entity_id, similarity_score))

        # Sort by similarity and return top-k
        candidate_entities.sort(key=lambda x: x[1], reverse=True)
        return candidate_entities[:top_k]

    def explain_recommendation(self, user_id, recommended_entity):
        """Generate explanation for recommendation"""
        user_profile = self.user_profiles[user_id]
        entity_features = self.get_entity_features(recommended_entity)

        explanation_parts = []

        # Find similar liked entities
        similar_entities = []
        for liked_entity in user_profile['liked_entities']:
            liked_features = self.get_entity_features(liked_entity)
            similarity = self.calculate_content_similarity(entity_features, liked_features)
            if similarity > 0:
                similar_entities.append((liked_entity, similarity))

        if similar_entities:
            similar_entities.sort(key=lambda x: x[1], reverse=True)
            top_similar = similar_entities[0]
            similar_info = self.kg.get_entity(top_similar[0])
            similar_name = similar_info['properties'].get('name', top_similar[0]) if similar_info else top_similar[0]
            explanation_parts.append(f"Similar to '{similar_name}' that you liked")

        # Check shared relationships
        entity_rels = entity_features.get('relationships', [])
        shared_rels = []
        for rel in entity_rels:
            if rel['object'] in user_profile['liked_entities']:
                shared_rels.append(rel)

        if shared_rels:
            rel_info = self.kg.get_entity(shared_rels[0]['object'])
            rel_name = rel_info['properties'].get('name', shared_rels[0]['object']) if rel_info else shared_rels[0]['object']
            explanation_parts.append(f"Related to '{rel_name}' that you liked")

        if not explanation_parts:
            explanation_parts.append("Based on your interests and preferences")

        return " · ".join(explanation_parts)

# Exercise 9.2: Recommendation system
def exercise_recommendation_system():
    """Exercise: Build recommendation system with knowledge graph"""

    # Create recommender
    recommender = KnowledgeGraphRecommender(kg)

    # Build user profiles
    user_interactions = {
        'user_tech': [
            {'entity': 'steve_jobs', 'type': 'like', 'score': 1.0},
            {'entity': 'apple_inc', 'type': 'like', 'score': 1.0},
            {'entity': 'steve_wozniak', 'type': 'like', 'score': 0.8},
            {'entity': 'microsoft', 'type': 'dislike', 'score': -0.5}
        ],
        'user_business': [
            {'entity': 'apple_inc', 'type': 'like', 'score': 1.0},
            {'entity': 'microsoft', 'type': 'like', 'score': 0.9},
            {'entity': 'bill_gates', 'type': 'like', 'score': 0.8}
        ]
    }

    # Build profiles
    profiles = {}
    for user_id, interactions in user_interactions.items():
        profile = recommender.build_user_profile(user_id, interactions)
        profiles[user_id] = profile
        print(f"👤 Built profile for {user_id}:")
        print(f"  Liked entities: {len(profile['liked_entities'])}")
        print(f"  Disliked entities: {len(profile['disliked_entities'])}")

    # Generate recommendations
    print(f"\n🎯 Recommendations:")

    for user_id in user_interactions.keys():
        print(f"\nRecommendations for {user_id}:")

        recommendations = recommender.recommend_entities(user_id, top_k=5)

        for i, (entity_id, similarity) in enumerate(recommendations):
            entity_info = kg.get_entity(entity_id)
            entity_name = entity_info['properties'].get('name', entity_id) if entity_info else entity_id
            explanation = recommender.explain_recommendation(user_id, entity_id)

            print(f"  {i+1}. {entity_name} (similarity: {similarity:.3f})")
            print(f"     Why: {explanation}")

    # Type-specific recommendations
    print(f"\n🔍 Type-specific recommendations:")

    for user_id in user_interactions.keys():
        print(f"\nCompany recommendations for {user_id}:")

        company_recs = recommender.recommend_entities(user_id, entity_type='Company', top_k=3)

        for i, (entity_id, similarity) in enumerate(company_recs):
            entity_info = kg.get_entity(entity_id)
            entity_name = entity_info['properties'].get('name', entity_id) if entity_info else entity_id
            print(f"  {i+1}. {entity_name} (similarity: {similarity:.3f})")

    return recommender

# Run recommendation system exercise
recommender = exercise_recommendation_system()
```

---

## **10. Advanced Applications**

### **10.1 Knowledge Graph Analytics Dashboard**

```python
class KnowledgeGraphDashboard:
    """Interactive dashboard for knowledge graph analytics"""

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.analytics = GraphAnalytics(knowledge_graph)

    def generate_dashboard_data(self):
        """Generate data for dashboard"""
        dashboard_data = {
            'graph_statistics': self.analytics.analyze_graph_structure(),
            'centrality_measures': self.analytics.calculate_centrality_measures(),
            'community_analysis': self.analytics.find_communities(),
            'entity_type_distribution': self.get_entity_type_distribution(),
            'relationship_type_distribution': self.get_relationship_type_distribution(),
            'anomaly_detection': self.analytics.detect_anomalies(),
            'temporal_analysis': self.get_temporal_analysis()
        }

        return dashboard_data

    def get_entity_type_distribution(self):
        """Get distribution of entity types"""
        type_counts = {}

        for node in self.kg.graph.nodes:
            entity_info = self.kg.get_entity(node)
            if entity_info:
                entity_type = entity_info['type']
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

        return type_counts

    def get_relationship_type_distribution(self):
        """Get distribution of relationship types"""
        rel_counts = {}

        for edge in self.kg.graph.edges(data=True):
            subject, obj, data = edge
            relation = data.get('relationship', 'RELATED_TO')
            rel_counts[relation] = rel_counts.get(relation, 0) + 1

        return rel_counts

    def get_temporal_analysis(self):
        """Analyze temporal patterns in the graph"""
        temporal_data = {
            'entities_by_year': {},
            'relationships_by_year': {},
            'growth_trends': {}
        }

        # Analyze entity creation over time
        for node in self.kg.graph.nodes:
            entity_info = self.kg.get_entity(node)
            if entity_info:
                properties = entity_info['properties']

                # Look for year-related properties
                for prop_name, prop_value in properties.items():
                    if 'year' in prop_name.lower() and isinstance(prop_value, (int, str)):
                        try:
                            year = int(str(prop_value))
                            if 1900 <= year <= 2030:  # Reasonable year range
                                temporal_data['entities_by_year'][year] = temporal_data['entities_by_year'].get(year, 0) + 1
                        except ValueError:
                            continue

        # Analyze relationships over time
        for edge in self.kg.graph.edges(data=True):
            subject, obj, data = edge
            properties = data.get('properties', {})

            for prop_name, prop_value in properties.items():
                if 'year' in prop_name.lower() and isinstance(prop_value, (int, str)):
                    try:
                        year = int(str(prop_value))
                        if 1900 <= year <= 2030:
                            temporal_data['relationships_by_year'][year] = temporal_data['relationships_by_year'].get(year, 0) + 1
                    except ValueError:
                        continue

        return temporal_data

    def create_visualization_data(self):
        """Create data for visualizations"""
        viz_data = {
            'network_graph': self.create_network_graph_data(),
            'centrality_chart': self.create_centrality_chart_data(),
            'distribution_charts': self.create_distribution_charts(),
            'temporal_charts': self.create_temporal_charts()
        }

        return viz_data

    def create_network_graph_data(self):
        """Create network graph data for visualization"""
        nodes = []
        edges = []

        # Create nodes
        for i, node in enumerate(self.kg.graph.nodes):
            entity_info = self.kg.get_entity(node)
            nodes.append({
                'id': i,
                'label': entity_info['properties'].get('name', node) if entity_info else node,
                'type': entity_info['type'] if entity_info else 'Unknown',
                'group': entity_info['type'] if entity_info else 'Unknown'
            })

        # Create edges
        for edge in self.kg.graph.edges(data=True):
            subject, obj, data = edge
            edges.append({
                'from': list(self.kg.graph.nodes).index(subject),
                'to': list(self.kg.graph.nodes).index(obj),
                'label': data.get('relationship', 'RELATED_TO'),
                'width': 1
            })

        return {'nodes': nodes, 'edges': edges}

    def create_centrality_chart_data(self):
        """Create centrality measures chart data"""
        centrality_measures = self.analytics.calculate_centrality_measures()
        chart_data = {}

        # Take top 10 nodes for each measure
        for measure, scores in centrality_measures.items():
            top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
            chart_data[measure] = [
                {
                    'entity': node,
                    'score': score,
                    'name': self.kg.get_entity(node)['properties'].get('name', node) if self.kg.get_entity(node) else node
                }
                for node, score in top_nodes
            ]

        return chart_data

    def create_distribution_charts(self):
        """Create distribution chart data"""
        return {
            'entity_types': self.get_entity_type_distribution(),
            'relationship_types': self.get_relationship_type_distribution()
        }

    def create_temporal_charts(self):
        """Create temporal analysis chart data"""
        temporal_data = self.get_temporal_analysis()

        entities_timeline = [
            {'year': year, 'count': count}
            for year, count in sorted(temporal_data['entities_by_year'].items())
        ]

        relationships_timeline = [
            {'year': year, 'count': count}
            for year, count in sorted(temporal_data['relationships_by_year'].items())
        ]

        return {
            'entities_timeline': entities_timeline,
            'relationships_timeline': relationships_timeline
        }

    def generate_summary_report(self):
        """Generate summary report of knowledge graph"""
        print("📊 Knowledge Graph Dashboard Summary")
        print("=" * 50)

        stats = self.analytics.analyze_graph_structure()
        entity_types = self.get_entity_type_distribution()
        rel_types = self.get_relationship_type_distribution()

        print(f"\n📈 Graph Statistics:")
        print(f"  Nodes (Entities): {stats['num_nodes']}")
        print(f"  Edges (Relationships): {stats['num_edges']}")
        print(f"  Density: {stats['density']:.3f}")

        if 'is_connected' in stats:
            print(f"  Connected: {stats['is_connected']}")
            if not stats['is_connected']:
                print(f"  Components: {stats['num_components']}")

        print(f"\n🏷️ Entity Types:")
        for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {entity_type}: {count}")

        print(f"\n🔗 Relationship Types:")
        for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {rel_type}: {count}")

        # Top entities by centrality
        centrality = self.analytics.calculate_centrality_measures()
        if 'degree_centrality' in centrality:
            print(f"\n⭐ Most Connected Entities:")
            top_entities = sorted(centrality['degree_centrality'].items(),
                                key=lambda x: x[1], reverse=True)[:5]
            for entity, score in top_entities:
                entity_info = self.kg.get_entity(entity)
                name = entity_info['properties'].get('name', entity) if entity_info else entity
                print(f"  {name}: {score:.3f}")

        return {
            'statistics': stats,
            'entity_types': entity_types,
            'relationship_types': rel_types,
            'centrality': centrality
        }

# Exercise 10.1: Dashboard creation
def exercise_dashboard():
    """Exercise: Create knowledge graph analytics dashboard"""

    # Create dashboard
    dashboard = KnowledgeGraphDashboard(kg)

    # Generate summary report
    summary = dashboard.generate_summary_report()

    # Generate visualization data
    viz_data = dashboard.create_visualization_data()

    print(f"\n📊 Dashboard Data Generated:")
    print(f"  Network graph: {len(viz_data['network_graph']['nodes'])} nodes, {len(viz_data['network_graph']['edges'])} edges")
    print(f"  Centrality charts: {len(viz_data['centrality_chart'])} measures")
    print(f"  Distribution charts: {len(viz_data['distribution_charts'])} charts")
    print(f"  Temporal charts: {len(viz_data['temporal_charts'])} timelines")

    return dashboard, viz_data

# Run dashboard exercise
dashboard, viz_data = exercise_dashboard()
```

---

## **Summary**

This comprehensive practice guide covered knowledge graphs through hands-on exercises:

### **What We Accomplished:**

1. **Basic Knowledge Graph Construction** - Built simple and RDF-based knowledge graphs
2. **Entity Extraction & Linking** - Implemented NER and entity resolution
3. **Graph Database Operations** - Explored Neo4j integration and graph analytics
4. **Knowledge Graph Embeddings** - Trained TransE models for link prediction
5. **Graph Neural Networks** - Implemented GCNs for node classification
6. **RAG Enhancement** - Created graph-enhanced retrieval systems
7. **Question Answering** - Built QA systems using knowledge graphs
8. **Bias Detection** - Analyzed bias and fairness in knowledge graphs
9. **Advanced Applications** - Completed missing links and built recommenders
10. **Analytics Dashboard** - Created comprehensive analytics visualizations

### **Key Skills Developed:**

- ✅ Knowledge graph design and implementation
- ✅ Entity extraction and linking techniques
- ✅ Graph neural networks for knowledge graphs
- ✅ Bias detection and fairness analysis
- ✅ RAG system enhancement with knowledge graphs
- ✅ Advanced applications (QA, recommendations, completion)

The knowledge graph module is now complete with comprehensive theory (2,141 lines) and practice materials, providing students with both conceptual understanding and practical implementation skills for modern knowledge graph systems.
