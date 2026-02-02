# Graph Neural Networks Quick Reference Cheatsheet

## Table of Contents

1. [Core GNN Algorithms](#core-gnn-algorithms)
2. [Graph Representations](#graph-representations)
3. [Implementation Patterns](#implementation-patterns)
4. [Architecture Comparison](#architecture-comparison)
5. [Training Guidelines](#training-guidelines)
6. [Performance Optimization](#performance-optimization)
7. [Common Applications](#common-applications)
8. [Debugging Checklist](#debugging-checklist)
9. [Deployment Guide](#deployment-guide)
10. [Quick Code Snippets](#quick-code-snippets)

## Core GNN Algorithms {#core-gnn-algorithms}

### Graph Convolutional Networks (GCN)

```python
# Basic GCN Layer
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()

    def forward(self, H, A):
        # Normalized adjacency: D^(-1/2) * A * D^(-1/2)
        A_hat = A + torch.eye(A.size(0))
        D_hat = torch.diag(torch.pow(A_hat.sum(1), -0.5))
        A_norm = torch.mm(torch.mm(D_hat, A_hat), D_hat)

        # GCN operation: A_norm * H * W
        return torch.mm(A_norm, torch.mm(H, self.weight))
```

**Key Formula**: `H^(l+1) = σ(D^(-1/2) * A * D^(-1/2) * H^(l) * W^(l))`

### Graph Attention Networks (GAT)

```python
# Single-head attention
def attention_coefficients(H, A, W, a):
    h_trans = torch.mm(H, W)  # (N, F')
    N = h_trans.size(0)

    # Compute attention scores
    e = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if A[i, j] > 0:
                e[i, j] = F.leaky_relu(a.T @ torch.cat([h_trans[i], h_trans[j]]))

    # Mask non-edges
    e = torch.where(A > 0, e, torch.tensor(-1e9))

    # Normalize with softmax
    attention = F.softmax(e, dim=1)
    return attention

# Multi-head aggregation
def multi_head_attention(H, A, W_list, a_list):
    head_outputs = []
    for W, a in zip(W_list, a_list):
        attention = attention_coefficients(H, A, W, a)
        head_out = torch.mm(attention, torch.mm(H, W))
        head_outputs.append(F.elu(head_out))
    return torch.cat(head_outputs, dim=1)
```

**Key Formula**: `attention_ij = softmax(LeakyReLU(a^T[W*h_i || W*h_j]))`

### GraphSAGE (Sampling and Aggregation)

```python
class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim, output_dim, aggregator='mean'):
        super().__init__()
        self.aggregator = aggregator
        self.self_transform = nn.Linear(input_dim, output_dim)
        self.neighbor_transform = nn.Linear(input_dim, output_dim)

    def forward(self, node_features, neighbor_features):
        # Self representation
        self_repr = F.relu(self.self_transform(node_features))

        # Aggregate neighbors
        if self.aggregator == 'mean':
            neighbor_repr = torch.mean(neighbor_features, dim=1)
        elif self.aggregator == 'max':
            neighbor_repr = torch.max(neighbor_features, dim=1)[0]
        elif self.aggregator == 'lstm':
            neighbor_repr = self.lstm_aggregate(neighbor_features)

        neighbor_repr = F.relu(self.neighbor_transform(neighbor_repr))
        return self_repr + neighbor_repr
```

**Key Concepts**:

- **Sampling**: Sample fixed-size neighborhood for each node
- **Aggregation**: Mean, Max, or LSTM aggregation functions
- **Inductive**: Can generalize to unseen graphs

### Message Passing Neural Networks (MPNN)

```python
class MessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def message_function(self, h_i, h_j, e_ij):
        return self.message_mlp(torch.cat([h_i, h_j, e_ij]))

    def update_function(self, h_i, messages):
        aggregated = torch.sum(messages, dim=0)
        return self.update_mlp(torch.cat([h_i, aggregated]))
```

**Framework**:

1. **Message**: `m_ij^((l)) = M_l(h_i^(l), h_j^(l), e_ij)`
2. **Aggregate**: `h_i^(l+1) = U_l(h_i^(l), Σ_j m_ij^(l))`
3. **Readout**: `ŷ = R({h_i^(L) | v_i ∈ G})`

## Graph Representations {#graph-representations}

### Data Structures

| Format               | Description                     | Use Case                              |
| -------------------- | ------------------------------- | ------------------------------------- |
| **Adjacency Matrix** | Dense N×N matrix                | Small graphs, mathematical operations |
| **Edge List**        | List of (source, target) tuples | Large sparse graphs                   |
| **Adjacency List**   | Dict mapping node → neighbors   | Efficient neighborhood queries        |
| **Sparse Tensor**    | COO format for PyTorch          | GPU-efficient processing              |

### Feature Encoding

#### Node Features

```python
# Common node feature types
node_features = {
    'categorical': torch.nn.Embedding(num_categories, embedding_dim),
    'numerical': torch.nn.Linear(1, feature_dim),
    'one_hot': F.one_hot(torch.tensor(categories), num_classes),
    'normalized': (values - mean) / std
}
```

#### Edge Features

```python
# Edge feature encoding
edge_features = {
    'bond_type': one_hot(bond_types, num_bond_types),
    'weight': normalize(weights),
    'distance': rbf_kernel(distances),
    'temporal': sinusoidal_encoding(timestamps)
}
```

### Graph Types

| Type              | Description                | Examples                                |
| ----------------- | -------------------------- | --------------------------------------- |
| **Homogeneous**   | Single node and edge type  | Social networks, citation networks      |
| **Heterogeneous** | Multiple node/edge types   | Knowledge graphs, multi-relational data |
| **Dynamic**       | Temporal edge/node changes | Social media, transaction networks      |
| **Spatial**       | Geometric constraints      | Molecular graphs, road networks         |

## Implementation Patterns {#implementation-patterns}

### Basic GCN Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            GCNLayer(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers - 1)
        ])
        self.layers.append(GCNLayer(hidden_dim, output_dim))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.dropout(F.relu(layer(x, adj)))
        x = self.layers[-1](x, adj)  # No activation on output
        return F.log_softmax(x, dim=1)
```

### Attention Mechanism Pattern

```python
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Generate Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )

        return self.W_o(context)
```

### Batch Processing Pattern

```python
def collate_graphs(batch):
    """Collate a batch of graphs into a single large graph"""
    # Find maximum sizes
    max_nodes = max(g['num_nodes'] for g in batch)
    max_edges = max(g['edge_index'].size(1) for g in batch)

    # Initialize batch tensors
    batch_size = len(batch)
    batch_node_features = torch.zeros(batch_size * max_nodes, batch[0]['node_features'].size(1))
    batch_edge_index = torch.zeros(2, batch_size * max_edges, dtype=torch.long)
    batch_edge_features = torch.zeros(batch_size * max_edges, batch[0]['edge_features'].size(1))
    batch_graph_index = torch.zeros(batch_size * max_nodes, dtype=torch.long)

    # Fill batch
    node_offset = 0
    edge_offset = 0

    for i, graph in enumerate(batch):
        # Node features
        n_nodes = graph['num_nodes']
        batch_node_features[node_offset:node_offset + n_nodes] = graph['node_features']

        # Edge index (with offset)
        e_edges = graph['edge_index'].size(1)
        batch_edge_index[:, edge_offset:edge_offset + e_edges] = graph['edge_index'] + node_offset

        # Edge features
        batch_edge_features[edge_offset:edge_offset + e_edges] = graph['edge_features']

        # Graph index
        batch_graph_index[node_offset:node_offset + n_nodes] = i

        node_offset += max_nodes
        edge_offset += max_edges

    return {
        'node_features': batch_node_features,
        'edge_index': batch_edge_index,
        'edge_features': batch_edge_features,
        'graph_index': batch_graph_index,
        'num_graphs': batch_size
    }
```

## Architecture Comparison {#architecture-comparison}

### GNN Variants Comparison

| Algorithm             | Expressiveness | Computational Cost | Scalability | Interpretability | Best For                                     |
| --------------------- | -------------- | ------------------ | ----------- | ---------------- | -------------------------------------------- |
| **GCN**               | ⭐⭐⭐         | ⭐⭐⭐⭐⭐         | ⭐⭐⭐⭐    | ⭐⭐             | General purpose, node classification         |
| **GraphSAGE**         | ⭐⭐⭐         | ⭐⭐⭐⭐           | ⭐⭐⭐⭐⭐  | ⭐⭐             | Large-scale graphs, inductive learning       |
| **GAT**               | ⭐⭐⭐⭐       | ⭐⭐⭐             | ⭐⭐⭐      | ⭐⭐⭐⭐⭐       | Attention-requiring tasks, interpretability  |
| **GIN**               | ⭐⭐⭐⭐⭐     | ⭐⭐⭐             | ⭐⭐⭐      | ⭐⭐             | Graph classification, maximum expressiveness |
| **Graph Transformer** | ⭐⭐⭐⭐⭐     | ⭐                 | ⭐⭐        | ⭐⭐⭐           | Long-range dependencies, complex patterns    |

### Layer-wise Complexity

| Operation                | Time Complexity | Space Complexity | Notes                          |
| ------------------------ | --------------- | ---------------- | ------------------------------ |
| **Message Passing**      | O(E×F)          | O(V×F)           | Linear in edges                |
| **Graph Attention**      | O(V²×F)         | O(V²)            | Quadratic in nodes             |
| **Spectral Convolution** | O(V³)           | O(V²)            | Not practical for large graphs |
| **Sampling**             | O(K×V×F)        | O(K×V×F)         | K = sample size                |

### Memory Requirements

| Graph Size            | GCN | GraphSAGE | GAT | Graph Transformer |
| --------------------- | --- | --------- | --- | ----------------- |
| **Small (<1K nodes)** | ✅  | ✅        | ✅  | ✅                |
| **Medium (1K-100K)**  | ⚠️  | ✅        | ⚠️  | ❌                |
| **Large (>100K)**     | ❌  | ✅        | ❌  | ❌                |

## Training Guidelines {#training-guidelines}

### Loss Functions by Task

| Task                     | Loss Function      | Notes                       |
| ------------------------ | ------------------ | --------------------------- |
| **Node Classification**  | CrossEntropy / NLL | For class labels            |
| **Graph Classification** | CrossEntropy / MSE | For graph-level targets     |
| **Link Prediction**      | BCE / Margin       | Predict edge existence      |
| **Node Regression**      | MSE / MAE          | Continuous node values      |
| **Graph Regression**     | MSE / MAE          | Continuous graph properties |

### Hyperparameter Guidelines

```python
# Recommended hyperparameters
hyperparams = {
    'learning_rate': {
        'node_classification': 0.01,
        'graph_classification': 0.001,
        'link_prediction': 0.005
    },
    'hidden_dim': {
        'small_graphs': 64,
        'medium_graphs': 128,
        'large_graphs': 256
    },
    'num_layers': {
        'shallow': 2-3,
        'medium': 3-5,
        'deep': 5-8 (with residual connections)
    },
    'dropout': {
        'training': 0.1-0.5,
        'final_layer': 0.0
    },
    'weight_decay': 5e-4,  # L2 regularization
    'batch_size': {
        'small': 256,
        'medium': 128,
        'large': 32
    }
}
```

### Training Loop Template

```python
def train_gnn(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(data.x, data.edge_index)

    # Loss computation
    if data.y.dim() == 1:  # Node classification
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
    else:  # Graph classification
        loss = criterion(output, data.y)

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()

def evaluate_gnn(model, data, mask=None):
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)

        if mask is not None:
            pred = output[mask].argmax(dim=1)
            true = data.y[mask]
        else:
            pred = output.argmax(dim=1)
            true = data.y

        accuracy = (pred == true).float().mean()
    return accuracy.item()
```

### Regularization Techniques

```python
# Dropout
model = nn.Sequential(
    GCNLayer(input_dim, hidden_dim),
    nn.Dropout(0.5),
    GCNLayer(hidden_dim, output_dim)
)

# Weight decay
optimizer = torch.optim.Adam(model.parameters(),
                           lr=0.01,
                           weight_decay=5e-4)

# Graph dropout
def graph_dropout(A, p=0.1):
    """Randomly remove edges"""
    mask = torch.rand(A.shape) > p
    return A * mask

# Residual connections
class ResidualGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gcn = GCNLayer(input_dim, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, A):
        identity = self.linear(x)
        out = self.gcn(x, A)
        return F.relu(identity + out)
```

## Performance Optimization {#performance-optimization}

### Sparse Operations

```python
# Efficient sparse matrix multiplication
def sparse_gcn_forward(x, edge_index, edge_weight, weight):
    # Convert to sparse tensor
    row, col = edge_index
    sparse_A = torch.sparse_coo_tensor(
        edge_index, edge_weight, (x.size(0), x.size(0))
    )

    # Sparse matrix multiplication
    h = torch.mm(x, weight)
    h = torch.sparse.mm(sparse_A, h)

    return h

# Batch processing for large graphs
def process_large_graph_batch(graph, batch_size=1000):
    """Process large graphs in batches"""
    node_embeddings = []

    for start_idx in range(0, graph.num_nodes, batch_size):
        end_idx = min(start_idx + batch_size, graph.num_nodes)

        # Get subgraph
        batch_nodes = torch.arange(start_idx, end_idx)
        batch_edges = get_edges_in_subgraph(graph, batch_nodes)

        # Process batch
        batch_emb = model(graph.x[batch_nodes], batch_edges)
        node_embeddings.append(batch_emb)

    return torch.cat(node_embeddings, dim=0)
```

### GPU Optimization

```python
# Memory-efficient attention
class MemoryEfficientAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()

        # Chunked computation for memory efficiency
        chunk_size = min(1024, seq_len)

        outputs = []
        for i in range(0, seq_len, chunk_size):
            chunk = x[:, i:i+chunk_size, :]

            # Compute attention for chunk
            q = self.q_proj(chunk).view(batch_size, chunk.size(1), self.num_heads, self.head_dim)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                scores = scores.masked_fill(
                    attention_mask[:, i:i+chunk_size, :, :] == 0, -1e9
                )

            attn_weights = F.softmax(scores, dim=-1)
            chunk_output = torch.matmul(attn_weights, v)

            outputs.append(chunk_output)

        # Concatenate chunks
        combined = torch.cat(outputs, dim=1)
        return self.out_proj(combined.view(batch_size, seq_len, -1))
```

### Profiling and Debugging

```python
import torch.profiler

def profile_gnn_training(model, data):
    """Profile GNN training performance"""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    ) as prof:
        for step in range(5):
            # Training step
            output = model(data.x, data.edge_index)
            loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
            loss.backward()

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Memory usage tracking
def track_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
```

## Common Applications {#common-applications}

### Social Networks

```python
# Friend recommendation
def recommend_friends(user_id, user_features, adj_matrix, model):
    """Recommend friends using GNN embeddings"""
    user_emb = model(user_features, adj_matrix)
    user_vec = user_emb[user_id]

    # Compute similarity with all users
    similarities = torch.cosine_similarity(user_vec, user_emb)

    # Exclude existing friends
    friends = torch.where(adj_matrix[user_id] > 0)[0]
    similarities[friends] = -1  # Exclude friends

    # Get top recommendations
    top_k = torch.argsort(similarities, descending=True)[:10]
    return top_k

# Community detection
def detect_communities(graph_embeddings, num_communities=10):
    """Detect communities using embeddings"""
    kmeans = KMeans(n_clusters=num_communities)
    communities = kmeans.fit_predict(graph_embeddings.numpy())
    return communities
```

### Molecular Property Prediction

```python
# Molecular property prediction
class MolecularGNN(nn.Module):
    def __init__(self, atom_dim, bond_dim, hidden_dim, output_dim):
        super().__init__()
        self.atom_embedding = nn.Linear(atom_dim, hidden_dim)
        self.bond_embedding = nn.Linear(bond_dim, hidden_dim)
        self.message_passing = MessagePassingLayer(hidden_dim, hidden_dim, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, atom_features, bond_features, edge_index):
        # Embed atoms and bonds
        h_atoms = self.atom_embedding(atom_features)
        h_bonds = self.bond_embedding(bond_features)

        # Message passing
        for _ in range(3):  # 3 propagation steps
            h_atoms = self.message_passing(h_atoms, edge_index, h_bonds)

        # Global pooling
        h_graph = torch.mean(h_atoms, dim=0, keepdim=True)

        # Predict properties
        properties = self.readout(h_graph)
        return properties
```

### Knowledge Graph Completion

```python
# Knowledge graph embedding
class KGEmbedding(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

    def forward(self, head, relation, tail):
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        # TransE scoring function
        score = torch.norm(h + r - t, p=2, dim=1)
        return -score  # Negative distance as score
```

### Fraud Detection

```python
# Transaction graph fraud detection
class FraudGNN(nn.Module):
    def __init__(self, user_dim, transaction_dim, hidden_dim):
        super().__init__()
        self.user_encoder = nn.Linear(user_dim, hidden_dim)
        self.transaction_encoder = nn.Linear(transaction_dim, hidden_dim)
        self.fraud_gnn = GCN(hidden_dim, hidden_dim, hidden_dim, num_layers=3)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, user_features, transaction_features, graph):
        # Encode users and transactions
        user_emb = self.user_encoder(user_features)
        trans_emb = self.transaction_encoder(transaction_features)

        # Combine embeddings (simplified)
        combined_emb = user_emb + trans_emb[:user_emb.size(0)]

        # GNN for relational reasoning
        fraud_emb = self.fraud_gnn(combined_emb, graph)

        # Classify
        fraud_score = torch.sigmoid(self.classifier(fraud_emb))
        return fraud_score
```

## Debugging Checklist {#debugging-checklist}

### Common Issues and Solutions

#### 1. Poor Performance

**Symptoms**: Low accuracy, high loss, slow convergence

**Check List**:

- [ ] **Feature Normalization**: Normalize node and edge features
- [ ] **Adjacency Matrix**: Verify no isolated nodes, proper normalization
- [ ] **Learning Rate**: Try values from 1e-5 to 1e-1
- [ ] **Model Capacity**: Increase/decrease hidden dimensions
- [ ] **Regularization**: Add dropout, weight decay
- [ ] **Data Split**: Ensure no data leakage between train/val/test

**Quick Fix**:

```python
# Add normalization
node_features = F.normalize(node_features, dim=1)
edge_weights = F.softmax(edge_weights, dim=0)

# Increase model capacity
model = GCN(input_dim, hidden_dim=256, output_dim, num_layers=4)

# Add regularization
optimizer = torch.optim.Adam(model.parameters(),
                           lr=0.01,
                           weight_decay=1e-4)
```

#### 2. Over-smoothing

**Symptoms**: All node embeddings become similar

**Symptoms**:

- Node embeddings converge to similar values
- Performance degrades with more layers
- Low variance in node representations

**Solutions**:

```python
# Add residual connections
class ResidualGCN(nn.Module):
    def forward(self, x, A):
        identity = x
        out = F.relu(self.gcn1(x, A))
        out = self.gcn2(out, A)
        return identity + out

# Use skip connections
class SkipGCN(nn.Module):
    def forward(self, x, A):
        h1 = self.layer1(x, A)
        h2 = self.layer2(h1, A)
        h3 = self.layer3(h2, A)
        return x + 0.1 * (h1 + h2 + h3)  # Skip connections

# Limit depth
model = GCN(input_dim, hidden_dim, output_dim, num_layers=2)  # Shallow
```

#### 3. Memory Issues

**Symptoms**: Out of memory errors, slow training

**Solutions**:

```python
# Reduce batch size
batch_size = 32  # Smaller batches

# Use sparse operations
from torch_sparse import SparseTensor
adj_sparse = SparseTensor(row=edge_index[0],
                         col=edge_index[1],
                         value=edge_weight)

# Implement neighbor sampling
def sample_neighbors(nodes, num_neighbors):
    neighbors = {}
    for node in nodes:
        all_neighbors = adj_list[node]
        if len(all_neighbors) > num_neighbors:
            neighbors[node] = random.sample(all_neighbors, num_neighbors)
        else:
            neighbors[node] = all_neighbors
    return neighbors

# Use gradient checkpointing
model = nn.DataParallel(model)
torch.utils.checkpoint.checkpoint_sequential(model, segments=2)
```

#### 4. Graph Structure Issues

**Symptoms**: Poor performance on specific graph types

**Solutions**:

```python
# Add self-loops
adj_with_self = adj_matrix + torch.eye(adj_matrix.size(0))

# Handle isolated nodes
def normalize_adjacency(A):
    A = A + torch.eye(A.size(0))  # Add self-loops
    degree = A.sum(1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    return torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)

# Check graph connectivity
import networkx as nx
G = nx.from_numpy_array(adj_matrix.numpy())
num_components = nx.number_connected_components(G)
print(f"Number of connected components: {num_components}")
```

### Validation Tests

```python
# 1. Model capacity test
def test_model_capacity():
    """Test if model can fit a simple dataset"""
    model = GCN(16, 64, 3, num_layers=3)
    x = torch.randn(100, 16)
    y = torch.randint(0, 3, (100,))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(50):
        optimizer.zero_grad()
        output = model(x, torch.randint(0, 2, (100, 100)))
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc = (output.argmax(dim=1) == y).float().mean()
            print(f"Epoch {epoch}: Loss={loss.item():.3f}, Acc={acc.item():.3f}")

# 2. Gradient flow test
def test_gradient_flow():
    """Test if gradients flow properly through the model"""
    model = GCN(16, 32, 3, num_layers=3)
    x = torch.randn(50, 16)
    y = torch.randint(0, 3, (50,))

    output = model(x, torch.randint(0, 2, (50, 50)))
    loss = F.cross_entropy(output, y)
    loss.backward()

    # Check for NaN gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
            if torch.isinf(param.grad).any():
                print(f"Inf gradient in {name}")

# 3. Node degree distribution
def analyze_graph_properties(adj_matrix):
    """Analyze basic graph properties"""
    degrees = adj_matrix.sum(dim=1)

    print(f"Graph Properties:")
    print(f"  Nodes: {adj_matrix.size(0)}")
    print(f"  Edges: {adj_matrix.sum().item() / 2:.0f}")
    print(f"  Density: {adj_matrix.sum() / (adj_matrix.size(0) ** 2):.4f}")
    print(f"  Avg degree: {degrees.mean():.2f}")
    print(f"  Max degree: {degrees.max():.0f}")
    print(f"  Min degree: {degrees.min():.0f}")

    # Check for isolated nodes
    isolated = (degrees == 0).sum().item()
    print(f"  Isolated nodes: {isolated}")

    return degrees
```

## Deployment Guide {#deployment-guide}

### Model Serving

```python
# FastAPI GNN inference service
from fastapi import FastAPI, HTTPException
import torch
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="GNN Inference API")

class GraphInput(BaseModel):
    node_features: list
    edge_list: list
    edge_features: list = None

class GNNInferenceService:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def preprocess(self, graph_input):
        """Convert API input to model input"""
        node_features = torch.tensor(graph_input.node_features, dtype=torch.float32)
        edge_index = torch.tensor(graph_input.edge_list, dtype=torch.long)

        if graph_input.edge_features:
            edge_features = torch.tensor(graph_input.edge_features, dtype=torch.float32)
        else:
            edge_features = torch.ones(edge_index.size(1), 1)  # Default edge weight

        return node_features, edge_index, edge_features

    def predict(self, graph_input):
        """Run inference"""
        try:
            x, edge_index, edge_features = self.preprocess(graph_input)

            with torch.no_grad():
                output = self.model(x, edge_index, edge_features)

            return output.cpu().numpy()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

# Initialize service
gnn_service = GNNInferenceService("path/to/model.pth")

@app.post("/predict")
def predict_node_classification(graph: GraphInput):
    """Predict node classifications"""
    output = gnn_service.predict(graph)
    predictions = np.argmax(output, axis=1).tolist()
    probabilities = torch.softmax(torch.tensor(output), dim=1).numpy().tolist()

    return {
        "predictions": predictions,
        "probabilities": probabilities
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": gnn_service.model is not None}
```

### Model Optimization

```python
# Model quantization
def quantize_model(model):
    """Quantize model for inference"""
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    quantized_model = torch.quantization.prepare(model, inplace=False)
    quantized_model = torch.quantization.convert(quantized_model, inplace=False)
    return quantized_model

# Model pruning
def prune_model(model, amount=0.3):
    """Prune model weights"""
    import torch.nn.utils.prune as prune

    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)

    return model

# ONNX export
def export_to_onnx(model, sample_input, path):
    """Export model to ONNX format"""
    torch.onnx.export(
        model,
        sample_input,
        path,
        input_names=['node_features', 'edge_index'],
        output_names=['predictions'],
        dynamic_axes={
            'node_features': {0: 'num_nodes'},
            'edge_index': {1: 'num_edges'},
            'predictions': {0: 'num_nodes'}
        },
        opset_version=11
    )
```

### Batch Processing

```python
class GNNBatchProcessor:
    def __init__(self, model, batch_size=1000):
        self.model = model
        self.batch_size = batch_size

    def process_large_graph(self, x, edge_index):
        """Process large graphs in batches"""
        num_nodes = x.size(0)
        node_batches = torch.split(torch.arange(num_nodes), self.batch_size)

        all_outputs = []
        for batch_nodes in node_batches:
            # Get subgraph for this batch
            batch_mask = torch.zeros(num_nodes, dtype=torch.bool)
            batch_mask[batch_nodes] = True

            # Get edges within batch
            batch_edges = self._get_subgraph_edges(edge_index, batch_mask)

            if batch_edges.size(1) > 0:
                batch_x = x[batch_nodes]
                batch_output = self.model(batch_x, batch_edges)
                all_outputs.append(batch_output)
            else:
                # Handle isolated nodes
                zero_output = torch.zeros(len(batch_nodes), self.model.output_dim)
                all_outputs.append(zero_output)

        return torch.cat(all_outputs, dim=0)

    def _get_subgraph_edges(self, edge_index, node_mask):
        """Get edges that connect nodes within the batch"""
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        subgraph_edges = edge_index[:, edge_mask]

        # Remap node indices
        node_mapping = torch.full((node_mask.size(0),), -1, dtype=torch.long)
        node_mapping[node_mask] = torch.arange(node_mask.sum())
        remapped_edges = node_mapping[subgraph_edges]

        return remapped_edges
```

## Quick Code Snippets {#quick-code-snippets}

### Essential Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GraphSAGEConv
import numpy as np
import networkx as nx
from sklearn.metrics import accuracy_score, roc_auc_score
```

### Data Loading

```python
# Load graph from edge list
def load_graph_from_edges(edge_list, node_features=None):
    """Load graph from edge list format"""
    edges = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    if node_features is not None:
        x = torch.tensor(node_features, dtype=torch.float32)
    else:
        x = torch.randn(len(set(sum(edge_list))), 10)  # Default features

    graph = Data(x=x, edge_index=edges)
    return graph

# Create synthetic graph
def create_synthetic_graph(n_nodes=100, edge_prob=0.1):
    """Create random graph for testing"""
    G = nx.erdos_renyi_graph(n_nodes, edge_prob)
    edge_list = list(G.edges())

    # Random node features
    node_features = np.random.randn(n_nodes, 16)

    return load_graph_from_edges(edge_list, node_features)
```

### Model Initialization

```python
# Quick GCN model
def create_gcn(input_dim, hidden_dim, output_dim):
    """Create a simple GCN model"""
    return nn.Sequential(
        GCNConv(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        GCNConv(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        GCNConv(hidden_dim, output_dim)
    )

# Quick GAT model
def create_gat(input_dim, hidden_dim, output_dim, num_heads=4):
    """Create a simple GAT model"""
    return nn.Sequential(
        GATConv(input_dim, hidden_dim // num_heads, heads=num_heads),
        nn.ELU(),
        GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads),
        nn.ELU(),
        GATConv(hidden_dim, output_dim, heads=1, concat=False)
    )
```

### Training Utilities

```python
# Quick training function
def train_model(model, data, epochs=100, lr=0.01):
    """Train GNN model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred = out[data.val_mask].argmax(dim=1)
                acc = (pred == data.y[data.val_mask]).float().mean()
                print(f'Epoch {epoch}: Loss {loss:.4f}, Val Acc {acc:.4f}')

# Evaluation function
def evaluate_model(model, data):
    """Evaluate GNN model"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        return {
            'train_acc': train_acc.item(),
            'val_acc': val_acc.item(),
            'test_acc': test_acc.item()
        }
```

### Visualization

```python
# Plot training curves
def plot_training_curves(train_losses, val_accs):
    """Plot training progress"""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2.plot(val_accs)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

# Visualize graph structure
def visualize_graph(adj_matrix, node_labels=None, figsize=(10, 8)):
    """Visualize graph structure"""
    G = nx.from_numpy_array(adj_matrix.numpy())
    plt.figure(figsize=figsize)

    pos = nx.spring_layout(G, seed=42)

    if node_labels is not None:
        nx.draw(G, pos, node_color=node_labels, node_size=100,
                cmap=plt.cm.Set3, with_labels=True)
    else:
        nx.draw(G, pos, node_size=50, alpha=0.8)

    plt.title('Graph Structure')
    plt.axis('off')
    plt.show()
```

---

## Quick Reference Summary

### Essential Formulas

1. **GCN**: `H^(l+1) = σ(D^(-1/2) * A * D^(-1/2) * H^(l) * W^(l))`
2. **GAT**: `attention_ij = softmax(LeakyReLU(a^T[W*h_i || W*h_j]))`
3. **Message Passing**: `h_i^(l+1) = U_l(h_i^(l), Σ_j M_l(h_i^(l), h_j^(l), e_ij))`

### Hyperparameter Defaults

- **Learning Rate**: 0.01 (node), 0.001 (graph)
- **Hidden Dimension**: 64-256
- **Dropout**: 0.1-0.5
- **Weight Decay**: 5e-4
- **Layers**: 2-4 (GCN), 1-2 (GAT)

### Performance Benchmarks

| Dataset     | Task                 | GCN Acc | GAT Acc | GraphSAGE Acc |
| ----------- | -------------------- | ------- | ------- | ------------- |
| Cora        | Node Classification  | 81.5%   | 83.0%   | 81.8%         |
| CiteSeer    | Node Classification  | 70.3%   | 72.5%   | 71.2%         |
| PubMed      | Node Classification  | 79.0%   | 79.4%   | 80.1%         |
| IMDB-BINARY | Graph Classification | 70.0%   | 74.3%   | 71.2%         |

### Memory Guidelines

- **Small Graphs** (<1K nodes): Standard GNN
- **Medium Graphs** (1K-100K): GraphSAGE + sampling
- **Large Graphs** (>100K): Distributed training + checkpointing

This cheatsheet provides quick access to essential GNN concepts, implementations, and best practices for rapid development and debugging.
