# Graph Neural Networks Practice Exercises

## Table of Contents

1. [Basic Graph Operations](#basic-graph-operations)
2. [Implementing Graph Convolutional Networks](#implementing-graph-convolutional-networks)
3. [Graph Attention Networks Implementation](#graph-attention-networks-implementation)
4. [Message Passing Neural Networks](#message-passing-neural-networks)
5. [Advanced GNN Architectures](#advanced-gnn-architectures)
6. [Real-world Applications](#real-world-applications)
7. [Performance Optimization](#performance-optimization)
8. [End-to-End Projects](#end-to-end-projects)

## Basic Graph Operations {#basic-graph-operations}

### Exercise 1: Graph Data Structures

**Task**: Implement efficient graph representations and operations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from collections import defaultdict
import scipy.sparse as sp

class GraphDataStructure:
    """Efficient graph data structure for GNN training"""

    def __init__(self, num_nodes, edges, edge_features=None, node_features=None):
        self.num_nodes = num_nodes
        self.edges = edges
        self.edge_features = edge_features
        self.node_features = node_features

        # Create adjacency matrix
        self.adj_matrix = self._create_adjacency_matrix(edges)

        # Create sparse representation for efficiency
        self.adj_sparse = sp.coo_matrix(self.adj_matrix)

    def _create_adjacency_matrix(self, edges):
        """Create adjacency matrix from edge list"""
        adj = np.zeros((self.num_nodes, self.num_nodes))
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1  # Undirected graph
        return adj

    def get_neighbors(self, node):
        """Get neighbors of a specific node"""
        neighbors = []
        for i in range(self.num_nodes):
            if self.adj_matrix[node][i] == 1:
                neighbors.append(i)
        return neighbors

    def get_degree_matrix(self):
        """Create degree matrix"""
        degrees = np.sum(self.adj_matrix, axis=1)
        return np.diag(degrees)

    def normalize_adjacency(self):
        """Create normalized adjacency matrix"""
        degree_matrix = self.get_degree_matrix()
        degree_matrix_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))

        # Handle isolated nodes (degree = 0)
        degree_matrix_inv_sqrt = np.nan_to_num(degree_matrix_inv_sqrt)

        normalized_adj = np.dot(
            np.dot(degree_matrix_inv_sqrt, self.adj_matrix),
            degree_matrix_inv_sqrt
        )
        return normalized_adj

    def to_pytorch_sparse(self):
        """Convert to PyTorch sparse tensor"""
        indices = torch.tensor(self.adj_sparse.nonzero())
        values = torch.tensor(self.adj_sparse.data)
        shape = torch.Size(self.adj_sparse.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

# Test the graph data structure
def test_graph_data_structure():
    # Create a simple graph
    num_nodes = 6
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 3)]

    graph = GraphDataStructure(num_nodes, edges)

    print("Adjacency Matrix:")
    print(graph.adj_matrix)
    print("\nNormalized Adjacency Matrix:")
    print(graph.normalize_adjacency())

    # Test neighbor retrieval
    print(f"\nNeighbors of node 1: {graph.get_neighbors(1)}")

    # Test sparse representation
    sparse_tensor = graph.to_pytorch_sparse()
    print(f"\nSparse tensor shape: {sparse_tensor.shape}")

# Run the test
test_graph_data_structure()
```

### Exercise 2: Synthetic Graph Generation

**Task**: Create functions to generate different types of synthetic graphs for testing.

```python
def generate_erdos_renyi_graph(n, p, seed=42):
    """Generate Erdős–Rényi random graph"""
    np.random.seed(seed)
    edges = []

    for i in range(n):
        for j in range(i+1, n):
            if np.random.random() < p:
                edges.append((i, j))

    return GraphDataStructure(n, edges)

def generate_barabasi_albert_graph(n, m, seed=42):
    """Generate Barabási–Albert preferential attachment graph"""
    np.random.seed(seed)
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    edges = list(G.edges())

    return GraphDataStructure(n, edges)

def generate_stochastic_block_model(n, num_blocks, p_intra, p_inter, seed=42):
    """Generate stochastic block model graph"""
    np.random.seed(seed)

    # Randomly assign nodes to blocks
    node_to_block = np.random.randint(0, num_blocks, n)
    edges = []

    for i in range(n):
        for j in range(i+1, n):
            if node_to_block[i] == node_to_block[j]:
                # Intra-block edge
                if np.random.random() < p_intra:
                    edges.append((i, j))
            else:
                # Inter-block edge
                if np.random.random() < p_inter:
                    edges.append((i, j))

    return GraphDataStructure(n, edges)

# Generate different graph types for testing
def generate_test_graphs():
    """Generate a variety of test graphs"""
    graphs = {}

    # Small regular graph
    graphs['small_regular'] = generate_erdos_renyi_graph(10, 0.3)

    # Medium BA graph
    graphs['ba_medium'] = generate_barabasi_albert_graph(50, 3)

    # Large stochastic block model
    graphs['sbm_large'] = generate_stochastic_block_model(100, 4, 0.8, 0.1)

    return graphs

# Test synthetic graph generation
test_graphs = generate_test_graphs()
print("Generated graphs:")
for name, graph in test_graphs.items():
    print(f"{name}: {graph.num_nodes} nodes, {len(graph.edges)} edges")
```

## Implementing Graph Convolutional Networks {#implementing-graph-convolutional-networks}

### Exercise 3: Basic GCN Layer Implementation

**Task**: Implement a complete GCN layer from scratch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicGCNLayer(nn.Module):
    """Basic Graph Convolutional Network Layer"""

    def __init__(self, input_dim, output_dim, bias=True, activation=None):
        super(BasicGCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        # Learnable parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform distribution"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, node_features, adjacency_matrix):
        """
        Forward pass of GCN layer

        Args:
            node_features: Tensor of shape (num_nodes, input_dim)
            adjacency_matrix: Tensor or numpy array of shape (num_nodes, num_nodes)

        Returns:
            Tensor of shape (num_nodes, output_dim)
        """
        # Convert adjacency matrix to tensor if needed
        if not isinstance(adjacency_matrix, torch.Tensor):
            adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)

        # Add self-loops to adjacency matrix
        num_nodes = adjacency_matrix.shape[0]
        adj_with_self_loops = adjacency_matrix + torch.eye(num_nodes)

        # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
        degree = torch.sum(adj_with_self_loops, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0  # Handle isolated nodes

        # Create normalized adjacency matrix
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_normalized = torch.mm(torch.mm(D_inv_sqrt, adj_with_self_loops), D_inv_sqrt)

        # Compute new node features: A_hat * X * W
        support = torch.mm(node_features, self.weight)
        output = torch.mm(adj_normalized, support)

        # Add bias
        if self.bias is not None:
            output = output + self.bias

        # Apply activation function
        if self.activation is not None:
            output = self.activation(output)

        return output

class SimpleGCN(nn.Module):
    """Simple Graph Convolutional Network for node classification"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(SimpleGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Create layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(BasicGCNLayer(input_dim, hidden_dim, activation=F.relu))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(BasicGCNLayer(hidden_dim, hidden_dim, activation=F.relu))

        # Output layer (no activation for classification)
        self.layers.append(BasicGCNLayer(hidden_dim, output_dim))

    def forward(self, node_features, adjacency_matrix):
        x = node_features
        x = self.dropout(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, adjacency_matrix)
            if i < len(self.layers) - 1:  # Don't dropout on last layer
                x = self.dropout(x)

        return F.log_softmax(x, dim=1)

# Test GCN implementation
def test_basic_gcn():
    # Create synthetic data
    num_nodes = 20
    input_dim = 16
    hidden_dim = 32
    output_dim = 7
    num_classes = 3

    # Generate random node features
    node_features = torch.randn(num_nodes, input_dim)

    # Generate random adjacency matrix
    adjacency_matrix = torch.randn(num_nodes, num_nodes)
    adjacency_matrix = (adjacency_matrix > 0.5).float()  # Make it binary

    # Create GCN model
    gcn = SimpleGCN(input_dim, hidden_dim, output_dim)

    # Forward pass
    output = gcn(node_features, adjacency_matrix)

    print(f"Input shape: {node_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in gcn.parameters())}")

test_basic_gcn()
```

### Exercise 4: GCN with Edge Features

**Task**: Extend GCN to handle edge features (edge-weighted graphs).

```python
class EdgeFeatureGCNLayer(nn.Module):
    """GCN Layer that incorporates edge features"""

    def __init__(self, input_dim, output_dim, edge_dim, bias=True):
        super(EdgeFeatureGCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim

        # Learnable parameters
        self.node_weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.edge_weight = nn.Parameter(torch.FloatTensor(edge_dim, output_dim))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.node_weight)
        nn.init.xavier_uniform_(self.edge_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, node_features, adjacency_matrix, edge_features):
        """
        Forward pass with edge features

        Args:
            node_features: (num_nodes, input_dim)
            adjacency_matrix: (num_nodes, num_nodes)
            edge_features: (num_nodes, num_nodes, edge_dim)

        Returns:
            (num_nodes, output_dim)
        """
        # Transform node features
        node_transformed = torch.mm(node_features, self.node_weight)

        # Create edge-weighted adjacency matrix
        edge_weights = torch.sum(edge_features * self.edge_weight.unsqueeze(0), dim=-1)
        weighted_adj = adjacency_matrix * edge_weights

        # Normalize
        degree = torch.sum(weighted_adj, dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_normalized = torch.mm(torch.mm(D_inv_sqrt, weighted_adj), D_inv_sqrt)

        # Aggregate messages
        output = torch.mm(adj_normalized, node_transformed)

        if self.bias is not None:
            output = output + self.bias

        return F.relu(output)

def test_edge_feature_gcn():
    """Test GCN with edge features"""
    num_nodes = 10
    input_dim = 8
    edge_dim = 4
    output_dim = 16

    # Create data
    node_features = torch.randn(num_nodes, input_dim)
    adjacency_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    edge_features = torch.randn(num_nodes, num_nodes, edge_dim)

    # Create model
    model = EdgeFeatureGCNLayer(input_dim, output_dim, edge_dim)

    # Forward pass
    output = model(node_features, adjacency_matrix, edge_features)

    print(f"Edge Feature GCN output shape: {output.shape}")

test_edge_feature_gcn()
```

## Graph Attention Networks Implementation {#graph-attention-networks-implementation}

### Exercise 5: Single-Head Graph Attention

**Task**: Implement single-head graph attention mechanism.

```python
class GraphAttentionLayer(nn.Module):
    """Single-head Graph Attention Network Layer"""

    def __init__(self, input_dim, output_dim, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha  # LeakyReLU negative slope

        # Attention mechanism parameters
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.a = nn.Parameter(torch.FloatTensor(2 * output_dim, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, node_features, adjacency_matrix):
        """
        Forward pass of single-head attention

        Args:
            node_features: (num_nodes, input_dim)
            adjacency_matrix: (num_nodes, num_nodes) binary matrix

        Returns:
            (num_nodes, output_dim)
        """
        num_nodes = node_features.size(0)

        # Linear transformation
        h = torch.mm(node_features, self.W)  # (num_nodes, output_dim)

        # Prepare attention mechanism
        # Repeat h for each node to compute pairwise attention
        h_repeated = h.unsqueeze(1).repeat(1, num_nodes, 1)  # (num_nodes, num_nodes, output_dim)
        h_concatenated = torch.cat([h_repeated, h_repeated.transpose(0, 1)], dim=2)  # (num_nodes, num_nodes, 2*output_dim)

        # Compute attention coefficients
        e = F.leaky_relu(torch.mm(h_concatenated.view(-1, 2 * self.output_dim), self.a), self.alpha)
        e = e.view(num_nodes, num_nodes)  # (num_nodes, num_nodes)

        # Set attention to -inf for non-edges
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adjacency_matrix > 0, e, zero_vec)

        # Normalize attention coefficients (softmax over neighbors)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # Apply attention to node features
        h_prime = torch.mm(attention, h)

        return F.elu(h_prime)

def test_graph_attention_layer():
    """Test single-head graph attention"""
    num_nodes = 8
    input_dim = 6
    output_dim = 12

    # Create data
    node_features = torch.randn(num_nodes, input_dim)
    adjacency_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()

    # Ensure diagonal is 1 (self-loops)
    adjacency_matrix.fill_diagonal_(1)

    # Create attention layer
    gat_layer = GraphAttentionLayer(input_dim, output_dim)

    # Forward pass
    output = gat_layer(node_features, adjacency_matrix)

    print(f"GAT Layer output shape: {output.shape}")

    # Verify attention weights sum to 1 for each node
    print("Attention weights validation:")
    for i in range(min(3, num_nodes)):  # Check first 3 nodes
        attention_row = torch.where(adjacency_matrix[i] > 0,
                                   F.softmax(gat_layer.a.t() @ torch.cat([output[i], output[i]], dim=0), dim=0),
                                   torch.zeros(num_nodes))
        print(f"Node {i} attention sum: {attention_row.sum().item():.3f}")

test_graph_attention_layer()
```

### Exercise 6: Multi-Head Graph Attention

**Task**: Implement multi-head attention to capture different types of relationships.

```python
class MultiHeadGraphAttention(nn.Module):
    """Multi-head Graph Attention Network"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, dropout=0.1, alpha=0.2):
        super(MultiHeadGraphAttention, self).__init__()
        self.num_heads = num_heads
        self.output_dim_per_head = output_dim // num_heads

        self.layers = nn.ModuleList([
            GraphAttentionLayer(input_dim, self.output_dim_per_head, dropout, alpha)
            for _ in range(num_heads)
        ])

        self.output_linear = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha

    def forward(self, node_features, adjacency_matrix):
        """
        Multi-head attention forward pass

        Args:
            node_features: (num_nodes, input_dim)
            adjacency_matrix: (num_nodes, num_nodes)

        Returns:
            (num_nodes, output_dim)
        """
        # Apply each attention head
        head_outputs = []
        for layer in self.layers:
            head_output = layer(node_features, adjacency_matrix)
            head_outputs.append(head_output)

        # Concatenate head outputs
        concatenated = torch.cat(head_outputs, dim=1)

        # Output transformation
        output = self.output_linear(concatenated)
        output = self.dropout(output)

        return output

class AdvancedGAT(nn.Module):
    """Advanced Graph Attention Network with residual connections"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, num_layers=2, dropout=0.1):
        super(AdvancedGAT, self).__init__()

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Attention layers
        self.attention_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer: input_dim -> hidden_dim
                self.attention_layers.append(
                    MultiHeadGraphAttention(hidden_dim, hidden_dim, hidden_dim, num_heads, dropout)
                )
            elif i == num_layers - 1:
                # Last layer: hidden_dim -> output_dim
                self.attention_layers.append(
                    MultiHeadGraphAttention(hidden_dim, hidden_dim, output_dim, num_heads, dropout)
                )
            else:
                # Middle layers: hidden_dim -> hidden_dim
                self.attention_layers.append(
                    MultiHeadGraphAttention(hidden_dim, hidden_dim, hidden_dim, num_heads, dropout)
                )

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
        ])

    def forward(self, node_features, adjacency_matrix):
        x = self.input_projection(node_features)
        x = self.dropout(x)

        for i, layer in enumerate(self.attention_layers):
            # Apply attention
            x_new = layer(x, adjacency_matrix)

            # Residual connection for all but last layer
            if i < len(self.attention_layers) - 1:
                x_new = x_new + x  # Residual connection
                x_new = F.elu(x_new)
                x_new = self.layer_norms[i](x_new)
                x_new = self.dropout(x_new)

            x = x_new

        return x

def test_multi_head_gat():
    """Test multi-head GAT implementation"""
    num_nodes = 12
    input_dim = 8
    hidden_dim = 64
    output_dim = 32
    num_heads = 4

    # Create data
    node_features = torch.randn(num_nodes, input_dim)
    adjacency_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    adjacency_matrix.fill_diagonal_(1)

    # Create model
    gat = AdvancedGAT(input_dim, hidden_dim, output_dim, num_heads)

    # Forward pass
    output = gat(node_features, adjacency_matrix)

    print(f"Multi-head GAT output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in gat.parameters())}")

test_multi_head_gat()
```

## Message Passing Neural Networks {#message-passing-neural-networks}

### Exercise 7: General Message Passing Framework

**Task**: Implement a flexible message passing framework.

```python
class MessagePassingLayer(nn.Module):
    """General Message Passing Neural Network Layer"""

    def __init__(self, node_dim, edge_dim, hidden_dim, message_type='add', update_type='gru'):
        super(MessagePassingLayer, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.message_type = message_type
        self.update_type = update_type

        # Message function components
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update function
        if update_type == 'gru':
            self.update_gru = nn.GRU(hidden_dim, node_dim, batch_first=True)
        elif update_type == 'mlp':
            self.update_mlp = nn.Sequential(
                nn.Linear(node_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, node_dim)
            )

        # Output projection
        self.output_projection = nn.Linear(node_dim, hidden_dim)

    def message_function(self, h_i, h_j, e_ij):
        """
        Compute message from node j to node i

        Args:
            h_i: Features of node i (node_dim,)
            h_j: Features of node j (node_dim,)
            e_ij: Edge features between i and j (edge_dim,)

        Returns:
            Message tensor (hidden_dim,)
        """
        # Concatenate node features and edge features
        message_input = torch.cat([h_i, h_j, e_ij], dim=-1)
        message = self.message_mlp(message_input)
        return message

    def aggregate_messages(self, messages, aggregation_type='sum'):
        """
        Aggregate messages from all neighbors

        Args:
            messages: Tensor of shape (num_neighbors, hidden_dim)
            aggregation_type: 'sum', 'mean', 'max', 'attention'

        Returns:
            Aggregated message (hidden_dim,)
        """
        if aggregation_type == 'sum':
            return torch.sum(messages, dim=0)
        elif aggregation_type == 'mean':
            return torch.mean(messages, dim=0)
        elif aggregation_type == 'max':
            return torch.max(messages, dim=0)[0]
        elif aggregation_type == 'attention':
            # Simple attention aggregation
            attention_weights = F.softmax(torch.mv(messages, self.attention_vector), dim=0)
            return torch.sum(attention_weights.unsqueeze(1) * messages, dim=0)

    def update_function(self, h_i, aggregated_messages):
        """
        Update node i representation

        Args:
            h_i: Current representation of node i (node_dim,)
            aggregated_messages: Aggregated messages from neighbors (hidden_dim,)

        Returns:
            Updated representation (node_dim,)
        """
        if self.update_type == 'gru':
            # GRU update: h_i' = GRU(h_i, aggregated_messages)
            _, hidden = self.update_gru(aggregated_messages.unsqueeze(0), h_i.unsqueeze(0))
            return hidden.squeeze(0)
        elif self.update_type == 'mlp':
            # MLP update: h_i' = MLP([h_i; aggregated_messages])
            combined = torch.cat([h_i, aggregated_messages], dim=-1)
            return self.update_mlp(combined)

    def forward(self, node_features, edge_index, edge_features):
        """
        Message passing forward pass

        Args:
            node_features: (num_nodes, node_dim)
            edge_index: (2, num_edges) edge list
            edge_features: (num_edges, edge_dim)

        Returns:
            Updated node features (num_nodes, hidden_dim)
        """
        num_nodes = node_features.size(0)
        num_edges = edge_index.size(1)

        # Prepare edge list and features
        edge_list = edge_index.t()  # (num_edges, 2)

        # Compute messages for each edge
        messages = []
        message_nodes = []  # Track which node receives each message

        for edge_idx in range(num_edges):
            i, j = edge_list[edge_idx]
            h_i = node_features[i]
            h_j = node_features[j]
            e_ij = edge_features[edge_idx]

            # Message from j to i
            message = self.message_function(h_i, h_j, e_ij)
            messages.append(message)
            message_nodes.append(i)

        # Stack messages
        messages = torch.stack(messages)  # (num_edges, hidden_dim)
        message_nodes = torch.tensor(message_nodes)  # (num_edges,)

        # Aggregate messages for each node
        updated_features = torch.zeros_like(node_features)

        for node in range(num_nodes):
            # Get messages for this node
            node_mask = message_nodes == node
            node_messages = messages[node_mask]

            if len(node_messages) > 0:
                # Aggregate messages
                aggregated = self.aggregate_messages(node_messages, 'sum')
                # Update node representation
                updated_features[node] = self.update_function(
                    node_features[node], aggregated
                )
            else:
                # No neighbors, keep original representation
                updated_features[node] = node_features[node]

        # Project to output dimension
        output = self.output_projection(updated_features)
        return F.relu(output)

def test_message_passing():
    """Test message passing implementation"""
    num_nodes = 6
    node_dim = 8
    edge_dim = 4
    hidden_dim = 16

    # Create simple graph: triangle + extra nodes
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5)]
    edge_index = torch.tensor(edges).t().contiguous()  # (2, num_edges)

    # Create features
    node_features = torch.randn(num_nodes, node_dim)
    edge_features = torch.randn(len(edges), edge_dim)

    # Create message passing layer
    mp_layer = MessagePassingLayer(node_dim, edge_dim, hidden_dim)

    # Forward pass
    output = mp_layer(node_features, edge_index, edge_features)

    print(f"Message Passing output shape: {output.shape}")
    print(f"Number of edges processed: {len(edges)}")

test_message_passing()
```

### Exercise 8: Specialized Message Passing Variants

**Task**: Implement specialized message passing variants for different graph types.

```python
class MolecularMessagePassing(nn.Module):
    """Specialized message passing for molecular graphs"""

    def __init__(self, atom_dim, bond_dim, hidden_dim):
        super(MolecularMessagePassing, self).__init__()

        # Atom representation updates
        self.atom_update = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Bond-aware message computation
        self.bond_message_mlp = nn.Sequential(
            nn.Linear(2 * atom_dim + bond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Distance-based attention (for 3D molecular structures)
        self.distance_attention = nn.Linear(3, 1)  # 3D coordinates -> attention weight

    def forward(self, atom_features, bond_index, bond_features, coordinates=None):
        """
        Molecular message passing with bond and distance awareness

        Args:
            atom_features: (num_atoms, atom_dim)
            bond_index: (2, num_bonds)
            bond_features: (num_bonds, bond_dim)
            coordinates: (num_atoms, 3) optional 3D coordinates

        Returns:
            Updated atom features (num_atoms, hidden_dim)
        """
        num_atoms = atom_features.size(0)
        num_bonds = bond_index.size(1)

        # Compute bond-aware messages
        messages = []
        receiver_atoms = []

        for bond_idx in range(num_bonds):
            donor, receiver = bond_index[:, bond_idx]

            donor_features = atom_features[donor]
            receiver_features = atom_features[receiver]
            bond_features_batch = bond_features[bond_idx]

            # Create message input
            message_input = torch.cat([donor_features, receiver_features, bond_features_batch])
            message = self.bond_message_mlp(message_input)

            messages.append(message)
            receiver_atoms.append(receiver)

        # Aggregate messages per atom
        aggregated_messages = torch.zeros(num_atoms, messages[0].size(0))

        for msg, receiver in zip(messages, receiver_atoms):
            aggregated_messages[receiver] += msg

        # Distance-based weighting (if coordinates provided)
        if coordinates is not None:
            # This would require more complex implementation
            # For simplicity, we'll just add distance bias
            pass

        # Update atom representations using GRU
        updated_features, _ = self.atom_update(aggregated_messages.unsqueeze(0))

        return updated_features.squeeze(0)

class SocialNetworkMessagePassing(nn.Module):
    """Specialized message passing for social networks"""

    def __init__(self, user_dim, relation_dim, hidden_dim, num_relation_types=5):
        super(SocialNetworkMessagePassing, self).__init__()

        self.num_relation_types = num_relation_types

        # Different message functions for different relation types
        self.relation_message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * user_dim + relation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_relation_types)
        ])

        # Attention over relation types
        self.relation_attention = nn.Linear(hidden_dim * num_relation_types, num_relation_types)

        # User update function
        self.user_update = nn.Sequential(
            nn.Linear(user_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, user_features, relation_index, relation_types, relation_features):
        """
        Social network message passing with relation types

        Args:
            user_features: (num_users, user_dim)
            relation_index: (2, num_relations)
            relation_types: (num_relations,) relation type indices
            relation_features: (num_relations, relation_dim)

        Returns:
            Updated user features (num_users, hidden_dim)
        """
        num_users = user_features.size(0)
        num_relations = relation_index.size(1)

        # Compute messages for each relation type
        relation_messages = {t: [] for t in range(self.num_relation_types)}
        relation_receivers = {t: [] for t in range(self.num_relation_types)}

        for rel_idx in range(num_relations):
            follower, followed = relation_index[:, rel_idx]
            relation_type = relation_types[rel_idx]
            relation_feature = relation_features[rel_idx]

            # Compute message based on relation type
            follower_features = user_features[follower]
            followed_features = user_features[followed]

            message_input = torch.cat([follower_features, followed_features, relation_feature])
            message = self.relation_message_mlps[relation_type](message_input)

            relation_messages[relation_type].append(message)
            relation_receivers[relation_type].append(follower)

        # Aggregate messages within each relation type
        relation_aggregated = {}
        for rel_type in range(self.num_relation_types):
            if relation_messages[rel_type]:
                messages = torch.stack(relation_messages[rel_type])
                receivers = torch.tensor(relation_receivers[rel_type])

                # Sum aggregation
                aggregated = torch.zeros(num_users, messages.size(1))
                for msg, receiver in zip(messages, receivers):
                    aggregated[receiver] += msg

                relation_aggregated[rel_type] = aggregated
            else:
                relation_aggregated[rel_type] = torch.zeros(num_users, messages.size(1))

        # Concatenate all relation-type messages
        all_relation_messages = torch.cat([relation_aggregated[t] for t in range(self.num_relation_types)], dim=1)

        # Apply attention over relation types
        relation_weights = F.softmax(self.relation_attention(all_relation_messages), dim=1)
        weighted_messages = all_relation_messages * relation_weights

        # Update user features
        combined = torch.cat([user_features, weighted_messages], dim=1)
        updated_features = self.user_update(combined)

        return updated_features

def test_specialized_mp():
    """Test specialized message passing implementations"""

    # Test molecular message passing
    print("=== Testing Molecular Message Passing ===")
    num_atoms = 8
    atom_dim = 16
    bond_dim = 6
    hidden_dim = 32

    # Create molecular graph (benzene ring)
    bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6), (2, 7)]  # Ring + substituents
    bond_index = torch.tensor(bonds).t().contiguous()

    atom_features = torch.randn(num_atoms, atom_dim)
    bond_features = torch.randn(len(bonds), bond_dim)
    coordinates = torch.randn(num_atoms, 3)  # Random 3D coordinates

    mol_mp = MolecularMessagePassing(atom_dim, bond_dim, hidden_dim)
    mol_output = mol_mp(atom_features, bond_index, bond_features, coordinates)

    print(f"Molecular MP output shape: {mol_output.shape}")

    # Test social network message passing
    print("\n=== Testing Social Network Message Passing ===")
    num_users = 10
    user_dim = 12
    relation_dim = 4
    num_relation_types = 3

    # Create social network relations
    relations = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (6, 7), (7, 8), (8, 6)]
    relation_index = torch.tensor(relations).t().contiguous()
    relation_types = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])  # Friend, family, colleague
    relation_features = torch.randn(len(relations), relation_dim)

    user_features = torch.randn(num_users, user_dim)

    social_mp = SocialNetworkMessagePassing(user_dim, relation_dim, hidden_dim, num_relation_types)
    social_output = social_mp(user_features, relation_index, relation_types, relation_features)

    print(f"Social MP output shape: {social_output.shape}")

test_specialized_mp()
```

## Advanced GNN Architectures {#advanced-gnn-architectures}

### Exercise 9: GraphSAGE Implementation

**Task**: Implement GraphSAGE with sampling and aggregation.

```python
import random
from collections import defaultdict, deque

class GraphSAGESampler:
    """GraphSAGE neighbor sampler"""

    def __init__(self, graph, adj_list):
        self.graph = graph
        self.adj_list = adj_list

    def sample_neighbors(self, nodes, num_neighbors):
        """
        Sample neighbors for given nodes

        Args:
            nodes: List of node IDs
            num_neighbors: Maximum number of neighbors to sample

        Returns:
            Dictionary mapping node ID to sampled neighbors
        """
        sampled_neighbors = {}

        for node in nodes:
            neighbors = self.adj_list.get(node, [])
            if len(neighbors) <= num_neighbors:
                sampled_neighbors[node] = neighbors
            else:
                # Random sampling
                sampled_neighbors[node] = random.sample(neighbors, num_neighbors)

        return sampled_neighbors

    def sample_subgraph(self, start_nodes, num_hops, num_neighbors):
        """
        Sample subgraph starting from start_nodes

        Args:
            start_nodes: Starting nodes for sampling
            num_hops: Number of hops to sample
            num_neighbors: Max neighbors per hop

        Returns:
            Dictionary containing sampled nodes and edges
        """
        # BFS sampling
        current_nodes = start_nodes
        all_nodes = set(start_nodes)

        edges = set()

        for hop in range(num_hops):
            next_nodes = []
            for node in current_nodes:
                # Sample neighbors
                neighbors = self.adj_list.get(node, [])
                if len(neighbors) > num_neighbors:
                    neighbors = random.sample(neighbors, num_neighbors)

                for neighbor in neighbors:
                    if neighbor not in all_nodes:
                        all_nodes.add(neighbor)
                        next_nodes.append(neighbor)
                    edges.add((min(node, neighbor), max(node, neighbor)))

            current_nodes = next_nodes
            if not current_nodes:
                break

        return {
            'nodes': list(all_nodes),
            'edges': list(edges)
        }

class GraphSAGELayer(nn.Module):
    """GraphSAGE layer with different aggregation functions"""

    def __init__(self, input_dim, output_dim, aggregator='mean', dropout=0.5):
        super(GraphSAGELayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator = aggregator
        self.dropout = nn.Dropout(dropout)

        # Self transformation
        self.self_transform = nn.Linear(input_dim, output_dim)

        # Neighbor transformation
        self.neighbor_transform = nn.Linear(input_dim, output_dim)

        # Aggregation function
        if aggregator == 'mean':
            self.aggregate = self.mean_aggregate
        elif aggregator == 'max':
            self.aggregate = self.max_aggregate
        elif aggregator == 'lstm':
            self.lstm = nn.LSTM(output_dim, output_dim, batch_first=True)
            self.aggregate = self.lstm_aggregate
        elif aggregator == 'pool':
            self.pool_mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
            self.aggregate = self.pool_aggregate

    def mean_aggregate(self, neighbor_features):
        """Mean aggregation"""
        return torch.mean(neighbor_features, dim=0)

    def max_aggregate(self, neighbor_features):
        """Max pooling aggregation"""
        return torch.max(neighbor_features, dim=0)[0]

    def lstm_aggregate(self, neighbor_features):
        """LSTM aggregation"""
        # Sort neighbor features for LSTM
        sorted_features, _ = torch.sort(neighbor_features, dim=0)
        lstm_out, _ = self.lstm(sorted_features.unsqueeze(0))
        return lstm_out.squeeze(0)[-1]  # Take last hidden state

    def pool_aggregate(self, neighbor_features):
        """Pooling aggregation"""
        pooled = self.pool_mlp(neighbor_features)
        return torch.max(pooled, dim=0)[0]

    def forward(self, node_features, neighbor_features):
        """
        GraphSAGE forward pass

        Args:
            node_features: (num_nodes, input_dim) - self features
            neighbor_features: (num_nodes, num_neighbors, input_dim) - neighbor features

        Returns:
            (num_nodes, output_dim) - updated node features
        """
        # Transform self features
        self_repr = self.self_transform(node_features)
        self_repr = F.relu(self_repr)

        # Aggregate neighbor features
        neighbor_repr = torch.zeros_like(self_repr)

        for i in range(node_features.size(0)):
            if neighbor_features[i].size(0) > 0:
                # Transform neighbor features
                neighbor_transformed = self.neighbor_transform(neighbor_features[i])
                neighbor_transformed = F.relu(neighbor_transformed)

                # Aggregate
                neighbor_repr[i] = self.aggregate(neighbor_transformed)

        # Combine self and neighbor representations
        output = self_repr + neighbor_repr
        output = self.dropout(output)

        return output

class GraphSAGE(nn.Module):
    """Complete GraphSAGE model"""

    def __init__(self, input_dim, hidden_dims, output_dim, aggregator='mean', dropout=0.5):
        super(GraphSAGE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Create layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(GraphSAGELayer(input_dim, hidden_dims[0], aggregator, dropout))

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.layers.append(GraphSAGELayer(hidden_dims[i-1], hidden_dims[i], aggregator, dropout))

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, neighbor_features_list):
        """
        Forward pass with sampled subgraphs

        Args:
            node_features: (num_nodes, input_dim)
            neighbor_features_list: List of neighbor feature tensors for each layer

        Returns:
            (num_nodes, output_dim)
        """
        x = node_features

        # Process each layer
        for i, (layer, neighbor_features) in enumerate(zip(self.layers, neighbor_features_list)):
            x = layer(x, neighbor_features)

        # Final classification
        x = self.dropout(x)
        output = self.output_layer(x)

        return F.log_softmax(output, dim=1)

def test_graphsage():
    """Test GraphSAGE implementation"""
    print("=== Testing GraphSAGE ===")

    # Create synthetic graph
    num_nodes = 20
    input_dim = 16
    hidden_dims = [32, 64]
    output_dim = 7
    num_classes = 3

    # Create adjacency list
    adj_list = defaultdict(list)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 6), (6, 7), (7, 8), (8, 5)]

    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    # Create sampler
    sampler = GraphSAGESampler(None, adj_list)

    # Sample subgraph
    start_nodes = [0, 5]
    subgraph = sampler.sample_subgraph(start_nodes, num_hops=2, num_neighbors=3)
    sampled_nodes = subgraph['nodes']

    print(f"Sampled {len(sampled_nodes)} nodes: {sampled_nodes}")

    # Create feature tensors for sampled nodes
    node_features = torch.randn(len(sampled_nodes), input_dim)

    # Create neighbor features for each layer
    neighbor_features_list = []
    for layer_idx in range(len(hidden_dims)):
        layer_neighbors = []
        for node in sampled_nodes:
            # Sample neighbors for this layer
            neighbors = sampler.sample_neighbors([node], 3)[node]
            neighbor_indices = [sampled_nodes.index(n) for n in neighbors if n in sampled_nodes]

            if neighbor_indices:
                neighbor_features = torch.randn(len(neighbor_indices), input_dim)
            else:
                neighbor_features = torch.zeros(0, input_dim)

            layer_neighbors.append(neighbor_features)

        neighbor_features_list.append(layer_neighbors)

    # Create GraphSAGE model
    modelsage = GraphSAGE(input_dim, hidden_dims, output_dim)

    # Forward pass
    output = modelsage(node_features, neighbor_features_list)

    print(f"GraphSAGE output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in modelsage.parameters())}")

test_graphsage()
```

### Exercise 10: Graph Transformer Architecture

**Task**: Implement Graph Transformer with global attention.

```python
import math

class GraphPositionalEncoding(nn.Module):
    """Positional encoding for graph nodes"""

    def __init__(self, hidden_dim, max_seq_length=1000):
        super(GraphPositionalEncoding, self).__init__()

        self.hidden_dim = hidden_dim

        # Compute positional encoding matrix
        pe = torch.zeros(max_seq_length, hidden_dim)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() *
                           -(math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, node_ids):
        """
        Add positional encoding to node embeddings

        Args:
            node_ids: (num_nodes,) node indices

        Returns:
            (num_nodes, hidden_dim) positional encodings
        """
        return self.pe[node_ids]

class GlobalGraphAttention(nn.Module):
    """Global attention mechanism for graph transformers"""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(GlobalGraphAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, node_features, attention_mask=None):
        """
        Global attention over all nodes

        Args:
            node_features: (num_nodes, hidden_dim)
            attention_mask: (num_nodes, num_nodes) optional mask

        Returns:
            (num_nodes, hidden_dim) attended features
        """
        batch_size, num_nodes, hidden_dim = node_features.size()

        # Project to Q, K, V
        Q = self.q_proj(node_features)  # (batch_size, num_nodes, hidden_dim)
        K = self.k_proj(node_features)  # (batch_size, num_nodes, hidden_dim)
        V = self.v_proj(node_features)  # (batch_size, num_nodes, hidden_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, num_nodes, num_nodes)

        # Apply mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)  # (batch_size, num_heads, num_nodes, head_dim)

        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, num_nodes, hidden_dim
        )

        # Output projection
        output = self.out_proj(attended_values)

        return output

class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer with global attention and graph structure awareness"""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, alpha=0.01):
        super(GraphTransformerLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Global attention
        self.global_attention = GlobalGraphAttention(hidden_dim, num_heads, dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha  # Residual connection scale

    def forward(self, node_features, adjacency_matrix=None):
        """
        Graph Transformer layer forward pass

        Args:
            node_features: (num_nodes, hidden_dim)
            adjacency_matrix: (num_nodes, num_nodes) optional for local structure

        Returns:
            (num_nodes, hidden_dim) updated features
        """
        # Add batch dimension for attention
        node_features_batch = node_features.unsqueeze(0)  # (1, num_nodes, hidden_dim)

        # Global self-attention
        attended_features = self.global_attention(node_features_batch)
        attended_features = attended_features.squeeze(0)  # Remove batch dimension

        # First residual connection and layer norm
        output1 = self.norm1(node_features + self.alpha * self.dropout(attended_features))

        # Feed-forward network
        ffn_output = self.ffn(output1)

        # Second residual connection and layer norm
        output2 = self.norm2(output1 + self.alpha * self.dropout(ffn_output))

        return output2

class GraphTransformer(nn.Module):
    """Complete Graph Transformer model"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, num_heads=8, dropout=0.1):
        super(GraphTransformer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = GraphPositionalEncoding(hidden_dim)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, node_ids, adjacency_matrix=None):
        """
        Graph Transformer forward pass

        Args:
            node_features: (num_nodes, input_dim)
            node_ids: (num_nodes,) node identifiers for positional encoding
            adjacency_matrix: (num_nodes, num_nodes) optional graph structure

        Returns:
            (num_nodes, output_dim) node predictions
        """
        # Input projection
        x = self.input_projection(node_features)

        # Add positional encoding
        pos_enc = self.pos_encoding(node_ids)
        x = x + pos_enc

        # Apply dropout
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, adjacency_matrix)

        # Output projection
        output = self.output_projection(x)

        return F.log_softmax(output, dim=1)

def test_graph_transformer():
    """Test Graph Transformer implementation"""
    print("=== Testing Graph Transformer ===")

    # Create synthetic data
    num_nodes = 15
    input_dim = 8
    hidden_dim = 64
    output_dim = 5
    num_layers = 4
    num_heads = 8

    # Create data
    node_features = torch.randn(num_nodes, input_dim)
    node_ids = torch.arange(num_nodes)
    adjacency_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    adjacency_matrix.fill_diagonal_(1)  # Add self-loops

    # Create model
    graph_transformer = GraphTransformer(
        input_dim, hidden_dim, output_dim, num_layers, num_heads
    )

    # Forward pass
    output = graph_transformer(node_features, node_ids, adjacency_matrix)

    print(f"Graph Transformer output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in graph_transformer.parameters())}")

    # Test positional encoding
    pos_enc = graph_transformer.pos_encoding(node_ids)
    print(f"Positional encoding shape: {pos_enc.shape}")

test_graph_transformer()
```

## Real-world Applications {#real-world-applications}

### Exercise 11: Social Network Analysis

**Task**: Implement social network analysis using GNNs for community detection and link prediction.

```python
import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

class SocialNetworkAnalyzer:
    """Social Network Analysis using Graph Neural Networks"""

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # User embedding network
        self.user_embedding_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Social GNN for link prediction
        self.social_gcn = SimpleGCN(input_dim, hidden_dim, output_dim, num_layers=3)

        # Community detection GNN
        self.community_gcn = SimpleGCN(input_dim, hidden_dim, 10, num_layers=3)  # 10 communities

        # Link prediction decoder
        self.link_decoder = nn.Sequential(
            nn.Linear(2 * output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def create_user_features(self, user_profiles):
        """Create feature vectors for users based on profiles"""
        # Simple feature engineering from user profiles
        features = []
        for profile in user_profiles:
            feature_vector = [
                profile.get('age', 30) / 100,  # Normalized age
                profile.get('posts_count', 100) / 1000,  # Normalized activity
                profile.get('friends_count', 200) / 1000,  # Normalized social connectivity
                len(profile.get('interests', [])) / 10,  # Interest diversity
                profile.get('engagement_score', 0.5),  # Engagement level
            ]
            features.append(feature_vector)

        # Pad or truncate to input_dim
        feature_matrix = np.array(features)
        if feature_matrix.shape[1] < self.input_dim:
            padding = np.zeros((feature_matrix.shape[0], self.input_dim - feature_matrix.shape[1]))
            feature_matrix = np.hstack([feature_matrix, padding])
        elif feature_matrix.shape[1] > self.input_dim:
            feature_matrix = feature_matrix[:, :self.input_dim]

        return torch.tensor(feature_matrix, dtype=torch.float32)

    def create_social_graph(self, user_connections):
        """Create adjacency matrix from user connections"""
        num_users = len(user_connections)
        adj_matrix = np.zeros((num_users, num_users))

        for connections in user_connections:
            for connection in connections:
                adj_matrix[connections['user_id']][connection] = 1
                adj_matrix[connection][connections['user_id']] = 1  # Undirected

        return torch.tensor(adj_matrix, dtype=torch.float32)

    def predict_links(self, node_features, adjacency_matrix, test_edges):
        """Predict probability of edges existing between node pairs"""
        # Get node embeddings
        embeddings = self.social_gcn(node_features, adjacency_matrix)

        predictions = []
        for user_i, user_j in test_edges:
            # Combine embeddings
            combined = torch.cat([embeddings[user_i], embeddings[user_j]])

            # Predict link probability
            prob = self.link_decoder(combined)
            predictions.append(prob.item())

        return np.array(predictions)

    def detect_communities(self, node_features, adjacency_matrix, threshold=0.5):
        """Detect communities in the social network"""
        # Get community probabilities
        community_probs = self.community_gcn(node_features, adjacency_matrix)

        # Assign each node to most likely community
        communities = torch.argmax(community_probs, dim=1)

        # Group nodes by community
        community_groups = {}
        for i, community in enumerate(communities):
            if community.item() not in community_groups:
                community_groups[community.item()] = []
            community_groups[community.item()].append(i)

        return community_groups, community_probs

def simulate_social_network():
    """Create a simulated social network for testing"""
    num_users = 100

    # Generate random user profiles
    user_profiles = []
    for i in range(num_users):
        profile = {
            'user_id': i,
            'age': np.random.randint(18, 65),
            'posts_count': np.random.randint(10, 1000),
            'friends_count': np.random.randint(5, 500),
            'interests': np.random.choice(['sports', 'music', 'tech', 'food', 'travel'],
                                        size=np.random.randint(1, 5)),
            'engagement_score': np.random.random()
        }
        user_profiles.append(profile)

    # Generate connections (social network edges)
    user_connections = {i: [] for i in range(num_users)}

    # Create community structure
    communities = np.random.choice(5, num_users)  # 5 communities

    for i in range(num_users):
        # Within-community connections (higher probability)
        community_members = np.where(communities == communities[i])[0]
        within_community = [j for j in community_members if j != i]

        for j in within_community:
            if np.random.random() < 0.1:  # 10% within-community connection rate
                user_connections[i].append(j)

        # Cross-community connections (lower probability)
        other_communities = np.where(communities != communities[i])[0]
        for j in other_communities:
            if np.random.random() < 0.02:  # 2% cross-community connection rate
                user_connections[i].append(j)

    # Remove duplicates
    for i in user_connections:
        user_connections[i] = list(set(user_connections[i]))

    return user_profiles, user_connections, communities

def test_social_network_analysis():
    """Test social network analysis with GNNs"""
    print("=== Testing Social Network Analysis ===")

    # Create simulated social network
    user_profiles, user_connections, true_communities = simulate_social_network()

    # Create analyzer
    analyzer = SocialNetworkAnalyzer()

    # Create user features
    user_features = analyzer.create_user_features(user_profiles)
    print(f"User features shape: {user_features.shape}")

    # Create adjacency matrix
    adj_matrix = analyzer.create_social_graph(user_connections)
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Total connections: {torch.sum(adj_matrix).item() / 2:.0f}")

    # Test community detection
    detected_communities, community_probs = analyzer.detect_communities(user_features, adj_matrix)

    print(f"Detected {len(detected_communities)} communities:")
    for comm_id, members in detected_communities.items():
        print(f"  Community {comm_id}: {len(members)} members")

    # Test link prediction
    # Create test edges (some existing, some non-existing)
    all_possible_edges = []
    for i in range(len(user_profiles)):
        for j in range(i+1, len(user_profiles)):
            all_possible_edges.append((i, j))

    # Split into train/test
    train_edges, test_edges = train_test_split(all_possible_edges, test_size=0.2, random_state=42)

    # Remove some train edges to simulate missing connections
    remove_fraction = 0.3
    np.random.seed(42)
    remove_indices = np.random.choice(len(train_edges), int(len(train_edges) * remove_fraction), replace=False)
    missing_train_edges = [train_edges[i] for i in remove_indices]

    # Predict link probabilities
    link_probs = analyzer.predict_links(user_features, adj_matrix, missing_train_edges)

    print(f"\nLink prediction on {len(missing_train_edges)} test edges:")
    print(f"Average link probability: {np.mean(link_probs):.3f}")
    print(f"High probability links (>0.7): {np.sum(np.array(link_probs) > 0.7)}")

test_social_network_analysis()
```

### Exercise 12: Molecular Property Prediction

**Task**: Implement molecular property prediction using Graph Neural Networks.

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import matplotlib.pyplot as plt

class MolecularGraphConverter:
    """Convert molecular graphs to GNN-compatible format"""

    def __init__(self):
        # Atom feature dimensions
        self.atom_features = {
            'atomic_num': 118,  # All possible elements
            'degree': 11,       # 0-10
            'formal_charge': 7, # -3 to +3
            'num_h': 7,         # 0-6
            'num_radical_e': 7, # 0-6
            'hybridization': 6, # sp, sp2, sp3, sp3d, sp3d2, other
            'is_aromatic': 2,   # True/False
            'is_in_ring': 2     # True/False
        }

        # Bond feature dimensions
        self.bond_features = {
            'bond_type': 5,      # Single, Double, Triple, Aromatic, None
            'is_conjugated': 2,  # True/False
            'is_in_ring': 2,     # True/False
        }

    def atom_to_features(self, atom):
        """Convert RDKit atom to feature vector"""
        features = []

        # Atomic number (one-hot encoded)
        atomic_num = np.zeros(118)
        atomic_num[atom.GetAtomicNum() - 1] = 1
        features.extend(atomic_num)

        # Degree
        degree = np.zeros(11)
        degree[min(atom.GetDegree(), 10)] = 1
        features.extend(degree)

        # Formal charge
        formal_charge = np.zeros(7)
        charge = atom.GetFormalCharge() + 3  # Shift to make positive
        charge = np.clip(charge, 0, 6)
        formal_charge[charge] = 1
        features.extend(formal_charge)

        # Number of hydrogens
        num_h = np.zeros(7)
        num_h[min(atom.GetNumImplicitHs(), 6)] = 1
        features.extend(num_h)

        # Radical electrons
        num_radical = np.zeros(7)
        num_radical[min(atom.GetNumRadicalElectrons(), 6)] = 1
        features.extend(num_radical)

        # Hybridization
        hybridization = np.zeros(6)
        hybrid_map = {
            Chem.HybridizationType.SP: 0,
            Chem.HybridizationType.SP2: 1,
            Chem.HybridizationType.SP3: 2,
            Chem.HybridizationType.SP3D: 3,
            Chem.HybridizationType.SP3D2: 4,
        }
        hybrid_val = hybrid_map.get(atom.GetHybridization(), 5)
        hybridization[hybrid_val] = 1
        features.extend(hybridization)

        # Aromatic
        is_aromatic = np.zeros(2)
        is_aromatic[int(atom.GetIsAromatic())] = 1
        features.extend(is_aromatic)

        # In ring
        is_in_ring = np.zeros(2)
        is_in_ring[int(atom.IsInRing())] = 1
        features.extend(is_in_ring)

        return np.array(features)

    def bond_to_features(self, bond):
        """Convert RDKit bond to feature vector"""
        features = []

        # Bond type (one-hot encoded)
        bond_type = np.zeros(5)
        bond_type_map = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3,
        }
        bond_type_val = bond_type_map.get(bond.GetBondType(), 4)
        bond_type[bond_type_val] = 1
        features.extend(bond_type)

        # Is conjugated
        is_conjugated = np.zeros(2)
        is_conjugated[int(bond.GetIsConjugated())] = 1
        features.extend(is_conjugated)

        # Is in ring
        is_in_ring = np.zeros(2)
        is_in_ring[int(bond.IsInRing())] = 1
        features.extend(is_in_ring)

        return np.array(features)

    def molecule_to_graph(self, mol):
        """Convert RDKit molecule to graph format"""
        if mol is None:
            return None

        num_atoms = mol.GetNumAtoms()

        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.atom_to_features(atom))
        atom_features = np.array(atom_features)

        # Get bond features and edge list
        edge_features = []
        edge_list = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_list.append([i, j])
            edge_features.append(self.bond_to_features(bond))

        if edge_list:
            edge_list = np.array(edge_list).T  # (2, num_edges)
            edge_features = np.array(edge_features)
        else:
            edge_list = np.array([[], []])  # No edges
            edge_features = np.array([]).reshape(0, len(self.bond_features))

        return {
            'atom_features': atom_features,
            'edge_list': edge_list,
            'edge_features': edge_features,
            'num_atoms': num_atoms
        }

class MolecularGNN(nn.Module):
    """Graph Neural Network for molecular property prediction"""

    def __init__(self, atom_dim, bond_dim, hidden_dim, output_dim, num_layers=3):
        super(MolecularGNN, self).__init__()

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim

        # Message passing layers
        self.message_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.message_layers.append(
                MessagePassingLayer(hidden_dim, bond_dim, hidden_dim)
            )

        # Global readouts
        self.global_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = nn.Dropout(0.3)

    def forward(self, batch_data):
        """
        Forward pass for molecular batch

        Args:
            batch_data: Dictionary containing:
                - atom_features: (num_nodes, atom_dim)
                - edge_list: (2, num_edges)
                - edge_features: (num_edges, bond_dim)
                - mol_index: (num_nodes,) molecule index for each atom

        Returns:
            (batch_size, output_dim) property predictions
        """
        atom_features = batch_data['atom_features']
        edge_list = batch_data['edge_list']
        edge_features = batch_data['edge_features']
        mol_index = batch_data['mol_index']

        # Apply message passing layers
        node_representations = atom_features
        for layer in self.message_layers:
            node_representations = layer(node_representations, edge_list, edge_features)
            node_representations = self.dropout(node_representations)

        # Global pooling (sum) per molecule
        batch_size = mol_index.max().item() + 1
        molecule_representations = torch.zeros(batch_size, self.hidden_dim)

        for mol_id in range(batch_size):
            mol_nodes = (mol_index == mol_id)
            if mol_nodes.any():
                molecule_representations[mol_id] = torch.sum(
                    node_representations[mol_nodes], dim=0
                )

        # Predict properties
        predictions = self.global_readout(molecule_representations)

        return predictions

def create_molecular_dataset():
    """Create a sample molecular dataset"""
    # Define sample molecules with SMILES
    molecules_smiles = [
        'CCO',  # Ethanol
        'CCN',  # Ethanamine
        'CC(=O)O',  # Acetic acid
        'CC(=O)C',  # Acetone
        'C1=CC=CC=C1',  # Benzene
        'CC1=CC=CC=C1',  # Toluene
        'C1=CC=CC=C1C(=O)O',  # Benzoic acid
        'CC(C)O',  # Isopropanol
        'CC(C)C',  # Isobutane
        'C1CCC1',  # Cyclobutane
        'C1=CC=CC=C1C(=O)C',  # Acetophenone
        'CC1=CC=CC=C1O',  # Phenol
        'C1=CC=C(C=C1)C(=O)O',  # Terephthalic acid
        'CC(C)(C)O',  # tert-Butanol
        'C1=CC=CC=C1C=CC=C1',  # Biphenyl
    ]

    # Create molecular graphs
    converter = MolecularGraphConverter()
    molecules = []

    for smiles in molecules_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            graph = converter.molecule_to_graph(mol)
            if graph is not None:
                molecules.append({
                    'smiles': smiles,
                    'graph': graph,
                    'properties': {
                        'molecular_weight': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'num_hbd': Descriptors.NumHDonors(mol),
                        'num_hba': Descriptors.NumHAcceptors(mol),
                        'tpsa': Descriptors.TPSA(mol),
                        'num_rot_bonds': Descriptors.NumRotatableBonds(mol)
                    }
                })

    return molecules

def collate_molecular_batch(molecules):
    """Collate molecular batch for GNN training"""
    # Find maximum sizes
    max_atoms = max(mol['graph']['num_atoms'] for mol in molecules)
    max_edges = max(mol['graph']['edge_list'].shape[1] for mol in molecules if mol['graph']['edge_list'].shape[1] > 0)

    # Initialize batch tensors
    atom_dim = molecules[0]['graph']['atom_features'].shape[1]
    bond_dim = molecules[0]['graph']['edge_features'].shape[1]

    batch_atom_features = torch.zeros(len(molecules) * max_atoms, atom_dim)
    batch_edge_list = torch.zeros(2, len(molecules) * max_edges, dtype=torch.long)
    batch_edge_features = torch.zeros(len(molecules) * max_edges, bond_dim)
    batch_mol_index = torch.zeros(len(molecules) * max_atoms, dtype=torch.long)
    batch_properties = []

    current_atom_offset = 0
    current_edge_offset = 0

    for mol_id, molecule in enumerate(molecules):
        graph = molecule['graph']
        num_atoms = graph['num_atoms']
        num_edges = graph['edge_list'].shape[1]

        # Add atom features
        batch_atom_features[current_atom_offset:current_atom_offset + num_atoms] = torch.tensor(
            graph['atom_features'], dtype=torch.float32
        )

        # Add edges (with offset)
        if num_edges > 0:
            batch_edge_list[:, current_edge_offset:current_edge_offset + num_edges] = (
                torch.tensor(graph['edge_list'], dtype=torch.long) + current_atom_offset
            )
            batch_edge_features[current_edge_offset:current_edge_offset + num_edges] = torch.tensor(
                graph['edge_features'], dtype=torch.float32
            )

        # Add molecule index
        batch_mol_index[current_atom_offset:current_atom_offset + num_atoms] = mol_id

        # Add properties
        batch_properties.append(list(molecule['properties'].values()))

        # Update offsets
        current_atom_offset += max_atoms
        current_edge_offset += max_edges

    return {
        'atom_features': batch_atom_features,
        'edge_list': batch_edge_list,
        'edge_features': batch_edge_features,
        'mol_index': batch_mol_index,
        'properties': torch.tensor(batch_properties, dtype=torch.float32)
    }

def test_molecular_prediction():
    """Test molecular property prediction"""
    print("=== Testing Molecular Property Prediction ===")

    # Create molecular dataset
    molecules = create_molecular_dataset()
    print(f"Created dataset with {len(molecules)} molecules")

    # Print some examples
    print("\nSample molecules:")
    for i, mol in enumerate(molecules[:5]):
        print(f"  {i+1}. {mol['smiles']} - MW: {mol['properties']['molecular_weight']:.1f}")

    # Create GNN model
    converter = MolecularGraphConverter()
    first_mol = molecules[0]['graph']
    atom_dim = first_mol['atom_features'].shape[1]
    bond_dim = first_mol['edge_features'].shape[1]
    hidden_dim = 64
    output_dim = 6  # 6 properties to predict

    model = MolecularGNN(atom_dim, bond_dim, hidden_dim, output_dim)

    print(f"\nModel architecture:")
    print(f"  Atom dimension: {atom_dim}")
    print(f"  Bond dimension: {bond_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Test forward pass
    batch_data = collate_molecular_batch(molecules[:5])

    print(f"\nBatch data shapes:")
    for key, value in batch_data.items():
        if key != 'properties':
            print(f"  {key}: {value.shape}")

    # Forward pass
    predictions = model(batch_data)
    true_properties = batch_data['properties']

    print(f"\nPrediction results (first 3 molecules):")
    property_names = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds']

    for i in range(min(3, len(molecules))):
        print(f"\nMolecule {i+1} ({molecules[i]['smiles']}):")
        for j, prop_name in enumerate(property_names):
            pred = predictions[i, j].item()
            true = true_properties[i, j].item()
            error = abs(pred - true)
            print(f"  {prop_name}: Pred={pred:.2f}, True={true:.2f}, Error={error:.2f}")

test_molecular_prediction()
```

## Performance Optimization {#performance-optimization}

### Exercise 13: Efficient Graph Processing

**Task**: Implement optimized graph operations for large-scale GNNs.

```python
import torch
from torch_scatter import scatter_mean, scatter_sum, scatter_max
from torch_sparse import SparseTensor

class OptimizedGCNLayer(nn.Module):
    """Optimized GCN layer using sparse operations"""

    def __init__(self, input_dim, output_dim, bias=True):
        super(OptimizedGCNLayer, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, node_features, edge_index, edge_weight=None):
        """
        Optimized forward pass using torch_scatter

        Args:
            node_features: (num_nodes, input_dim)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,) optional edge weights

        Returns:
            (num_nodes, output_dim)
        """
        # Transform features
        h = torch.mm(node_features, self.weight)  # (num_nodes, output_dim)

        # Add self-loops
        edge_index_with_self = torch.cat([
            edge_index,
            torch.stack([torch.arange(h.size(0)), torch.arange(h.size(0))])
        ], dim=1)

        if edge_weight is not None:
            edge_weight_with_self = torch.cat([
                edge_weight,
                torch.ones(h.size(0))  # Self-loop weights
            ])
        else:
            edge_weight_with_self = torch.ones(edge_index_with_self.size(1))

        # Source and target nodes
        row, col = edge_index_with_self

        # Aggregate messages from neighbors
        h_neighbor = h[col] * edge_weight_with_self.unsqueeze(1)
        h_new = scatter_sum(h_neighbor, row, dim=0, dim_size=h.size(0))

        # Normalize by degree
        degree = scatter_sum(edge_weight_with_self, row, dim=0, dim_size=h.size(0))
        degree = torch.clamp(degree, min=1)  # Avoid division by zero
        h_new = h_new / degree.unsqueeze(1)

        # Add bias
        if self.bias is not None:
            h_new = h_new + self.bias

        return F.relu(h_new)

class BatchGraphProcessor:
    """Efficient batch processing of multiple graphs"""

    def __init__(self, max_nodes_per_batch=10000):
        self.max_nodes_per_batch = max_nodes_per_batch

    def process_large_graph(self, graphs):
        """Process large graphs by batching"""
        results = []

        # Sort graphs by size
        sorted_graphs = sorted(enumerate(graphs), key=lambda x: x[1]['num_nodes'])

        current_batch = []
        current_node_count = 0

        for graph_id, graph in sorted_graphs:
            node_count = graph['num_nodes']

            # If adding this graph would exceed limit, process current batch
            if current_node_count + node_count > self.max_nodes_per_batch and current_batch:
                batch_result = self.process_batch(current_batch)
                results.append(batch_result)
                current_batch = []
                current_node_count = 0

            current_batch.append((graph_id, graph))
            current_node_count += node_count

        # Process remaining batch
        if current_batch:
            batch_result = self.process_batch(current_batch)
            results.append(batch_result)

        return results

    def process_batch(self, batch):
        """Process a batch of graphs"""
        # Combine all graphs in batch
        total_nodes = sum(graph['num_nodes'] for _, graph in batch)
        total_edges = sum(graph['edge_index'].size(1) for _, graph in batch)

        # Create combined edge index and features
        combined_edge_index = torch.zeros(2, total_edges, dtype=torch.long)
        combined_edge_features = torch.zeros(total_edges, 64)  # Example feature dim
        node_offset = 0
        edge_offset = 0

        for graph_id, graph in batch:
            # Adjust edge indices
            edge_index = graph['edge_index'] + node_offset
            combined_edge_index[:, edge_offset:edge_offset + edge_index.size(1)] = edge_index

            # Add edge features
            if graph['edge_features'].size(0) > 0:
                combined_edge_features[edge_offset:edge_offset + edge_index.size(1)] = graph['edge_features']

            node_offset += graph['num_nodes']
            edge_offset += edge_index.size(1)

        return {
            'batch': batch,
            'combined_edge_index': combined_edge_index,
            'combined_edge_features': combined_edge_features
        }

class MemoryEfficientGNN:
    """Memory-efficient implementation of GNN training"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Lazy layer initialization
        self.layers = nn.ModuleDict()
        self.initialized = False

    def init_layers(self):
        """Initialize layers when first called"""
        self.layers['gcn1'] = OptimizedGCNLayer(self.input_dim, self.hidden_dim)
        self.layers['gcn2'] = OptimizedGCNLayer(self.hidden_dim, self.hidden_dim)
        self.layers['gcn3'] = OptimizedGCNLayer(self.hidden_dim, self.output_dim)
        self.initialized = True

    def forward(self, x, edge_index, edge_weight=None, batch_size=None):
        """
        Forward pass with memory optimization

        Args:
            x: (num_nodes, input_dim)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,) optional
            batch_size: int, process in batches if specified
        """
        if not self.initialized:
            self.init_layers()

        if batch_size is None or batch_size >= x.size(0):
            # Process all at once
            return self._forward_single(x, edge_index, edge_weight)
        else:
            # Process in batches
            return self._forward_batched(x, edge_index, edge_weight, batch_size)

    def _forward_single(self, x, edge_index, edge_weight):
        """Single forward pass"""
        x = self.layers['gcn1'](x, edge_index, edge_weight)
        x = self.layers['gcn2'](x, edge_index, edge_weight)
        x = self.layers['gcn3'](x, edge_index, edge_weight)
        return x

    def _forward_batched(self, x, edge_index, edge_weight, batch_size):
        """Batch processing for memory efficiency"""
        num_nodes = x.size(0)
        all_outputs = []

        for start_idx in range(0, num_nodes, batch_size):
            end_idx = min(start_idx + batch_size, num_nodes)

            # Get batch nodes
            batch_x = x[start_idx:end_idx]

            # Get edges within batch
            batch_mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
            batch_mask = batch_mask & ((edge_index[1] >= start_idx) & (edge_index[1] < end_idx))

            batch_edge_index = edge_index[:, batch_mask] - start_idx
            if edge_weight is not None:
                batch_edge_weight = edge_weight[batch_mask]
            else:
                batch_edge_weight = None

            # Forward pass for batch
            batch_output = self._forward_single(batch_x, batch_edge_index, batch_edge_weight)
            all_outputs.append(batch_output)

        return torch.cat(all_outputs, dim=0)

def test_optimized_gnn():
    """Test optimized GNN implementations"""
    print("=== Testing Optimized GNN ===")

    # Create synthetic large graph
    num_nodes = 5000
    num_edges = 15000

    # Generate random sparse graph
    edge_indices = torch.randint(0, num_nodes, (2, num_edges))
    edge_weights = torch.rand(num_edges)
    node_features = torch.randn(num_nodes, 32)

    print(f"Graph size: {num_nodes} nodes, {num_edges} edges")

    # Test optimized GCN layer
    optimized_gcn = OptimizedGCNLayer(32, 64)

    print("\nTesting Optimized GCN Layer:")
    import time

    start_time = time.time()
    output1 = optimized_gcn(node_features, edge_indices, edge_weights)
    time1 = time.time() - start_time

    print(f"Output shape: {output1.shape}")
    print(f"Time: {time1:.3f}s")

    # Test batch processing
    memory_gnn = MemoryEfficientGNN(32, 64, 10)

    print("\nTesting Memory Efficient GNN:")

    # Single forward pass
    start_time = time.time()
    output_single = memory_gnn(node_features, edge_indices, edge_weights)
    time_single = time.time() - start_time

    print(f"Single pass output shape: {output_single.shape}")
    print(f"Single pass time: {time_single:.3f}s")

    # Batched forward pass
    start_time = time.time()
    output_batched = memory_gnn(node_features, edge_indices, edge_weights, batch_size=1000)
    time_batched = time.time() - start_time

    print(f"Batched output shape: {output_batched.shape}")
    print(f"Batched pass time: {time_batched:.3f}s")

    # Verify results are similar
    diff = torch.abs(output_single - output_batched).max().item()
    print(f"Maximum difference: {diff:.6f}")

test_optimized_gnn()
```

### Exercise 14: Distributed Graph Neural Networks

**Task**: Implement distributed training for large-scale graph neural networks.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

class DistributedGraphSampler:
    """Distributed sampling for large graphs"""

    def __init__(self, graph, num_replicas, rank, shuffle=True):
        self.graph = graph
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle

        # Calculate sample size per replica
        self.num_samples = math.ceil(len(graph.nodes) / num_replicas)

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.graph.nodes), generator=g)
        else:
            indices = torch.arange(len(self.graph.nodes))

        # Split indices among replicas
        start_idx = self.rank * self.num_samples
        end_idx = min(start_idx + self.num_samples, len(indices))
        replica_indices = indices[start_idx:end_idx]

        return iter(replica_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistributedGraphGNN:
    """Distributed Graph Neural Network for large-scale training"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Initialize model components
        self._init_model()

    def _init_model(self):
        """Initialize model with layers"""
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(OptimizedGCNLayer(self.input_dim, self.hidden_dim))

        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.layers.append(OptimizedGCNLayer(self.hidden_dim, self.hidden_dim))

        # Output layer
        self.layers.append(OptimizedGCNLayer(self.hidden_dim, self.output_dim))

    def distribute_model(self, device):
        """Distribute model across devices"""
        self.model = DDP(self._create_model(), device_ids=[device])
        return self.model

    def _create_model(self):
        """Create model for distribution"""
        model = GraphNeuralNetwork(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers)
        return model

    def aggregate_gradients(self):
        """Aggregate gradients across all replicas"""
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()

class GraphNeuralNetwork(nn.Module):
    """Base graph neural network"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GraphNeuralNetwork, self).__init__()

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(OptimizedGCNLayer(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(OptimizedGCNLayer(hidden_dim, hidden_dim))

        # Output layer
        self.layers.append(OptimizedGCNLayer(hidden_dim, output_dim))

    def forward(self, x, edge_index, edge_weight=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if i < len(self.layers) - 1:  # Don't activate last layer
                x = F.relu(x)
        return x

def setup_distributed(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Setup CUDA
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def distributed_train_step(rank, world_size, model, data, optimizer, epoch):
    """Single training step in distributed setting"""
    # Setup distributed sampler
    sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(data, batch_size=32, sampler=sampler)

    model.train()

    # Set epoch for proper shuffling
    sampler.set_epoch(epoch)

    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Forward pass
        output = model(batch['features'], batch['edge_index'], batch['edge_weight'])
        loss = F.mse_loss(output, batch['target'])

        # Backward pass
        loss.backward()

        # Aggregate gradients
        model.module.aggregate_gradients()

        # Update parameters
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0 and rank == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Average loss across all replicas
    avg_loss = total_loss / max(num_batches, 1)
    if world_size > 1:
        loss_tensor = torch.tensor(avg_loss, device=f'cuda:{rank}')
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size

    return avg_loss

def run_distributed_training(num_epochs=5):
    """Run distributed training across multiple processes"""
    world_size = torch.cuda.device_count()
    print(f"Running distributed training with {world_size} GPUs")

    if world_size < 2:
        print("Not enough GPUs for distributed training")
        return

    # Spawn processes
    mp.spawn(
        distributed_worker,
        args=(world_size, num_epochs),
        nprocs=world_size,
        join=True
    )

def distributed_worker(rank, world_size, num_epochs):
    """Worker function for distributed training"""
    # Setup
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Create synthetic data
    num_nodes = 1000
    input_dim = 32
    hidden_dim = 64
    output_dim = 16

    # Generate graph data
    node_features = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 5000))
    edge_weight = torch.rand(5000)
    targets = torch.randn(num_nodes, output_dim)

    graph_data = {
        'features': node_features,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'target': targets
    }

    # Create model
    model = DistributedGraphGNN(input_dim, hidden_dim, output_dim)
    model = model.distribute_model(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    if rank == 0:
        print(f"Starting distributed training on device {device}")

    for epoch in range(num_epochs):
        loss = distributed_train_step(rank, world_size, model, graph_data, optimizer, epoch)

        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {loss:.4f}")

    # Cleanup
    cleanup_distributed()

    if rank == 0:
        print("Distributed training completed")

def test_distributed_concepts():
    """Test distributed GNN concepts (without actual distribution)"""
    print("=== Testing Distributed GNN Concepts ===")

    # Simulate distributed training concepts
    input_dim = 32
    hidden_dim = 64
    output_dim = 16
    world_size = 4

    # Create model
    model = DistributedGraphGNN(input_dim, hidden_dim, output_dim)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model layers: {len(model.layers)}")

    # Simulate gradient aggregation
    def simulate_gradient_aggregation():
        """Simulate gradient aggregation across replicas"""
        # Create dummy gradients
        gradients = [torch.randn(1000) for _ in range(world_size)]

        # Simulate all_reduce operation
        aggregated = torch.zeros_like(gradients[0])
        for grad in gradients:
            aggregated += grad
        aggregated /= world_size

        print(f"Simulated gradient aggregation:")
        print(f"  Individual gradients mean: {[g.mean().item() for g in gradients]}")
        print(f"  Aggregated gradient mean: {aggregated.mean().item()}")

        return aggregated

    aggregated_grad = simulate_gradient_aggregation()

    # Simulate distributed sampling
    total_nodes = 1000
    nodes_per_replica = total_nodes // world_size

    print(f"\nSimulated distributed sampling:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  World size: {world_size}")
    print(f"  Nodes per replica: {nodes_per_replica}")

    for rank in range(world_size):
        start_idx = rank * nodes_per_replica
        end_idx = start_idx + nodes_per_replica
        replica_nodes = list(range(start_idx, min(end_idx, total_nodes)))
        print(f"  Rank {rank}: {len(replica_nodes)} nodes")

test_distributed_concepts()
```

## End-to-End Projects {#end-to-end-projects}

### Exercise 15: Complete Social Network Analysis System

**Task**: Build a complete social network analysis system using Graph Neural Networks.

```python
import pandas as pd
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
import seaborn as sns

class SocialNetworkAnalysisSystem:
    """Complete social network analysis system using GNNs"""

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize models
        self.user_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.social_gnn = SimpleGCN(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=4
        )

        self.link_predictor = nn.Sequential(
            nn.Linear(2 * output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.community_detector = SimpleGCN(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            output_dim=10,  # 10 communities
            num_layers=3
        )

        self.influence_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def create_synthetic_social_network(self, num_users=1000, avg_degree=20):
        """Create a realistic synthetic social network"""
        np.random.seed(42)

        # Generate user profiles with realistic distributions
        user_profiles = []
        for i in range(num_users):
            age = np.random.normal(30, 10)  # Age around 30
            age = max(18, min(80, age))  # Clamp to reasonable range

            posts_per_month = np.random.exponential(10)  # Power-law distribution
            posts_per_month = min(500, posts_per_month)

            followers = int(np.random.exponential(100))  # Power-law follower count
            followers = min(10000, followers)

            interests = np.random.choice(
                ['tech', 'sports', 'music', 'food', 'travel', 'politics', 'art', 'science'],
                size=np.random.randint(1, 5)
            )

            engagement_score = np.random.beta(2, 5)  # Biased towards lower engagement

            user_profiles.append({
                'user_id': i,
                'age': age,
                'posts_per_month': posts_per_month,
                'follower_count': followers,
                'interests': interests,
                'engagement_score': engagement_score,
                'account_age_days': np.random.randint(30, 2000)
            })

        # Create community structure
        communities = np.random.choice(10, num_users)  # 10 communities

        # Generate connections with community bias
        adj_matrix = np.zeros((num_users, num_users))

        for i in range(num_users):
            # Connect to users in same community (higher probability)
            same_community = np.where(communities == communities[i])[0]
            same_community = same_community[same_community != i]

            for j in same_community:
                if np.random.random() < 0.05:  # 5% same-community connection rate
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

            # Connect to users in different communities (lower probability)
            diff_community = np.where(communities != communities[i])[0]
            for j in diff_community:
                if np.random.random() < 0.01:  # 1% cross-community connection rate
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

        # Add influence scores based on follower count and engagement
        influence_scores = []
        for profile in user_profiles:
            # Influencers have high followers and engagement
            base_influence = np.log(profile['follower_count'] + 1) * profile['engagement_score']
            influence = base_influence + np.random.normal(0, 0.1)
            influence_scores.append(max(0, influence))

        return {
            'user_profiles': user_profiles,
            'adjacency_matrix': adj_matrix,
            'communities': communities,
            'influence_scores': influence_scores
        }

    def create_user_features(self, user_profiles):
        """Create feature vectors for users"""
        features = []

        for profile in user_profiles:
            # Numerical features
            feature_vector = [
                profile['age'] / 100,  # Normalized age
                np.log(profile['posts_per_month'] + 1) / 10,  # Log-normalized posts
                np.log(profile['follower_count'] + 1) / 10,  # Log-normalized followers
                profile['engagement_score'],  # Engagement level
                profile['account_age_days'] / 2000,  # Normalized account age
            ]

            # Interest features (one-hot encoding)
            all_interests = ['tech', 'sports', 'music', 'food', 'travel', 'politics', 'art', 'science']
            for interest in all_interests:
                feature_vector.append(1 if interest in profile['interests'] else 0)

            features.append(feature_vector)

        # Convert to tensor and pad/truncate to input_dim
        feature_matrix = torch.tensor(features, dtype=torch.float32)

        if feature_matrix.size(1) < self.input_dim:
            padding = torch.zeros(feature_matrix.size(0), self.input_dim - feature_matrix.size(1))
            feature_matrix = torch.cat([feature_matrix, padding], dim=1)
        elif feature_matrix.size(1) > self.input_dim:
            feature_matrix = feature_matrix[:, :self.input_dim]

        return feature_matrix

    def train_link_prediction(self, user_features, adj_matrix, train_ratio=0.8):
        """Train link prediction model"""
        print("Training link prediction model...")

        # Create positive and negative edges
        num_users = adj_matrix.size(0)
        positive_edges = []
        negative_edges = []

        for i in range(num_users):
            for j in range(i + 1, num_users):
                if adj_matrix[i, j] == 1:
                    positive_edges.append((i, j))
                else:
                    # Sample negative edges with same distribution as positive
                    if np.random.random() < 0.1:  # Sample 10% of possible negative edges
                        negative_edges.append((i, j))

        # Split into train and test
        n_pos = len(positive_edges)
        n_neg = len(negative_edges)

        pos_train, pos_test = train_test_split(positive_edges, test_size=1-train_ratio)
        neg_train, neg_test = train_test_split(negative_edges, test_size=1-train_ratio)

        # Create labels
        train_edges = pos_train + neg_train
        train_labels = [1] * len(pos_train) + [0] * len(neg_train)

        test_edges = pos_test + neg_test
        test_labels = [1] * len(pos_test) + [0] * len(neg_test)

        # Convert to tensors
        train_edges = torch.tensor(train_edges)
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        test_edges = torch.tensor(test_edges)
        test_labels = torch.tensor(test_labels, dtype=torch.float32)

        # Train model
        optimizer = torch.optim.Adam(
            list(self.user_encoder.parameters()) +
            list(self.social_gnn.parameters()) +
            list(self.link_predictor.parameters()),
            lr=0.001
        )

        self.user_encoder.train()
        self.social_gnn.train()
        self.link_predictor.train()

        num_epochs = 100
        batch_size = 64

        for epoch in range(num_epochs):
            # Shuffle training data
            perm = torch.randperm(len(train_edges))

            epoch_loss = 0
            num_batches = 0

            for i in range(0, len(train_edges), batch_size):
                batch_idx = perm[i:i + batch_size]
                batch_edges = train_edges[batch_idx]
                batch_labels = train_labels[batch_idx]

                optimizer.zero_grad()

                # Forward pass
                user_repr = self.user_encoder(user_features)

                # Create adjacency matrix (simplified for batch processing)
                batch_adj = adj_matrix  # Use full adjacency for simplicity

                # Get social embeddings
                social_repr = self.social_gnn(user_repr, batch_adj)

                # Predict links
                link_probs = []
                for edge in batch_edges:
                    u, v = edge
                    combined = torch.cat([social_repr[u], social_repr[v]])
                    prob = self.link_predictor(combined)
                    link_probs.append(prob)

                link_probs = torch.stack(link_probs).squeeze()

                # Compute loss
                loss = F.binary_cross_entropy(link_probs, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss / max(num_batches, 1):.4f}")

        # Evaluation
        self.user_encoder.eval()
        self.social_gnn.eval()
        self.link_predictor.eval()

        with torch.no_grad():
            # Get embeddings
            user_repr = self.user_encoder(user_features)
            social_repr = self.social_gnn(user_repr, adj_matrix)

            # Predict test edges
            test_probs = []
            for edge in test_edges:
                u, v = edge
                combined = torch.cat([social_repr[u], social_repr[v]])
                prob = self.link_predictor(combined)
                test_probs.append(prob.item())

            test_probs = np.array(test_probs)

            # Calculate metrics
            auc_score = roc_auc_score(test_labels, test_probs)
            print(f"Test AUC: {auc_score:.4f}")

            # Precision-recall analysis
            precision, recall, _ = precision_recall_curve(test_labels, test_probs)
            print(f"Max precision: {max(precision):.4f}")
            print(f"Max recall: {max(recall):.4f}")

        return {
            'auc_score': auc_score,
            'test_edges': len(test_edges),
            'positive_test_edges': sum(test_labels),
            'negative_test_edges': len(test_labels) - sum(test_labels)
        }

    def detect_communities(self, user_features, adj_matrix, true_communities=None):
        """Detect communities using GNNs"""
        print("Detecting communities...")

        # Get community predictions
        user_repr = self.user_encoder(user_features)
        community_probs = self.community_detector(user_repr, adj_matrix)
        predicted_communities = torch.argmax(community_probs, dim=1)

        if true_communities is not None:
            # Calculate community detection metrics
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

            ari_score = adjusted_rand_score(true_communities, predicted_communities.numpy())
            nmi_score = normalized_mutual_info_score(true_communities, predicted_communities.numpy())

            print(f"Adjusted Rand Index: {ari_score:.4f}")
            print(f"Normalized Mutual Information: {nmi_score:.4f}")

        # Analyze community sizes
        community_sizes = {}
        for comm_id in range(10):  # We assumed 10 communities
            size = torch.sum(predicted_communities == comm_id).item()
            if size > 0:
                community_sizes[f'Community_{comm_id}'] = size

        print("Community sizes:")
        for comm_name, size in community_sizes.items():
            print(f"  {comm_name}: {size} users")

        return {
            'predicted_communities': predicted_communities.numpy(),
            'community_sizes': community_sizes,
            'community_probs': community_probs
        }

    def predict_user_influence(self, user_features, adj_matrix, true_influence=None):
        """Predict user influence scores"""
        print("Predicting user influence...")

        # Get user representations
        user_repr = self.user_encoder(user_features)
        social_repr = self.social_gnn(user_repr, adj_matrix)

        # Predict influence
        predicted_influence = self.influence_predictor(social_repr).squeeze()

        if true_influence is not None:
            # Calculate correlation
            correlation = np.corrcoef(true_influence, predicted_influence.numpy())[0, 1]
            print(f"Influence prediction correlation: {correlation:.4f}")

        return {
            'predicted_influence': predicted_influence.numpy(),
            'influence_ranking': torch.argsort(predicted_influence, descending=True).numpy()
        }

    def analyze_network_properties(self, adj_matrix):
        """Analyze network properties"""
        print("Analyzing network properties...")

        # Convert to NetworkX graph for analysis
        G = nx.from_numpy_array(adj_matrix.numpy())

        properties = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'is_connected': nx.is_connected(G),
        }

        if nx.is_connected(G):
            properties['average_shortest_path'] = nx.average_shortest_path_length(G)
            properties['diameter'] = nx.diameter(G)

        # Degree statistics
        degrees = [d for n, d in G.degree()]
        properties['average_degree'] = np.mean(degrees)
        properties['max_degree'] = max(degrees)
        properties['min_degree'] = min(degrees)

        print("Network properties:")
        for prop, value in properties.items():
            if isinstance(value, float):
                print(f"  {prop}: {value:.4f}")
            else:
                print(f"  {prop}: {value}")

        return properties

    def run_complete_analysis(self):
        """Run complete social network analysis"""
        print("=== Starting Complete Social Network Analysis ===\n")

        # Create synthetic network
        network_data = self.create_synthetic_social_network(num_users=1000)
        user_profiles = network_data['user_profiles']
        adj_matrix = network_data['adjacency_matrix']
        true_communities = network_data['communities']
        true_influence = network_data['influence_scores']

        # Convert to tensor
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

        # Create user features
        user_features = self.create_user_features(user_profiles)

        print(f"Network: {len(user_profiles)} users, {adj_matrix.sum().item() / 2:.0f} connections")
        print(f"Feature dimensions: {user_features.shape}\n")

        # 1. Analyze network properties
        network_props = self.analyze_network_properties(adj_matrix)
        print()

        # 2. Train link prediction
        link_results = self.train_link_prediction(user_features, adj_matrix)
        print()

        # 3. Detect communities
        community_results = self.detect_communities(
            user_features, adj_matrix, true_communities
        )
        print()

        # 4. Predict influence
        influence_results = self.predict_user_influence(
            user_features, adj_matrix, true_influence
        )
        print()

        # 5. Generate top influencers report
        top_influencers = influence_results['influence_ranking'][:10]
        print("Top 10 Predicted Influencers:")
        for i, user_id in enumerate(top_influencers):
            profile = user_profiles[user_id]
            pred_influence = influence_results['predicted_influence'][user_id]
            true_inf = true_influence[user_id]
            print(f"  {i+1}. User {user_id}: Predicted={pred_influence:.3f}, True={true_inf:.3f}")
            print(f"     Age: {profile['age']:.0f}, Followers: {profile['follower_count']}, Posts: {profile['posts_per_month']:.0f}")

        return {
            'network_properties': network_props,
            'link_prediction': link_results,
            'community_detection': community_results,
            'influence_prediction': influence_results,
            'top_influencers': top_influencers
        }

def test_complete_social_analysis():
    """Test the complete social network analysis system"""
    # Create and run analysis
    analyzer = SocialNetworkAnalysisSystem()
    results = analyzer.run_complete_analysis()

    return results

# Run the complete analysis
if __name__ == "__main__":
    test_complete_social_analysis()
```

## Summary

This comprehensive practice guide covers Graph Neural Networks from basic concepts to advanced applications:

### Core Implementations

- **Basic GCN**: Graph convolutional operations with normalization
- **Graph Attention**: Multi-head attention mechanisms for learned neighbor importance
- **Message Passing**: Flexible framework for various graph neural network architectures
- **GraphSAGE**: Sampling and aggregation for large-scale graphs
- **Graph Transformers**: Global attention with positional encodings

### Specialized Applications

- **Social Network Analysis**: Community detection, link prediction, influence scoring
- **Molecular Property Prediction**: Chemical graph processing with RDKit integration
- **Distributed Training**: Multi-GPU training for large-scale graphs
- **Performance Optimization**: Memory-efficient implementations

### Key Learning Outcomes

1. **Graph Representations**: Understanding different graph data structures and conversions
2. **Message Passing**: Core operation enabling relational reasoning across graphs
3. **Attention Mechanisms**: Learning importance weights for different neighbors
4. **Scalability**: Techniques for handling large graphs through sampling and batching
5. **Real-world Applications**: End-to-end systems for social networks, molecular science, and recommendation systems

### Advanced Concepts

- **Expressivity Analysis**: Understanding what graphs GNNs can and cannot distinguish
- **Training Strategies**: Supervised, semi-supervised, and self-supervised learning approaches
- **Evaluation Metrics**: Node-level, edge-level, and graph-level task evaluation
- **Deployment Considerations**: Model compression, inference optimization, and distributed serving

This practice guide provides hands-on experience with implementing, training, and deploying Graph Neural Networks across diverse domains and applications.
