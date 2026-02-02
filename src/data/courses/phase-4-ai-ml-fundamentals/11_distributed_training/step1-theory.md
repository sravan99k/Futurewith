---
title: Advanced Neural Networks Theory
level: Advanced
estimated_time: 90 minutes
prerequisites:
  [
    Deep Learning fundamentals,
    Neural Networks,
    Python programming,
    Linear algebra,
    Calculus,
  ]
skills_gained:
  [
    Specialized architectures,
    Graph neural networks,
    Autoencoders,
    GANs,
    Attention mechanisms,
    Multi-modal networks,
    Theoretical foundations,
  ]
success_criteria:
  [
    "Implement Capsule Networks with dynamic routing",
    "Build and train Graph Neural Networks",
    "Create Variational Autoencoders",
    "Develop Generative Adversarial Networks",
    "Apply attention mechanisms effectively",
    "Understand theoretical foundations",
  ]
version: 1.0
last_updated: 2025-11-11
---

# Advanced Neural Networks Theory

## Learning Goals

By the end of this comprehensive guide, you will be able to:

- Implement and understand Capsule Networks with dynamic routing algorithms
- Build and train Graph Neural Networks for graph-structured data
- Create sophisticated autoencoders including VAEs and conditional models
- Develop and train Generative Adversarial Networks including DCGANs and WGANs
- Apply attention mechanisms and understand multi-modal neural networks
- Grasp the theoretical foundations of advanced neural network architectures
- Design custom neural architectures for specific problems
- Evaluate and troubleshoot advanced neural network training

## TL;DR

Advanced neural networks extend basic architectures with specialized components: Capsule Networks for pose understanding, Graph Neural Networks for relational data, Variational Autoencoders for probabilistic generation, and sophisticated GAN variants - all requiring deep understanding of mathematical foundations and careful implementation.

## Common Confusions & Mistakes

- **Confusion: "Capsule Networks vs CNNs"** — Capsules capture spatial relationships and pose (orientation, size) while CNNs only detect features; capsules use dynamic routing instead of max pooling.

- **Confusion: "Graph Neural Networks vs Traditional GNNs"** — GNNs handle graph structure explicitly, not just sequential data; they use message passing and graph convolutions for relational reasoning.

- **Confusion: "VAE vs Regular Autoencoder"** — VAEs learn probabilistic latent space with KL divergence loss, enabling sampling; regular autoencoders have deterministic reconstruction.

- **Confusion: "GAN vs VAE Generation"** — GANs use adversarial training (generator vs discriminator) for sharp samples; VAEs use probabilistic inference for smooth, diverse samples.

- **Quick Debug Tip:** If your GAN training is unstable, try WGAN with gradient penalty or reduce learning rates; for GNNs, always normalize adjacency matrices properly.

- **Memory Error:** High-dimensional latent spaces can cause memory issues; start with lower dimensions and batch sizes when experimenting.

- **Training Instability:** Advanced architectures require careful hyperparameter tuning; use learning rate schedules and gradient clipping when needed.

## Micro-Quiz (80% mastery required)

1. **Q:** What problem does dynamic routing solve in Capsule Networks? **A:** It learns how much to trust each capsule's prediction based on agreement with higher-level capsules.

2. **Q:** In a Graph Neural Network, what does message passing accomplish? **A:** It allows nodes to exchange information with neighbors, enabling relational reasoning across the graph structure.

3. **Q:** What is the key difference between VAE and regular autoencoder loss functions? **A:** VAEs include KL divergence loss to regularize the latent space to a standard normal distribution.

4. **Q:** How does a WGAN improve upon basic GAN training stability? **A:** Uses Wasserstein distance and weight clipping to provide meaningful gradients even when discriminator is optimal.

5. **Q:** What does the reparameterization trick enable in VAEs? **A:** Allows backpropagation through stochastic sampling by separating the stochastic part.

## Reflection Prompts

- **Technical Depth:** Explain how Capsule Networks' dynamic routing algorithm relates to the information flow in traditional CNNs.

- **Application Design:** How would you choose between a VAE and a GAN for a specific data generation task?

- **Integration Challenge:** How could you combine Graph Neural Networks with attention mechanisms for more powerful graph analysis?

## Table of Contents

1. [Specialized Neural Network Architectures](#specialized-architectures)
2. [Graph Neural Networks (GNNs)](#gnns)
3. [Autoencoders and Variational Autoencoders](#autoencoders)
4. [Generative Adversarial Networks (GANs)](#gans)
5. [Attention Mechanisms and Transformers](#attention-transformers)
6. [Multi-modal Neural Networks](#multimodal-networks)
7. [Neural Architecture Search](#neural-architecture-search)
8. [Theoretical Foundations](#theoretical-foundations)
9. [Advanced Optimization Theory](#optimization-theory)
10. [Information Theory in Neural Networks](#information-theory)

---

## 1. Specialized Neural Network Architectures {#specialized-architectures}

### 1.1 Capsule Networks

Capsule networks address limitations of CNNs in understanding spatial relationships and pose information.

#### Mathematical Foundation

A capsule is a group of neurons whose activity vector represents instantiation parameters of a specific type of entity such as an object or object part. The length of the activity vector represents the probability that the entity exists, while the orientation represents its pose parameters.

```python
class CapsuleLayer:
    def __init__(self, num_capsules, dim_capsules, num_routes, in_channels):
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.num_routes = num_routes
        self.in_channels = in_channels

        # Weight matrix for each input capsule to output capsule
        self.W = np.random.randn(in_channels, num_capsules, dim_capsules, num_routes)

    def squash(self, s):
        """Squashing function to ensure capsule vectors have unit length"""
        squared_norm = np.sum(s**2, axis=-1, keepdims=True)
        return (squared_norm / (1 + squared_norm)) * (s / np.sqrt(squared_norm))

    def forward(self, u):
        """
        Dynamic routing algorithm
        Args:
            u: Input predictions from lower layer (batch_size, in_channels, 1, dim_capsules)
        """
        # Add batch dimension if needed
        if u.ndim == 3:
            u = u[:, :, np.newaxis, :]

        batch_size, _, _, _ = u.shape

        # Compute predictions
        # u_hat shape: (batch_size, in_channels, num_capsules, dim_capsules)
        u_hat = np.einsum('bijd,ijdk->bikd', u, self.W)

        # Initialize coupling coefficients
        b = np.zeros((batch_size, self.in_channels, self.num_capsules))
        c = self.softmax(b, axis=2)

        # Routing iterations
        for _ in range(3):
            # Weighted sum of predictions
            s = np.einsum('bij,bijd->bjd', c, u_hat)

            # Apply squash function
            v = self.squash(s)

            # Update coupling coefficients
            if _ < 2:  # Don't update on last iteration
                v_expanded = np.expand_dims(v, axis=1)
                u_hat_expanded = np.expand_dims(u_hat, axis=3)
                b += np.sum(u_hat_expanded * v_expanded, axis=(2, 4))
                c = self.softmax(b, axis=2)

        return v

    def softmax(self, x, axis=None):
        """Stable softmax implementation"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class PrimaryCapsuleLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 num_capsules, dim_capsules):
        self.conv = Conv1D(in_channels, out_channels, kernel_size, stride)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    def forward(self, x):
        # Apply convolution
        x = self.conv.forward(x)
        batch_size, height, width, out_channels = x.shape

        # Reshape to capsules
        x = x.reshape(batch_size, height, width, out_channels)

        # Flatten spatial dimensions
        x = x.reshape(batch_size, -1, out_channels)

        # Project to capsule dimensions
        capsule_output = np.zeros((batch_size, x.shape[1], self.num_capsules, self.dim_capsules))
        for i in range(self.num_capsules):
            capsule_output[:, :, i, :] = np.tanh(x * 0.5)  # Simple projection

        return capsule_output

class DeepCapsuleNetwork:
    def __init__(self, input_dim, num_classes, primary_capsules=32,
                 digit_capsules=10, dim_capsules=16):
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Primary capsule layer
        self.primary_caps = PrimaryCapsuleLayer(1, 256, 9, 1, primary_capsules, dim_capsules)

        # Convolutional layer before primary capsules
        self.pre_conv = Conv1D(1, 256, 9, 1)

        # Digit capsule layer
        self.digit_caps = CapsuleLayer(
            num_capsules=digit_capsules,
            dim_capsules=dim_capsules,
            num_routes=primary_capsules * 16 * 16,  # Flattened primary capsule output
            in_channels=primary_capsules
        )

        # Reconstruction network
        self.reconstruction_layers = [
            DenseLayer(dim_capsules * num_classes, 512, 'relu'),
            DenseLayer(512, 1024, 'relu'),
            DenseLayer(1024, input_dim, 'sigmoid')
        ]

    def forward(self, x):
        # Reshape input if needed
        if len(x.shape) == 2:
            x = x.reshape(-1, 1, x.shape[1])

        # Pre-convolution
        x = self.pre_conv.forward(x)
        batch_size, height, width, _ = x.shape

        # Primary capsules
        x = self.primary_caps.forward(x)

        # Digit capsules
        digit_capsules = self.digit_caps.forward(x)

        return digit_capsules

    def compute_margin_loss(self, digit_capsules, labels):
        """Margin loss for capsule networks"""
        batch_size, num_capsules, dim_capsules = digit_capsules.shape

        # Compute lengths of digit capsules
        lengths = np.sqrt(np.sum(digit_capsules**2, axis=2))

        # Positive and negative margin losses
        m_plus = 0.9
        m_minus = 0.1
        lambda_val = 0.5

        L_k = np.zeros(batch_size)

        for i in range(num_capsules):
            # Present class
            L_k += np.where(labels == i,
                          m_plus - lengths[:, i],
                          0)**2

            # Absent class
            L_k += lambda_val * np.maximum(0, lengths[:, i] - m_minus)**2

        return np.mean(L_k)

    def reconstruct(self, digit_capsules, labels):
        """Reconstruct input from digit capsules"""
        # Select relevant capsules based on labels
        batch_size, num_capsules, dim_capsules = digit_capsules.shape
        selected_capsules = np.zeros((batch_size, num_capsules * dim_capsules))

        for i in range(batch_size):
            label_idx = labels[i]
            selected_capsules[i, :] = digit_capsules[i, label_idx, :]

        # Reconstruction network
        x = selected_capsules
        for layer in self.reconstruction_layers:
            x = layer.forward(x)

        return x
```

### 1.2 Normalizing Flows

Normalizing flows enable complex probability distributions through a sequence of invertible transformations.

```python
class NormalizingFlow:
    def __init__(self, num_flows, base_distribution='normal'):
        self.num_flows = num_flows
        self.base_distribution = base_distribution
        self.flow_layers = []

    def add_flow(self, flow_layer):
        """Add a flow layer to the normalizing flow"""
        self.flow_layers.append(flow_layer)

    def forward(self, z):
        """Forward pass through flow"""
        log_det = 0
        for flow in self.flow_layers:
            z, ld = flow.forward(z)
            log_det += ld
        return z, log_det

    def inverse(self, x):
        """Inverse pass through flow"""
        log_det = 0
        for flow in reversed(self.flow_layers):
            x, ld = flow.inverse(x)
            log_det += ld
        return x, log_det

class PlanarFlow:
    def __init__(self, dim):
        self.dim = dim

        # Learnable parameters
        self.w = np.random.randn(dim)
        self.b = np.random.randn(1)
        self.u = np.random.randn(dim)

        # Ensure w^T * u > -1 for invertibility
        while np.dot(self.w, self.u) <= -1:
            self.u = np.random.randn(dim)

    def forward(self, z):
        """Forward pass: z -> f(z) = z + u * tanh(w^T z + b)"""
        linear = np.dot(z, self.w) + self.b
        transformation = self.u * np.tanh(linear)
        z_prime = z + transformation

        # Log determinant of Jacobian
        psi = self.w * (1 - np.tanh(linear)**2)
        log_det = np.log(np.abs(1 + np.dot(self.u, psi)))

        return z_prime, log_det

    def inverse(self, z_prime):
        """Inverse pass: f^{-1}(z')"""
        # Iterative solution for inverse
        z = z_prime.copy()
        for _ in range(50):  # Fixed point iteration
            linear = np.dot(z, self.w) + self.b
            z = z_prime - self.u * np.tanh(linear)
        return z, 0  # Return 0 for log_det (computed in forward)

class RadialFlow:
    def __init__(self, dim):
        self.dim = dim

        # Learnable parameters
        self.z0 = np.random.randn(dim)
        self.alpha = np.random.rand(1)  # Ensure alpha > 0
        self.beta = np.random.randn(1)

    def forward(self, z):
        """Forward pass: f(z) = z + beta * r * r0 / (||r0||^2 + alpha) where r = z - z0"""
        r = z - self.z0
        r_norm_sq = np.sum(r**2)

        # Transformation
        scale = self.beta * r / (r_norm_sq + self.alpha)
        z_prime = z + scale

        # Log determinant of Jacobian
        log_det = np.log(np.abs(1 + self.beta * (self.dim - 1) / (r_norm_sq + self.alpha)))

        return z_prime, log_det

    def inverse(self, z_prime):
        """Inverse pass (iterative approximation)"""
        z = z_prime.copy()
        for _ in range(50):
            r = z - self.z0
            r_norm_sq = np.sum(r**2)
            scale = self.beta * r / (r_norm_sq + self.alpha)
            z = z_prime - scale
        return z, 0

class CouplingLayer:
    def __init__(self, dim, hidden_dim, scale_shift='affine'):
        self.dim = dim
        self.scale_shift = scale_shift

        # Split dimensions
        self.dim1 = dim // 2
        self.dim2 = dim - self.dim1

        # Neural networks for scale and shift parameters
        self.scale_net = self._build_network(self.dim1, hidden_dim, self.dim2, 'tanh')
        self.shift_net = self._build_network(self.dim1, hidden_dim, self.dim2, 'linear')

    def _build_network(self, input_dim, hidden_dim, output_dim, activation):
        """Build a small neural network"""
        layers = [
            DenseLayer(input_dim, hidden_dim, activation),
            DenseLayer(hidden_dim, hidden_dim, activation),
            DenseLayer(hidden_dim, output_dim, activation if activation == 'tanh' else 'linear')
        ]
        return layers

    def forward(self, z):
        """Forward pass with coupling"""
        z1, z2 = z[:, :self.dim1], z[:, self.dim1:]

        # Compute scale and shift
        s = z1
        for layer in self.scale_net:
            s = layer.forward(s)

        t = z1
        for layer in self.shift_net:
            t = layer.forward(t)

        if self.scale_shift == 'affine':
            # Affine coupling
            z2_prime = z2 * s + t
            log_det = np.sum(np.log(np.abs(s)), axis=1)
        else:  # additive
            # Additive coupling
            z2_prime = z2 + t
            log_det = np.zeros(z1.shape[0])

        z_prime = np.concatenate([z1, z2_prime], axis=1)
        return z_prime, log_det

    def inverse(self, z_prime):
        """Inverse pass with coupling"""
        z1, z2_prime = z_prime[:, :self.dim1], z_prime[:, self.dim1:]

        # Compute scale and shift
        s = z1
        for layer in self.scale_net:
            s = layer.forward(s)

        t = z1
        for layer in self.shift_net:
            t = layer.forward(t)

        if self.scale_shift == 'affine':
            z2 = (z2_prime - t) / s
            log_det = -np.sum(np.log(np.abs(s)), axis=1)
        else:  # additive
            z2 = z2_prime - t
            log_det = np.zeros(z1.shape[0])

        z = np.concatenate([z1, z2], axis=1)
        return z, log_det
```

---

## 2. Graph Neural Networks (GNNs) {#gnns}

Graph Neural Networks extend neural networks to graph-structured data.

### 2.1 Message Passing Framework

```python
class MessagePassingLayer:
    def __init__(self, node_dim, edge_dim, hidden_dim, aggr_type='sum'):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.aggr_type = aggr_type

        # Message function: m_{uv} = M(h_u, h_v, e_{uv})
        self.message_net = self._build_message_network()

        # Update function: h_u' = U(h_u, Σ m_{uv})
        self.update_net = self._build_update_network()

    def _build_message_network(self):
        """Build message network"""
        layers = [
            DenseLayer(self.node_dim + self.node_dim + self.edge_dim, self.hidden_dim, 'relu'),
            DenseLayer(self.hidden_dim, self.hidden_dim, 'relu')
        ]
        return layers

    def _build_update_network(self):
        """Build update network"""
        layers = [
            DenseLayer(self.node_dim + self.hidden_dim, self.hidden_dim, 'relu'),
            DenseLayer(self.hidden_dim, self.node_dim, 'linear')
        ]
        return layers

    def forward(self, h, edge_index, edge_attr):
        """
        Forward pass
        Args:
            h: Node features (num_nodes, node_dim)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge features (num_edges, edge_dim)
        """
        num_nodes = h.shape[0]

        # Initialize messages
        messages = np.zeros((edge_index.shape[1], self.hidden_dim))

        # Compute messages for each edge
        for i, (u, v) in enumerate(edge_index.T):
            # Message from node u to node v
            message_input = np.concatenate([h[u], h[v], edge_attr[i]])

            for layer in self.message_net:
                message_input = layer.forward(message_input)

            messages[i] = message_input

        # Aggregate messages for each node
        aggregated = np.zeros((num_nodes, self.hidden_dim))

        for v in range(num_nodes):
            # Get incoming messages to node v
            incoming_edges = edge_index[1] == v
            incoming_messages = messages[incoming_edges]

            if self.aggr_type == 'sum':
                aggregated[v] = np.sum(incoming_messages, axis=0)
            elif self.aggr_type == 'mean':
                aggregated[v] = np.mean(incoming_messages, axis=0)
            elif self.aggr_type == 'max':
                aggregated[v] = np.max(incoming_messages, axis=0)

        # Update node features
        h_new = np.zeros_like(h)
        for u in range(num_nodes):
            update_input = np.concatenate([h[u], aggregated[u]])

            for layer in self.update_net:
                update_input = layer.forward(update_input)

            h_new[u] = update_input

        return h_new

class GraphAttentionLayer:
    def __init__(self, node_dim, hidden_dim, heads=8, dropout=0.1):
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.dropout = dropout

        # Attention mechanisms for each head
        self.attention_layers = []
        for _ in range(heads):
            att_layer = MultiHeadAttention(
                in_dim=node_dim,
                out_dim=self.head_dim,
                key_dim=self.head_dim,
                value_dim=self.head_dim,
                heads=1
            )
            self.attention_layers.append(att_layer)

    def forward(self, h, edge_index, edge_attr):
        """
        Forward pass with graph attention
        Args:
            h: Node features (num_nodes, node_dim)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge features (num_edges, edge_dim)
        """
        num_nodes = h.shape[0]

        # Process each attention head
        head_outputs = []
        for att_layer in self.attention_layers:
            # For simplicity, treat as self-attention on graph
            # In practice, you'd implement proper graph attention
            output = self._compute_graph_attention(att_layer, h, edge_index)
            head_outputs.append(output)

        # Concatenate head outputs
        h_concat = np.concatenate(head_outputs, axis=1)

        # Final linear transformation
        h_new = self._final_transformation(h_concat)

        return h_new

    def _compute_graph_attention(self, att_layer, h, edge_index):
        """Compute graph attention for a single head"""
        # Simplified self-attention
        # In practice, you need to implement proper edge-based attention

        # Compute attention scores (simplified)
        attention_scores = np.dot(h, h.T) / np.sqrt(self.head_dim)

        # Apply mask for non-connected nodes (simplified)
        # In practice, you'd use the edge_index to create proper masks

        # Softmax
        attention_weights = self._softmax_stable(attention_scores)

        # Apply dropout
        if self.dropout > 0:
            mask = np.random.rand(*attention_weights.shape) > self.dropout
            attention_weights = attention_weights * mask

        # Compute weighted sum
        h_new = np.dot(attention_weights, h)

        return h_new

    def _softmax_stable(self, x):
        """Stable softmax implementation"""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _final_transformation(self, h_concat):
        """Final linear transformation"""
        # Simplified final layer
        W = np.random.randn(h_concat.shape[1], self.hidden_dim) * 0.1
        return np.maximum(0, np.dot(h_concat, W))

class GraphConvolutionalNetwork:
    def __init__(self, input_dim, hidden_dims, output_dim, num_classes):
        self.layers = []

        # Input layer
        self.layers.append(GraphConvLayer(input_dim, hidden_dims[0]))

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.layers.append(GraphConvLayer(hidden_dims[i-1], hidden_dims[i]))

        # Output layer
        self.layers.append(GraphConvLayer(hidden_dims[-1], output_dim))

        # Final classification layer
        self.classifier = DenseLayer(output_dim, num_classes, 'linear')

    def forward(self, h, edge_index, edge_attr=None):
        """Forward pass through GCN"""
        for layer in self.layers:
            h = layer.forward(h, edge_index, edge_attr)

        # Global pooling and classification
        h_pooled = np.mean(h, axis=0, keepdims=True)  # Global average pooling
        output = self.classifier.forward(h_pooled)

        return output

class GraphConvLayer:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        if bias:
            self.bias = np.zeros(out_features)
        else:
            self.bias = None

    def forward(self, h, edge_index, edge_attr=None):
        """
        Simplified Graph Convolutional Layer forward pass
        Args:
            h: Node features (num_nodes, in_features)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge features (num_edges, edge_attr_dim)
        """
        num_nodes = h.shape[0]

        # Compute adjacency matrix (simplified)
        adj = np.zeros((num_nodes, num_nodes))
        for i, (u, v) in enumerate(edge_index.T):
            adj[u, v] = 1
            adj[v, u] = 1  # Undirected graph

        # Add self-connections
        adj = adj + np.eye(num_nodes)

        # Compute degree matrix
        deg = np.diag(np.sum(adj, axis=1))
        deg_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(deg)))

        # Normalize adjacency matrix
        adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt

        # Apply GCN transformation
        h_transformed = np.dot(h, self.weight)
        h_new = np.dot(adj_norm, h_transformed)

        if self.bias is not None:
            h_new += self.bias

        # Activation
        h_new = np.maximum(0, h_new)  # ReLU

        return h_new

class GraphSAGELayer:
    """Graph Sample and Aggregate layer"""
    def __init__(self, in_features, out_features, aggregator='mean'):
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator

        # Neighbor transformation
        self.neighbor_transform = DenseLayer(in_features, out_features, 'relu')

        # Self transformation
        self.self_transform = DenseLayer(in_features, out_features, 'relu')

    def forward(self, h, edge_index, edge_attr=None):
        """Forward pass"""
        num_nodes = h.shape[0]

        # Initialize aggregated features
        h_agg = np.zeros((num_nodes, self.out_features))

        for u in range(num_nodes):
            # Get neighbors of node u
            neighbors = edge_index[0][edge_index[1] == u]

            if len(neighbors) == 0:
                # No neighbors, use self feature
                h_agg[u] = self.self_transform.forward(h[u])
            else:
                # Aggregate neighbor features
                neighbor_features = h[neighbors]

                # Transform neighbor features
                neighbor_features_transformed = np.zeros((len(neighbors), self.out_features))
                for i, neighbor in enumerate(neighbors):
                    neighbor_features_transformed[i] = self.neighbor_transform.forward(h[neighbor])

                # Aggregate
                if self.aggregator == 'mean':
                    neighbor_agg = np.mean(neighbor_features_transformed, axis=0)
                elif self.aggregator == 'sum':
                    neighbor_agg = np.sum(neighbor_features_transformed, axis=0)
                elif self.aggregator == 'max':
                    neighbor_agg = np.max(neighbor_features_transformed, axis=0)

                # Concatenate self and neighbor features
                self_features = self.self_transform.forward(h[u])
                combined = np.concatenate([self_features, neighbor_agg])

                # Final transformation
                h_agg[u] = combined

        return h_agg
```

### 2.2 Graph Autoencoders

```python
class GraphAutoencoder:
    def __init__(self, node_features, hidden_dims, latent_dim):
        self.node_features = node_features
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = GraphEncoder(node_features, hidden_dims, latent_dim)

        # Decoder
        self.decoder = GraphDecoder(latent_dim)

    def forward(self, h, edge_index, edge_attr=None):
        """Forward pass through graph autoencoder"""
        # Encode node features
        z = self.encoder.forward(h, edge_index)

        # Reconstruct adjacency matrix
        adj_recon = self.decoder.forward(z, edge_index)

        return adj_recon, z

    def compute_loss(self, h, edge_index, edge_attr=None, adj_target=None):
        """Compute graph autoencoder loss"""
        adj_recon, z = self.forward(h, edge_index, edge_attr)

        # Reconstruction loss
        num_nodes = h.shape[0]
        adj_target_dense = np.zeros((num_nodes, num_nodes))
        for i, (u, v) in enumerate(edge_index.T):
            adj_target_dense[u, v] = 1
            adj_target_dense[v, u] = 1

        # Binary cross-entropy loss
        recon_loss = -np.mean(adj_target_dense * np.log(adj_recon + 1e-8) +
                             (1 - adj_target_dense) * np.log(1 - adj_recon + 1e-8))

        return recon_loss

class GraphEncoder:
    def __init__(self, input_dim, hidden_dims, latent_dim):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Build encoder layers
        self.layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(GraphConvLayer(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Final layer to latent space
        self.latent_layer = GraphConvLayer(prev_dim, latent_dim)

    def forward(self, h, edge_index):
        """Encode node features to latent space"""
        x = h

        for layer in self.layers:
            x = layer.forward(x, edge_index)

        z = self.latent_layer.forward(x, edge_index)

        # Global pooling to get graph-level representation
        z_graph = np.mean(z, axis=0, keepdims=True)

        return z_graph

class GraphDecoder:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def forward(self, z, edge_index):
        """Reconstruct adjacency matrix from latent representations"""
        num_nodes = len(edge_index[0])
        batch_size = z.shape[0]

        # Compute dot product between latent representations
        z_expanded = np.expand_dims(z, axis=1)  # (1, 1, latent_dim)
        z_nodes = np.tile(z, (num_nodes, 1))  # (num_nodes, latent_dim)

        # Compute similarity scores
        scores = np.dot(z_nodes, z_nodes.T)  # (num_nodes, num_nodes)

        # Apply sigmoid to get probabilities
        adj_recon = 1.0 / (1.0 + np.exp(-scores))

        return adj_recon
```

---

## 3. Autoencoders and Variational Autoencoders {#autoencoders}

### 3.1 Denoising Autoencoders

```python
class DenoisingAutoencoder:
    def __init__(self, input_dim, hidden_dims, latent_dim, noise_factor=0.1):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.noise_factor = noise_factor

        # Encoder
        self.encoder = self._build_encoder()

        # Decoder
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        """Build encoder network"""
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(DenseLayer(prev_dim, hidden_dim, 'relu'))
            prev_dim = hidden_dim

        # Latent layer
        layers.append(DenseLayer(prev_dim, self.latent_dim, 'linear'))

        return layers

    def _build_decoder(self):
        """Build decoder network"""
        layers = []
        prev_dim = self.latent_dim

        for hidden_dim in reversed(self.hidden_dims):
            layers.append(DenseLayer(prev_dim, hidden_dim, 'relu'))
            prev_dim = hidden_dim

        # Output layer
        layers.append(DenseLayer(prev_dim, self.input_dim, 'sigmoid'))

        return layers

    def add_noise(self, x, noise_type='gaussian'):
        """Add noise to input"""
        if noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_factor, x.shape)
            x_noisy = x + noise
        elif noise_type == 'masking':
            # Masking noise (dropout)
            mask = np.random.rand(*x.shape) > self.noise_factor
            x_noisy = x * mask
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            x_noisy = x.copy()
            num_corrupted = int(self.noise_factor * x.size)
            indices = np.random.choice(x.size, num_corrupted, replace=False)
            x_noisy.flat[indices] = np.random.choice([0, 1], num_corrupted)
        else:
            x_noisy = x

        return np.clip(x_noisy, 0, 1)  # Ensure values are in [0, 1]

    def encode(self, x):
        """Encode input to latent space"""
        for layer in self.encoder:
            x = layer.forward(x)
        return x

    def decode(self, z):
        """Decode from latent space"""
        for layer in self.decoder:
            x = layer.forward(x)
        return x

    def forward(self, x, apply_noise=True):
        """Forward pass"""
        if apply_noise:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x

        z = self.encode(x_noisy)
        x_recon = self.decode(z)

        return x_recon, z

    def reconstruct(self, x):
        """Reconstruct input"""
        x_recon, _ = self.forward(x, apply_noise=False)
        return x_recon

    def compute_loss(self, x, x_recon, beta=0.0):
        """Compute denoising autoencoder loss"""
        # Reconstruction loss
        recon_loss = np.mean((x - x_recon) ** 2)

        # Optional: latent regularization
        if beta > 0:
            z = self.encode(x)
            latent_reg = beta * np.mean(np.sum(z**2, axis=1))
            total_loss = recon_loss + latent_reg
        else:
            total_loss = recon_loss

        return total_loss, recon_loss

class ContractiveAutoencoder(DenoisingAutoencoder):
    """Contractive autoencoder with Jacobian penalty"""
    def __init__(self, input_dim, hidden_dims, latent_dim, lamda=1e-4):
        super().__init__(input_dim, hidden_dims, latent_dim)
        self.lamda = lamda

    def compute_jacobian_penalty(self, x):
        """Compute Frobenius norm of Jacobian of encoder with respect to input"""
        # Simplified numerical approximation
        epsilon = 1e-5
        batch_size = x.shape[0]

        # Encode original input
        z_orig = self.encode(x)

        # Compute Jacobian approximately
        jacobian_norm = 0
        for i in range(x.shape[1]):
            # Perturb one input dimension
            x_perturbed = x.copy()
            x_perturbed[:, i] += epsilon

            # Encode perturbed input
            z_perturbed = self.encode(x_perturbed)

            # Compute difference
            diff = (z_perturbed - z_orig) / epsilon

            # Add to Jacobian norm
            jacobian_norm += np.sum(diff ** 2)

        return jacobian_norm / x.shape[1]

    def compute_loss(self, x, x_recon, beta=0.0, use_jacobian=False):
        """Compute contractive autoencoder loss"""
        recon_loss, _ = super().compute_loss(x, x_recon, beta)

        if use_jacobian:
            jacobian_penalty = self.compute_jacobian_penalty(x)
            total_loss = recon_loss + self.lamda * jacobian_penalty
        else:
            total_loss = recon_loss

        return total_loss, recon_loss
```

### 3.2 Variational Autoencoders (VAE)

```python
class VariationalAutoencoder:
    def __init__(self, input_dim, hidden_dims, latent_dim, beta=1.0):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder networks for mean and log variance
        self.encoder_mean = self._build_encoder()
        self.encoder_logvar = self._build_encoder()

        # Decoder
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        """Build encoder network"""
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(DenseLayer(prev_dim, hidden_dim, 'relu'))
            prev_dim = hidden_dim

        # Output layer
        layers.append(DenseLayer(prev_dim, self.latent_dim, 'linear'))

        return layers

    def _build_decoder(self):
        """Build decoder network"""
        layers = []
        prev_dim = self.latent_dim

        for hidden_dim in reversed(self.hidden_dims):
            layers.append(DenseLayer(prev_dim, hidden_dim, 'relu'))
            prev_dim = hidden_dim

        # Output layer
        layers.append(DenseLayer(prev_dim, self.input_dim, 'sigmoid'))

        return layers

    def encode(self, x):
        """Encode to mean and log variance parameters"""
        # Compute mean
        z_mean = x
        for layer in self.encoder_mean:
            z_mean = layer.forward(z_mean)

        # Compute log variance
        z_logvar = x
        for layer in self.encoder_logvar:
            z_logvar = layer.forward(z_logvar)

        return z_mean, z_logvar

    def reparameterize(self, z_mean, z_logvar):
        """Reparameterization trick"""
        # Sample epsilon from standard normal
        eps = np.random.randn(*z_mean.shape)

        # Reparameterize
        z = z_mean + np.sqrt(np.exp(z_logvar)) * eps

        return z

    def decode(self, z):
        """Decode from latent space"""
        x_recon = z
        for layer in self.decoder:
            x_recon = layer.forward(x_recon)

        return x_recon

    def forward(self, x):
        """Forward pass"""
        # Encode
        z_mean, z_logvar = self.encode(x)

        # Sample
        z = self.reparameterize(z_mean, z_logvar)

        # Decode
        x_recon = self.decode(z)

        return x_recon, z_mean, z_logvar

    def compute_loss(self, x, x_recon, z_mean, z_logvar):
        """Compute VAE loss: reconstruction + KL divergence"""
        # Reconstruction loss
        recon_loss = -np.mean(np.sum(x * np.log(x_recon + 1e-8) +
                                   (1 - x) * np.log(1 - x_recon + 1e-8), axis=1))

        # KL divergence: KL(q(z|x) || p(z))
        # p(z) ~ N(0, I)
        kl_loss = -0.5 * np.mean(np.sum(1 + z_logvar - z_mean**2 - np.exp(z_logvar), axis=1))

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def sample(self, n_samples):
        """Sample from the prior and decode"""
        # Sample from prior p(z) ~ N(0, I)
        z = np.random.randn(n_samples, self.latent_dim)

        # Decode
        samples = self.decode(z)

        return samples

    def interpolate(self, x1, x2, n_interpolations=10):
        """Interpolate between two points in latent space"""
        # Encode both points
        z1_mean, z1_logvar = self.encode(x1)
        z2_mean, z2_logvar = self.encode(x2)

        # Sample
        z1 = self.reparameterize(z1_mean, z1_logvar)
        z2 = self.reparameterize(z2_mean, z2_logvar)

        # Interpolate
        interpolations = []
        for i in range(n_interpolations):
            alpha = i / (n_interpolations - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2

            # Decode
            x_interp = self.decode(z_interp)
            interpolations.append(x_interp)

        return interpolations

class BetaVAE(VariationalAutoencoder):
    """β-VAE with controllable disentanglement"""
    def __init__(self, input_dim, hidden_dims, latent_dim, beta=1.0, gamma=100.0, capacity=25.0):
        super().__init__(input_dim, hidden_dims, latent_dim, beta)
        self.gamma = gamma
        self.capacity = capacity
        self.capacity_update = 0.0

    def compute_beta_vae_loss(self, x, x_recon, z_mean, z_logvar, epoch, total_epochs):
        """Compute β-VAE loss with capacity control"""
        recon_loss, _, kl_loss = self.compute_loss(x, x_recon, z_mean, z_logvar)

        # Update capacity
        self.capacity_update = self.capacity * (epoch / total_epochs)

        # Modified KL divergence with capacity
        c = self.capacity_update
        kl_loss_modified = np.abs(kl_loss - c)

        # Total loss
        total_loss = recon_loss + self.gamma * kl_loss_modified

        return total_loss, recon_loss, kl_loss_modified

class ConditionalVAE(VariationalAutoencoder):
    """Conditional VAE for controlled generation"""
    def __init__(self, input_dim, hidden_dims, latent_dim, condition_dim, beta=1.0):
        self.condition_dim = condition_dim
        super().__init__(input_dim, hidden_dims, latent_dim, beta)

        # Modify encoders and decoder to include conditions
        self.encoder_mean = self._build_conditional_encoder()
        self.encoder_logvar = self._build_conditional_encoder()
        self.decoder = self._build_conditional_decoder()

    def _build_conditional_encoder(self):
        """Build encoder that takes both input and condition"""
        layers = []
        prev_dim = self.input_dim + self.condition_dim  # Concatenate input and condition

        for hidden_dim in self.hidden_dims:
            layers.append(DenseLayer(prev_dim, hidden_dim, 'relu'))
            prev_dim = hidden_dim

        layers.append(DenseLayer(prev_dim, self.latent_dim, 'linear'))

        return layers

    def _build_conditional_decoder(self):
        """Build decoder that takes both latent and condition"""
        layers = []
        prev_dim = self.latent_dim + self.condition_dim  # Concatenate latent and condition

        for hidden_dim in reversed(self.hidden_dims):
            layers.append(DenseLayer(prev_dim, hidden_dim, 'relu'))
            prev_dim = hidden_dim

        layers.append(DenseLayer(prev_dim, self.input_dim, 'sigmoid'))

        return layers

    def forward(self, x, conditions):
        """Forward pass with conditions"""
        # Concatenate input and condition
        x_cond = np.concatenate([x, conditions], axis=1)

        # Encode
        z_mean, z_logvar = self.encode(x_cond)

        # Sample
        z = self.reparameterize(z_mean, z_logvar)

        # Concatenate latent and condition
        z_cond = np.concatenate([z, conditions], axis=1)

        # Decode
        x_recon = self.decode(z_cond)

        return x_recon, z_mean, z_logvar

    def sample_with_condition(self, conditions, n_samples=1):
        """Sample with specific conditions"""
        # Sample from prior
        z = np.random.randn(n_samples, self.latent_dim)

        # Concatenate with conditions
        z_cond = np.concatenate([z, conditions], axis=1)

        # Decode
        samples = self.decode(z_cond)

        return samples
```

---

## 4. Generative Adversarial Networks (GANs) {#gans}

### 4.1 Basic GAN

```python
class Generator:
    def __init__(self, latent_dim, hidden_dims, output_dim):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build generator network
        self.layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            self.layers.append(DenseLayer(prev_dim, hidden_dim, 'relu'))
            prev_dim = hidden_dim

        # Output layer
        self.layers.append(DenseLayer(prev_dim, output_dim, 'tanh'))

    def forward(self, z):
        """Generate samples from latent code"""
        x = z
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def generate(self, n_samples):
        """Generate random samples"""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.forward(z)

class Discriminator:
    def __init__(self, input_dim, hidden_dims):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build discriminator network
        self.layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(DenseLayer(prev_dim, hidden_dim, 'relu'))
            prev_dim = hidden_dim

        # Output layer
        self.layers.append(DenseLayer(prev_dim, 1, 'sigmoid'))

    def forward(self, x):
        """Discriminate real vs fake"""
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def discriminate(self, x):
        """Return discrimination probabilities"""
        return self.forward(x)

class GAN:
    def __init__(self, latent_dim, generator_dims, discriminator_dims, input_dim):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, generator_dims, input_dim)
        self.discriminator = Discriminator(input_dim, discriminator_dims)

    def train_generator(self, batch_size):
        """Train generator by fooling discriminator"""
        # Generate fake samples
        z = np.random.randn(batch_size, self.latent_dim)
        fake_samples = self.generator.forward(z)

        # Discriminator predictions on fake samples
        fake_labels = self.discriminator.discriminate(fake_samples)

        # Generator wants discriminator to predict real (1)
        generator_loss = -np.mean(np.log(fake_labels + 1e-8))

        return generator_loss, fake_samples

    def train_discriminator(self, real_samples, batch_size):
        """Train discriminator to distinguish real from fake"""
        # Generate fake samples
        z = np.random.randn(batch_size, self.latent_dim)
        fake_samples = self.generator.forward(z)

        # Discriminator predictions
        real_pred = self.discriminator.discriminate(real_samples)
        fake_pred = self.discriminator.discriminate(fake_samples)

        # Discriminator loss (binary cross-entropy)
        real_loss = -np.mean(np.log(real_pred + 1e-8))
        fake_loss = -np.mean(np.log(1 - fake_pred + 1e-8))

        total_loss = 0.5 * (real_loss + fake_loss)

        return total_loss, real_loss, fake_loss

    def train_step(self, real_samples, batch_size):
        """One training step"""
        # Train discriminator
        d_loss, d_real_loss, d_fake_loss = self.train_discriminator(real_samples, batch_size)

        # Train generator
        g_loss, fake_samples = self.train_generator(batch_size)

        return {
            'generator_loss': g_loss,
            'discriminator_loss': d_loss,
            'discriminator_real_loss': d_real_loss,
            'discriminator_fake_loss': d_fake_loss,
            'generated_samples': fake_samples
        }

    def generate_samples(self, n_samples):
        """Generate new samples"""
        return self.generator.generate(n_samples)
```

### 4.2 Deep Convolutional GAN (DCGAN)

```python
class DCGenerator:
    def __init__(self, latent_dim, base_channels=64):
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        # Generator layers (upsampling)
        self.layers = [
            # Input: (latent_dim,) -> (4, 4, base_channels*8)
            DenseLayer(latent_dim, 4 * 4 * base_channels * 8, 'linear'),
            ReshapeLayer((4, 4, base_channels * 8)),

            # Transposed convolutions for upsampling
            TransposedConvLayer(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            BatchNormalizationLayer(base_channels * 4),
            ReLULayer(),

            TransposedConvLayer(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            BatchNormalizationLayer(base_channels * 2),
            ReLULayer(),

            TransposedConvLayer(base_channels * 2, base_channels, 4, stride=2, padding=1),
            BatchNormalizationLayer(base_channels),
            ReLULayer(),

            # Output: (64, 64, 3)
            TransposedConvLayer(base_channels, 3, 4, stride=2, padding=1),
            TanhLayer()
        ]

    def forward(self, z):
        """Generate images from latent code"""
        x = z
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                x = layer.forward(x)
            else:
                x = layer(x)
        return x

class DCDiscriminator:
    def __init__(self, base_channels=64):
        self.base_channels = base_channels

        # Discriminator layers (downsampling)
        self.layers = [
            # Input: (64, 64, 3) -> (32, 32, base_channels)
            ConvLayer(3, base_channels, 4, stride=2, padding=1),
            LeakyReLULayer(0.2),

            # (32, 32, base_channels) -> (16, 16, base_channels*2)
            ConvLayer(base_channels, base_channels * 2, 4, stride=2, padding=1),
            BatchNormalizationLayer(base_channels * 2),
            LeakyReLULayer(0.2),

            # (16, 16, base_channels*2) -> (8, 8, base_channels*4)
            ConvLayer(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            BatchNormalizationLayer(base_channels * 4),
            LeakyReLULayer(0.2),

            # (8, 8, base_channels*4) -> (4, 4, base_channels*8)
            ConvLayer(base_channels * 4, base_channels * 8, 4, stride=2, padding=1),
            BatchNormalizationLayer(base_channels * 8),
            LeakyReLULayer(0.2),

            # Flatten and classify
            FlattenLayer(),
            DenseLayer(4 * 4 * base_channels * 8, 1, 'sigmoid')
        ]

    def forward(self, x):
        """Discriminate real vs fake images"""
        y = x
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                y = layer.forward(y)
            else:
                y = layer(y)
        return y

# Supporting layers for DCGAN
class ReshapeLayer:
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.target_shape)

class TransposedConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights
        self.weight = np.random.randn(kernel_size, kernel_size, out_channels, in_channels) * 0.02
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        # Simplified transposed convolution
        batch_size, in_height, in_width, in_channels = x.shape

        # Calculate output dimensions
        out_height = (in_height - 1) * self.stride + self.kernel_size - 2 * self.padding
        out_width = (in_width - 1) * self.stride + self.kernel_size - 2 * self.padding

        # Initialize output
        output = np.zeros((batch_size, out_height, out_width, self.out_channels))

        # Simplified transposed convolution (for demonstration)
        # In practice, this would be more complex
        for b in range(batch_size):
            for i in range(out_height):
                for j in range(out_width):
                    for oc in range(self.out_channels):
                        val = 0
                        for kh in range(self.kernel_size):
                            for kw in range(self.kernel_size):
                                for ic in range(self.in_channels):
                                    in_h = (i - kh + self.padding) // self.stride
                                    in_w = (j - kw + self.padding) // self.stride

                                    if 0 <= in_h < in_height and 0 <= in_w < in_width:
                                        val += x[b, in_h, in_w, ic] * self.weight[kh, kw, oc, ic]

                        output[b, i, j, oc] = val + self.bias[oc]

        return output

class LeakyReLULayer:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)

class TanhLayer:
    def forward(self, x):
        return np.tanh(x)

class ReLULayer:
    def forward(self, x):
        return np.maximum(0, x)

class FlattenLayer:
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
```

### 4.3 Wasserstein GAN (WGAN)

```python
class WGAN:
    def __init__(self, latent_dim, generator_dims, discriminator_dims, input_dim,
                 clip_value=0.01):
        self.latent_dim = latent_dim
        self.clip_value = clip_value

        # Generator and discriminator
        self.generator = Generator(latent_dim, generator_dims, input_dim)
        self.discriminator = WDiscriminator(input_dim, discriminator_dims)

    def train_discriminator(self, real_samples, batch_size, n_critic=5):
        """Train discriminator with Wasserstein loss"""
        total_loss = 0

        for _ in range(n_critic):
            # Generate fake samples
            z = np.random.randn(batch_size, self.latent_dim)
            fake_samples = self.generator.forward(z)

            # Discriminator predictions
            real_pred = self.discriminator.discriminate(real_samples)
            fake_pred = self.discriminator.discriminate(fake_samples)

            # Wasserstein loss
            loss = -np.mean(real_pred) + np.mean(fake_pred)

            # Clip discriminator weights
            self.discriminator.clip_weights(self.clip_value)

            total_loss += loss

        return total_loss / n_critic

    def train_generator(self, batch_size):
        """Train generator with Wasserstein loss"""
        # Generate samples
        z = np.random.randn(batch_size, self.latent_dim)
        fake_samples = self.generator.forward(z)

        # Discriminator predictions on fake samples
        fake_pred = self.discriminator.discriminate(fake_samples)

        # Generator wants to maximize discriminator output
        loss = -np.mean(fake_pred)

        return loss, fake_samples

class WDiscriminator:
    def __init__(self, input_dim, hidden_dims):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build discriminator without final sigmoid
        self.layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(DenseLayer(prev_dim, hidden_dim, 'relu'))
            prev_dim = hidden_dim

        # Linear output (no activation)
        self.output_layer = DenseLayer(prev_dim, 1, 'linear')

    def forward(self, x):
        """Discriminate with linear output"""
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        y = self.output_layer.forward(y)
        return y

    def discriminate(self, x):
        """Return discrimination scores"""
        return self.forward(x)

    def clip_weights(self, clip_value):
        """Clip discriminator weights"""
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                np.clip(layer.weights, -clip_value, clip_value, out=layer.weights)
                if layer.use_bias:
                    np.clip(layer.bias, -clip_value, clip_value, out=layer.bias)

        if hasattr(self.output_layer, 'weights'):
            np.clip(self.output_layer.weights, -clip_value, clip_value, out=self.output_layer.weights)
            if self.output_layer.use_bias:
                np.clip(self.output_layer.bias, -clip_value, clip_value, out=self.output_layer.bias)

class WGAN_GP:
    """WGAN with Gradient Penalty"""
    def __init__(self, latent_dim, generator_dims, discriminator_dims, input_dim,
                 lambda_gp=10.0):
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp

        self.generator = Generator(latent_dim, generator_dims, input_dim)
        self.discriminator = WDiscriminator(input_dim, discriminator_dims)

    def compute_gradient_penalty(self, real_samples, fake_samples, batch_size):
        """Compute gradient penalty for improved WGAN"""
        # Generate random samples for interpolation
        alpha = np.random.rand(batch_size, 1, 1, 1)  # Assuming image input

        # Interpolate between real and fake samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples

        # Compute discriminator output on interpolates
        interpolate_preds = self.discriminator.discriminate(interpolates)

        # Compute gradients
        gradients = self.compute_gradients(interpolates, interpolate_preds)

        # Compute gradient penalty
        gradient_norm = np.sqrt(np.sum(gradients**2, axis=(1, 2, 3)))  # For images
        gradient_penalty = self.lambda_gp * np.mean((gradient_norm - 1)**2)

        return gradient_penalty

    def compute_gradients(self, x, preds):
        """Compute gradients with respect to inputs"""
        # Simplified gradient computation
        # In practice, you'd use automatic differentiation
        batch_size = x.shape[0]
        gradients = np.zeros_like(x)

        epsilon = 1e-5
        for i in range(batch_size):
            for j in range(x.shape[1]):  # Assuming flattened or 1D
                x_plus = x[i].copy()
                x_minus = x[i].copy()
                x_plus[j] += epsilon
                x_minus[j] -= epsilon

                pred_plus = self.discriminator.discriminate(x_plus[np.newaxis, :])
                pred_minus = self.discriminator.discriminate(x_minus[np.newaxis, :])

                gradients[i, j] = (pred_plus - pred_minus) / (2 * epsilon)

        return gradients

    def train_discriminator(self, real_samples, batch_size):
        """Train discriminator with gradient penalty"""
        # Generate fake samples
        z = np.random.randn(batch_size, self.latent_dim)
        fake_samples = self.generator.forward(z)

        # Discriminator predictions
        real_pred = self.discriminator.discriminate(real_samples)
        fake_pred = self.discriminator.discriminate(fake_samples)

        # Wasserstein loss
        wasserstein_loss = -np.mean(real_pred) + np.mean(fake_pred)

        # Gradient penalty
        gp = self.compute_gradient_penalty(real_samples, fake_samples, batch_size)

        # Total loss
        total_loss = wasserstein_loss + gp

        return total_loss, wasserstein_loss, gp

    def train_generator(self, batch_size):
        """Train generator"""
        z = np.random.randn(batch_size, self.latent_dim)
        fake_samples = self.generator.forward(z)
        fake_pred = self.discriminator.discriminate(fake_samples)

        # Generator wants to maximize discriminator output
        loss = -np.mean(fake_pred)

        return loss, fake_samples
```

This comprehensive guide to advanced neural network theory covers specialized architectures like Capsule Networks, normalizing flows, graph neural networks, and advanced autoencoders. Each section provides both theoretical foundations and practical implementations, enabling deep understanding of these sophisticated approaches to neural network design and training.

## Mini Sprint Project (30-45 minutes)

**Objective:** Build a Simple Variational Autoencoder

**Data/Input sample:** Small dataset of grayscale images (28x28) or simple 2D patterns

**Steps / Milestones:**

- **Step A:** Implement the VAE encoder network (input → mean/logvar)
- **Step B:** Create the reparameterization trick for sampling
- **Step C:** Build the decoder network (latent → reconstruction)
- **Step D:** Implement the combined loss (reconstruction + KL divergence)
- **Step E:** Train the model and visualize reconstructions
- **Step F:** Generate new samples from the learned latent space

**Success criteria:** Working VAE that can reconstruct input images and generate new samples from the latent space

**Code Framework:**

```python
# VAE Implementation Framework
class SimpleVAE:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        # Your implementation here

    def encode(self, x):
        # Convert input to mean and log variance

    def reparameterize(self, mu, logvar):
        # Reparameterization trick for backpropagation

    def decode(self, z):
        # Generate reconstruction from latent space

    def forward(self, x):
        # Complete forward pass

    def loss_function(self, x, x_recon, mu, logvar):
        # Combined reconstruction and KL loss
```

## Full Project Extension (6-12 hours)

**Project brief:** Multi-Architecture Neural Network Research Platform

**Deliverables:**

- Complete implementation of 3+ advanced architectures (Capsule Networks, GNNs, VAEs)
- Comparative analysis framework for different approaches
- Interactive visualization system for latent spaces and model behavior
- Comprehensive training pipeline with experiment tracking
- Research report comparing architectures on standard benchmarks

**Skills demonstrated:**

- Advanced neural network architecture design
- Multi-domain implementation (images, graphs, probabilistic models)
- Experimental design and comparative analysis
- Visualization and interpretation of complex model behaviors
- Research methodology and documentation
- Performance optimization and debugging

**Project Structure:**

```
research_platform/
├── models/
│   ├── capsule_networks.py
│   ├── graph_neural_networks.py
│   ├── variational_autoencoders.py
│   └── generative_adversarial_networks.py
├── data/
│   ├── datasets.py
│   └── preprocessing.py
├── training/
│   ├── train_utils.py
│   ├── experiment_tracker.py
│   └── evaluation.py
├── visualization/
│   ├── latent_spaces.py
│   ├── training_curves.py
│   └── model_analysis.py
└── research/
    ├── comparative_study.py
    ├── benchmark_results.md
    └── final_report.md
```

**Key Challenges:**

- Implementing mathematically sophisticated algorithms correctly
- Managing computational requirements for multiple architectures
- Designing fair comparisons between different approaches
- Interpreting and visualizing high-dimensional model behavior
- Writing comprehensive research documentation

**Success Criteria:**

- All architectures implemented and tested successfully
- Comparative analysis reveals meaningful insights
- Visualizations clearly demonstrate model capabilities
- Research report meets academic standards
- Code is well-documented and reproducible
