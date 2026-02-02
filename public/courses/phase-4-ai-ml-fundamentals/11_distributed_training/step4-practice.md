# Advanced Neural Networks Practice

## Table of Contents

1. [Implementing Attention Mechanisms](#attention-implementation)
2. [Building Transformers from Scratch](#transformers-practice)
3. [Creating Autoencoder Implementations](#autoencoders-practice)
4. [Building GAN Implementations](#gans-practice)
5. [Multi-modal Network Examples](#multimodal-practice)
6. [Advanced Architectural Experiments](#architectural-experiments)
7. [Performance Optimization Techniques](#performance-optimization)
8. [Debugging and Monitoring](#debugging-monitoring)
9. [Production Deployment](#production-deployment)

---

## 1. Implementing Attention Mechanisms {#attention-implementation}

### 1.1 Scaled Dot-Product Attention

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import time

class ScaledDotProductAttention:
    def __init__(self, d_k: int, dropout_rate: float = 0.0):
        self.d_k = d_k  # Key dimension
        self.dropout_rate = dropout_rate
        self.attention_weights = None
        self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention

        Args:
            query: (batch_size, seq_len, d_k)
            key: (batch_size, seq_len, d_k)
            value: (batch_size, seq_len, d_v)
            mask: (batch_size, seq_len, seq_len) - where True means mask out

        Returns:
            output: (batch_size, seq_len, d_v)
            attention_weights: (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, d_k = query.shape

        # 1. Compute attention scores: QK^T / sqrt(d_k)
        scores = np.dot(query, key.transpose(0, 2, 1)) / np.sqrt(self.d_k)

        # 2. Apply mask if provided
        if mask is not None:
            # Use a very large negative number for masked positions
            scores = np.where(mask, -1e9, scores)

        # 3. Apply softmax
        attention_weights = self.softmax(scores)

        # 4. Apply dropout
        if self.dropout is not None:
            attention_weights = self.dropout.forward(attention_weights)

        # 5. Apply attention to values: softmax(QK^T)V
        output = np.dot(attention_weights, value)

        self.attention_weights = attention_weights.copy()
        return output, attention_weights

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def visualize_attention(self, save_path: str = "attention_heatmap.png"):
        """Visualize attention weights"""
        if self.attention_weights is None:
            print("No attention weights to visualize. Run forward pass first.")
            return

        # Take first sample and head
        attention = self.attention_weights[0, 0]  # (seq_len, seq_len)

        plt.figure(figsize=(10, 8))
        plt.imshow(attention, cmap='Blues', aspect='auto')
        plt.colorbar()
        plt.title('Attention Weights')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.0):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_k  # Assume d_v = d_k

        # Linear projections for Q, K, V
        self.W_q = LinearLayer(d_model, d_model, bias=False)
        self.W_k = LinearLayer(d_model, d_model, bias=False)
        self.W_v = LinearLayer(d_model, d_model, bias=False)
        self.W_o = LinearLayer(d_model, d_model, bias=False)

        # Attention mechanism
        self.attention = ScaledDotProductAttention(self.d_k, dropout_rate)

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-head attention forward pass

        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)
            value: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len) or (batch_size, seq_len, seq_len)

        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: List of attention matrices per head
        """
        batch_size, seq_len, _ = query.shape

        # 1. Linear projections to get Q, K, V
        Q = self.W_q.forward(query)  # (batch_size, seq_len, d_model)
        K = self.W_k.forward(key)
        V = self.W_v.forward(value)

        # 2. Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # 3. Apply attention for each head
        head_outputs = []
        head_attentions = []

        for h in range(self.num_heads):
            Q_h = Q[:, h]  # (batch_size, seq_len, d_k)
            K_h = K[:, h]
            V_h = V[:, h]

            # Apply mask to each head
            mask_h = mask if mask is None or mask.shape[1] == 1 else mask[:, h:h+1, :]

            output_h, attention_h = self.attention.forward(Q_h, K_h, V_h, mask_h)
            head_outputs.append(output_h)
            head_attentions.append(attention_h)

        # 4. Concatenate heads
        concatenated = np.concatenate(head_outputs, axis=-1)  # (batch_size, seq_len, d_model)

        # 5. Output linear transformation
        output = self.W_o.forward(concatenated)

        return output, head_attentions

# Helper classes
class LinearLayer:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize weights
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        if bias:
            self.bias = np.zeros(out_features)
        else:
            self.bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Linear forward pass"""
        if x.ndim == 2:
            # (batch_size, in_features) -> (batch_size, out_features)
            output = np.dot(x, self.weight)
        else:
            # (batch_size, seq_len, in_features) -> (batch_size, seq_len, out_features)
            output = np.dot(x, self.weight)

        if self.use_bias:
            if x.ndim == 2:
                output += self.bias
            else:
                output += self.bias[np.newaxis, :]

        return output

class Dropout:
    def __init__(self, dropout_rate: float):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.dropout_rate == 0:
            return x

        # Create dropout mask
        self.mask = (np.random.rand(*x.shape) > self.dropout_rate) / (1 - self.dropout_rate)
        return x * self.mask

# Practice Exercise: Implement Causal Self-Attention
class CausalSelfAttention:
    """Self-attention with causal masking for autoregressive models"""
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.0):
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)

    def create_causal_mask(self, seq_len: int, batch_size: int) -> np.ndarray:
        """Create causal mask to prevent attending to future positions"""
        # Create mask where (i, j) = 0 if j > i (future position)
        mask = np.tril(np.ones((seq_len, seq_len), dtype=bool), k=0)
        return np.broadcast_to(mask, (batch_size, 1, seq_len))

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Causal self-attention forward pass"""
        batch_size, seq_len, d_model = x.shape
        causal_mask = self.create_causal_mask(seq_len, batch_size)

        # Use x as query, key, and value
        output, attention_weights = self.mha.forward(x, x, x, mask=causal_mask)

        return output, attention_weights

# Test implementation
def test_attention_mechanisms():
    """Test attention implementations"""
    print("Testing Attention Mechanisms")
    print("=" * 40)

    # Test parameters
    batch_size, seq_len, d_model, num_heads = 2, 10, 64, 8
    d_k = d_model // num_heads

    # Create dummy input
    x = np.random.randn(batch_size, seq_len, d_model)

    # Test Scaled Dot-Product Attention
    print("1. Testing Scaled Dot-Product Attention...")
    attention = ScaledDotProductAttention(d_k)
    output, weights = attention.forward(x, x, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {weights.shape}")

    # Test Multi-Head Attention
    print("\n2. Testing Multi-Head Attention...")
    mha = MultiHeadAttention(d_model, num_heads)
    mha_output, mha_attentions = mha.forward(x, x, x)
    print(f"   Multi-head output shape: {mha_output.shape}")
    print(f"   Number of attention heads: {len(mha_attentions)}")

    # Test Causal Self-Attention
    print("\n3. Testing Causal Self-Attention...")
    causal_attn = CausalSelfAttention(d_model, num_heads)
    causal_output, causal_weights = causal_attn.forward(x)
    print(f"   Causal attention output shape: {causal_output.shape}")
    print(f"   Causal weights shape: {causal_weights[0].shape}")

    # Visualize attention patterns
    print("\n4. Visualizing attention patterns...")
    attention.visualize_attention("attention_example.png")
    mha.visualize_attention("multihead_attention.png")

    print("\nAttention mechanism testing completed!")

if __name__ == "__main__":
    test_attention_mechanisms()
```

### 1.2 Cross-Attention Implementation

```python
class CrossAttention:
    """Cross-attention for encoder-decoder models"""
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.0):
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.d_model = d_model
        self.num_heads = num_heads

    def forward(self, query: np.ndarray, encoder_output: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cross-attention forward pass

        Args:
            query: (batch_size, target_seq_len, d_model) - decoder hidden states
            encoder_output: (batch_size, source_seq_len, d_model) - encoder outputs
            mask: (batch_size, target_seq_len, source_seq_len) - attention mask

        Returns:
            output: (batch_size, target_seq_len, d_model)
            attention_weights: (batch_size, target_seq_len, source_seq_len)
        """
        # Cross-attention: query from decoder, key-value from encoder
        output, attention_weights = self.mha.forward(query, encoder_output, encoder_output, mask)

        return output, attention_weights

    def create_padding_mask(self, source_lengths: np.ndarray, source_seq_len: int) -> np.ndarray:
        """
        Create padding mask for source sequence

        Args:
            source_lengths: (batch_size,) - actual lengths of sequences
            source_seq_len: maximum sequence length

        Returns:
            mask: (batch_size, 1, source_seq_len) - True for padding positions
        """
        batch_size = len(source_lengths)
        mask = np.ones((batch_size, 1, source_seq_len), dtype=bool)

        for i, length in enumerate(source_lengths):
            mask[i, 0, length:] = False  # True means keep, False means mask

        return mask

    def forward_with_padding(self, query: np.ndarray, encoder_output: np.ndarray,
                           source_lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cross-attention with automatic padding mask creation"""
        batch_size, source_seq_len, _ = encoder_output.shape
        padding_mask = self.create_padding_mask(source_lengths, source_seq_len)

        # Combine with causal mask if needed (for decoder self-attention)
        # In cross-attention, we only need padding mask
        output, attention_weights = self.forward(query, encoder_output, padding_mask)

        return output, attention_weights

# Practice: Implement Attention Visualization
class AttentionVisualizer:
    """Visualize attention patterns for analysis"""
    def __init__(self, save_dir: str = "attention_visuals/"):
        import os
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def visualize_multihead_attention(self, attention_weights: list,
                                     tokens: Optional[list] = None,
                                     save_name: str = "multihead_attention.png"):
        """Visualize attention weights from multiple heads"""
        num_heads = len(attention_weights)

        if num_heads <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
        else:
            n_cols = min(4, num_heads)
            n_rows = (num_heads + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols > 1:
                axes = axes[np.newaxis, :]
            elif n_rows == 1 and n_cols == 1:
                axes = axes[np.newaxis, np.newaxis]

        for i, attn_weights in enumerate(attention_weights):
            if num_heads <= 4:
                ax = axes[i]
            else:
                row, col = i // n_cols, i % n_cols
                ax = axes[row, col]

            # Take first sample
            if len(attn_weights.shape) == 3:
                attn_matrix = attn_weights[0]  # (seq_len, seq_len)
            else:
                attn_matrix = attn_weights  # (seq_len, seq_len)

            im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto')
            ax.set_title(f'Head {i+1}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')

            # Add token labels if provided
            if tokens:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right')
                ax.set_yticklabels(tokens)

            plt.colorbar(im, ax=ax)

        # Hide unused subplots
        if num_heads <= 4:
            for i in range(num_heads, 4):
                axes[i].set_visible(False)
        else:
            for i in range(num_heads, n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        plt.show()

    def compare_attention_patterns(self, attention_sets: dict, save_name: str = "attention_comparison.png"):
        """
        Compare attention patterns between different models/layers

        Args:
            attention_sets: dict with keys as model names and values as attention weights
        """
        n_models = len(attention_sets)

        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]

        for i, (model_name, attention_weights) in enumerate(attention_sets.items()):
            # Take first head and first sample
            attn_matrix = attention_weights[0][0]  # (seq_len, seq_len)

            im = axes[i].imshow(attn_matrix, cmap='Blues', aspect='auto')
            axes[i].set_title(f'{model_name} - Head 1')
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Query Position')
            plt.colorbar(im, ax=axes[i])

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        plt.show()

# Test cross-attention implementation
def test_cross_attention():
    """Test cross-attention implementation"""
    print("Testing Cross-Attention")
    print("=" * 30)

    # Parameters
    batch_size, target_len, source_len, d_model, num_heads = 2, 8, 10, 64, 8

    # Create dummy inputs
    decoder_input = np.random.randn(batch_size, target_len, d_model)
    encoder_output = np.random.randn(batch_size, source_len, d_model)
    source_lengths = np.array([8, 10])  # Variable length sequences

    # Test cross-attention
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")

    cross_attn = CrossAttention(d_model, num_heads)
    output, attention_weights = cross_attn.forward_with_padding(
        decoder_input, encoder_output, source_lengths
    )

    print(f"Cross-attention output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights[0].shape}")

    # Visualize attention
    visualizer = AttentionVisualizer()
    visualizer.visualize_multihead_attention(attention_weights,
                                           save_name="cross_attention_example.png")

    print("Cross-attention test completed!")

if __name__ == "__main__":
    test_cross_attention()
```

---

## 2. Building Transformers from Scratch {#transformers-practice}

### 2.1 Complete Transformer Implementation

```python
class PositionalEncoding:
    """Sinusoidal positional encoding"""
    def __init__(self, d_model: int, max_seq_len: int = 1000):
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, batch_size: int = 1) -> np.ndarray:
        """Generate positional encoding for given sequence length"""
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(self.d_model)[np.newaxis, :]

        # Create angle values
        angle_rads = pos / np.power(10000, (2 * (i//2)) / self.d_model)

        # Apply sin to even indices, cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        # Add batch dimension
        pos_encoding = angle_rads[np.newaxis, ...]
        pos_encoding = np.tile(pos_encoding, (batch_size, 1, 1))

        return pos_encoding

class LayerNormalization:
    """Layer normalization"""
    def __init__(self, d_model: int, epsilon: float = 1e-6):
        self.d_model = d_model
        self.epsilon = epsilon
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization"""
        # Compute mean and variance along the last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_normalized = (x - mean) / np.sqrt(variance + self.epsilon)

        # Scale and shift
        return self.gamma * x_normalized + self.beta

class TransformerEncoderLayer:
    """Transformer encoder layer"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.0):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)

        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout_rate)

        # Layer normalization
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

        # Dropout
        self.dropout = Dropout(dropout_rate)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through encoder layer"""
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention.forward(x, x, x, mask)
        attn_output = self.dropout.forward(attn_output)
        x = self.norm1(x + attn_output)

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward.forward(x)
        ff_output = self.dropout.forward(ff_output)
        x = self.norm2(x + ff_output)

        return x

class TransformerDecoderLayer:
    """Transformer decoder layer"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.0):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)

        # Cross-attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)

        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout_rate)

        # Layer normalization
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)

        # Dropout
        self.dropout = Dropout(dropout_rate)

    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                self_mask: Optional[np.ndarray] = None,
                cross_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through decoder layer"""
        # Causal self-attention
        self_attn_output, _ = self.self_attention.forward(x, x, x, self_mask)
        self_attn_output = self.dropout.forward(self_attn_output)
        x = self.norm1(x + self_attn_output)

        # Cross-attention
        cross_attn_output, _ = self.cross_attention.forward(x, encoder_output, encoder_output, cross_mask)
        cross_attn_output = self.dropout.forward(cross_attn_output)
        x = self.norm2(x + cross_attn_output)

        # Feed-forward
        ff_output = self.feed_forward.forward(x)
        ff_output = self.dropout.forward(ff_output)
        x = self.norm3(x + ff_output)

        return x

class FeedForwardNetwork:
    """Position-wise feed-forward network"""
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.0):
        self.linear1 = LinearLayer(d_model, d_ff, bias=True)
        self.linear2 = LinearLayer(d_ff, d_model, bias=True)
        self.dropout = Dropout(dropout_rate)
        self.activation = ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through feed-forward network"""
        x = self.linear1.forward(x)
        x = self.activation.forward(x)
        x = self.dropout.forward(x)
        x = self.linear2.forward(x)
        return x

class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

class Transformer:
    """Complete Transformer architecture"""
    def __init__(self, d_model: int, num_heads: int, num_layers: int,
                 d_ff: int, vocab_size: int, max_seq_len: int, dropout_rate: float = 0.0):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        # Token embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Encoder and decoder
        self.encoder_layers = []
        self.decoder_layers = []

        for _ in range(num_layers):
            self.encoder_layers.append(
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout_rate)
            )
            self.decoder_layers.append(
                TransformerDecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            )

        # Final layer
        self.output_projection = LinearLayer(d_model, vocab_size, bias=False)

    def create_padding_mask(self, x: np.ndarray, pad_token_id: int = 0) -> np.ndarray:
        """Create padding mask"""
        mask = (x != pad_token_id)[:, np.newaxis, :]  # (batch_size, 1, seq_len)
        return mask

    def create_look_ahead_mask(self, size: int) -> np.ndarray:
        """Create look-ahead mask for decoder"""
        return np.tril(np.ones((size, size), dtype=bool), k=0)

    def encode(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode input sequence"""
        # Token embeddings
        x = self.token_embedding.forward(x)

        # Scale by sqrt(d_model)
        x = x * np.sqrt(self.d_model)

        # Add positional encoding
        seq_len = x.shape[1]
        x += self.pos_encoding.forward(seq_len, x.shape[0])

        # Apply dropout
        x = Dropout(self.dropout_rate).forward(x)

        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer.forward(x, mask)

        return x

    def decode(self, x: np.ndarray, encoder_output: np.ndarray,
               self_mask: Optional[np.ndarray] = None,
               cross_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Decode using encoder output"""
        # Token embeddings
        x = self.token_embedding.forward(x)

        # Scale and add positional encoding
        x = x * np.sqrt(self.d_model)
        seq_len = x.shape[1]
        x += self.pos_encoding.forward(seq_len, x.shape[0])

        # Apply dropout
        x = Dropout(self.dropout_rate).forward(x)

        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer.forward(x, encoder_output, self_mask, cross_mask)

        return x

    def forward(self, encoder_input: np.ndarray, decoder_input: np.ndarray,
                encoder_mask: Optional[np.ndarray] = None,
                decoder_mask: Optional[np.ndarray] = None,
                cross_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Full Transformer forward pass"""
        # Encode
        encoder_output = self.encode(encoder_input, encoder_mask)

        # Decode
        decoder_output = self.decode(decoder_input, encoder_output, decoder_mask, cross_mask)

        # Output projection
        output = self.output_projection.forward(decoder_output)

        return output

    def generate(self, encoder_input: np.ndarray, start_token: int,
                 end_token: int, max_length: int) -> np.ndarray:
        """Generate sequence autoregressively"""
        batch_size = encoder_input.shape[0]
        generated = np.full((batch_size, 1), start_token, dtype=int)

        # Encode input
        encoder_output = self.encode(encoder_input)

        for _ in range(max_length - 1):
            # Create masks
            self_mask = self.create_look_ahead_mask(generated.shape[1])
            self_mask = np.broadcast_to(self_mask, (batch_size, 1, generated.shape[1]))

            # Forward pass
            output = self.decode(generated, encoder_output, self_mask)

            # Get next token (greedy decoding)
            next_token = np.argmax(output[:, -1, :], axis=-1, keepdims=True)
            generated = np.concatenate([generated, next_token], axis=1)

            # Stop if all sequences have end token
            if np.all(next_token == end_token):
                break

        return generated

class TokenEmbedding:
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Token embedding forward pass"""
        return self.embedding[x]

# Test Transformer implementation
def test_transformer():
    """Test Transformer implementation"""
    print("Testing Transformer Implementation")
    print("=" * 40)

    # Parameters
    batch_size, seq_len, vocab_size, d_model, num_heads, num_layers, d_ff = 2, 10, 1000, 64, 8, 3, 256

    # Create dummy data
    encoder_input = np.random.randint(0, vocab_size, (batch_size, seq_len))
    decoder_input = np.random.randint(0, vocab_size, (batch_size, seq_len))

    # Create Transformer
    transformer = Transformer(d_model, num_heads, num_layers, d_ff, vocab_size, 1000)

    # Test forward pass
    print("Testing forward pass...")
    output = transformer.forward(encoder_input, decoder_input)
    print(f"Input shapes: encoder {encoder_input.shape}, decoder {decoder_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test generation
    print("\nTesting sequence generation...")
    generated = transformer.generate(encoder_input[:, :5], start_token=1, end_token=2, max_length=15)
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated sequence: {generated[0]}")

    print("\nTransformer test completed!")

if __name__ == "__main__":
    test_transformer()
```

### 2.2 Vision Transformer (ViT) Practice

```python
class PatchEmbedding:
    """Patch embedding for Vision Transformer"""
    def __init__(self, patch_size: int, d_model: int):
        self.patch_size = patch_size
        self.d_model = d_model
        self.projection = LinearLayer(patch_size * patch_size * 3, d_model, bias=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Convert image patches to embedding vectors

        Args:
            x: (batch_size, height, width, channels) - images

        Returns:
            embeddings: (batch_size, num_patches, d_model)
        """
        batch_size, height, width, channels = x.shape

        # Extract patches
        patches = self.extract_patches(x)

        # Flatten patches
        patches = patches.reshape(batch_size, -1, channels * self.patch_size * self.patch_size)

        # Project to embedding dimension
        embeddings = self.projection.forward(patches)

        return embeddings

    def extract_patches(self, x: np.ndarray) -> np.ndarray:
        """Extract patches from image"""
        batch_size, height, width, channels = x.shape

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * self.patch_size
                end_h = start_h + self.patch_size
                start_w = j * self.patch_size
                end_w = start_w + self.patch_size

                patch = x[:, start_h:end_h, start_w:end_w, :]
                patches.append(patch)

        return np.stack(patches, axis=1)  # (batch_size, num_patches, patch_h, patch_w, channels)

class VisionTransformer:
    """Vision Transformer implementation"""
    def __init__(self, patch_size: int, d_model: int, num_heads: int,
                 num_layers: int, num_classes: int, image_size: int, dropout_rate: float = 0.0):
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.image_size = image_size

        # Number of patches
        num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_size, d_model)

        # Class token
        self.class_token = np.random.randn(1, d_model) * 0.02

        # Positional embedding
        self.pos_embedding = np.random.randn(num_patches + 1, d_model) * 0.02

        # Transformer encoder layers
        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append(
                TransformerEncoderLayer(d_model, num_heads, d_model * 4, dropout_rate)
            )

        # Classification head
        self.classifier = LinearLayer(d_model, num_classes, bias=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through ViT

        Args:
            x: (batch_size, height, width, channels) - images

        Returns:
            logits: (batch_size, num_classes) - classification logits
        """
        batch_size = x.shape[0]

        # Patch embedding
        embeddings = self.patch_embedding.forward(x)  # (batch_size, num_patches, d_model)

        # Add class token
        class_tokens = np.tile(self.class_token, (batch_size, 1, 1))
        embeddings = np.concatenate([class_tokens, embeddings], axis=1)

        # Add positional embedding
        embeddings = embeddings + self.pos_embedding[np.newaxis, :, :]

        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            embeddings = layer.forward(embeddings)

        # Use class token for classification
        class_token_output = embeddings[:, 0]  # (batch_size, d_model)

        # Classification
        logits = self.classifier.forward(class_token_output)

        return logits

# Practice: Image Classification with ViT
class ViTTrainer:
    """Trainer for Vision Transformer"""
    def __init__(self, model: VisionTransformer, learning_rate: float = 1e-4):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = AdamOptimizer(learning_rate)

    def train_epoch(self, train_loader, loss_fn):
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0

        for batch_images, batch_labels in train_loader:
            # Forward pass
            logits = self.model.forward(batch_images)

            # Compute loss
            loss = loss_fn(logits, batch_labels)

            # Compute gradients (simplified)
            gradients = self.compute_gradients(logits, batch_images, batch_labels, loss_fn)

            # Update parameters
            self.update_parameters(gradients)

            total_loss += loss
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, test_loader, metric_fn):
        """Evaluate model"""
        total_metric = 0
        num_samples = 0

        for batch_images, batch_labels in test_loader:
            logits = self.model.forward(batch_images)
            predictions = np.argmax(logits, axis=1)

            batch_metric = metric_fn(predictions, batch_labels)
            total_metric += batch_metric * len(batch_labels)
            num_samples += len(batch_labels)

        return total_metric / num_samples

    def compute_gradients(self, logits, images, labels, loss_fn):
        """Simplified gradient computation"""
        # In practice, you'd use automatic differentiation
        batch_size = len(labels)

        # One-hot encode labels
        y_true = np.zeros_like(logits)
        y_true[np.arange(batch_size), labels] = 1

        # Compute gradients of loss with respect to logits
        d_logits = (logits - y_true) / batch_size  # Simplified cross-entropy gradient

        # Backpropagate through the model (simplified)
        gradients = self.backpropagate(d_logits, images)

        return gradients

    def backpropagate(self, d_logits, images):
        """Simplified backpropagation"""
        # This is a placeholder - real implementation would compute gradients
        # through all layers of the model
        gradients = {}
        return gradients

    def update_parameters(self, gradients):
        """Update model parameters using optimizer"""
        # Simplified parameter update
        pass

# Test ViT implementation
def test_vit():
    """Test Vision Transformer implementation"""
    print("Testing Vision Transformer")
    print("=" * 30)

    # Parameters
    batch_size, image_size, patch_size, d_model, num_heads, num_layers, num_classes = 2, 32, 4, 64, 8, 3, 10

    # Create dummy image data (RGB images)
    images = np.random.rand(batch_size, image_size, image_size, 3)

    # Create ViT model
    vit = VisionTransformer(patch_size, d_model, num_heads, num_layers, num_classes, image_size)

    # Test forward pass
    print(f"Input image shape: {images.shape}")
    logits = vit.forward(images)
    print(f"Output logits shape: {logits.shape}")
    print(f"Predicted class probabilities sum: {np.sum(np.exp(logits), axis=1)}")

    # Test with different image sizes
    larger_image = np.random.rand(1, 64, 64, 3)  # Larger image
    larger_logits = vit.forward(larger_image)
    print(f"Larger image output shape: {larger_logits.shape}")

    print("\nVision Transformer test completed!")

if __name__ == "__main__":
    test_vit()
```

This practice section provides comprehensive hands-on experience with implementing attention mechanisms and transformers from scratch. The implementations are designed to be educational and demonstrate the key concepts while remaining computationally feasible to understand and modify.
