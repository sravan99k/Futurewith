# Generative AI Cheat Sheet 🚀

_Quick Reference Guide for Creative AI Models_

---

## 📋 Table of Contents

1. [Core Concepts](#core-concepts)
2. [GANs (Generative Adversarial Networks)](#gans)
3. [VAEs (Variational Autoencoders)](#vaes)
4. [Diffusion Models](#diffusion-models)
5. [Style Transfer](#style-transfer)
6. [Text Generation](#text-generation)
7. [Creative Applications](#creative-applications)
8. [Generation Techniques](#generation-techniques)
9. [Implementation Patterns](#implementation-patterns)
10. [Quick Start Code](#quick-start-code)
11. [Common Issues & Solutions](#common-issues--solutions)
12. [Resources](#resources)

---

## 🎯 Core Concepts

### What is Generative AI?

**Definition**: AI that creates new content (images, text, music, video) by learning patterns from existing data.

**Key Principle**: Learn probability distribution P(data) and sample from it to generate new data.

### Main Approaches

```python
# Core generative model comparison
models = {
    'GANs': 'Adversarial training - generator vs discriminator',
    'VAEs': 'Probabilistic encoding/decoding with latent space',
    'Diffusion': 'Forward/backward noising process',
    'Autoregressive': 'Sequential token prediction',
    'Flow-based': 'Invertible transformations'
}
```

### Quality Metrics

- **FID (Fréchet Inception Distance)**: Lower is better
- **IS (Inception Score)**: Higher is better
- **LPIPS**: Measures perceptual similarity
- **BLEU/ROUGE**: For text generation quality

---

## 🎨 GANs (Generative Adversarial Networks)

### Architecture

```python
# GAN Training Loop
for epoch in range(epochs):
    # 1. Train Discriminator
    real_data = next(real_batch)
    fake_data = generator(noise_batch)

    d_real = discriminator(real_data)
    d_fake = discriminator(fake_data.detach())

    d_loss = BCE(d_real, 1) + BCE(d_fake, 0)
    d_loss.backward()

    # 2. Train Generator
    fake_data = generator(noise_batch)
    d_fake = discriminator(fake_data)
    g_loss = BCE(d_fake, 1)
    g_loss.backward()
```

### Common GAN Variants

| Variant      | Use Case         | Key Feature            |
| ------------ | ---------------- | ---------------------- |
| **DCGAN**    | Image generation | Convolutional networks |
| **StyleGAN** | High-res images  | Style mixing control   |
| **CycleGAN** | Style transfer   | Unpaired data          |
| **cGAN**     | Conditional gen  | Label conditioning     |
| **WGAN**     | Stable training  | Wasserstein loss       |

### DCGAN Generator Example

```python
def build_generator():
    model = Sequential([
        Dense(7*7*256, input_shape=(100,)),
        BatchNormalization(),
        LeakyReLU(),
        Reshape((7, 7, 256)),

        Conv2DTranspose(128, (5,5), strides=(1,1), padding='same'),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(64, (5,5), strides=(2,2), padding='same'),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', activation='tanh')
    ])
    return model
```

### Common Issues & Fixes

| Problem                   | Solution                                  |
| ------------------------- | ----------------------------------------- |
| Mode collapse             | Use feature matching, unrolled GANs       |
| Training instability      | Use Wasserstein loss, spectral norm       |
| Discriminator overfitting | Use dropout, data augmentation            |
| Poor quality              | Increase batch size, improve architecture |

---

## 🔧 VAEs (Variational Autoencoders)

### Key Components

```python
class VAE:
    def __init__(self, latent_dim=2):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.latent_dim = latent_dim

    def sample(self, z_mean, z_log_var):
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_loss(self, x, x_recon, z_mean, z_log_var):
        recon_loss = tf.reduce_sum(BCE(x, x_recon))
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - z_mean^2 - tf.exp(z_log_var))
        return recon_loss + kl_loss
```

### VAE Loss Function

```
Total Loss = Reconstruction Loss + β × KL Divergence Loss
```

**Reconstruction Loss**: How well decoder reconstructs input
**KL Divergence**: Regularizes latent space to be normally distributed

### VAE Variants

| Type                | Purpose            | Key Change                  |
| ------------------- | ------------------ | --------------------------- |
| **β-VAE**           | Disentanglement    | β weight on KL loss         |
| **Conditional VAE** | Controllable gen   | Labels in encoder/decoder   |
| **∂-VAE**           | Dynamic generation | Learned variance parameters |
| **TF-VAE**          | Better samples     | Teacher forcing             |

### Decoder/Encoder Templates

```python
# Encoder Template
def build_encoder():
    return Sequential([
        Conv2D(32, 3, activation='relu', strides=2, padding='same'),
        Conv2D(64, 3, activation='relu', strides=2, padding='same'),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(latent_dim, name='z_mean'),
        Dense(latent_dim, name='z_log_var')
    ])

# Decoder Template
def build_decoder():
    return Sequential([
        Dense(7*7*64, activation='relu'),
        Reshape((7, 7, 64)),
        Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same'),
        Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same'),
        Conv2D(1, 3, activation='sigmoid', padding='same')
    ])
```

---

## 🌊 Diffusion Models

### Forward & Reverse Process

```python
class DiffusionModel:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.beta = self.linear_schedule(timesteps)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = np.cumprod(self.alpha)

    def add_noise(self, x, t):
        noise = tf.random.normal(tf.shape(x))
        alpha_t = self.alpha_cumprod[t]
        return tf.sqrt(alpha_t) * x + tf.sqrt(1 - alpha_t) * noise

    def denoise(self, x, t, predicted_noise):
        alpha_t = self.alpha[t]
        alpha_cumprod_t = self.alpha_cumprod[t]
        return (x - (1 - alpha_t) * predicted_noise) / tf.sqrt(alpha_t)
```

### Training Process

```python
def train_step(batch):
    # 1. Sample random timesteps
    t = tf.random.uniform([batch_size]) * timesteps
    t = tf.cast(t, tf.int32)

    # 2. Add noise according to schedule
    noisy_batch = self.add_noise(batch, t)

    # 3. Predict noise
    predicted_noise = self.unet(noisy_batch, t)

    # 4. Compute loss (MSE between actual and predicted noise)
    loss = tf.reduce_mean(tf.square(noise - predicted_noise))
    return loss

def sample(self, shape):
    # Start with pure noise
    x = tf.random.normal(shape)

    # Iterative denoising
    for t in reversed(range(self.timesteps)):
        predicted_noise = self.unet(x, t)
        x = self.denoise(x, t, predicted_noise)

    return x
```

### DDPM vs DDIM Sampling

| Aspect        | DDPM | DDIM   |
| ------------- | ---- | ------ |
| Steps         | 1000 | 50-250 |
| Speed         | Slow | Fast   |
| Quality       | High | High   |
| Deterministic | No   | Yes    |

### Conditioning Methods

```python
# Classifier-Free Guidance
def classifier_free_guidance(unet, x, t, cond, guidance_scale=7.5):
    # Get conditional prediction
    cond_pred = unet(x, t, cond)

    # Get unconditional prediction (empty cond)
    uncond_pred = unet(x, t, empty_cond)

    # Combine with guidance
    return uncond_pred + guidance_scale * (cond_pred - uncond_pred)
```

---

## 🎭 Style Transfer

### Neural Style Transfer

```python
class StyleTransfer:
    def __init__(self):
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    def compute_loss(self, combination_image):
        # Content loss
        content_loss = tf.reduce_sum(tf.square(self.content_outputs - self.content_targets))

        # Style loss (Gram matrix differences)
        style_loss = sum([self.gram_style_loss(self.style_outputs[i], self.style_targets[i])
                         for i in range(len(self.style_layers))])

        return content_loss + style_weight * style_loss + total_variation_weight * total_variation_loss
```

### Fast Style Transfer

```python
# Real-time style transfer network
class StyleTransferNet:
    def __init__(self):
        self.build_network()

    def build_network(self):
        # Encoder-Decoder architecture with residual blocks
        inputs = Input(shape=(256, 256, 3))

        # Encoder
        x = Conv2D(32, 9, padding='same')(inputs)
        x = ReLU()(x)
        x = Conv2D(64, 3, strides=2, padding='same')(x)
        x = ReLU()(x)
        x = Conv2D(128, 3, strides=2, padding='same')(x)
        x = ReLU()(x)

        # Residual blocks
        x = self.residual_block(x, 128)
        x = self.residual_block(x, 128)
        x = self.residual_block(x, 128)
        x = self.residual_block(x, 128)
        x = self.residual_block(x, 128)

        # Decoder
        x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
        x = ReLU()(x)
        x = Conv2DTranspose(32, 3, strides=2, padding='same')(x)
        x = ReLU()(x)
        outputs = Conv2D(3, 9, padding='same', activation='tanh')(x)

        return Model(inputs, outputs)
```

### CycleGAN for Style Transfer

```python
def cycle_consistency_loss(real_A, reconstructed_A, real_B, reconstructed_B, lambda_cycle=10):
    loss_A = tf.reduce_mean(tf.abs(real_A - reconstructed_A))
    loss_B = tf.reduce_mean(tf.abs(real_B - reconstructed_B))
    return lambda_cycle * (loss_A + loss_B)
```

---

## 📝 Text Generation

### Transformer-Based Generation

```python
class TextGenerator:
    def generate(self, prompt, max_length=100, temperature=1.0, top_k=50):
        tokens = self.tokenize(prompt)

        for _ in range(max_length):
            # Get predictions
            logits = self.model(tokens)
            next_token_logits = logits[0, -1, :] / temperature

            # Apply top-k filtering
            next_token_logits = self.top_k_filtering(next_token_logits, top_k)

            # Sample
            probs = tf.nn.softmax(next_token_logits)
            next_token = tf.random.categorical([probs], 1)[0, 0]

            # Add to sequence
            tokens = tf.concat([tokens, [[next_token]]], axis=1)

            if next_token == self.eos_token_id:
                break

        return self.detokenize(tokens[0])
```

### GPT Generation Strategies

| Strategy            | Description                         | Code                   |
| ------------------- | ----------------------------------- | ---------------------- |
| **Greedy**          | Always pick highest prob            | `tf.argmax(logits)`    |
| **Top-k**           | Sample from top k tokens            | Filter logits, sample  |
| **Top-p (Nucleus)** | Sample from tokens with cumprob < p | Dynamic filtering      |
| **Temperature**     | Scale probabilities                 | `logits / temperature` |
| **Beam Search**     | Keep k best sequences               | Complex implementation |

### Tokenization

```python
# Byte Pair Encoding (BPE)
class BPETokenizer:
    def __init__(self, vocab_size=30000):
        self.vocab = {}
        self.merges = []
        self.special_tokens = ['<pad>', '<unk>', '<s>', '</s>']

    def encode(self, text):
        # Convert to tokens
        tokens = self.basic_tokenizer(text)

        # Apply merges
        for pair in self.merges:
            tokens = self.merge_pair(tokens, pair)

        return tokens

    def decode(self, tokens):
        # Join tokens and clean up
        text = ''.join(self.inv_vocab[token] for token in tokens)
        return text
```

---

## 🎨 Creative Applications

### Multi-Modal Generation

```python
class CreativeAI:
    def __init__(self):
        self.text_gen = TextGenerator()
        self.image_gen = ImageGenerator()
        self.music_gen = MusicGenerator()

    def generate_story_with_media(self, concept):
        # 1. Generate story
        story = self.text_gen.generate_story(concept, length=1000)

        # 2. Extract scenes and descriptions
        scenes = self.extract_scenes(story)

        # 3. Generate images for each scene
        scene_images = [self.image_gen.generate(scene['desc']) for scene in scenes]

        # 4. Generate background music
        music = self.music_gen.generate(story['mood'], duration=300)

        return {
            'story': story,
            'scenes': scenes,
            'images': scene_images,
            'music': music
        }
```

### Style Mixing

```python
def mix_styles(style1_features, style2_features, content_features, alpha=0.5):
    """Mix styles with controllable blending"""

    # Interpolate style features
    mixed_style = alpha * style1_features + (1 - alpha) * style2_features

    # Apply to content
    result = content_features + mixed_style * 0.1

    return result
```

### Content-Aware Generation

```python
class ContentAwareGenerator:
    def generate_with_constraints(self, prompt, constraints):
        # Parse constraints
        color_constraint = constraints.get('dominant_color')
        style_constraint = constraints.get('art_style')
        mood_constraint = constraints.get('mood')

        # Adjust generation
        if color_constraint:
            prompt = f"{prompt}, {color_constraint} colors"

        if style_constraint:
            prompt = f"{style_constraint} style, {prompt}"

        # Generate
        image = self.diffusion_model.generate(prompt)

        return image
```

---

## 🔄 Generation Techniques

### Sampling Methods

#### 1. Temperature Sampling

```python
def temperature_sampling(logits, temperature=1.0):
    """Lower temp = more focused, Higher temp = more random"""
    scaled_logits = logits / temperature
    return tf.nn.softmax(scaled_logits)
```

#### 2. Top-k Sampling

```python
def top_k_sampling(logits, k=50):
    """Sample from top k most likely tokens"""
    # Keep only top k
    top_k_logits, indices = tf.nn.top_k(logits, k=k)

    # Create mask
    mask = tf.scatter_nd(
        indices[:, None],
        tf.ones_like(top_k_logits, dtype=tf.bool),
        tf.shape(logits)
    )

    # Apply mask
    filtered_logits = tf.where(mask, logits, tf.float32.min)
    return tf.nn.softmax(filtered_logits)
```

#### 3. Nucleus (Top-p) Sampling

```python
def nucleus_sampling(logits, p=0.9):
    """Sample from tokens with cumulative probability > p"""
    sorted_logits = tf.sort(logits, direction='DESCENDING')
    sorted_indices = tf.argsort(logits, direction='DESCENDING')

    # Find cutoff
    cumsum = tf.cumsum(tf.nn.softmax(sorted_logits))
    cutoff = tf.reduce_min(tf.where(cumsum >= p, sorted_indices, tf.shape(logits)[0]-1))

    # Create mask
    mask = tf.scatter_nd(
        sorted_indices[:cutoff+1][:, None],
        tf.ones(cutoff+1, dtype=tf.bool),
        tf.shape(logits)
    )

    # Apply mask
    filtered_logits = tf.where(mask, logits, tf.float32.min)
    return tf.nn.softmax(filtered_logits)
```

### Advanced Techniques

#### 1. Classifier-Free Guidance

```python
def classifier_free_guidance_step(model, x, t, cond, guidance_scale=7.5):
    # Conditional prediction
    cond_pred = model(x, t, cond)

    # Unconditional prediction (empty condition)
    uncond_pred = model(x, t, empty_condition)

    # Guided prediction
    return uncond_pred + guidance_scale * (cond_pred - uncond_pred)
```

#### 2. Prompt Engineering

```python
class PromptEngineer:
    def create_advanced_prompt(self, base_prompt, modifiers):
        # Structured modifiers
        style = modifiers.get('style', '')
        quality = modifiers.get('quality', '')
        lighting = modifiers.get('lighting', '')
        composition = modifiers.get('composition', '')

        # Compose enhanced prompt
        enhanced_prompt = f"{base_prompt}, {style}, {quality}, {lighting}, {composition}"

        # Negative prompts
        negative_prompt = modifiers.get('negative', 'blurry, low quality, distorted')

        return {
            'positive': enhanced_prompt,
            'negative': negative_prompt
        }
```

#### 3. Iterative Refinement

```python
def iterative_refinement(model, initial_output, feedback, iterations=3):
    """Refine generation based on feedback"""
    current = initial_output

    for i in range(iterations):
        # Analyze feedback
        corrections = self.analyze_feedback(feedback)

        # Adjust generation
        current = model.refine(current, corrections)

        # Update feedback
        feedback = self.update_feedback(feedback, corrections)

    return current
```

---

## 💻 Implementation Patterns

### Model Architecture Patterns

#### 1. Encoder-Decoder Pattern (VAEs, Autoencoders)

```python
class EncoderDecoder:
    def __init__(self, latent_dim):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.latent_dim = latent_dim

    def build_encoder(self):
        return Sequential([...])

    def build_decoder(self):
        return Sequential([...])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reconstruct(self, x):
        z = self.encode(x)
        return self.decode(z)
```

#### 2. Adversarial Pattern (GANs)

```python
class GAN:
    def __init__(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.adversarial_model = self.build_adversarial_model()

    def train_step(self, real_data, batch_size):
        # Train discriminator
        noise = tf.random.normal([batch_size, 100])
        fake_data = self.generator(noise)

        d_real = self.discriminator(real_data)
        d_fake = self.discriminator(fake_data)

        d_loss = self.discriminator_loss(d_real, d_fake)

        # Train generator
        noise = tf.random.normal([batch_size, 100])
        g_loss = self.adversarial_model([noise, real_data])

        return d_loss, g_loss
```

#### 3. U-Net Pattern (Diffusion Models)

```python
class UNet:
    def __init__(self):
        self.down_blocks = [...]
        self.mid_block = [...]
        self.up_blocks = [...]

    def call(self, x, time_embed):
        skip_connections = []

        # Encoder
        x = self.down_blocks[0](x)
        skip_connections.append(x)
        # ... more downsampling

        # Middle
        x = self.mid_block(x, time_embed)

        # Decoder
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, skip_connections[-i-1])

        return x
```

### Training Patterns

#### 1. Progressive Training

```python
class ProgressiveGAN:
    def __init__(self):
        self.generators = []
        self.discriminators = []
        self.current_resolution = 4

    def grow_model(self):
        # Add new layers for higher resolution
        self.add_generator_layer(self.current_resolution * 2)
        self.add_discriminator_layer(self.current_resolution * 2)
        self.current_resolution *= 2

    def train_at_resolution(self, resolution, epochs):
        # Train at current resolution
        for epoch in range(epochs):
            # Mix old and new resolutions during fade-in
            alpha = epoch / epochs if epoch < self.fade_in else 1.0

            loss = self.train_step(resolution, alpha)
            self.log_training(loss)
```

#### 2. Curriculum Learning

```python
class CurriculumTrainer:
    def __init__(self):
        self.easy_samples = []
        self.hard_samples = []
        self.current_difficulty = 0

    def get_training_batch(self, batch_size):
        # Start with easier samples
        if self.current_difficulty < self.max_difficulty:
            easy_ratio = 1.0
        else:
            easy_ratio = 0.7  # 70% easy, 30% hard

        easy_count = int(batch_size * easy_ratio)
        hard_count = batch_size - easy_count

        easy_batch = random.sample(self.easy_samples, easy_count)
        hard_batch = random.sample(self.hard_samples, hard_count)

        return easy_batch + hard_batch

    def increase_difficulty(self):
        # Gradually increase difficulty
        self.current_difficulty += 1
```

#### 3. Multi-Scale Training

```python
class MultiScaleTrainer:
    def train_step(self, batch):
        losses = {}

        for scale in self.scales:
            # Resize to scale
            scaled_batch = resize_image(batch, scale)

            # Train at scale
            loss = self.train_at_scale(scaled_batch, scale)
            losses[f'scale_{scale}'] = loss

        return losses
```

### Memory Optimization Patterns

#### 1. Gradient Checkpointing

```python
class MemoryEfficientGAN:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()

    @tf.function
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Forward pass (memory intensive)
            fake_data = self.generator(noise, training=True)

            # Discriminator forward pass
            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(fake_data, training=True)

            # Compute losses
            g_loss = self.generator_loss(fake_output)
            d_loss = self.discriminator_loss(real_output, fake_output)

        # Backward pass
        g_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        return g_loss, d_loss
```

#### 2. Mixed Precision Training

```python
class MixedPrecisionTrainer:
    def __init__(self):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        self.generator = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    @tf.function
    def train_step(self, real_data):
        noise = tf.random.normal([32, 100])

        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            loss = self.compute_loss(fake_data)

        # Scale gradients to prevent underflow
        scaled_loss = self.optimizer.get_scaled_loss(loss)
        grads = tape.gradient(scaled_loss, self.generator.trainable_variables)
        grads = self.optimizer.get_unscaled_gradients(grads)

        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return loss
```

---

## ⚡ Quick Start Code

### 1. Basic GAN Training

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Configuration
BATCH_SIZE = 128
NOISE_DIM = 100
EPOCHS = 50

# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_DIM,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Reshape((7, 7, 256)),
    # Add decoder layers here...
])

# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 5, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),
    # Add more layers...
])

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training loop
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Main training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)

        # Generate and save sample images
        if epoch % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed)

        print(f'Epoch {epoch}, Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}')
```

### 2. VAE Implementation

```python
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_inputs = tf.keras.layers.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name="z_log_var")(x)
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var])
        return encoder

    def build_decoder(self):
        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = tf.keras.layers.Reshape((7, 7, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = tf.keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs)
        return decoder

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        decoded = self.decode(z)
        return decoded

# Loss function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    # Reconstruction loss
    recon_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_logit), axis=(1,2))

    # KL divergence loss
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)

    return tf.reduce_mean(recon_loss + kl_loss)

# Training
@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### 3. Diffusion Model (Simplified)

```python
class SimpleDiffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.beta = np.linspace(0.0001, 0.02, timesteps)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = np.cumprod(self.alpha)

        self.unet = self.build_unet()

    def build_unet(self):
        # Simplified UNet architecture
        inputs = tf.keras.layers.Input(shape=(28, 28, 1))
        time_inputs = tf.keras.layers.Input(shape=())

        # Time embedding
        time_embed = tf.keras.layers.Dense(128)(time_inputs)
        time_embed = tf.keras.layers.Dense(128)(time_embed)

        # Downsample
        x = tf.keras.layers.Conv2D(64, 3, padding='same')(inputs)
        skip1 = self.residual_block(x, 64, time_embed)
        x = tf.keras.layers.AveragePooling2D(2)(skip1)

        x = self.residual_block(x, 128, time_embed)
        skip2 = x
        x = tf.keras.layers.AveragePooling2D(2)(x)

        x = self.residual_block(x, 256, time_embed)

        # Upsample
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Concatenate()([x, skip2])
        x = self.residual_block(x, 128, time_embed)

        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Concatenate()([x, skip1])
        x = self.residual_block(x, 64, time_embed)

        outputs = tf.keras.layers.Conv2D(1, 3, padding='same')(x)
        return tf.keras.Model([inputs, time_inputs], outputs)

    def residual_block(self, x, channels, time_embed):
        residual = x
        x = tf.keras.layers.Conv2D(channels, 3, padding='same')(x)
        x = tf.keras.layers.Dense(channels)(time_embed)[:, None, None, :] + x
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Conv2D(channels, 3, padding='same')(x)

        if residual.shape != x.shape:
            residual = tf.keras.layers.Conv2D(channels, 1)(residual)

        return tf.keras.layers.Add()([residual, x])

    def add_noise(self, x, t):
        noise = tf.random.normal(tf.shape(x))
        alpha_t = self.alpha_cumprod[t]
        return tf.sqrt(alpha_t) * x + tf.sqrt(1 - alpha_t) * noise, noise

    def train_step(self, x):
        t = tf.random.uniform([tf.shape(x)[0]], 0, self.timesteps, dtype=tf.int32)
        noisy_x, noise = self.add_noise(x, t)

        with tf.GradientTape() as tape:
            predicted_noise = self.unet([noisy_x, t])
            loss = tf.reduce_mean(tf.square(noise - predicted_noise))

        gradients = tape.gradient(loss, self.unet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))
        return loss

    def sample(self, shape):
        x = tf.random.normal(shape)

        for i in reversed(range(self.timesteps)):
            t = tf.fill([shape[0]], i)

            predicted_noise = self.unet([x, t])

            alpha_t = self.alpha[i]
            alpha_cumprod_t = self.alpha_cumprod[i]
            beta_t = self.beta[i]

            if i > 0:
                noise = tf.random.normal(shape)
                variance = beta_t * (1 - alpha_cumprod_t) / (1 - alpha_cumprod_t)
                x = (x - (1 - alpha_t) / tf.sqrt(1 - alpha_cumprod_t) * predicted_noise) / tf.sqrt(alpha_t)
                x = x + tf.sqrt(variance) * noise
            else:
                x = (x - (1 - alpha_t) / tf.sqrt(1 - alpha_cumprod_t) * predicted_noise) / tf.sqrt(alpha_t)

        return x
```

### 4. Style Transfer

```python
class StyleTransfer:
    def __init__(self):
        # Use VGG19 for feature extraction
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

        # Get style and content layers
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.content_layers = ['block5_conv2']

        # Build feature extraction model
        outputs = [vgg.get_layer(name).output for name in self.style_layers + self.content_layers]
        self.feature_extractor = tf.keras.Model([vgg.input], outputs)

    def gram_matrix(self, features):
        """Compute Gram matrix for style loss"""
        result = tf.linalg.einsum('bijc,bijd->bcd', features, features)
        input_shape = tf.shape(features)
        height = input_shape[1]
        width = input_shape[2]
        return result / (height * width)

    def compute_loss(self, combination_image, style_targets, content_targets):
        # Get features
        combination_features = self.feature_extractor(combination_image)
        style_features = combination_features[:len(self.style_layers)]
        content_features = combination_features[len(self.style_layers):]

        # Style loss
        style_loss = tf.add_n([tf.reduce_mean(tf.square(self.gram_matrix(style_features[i]) - style_targets[i]))
                              for i in range(len(style_features))])

        # Content loss
        content_loss = tf.add_n([tf.reduce_mean(tf.square(content_features[i] - content_targets[i]))
                                for i in range(len(content_features))])

        return style_loss * 1e-4 + content_loss * 1e4

    def transfer_style(self, content_image, style_image, epochs=1000):
        # Extract style features
        style_features = self.feature_extractor(style_image)[:len(self.style_layers)]
        style_targets = [self.gram_matrix(feature) for feature in style_features]

        # Extract content features
        content_features = self.feature_extractor(content_image)[len(self.style_layers):]
        content_targets = [feature for feature in content_features]

        # Initialize combination image
        combination_image = tf.Variable(content_image)

        optimizer = tf.keras.optimizers.Adam(learning_rate=2.0, beta_1=0.99, epsilon=1e-1)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss = self.compute_loss(combination_image, style_targets, content_targets)

            grad = tape.gradient(loss, combination_image)
            optimizer.apply_gradients([(grad, combination_image)])

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        return combination_image
```

---

## 🚨 Common Issues & Solutions

### GAN Training Issues

#### 1. Mode Collapse

**Symptoms**: Generator produces limited variety of samples
**Solutions**:

```python
# Use feature matching
def feature_matching_loss(real_features, fake_features):
    return tf.reduce_mean(tf.abs(tf.reduce_mean(real_features, axis=0) -
                                tf.reduce_mean(fake_features, axis=0)))

# Use unrolled GANs
for _ in range(unroll_steps):
    fake_data = generator(noise)
    d_loss = discriminator_loss(discriminator(real_data), discriminator(fake_data))
    discriminator_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
```

#### 2. Training Instability

**Solutions**:

```python
# Use Wasserstein loss
def wasserstein_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

# Use spectral normalization
def spectral_norm(layer):
    return tf.keras.utils.get_registered_object(f"spectral_norm_{layer.name}")

# Use gradient penalty
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = tf.random.uniform([BATCH_SIZE, 1, 1, 1], 0., 1.)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
    return gradient_penalty
```

### VAE Issues

#### 1. Poor Reconstruction

**Solutions**:

- Increase model capacity
- Adjust KL weight (β-VAE)
- Use more sophisticated architectures

```python
# Beta-VAE
class BetaVAE(VAE):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def compute_loss(self, x, x_recon, z_mean, z_log_var):
        recon_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_recon), axis=(1, 2))
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - z_mean**2 - tf.exp(z_log_var), axis=1)
        return tf.reduce_mean(recon_loss + self.beta * kl_loss)
```

#### 2. Posterior Collapse

**Solutions**:

- Decrease KL weight
- Use free bits trick
- Use β-VAE with gradual KL annealing

```python
# Free bits VAE
def free_bits_kl(kl_loss, free_bits=1.0):
    kl_loss = tf.maximum(kl_loss, free_bits)
    return kl_loss
```

### Diffusion Model Issues

#### 1. Slow Sampling

**Solutions**:

```python
# DDIM Sampling (faster)
def ddim_sample(self, shape, eta=0):
    x = tf.random.normal(shape)

    for i in reversed(range(self.timesteps)):
        t = tf.fill([shape[0]], i)
        predicted_noise = self.unet([x, t])

        alpha_cumprod_t = self.alpha_cumprod[i]
        alpha_cumprod_prev = self.alpha_cumprod[max(0, i - self.schedule_sampler.step_size)]

        x = predicted_noise * tf.sqrt(1 - alpha_cumprod_prev / alpha_cumprod_t)
        x = x + tf.sqrt(alpha_cumprod_prev * (1 - alpha_cumprod_t) / alpha_cumprod_t) * predicted_noise

        if i > 0 and eta > 0:
            noise = tf.random.normal(shape)
            variance = eta * tf.sqrt((1 - alpha_cumprod_prev) * (1 - alpha_cumprod_t) / alpha_cumprod_t)
            x = x + variance * noise

    return x

# Distillation for faster sampling
class DistilledDiffusion:
    def __init__(self, original_model, steps):
        self.student_model = self.build_student_model()
        self.original_model = original_model
        self.target_steps = steps
```

#### 2. Quality vs Speed Trade-off

**Solutions**:

- Use classifier-free guidance
- Adjust guidance scale
- Use progressive distillation

```python
# Optimal guidance scale
def optimal_guidance(model_output, guidance_scale=7.5):
    if guidance_scale == 0:
        return model_output

    # Classifier-free guidance
    uncond_output, cond_output = model_output
    return uncond_output + guidance_scale * (cond_output - uncond_output)
```

---

## 📚 Resources

### Pre-trained Models

#### Text Models

```python
# Hugging Face Transformers
from transformers import (
    # GPT family
    GPT2LMHeadModel, GPT2Tokenizer,
    GPTNeoForCausalLM,

    # T5 family
    T5ForConditionalGeneration, T5Tokenizer,

    # LLaMA family
    LlamaForCausalLM, LlamaTokenizer,

    # Chat models
    AutoModelForCausalLM, AutoTokenizer,
)

# Load and generate
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

inputs = tokenizer.encode("Once upon a time", return_tensors="pt")
outputs = model.generate(inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### Image Models

```python
# Stable Diffusion
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# Text-to-image
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipe("A fantasy landscape").images[0]

# Image-to-image
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
input_image = load_image("input.jpg")
result = img2img_pipe(prompt="in the style of Van Gogh", image=input_image).images[0]
```

#### Multi-Modal Models

```python
# CLIP for image-text understanding
from transformers import CLIPModel, CLIPProcessor

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# BLIP for image captioning
from transformers import BlipForConditionalGeneration, BlipProcessor

blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Generate caption
inputs = blip_processor(image, return_tensors="pt")
out = blip_model.generate(**inputs, max_length=100)
caption = blip_processor.decode(out[0], skip_special_tokens=True)
```

### Datasets

#### Text Datasets

```python
# Common datasets
datasets = {
    'books': 'Project Gutenberg',
    'web_text': 'Common Crawl',
    'conversational': 'Dialogue datasets',
    'code': 'GitHub, Stack Overflow',
    'scientific': 'ArXiv papers'
}

# Load with Hugging Face
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
```

#### Image Datasets

```python
# Popular datasets
image_datasets = {
    'faces': 'FFHQ, CelebA-HQ',
    'art': 'WikiArt, ArtBench',
    'general': 'ImageNet, LAION',
    'artistic': 'Artist styles collection'
}

# Custom dataset
class CustomImageDataset(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, target_size=(256, 256)):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        for path in batch_paths:
            image = tf.keras.preprocessing.image.load_img(path, target_size=self.target_size)
            image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
            images.append(image)

        return np.array(images), np.array(batch_labels)
```

### Development Tools

#### Experiment Tracking

```python
# Weights & Biases
import wandb

wandb.init(project="generative-ai")

# TensorBoard
from tensorflow.keras.callbacks import TensorBoard
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# MLflow
import mlflow
mlflow.start_run()
```

#### Model Analysis

```python
# FID calculation
def calculate_fid(real_features, generated_features):
    # Fréchet distance between real and generated
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features.T)
    mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features.T)

    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# LPIPS for perceptual similarity
import lpips
loss_fn = lpips.LPIPS(net='alex')
distance = loss_fn(real_image, generated_image)
```

#### Performance Monitoring

```python
class ModelMonitor:
    def __init__(self):
        self.metrics = {}
        self.performance_history = []

    def log_training(self, epoch, metrics):
        self.performance_history.append({
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': time.time()
        })

    def analyze_training(self):
        if len(self.performance_history) < 10:
            return "Need more data"

        recent_metrics = self.performance_history[-10:]

        # Detect overfitting
        train_loss = [m['metrics']['train_loss'] for m in recent_metrics]
        val_loss = [m['metrics']['val_loss'] for m in recent_metrics]

        if val_loss[-1] > val_loss[0] and train_loss[-1] < train_loss[0]:
            return "Overfitting detected"

        return "Training stable"
```

### Online Resources

#### Learning Platforms

- **Papers With Code**: Latest research with implementations
- **Distill.pub**: Interactive machine learning explanations
- **ML Explained**: Visual explanations of ML concepts
- **Towards Data Science**: Community articles

#### Model Hubs

- **Hugging Face Hub**: Pre-trained models
- **TensorFlow Hub**: TensorFlow models
- **PyTorch Hub**: PyTorch models
- **Paperspace**: Model deployment

#### Communities

- **Reddit**: r/MachineLearning, r/artificial
- **Discord**: AI/ML communities
- **Twitter**: #MachineLearning #GenAI
- **Kaggle**: Competitions and datasets

---

## ⚡ Quick Commands

### Environment Setup

```bash
# Create environment
python -m venv genai_env
source genai_env/bin/activate  # Linux/Mac
# genai_env\Scripts\activate   # Windows

# Install core packages
pip install tensorflow torch transformers diffusers
pip install opencv-python pillow matplotlib seaborn
pip install jupyter wandb datasets

# GPU support
pip install tensorflow-gpu  # NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Data Preparation

```python
# Quick data loading
def load_data(data_path, target_size=(256, 256)):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=20
    )

    train_gen = datagen.flow_from_directory(
        data_path,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_path,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen

# Text preprocessing
def prepare_text_data(texts, tokenizer, max_length=512):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )
    return encoded
```

### Model Saving/Loading

```python
# Save models
generator.save('models/generator.h5')
discriminator.save('models/discriminator.h5')

# Load models
generator = tf.keras.models.load_model('models/generator.h5')
discriminator = tf.keras.models.load_model('models/discriminator.h5')

# Save with custom objects
model.save('model.h5', save_format='tf')
```

### Inference Examples

```python
# GAN generation
def generate_samples(generator, num_samples=16):
    noise = tf.random.normal([num_samples, 100])
    samples = generator(noise, training=False)
    return samples

# VAE sampling
def sample_vae(vae, num_samples=16):
    z = tf.random.normal([num_samples, vae.latent_dim])
    return vae.decoder(z)

# Diffusion sampling
def generate_diffusion(diffusion_model, shape=(4, 28, 28, 1)):
    return diffusion_model.sample(shape)
```

---

## 🎯 Pro Tips

### 1. Start Simple

- Begin with basic architectures
- Use small datasets for experimentation
- Implement training loops from scratch first

### 2. Monitor Training

- Track multiple metrics simultaneously
- Visualize generated samples regularly
- Use experiment tracking tools

### 3. Hyperparameter Tuning

- Use learning rate schedulers
- Experiment with batch sizes
- Adjust loss function weights

### 4. Data Quality

- Preprocess data consistently
- Handle outliers appropriately
- Use data augmentation

### 5. Performance Optimization

- Use mixed precision training
- Implement gradient accumulation
- Use efficient data loading

### 6. Deployment Considerations

- Optimize models for inference
- Consider model compression
- Plan for scalability

---

## 🔥 Advanced Techniques

### 1. Attention Mechanisms

```python
# Self-attention for images
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.query_conv = tf.keras.layers.Conv2D(channels // 8, 1)
        self.key_conv = tf.keras.layers.Conv2D(channels // 8, 1)
        self.value_conv = tf.keras.layers.Conv2D(channels, 1)
        self.output_conv = tf.keras.layers.Conv2D(channels, 1)

    def call(self, x):
        batch_size, height, width, channels = x.shape

        # Get query, key, value
        query = tf.reshape(self.query_conv(x), [batch_size, height * width, channels // 8])
        key = tf.reshape(self.key_conv(x), [batch_size, height * width, channels // 8])
        value = tf.reshape(self.value_conv(x), [batch_size, height * width, channels])

        # Compute attention
        attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.sqrt(channels // 8))
        context = tf.matmul(attention, value)
        context = tf.reshape(context, [batch_size, height, width, channels])

        output = self.output_conv(context)
        return x + output
```

### 2. Advanced Loss Functions

```python
# Perceptual loss
class PerceptualLoss(tf.keras.layers.Layer):
    def __init__(self, layers=['block1_conv1', 'block2_conv1']):
        super(PerceptualLoss, self).__init__()
        self.vgg = self.build_vgg(layers)
        self.vgg.trainable = False

    def build_vgg(self, layers):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        outputs = [vgg.get_layer(name).output for name in layers]
        return tf.keras.Model([vgg.input], outputs)

    def call(self, img1, img2):
        feat1 = self.vgg(img1)
        feat2 = self.vgg(img2)
        return tf.reduce_mean(tf.abs(feat1 - feat2))

# Adversarial loss variations
def relativistic_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(tf.nn.relu(1 - real_logits))
    fake_loss = tf.reduce_mean(tf.nn.relu(1 + fake_logits))
    return (real_loss + fake_loss) / 2
```

### 3. Custom Layers

```python
# Spectral normalization
class SpectralNorm(tf.keras.constraints.Constraint):
    def __init__(self, n_iter=1):
        self.n_iter = n_iter

    def __call__(self, w):
        w_shape = w.shape
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.random.normal([w_shape[0]])

        for _ in range(self.n_iter):
            v = tf.nn.l2_normalize(tf.matmul(w, u, transpose_a=True))
            u = tf.nn.l2_normalize(tf.matmul(w, v))

        sigma = tf.matmul(u, tf.matmul(w, v))
        w = w / sigma

        return tf.reshape(w, w_shape)

# Apply spectral norm
class SpectralConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(filters, kernel_size, **kwargs)
        self.spectral_norm = SpectralNorm()

    def build(self, input_shape):
        super().build(input_shape)
        if hasattr(self, 'kernel'):
            self.kernel = self.spectral_norm(self.kernel)
```

---

## 📊 Performance Benchmarks

### Generation Quality Comparison

| Model Type     | FID Score | Sample Speed | Control Level |
| -------------- | --------- | ------------ | ------------- |
| GAN            | 5-15      | Very Fast    | Medium        |
| VAE            | 15-30     | Fast         | High          |
| Diffusion      | 2-8       | Slow         | High          |
| Autoregressive | 10-25     | Very Slow    | Very High     |

### Hardware Requirements

| Model Type        | GPU Memory | Training Time | Inference Time |
| ----------------- | ---------- | ------------- | -------------- |
| Small GAN         | 4GB        | 1-10 hours    | <1ms           |
| VAE               | 2GB        | 30min-5h      | <1ms           |
| Diffusion         | 8GB+       | 10-100h       | 1-30s          |
| Large Transformer | 16GB+      | Days          | 100ms-1s       |

### Dataset Size Recommendations

| Model Type  | Minimum    | Recommended  | Large Scale  |
| ----------- | ---------- | ------------ | ------------ |
| GAN         | 1K images  | 10K+ images  | 1M+ images   |
| VAE         | 500 images | 5K+ images   | 500K+ images |
| Diffusion   | 10K images | 100K+ images | 10M+ images  |
| Text Models | 1M tokens  | 100M+ tokens | 100B+ tokens |

---

## 🎓 Next Steps

### Continue Learning

1. **Advanced Architectures**: Try Vision Transformers, Swin Transformers
2. **Emerging Techniques**: Explore neural radiance fields (NeRF)
3. **Optimization**: Learn about neural architecture search (NAS)
4. **Deployment**: Study model quantization and pruning

### Build Projects

1. **Art Generator**: Combine multiple models for artistic creation
2. **Style Transfer App**: Real-time video style transfer
3. **Text-to-Image**: Build your own Stable Diffusion variant
4. **Music Generation**: Create AI composers with VAEs and Transformers

### Contribute to Open Source

1. **Framework Development**: Contribute to TensorFlow, PyTorch
2. **Model Implementation**: Reproduce papers with code
3. **Dataset Creation**: Build and share high-quality datasets
4. **Documentation**: Help improve docs and tutorials

---

**Remember**: The best way to learn generative AI is by experimenting, creating, and iterating. Start with simple projects and gradually build complexity. The field is evolving rapidly, so stay curious and keep learning!

---

_Last Updated: November 2025_
_This cheat sheet covers the essential techniques and patterns for working with generative AI models. Keep it as a reference while building your own generative AI projects!_
