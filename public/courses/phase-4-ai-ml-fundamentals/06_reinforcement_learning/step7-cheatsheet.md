# Reinforcement Learning Cheat Sheet

## 📚 Table of Contents

1. [RL Fundamentals](#rl-fundamentals)
2. [Key Algorithms](#key-algorithms)
3. [Environment Setup](#environment-setup)
4. [Reward Systems](#reward-systems)
5. [Exploration Strategies](#exploration-strategies)
6. [Implementation Patterns](#implementation-patterns)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Common Issues & Solutions](#common-issues--solutions)
9. [Code Templates](#code-templates)
10. [Resources & References](#resources--references)

---

## RL Fundamentals

### Core Concepts

```python
# Basic RL Framework
State (S) → Agent → Action (A) → Environment → Reward (R) → Next State (S')
```

### Mathematical Foundation

- **Markov Decision Process (MDP)**: (S, A, P, R, γ)
  - S: State space
  - A: Action space
  - P: Transition probability P(s'|s,a)
  - R: Reward function R(s,a)
  - γ: Discount factor (0 ≤ γ ≤ 1)

### Value Functions

- **State Value**: V^π(s) = E[Σ γ^t r_t | s_0 = s, π]
- **Action Value**: Q^π(s,a) = E[Σ γ^t r_t | s_0 = s, a_0 = a, π]
- **Optimal Value**: V\*(s) = max_π V^π(s)

---

## Key Algorithms

### 1. Q-Learning (Value-Based)

#### Basic Q-Learning

```python
# Q-Learning Update Rule
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

# Key Parameters
alpha = 0.1    # Learning rate
gamma = 0.95   # Discount factor
epsilon = 0.1  # Exploration rate
```

#### Implementation Pattern

```python
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount=0.95):
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = 0.1

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(self.q_table[state]))
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount * max_next_q - current_q)
        self.q_table[state, action] = new_q
```

### 2. Deep Q-Network (DQN)

#### Architecture Components

- **Experience Replay**: Store transitions (s,a,r,s') in replay buffer
- **Target Network**: Separate network for stable Q-value targets
- **ε-Greedy Exploration**: Balance exploration vs exploitation

#### DQN Implementation

```python
class DQN:
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.q_network = self._build_network(state_dim, action_dim, hidden_dim)
        self.target_network = self._build_network(state_dim, action_dim, hidden_dim)
        self.memory = deque(maxlen=10000)

    def _build_network(self, state_dim, action_dim, hidden_dim):
        model = Sequential([
            Dense(hidden_dim, activation='relu', input_shape=(state_dim,)),
            Dense(hidden_dim, activation='relu'),
            Dense(action_dim, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.discount * np.max(next_q_values[i])

        self.q_network.fit(states, current_q_values, verbose=0)
```

### 3. Policy Gradient Methods

#### REINFORCE Algorithm

```python
class REINFORCE:
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.policy_network = self._build_policy_network(state_dim, action_dim, hidden_dim)
        self.discount = 0.99

    def _build_policy_network(self, state_dim, action_dim, hidden_dim):
        model = Sequential([
            Dense(hidden_dim, activation='relu', input_shape=(state_dim,)),
            Dense(hidden_dim, activation='relu'),
            Dense(action_dim, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001))
        return model

    def train(self, states, actions, rewards):
        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.discount * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        # Normalize rewards
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)

        # Update policy
        actions_one_hot = tf.keras.utils.to_categorical(actions, self.action_dim)
        with tf.GradientTape() as tape:
            probabilities = self.policy_network(states)
            log_probs = tf.math.log(probabilities + 1e-8)
            selected_log_probs = tf.reduce_sum(actions_one_hot * log_probs, axis=1)
            policy_loss = -tf.reduce_mean(selected_log_probs * discounted_rewards)

        gradients = tape.gradient(policy_loss, self.policy_network.trainable_variables)
        self.policy_network.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
```

### 4. Actor-Critic Methods

#### A2C (Advantage Actor-Critic)

```python
class A2C:
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.actor = self._build_actor(state_dim, action_dim, hidden_dim)
        self.critic = self._build_critic(state_dim, hidden_dim)
        self.discount = 0.99

    def _build_actor(self, state_dim, action_dim, hidden_dim):
        return Sequential([
            Dense(hidden_dim, activation='relu', input_shape=(state_dim,)),
            Dense(hidden_dim, activation='relu'),
            Dense(action_dim, activation='softmax')
        ])

    def _build_critic(self, state_dim, hidden_dim):
        return Sequential([
            Dense(hidden_dim, activation='relu', input_shape=(state_dim,)),
            Dense(hidden_dim, activation='relu'),
            Dense(1, activation='linear')
        ])

    def train(self, states, actions, rewards, next_states, dones):
        # Calculate advantages
        values = self.critic(states).flatten()
        next_values = self.critic(next_states).flatten()

        targets = rewards + self.discount * next_values * (1 - dones)
        advantages = targets - values

        # Update critic
        self.critic.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.critic.fit(states, targets, verbose=0)

        # Update actor
        actions_one_hot = tf.keras.utils.to_categorical(actions, self.action_dim)
        with tf.GradientTape() as tape:
            probabilities = self.actor(states)
            log_probs = tf.math.log(probabilities + 1e-8)
            selected_log_probs = tf.reduce_sum(actions_one_hot * log_probs, axis=1)
            actor_loss = -tf.reduce_mean(selected_log_probs * advantages)

        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
```

---

## Environment Setup

### OpenAI Gym/Gymnasium

#### Basic Environment Usage

```python
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1')

# Reset environment
state, info = env.reset()

# Environment interaction
for step in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        state, info = env.reset()

env.close()
```

#### Custom Environment Template

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnvironment(gym.Env):
    def __init__(self, state_size=4, action_size=2):
        super(CustomEnvironment, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Define action and observation spaces
        self.action_space = spaces.Discrete(action_size)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)

        # Environment parameters
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self._generate_state()
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        # Apply action and calculate reward
        reward = self._calculate_reward(action)
        done = self._is_done()

        # Update state
        self.state = self._update_state(action)

        return self.state, reward, done, False, {}

    def render(self):
        # Visualization code
        pass

    def _generate_state(self):
        return np.random.normal(0, 1, self.state_size)

    def _calculate_reward(self, action):
        # Implement your reward logic
        return 1.0 if action == 0 else 0.0

    def _is_done(self):
        return self.current_step >= self.max_steps

    def _update_state(self, action):
        # Update state based on action
        return self.state + np.random.normal(0, 0.1, self.state_size)
```

### Popular RL Environments

#### Classic Control

```python
# CartPole
env = gym.make('CartPole-v1')  # Balance a pole on a cart
# MountainCar
env = gym.make('MountainCar-v0')  # Reach the top of a hill
# Pendulum
env = gym.make('Pendulum-v1')  # Swing up a pendulum
```

#### Atari Games

```python
# Pong
env = gym.make('Pong-v4')
# Breakout
env = gym.make('Breakout-v4')
# Space Invaders
env = gym.make('SpaceInvaders-v4')
```

#### MuJoCo (Continuous Control)

```python
# HalfCheetah
env = gym.make('HalfCheetah-v4')
# Walker2d
env = gym.make('Walker2d-v4')
# Humanoid
env = gym.make('Humanoid-v4')
```

---

## Reward Systems

### Reward Types

#### 1. Sparse Rewards

```python
# Binary reward: 1 for success, 0 for failure
def sparse_reward(terminated, success):
    return 1.0 if (terminated and success) else 0.0
```

#### 2. Dense Rewards

```python
# Continuous reward based on distance to goal
def distance_based_reward(current_pos, goal_pos):
    distance = np.linalg.norm(current_pos - goal_pos)
    return -distance  # Negative distance as reward
```

#### 3. Shaped Rewards

```python
def shaped_reward(velocity, position, goal_position):
    # Combine multiple reward components
    progress_reward = np.dot(velocity, goal_position - position)
    velocity_reward = -np.sum(velocity**2)
    position_reward = -np.sum((position - goal_position)**2)

    return 0.4 * progress_reward + 0.3 * velocity_reward + 0.3 * position_reward
```

### Reward Engineering Tips

#### 1. Reward Scaling

```python
def scale_reward(reward, min_reward=-10, max_reward=10):
    """Scale reward to reasonable range"""
    return np.clip(reward, min_reward, max_reward)

def normalize_reward(reward, mean=0, std=1):
    """Normalize reward using running statistics"""
    return (reward - mean) / (std + 1e-8)
```

#### 2. Reward Function Design

```python
class RewardFunction:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, state, action, next_state, done):
        # Primary objective
        primary = self.primary_objective(next_state, done)

        # Secondary objectives
        efficiency = self.efficiency_reward(state, action)
        safety = self.safety_reward(state, action)

        # Combine rewards
        return (self.weights['primary'] * primary +
                self.weights['efficiency'] * efficiency +
                self.weights['safety'] * safety)

    def primary_objective(self, state, done):
        # Implement main goal reward
        return 1.0 if done and self.is_successful(state) else 0.0

    def efficiency_reward(self, state, action):
        # Reward efficient actions
        return -np.sum(action**2)

    def safety_reward(self, state, action):
        # Penalize unsafe actions
        return -10.0 if self.is_dangerous(state, action) else 0.0
```

---

## Exploration Strategies

### 1. ε-Greedy Exploration

```python
class EpsilonGreedy:
    def __init__(self, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, q_values):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(q_values))
        return np.argmax(q_values)

    def decay(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
```

### 2. Boltzmann Exploration

```python
class Boltzmann:
    def __init__(self, temperature=1.0, temperature_decay=0.99):
        self.temperature = temperature
        self.temperature_decay = temperature_decay

    def select_action(self, q_values):
        # Convert Q-values to probabilities using softmax
        exp_values = np.exp(q_values / (self.temperature + 1e-8))
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(len(q_values), p=probabilities)

    def decay(self):
        self.temperature *= self.temperature_decay
```

### 3. Upper Confidence Bound (UCB)

```python
import math

class UCB:
    def __init__(self, c=1.4):
        self.c = c  # Exploration parameter
        self.counts = {}  # Action counts
        self.values = {}  # Action values

    def select_action(self, state, q_values):
        action_counts = np.array([self.counts.get((state, a), 0) for a in range(len(q_values))])

        # If any action hasn't been tried, prioritize it
        if np.min(action_counts) == 0:
            return np.argmin(action_counts)

        # Calculate UCB values
        total_counts = np.sum(action_counts)
        ucb_values = q_values + self.c * np.sqrt(math.log(total_counts) / action_counts)

        return np.argmax(ucb_values)

    def update(self, state, action, value):
        key = (state, action)
        if key not in self.counts:
            self.counts[key] = 0
            self.values[key] = 0.0

        self.counts[key] += 1
        # Running average update
        self.values[key] += (value - self.values[key]) / self.counts[key]
```

### 4. Thompson Sampling

```python
class ThompsonSampling:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.successes = np.zeros(num_actions)
        self.failures = np.zeros(num_actions)

    def select_action(self):
        # Sample from beta distribution for each action
        samples = np.random.beta(self.successes + 1, self.failures + 1)
        return np.argmax(samples)

    def update(self, action, reward):
        if reward == 1:
            self.successes[action] += 1
        else:
            self.failures[action] += 1
```

---

## Implementation Patterns

### 1. Agent Training Loop

```python
def train_agent(agent, env, episodes=1000, max_steps=1000):
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Agent selects action
            action = agent.select_action(state)

            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Agent learns
            agent.learn(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

    return episode_rewards
```

### 2. Experience Replay Buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
```

### 3. Prioritized Experience Replay

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1.0))

    def sample(self, batch_size, beta=0.4):
        # Calculate probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / np.sum(probs)

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Get samples
        batch = [self.buffer[i] for i in indices]
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / np.max(weights)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = abs(td_error) + 1e-6
```

### 4. Model Checkpointing

```python
class ModelCheckpoint:
    def __init__(self, save_path, save_freq=100):
        self.save_path = save_path
        self.save_freq = save_freq
        self.best_reward = -np.inf

    def save_if_best(self, agent, episode_reward, episode):
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            agent.save(self.save_path)
            print(f"Saved model at episode {episode} with reward {episode_reward:.2f}")

    def save_periodically(self, agent, episode):
        if episode % self.save_freq == 0:
            path = f"{self.save_path}_episode_{episode}.h5"
            agent.save(path)
```

---

## Evaluation Metrics

### 1. Performance Metrics

```python
def evaluate_agent(agent, env, episodes=100):
    """Evaluate agent performance over multiple episodes"""
    episode_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'success_rate': np.mean(np.array(episode_rewards) > threshold)
    }
```

### 2. Learning Curves

```python
import matplotlib.pyplot as plt

def plot_learning_curve(rewards, window=100):
    """Plot learning curve with moving average"""
    plt.figure(figsize=(12, 6))

    # Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label='Episode Reward')

    # Moving average
    moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    plt.plot(moving_avg, label=f'{window}-Episode Moving Average')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)

    # Histogram of final performance
    plt.subplot(1, 2, 2)
    plt.hist(rewards[-100:], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Final Performance Distribution')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
```

### 3. Confidence Intervals

```python
def confidence_interval(rewards, confidence=0.95):
    """Calculate confidence interval for performance"""
    n = len(rewards)
    mean = np.mean(rewards)
    sem = stats.sem(rewards)  # Standard error of mean

    h = sem * stats.t.ppf((1 + confidence) / 2, n-1)
    return mean - h, mean + h
```

---

## Common Issues & Solutions

### 1. Sample Efficiency

**Problem**: Agent needs too many samples to learn
**Solutions**:

- Use experience replay
- Implement prioritized replay
- Use function approximation (Deep RL)
- Leverage demonstrations (Imitation Learning)

### 2. Exploration vs Exploitation

**Problem**: Agent gets stuck in local optima
**Solutions**:

- Implement proper exploration strategies
- Use entropy regularization
- Implement curiosity-driven rewards
- Curriculum learning

### 3. Overestimation Bias

**Problem**: Q-values are systematically overestimated
**Solutions**:

- Double DQN
- Dueling DQN architecture
- Target networks with slower updates
- Use multiple critics

### 4. Training Instability

**Problem**: Policy oscillates or diverges
**Solutions**:

- Gradient clipping
- Learning rate scheduling
- Normalize observations
- Use advantage actor-critic methods

### 5. Reward Hacking

**Problem**: Agent finds unintended ways to maximize reward
**Solutions**:

- Careful reward shaping
- Use reward constraints
- Human-in-the-loop feedback
- Multi-objective optimization

---

## Code Templates

### Complete DQN Training Template

```python
import gymnasium as gym
import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow import keras

class DQNTrainer:
    def __init__(self, env_name='CartPole-v1'):
        self.env = gym.make(env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        # Hyperparameters
        self.learning_rate = 0.001
        self.discount = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.train_start = 1000
        self.memory = deque(maxlen=10000)

        # Networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()

        # Training metrics
        self.episode_rewards = []
        self.losses = []

    def _build_network(self):
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.discount * np.max(next_q_values[i])

        history = self.q_network.fit(states, current_q_values, verbose=0)
        self.losses.append(history.history['loss'][0])

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def train(self, episodes=1000):
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.remember(state, action, reward, next_state, done)
                self.replay()

                state = next_state
                total_reward += reward

            self.episode_rewards.append(total_reward)

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Update target network
            if episode % 10 == 0:
                self.update_target_network()

            # Logging
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        return self.episode_rewards, self.losses

# Usage
if __name__ == "__main__":
    trainer = DQNTrainer('CartPole-v1')
    rewards, losses = trainer.train(episodes=1000)
```

### Complete A3C Training Template

```python
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import threading
import time

class ActorCriticAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Build actor and critic networks
        self.actor, self.critic = self._build_networks()

    def _build_networks(self):
        # Shared base network
        inputs = keras.Input(shape=(self.state_size,))
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.Dense(128, activation='relu')(x)

        # Actor head (policy)
        policy = keras.layers.Dense(self.action_size, activation='softmax')(x)

        # Critic head (value)
        value = keras.layers.Dense(1, activation='linear')(x)

        actor = keras.Model(inputs=inputs, outputs=policy)
        critic = keras.Model(inputs=inputs, outputs=value)

        actor.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
        critic.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

        return actor, critic

    def get_action(self, state):
        policy = self.actor.predict(state.reshape(1, -1), verbose=0)[0]
        action = np.random.choice(self.action_size, p=policy)
        return action

    def update_networks(self, states, actions, rewards, next_states, dones):
        # Calculate targets and advantages
        values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(next_states, verbose=0).flatten()

        targets = rewards + 0.99 * next_values * (1 - dones)
        advantages = targets - values

        # Prepare one-hot actions
        actions_one_hot = keras.utils.to_categorical(actions, self.action_size)

        # Update critic (MSE loss)
        self.critic.fit(states, targets, verbose=0)

        # Update actor (policy gradient with advantage)
        with tf.GradientTape() as tape:
            policies = self.actor(states, training=True)
            log_probs = tf.math.log(policies + 1e-8)
            selected_log_probs = tf.reduce_sum(actions_one_hot * log_probs, axis=1)
            actor_loss = -tf.reduce_mean(selected_log_probs * advantages)

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

class A3CTrainer:
    def __init__(self, env_name, num_workers=4):
        self.env_name = env_name
        self.num_workers = num_workers
        self.global_episode = 0

    def worker(self, worker_id):
        env = gym.make(self.env_name)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        # Create local agent
        agent = ActorCriticAgent(state_size, action_size)

        while True:
            state, _ = env.reset()
            episode_reward = 0
            states, actions, rewards, next_states, dones = [], [], [], [], []

            while True:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                episode_reward += reward
                state = next_state

                if done:
                    # Convert lists to arrays
                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    next_states = np.array(next_states)
                    dones = np.array(dones)

                    # Update networks
                    agent.update_networks(states, actions, rewards, next_states, dones)

                    print(f"Worker {worker_id}, Episode {self.global_episode}, Reward: {episode_reward:.2f}")
                    self.global_episode += 1
                    break

    def train(self):
        threads = []
        for worker_id in range(self.num_workers):
            thread = threading.Thread(target=self.worker, args=(worker_id,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

# Usage
if __name__ == "__main__":
    trainer = A3CTrainer('CartPole-v1', num_workers=4)
    trainer.train()
```

---

## Resources & References

### Books

1. **"Reinforcement Learning: An Introduction"** by Sutton & Barto
2. **"Deep Reinforcement Learning"** by Aske Plaat
3. **"Algorithms for Reinforcement Learning"** by Csaba Szepesvari

### Papers

1. **Deep Q-Networks**: "Human-level control through deep reinforcement learning"
2. **Policy Gradients**: "Simple statistical gradient-following algorithms for connectionist reinforcement learning"
3. **Actor-Critic**: "Actor-Critic Algorithms"
4. **Proximal Policy Optimization**: "Proximal Policy Optimization Algorithms"

### Libraries & Frameworks

- **Stable Baselines3**: Professional RL library
- **RLlib**: Distributed RL library by Ray
- **Gym/Gymnasium**: Environment interface
- **TensorForce**: TensorFlow-based RL library
- **OpenAI Baselines**: High-quality implementations

### Online Resources

- OpenAI Spinning Up in Deep RL
- DeepMind Lab: 3D learning environment
- Unity ML-Agents: Game-based environments
- Google's Dopamine: Research framework

### Practice Platforms

- **OpenAI Gym**: Standardized environments
- **RoboSchool**: Robot simulation
- **DeepMind Control Suite**: Continuous control
- **Atari Learning Environment**: Game-based benchmarks

---

## Quick Reference Checklist

### ✅ Environment Setup

- [ ] Install required libraries (gym, numpy, tensorflow/pytorch)
- [ ] Set up environment with proper rendering
- [ ] Implement custom environment if needed

### ✅ Algorithm Selection

- [ ] Discrete actions → Q-Learning/DQN
- [ ] Continuous actions → Policy Gradient/Actor-Critic
- [ ] Model-free vs Model-based requirements
- [ ] Sample efficiency needs

### ✅ Hyperparameter Tuning

- [ ] Learning rate (0.001 - 0.01)
- [ ] Discount factor (0.95 - 0.99)
- [ ] Exploration rate/strategy
- [ ] Network architecture

### ✅ Training Process

- [ ] Monitor learning curves
- [ ] Implement early stopping
- [ ] Save/checkpoint models
- [ ] Evaluate on test environment

### ✅ Evaluation

- [ ] Run multiple episodes for statistical significance
- [ ] Compare against baseline agents
- [ ] Analyze final performance distribution
- [ ] Document hyperparameter settings

---

_Last Updated: November 2025_
_This cheat sheet covers core RL concepts, algorithms, and implementation patterns for practical reinforcement learning applications._
