# AI Agents Fundamentals - Theory Guide

## Table of Contents

1. [Introduction to AI Agents](#introduction)
2. [Agent Architecture](#architecture)
3. [Types of AI Agents](#types)
4. [Decision Making and Planning](#decision-making)
5. [Learning and Adaptation](#learning)
6. [Communication and Coordination](#communication)
7. [Multi-Agent Systems](#multi-agent)
8. [Evaluation and Metrics](#evaluation)
9. [Current Trends and Applications](#trends)

## Introduction to AI Agents {#introduction}

### What is an AI Agent?

An AI agent is an autonomous entity that perceives its environment, processes information, and takes actions to achieve specific goals. Agents operate based on their design, learning capabilities, and the information they gather from their surroundings.

**Key Characteristics:**

- **Autonomy**: Operates independently without constant human intervention
- **Reactivity**: Responds to changes in the environment
- **Pro-activeness**: Takes initiative to achieve goals
- **Social Ability**: Interacts with other agents and humans

### Components of an AI Agent

```python
class AIAgent:
    """Basic AI Agent Structure"""

    def __init__(self, name, goals):
        self.name = name
        self.goals = goals
        self.percepts = []
        self.actions = []
        self.beliefs = {}
        self.intentions = []

    def perceive(self, environment):
        """Process environmental information"""
        # Sensor processing
        self.percepts = self.sensor_function(environment)

    def think(self):
        """Decision making process"""
        # Update beliefs based on percepts
        self.update_beliefs()

        # Plan actions to achieve goals
        self.plan_actions()

        # Select best action
        self.select_action()

    def act(self, environment):
        """Execute chosen action"""
        if self.intentions:
            action = self.intentions[0]
            result = self.actuator_function(environment, action)
            self.actions.append((action, result))

    def run(self, environment, steps=100):
        """Main agent loop"""
        for _ in range(steps):
            self.perceive(environment)
            self.think()
            self.act(environment)
```

### Agent vs. Program vs. System

- **Program**: Set of instructions that executes deterministically
- **Agent**: Program that can perceive and act autonomously
- **System**: Collection of interacting components

## Agent Architecture {#architecture}

### Simple Reflex Agent

The simplest form of agent that maps percepts directly to actions using rules.

```python
class SimpleReflexAgent:
    """Simple Reflex Agent Implementation"""

    def __init__(self, rules):
        self.rules = rules  # List of condition-action rules

    def agent_function(self, percept):
        """Map percept to action"""
        for condition, action in self.rules:
            if condition(percept):
                return action
        return None  # No matching rule

# Example: Vacuum cleaner agent
def vacuum_agent_function(percept):
    """Vacuum cleaner agent function"""
    location, status = percept

    if status == 'dirty':
        return 'suck'
    elif location == 'A':
        return 'move_right'
    elif location == 'B':
        return 'move_left'
    else:
        return 'stay'

# Rules for the vacuum cleaner
vacuum_rules = [
    (lambda p: p[1] == 'dirty', 'suck'),
    (lambda p: p[0] == 'A', 'move_right'),
    (lambda p: p[0] == 'B', 'move_left')
]
```

### Model-Based Reflex Agent

Enhanced reflex agent that maintains internal state to handle partially observable environments.

```python
class ModelBasedReflexAgent:
    """Model-Based Reflex Agent with Internal State"""

    def __init__(self, rules):
        self.rules = rules
        self.state = {}  # Internal state
        self.action_history = []

    def update_state(self, percept, action):
        """Update internal state based on percept and action"""
        # Update state based on action effects
        if action == 'move_right':
            self.state['location'] = 'B'
        elif action == 'move_left':
            self.state['location'] = 'A'
        elif action == 'suck':
            self.state['status'] = 'clean'

    def agent_function(self, percept):
        """Map percept to action considering internal state"""
        # Use both percept and state for decision making
        context = {**dict(percept), **self.state}

        for condition, action in self.rules:
            if condition(context):
                self.update_state(percept, action)
                return action
        return None

# Model-based rules
model_based_rules = [
    (lambda c: c.get('status') == 'dirty', 'suck'),
    (lambda c: c.get('location') == 'A', 'move_right'),
    (lambda c: c.get('location') == 'B', 'move_left')
]
```

### Goal-Based Agent

Agent that uses goals to guide decision making and can plan sequences of actions.

```python
class GoalBasedAgent:
    """Goal-Based Agent with Planning Capabilities"""

    def __init__(self, rules, goals):
        self.rules = rules
        self.goals = goals
        self.state = {}
        self.action_history = []

    def agent_function(self, percept):
        """Goal-oriented decision making"""
        # Update state
        self.update_state(percept)

        # Check if any goal is achieved
        for goal in self.goals:
            if self.goal_achieved(goal):
                continue

        # Find actions that lead to goal
        plan = self.plan_to_goal()
        if plan:
            return plan[0]  # Execute first action in plan

        # Fall back to rules
        return self.apply_rules()

    def plan_to_goal(self):
        """Simple planning to achieve goals"""
        # Breadth-first search for goal achievement
        for goal in self.goals:
            if not self.goal_achieved(goal):
                return self.find_action_sequence(goal)
        return None

    def find_action_sequence(self, goal):
        """Find sequence of actions to achieve goal"""
        # Simplified planning logic
        if goal == 'clean_house':
            return ['suck', 'move_right', 'suck', 'move_left']
        return None
```

### Utility-Based Agent

Agent that maximizes utility function to make optimal decisions.

```python
import numpy as np

class UtilityBasedAgent:
    """Utility-Based Agent for Optimal Decision Making"""

    def __init__(self, rules, utility_function):
        self.rules = rules
        self.utility_function = utility_function
        self.state = {}
        self.expected_utilities = {}

    def agent_function(self, percept):
        """Maximize expected utility"""
        self.update_state(percept)

        # Calculate utility for each possible action
        action_utilities = {}
        for rule in self.rules:
            action = rule[1]
            action_utilities[action] = self.calculate_expected_utility(action)

        # Select action with highest utility
        best_action = max(action_utilities, key=action_utilities.get)
        return best_action

    def calculate_expected_utility(self, action):
        """Calculate expected utility of an action"""
        # Simulate action outcomes
        outcomes = self.simulate_action_outcomes(action)

        # Calculate weighted utility
        expected_utility = 0
        for outcome, probability in outcomes.items():
            utility = self.utility_function(outcome)
            expected_utility += probability * utility

        return expected_utility

    def simulate_action_outcomes(self, action):
        """Simulate possible outcomes of an action"""
        # Simple simulation based on current state
        if action == 'suck':
            return {'clean': 0.9, 'dirty': 0.1}
        elif action == 'move_right':
            return {'location_B': 1.0}
        elif action == 'move_left':
            return {'location_A': 1.0}
        return {'unchanged': 1.0}

# Example utility function
def vacuum_utility_function(outcome):
    """Utility function for vacuum cleaner"""
    if 'clean' in outcome and outcome['clean']:
        return 10  # High utility for clean state
    elif 'location' in outcome:
        return 1   # Low utility for movement
    return 0      # No utility for other outcomes
```

### Learning Agent

Agent that improves performance through experience and learning.

```python
class LearningAgent:
    """Learning Agent with Experience-Based Improvement"""

    def __init__(self, initial_performance_element):
        self.performance_element = initial_performance_element
        self.critic = None
        self.learning_element = None
        self.problem_generator = None

    def agent_function(self, percept):
        """Learning agent cycle"""
        # Generate problem
        if self.problem_generator:
            problem = self.problem_generator(percept)
        else:
            problem = percept

        # Performance element makes decision
        action = self.performance_element(problem)

        # Critic evaluates performance
        if self.critic:
            feedback = self.critic(percept, action)

            # Learning element updates based on feedback
            if self.learning_element:
                self.learning_element(feedback)

        return action

# Example: Q-Learning Agent
class QLearningAgent(LearningAgent):
    """Q-Learning Agent Implementation"""

    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.9):
        # Initialize Q-table
        self.q_table = np.zeros((len(states), len(actions)))
        self.states = states
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.current_state = 0
        self.last_action = 0

    def performance_element(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < 0.1:  # Exploration
            return np.random.choice(len(self.actions))
        else:  # Exploitation
            return np.argmax(self.q_table[state])

    def critic(self, state, action, reward, next_state):
        """Update Q-values based on received reward"""
        # Q-learning update rule
        best_next_action = np.max(self.q_table[next_state])
        td_target = reward + self.discount_factor * best_next_action
        td_error = td_target - self.q_table[state, action]

        # Update Q-value
        self.q_table[state, action] += self.learning_rate * td_error

    def learning_element(self, feedback):
        """Process learning feedback"""
        # In Q-learning, learning is handled in critic
        pass
```

## Types of AI Agents {#types}

### Reactive Agents

Agents that respond directly to environmental stimuli without internal representations.

```python
class ReactiveAgent:
    """Reactive Agent Implementation"""

    def __init__(self, behavior_rules):
        self.behavior_rules = behavior_rules

    def step(self, sensor_input):
        """Direct sensor to action mapping"""
        for rule in self.behavior_rules:
            if rule['condition'](sensor_input):
                return rule['action']
        return None

# Example: Bug2 algorithm for robot navigation
class Bug2ReactiveAgent(ReactiveAgent):
    """Bug2 Reactive Navigation Agent"""

    def __init__(self):
        behavior_rules = [
            {
                'condition': lambda s: not s['goal_reached'] and not s['obstacle_detected'],
                'action': 'move_towards_goal'
            },
            {
                'condition': lambda s: s['obstacle_detected'],
                'action': 'follow_obstacle_boundary'
            },
            {
                'condition': lambda s: s['goal_reached'],
                'action': 'stop'
            }
        ]
        super().__init__(behavior_rules)
```

### Deliberative Agents

Agents that maintain internal models and plan actions using reasoning.

```python
import heapq

class DeliberativeAgent:
    """Deliberative Agent with Planning Capabilities"""

    def __init__(self, world_model, planner):
        self.world_model = world_model
        self.planner = planner
        self.current_plan = []

    def step(self, percept):
        """Deliberative planning cycle"""
        # Update world model with percept
        self.world_model.update(percept)

        # Check if current plan is still valid
        if not self.plan_is_valid():
            # Generate new plan
            self.current_plan = self.planner.plan(
                self.world_model.current_state,
                self.world_model.goal_state
            )

        # Execute next action in plan
        if self.current_plan:
            action = self.current_plan[0]
            return action
        return None

    def plan_is_valid(self):
        """Check if current plan is still valid"""
        if not self.current_plan:
            return False

        # Check if next action is still applicable
        next_action = self.current_plan[0]
        return self.world_model.is_action_applicable(next_action)

# Example: A* Planner
class AStarPlanner:
    """A* Planning Algorithm"""

    def __init__(self, heuristic_function):
        self.heuristic = heuristic_function

    def plan(self, start_state, goal_state):
        """Generate plan using A* algorithm"""
        open_set = [(0, start_state, [])]
        closed_set = set()

        while open_set:
            f_score, current_state, path = heapq.heappop(open_set)

            if current_state == goal_state:
                return path

            if current_state in closed_set:
                continue

            closed_set.add(current_state)

            # Generate successors
            for action, successor in current_state.get_successors():
                if successor not in closed_set:
                    g_score = len(path) + 1
 self.heuristic                    h_score =(successor, goal_state)
                    f_score = g_score + h_score
                    new_path = path + [action]
                    heapq.heappush(open_set, (f_score, successor, new_path))

        return []  # No plan found
```

### Hybrid Agents

Agents that combine reactive and deliberative components.

```python
class HybridAgent:
    """Hybrid Agent combining reactive and deliberative approaches"""

    def __init__(self, reactive_layer, deliberative_layer):
        self.reactive_layer = reactive_layer
        self.deliberative_layer = deliberative_layer
        self.current_behavior = 'reactive'  # or 'deliberative'

    def step(self, percept):
        """Hybrid decision making"""
        # Assess situation urgency
        urgency = self.assess_urgency(percept)

        if urgency > 0.8:  # High urgency
            self.current_behavior = 'reactive'
            return self.reactive_layer.step(percept)
        else:
            self.current_behavior = 'deliberative'
            return self.deliberative_layer.step(percept)

    def assess_urgency(self, percept):
        """Assess situation urgency (0-1 scale)"""
        # Simple urgency assessment
        if percept.get('danger_detected', False):
            return 1.0
        elif percept.get('goal_very_close', False):
            return 0.9
        else:
            return 0.3

# Example: Subsumption Architecture
class SubsumptionAgent:
    """Subsumption Architecture for Hybrid Agents"""

    def __init__(self):
        self.layers = []
        self.active_layer = 0

    def add_layer(self, layer):
        """Add behavior layer (higher index = higher priority)"""
        self.layers.append(layer)

    def step(self, percept):
        """Subsumption arbitration"""
        for i, layer in enumerate(self.layers):
            action = layer.step(percept)
            if action is not None:
                self.active_layer = i
                return action
        return None
```

### Distributed Agents

Agents that operate across multiple systems or locations.

```python
import asyncio
import json
from typing import Dict, Any

class DistributedAgent:
    """Distributed Agent Implementation"""

    def __init__(self, agent_id, communication_protocol):
        self.agent_id = agent_id
        self.communication_protocol = communication_protocol
        self.local_state = {}
        self.known_agents = set()

    async def send_message(self, recipient_id, message):
        """Send message to another agent"""
        await self.communication_protocol.send(self.agent_id, recipient_id, message)

    async def receive_message(self, sender_id, message):
        """Receive message from another agent"""
        # Process received message
        self.process_message(sender_id, message)

    def process_message(self, sender_id, message):
        """Process incoming message"""
        message_type = message.get('type')

        if message_type == 'state_update':
            # Update knowledge about other agent
            self.known_agents.add(sender_id)
            self.update_global_state(message['data'])

        elif message_type == 'request':
            # Respond to request
            response = self.handle_request(message)
            asyncio.create_task(self.send_message(sender_id, response))

    def update_global_state(self, remote_state):
        """Update global state with remote information"""
        # Merge local and remote state
        for key, value in remote_state.items():
            if key in self.local_state:
                # Resolve conflicts using consensus algorithm
                self.local_state[key] = self.resolve_conflict(
                    self.local_state[key], value
                )
            else:
                self.local_state[key] = value

    def resolve_conflict(self, local_value, remote_value):
        """Simple conflict resolution"""
        # Could implement voting, timestamp-based, or other consensus methods
        return remote_value  # Simple last-write-wins

    def handle_request(self, request):
        """Handle request from other agent"""
        request_type = request.get('request_type')

        if request_type == 'state_query':
            return {
                'type': 'state_response',
                'data': self.local_state
            }
        return None

# Multi-Agent System Coordinator
class MultiAgentCoordinator:
    """Coordinator for multi-agent systems"""

    def __init__(self):
        self.agents = {}
        self.coordination_protocols = {}

    def register_agent(self, agent):
        """Register agent in the system"""
        self.agents[agent.agent_id] = agent

    async def coordinate_agents(self, goal):
        """Coordinate agents to achieve common goal"""
        # Simple coordination: broadcast goal to all agents
        for agent in self.agents.values():
            await agent.send_message('coordinator', {
                'type': 'goal_assignment',
                'goal': goal
            })

        # Wait for responses and coordinate actions
        responses = await self.collect_responses()
        return self.synthesize_actions(responses)

    async def collect_responses(self, timeout=10):
        """Collect responses from agents"""
        # Implementation would collect responses from all agents
        responses = {}
        # ... coordination logic
        return responses

    def synthesize_actions(self, responses):
        """Synthesize coordinated actions from responses"""
        # Combine individual agent plans into coordinated plan
        coordinated_plan = []
        # ... synthesis logic
        return coordinated_plan
```

## Decision Making and Planning {#decision-making}

### Decision Theory

Framework for making rational decisions under uncertainty.

```python
import numpy as np
from typing import List, Dict, Tuple

class DecisionMaker:
    """Decision Theory-based Agent"""

    def __init__(self, utility_function, belief_function):
        self.utility_function = utility_function
        self.belief_function = belief_function

    def make_decision(self, actions, state_beliefs):
        """Make decision using decision theory"""
        action_utilities = {}

        for action in actions:
            # Calculate expected utility of action
            expected_utility = self.calculate_expected_utility(
                action, state_beliefs
            )
            action_utilities[action] = expected_utility

        # Select action with maximum expected utility
        best_action = max(action_utilities, key=action_utilities.get)
        return best_action, action_utilities

    def calculate_expected_utility(self, action, state_beliefs):
        """Calculate expected utility of an action"""
        expected_utility = 0

        for state, probability in state_beliefs.items():
            # Calculate utility of action in this state
            utility = self.utility_function(action, state)
            expected_utility += probability * utility

        return expected_utility

# Example: Medical Diagnosis Agent
class MedicalDiagnosisAgent(DecisionMaker):
    """Medical Diagnosis Agent using Decision Theory"""

    def __init__(self):
        # Define utility function for medical decisions
        def medical_utility(action, state):
            if action == 'treat' and state['disease'] == 'present':
                return 10  # High utility for correct treatment
            elif action == 'treat' and state['disease'] == 'absent':
                return -5  # Negative utility for unnecessary treatment
            elif action == 'test' and state['disease'] == 'present':
                return 3   # Positive utility for early detection
            elif action == 'test' and state['disease'] == 'absent':
                return 1   # Positive utility for confirmation
            return 0

        # Define belief function
        def belief_function(symptoms):
            # Simple belief updating based on symptoms
            if symptoms['fever'] and symptoms['cough']:
                return {'disease': 'present': 0.8, 'disease': 'absent': 0.2}
            elif symptoms['fever']:
                return {'disease': 'present': 0.6, 'disease': 'absent': 0.4}
            else:
                return {'disease': 'present': 0.1, 'disease': 'absent': 0.9}

        super().__init__(medical_utility, belief_function)
```

### Planning Algorithms

**Forward Chaining Planner:**

```python
class ForwardChainingPlanner:
    """Forward Chaining Planning Algorithm"""

    def __init__(self, actions, goals):
        self.actions = actions
        self.goals = goals
        self.initial_state = set()
        self.goal_state = set(goals)

    def plan(self, initial_state):
        """Generate plan using forward chaining"""
        current_state = set(initial_state)
        plan = []
        achieved_goals = set()

        while not achieved_goals.issuperset(self.goal_state):
            action_applied = False

            for action in self.actions:
                if self.is_action_applicable(action, current_state):
                    # Apply action
                    new_state = self.apply_action(action, current_state)

                    # Check if action contributes to goals
                    if self.action_contributes_to_goals(action, self.goal_state):
                        plan.append(action)
                        current_state = new_state
                        action_applied = True
                        break

            if not action_applied:
                # No applicable action found
                break

        return plan if achieved_goals.issuperset(self.goal_state) else None

    def is_action_applicable(self, action, state):
        """Check if action is applicable in current state"""
        return action['preconditions'].issubset(state)

    def apply_action(self, action, state):
        """Apply action to state"""
        new_state = state.copy()
        new_state.update(action['effects'])
        return new_state

    def action_contributes_to_goals(self, action, goals):
        """Check if action contributes to achieving goals"""
        return not action['effects'].isdisjoint(goals)
```

**Backward Chaining Planner:**

```python
class BackwardChainingPlanner:
    """Backward Chaining Planning Algorithm"""

    def __init__(self, actions, goals):
        self.actions = actions
        self.goals = set(goals)

    def plan(self, initial_state, goal_state):
        """Generate plan using backward chaining"""
        return self.backward_chain(goal_state, set(initial_state))

    def backward_chain(self, goals, current_state):
        """Recursive backward chaining"""
        if goals.issubset(current_state):
            return []  # Goals already achieved

        for action in self.actions:
            if not action['effects'].isdisjoint(goals):
                # Action contributes to goals
                remaining_goals = goals - action['effects']
                sub_plan = self.backward_chain(remaining_goals, current_state)

                if sub_plan is not None:
                    # Check if preconditions can be satisfied
                    if self.can_satisfy_preconditions(action['preconditions'], current_state, sub_plan):
                        return [action] + sub_plan

        return None  # No plan found

    def can_satisfy_preconditions(self, preconditions, current_state, sub_plan):
        """Check if preconditions can be satisfied by sub-plan"""
        # Simplified: assume sub-plan can achieve preconditions
        return True
```

### Reinforcement Learning for Planning

```python
import gymnasium as gym

class RLPlanningAgent:
    """Reinforcement Learning Agent for Planning"""

    def __init__(self, environment, learning_algorithm='q_learning'):
        self.env = environment
        self.learning_algorithm = learning_algorithm
        self.q_table = {}
        self.policy = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1

    def learn(self, episodes=1000):
        """Learn policy through interaction"""
        for episode in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # Update Q-values
                self.update_q_value(state, action, reward, next_state)

                state = next_state

        # Extract policy from Q-values
        self.extract_policy()

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(state)

    def get_best_action(self, state):
        """Get best action for state"""
        if state not in self.q_table:
            return self.env.action_space.sample()

        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning update rule"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in range(self.env.action_space.n)}

        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in range(self.env.action_space.n)}

        # Q-learning update
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state][action] = new_q

    def extract_policy(self):
        """Extract policy from Q-values"""
        for state in self.q_table:
            self.policy[state] = max(
                self.q_table[state],
                key=self.q_table[state].get
            )

    def plan(self, initial_state):
        """Generate plan using learned policy"""
        plan = []
        state = initial_state
        visited_states = set()
        max_steps = 100

        for _ in range(max_steps):
            if state in visited_states:
                break  # Avoid cycles

            visited_states.add(state)

            if state not in self.policy:
                break  # No policy for this state

            action = self.policy[state]
            plan.append(action)

            # Simulate action to get next state
            next_state = self.simulate_action(state, action)
            state = next_state

        return plan

    def simulate_action(self, state, action):
        """Simulate action to predict next state"""
        # Simplified simulation
        # In practice, you'd use a model of the environment
        return state  # Placeholder
```

## Learning and Adaptation {#learning}

### Supervised Learning for Agents

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

class SupervisedLearningAgent:
    """Agent that learns from labeled examples"""

    def __init__(self, learning_algorithm='random_forest'):
        self.learning_algorithm = learning_algorithm
        self.model = None
        self.training_data = []
        self.training_labels = []

    def train(self, features, labels):
        """Train the learning model"""
        if self.learning_algorithm == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.learning_algorithm == 'neural_network':
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)

        self.model.fit(features, labels)

    def predict(self, features):
        """Make predictions using trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(features)

    def predict_proba(self, features):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict_proba(features)

    def add_training_example(self, features, label):
        """Add single training example"""
        self.training_data.append(features)
        self.training_labels.append(label)

    def incremental_train(self):
        """Train model with accumulated examples"""
        if self.training_data:
            features = np.array(self.training_data)
            labels = np.array(self.training_labels)
            self.train(features, labels)

# Example: Image Classification Agent
class ImageClassificationAgent(SupervisedLearningAgent):
    """Agent that classifies images"""

    def __init__(self):
        super().__init__(learning_algorithm='neural_network')
        self.image_preprocessor = None

    def preprocess_image(self, image):
        """Preprocess image for classification"""
        # Resize, normalize, etc.
        processed_image = self.image_preprocessor.transform(image)
        return processed_image

    def classify_image(self, image):
        """Classify an image"""
        processed_image = self.preprocess_image(image)
        prediction = self.predict(processed_image.reshape(1, -1))
        probabilities = self.predict_proba(processed_image.reshape(1, -1))

        return {
            'class': prediction[0],
            'confidence': probabilities[0].max(),
            'all_probabilities': probabilities[0]
        }
```

### Unsupervised Learning for Agents

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class UnsupervisedLearningAgent:
    """Agent that learns patterns from unlabeled data"""

    def __init__(self, learning_method='clustering'):
        self.learning_method = learning_method
        self.model = None
        self.data = None

    def fit(self, data):
        """Fit unsupervised model to data"""
        self.data = data

        if self.learning_method == 'clustering':
            self.model = KMeans(n_clusters=5, random_state=42)
            clusters = self.model.fit_predict(data)

        elif self.learning_method == 'pca':
            self.model = PCA(n_components=2, random_state=42)
            reduced_data = self.model.fit_transform(data)

        elif self.learning_method == 'tsne':
            self.model = TSNE(n_components=2, random_state=42)
            reduced_data = self.model.fit_transform(data)

        return self.model

    def predict_cluster(self, sample):
        """Predict cluster for a new sample"""
        if self.learning_method == 'clustering':
            return self.model.predict(sample.reshape(1, -1))[0]
        else:
            # For dimensionality reduction, find nearest neighbor
            distances = np.linalg.norm(self.data - sample, axis=1)
            nearest_index = np.argmin(distances)
            return self.model.labels_[nearest_index]

    def get_cluster_centers(self):
        """Get cluster centers"""
        if self.learning_method == 'clustering':
            return self.model.cluster_centers_
        return None

    def analyze_patterns(self):
        """Analyze discovered patterns"""
        if self.learning_method == 'clustering':
            return {
                'n_clusters': len(self.model.cluster_centers_),
                'inertia': self.model.inertia_,
                'cluster_sizes': np.bincount(self.model.labels_)
            }
        return {}
```

### Online Learning

```python
class OnlineLearningAgent:
    """Agent that learns incrementally from streaming data"""

    def __init__(self, update_strategy=' SGD'):
        self.update_strategy = update_strategy
        self.weights = None
        self.learning_rate = 0.01
        self.performance_history = []

    def initialize(self, input_dim):
        """Initialize model parameters"""
        self.weights = np.random.normal(0, 0.1, input_dim)

    def update(self, sample, target, error):
        """Update model with new sample"""
        if self.update_strategy == 'SGD':
            # Stochastic Gradient Descent update
            gradient = self.compute_gradient(sample, target, error)
            self.weights -= self.learning_rate * gradient

        elif self.update_strategy == 'adaptive':
            # Adaptive learning rate update
            adaptive_lr = self.learning_rate / (1 + len(self.performance_history) * 0.01)
            gradient = self.compute_gradient(sample, target, error)
            self.weights -= adaptive_lr * gradient

        # Record performance
        self.performance_history.append(error)

    def compute_gradient(self, sample, target, error):
        """Compute gradient for parameter update"""
        # Simplified gradient computation
        return error * sample

    def predict(self, sample):
        """Make prediction on new sample"""
        return np.dot(sample, self.weights)

    def get_learning_curve(self):
        """Get learning curve showing performance over time"""
        return self.performance_history

# Example: Adaptive Trading Agent
class AdaptiveTradingAgent(OnlineLearningAgent):
    """Agent that adapts trading strategy based on market feedback"""

    def __init__(self):
        super().__init__(update_strategy='adaptive')
        self.market_features = ['price', 'volume', 'momentum', 'volatility']
        self.trading_threshold = 0.5

    def process_market_data(self, features, price_change):
        """Process market data and learn"""
        # Normalize features
        normalized_features = self.normalize_features(features)

        # Make prediction
        signal_strength = self.predict(normalized_features)

        # Determine action
        action = 'buy' if signal_strength > self.trading_threshold else 'sell'

        # Calculate reward (simplified)
        reward = price_change if action == 'buy' else -price_change

        # Update model
        error = reward - signal_strength
        self.update(normalized_features, reward, error)

        return action, signal_strength

    def normalize_features(self, features):
        """Normalize market features"""
        # Simple normalization
        normalized = np.array(features)
        return normalized / (np.linalg.norm(normalized) + 1e-8)
```

### Meta-Learning

```python
class MetaLearningAgent:
    """Agent that learns how to learn"""

    def __init__(self, base_learner, meta_learner):
        self.base_learner = base_learner
        self.meta_learner = meta_learner
        self.task_history = []
        self.performance_history = []

    def learn_task(self, task_data, task_id):
        """Learn a new task"""
        # Split task data
        train_data, test_data = self.split_task_data(task_data)

        # Learn base model
        self.base_learner.train(train_data['features'], train_data['labels'])

        # Evaluate on test data
        performance = self.evaluate_performance(test_data)

        # Store task information
        task_info = {
            'task_id': task_id,
            'performance': performance,
            'task_data': task_data
        }
        self.task_history.append(task_info)
        self.performance_history.append(performance)

        # Update meta-learner
        self.update_meta_learner(task_info)

        return performance

    def split_task_data(self, task_data):
        """Split task data into train and test"""
        # Simple 80-20 split
        n_samples = len(task_data['features'])
        split_idx = int(0.8 * n_samples)

        train_data = {
            'features': task_data['features'][:split_idx],
            'labels': task_data['labels'][:split_idx]
        }

        test_data = {
            'features': task_data['features'][split_idx:],
            'labels': task_data['labels'][split_idx:]
        }

        return train_data, test_data

    def evaluate_performance(self, test_data):
        """Evaluate model performance"""
        predictions = self.base_learner.predict(test_data['features'])
        accuracy = np.mean(predictions == test_data['labels'])
        return accuracy

    def update_meta_learner(self, task_info):
        """Update meta-learner based on task performance"""
        # Extract meta-features from task
        meta_features = self.extract_meta_features(task_info)

        # Update meta-learner
        if hasattr(self.meta_learner, 'add_training_example'):
            self.meta_learner.add_training_example(meta_features, task_info['performance'])

    def extract_meta_features(self, task_info):
        """Extract meta-features that describe the task"""
        features = []

        # Task complexity features
        task_data = task_info['task_data']
        features.append(len(task_data['features']))  # Number of samples
        features.append(len(np.unique(task_data['labels'])))  # Number of classes

        # Data distribution features
        features.append(np.mean(task_data['features']))  # Feature mean
        features.append(np.std(task_data['features']))   # Feature std

        return np.array(features)

    def select_best_learner(self, new_task_features):
        """Select best learner for new task"""
        if hasattr(self.meta_learner, 'predict'):
            predicted_performance = self.meta_learner.predict(new_task_features.reshape(1, -1))

            # Select learner based on predicted performance
            if predicted_performance > 0.7:
                return 'high_performance_learner'
            else:
                return 'robust_learner'
        else:
            return 'default_learner'
```

## Communication and Coordination {#communication}

### Agent Communication Protocols

```python
import json
from enum import Enum
from typing import Dict, Any, List

class MessageType(Enum):
    REQUEST = "request"
    INFORM = "inform"
    QUERY = "query"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"

class AgentCommunication:
    """Agent Communication Framework"""

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.message_queue = []
        self.active_conversations = {}

    def send_message(self, recipient_id, message_type, content, conversation_id=None):
        """Send message to another agent"""
        message = {
            'sender': self.agent_id,
            'recipient': recipient_id,
            'type': message_type.value,
            'content': content,
            'conversation_id': conversation_id or f"conv_{self.agent_id}_{recipient_id}",
            'timestamp': self.get_timestamp()
        }

        return message

    def receive_message(self, message):
        """Process received message"""
        self.message_queue.append(message)

        # Handle different message types
        if message['type'] == MessageType.REQUEST.value:
            return self.handle_request(message)
        elif message['type'] == MessageType.QUERY.value:
            return self.handle_query(message)
        elif message['type'] == MessageType.PROPOSE.value:
            return self.handle_proposal(message)
        else:
            return self.handle_inform(message)

    def handle_request(self, message):
        """Handle request message"""
        request_content = message['content']

        # Process request
        if request_content['action'] == 'provide_information':
            response_content = self.provide_information(request_content['query'])
        elif request_content['action'] == 'coordinate_action':
            response_content = self.coordinate_action(request_content['action_details'])
        else:
            response_content = {'status': 'unknown_action'}

        return self.send_message(
            message['sender'],
            MessageType.INFORM,
            response_content,
            message['conversation_id']
        )

    def handle_query(self, message):
        """Handle query message"""
        query_content = message['content']
        answer = self.answer_query(query_content['question'])

        return self.send_message(
            message['sender'],
            MessageType.INFORM,
            {'answer': answer},
            message['conversation_id']
        )

    def handle_proposal(self, message):
        """Handle proposal message"""
        proposal = message['content']
        decision = self.evaluate_proposal(proposal)

        response_type = MessageType.ACCEPT if decision else MessageType.REJECT

        return self.send_message(
            message['sender'],
            response_type,
            {'decision': decision, 'reason': 'evaluated'},
            message['conversation_id']
        )

    def provide_information(self, query):
        """Provide information based on query"""
        # Implementation depends on agent's knowledge
        return {'information': 'relevant data'}

    def coordinate_action(self, action_details):
        """Coordinate on action"""
        # Implementation for action coordination
        return {'coordinated': True}

    def answer_query(self, question):
        """Answer query"""
        # Implementation for query answering
        return 'answer to the question'

    def evaluate_proposal(self, proposal):
        """Evaluate proposal"""
        # Decision logic for accepting/rejecting proposals
        return True  # or False

    def get_timestamp(self):
        """Get current timestamp"""
        import time
        return time.time()

# Example: Negotiation Protocol
class NegotiationProtocol(AgentCommunication):
    """Negotiation Protocol for Multi-Agent Systems"""

    def __init__(self, agent_id, negotiation_topics):
        super().__init__(agent_id)
        self.negotiation_topics = negotiation_topics
        self.offers = {}
        self.acceptance_threshold = 0.8

    def make_offer(self, recipient_id, topic, offer_value):
        """Make an offer in negotiation"""
        offer = {
            'topic': topic,
            'value': offer_value,
            'utility': self.calculate_utility(topic, offer_value),
            'timestamp': self.get_timestamp()
        }

        return self.send_message(
            recipient_id,
            MessageType.PROPOSE,
            offer
        )

    def evaluate_proposal(self, proposal):
        """Evaluate negotiation proposal"""
        topic = proposal['topic']
        offered_value = proposal['value']
        offered_utility = proposal['utility']

        # Calculate our utility for this offer
        our_utility = self.calculate_utility(topic, offered_value)

        # Accept if utility meets threshold
        return our_utility >= self.acceptance_threshold

    def calculate_utility(self, topic, value):
        """Calculate utility of a value for given topic"""
        # Utility function implementation
        # This would be domain-specific
        if topic in self.negotiation_topics:
            return min(value / 100, 1.0)  # Simplified utility
        return 0.0
```

### Coordination Mechanisms

**Contract Net Protocol:**

```python
class ContractNetProtocol:
    """Contract Net Protocol for Task Allocation"""

    def __init__(self, coordinator_id):
        self.coordinator_id = coordinator_id
        self.active_contracts = {}
        self.bidding_agents = set()

    def announce_task(self, task_description, deadline):
        """Announce task to eligible agents"""
        announcement = {
            'task_id': self.generate_task_id(),
            'description': task_description,
            'deadline': deadline,
            'announcement_time': self.get_timestamp()
        }

        # Broadcast to all agents
        for agent_id in self.bidding_agents:
            self.send_message(agent_id, MessageType.REQUEST, {
                'action': 'bid_on_task',
                'announcement': announcement
            })

        return announcement['task_id']

    def receive_bid(self, agent_id, bid):
        """Receive bid from agent"""
        task_id = bid['task_id']

        if task_id not in self.active_contracts:
            self.active_contracts[task_id] = []

        self.active_contracts[task_id].append({
            'agent_id': agent_id,
            'bid_value': bid['bid_value'],
            'estimated_completion': bid['estimated_completion'],
            'bid_timestamp': self.get_timestamp()
        })

    def award_contract(self, task_id, winning_agent_id):
        """Award contract to winning agent"""
        winning_bid = None
        for bid in self.active_contracts[task_id]:
            if bid['agent_id'] == winning_agent_id:
                winning_bid = bid
                break

        if winning_bid:
            # Send contract award
            contract = {
                'task_id': task_id,
                'contractor_id': winning_agent_id,
                'terms': winning_bid,
                'award_time': self.get_timestamp()
            }

            self.send_message(winning_agent_id, MessageType.ACCEPT, {
                'action': 'contract_awarded',
                'contract': contract
            })

            return contract
        return None

    def generate_task_id(self):
        """Generate unique task ID"""
        import uuid
        return str(uuid.uuid4())
```

**Market-Based Coordination:**

```python
class MarketBasedCoordination:
    """Market-Based Coordination Mechanism"""

    def __init__(self):
        self.agents = {}
        self.market_prices = {}
        self.transaction_history = []

    def register_agent(self, agent_id, agent_type, initial_budget):
        """Register agent in the market"""
        self.agents[agent_id] = {
            'type': agent_type,
            'budget': initial_budget,
            'offers': {},
            'demands': {}
        }

    def submit_offer(self, agent_id, good, price, quantity):
        """Submit offer to market"""
        if agent_id not in self.agents:
            raise ValueError("Agent not registered")

        agent = self.agents[agent_id]

        if agent['budget'] >= price * quantity:
            offer = {
                'agent_id': agent_id,
                'good': good,
                'price': price,
                'quantity': quantity,
                'timestamp': self.get_timestamp()
            }

            # Store offer
            if good not in agent['offers']:
                agent['offers'][good] = []
            agent['offers'][good].append(offer)

            return offer
        return None

    def match_offers(self, good):
        """Match offers and demands for a good"""
        # Get all offers for the good
        all_offers = []
        for agent in self.agents.values():
            if good in agent['offers']:
                all_offers.extend(agent['offers'][good])

        # Sort by price (ascending for offers)
        all_offers.sort(key=lambda x: x['price'])

        # Match with demands (simplified)
        transactions = []
        for offer in all_offers:
            # Find matching demand
            demand = self.find_matching_demand(good, offer['price'])
            if demand:
                transaction = self.execute_transaction(offer, demand)
                if transaction:
                    transactions.append(transaction)

        return transactions

    def find_matching_demand(self, good, max_price):
        """Find demand that matches offer price"""
        # Simplified demand matching
        for agent_id, agent in self.agents.items():
            if good in agent['demands']:
                for demand in agent['demands'][good]:
                    if demand['max_price'] >= max_price:
                        return demand
        return None

    def execute_transaction(self, offer, demand):
        """Execute transaction between offer and demand"""
        buyer_id = demand['agent_id']
        seller_id = offer['agent_id']

        if buyer_id not in self.agents or seller_id not in self.agents:
            return None

        buyer = self.agents[buyer_id]
        seller = self.agents[seller_id]

        transaction = {
            'buyer_id': buyer_id,
            'seller_id': seller_id,
            'good': offer['good'],
            'quantity': min(offer['quantity'], demand['quantity']),
            'price': offer['price'],
            'timestamp': self.get_timestamp()
        }

        # Update budgets
        total_cost = transaction['price'] * transaction['quantity']
        if buyer['budget'] >= total_cost:
            buyer['budget'] -= total_cost
            seller['budget'] += total_cost

            # Record transaction
            self.transaction_history.append(transaction)

            return transaction
        return None
```

## Multi-Agent Systems {#multi-agent}

### Agent Environments

```python
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class AgentEnvironment(ABC):
    """Abstract base class for agent environments"""

    def __init__(self, agents: List, initial_state: Dict):
        self.agents = agents
        self.state = initial_state
        self.time_step = 0
        self.is_terminal = False

    @abstractmethod
    def get_percepts(self, agent_id: str) -> Dict:
        """Get percepts for specific agent"""
        pass

    @abstractmethod
    def execute_action(self, agent_id: str, action: Any) -> Dict:
        """Execute action for specific agent"""
        pass

    @abstractmethod
    def is_terminal_state(self) -> bool:
        """Check if environment is in terminal state"""
        pass

    def step(self):
        """Execute one time step"""
        if self.is_terminal:
            return

        # Get actions from all agents
        actions = {}
        for agent in self.agents:
            percepts = self.get_percepts(agent.agent_id)
            action = agent.select_action(percepts)
            actions[agent.agent_id] = action

        # Execute actions and get results
        results = {}
        for agent_id, action in actions.items():
            result = self.execute_action(agent_id, action)
            results[agent_id] = result

        # Update agent learning
        for agent in self.agents:
            agent.learn(self.get_percepts(agent.agent_id), results[agent.agent_id])

        # Update environment state
        self.update_state(actions, results)
        self.time_step += 1

        # Check terminal condition
        self.is_terminal = self.is_terminal_state()

    def run(self, max_steps=1000):
        """Run environment for specified number of steps"""
        for _ in range(max_steps):
            if self.is_terminal:
                break
            self.step()
        return self.state

# Example: Grid World Environment
class GridWorldEnvironment(AgentEnvironment):
    """Multi-Agent Grid World Environment"""

    def __init__(self, agents: List, grid_size=(10, 10), obstacles=None):
        if obstacles is None:
            obstacles = set()

        initial_state = {
            'grid_size': grid_size,
            'agent_positions': {agent.agent_id: (0, 0) for agent in agents},
            'obstacles': obstacles,
            'goals': {(grid_size[0]-1, grid_size[1]-1): 'goal'},
            'rewards': {}
        }

        super().__init__(agents, initial_state)

    def get_percepts(self, agent_id: str) -> Dict:
        """Get percepts for specific agent"""
        position = self.state['agent_positions'][agent_id]
        grid_size = self.state['grid_size']
        obstacles = self.state['obstacles']

        # Simple percept: position and nearby obstacles
        percepts = {
            'position': position,
            'grid_size': grid_size,
            'nearby_obstacles': self.get_nearby_obstacles(position),
            'goal_position': list(self.state['goals'].keys())[0] if self.state['goals'] else None
        }

        return percepts

    def execute_action(self, agent_id: str, action: str) -> Dict:
        """Execute action for specific agent"""
        position = self.state['agent_positions'][agent_id]
        new_position = self.compute_new_position(position, action)

        # Check if new position is valid
        if self.is_valid_position(new_position):
            self.state['agent_positions'][agent_id] = new_position
            reward = self.calculate_reward(agent_id, new_position)
        else:
            reward = -1  # Penalty for invalid move

        # Update rewards
        self.state['rewards'][agent_id] = reward

        return {
            'new_position': self.state['agent_positions'][agent_id],
            'reward': reward,
            'success': self.is_valid_position(new_position)
        }

    def compute_new_position(self, position, action):
        """Compute new position based on action"""
        x, y = position

        if action == 'up':
            return (x, y - 1)
        elif action == 'down':
            return (x, y + 1)
        elif action == 'left':
            return (x - 1, y)
        elif action == 'right':
            return (x + 1, y)
        else:
            return position

    def is_valid_position(self, position):
        """Check if position is valid"""
        x, y = position
        grid_width, grid_height = self.state['grid_size']

        # Check bounds
        if x < 0 or x >= grid_width or y < 0 or y >= grid_height:
            return False

        # Check obstacles
        if position in self.state['obstacles']:
            return False

        return True

    def get_nearby_obstacles(self, position, radius=1):
        """Get obstacles within radius of position"""
        nearby = set()
        x, y = position

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                obstacle_pos = (x + dx, y + dy)
                if obstacle_pos in self.state['obstacles']:
                    nearby.add(obstacle_pos)

        return nearby

    def calculate_reward(self, agent_id, position):
        """Calculate reward for agent at position"""
        goal_position = list(self.state['goals'].keys())[0] if self.state['goals'] else None

        if goal_position and position == goal_position:
            return 100  # Goal reached
        else:
            # Small penalty for each step
            return -1

    def is_terminal_state(self) -> bool:
        """Check if all agents reached goal or max steps reached"""
        if self.time_step >= 1000:  # Max steps
            return True

        # Check if all agents reached goal
        goal_position = list(self.state['goals'].keys())[0] if self.state['goals'] else None

        if goal_position:
            for position in self.state['agent_positions'].values():
                if position != goal_position:
                    return False
            return True

        return False
```

### Cooperation and Competition

```python
class CooperativeAgent:
    """Agent designed for cooperative multi-agent scenarios"""

    def __init__(self, agent_id, cooperation_strategy='altruistic'):
        self.agent_id = agent_id
        self.cooperation_strategy = cooperation_strategy
        self.team_performance_history = []
        self.individual_performance_history = []

    def select_action(self, percepts):
        """Select action considering cooperation"""
        other_agents = percepts.get('other_agents', {})

        if self.cooperation_strategy == 'altruistic':
            return self.select_altruistic_action(other_agents)
        elif self.cooperation_strategy == 'fair':
            return self.select_fair_action(other_agents)
        elif self.cooperation_strategy == 'competitive':
            return self.select_competitive_action(other_agents)
        else:
            return self.select_individual_action(percepts)

    def select_altruistic_action(self, other_agents):
        """Select action that benefits others most"""
        best_action = None
        max_benefit = -float('inf')

        for action, benefit in self.compute_cooperative_benefits(other_agents).items():
            if benefit > max_benefit:
                max_benefit = benefit
                best_action = action

        return best_action

    def compute_cooperative_benefits(self, other_agents):
        """Compute benefits of actions for the team"""
        # Simplified benefit calculation
        benefits = {}

        for action in ['cooperate', 'compete', 'help', 'share']:
            total_benefit = 0
            for agent_id, agent_state in other_agents.items():
                # Calculate how this action helps other agents
                benefit = self.calculate_action_benefit(action, agent_state)
                total_benefit += benefit
            benefits[action] = total_benefit

        return benefits

    def calculate_action_benefit(self, action, agent_state):
        """Calculate benefit of action for specific agent"""
        # Domain-specific benefit calculation
        if action == 'help' and agent_state.get('needs_help', False):
            return 10
        elif action == 'share' and agent_state.get('has_surplus', False):
            return 5
        else:
            return 0

    def learn_cooperation(self, team_performance, individual_performance):
        """Learn cooperation strategies"""
        self.team_performance_history.append(team_performance)
        self.individual_performance_history.append(individual_performance)

        # Adjust cooperation strategy based on performance
        if len(self.team_performance_history) > 10:
            recent_team_perf = np.mean(self.team_performance_history[-10:])
            recent_individual_perf = np.mean(self.individual_performance_history[-10:])

            # If team performance is much better than individual, be more cooperative
            if recent_team_perf > recent_individual_perf * 1.5:
                self.cooperation_strategy = 'altruistic'
            elif recent_individual_perf > recent_team_perf * 1.5:
                self.cooperation_strategy = 'competitive'

class CompetitiveAgent:
    """Agent designed for competitive multi-agent scenarios"""

    def __init__(self, agent_id, competition_style='aggressive'):
        self.agent_id = agent_id
        self.competition_style = competition_style
        self.victories = 0
        self.defeats = 0

    def select_action(self, percepts):
        """Select action considering competition"""
        other_agents = percepts.get('other_agents', {})

        if self.competition_style == 'aggressive':
            return self.select_aggressive_action(other_agents)
        elif self.competition_style == 'defensive':
            return self.select_defensive_action(other_agents)
        elif self.competition_style == 'opportunistic':
            return self.select_opportunistic_action(other_agents)
        else:
            return self.select_neutral_action(percepts)

    def select_aggressive_action(self, other_agents):
        """Select action that maximizes advantage over others"""
        best_action = None
        max_advantage = -float('inf')

        for action in self.get_possible_actions():
            advantage = self.calculate_competitive_advantage(action, other_agents)
            if advantage > max_advantage:
                max_advantage = advantage
                best_action = action

        return best_action

    def calculate_competitive_advantage(self, action, other_agents):
        """Calculate competitive advantage of action"""
        advantage = 0

        for agent_id, agent_state in other_agents.items():
            # Calculate how this action gives advantage over this agent
            my_gain = self.calculate_action_gain(action, agent_state)
            their_gain = agent_state.get('potential_gain', 0)
            advantage += my_gain - their_gain

        return advantage

    def calculate_action_gain(self, action, opponent_state):
        """Calculate personal gain from action"""
        # Simplified gain calculation
        if action == 'attack' and opponent_state.get('vulnerable', False):
            return 10
        elif action == 'defend' and opponent_state.get('threatening', False):
            return 5
        else:
            return 1

    def learn_competition(self, victory, performance_metrics):
        """Learn from competitive outcomes"""
        if victory:
            self.victories += 1
        else:
            self.defeats += 1

        # Adjust competition style based on success
        win_rate = self.victories / (self.victories + self.defeats)

        if win_rate < 0.3 and self.competition_style == 'aggressive':
            self.competition_style = 'defensive'
        elif win_rate > 0.7 and self.competition_style == 'defensive':
            self.competition_style = 'aggressive'
```

## Evaluation and Metrics {#evaluation}

### Performance Metrics for Agents

```python
import numpy as np
from typing import Dict, List, Any

class AgentMetrics:
    """Metrics for evaluating agent performance"""

    def __init__(self):
        self.metrics_history = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []

    def calculate_performance_metrics(self, agent_id, episode_data):
        """Calculate comprehensive performance metrics"""
        metrics = {}

        # Basic metrics
        metrics['total_reward'] = episode_data.get('total_reward', 0)
        metrics['episode_length'] = episode_data.get('episode_length', 0)
        metrics['success'] = episode_data.get('success', False)

        # Efficiency metrics
        metrics['reward_per_step'] = (
            metrics['total_reward'] / max(metrics['episode_length'], 1)
        )

        # Learning progress
        if len(self.episode_rewards) > 0:
            metrics['improvement_rate'] = self.calculate_improvement_rate()

        # Goal achievement
        metrics['goal_achievement_rate'] = self.calculate_goal_achievement_rate()

        # Resource utilization
        metrics['resource_efficiency'] = self.calculate_resource_efficiency(episode_data)

        return metrics

    def calculate_improvement_rate(self):
        """Calculate rate of improvement over episodes"""
        if len(self.episode_rewards) < 10:
            return 0

        recent_rewards = self.episode_rewards[-10:]
        early_rewards = self.episode_rewards[:10]

        recent_avg = np.mean(recent_rewards)
        early_avg = np.mean(early_rewards)

        if early_avg == 0:
            return 0

        improvement_rate = (recent_avg - early_avg) / abs(early_avg)
        return improvement_rate

    def calculate_goal_achievement_rate(self, window_size=100):
        """Calculate goal achievement rate over recent episodes"""
        recent_successes = self.success_rates[-window_size:]
        return np.mean(recent_successes) if recent_successes else 0

    def calculate_resource_efficiency(self, episode_data):
        """Calculate resource efficiency metrics"""
        resources_used = episode_data.get('resources_used', {})
        goals_achieved = episode_data.get('goals_achieved', 0)

        if goals_achieved == 0:
            return 0

        total_resources = sum(resources_used.values())
        return goals_achieved / 1)

 max(total_resources,    def update_metrics(self, episode_data):
        """Update metrics with new episode data"""
        metrics = self.calculate_performance_metrics('agent', episode_data)
        self.metrics_history.append(metrics)

        # Update specific metric histories
        self.episode_rewards.append(episode_data.get('total_reward', 0))
        self.episode_lengths.append(episode_data.get('episode_length', 0))
        self.success_rates.append(episode_data.get('success', False))

    def get_performance_summary(self, agent_id):
        """Get performance summary for agent"""
        if not self.metrics_history:
            return {}

        recent_metrics = self.metrics_history[-100:]  # Last 100 episodes

        summary = {
            'agent_id': agent_id,
            'total_episodes': len(self.metrics_history),
            'recent_performance': {
                'avg_reward': np.mean([m['total_reward'] for m in recent_metrics]),
                'avg_episode_length': np.mean([m['episode_length'] for m in recent_metrics]),
                'success_rate': np.mean([m['success'] for m in recent_metrics]),
                'avg_efficiency': np.mean([m['reward_per_step'] for m in recent_metrics])
            },
            'learning_progress': {
                'improvement_rate': self.calculate_improvement_rate(),
                'goal_achievement_rate': self.calculate_goal_achievement_rate()
            }
        }

        return summary

class MultiAgentMetrics:
    """Metrics for evaluating multi-agent systems"""

    def __init__(self):
        self.system_metrics = []
        self.agent_interactions = []
        self.cooperation_metrics = []
        self.competition_metrics = []

    def evaluate_system_performance(self, agents, environment_state):
        """Evaluate overall system performance"""
        metrics = {}

        # Individual agent performance
        individual_performances = []
        for agent in agents:
            agent_perf = self.evaluate_individual_agent(agent, environment_state)
            individual_performances.append(agent_perf)

        metrics['individual_performances'] = individual_performances
        metrics['avg_individual_performance'] = np.mean(individual_performances)

        # System-level metrics
        metrics['system_efficiency'] = self.calculate_system_efficiency(agents)
        metrics['resource_utilization'] = self.calculate_resource_utilization(agents)
        metrics['goal_achievement'] = self.calculate_system_goal_achievement(agents)

        # Interaction metrics
        metrics['cooperation_level'] = self.calculate_cooperation_level(agents)
        metrics['conflict_frequency'] = self.calculate_conflict_frequency(agents)

        return metrics

    def evaluate_individual_agent(self, agent, environment_state):
        """Evaluate individual agent performance"""
        # Implementation depends on agent type and environment
        # This is a simplified example
        performance_score = 0

        # Goal achievement
        if hasattr(agent, 'goals_achieved'):
            performance_score += agent.goals_achieved * 10

        # Resource efficiency
        if hasattr(agent, 'resources_used'):
            if agent.goals_achieved > 0:
                efficiency = agent.goals_achieved / sum(agent.resources_used.values())
                performance_score += efficiency * 5

        return performance_score

    def calculate_system_efficiency(self, agents):
        """Calculate overall system efficiency"""
        total_goals = sum(getattr(agent, 'goals_achieved', 0) for agent in agents)
        total_resources = sum(
            sum(getattr(agent, 'resources_used', {}).values())
            for agent in agents
        )

        if total_resources == 0:
            return 0

        return total_goals / total_resources

    def calculate_resource_utilization(self, agents):
        """Calculate resource utilization efficiency"""
        resource_usage = {}

        for agent in agents:
            resources = getattr(agent, 'resources_used', {})
            for resource_type, amount in resources.items():
                if resource_type not in resource_usage:
                    resource_usage[resource_type] = 0
                resource_usage[resource_type] += amount

        # Calculate utilization rates (simplified)
        utilization_rates = []
        for resource_type, total_used in resource_usage.items():
            # Assuming total available is 100 for each resource
            utilization_rate = total_used / 100
            utilization_rates.append(utilization_rate)

        return np.mean(utilization_rates) if utilization_rates else 0

    def calculate_cooperation_level(self, agents):
        """Calculate level of cooperation among agents"""
        cooperation_count = 0
        total_interactions = 0

        # Count cooperative interactions
        for agent in agents:
            interactions = getattr(agent, 'interactions', [])
            for interaction in interactions:
                if interaction.get('type') == 'cooperation':
                    cooperation_count += 1
                total_interactions += 1

        if total_interactions == 0:
            return 0

        return cooperation_count / total_interactions

    def calculate_conflict_frequency(self, agents):
        """Calculate frequency of conflicts among agents"""
        conflict_count = 0
        total_interactions = 0

        # Count conflicts
        for agent in agents:
            interactions = getattr(agent, 'interactions', [])
            for interaction in interactions:
                if interaction.get('type') == 'conflict':
                    conflict_count += 1
                total_interactions += 1

        if total_interactions == 0:
            return 0

        return conflict_count / total_interactions
```

### Benchmarking and Testing

```python
class AgentBenchmark:
    """Benchmarking framework for AI agents"""

    def __init__(self, benchmark_suites):
        self.benchmark_suites = benchmark_suites
        self.results = {}

    def run_benchmark(self, agent, suite_name):
        """Run specific benchmark suite"""
        if suite_name not in self.benchmark_suites:
            raise ValueError(f"Unknown benchmark suite: {suite_name}")

        suite = self.benchmark_suites[suite_name]
        results = []

        for test_case in suite['test_cases']:
            result = self.run_test_case(agent, test_case)
            results.append(result)

        self.results[suite_name] = results
        return results

    def run_test_case(self, agent, test_case):
        """Run individual test case"""
        environment = test_case['environment']
        expected_behavior = test_case['expected_behavior']
        evaluation_criteria = test_case['evaluation_criteria']

        # Run agent in environment
        agent_behavior = self.evaluate_agent_behavior(agent, environment)

        # Compare with expected behavior
        compliance_score = self.calculate_compliance_score(
            agent_behavior, expected_behavior, evaluation_criteria
        )

        return {
            'test_case_id': test_case['id'],
            'compliance_score': compliance_score,
            'agent_behavior': agent_behavior,
            'passed': compliance_score >= test_case.get('pass_threshold', 0.7)
        }

    def evaluate_agent_behavior(self, agent,Evaluate agent environment):
        """ behavior in environment"""
        # Simplified behavior evaluation
        percepts = environment.get_percepts(agent.agent_id)
        action = agent.select_action(percepts)

        return {
            'action': action,
            'response_time': getattr(agent, 'response_time', 0),
            'reasoning_steps': getattr(agent, 'reasoning_steps', 1)
        }

    def calculate_compliance_score(self, agent_behavior, expected_behavior, criteria):
        """Calculate compliance score with expected behavior"""
        score = 0
        total_weight = 0

        for criterion, weight in criteria.items():
            if criterion == 'action_correctness':
                if agent_behavior['action'] == expected_behavior['action']:
                    score += weight
                total_weight += weight

            elif criterion == 'response_time':
                max_time = expected_behavior.get('max_response_time', 1.0)
                if agent_behavior['response_time'] <= max_time:
                    score += weight
                total_weight += weight

            elif criterion == 'reasoning_quality':
                min_steps = expected_behavior.get('min_reasoning_steps', 1)
                if agent_behavior['reasoning_steps'] >= min_steps:
                    score += weight
                total_weight += weight

        return score / max(total_weight, 1)

# Example benchmark suite
benchmark_suite = {
    'basic_navigation': {
        'test_cases': [
            {
                'id': 'navigate_to_goal',
                'environment': {'type': 'grid_world', 'size': (5, 5)},
                'expected_behavior': {
                    'action': 'move_to_goal',
                    'max_response_time': 0.1,
                    'min_reasoning_steps': 2
                },
                'evaluation_criteria': {
                    'action_correctness': 0.5,
                    'response_time': 0.3,
                    'reasoning_quality': 0.2
                },
                'pass_threshold': 0.8
            }
        ]
    }
}
```

## Current Trends and Applications {#trends}

### Large Language Model Agents

```python
class LLMAgent:
    """Agent powered by Large Language Model"""

    def __init__(self, llm_model, system_prompt, tools=None):
        self.llm = llm_model
        self.system_prompt = system_prompt
        self.tools = tools or {}
        self.conversation_history = []
        self.memory = {}

    def think(self, user_input):
        """LLM-based reasoning and planning"""
        # Build context
        context = self.build_context(user_input)

        # Generate response using LLM
        response = self.llm.generate(
            prompt=context,
            system_prompt=self.system_prompt
        )

        # Extract actions from response
        actions = self.extract_actions(response)

        return actions

    def build_context(self, user_input):
        """Build context for LLM"""
        context_parts = [
            self.system_prompt,
            f"User: {user_input}",
            "Conversation history:",
        ]

        # Add recent conversation history
        for message in self.conversation_history[-5:]:  # Last 5 messages
            context_parts.append(f"{message['role']}: {message['content']}")

        # Add relevant memory
        if self.memory:
            context_parts.append("Relevant memory:")
            for key, value in self.memory.items():
                context_parts.append(f"{key}: {value}")

        return "\\n".join(context_parts)

    def extract_actions(self, llm_response):
        """Extract actionable items from LLM response"""
        # Simple extraction - in practice, this would be more sophisticated
        actions = []

        # Look for tool calls
        if "tool:" in llm_response.lower():
            # Extract tool names and parameters
            lines = llm_response.split("\\n")
            for line in lines:
                if line.strip().startswith("tool:"):
                    tool_name = line.split(":")[1].strip()
                    actions.append({'type': 'tool_call', 'tool': tool_name})

        # Look for planning actions
        if "plan:" in llm_response.lower():
            actions.append({'type': 'planning', 'content': llm_response})

        return actions

    def execute_tool(self, tool_name, parameters):
        """Execute tool if available"""
        if tool_name in self.tools:
            return self.tools[tool_name](**parameters)
        else:
            return {'error': f'Tool {tool_name} not found'}

    def learn(self, user_input, response, outcome):
        """Learn from interactions"""
        # Store in memory
        memory_key = f"interaction_{len(self.conversation_history)}"
        self.memory[memory_key] = {
            'input': user_input,
            'response': response,
            'outcome': outcome
        }

        # Update conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_input
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })

        # Prune history if too long
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

# Example tool system for LLM agents
def create_agent_tools():
    """Create set of tools for LLM agent"""
    return {
        'web_search': lambda query: f"Search results for: {query}",
        'calculator': lambda expression: eval(expression),
        'weather': lambda location: f"Weather in {location}: Sunny",
        'calendar': lambda action, date: f"Calendar {action} for {date}",
        'email': lambda to, subject, body: f"Email sent to {to}"
    }

class MultiModalAgent(LLMAgent):
    """Agent that can process multiple modalities"""

    def __init__(self, llm_model, vision_model, audio_model, system_prompt):
        super().__init__(llm_model, system_prompt)
        self.vision_model = vision_model
        self.audio_model = audio_model

    def process_multimodal_input(self, text=None, image=None, audio=None):
        """Process input from multiple modalities"""
        processed_inputs = {}

        if text:
            processed_inputs['text'] = text

        if image:
            # Process image using vision model
            image_description = self.vision_model.analyze(image)
            processed_inputs['image'] = image_description

        if audio:
            # Process audio using audio model
            audio_transcript = self.audio_model.transcribe(audio)
            processed_inputs['audio'] = audio_transcript

        # Combine modalities for LLM processing
        combined_input = self.combine_modalities(processed_inputs)

        return self.think(combined_input)

    def combine_modalities(self, processed_inputs):
        """Combine processed modalities into unified input"""
        parts = []

        if 'text' in processed_inputs:
            parts.append(f"Text: {processed_inputs['text']}")

        if 'image' in processed_inputs:
            parts.append(f"Image: {processed_inputs['image']}")

        if 'audio' in processed_inputs:
            parts.append(f"Audio: {processed_inputs['audio']}")

        return " | ".join(parts)
```

### Swarm Intelligence

```python
class SwarmAgent:
    """Agent in a swarm intelligence system"""

    def __init__(self, agent_id, position, velocity, behavior_params):
        self.agent_id = agent_id
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.behavior_params = behavior_params
        self.neighbors = []
        self.pheromone_trail = 0

    def update_neighbors(self, all_agents, detection_radius):
        """Update list of neighboring agents"""
        self.neighbors = []

        for agent in all_agents:
            if agent.agent_id != self.agent_id:
                distance = np.linalg.norm(self.position - agent.position)
                if distance <= detection_radius:
                    self.neighbors.append(agent)

    def calculate_swarm_forces(self):
        """Calculate forces from swarm behavior"""
        forces = np.zeros(2)

        # Separation force (avoid crowding)
        separation_force = self.calculate_separation_force()
        forces += separation_force * self.behavior_params['separation_weight']

        # Alignment force (match velocity with neighbors)
        alignment_force = self.calculate_alignment_force()
        forces += alignment_force * self.behavior_params['alignment_weight']

        # Cohesion force (move toward center of mass)
        cohesion_force = self.calculate_cohesion_force()
        forces += cohesion_force * self.behavior_params['cohesion_weight']

        return forces

    def calculate_separation_force(self):
        """Calculate separation force"""
        separation_force = np.zeros(2)

        for neighbor in self.neighbors:
            diff = self.position - neighbor.position
            distance = np.linalg.norm(diff)
            if distance > 0:
                separation_force += diff / (distance ** 2)

        return separation_force

    def calculate_alignment_force(self):
        """Calculate alignment force"""
        if not self.neighbors:
            return np.zeros(2)

        avg_velocity = np.mean([neighbor.velocity for neighbor in self.neighbors], axis=0)
        return avg_velocity - self.velocity

    def calculate_cohesion_force(self):
        """Calculate cohesion force"""
        if not self.neighbors:
            return np.zeros(2)

        center_of_mass = np.mean([neighbor.position for neighbor in self.neighbors], axis=0)
        return center_of_mass - self.position

    def update_state(self, forces, time_step=1):
        """Update agent state based on forces"""
        # Update velocity
        self.velocity += forces * time_step

        # Limit velocity
        max_speed = self.behavior_params.get('max_speed', 1.0)
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = self.velocity * max_speed / speed

        # Update position
        self.position += self.velocity * time_step

class SwarmIntelligence:
    """Swarm Intelligence System"""

    def __init__(self, num_agents, bounds, behavior_params):
        self.agents = self.initialize_agents(num_agents, bounds, behavior_params)
        self.bounds = bounds
        self.global_best_position = None
        self.global_best_fitness = float('inf')

    def initialize_agents(self, num_agents, bounds, behavior_params):
        """Initialize swarm of agents"""
        agents = []

        for i in range(num_agents):
            position = np.random.uniform(bounds[0], bounds[1], 2)
            velocity = np.random.uniform(-1, 1, 2)

            agent = SwarmAgent(i, position, velocity, behavior_params)
            agents.append(agent)

        return agents

    def optimize_function(self, position):
        """Function to optimize (e.g., Rastrigin function)"""
        x, y = position
        return 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))

    def step(self):
        """Execute one optimization step"""
        # Update neighbors for all agents
        detection_radius = 5.0
        for agent in self.agents:
            agent.update_neighbors(self.agents, detection_radius)

        # Update all agents
        for agent in self.agents:
            forces = agent.calculate_swarm_forces()
            agent.update_state(forces)

            # Evaluate fitness
            fitness = self.optimize_function(agent.position)

            # Update global best
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = agent.position.copy()

        return self.global_best_position, self.global_best_fitness

    def run_optimization(self, max_iterations=1000):
        """Run swarm optimization"""
        for iteration in range(max_iterations):
            best_pos, best_fit = self.step()

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Best fitness = {best_fit:.6f}")

        return best_pos, best_fit
```

This comprehensive theory guide covers all fundamental aspects of AI agents, from basic concepts to cutting-edge applications. It provides both theoretical understanding and practical implementation examples that are essential for mastering AI agent development.
