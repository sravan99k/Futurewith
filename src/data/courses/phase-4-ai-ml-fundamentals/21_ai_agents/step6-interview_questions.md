# AI Agents Interview Preparation

**Version:** 1.0 | **Date:** November 2025

## Comprehensive Interview Guide for AI Agents

---

## 📋 Table of Contents

1. [Technical Concepts & Theory](#technical-concepts)
2. [System Design Questions](#system-design)
3. [Coding Challenges](#coding-challenges)
4. [Architecture & Implementation](#architecture-implementation)
5. [Practical Scenarios](#practical-scenarios)
6. [Behavioral Questions](#behavioral-questions)
7. [Industry-Specific Questions](#industry-specific)
8. [Advanced Topics](#advanced-topics)

---

## Technical Concepts & Theory {#technical-concepts}

### Q1: Explain the different types of AI agents and their characteristics.

**Answer:**
AI agents can be categorized into several types based on their capabilities and architecture:

**1. Simple Reflex Agents**

- Operate on a stimulus-response basis
- Use condition-action rules (if-then statements)
- Fast response but limited adaptability
- Example: Thermostat, basic robot obstacle avoidance
- Pros: Simple, predictable, computationally efficient
- Cons: Cannot handle unexpected situations, no learning capability

**2. Model-Based Agents**

- Maintain an internal model of the world
- Track changes in environment over time
- Use history to supplement current perception
- Example: Self-driving cars tracking other vehicles
- Pros: Can handle partially observable environments
- Cons: Requires more computational resources

**3. Goal-Based Agents**

- Work towards specific objectives
- Consider future consequences of actions
- Use planning algorithms
- Example: Delivery robots, game-playing agents
- Pros: Can achieve complex objectives
- Cons: Planning can be computationally expensive

**4. Utility-Based Agents**

- Optimize for maximum utility or value
- Balance multiple competing objectives
- Consider trade-offs between different outcomes
- Example: Financial trading agents, recommendation systems
- Pros: Can handle multi-objective optimization
- Cons: Requires careful utility function design

**5. Learning Agents**

- Improve performance through experience
- Adapt behavior based on feedback
- Can handle novel situations
- Example: Game-playing AI, autonomous vehicles
- Pros: Highly adaptable, can discover new strategies
- Cons: Training can be time-consuming, initial performance may be poor

**Follow-up Questions:**

- Can you provide real-world examples of each type?
- How would you combine multiple agent types in a system?
- What are the computational trade-offs between different types?

### Q2: What are the key differences between multi-agent systems and single-agent systems?

**Answer:**

| Aspect            | Single Agent                         | Multi-Agent System                 |
| ----------------- | ------------------------------------ | ---------------------------------- |
| **Complexity**    | Lower complexity                     | Higher complexity                  |
| **Communication** | No communication needed              | Requires inter-agent communication |
| **Coordination**  | Internal coordination                | External coordination mechanisms   |
| **Scalability**   | Limited by single processing unit    | Can scale horizontally             |
| **Robustness**    | Single point of failure              | Distributed redundancy             |
| **Performance**   | Limited by single agent capabilities | Can achieve emergent behaviors     |
| **Design**        | Simpler design                       | Complex protocol design            |
| **Debugging**     | Easier to debug                      | Harder to debug and test           |

**Key Challenges in Multi-Agent Systems:**

1. **Coordination**: How agents work together toward common goals
2. **Communication**: Efficient message passing and protocol design
3. **Competition**: Handling conflicting interests between agents
4. **Emergence**: Unpredictable behaviors arising from agent interactions
5. **Scalability**: Managing large numbers of agents efficiently

**Advantages of Multi-Agent Systems:**

1. **Parallel Processing**: Tasks can be distributed across agents
2. **Robustness**: Failure of one agent doesn't crash entire system
3. **Scalability**: Can add more agents as needed
4. **Specialization**: Different agents can specialize in different tasks
5. **Flexibility**: Can adapt to changing requirements

### Q3: Explain the concept of emergence in multi-agent systems and provide examples.

**Answer:**

**Emergence** is the phenomenon where complex behaviors, patterns, or properties arise from simple interactions between agents, without central control or explicit programming of these behaviors.

**Characteristics of Emergent Behavior:**

1. **Non-linear**: Small changes can have large effects
2. **Unpredictable**: Difficult to foresee emergent outcomes
3. **Bottom-up**: Arises from local interactions, not global control
4. **Adaptive**: System can self-organize and adapt
5. **Robust**: System maintains function despite individual agent failures

**Types of Emergent Behavior:**

**1. Flocking Behavior (Boids)**

```python
# Simple boids rules that create complex flocking
def flocking_rules(agent, neighbors):
    separation = steer_away_from(neighbors, min_distance=20)
    alignment = align_with(neighbors)
    cohesion = move_toward_center(neighbors)

    # Weighted combination creates flocking behavior
    return separation * 1.5 + alignment * 1.0 + cohesion * 1.0
```

**2. Swarm Intelligence**

- Ant colony optimization
- Particle swarm optimization
- Bee colony algorithms
- Can solve complex optimization problems

**3. Market Dynamics**

- Price discovery through buyer-seller interactions
- Supply and demand equilibrium
- Bubble formation and crash

**4. Social Behaviors**

- Information propagation in social networks
- Opinion formation and consensus
- Cultural evolution

**Design Principles for Emergent Systems:**

1. **Local Rules**: Define simple rules for individual agents
2. **Limited Communication**: Restrict information sharing
3. **Feedback Loops**: Include mechanisms for feedback
4. **Diversity**: Allow for variation in agent behaviors
5. **Iteration**: Enable repeated interactions over time

**Controlling Emergence:**

- Carefully tune agent parameters
- Implement global constraints
- Use monitoring and intervention mechanisms
- Design for desired emergent behaviors

### Q4: What are the main challenges in designing communication protocols for agents?

**Answer:**

**1. Message Delivery Guarantees**

- **Problem**: Ensuring messages reach intended recipients
- **Solutions**: Acknowledgments, timeouts, retry mechanisms
- **Trade-offs**: Reliability vs. performance

**2. Message Ordering**

- **Problem**: Maintaining logical message sequence
- **Solutions**: Sequence numbers, timestamps, causal ordering
- **Considerations**: Network delays, processing speeds

**3. Deadlock Prevention**

- **Problem**: Agents waiting for messages that never arrive
- **Solutions**: Timeouts, message queues, circular dependency detection
- **Monitoring**: Detect and resolve deadlocks

**4. Message Broadcast**

- **Problem**: Efficient one-to-many communication
- **Solutions**: Topic-based publish-subscribe, multicast protocols
- **Scalability**: Handle large numbers of subscribers

**5. Protocol Negotiation**

- **Problem**: Agents need to agree on communication protocol
- **Solutions**: Protocol discovery, version negotiation
- **Backward compatibility**: Support multiple protocol versions

**Common Communication Patterns:**

**Request-Response Pattern:**

```python
# Implementation challenges
class RequestResponseProtocol:
    def __init__(self):
        self.pending_requests = {}
        self.timeouts = {}

    async def send_request(self, receiver, request):
        request_id = generate_unique_id()
        self.pending_requests[request_id] = request

        # Set timeout
        self.timeouts[request_id] = asyncio.get_event_loop().time() + 30

        await self.send_message(receiver, {
            'type': 'request',
            'id': request_id,
            'content': request
        })

        return request_id
```

**Publish-Subscribe Pattern:**

```python
class PubSubProtocol:
    def __init__(self):
        self.topics = defaultdict(list)  # topic -> subscribers
        self.message_history = {}

    def subscribe(self, agent_id, topic):
        self.topics[topic].append(agent_id)

    def publish(self, sender, topic, message):
        # Deliver to all subscribers
        for subscriber in self.topics[topic]:
            self.send_message(sender, subscriber, {
                'type': 'publication',
                'topic': topic,
                'message': message
            })
```

### Q5: Explain reinforcement learning in the context of AI agents.

**Answer:**

**Reinforcement Learning (RL)** is a learning paradigm where agents learn to make decisions by interacting with an environment to maximize cumulative reward.

**Key Components:**

1. **Agent**: The learning entity
2. **Environment**: The world the agent operates in
3. **State (S)**: Current situation of the agent
4. **Action (A)**: What the agent can do
5. **Reward (R)**: Feedback from the environment
6. **Policy (π)**: Strategy that defines agent's behavior

**RL Algorithms:**

**1. Q-Learning**

```python
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * max_next_q
        self.q_table[state, action] += self.alpha * (td_target - current_q)
```

**2. Policy Gradient Methods**

```python
class PolicyGradientAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_network = self.build_network()

    def compute_loss(self, states, actions, rewards):
        # REINFORCE algorithm
        log_probs = self.compute_log_probs(states, actions)
        returns = self.compute_returns(rewards)
        loss = -np.mean(log_probs * returns)
        return loss
```

**3. Actor-Critic Methods**

```python
class ActorCriticAgent:
    def __init__(self):
        self.actor = PolicyNetwork()    # Learns policy
        self.critic = ValueNetwork()    # Learns value function

    def update(self, state, action, reward, next_state):
        # Actor update
        actor_loss = self.compute_actor_loss(state, action, reward)
        self.actor.update(actor_loss)

        # Critic update
        critic_loss = self.compute_critic_loss(state, reward, next_state)
        self.critic.update(critic_loss)
```

**Applications in AI Agents:**

1. **Game Playing**: AlphaGo, chess agents
2. **Robotics**: Robot navigation, manipulation
3. **Trading**: Portfolio management, algorithmic trading
4. **Recommendation**: Personalized content delivery
5. **Resource Management**: Energy optimization, traffic control

**Challenges:**

1. **Exploration vs. Exploitation**: Balancing trying new actions vs. using known good actions
2. **Credit Assignment**: Determining which actions led to rewards
3. **Non-stationary Environments**: Environment changes over time
4. **Sample Efficiency**: Learning quickly from limited experience
5. **Scalability**: Handling large state and action spaces

---

## System Design Questions {#system-design}

### Q1: Design a distributed chatbot system with multiple specialized agents.

**Answer:**

**System Overview:**

```
User Input → Intent Classifier Agent → Router Agent → Specialized Agents
                                          ↓
Response Aggregator Agent ← Multiple Specialized Agents
                                          ↓
User Output
```

**Architecture Components:**

**1. Intent Classification Agent**

```python
class IntentClassifierAgent:
    def __init__(self):
        self.model = load_classification_model()
        self.intents = {
            'weather': WeatherAgent(),
            'news': NewsAgent(),
            'shopping': ShoppingAgent(),
            'technical': TechnicalSupportAgent()
        }

    def classify_intent(self, user_input):
        features = self.extract_features(user_input)
        intent_scores = self.model.predict(features)
        return max(intent_scores, key=intent_scores.get)
```

**2. Specialized Agents**

```python
class WeatherAgent:
    def handle_request(self, query):
        location = self.extract_location(query)
        weather_data = self.get_weather_data(location)
        return self.format_weather_response(weather_data)

class NewsAgent:
    def handle_request(self, query):
        topic = self.extract_topic(query)
        articles = self.fetch_news_articles(topic)
        return self.format_news_response(articles)
```

**3. Response Aggregation**

```python
class ResponseAggregator:
    def __init__(self):
        self.response_cache = {}
        self.confidence_threshold = 0.7

    def aggregate_responses(self, primary_response, backup_responses):
        if primary_response.confidence > self.confidence_threshold:
            return primary_response
        else:
            return self.combine_responses(backup_responses)
```

**Design Considerations:**

**Scalability:**

- Horizontal scaling of specialized agents
- Load balancing across agent instances
- Microservices architecture
- Message queuing for high throughput

**Reliability:**

- Circuit breaker pattern for failing agents
- Timeout mechanisms
- Graceful degradation
- Health checking and monitoring

**Performance:**

- Response caching
- Parallel agent execution
- Async processing
- Request batching

**Fault Tolerance:**

- Multiple instances of critical agents
- Database replication
- Message persistence
- Automatic failover

### Q2: Design a multi-agent system for autonomous vehicle coordination at intersections.

**Answer:**

**System Architecture:**

**1. Vehicle Agents**

```python
class VehicleAgent:
    def __init__(self, vehicle_id, position, velocity, destination):
        self.id = vehicle_id
        self.position = position
        self.velocity = velocity
        self.destination = destination
        self.state = VehicleState.APPROACHING

    def negotiate_intersection(self, other_vehicles):
        # Use intersection protocol to coordinate
        my_plan = self.calculate_optimal_route()
        self.send_negociation_offer(my_plan, other_vehicles)
```

**2. Intersection Manager Agent**

```python
class IntersectionManager:
    def __init__(self, intersection_id, location):
        self.id = intersection_id
        self.location = location
        self.vehicles = {}  # vehicle_id -> vehicle_state
        self.conflict_graph = ConflictGraph()

    def coordinate_vehicles(self):
        # Detect potential conflicts
        conflicts = self.conflict_graph.find_conflicts(self.vehicles)

        # Resolve conflicts using priority or auction
        if conflicts:
            self.resolve_conflicts(conflicts)

    def allocate_right_of_way(self, vehicle_requests):
        # Implement traffic light or all-way stop logic
        return self.traffic_algorithm.allocate(vehicle_requests)
```

**Coordination Protocols:**

**1. Reservation-Based Protocol**

```python
class ReservationProtocol:
    def __init__(self):
        self.reservation_table = {}  # time_slot -> vehicle_id

    def request_reservation(self, vehicle_id, desired_time):
        if self.is_slot_available(desired_time):
            self.reservation_table[desired_time] = vehicle_id
            return True
        return False
```

**2. Auction-Based Protocol**

```python
class AuctionProtocol:
    def conduct_auction(self, conflict_vehicles):
        bids = {}
        for vehicle in conflict_vehicles:
            bids[vehicle.id] = self.calculate_bid(vehicle)

        winner = max(bids, key=bids.get)
        return winner
```

**Key Algorithms:**

**1. Conflict Detection**

```python
def detect_conflicts(vehicles, intersection_geometry):
    conflicts = []

    for i, v1 in enumerate(vehicles):
        for j, v2 in enumerate(vehicles[i+1:], i+1):
            if will_collide(v1, v2, intersection_geometry):
                conflicts.append((v1, v2))

    return conflicts
```

**2. Trajectory Planning**

```python
def plan_safe_trajectory(vehicle, obstacles, intersection_geometry):
    # Use motion planning algorithm (A*, RRT, etc.)
    constraints = [
        speed_limit,
        acceleration_limit,
        collision_avoidance,
        comfort_constraints
    ]

    return motion_planner.plan(vehicle, obstacles, constraints)
```

**Safety Mechanisms:**

1. **Redundancy**: Multiple sensors and processing units
2. **Communication Security**: Encrypted inter-vehicle communication
3. **Fail-Safe Modes**: Manual override and emergency braking
4. **Validation**: Trajectory verification before execution

### Q3: Design a system for coordinating multiple cleaning robots in a large building.

**Answer:**

**System Components:**

**1. Central Coordinator Agent**

```python
class BuildingCoordinator:
    def __init__(self, building_map, cleaning_schedule):
        self.building_map = building_map
        self.cleaning_schedule = cleaning_schedule
        self.robots = {}  # robot_id -> robot_state
        self.assigned_areas = {}  # area_id -> assigned_robot

    def coordinate_cleaning(self):
        # Update robot positions and battery levels
        self.update_robot_status()

        # Identify areas needing cleaning
        dirty_areas = self.identify_dirty_areas()

        # Assign robots to areas
        assignments = self.optimize_assignments(dirty_areas)

        # Send assignments to robots
        for robot_id, assignment in assignments.items():
            self.send_assignment(robot_id, assignment)
```

**2. Robot Agents**

```python
class CleaningRobotAgent:
    def __init__(self, robot_id, initial_position):
        self.id = robot_id
        self.position = initial_position
        self.battery_level = 100
        self.capabilities = ['vacuum', 'mop', 'sanitize']
        self.current_task = None

    def handle_assignment(self, assignment):
        self.current_task = assignment

        # Plan path to cleaning area
        path = self.plan_path(assignment.area)

        # Execute cleaning task
        while not self.is_area_clean(assignment.area):
            self.execute_cleaning_step()

        # Report completion
        self.send_completion_report()
```

**Coordination Strategies:**

**1. Area Partitioning**

```python
def partition_building(building_map, num_robots):
    areas = divide_building_into_regions(building_map, num_robots)

    # Consider factors:
    # - Area size and complexity
    # - Cleaning requirements
    # - Robot capabilities
    # - Battery constraints

    optimal_assignment = optimize_area_assignment(areas, robots)
    return optimal_assignment
```

**2. Dynamic Task Allocation**

```python
def dynamic_allocation(dirty_areas, available_robots):
    assignments = []

    for area in dirty_areas:
        # Find best robot for this area
        best_robot = find_best_robot(area, available_robots)

        if best_robot:
            assignments.append((area, best_robot))
            available_robots.remove(best_robot)

    return assignments
```

**Communication Protocols:**

**1. Position Sharing**

```python
class PositionSharingProtocol:
    def broadcast_position(self, robot_id, position, timestamp):
        message = {
            'type': 'position_update',
            'robot_id': robot_id,
            'position': position,
            'timestamp': timestamp
        }

        self.broadcast_to_all_robots(message)
```

**2. Conflict Resolution**

```python
class ConflictResolver:
    def resolve_area_conflict(self, robot1, robot2, area):
        # Assign area based on:
        # 1. Distance to area
        # 2. Battery level
        # 3. Current task priority
        # 4. Cleaning efficiency

        score1 = self.calculate_assignment_score(robot1, area)
        score2 = self.calculate_assignment_score(robot2, area)

        if score1 > score2:
            return robot1
        else:
            return robot2
```

**Optimization Features:**

1. **Energy Management**: Route optimization for charging
2. **Traffic Avoidance**: Prevent robot collisions
3. **Adaptive Scheduling**: Adjust based on real-time conditions
4. **Maintenance Scheduling**: Plan for robot maintenance
5. **Performance Metrics**: Track cleaning efficiency

---

## Coding Challenges {#coding-challenges}

### Challenge 1: Implement a Simple Reflux Agent

**Problem:**
Implement a simple reflex agent that reacts to environmental stimuli based on predefined rules.

**Solution:**

```python
import time
import random
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class AgentState(Enum):
    IDLE = "idle"
    CLEANING = "cleaning"
    CHARGING = "charging"
    ERROR = "error"

@dataclass
class Perception:
    sensor_data: Dict[str, Any]
    timestamp: float
    confidence: float = 1.0

@dataclass
class Action:
    action_type: str
    parameters: Dict[str, Any]
    priority: int = 1

class SimpleReflexAgent:
    def __init__(self, agent_id: str):
        self.id = agent_id
        self.state = AgentState.IDLE
        self.position = [0, 0]
        self.battery_level = 100
        self.cleaned_areas = set()

    def perceive(self, environment: Dict[str, Any]) -> Perception:
        """Process environmental inputs"""
        sensor_data = {
            'position': self.position,
            'battery_level': self.battery_level,
            'dirt_detected': environment.get('dirt_detected', False),
            'obstacles': environment.get('obstacles', []),
            'charging_station': environment.get('charging_station', [10, 10])
        }

        return Perception(
            sensor_data=sensor_data,
            timestamp=time.time(),
            confidence=self._calculate_confidence(sensor_data)
        )

    def think(self, perception: Perception) -> Action:
        """Decide on action based on perception"""
        data = perception.sensor_data

        # Rule 1: Return to charging if battery is low
        if data['battery_level'] < 20:
            return Action(
                action_type="return_to_charging",
                parameters={'destination': data['charging_station']},
                priority=10
            )

        # Rule 2: Clean if dirt is detected
        if data['dirt_detected']:
            return Action(
                action_type="clean",
                parameters={'cleaning_method': 'vacuum'},
                priority=8
            )

        # Rule 3: Avoid obstacles
        if data['obstacles']:
            return Action(
                action_type="navigate_around",
                parameters={'obstacles': data['obstacles']},
                priority=7
            )

        # Rule 4: Default action - explore
        return Action(
            action_type="explore",
            parameters={'exploration_radius': 5},
            priority=1
        )

    def act(self, action: Action) -> Dict[str, Any]:
        """Execute the decided action"""
        start_time = time.time()

        try:
            if action.action_type == "return_to_charging":
                result = self._action_return_to_charging(action.parameters)
            elif action.action_type == "clean":
                result = self._action_clean(action.parameters)
            elif action.action_type == "navigate_around":
                result = self._action_navigate_around(action.parameters)
            elif action.action_type == "explore":
                result = self._action_explore(action.parameters)
            else:
                result = {'success': False, 'error': f'Unknown action: {action.action_type}'}

            # Update state based on action result
            if result.get('success', False):
                self.state = AgentState.IDLE

            execution_time = time.time() - start_time
            result['execution_time'] = execution_time

            return result

        except Exception as e:
            self.state = AgentState.ERROR
            return {'success': False, 'error': str(e), 'action': action}

    def _action_return_to_charging(self, params: Dict) -> Dict[str, Any]:
        """Navigate to charging station"""
        destination = params.get('destination')

        # Simulate navigation
        time.sleep(0.2)
        self.position = destination[:2]  # Update position
        self.battery_level = 100  # Recharge

        return {
            'success': True,
            'action': 'return_to_charging',
            'destination': destination,
            'message': 'Robot returned to charging station'
        }

    def _action_clean(self, params: Dict) -> Dict[str, Any]:
        """Clean the current area"""
        cleaning_method = params.get('cleaning_method', 'vacuum')

        # Simulate cleaning
        time.sleep(0.1)

        # Mark area as cleaned
        self.cleaned_areas.add(tuple(self.position))

        return {
            'success': True,
            'action': 'clean',
            'method': cleaning_method,
            'cleaned_position': self.position[:2],
            'message': f'Area cleaned using {cleaning_method}'
        }

    def _action_navigate_around(self, params: Dict) -> Dict[str, Any]:
        """Navigate around obstacles"""
        obstacles = params.get('obstacles', [])

        # Simulate obstacle avoidance
        time.sleep(0.05)

        # Simple obstacle avoidance - move to safe position
        self.position[0] += 1  # Move right

        return {
            'success': True,
            'action': 'navigate_around',
            'obstacles_count': len(obstacles),
            'message': f'Navigated around {len(obstacles)} obstacles'
        }

    def _action_explore(self, params: Dict) -> Dict[str, Any]:
        """Explore the environment"""
        radius = params.get('exploration_radius', 5)

        # Simulate exploration - random movement
        time.sleep(0.05)

        # Random movement within radius
        delta_x = random.uniform(-radius/2, radius/2)
        delta_y = random.uniform(-radius/2, radius/2)

        self.position[0] += delta_x
        self.position[1] += delta_y

        return {
            'success': True,
            'action': 'explore',
            'movement': [delta_x, delta_y],
            'new_position': self.position[:2],
            'message': f'Explored area within radius {radius}'
        }

    def _calculate_confidence(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate confidence in sensor readings"""
        confidence = 0.9  # Base confidence

        # Adjust based on battery level
        battery = sensor_data.get('battery_level', 100)
        if battery < 20:
            confidence *= 0.7  # Low battery reduces confidence
        elif battery < 50:
            confidence *= 0.9

        return confidence

# Test the agent
def test_simple_reflex_agent():
    """Test the simple reflex agent"""
    agent = SimpleReflexAgent("robot_001")

    # Test scenarios
    test_scenarios = [
        {
            'name': 'Low Battery',
            'environment': {
                'dirt_detected': False,
                'battery_level': 15,
                'obstacles': [],
                'charging_station': [10, 10]
            },
            'expected_action': 'return_to_charging'
        },
        {
            'name': 'Dirt Detected',
            'environment': {
                'dirt_detected': True,
                'battery_level': 80,
                'obstacles': [],
                'charging_station': [10, 10]
            },
            'expected_action': 'clean'
        },
        {
            'name': 'Obstacles Present',
            'environment': {
                'dirt_detected': False,
                'battery_level': 90,
                'obstacles': [[1, 1], [2, 2]],
                'charging_station': [10, 10]
            },
            'expected_action': 'navigate_around'
        },
        {
            'name': 'Normal Operation',
            'environment': {
                'dirt_detected': False,
                'battery_level': 85,
                'obstacles': [],
                'charging_station': [10, 10]
            },
            'expected_action': 'explore'
        }
    ]

    print("Testing Simple Reflex Agent:")
    print("=" * 50)

    for scenario in test_scenarios:
        print(f"\nTest: {scenario['name']}")

        # Run agent cycle
        perception = agent.perceive(scenario['environment'])
        action = agent.think(perception)
        result = agent.act(action)

        print(f"Environment: {scenario['environment']}")
        print(f"Perception: {perception.sensor_data}")
        print(f"Action: {action.action_type}")
        print(f"Result: {result['message']}")
        print(f"Agent State: {agent.state.value}")
        print(f"Battery: {agent.battery_level}%")

        # Verify expected behavior
        if action.action_type == scenario['expected_action']:
            print("✓ PASS")
        else:
            print(f"✗ FAIL - Expected {scenario['expected_action']}, got {action.action_type}")

    print("\n" + "=" * 50)
    print(f"Final Agent Status:")
    print(f"ID: {agent.id}")
    print(f"State: {agent.state.value}")
    print(f"Position: {agent.position}")
    print(f"Battery: {agent.battery_level}%")
    print(f"Cleaned Areas: {len(agent.cleaned_areas)}")

if __name__ == "__main__":
    test_simple_reflex_agent()
```

### Challenge 2: Implement a Multi-Agent Communication System

**Problem:**
Create a message bus system that enables communication between multiple agents.

**Solution:**

```python
import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AgentMessage:
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    content: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = None
    expires_at: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class MessageBus:
    """Centralized message bus for agent communication"""

    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.subscribers = {}  # agent_id -> {message_types -> callback}
        self.message_queue = asyncio.Queue()
        self.message_history = []
        self.delivery_stats = {
            'total_messages': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'average_delivery_time': 0.0
        }

    async def subscribe(self, agent_id: str, message_types: Set[MessageType],
                       callback: Callable[[AgentMessage], None]):
        """Subscribe agent to message types"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = {}

        for msg_type in message_types:
            self.subscribers[agent_id][msg_type] = callback

    async def publish(self, message: AgentMessage) -> bool:
        """Publish message to message bus"""
        try:
            self.message_queue.put_nowait(message)
            self.delivery_stats['total_messages'] += 1

            # Process message immediately
            delivery_time = await self._deliver_message(message)

            if delivery_time:
                self.delivery_stats['successful_deliveries'] += 1

                # Update average delivery time
                n = self.delivery_stats['successful_deliveries']
                current_avg = self.delivery_stats['average_delivery_time']
                self.delivery_stats['average_delivery_time'] = (
                    (current_avg * (n - 1) + delivery_time) / n
                )
            else:
                self.delivery_stats['failed_deliveries'] += 1

            # Add to history
            self.message_history.append({
                'message': message,
                'delivery_time': delivery_time,
                'timestamp': time.time()
            })

            # Limit history size
            if len(self.message_history) > 100:
                self.message_history = self.message_history[-50:]

            return True

        except asyncio.QueueFull:
            return False

    async def send_direct(self, sender_id: str, receiver_id: str,
                         message_type: MessageType, content: Dict[str, Any]) -> str:
        """Send direct message to specific agent"""
        message_id = str(uuid.uuid4())

        message = AgentMessage(
            message_id=message_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content
        )

        success = await self.publish(message)
        return message_id if success else None

    async def broadcast(self, sender_id: str, message_type: MessageType,
                       content: Dict[str, Any]) -> str:
        """Broadcast message to all subscribers"""
        message_id = str(uuid.uuid4())

        message = AgentMessage(
            message_id=message_id,
            sender_id=sender_id,
            receiver_id=None,  # Broadcast
            message_type=message_type,
            content=content
        )

        success = await self.publish(message)
        return message_id if success else None

    async def _deliver_message(self, message: AgentMessage) -> Optional[float]:
        """Deliver message to intended recipients"""
        start_time = time.time()

        try:
            if message.receiver_id:
                # Direct message
                return await self._deliver_direct_message(message)
            else:
                # Broadcast message
                return await self._deliver_broadcast_message(message)
        except Exception as e:
            print(f"Delivery error: {e}")
            return None

    async def _deliver_direct_message(self, message: AgentMessage) -> Optional[float]:
        """Deliver direct message"""
        receiver_id = message.receiver_id

        if receiver_id not in self.subscribers:
            return None

        # Check if receiver is subscribed to this message type
        if message.message_type not in self.subscribers[receiver_id]:
            return None

        try:
            callback = self.subscribers[receiver_id][message.message_type]
            await callback(message)
            return time.time() - (message.timestamp + start_time)
        except Exception:
            return None

    async def _deliver_broadcast_message(self, message: AgentMessage) -> float:
        """Deliver broadcast message"""
        delivery_times = []

        for receiver_id, subscriptions in self.subscribers.items():
            if message.message_type in subscriptions:
                try:
                    callback = subscriptions[message.message_type]
                    await callback(message)
                    delivery_times.append(time.time() - message.timestamp)
                except Exception:
                    pass

        if delivery_times:
            return sum(delivery_times) / len(delivery_times)
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        return {
            **self.delivery_stats,
            'subscribers': len(self.subscribers),
            'queue_size': self.message_queue.qsize(),
            'message_history_size': len(self.message_history)
        }

class CommunicatingAgent:
    """Agent with communication capabilities"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.id = agent_id
        self.message_bus = message_bus
        self.received_messages = []
        self.message_handlers = {}

        # Default message handler
        self.message_handlers[MessageType.REQUEST] = self._handle_request
        self.message_handlers[MessageType.RESPONSE] = self._handle_response
        self.message_handlers[MessageType.BROADCAST] = self._handle_broadcast
        self.message_handlers[MessageType.NOTIFICATION] = self._handle_notification

    async def register(self, message_types: Set[MessageType]):
        """Register for communication"""
        await self.message_bus.subscribe(self.id, message_types, self._handle_message)

    async def send_request(self, receiver_id: str, action: str, parameters: Dict[str, Any]):
        """Send request to another agent"""
        content = {
            'action': action,
            'parameters': parameters,
            'conversation_id': str(uuid.uuid4())
        }

        return await self.message_bus.send_direct(
            self.id, receiver_id, MessageType.REQUEST, content
        )

    async def send_response(self, receiver_id: str, conversation_id: str, result: Dict[str, Any]):
        """Send response to request"""
        content = {
            'conversation_id': conversation_id,
            'result': result
        }

        await self.message_bus.send_direct(
            self.id, receiver_id, MessageType.RESPONSE, content
        )

    async def broadcast(self, event: str, data: Dict[str, Any]):
        """Broadcast message to all agents"""
        content = {
            'event': event,
            'data': data
        }

        await self.message_bus.broadcast(self.id, MessageType.BROADCAST, content)

    async def _handle_message(self, message: AgentMessage):
        """Handle incoming message"""
        self.received_messages.append(message)

        if message.message_type in self.message_handlers:
            await self.message_handlers[message.message_type](message)

    async def _handle_request(self, message: AgentMessage):
        """Handle incoming request"""
        content = message.content
        action = content.get('action')
        parameters = content.get('parameters', {})
        conversation_id = content.get('conversation_id')

        # Process request (simplified)
        result = {
            'success': True,
            'action': action,
            'processed_by': self.id,
            'data': f"Processed {action} with parameters {parameters}"
        }

        # Send response
        await self.send_response(message.sender_id, conversation_id, result)

    async def _handle_response(self, message: AgentMessage):
        """Handle incoming response"""
        content = message.content
        result = content.get('result')
        print(f"Agent {self.id} received response: {result}")

    async def _handle_broadcast(self, message: AgentMessage):
        """Handle incoming broadcast"""
        content = message.content
        event = content.get('event')
        data = content.get('data')
        print(f"Agent {self.id} received broadcast: {event} - {data}")

    async def _handle_notification(self, message: AgentMessage):
        """Handle incoming notification"""
        print(f"Agent {self.id} received notification")

# Test the communication system
async def test_multi_agent_communication():
    """Test the multi-agent communication system"""

    # Create message bus
    message_bus = MessageBus()

    # Create agents
    agent1 = CommunicatingAgent("agent_001", message_bus)
    agent2 = CommunicatingAgent("agent_002", message_bus)
    agent3 = CommunicatingAgent("agent_003", message_bus)

    # Register agents
    await agent1.register({MessageType.REQUEST, MessageType.RESPONSE, MessageType.BROADCAST})
    await agent2.register({MessageType.REQUEST, MessageType.RESPONSE, MessageType.BROADCAST})
    await agent3.register({MessageType.BROADCAST})

    print("Testing Multi-Agent Communication System:")
    print("=" * 50)

    # Test 1: Direct request-response
    print("\n1. Testing Direct Request-Response:")
    message_id = await agent1.send_request(
        "agent_002",
        "get_status",
        {"query": "current_state"}
    )
    print(f"Sent request with ID: {message_id}")

    # Wait for processing
    await asyncio.sleep(0.1)

    # Test 2: Broadcast
    print("\n2. Testing Broadcast:")
    await agent3.broadcast(
        "system_announcement",
        {"message": "Hello from agent_003", "timestamp": time.time()}
    )

    # Test 3: Multiple requests
    print("\n3. Testing Multiple Simultaneous Requests:")
    tasks = []
    for i in range(3):
        task = agent1.send_request(
            "agent_002",
            f"process_task_{i}",
            {"task_id": i, "data": f"data_{i}"}
        )
        tasks.append(task)

    message_ids = await asyncio.gather(*tasks)
    print(f"Sent {len(message_ids)} requests: {message_ids}")

    # Wait for processing
    await asyncio.sleep(0.2)

    # Print statistics
    print(f"\nMessage Bus Statistics:")
    stats = message_bus.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nAgent Message Counts:")
    print(f"  agent_001 received: {len(agent1.received_messages)} messages")
    print(f"  agent_002 received: {len(agent2.received_messages)} messages")
    print(f"  agent_003 received: {len(agent3.received_messages)} messages")

if __name__ == "__main__":
    asyncio.run(test_multi_agent_communication())
```

### Challenge 3: Implement Goal-Oriented Planning

**Problem:**
Create an agent that can plan and execute sequences of actions to achieve goals.

**Solution:**

```python
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import heapq

class GoalStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class Goal:
    goal_id: str
    description: str
    priority: float
    deadline: Optional[float] = None
    status: GoalStatus = GoalStatus.ACTIVE
    progress: float = 0.0
    sub_goals: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)
    estimated_duration: float = 1.0

@dataclass
class PlanStep:
    step_id: str
    action: str
    parameters: Dict[str, Any]
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    estimated_duration: float = 1.0
    resource_requirements: Dict[str, int] = field(default_factory=dict)

@dataclass
class Plan:
    plan_id: str
    goal_id: str
    steps: List[PlanStep]
    current_step_index: int = 0
    status: str = "active"
    total_duration: float = 0.0

class GoalOrientedAgent:
    """Agent that plans and executes goals"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.goals = {}  # goal_id -> Goal
        self.active_plans = {}  # goal_id -> Plan
        self.completed_plans = []
        self.resource_allocations = {}  # resource -> allocated_amount
        self.available_resources = {
            'time': 100.0,
            'energy': 100.0,
            'tools': ['cleaner', 'navigator'],
            'materials': ['soap', 'water']
        }
        self.goal_selection_strategy = "priority_deadline"

    def add_goal(self, goal: Goal):
        """Add a new goal"""
        self.goals[goal.goal_id] = goal
        print(f"Added goal: {goal.description} (Priority: {goal.priority})")

    def create_plan(self, goal_id: str) -> Optional[Plan]:
        """Create a plan to achieve a goal"""
        if goal_id not in self.goals:
            return None

        goal = self.goals[goal_id]

        # Check if dependencies are satisfied
        if not self._check_dependencies(goal):
            return None

        # Generate plan based on goal type
        if 'clean' in goal.description.lower():
            plan = self._create_cleaning_plan(goal)
        elif 'deliver' in goal.description.lower():
            plan = self._create_delivery_plan(goal)
        elif 'build' in goal.description.lower():
            plan = self._create_construction_plan(goal)
        else:
            plan = self._create_generic_plan(goal)

        if plan:
            self.active_plans[goal_id] = plan
            goal.status = GoalStatus.ACTIVE
            print(f"Created plan for goal '{goal.description}' with {len(plan.steps)} steps")

        return plan

    def execute_next_step(self, goal_id: str) -> Tuple[bool, str]:
        """Execute the next step in the plan"""
        if goal_id not in self.active_plans:
            return False, "No active plan for this goal"

        plan = self.active_plans[goal_id]

        if plan.current_step_index >= len(plan.steps):
            # Plan completed
            self._complete_plan(goal_id)
            return True, "Plan completed"

        current_step = plan.steps[plan.current_step_index]

        # Check if preconditions are met
        if not self._check_preconditions(current_step):
            return False, f"Preconditions not met for step {current_step.step_id}"

        # Check resource availability
        if not self._check_resources(current_step):
            return False, f"Insufficient resources for step {current_step.step_id}"

        # Execute the step
        success, message = self._execute_step(current_step)

        if success:
            # Update progress
            plan.current_step_index += 1
            progress = (plan.current_step_index / len(plan.steps)) * 100
            self.goals[goal_id].progress = progress

            # Apply effects
            self._apply_effects(current_step)

            # Consume resources
            self._consume_resources(current_step)

            print(f"Step {current_step.step_id} completed: {message}")
        else:
            print(f"Step {current_step.step_id} failed: {message}")

        return success, message

    def _create_cleaning_plan(self, goal: Goal) -> Plan:
        """Create plan for cleaning goal"""
        steps = [
            PlanStep(
                step_id="assess_area",
                action="assess_cleaning_requirements",
                parameters={'area': 'target_area'},
                preconditions=[],
                effects=['area_assessed'],
                estimated_duration=0.5
            ),
            PlanStep(
                step_id="gather_supplies",
                action="gather_cleaning_supplies",
                parameters={'supplies': ['soap', 'water']},
                preconditions=['area_assessed'],
                effects=['supplies_gathered'],
                estimated_duration=1.0,
                resource_requirements={'time': 1.0}
            ),
            PlanStep(
                step_id="clean_surface",
                action="clean_surface",
                parameters={'cleaning_method': 'scrub'},
                preconditions=['supplies_gathered'],
                effects=['surface_cleaned'],
                estimated_duration=3.0,
                resource_requirements={'time': 3.0, 'energy': 2.0}
            ),
            PlanStep(
                step_id="verify_cleanliness",
                action="verify_cleanliness",
                parameters={'cleaning_standard': 'high'},
                preconditions=['surface_cleaned'],
                effects=['cleanliness_verified'],
                estimated_duration=0.5
            )
        ]

        total_duration = sum(step.estimated_duration for step in steps)

        return Plan(
            plan_id=f"plan_{goal.goal_id}",
            goal_id=goal.goal_id,
            steps=steps,
            total_duration=total_duration
        )

    def _create_delivery_plan(self, goal: Goal) -> Plan:
        """Create plan for delivery goal"""
        steps = [
            PlanStep(
                step_id="navigate_to_pickup",
                action="navigate_to_location",
                parameters={'destination': 'pickup_location'},
                preconditions=[],
                effects=['at_pickup_location'],
                estimated_duration=2.0,
                resource_requirements={'time': 2.0, 'energy': 1.0}
            ),
            PlanStep(
                step_id="pickup_item",
                action="pickup_object",
                parameters={'object_type': 'package'},
                preconditions=['at_pickup_location'],
                effects=['item_picked_up'],
                estimated_duration=0.5,
                resource_requirements={'energy': 0.5}
            ),
            PlanStep(
                step_id="navigate_to_dropoff",
                action="navigate_to_location",
                parameters={'destination': 'dropoff_location'},
                preconditions=['item_picked_up'],
                effects=['at_dropoff_location'],
                estimated_duration=2.5,
                resource_requirements={'time': 2.5, 'energy': 1.5}
            ),
            PlanStep(
                step_id="deliver_item",
                action="deliver_object",
                parameters={'confirmation_required': True},
                preconditions=['at_dropoff_location'],
                effects=['item_delivered'],
                estimated_duration=0.5
            )
        ]

        total_duration = sum(step.estimated_duration for step in steps)

        return Plan(
            plan_id=f"plan_{goal.goal_id}",
            goal_id=goal.goal_id,
            steps=steps,
            total_duration=total_duration
        )

    def _create_construction_plan(self, goal: Goal) -> Plan:
        """Create plan for construction goal"""
        steps = [
            PlanStep(
                step_id="gather_materials",
                action="gather_materials",
                parameters={'materials': ['wood', 'nails', 'tools']},
                preconditions=[],
                effects=['materials_gathered'],
                estimated_duration=1.5,
                resource_requirements={'time': 1.5}
            ),
            PlanStep(
                step_id="prepare_workspace",
                action="prepare_workspace",
                parameters={'workspace': 'construction_area'},
                preconditions=['materials_gathered'],
                effects=['workspace_prepared'],
                estimated_duration=1.0,
                resource_requirements={'time': 1.0}
            ),
            PlanStep(
                step_id="assemble_structure",
                action="assemble_structure",
                parameters={'assembly_method': 'step_by_step'},
                preconditions=['workspace_prepared'],
                effects=['structure_assembled'],
                estimated_duration=5.0,
                resource_requirements={'time': 5.0, 'energy': 3.0}
            ),
            PlanStep(
                step_id="verify_construction",
                action="verify_construction",
                parameters={'quality_standard': 'structural'},
                preconditions=['structure_assembled'],
                effects=['construction_verified'],
                estimated_duration=1.0
            )
        ]

        total_duration = sum(step.estimated_duration for step in steps)

        return Plan(
            plan_id=f"plan_{goal.goal_id}",
            goal_id=goal.goal_id,
            steps=steps,
            total_duration=total_duration
        )

    def _create_generic_plan(self, goal: Goal) -> Plan:
        """Create generic plan for unknown goal type"""
        steps = [
            PlanStep(
                step_id="analyze_goal",
                action="analyze_objective",
                parameters={'goal_description': goal.description},
                preconditions=[],
                effects=['goal_analyzed'],
                estimated_duration=1.0
            ),
            PlanStep(
                step_id="execute_task",
                action="perform_task",
                parameters={'task_type': 'generic'},
                preconditions=['goal_analyzed'],
                effects=['task_performed'],
                estimated_duration=2.0,
                resource_requirements={'time': 2.0, 'energy': 1.0}
            ),
            PlanStep(
                step_id="verify_completion",
                action="verify_completion",
                parameters={'completion_criteria': 'basic'},
                preconditions=['task_performed'],
                effects=['completion_verified'],
                estimated_duration=0.5
            )
        ]

        total_duration = sum(step.estimated_duration for step in steps)

        return Plan(
            plan_id=f"plan_{goal.goal_id}",
            goal_id=goal.goal_id,
            steps=steps,
            total_duration=total_duration
        )

    def _check_dependencies(self, goal: Goal) -> bool:
        """Check if goal dependencies are satisfied"""
        for dep_id in goal.sub_goals:
            if dep_id not in self.goals:
                continue
            if self.goals[dep_id].status != GoalStatus.COMPLETED:
                return False
        return True

    def _check_preconditions(self, step: PlanStep) -> bool:
        """Check if step preconditions are met"""
        for precondition in step.preconditions:
            if precondition not in self._achieved_effects:
                return False
        return True

    def _check_resources(self, step: PlanStep) -> bool:
        """Check if required resources are available"""
        for resource, amount in step.resource_requirements.items():
            available = self.available_resources.get(resource, 0)
            if available < amount:
                return False
        return True

    def _execute_step(self, step: PlanStep) -> Tuple[bool, str]:
        """Execute a plan step"""
        action = step.action
        parameters = step.parameters

        # Simulate step execution
        time.sleep(step.estimated_duration * 0.1)  # Simulate processing time

        # Simple success/failure logic based on action type
        success_rate = 0.9  # 90% success rate

        if action == "navigate_to_location":
            # Navigation usually succeeds
            return True, f"Successfully navigated to {parameters.get('destination', 'location')}"
        elif action == "clean_surface":
            # Cleaning usually succeeds if supplies are available
            return success_rate > 0.5, f"Cleaned surface using {parameters.get('cleaning_method', 'default')} method"
        elif action == "assemble_structure":
            # Construction can fail
            import random
            if random.random() < success_rate:
                return True, "Structure assembled successfully"
            else:
                return False, "Assembly failed - need to retry"
        else:
            return True, f"Executed {action} with parameters {parameters}"

    def _apply_effects(self, step: PlanStep):
        """Apply the effects of a completed step"""
        for effect in step.effects:
            self._achieved_effects.add(effect)

    def _consume_resources(self, step: PlanStep):
        """Consume resources for the step"""
        for resource, amount in step.resource_requirements.items():
            if resource in self.available_resources:
                self.available_resources[resource] = max(0,
                    self.available_resources[resource] - amount)

    def _complete_plan(self, goal_id: str):
        """Mark plan as completed"""
        if goal_id in self.active_plans:
            plan = self.active_plans[goal_id]
            plan.status = "completed"

            self.completed_plans.append(plan)
            del self.active_plans[goal_id]

            # Mark goal as completed
            if goal_id in self.goals:
                self.goals[goal_id].status = GoalStatus.COMPLETED
                self.goals[goal_id].progress = 100.0

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'agent_id': self.id,
            'goals': {
                goal_id: {
                    'description': goal.description,
                    'priority': goal.priority,
                    'status': goal.status.value,
                    'progress': goal.progress
                }
                for goal_id, goal in self.goals.items()
            },
            'active_plans': {
                goal_id: {
                    'plan_id': plan.plan_id,
                    'current_step': plan.current_step_index + 1,
                    'total_steps': len(plan.steps),
                    'progress': (plan.current_step_index / len(plan.steps)) * 100
                }
                for goal_id, plan in self.active_plans.items()
            },
            'available_resources': self.available_resources,
            'completed_plans': len(self.completed_plans)
        }

# Test the goal-oriented agent
def test_goal_oriented_agent():
    """Test the goal-oriented agent"""
    agent = GoalOrientedAgent("planner_001")

    # Initialize achieved effects set
    agent._achieved_effects = set()

    # Create goals
    goals = [
        Goal(
            goal_id="goal_001",
            description="Clean the kitchen floor",
            priority=8.0,
            deadline=time.time() + 3600  # 1 hour
        ),
        Goal(
            goal_id="goal_002",
            description="Deliver package to customer",
            priority=9.0,
            deadline=time.time() + 1800  # 30 minutes
        ),
        Goal(
            goal_id="goal_003",
            description="Build a small shelf",
            priority=6.0
        )
    ]

    # Add goals
    for goal in goals:
        agent.add_goal(goal)

    print("\nGoal-Oriented Agent Test:")
    print("=" * 50)

    # Execute plans
    for goal_id in agent.goals:
        print(f"\nProcessing goal: {agent.goals[goal_id].description}")

        # Create plan
        plan = agent.create_plan(goal_id)
        if not plan:
            print(f"Could not create plan for goal {goal_id}")
            continue

        # Execute plan steps
        while goal_id in agent.active_plans:
            success, message = agent.execute_next_step(goal_id)

            if not success:
                print(f"Step failed: {message}")
                # Could implement retry logic here
                break

            if success and message == "Plan completed":
                print(f"Goal completed: {message}")
                break

    # Print final status
    print(f"\nFinal Agent Status:")
    status = agent.get_status()
    for key, value in status.items():
        if key != 'goals' and key != 'active_plans':
            print(f"  {key}: {value}")

    print(f"\nGoals Summary:")
    for goal_id, goal_info in status['goals'].items():
        print(f"  {goal_info['description']}: {goal_info['status']} ({goal_info['progress']:.1f}%)")

if __name__ == "__main__":
    test_goal_oriented_agent()
```

---

This interview preparation guide provides comprehensive coverage of AI agent concepts, practical coding challenges, and real-world scenarios that students are likely to encounter in technical interviews. The content is designed to test both theoretical understanding and practical implementation skills in AI agent development.
