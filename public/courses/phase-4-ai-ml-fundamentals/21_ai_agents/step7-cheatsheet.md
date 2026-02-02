# AI Agents Cheatsheet

**Version:** 1.0 | **Date:** November 2025

## Quick Reference Guide for AI Agents

---

## 🏗️ Agent Architecture Patterns

### Basic Agent Components

```python
# Core agent structure
class Agent:
    def perceive(self, environment) -> Perception
    def think(self, perception) -> Action
    def act(self, action) -> Result
    def run_cycle(self, environment) -> CycleResult
```

### Agent States

```python
class AgentState(Enum):
    IDLE = "idle"           # Waiting for tasks
    BUSY = "busy"           # Currently working
    WAITING = "waiting"     # Blocked/paused
    ERROR = "error"         # Error state
    LEARNING = "learning"   # Adapting/learning
```

### Perception → Action Patterns

| Pattern          | Use Case                | Complexity | Example                  |
| ---------------- | ----------------------- | ---------- | ------------------------ |
| **Reactive**     | Simple stimuli-response | Low        | Robot avoiding obstacles |
| **Deliberative** | Planning required       | Medium     | Route planning           |
| **Hybrid**       | Real-time + planning    | High       | Autonomous vehicles      |
| **Learning**     | Adaptive behavior       | Very High  | Self-improving systems   |

---

## 🤝 Communication Patterns

### Message Types

```python
class MessageType(Enum):
    REQUEST = "request"         # Ask for something
    RESPONSE = "response"       # Reply to request
    BROADCAST = "broadcast"     # Announce to all
    NOTIFICATION = "notify"     # Inform about events
    HEARTBEAT = "heartbeat"     # Status updates
```

### Communication Patterns

| Pattern               | Description              | Use Case              | Example          |
| --------------------- | ------------------------ | --------------------- | ---------------- |
| **Direct**            | Point-to-point messaging | Task delegation       | Manager → Worker |
| **Publish-Subscribe** | Topic-based messaging    | Event notifications   | Stock updates    |
| **Broadcast**         | One-to-all messaging     | System announcements  | Emergency alerts |
| **Request-Response**  | Query-reply pattern      | Information retrieval | Database queries |

### Message Bus Commands

```python
# Subscribe to messages
await message_bus.subscribe(agent_id, {MessageType.REQUEST}, callback)

# Send direct message
message_id = await message_bus.send_direct(
    sender_id, receiver_id, MessageType.REQUEST, content
)

# Broadcast message
await message_bus.broadcast(sender_id, MessageType.BROADCAST, content)
```

---

## 🎯 Goal Management

### Goal Structure

```python
@dataclass
class Goal:
    goal_id: str
    description: str
    priority: float
    deadline: Optional[float]
    status: str = "active"
    progress: float = 0.0
    dependencies: List[str] = field(default_factory=list)
```

### Goal Selection Strategies

| Strategy               | Description             | Formula                                   | When to Use          |
| ---------------------- | ----------------------- | ----------------------------------------- | -------------------- |
| **Priority-Deadline**  | Urgency + importance    | `score = priority × (1 + urgency_factor)` | Time-critical tasks  |
| **Progress-Based**     | Complete started tasks  | `score = 1 / (progress + ε)`              | Iterative projects   |
| **Resource-Efficient** | Minimize resource usage | `score = value / resources`               | Resource-constrained |
| **Learning-Value**     | Maximize knowledge gain | `score = learning_potential`              | Research tasks       |

### Plan Execution

```python
# Plan structure
plan = {
    'goal_id': 'goal_001',
    'steps': [
        {'action': 'assess', 'duration': 0.5},
        {'action': 'execute', 'duration': 2.0},
        {'action': 'verify', 'duration': 0.5}
    ],
    'current_step': 0,
    'status': 'in_progress'
}
```

---

## 🧠 Learning Algorithms

### Q-Learning Agent

```python
class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = lr
        self.discount_factor = gamma
        self.epsilon = epsilon

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state, action]
        target_q = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (target_q - current_q)
```

### Policy Gradient Agent

```python
class PolicyGradientAgent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.policy_network = self._build_network()
        self.learning_rate = lr

    def select_action(self, state, training=True):
        logits = self.forward(state, self.policy_network)
        if training:
            logits += np.random.normal(0, 0.1, size=logits.shape)  # Exploration
        return logits

    def update(self, states, actions, rewards):
        returns = self._compute_returns(rewards)
        policy_loss = -np.mean(self._compute_log_probs(states, actions) * returns)
        self._update_network(policy_loss)
```

---

## ⚡ Modern Frameworks

### LangChain Agents

```python
from langchain.agents import create_openai_functions_agent
from langchain.tools import Tool

# Setup
llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [
    Tool(name="calculator", description="Math operations", func=calculate),
    Tool(name="web_search", description="Search web", func=search_web)
]

# Create agent
prompt = PromptTemplate.from_template("You are a helpful agent. Use tools when needed.")
agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run
result = executor.invoke({"input": "What's 15% of 200?"})
```

### AutoGen Agents

```python
import autogen

# Create agents
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4"}
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10
)

# Start conversation
user_proxy.initiate_chat(assistant, message="Hello, how can you help me?")
```

### CrewAI Agents

```python
from crewai import Agent, Task, Crew

# Create agents
researcher = Agent(
    role="Research Analyst",
    goal="Conduct thorough research",
    backstory="Expert in data analysis",
    verbose=True
)

# Create tasks
research_task = Task(
    description="Research market trends",
    agent=researcher,
    expected_output="Comprehensive market analysis"
)

# Create crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=2
)

result = crew.kickoff()
```

---

## 🔧 Common Agent Patterns

### State Machine Pattern

```python
class StateMachineAgent:
    def __init__(self):
        self.states = {
            'idle': self._idle_state,
            'working': self._working_state,
            'waiting': self._waiting_state,
            'error': self._error_state
        }
        self.current_state = 'idle'

    def run(self):
        while self.current_state != 'stop':
            state_function = self.states[self.current_state]
            self.current_state = state_function()
```

### Observer Pattern

```python
class ObservableAgent:
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, event):
        for observer in self.observers:
            observer.update(event)

    def change_state(self, new_state):
        self.state = new_state
        self.notify_observers(f"State changed to {new_state}")
```

### Strategy Pattern

```python
class StrategyAgent:
    def __init__(self, strategy):
        self.strategy = strategy

    def set_strategy(self, strategy):
        self.strategy = strategy

    def execute_task(self, task):
        return self.strategy.execute(task)

# Different strategies
class AggressiveStrategy:
    def execute(self, task):
        return f"Aggressively executing: {task}"

class ConservativeStrategy:
    def execute(self, task):
        return f"Carefully executing: {task}"
```

---

## 🛡️ Safety & Ethics

### Safety Levels

```python
class SafetyLevel(Enum):
    LOW = "low"           # Basic precautions
    MEDIUM = "medium"     # Standard safety measures
    HIGH = "high"         # Enhanced safety protocols
    CRITICAL = "critical" # Maximum safety required
```

### Ethical Frameworks

| Framework         | Principle              | Implementation            | Use Case               |
| ----------------- | ---------------------- | ------------------------- | ---------------------- |
| **Utilitarian**   | Maximize overall good  | Cost-benefit analysis     | Resource allocation    |
| **Deontological** | Follow moral rules     | Rule-based constraints    | Legal compliance       |
| **Virtue Ethics** | Embody good traits     | Character-based decisions | Personal AI assistants |
| **Care Ethics**   | Maintain relationships | Relationship-focused      | Social robots          |

### Bias Detection Checklist

```python
bias_checks = {
    'demographic_bias': check_demographic_distribution,
    'confirmation_bias': check_contradicting_evidence,
    'recency_bias': check_historical_weight,
    'availability_bias': check_representative_samples,
    'affinity_bias': check_similar_group_preference
}
```

---

## 📊 Performance Metrics

### Agent Metrics

```python
class AgentMetrics:
    def __init__(self):
        self.metrics = {
            'tasks_completed': 0,
            'success_rate': 0.0,
            'average_response_time': 0.0,
            'energy_consumed': 0.0,
            'learning_progress': 0.0,
            'collaboration_score': 0.0
        }

    def update(self, task_result, response_time, energy_cost):
        self.metrics['tasks_completed'] += 1
        self.metrics['success_rate'] = self._calculate_success_rate()
        self.metrics['average_response_time'] = self._calculate_avg_response_time()
        self.metrics['energy_consumed'] += energy_cost
```

### System Metrics

| Metric           | Description             | Calculation                     | Target     |
| ---------------- | ----------------------- | ------------------------------- | ---------- |
| **Throughput**   | Tasks per time unit     | `completed_tasks / time_period` | >100/day   |
| **Latency**      | Response time           | `task_completion_time`          | <2 seconds |
| **Availability** | Uptime percentage       | `up_time / total_time`          | >99.5%     |
| **Accuracy**     | Correct task completion | `correct_tasks / total_tasks`   | >95%       |

---

## 🔍 Troubleshooting Guide

### Common Issues & Solutions

| Issue                    | Symptoms                  | Cause                       | Solution                             |
| ------------------------ | ------------------------- | --------------------------- | ------------------------------------ |
| **Message Loss**         | Missing communications    | Queue overflow              | Increase queue size, implement retry |
| **Deadlocks**            | Agents stuck waiting      | Circular dependencies       | Add timeout mechanisms               |
| **Performance Issues**   | Slow responses            | Resource contention         | Implement load balancing             |
| **Learning Instability** | Oscillating behavior      | High learning rate          | Reduce learning rate                 |
| **Communication Loops**  | Infinite message exchange | Lack of message ID tracking | Add message deduplication            |

### Debug Commands

```python
# Enable detailed logging
logging.getLogger().setLevel(logging.DEBUG)

# Monitor agent state
agent_status = agent.get_status()
print(f"State: {agent_status['state']}, Energy: {agent_status['energy']}")

# Check message queue
message_bus_stats = message_bus.get_message_stats()
print(f"Queue size: {message_bus_stats['queue_size']}")

# Analyze learning progress
if hasattr(agent, 'learning_metrics'):
    print(f"Learning progress: {agent.learning_metrics}")
```

### Performance Optimization

```python
# Cache frequently accessed data
@cache
def expensive_computation(data):
    return complex_calculation(data)

# Batch similar operations
def batch_process(requests):
    batched = group_by_type(requests)
    results = {}
    for request_type, batch in batched.items():
        results[request_type] = process_batch(batch)
    return results

# Use async operations for I/O
async def handle_concurrent_requests(requests):
    tasks = [process_request(req) for req in requests]
    return await asyncio.gather(*tasks)
```

---

## 🚀 Deployment Checklist

### Pre-Deployment

- [ ] All agents have unique IDs
- [ ] Communication protocols tested
- [ ] Safety constraints validated
- [ ] Performance benchmarks met
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Monitoring dashboards setup

### Production Setup

```python
# Environment configuration
production_config = {
    'max_agents': 100,
    'message_queue_size': 10000,
    'timeout': 30,
    'retry_attempts': 3,
    'log_level': 'INFO',
    'safety_monitoring': True
}

# Resource limits
resource_limits = {
    'max_memory_mb': 2048,
    'max_cpu_percent': 80,
    'max_concurrent_tasks': 50
}

# Monitoring setup
monitoring_config = {
    'metrics_interval': 60,
    'alert_thresholds': {
        'error_rate': 0.05,
        'response_time': 5000,  # milliseconds
        'memory_usage': 0.8     # 80%
    }
}
```

### Monitoring Dashboard

```python
# Key metrics to monitor
dashboard_metrics = [
    'active_agents',
    'message_throughput',
    'average_response_time',
    'error_rate',
    'task_success_rate',
    'resource_utilization',
    'learning_progress'
]

# Alert conditions
alert_conditions = {
    'high_error_rate': error_rate > 0.05,
    'slow_responses': avg_response_time > 5.0,
    'resource_exhaustion': memory_usage > 0.9,
    'communication_failure': message_failure_rate > 0.1
}
```

---

## 📚 Framework Comparison

| Feature           | LangChain  | AutoGen          | CrewAI          | Custom              |
| ----------------- | ---------- | ---------------- | --------------- | ------------------- |
| **Complexity**    | Medium     | High             | Low             | Variable            |
| **Setup Time**    | 15 min     | 30 min           | 10 min          | 60+ min             |
| **Customization** | High       | Very High        | Medium          | Maximum             |
| **Documentation** | Excellent  | Good             | Good            | N/A                 |
| **Community**     | Large      | Growing          | Small           | N/A                 |
| **Best For**      | LLM agents | Multi-agent chat | Task automation | Unique requirements |

---

## 🎛️ Configuration Templates

### Development Environment

```yaml
# dev_config.yaml
development:
  log_level: DEBUG
  max_agents: 10
  timeout: 60
  safety_monitoring: false
  learning_rate: 0.01
  exploration_rate: 0.2
```

### Production Environment

```yaml
# prod_config.yaml
production:
  log_level: INFO
  max_agents: 1000
  timeout: 30
  safety_monitoring: true
  learning_rate: 0.001
  exploration_rate: 0.05
  monitoring_enabled: true
  alerts_enabled: true
```

### Testing Environment

```yaml
# test_config.yaml
testing:
  log_level: WARNING
  max_agents: 5
  timeout: 10
  safety_monitoring: true
  mock_external_services: true
  deterministic_execution: true
```

---

## 🔗 Useful Resources

### Documentation

- [LangChain Documentation](https://python.langchain.com/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [FIPA Specifications](http://www.fipa.org/)

### Libraries & Tools

```bash
# Core dependencies
pip install numpy pandas scikit-learn

# Framework-specific
pip install langchain langchain-openai
pip install autogen
pip install crewai

# Utilities
pip install matplotlib seaborn  # Visualization
pip install networkx             # Graph operations
pip install asyncio aiohttp      # Async communication
pip install pytest pytest-asyncio # Testing
```

### Best Practices Summary

1. **Start Simple**: Begin with basic agents, add complexity gradually
2. **Test Thoroughly**: Use unit tests for individual components
3. **Monitor Performance**: Implement metrics from day one
4. **Design for Failure**: Add error handling and recovery mechanisms
5. **Document Decisions**: Record why specific approaches were chosen
6. **Security First**: Implement safety constraints early
7. **Scale Gradually**: Test with increasing agent counts
8. **Maintain State**: Keep track of agent and system state
9. **Optimize Communication**: Minimize message overhead
10. **Plan for Maintenance**: Design for easy debugging and updates

---

## 🆘 Quick Reference Commands

### Agent Lifecycle

```python
# Create agent
agent = SimpleAgent("agent_001")

# Run single cycle
result = agent.run_cycle(environment)

# Get status
status = agent.get_status()

# Add goal
goal = Goal("goal_001", "Complete task", priority=8.0)
agent.add_goal(goal)
```

### Communication

```python
# Subscribe to messages
await message_bus.subscribe(agent_id, {MessageType.REQUEST}, callback)

# Send message
await agent.send_request(receiver_id, "get_status", {})

# Broadcast
await agent.broadcast(MessageType.BROADCAST, {"event": "announcement"})
```

### Learning

```python
# Update Q-values
agent.update(state, action, reward, next_state, done)

# Save model
agent.save_model("agent_model.pkl")

# Load model
agent.load_model("agent_model.pkl")
```

### Safety

```python
# Add safety constraint
constraint = SafetyConstraint("no_harm", "Do no harm", SafetyLevel.HIGH, check_function)
safety_monitor.add_constraint(constraint)

# Check safety
safety_result = safety_monitor.check_safety(action, context)
if not safety_result['safe']:
    print("Action blocked due to safety concerns")
```

---

This cheatsheet provides a comprehensive quick reference for AI agents, covering all essential concepts, patterns, frameworks, and best practices in a condensed format for easy lookup during development and troubleshooting.
