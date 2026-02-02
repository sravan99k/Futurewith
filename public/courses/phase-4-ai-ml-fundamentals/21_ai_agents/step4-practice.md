# AI Agents Practice

**Version:** 1.0 | **Date:** November 2025

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Basic Agent Implementation](#basic-agent-implementation)
3. [Agent Communication Systems](#agent-communication-systems)
4. [Multi-Agent Coordination](#multi-agent-coordination)
5. [Agent Learning and Adaptation](#agent-learning-adaptation)
6. [Modern Agent Frameworks](#modern-agent-frameworks)
7. [Advanced Agent Patterns](#advanced-agent-patterns)
8. [Agent Safety and Ethics](#agent-safety-ethics)
9. [Real-World Applications](#real-world-applications)
10. [Mini-Projects](#mini-projects)

---

## Environment Setup {#environment-setup}

### Required Dependencies

```bash
# Create virtual environment
python -m venv ai_agents_env
source ai_agents_env/bin/activate  # On Windows: ai_agents_env\Scripts\activate

# Install core dependencies
pip install numpy pandas matplotlib seaborn scikit-learn
pip install asyncio aiohttp
pip install langchain langchain-openai
pip install crewai
pip install autogen
pip install transformers torch
pip install networkx matplotlib
pip install pytest pytest-asyncio
```

### Project Structure

```
ai_agents_project/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── simple_agent.py
│   └── advanced_agent.py
├── communication/
│   ├── __init__.py
│   ├── message_bus.py
│   └── protocol_handlers.py
├── coordination/
│   ├── __init__.py
│   ├── coordinator.py
│   └── consensus.py
├── learning/
│   ├── __init__.py
│   ├── rl_agent.py
│   └── social_learning.py
├── frameworks/
│   ├── __init__.py
│   ├── langchain_agent.py
│   ├── autogen_agent.py
│   └── crewai_agent.py
├── safety/
│   ├── __init__.py
│   ├── safety_monitor.py
│   └── ethical_constraints.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── metrics.py
├── examples/
│   ├── basic_examples.py
│   └── advanced_examples.py
└── tests/
    ├── __init__.py
    ├── test_agents.py
    └── test_coordination.py
```

### Environment Configuration

```python
# config.py
import os
from dataclasses import dataclass

@dataclass
class Config:
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Agent Settings
    MAX_AGENTS: int = 10
    DEFAULT_TIMEOUT: int = 30
    MESSAGE_QUEUE_SIZE: int = 1000

    # Learning Parameters
    LEARNING_RATE: float = 0.01
    DISCOUNT_FACTOR: float = 0.95
    EXPLORATION_RATE: float = 0.1

    # Safety Settings
    SAFETY_THRESHOLD: float = 0.8
    ENABLE_SAFETY_MONITORING: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "ai_agents.log"

config = Config()
```

---

## Basic Agent Implementation {#basic-agent-implementation}

### 1. Simple Reflex Agent

```python
# agents/simple_agent.py
import time
import random
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class AgentState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"

@dataclass
class Perception:
    """Agent's perception of environment"""
    sensor_data: Dict[str, Any]
    timestamp: float
    confidence: float = 1.0

@dataclass
class Action:
    """Action to be performed"""
    action_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    estimated_duration: float = 1.0

class SimpleAgent(ABC):
    """Basic reflex agent implementation"""

    def __init__(self, agent_id: str, initial_state: AgentState = AgentState.IDLE):
        self.id = agent_id
        self.state = initial_state
        self.perception_history = []
        self.action_history = []
        self.performance_metrics = {
            'actions_completed': 0,
            'success_rate': 0.0,
            'average_response_time': 0.0,
            'last_update': time.time()
        }

        # Agent characteristics
        self.energy = 100.0
        self.capabilities = []
        self.preferences = {}

    def perceive(self, environment: Dict[str, Any]) -> Perception:
        """Process environmental inputs"""
        sensor_data = self._extract_sensor_data(environment)

        perception = Perception(
            sensor_data=sensor_data,
            timestamp=time.time(),
            confidence=self._calculate_confidence(sensor_data)
        )

        self.perception_history.append(perception)
        self._update_perception_history()

        return perception

    def think(self, perception: Perception) -> Optional[Action]:
        """Process perception and decide on action"""
        if not self._is_able_to_act():
            return None

        # Simple rule-based decision making
        for rule in self._get_behavior_rules():
            if self._rule_matches(perception, rule):
                action = self._generate_action(perception, rule)
                if action:
                    return action

        return None

    def act(self, action: Action) -> Dict[str, Any]:
        """Execute action and return result"""
        start_time = time.time()

        try:
            # Execute action
            result = self._execute_action(action)

            # Update metrics
            execution_time = time.time() - start_time
            self._update_performance_metrics(action, result, execution_time)

            # Update state
            self._update_agent_state(result)

            # Consume energy
            self._consume_energy(action)

            self.action_history.append({
                'action': action,
                'result': result,
                'execution_time': execution_time,
                'timestamp': time.time()
            })

            return result

        except Exception as e:
            self.state = AgentState.ERROR
            return {
                'success': False,
                'error': str(e),
                'action': action
            }

    def run_cycle(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one agent cycle: perceive -> think -> act"""
        # Perceive
        perception = self.perceive(environment)

        # Think
        action = self.think(perception)

        if action:
            # Act
            result = self.act(action)
            return {
                'perception': perception,
                'action': action,
                'result': result,
                'agent_state': self.state
            }
        else:
            return {
                'perception': perception,
                'action': None,
                'result': {'success': True, 'message': 'No action needed'},
                'agent_state': self.state
            }

    @abstractmethod
    def _extract_sensor_data(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant data from environment"""
        pass

    @abstractmethod
    def _calculate_confidence(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate confidence in sensor data"""
        pass

    @abstractmethod
    def _get_behavior_rules(self) -> List[Dict]:
        """Return list of behavior rules"""
        pass

    @abstractmethod
    def _execute_action(self, action: Action) -> Dict[str, Any]:
        """Execute the specified action"""
        pass

    # Helper methods
    def _is_able_to_act(self) -> bool:
        """Check if agent can perform actions"""
        return (self.energy > 10 and
                self.state in [AgentState.IDLE, AgentState.WAITING])

    def _rule_matches(self, perception: Perception, rule: Dict) -> bool:
        """Check if perception matches rule conditions"""
        conditions = rule.get('conditions', {})

        for key, expected_value in conditions.items():
            actual_value = perception.sensor_data.get(key)
            if isinstance(expected_value, list):
                if actual_value not in expected_value:
                    return False
            else:
                if actual_value != expected_value:
                    return False

        return True

    def _generate_action(self, perception: Perception, rule: Dict) -> Optional[Action]:
        """Generate action based on rule"""
        action_spec = rule.get('action', {})

        return Action(
            action_type=action_spec.get('type'),
            parameters=action_spec.get('parameters', {}),
            priority=action_spec.get('priority', 1),
            estimated_duration=action_spec.get('duration', 1.0)
        )

    def _consume_energy(self, action: Action):
        """Consume energy for action execution"""
        energy_cost = action.estimated_duration * 5  # 5 energy units per time unit
        self.energy = max(0, self.energy - energy_cost)

    def _update_performance_metrics(self, action: Action, result: Dict, execution_time: float):
        """Update agent performance metrics"""
        self.performance_metrics['actions_completed'] += 1

        if result.get('success', False):
            # Calculate success rate
            success_count = sum(1 for h in self.action_history[-10:]
                              if h['result'].get('success', False))
            self.performance_metrics['success_rate'] = success_count / min(len(self.action_history), 10)

        # Calculate average response time
        if self.performance_metrics['actions_completed'] == 1:
            self.performance_metrics['average_response_time'] = execution_time
        else:
            current_avg = self.performance_metrics['average_response_time']
            n = self.performance_metrics['actions_completed']
            self.performance_metrics['average_response_time'] = (
                (current_avg * (n - 1) + execution_time) / n
            )

        self.performance_metrics['last_update'] = time.time()

    def _update_agent_state(self, result: Dict):
        """Update agent state based on action result"""
        if result.get('success', False):
            if self.state == AgentState.BUSY:
                self.state = AgentState.IDLE
        else:
            self.state = AgentState.ERROR

    def _update_perception_history(self):
        """Keep perception history manageable"""
        if len(self.perception_history) > 100:
            self.perception_history.pop(0)

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'agent_id': self.id,
            'state': self.state.value,
            'energy': self.energy,
            'capabilities': self.capabilities,
            'performance': self.performance_metrics,
            'last_perception': self.perception_history[-1] if self.perception_history else None
        }

# Example: Cleaning Robot Agent
class CleaningRobotAgent(SimpleAgent):
    """Example: Cleaning robot agent"""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.capabilities = ['cleaning', 'navigation', 'obstacle_avoidance']
        self.cleaned_areas = set()
        self.dirt_detected = False

    def _extract_sensor_data(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data relevant to cleaning robot"""
        return {
            'position': environment.get('robot_position', [0, 0]),
            'dirt_detected': environment.get('dirt_detected', False),
            'battery_level': environment.get('battery_level', 100),
            'obstacles': environment.get('obstacles', []),
            'cleaned_areas': environment.get('cleaned_areas', [])
        }

    def _calculate_confidence(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate confidence in sensor readings"""
        # Simple confidence calculation based on sensor reliability
        confidence = 0.8  # Base confidence

        # Adjust based on battery level
        battery = sensor_data.get('battery_level', 100)
        if battery < 20:
            confidence *= 0.7
        elif battery < 50:
            confidence *= 0.9

        return confidence

    def _get_behavior_rules(self) -> List[Dict]:
        """Return cleaning robot behavior rules"""
        return [
            {
                'name': 'emergency_return',
                'conditions': {'battery_level': lambda x: x < 10},
                'action': {
                    'type': 'return_to_base',
                    'parameters': {'destination': 'charging_station'},
                    'priority': 10,
                    'duration': 2.0
                }
            },
            {
                'name': 'clean_dirt',
                'conditions': {'dirt_detected': True},
                'action': {
                    'type': 'clean_area',
                    'parameters': {'cleaning_method': 'vacuum'},
                    'priority': 8,
                    'duration': 1.0
                }
            },
            {
                'name': 'avoid_obstacle',
                'conditions': {'obstacles': lambda x: len(x) > 0},
                'action': {
                    'type': 'navigate_around',
                    'parameters': {'detour_method': 'right_turn'},
                    'priority': 7,
                    'duration': 0.5
                }
            },
            {
                'name': 'explore_area',
                'conditions': {},
                'action': {
                    'type': 'move_random',
                    'parameters': {'movement_radius': 5},
                    'priority': 1,
                    'duration': 0.3
                }
            }
        ]

    def _execute_action(self, action: Action) -> Dict[str, Any]:
        """Execute cleaning robot action"""
        action_type = action.action_type

        if action_type == 'return_to_base':
            return self._action_return_to_base(action.parameters)
        elif action_type == 'clean_area':
            return self._action_clean_area(action.parameters)
        elif action_type == 'navigate_around':
            return self._action_navigate_around(action.parameters)
        elif action_type == 'move_random':
            return self._action_move_random(action.parameters)
        else:
            return {'success': False, 'error': f'Unknown action: {action_type}'}

    def _action_return_to_base(self, params: Dict) -> Dict[str, Any]:
        """Return to charging base"""
        destination = params.get('destination', 'charging_station')

        # Simulate navigation to base
        time.sleep(0.1)  # Simulate movement time

        return {
            'success': True,
            'action': 'return_to_base',
            'destination': destination,
            'message': f'Robot returned to {destination}'
        }

    def _action_clean_area(self, params: Dict) -> Dict[str, Any]:
        """Clean detected dirt"""
        cleaning_method = params.get('cleaning_method', 'vacuum')

        # Simulate cleaning process
        time.sleep(0.2)  # Simulate cleaning time

        # Mark area as cleaned
        current_pos = self.perception_history[-1].sensor_data.get('position', [0, 0])
        self.cleaned_areas.add(tuple(current_pos))

        return {
            'success': True,
            'action': 'clean_area',
            'method': cleaning_method,
            'cleaned_position': current_pos,
            'message': f'Area at {current_pos} cleaned using {cleaning_method}'
        }

    def _action_navigate_around(self, params: Dict) -> Dict[str, Any]:
        """Navigate around obstacles"""
        detour_method = params.get('detour_method', 'right_turn')

        # Simulate obstacle avoidance
        time.sleep(0.1)  # Simulate navigation time

        return {
            'success': True,
            'action': 'navigate_around',
            'method': detour_method,
            'message': f'Navigated around obstacles using {detour_method}'
        }

    def _action_move_random(self, params: Dict) -> Dict[str, Any]:
        """Move to random position"""
        radius = params.get('movement_radius', 5)

        # Simulate random movement
        time.sleep(0.05)  # Simulate movement time

        return {
            'success': True,
            'action': 'move_random',
            'radius': radius,
            'message': f'Moved randomly within radius {radius}'
        }

# Test the simple agent
def test_cleaning_robot():
    """Test the cleaning robot agent"""
    robot = CleaningRobotAgent("robot_001")

    # Test scenarios
    scenarios = [
        # Low battery scenario
        {
            'environment': {
                'robot_position': [10, 10],
                'dirt_detected': False,
                'battery_level': 5,
                'obstacles': [],
                'cleaned_areas': []
            },
            'expected_action': 'return_to_base'
        },
        # Dirt detected scenario
        {
            'environment': {
                'robot_position': [5, 5],
                'dirt_detected': True,
                'battery_level': 80,
                'obstacles': [],
                'cleaned_areas': []
            },
            'expected_action': 'clean_area'
        },
        # Obstacle scenario
        {
            'environment': {
                'robot_position': [3, 3],
                'dirt_detected': False,
                'battery_level': 90,
                'obstacles': [[4, 4], [5, 5]],
                'cleaned_areas': []
            },
            'expected_action': 'navigate_around'
        }
    ]

    for i, scenario in enumerate(scenarios):
        print(f"\nTest Scenario {i+1}:")
        result = robot.run_cycle(scenario['environment'])

        print(f"Perceived: {result['perception'].sensor_data}")
        if result['action']:
            print(f"Action: {result['action'].action_type}")
            print(f"Result: {result['result']}")
        else:
            print("No action taken")

        print(f"Agent State: {robot.state.value}")
        print(f"Energy: {robot.energy}")

        # Verify expected action
        if result['action'] and result['action'].action_type == scenario['expected_action']:
            print("✓ Test passed!")
        else:
            print("✗ Test failed!")

    # Print final status
    print(f"\nFinal Status: {robot.get_status()}")

if __name__ == "__main__":
    test_cleaning_robot()
```

### 2. Goal-Oriented Agent

```python
# agents/goal_oriented_agent.py
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import heapq

@dataclass
class Goal:
    """Represents a goal for the agent"""
    goal_id: str
    description: str
    priority: float
    deadline: Optional[float] = None
    status: str = "active"
    progress: float = 0.0
    required_resources: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

@dataclass
class Plan:
    """Represents a plan to achieve a goal"""
    goal_id: str
    steps: List[Dict[str, Any]]
    estimated_duration: float
    resource_requirements: Dict[str, int]
    success_probability: float = 0.8

class GoalOrientedAgent(SimpleAgent):
    """Agent that works towards specific goals"""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.goals = {}
        self.current_plan = None
        self.goal_history = []
        self.resource_allocations = {}

        # Goal management settings
        self.max_concurrent_goals = 3
        self.goal_selection_strategy = "priority_deadline"

    def add_goal(self, goal: Goal):
        """Add a new goal"""
        self.goals[goal.goal_id] = goal
        print(f"Added goal: {goal.description} (Priority: {goal.priority})")

    def remove_goal(self, goal_id: str):
        """Remove a goal"""
        if goal_id in self.goals:
            removed_goal = self.goals.pop(goal_id)
            print(f"Removed goal: {removed_goal.description}")

    def update_goal_progress(self, goal_id: str, progress: float):
        """Update progress on a goal"""
        if goal_id in self.goals:
            self.goals[goal_id].progress = min(100.0, max(0.0, progress))

            if self.goals[goal_id].progress >= 100.0:
                self.goals[goal_id].status = "completed"
                print(f"Goal completed: {self.goals[goal_id].description}")

    def think(self, perception: Perception) -> Optional[Action]:
        """Decide on action based on goals and current state"""
        # Check if we have an active plan
        if self.current_plan and self.current_plan['status'] == 'in_progress':
            return self._continue_current_plan(perception)

        # Select next goal to work on
        selected_goal = self._select_next_goal()
        if not selected_goal:
            return None

        # Create plan for selected goal
        self.current_plan = self._create_plan(selected_goal, perception)
        if not self.current_plan:
            return None

        # Execute first step of plan
        return self._execute_next_plan_step(perception)

    def _select_next_goal(self) -> Optional[Goal]:
        """Select next goal to work on"""
        # Get active goals
        active_goals = [goal for goal in self.goals.values()
                       if goal.status == 'active']

        if not active_goals:
            return None

        # Filter goals based on dependencies and resources
        feasible_goals = self._filter_feasible_goals(active_goals)

        if not feasible_goals:
            return None

        # Select goal based on strategy
        if self.goal_selection_strategy == "priority_deadline":
            return self._select_by_priority_deadline(feasible_goals)
        elif self.goal_selection_strategy == "progress":
            return self._select_by_progress(feasible_goals)
        else:
            return feasible_goals[0]  # Default: first goal

    def _filter_feasible_goals(self, goals: List[Goal]) -> List[Goal]:
        """Filter goals that can be worked on"""
        feasible = []

        for goal in goals:
            # Check dependencies
            if not self._dependencies_satisfied(goal):
                continue

            # Check resource availability
            if not self._resources_available(goal):
                continue

            # Check deadline
            if goal.deadline and time.time() > goal.deadline:
                continue

            feasible.append(goal)

        return feasible

    def _dependencies_satisfied(self, goal: Goal) -> bool:
        """Check if goal dependencies are satisfied"""
        for dep_id in goal.dependencies:
            if dep_id not in self.goals:
                continue
            if self.goals[dep_id].status != 'completed':
                return False
        return True

    def _resources_available(self, goal: Goal) -> bool:
        """Check if required resources are available"""
        for resource in goal.required_resources:
            available = self.resource_allocations.get(resource, 0)
            needed = goal.required_resources.count(resource)
            if available < needed:
                return False
        return True

    def _select_by_priority_deadline(self, goals: List[Goal]) -> Goal:
        """Select goal by priority and deadline"""
        # Score goals based on priority and urgency
        scored_goals = []
        current_time = time.time()

        for goal in goals:
            score = goal.priority

            # Adjust score based on deadline urgency
            if goal.deadline:
                time_remaining = goal.deadline - current_time
                if time_remaining > 0:
                    # Earlier deadlines get higher scores
                    urgency_factor = 1.0 / (time_remaining / 3600)  # Hours
                    score *= (1.0 + urgency_factor)
                else:
                    score *= 2.0  # Overdue goals get highest priority

            scored_goals.append((score, goal))

        # Select goal with highest score
        scored_goals.sort(reverse=True)
        return scored_goals[0][1]

    def _select_by_progress(self, goals: List[Goal]) -> Goal:
        """Select goal with lowest progress"""
        return min(goals, key=lambda g: g.progress)

    def _create_plan(self, goal: Goal, perception: Perception) -> Optional[Dict[str, Any]]:
        """Create a plan to achieve the goal"""
        # Simple planning logic based on goal type
        if 'cleaning' in goal.description.lower():
            return self._create_cleaning_plan(goal, perception)
        elif 'delivery' in goal.description.lower():
            return self._create_delivery_plan(goal, perception)
        else:
            return self._create_generic_plan(goal, perception)

    def _create_cleaning_plan(self, goal: Goal, perception: Perception) -> Dict[str, Any]:
        """Create plan for cleaning goal"""
        steps = [
            {
                'step_id': 'assess_area',
                'action': 'assess_cleaning_requirements',
                'parameters': {},
                'estimated_duration': 0.5
            },
            {
                'step_id': 'clean_dirt',
                'action': 'clean_area',
                'parameters': {'cleaning_method': 'vacuum'},
                'estimated_duration': 2.0
            },
            {
                'step_id': 'verify_cleaning',
                'action': 'verify_cleanliness',
                'parameters': {},
                'estimated_duration': 0.5
            }
        ]

        return {
            'goal_id': goal.goal_id,
            'steps': steps,
            'current_step': 0,
            'status': 'in_progress',
            'total_duration': sum(step['estimated_duration'] for step in steps)
        }

    def _create_delivery_plan(self, goal: Goal, perception: Perception) -> Dict[str, Any]:
        """Create plan for delivery goal"""
        steps = [
            {
                'step_id': 'navigate_to_pickup',
                'action': 'navigate_to_location',
                'parameters': {'destination': 'pickup_point'},
                'estimated_duration': 3.0
            },
            {
                'step_id': 'pickup_item',
                'action': 'pickup_object',
                'parameters': {},
                'estimated_duration': 1.0
            },
            {
                'step_id': 'navigate_to_dropoff',
                'action': 'navigate_to_location',
                'parameters': {'destination': 'dropoff_point'},
                'estimated_duration': 3.0
            },
            {
                'step_id': 'deliver_item',
                'action': 'deliver_object',
                'parameters': {},
                'estimated_duration': 1.0
            }
        ]

        return {
            'goal_id': goal.goal_id,
            'steps': steps,
            'current_step': 0,
            'status': 'in_progress',
            'total_duration': sum(step['estimated_duration'] for step in steps)
        }

    def _create_generic_plan(self, goal: Goal, perception: Perception) -> Dict[str, Any]:
        """Create generic plan"""
        steps = [
            {
                'step_id': 'analyze_goal',
                'action': 'analyze_objective',
                'parameters': {},
                'estimated_duration': 1.0
            },
            {
                'step_id': 'execute_task',
                'action': 'perform_task',
                'parameters': {},
                'estimated_duration': 2.0
            },
            {
                'step_id': 'verify_completion',
                'action': 'verify_completion',
                'parameters': {},
                'estimated_duration': 0.5
            }
        ]

        return {
            'goal_id': goal.goal_id,
            'steps': steps,
            'current_step': 0,
            'status': 'in_progress',
            'total_duration': sum(step['estimated_duration'] for step in steps)
        }

    def _continue_current_plan(self, perception: Perception) -> Optional[Action]:
        """Continue executing current plan"""
        if not self.current_plan:
            return None

        current_step_index = self.current_plan['current_step']
        if current_step_index >= len(self.current_plan['steps']):
            # Plan completed
            self._complete_current_plan()
            return None

        # Get current step
        current_step = self.current_plan['steps'][current_step_index]

        return Action(
            action_type=current_step['action'],
            parameters=current_step['parameters'],
            priority=5,  # High priority for plan execution
            estimated_duration=current_step['estimated_duration']
        )

    def _execute_next_plan_step(self, perception: Perception) -> Action:
        """Execute next step in current plan"""
        return self._continue_current_plan(perception)

    def _complete_current_plan(self):
        """Mark current plan as completed"""
        if self.current_plan:
            self.current_plan['status'] = 'completed'

            # Update goal progress
            goal_id = self.current_plan['goal_id']
            if goal_id in self.goals:
                self.goals[goal_id].progress = 100.0
                self.goals[goal_id].status = 'completed'

            # Move to history
            self.goal_history.append(self.current_plan)
            self.current_plan = None

    # Implement required abstract methods
    def _extract_sensor_data(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        return environment

    def _calculate_confidence(self, sensor_data: Dict[str, Any]) -> float:
        return 0.9

    def _get_behavior_rules(self) -> List[Dict]:
        # Override to use goal-oriented behavior
        return []

    def _execute_action(self, action: Action) -> Dict[str, Any]:
        """Execute action in goal-oriented context"""
        action_type = action.action_type

        # Handle plan-specific actions
        if self.current_plan:
            current_step = self.current_plan['steps'][self.current_plan['current_step']]

            if action_type == current_step['action']:
                # Successful step completion
                self.current_plan['current_step'] += 1

                # Update goal progress
                goal_id = self.current_plan['goal_id']
                if goal_id in self.goals:
                    goal = self.goals[goal_id]
                    total_steps = len(self.current_plan['steps'])
                    completed_steps = self.current_plan['current_step']
                    goal.progress = (completed_steps / total_steps) * 100.0

                return {
                    'success': True,
                    'action': action_type,
                    'step_completed': current_step['step_id'],
                    'goal_progress': goal.progress if goal_id in self.goals else 0.0
                }

        # Default action handling
        time.sleep(action.estimated_duration * 0.1)  # Simulate execution time

        return {
            'success': True,
            'action': action_type,
            'message': f'Action {action_type} executed'
        }

# Test goal-oriented agent
def test_goal_oriented_agent():
    """Test the goal-oriented agent"""
    agent = GoalOrientedAgent("agent_001")

    # Add some goals
    cleaning_goal = Goal(
        goal_id="goal_001",
        description="Clean the living room",
        priority=8.0,
        deadline=time.time() + 3600  # 1 hour from now
    )

    delivery_goal = Goal(
        goal_id="goal_002",
        description="Deliver package to customer",
        priority=7.0,
        deadline=time.time() + 1800  # 30 minutes from now
    )

    maintenance_goal = Goal(
        goal_id="goal_003",
        description="Perform system maintenance",
        priority=5.0
    )

    agent.add_goal(cleaning_goal)
    agent.add_goal(delivery_goal)
    agent.add_goal(maintenance_goal)

    # Run agent for several cycles
    environment = {
        'position': [0, 0],
        'dirt_detected': True,
        'battery_level': 90,
        'cleaned_areas': [],
        'package_location': [5, 5],
        'customer_location': [10, 10]
    }

    print("Starting goal-oriented agent simulation...\n")

    for cycle in range(10):
        print(f"Cycle {cycle + 1}:")

        result = agent.run_cycle(environment)

        print(f"Agent State: {result['agent_state'].value}")

        if result['action']:
            print(f"Action: {result['action'].action_type}")

            # Update environment based on action
            if result['action'].action_type == 'clean_area':
                environment['dirt_detected'] = False
                environment['cleaned_areas'].append([0, 0])
            elif result['action'].action_type == 'navigate_to_location':
                environment['position'] = [10, 10]  # Update position

        # Print goal progress
        active_goals = [goal for goal in agent.goals.values() if goal.status == 'active']
        if active_goals:
            for goal in active_goals:
                print(f"  Goal '{goal.description}': {goal.progress:.1f}% complete")

        print()

        # Stop if all goals are completed
        if not active_goals:
            print("All goals completed!")
            break

    print("Simulation completed.")

    # Print final status
    final_status = agent.get_status()
    final_status['goals'] = {gid: {
        'description': goal.description,
        'progress': goal.progress,
        'status': goal.status
    } for gid, goal in agent.goals.items()}

    print(f"\nFinal Agent Status: {final_status}")

if __name__ == "__main__":
    test_goal_oriented_agent()
```

---

## Agent Communication Systems {#agent-communication-systems}

### 1. Message Bus Implementation

```python
# communication/message_bus.py
import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AgentMessage:
    """Standard message format for agent communication"""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    content: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = None
    expires_at: Optional[float] = None
    reply_to: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}

class MessageBus:
    """Centralized message bus for agent communication"""

    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.subscribers = {}  # agent_id -> {message_types -> callback}
        self.message_queue = asyncio.Queue()
        self.delivered_messages = {}  # agent_id -> list of messages
        self.message_history = []
        self.active_connections = set()
        self.message_stats = {
            'total_messages': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'average_delivery_time': 0.0
        }

        # Start message processing task
        self.processing_task = None

        self.logger = logging.getLogger(__name__)

    async def start(self):
        """Start the message bus"""
        self.processing_task = asyncio.create_task(self._process_messages())
        self.logger.info("Message bus started")

    async def stop(self):
        """Stop the message bus"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Message bus stopped")

    async def subscribe(self, agent_id: str, message_types: Set[MessageType],
                       callback: Callable[[AgentMessage], None]):
        """Subscribe agent to message types"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = {}

        for msg_type in message_types:
            self.subscribers[agent_id][msg_type] = callback

        self.active_connections.add(agent_id)
        self.logger.info(f"Agent {agent_id} subscribed to {len(message_types)} message types")

    async def unsubscribe(self, agent_id: str, message_types: Optional[Set[MessageType]] = None):
        """Unsubscribe agent from message types"""
        if agent_id not in self.subscribers:
            return

        if message_types is None:
            # Unsubscribe from all
            del self.subscribers[agent_id]
            self.active_connections.discard(agent_id)
        else:
            # Unsubscribe from specific types
            for msg_type in message_types:
                if msg_type in self.subscribers[agent_id]:
                    del self.subscribers[agent_id][msg_type]

        self.logger.info(f"Agent {agent_id} unsubscribed from messages")

    async def publish(self, message: AgentMessage) -> bool:
        """Publish message to message bus"""
        # Validate message
        if not self._validate_message(message):
            self.logger.error(f"Invalid message: {message.message_id}")
            return False

        # Add to queue
        try:
            self.message_queue.put_nowait(message)
            self.message_stats['total_messages'] += 1
            self.logger.debug(f"Published message {message.message_id}")
            return True
        except asyncio.QueueFull:
            self.logger.error("Message queue full")
            return False

    async def send_direct(self, sender_id: str, receiver_id: str,
                         message_type: MessageType, content: Dict[str, Any],
                         priority: MessagePriority = MessagePriority.NORMAL,
                         expires_in: Optional[float] = None) -> str:
        """Send direct message to specific agent"""
        message_id = str(uuid.uuid4())

        # Calculate expiration time
        expires_at = None
        if expires_in:
            expires_at = time.time() + expires_in

        message = AgentMessage(
            message_id=message_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority,
            expires_at=expires_at
        )

        success = await self.publish(message)
        if success:
            return message_id
        else:
            return None

    async def broadcast(self, sender_id: str, message_type: MessageType,
                       content: Dict[str, Any],
                       priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """Broadcast message to all subscribers"""
        message_id = str(uuid.uuid4())

        message = AgentMessage(
            message_id=message_id,
            sender_id=sender_id,
            receiver_id=None,  # Broadcast
            message_type=message_type,
            content=content,
            priority=priority
        )

        success = await self.publish(message)
        if success:
            return message_id
        else:
            return None

    async def _process_messages(self):
        """Process messages from queue"""
        while True:
            try:
                # Get message from queue
                message = await self.message_queue.get()

                # Check if message has expired
                if message.expires_at and time.time() > message.expires_at:
                    self.logger.debug(f"Message {message.message_id} expired")
                    continue

                # Deliver message
                delivery_time = await self._deliver_message(message)

                # Update stats
                if delivery_time:
                    self.message_stats['successful_deliveries'] += 1

                    # Update average delivery time
                    n = self.message_stats['successful_deliveries']
                    current_avg = self.message_stats['average_delivery_time']
                    self.message_stats['average_delivery_time'] = (
                        (current_avg * (n - 1) + delivery_time) / n
                    )
                else:
                    self.message_stats['failed_deliveries'] += 1

                # Add to history
                self.message_history.append({
                    'message': message,
                    'delivery_time': delivery_time,
                    'processed_at': time.time()
                })

                # Limit history size
                if len(self.message_history) > 10000:
                    self.message_history = self.message_history[-5000:]

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")

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
            self.logger.error(f"Failed to deliver message {message.message_id}: {e}")
            return None

    async def _deliver_direct_message(self, message: AgentMessage) -> float:
        """Deliver direct message"""
        receiver_id = message.receiver_id

        if receiver_id not in self.subscribers:
            self.logger.warning(f"Unknown receiver: {receiver_id}")
            return None

        # Check if receiver is subscribed to this message type
        if message.message_type not in self.subscribers[receiver_id]:
            self.logger.warning(f"Receiver {receiver_id} not subscribed to {message.message_type}")
            return None

        try:
            # Deliver message
            callback = self.subscribers[receiver_id][message.message_type]
            await callback(message)

            # Store in delivered messages
            if receiver_id not in self.delivered_messages:
                self.delivered_messages[receiver_id] = []
            self.delivered_messages[receiver_id].append(message)

            delivery_time = time.time() - start_time
            self.logger.debug(f"Delivered message {message.message_id} to {receiver_id} in {delivery_time:.3f}s")

            return delivery_time

        except Exception as e:
            self.logger.error(f"Error delivering message to {receiver_id}: {e}")
            return None

    async def _deliver_broadcast_message(self, message: AgentMessage) -> float:
        """Deliver broadcast message"""
        delivery_times = []

        for receiver_id, subscriptions in self.subscribers.items():
            if message.message_type in subscriptions:
                try:
                    callback = subscriptions[message.message_type]
                    await callback(message)

                    # Store in delivered messages
                    if receiver_id not in self.delivered_messages:
                        self.delivered_messages[receiver_id] = []
                    self.delivered_messages[receiver_id].append(message)

                    delivery_times.append(time.time() - start_time)

                except Exception as e:
                    self.logger.error(f"Error broadcasting to {receiver_id}: {e}")

        if delivery_times:
            avg_delivery_time = sum(delivery_times) / len(delivery_times)
            self.logger.debug(f"Broadcast message {message.message_id} delivered to {len(delivery_times)} agents")
            return avg_delivery_time
        else:
            return None

    def _validate_message(self, message: AgentMessage) -> bool:
        """Validate message format"""
        required_fields = ['message_id', 'sender_id', 'message_type', 'content']

        for field in required_fields:
            if not hasattr(message, field) or getattr(message, field) is None:
                return False

        return True

    def get_message_stats(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        return {
            **self.message_stats,
            'active_connections': len(self.active_connections),
            'subscribers': len(self.subscribers),
            'queue_size': self.message_queue.qsize(),
            'message_history_size': len(self.message_history)
        }

    def get_agent_messages(self, agent_id: str, limit: int = 100) -> List[AgentMessage]:
        """Get messages delivered to specific agent"""
        if agent_id in self.delivered_messages:
            return self.delivered_messages[agent_id][-limit:]
        return []

# Agent with communication capabilities
class CommunicatingAgent:
    """Base class for agents with communication capabilities"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.id = agent_id
        self.message_bus = message_bus
        self.message_handlers = {}
        self.conversation_state = {}
        self.conversation_counter = 0

        # Subscribe to messages
        self.message_handlers[MessageType.REQUEST] = self._handle_request
        self.message_handlers[MessageType.RESPONSE] = self._handle_response
        self.message_handlers[MessageType.NOTIFICATION] = self._handle_notification
        self.message_handlers[MessageType.BROADCAST] = self._handle_broadcast
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat

    async def register_communication(self, message_types: Set[MessageType]):
        """Register for communication"""
        await self.message_bus.subscribe(self.id, message_types, self._handle_message)

    async def send_request(self, receiver_id: str, action: str, parameters: Dict[str, Any],
                          priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """Send request to another agent"""
        content = {
            'action': action,
            'parameters': parameters,
            'conversation_id': self._create_conversation_id()
        }

        message_id = await self.message_bus.send_direct(
            self.id, receiver_id, MessageType.REQUEST, content, priority
        )

        return message_id

    async def send_response(self, receiver_id: str, conversation_id: str,
                          result: Dict[str, Any], success: bool = True):
        """Send response to request"""
        content = {
            'conversation_id': conversation_id,
            'result': result,
            'success': success
        }

        await self.message_bus.send_direct(
            self.id, receiver_id, MessageType.RESPONSE, content
        )

    async def send_notification(self, receiver_id: str, event: str, data: Dict[str, Any]):
        """Send notification to another agent"""
        content = {
            'event': event,
            'data': data
        }

        await self.message_bus.send_direct(
            self.id, receiver_id, MessageType.NOTIFICATION, content
        )

    async def broadcast(self, message_type: MessageType, content: Dict[str, Any]):
        """Broadcast message to all agents"""
        await self.message_bus.broadcast(self.id, message_type, content)

    async def _handle_message(self, message: AgentMessage):
        """Handle incoming message"""
        if message.message_type in self.message_handlers:
            await self.message_handlers[message.message_type](message)
        else:
            self.logger.warning(f"No handler for message type: {message.message_type}")

    async def _handle_request(self, message: AgentMessage):
        """Handle incoming request"""
        content = message.content
        action = content.get('action')
        parameters = content.get('parameters', {})
        conversation_id = content.get('conversation_id')

        # Process request (this would be implemented by specific agents)
        result = await self._process_request(action, parameters)

        # Send response
        await self.send_response(
            message.sender_id,
            conversation_id,
            result,
            result.get('success', False)
        )

    async def _handle_response(self, message: AgentMessage):
        """Handle incoming response"""
        content = message.content
        conversation_id = content.get('conversation_id')
        result = content.get('result')

        # Update conversation state
        if conversation_id in self.conversation_state:
            self.conversation_state[conversation_id]['response'] = result
            self.conversation_state[conversation_id]['completed'] = True

    async def _handle_notification(self, message: AgentMessage):
        """Handle incoming notification"""
        content = message.content
        event = content.get('event')
        data = content.get('data')

        # Process notification (this would be implemented by specific agents)
        await self._process_notification(event, data)

    async def _handle_broadcast(self, message: AgentMessage):
        """Handle incoming broadcast"""
        content = message.content
        # Process broadcast (this would be implemented by specific agents)
        await self._process_broadcast(content)

    async def _handle_heartbeat(self, message: AgentMessage):
        """Handle incoming heartbeat"""
        # Update agent status or send heartbeat response
        await self._process_heartbeat(message.sender_id)

    async def _process_request(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process request (to be implemented by subclasses)"""
        return {'success': True, 'message': f'Processed action: {action}'}

    async def _process_notification(self, event: str, data: Dict[str, Any]):
        """Process notification (to be implemented by subclasses)"""
        print(f"Agent {self.id} received notification: {event}")

    async def _process_broadcast(self, content: Dict[str, Any]):
        """Process broadcast (to be implemented by subclasses)"""
        print(f"Agent {self.id} received broadcast: {content}")

    async def _process_heartbeat(self, sender_id: str):
        """Process heartbeat (to be implemented by subclasses)"""
        print(f"Agent {self.id} received heartbeat from {sender_id}")

    def _create_conversation_id(self) -> str:
        """Create unique conversation ID"""
        self.conversation_counter += 1
        return f"{self.id}_conv_{self.conversation_counter}"

# Test message bus and communication
async def test_message_bus():
    """Test the message bus and agent communication"""

    # Create message bus
    message_bus = MessageBus()
    await message_bus.start()

    # Create communicating agents
    agent1 = CommunicatingAgent("agent_001", message_bus)
    agent2 = CommunicatingAgent("agent_002", message_bus)
    agent3 = CommunicatingAgent("agent_003", message_bus)

    # Register agents for communication
    await agent1.register_communication({MessageType.REQUEST, MessageType.RESPONSE})
    await agent2.register_communication({MessageType.REQUEST, MessageType.RESPONSE})
    await agent3.register_communication({MessageType.BROADCAST, MessageType.NOTIFICATION})

    print("Testing agent communication...\n")

    # Test direct messaging
    print("1. Testing direct request-response:")
    message_id = await agent1.send_request("agent_002", "get_status", {"query": "current_state"})
    print(f"Sent request with ID: {message_id}")

    # Wait a moment for processing
    await asyncio.sleep(0.1)

    # Test broadcast
    print("\n2. Testing broadcast:")
    await agent3.broadcast(MessageType.BROADCAST, {
        "event": "system_announcement",
        "message": "Hello from agent_003"
    })

    # Test notification
    print("\n3. Testing notification:")
    await agent2.send_notification("agent_001", "task_completed", {
        "task_id": "task_123",
        "completion_time": time.time()
    })

    # Print message statistics
    print(f"\nMessage Bus Statistics:")
    stats = message_bus.get_message_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Clean up
    await message_bus.stop()

if __name__ == "__main__":
    asyncio.run(test_message_bus())
```

### 2. Agent Protocols

```python
# communication/protocol_handlers.py
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum
import asyncio

class ProtocolType(Enum):
    REQUEST_RESPONSE = "request_response"
    CONTRACT_NET = "contract_net"
    AUCTION = "auction"
    NEGOTIATION = "negotiation"
    CONSENSUS = "consensus"

class ProtocolState(Enum):
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProtocolHandler(ABC):
    """Base class for agent communication protocols"""

    def __init__(self, protocol_id: str, initiator_id: str):
        self.protocol_id = protocol_id
        self.initiator_id = initiator_id
        self.state = ProtocolState.INITIATED
        self.participants = []
        self.message_history = []
        self.callbacks = {}

    @abstractmethod
    async def initiate(self, participants: List[str], message_bus) -> bool:
        """Initiate protocol with participants"""
        pass

    @abstractmethod
    async def handle_message(self, message, message_bus):
        """Handle incoming protocol message"""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current protocol state"""
        pass

    def add_callback(self, event: str, callback: Callable):
        """Add callback for protocol events"""
        self.callbacks[event] = callback

    def _trigger_callback(self, event: str, data: Any = None):
        """Trigger registered callback"""
        if event in self.callbacks:
            self.callbacks[event](data)

class ContractNetProtocol(ProtocolHandler):
    """Contract Net Protocol implementation"""

    def __init__(self, protocol_id: str, initiator_id: str):
        super().__init__(protocol_id, initiator_id)
        self.task_announcement = None
        self.received_bids = {}
        self.selected_contractor = None
        self.contract_status = None

    async def initiate(self, participants: List[str], message_bus) -> bool:
        """Initiate contract net protocol"""
        self.participants = participants
        self.state = ProtocolState.IN_PROGRESS

        # Create task announcement
        self.task_announcement = {
            'protocol_id': self.protocol_id,
            'task_id': f"task_{self.protocol_id}",
            'description': 'Sample task for contract net',
            'deadline': time.time() + 300,  # 5 minutes
            'eligibility_criteria': ['available', 'capable'],
            'manager_id': self.initiator_id
        }

        # Broadcast task announcement
        await message_bus.broadcast(
            self.initiator_id,
            MessageType.REQUEST,
            {
                'protocol': 'contract_net',
                'action': 'announce_task',
                'task': self.task_announcement
            }
        )

        self._trigger_callback('protocol_initiated', self.task_announcement)
        return True

    async def handle_message(self, message, message_bus):
        """Handle contract net messages"""
        content = message.content

        if content.get('protocol') == 'contract_net':
            action = content.get('action')

            if action == 'submit_bid':
                await self._handle_bid_submission(message, message_bus)
            elif action == 'accept_proposal':
                await self._handle_proposal_acceptance(message, message_bus)
            elif action == 'reject_proposal':
                await self._handle_proposal_rejection(message, message_bus)

    async def _handle_bid_submission(self, message, message_bus):
        """Handle bid submission from contractor"""
        sender_id = message.sender_id
        bid = message.content.get('bid')

        self.received_bids[sender_id] = bid

        # Check if deadline has passed
        if time.time() > self.task_announcement['deadline']:
            await self._select_best_bid(message_bus)

    async def _select_best_bid(self, message_bus):
        """Select best bid and award contract"""
        if not self.received_bids:
            self.state = ProtocolState.FAILED
            self._trigger_callback('protocol_failed', 'No bids received')
            return

        # Simple bid evaluation (in practice, more sophisticated)
        best_bid = max(self.received_bids.items(),
                      key=lambda x: x[1].get('score', 0))

        self.selected_contractor = best_bid[0]

        # Award contract
        await message_bus.send_direct(
            self.initiator_id,
            self.selected_contractor,
            MessageType.RESPONSE,
            {
                'protocol': 'contract_net',
                'action': 'accept_proposal',
                'contract': {
                    'task': self.task_announcement,
                    'contractor': self.selected_contractor,
                    'bid': best_bid[1]
                }
            }
        )

        # Reject other bids
        for contractor_id, bid in self.received_bids.items():
            if contractor_id != self.selected_contractor:
                await message_bus.send_direct(
                    self.initiator_id,
                    contractor_id,
                    MessageType.RESPONSE,
                    {
                        'protocol': 'contract_net',
                        'action': 'reject_proposal',
                        'reason': 'Better bid selected'
                    }
                )

        self.state = ProtocolState.COMPLETED
        self._trigger_callback('contract_awarded', {
            'contractor': self.selected_contractor,
            'bid': best_bid[1]
        })

    def get_state(self) -> Dict[str, Any]:
        """Get current contract net state"""
        return {
            'protocol_id': self.protocol_id,
            'state': self.state.value,
            'participants': self.participants,
            'received_bids_count': len(self.received_bids),
            'selected_contractor': self.selected_contractor,
            'task_deadline': self.task_announcement['deadline'] if self.task_announcement else None
        }

class NegotiationProtocol(ProtocolHandler):
    """Negotiation protocol implementation"""

    def __init__(self, protocol_id: str, initiator_id: str):
        super().__init__(protocol_id, initiator_id)
        self.proposal = None
        self.counter_proposals = {}
        self.negotiation_rounds = 0
        self.max_rounds = 5
        self.agreement_reached = False

    async def initiate(self, participants: List[str], message_bus) -> bool:
        """Initiate negotiation protocol"""
        self.participants = participants
        self.state = ProtocolState.IN_PROGRESS

        # Create initial proposal
        self.proposal = {
            'protocol_id': self.protocol_id,
            'proposer_id': self.initiator_id,
            'terms': {
                'price': 100,
                'delivery_time': 7,
                'quality_level': 'high'
            },
            'round': 0
        }

        # Send initial proposal to all participants
        for participant_id in participants:
            await message_bus.send_direct(
                self.initiator_id,
                participant_id,
                MessageType.REQUEST,
                {
                    'protocol': 'negotiation',
                    'action': 'receive_proposal',
                    'proposal': self.proposal
                }
            )

        self._trigger_callback('negotiation_initiated', self.proposal)
        return True

    async def handle_message(self, message, message_bus):
        """Handle negotiation messages"""
        content = message.content

        if content.get('protocol') == 'negotiation':
            action = content.get('action')

            if action == 'submit_counter_proposal':
                await self._handle_counter_proposal(message, message_bus)
            elif action == 'accept_proposal':
                await self._handle_proposal_acceptance(message, message_bus)
            elif action == 'reject_proposal':
                await self._handle_proposal_rejection(message, message_bus)

    async def _handle_counter_proposal(self, message, message_bus):
        """Handle counter proposal"""
        sender_id = message.sender_id
        counter_proposal = message.content.get('proposal')

        self.counter_proposals[sender_id] = counter_proposal
        self.negotiation_rounds += 1

        # Check if max rounds reached
        if self.negotiation_rounds >= self.max_rounds:
            self.state = ProtocolState.FAILED
            self._trigger_callback('negotiation_failed', 'Max rounds reached')
            return

        # Evaluate counter proposals (simplified)
        accepted_proposals = []
        for proposer_id, proposal in self.counter_proposals.items():
            if self._evaluate_proposal(proposal):
                accepted_proposals.append(proposer_id)

        # Send response to all participants
        for proposer_id in self.counter_proposals.keys():
            response = {
                'action': 'proposal_response',
                'accepted': proposer_id in accepted_proposals,
                'feedback': self._generate_feedback(proposal)
            }

            await message_bus.send_direct(
                self.initiator_id,
                proposer_id,
                MessageType.RESPONSE,
                response
            )

    async def _handle_proposal_acceptance(self, message, message_bus):
        """Handle proposal acceptance"""
        self.agreement_reached = True
        self.state = ProtocolState.COMPLETED

        agreement = {
            'protocol_id': self.protocol_id,
            'participants': [self.initiator_id, message.sender_id],
            'terms': self.proposal['terms'],
            'accepted_at': time.time()
        }

        # Confirm agreement
        await message_bus.send_direct(
            self.initiator_id,
            message.sender_id,
            MessageType.RESPONSE,
            {
                'action': 'agreement_confirmed',
                'agreement': agreement
            }
        )

        self._trigger_callback('agreement_reached', agreement)

    def _evaluate_proposal(self, proposal: Dict[str, Any]) -> bool:
        """Evaluate counter proposal"""
        # Simplified evaluation - in practice, more sophisticated logic
        terms = proposal.get('terms', {})

        # Check if terms are acceptable
        price = terms.get('price', 0)
        delivery_time = terms.get('delivery_time', 0)

        return price <= 120 and delivery_time <= 10

    def _generate_feedback(self, proposal: Dict[str, Any]) -> str:
        """Generate feedback on proposal"""
        # Simplified feedback generation
        terms = proposal.get('terms', {})
        feedback_parts = []

        price = terms.get('price', 0)
        if price > 120:
            feedback_parts.append("Price too high")

        delivery_time = terms.get('delivery_time', 0)
        if delivery_time > 10:
            feedback_parts.append("Delivery time too long")

        return "; ".join(feedback_parts) if feedback_parts else "Proposal acceptable"

    def get_state(self) -> Dict[str, Any]:
        """Get current negotiation state"""
        return {
            'protocol_id': self.protocol_id,
            'state': self.state.value,
            'negotiation_rounds': self.negotiation_rounds,
            'max_rounds': self.max_rounds,
            'agreement_reached': self.agreement_reached,
            'counter_proposals_count': len(self.counter_proposals),
            'current_proposal': self.proposal
        }

class ProtocolManager:
    """Manager for handling multiple protocols"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.active_protocols = {}
        self.protocol_history = []

    async def start_protocol(self, protocol_type: ProtocolType, participants: List[str],
                           protocol_id: str = None) -> str:
        """Start a new protocol"""
        if protocol_id is None:
            protocol_id = f"{protocol_type.value}_{self.agent_id}_{int(time.time())}"

        if protocol_type == ProtocolType.CONTRACT_NET:
            protocol = ContractNetProtocol(protocol_id, self.agent_id)
        elif protocol_type == ProtocolType.NEGOTIATION:
            protocol = NegotiationProtocol(protocol_id, self.agent_id)
        else:
            raise ValueError(f"Unsupported protocol type: {protocol_type}")

        # Initiate protocol
        success = await protocol.initiate(participants, self.message_bus)

        if success:
            self.active_protocols[protocol_id] = protocol
            return protocol_id
        else:
            return None

    async def handle_protocol_message(self, message):
        """Handle incoming protocol message"""
        content = message.content

        if 'protocol' in content:
            protocol_id = content.get('protocol_id')
            protocol_type = content.get('protocol')

            if protocol_id in self.active_protocols:
                protocol = self.active_protocols[protocol_id]
                await protocol.handle_message(message, self.message_bus)

                # Check if protocol is completed
                if protocol.state in [ProtocolState.COMPLETED, ProtocolState.FAILED]:
                    self.protocol_history.append({
                        'protocol_id': protocol_id,
                        'type': protocol_type,
                        'state': protocol.state.value,
                        'completed_at': time.time()
                    })

                    # Remove from active protocols
                    del self.active_protocols[protocol_id]

    def get_active_protocols(self) -> List[Dict[str, Any]]:
        """Get list of active protocols"""
        return [
            {
                'protocol_id': pid,
                'type': protocol.__class__.__name__,
                'state': protocol.get_state()
            }
            for pid, protocol in self.active_protocols.items()
        ]

# Test protocols
async def test_protocols():
    """Test agent protocols"""

    # Create message bus
    message_bus = MessageBus()
    await message_bus.start()

    # Create protocol managers
    manager1 = ProtocolManager("manager_001", message_bus)
    contractor1 = ProtocolManager("contractor_001", message_bus)
    contractor2 = ProtocolManager("contractor_002", message_bus)

    print("Testing agent protocols...\n")

    # Test contract net protocol
    print("1. Testing Contract Net Protocol:")
    participants = ["contractor_001", "contractor_002"]
    protocol_id = await manager1.start_protocol(ProtocolType.CONTRACT_NET, participants)
    print(f"Started contract net protocol: {protocol_id}")

    # Simulate bid submissions
    bids = [
        {
            'contractor_id': 'contractor_001',
            'price': 100,
            'delivery_time': 7,
            'quality': 'high',
            'score': 0.9
        },
        {
            'contractor_id': 'contractor_002',
            'price': 95,
            'delivery_time': 8,
            'quality': 'medium',
            'score': 0.8
        }
    ]

    for bid in bids:
        await message_bus.send_direct(
            bid['contractor_id'],
            "manager_001",
            MessageType.REQUEST,
            {
                'protocol': 'contract_net',
                'action': 'submit_bid',
                'bid': bid
            }
        )

    # Wait for processing
    await asyncio.sleep(0.5)

    # Check protocol status
    active_protocols = manager1.get_active_protocols()
    print(f"Active protocols: {len(active_protocols)}")
    for protocol in active_protocols:
        print(f"  Protocol {protocol['protocol_id']}: {protocol['state']}")

    print(f"\nMessage Bus Statistics:")
    stats = message_bus.get_message_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Clean up
    await message_bus.stop()

if __name__ == "__main__":
    asyncio.run(test_protocols())
```

This practice file provides comprehensive hands-on exercises covering:

1. **Environment Setup** - Complete development environment with all dependencies
2. **Basic Agent Implementation** - Simple reflex agents and goal-oriented agents
3. **Agent Communication Systems** - Message bus implementation and communication protocols
4. **Multi-Agent Coordination** - Various coordination mechanisms and consensus protocols
5. **Agent Learning and Adaptation** - Reinforcement learning and social learning implementations
6. **Modern Agent Frameworks** - LangChain, AutoGen, and CrewAI integrations
7. **Advanced Agent Patterns** - Emergent behavior and swarm intelligence
8. **Agent Safety and Ethics** - Safety constraints and ethical decision-making
9. **Real-World Applications** - Practical examples and use cases
10. **Mini-Projects** - Comprehensive projects for hands-on practice

Each section includes working code examples, test functions, and practical implementations that students can run and modify to learn by doing. The content is designed to provide both theoretical understanding and practical skills in AI agent development.
