# AI Agents Theory

**Version:** 1.0 | **Date:** November 2025

## Table of Contents

1. [Introduction to AI Agents](#introduction)
2. [Agent Architectures and Design Patterns](#architectures)
3. [Multi-Agent Systems Theory](#multi-agent-systems)
4. [Agent Communication and Coordination](#communication)
5. [Agent Learning and Adaptation](#learning)
6. [Agent Frameworks and Technologies](#frameworks)
7. [Advanced Agent Concepts](#advanced-concepts)
8. [Agent Safety and Ethics](#safety-ethics)
9. [Agent Applications and Use Cases](#applications)
10. [Future Directions in AI Agents](#future-directions)

---

## Introduction to AI Agents {#introduction}

### What are AI Agents?

AI agents are autonomous software systems that can perceive their environment, make decisions, and take actions to achieve specific goals. Unlike traditional software that follows predetermined instructions, AI agents can:

- **Perceive** their environment through sensors or data inputs
- **Reason** about their current state and available options
- **Plan** optimal strategies to achieve their objectives
- **Act** on their environment through various tools and interfaces
- **Learn** from experiences and adapt their behavior over time
- **Communicate** with other agents or human users

### Core Components of AI Agents

#### 1. Perception Module

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class SensorData:
    """Standard format for sensor inputs"""
    source: str
    data: Any
    timestamp: float
    confidence: float = 1.0

class PerceptionModule(ABC):
    """Abstract base class for agent perception"""

    @abstractmethod
    def perceive(self, environment: Dict) -> List[SensorData]:
        """Process environmental inputs into structured data"""
        pass

    @abstractmethod
    def filter_noise(self, data: List[SensorData]) -> List[SensorData]:
        """Remove irrelevant or low-confidence data"""
        pass
```

#### 2. Memory and Knowledge Base

```python
from datetime import datetime
from typing import Optional
import json

class AgentMemory:
    """Episodic and semantic memory for AI agents"""

    def __init__(self, max_size: int = 10000):
        self.episodic_memory = []  # Timeline of experiences
        self.semantic_memory = {}  # Facts and concepts
        self.working_memory = {}   # Current context
        self.max_size = max_size

    def store_experience(self, experience: Dict):
        """Store episodic memory"""
        experience['timestamp'] = datetime.now()
        self.episodic_memory.append(experience)

        # Manage memory size
        if len(self.episodic_memory) > self.max_size:
            self.episodic_memory.pop(0)

    def recall(self, query: str) -> List[Dict]:
        """Retrieve relevant memories"""
        # Simple keyword matching - can be enhanced with embeddings
        relevant = []
        for exp in self.episodic_memory:
            if any(keyword in str(exp).lower() for keyword in query.lower().split()):
                relevant.append(exp)
        return relevant

    def update_semantic_knowledge(self, fact: str, confidence: float = 1.0):
        """Update semantic memory with new facts"""
        self.semantic_memory[fact] = {
            'confidence': confidence,
            'last_updated': datetime.now()
        }
```

#### 3. Decision Making Engine

```python
from enum import Enum
from typing import Callable

class DecisionStrategy(Enum):
    RULE_BASED = "rule_based"
    UTILITY_BASED = "utility_based"
    GOAL_BASED = "goal_based"
    LEARNING_BASED = "learning_based"

class DecisionEngine:
    """Core decision-making system for AI agents"""

    def __init__(self, strategy: DecisionStrategy):
        self.strategy = strategy
        self.rules = []
        self.utilities = {}
        self.goals = []
        self.learning_model = None

    def add_rule(self, condition: Callable, action: Callable):
        """Add rule-based decision rule"""
        self.rules.append((condition, action))

    def set_utilities(self, utilities: Dict[str, float]):
        """Set utility values for outcomes"""
        self.utilities.update(utilities)

    def add_goal(self, goal: str, priority: float = 1.0):
        """Add goal with priority"""
        self.goals.append({
            'description': goal,
            'priority': priority,
            'status': 'active'
        })

    def decide(self, state: Dict) -> str:
        """Make decision based on current strategy"""
        if self.strategy == DecisionStrategy.RULE_BASED:
            return self._rule_based_decision(state)
        elif self.strategy == DecisionStrategy.UTILITY_BASED:
            return self._utility_based_decision(state)
        elif self.strategy == DecisionStrategy.GOAL_BASED:
            return self._goal_based_decision(state)
        else:
            return self._learning_based_decision(state)
```

#### 4. Action Execution System

```python
import asyncio
from typing import Protocol

class Action(Protocol):
    """Protocol for agent actions"""
    def execute(self, agent_context: Dict) -> Dict:
        """Execute action and return result"""
        ...

class ActionExecutor:
    """System for executing agent actions"""

    def __init__(self):
        self.action_library = {}
        self.execution_history = []

    def register_action(self, name: str, action: Action):
        """Register new action type"""
        self.action_library[name] = action

    async def execute_action(self, action_name: str, params: Dict, context: Dict) -> Dict:
        """Execute action asynchronously"""
        if action_name not in self.action_library:
            raise ValueError(f"Unknown action: {action_name}")

        try:
            action = self.action_library[action_name]
            result = await action.execute(context | params)

            # Log execution
            self.execution_history.append({
                'action': action_name,
                'params': params,
                'result': result,
                'timestamp': datetime.now()
            })

            return result
        except Exception as e:
            return {'error': str(e), 'success': False}
```

### Agent Taxonomy

#### By Autonomy Level

1. **Reactive Agents**: Respond immediately to environmental stimuli
2. **Deliberative Agents**: Plan and reason before acting
3. **Hybrid Agents**: Combine reactive and deliberative approaches
4. **Learning Agents**: Continuously improve their behavior

#### By Environment Type

1. **Discrete Environments**: Finite state spaces (chess, puzzles)
2. **Continuous Environments**: Infinite state spaces (robotics, trading)
3. **Multi-Agent Environments**: Other agents present
4. **Stochastic Environments**: Uncertainty in outcomes

#### By Learning Capability

1. **Simple Reflex Agents**: Rule-based responses
2. **Model-Based Agents**: Maintain internal world model
3. **Goal-Based Agents**: Work towards specific objectives
4. **Utility-Based Agents**: Optimize for utility functions

---

## Agent Architectures and Design Patterns {#architectures}

### 1. Subsumption Architecture

```python
class SubsumptionLayer:
    """Individual layer in subsumption architecture"""

    def __init__(self, priority: int, behavior: Callable):
        self.priority = priority
        self.behavior = behavior
        self.active = True

    def should_suppress_lower_layers(self, input_data: Dict) -> bool:
        """Determine if this layer should take control"""
        return self.behavior(input_data) is not None

    def process(self, input_data: Dict, timestamp: float) -> Optional[Dict]:
        """Process input and potentially generate output"""
        if self.active:
            return self.behavior(input_data)
        return None

class SubsumptionArchitecture:
    """Hierarchical agent architecture with behavior arbitration"""

    def __init__(self):
        self.layers = []
        self.current_behavior = None

    def add_layer(self, layer: SubsumptionLayer):
        """Add behavior layer (higher priority = higher in hierarchy)"""
        self.layers.append(layer)
        self.layers.sort(key=lambda x: x.priority, reverse=True)

    def process_input(self, input_data: Dict) -> Optional[Dict]:
        """Process input through architecture layers"""
        for layer in self.layers:
            if layer.should_suppress_lower_layers(input_data):
                self.current_behavior = layer
                return layer.process(input_data, time.time())
        return None
```

### 2. Belief-Desire-Intention (BDI) Architecture

```python
from typing import Set, List
from dataclasses import dataclass

@dataclass
class Belief:
    """Agent's belief about the world"""
    proposition: str
    confidence: float
    source: str
    timestamp: float

@dataclass
class Desire:
    """Agent's goals and objectives"""
    goal: str
    priority: float
    status: str  # 'active', 'completed', 'abandoned'

@dataclass
class Intention:
    """Agent's committed plans"""
    plan: str
    steps: List[Dict]
    current_step: int
    status: str  # 'active', 'completed', 'failed'

class BDIAgent:
    """Belief-Desire-Intention agent implementation"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.beliefs = []
        self.desires = []
        self.intentions = []
        self.intention_stack = []

    def update_belief(self, proposition: str, confidence: float, source: str):
        """Update agent's beliefs about the world"""
        # Remove conflicting beliefs
        self.beliefs = [b for b in self.beliefs if b.proposition != proposition]

        # Add new belief
        self.beliefs.append(Belief(
            proposition=proposition,
            confidence=confidence,
            source=source,
            timestamp=time.time()
        ))

    def adopt_desire(self, goal: str, priority: float = 1.0):
        """Adopt new goal"""
        self.desires.append(Desire(goal, priority, 'active'))

    def form_intention(self, goal: str, plan: str, steps: List[Dict]):
        """Form intention to achieve goal"""
        self.intentions.append(Intention(plan, steps, 0, 'active'))

    def reconsider(self):
        """Reconsider current intentions based on new beliefs"""
        # Check if desires are still achievable
        for desire in self.desires:
            if desire.status == 'active':
                # Use beliefs to assess feasibility
                relevant_beliefs = [b for b in self.beliefs
                                  if desire.goal.lower() in b.proposition.lower()]

                if not relevant_beliefs or max(b.confidence for b in relevant_beliefs) < 0.5:
                    desire.status = 'abandoned'

    def execute_intention(self):
        """Execute current intention"""
        if not self.intentions:
            return

        current_intention = self.intentions[0]
        if current_intention.current_step < len(current_intention.steps):
            # Execute next step
            step = current_intention.steps[current_intention.current_step]
            # ... execute step logic
            current_intention.current_step += 1

            if current_intention.current_step >= len(current_intention.steps):
                current_intention.status = 'completed'
```

### 3. Blackboard Architecture

```python
import threading
from queue import Queue, Empty
from typing import Dict, List, Any

class KnowledgeSource:
    """Source of knowledge in blackboard system"""

    def __init__(self, name: str):
        self.name = name
        self.knowledge_area = None
        self.enabled = True

    def contribute_knowledge(self, blackboard: 'Blackboard') -> List[Dict]:
        """Add knowledge to blackboard"""
        if not self.enabled:
            return []

        # Analyze current blackboard state
        current_state = blackboard.get_current_state()

        # Generate new knowledge
        new_knowledge = self._generate_knowledge(current_state)

        return new_knowledge

    def _generate_knowledge(self, state: Dict) -> List[Dict]:
        """Generate new knowledge (to be implemented by subclasses)"""
        return []

class Blackboard:
    """Shared knowledge repository for multiple agents"""

    def __init__(self):
        self.knowledge_areas = {}
        self.event_queue = Queue()
        self.knowledge_sources = []
        self.lock = threading.Lock()
        self.subscribers = []

    def register_knowledge_source(self, ks: KnowledgeSource):
        """Register new knowledge source"""
        self.knowledge_sources.append(ks)

    def post_knowledge(self, area: str, knowledge: Dict):
        """Add knowledge to specific area"""
        with self.lock:
            if area not in self.knowledge_areas:
                self.knowledge_areas[area] = []

            self.knowledge_areas[area].append({
                'content': knowledge,
                'timestamp': time.time(),
                'source': 'unknown'
            })

            # Notify subscribers
            for callback in self.subscribers:
                try:
                    callback(area, knowledge)
                except Exception as e:
                    print(f"Subscriber callback error: {e}")

    def read_knowledge(self, area: str) -> List[Dict]:
        """Read knowledge from specific area"""
        with self.lock:
            return self.knowledge_areas.get(area, [])

    def get_current_state(self) -> Dict:
        """Get current state of blackboard"""
        with self.lock:
            return self.knowledge_areas.copy()

    def run_cycle(self):
        """Execute one cycle of blackboard system"""
        for ks in self.knowledge_sources:
            if ks.enabled:
                new_knowledge = ks.contribute_knowledge(self)
                for knowledge in new_knowledge:
                    self.post_knowledge(ks.knowledge_area, knowledge)
```

---

## Multi-Agent Systems Theory {#multi-agent-systems}

### 1. Agent Communication Protocols

#### FIPA ACL (Agent Communication Language)

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional

class Performative(Enum):
    REQUEST = "request"
    INFORM = "inform"
    QUERY = "query"
    PROPOSE = "propose"
    ACCEPT_PROPOSAL = "accept-proposal"
    REJECT_PROPOSAL = "reject-proposal"
    CONFIRM = "confirm"
    DISCONFIRM = "disconfirm"

@dataclass
class ACLMessage:
    """FIPA ACL message format"""
    performative: Performative
    sender: str
    receiver: str
    content: str
    conversation_id: str
    reply_to: Optional[str] = None
    reply_with: Optional[str] = None
    in_reply_to: Optional[str] = None
    reply_by: Optional[float] = None
    encoding: str = "string"
    language: str = "fipa-sl"
    ontology: str = "default"
    protocol: str = "fipa-request"
    conversation_state: str = "not-started"

class CommunicationAgent:
    """Agent with communication capabilities"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.message_queue = []
        self.conversation_history = {}
        self.communication_partners = set()

    def send_message(self, receiver: str, performative: Performative,
                    content: str, conversation_id: str) -> ACLMessage:
        """Send message to another agent"""
        message = ACLMessage(
            performative=performative,
            sender=self.id,
            receiver=receiver,
            content=content,
            conversation_id=conversation_id
        )

        # Add to conversation history
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        self.conversation_history[conversation_id].append(message)

        return message

    def receive_message(self, message: ACLMessage):
        """Receive message from another agent"""
        self.message_queue.append(message)

        # Update conversation history
        if message.conversation_id not in self.conversation_history:
            self.conversation_history[message.conversation_id] = []
        self.conversation_history[message.conversation_id].append(message)

        # Process message
        self._process_message(message)

    def _process_message(self, message: ACLMessage):
        """Process received message"""
        if message.performative == Performative.REQUEST:
            self._handle_request(message)
        elif message.performative == Performative.INFORM:
            self._handle_inform(message)
        elif message.performative == Performative.QUERY:
            self._handle_query(message)
        # ... handle other performatives

    def _handle_request(self, message: ACLMessage):
        """Handle request message"""
        # Implement request handling logic
        pass

    def _handle_inform(self, message: ACLMessage):
        """Handle inform message"""
        # Update knowledge base with new information
        pass
```

### 2. Coordination Mechanisms

#### Contract Net Protocol

```python
from typing import List, Dict, Tuple
from enum import Enum

class TaskState(Enum):
    OPEN = "open"
    ANNOUNCED = "announced"
    BIDDING = "bidding"
    AWARDED = "awarded"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TaskAnnouncement:
    """Task announcement in contract net"""
    task_id: str
    description: str
    eligibility_criteria: List[str]
    deadline: float
    manager_id: str

@dataclass
class Bid:
    """Bid submitted by contractor"""
    task_id: str
    contractor_id: str
    proposal: str
    cost_estimate: float
    completion_time: float

class ContractNetManager:
    """Manager agent in contract net protocol"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.open_tasks = {}
        self.received_bids = {}
        self.active_contracts = {}

    def announce_task(self, task: TaskAnnouncement):
        """Announce task to potential contractors"""
        self.open_tasks[task.task_id] = task

        # Broadcast to all contractors
        contractors = self._get_potential_contractors(task.eligibility_criteria)
        for contractor_id in contractors:
            self._send_task_announcement(contractor_id, task)

    def receive_bid(self, bid: Bid):
        """Receive bid from contractor"""
        if bid.task_id not in self.received_bids:
            self.received_bids[bid.task_id] = []

        self.received_bids[bid.task_id].append(bid)

    def award_contract(self, task_id: str, winning_bid: Bid):
        """Award contract to winning bidder"""
        task = self.open_tasks[task_id]

        # Remove other bids
        if task_id in self.received_bids:
            for bid in self.received_bids[task_id]:
                if bid.contractor_id != winning_bid.contractor_id:
                    self._send_rejection(bid.contractor_id, task_id)

        # Send acceptance
        self._send_acceptance(winning_bid.contractor_id, task_id, winning_bid)

        # Update state
        self.open_tasks[task_id].state = TaskState.AWARDED
        self.active_contracts[task_id] = winning_bid

    def _get_potential_contractors(self, criteria: List[str]) -> List[str]:
        """Get list of contractors meeting criteria"""
        # Implementation depends on contractor registry
        return []

    def _send_task_announcement(self, contractor_id: str, task: TaskAnnouncement):
        """Send task announcement to contractor"""
        # Implementation depends on communication system
        pass

class ContractNetContractor:
    """Contractor agent in contract net protocol"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.capabilities = []
        self.active_bids = {}
        self.completed_tasks = []

    def receive_task_announcement(self, task: TaskAnnouncement):
        """Receive task announcement from manager"""
        if self._can_handle_task(task):
            bid = self._create_bid(task)
            self._submit_bid(bid)

    def _can_handle_task(self, task: TaskAnnouncement) -> bool:
        """Check if agent can handle the task"""
        # Check if all eligibility criteria are met
        for criterion in task.eligibility_criteria:
            if criterion not in self.capabilities:
                return False
        return True

    def _create_bid(self, task: TaskAnnouncement) -> Bid:
        """Create bid for task"""
        # Estimate cost and completion time
        cost_estimate = self._estimate_cost(task)
        completion_time = self._estimate_time(task)

        return Bid(
            task_id=task.task_id,
            contractor_id=self.id,
            proposal=self._create_proposal(task),
            cost_estimate=cost_estimate,
            completion_time=completion_time
        )

    def _estimate_cost(self, task: TaskAnnouncement) -> float:
        """Estimate cost of completing task"""
        # Implementation depends on agent's pricing model
        return 100.0  # Placeholder

    def _estimate_time(self, task: TaskAnnouncement) -> float:
        """Estimate time to complete task"""
        # Implementation depends on agent's performance
        return 3600.0  # Placeholder (1 hour)

    def _create_proposal(self, task: TaskAnnouncement) -> str:
        """Create detailed proposal"""
        return f"Proposal for task {task.task_id}"

    def _submit_bid(self, bid: Bid):
        """Submit bid to manager"""
        # Implementation depends on communication system
        pass
```

### 3. Coordination Through Mediation

```python
class MediatorAgent:
    """Mediating agent for coordinating multiple agents"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.mediated_agents = set()
        self.coordination_rules = []
        self.global_state = {}

    def register_agent(self, agent_id: str, capabilities: List[str]):
        """Register agent with mediator"""
        self.mediated_agents.add(agent_id)
        self.global_state[agent_id] = {
            'capabilities': capabilities,
            'status': 'idle',
            'current_task': None
        }

    def coordinate_action(self, action: Dict) -> Dict:
        """Coordinate action among multiple agents"""
        # Analyze action requirements
        required_capabilities = self._extract_required_capabilities(action)

        # Find suitable agents
        suitable_agents = self._find_suitable_agents(required_capabilities)

        if not suitable_agents:
            return {'success': False, 'reason': 'No suitable agents found'}

        # Apply coordination rules
        selected_agents = self._apply_coordination_rules(suitable_agents, action)

        # Distribute subtasks
        subtask_plan = self._create_subtask_plan(action, selected_agents)

        # Execute plan
        results = self._execute_coordinated_plan(subtask_plan)

        return results

    def _extract_required_capabilities(self, action: Dict) -> List[str]:
        """Extract capabilities required for action"""
        # Implementation depends on action type
        return action.get('required_capabilities', [])

    def _find_suitable_agents(self, required_capabilities: List[str]) -> List[str]:
        """Find agents with required capabilities"""
        suitable_agents = []

        for agent_id, agent_state in self.global_state.items():
            if agent_state['status'] == 'idle':
                if all(cap in agent_state['capabilities']
                      for cap in required_capabilities):
                    suitable_agents.append(agent_id)

        return suitable_agents

    def _apply_coordination_rules(self, agents: List[str], action: Dict) -> List[str]:
        """Apply coordination rules to select agents"""
        # Apply load balancing, priority, etc.
        return agents  # Simplified - select all for now

    def _create_subtask_plan(self, action: Dict, agents: List[str]) -> Dict:
        """Create plan for coordinated execution"""
        return {
            'main_action': action,
            'subtasks': [
                {
                    'subtask_id': f"{action['id']}_sub_{i}",
                    'assigned_agent': agent,
                    'description': f"Part {i} of {action['description']}"
                }
                for i, agent in enumerate(agents)
            ]
        }
```

---

## Agent Communication and Coordination {#communication}

### 1. Message Passing Systems

```python
import asyncio
import json
from typing import Dict, List, Callable, Any
from datetime import datetime
from dataclasses import dataclass

@dataclass
class AgentMessage:
    """Standard message format for agent communication"""
    sender_id: str
    receiver_id: str
    message_type: str
    content: Any
    timestamp: float
    conversation_id: str
    message_id: str
    reply_to: Optional[str] = None
    metadata: Dict = None

class MessageBus:
    """Centralized message bus for agent communication"""

    def __init__(self):
        self.subscribers = {}  # agent_id -> [message_types]
        self.message_queue = asyncio.Queue()
        self.delivered_messages = {}

    def subscribe(self, agent_id: str, message_types: List[str],
                  callback: Callable[[AgentMessage], None]):
        """Subscribe agent to message types"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = {}

        for msg_type in message_types:
            self.subscribers[agent_id][msg_type] = callback

    async def publish(self, message: AgentMessage):
        """Publish message to message bus"""
        await self.message_queue.put(message)

        # Process immediately if there are subscribers
        await self._process_message(message)

    async def _process_message(self, message: AgentMessage):
        """Process message and deliver to subscribers"""
        # Check if receiver is subscribed to this message type
        if (message.receiver_id in self.subscribers and
            message.message_type in self.subscribers[message.receiver_id]):

            callback = self.subscribers[message.receiver_id][message.message_type]
            try:
                await callback(message)
            except Exception as e:
                print(f"Error delivering message {message.message_id}: {e}")

        # Store in delivered messages
        if message.receiver_id not in self.delivered_messages:
            self.delivered_messages[message.receiver_id] = []
        self.delivered_messages[message.receiver_id].append(message)

class CommunicatingAgent:
    """Base class for agents with communication capabilities"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.id = agent_id
        self.message_bus = message_bus
        self.conversation_handlers = {}
        self.message_handlers = {}

        # Subscribe to own messages
        self.message_bus.subscribe(self.id, ['direct'], self._handle_direct_message)

    def register_conversation_handler(self, conversation_id: str,
                                    handler: Callable[[AgentMessage], None]):
        """Register handler for specific conversation"""
        self.conversation_handlers[conversation_id] = handler

    def register_message_handler(self, message_type: str,
                               handler: Callable[[AgentMessage], None]):
        """Register handler for message type"""
        self.message_handlers[message_type] = handler

    async def send_message(self, receiver_id: str, message_type: str,
                         content: Any, conversation_id: str = None) -> AgentMessage:
        """Send message to another agent"""
        message = AgentMessage(
            sender_id=self.id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.now().timestamp(),
            conversation_id=conversation_id or f"conv_{self.id}_{receiver_id}",
            message_id=f"msg_{self.id}_{datetime.now().timestamp()}"
        )

        await self.message_bus.publish(message)
        return message

    async def _handle_direct_message(self, message: AgentMessage):
        """Handle direct messages"""
        # Route to conversation handler if available
        if message.conversation_id in self.conversation_handlers:
            await self.conversation_handlers[message.conversation_id](message)
        # Route to message type handler if available
        elif message.message_type in self.message_handlers:
            await self.message_handlers[message.message_type](message)
        # Default handler
        else:
            await self._default_message_handler(message)

    async def _default_message_handler(self, message: AgentMessage):
        """Default message handler"""
        print(f"Agent {self.id} received message: {message.message_type}")
```

### 2. Distributed Consensus

```python
import random
from typing import List, Set
from enum import Enum

class ConsensusState(Enum):
    IDLE = "idle"
    PROPOSING = "proposing"
    VOTING = "voting"
    COMMITTED = "committed"
    ABORTED = "aborted"

@dataclass
class ConsensusProposal:
    """Proposal for distributed consensus"""
    proposal_id: str
    proposer_id: str
    value: Any
    round: int
    timestamp: float

@dataclass
class ConsensusVote:
    """Vote in consensus protocol"""
    proposal_id: str
    voter_id: str
    vote: bool  # True = accept, False = reject
    round: int
    timestamp: float

class ConsensusAgent:
    """Agent implementing distributed consensus protocol"""

    def __init__(self, agent_id: str, peers: List[str], quorum_size: int):
        self.id = agent_id
        self.peers = peers
        self.quorum_size = quorum_size
        self.current_proposal = None
        self.votes_received = {}
        self.proposals_received = {}
        self.decision = None
        self.state = ConsensusState.IDLE

    async def propose_value(self, value: Any) -> bool:
        """Propose value for consensus"""
        if self.state != ConsensusState.IDLE:
            return False

        self.current_proposal = ConsensusProposal(
            proposal_id=f"prop_{self.id}_{time.time()}",
            proposer_id=self.id,
            value=value,
            round=1,
            timestamp=time.time()
        )

        self.state = ConsensusState.PROPOSING
        self.votes_received[self.current_proposal.proposal_id] = []

        # Broadcast proposal to all peers
        await self._broadcast_proposal(self.current_proposal)
        return True

    async def receive_proposal(self, proposal: ConsensusProposal):
        """Receive consensus proposal"""
        if (proposal.proposal_id in self.proposals_received or
            self.state == ConsensusState.COMMITTED):
            return

        self.proposals_received[proposal.proposal_id] = proposal

        # Vote on proposal
        vote = await self._evaluate_proposal(proposal)
        await self._send_vote(proposal, vote)

        # If we have a proposal and received enough votes, decide
        if self.current_proposal:
            await self._check_decision(proposal.proposal_id)

    async def receive_vote(self, vote: ConsensusVote):
        """Receive vote from peer"""
        if vote.proposal_id not in self.votes_received:
            self.votes_received[vote.proposal_id] = []

        self.votes_received[vote.proposal_id].append(vote)
        await self._check_decision(vote.proposal_id)

    async def _evaluate_proposal(self, proposal: ConsensusProposal) -> bool:
        """Evaluate proposal and return vote"""
        # Simple evaluation - in practice, more sophisticated logic
        return True  # Accept all proposals for now

    async def _broadcast_proposal(self, proposal: ConsensusProposal):
        """Broadcast proposal to all peers"""
        for peer_id in self.peers:
            if peer_id != self.id:
                await self._send_proposal(peer_id, proposal)

    async def _send_vote(self, proposal: ConsensusProposal, vote: bool):
        """Send vote to all peers"""
        vote_msg = ConsensusVote(
            proposal_id=proposal.proposal_id,
            voter_id=self.id,
            vote=vote,
            round=proposal.round,
            timestamp=time.time()
        )

        for peer_id in self.peers:
            if peer_id != self.id:
                await self._send_vote_msg(peer_id, vote_msg)

    async def _check_decision(self, proposal_id: str):
        """Check if consensus can be reached"""
        if proposal_id not in self.votes_received:
            return

        votes = self.votes_received[proposal_id]
        accepting_votes = sum(1 for v in votes if v.vote)

        # Check if we have quorum
        if accepting_votes >= self.quorum_size:
            self.state = ConsensusState.COMMITTED
            if proposal_id == self.current_proposal.proposal_id:
                self.decision = self.current_proposal.value
            else:
                # Our proposal lost, use winning proposal
                if proposal_id in self.proposals_received:
                    self.decision = self.proposals_received[proposal_id].value
```

---

## Agent Learning and Adaptation {#learning}

### 1. Reinforcement Learning for Agents

```python
import numpy as np
from collections import defaultdict, deque
import random

class QLearningAgent:
    """Q-Learning agent for discrete environments"""

    def __init__(self, state_space_size: int, action_space_size: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.experience_buffer = deque(maxlen=10000)

    def select_action(self, state: int, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state: int, action: int, reward: float,
             next_state: int, done: bool):
        """Update Q-values using Q-learning"""
        # Store experience
        self.experience_buffer.append((state, action, reward, next_state, done))

        # Q-learning update
        current_q = self.q_table[state, action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])

        # Update Q-value
        self.q_table[state, action] += self.learning_rate * (target_q - current_q)

        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def replay_experience(self, batch_size: int = 32):
        """Replay experiences for better learning"""
        if len(self.experience_buffer) < batch_size:
            return

        batch = random.sample(self.experience_buffer, batch_size)

        for state, action, reward, next_state, done in batch:
            current_q = self.q_table[state, action]

            if done:
                target_q = reward
            else:
                target_q = reward + self.discount_factor * np.max(self.q_table[next_state])

            self.q_table[state, action] += self.learning_rate * (target_q - current_q)

class PolicyGradientAgent:
    """Policy gradient agent for continuous action spaces"""

    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 0.001, hidden_size: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # Initialize policy network
        self.policy_network = self._build_network(hidden_size)
        self.value_network = self._build_network(hidden_size)

        # Learning metrics
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []

    def _build_network(self, hidden_size: int):
        """Build neural network for policy/value function"""
        # Simplified - in practice use proper deep learning framework
        return {
            'weights1': np.random.randn(self.state_dim, hidden_size) * 0.1,
            'bias1': np.zeros(hidden_size),
            'weights2': np.random.randn(hidden_size, self.action_dim) * 0.1,
            'bias2': np.zeros(self.action_dim)
        }

    def forward(self, state: np.ndarray, network: dict) -> np.ndarray:
        """Forward pass through network"""
        hidden = np.maximum(0, np.dot(state, network['weights1']) + network['bias1'])
        output = np.dot(hidden, network['weights2']) + network['bias2']
        return output

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using policy network"""
        logits = self.forward(state, self.policy_network)

        if training:
            # Add noise for exploration
            noise = np.random.normal(0, 0.1, size=logits.shape)
            action = logits + noise
        else:
            action = logits

        return action

    def compute_loss(self, states: np.ndarray, actions: np.ndarray,
                    rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray):
        """Compute policy and value function losses"""
        batch_size = len(states)

        # Compute policy loss (simplified)
        actions_log_prob = -0.5 * np.sum((actions - np.mean(actions, axis=0))**2, axis=1)
        returns = self._compute_returns(rewards, dones)

        policy_loss = -np.mean(actions_log_prob * returns)

        # Compute value loss
        current_values = np.array([self._compute_value(state, self.value_network)
                                 for state in states])
        value_loss = np.mean((current_values - returns)**2)

        return policy_loss, value_loss

    def _compute_returns(self, rewards: np.ndarray, dones: np.ndarray,
                        gamma: float = 0.99) -> np.ndarray:
        """Compute discounted returns"""
        returns = np.zeros_like(rewards)
        running_return = 0

        for i in reversed(range(len(rewards))):
            running_return = rewards[i] + gamma * running_return * (1 - dones[i])
            returns[i] = running_return

        return returns

    def _compute_value(self, state: np.ndarray, network: dict) -> float:
        """Compute state value"""
        return np.mean(self.forward(state, network))
```

### 2. Social Learning and Imitation

```python
class ImitationLearningAgent:
    """Agent that learns from observing other agents"""

    def __init__(self, agent_id: str, demonstration_buffer_size: int = 1000):
        self.id = agent_id
        self.demonstration_buffer = deque(maxlen=demonstration_buffer_size)
        self.behavior_model = None
        self.observation_history = []

    def observe_agent(self, demonstrator_id: str, state: Dict, action: Dict,
                     outcome: Dict):
        """Record observation of another agent's behavior"""
        observation = {
            'demonstrator': demonstrator_id,
            'state': state,
            'action': action,
            'outcome': outcome,
            'timestamp': time.time()
        }

        self.observation_history.append(observation)

        # Store in demonstration buffer
        self.demonstration_buffer.append(observation)

    def learn_from_demonstrations(self, batch_size: int = 32):
        """Learn behavior model from demonstrations"""
        if len(self.demonstration_buffer) < batch_size:
            return

        batch = random.sample(list(self.demonstration_buffer), batch_size)

        # Extract features and labels
        states = []
        actions = []

        for demo in batch:
            states.append(self._extract_state_features(demo['state']))
            actions.append(demo['action'])

        # Train behavior model (simplified)
        self.behavior_model = self._train_behavior_model(states, actions)

    def _extract_state_features(self, state: Dict) -> np.ndarray:
        """Extract features from state for learning"""
        # Simplified feature extraction
        return np.array(list(state.values()))

    def _train_behavior_model(self, states: List[np.ndarray],
                            actions: List[Dict]) -> Dict:
        """Train model to predict actions from states"""
        # Simplified training - in practice use proper ML framework
        X = np.array(states)
        y = np.array([list(action.values()) for action in actions])

        # Simple linear model
        weights = np.linalg.lstsq(X, y, rcond=None)[0]

        return {'weights': weights}

    def predict_action(self, state: Dict) -> Dict:
        """Predict action using learned behavior model"""
        if self.behavior_model is None:
            return self._default_action(state)

        state_features = self._extract_state_features(state)

        # Predict action using learned model
        predicted_values = np.dot(state_features, self.behavior_model['weights'])

        # Convert back to action format
        action_keys = list(self.demonstration_buffer[0]['action'].keys())
        predicted_action = dict(zip(action_keys, predicted_values))

        return predicted_action

    def _default_action(self, state: Dict) -> Dict:
        """Default action when no model is available"""
        # Return safe default action
        return {key: 0 for key in state.keys()}

class SocialLearningNetwork:
    """Network of agents that learn from each other"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.peers = {}
        self.knowledge_base = {}
        self.social_network = set()
        self.reputation_scores = {}

    def connect_to_peer(self, peer_id: str, initial_reputation: float = 0.5):
        """Establish connection to peer agent"""
        self.peers[peer_id] = {
            'connection_established': time.time(),
            'last_interaction': None,
            'knowledge_shared': 0,
            'interactions_count': 0
        }

        self.social_network.add(peer_id)
        self.reputation_scores[peer_id] = initial_reputation

    def share_knowledge(self, peer_id: str, knowledge: Dict) -> bool:
        """Share knowledge with peer agent"""
        if peer_id not in self.peers:
            return False

        # Update knowledge base
        knowledge['shared_by'] = self.id
        knowledge['shared_at'] = time.time()
        self.knowledge_base[f"shared_{len(self.knowledge_base)}"] = knowledge

        # Update peer statistics
        self.peers[peer_id]['knowledge_shared'] += 1
        self.peers[peer_id]['last_interaction'] = time.time()

        return True

    def learn_from_peer(self, peer_id: str, knowledge: Dict) -> bool:
        """Learn knowledge from peer agent"""
        if peer_id not in self.peers:
            return False

        # Evaluate knowledge quality
        quality_score = self._evaluate_knowledge_quality(knowledge)

        # Update peer reputation based on knowledge quality
        current_reputation = self.reputation_scores[peer_id]
        new_reputation = current_reputation * 0.9 + quality_score * 0.1
        self.reputation_scores[peer_id] = new_reputation

        # Decide whether to accept knowledge
        if quality_score > 0.5:
            self.knowledge_base[f"learned_{len(self.knowledge_base)}"] = knowledge
            self.peers[peer_id]['interactions_count'] += 1
            return True

        return False

    def _evaluate_knowledge_quality(self, knowledge: Dict) -> float:
        """Evaluate quality of received knowledge"""
        # Simplified quality evaluation
        # In practice, use more sophisticated methods
        if 'accuracy' in knowledge:
            return knowledge['accuracy']
        elif 'source_reliability' in knowledge:
            return knowledge['source_reliability']
        else:
            return 0.5  # Default quality

    def get_trusted_peers(self, min_reputation: float = 0.7) -> List[str]:
        """Get list of trusted peer agents"""
        return [peer_id for peer_id, reputation in self.reputation_scores.items()
                if reputation >= min_reputation]
```

---

## Agent Frameworks and Technologies {#frameworks}

### 1. LangChain Agents

```python
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish

class CustomAgentTool:
    """Base class for custom agent tools"""

    def __init__(self, name: str, description: str, func: callable):
        self.name = name
        self.description = description
        self.func = func

    def run(self, **kwargs) -> str:
        """Execute tool function"""
        return self.func(**kwargs)

class LangChainAgent:
    """Agent implementation using LangChain framework"""

    def __init__(self, openai_api_key: str, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=0
        )

        self.tools = []
        self.agent = None
        self.agent_executor = None

    def add_tool(self, name: str, description: str, func: callable):
        """Add custom tool to agent"""
        tool = CustomAgentTool(name, description, func)
        self.tools.append(tool)

    def add_web_search_tool(self):
        """Add web search capability"""
        from langchain_community.tools import DuckDuckGoSearchRun

        search = DuckDuckGoSearchRun()
        self.tools.append(Tool(
            name="web_search",
            description="Search the web for information",
            func=search.run
        ))

    def add_calculator_tool(self):
        """Add calculator capability"""
        def calculate(expression: str) -> str:
            """Calculate mathematical expression"""
            try:
                result = eval(expression)
                return f"The result of {expression} is {result}"
            except Exception as e:
                return f"Error: {e}"

        self.tools.append(Tool(
            name="calculator",
            description="Calculate mathematical expressions",
            func=calculate
        ))

    def create_agent(self):
        """Create the agent with LangChain"""
        # Define prompt template
        prompt = PromptTemplate.from_template("""
        You are a helpful AI agent that can use tools to accomplish tasks.

        Available tools: {tools}

        User request: {input}

        Think step by step about how to solve this problem.
        Use the tools available to you when needed.
        If you need additional information, ask for it.

        {agent_scratchpad}
        """)

        # Create agent
        self.agent = create_openai_functions_agent(self.llm, self.tools, prompt)

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            max_execution_time=30
        )

    async def run(self, query: str) -> Dict[str, Any]:
        """Run agent with query"""
        if self.agent_executor is None:
            self.create_agent()

        try:
            result = await self.agent_executor.ainvoke({"input": query})
            return {
                "success": True,
                "result": result,
                "output": result.get("output", "No output generated")
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": None
            }

class ReActAgent(LangChainAgent):
    """ReAct (Reasoning and Acting) agent implementation"""

    def create_agent(self):
        """Create ReAct-style agent"""
        prompt = PromptTemplate.from_template("""
        You are an AI assistant with access to tools that can help you answer questions.

        You have access to the following tools:
        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of the tools
        Observation: the result of the action
        ... (this Thought/Action/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: your final answer to the original input question

        Question: {input}

        Think step by step and take action using tools when necessary.

        {agent_scratchpad}
        """)

        self.agent = create_openai_functions_agent(self.llm, self.tools, prompt)

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=20,
            max_execution_time=60
        )
```

### 2. AutoGen Multi-Agent Framework

```python
import autogen
from typing import Dict, List, Any, Optional

class AutoGenAgent:
    """Agent implementation using Microsoft AutoGen framework"""

    def __init__(self, name: str, llm_config: Dict, system_message: str = ""):
        self.name = name
        self.llm_config = llm_config
        self.system_message = system_message or f"You are {name}, a helpful AI assistant."

        # Create agent
        self.agent = autogen.AssistantAgent(
            name=name,
            llm_config=llm_config,
            system_message=self.system_message
        )

        # Initialize user proxy for tool execution
        self.user_proxy = autogen.UserProxyAgent(
            name=f"{name}_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10
        )

        self.conversation_history = []

    def setup_tool_execution(self, functions: List[callable]):
        """Setup function execution for the agent"""
        # Register functions as tools
        autogen.function_to_json(functions)

        # Update agent configuration to include functions
        self.agent.update_system_message(
            self.system_message + "\n\nYou have access to the following functions: " +
            str(functions)
        )

    def start_conversation(self, initial_message: str) -> Dict[str, Any]:
        """Start conversation with the agent"""
        try:
            # Initiate conversation
            self.user_proxy.initiate_chat(
                self.agent,
                message=initial_message
            )

            # Collect conversation history
            chat_history = self.user_proxy.chat_history

            self.conversation_history = chat_history

            return {
                "success": True,
                "conversation_id": f"{self.name}_{time.time()}",
                "history": chat_history,
                "final_message": chat_history[-1]["content"] if chat_history else ""
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "conversation_id": None,
                "history": [],
                "final_message": None
            }

class MultiAgentOrchestrator:
    """Orchestrator for coordinating multiple AutoGen agents"""

    def __init__(self, agents: List[AutoGenAgent]):
        self.agents = agents
        self.agent_groups = {}
        self.conversation_flows = {}

    def create_agent_group(self, group_name: str, agent_names: List[str]):
        """Create a group of agents that can collaborate"""
        selected_agents = [agent for agent in self.agents
                          if agent.name in agent_names]

        self.agent_groups[group_name] = {
            'agents': selected_agents,
            'conversation_history': [],
            'shared_context': {}
        }

    def coordinate_group_task(self, group_name: str, task: str) -> Dict[str, Any]:
        """Coordinate task execution among agent group"""
        if group_name not in self.agent_groups:
            return {"success": False, "error": "Group not found"}

        group = self.agent_groups[group_name]

        try:
            # Sequential task execution among group members
            current_context = {"task": task, "shared_context": {}}

            for agent in group['agents']:
                # Get agent's response
                response = agent.start_conversation(
                    f"Given context: {current_context}\n\nYour task: {task}"
                )

                if not response["success"]:
                    return response

                # Update shared context
                group['shared_context'][agent.name] = response["final_message"]
                current_context["shared_context"] = group['shared_context']

                # Add to conversation history
                group['conversation_history'].append({
                    'agent': agent.name,
                    'task': task,
                    'response': response["final_message"],
                    'timestamp': time.time()
                })

            return {
                "success": True,
                "group_name": group_name,
                "final_result": current_context["shared_context"],
                "conversation_history": group['conversation_history']
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "group_name": group_name
            }

    def create_fallback_chain(self, agent_names: List[str], task: str) -> Dict[str, Any]:
        """Create fallback chain where agents try sequentially"""
        for agent in self.agents:
            if agent.name in agent_names:
                response = agent.start_conversation(task)

                if response["success"]:
                    return {
                        "success": True,
                        "agent_used": agent.name,
                        "result": response["final_message"],
                        "conversation_history": response["history"]
                    }

        return {
            "success": False,
            "error": "All agents in fallback chain failed",
            "agent_names": agent_names
        }
```

### 3. CrewAI Framework Implementation

```python
from crewai import Agent, Task, Crew
from typing import Dict, List, Any

class CrewAIAgent:
    """Agent implementation using CrewAI framework"""

    def __init__(self, name: str, role: str, goal: str, backstory: str,
                 llm_model: str = "gpt-3.5-turbo"):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory

        # Create CrewAI agent
        self.agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=True,
            allow_delegation=True,
            max_iter=3,
            memory=True,
            cache=True
        )

        self.completed_tasks = []
        self.performance_metrics = {}

    def add_tool(self, tool_name: str, tool_description: str, tool_function: callable):
        """Add custom tool to agent"""
        # Define tool function for CrewAI
        def tool_wrapper(**kwargs):
            return tool_function(**kwargs)

        tool_wrapper.__name__ = tool_name
        tool_wrapper.__doc__ = tool_description

        # Add to agent
        self.agent.tools.append(tool_wrapper)

    def assign_task(self, description: str, expected_output: str = "",
                   agent_name: str = None) -> Task:
        """Assign task to agent"""
        task = Task(
            description=description,
            expected_output=expected_output,
            agent=self.agent if agent_name is None else None
        )

        return task

class CrewManager:
    """Manager for coordinating multiple CrewAI agents"""

    def __init__(self):
        self.agents = []
        self.crew = None
        self.tasks = []
        self.execution_log = []

    def add_agent(self, agent: CrewAIAgent):
        """Add agent to crew"""
        self.agents.append(agent.agent)

    def create_task(self, description: str, agent_name: str = None,
                   expected_output: str = "") -> Task:
        """Create task for execution"""
        if agent_name:
            # Assign to specific agent
            agent_obj = next((a for a in self.agents if a.role == agent_name), None)
            if agent_obj:
                task = Task(
                    description=description,
                    expected_output=expected_output,
                    agent=agent_obj
                )
            else:
                raise ValueError(f"Agent with name {agent_name} not found")
        else:
            # Auto-assign to best agent
            task = Task(
                description=description,
                expected_output=expected_output,
                agent=None  # Will be assigned by crew
            )

        self.tasks.append(task)
        return task

    def execute_crew(self, process: str = "sequential") -> Dict[str, Any]:
        """Execute tasks with crew"""
        try:
            # Create crew
            self.crew = Crew(
                agents=self.agents,
                tasks=self.tasks,
                verbose=2,
                process=process  # "sequential" or "hierarchical"
            )

            # Execute crew
            result = self.crew.kickoff()

            # Log execution
            self.execution_log.append({
                'timestamp': time.time(),
                'process': process,
                'agents_count': len(self.agents),
                'tasks_count': len(self.tasks),
                'result': result
            })

            return {
                "success": True,
                "result": result,
                "execution_time": time.time() - self.execution_log[-1]['timestamp'],
                "agents_used": [agent.role for agent in self.agents],
                "tasks_completed": len(self.tasks)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": None,
                "result": None
            }

    def get_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        """Get performance metrics for specific agent"""
        agent_obj = next((a for a in self.agents if a.role == agent_name), None)
        if agent_obj:
            return {
                "agent_name": agent_name,
                "tasks_completed": len(agent_obj.completed_tasks),
                "performance_score": agent_obj.performance_metrics.get("score", 0),
                "collaboration_score": agent_obj.performance_metrics.get("collaboration", 0)
            }
        return {"error": "Agent not found"}

    def create_hierarchical_crew(self, manager_agent: CrewAIAgent,
                               worker_agents: List[CrewAIAgent]) -> Dict[str, Any]:
        """Create hierarchical crew with manager and workers"""
        self.add_agent(manager_agent)

        for worker in worker_agents:
            self.add_agent(worker)

        # Create crew with hierarchical process
        self.crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=2,
            process="hierarchical",
            manager_llm=manager_agent.agent.llm  # Manager controls execution
        )

        result = self.crew.kickoff()

        return {
            "success": True,
            "result": result,
            "manager": manager_agent.name,
            "workers": [agent.name for agent in worker_agents],
            "execution_type": "hierarchical"
        }
```

---

## Advanced Agent Concepts {#advanced-concepts}

### 1. Emergent Behavior and Self-Organization

```python
import numpy as np
from typing import Dict, List, Tuple
from enum import Enum

class EmergenceType(Enum):
    FLOCKING = "flocking"
    SWARMING = "swarming"
    CLUSTERING = "clustering"
    PATTERN_FORMATION = "pattern_formation"
    COLLECTIVE_INTELLIGENCE = "collective_intelligence"

class EmergentAgent:
    """Agent capable of emergent behavior"""

    def __init__(self, agent_id: str, position: Tuple[float, float],
                 velocity: Tuple[float, float]):
        self.id = agent_id
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.acceleration = np.array([0.0, 0.0])

        # Behavior parameters
        self.neighbor_radius = 50.0
        self.separation_distance = 20.0
        self.max_speed = 5.0
        self.max_force = 0.5

        # State tracking
        self.history = [self.position.copy()]
        self.behavior_state = "normal"

    def update(self, neighbors: List['EmergentAgent'], forces: Dict[str, np.ndarray]):
        """Update agent state based on neighbors and forces"""
        # Reset acceleration
        self.acceleration = np.array([0.0, 0.0])

        # Apply various behavioral forces
        for behavior, force in forces.items():
            self.acceleration += force

        # Update velocity
        self.velocity += self.acceleration

        # Limit velocity
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

        # Update position
        self.position += self.velocity

        # Store history
        self.history.append(self.position.copy())

        # Keep history size manageable
        if len(self.history) > 1000:
            self.history.pop(0)

    def get_neighbors(self, all_agents: List['EmergentAgent']) -> List['EmergentAgent']:
        """Get nearby agents for local interaction"""
        neighbors = []
        for agent in all_agents:
            if agent.id != self.id:
                distance = np.linalg.norm(self.position - agent.position)
                if distance <= self.neighbor_radius:
                    neighbors.append(agent)
        return neighbors

    def compute_flocking_forces(self, neighbors: List['EmergentAgent']) -> Dict[str, np.ndarray]:
        """Compute boids-style flocking forces"""
        forces = {}

        if not neighbors:
            return forces

        # Alignment force
        alignment = np.array([0.0, 0.0])
        alignment_count = 0

        # Cohesion force
        cohesion = np.array([0.0, 0.0])
        cohesion_count = 0

        # Separation force
        separation = np.array([0.0, 0.0])
        separation_count = 0

        for neighbor in neighbors:
            to_neighbor = neighbor.position - self.position
            distance = np.linalg.norm(to_neighbor)

            # Alignment
            alignment += neighbor.velocity
            alignment_count += 1

            # Cohesion
            cohesion += neighbor.position
            cohesion_count += 1

            # Separation
            if distance < self.separation_distance and distance > 0:
                separation -= to_neighbor / (distance ** 2)
                separation_count += 1

        # Compute final forces
        if alignment_count > 0:
            alignment = (alignment / alignment_count) - self.velocity
            alignment = self._limit_vector(alignment, self.max_force)

        if cohesion_count > 0:
            cohesion = (cohesion / cohesion_count) - self.position
            cohesion = self._limit_vector(cohesion, self.max_force)

        if separation_count > 0:
            separation = self._limit_vector(separation, self.max_force)

        forces['alignment'] = alignment
        forces['cohesion'] = cohesion
        forces['separation'] = separation

        return forces

    def _limit_vector(self, vector: np.ndarray, max_magnitude: float) -> np.ndarray:
        """Limit vector magnitude"""
        magnitude = np.linalg.norm(vector)
        if magnitude > max_magnitude:
            vector = (vector / magnitude) * max_magnitude
        return vector

class EmergentSystem:
    """System for simulating emergent agent behaviors"""

    def __init__(self, num_agents: int = 50, world_size: Tuple[int, int] = (800, 600)):
        self.num_agents = num_agents
        self.world_size = world_size
        self.agents = []
        self.time_step = 0
        self.global_forces = {}

        # Initialize agents
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize random agents"""
        import random

        for i in range(self.num_agents):
            position = (
                random.uniform(0, self.world_size[0]),
                random.uniform(0, self.world_size[1])
            )
            velocity = (
                random.uniform(-2, 2),
                random.uniform(-2, 2)
            )

            agent = EmergentAgent(f"agent_{i}", position, velocity)
            self.agents.append(agent)

    def simulate_step(self):
        """Execute one simulation step"""
        for agent in self.agents:
            # Get neighbors
            neighbors = agent.get_neighbors(self.agents)

            # Compute local forces
            local_forces = agent.compute_flocking_forces(neighbors)

            # Add global forces
            all_forces = {**local_forces, **self.global_forces}

            # Update agent
            agent.update(neighbors, all_forces)

        self.time_step += 1

    def add_global_force(self, force_name: str, force_vector: np.ndarray):
        """Add force that affects all agents"""
        self.global_forces[force_name] = force_vector

    def detect_patterns(self) -> Dict[str, Any]:
        """Detect emergent patterns in the system"""
        # Compute system metrics
        metrics = {}

        # Compute average velocity
        velocities = [agent.velocity for agent in self.agents]
        avg_velocity = np.mean(velocities, axis=0)
        metrics['average_velocity'] = avg_velocity

        # Compute velocity variance
        velocity_magnitudes = [np.linalg.norm(v) for v in velocities]
        metrics['velocity_variance'] = np.var(velocity_magnitudes)

        # Compute cluster coefficient
        metrics['clustering_coefficient'] = self._compute_clustering_coefficient()

        # Detect pattern type
        metrics['emergence_type'] = self._classify_emergence(avg_velocity, velocity_magnitudes)

        return metrics

    def _compute_clustering_coefficient(self) -> float:
        """Compute clustering coefficient of agent network"""
        # Simplified clustering coefficient
        edges = 0
        possible_edges = 0

        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents):
                if i < j:
                    distance = np.linalg.norm(agent1.position - agent2.position)
                    if distance < agent1.neighbor_radius:
                        edges += 1
                    possible_edges += 1

        return edges / possible_edges if possible_edges > 0 else 0

    def _classify_emergence(self, avg_velocity: np.ndarray,
                          velocity_magnitudes: List[float]) -> str:
        """Classify type of emergent behavior"""
        avg_speed = np.linalg.norm(avg_velocity)
        speed_variance = np.var(velocity_magnitudes)

        if avg_speed > 2.0 and speed_variance < 1.0:
            return "flocking"
        elif avg_speed < 1.0 and speed_variance < 0.5:
            return "clustering"
        elif speed_variance > 2.0:
            return "swarming"
        else:
            return "disordered"
```

### 2. Swarm Intelligence

```python
from typing import Dict, List, Tuple
import random
import math

class SwarmAgent:
    """Agent in swarm intelligence system"""

    def __init__(self, agent_id: str, position: Tuple[float, float],
                 task: str = "explore"):
        self.id = agent_id
        self.position = np.array(position)
        self.velocity = np.array([
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ])

        self.fitness = 0.0
        self.best_position = position
        self.best_fitness = 0.0

        # Swarm parameters
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.max_velocity = 2.0

        self.task = task
        self.energy = 100.0
        self.memory = {}

    def update_velocity(self, global_best: np.ndarray):
        """Update velocity using PSO formula"""
        r1, r2 = random.random(), random.random()

        cognitive = self.cognitive_coefficient * r1 * (self.best_position - self.position)
        social = self.social_coefficient * r2 * (global_best - self.position)

        self.velocity = (self.inertia_weight * self.velocity + cognitive + social)

        # Limit velocity
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > self.max_velocity:
            self.velocity = (self.velocity / velocity_magnitude) * self.max_velocity

    def update_position(self, bounds: Tuple[float, float, float, float]):
        """Update position within bounds"""
        self.position += self.velocity

        # Apply bounds
        x_min, x_max, y_min, y_max = bounds
        self.position[0] = max(x_min, min(x_max, self.position[0]))
        self.position[1] = max(y_min, min(y_max, self.position[1]))

    def evaluate_fitness(self, fitness_function: callable):
        """Evaluate agent fitness"""
        current_fitness = fitness_function(self.position)
        self.fitness = current_fitness

        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            self.best_position = self.position.copy()

    def execute_task(self, task_type: str, environment: Dict) -> Dict:
        """Execute assigned task"""
        task_result = {
            'agent_id': self.id,
            'task': task_type,
            'success': False,
            'energy_cost': 0,
            'reward': 0
        }

        if task_type == "exploration":
            task_result.update(self._explore(environment))
        elif task_type == "foraging":
            task_result.update(self._forage(environment))
        elif task_type == "construction":
            task_result.update(self._construct(environment))
        elif task_type == "patrol":
            task_result.update(self._patrol(environment))

        # Update energy
        self.energy -= task_result['energy_cost']
        self.energy = max(0, self.energy)

        return task_result

    def _explore(self, environment: Dict) -> Dict:
        """Explore environment to discover new resources"""
        # Random movement for exploration
        exploration_reward = random.uniform(0, 10)

        return {
            'success': True,
            'energy_cost': 5,
            'reward': exploration_reward,
            'discovered_resources': random.randint(0, 3)
        }

    def _forage(self, environment: Dict) -> Dict:
        """Search and collect resources"""
        # Check if resources are available
        resources_nearby = self._detect_resources(environment)

        if resources_nearby:
            collected = min(resources_nearby, self.energy // 10)
            reward = collected * 20
            return {
                'success': True,
                'energy_cost': collected * 10,
                'reward': reward,
                'resources_collected': collected
            }
        else:
            return {
                'success': False,
                'energy_cost': 8,
                'reward': 0,
                'resources_collected': 0
            }

    def _construct(self, environment: Dict) -> Dict:
        """Build structures or artifacts"""
        if self.energy >= 20:
            return {
                'success': True,
                'energy_cost': 20,
                'reward': 30,
                'structures_built': 1
            }
        else:
            return {
                'success': False,
                'energy_cost': 5,
                'reward': 0,
                'structures_built': 0
            }

    def _patrol(self, environment: Dict) -> Dict:
        """Monitor and secure area"""
        threat_detected = random.choice([True, False])

        if threat_detected:
            return {
                'success': True,
                'energy_cost': 15,
                'reward': 25,
                'threats_detected': 1
            }
        else:
            return {
                'success': True,
                'energy_cost': 10,
                'reward': 5,
                'threats_detected': 0
            }

    def _detect_resources(self, environment: Dict) -> int:
        """Detect available resources in vicinity"""
        # Simplified resource detection
        return random.randint(0, 5)

class SwarmIntelligenceSystem:
    """Swarm intelligence coordination system"""

    def __init__(self, num_agents: int = 30, world_bounds: Tuple[float, float, float, float] = (0, 100, 0, 100)):
        self.agents = []
        self.world_bounds = world_bounds
        self.global_best_position = None
        self.global_best_fitness = float('-inf')

        # Task management
        self.active_tasks = []
        self.task_queue = []
        self.completed_tasks = []

        # Performance tracking
        self.swarm_performance = {
            'total_rewards': 0,
            'tasks_completed': 0,
            'energy_consumed': 0,
            'resources_collected': 0
        }

        # Initialize swarm
        self._initialize_swarm(num_agents)

    def _initialize_swarm(self, num_agents: int):
        """Initialize swarm with agents"""
        for i in range(num_agents):
            position = (
                random.uniform(self.world_bounds[0], self.world_bounds[1]),
                random.uniform(self.world_bounds[2], self.world_bounds[3])
            )

            agent = SwarmAgent(f"swarm_{i}", position)
            self.agents.append(agent)

    def optimize_with_pso(self, fitness_function: callable, iterations: int = 100):
        """Optimize using Particle Swarm Optimization"""
        for iteration in range(iterations):
            # Evaluate all agents
            for agent in self.agents:
                agent.evaluate_fitness(fitness_function)

                # Update global best
                if agent.fitness > self.global_best_fitness:
                    self.global_best_fitness = agent.fitness
                    self.global_best_position = agent.position.copy()

            # Update all agents
            for agent in self.agents:
                agent.update_velocity(self.global_best_position)
                agent.update_position(self.world_bounds)

    def coordinate_tasks(self, task_distribution: str = "adaptive"):
        """Coordinate task execution among swarm"""
        if task_distribution == "adaptive":
            self._adaptive_task_distribution()
        elif task_distribution == "load_balanced":
            self._load_balanced_distribution()
        elif task_distribution == "specialized":
            self._specialized_distribution()

    def _adaptive_task_distribution(self):
        """Adaptively distribute tasks based on agent capabilities"""
        # Analyze current swarm state
        energy_levels = [agent.energy for agent in self.agents]
        task_capabilities = self._assess_task_capabilities()

        # Assign tasks based on energy and capability
        for task in self.task_queue:
            suitable_agents = [
                agent for agent in self.agents
                if agent.energy >= 10 and agent.task in task_capabilities.get(task['type'], [])
            ]

            if suitable_agents:
                # Select best agent (highest energy + capability match)
                best_agent = max(suitable_agents, key=lambda a: a.energy)
                self._assign_task_to_agent(best_agent, task)
                self.task_queue.remove(task)

    def _load_balanced_distribution(self):
        """Distribute tasks to balance load"""
        for task in self.task_queue:
            # Select agent with lowest current load
            load_levels = {agent.id: len(agent.memory) for agent in self.agents}
            least_loaded_agent = min(load_levels, key=load_levels.get)

            selected_agent = next(a for a in self.agents if a.id == least_loaded_agent)
            self._assign_task_to_agent(selected_agent, task)
            self.task_queue.remove(task)

    def _specialized_distribution(self):
        """Distribute tasks based on agent specialization"""
        for task in self.task_queue:
            task_type = task['type']

            # Find specialized agents for task type
            specialized_agents = [
                agent for agent in self.agents
                if agent.task == task_type
            ]

            if specialized_agents:
                # Select random specialized agent
                selected_agent = random.choice(specialized_agents)
                self._assign_task_to_agent(selected_agent, task)
                self.task_queue.remove(task)

    def _assign_task_to_agent(self, agent: SwarmAgent, task: Dict):
        """Assign task to specific agent"""
        result = agent.execute_task(task['type'], task.get('environment', {}))

        # Update performance metrics
        self.swarm_performance['total_rewards'] += result['reward']
        self.swarm_performance['energy_consumed'] += result['energy_cost']

        if result['success']:
            self.completed_tasks.append({
                'task': task,
                'agent': agent.id,
                'result': result,
                'timestamp': time.time()
            })

            if 'resources_collected' in result:
                self.swarm_performance['resources_collected'] += result['resources_collected']

        self.active_tasks.append({
            'agent_id': agent.id,
            'task': task,
            'result': result,
            'assigned_at': time.time()
        })

    def _assess_task_capabilities(self) -> Dict[str, List[str]]:
        """Assess which agents are capable of which tasks"""
        capabilities = {
            'exploration': [],
            'foraging': [],
            'construction': [],
            'patrol': []
        }

        for agent in self.agents:
            if agent.energy >= 10:
                # Add agent to relevant capability lists
                for task_type in capabilities.keys():
                    # Simplified capability assessment
                    if random.random() > 0.3:  # 70% chance of capability
                        capabilities[task_type].append(agent.task)

        return capabilities

    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current status of swarm"""
        total_energy = sum(agent.energy for agent in self.agents)
        avg_energy = total_energy / len(self.agents)

        # Count agents by task
        task_distribution = {}
        for agent in self.agents:
            task = agent.task
            task_distribution[task] = task_distribution.get(task, 0) + 1

        return {
            'swarm_size': len(self.agents),
            'total_energy': total_energy,
            'average_energy': avg_energy,
            'task_distribution': task_distribution,
            'performance': self.swarm_performance,
            'global_best_fitness': self.global_best_fitness,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks)
        }
```

---

## Agent Safety and Ethics {#safety-ethics}

### 1. Safety Constraints and Guardrails

```python
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass

class SafetyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SafetyConstraint:
    """Safety constraint specification"""
    name: str
    description: str
    level: SafetyLevel
    constraint_function: Callable[[Dict], bool]
    severity: str
    allowed_exceptions: List[str]

class SafetyMonitor:
    """Monitor for agent safety constraints"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.constraints = {}
        self.violation_history = []
        self.safety_score = 100.0
        self.monitoring_active = True

    def add_constraint(self, constraint: SafetyConstraint):
        """Add safety constraint"""
        self.constraints[constraint.name] = constraint

    def check_safety(self, action: Dict, context: Dict) -> Dict[str, Any]:
        """Check if action is safe to execute"""
        violations = []

        for constraint_name, constraint in self.constraints.items():
            try:
                if not constraint.constraint_function(context):
                    violation = {
                        'constraint': constraint_name,
                        'action': action,
                        'context': context,
                        'severity': constraint.severity,
                        'timestamp': time.time()
                    }
                    violations.append(violation)
            except Exception as e:
                violation = {
                    'constraint': constraint_name,
                    'error': str(e),
                    'severity': 'high',
                    'timestamp': time.time()
                }
                violations.append(violation)

        # Process violations
        if violations:
            self._process_violations(violations)

            return {
                'safe': False,
                'violations': violations,
                'action_blocked': True,
                'safety_score': self.safety_score
            }

        return {
            'safe': True,
            'violations': [],
            'action_blocked': False,
            'safety_score': self.safety_score
        }

    def _process_violations(self, violations: List[Dict]):
        """Process safety violations"""
        for violation in violations:
            # Add to history
            self.violation_history.append(violation)

            # Deduct safety score based on severity
            if violation['severity'] == 'low':
                self.safety_score -= 1
            elif violation['severity'] == 'medium':
                self.safety_score -= 5
            elif violation['severity'] == 'high':
                self.safety_score -= 10
            elif violation['severity'] == 'critical':
                self.safety_score -= 20

            # Ensure score doesn't go below 0
            self.safety_score = max(0, self.safety_score)

    def get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report"""
        recent_violations = [
            v for v in self.violation_history
            if time.time() - v['timestamp'] < 3600  # Last hour
        ]

        violations_by_severity = {}
        for violation in recent_violations:
            severity = violation['severity']
            violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1

        return {
            'agent_id': self.agent_id,
            'current_safety_score': self.safety_score,
            'total_violations': len(self.violation_history),
            'recent_violations': len(recent_violations),
            'violations_by_severity': violations_by_severity,
            'constraints_count': len(self.constraints),
            'monitoring_active': self.monitoring_active,
            'recommendation': self._get_safety_recommendation()
        }

    def _get_safety_recommendation(self) -> str:
        """Get safety improvement recommendation"""
        if self.safety_score >= 90:
            return "Excellent safety compliance"
        elif self.safety_score >= 70:
            return "Good safety compliance, minor improvements needed"
        elif self.safety_score >= 50:
            return "Moderate safety concerns, review constraints"
        elif self.safety_score >= 25:
            return "Significant safety issues, immediate attention required"
        else:
            return "Critical safety violations, agent should be halted"

class EthicalAgent:
    """Agent with ethical decision-making capabilities"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.ethical_framework = "utilitarian"  # utilitarian, deontological, virtue_ethics
        self.ethical_values = {
            'harm_prevention': 1.0,
            'fairness': 1.0,
            'autonomy_respect': 1.0,
            'privacy_protection': 1.0,
            'transparency': 1.0
        }
        self.ethical_decisions = []
        self.bias_detector = BiasDetector()

    def evaluate_ethical_impact(self, action: Dict, stakeholders: List[str]) -> Dict[str, Any]:
        """Evaluate ethical implications of action"""
        evaluation = {
            'action': action,
            'stakeholders': stakeholders,
            'ethical_score': 0.0,
            'concerns': [],
            'recommendations': []
        }

        # Check for bias
        bias_assessment = self.bias_detector.assess_bias(action, stakeholders)
        if bias_assessment['bias_detected']:
            evaluation['concerns'].append(f"Bias detected: {bias_assessment['bias_type']}")
            evaluation['ethical_score'] -= bias_assessment['severity'] * 0.3

        # Evaluate based on ethical framework
        if self.ethical_framework == "utilitarian":
            evaluation.update(self._utilitarian_evaluation(action, stakeholders))
        elif self.ethical_framework == "deontological":
            evaluation.update(self._deontological_evaluation(action, stakeholders))

        # Add transparency check
        transparency_score = self._evaluate_transparency(action)
        evaluation['transparency_score'] = transparency_score

        if transparency_score < 0.7:
            evaluation['concerns'].append("Low transparency in decision making")

        # Store decision
        self.ethical_decisions.append({
            'timestamp': time.time(),
            'action': action,
            'evaluation': evaluation
        })

        return evaluation

    def _utilitarian_evaluation(self, action: Dict, stakeholders: List[str]) -> Dict[str, Any]:
        """Evaluate action using utilitarian ethics"""
        potential_harm = self._assess_potential_harm(action, stakeholders)
        potential_benefit = self._assess_potential_benefit(action, stakeholders)

        net_benefit = potential_benefit - potential_harm

        return {
            'potential_harm': potential_harm,
            'potential_benefit': potential_benefit,
            'net_benefit': net_benefit,
            'justified': net_benefit > 0
        }

    def _deontological_evaluation(self, action: Dict, stakeholders: List[str]) -> Dict[str, Any]:
        """Evaluate action using deontological ethics"""
        rights_violations = self._check_rights_violations(action, stakeholders)
        duty_compliance = self._check_duty_compliance(action, stakeholders)

        return {
            'rights_violations': rights_violations,
            'duty_compliance': duty_compliance,
            'categorical_imperative_satisfied': len(rights_violations) == 0,
            'justified': len(rights_violations) == 0 and duty_compliance >= 0.7
        }

    def _assess_potential_harm(self, action: Dict, stakeholders: List[str]) -> float:
        """Assess potential harm to stakeholders"""
        # Simplified harm assessment
        harm_factors = {
            'physical_harm': 0.0,
            'emotional_harm': 0.0,
            'financial_harm': 0.0,
            'privacy_violation': 0.0,
            'social_harm': 0.0
        }

        # Analyze action for harm indicators
        if 'resource_consumption' in action:
            if action['resource_consumption'] > 100:
                harm_factors['financial_harm'] += 0.2

        if 'data_access' in action:
            if action['data_access']['personal_data']:
                harm_factors['privacy_violation'] += 0.3

        return sum(harm_factors.values())

    def _assess_potential_benefit(self, action: Dict, stakeholders: List[str]) -> float:
        """Assess potential benefit to stakeholders"""
        # Simplified benefit assessment
        benefit_score = 0.0

        if 'efficiency_gain' in action:
            benefit_score += min(action['efficiency_gain'], 1.0)

        if 'problem_solving' in action:
            if action['problem_solving']['impact'] == 'high':
                benefit_score += 0.8

        return min(benefit_score, 1.0)

    def _check_rights_violations(self, action: Dict, stakeholders: List[str]) -> List[str]:
        """Check for rights violations"""
        violations = []

        if 'autonomy' in action:
            if action['autonomy']['forced_participation']:
                violations.append('Autonomy violation')

        if 'privacy' in action:
            if not action['privacy']['consent_obtained']:
                violations.append('Privacy rights violation')

        return violations

    def _check_duty_compliance(self, action: Dict, stakeholders: List[str]) -> float:
        """Check compliance with duties"""
        compliance_score = 1.0

        # Check for duty to do no harm
        if 'harmful_action' in action and action['harmful_action']:
            compliance_score -= 0.5

        # Check for duty to be transparent
        if 'transparency' in action:
            if action['transparency']['sufficient_information']:
                compliance_score += 0.2
            else:
                compliance_score -= 0.3

        return max(0.0, compliance_score)

    def _evaluate_transparency(self, action: Dict) -> float:
        """Evaluate transparency of action"""
        transparency_score = 0.5  # Base score

        if 'decision_rationale' in action:
            transparency_score += 0.3

        if 'data_sources' in action:
            if action['data_sources']['clearly_indicated']:
                transparency_score += 0.2

        return min(transparency_score, 1.0)

class BiasDetector:
    """Detector for various types of bias in agent decisions"""

    def __init__(self):
        self.bias_patterns = {
            'demographic_bias': self._check_demographic_bias,
            'confirmation_bias': self._check_confirmation_bias,
            'recency_bias': self._check_recency_bias,
            'availability_bias': self._check_availability_bias
        }

    def assess_bias(self, action: Dict, stakeholders: List[str]) -> Dict[str, Any]:
        """Assess potential bias in action"""
        bias_assessment = {
            'bias_detected': False,
            'bias_type': None,
            'severity': 0.0,
            'recommendations': []
        }

        for bias_type, check_function in self.bias_patterns.items():
            result = check_function(action, stakeholders)
            if result['detected']:
                bias_assessment['bias_detected'] = True
                bias_assessment['bias_type'] = bias_type
                bias_assessment['severity'] = result['severity']
                bias_assessment['recommendations'] = result['recommendations']
                break

        return bias_assessment

    def _check_demographic_bias(self, action: Dict, stakeholders: List[str]) -> Dict[str, Any]:
        """Check for demographic bias"""
        # Simplified demographic bias check
        if 'resource_allocation' in action:
            allocation = action['resource_allocation']
            if 'equal_distribution' in allocation and not allocation['equal_distribution']:
                return {
                    'detected': True,
                    'severity': 0.6,
                    'recommendations': ['Ensure equal opportunity', 'Review allocation criteria']
                }

        return {'detected': False, 'severity': 0.0, 'recommendations': []}

    def _check_confirmation_bias(self, action: Dict, stakeholders: List[str]) -> Dict[str, Any]:
        """Check for confirmation bias"""
        # Check if action confirms existing beliefs without evidence
        if 'decision_basis' in action:
            if action['decision_basis'].get('ignores_contradicting_evidence'):
                return {
                    'detected': True,
                    'severity': 0.7,
                    'recommendations': ['Seek contrary evidence', 'Consider alternative perspectives']
                }

        return {'detected': False, 'severity': 0.0, 'recommendations': []}

    def _check_recency_bias(self, action: Dict, stakeholders: List[str]) -> Dict[str, Any]:
        """Check for recency bias"""
        # Check if action overweights recent events
        if 'evidence_timeline' in action:
            timeline = action['evidence_timeline']
            recent_weight = timeline.get('recent_evidence_weight', 0)
            if recent_weight > 0.8:
                return {
                    'detected': True,
                    'severity': 0.5,
                    'recommendations': ['Weight historical evidence equally', 'Use moving averages']
                }

        return {'detected': False, 'severity': 0.0, 'recommendations': []}

    def _check_availability_bias(self, action: Dict, stakeholders: List[str]) -> Dict[str, Any]:
        """Check for availability bias"""
        # Check if decision based on easily recalled examples
        if 'decision_basis' in action:
            if action['decision_basis'].get('based_on_memorable_examples'):
                return {
                    'detected': True,
                    'severity': 0.4,
                    'recommendations': ['Use representative samples', 'Quantify all evidence']
                }

        return {'detected': False, 'severity': 0.0, 'recommendations': []}
```

---

This theory file provides a comprehensive foundation for understanding AI agents, their architectures, communication patterns, learning mechanisms, frameworks, and safety considerations. The file is structured to build knowledge systematically from basic concepts to advanced implementations.

The theory covers:

1. Core components and taxonomies of AI agents
2. Major architectural patterns (Subsumption, BDI, Blackboard)
3. Multi-agent system coordination and communication
4. Learning mechanisms including reinforcement learning and social learning
5. Modern frameworks like LangChain, AutoGen, and CrewAI
6. Advanced concepts like emergent behavior and swarm intelligence
7. Safety, ethics, and bias detection

Each section includes both theoretical explanations and practical code implementations to bridge the gap between theory and practice. The content is designed to provide students with both conceptual understanding and hands-on knowledge of AI agent systems.
