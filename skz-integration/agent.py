"""
Base Agent Model and Framework for Autonomous Academic Publishing Agents
Implements the Agent-Arena-Relation (AAR) core architecture
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional
import threading
import queue
import time

db = SQLAlchemy()

class AgentStatus(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class MessageType(Enum):
    COMMAND = "command"
    QUERY = "query"
    EVENT = "event"
    COORDINATION = "coordination"
    RESPONSE = "response"

class AgentCapability(Enum):
    RESEARCH_DISCOVERY = "research_discovery"
    MANUSCRIPT_ENHANCEMENT = "manuscript_enhancement"
    CONTENT_QUALITY = "content_quality"
    EDITORIAL_ORCHESTRATION = "editorial_orchestration"
    REVIEWER_COORDINATION = "reviewer_coordination"
    STAKEHOLDER_COMMUNICATION = "stakeholder_communication"
    ANALYTICS_REPORTING = "analytics_reporting"
    STRATEGIC_PLANNING = "strategic_planning"

class Agent(db.Model):
    """Base Agent Model implementing AAR (Agent-Arena-Relation) core"""
    
    __tablename__ = 'agents'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False)
    agent_type = db.Column(db.String(50), nullable=False)
    capabilities = db.Column(db.Text)  # JSON string of capabilities
    status = db.Column(db.Enum(AgentStatus), default=AgentStatus.INITIALIZING)
    arena_context = db.Column(db.Text)  # JSON string of arena/environment context
    performance_metrics = db.Column(db.Text)  # JSON string of performance data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_heartbeat = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    sent_messages = db.relationship('Message', foreign_keys='Message.sender_id', backref='sender')
    received_messages = db.relationship('Message', foreign_keys='Message.recipient_id', backref='recipient')
    
    def __init__(self, name: str, agent_type: str, capabilities: List[AgentCapability], arena_context: Dict = None):
        self.name = name
        self.agent_type = agent_type
        self.capabilities = json.dumps([cap.value for cap in capabilities])
        self.arena_context = json.dumps(arena_context or {})
        self.performance_metrics = json.dumps({
            'tasks_completed': 0,
            'success_rate': 0.0,
            'average_response_time': 0.0,
            'error_count': 0
        })
        self.status = AgentStatus.INITIALIZING
        
        # Initialize message queue for agent communication
        self.message_queue = queue.Queue()
        self.running = False
        self.thread = None
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities as list"""
        return json.loads(self.capabilities) if self.capabilities else []
    
    def get_arena_context(self) -> Dict:
        """Get arena context as dictionary"""
        return json.loads(self.arena_context) if self.arena_context else {}
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics as dictionary"""
        return json.loads(self.performance_metrics) if self.performance_metrics else {}
    
    def update_performance_metrics(self, metrics: Dict):
        """Update performance metrics"""
        current_metrics = self.get_performance_metrics()
        current_metrics.update(metrics)
        self.performance_metrics = json.dumps(current_metrics)
        self.updated_at = datetime.utcnow()
    
    def start(self):
        """Start the agent's processing thread"""
        if not self.running:
            self.running = True
            self.status = AgentStatus.ACTIVE
            self.thread = threading.Thread(target=self._run_loop)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        """Stop the agent's processing thread"""
        self.running = False
        self.status = AgentStatus.SHUTDOWN
        if self.thread:
            self.thread.join(timeout=5)
    
    def _run_loop(self):
        """Main agent processing loop"""
        while self.running:
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.utcnow()
                
                # Process messages from queue
                try:
                    message = self.message_queue.get(timeout=1)
                    self.status = AgentStatus.BUSY
                    self._process_message(message)
                    self.status = AgentStatus.ACTIVE
                    self.message_queue.task_done()
                except queue.Empty:
                    self.status = AgentStatus.IDLE
                    continue
                    
            except Exception as e:
                self.status = AgentStatus.ERROR
                print(f"Agent {self.name} error: {e}")
                time.sleep(1)
    
    def _process_message(self, message):
        """Process incoming message - to be overridden by specific agents"""
        # Default implementation - echo back
        if message.message_type == MessageType.QUERY:
            response = Message(
                sender_id=self.id,
                recipient_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                content=json.dumps({
                    "status": "processed",
                    "original_message": message.content,
                    "agent": self.name
                }),
                correlation_id=message.id
            )
            # In a real implementation, this would be sent through the message broker
            print(f"Agent {self.name} processed message: {message.content}")
    
    def send_message(self, recipient_id: str, message_type: MessageType, content: Dict, correlation_id: str = None):
        """Send message to another agent"""
        message = Message(
            sender_id=self.id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=json.dumps(content),
            correlation_id=correlation_id
        )
        db.session.add(message)
        db.session.commit()
        return message
    
    def receive_message(self, message):
        """Receive message and add to processing queue"""
        self.message_queue.put(message)
    
    def to_dict(self) -> Dict:
        """Convert agent to dictionary representation"""
        return {
            'id': self.id,
            'name': self.name,
            'agent_type': self.agent_type,
            'capabilities': self.get_capabilities(),
            'status': self.status.value,
            'arena_context': self.get_arena_context(),
            'performance_metrics': self.get_performance_metrics(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_heartbeat': self.last_heartbeat.isoformat()
        }

class Message(db.Model):
    """Message model for inter-agent communication"""
    
    __tablename__ = 'messages'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    sender_id = db.Column(db.String(36), db.ForeignKey('agents.id'), nullable=False)
    recipient_id = db.Column(db.String(36), db.ForeignKey('agents.id'), nullable=False)
    message_type = db.Column(db.Enum(MessageType), nullable=False)
    content = db.Column(db.Text, nullable=False)
    correlation_id = db.Column(db.String(36))  # For request-response correlation
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='pending')
    
    def get_content(self) -> Dict:
        """Get message content as dictionary"""
        return json.loads(self.content) if self.content else {}
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary representation"""
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'message_type': self.message_type.value,
            'content': self.get_content(),
            'correlation_id': self.correlation_id,
            'created_at': self.created_at.isoformat(),
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'status': self.status
        }

class WorkflowInstance(db.Model):
    """Workflow instance for tracking multi-agent processes"""
    
    __tablename__ = 'workflow_instances'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_type = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default='active')
    context_data = db.Column(db.Text)  # JSON string of workflow context
    current_step = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    def get_context_data(self) -> Dict:
        """Get workflow context data as dictionary"""
        return json.loads(self.context_data) if self.context_data else {}
    
    def update_context_data(self, data: Dict):
        """Update workflow context data"""
        current_data = self.get_context_data()
        current_data.update(data)
        self.context_data = json.dumps(current_data)
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert workflow instance to dictionary representation"""
        return {
            'id': self.id,
            'workflow_type': self.workflow_type,
            'status': self.status,
            'context_data': self.get_context_data(),
            'current_step': self.current_step,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class AgentRegistry:
    """Registry for managing agent instances"""
    
    _instance = None
    _agents = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
        return cls._instance
    
    def register_agent(self, agent: Agent):
        """Register an agent instance"""
        self._agents[agent.id] = agent
        agent.start()
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent instance"""
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            agent.stop()
            del self._agents[agent_id]
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent instance by ID"""
        return self._agents.get(agent_id)
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[Agent]:
        """Get agents with specific capability"""
        matching_agents = []
        for agent in self._agents.values():
            if capability.value in agent.get_capabilities():
                matching_agents.append(agent)
        return matching_agents
    
    def get_all_agents(self) -> List[Agent]:
        """Get all registered agents"""
        return list(self._agents.values())
    
    def route_message(self, message: Message):
        """Route message to appropriate agent"""
        recipient = self.get_agent(message.recipient_id)
        if recipient:
            recipient.receive_message(message)
        else:
            print(f"Agent {message.recipient_id} not found for message routing")

# Global agent registry instance
agent_registry = AgentRegistry()

