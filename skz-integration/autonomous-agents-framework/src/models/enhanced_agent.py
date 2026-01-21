"""
Enhanced Agent Base Class
Phase 2 Critical Component - Integrates all autonomous capabilities
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import threading
from abc import ABC, abstractmethod

from .memory_system import PersistentMemorySystem
from .ml_decision_engine import DecisionEngine, DecisionContext, DecisionResult
from .learning_framework import LearningFramework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    """Represents the current state of an agent"""
    agent_id: str
    status: str  # 'active', 'idle', 'busy', 'error'
    current_task: Optional[str]
    performance_metrics: Dict[str, Any]
    last_activity: datetime
    memory_usage: Dict[str, Any]
    learning_stats: Dict[str, Any]

@dataclass
class AgentAction:
    """Represents an action an agent can take"""
    action_id: str
    action_type: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    constraints: Dict[str, Any]
    priority: float

class EnhancedAgent(ABC):
    """
    Enhanced agent base class with Phase 2 autonomous capabilities
    Integrates persistent memory, ML decision making, and learning
    """
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.lock = threading.RLock()
        
        # Initialize Phase 2 components
        self.memory_system = PersistentMemorySystem(f"agent_memory_{agent_id}.db")
        self.decision_engine = DecisionEngine()
        self.learning_framework = LearningFramework(agent_id)
        
        # Agent state
        self.state = AgentState(
            agent_id=agent_id,
            status='active',
            current_task=None,
            performance_metrics={'success_rate': 0.0, 'total_actions': 0},
            last_activity=datetime.now(),
            memory_usage={},
            learning_stats={}
        )
        
        # Task queue
        self.task_queue = []
        self.completed_tasks = []
        
        logger.info(f"Initialized enhanced agent {agent_id} of type {agent_type}")
    
    @abstractmethod
    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task - to be implemented by specific agents"""
        pass
    
    @abstractmethod
    def get_available_actions(self, context: Dict[str, Any]) -> List[AgentAction]:
        """Get available actions for a given context - to be implemented by specific agents"""
        pass
    
    def execute_action(self, action: AgentAction) -> Dict[str, Any]:
        """Execute an action with autonomous decision making"""
        with self.lock:
            try:
                # Update state
                self.state.status = 'busy'
                self.state.current_task = action.action_type
                self.state.last_activity = datetime.now()
                
                # Store action in memory
                self.memory_system.store_memory(
                    agent_id=self.agent_id,
                    memory_type='action',
                    content={
                        'action_type': action.action_type,
                        'input_data': action.input_data,
                        'timestamp': datetime.now().isoformat()
                    },
                    importance_score=action.priority,
                    tags=[action.action_type, 'action']
                )
                
                # Get decision context
                context = DecisionContext(
                    agent_id=self.agent_id,
                    action_type=action.action_type,
                    input_data=action.input_data,
                    available_options=self._get_action_options(action),
                    constraints=action.constraints,
                    goals=self._get_agent_goals(),
                    risk_tolerance=0.5
                )
                
                # Make autonomous decision
                decision_result = self.decision_engine.make_decision(context)
                
                # Execute the decision
                result = self._execute_decision(decision_result.decision, action)
                
                # Learn from experience
                success = self._evaluate_success(result, action.expected_output)
                self.learning_framework.learn_from_experience(
                    action_type=action.action_type,
                    input_data=action.input_data,
                    output_data=result,
                    success=success,
                    performance_metrics=self._calculate_performance_metrics(result, action),
                    feedback={'decision_confidence': decision_result.confidence_score}
                )
                
                # Update performance metrics
                self._update_performance_metrics(success)
                
                # Update state
                self.state.status = 'active'
                self.state.current_task = None
                self.state.performance_metrics = self._get_performance_metrics()
                self.state.memory_usage = self.memory_system.get_memory_stats()
                self.state.learning_stats = self.learning_framework.get_learning_stats()
                
                logger.info(f"Agent {self.agent_id} executed action {action.action_type} with success: {success}")
                
                return {
                    'success': success,
                    'result': result,
                    'decision_confidence': decision_result.confidence_score,
                    'reasoning': decision_result.reasoning,
                    'performance_metrics': self._calculate_performance_metrics(result, action)
                }
                
            except Exception as e:
                logger.error(f"Error executing action for agent {self.agent_id}: {str(e)}")
                self.state.status = 'error'
                return {'success': False, 'error': str(e)}
    
    def _get_action_options(self, action: AgentAction) -> List[Dict[str, Any]]:
        """Get options for an action"""
        # Get similar past experiences
        similar_memories = self.memory_system.retrieve_memory(
            agent_id=self.agent_id,
            memory_type='action',
            limit=5
        )
        
        options = []
        
        # Add historical successful options
        for memory in similar_memories:
            if memory.content.get('action_type') == action.action_type:
                options.append({
                    'type': 'historical_success',
                    'data': memory.content,
                    'confidence': memory.importance_score,
                    'quality_score': 0.8,
                    'risk_score': 0.2,
                    'efficiency_score': 0.7
                })
        
        # Add learning-based recommendations
        recommendations = self.learning_framework.get_learning_recommendations(action.input_data)
        for rec in recommendations:
            options.append({
                'type': 'learning_recommendation',
                'data': rec['action'],
                'confidence': rec['confidence'],
                'quality_score': 0.7,
                'risk_score': 0.3,
                'efficiency_score': 0.6
            })
        
        # Add default option
        options.append({
            'type': 'default',
            'data': action.expected_output,
            'confidence': 0.5,
            'quality_score': 0.5,
            'risk_score': 0.5,
            'efficiency_score': 0.5
        })
        
        return options
    
    def _get_agent_goals(self) -> List[str]:
        """Get agent-specific goals"""
        base_goals = ['quality', 'efficiency', 'safety']
        
        # Add agent-specific goals
        if 'research' in self.agent_type.lower():
            base_goals.extend(['innovation', 'discovery'])
        elif 'quality' in self.agent_type.lower():
            base_goals.extend(['compliance', 'validation'])
        elif 'coordination' in self.agent_type.lower():
            base_goals.extend(['collaboration', 'optimization'])
        
        return base_goals
    
    def _execute_decision(self, decision: Dict[str, Any], action: AgentAction) -> Dict[str, Any]:
        """Execute the decided action"""
        # Process the task with the decided approach
        task_data = {
            **action.input_data,
            'decision_data': decision.get('data', {}),
            'decision_type': decision.get('type', 'default')
        }
        
        return self.process_task(task_data)
    
    def _evaluate_success(self, result: Dict[str, Any], expected_output: Dict[str, Any]) -> bool:
        """Evaluate if the result meets expectations"""
        # Simple evaluation - can be enhanced
        if not result:
            return False
        
        # Check if key expected outputs are present
        for key, expected_value in expected_output.items():
            if key not in result:
                return False
            if isinstance(expected_value, (int, float)):
                # Numeric comparison with tolerance
                actual_value = result[key]
                if isinstance(actual_value, (int, float)):
                    tolerance = 0.1
                    if abs(actual_value - expected_value) > tolerance:
                        return False
                else:
                    return False
            elif isinstance(expected_value, str):
                # String comparison
                if result[key] != expected_value:
                    return False
        
        return True
    
    def _calculate_performance_metrics(self, result: Dict[str, Any], action: AgentAction) -> Dict[str, Any]:
        """Calculate performance metrics for the action"""
        metrics = {
            'action_type': action.action_type,
            'execution_time': datetime.now().isoformat(),
            'result_size': len(str(result)),
            'priority': action.priority
        }
        
        # Add agent-specific metrics
        if hasattr(self, 'calculate_agent_metrics'):
            metrics.update(self.calculate_agent_metrics(result, action))
        
        return metrics
    
    def _update_performance_metrics(self, success: bool):
        """Update agent performance metrics"""
        self.state.performance_metrics['total_actions'] += 1
        
        if success:
            # Update success rate
            total = self.state.performance_metrics['total_actions']
            current_success_rate = self.state.performance_metrics['success_rate']
            new_success_rate = ((current_success_rate * (total - 1)) + 1) / total
            self.state.performance_metrics['success_rate'] = new_success_rate
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'success_rate': self.state.performance_metrics['success_rate'],
            'total_actions': self.state.performance_metrics['total_actions'],
            'last_activity': self.state.last_activity.isoformat(),
            'status': self.state.status
        }
    
    def get_agent_state(self) -> AgentState:
        """Get current agent state"""
        with self.lock:
            return self.state
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities and statistics"""
        with self.lock:
            return {
                'agent_id': self.agent_id,
                'agent_type': self.agent_type,
                'capabilities': self.capabilities,
                'state': self.state,
                'memory_stats': self.memory_system.get_memory_stats(),
                'learning_stats': self.learning_framework.get_learning_stats(),
                'decision_stats': self.decision_engine.get_decision_stats()
            }
    
    def add_task(self, task_data: Dict[str, Any], priority: float = 0.5) -> str:
        """Add a task to the agent's queue"""
        with self.lock:
            task_id = f"task_{self.agent_id}_{uuid.uuid4().hex[:8]}"
            
            task = {
                'id': task_id,
                'data': task_data,
                'priority': priority,
                'created_at': datetime.now(),
                'status': 'pending'
            }
            
            self.task_queue.append(task)
            self.task_queue.sort(key=lambda x: x['priority'], reverse=True)
            
            logger.info(f"Added task {task_id} to agent {self.agent_id}")
            return task_id
    
    def process_next_task(self) -> Optional[Dict[str, Any]]:
        """Process the next task in the queue"""
        with self.lock:
            if not self.task_queue:
                return None
            
            task = self.task_queue.pop(0)
            task['status'] = 'processing'
            
            try:
                # Create action from task
                action = AgentAction(
                    action_id=f"action_{task['id']}",
                    action_type='process_task',
                    input_data=task['data'],
                    expected_output={},
                    constraints={},
                    priority=task['priority']
                )
                
                # Execute action
                result = self.execute_action(action)
                
                # Update task
                task['status'] = 'completed' if result['success'] else 'failed'
                task['result'] = result
                self.completed_tasks.append(task)
                
                return result
                
            except Exception as e:
                task['status'] = 'error'
                task['error'] = str(e)
                logger.error(f"Error processing task {task['id']}: {str(e)}")
                return {'success': False, 'error': str(e)}
    
    def get_task_queue_status(self) -> Dict[str, Any]:
        """Get task queue status"""
        with self.lock:
            return {
                'pending_tasks': len(self.task_queue),
                'completed_tasks': len(self.completed_tasks),
                'current_task': self.state.current_task,
                'agent_status': self.state.status
            }
    
    def cleanup_old_memories(self, days_old: int = 30):
        """Clean up old memories"""
        self.memory_system.cleanup_old_memories(days_old)
        logger.info(f"Cleaned up memories older than {days_old} days for agent {self.agent_id}")
    
    def save_agent_state(self, filepath: str):
        """Save agent state to file"""
        with self.lock:
            state = {
                'agent_id': self.agent_id,
                'agent_type': self.agent_type,
                'capabilities': self.capabilities,
                'state': self.state,
                'task_queue': self.task_queue,
                'completed_tasks': self.completed_tasks
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, default=str, indent=2)
            
            # Save learning state
            self.learning_framework.save_learning_state(filepath.replace('.json', '_learning.pkl'))
            
            logger.info(f"Saved agent state to {filepath}")
    
    def load_agent_state(self, filepath: str):
        """Load agent state from file"""
        with self.lock:
            try:
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                self.agent_id = state['agent_id']
                self.agent_type = state['agent_type']
                self.capabilities = state['capabilities']
                self.task_queue = state['task_queue']
                self.completed_tasks = state['completed_tasks']
                
                # Load learning state
                self.learning_framework.load_learning_state(filepath.replace('.json', '_learning.pkl'))
                
                logger.info(f"Loaded agent state from {filepath}")
            except FileNotFoundError:
                logger.info(f"No existing state file found for agent {self.agent_id}")
    
    def get_autonomous_capabilities(self) -> Dict[str, Any]:
        """Get autonomous capabilities summary"""
        return {
            'persistent_memory': True,
            'ml_decision_making': True,
            'learning_capabilities': True,
            'autonomous_planning': True,
            'memory_stats': self.memory_system.get_memory_stats(),
            'learning_stats': self.learning_framework.get_learning_stats(),
            'decision_stats': self.decision_engine.get_decision_stats(),
            'performance_metrics': self._get_performance_metrics()
        }