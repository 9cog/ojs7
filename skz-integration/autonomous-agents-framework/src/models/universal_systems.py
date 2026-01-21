"""
Universal Systems for SKZ Autonomous Agents
Provides the exact interface specified in the issue requirements
"""

from .memory_system import PersistentMemorySystem
from .learning_framework import LearningFramework
from .decision_engine import DecisionEngine
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector storage component for the persistent memory system"""
    
    def __init__(self, memory_system: PersistentMemorySystem = None):
        self.memory_system = memory_system
    
    def store_vector(self, content_hash: str, embedding, metadata=None):
        """Store a vector embedding"""
        if self.memory_system is None:
            raise RuntimeError("VectorStore not connected to memory system")
        return self.memory_system.store_vector_embedding(content_hash, embedding, metadata)
    
    def find_similar(self, query_embedding, limit=5):
        """Find similar vectors"""
        if self.memory_system is None:
            raise RuntimeError("VectorStore not connected to memory system")
        return self.memory_system.find_similar_vectors(query_embedding, limit)

class KnowledgeGraph:
    """Knowledge graph component for the persistent memory system"""
    
    def __init__(self, memory_system: PersistentMemorySystem = None):
        self.memory_system = memory_system
    
    def add_relationship(self, source_id: str, target_id: str, relationship_type: str, 
                        confidence_score: float = 1.0, metadata=None):
        """Add a knowledge relationship"""
        if self.memory_system is None:
            raise RuntimeError("KnowledgeGraph not connected to memory system")
        return self.memory_system.store_knowledge_relationship(
            source_id, target_id, relationship_type, confidence_score, metadata
        )

class ExperienceDatabase:
    """Experience database component for the persistent memory system"""
    
    def __init__(self, memory_system: PersistentMemorySystem = None):
        self.memory_system = memory_system
    
    def log_experience(self, agent_id: str, action_type: str, input_data, output_data, 
                      success: bool, performance_metrics=None):
        """Log an experience"""
        if self.memory_system is None:
            raise RuntimeError("ExperienceDatabase not connected to memory system")
        return self.memory_system.log_experience(
            agent_id, action_type, input_data, output_data, success, performance_metrics
        )
    
    def get_experiences(self, agent_id: str, action_type: str = None, limit: int = 50):
        """Get agent experiences"""
        if self.memory_system is None:
            raise RuntimeError("ExperienceDatabase not connected to memory system")
        return self.memory_system.get_agent_experiences(agent_id, action_type, limit)

class ContextMemory:
    """Context memory component for the persistent memory system"""
    
    def __init__(self, memory_system: PersistentMemorySystem = None):
        self.memory_system = memory_system
    
    def store_context(self, agent_id: str, content, metadata=None, importance_score: float = 0.5, tags=None):
        """Store context information"""
        if self.memory_system is None:
            raise RuntimeError("ContextMemory not connected to memory system")
        return self.memory_system.store_memory(
            agent_id, 'context', content, metadata, importance_score, tags
        )
    
    def retrieve_context(self, agent_id: str, query: str = None, limit: int = 10, min_importance: float = 0.0):
        """Retrieve context information"""
        if self.memory_system is None:
            raise RuntimeError("ContextMemory not connected to memory system")
        return self.memory_system.retrieve_memory(agent_id, 'context', query, limit, min_importance)

class ReinforcementLearner:
    """Reinforcement learning component wrapper"""
    
    def __init__(self, learning_framework: LearningFramework):
        self.learner = learning_framework.reinforcement_learner

class SupervisedLearner:
    """Supervised learning component wrapper"""
    
    def __init__(self, learning_framework: LearningFramework):
        self.learner = learning_framework.supervised_learner

class UnsupervisedLearner:
    """Unsupervised learning component wrapper"""
    
    def __init__(self, learning_framework: LearningFramework):
        self.learner = learning_framework.unsupervised_learner

class MetaLearner:
    """Meta learning component wrapper"""
    
    def __init__(self, learning_framework: LearningFramework):
        self.learner = learning_framework.meta_learner

class GoalManager:
    """Goal management component wrapper"""
    
    def __init__(self, decision_engine: DecisionEngine = None, agent_id: str = None, db_path: str = None):
        if decision_engine is not None:
            # Wrapper mode - delegate to existing decision engine component
            self.manager = decision_engine.goal_manager
            self.agent_id = decision_engine.agent_id
            self.db_path = decision_engine.db_path
        else:
            # Standalone mode for the exact interface specified in issue
            if agent_id is None:
                agent_id = "universal_agent"
            if db_path is None:
                db_path = f"goal_manager_{agent_id}.db"
            
            from .decision_engine import GoalManager as CoreGoalManager
            self.manager = CoreGoalManager(agent_id, db_path)
            self.agent_id = agent_id
            self.db_path = db_path
    
    def __getattr__(self, name):
        # Delegate all method calls to the underlying manager
        return getattr(self.manager, name)

class ConstraintHandler:
    """Constraint handling component wrapper"""
    
    def __init__(self, decision_engine: DecisionEngine = None, agent_id: str = None, db_path: str = None):
        if decision_engine is not None:
            # Wrapper mode - delegate to existing decision engine component
            self.handler = decision_engine.constraint_handler
            self.agent_id = decision_engine.agent_id
            self.db_path = decision_engine.db_path
        else:
            # Standalone mode for the exact interface specified in issue
            if agent_id is None:
                agent_id = "universal_agent"
            if db_path is None:
                db_path = f"constraint_handler_{agent_id}.db"
            
            from .decision_engine import ConstraintHandler as CoreConstraintHandler
            self.handler = CoreConstraintHandler(agent_id, db_path)
            self.agent_id = agent_id
            self.db_path = db_path
    
    def __getattr__(self, name):
        # Delegate all method calls to the underlying handler
        return getattr(self.handler, name)

class RiskAssessor:
    """Risk assessment component wrapper"""
    
    def __init__(self, decision_engine: DecisionEngine = None, agent_id: str = None, db_path: str = None):
        if decision_engine is not None:
            # Wrapper mode - delegate to existing decision engine component
            self.assessor = decision_engine.risk_assessor
            self.agent_id = decision_engine.agent_id
            self.db_path = decision_engine.db_path
        else:
            # Standalone mode for the exact interface specified in issue
            if agent_id is None:
                agent_id = "universal_agent"
            if db_path is None:
                db_path = f"risk_assessor_{agent_id}.db"
            
            from .decision_engine import RiskAssessor as CoreRiskAssessor
            self.assessor = CoreRiskAssessor(agent_id, db_path)
            self.agent_id = agent_id
            self.db_path = db_path
    
    def __getattr__(self, name):
        # Delegate all method calls to the underlying assessor
        return getattr(self.assessor, name)

class AdaptivePlanner:
    """Adaptive planning component wrapper"""
    
    def __init__(self, decision_engine: DecisionEngine = None, agent_id: str = None, db_path: str = None):
        if decision_engine is not None:
            # Wrapper mode - delegate to existing decision engine component
            self.planner = decision_engine.adaptive_planner
            self.agent_id = decision_engine.agent_id
            self.db_path = decision_engine.db_path
        else:
            # Standalone mode for the exact interface specified in issue
            if agent_id is None:
                agent_id = "universal_agent"
            if db_path is None:
                db_path = f"adaptive_planner_{agent_id}.db"
            
            from .decision_engine import AdaptivePlanner as CoreAdaptivePlanner
            self.planner = CoreAdaptivePlanner(agent_id, db_path)
            self.agent_id = agent_id
            self.db_path = db_path
    
    def __getattr__(self, name):
        # Delegate all method calls to the underlying planner
        return getattr(self.planner, name)

def create_universal_memory_system(agent_id: str, db_path: str = None) -> PersistentMemorySystem:
    """
    Create universal memory system with all required components
    
    Returns:
        PersistentMemorySystem with VectorStore, KnowledgeGraph, ExperienceDatabase, ContextMemory
    """
    if db_path is None:
        db_path = f"agent_memory_{agent_id}.db"
    
    memory_system = PersistentMemorySystem(db_path)
    
    # Add component references for easy access
    memory_system.vector_store = VectorStore(memory_system)
    memory_system.knowledge_graph = KnowledgeGraph(memory_system)
    memory_system.experience_db = ExperienceDatabase(memory_system)
    memory_system.context_memory = ContextMemory(memory_system)
    
    logger.info(f"Created universal memory system for agent {agent_id}")
    return memory_system

def create_universal_learning_framework(agent_id: str, db_path: str = None) -> LearningFramework:
    """
    Create universal learning framework with all required learners
    
    Returns:
        LearningFramework with ReinforcementLearner, SupervisedLearner, UnsupervisedLearner, MetaLearner
    """
    if db_path is None:
        db_path = f"learning_framework_{agent_id}.db"
    
    learning_framework = LearningFramework(agent_id, db_path)
    
    # Add wrapper components for easy access
    learning_framework.reinforcement_learner_wrapper = ReinforcementLearner(learning_framework)
    learning_framework.supervised_learner_wrapper = SupervisedLearner(learning_framework)
    learning_framework.unsupervised_learner_wrapper = UnsupervisedLearner(learning_framework)
    learning_framework.meta_learner_wrapper = MetaLearner(learning_framework)
    
    logger.info(f"Created universal learning framework for agent {agent_id}")
    return learning_framework

def create_universal_decision_engine(agent_id: str, db_path: str = None) -> DecisionEngine:
    """
    Create universal decision engine with all required components
    
    Returns:
        DecisionEngine with GoalManager, ConstraintHandler, RiskAssessor, AdaptivePlanner
    """
    if db_path is None:
        db_path = f"decision_engine_{agent_id}.db"
    
    decision_engine = DecisionEngine(agent_id, db_path)
    
    # Add wrapper components for easy access
    decision_engine.goal_manager_wrapper = GoalManager(decision_engine)
    decision_engine.constraint_handler_wrapper = ConstraintHandler(decision_engine)
    decision_engine.risk_assessor_wrapper = RiskAssessor(decision_engine)
    decision_engine.adaptive_planner_wrapper = AdaptivePlanner(decision_engine)
    
    logger.info(f"Created universal decision engine for agent {agent_id}")
    return decision_engine

def create_decision_engine_with_components(goal_manager=None, constraint_handler=None, 
                                         risk_assessor=None, adaptive_planner=None) -> DecisionEngine:
    """
    Create decision engine using the exact interface specified in the issue:
    
    decision_engine = DecisionEngine(
        goal_manager=GoalManager(),
        constraint_handler=ConstraintHandler(),
        risk_assessor=RiskAssessor(),
        adaptive_planner=AdaptivePlanner()
    )
    
    Returns:
        DecisionEngine with provided components
    """
    # Create default components if not provided
    if goal_manager is None:
        goal_manager = GoalManager()
    if constraint_handler is None:
        constraint_handler = ConstraintHandler()
    if risk_assessor is None:
        risk_assessor = RiskAssessor()
    if adaptive_planner is None:
        adaptive_planner = AdaptivePlanner()
    
    # Create decision engine with the provided components
    decision_engine = DecisionEngine(
        goal_manager=goal_manager,
        constraint_handler=constraint_handler,
        risk_assessor=risk_assessor,
        adaptive_planner=adaptive_planner
    )
    
    logger.info(f"Created decision engine with provided components for agent {decision_engine.agent_id}")
    return decision_engine