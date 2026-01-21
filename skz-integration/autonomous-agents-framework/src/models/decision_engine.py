"""
Universal Decision Engine for Autonomous Agents
Critical component providing goal management, constraint handling, risk assessment, and adaptive planning
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import os
from .ab_testing import choose_variant
from .model_registry import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class RiskLevel(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Goal:
    """Represents a goal for an agent"""
    id: str
    agent_id: str
    description: str
    priority: Priority
    target_metrics: Dict[str, Any]
    deadline: Optional[datetime]
    status: str  # 'active', 'completed', 'paused', 'failed'
    progress: float  # 0.0 to 1.0
    created_at: datetime
    updated_at: datetime

@dataclass
class Constraint:
    """Represents a constraint for decision making"""
    id: str
    agent_id: str
    constraint_type: str  # 'resource', 'time', 'quality', 'policy'
    description: str
    parameters: Dict[str, Any]
    strict: bool  # Whether constraint is hard (strict) or soft
    priority: Priority
    active: bool
    created_at: datetime

@dataclass
class RiskFactor:
    """Represents a risk factor"""
    id: str
    agent_id: str
    risk_type: str
    description: str
    probability: float  # 0.0 to 1.0
    impact: float  # 0.0 to 1.0
    risk_level: RiskLevel
    mitigation_strategies: List[str]
    monitoring_metrics: List[str]
    created_at: datetime

@dataclass
class Plan:
    """Represents an adaptive plan"""
    id: str
    agent_id: str
    goal_id: str
    description: str
    steps: List[Dict[str, Any]]
    estimated_duration: int  # minutes
    resource_requirements: Dict[str, Any]
    success_probability: float
    contingency_plans: List[Dict[str, Any]]
    status: str  # 'draft', 'active', 'completed', 'failed'
    created_at: datetime
    updated_at: datetime

class GoalManager:
    """Manages agent goals and tracks progress"""
    
    def __init__(self, agent_id: str, db_path: str):
        self.agent_id = agent_id
        self.db_path = db_path
        self.lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """Initialize goals database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    target_metrics TEXT NOT NULL,
                    deadline TEXT,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_agent ON goals(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)")
    
    def create_goal(self, description: str, target_metrics: Dict[str, Any], 
                   priority: Priority = Priority.MEDIUM, deadline: Optional[datetime] = None) -> str:
        """Create a new goal"""
        with self.lock:
            # Handle string priority input
            if isinstance(priority, str):
                priority_map = {
                    'low': Priority.LOW,
                    'medium': Priority.MEDIUM,
                    'high': Priority.HIGH,
                    'critical': Priority.CRITICAL
                }
                priority = priority_map.get(priority.lower(), Priority.MEDIUM)
            
            goal_id = f"goal_{self.agent_id}_{datetime.now().timestamp()}"
            goal = Goal(
                id=goal_id,
                agent_id=self.agent_id,
                description=description,
                priority=priority,
                target_metrics=target_metrics,
                deadline=deadline,
                status="active",
                progress=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO goals 
                    (id, agent_id, description, priority, target_metrics, deadline, status, progress, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    goal.id, goal.agent_id, goal.description, goal.priority.value,
                    json.dumps(goal.target_metrics), 
                    goal.deadline.isoformat() if goal.deadline else None,
                    goal.status, goal.progress, goal.created_at.isoformat(), goal.updated_at.isoformat()
                ))
                conn.commit()
            
            logger.info(f"Created goal {goal_id} for agent {self.agent_id}")
            return goal_id
    
    def update_goal_progress(self, goal_id: str, progress: float, status: Optional[str] = None) -> bool:
        """Update goal progress"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                updates = ["progress = ?", "updated_at = ?"]
                values = [progress, datetime.now().isoformat()]
                
                if status:
                    updates.append("status = ?")
                    values.append(status)
                
                values.append(goal_id)
                values.append(self.agent_id)
                
                cursor = conn.execute(f"""
                    UPDATE goals SET {', '.join(updates)}
                    WHERE id = ? AND agent_id = ?
                """, values)
                
                success = cursor.rowcount > 0
                conn.commit()
                
                if success:
                    logger.info(f"Updated goal {goal_id} progress to {progress}")
                return success
    
    def get_active_goals(self) -> List[Goal]:
        """Get all active goals for the agent"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM goals 
                    WHERE agent_id = ? AND status = 'active'
                    ORDER BY priority DESC, created_at ASC
                """, (self.agent_id,))
                
                goals = []
                for row in cursor.fetchall():
                    goals.append(Goal(
                        id=row[0], agent_id=row[1], description=row[2],
                        priority=Priority(row[3]), target_metrics=json.loads(row[4]),
                        deadline=datetime.fromisoformat(row[5]) if row[5] else None,
                        status=row[6], progress=row[7],
                        created_at=datetime.fromisoformat(row[8]),
                        updated_at=datetime.fromisoformat(row[9])
                    ))
                
                return goals

class ConstraintHandler:
    """Handles constraints for decision making"""
    
    def __init__(self, agent_id: str, db_path: str):
        self.agent_id = agent_id
        self.db_path = db_path
        self.lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """Initialize constraints database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS constraints (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    constraint_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    strict BOOLEAN NOT NULL,
                    priority TEXT NOT NULL,
                    active BOOLEAN NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_constraints_agent ON constraints(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_constraints_active ON constraints(active)")
    
    def add_constraint(self, constraint_type: str, description: str, parameters: Dict[str, Any],
                      strict: bool = True, priority: Priority = Priority.MEDIUM) -> str:
        """Add a new constraint"""
        with self.lock:
            # Handle string priority input
            if isinstance(priority, str):
                priority_map = {
                    'low': Priority.LOW,
                    'medium': Priority.MEDIUM,
                    'high': Priority.HIGH,
                    'critical': Priority.CRITICAL
                }
                priority = priority_map.get(priority.lower(), Priority.MEDIUM)
            
            constraint_id = f"constraint_{self.agent_id}_{datetime.now().timestamp()}"
            constraint = Constraint(
                id=constraint_id,
                agent_id=self.agent_id,
                constraint_type=constraint_type,
                description=description,
                parameters=parameters,
                strict=strict,
                priority=priority,
                active=True,
                created_at=datetime.now()
            )
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO constraints 
                    (id, agent_id, constraint_type, description, parameters, strict, priority, active, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    constraint.id, constraint.agent_id, constraint.constraint_type,
                    constraint.description, json.dumps(constraint.parameters),
                    constraint.strict, constraint.priority.value, constraint.active,
                    constraint.created_at.isoformat()
                ))
                conn.commit()
            
            logger.info(f"Added constraint {constraint_id} for agent {self.agent_id}")
            return constraint_id
    
    def validate_decision(self, decision_context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a decision against active constraints"""
        with self.lock:
            violations = []
            can_proceed = True
            
            constraints = self.get_active_constraints()
            
            for constraint in constraints:
                is_violated = self._check_constraint_violation(constraint, decision_context)
                
                if is_violated:
                    violation_msg = f"Constraint '{constraint.description}' violated"
                    violations.append(violation_msg)
                    
                    if constraint.strict:
                        can_proceed = False
            
            return can_proceed, violations
    
    def _check_constraint_violation(self, constraint: Constraint, decision_context: Dict[str, Any]) -> bool:
        """Check if a specific constraint is violated"""
        # Resource constraints
        if constraint.constraint_type == "resource":
            required_resources = decision_context.get("required_resources", {})
            available_resources = constraint.parameters.get("available_resources", {})
            
            for resource, amount in required_resources.items():
                if resource in available_resources:
                    if amount > available_resources[resource]:
                        return True
        
        # Time constraints
        elif constraint.constraint_type == "time":
            estimated_duration = decision_context.get("estimated_duration", 0)
            max_duration = constraint.parameters.get("max_duration", float('inf'))
            
            if estimated_duration > max_duration:
                return True
        
        # Quality constraints
        elif constraint.constraint_type == "quality":
            quality_score = decision_context.get("quality_score", 0.0)
            min_quality = constraint.parameters.get("min_quality", 0.0)
            
            if quality_score < min_quality:
                return True
        
        # Policy constraints
        elif constraint.constraint_type == "policy":
            action_type = decision_context.get("action_type", "")
            forbidden_actions = constraint.parameters.get("forbidden_actions", [])
            
            if action_type in forbidden_actions:
                return True
        
        return False
    
    def get_active_constraints(self) -> List[Constraint]:
        """Get all active constraints"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM constraints 
                    WHERE agent_id = ? AND active = 1
                    ORDER BY priority DESC, created_at ASC
                """, (self.agent_id,))
                
                constraints = []
                for row in cursor.fetchall():
                    constraints.append(Constraint(
                        id=row[0], agent_id=row[1], constraint_type=row[2],
                        description=row[3], parameters=json.loads(row[4]),
                        strict=bool(row[5]), priority=Priority(row[6]),
                        active=bool(row[7]), created_at=datetime.fromisoformat(row[8])
                    ))
                
                return constraints

class RiskAssessor:
    """Assesses and manages risks in decision making"""
    
    def __init__(self, agent_id: str, db_path: str):
        self.agent_id = agent_id
        self.db_path = db_path
        self.lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """Initialize risk assessment database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_factors (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    risk_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    probability REAL NOT NULL,
                    impact REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    mitigation_strategies TEXT NOT NULL,
                    monitoring_metrics TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_agent ON risk_factors(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_level ON risk_factors(risk_level)")
    
    def assess_decision_risk(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk for a decision"""
        with self.lock:
            risk_factors = self.get_relevant_risk_factors(decision_context)
            
            # Calculate overall risk score
            total_risk_score = 0.0
            max_risk_level = RiskLevel.MINIMAL
            active_risks = []
            
            for risk_factor in risk_factors:
                risk_score = risk_factor.probability * risk_factor.impact
                total_risk_score += risk_score
                
                if risk_factor.risk_level.value > max_risk_level.value:
                    max_risk_level = risk_factor.risk_level
                
                if risk_score > 0.3:  # Significant risk threshold
                    active_risks.append({
                        'id': risk_factor.id,
                        'type': risk_factor.risk_type,
                        'description': risk_factor.description,
                        'score': risk_score,
                        'level': risk_factor.risk_level.value,
                        'mitigation': risk_factor.mitigation_strategies
                    })
            
            # Normalize risk score
            normalized_risk = min(total_risk_score / len(risk_factors) if risk_factors else 0.0, 1.0)
            
            return {
                'overall_risk_score': normalized_risk,
                'risk_level': max_risk_level.value,
                'active_risks': active_risks,
                'risk_factors_count': len(risk_factors),
                'recommendation': self._get_risk_recommendation(normalized_risk, max_risk_level)
            }
    
    def add_risk_factor(self, risk_type: str, description: str, probability: float, impact: float,
                       mitigation_strategies: Optional[List[str]] = None, monitoring_metrics: Optional[List[str]] = None) -> str:
        """Add a new risk factor"""
        with self.lock:
            risk_id = f"risk_{self.agent_id}_{datetime.now().timestamp()}"
            
            # Calculate risk level based on probability and impact
            risk_score = probability * impact
            if risk_score >= 0.8:
                risk_level = RiskLevel.CRITICAL
            elif risk_score >= 0.6:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 0.4:
                risk_level = RiskLevel.MEDIUM
            elif risk_score >= 0.2:
                risk_level = RiskLevel.LOW
            else:
                risk_level = RiskLevel.MINIMAL
            
            risk_factor = RiskFactor(
                id=risk_id,
                agent_id=self.agent_id,
                risk_type=risk_type,
                description=description,
                probability=probability,
                impact=impact,
                risk_level=risk_level,
                mitigation_strategies=mitigation_strategies or [],
                monitoring_metrics=monitoring_metrics or [],
                created_at=datetime.now()
            )
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO risk_factors 
                    (id, agent_id, risk_type, description, probability, impact, risk_level, 
                     mitigation_strategies, monitoring_metrics, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    risk_factor.id, risk_factor.agent_id, risk_factor.risk_type,
                    risk_factor.description, risk_factor.probability, risk_factor.impact,
                    risk_factor.risk_level.value, json.dumps(risk_factor.mitigation_strategies),
                    json.dumps(risk_factor.monitoring_metrics), risk_factor.created_at.isoformat()
                ))
                conn.commit()
            
            logger.info(f"Added risk factor {risk_id} for agent {self.agent_id}")
            return risk_id
    
    def get_relevant_risk_factors(self, decision_context: Dict[str, Any]) -> List[RiskFactor]:
        """Get risk factors relevant to a decision context"""
        with self.lock:
            # For now, return all risk factors. In production, this would be filtered
            # based on the decision context
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM risk_factors 
                    WHERE agent_id = ?
                    ORDER BY probability * impact DESC
                """, (self.agent_id,))
                
                risk_factors = []
                for row in cursor.fetchall():
                    risk_factors.append(RiskFactor(
                        id=row[0], agent_id=row[1], risk_type=row[2],
                        description=row[3], probability=row[4], impact=row[5],
                        risk_level=RiskLevel(row[6]), 
                        mitigation_strategies=json.loads(row[7]),
                        monitoring_metrics=json.loads(row[8]),
                        created_at=datetime.fromisoformat(row[9])
                    ))
                
                return risk_factors
    
    def _get_risk_recommendation(self, risk_score: float, risk_level: RiskLevel) -> str:
        """Get risk-based recommendation"""
        if risk_level == RiskLevel.CRITICAL or risk_score >= 0.8:
            return "HIGH_RISK: Recommend against proceeding without significant risk mitigation"
        elif risk_level == RiskLevel.HIGH or risk_score >= 0.6:
            return "MODERATE_RISK: Proceed with caution and enhanced monitoring"
        elif risk_level == RiskLevel.MEDIUM or risk_score >= 0.4:
            return "ACCEPTABLE_RISK: Proceed with standard monitoring"
        else:
            return "LOW_RISK: Safe to proceed"

class AdaptivePlanner:
    """Creates and adapts plans based on goals, constraints, and risks"""
    
    def __init__(self, agent_id: str, db_path: str):
        self.agent_id = agent_id
        self.db_path = db_path
        self.lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """Initialize planning database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plans (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    goal_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    steps TEXT NOT NULL,
                    estimated_duration INTEGER NOT NULL,
                    resource_requirements TEXT NOT NULL,
                    success_probability REAL NOT NULL,
                    contingency_plans TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_plans_agent ON plans(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_plans_goal ON plans(goal_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status)")
    
    def create_plan(self, goal: Goal, constraints: List[Constraint], 
                   risk_assessment: Dict[str, Any]) -> str:
        """Create an adaptive plan for achieving a goal"""
        with self.lock:
            plan_id = f"plan_{self.agent_id}_{datetime.now().timestamp()}"
            
            # Generate plan steps based on goal and constraints
            steps = self._generate_plan_steps(goal, constraints, risk_assessment)
            
            # Estimate duration and resources
            estimated_duration = self._estimate_duration(steps)
            resource_requirements = self._estimate_resources(steps)
            
            # Calculate success probability
            success_probability = self._calculate_success_probability(goal, constraints, risk_assessment)
            
            # Generate contingency plans
            contingency_plans = self._generate_contingency_plans(goal, risk_assessment)
            
            plan = Plan(
                id=plan_id,
                agent_id=self.agent_id,
                goal_id=goal.id,
                description=f"Plan to achieve: {goal.description}",
                steps=steps,
                estimated_duration=estimated_duration,
                resource_requirements=resource_requirements,
                success_probability=success_probability,
                contingency_plans=contingency_plans,
                status="draft",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO plans 
                    (id, agent_id, goal_id, description, steps, estimated_duration, 
                     resource_requirements, success_probability, contingency_plans, 
                     status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    plan.id, plan.agent_id, plan.goal_id, plan.description,
                    json.dumps(plan.steps), plan.estimated_duration,
                    json.dumps(plan.resource_requirements), plan.success_probability,
                    json.dumps(plan.contingency_plans), plan.status,
                    plan.created_at.isoformat(), plan.updated_at.isoformat()
                ))
                conn.commit()
            
            logger.info(f"Created plan {plan_id} for goal {goal.id}")
            return plan_id
    
    def adapt_plan(self, plan_id: str, execution_feedback: Dict[str, Any]) -> bool:
        """Adapt a plan based on execution feedback"""
        with self.lock:
            plan = self.get_plan(plan_id)
            if not plan:
                return False
            
            # Analyze feedback and adapt plan
            adaptations = self._analyze_execution_feedback(execution_feedback)
            
            if adaptations['requires_adaptation']:
                # Update plan steps
                updated_steps = self._adapt_plan_steps(plan.steps, adaptations)
                
                # Recalculate estimates
                new_duration = self._estimate_duration(updated_steps)
                new_success_probability = max(0.1, plan.success_probability + adaptations['probability_adjustment'])
                
                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE plans SET 
                        steps = ?, estimated_duration = ?, success_probability = ?, updated_at = ?
                        WHERE id = ? AND agent_id = ?
                    """, (
                        json.dumps(updated_steps), new_duration, new_success_probability,
                        datetime.now().isoformat(), plan_id, self.agent_id
                    ))
                    conn.commit()
                
                logger.info(f"Adapted plan {plan_id} based on execution feedback")
                return True
            
            return False
    
    def _generate_plan_steps(self, goal: Goal, constraints: List[Constraint], 
                           risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate plan steps based on goal and context"""
        steps = []
        
        # Basic planning logic - can be enhanced with more sophisticated algorithms
        target_metrics = goal.target_metrics
        
        # Step 1: Analysis and preparation
        steps.append({
            'step_number': 1,
            'description': 'Analyze requirements and prepare resources',
            'action_type': 'analysis',
            'estimated_duration': 30,  # minutes
            'required_resources': {'cpu': 0.2, 'memory': 0.1},
            'success_criteria': ['requirements_analyzed', 'resources_prepared'],
            'risk_factors': [r['type'] for r in risk_assessment.get('active_risks', [])]
        })
        
        # Step 2: Implementation based on goal type
        if 'research' in goal.description.lower():
            steps.append({
                'step_number': 2,
                'description': 'Conduct research and gather information',
                'action_type': 'research',
                'estimated_duration': 120,
                'required_resources': {'cpu': 0.5, 'memory': 0.3, 'network': 0.4},
                'success_criteria': ['data_gathered', 'analysis_complete'],
                'risk_factors': ['data_quality', 'research_scope']
            })
        elif 'review' in goal.description.lower():
            steps.append({
                'step_number': 2,
                'description': 'Conduct review and evaluation',
                'action_type': 'review',
                'estimated_duration': 90,
                'required_resources': {'cpu': 0.3, 'memory': 0.2},
                'success_criteria': ['review_complete', 'evaluation_done'],
                'risk_factors': ['reviewer_availability', 'quality_standards']
            })
        else:
            steps.append({
                'step_number': 2,
                'description': 'Execute primary action',
                'action_type': 'execution',
                'estimated_duration': 60,
                'required_resources': {'cpu': 0.4, 'memory': 0.2},
                'success_criteria': ['action_complete'],
                'risk_factors': ['execution_complexity']
            })
        
        # Step 3: Validation and finalization
        steps.append({
            'step_number': 3,
            'description': 'Validate results and finalize',
            'action_type': 'validation',
            'estimated_duration': 45,
            'required_resources': {'cpu': 0.2, 'memory': 0.1},
            'success_criteria': ['results_validated', 'quality_confirmed'],
            'risk_factors': ['validation_criteria', 'quality_metrics']
        })
        
        return steps
    
    def _estimate_duration(self, steps: List[Dict[str, Any]]) -> int:
        """Estimate total duration for plan steps"""
        return sum(step.get('estimated_duration', 30) for step in steps)
    
    def _estimate_resources(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate resource requirements for plan"""
        total_resources = {'cpu': 0.0, 'memory': 0.0, 'network': 0.0, 'storage': 0.0}
        
        for step in steps:
            step_resources = step.get('required_resources', {})
            for resource, amount in step_resources.items():
                total_resources[resource] = max(total_resources.get(resource, 0), amount)
        
        return total_resources
    
    def _calculate_success_probability(self, goal: Goal, constraints: List[Constraint], 
                                     risk_assessment: Dict[str, Any]) -> float:
        """Calculate probability of plan success"""
        base_probability = 0.8  # Start optimistic
        
        # Adjust for goal complexity (based on priority and metrics)
        if goal.priority == Priority.CRITICAL:
            base_probability -= 0.1
        elif goal.priority == Priority.LOW:
            base_probability += 0.05
        
        # Adjust for constraints
        strict_constraints = sum(1 for c in constraints if c.strict)
        base_probability -= strict_constraints * 0.05
        
        # Adjust for risks
        risk_score = risk_assessment.get('overall_risk_score', 0.0)
        base_probability -= risk_score * 0.3
        
        return max(0.1, min(0.95, base_probability))
    
    def _generate_contingency_plans(self, goal: Goal, risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate contingency plans for high-risk scenarios"""
        contingencies = []
        
        active_risks = risk_assessment.get('active_risks', [])
        
        for risk in active_risks:
            if risk['score'] > 0.5:  # High-impact risks
                contingencies.append({
                    'trigger_condition': f"Risk '{risk['type']}' manifests",
                    'alternative_approach': f"Implement mitigation: {risk['mitigation'][0] if risk['mitigation'] else 'fallback_strategy'}",
                    'resource_adjustment': {'time_buffer': 1.5, 'resource_buffer': 1.2},
                    'success_probability_adjustment': -0.2
                })
        
        # Default contingency for unexpected failures
        contingencies.append({
            'trigger_condition': "Primary plan fails",
            'alternative_approach': "Simplified approach with reduced scope",
            'resource_adjustment': {'time_buffer': 1.2, 'resource_buffer': 1.1},
            'success_probability_adjustment': -0.3
        })
        
        return contingencies
    
    def _analyze_execution_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze execution feedback to determine needed adaptations"""
        adaptations = {
            'requires_adaptation': False,
            'probability_adjustment': 0.0,
            'step_modifications': [],
            'resource_adjustments': {}
        }
        
        # Check if execution is behind schedule
        if feedback.get('time_ratio', 1.0) > 1.2:
            adaptations['requires_adaptation'] = True
            adaptations['probability_adjustment'] -= 0.1
            adaptations['step_modifications'].append('extend_timeframes')
        
        # Check if resources are insufficient
        if feedback.get('resource_utilization', 0.5) > 0.9:
            adaptations['requires_adaptation'] = True
            adaptations['resource_adjustments']['increase_allocation'] = True
        
        # Check for quality issues
        if feedback.get('quality_score', 0.8) < 0.6:
            adaptations['requires_adaptation'] = True
            adaptations['probability_adjustment'] -= 0.15
            adaptations['step_modifications'].append('add_quality_checks')
        
        return adaptations
    
    def _adapt_plan_steps(self, current_steps: List[Dict[str, Any]], 
                         adaptations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adapt plan steps based on analysis"""
        adapted_steps = current_steps.copy()
        
        if 'extend_timeframes' in adaptations.get('step_modifications', []):
            for step in adapted_steps:
                step['estimated_duration'] = int(step['estimated_duration'] * 1.3)
        
        if 'add_quality_checks' in adaptations.get('step_modifications', []):
            # Add quality validation step
            quality_step = {
                'step_number': len(adapted_steps) + 1,
                'description': 'Additional quality validation',
                'action_type': 'quality_check',
                'estimated_duration': 30,
                'required_resources': {'cpu': 0.1, 'memory': 0.05},
                'success_criteria': ['quality_verified', 'standards_met'],
                'risk_factors': ['quality_metrics']
            }
            adapted_steps.append(quality_step)
        
        return adapted_steps
    
    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by ID"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM plans WHERE id = ? AND agent_id = ?
                """, (plan_id, self.agent_id))
                
                row = cursor.fetchone()
                if row:
                    return Plan(
                        id=row[0], agent_id=row[1], goal_id=row[2],
                        description=row[3], steps=json.loads(row[4]),
                        estimated_duration=row[5], resource_requirements=json.loads(row[6]),
                        success_probability=row[7], contingency_plans=json.loads(row[8]),
                        status=row[9], created_at=datetime.fromisoformat(row[10]),
                        updated_at=datetime.fromisoformat(row[11])
                    )
                
                return None

class DecisionEngine:
    """
    Universal decision engine that coordinates goal management, constraint handling,
    risk assessment, and adaptive planning
    """
    
    def __init__(self, agent_id: Optional[str] = None, db_path: str = "decision_engine.db", 
                 goal_manager=None, constraint_handler=None, risk_assessor=None, adaptive_planner=None):
        self.lock = threading.RLock()
        
        # Support both the new interface (with components as parameters) and backward compatibility
        if goal_manager is not None and constraint_handler is not None and risk_assessor is not None and adaptive_planner is not None:
            # New interface as specified in the issue
            self.goal_manager = goal_manager
            self.constraint_handler = constraint_handler
            self.risk_assessor = risk_assessor
            self.adaptive_planner = adaptive_planner
            
            # Extract agent_id from components if not provided
            if agent_id is None:
                self.agent_id = getattr(goal_manager, 'agent_id', 'universal_agent')
            else:
                self.agent_id = agent_id
            
            self.db_path = db_path
            logger.info(f"Initialized DecisionEngine for agent {self.agent_id} with provided components")
        else:
            # Backward compatibility - original interface
            if agent_id is None:
                raise ValueError("agent_id is required when not providing component instances")
            
            self.agent_id = agent_id
            self.db_path = db_path
            
            # Initialize components
            self.goal_manager = GoalManager(agent_id, db_path)
            self.constraint_handler = ConstraintHandler(agent_id, db_path)
            self.risk_assessor = RiskAssessor(agent_id, db_path)
            self.adaptive_planner = AdaptivePlanner(agent_id, db_path)
            
            logger.info(f"Initialized DecisionEngine for agent {agent_id} with auto-created components")
    
    def make_decision(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a comprehensive decision using all components"""
        with self.lock:
            logger.info(f"Making decision for agent {self.agent_id}")
            
            # Step 1: Get relevant goals
            active_goals = self.goal_manager.get_active_goals()
            
            # Step 2: Check constraints
            can_proceed, constraint_violations = self.constraint_handler.validate_decision(decision_context)
            
            # Step 3: Assess risks
            risk_assessment = self.risk_assessor.assess_decision_risk(decision_context)
            
            # Step 4: Create or adapt plan if proceeding
            recommended_plan = None
            if can_proceed and active_goals:
                primary_goal = active_goals[0]  # Highest priority goal
                constraints = self.constraint_handler.get_active_constraints()
                plan_id = self.adaptive_planner.create_plan(primary_goal, constraints, risk_assessment)
                recommended_plan = self.adaptive_planner.get_plan(plan_id)
            
            variant, _ = choose_variant(decision_context if isinstance(decision_context, dict) else {})
            model_name = os.getenv("DECISION_MODEL_NAME", "ojs_decision_model")
            model_version_hint = os.getenv("DECISION_MODEL_VERSION") or None
            handle = load_model(model_name, model_version_hint)
            resolved_version = handle.version if handle else "unavailable"

            model_score = None
            try:
                if handle and hasattr(handle, "model") and handle.model is not None:
                    from .feature_engineering import basic_manuscript_features, FEATURE_KEYS, to_vector
                    feats = basic_manuscript_features(decision_context if isinstance(decision_context, dict) else {})
                    X = [to_vector(feats, FEATURE_KEYS)]
                    model = handle.model
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X)
                        try:
                            model_score = float(proba[0][1])
                        except Exception:
                            model_score = float(proba[0]) if hasattr(proba, "__getitem__") else None
                    elif hasattr(model, "predict"):
                        pred = model.predict(X)
                        model_score = float(pred[0]) if hasattr(pred, "__getitem__") else None
            except Exception:
                if os.getenv("ENVIRONMENT", "").lower() == "production":
                    raise
                model_score = None

            # Step 5: Formulate decision
            decision = {
                'agent_id': self.agent_id,
                'timestamp': datetime.now().isoformat(),
                'decision_context': decision_context,
                'can_proceed': can_proceed,
                'score': model_score,
                'score': model_score,
                'confidence_score': self._calculate_confidence(can_proceed, risk_assessment, constraint_violations),
                'active_goals': [{'id': g.id, 'description': g.description, 'priority': g.priority.value} for g in active_goals],
                'constraint_violations': constraint_violations,
                'risk_assessment': risk_assessment,
                'recommended_plan': {
                    'id': recommended_plan.id,
                    'description': recommended_plan.description,
                    'steps': len(recommended_plan.steps),
                    'estimated_duration': recommended_plan.estimated_duration,
                    'success_probability': recommended_plan.success_probability
                } if recommended_plan else None,
                'recommendations': self._generate_recommendations(can_proceed, risk_assessment, constraint_violations),
                'variant': variant,
                'model_version': resolved_version
            }
            
            logger.info(f"Decision made for agent {self.agent_id}: {'PROCEED' if can_proceed else 'HALT'}")
            return decision
    
    def _calculate_confidence(self, can_proceed: bool, risk_assessment: Dict[str, Any], 
                            constraint_violations: List[str]) -> float:
        """Calculate confidence score for the decision"""
        base_confidence = 0.8 if can_proceed else 0.2
        
        # Adjust for risk level
        risk_score = risk_assessment.get('overall_risk_score', 0.0)
        base_confidence -= risk_score * 0.3
        
        # Adjust for constraint violations
        base_confidence -= len(constraint_violations) * 0.1
        
        return max(0.1, min(0.95, base_confidence))
    
    def _generate_recommendations(self, can_proceed: bool, risk_assessment: Dict[str, Any],
                                constraint_violations: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if not can_proceed:
            recommendations.append("HALT: Address constraint violations before proceeding")
            for violation in constraint_violations:
                recommendations.append(f"Resolve: {violation}")
        
        risk_level = risk_assessment.get('risk_level', 'minimal')
        if risk_level in ['high', 'critical']:
            recommendations.append("HIGH_RISK: Implement risk mitigation strategies")
            recommendations.append(risk_assessment.get('recommendation', 'Monitor risks closely'))
        
        if can_proceed and risk_level in ['minimal', 'low', 'medium']:
            recommendations.append("PROCEED: Execute with standard monitoring")
            recommendations.append("Monitor progress against plan milestones")
        
        return recommendations
    
    def update_goal_progress(self, goal_id: str, progress: float, execution_feedback: Optional[Dict[str, Any]] = None):
        """Update goal progress and adapt plans if needed"""
        with self.lock:
            # Update goal progress
            success = self.goal_manager.update_goal_progress(goal_id, progress)
            
            if success and execution_feedback:
                # Find and adapt relevant plans
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT id FROM plans WHERE goal_id = ? AND agent_id = ? AND status = 'active'
                    """, (goal_id, self.agent_id))
                    
                    for row in cursor.fetchall():
                        plan_id = row[0]
                        self.adaptive_planner.adapt_plan(plan_id, execution_feedback)
            
            return success
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the decision engine"""
        with self.lock:
            active_goals = self.goal_manager.get_active_goals()
            active_constraints = self.constraint_handler.get_active_constraints()
            
            return {
                'agent_id': self.agent_id,
                'timestamp': datetime.now().isoformat(),
                'active_goals_count': len(active_goals),
                'active_constraints_count': len(active_constraints),
                'components_status': {
                    'goal_manager': 'active',
                    'constraint_handler': 'active',
                    'risk_assessor': 'active',
                    'adaptive_planner': 'active'
                },
                'recent_goals': [
                    {
                        'id': goal.id,
                        'description': goal.description,
                        'progress': goal.progress,
                        'priority': goal.priority.value
                    } for goal in active_goals[:3]
                ]
            }
