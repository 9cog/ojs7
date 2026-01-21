"""
Learning Framework for Autonomous Agents
Phase 2 Critical Component - Enables continuous learning and improvement
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import threading
from collections import defaultdict
import pickle
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningExperience:
    """Represents a learning experience"""
    id: str
    agent_id: str
    action_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    performance_metrics: Dict[str, Any]
    feedback: Dict[str, Any]
    created_at: datetime

@dataclass
class LearningPattern:
    """Represents a learned pattern"""
    id: str
    agent_id: str
    pattern_type: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    success_rate: float
    confidence: float
    created_at: datetime
    last_used: datetime

class ReinforcementLearner:
    """Reinforcement learning for agent behavior optimization"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate
        self.lock = threading.RLock()
        
    def get_action(self, state: str, available_actions: List[str]) -> str:
        """Get action using epsilon-greedy policy"""
        with self.lock:
            if np.random.random() < self.epsilon:
                # Exploration: random action
                return np.random.choice(available_actions)
            else:
                # Exploitation: best action
                return max(available_actions, key=lambda a: self.q_table[state][a])
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning"""
        with self.lock:
            current_q = self.q_table[state][action]
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
            
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
            self.q_table[state][action] = new_q
    
    def get_policy(self) -> Dict[str, Dict[str, float]]:
        """Get current policy"""
        with self.lock:
            return dict(self.q_table)
    
    def save_policy(self, filepath: str):
        """Save policy to file"""
        with self.lock:
            with open(filepath, 'wb') as f:
                pickle.dump(dict(self.q_table), f)
    
    def load_policy(self, filepath: str):
        """Load policy from file"""
        with self.lock:
            try:
                with open(filepath, 'rb') as f:
                    self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(f))
            except FileNotFoundError:
                logger.info(f"No existing policy found for {self.agent_id}")

class SupervisedLearner:
    """Supervised learning for pattern recognition and classification"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.patterns = {}
        self.classifiers = {}
        self.lock = threading.RLock()
        
    def learn_pattern(self, pattern_type: str, input_data: Dict[str, Any], 
                     output_data: Dict[str, Any], success: bool):
        """Learn a pattern from experience"""
        with self.lock:
            if pattern_type not in self.patterns:
                self.patterns[pattern_type] = []
            
            pattern = {
                'input': input_data,
                'output': output_data,
                'success': success,
                'created_at': datetime.now()
            }
            
            self.patterns[pattern_type].append(pattern)
            
            # Keep only recent patterns
            if len(self.patterns[pattern_type]) > 100:
                self.patterns[pattern_type] = self.patterns[pattern_type][-50:]
    
    def find_similar_patterns(self, input_data: Dict[str, Any], pattern_type: str, 
                            threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find similar patterns"""
        with self.lock:
            if pattern_type not in self.patterns:
                return []
            
            similar_patterns = []
            for pattern in self.patterns[pattern_type]:
                similarity = self._calculate_similarity(input_data, pattern['input'])
                if similarity >= threshold:
                    pattern_copy = pattern.copy()
                    pattern_copy['similarity'] = similarity
                    similar_patterns.append(pattern_copy)
            
            # Sort by similarity
            similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_patterns[:5]
    
    def _calculate_similarity(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """Calculate similarity between two data structures"""
        # Simple similarity calculation
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())
        
        if not keys1 and not keys2:
            return 1.0
        if not keys1 or not keys2:
            return 0.0
        
        # Key overlap
        key_overlap = len(keys1.intersection(keys2)) / len(keys1.union(keys2))
        
        # Value similarity for common keys
        value_similarity = 0.0
        common_keys = keys1.intersection(keys2)
        
        for key in common_keys:
            val1 = data1[key]
            val2 = data2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    value_similarity += 1 - abs(val1 - val2) / max_val
                else:
                    value_similarity += 1.0
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity
                if val1.lower() == val2.lower():
                    value_similarity += 1.0
                else:
                    # Simple substring matching
                    val1_lower = val1.lower()
                    val2_lower = val2.lower()
                    if val1_lower in val2_lower or val2_lower in val1_lower:
                        value_similarity += 0.5
            else:
                # Type mismatch
                value_similarity += 0.0
        
        if common_keys:
            value_similarity /= len(common_keys)
        
        # Combine key and value similarity
        return (key_overlap + value_similarity) / 2

class UnsupervisedLearner:
    """Unsupervised learning for discovering patterns and clusters"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.clusters = defaultdict(list)
        self.cluster_centers = {}
        self.lock = threading.RLock()
        
    def discover_patterns(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover patterns in data"""
        with self.lock:
            if not data_points:
                return []
            
            # Simple clustering based on key-value patterns
            clusters = defaultdict(list)
            
            for point in data_points:
                # Create cluster key based on data structure
                cluster_key = self._create_cluster_key(point)
                clusters[cluster_key].append(point)
            
            # Identify significant patterns
            patterns = []
            for cluster_key, cluster_points in clusters.items():
                if len(cluster_points) >= 2:  # Minimum cluster size
                    pattern = {
                        'cluster_key': cluster_key,
                        'frequency': len(cluster_points),
                        'representative_point': cluster_points[0],
                        'all_points': cluster_points,
                        'confidence': min(1.0, len(cluster_points) / 10.0)
                    }
                    patterns.append(pattern)
            
            # Sort by frequency
            patterns.sort(key=lambda x: x['frequency'], reverse=True)
            return patterns
    
    def _create_cluster_key(self, data_point: Dict[str, Any]) -> str:
        """Create a cluster key based on data structure"""
        # Create a hash of the data structure
        keys = sorted(data_point.keys())
        key_str = '_'.join(keys)
        
        # Add value types for better clustering
        value_types = []
        for key in keys:
            value = data_point[key]
            if isinstance(value, (int, float)):
                value_types.append('numeric')
            elif isinstance(value, str):
                value_types.append('string')
            elif isinstance(value, bool):
                value_types.append('boolean')
            elif isinstance(value, list):
                value_types.append('list')
            elif isinstance(value, dict):
                value_types.append('dict')
            else:
                value_types.append('other')
        
        return f"{key_str}_{'_'.join(value_types)}"
    
    def find_anomalies(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find anomalous data points"""
        with self.lock:
            if len(data_points) < 3:
                return []
            
            # Simple anomaly detection based on frequency
            point_counts = defaultdict(int)
            for point in data_points:
                cluster_key = self._create_cluster_key(point)
                point_counts[cluster_key] += 1
            
            # Find low-frequency patterns (potential anomalies)
            total_points = len(data_points)
            anomalies = []
            
            for point in data_points:
                cluster_key = self._create_cluster_key(point)
                frequency = point_counts[cluster_key] / total_points
                
                if frequency < 0.1:  # Less than 10% frequency
                    anomaly = {
                        'point': point,
                        'cluster_key': cluster_key,
                        'frequency': frequency,
                        'anomaly_score': 1.0 - frequency
                    }
                    anomalies.append(anomaly)
            
            # Sort by anomaly score
            anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
            return anomalies

class MetaLearner:
    """Meta-learning for optimizing learning strategies"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.learning_strategies = {}
        self.performance_history = []
        self.lock = threading.RLock()
        
    def optimize_learning_strategy(self, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize learning strategy based on performance"""
        with self.lock:
            self.performance_history.append({
                'performance': current_performance,
                'timestamp': datetime.now()
            })
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-50:]
            
            # Analyze performance trends
            if len(self.performance_history) >= 5:
                recent_performance = self.performance_history[-5:]
                avg_performance = np.mean([p['performance'].get('success_rate', 0.5) 
                                        for p in recent_performance])
                
                # Adjust learning parameters based on performance
                strategy_adjustments = {}
                
                if avg_performance < 0.6:
                    # Poor performance - increase exploration
                    strategy_adjustments['increase_exploration'] = True
                    strategy_adjustments['learning_rate'] = 0.15
                    strategy_adjustments['epsilon'] = 0.2
                elif avg_performance > 0.8:
                    # Good performance - fine-tune
                    strategy_adjustments['fine_tune'] = True
                    strategy_adjustments['learning_rate'] = 0.05
                    strategy_adjustments['epsilon'] = 0.05
                else:
                    # Moderate performance - maintain
                    strategy_adjustments['maintain'] = True
                    strategy_adjustments['learning_rate'] = 0.1
                    strategy_adjustments['epsilon'] = 0.1
                
                return strategy_adjustments
            
            return {'maintain': True, 'learning_rate': 0.1, 'epsilon': 0.1}
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about learning performance"""
        with self.lock:
            if not self.performance_history:
                return {
                    'status': 'no_data',
                    'current_performance': 0.5,
                    'average_performance': 0.5,
                    'performance_trend': 'stable',
                    'learning_efficiency': 0,
                    'recommendations': ['Need more data for meaningful analysis']
                }
            
            recent_performance = [p['performance'].get('success_rate', 0.5) 
                                for p in self.performance_history[-10:]]
            
            insights = {
                'current_performance': recent_performance[-1] if recent_performance else 0.5,
                'average_performance': np.mean(recent_performance),
                'performance_trend': 'improving' if len(recent_performance) >= 2 and 
                                   recent_performance[-1] > recent_performance[0] else 'stable',
                'learning_efficiency': len(self.performance_history),
                'recommendations': self._generate_recommendations(recent_performance)
            }
            
            return insights
    
    def _generate_recommendations(self, performance_history: List[float]) -> List[str]:
        """Generate learning recommendations"""
        recommendations = []
        
        if len(performance_history) < 3:
            recommendations.append("Need more data for meaningful analysis")
            return recommendations
        
        avg_performance = np.mean(performance_history)
        
        if avg_performance < 0.6:
            recommendations.extend([
                "Consider increasing exploration rate",
                "Review and update training data",
                "Check for systematic errors in decision logic"
            ])
        elif avg_performance > 0.8:
            recommendations.extend([
                "Performance is excellent - consider fine-tuning",
                "Explore advanced optimization techniques",
                "Consider reducing exploration for efficiency"
            ])
        else:
            recommendations.extend([
                "Performance is stable - continue current approach",
                "Monitor for improvement opportunities",
                "Consider incremental optimizations"
            ])
        
        return recommendations

class LearningFramework:
    """Main learning framework that coordinates all learning components"""
    
    def __init__(self, agent_id: str = None, db_path: str = "learning_framework.db",
                 reinforcement_learner: 'ReinforcementLearner' = None,
                 supervised_learner: 'SupervisedLearner' = None,
                 unsupervised_learner: 'UnsupervisedLearner' = None,
                 meta_learner: 'MetaLearner' = None):
        """
        Initialize learning framework with optional injectable learners
        
        Args:
            agent_id: Agent identifier (required if learners not provided)
            db_path: Database path for storing learning data
            reinforcement_learner: Optional injectable reinforcement learner
            supervised_learner: Optional injectable supervised learner  
            unsupervised_learner: Optional injectable unsupervised learner
            meta_learner: Optional injectable meta learner
        """
        # Determine agent_id from provided learners if not specified
        if agent_id is None and reinforcement_learner is not None:
            agent_id = getattr(reinforcement_learner, 'agent_id', 'default_agent')
        elif agent_id is None:
            agent_id = 'default_agent'
            
        self.agent_id = agent_id
        self.db_path = db_path
        self.lock = threading.RLock()
        
        # Use provided learners or create new ones
        self.reinforcement_learner = reinforcement_learner if reinforcement_learner is not None else ReinforcementLearner(agent_id)
        self.supervised_learner = supervised_learner if supervised_learner is not None else SupervisedLearner(agent_id)
        self.unsupervised_learner = unsupervised_learner if unsupervised_learner is not None else UnsupervisedLearner(agent_id)
        self.meta_learner = meta_learner if meta_learner is not None else MetaLearner(agent_id)
        
        self._init_database()
        
    def _init_database(self):
        """Initialize learning database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_experiences (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    output_data TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    feedback TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used TEXT NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experience_agent ON learning_experiences(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pattern_agent ON learning_patterns(agent_id)")
            
            conn.commit()
    
    def learn_from_experience(self, action_type: str, input_data: Dict[str, Any],
                             output_data: Dict[str, Any], success: bool,
                             performance_metrics: Dict[str, Any] = None,
                             feedback: Dict[str, Any] = None) -> str:
        """Learn from an experience"""
        with self.lock:
            experience_id = f"exp_{self.agent_id}_{action_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Store experience
            experience = LearningExperience(
                id=experience_id,
                agent_id=self.agent_id,
                action_type=action_type,
                input_data=input_data,
                output_data=output_data,
                success=success,
                performance_metrics=performance_metrics or {},
                feedback=feedback or {},
                created_at=datetime.now()
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO learning_experiences 
                    (id, agent_id, action_type, input_data, output_data, success, performance_metrics, feedback, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experience.id,
                    experience.agent_id,
                    experience.action_type,
                    json.dumps(experience.input_data),
                    json.dumps(experience.output_data),
                    experience.success,
                    json.dumps(experience.performance_metrics),
                    json.dumps(experience.feedback),
                    experience.created_at.isoformat()
                ))
                conn.commit()
            
            # Update learning components
            self._update_learning_components(experience)
            
            logger.info(f"Learned from experience {experience_id}")
            return experience_id
    
    def _update_learning_components(self, experience: LearningExperience):
        """Update all learning components with new experience"""
        # Update reinforcement learner
        reward = 1.0 if experience.success else -1.0
        state = f"{experience.action_type}_{hash(str(experience.input_data)) % 1000}"
        next_state = f"{experience.action_type}_{hash(str(experience.output_data)) % 1000}"
        
        self.reinforcement_learner.update_q_value(state, experience.action_type, reward, next_state)
        
        # Update supervised learner
        self.supervised_learner.learn_pattern(
            experience.action_type,
            experience.input_data,
            experience.output_data,
            experience.success
        )
        
        # Update meta learner
        performance = {
            'success_rate': 1.0 if experience.success else 0.0,
            'action_type': experience.action_type,
            'timestamp': experience.created_at
        }
        self.meta_learner.optimize_learning_strategy(performance)
    
    def get_learning_recommendations(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get learning-based recommendations"""
        with self.lock:
            recommendations = []
            
            # Get similar patterns
            similar_patterns = self.supervised_learner.find_similar_patterns(
                current_context, 'general', threshold=0.6
            )
            
            for pattern in similar_patterns:
                if pattern['success']:
                    recommendations.append({
                        'type': 'pattern_based',
                        'confidence': pattern['similarity'],
                        'action': pattern['output'],
                        'reasoning': f"Similar successful pattern found (similarity: {pattern['similarity']:.2f})"
                    })
            
            # Get meta-learning insights
            insights = self.meta_learner.get_learning_insights()
            if insights.get('recommendations'):
                recommendations.append({
                    'type': 'meta_learning',
                    'confidence': 0.8,
                    'action': {'strategy_adjustments': insights['recommendations']},
                    'reasoning': f"Meta-learning insights: {insights['performance_trend']} performance"
                })
            
            return recommendations
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning framework statistics"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                # Experience stats
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM learning_experiences WHERE agent_id = ?
                """, (self.agent_id,))
                total_experiences = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM learning_experiences 
                    WHERE agent_id = ? AND success = 1
                """, (self.agent_id,))
                successful_experiences = cursor.fetchone()[0]
                
                success_rate = successful_experiences / total_experiences if total_experiences > 0 else 0.0
                
                # Pattern stats
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM learning_patterns WHERE agent_id = ?
                """, (self.agent_id,))
                total_patterns = cursor.fetchone()[0]
                
                return {
                    'total_experiences': total_experiences,
                    'successful_experiences': successful_experiences,
                    'success_rate': success_rate,
                    'total_patterns': total_patterns,
                    'learning_efficiency': self.meta_learner.get_learning_insights()['learning_efficiency'],
                    'current_performance': self.meta_learner.get_learning_insights()['current_performance']
                }
    
    def save_learning_state(self, filepath: str):
        """Save learning state to file"""
        with self.lock:
            state = {
                'reinforcement_learner': self.reinforcement_learner.get_policy(),
                'supervised_learner': self.supervised_learner.patterns,
                'meta_learner': self.meta_learner.performance_history
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
    
    def load_learning_state(self, filepath: str):
        """Load learning state from file"""
        with self.lock:
            try:
                with open(filepath, 'rb') as f:
                    state = pickle.load(f)
                
                # Restore reinforcement learner
                self.reinforcement_learner.q_table = defaultdict(lambda: defaultdict(float), 
                                                              state.get('reinforcement_learner', {}))
                
                # Restore supervised learner
                self.supervised_learner.patterns = state.get('supervised_learner', {})
                
                # Restore meta learner
                self.meta_learner.performance_history = state.get('meta_learner', [])
                
                logger.info(f"Loaded learning state for agent {self.agent_id}")
            except FileNotFoundError:
                logger.info(f"No existing learning state found for agent {self.agent_id}")