"""
Learning System Enhancement for SKZ Autonomous Agents
Advanced machine learning and adaptive learning capabilities
"""
import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class LearningEvent:
    """Single learning event record"""
    event_id: str
    agent_id: str
    action_type: str
    input_features: Dict[str, Any]
    output_result: Any
    feedback_score: float  # -1 to 1 scale
    context: Dict[str, Any]
    timestamp: str
    success: bool
    execution_time: float

@dataclass
class LearningPattern:
    """Discovered learning pattern"""
    pattern_id: str
    pattern_type: str
    conditions: Dict[str, Any]
    action_recommendations: List[str]
    confidence: float
    frequency: int
    success_rate: float
    last_updated: str

@dataclass
class PerformanceMetrics:
    """Agent performance metrics"""
    agent_id: str
    total_actions: int
    success_rate: float
    avg_execution_time: float
    improvement_trend: float
    learning_velocity: float
    adaptation_score: float
    timestamp: str

class LearningSystem:
    """Advanced learning system for autonomous agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_size = config.get('memory_size', 10000)
        self.pattern_threshold = config.get('pattern_threshold', 5)
        self.learning_rate = config.get('learning_rate', 0.01)
        
        # Memory stores
        self.event_memory = deque(maxlen=self.memory_size)
        self.patterns = {}
        self.agent_models = {}
        self.performance_history = defaultdict(list)
        
    async def record_learning_event(self, event: LearningEvent):
        """Record a learning event for analysis"""
        try:
            self.event_memory.append(event)
            await self._update_performance_metrics(event)
            await self._detect_patterns()
            logger.info(f"Recorded learning event {event.event_id}")
        except Exception as e:
            logger.error(f"Error recording learning event: {e}")
    
    async def get_action_recommendation(self, agent_id: str, context: Dict[str, Any], action_type: str) -> Dict[str, Any]:
        """Get ML-based action recommendations"""
        try:
            # Extract features from context
            features = self._extract_features(context)
            
            # Find matching patterns
            matching_patterns = await self._find_matching_patterns(agent_id, action_type, features)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(matching_patterns, context)
            
            # Calculate confidence
            confidence = self._calculate_recommendation_confidence(matching_patterns)
            
            return {
                'recommendations': recommendations,
                'confidence': confidence,
                'reasoning': [p.pattern_id for p in matching_patterns],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {'recommendations': [], 'confidence': 0.0, 'reasoning': [], 'timestamp': datetime.now().isoformat()}
    
    async def _detect_patterns(self):
        """Detect learning patterns from event memory"""
        try:
            # Group events by agent and action type
            grouped_events = defaultdict(list)
            
            for event in self.event_memory:
                key = f"{event.agent_id}_{event.action_type}"
                grouped_events[key].append(event)
            
            # Analyze patterns for each group
            for group_key, events in grouped_events.items():
                if len(events) >= self.pattern_threshold:
                    patterns = await self._analyze_event_patterns(events)
                    for pattern in patterns:
                        self.patterns[pattern.pattern_id] = pattern
                        
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
    
    async def _analyze_event_patterns(self, events: List[LearningEvent]) -> List[LearningPattern]:
        """Analyze patterns in event sequences"""
        patterns = []
        
        try:
            # Successful vs failed event analysis
            successful_events = [e for e in events if e.success]
            failed_events = [e for e in events if not e.success]
            
            # Pattern 1: Success conditions
            if len(successful_events) >= self.pattern_threshold:
                success_pattern = await self._extract_success_pattern(successful_events)
                if success_pattern:
                    patterns.append(success_pattern)
            
            # Pattern 2: Failure conditions
            if len(failed_events) >= self.pattern_threshold:
                failure_pattern = await self._extract_failure_pattern(failed_events)
                if failure_pattern:
                    patterns.append(failure_pattern)
            
            # Pattern 3: Temporal patterns
            temporal_patterns = await self._extract_temporal_patterns(events)
            patterns.extend(temporal_patterns)
            
        except Exception as e:
            logger.error(f"Error analyzing event patterns: {e}")
        
        return patterns
    
    async def _extract_success_pattern(self, successful_events: List[LearningEvent]) -> Optional[LearningPattern]:
        """Extract patterns from successful events"""
        try:
            # Find common features in successful events
            common_features = self._find_common_features([e.input_features for e in successful_events])
            
            if not common_features:
                return None
            
            # Calculate success metrics
            avg_score = sum(e.feedback_score for e in successful_events) / len(successful_events)
            success_rate = len([e for e in successful_events if e.feedback_score > 0]) / len(successful_events)
            
            pattern_id = self._generate_pattern_id(successful_events[0].agent_id, 'success', common_features)
            
            return LearningPattern(
                pattern_id=pattern_id,
                pattern_type='success_conditions',
                conditions=common_features,
                action_recommendations=['continue_similar_approach', 'increase_confidence'],
                confidence=min(1.0, avg_score),
                frequency=len(successful_events),
                success_rate=success_rate,
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error extracting success pattern: {e}")
            return None
    
    async def _extract_failure_pattern(self, failed_events: List[LearningEvent]) -> Optional[LearningPattern]:
        """Extract patterns from failed events"""
        try:
            # Find common features in failed events
            common_features = self._find_common_features([e.input_features for e in failed_events])
            
            if not common_features:
                return None
            
            pattern_id = self._generate_pattern_id(failed_events[0].agent_id, 'failure', common_features)
            
            return LearningPattern(
                pattern_id=pattern_id,
                pattern_type='failure_conditions',
                conditions=common_features,
                action_recommendations=['avoid_similar_approach', 'increase_validation', 'seek_human_input'],
                confidence=0.8,
                frequency=len(failed_events),
                success_rate=0.0,
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error extracting failure pattern: {e}")
            return None
    
    async def _extract_temporal_patterns(self, events: List[LearningEvent]) -> List[LearningPattern]:
        """Extract temporal patterns from events"""
        patterns = []
        
        try:
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda x: x.timestamp)
            
            # Look for improvement trends
            if len(sorted_events) >= 10:
                recent_events = sorted_events[-10:]
                older_events = sorted_events[-20:-10] if len(sorted_events) >= 20 else []
                
                if older_events:
                    recent_avg = sum(e.feedback_score for e in recent_events) / len(recent_events)
                    older_avg = sum(e.feedback_score for e in older_events) / len(older_events)
                    
                    if recent_avg > older_avg + 0.1:  # Significant improvement
                        improvement_pattern = LearningPattern(
                            pattern_id=f"{events[0].agent_id}_improvement_trend",
                            pattern_type='improvement_trend',
                            conditions={'trend': 'improving', 'improvement_rate': recent_avg - older_avg},
                            action_recommendations=['maintain_current_approach', 'increase_confidence'],
                            confidence=0.9,
                            frequency=len(recent_events),
                            success_rate=len([e for e in recent_events if e.success]) / len(recent_events),
                            last_updated=datetime.now().isoformat()
                        )
                        patterns.append(improvement_pattern)
                        
        except Exception as e:
            logger.error(f"Error extracting temporal patterns: {e}")
        
        return patterns
    
    def _find_common_features(self, feature_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common features across multiple feature sets"""
        if not feature_sets:
            return {}
        
        common = {}
        
        # Find keys that appear in most feature sets
        all_keys = set()
        for fs in feature_sets:
            all_keys.update(fs.keys())
        
        for key in all_keys:
            values = [fs.get(key) for fs in feature_sets if key in fs]
            
            if len(values) >= len(feature_sets) * 0.7:  # Appears in 70% of sets
                # For numeric values, use range
                if all(isinstance(v, (int, float)) for v in values):
                    common[key] = {'min': min(values), 'max': max(values), 'avg': sum(values) / len(values)}
                # For categorical values, use most common
                elif all(isinstance(v, str) for v in values):
                    most_common = max(set(values), key=values.count)
                    common[key] = most_common
        
        return common
    
    def _generate_pattern_id(self, agent_id: str, pattern_type: str, conditions: Dict[str, Any]) -> str:
        """Generate unique pattern ID"""
        content = f"{agent_id}_{pattern_type}_{json.dumps(conditions, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def _find_matching_patterns(self, agent_id: str, action_type: str, features: Dict[str, Any]) -> List[LearningPattern]:
        """Find patterns matching current context"""
        matching = []
        
        for pattern in self.patterns.values():
            if self._pattern_matches_context(pattern, agent_id, action_type, features):
                matching.append(pattern)
        
        return sorted(matching, key=lambda x: x.confidence, reverse=True)
    
    def _pattern_matches_context(self, pattern: LearningPattern, agent_id: str, action_type: str, features: Dict[str, Any]) -> bool:
        """Check if pattern matches current context"""
        try:
            # Check conditions match
            for key, condition in pattern.conditions.items():
                if key not in features:
                    continue
                
                feature_value = features[key]
                
                # Handle range conditions
                if isinstance(condition, dict) and 'min' in condition and 'max' in condition:
                    if not (condition['min'] <= feature_value <= condition['max']):
                        return False
                # Handle exact matches
                elif condition != feature_value:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _extract_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from context for pattern matching"""
        features = {}
        
        try:
            # Extract numeric features
            for key, value in context.items():
                if isinstance(value, (int, float)):
                    features[key] = value
                elif isinstance(value, str):
                    features[f"{key}_str"] = value
                elif isinstance(value, bool):
                    features[f"{key}_bool"] = value
                elif isinstance(value, list) and value:
                    features[f"{key}_length"] = len(value)
                    if all(isinstance(v, (int, float)) for v in value):
                        features[f"{key}_avg"] = sum(value) / len(value)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
        
        return features
    
    async def _generate_recommendations(self, patterns: List[LearningPattern], context: Dict[str, Any]) -> List[str]:
        """Generate action recommendations from matching patterns"""
        recommendations = []
        
        for pattern in patterns:
            recommendations.extend(pattern.action_recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _calculate_recommendation_confidence(self, patterns: List[LearningPattern]) -> float:
        """Calculate confidence in recommendations"""
        if not patterns:
            return 0.0
        
        # Weight by pattern confidence and frequency
        weighted_sum = sum(p.confidence * min(1.0, p.frequency / 10.0) for p in patterns)
        weight_sum = sum(min(1.0, p.frequency / 10.0) for p in patterns)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    async def _update_performance_metrics(self, event: LearningEvent):
        """Update performance metrics for agent"""
        try:
            # Add to performance history
            self.performance_history[event.agent_id].append({
                'timestamp': event.timestamp,
                'success': event.success,
                'feedback_score': event.feedback_score,
                'execution_time': event.execution_time
            })
            
            # Keep only recent history
            max_history = self.config.get('max_performance_history', 1000)
            if len(self.performance_history[event.agent_id]) > max_history:
                self.performance_history[event.agent_id] = self.performance_history[event.agent_id][-max_history:]
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def get_performance_metrics(self, agent_id: str) -> PerformanceMetrics:
        """Get current performance metrics for agent"""
        try:
            history = self.performance_history.get(agent_id, [])
            
            if not history:
                return PerformanceMetrics(
                    agent_id=agent_id,
                    total_actions=0,
                    success_rate=0.0,
                    avg_execution_time=0.0,
                    improvement_trend=0.0,
                    learning_velocity=0.0,
                    adaptation_score=0.0,
                    timestamp=datetime.now().isoformat()
                )
            
            # Calculate metrics
            total_actions = len(history)
            success_rate = sum(1 for h in history if h['success']) / total_actions
            avg_execution_time = sum(h['execution_time'] for h in history) / total_actions
            
            # Calculate improvement trend
            if total_actions >= 20:
                recent = history[-10:]
                older = history[-20:-10]
                recent_score = sum(h['feedback_score'] for h in recent) / len(recent)
                older_score = sum(h['feedback_score'] for h in older) / len(older)
                improvement_trend = recent_score - older_score
            else:
                improvement_trend = 0.0
            
            # Calculate learning velocity (improvement rate over time)
            learning_velocity = improvement_trend / max(1, total_actions / 100)
            
            # Calculate adaptation score
            adaptation_score = min(1.0, (success_rate + abs(improvement_trend)) / 2.0)
            
            return PerformanceMetrics(
                agent_id=agent_id,
                total_actions=total_actions,
                success_rate=success_rate,
                avg_execution_time=avg_execution_time,
                improvement_trend=improvement_trend,
                learning_velocity=learning_velocity,
                adaptation_score=adaptation_score,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics(
                agent_id=agent_id,
                total_actions=0,
                success_rate=0.0,
                avg_execution_time=0.0,
                improvement_trend=0.0,
                learning_velocity=0.0,
                adaptation_score=0.0,
                timestamp=datetime.now().isoformat()
            )
    
    async def export_learning_data(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Export learning data for analysis or backup"""
        try:
            data = {
                'patterns': {pid: asdict(pattern) for pid, pattern in self.patterns.items()},
                'performance_history': dict(self.performance_history),
                'export_timestamp': datetime.now().isoformat(),
                'config': self.config
            }
            
            if agent_id:
                # Filter for specific agent
                agent_patterns = {pid: asdict(pattern) for pid, pattern in self.patterns.items() if agent_id in pattern.pattern_id}
                data['patterns'] = agent_patterns
                data['performance_history'] = {agent_id: self.performance_history.get(agent_id, [])}
            
            return data
            
        except Exception as e:
            logger.error(f"Error exporting learning data: {e}")
            return {}


# Utility functions
async def create_learning_event(agent_id: str, action_type: str, input_data: Dict, output_data: Any, feedback: float) -> LearningEvent:
    """Create a learning event"""
    return LearningEvent(
        event_id=f"{agent_id}_{datetime.now().timestamp()}",
        agent_id=agent_id,
        action_type=action_type,
        input_features=input_data,
        output_result=output_data,
        feedback_score=feedback,
        context={},
        timestamp=datetime.now().isoformat(),
        success=feedback > 0,
        execution_time=0.0
    )
