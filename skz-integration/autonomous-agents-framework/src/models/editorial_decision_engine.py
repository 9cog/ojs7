"""
Editorial Decision Support System for Editorial Orchestration Agent
ML-powered editorial decision recommendations and workflow optimization
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REVISE = "revise" 
    DESK_REJECT = "desk_reject"
    TRANSFER = "transfer"
    REVIEW = "review"

class UrgencyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ManuscriptMetrics:
    """Manuscript evaluation metrics"""
    manuscript_id: str
    technical_quality_score: float
    novelty_score: float
    significance_score: float
    clarity_score: float
    completeness_score: float
    ethical_compliance_score: float
    statistical_rigor_score: float
    literature_coverage_score: float
    methodology_score: float
    reproducibility_score: float
    overall_score: float

@dataclass
class ReviewData:
    """Review assessment data"""
    reviewer_id: str
    recommendation: str
    confidence: float
    review_quality: float
    technical_comments: List[str]
    major_issues: List[str]
    minor_issues: List[str]
    review_completeness: float
    review_timeliness: float

@dataclass
class EditorialContext:
    """Editorial decision context"""
    journal_standards: Dict[str, float]
    acceptance_rate: float
    current_workload: int
    special_issue: bool
    deadline_pressure: UrgencyLevel
    editor_expertise_match: float
    journal_scope_alignment: float

@dataclass
class DecisionRecommendation:
    """Editorial decision recommendation"""
    manuscript_id: str
    recommended_decision: DecisionType
    confidence: float
    reasoning: List[str]
    alternative_decisions: List[Tuple[DecisionType, float]]
    required_actions: List[str]
    estimated_timeline: int
    risk_factors: List[str]
    priority_score: float
    timestamp: str

class EditorialDecisionEngine:
    """Advanced editorial decision support system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Decision thresholds
        self.quality_thresholds = {
            DecisionType.ACCEPT: 8.5,
            DecisionType.REVISE: 6.0,
            DecisionType.REJECT: 4.0,
            DecisionType.DESK_REJECT: 2.0
        }
        
        # Weighting factors
        self.metric_weights = {
            'technical_quality': 0.25,
            'novelty': 0.20,
            'significance': 0.20,
            'clarity': 0.10,
            'methodology': 0.15,
            'reproducibility': 0.10
        }
        
    async def generate_decision_recommendation(self, 
                                            manuscript_metrics: ManuscriptMetrics,
                                            reviews: List[ReviewData],
                                            editorial_context: EditorialContext) -> DecisionRecommendation:
        """Generate comprehensive editorial decision recommendation"""
        
        try:
            # Calculate base decision scores
            quality_score = await self._calculate_quality_score(manuscript_metrics)
            review_consensus = await self._analyze_review_consensus(reviews)
            context_adjustment = await self._calculate_context_adjustment(editorial_context)
            
            # Generate primary recommendation
            primary_decision = await self._determine_primary_decision(
                quality_score, review_consensus, context_adjustment
            )
            
            # Calculate confidence
            confidence = await self._calculate_decision_confidence(
                manuscript_metrics, reviews, editorial_context, primary_decision
            )
            
            # Generate reasoning
            reasoning = await self._generate_decision_reasoning(
                manuscript_metrics, reviews, editorial_context, primary_decision
            )
            
            # Identify alternative decisions
            alternatives = await self._generate_alternative_decisions(
                quality_score, review_consensus, context_adjustment
            )
            
            # Determine required actions
            required_actions = await self._determine_required_actions(
                primary_decision, manuscript_metrics, reviews
            )
            
            # Calculate timeline and priority
            timeline = await self._estimate_decision_timeline(primary_decision, editorial_context)
            priority = await self._calculate_priority_score(manuscript_metrics, editorial_context)
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(
                manuscript_metrics, reviews, editorial_context, primary_decision
            )
            
            return DecisionRecommendation(
                manuscript_id=manuscript_metrics.manuscript_id,
                recommended_decision=primary_decision,
                confidence=confidence,
                reasoning=reasoning,
                alternative_decisions=alternatives,
                required_actions=required_actions,
                estimated_timeline=timeline,
                risk_factors=risk_factors,
                priority_score=priority,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error generating decision recommendation: {e}")
            return DecisionRecommendation(
                manuscript_id=manuscript_metrics.manuscript_id,
                recommended_decision=DecisionType.REVIEW,
                confidence=0.0,
                reasoning=["Error in decision analysis"],
                alternative_decisions=[],
                required_actions=["Manual review required"],
                estimated_timeline=14,
                risk_factors=["Decision system error"],
                priority_score=0.5,
                timestamp=datetime.now().isoformat()
            )
    
    async def _calculate_quality_score(self, metrics: ManuscriptMetrics) -> float:
        """Calculate weighted quality score"""
        
        # Weight individual metrics
        weighted_score = (
            metrics.technical_quality_score * self.metric_weights['technical_quality'] +
            metrics.novelty_score * self.metric_weights['novelty'] +
            metrics.significance_score * self.metric_weights['significance'] +
            metrics.clarity_score * self.metric_weights['clarity'] +
            metrics.methodology_score * self.metric_weights['methodology'] +
            metrics.reproducibility_score * self.metric_weights['reproducibility']
        )
        
        # Apply completeness and ethics modifiers
        completeness_modifier = metrics.completeness_score / 10.0
        ethics_modifier = metrics.ethical_compliance_score / 10.0
        
        final_score = weighted_score * completeness_modifier * ethics_modifier
        
        return min(10.0, max(0.0, final_score))
    
    async def _analyze_review_consensus(self, reviews: List[ReviewData]) -> Dict[str, float]:
        """Analyze reviewer consensus"""
        
        if not reviews:
            return {'consensus_strength': 0.0, 'average_recommendation': 5.0}
        
        # Map recommendations to numeric scores
        rec_mapping = {
            'accept': 9.0, 'minor_revisions': 7.0, 'major_revisions': 5.0,
            'reject': 3.0, 'desk_reject': 1.0
        }
        
        scores = [rec_mapping.get(review.recommendation.lower(), 5.0) for review in reviews]
        
        # Calculate consensus metrics
        avg_score = sum(scores) / len(scores)
        variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
        consensus_strength = max(0.0, 1.0 - (variance / 16.0))  # Normalize by max variance
        
        # Weight by reviewer confidence
        weighted_scores = [score * review.confidence for score, review in zip(scores, reviews)]
        weighted_avg = sum(weighted_scores) / sum(review.confidence for review in reviews)
        
        return {
            'consensus_strength': consensus_strength,
            'average_recommendation': avg_score,
            'weighted_average': weighted_avg,
            'reviewer_count': len(reviews)
        }
    
    async def _calculate_context_adjustment(self, context: EditorialContext) -> float:
        """Calculate contextual adjustment factor"""
        
        adjustment = 1.0
        
        # Journal standards adjustment
        if context.journal_standards.get('quality_bar', 7.0) > 8.0:
            adjustment *= 0.95  # Higher standards
        
        # Acceptance rate pressure
        if context.acceptance_rate < 0.15:  # Very selective journal
            adjustment *= 0.90
        elif context.acceptance_rate > 0.40:  # Less selective
            adjustment *= 1.05
        
        # Workload pressure
        if context.current_workload > 50:
            adjustment *= 0.98  # Slight bias toward efficiency
        
        # Special issue consideration
        if context.special_issue:
            adjustment *= 1.02  # Slight preference for inclusion
        
        # Deadline pressure
        urgency_multipliers = {
            UrgencyLevel.LOW: 1.0,
            UrgencyLevel.MEDIUM: 0.99,
            UrgencyLevel.HIGH: 0.97,
            UrgencyLevel.CRITICAL: 0.95
        }
        adjustment *= urgency_multipliers.get(context.deadline_pressure, 1.0)
        
        # Editor expertise match
        if context.editor_expertise_match > 0.8:
            adjustment *= 1.02
        elif context.editor_expertise_match < 0.5:
            adjustment *= 0.98
        
        return adjustment
    
    async def _determine_primary_decision(self, quality_score: float, 
                                        review_consensus: Dict[str, float],
                                        context_adjustment: float) -> DecisionType:
        """Determine primary editorial decision"""
        
        # Adjust quality score with context
        adjusted_score = quality_score * context_adjustment
        
        # Factor in reviewer consensus
        consensus_weight = review_consensus['consensus_strength']
        final_score = (
            adjusted_score * (1 - consensus_weight) +
            review_consensus['weighted_average'] * consensus_weight
        )
        
        # Apply decision thresholds
        if final_score >= self.quality_thresholds[DecisionType.ACCEPT]:
            return DecisionType.ACCEPT
        elif final_score >= self.quality_thresholds[DecisionType.REVISE]:
            return DecisionType.REVISE
        elif final_score >= self.quality_thresholds[DecisionType.REJECT]:
            return DecisionType.REJECT
        else:
            return DecisionType.DESK_REJECT
    
    async def _calculate_decision_confidence(self, 
                                           metrics: ManuscriptMetrics,
                                           reviews: List[ReviewData],
                                           context: EditorialContext,
                                           decision: DecisionType) -> float:
        """Calculate confidence in decision"""
        
        base_confidence = 0.7
        
        # Review quality factor
        if reviews:
            avg_review_quality = sum(r.review_quality for r in reviews) / len(reviews)
            review_factor = avg_review_quality / 10.0
            base_confidence += (review_factor - 0.5) * 0.2
        
        # Consensus factor
        if len(reviews) >= 2:
            recommendations = [r.recommendation for r in reviews]
            if len(set(recommendations)) == 1:  # Perfect consensus
                base_confidence += 0.15
            elif len(set(recommendations)) == len(recommendations):  # No consensus
                base_confidence -= 0.10
        
        # Editor expertise factor
        expertise_factor = context.editor_expertise_match
        base_confidence += (expertise_factor - 0.5) * 0.1
        
        # Manuscript completeness factor
        completeness_factor = metrics.completeness_score / 10.0
        base_confidence += (completeness_factor - 0.8) * 0.1
        
        return min(1.0, max(0.3, base_confidence))
    
    async def _generate_decision_reasoning(self, 
                                         metrics: ManuscriptMetrics,
                                         reviews: List[ReviewData],
                                         context: EditorialContext,
                                         decision: DecisionType) -> List[str]:
        """Generate reasoning for decision"""
        
        reasoning = []
        
        # Quality-based reasoning
        if metrics.overall_score >= 8.5:
            reasoning.append("High overall manuscript quality score")
        elif metrics.overall_score < 5.0:
            reasoning.append("Below-threshold manuscript quality")
        
        # Specific metric reasoning
        if metrics.novelty_score >= 8.0:
            reasoning.append("Strong novelty and innovation")
        if metrics.significance_score >= 8.0:
            reasoning.append("High scientific significance")
        if metrics.technical_quality_score < 6.0:
            reasoning.append("Technical quality concerns")
        if metrics.methodology_score < 6.0:
            reasoning.append("Methodological limitations")
        
        # Review-based reasoning
        if reviews:
            if all(r.recommendation in ['accept', 'minor_revisions'] for r in reviews):
                reasoning.append("Strong reviewer consensus for acceptance/minor revisions")
            elif all(r.recommendation in ['reject', 'major_revisions'] for r in reviews):
                reasoning.append("Reviewer consensus indicates significant concerns")
            
            major_issues = [issue for r in reviews for issue in r.major_issues]
            if len(major_issues) > 3:
                reasoning.append("Multiple major issues identified by reviewers")
        
        # Context-based reasoning
        if context.acceptance_rate < 0.20:
            reasoning.append("High selectivity standards applied")
        if context.special_issue:
            reasoning.append("Special issue considerations")
        if context.deadline_pressure in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]:
            reasoning.append("Timeline constraints considered")
        
        return reasoning[:5]  # Limit to top 5 reasons
    
    async def _generate_alternative_decisions(self, 
                                            quality_score: float,
                                            review_consensus: Dict[str, float],
                                            context_adjustment: float) -> List[Tuple[DecisionType, float]]:
        """Generate alternative decision options with probabilities"""
        
        alternatives = []
        base_score = quality_score * context_adjustment
        
        # Calculate probabilities for each decision
        for decision_type, threshold in self.quality_thresholds.items():
            distance = abs(base_score - threshold)
            probability = max(0.1, 1.0 - (distance / 5.0))
            alternatives.append((decision_type, probability))
        
        # Sort by probability and return top alternatives
        alternatives.sort(key=lambda x: x[1], reverse=True)
        return alternatives[:3]
    
    async def _determine_required_actions(self, 
                                        decision: DecisionType,
                                        metrics: ManuscriptMetrics,
                                        reviews: List[ReviewData]) -> List[str]:
        """Determine required actions for decision"""
        
        actions = []
        
        if decision == DecisionType.ACCEPT:
            actions.extend([
                "Prepare acceptance letter",
                "Schedule for production",
                "Notify authors of acceptance"
            ])
        elif decision == DecisionType.REVISE:
            actions.extend([
                "Compile reviewer comments",
                "Prepare revision request letter",
                "Set revision deadline"
            ])
            if metrics.statistical_rigor_score < 6.0:
                actions.append("Request statistical review")
        elif decision == DecisionType.REJECT:
            actions.extend([
                "Prepare detailed rejection letter",
                "Provide constructive feedback",
                "Suggest alternative venues if appropriate"
            ])
        elif decision == DecisionType.DESK_REJECT:
            actions.extend([
                "Quick turnaround rejection",
                "Brief feedback on fundamental issues"
            ])
        
        return actions
    
    async def _estimate_decision_timeline(self, decision: DecisionType, context: EditorialContext) -> int:
        """Estimate decision implementation timeline in days"""
        
        base_timelines = {
            DecisionType.ACCEPT: 3,
            DecisionType.REVISE: 7,
            DecisionType.REJECT: 5,
            DecisionType.DESK_REJECT: 1
        }
        
        base_time = base_timelines.get(decision, 5)
        
        # Adjust for workload
        if context.current_workload > 30:
            base_time += 2
        
        # Adjust for urgency
        urgency_adjustments = {
            UrgencyLevel.CRITICAL: -2,
            UrgencyLevel.HIGH: -1,
            UrgencyLevel.MEDIUM: 0,
            UrgencyLevel.LOW: 1
        }
        
        adjustment = urgency_adjustments.get(context.deadline_pressure, 0)
        
        return max(1, base_time + adjustment)
    
    async def _calculate_priority_score(self, metrics: ManuscriptMetrics, context: EditorialContext) -> float:
        """Calculate priority score for decision processing"""
        
        priority = 0.5  # Base priority
        
        # High quality manuscripts get higher priority
        if metrics.overall_score >= 8.0:
            priority += 0.3
        
        # Urgent context increases priority
        urgency_bonuses = {
            UrgencyLevel.CRITICAL: 0.4,
            UrgencyLevel.HIGH: 0.3,
            UrgencyLevel.MEDIUM: 0.1,
            UrgencyLevel.LOW: 0.0
        }
        priority += urgency_bonuses.get(context.deadline_pressure, 0.0)
        
        # Special issues get priority boost
        if context.special_issue:
            priority += 0.1
        
        return min(1.0, priority)
    
    async def _identify_risk_factors(self, 
                                   metrics: ManuscriptMetrics,
                                   reviews: List[ReviewData],
                                   context: EditorialContext,
                                   decision: DecisionType) -> List[str]:
        """Identify potential risk factors"""
        
        risks = []
        
        # Quality risks
        if metrics.ethical_compliance_score < 8.0:
            risks.append("Potential ethical compliance issues")
        if metrics.reproducibility_score < 6.0:
            risks.append("Reproducibility concerns")
        if metrics.statistical_rigor_score < 6.0:
            risks.append("Statistical methodology risks")
        
        # Review risks
        if reviews and len(reviews) < 2:
            risks.append("Limited peer review coverage")
        
        conflicting_reviews = len(set(r.recommendation for r in reviews)) > 2 if reviews else False
        if conflicting_reviews:
            risks.append("Conflicting reviewer recommendations")
        
        # Context risks
        if context.editor_expertise_match < 0.6:
            risks.append("Limited editor expertise alignment")
        if context.acceptance_rate < 0.15 and decision == DecisionType.ACCEPT:
            risks.append("Decision may affect journal selectivity")
        
        return risks[:3]  # Top 3 risks


# Utility functions
async def quick_decision_analysis(manuscript_data: Dict, reviews_data: List[Dict]) -> Dict:
    """Quick editorial decision analysis"""
    
    engine = EditorialDecisionEngine({})
    metrics = ManuscriptMetrics(**manuscript_data)
    reviews = [ReviewData(**r) for r in reviews_data]
    
    # Default context
    context = EditorialContext(
        journal_standards={'quality_bar': 7.0},
        acceptance_rate=0.25,
        current_workload=20,
        special_issue=False,
        deadline_pressure=UrgencyLevel.MEDIUM,
        editor_expertise_match=0.8,
        journal_scope_alignment=0.9
    )
    
    recommendation = await engine.generate_decision_recommendation(metrics, reviews, context)
    return asdict(recommendation)
