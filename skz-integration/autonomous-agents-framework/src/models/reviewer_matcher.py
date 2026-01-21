"""
Review Coordination Agent - Enhanced with Critical ML Features
Implements Reviewer Matching ML, Review Quality Prediction, and Workload Optimization
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from collections import defaultdict

from .enhanced_agent import EnhancedAgent
from .memory_system import PersistentMemorySystem
from .ml_decision_engine import DecisionEngine, DecisionContext

logger = logging.getLogger(__name__)

@dataclass
class ReviewerProfile:
    """Comprehensive reviewer profile structure"""
    reviewer_id: int
    name: str
    email: str
    expertise_areas: List[str]
    keywords: List[str]
    current_workload: int
    max_workload: int
    avg_review_time: int  # days
    quality_score: float  # 0-5 scale
    reliability_score: float  # 0-1 scale
    availability_status: str  # 'available', 'busy', 'unavailable'
    last_assignment_date: Optional[str]
    preferred_manuscript_types: List[str]
    language_preferences: List[str]
    timezone: str
    response_rate: float  # 0-1 scale
    past_collaborations: List[int]  # manuscript IDs
    conflict_of_interest: List[str]  # author names/institutions

@dataclass
class ManuscriptProfile:
    """Manuscript profile for matching"""
    manuscript_id: int
    title: str
    abstract: str
    keywords: List[str]
    subject_areas: List[str]
    authors: List[str]
    author_institutions: List[str]
    manuscript_type: str  # 'research', 'review', 'case_study', etc.
    urgency_level: str  # 'low', 'medium', 'high', 'critical'
    required_expertise: List[str]
    submission_date: str
    target_review_date: str
    language: str
    special_requirements: List[str]

@dataclass
class MatchingResult:
    """Result of reviewer-manuscript matching"""
    manuscript_id: int
    reviewer_id: int
    match_score: float
    confidence: float
    reasoning: Dict[str, float]
    potential_issues: List[str]
    estimated_review_time: int
    priority_score: float
    match_timestamp: str

@dataclass
class ReviewQualityPrediction:
    """Prediction of review quality"""
    reviewer_id: int
    manuscript_id: int
    predicted_quality: float
    predicted_depth: float
    predicted_timeliness: float
    confidence: float
    risk_factors: List[str]

@dataclass
class WorkloadOptimization:
    """Workload optimization results"""
    reviewer_assignments: Dict[int, List[int]]
    load_balance_score: float
    efficiency_improvement: float
    bottleneck_resolution: List[str]
    timeline_prediction: Dict[int, int]

class ReviewerMatcher:
    """Critical Feature 1: Reviewer Matching ML"""
    
    def __init__(self):
        self.expertise_analyzer = self._initialize_expertise_analyzer()
        self.workload_optimizer = self._initialize_workload_optimizer()
        self.quality_predictor = self._initialize_quality_predictor()
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.match_history = []
        
    def _initialize_expertise_analyzer(self) -> Dict[str, Any]:
        """Initialize expertise analysis system"""
        return {
            'weights': {
                'direct_match': 0.4,
                'related_match': 0.2,
                'keyword_overlap': 0.2,
                'past_experience': 0.2
            },
            'similarity_threshold': 0.6
        }
    
    def _initialize_workload_optimizer(self) -> Dict[str, Any]:
        """Initialize workload optimization system"""
        return {
            'max_concurrent_reviews': 3,
            'preferred_review_time': 21,
            'workload_balance_weight': 0.4,
            'efficiency_weight': 0.6
        }
    
    def _initialize_quality_predictor(self) -> Dict[str, Any]:
        """Initialize quality prediction system"""
        return {
            'quality_factors': ['past_performance', 'expertise_match', 'workload_status'],
            'prediction_model': None,  # Would be trained model in production
            'confidence_threshold': 0.7
        }
    
    def match_reviewers(self, manuscript: ManuscriptProfile, 
                       available_reviewers: List[ReviewerProfile], 
                       num_reviewers: int = 2) -> List[MatchingResult]:
        """Match reviewers to manuscript using ML algorithms"""
        try:
            # Calculate expertise similarity
            expertise_scores = self._calculate_expertise_similarity(manuscript, available_reviewers)
            
            # Calculate workload compatibility
            workload_scores = self._calculate_workload_compatibility(available_reviewers)
            
            # Predict review quality
            quality_predictions = self._predict_review_quality(manuscript, available_reviewers)
            
            # Check for conflicts of interest
            conflict_checks = self._check_conflicts_of_interest(manuscript, available_reviewers)
            
            # Calculate final matching scores
            matching_results = []
            
            for reviewer in available_reviewers:
                if conflict_checks.get(reviewer.reviewer_id, False):
                    continue  # Skip if conflict of interest
                
                expertise_score = expertise_scores.get(reviewer.reviewer_id, 0.0)
                workload_score = workload_scores.get(reviewer.reviewer_id, 0.0)
                quality_prediction = quality_predictions.get(reviewer.reviewer_id, 0.5)
                
                # Combined matching score
                match_score = (
                    expertise_score * 0.4 +
                    workload_score * 0.3 +
                    quality_prediction * 0.3
                )
                
                # Calculate confidence
                confidence = min(
                    expertise_score + 0.2,
                    workload_score + 0.1,
                    quality_prediction + 0.1
                ) / 3.0
                
                # Generate reasoning
                reasoning = {
                    'expertise_match': expertise_score,
                    'workload_compatibility': workload_score,
                    'quality_prediction': quality_prediction
                }
                
                # Identify potential issues
                potential_issues = self._identify_potential_issues(reviewer, manuscript)
                
                # Estimate review time
                estimated_time = self._estimate_review_time(reviewer, manuscript)
                
                # Calculate priority score
                priority_score = self._calculate_priority_score(manuscript, reviewer)
                
                result = MatchingResult(
                    manuscript_id=manuscript.manuscript_id,
                    reviewer_id=reviewer.reviewer_id,
                    match_score=match_score,
                    confidence=confidence,
                    reasoning=reasoning,
                    potential_issues=potential_issues,
                    estimated_review_time=estimated_time,
                    priority_score=priority_score,
                    match_timestamp=datetime.now().isoformat()
                )
                
                matching_results.append(result)
            
            # Sort by match score and return top matches
            matching_results.sort(key=lambda x: x.match_score, reverse=True)
            return matching_results[:num_reviewers]
            
        except Exception as e:
            logger.error(f"Error in reviewer matching: {e}")
            return []
    
    def _calculate_expertise_similarity(self, manuscript: ManuscriptProfile, 
                                      reviewers: List[ReviewerProfile]) -> Dict[int, float]:
        """Calculate expertise similarity using TF-IDF vectorization"""
        try:
            # Prepare documents for vectorization
            manuscript_text = f"{manuscript.title} {manuscript.abstract} {' '.join(manuscript.keywords)}"
            reviewer_texts = []
            reviewer_ids = []
            
            for reviewer in reviewers:
                reviewer_text = f"{' '.join(reviewer.expertise_areas)} {' '.join(reviewer.keywords)}"
                reviewer_texts.append(reviewer_text)
                reviewer_ids.append(reviewer.reviewer_id)
            
            if not reviewer_texts:
                return {}
            
            # Create TF-IDF vectors
            all_texts = [manuscript_text] + reviewer_texts
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity
            manuscript_vector = tfidf_matrix[0:1]
            reviewer_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(manuscript_vector, reviewer_vectors)[0]
            
            # Create similarity scores dictionary
            similarity_scores = {}
            for i, reviewer_id in enumerate(reviewer_ids):
                similarity_scores[reviewer_id] = similarities[i]
            
            return similarity_scores
            
        except Exception as e:
            logger.error(f"Error calculating expertise similarity: {e}")
            return {}
    
    def _calculate_workload_compatibility(self, reviewers: List[ReviewerProfile]) -> Dict[int, float]:
        """Calculate workload compatibility scores"""
        workload_scores = {}
        
        for reviewer in reviewers:
            # Calculate workload factor
            workload_ratio = reviewer.current_workload / reviewer.max_workload
            workload_factor = max(0, 1.0 - workload_ratio)
            
            # Factor in availability status
            availability_factor = {
                'available': 1.0,
                'busy': 0.5,
                'unavailable': 0.0
            }.get(reviewer.availability_status, 0.5)
            
            # Factor in response rate
            response_factor = reviewer.response_rate
            
            # Combined workload score
            workload_score = (
                workload_factor * 0.5 +
                availability_factor * 0.3 +
                response_factor * 0.2
            )
            
            workload_scores[reviewer.reviewer_id] = workload_score
        
        return workload_scores
    
    def _predict_review_quality(self, manuscript: ManuscriptProfile, 
                              reviewers: List[ReviewerProfile]) -> Dict[int, float]:
        """Predict review quality for each reviewer"""
        quality_predictions = {}
        
        for reviewer in reviewers:
            # Base quality from reviewer's historical performance
            base_quality = reviewer.quality_score / 5.0  # Normalize to 0-1
            
            # Factor in expertise match (higher expertise = better quality)
            manuscript_areas = set(manuscript.subject_areas + manuscript.keywords)
            reviewer_areas = set(reviewer.expertise_areas + reviewer.keywords)
            expertise_overlap = len(manuscript_areas & reviewer_areas) / max(len(manuscript_areas), 1)
            
            # Factor in workload (higher workload = potentially lower quality)
            workload_factor = max(0, 1.0 - (reviewer.current_workload / reviewer.max_workload))
            
            # Factor in reviewer reliability
            reliability_factor = reviewer.reliability_score
            
            # Combined quality prediction
            predicted_quality = (
                base_quality * 0.4 +
                expertise_overlap * 0.3 +
                workload_factor * 0.2 +
                reliability_factor * 0.1
            )
            
            quality_predictions[reviewer.reviewer_id] = min(predicted_quality, 1.0)
        
        return quality_predictions
    
    def _check_conflicts_of_interest(self, manuscript: ManuscriptProfile, 
                                   reviewers: List[ReviewerProfile]) -> Dict[int, bool]:
        """Check for conflicts of interest"""
        conflicts = {}
        
        manuscript_authors = set([author.lower() for author in manuscript.authors])
        manuscript_institutions = set([inst.lower() for inst in manuscript.author_institutions])
        
        for reviewer in reviewers:
            has_conflict = False
            
            # Check name conflicts
            reviewer_name = reviewer.name.lower()
            if reviewer_name in manuscript_authors:
                has_conflict = True
            
            # Check institution conflicts (would need reviewer institution data)
            # For now, check if reviewer is listed in COI
            for coi in reviewer.conflict_of_interest:
                if coi.lower() in manuscript_authors or any(coi.lower() in inst for inst in manuscript_institutions):
                    has_conflict = True
                    break
            
            conflicts[reviewer.reviewer_id] = has_conflict
        
        return conflicts
    
    def _identify_potential_issues(self, reviewer: ReviewerProfile, 
                                 manuscript: ManuscriptProfile) -> List[str]:
        """Identify potential issues with the reviewer assignment"""
        issues = []
        
        # Workload issues
        if reviewer.current_workload >= reviewer.max_workload:
            issues.append("Reviewer at maximum workload capacity")
        
        # Availability issues
        if reviewer.availability_status == 'busy':
            issues.append("Reviewer currently marked as busy")
        elif reviewer.availability_status == 'unavailable':
            issues.append("Reviewer currently unavailable")
        
        # Response rate issues
        if reviewer.response_rate < 0.7:
            issues.append("Reviewer has low response rate")
        
        # Expertise match issues
        manuscript_areas = set(manuscript.subject_areas)
        reviewer_areas = set(reviewer.expertise_areas)
        if not manuscript_areas & reviewer_areas:
            issues.append("Limited expertise overlap")
        
        # Language compatibility
        if manuscript.language not in reviewer.language_preferences:
            issues.append("Potential language compatibility issue")
        
        return issues
    
    def _estimate_review_time(self, reviewer: ReviewerProfile, 
                            manuscript: ManuscriptProfile) -> int:
        """Estimate review completion time in days"""
        # Base time from reviewer's average
        base_time = reviewer.avg_review_time
        
        # Adjust for manuscript urgency
        urgency_multipliers = {
            'critical': 0.5,
            'high': 0.7,
            'medium': 1.0,
            'low': 1.3
        }
        urgency_multiplier = urgency_multipliers.get(manuscript.urgency_level, 1.0)
        
        # Adjust for workload
        workload_multiplier = 1.0 + (reviewer.current_workload / reviewer.max_workload) * 0.5
        
        # Calculate estimated time
        estimated_time = int(base_time * urgency_multiplier * workload_multiplier)
        
        return max(estimated_time, 7)  # Minimum 7 days
    
    def _calculate_priority_score(self, manuscript: ManuscriptProfile, 
                                reviewer: ReviewerProfile) -> float:
        """Calculate priority score for this assignment"""
        # Manuscript urgency
        urgency_scores = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        urgency_score = urgency_scores.get(manuscript.urgency_level, 0.6)
        
        # Reviewer quality
        quality_score = reviewer.quality_score / 5.0
        
        # Deadline pressure
        try:
            target_date = datetime.fromisoformat(manuscript.target_review_date)
            days_until_deadline = (target_date - datetime.now()).days
            deadline_pressure = max(0, (30 - days_until_deadline) / 30)  # Higher pressure as deadline approaches
        except:
            deadline_pressure = 0.5
        
        # Combined priority score
        priority_score = (
            urgency_score * 0.4 +
            quality_score * 0.3 +
            deadline_pressure * 0.3
        )
        
        return min(priority_score, 1.0)

class ReviewQualityPredictor:
    """Critical Feature 2: Review Quality Prediction"""
    
    def __init__(self):
        self.reviewer_profiler = {}
        self.manuscript_analyzer = {}
        self.interaction_predictor = {}
        self.prediction_model = None
        
    def predict_review_quality(self, reviewer: ReviewerProfile, 
                             manuscript: ManuscriptProfile) -> ReviewQualityPrediction:
        """Predict the quality and characteristics of a review"""
        try:
            # Analyze reviewer profile
            reviewer_analysis = self._analyze_reviewer_profile(reviewer)
            
            # Analyze manuscript complexity
            manuscript_analysis = self._analyze_manuscript_complexity(manuscript)
            
            # Predict interaction quality
            interaction_quality = self._predict_interaction_quality(reviewer_analysis, manuscript_analysis)
            
            # Predict specific quality dimensions
            predicted_quality = self._predict_overall_quality(reviewer, manuscript, interaction_quality)
            predicted_depth = self._predict_review_depth(reviewer, manuscript)
            predicted_timeliness = self._predict_timeliness(reviewer, manuscript)
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(reviewer, manuscript)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(reviewer, manuscript)
            
            return ReviewQualityPrediction(
                reviewer_id=reviewer.reviewer_id,
                manuscript_id=manuscript.manuscript_id,
                predicted_quality=predicted_quality,
                predicted_depth=predicted_depth,
                predicted_timeliness=predicted_timeliness,
                confidence=confidence,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"Error predicting review quality: {e}")
            return ReviewQualityPrediction(
                reviewer.reviewer_id, manuscript.manuscript_id, 0.5, 0.5, 0.5, 0.0, []
            )
    
    def _analyze_reviewer_profile(self, reviewer: ReviewerProfile) -> Dict[str, float]:
        """Analyze reviewer profile for quality prediction"""
        return {
            'experience_level': min(len(reviewer.past_collaborations) / 50.0, 1.0),
            'expertise_breadth': min(len(reviewer.expertise_areas) / 10.0, 1.0),
            'historical_quality': reviewer.quality_score / 5.0,
            'reliability': reviewer.reliability_score,
            'response_rate': reviewer.response_rate,
            'workload_pressure': reviewer.current_workload / reviewer.max_workload
        }
    
    def _analyze_manuscript_complexity(self, manuscript: ManuscriptProfile) -> Dict[str, float]:
        """Analyze manuscript complexity"""
        abstract_length = len(manuscript.abstract.split())
        keyword_count = len(manuscript.keywords)
        
        return {
            'content_complexity': min(abstract_length / 300.0, 1.0),  # Longer abstracts = more complex
            'topic_specificity': min(keyword_count / 10.0, 1.0),
            'interdisciplinary_scope': min(len(manuscript.subject_areas) / 5.0, 1.0),
            'urgency_pressure': {
                'critical': 1.0, 'high': 0.8, 'medium': 0.5, 'low': 0.2
            }.get(manuscript.urgency_level, 0.5)
        }
    
    def _predict_interaction_quality(self, reviewer_analysis: Dict[str, float], 
                                   manuscript_analysis: Dict[str, float]) -> float:
        """Predict quality of reviewer-manuscript interaction"""
        # Match between reviewer experience and manuscript complexity
        experience_complexity_match = 1.0 - abs(
            reviewer_analysis['experience_level'] - manuscript_analysis['content_complexity']
        )
        
        # Expertise breadth vs interdisciplinary scope
        breadth_scope_match = min(
            reviewer_analysis['expertise_breadth'],
            manuscript_analysis['interdisciplinary_scope']
        )
        
        # Workload vs urgency compatibility
        workload_urgency_compat = 1.0 - (
            reviewer_analysis['workload_pressure'] * manuscript_analysis['urgency_pressure']
        )
        
        return (experience_complexity_match + breadth_scope_match + workload_urgency_compat) / 3.0
    
    def _predict_overall_quality(self, reviewer: ReviewerProfile, 
                               manuscript: ManuscriptProfile, 
                               interaction_quality: float) -> float:
        """Predict overall review quality"""
        # Base quality from reviewer's track record
        base_quality = reviewer.quality_score / 5.0
        
        # Adjust for expertise match
        manuscript_areas = set(manuscript.subject_areas + manuscript.keywords)
        reviewer_areas = set(reviewer.expertise_areas + reviewer.keywords)
        expertise_match = len(manuscript_areas & reviewer_areas) / max(len(manuscript_areas), 1)
        
        # Combine factors
        predicted_quality = (
            base_quality * 0.5 +
            expertise_match * 0.3 +
            interaction_quality * 0.2
        )
        
        return min(predicted_quality, 1.0)
    
    def _predict_review_depth(self, reviewer: ReviewerProfile, 
                            manuscript: ManuscriptProfile) -> float:
        """Predict depth and thoroughness of review"""
        # Reviewers with more experience tend to provide deeper reviews
        experience_factor = min(len(reviewer.past_collaborations) / 30.0, 1.0)
        
        # Higher quality reviewers provide more depth
        quality_factor = reviewer.quality_score / 5.0
        
        # Less workload pressure allows for deeper review
        workload_factor = 1.0 - (reviewer.current_workload / reviewer.max_workload)
        
        predicted_depth = (
            experience_factor * 0.4 +
            quality_factor * 0.4 +
            workload_factor * 0.2
        )
        
        return min(predicted_depth, 1.0)
    
    def _predict_timeliness(self, reviewer: ReviewerProfile, 
                          manuscript: ManuscriptProfile) -> float:
        """Predict review timeliness"""
        # Base timeliness from response rate
        base_timeliness = reviewer.response_rate
        
        # Adjust for workload
        workload_factor = 1.0 - (reviewer.current_workload / reviewer.max_workload)
        
        # Adjust for urgency
        urgency_boost = {
            'critical': 0.3, 'high': 0.2, 'medium': 0.1, 'low': 0.0
        }.get(manuscript.urgency_level, 0.0)
        
        predicted_timeliness = min(
            base_timeliness * 0.6 + workload_factor * 0.4 + urgency_boost,
            1.0
        )
        
        return predicted_timeliness
    
    def _calculate_prediction_confidence(self, reviewer: ReviewerProfile, 
                                       manuscript: ManuscriptProfile) -> float:
        """Calculate confidence in the prediction"""
        # More past data = higher confidence
        data_confidence = min(len(reviewer.past_collaborations) / 20.0, 1.0)
        
        # Clear expertise match = higher confidence
        manuscript_areas = set(manuscript.subject_areas)
        reviewer_areas = set(reviewer.expertise_areas)
        match_confidence = len(manuscript_areas & reviewer_areas) / max(len(manuscript_areas), 1)
        
        # Stable reviewer metrics = higher confidence
        stability_confidence = reviewer.reliability_score
        
        confidence = (data_confidence + match_confidence + stability_confidence) / 3.0
        
        return min(confidence, 1.0)
    
    def _identify_risk_factors(self, reviewer: ReviewerProfile, 
                             manuscript: ManuscriptProfile) -> List[str]:
        """Identify factors that could negatively impact review quality"""
        risk_factors = []
        
        # High workload
        if reviewer.current_workload >= reviewer.max_workload * 0.8:
            risk_factors.append("High reviewer workload may impact quality")
        
        # Low response rate
        if reviewer.response_rate < 0.7:
            risk_factors.append("Reviewer has history of delayed responses")
        
        # Limited expertise overlap
        manuscript_areas = set(manuscript.subject_areas)
        reviewer_areas = set(reviewer.expertise_areas)
        if len(manuscript_areas & reviewer_areas) == 0:
            risk_factors.append("Limited expertise overlap may affect review depth")
        
        # Quality concerns
        if reviewer.quality_score < 3.0:
            risk_factors.append("Reviewer has below-average quality ratings")
        
        # Urgency vs availability mismatch
        if manuscript.urgency_level in ['critical', 'high'] and reviewer.availability_status == 'busy':
            risk_factors.append("Urgent manuscript assigned to busy reviewer")
        
        return risk_factors

class WorkloadOptimizer:
    """Critical Feature 3: Workload Optimization"""
    
    def __init__(self):
        self.capacity_analyzer = {}
        self.assignment_optimizer = {}
        self.timeline_predictor = {}
        
    def optimize_workload(self, manuscripts: List[ManuscriptProfile], 
                         reviewers: List[ReviewerProfile]) -> WorkloadOptimization:
        """Optimize reviewer workload distribution"""
        try:
            # Analyze current capacity
            capacity_analysis = self._analyze_capacity(reviewers)
            
            # Optimize assignments
            optimized_assignments = self._optimize_assignments(manuscripts, reviewers, capacity_analysis)
            
            # Calculate load balance
            load_balance_score = self._calculate_load_balance(optimized_assignments, reviewers)
            
            # Calculate efficiency improvement
            efficiency_improvement = self._calculate_efficiency_improvement(
                optimized_assignments, manuscripts, reviewers
            )
            
            # Identify bottleneck resolutions
            bottleneck_resolution = self._identify_bottleneck_resolutions(
                optimized_assignments, capacity_analysis
            )
            
            # Predict timelines
            timeline_prediction = self._predict_timelines(optimized_assignments, reviewers)
            
            return WorkloadOptimization(
                reviewer_assignments=optimized_assignments,
                load_balance_score=load_balance_score,
                efficiency_improvement=efficiency_improvement,
                bottleneck_resolution=bottleneck_resolution,
                timeline_prediction=timeline_prediction
            )
            
        except Exception as e:
            logger.error(f"Error optimizing workload: {e}")
            return WorkloadOptimization({}, 0.0, 0.0, [], {})
    
    def _analyze_capacity(self, reviewers: List[ReviewerProfile]) -> Dict[str, Any]:
        """Analyze reviewer capacity"""
        total_capacity = sum(r.max_workload for r in reviewers)
        current_load = sum(r.current_workload for r in reviewers)
        available_capacity = total_capacity - current_load
        
        # Identify over/under-utilized reviewers
        overloaded = [r for r in reviewers if r.current_workload >= r.max_workload]
        underutilized = [r for r in reviewers if r.current_workload < r.max_workload * 0.5]
        
        return {
            'total_capacity': total_capacity,
            'current_load': current_load,
            'available_capacity': available_capacity,
            'utilization_rate': current_load / total_capacity if total_capacity > 0 else 0,
            'overloaded_reviewers': [r.reviewer_id for r in overloaded],
            'underutilized_reviewers': [r.reviewer_id for r in underutilized]
        }
    
    def _optimize_assignments(self, manuscripts: List[ManuscriptProfile], 
                            reviewers: List[ReviewerProfile],
                            capacity_analysis: Dict[str, Any]) -> Dict[int, List[int]]:
        """Optimize manuscript assignments to reviewers"""
        assignments = defaultdict(list)
        
        # Sort manuscripts by priority (urgency, deadline)
        sorted_manuscripts = sorted(
            manuscripts,
            key=lambda m: (
                {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(m.urgency_level, 1),
                -self._days_until_deadline(m.target_review_date)
            ),
            reverse=True
        )
        
        # Sort reviewers by availability and quality
        sorted_reviewers = sorted(
            reviewers,
            key=lambda r: (
                -(r.current_workload / r.max_workload),  # Prefer less loaded
                r.quality_score,  # Prefer higher quality
                r.response_rate   # Prefer reliable reviewers
            ),
            reverse=True
        )
        
        # Assign manuscripts using greedy algorithm
        for manuscript in sorted_manuscripts:
            best_reviewers = self._find_best_reviewers_for_manuscript(
                manuscript, sorted_reviewers, assignments
            )
            
            for reviewer_id in best_reviewers[:2]:  # Assign 2 reviewers per manuscript
                assignments[reviewer_id].append(manuscript.manuscript_id)
        
        return dict(assignments)
    
    def _find_best_reviewers_for_manuscript(self, manuscript: ManuscriptProfile,
                                          available_reviewers: List[ReviewerProfile],
                                          current_assignments: Dict[int, List[int]]) -> List[int]:
        """Find best reviewers for a specific manuscript"""
        scored_reviewers = []
        
        for reviewer in available_reviewers:
            # Check if reviewer has capacity
            current_load = len(current_assignments.get(reviewer.reviewer_id, []))
            if current_load >= reviewer.max_workload:
                continue
            
            # Calculate suitability score
            expertise_score = self._calculate_expertise_score(manuscript, reviewer)
            workload_score = 1.0 - (current_load / reviewer.max_workload)
            quality_score = reviewer.quality_score / 5.0
            
            total_score = (
                expertise_score * 0.4 +
                workload_score * 0.3 +
                quality_score * 0.3
            )
            
            scored_reviewers.append((reviewer.reviewer_id, total_score))
        
        # Sort by score and return top reviewers
        scored_reviewers.sort(key=lambda x: x[1], reverse=True)
        return [reviewer_id for reviewer_id, score in scored_reviewers]
    
    def _calculate_expertise_score(self, manuscript: ManuscriptProfile, 
                                 reviewer: ReviewerProfile) -> float:
        """Calculate expertise match score"""
        manuscript_areas = set(manuscript.subject_areas + manuscript.keywords)
        reviewer_areas = set(reviewer.expertise_areas + reviewer.keywords)
        
        if not manuscript_areas:
            return 0.5
        
        overlap = len(manuscript_areas & reviewer_areas)
        return overlap / len(manuscript_areas)
    
    def _calculate_load_balance(self, assignments: Dict[int, List[int]], 
                              reviewers: List[ReviewerProfile]) -> float:
        """Calculate load balance score (0-1, higher is better)"""
        if not assignments:
            return 1.0
        
        loads = []
        for reviewer in reviewers:
            current_load = len(assignments.get(reviewer.reviewer_id, []))
            load_ratio = current_load / reviewer.max_workload
            loads.append(load_ratio)
        
        # Calculate standard deviation of load ratios
        if len(loads) <= 1:
            return 1.0
        
        mean_load = sum(loads) / len(loads)
        variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
        std_dev = math.sqrt(variance)
        
        # Lower standard deviation = better balance
        balance_score = max(0, 1.0 - std_dev)
        return balance_score
    
    def _calculate_efficiency_improvement(self, assignments: Dict[int, List[int]],
                                        manuscripts: List[ManuscriptProfile],
                                        reviewers: List[ReviewerProfile]) -> float:
        """Calculate efficiency improvement from optimization"""
        # Baseline: random assignment efficiency
        total_manuscripts = len(manuscripts)
        total_capacity = sum(r.max_workload for r in reviewers)
        baseline_efficiency = min(total_manuscripts / total_capacity, 1.0) if total_capacity > 0 else 0
        
        # Optimized efficiency
        assigned_manuscripts = sum(len(ms_list) for ms_list in assignments.values())
        optimized_efficiency = assigned_manuscripts / total_manuscripts if total_manuscripts > 0 else 0
        
        # Calculate improvement
        improvement = max(0, optimized_efficiency - baseline_efficiency)
        return improvement
    
    def _identify_bottleneck_resolutions(self, assignments: Dict[int, List[int]],
                                       capacity_analysis: Dict[str, Any]) -> List[str]:
        """Identify how bottlenecks were resolved"""
        resolutions = []
        
        # Check if overloaded reviewers were helped
        overloaded = capacity_analysis.get('overloaded_reviewers', [])
        for reviewer_id in overloaded:
            new_load = len(assignments.get(reviewer_id, []))
            resolutions.append(f"Reviewer {reviewer_id} workload optimized to {new_load} manuscripts")
        
        # Check if underutilized reviewers were utilized
        underutilized = capacity_analysis.get('underutilized_reviewers', [])
        for reviewer_id in underutilized:
            new_load = len(assignments.get(reviewer_id, []))
            if new_load > 0:
                resolutions.append(f"Underutilized reviewer {reviewer_id} assigned {new_load} manuscripts")
        
        return resolutions
    
    def _predict_timelines(self, assignments: Dict[int, List[int]],
                         reviewers: List[ReviewerProfile]) -> Dict[int, int]:
        """Predict completion timelines for each reviewer"""
        timelines = {}
        
        for reviewer in reviewers:
            reviewer_assignments = assignments.get(reviewer.reviewer_id, [])
            if not reviewer_assignments:
                timelines[reviewer.reviewer_id] = 0
                continue
            
            # Estimate time based on number of assignments and reviewer's average time
            num_assignments = len(reviewer_assignments)
            avg_time = reviewer.avg_review_time
            
            # Account for parallel processing (can work on multiple reviews)
            max_parallel = min(3, num_assignments)  # Assume max 3 parallel reviews
            batches = math.ceil(num_assignments / max_parallel)
            
            estimated_days = batches * avg_time
            timelines[reviewer.reviewer_id] = estimated_days
        
        return timelines
    
    def _days_until_deadline(self, target_date: str) -> int:
        """Calculate days until target deadline"""
        try:
            deadline = datetime.fromisoformat(target_date)
            return (deadline - datetime.now()).days
        except:
            return 30  # Default to 30 days if parsing fails

class ReviewCoordinationAgent(EnhancedAgent):
    """
    Enhanced Review Coordination Agent with Critical ML Features
    Implements Reviewer Matching ML, Review Quality Prediction, and Workload Optimization
    """
    
    def __init__(self, agent_id: str = "review_coordination_001"):
        super().__init__(
            agent_id=agent_id,
            agent_type="review_coordination",
            capabilities=["reviewer_matching", "quality_prediction", "workload_optimization", "review_coordination"]
        )
        
        # Critical Feature 1: Reviewer Matching ML
        self.reviewer_matcher = ReviewerMatcher()
        
        # Critical Feature 2: Review Quality Prediction
        self.quality_predictor = ReviewQualityPredictor()
        
        # Critical Feature 3: Workload Optimization
        self.workload_optimizer = WorkloadOptimizer()
        
        logger.info("Review Coordination Agent initialized with critical ML features")
    
    def process_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process review coordination actions using enhanced ML capabilities"""
        try:
            action_type = action_data.get('action_type', 'match')
            
            if action_type == 'match_reviewers':
                return self._autonomous_reviewer_matching(action_data)
            elif action_type == 'predict_quality':
                return self._autonomous_quality_prediction(action_data)
            elif action_type == 'optimize_workload':
                return self._autonomous_workload_optimization(action_data)
            elif action_type == 'coordinate_reviews':
                return self._autonomous_review_coordination(action_data)
            else:
                return {"error": f"Unknown action type: {action_type}"}
                
        except Exception as e:
            logger.error(f"Error processing review coordination action: {e}")
            return {"error": str(e)}
    
    def _autonomous_reviewer_matching(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous reviewer matching using ML"""
        try:
            manuscript_data = action_data.get('manuscript', {})
            available_reviewers_data = action_data.get('available_reviewers', [])
            num_reviewers = action_data.get('num_reviewers', 2)
            
            # Convert to proper objects
            manuscript = ManuscriptProfile(**manuscript_data)
            reviewers = [ReviewerProfile(**r) for r in available_reviewers_data]
            
            # Perform matching
            matching_results = self.reviewer_matcher.match_reviewers(manuscript, reviewers, num_reviewers)
            
            # Store in memory
            self.memory_system.store_memory(
                agent_id=self.agent_id,
                memory_type='matching',
                content={'matching_results': [result.__dict__ for result in matching_results]},
                importance_score=0.8,
                tags=['reviewer_matching', str(manuscript.manuscript_id)]
            )
            
            return {
                'manuscript_id': manuscript.manuscript_id,
                'matching_results': [result.__dict__ for result in matching_results],
                'matching_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in autonomous reviewer matching: {e}")
            return {"error": str(e)}
    
    def _autonomous_quality_prediction(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous review quality prediction using ML"""
        try:
            reviewer_data = action_data.get('reviewer', {})
            manuscript_data = action_data.get('manuscript', {})
            
            # Convert to proper objects
            reviewer = ReviewerProfile(**reviewer_data)
            manuscript = ManuscriptProfile(**manuscript_data)
            
            # Predict quality
            quality_prediction = self.quality_predictor.predict_review_quality(reviewer, manuscript)
            
            # Store in memory
            self.memory_system.store_memory(
                agent_id=self.agent_id,
                memory_type='prediction',
                content={'quality_prediction': quality_prediction.__dict__},
                importance_score=0.7,
                tags=['quality_prediction', str(reviewer.reviewer_id), str(manuscript.manuscript_id)]
            )
            
            return {
                'quality_prediction': quality_prediction.__dict__,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in autonomous quality prediction: {e}")
            return {"error": str(e)}
    
    def _autonomous_workload_optimization(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous workload optimization using ML"""
        try:
            manuscripts_data = action_data.get('manuscripts', [])
            reviewers_data = action_data.get('reviewers', [])
            
            # Convert to proper objects
            manuscripts = [ManuscriptProfile(**m) for m in manuscripts_data]
            reviewers = [ReviewerProfile(**r) for r in reviewers_data]
            
            # Optimize workload
            optimization_result = self.workload_optimizer.optimize_workload(manuscripts, reviewers)
            
            # Store in memory
            self.memory_system.store_memory(
                agent_id=self.agent_id,
                memory_type='optimization',
                content={'workload_optimization': optimization_result.__dict__},
                importance_score=0.9,
                tags=['workload_optimization', 'load_balancing']
            )
            
            return {
                'workload_optimization': optimization_result.__dict__,
                'optimization_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in autonomous workload optimization: {e}")
            return {"error": str(e)}
    
    def _autonomous_review_coordination(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive autonomous review coordination"""
        try:
            coordination_data = action_data.get('coordination_data', {})
            
            # Extract data
            manuscripts_data = coordination_data.get('manuscripts', [])
            reviewers_data = coordination_data.get('reviewers', [])
            
            # Convert to objects
            manuscripts = [ManuscriptProfile(**m) for m in manuscripts_data]
            reviewers = [ReviewerProfile(**r) for r in reviewers_data]
            
            # Perform comprehensive coordination
            coordination_results = []
            
            # 1. Optimize overall workload
            workload_optimization = self.workload_optimizer.optimize_workload(manuscripts, reviewers)
            
            # 2. Match reviewers for each manuscript
            for manuscript in manuscripts:
                matching_results = self.reviewer_matcher.match_reviewers(manuscript, reviewers, 2)
                
                # 3. Predict quality for each match
                quality_predictions = []
                for match in matching_results:
                    reviewer = next(r for r in reviewers if r.reviewer_id == match.reviewer_id)
                    quality_pred = self.quality_predictor.predict_review_quality(reviewer, manuscript)
                    quality_predictions.append(quality_pred.__dict__)
                
                coordination_results.append({
                    'manuscript_id': manuscript.manuscript_id,
                    'reviewer_matches': [m.__dict__ for m in matching_results],
                    'quality_predictions': quality_predictions
                })
            
            comprehensive_result = {
                'workload_optimization': workload_optimization.__dict__,
                'manuscript_coordination': coordination_results,
                'total_manuscripts': len(manuscripts),
                'total_reviewers': len(reviewers),
                'coordination_score': self._calculate_coordination_score(workload_optimization, coordination_results),
                'coordination_timestamp': datetime.now().isoformat()
            }
            
            # Store comprehensive result
            self.memory_system.store_memory(
                agent_id=self.agent_id,
                memory_type='coordination',
                content={'comprehensive_coordination': comprehensive_result},
                importance_score=1.0,
                tags=['comprehensive_coordination', 'review_management']
            )
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Error in autonomous review coordination: {e}")
            return {"error": str(e)}
    
    def _calculate_coordination_score(self, workload_opt: WorkloadOptimization, 
                                    coordination_results: List[Dict[str, Any]]) -> float:
        """Calculate overall coordination effectiveness score"""
        try:
            # Workload balance component
            workload_score = workload_opt.load_balance_score
            
            # Matching quality component
            total_matches = 0
            total_match_score = 0
            
            for result in coordination_results:
                matches = result.get('reviewer_matches', [])
                for match in matches:
                    total_matches += 1
                    total_match_score += match.get('match_score', 0.5)
            
            avg_match_score = total_match_score / total_matches if total_matches > 0 else 0.5
            
            # Quality prediction confidence component
            total_predictions = 0
            total_confidence = 0
            
            for result in coordination_results:
                predictions = result.get('quality_predictions', [])
                for pred in predictions:
                    total_predictions += 1
                    total_confidence += pred.get('confidence', 0.5)
            
            avg_confidence = total_confidence / total_predictions if total_predictions > 0 else 0.5
            
            # Combined coordination score
            coordination_score = (
                workload_score * 0.4 +
                avg_match_score * 0.4 +
                avg_confidence * 0.2
            )
            
            return min(coordination_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating coordination score: {e}")
            return 0.5
            'review_quality': 0.4,
            'reliability': 0.3,
            'response_rate': 0.3
        }
        
        # Subject area hierarchies for cosmetic science
        self.subject_hierarchies = {
            'cosmetic_chemistry': [
                'formulation_science', 'ingredient_chemistry', 'stability_testing',
                'preservation', 'emulsification', 'rheology'
            ],
            'dermatology': [
                'skin_biology', 'irritation_testing', 'sensitization',
                'dermatitis', 'skin_barrier', 'wound_healing'
            ],
            'toxicology': [
                'safety_assessment', 'risk_assessment', 'QSAR',
                'in_vitro_testing', 'alternative_methods'
            ],
            'regulatory_science': [
                'regulatory_affairs', 'product_registration', 'labeling',
                'claims_substantiation', 'global_regulations'
            ],
            'consumer_science': [
                'sensory_evaluation', 'consumer_testing', 'market_research',
                'product_development', 'user_experience'
            ]
        }
        
    async def match_reviewers(self, manuscript: ManuscriptProfile, available_reviewers: List[ReviewerProfile], num_matches: int = 3) -> List[MatchingResult]:
        """
        Find optimal reviewer matches for a manuscript using ML or basic approach
        """
        logger.info(f"Matching reviewers for manuscript {manuscript.manuscript_id}")
        
        # Check if production ML models and semantic analysis are available
        if hasattr(self, 'sentence_transformer') and self.sentence_transformer is not None:
            return await self._match_reviewers_ml(manuscript, available_reviewers, num_matches)
        else:
            return await self._match_reviewers_basic(manuscript, available_reviewers, num_matches)
    
    async def _match_reviewers_ml(self, manuscript: ManuscriptProfile, available_reviewers: List[ReviewerProfile], num_matches: int = 3) -> List[MatchingResult]:
        """
        Production ML-based reviewer matching system
        """
        try:
            # PRODUCTION ENFORCEMENT: Require ML models in production
            if os.getenv('ENVIRONMENT', '').lower() == 'production':
                if not self.config.get('ml_models_available', False):
                    raise ValueError(
                        "PRODUCTION VIOLATION: ML models required for production reviewer matching. "
                        "Configure sentence transformers and expertise models. NEVER SACRIFICE QUALITY!!"
                    )
            
            # Import required ML libraries
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                from sklearn.metrics.pairwise import cosine_similarity
                from sklearn.feature_extraction.text import TfidfVectorizer
            except ImportError as e:
                if os.getenv('ENVIRONMENT', '').lower() == 'production':
                    raise ValueError(f"Required ML libraries not available in production: {e}")
                logger.warning(f"ML libraries not available, falling back to basic: {e}")
                return await self._match_reviewers_basic(manuscript, available_reviewers, num_matches)
            
            # Initialize ML models
            if not hasattr(self, 'sentence_transformer'):
                await self._initialize_ml_models()
            
            # 1. Semantic Similarity Analysis
            manuscript_text = f"{manuscript.title} {manuscript.abstract} {' '.join(manuscript.keywords)}"
            manuscript_embedding = self.sentence_transformer.encode([manuscript_text])
            
            # Calculate reviewer embeddings
            reviewer_embeddings = []
            for reviewer in available_reviewers:
                reviewer_text = f"{' '.join(reviewer.expertise_areas)} {' '.join(reviewer.keywords)} {reviewer.research_summary}"
                reviewer_embeddings.append(self.sentence_transformer.encode([reviewer_text]))
            
            # Calculate semantic similarity scores
            semantic_scores = []
            for i, reviewer_emb in enumerate(reviewer_embeddings):
                similarity = cosine_similarity(manuscript_embedding, reviewer_emb)[0][0]
                semantic_scores.append((available_reviewers[i], float(similarity)))
            
            # 2. Expertise Classification Matching
            expertise_scores = await self._calculate_expertise_match_scores(manuscript, available_reviewers)
            
            # 3. Historical Performance Analysis
            performance_scores = await self._calculate_performance_scores(available_reviewers)
            
            # 4. Multi-dimensional scoring with ML
            final_matches = []
            for reviewer, semantic_score in semantic_scores:
                expertise_score = expertise_scores.get(reviewer.reviewer_id, 0.0)
                performance_score = performance_scores.get(reviewer.reviewer_id, 0.5)
                
                # Weighted combination of scores
                final_score = (
                    0.4 * semantic_score +           # Semantic similarity
                    0.3 * expertise_score +          # Expertise match
                    0.2 * performance_score +        # Historical performance
                    0.1 * self._calculate_availability_score(reviewer)  # Availability
                )
                
                # Check conflicts of interest
                if not self._has_conflict_of_interest(manuscript, reviewer):
                    match_result = MatchingResult(
                        reviewer_id=reviewer.reviewer_id,
                        manuscript_id=manuscript.manuscript_id,
                        match_score=min(final_score, 1.0),
                        confidence_level='high',
                        reasoning=f"ML-based match: semantic={semantic_score:.3f}, expertise={expertise_score:.3f}, performance={performance_score:.3f}",
                        expertise_alignment=expertise_score,
                        availability_score=self._calculate_availability_score(reviewer),
                        predicted_quality=performance_score,
                        estimated_completion_time=self._estimate_completion_time(reviewer),
                        matching_timestamp=datetime.now().isoformat()
                    )
                    final_matches.append(match_result)
            
            # 5. Global assignment optimization
            final_matches.sort(key=lambda x: x.match_score, reverse=True)
            
            # Return top matches
            return final_matches[:num_matches]
            
        except Exception as e:
            logger.error(f"ML reviewer matching error: {e}")
            if os.getenv('ENVIRONMENT', '').lower() == 'production':
                raise ValueError(f"Production ML matching failed: {e}")
            return await self._match_reviewers_basic(manuscript, available_reviewers, num_matches)
    
    async def _initialize_ml_models(self):
        """Initialize ML models for reviewer matching"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a model optimized for academic/scientific text
            model_name = self.config.get('sentence_transformer_model', 'all-MiniLM-L6-v2')
            self.sentence_transformer = SentenceTransformer(model_name)
            
            logger.info(f"Initialized sentence transformer model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            if os.getenv('ENVIRONMENT', '').lower() == 'production':
                raise ValueError(f"Failed to initialize required ML models in production: {e}")
    
    async def _calculate_expertise_match_scores(self, manuscript: ManuscriptProfile, reviewers: List[ReviewerProfile]) -> Dict[str, float]:
        """Calculate expertise matching scores using ML classification"""
        scores = {}
        
        try:
            # Use TF-IDF for expertise domain matching
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create corpus
            manuscript_domains = ' '.join(manuscript.keywords + [manuscript.research_domain])
            reviewer_domains = []
            for reviewer in reviewers:
                reviewer_domain = ' '.join(reviewer.expertise_areas + reviewer.keywords)
                reviewer_domains.append(reviewer_domain)
            
            # Calculate TF-IDF similarity
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            corpus = [manuscript_domains] + reviewer_domains
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            manuscript_vector = tfidf_matrix[0:1]
            reviewer_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(manuscript_vector, reviewer_vectors)[0]
            
            for i, reviewer in enumerate(reviewers):
                scores[reviewer.reviewer_id] = float(similarities[i])
                
        except Exception as e:
            logger.error(f"Expertise matching error: {e}")
            # Fallback to basic keyword matching
            for reviewer in reviewers:
                scores[reviewer.reviewer_id] = self._calculate_basic_expertise_score(manuscript, reviewer)
        
        return scores
    
    async def _calculate_performance_scores(self, reviewers: List[ReviewerProfile]) -> Dict[str, float]:
        """Calculate historical performance scores"""
        scores = {}
        
        for reviewer in reviewers:
            # Calculate performance based on historical metrics
            completion_rate = reviewer.performance_metrics.get('completion_rate', 0.8)
            avg_quality = reviewer.performance_metrics.get('average_quality_score', 0.7)
            timeliness = reviewer.performance_metrics.get('timeliness_score', 0.6)
            
            # Weighted performance score
            performance_score = (
                0.4 * completion_rate +
                0.4 * avg_quality +
                0.2 * timeliness
            )
            
            scores[reviewer.reviewer_id] = performance_score
        
        return scores
    
    async def _match_reviewers_basic(self, manuscript: ManuscriptProfile, available_reviewers: List[ReviewerProfile], num_matches: int = 3) -> List[MatchingResult]:
        """
        Basic reviewer matching (FALLBACK - REPLACE WITH ML)
        """
        try:
            # Calculate match scores for all reviewers
            match_scores = []
            
            for reviewer in available_reviewers:
                if await self._check_eligibility(manuscript, reviewer):
                    match_score = await self._calculate_match_score(manuscript, reviewer)
                    match_scores.append(match_score)
                else:
                    logger.debug(f"Reviewer {reviewer.reviewer_id} not eligible for manuscript {manuscript.manuscript_id}")
            
            # Sort by match score and return top matches
            match_scores.sort(key=lambda x: x.match_score, reverse=True)
            
            return match_scores[:num_matches]
            
        except Exception as e:
            logger.error(f"Error in reviewer matching: {e}")
            return []
    
    async def _calculate_match_score(self, manuscript: ManuscriptProfile, reviewer: ReviewerProfile) -> MatchingResult:
        """Calculate comprehensive match score between manuscript and reviewer"""
        
        # Calculate individual component scores
        expertise_score = await self._calculate_expertise_match(manuscript, reviewer)
        workload_score = await self._calculate_workload_suitability(reviewer)
        quality_score = await self._calculate_quality_score(reviewer)
        availability_score = await self._calculate_availability_score(manuscript, reviewer)
        
        # Calculate reasoning breakdown
        reasoning = {
            'expertise_match': expertise_score,
            'workload_suitability': workload_score,
            'reviewer_quality': quality_score,
            'availability': availability_score,
            'urgency_alignment': await self._calculate_urgency_alignment(manuscript, reviewer),
            'past_performance': await self._calculate_past_performance(reviewer, manuscript.subject_areas)
        }
        
        # Weight and combine scores
        weighted_score = (
            expertise_score * 0.35 +
            workload_score * 0.25 +
            quality_score * 0.20 +
            availability_score * 0.20
        )
        
        # Calculate confidence based on data completeness
        confidence = self._calculate_confidence(reviewer, reasoning)
        
        # Identify potential issues
        potential_issues = await self._identify_potential_issues(manuscript, reviewer)
        
        # Estimate review time
        estimated_time = await self._estimate_review_time(manuscript, reviewer)
        
        # Calculate priority score
        priority_score = await self._calculate_priority_score(manuscript, weighted_score)
        
        return MatchingResult(
            manuscript_id=manuscript.manuscript_id,
            reviewer_id=reviewer.reviewer_id,
            match_score=weighted_score,
            confidence=confidence,
            reasoning=reasoning,
            potential_issues=potential_issues,
            estimated_review_time=estimated_time,
            priority_score=priority_score,
            match_timestamp=datetime.now().isoformat()
        )
    
    async def _calculate_expertise_match(self, manuscript: ManuscriptProfile, reviewer: ReviewerProfile) -> float:
        """Calculate expertise matching score"""
        
        score = 0.0
        
        # Direct expertise area matches
        direct_matches = len(set(manuscript.subject_areas) & set(reviewer.expertise_areas))
        if reviewer.expertise_areas:
            direct_score = direct_matches / len(reviewer.expertise_areas)
            score += direct_score * self.expertise_weights['direct_match']
        
        # Keyword overlap
        keyword_overlap = len(set(manuscript.keywords) & set(reviewer.keywords))
        if reviewer.keywords:
            keyword_score = keyword_overlap / len(reviewer.keywords)
            score += keyword_score * self.expertise_weights['keyword_overlap']
        
        # Hierarchical matching (related areas)
        related_score = await self._calculate_related_expertise(manuscript.subject_areas, reviewer.expertise_areas)
        score += related_score * self.expertise_weights['related_match']
        
        # Preferred manuscript types
        type_match = 1.0 if manuscript.manuscript_type in reviewer.preferred_manuscript_types else 0.5
        score += type_match * 0.1
        
        return min(1.0, score)
    
    async def _calculate_related_expertise(self, manuscript_areas: List[str], reviewer_areas: List[str]) -> float:
        """Calculate related expertise score using subject hierarchies"""
        
        related_score = 0.0
        total_comparisons = 0
        
        for ms_area in manuscript_areas:
            for rev_area in reviewer_areas:
                total_comparisons += 1
                
                # Check if areas are in same hierarchy
                for hierarchy, sub_areas in self.subject_hierarchies.items():
                    if ms_area in sub_areas and rev_area in sub_areas:
                        related_score += 0.8  # High related score
                    elif (ms_area == hierarchy and rev_area in sub_areas) or \
                         (rev_area == hierarchy and ms_area in sub_areas):
                        related_score += 0.6  # Medium related score
        
        return related_score / total_comparisons if total_comparisons > 0 else 0.0
    
    async def _calculate_workload_suitability(self, reviewer: ReviewerProfile) -> float:
        """Calculate workload suitability score"""
        
        # Current workload factor
        if reviewer.max_workload > 0:
            workload_ratio = reviewer.current_workload / reviewer.max_workload
            workload_score = max(0.0, 1.0 - workload_ratio)
        else:
            workload_score = 0.0
        
        # Response time factor (normalize to 0-1, assume 30 days is maximum acceptable)
        response_time_score = max(0.0, 1.0 - (reviewer.avg_review_time / 30.0))
        
        # Availability status
        availability_map = {'available': 1.0, 'busy': 0.5, 'unavailable': 0.0}
        availability_score = availability_map.get(reviewer.availability_status, 0.0)
        
        # Weight and combine
        total_score = (
            workload_score * self.workload_weights['current_load'] +
            response_time_score * self.workload_weights['response_time'] +
            availability_score * self.workload_weights['availability']
        )
        
        return total_score
    
    async def _calculate_quality_score(self, reviewer: ReviewerProfile) -> float:
        """Calculate reviewer quality score"""
        
        # Normalize quality score (assuming 0-5 scale)
        quality_normalized = reviewer.quality_score / 5.0
        
        # Combine quality factors
        total_score = (
            quality_normalized * self.quality_weights['review_quality'] +
            reviewer.reliability_score * self.quality_weights['reliability'] +
            reviewer.response_rate * self.quality_weights['response_rate']
        )
        
        return total_score
    
    async def _calculate_availability_score(self, manuscript: ManuscriptProfile, reviewer: ReviewerProfile) -> float:
        """Calculate availability alignment score"""
        
        # Parse dates
        try:
            target_date = datetime.fromisoformat(manuscript.target_review_date.replace('Z', '+00:00'))
            submission_date = datetime.fromisoformat(manuscript.submission_date.replace('Z', '+00:00'))
            
            # Calculate available time
            available_days = (target_date - datetime.now()).days
            
            # Compare with reviewer's average review time
            if available_days >= reviewer.avg_review_time:
                return 1.0
            elif available_days >= reviewer.avg_review_time * 0.8:
                return 0.8
            elif available_days >= reviewer.avg_review_time * 0.6:
                return 0.6
            else:
                return 0.3
                
        except Exception:
            return 0.5  # Default if date parsing fails
    
    async def _calculate_urgency_alignment(self, manuscript: ManuscriptProfile, reviewer: ReviewerProfile) -> float:
        """Calculate how well reviewer matches manuscript urgency"""
        
        urgency_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}
        manuscript_urgency = urgency_map.get(manuscript.urgency_level, 0.5)
        
        # Fast reviewers are better for urgent manuscripts
        reviewer_speed = max(0.0, 1.0 - (reviewer.avg_review_time / 21.0))  # 21 days as reference
        
        # For urgent manuscripts, prioritize fast reviewers
        if manuscript_urgency > 0.7:
            return reviewer_speed
        else:
            return 0.7 + 0.3 * reviewer_speed  # Speed is less critical for non-urgent
    
    async def _calculate_past_performance(self, reviewer: ReviewerProfile, subject_areas: List[str]) -> float:
        """Calculate past performance in similar areas"""
        
        # This would integrate with historical data
        # For now, use reviewer quality as proxy
        base_score = reviewer.quality_score / 5.0
        
        # Bonus for experience (number of past collaborations)
        experience_bonus = min(0.3, len(reviewer.past_collaborations) * 0.05)
        
        return min(1.0, base_score + experience_bonus)
    
    def _calculate_confidence(self, reviewer: ReviewerProfile, reasoning: Dict[str, float]) -> float:
        """Calculate confidence in the matching score"""
        
        confidence = 0.7  # Base confidence
        
        # Increase confidence with more complete data
        if reviewer.expertise_areas:
            confidence += 0.1
        if reviewer.keywords:
            confidence += 0.1
        if reviewer.past_collaborations:
            confidence += 0.1
        if reviewer.quality_score > 0:
            confidence += 0.1
        
        # Decrease confidence for potential issues
        if reviewer.current_workload >= reviewer.max_workload:
            confidence -= 0.2
        if reviewer.availability_status == 'unavailable':
            confidence -= 0.3
        
        return min(1.0, max(0.3, confidence))
    
    async def _check_eligibility(self, manuscript: ManuscriptProfile, reviewer: ReviewerProfile) -> bool:
        """Check if reviewer is eligible for this manuscript"""
        
        # Check conflicts of interest
        for author in manuscript.authors:
            if author.lower() in [coi.lower() for coi in reviewer.conflict_of_interest]:
                return False
        
        for institution in manuscript.author_institutions:
            if institution.lower() in [coi.lower() for coi in reviewer.conflict_of_interest]:
                return False
        
        # Check availability
        if reviewer.availability_status == 'unavailable':
            return False
        
        # Check workload
        if reviewer.current_workload >= reviewer.max_workload:
            return False
        
        # Check language compatibility
        if manuscript.language not in reviewer.language_preferences and reviewer.language_preferences:
            return False
        
        # Check if reviewer has reviewed this manuscript before
        if manuscript.manuscript_id in reviewer.past_collaborations:
            return False
        
        return True
    
    async def _identify_potential_issues(self, manuscript: ManuscriptProfile, reviewer: ReviewerProfile) -> List[str]:
        """Identify potential issues with this reviewer assignment"""
        
        issues = []
        
        # Workload concerns
        if reviewer.current_workload / reviewer.max_workload > 0.8:
            issues.append("High current workload may affect review timeline")
        
        # Response time concerns
        target_date = datetime.fromisoformat(manuscript.target_review_date.replace('Z', '+00:00'))
        available_days = (target_date - datetime.now()).days
        
        if available_days < reviewer.avg_review_time:
            issues.append("Tight timeline relative to reviewer's average response time")
        
        # Expertise gaps
        manuscript_areas = set(manuscript.subject_areas)
        reviewer_areas = set(reviewer.expertise_areas)
        
        if not manuscript_areas.intersection(reviewer_areas):
            issues.append("Limited direct expertise overlap")
        
        # Quality concerns
        if reviewer.quality_score < 3.0:
            issues.append("Below-average review quality history")
        
        # Reliability concerns
        if reviewer.reliability_score < 0.7:
            issues.append("Reliability concerns based on past performance")
        
        # Response rate concerns
        if reviewer.response_rate < 0.6:
            issues.append("Low response rate to review invitations")
        
        return issues
    
    async def _estimate_review_time(self, manuscript: ManuscriptProfile, reviewer: ReviewerProfile) -> int:
        """Estimate review completion time in days"""
        
        base_time = reviewer.avg_review_time
        
        # Adjust for manuscript complexity
        word_count_factor = len(manuscript.abstract.split()) / 250.0  # Assume 250 words average
        complexity_adjustment = min(1.5, word_count_factor)
        
        # Adjust for reviewer workload
        workload_factor = 1.0 + (reviewer.current_workload / reviewer.max_workload) * 0.5
        
        # Adjust for urgency
        urgency_map = {'low': 1.2, 'medium': 1.0, 'high': 0.8, 'critical': 0.6}
        urgency_factor = urgency_map.get(manuscript.urgency_level, 1.0)
        
        estimated_time = base_time * complexity_adjustment * workload_factor * urgency_factor
        
        return int(estimated_time)
    
    async def _calculate_priority_score(self, manuscript: ManuscriptProfile, match_score: float) -> float:
        """Calculate priority score for assignment optimization"""
        
        urgency_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}
        urgency_score = urgency_map.get(manuscript.urgency_level, 0.5)
        
        # Combine match quality with urgency
        priority = (match_score * 0.7) + (urgency_score * 0.3)
        
        return priority
    
    async def optimize_assignments(self, manuscripts: List[ManuscriptProfile], reviewers: List[ReviewerProfile], reviews_per_manuscript: int = 2) -> OptimizationResult:
        """
        Global optimization of reviewer assignments
        Uses greedy algorithm with look-ahead for better solutions
        """
        start_time = datetime.now()
        logger.info(f"Starting global optimization for {len(manuscripts)} manuscripts and {len(reviewers)} reviewers")
        
        try:
            assignments = []
            unassigned_manuscripts = []
            reviewer_workloads = {r.reviewer_id: r.current_workload for r in reviewers}
            
            # Calculate all possible matches
            all_matches = {}
            for manuscript in manuscripts:
                matches = await self.match_reviewers(manuscript, reviewers, len(reviewers))
                all_matches[manuscript.manuscript_id] = matches
            
            # Sort manuscripts by priority (urgent first)
            manuscripts_sorted = sorted(manuscripts, 
                                      key=lambda m: {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(m.urgency_level, 1),
                                      reverse=True)
            
            # Assign reviewers using greedy approach with constraints
            for manuscript in manuscripts_sorted:
                manuscript_assignments = 0
                available_matches = [
                    match for match in all_matches[manuscript.manuscript_id]
                    if reviewer_workloads[match.reviewer_id] < max(r.max_workload for r in reviewers if r.reviewer_id == match.reviewer_id)
                ]
                
                # Sort by match score and assign top available reviewers
                available_matches.sort(key=lambda x: x.match_score, reverse=True)
                
                for match in available_matches[:reviews_per_manuscript]:
                    if manuscript_assignments < reviews_per_manuscript:
                        # Check if reviewer is still available
                        reviewer = next(r for r in reviewers if r.reviewer_id == match.reviewer_id)
                        if reviewer_workloads[match.reviewer_id] < reviewer.max_workload:
                            assignments.append(match)
                            reviewer_workloads[match.reviewer_id] += 1
                            manuscript_assignments += 1
                
                # Track unassigned manuscripts
                if manuscript_assignments == 0:
                    unassigned_manuscripts.append(manuscript.manuscript_id)
            
            # Calculate optimization metrics
            overloaded_reviewers = [
                rid for rid, workload in reviewer_workloads.items()
                if workload > max(r.max_workload for r in reviewers if r.reviewer_id == rid)
            ]
            
            avg_match_score = sum(a.match_score for a in assignments) / len(assignments) if assignments else 0.0
            assignment_rate = (len(manuscripts) - len(unassigned_manuscripts)) / len(manuscripts)
            optimization_score = (avg_match_score + assignment_rate) / 2.0
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                total_manuscripts=len(manuscripts),
                total_reviewers=len(reviewers),
                assignments=assignments,
                unassigned_manuscripts=unassigned_manuscripts,
                overloaded_reviewers=overloaded_reviewers,
                optimization_score=optimization_score,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in assignment optimization: {e}")
            return OptimizationResult(
                total_manuscripts=len(manuscripts),
                total_reviewers=len(reviewers),
                assignments=[],
                unassigned_manuscripts=[m.manuscript_id for m in manuscripts],
                overloaded_reviewers=[],
                optimization_score=0.0,
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now().isoformat()
            )


# Utility functions
async def quick_reviewer_match(manuscript_data: Dict, reviewer_data: Dict) -> float:
    """Quick single reviewer-manuscript match score"""
    
    manuscript = ManuscriptProfile(**manuscript_data)
    reviewer = ReviewerProfile(**reviewer_data)
    
    matcher = ReviewerMatcher({})
    result = await matcher._calculate_match_score(manuscript, reviewer)
    
    return result.match_score

async def find_best_reviewers(manuscript_data: Dict, reviewers_data: List[Dict], top_k: int = 5) -> List[Dict]:
    """Find best reviewers for a manuscript"""
    
    manuscript = ManuscriptProfile(**manuscript_data)
    reviewers = [ReviewerProfile(**r) for r in reviewers_data]
    
    matcher = ReviewerMatcher({})
    matches = await matcher.match_reviewers(manuscript, reviewers, top_k)
    
    return [asdict(match) for match in matches]
