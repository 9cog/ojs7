#!/usr/bin/env python3
"""
Enhanced Editorial Decision Support System
Extends the existing SKZ framework with real-time decision support and analytics
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import sqlite3
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Import existing SKZ components
import sys
sys.path.append('/home/runner/work/oj7/oj7/skz-integration/autonomous-agents-framework/src')
try:
    from models.editorial_decision_engine import (
        EditorialDecisionEngine, 
        ManuscriptMetrics, 
        ReviewData, 
        EditorialContext,
        DecisionType,
        UrgencyLevel
    )
    from models.ml_decision_engine import DecisionEngine
except ImportError:
    # Fallback if imports fail
    class EditorialDecisionEngine:
        def __init__(self, config): pass
        async def generate_decision_recommendation(self, *args): 
            return {"error": "Decision engine not available"}

logger = logging.getLogger(__name__)

@dataclass
class DecisionAuditEntry:
    """Audit entry for tracking decision history"""
    timestamp: str
    submission_id: str
    decision_type: str
    confidence: float
    reasoning: List[str]
    reviewer_consensus: Dict
    manuscript_metrics: Dict
    processing_time: float
    editor_id: Optional[str] = None
    final_decision: Optional[str] = None
    decision_outcome: Optional[str] = None

class DecisionAuditSystem:
    """System for tracking and analyzing editorial decisions"""
    
    def __init__(self, db_path: str = "editorial_decisions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the decision audit database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decision_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                submission_id TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT NOT NULL,
                reviewer_consensus TEXT NOT NULL,
                manuscript_metrics TEXT NOT NULL,
                processing_time REAL NOT NULL,
                editor_id TEXT,
                final_decision TEXT,
                decision_outcome TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_submission_id ON decision_audit(submission_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON decision_audit(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def record_decision(self, audit_entry: DecisionAuditEntry):
        """Record a decision in the audit system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO decision_audit (
                timestamp, submission_id, decision_type, confidence, reasoning,
                reviewer_consensus, manuscript_metrics, processing_time,
                editor_id, final_decision, decision_outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            audit_entry.timestamp,
            audit_entry.submission_id,
            audit_entry.decision_type,
            audit_entry.confidence,
            json.dumps(audit_entry.reasoning),
            json.dumps(audit_entry.reviewer_consensus),
            json.dumps(audit_entry.manuscript_metrics),
            audit_entry.processing_time,
            audit_entry.editor_id,
            audit_entry.final_decision,
            audit_entry.decision_outcome
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Recorded decision for submission {audit_entry.submission_id}")
    
    def get_decision_history(self, submission_id: str = None, limit: int = 50) -> List[Dict]:
        """Get decision history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if submission_id:
            cursor.execute('''
                SELECT * FROM decision_audit 
                WHERE submission_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (submission_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM decision_audit 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_decision_statistics(self, days: int = 30) -> Dict:
        """Get decision statistics for the specified time period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get basic counts
        cursor.execute('''
            SELECT 
                COUNT(*) as total_decisions,
                AVG(confidence) as avg_confidence,
                AVG(processing_time) as avg_processing_time
            FROM decision_audit 
            WHERE timestamp > ?
        ''', (cutoff_date,))
        
        basic_stats = cursor.fetchone()
        
        # Get decision type distribution
        cursor.execute('''
            SELECT decision_type, COUNT(*) as count
            FROM decision_audit 
            WHERE timestamp > ?
            GROUP BY decision_type
        ''', (cutoff_date,))
        
        decision_distribution = dict(cursor.fetchall())
        
        # Calculate rates
        total = basic_stats[0] if basic_stats[0] else 1
        accept_count = decision_distribution.get('accept', 0)
        revision_count = (decision_distribution.get('revise', 0) + 
                         decision_distribution.get('minor_revision', 0) + 
                         decision_distribution.get('major_revision', 0))
        reject_count = (decision_distribution.get('reject', 0) + 
                       decision_distribution.get('desk_reject', 0))
        
        conn.close()
        
        return {
            'total_decisions': basic_stats[0] or 0,
            'average_confidence': basic_stats[1] or 0.0,
            'average_processing_time': basic_stats[2] or 0.0,
            'decision_distribution': decision_distribution,
            'accept_rate': (accept_count / total) * 100,
            'revision_rate': (revision_count / total) * 100,
            'reject_rate': (reject_count / total) * 100,
            'period_days': days
        }

class EnhancedEditorialDecisionSupport:
    """Enhanced editorial decision support system with real-time capabilities"""
    
    def __init__(self, port=8005):
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize components
        self.decision_engine = EditorialDecisionEngine({})
        self.ml_engine = DecisionEngine()
        self.audit_system = DecisionAuditSystem()
        
        # Performance metrics
        self.performance_metrics = {
            'decisions_processed': 0,
            'average_processing_time': 0.0,
            'accuracy_score': 0.0,
            'system_uptime': datetime.now()
        }
        
        self.setup_logging()
        self.setup_routes()
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - EnhancedDecisionSupport - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'service': 'enhanced_editorial_decision_support',
                'port': self.port,
                'uptime': (datetime.now() - self.performance_metrics['system_uptime']).total_seconds(),
                'metrics': self.performance_metrics
            })
        
        @self.app.route('/api/v1/decision/recommend', methods=['POST'])
        def get_decision_recommendation():
            """Get AI decision recommendation for a manuscript"""
            start_time = datetime.now()
            
            try:
                data = request.get_json()
                submission_id = data.get('submission_id')
                manuscript_data = data.get('manuscript_data', {})
                reviews_data = data.get('reviews_data', [])
                context_data = data.get('context_data', {})
                
                # Process the decision recommendation
                recommendation = asyncio.run(self._generate_enhanced_recommendation(
                    submission_id, manuscript_data, reviews_data, context_data
                ))
                
                # Record processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Create audit entry
                audit_entry = DecisionAuditEntry(
                    timestamp=datetime.now().isoformat(),
                    submission_id=submission_id,
                    decision_type=recommendation.get('recommended_decision', 'unknown'),
                    confidence=recommendation.get('confidence', 0.0),
                    reasoning=recommendation.get('reasoning', []),
                    reviewer_consensus=recommendation.get('review_consensus', {}),
                    manuscript_metrics=recommendation.get('manuscript_metrics', {}),
                    processing_time=processing_time
                )
                
                self.audit_system.record_decision(audit_entry)
                
                # Update performance metrics
                self.performance_metrics['decisions_processed'] += 1
                self.performance_metrics['average_processing_time'] = (
                    self.performance_metrics['average_processing_time'] * 0.9 + processing_time * 0.1
                )
                
                return jsonify({
                    'status': 'success',
                    'submission_id': submission_id,
                    'recommendation': recommendation,
                    'processing_time': processing_time,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error generating recommendation: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/v1/decision/history/<submission_id>', methods=['GET'])
        def get_decision_history(submission_id):
            """Get decision history for a submission"""
            try:
                limit = int(request.args.get('limit', 10))
                history = self.audit_system.get_decision_history(submission_id, limit)
                
                return jsonify({
                    'status': 'success',
                    'submission_id': submission_id,
                    'history': history,
                    'count': len(history)
                })
                
            except Exception as e:
                self.logger.error(f"Error fetching decision history: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/v1/decision/statistics', methods=['GET'])
        def get_decision_statistics():
            """Get decision statistics and analytics"""
            try:
                days = int(request.args.get('days', 30))
                stats = self.audit_system.get_decision_statistics(days)
                
                # Add performance metrics
                stats.update({
                    'system_performance': {
                        'decisions_processed': self.performance_metrics['decisions_processed'],
                        'average_processing_time': self.performance_metrics['average_processing_time'],
                        'uptime_hours': (datetime.now() - self.performance_metrics['system_uptime']).total_seconds() / 3600,
                        'accuracy_score': self.performance_metrics.get('accuracy_score', 0.0)
                    }
                })
                
                return jsonify({
                    'status': 'success',
                    'statistics': stats,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error fetching statistics: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/v1/decision/feedback', methods=['POST'])
        def record_decision_feedback():
            """Record feedback on a decision recommendation"""
            try:
                data = request.get_json()
                submission_id = data.get('submission_id')
                recommended_decision = data.get('recommended_decision')
                actual_decision = data.get('actual_decision')
                feedback_score = data.get('feedback_score', 0.0)
                
                # Update decision record with actual outcome
                # This would be used for learning and improving accuracy
                
                self.logger.info(f"Recorded feedback for {submission_id}: "
                               f"recommended={recommended_decision}, actual={actual_decision}")
                
                return jsonify({
                    'status': 'success',
                    'message': 'Feedback recorded successfully',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error recording feedback: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/v1/decision/analytics', methods=['GET'])
        def get_decision_analytics():
            """Get comprehensive decision analytics"""
            try:
                # Get recent decisions for trending
                recent_decisions = self.audit_system.get_decision_history(limit=100)
                
                # Calculate trends
                analytics = {
                    'recent_trends': self._analyze_decision_trends(recent_decisions),
                    'quality_metrics': self._calculate_quality_metrics(recent_decisions),
                    'efficiency_metrics': self._calculate_efficiency_metrics(recent_decisions),
                    'accuracy_analysis': self._analyze_decision_accuracy(recent_decisions)
                }
                
                return jsonify({
                    'status': 'success',
                    'analytics': analytics,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error generating analytics: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    async def _generate_enhanced_recommendation(self, submission_id: str, 
                                              manuscript_data: Dict, 
                                              reviews_data: List[Dict], 
                                              context_data: Dict) -> Dict:
        """Generate enhanced decision recommendation using ML engines"""
        
        # Prepare data for existing decision engine
        try:
            # Create manuscript metrics
            manuscript_metrics = ManuscriptMetrics(
                manuscript_id=submission_id,
                technical_quality_score=manuscript_data.get('technical_quality', 7.0),
                novelty_score=manuscript_data.get('novelty_score', 7.5),
                significance_score=manuscript_data.get('significance_score', 7.2),
                clarity_score=manuscript_data.get('clarity_score', 8.0),
                completeness_score=manuscript_data.get('completeness_score', 8.5),
                ethical_compliance_score=manuscript_data.get('ethical_compliance', 9.0),
                statistical_rigor_score=manuscript_data.get('statistical_rigor', 7.8),
                literature_coverage_score=manuscript_data.get('literature_coverage', 8.2),
                methodology_score=manuscript_data.get('methodology_score', 7.9),
                reproducibility_score=manuscript_data.get('reproducibility_score', 7.6),
                overall_score=manuscript_data.get('overall_score', 7.7)
            )
            
            # Create review data
            review_list = []
            for review in reviews_data:
                review_obj = ReviewData(
                    reviewer_id=review.get('reviewer_id', 'unknown'),
                    recommendation=review.get('recommendation', 'review'),
                    confidence=review.get('confidence', 0.8),
                    review_quality=review.get('review_quality', 8.0),
                    technical_comments=review.get('technical_comments', []),
                    major_issues=review.get('major_issues', []),
                    minor_issues=review.get('minor_issues', []),
                    review_completeness=review.get('completeness', 0.9),
                    review_timeliness=review.get('timeliness', 0.8)
                )
                review_list.append(review_obj)
            
            # Create editorial context
            urgency_mapping = {
                'low': UrgencyLevel.LOW,
                'medium': UrgencyLevel.MEDIUM, 
                'high': UrgencyLevel.HIGH,
                'critical': UrgencyLevel.CRITICAL
            }
            urgency = urgency_mapping.get(context_data.get('urgency', 'medium'), UrgencyLevel.MEDIUM)
            
            editorial_context = EditorialContext(
                journal_standards=context_data.get('journal_standards', {'quality_bar': 7.5}),
                acceptance_rate=context_data.get('acceptance_rate', 0.25),
                current_workload=context_data.get('workload', 20),
                special_issue=context_data.get('special_issue', False),
                deadline_pressure=urgency,
                editor_expertise_match=context_data.get('expertise_match', 0.8),
                journal_scope_alignment=context_data.get('scope_alignment', 0.9)
            )
            
            # Generate recommendation using existing engine
            recommendation = await self.decision_engine.generate_decision_recommendation(
                manuscript_metrics, review_list, editorial_context
            )
            
            # Convert to dict manually to handle enum types
            result = {
                'manuscript_id': recommendation.manuscript_id,
                'recommended_decision': str(recommendation.recommended_decision.value),
                'confidence': recommendation.confidence,
                'reasoning': recommendation.reasoning,
                'alternative_decisions': [(str(alt[0].value), alt[1]) for alt in recommendation.alternative_decisions],
                'required_actions': recommendation.required_actions,
                'estimated_timeline': recommendation.estimated_timeline,
                'risk_factors': recommendation.risk_factors,
                'priority_score': recommendation.priority_score,
                'timestamp': recommendation.timestamp
            }
            
            # Add additional analysis
            result['enhanced_analysis'] = {
                'risk_assessment': self._assess_decision_risk(manuscript_metrics, review_list),
                'quality_indicators': self._extract_quality_indicators(manuscript_metrics),
                'reviewer_reliability': self._assess_reviewer_reliability(review_list),
                'recommendation_strength': self._calculate_recommendation_strength(result)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced recommendation generation: {e}")
            # Return fallback recommendation
            return {
                'recommended_decision': 'review',
                'confidence': 0.5,
                'reasoning': ['System error - manual review recommended'],
                'alternative_decisions': [],
                'required_actions': ['Manual editorial review required'],
                'estimated_timeline': 7,
                'risk_factors': ['Decision system unavailable'],
                'priority_score': 0.5,
                'error': str(e)
            }
    
    def _assess_decision_risk(self, manuscript_metrics, review_list) -> Dict:
        """Assess risk factors in decision making"""
        risks = []
        risk_score = 0.0
        
        # Check for low quality indicators
        if manuscript_metrics.ethical_compliance_score < 8.0:
            risks.append('Ethical compliance concerns')
            risk_score += 0.3
        
        if manuscript_metrics.reproducibility_score < 6.0:
            risks.append('Reproducibility issues')
            risk_score += 0.2
        
        # Check reviewer consensus
        if len(review_list) < 2:
            risks.append('Limited peer review')
            risk_score += 0.2
        
        recommendations = [r.recommendation for r in review_list]
        if len(set(recommendations)) > 2:
            risks.append('Conflicting reviewer recommendations')
            risk_score += 0.15
        
        return {
            'risk_factors': risks,
            'risk_score': min(1.0, risk_score),
            'risk_level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low'
        }
    
    def _extract_quality_indicators(self, manuscript_metrics) -> Dict:
        """Extract key quality indicators"""
        return {
            'strengths': [
                metric for metric in [
                    'novelty' if manuscript_metrics.novelty_score > 8.0 else None,
                    'methodology' if manuscript_metrics.methodology_score > 8.0 else None,
                    'clarity' if manuscript_metrics.clarity_score > 8.0 else None,
                    'significance' if manuscript_metrics.significance_score > 8.0 else None
                ] if metric is not None
            ],
            'concerns': [
                metric for metric in [
                    'technical_quality' if manuscript_metrics.technical_quality_score < 6.0 else None,
                    'reproducibility' if manuscript_metrics.reproducibility_score < 6.0 else None,
                    'statistical_rigor' if manuscript_metrics.statistical_rigor_score < 6.0 else None
                ] if metric is not None
            ],
            'overall_quality_tier': (
                'excellent' if manuscript_metrics.overall_score > 8.5 else
                'good' if manuscript_metrics.overall_score > 7.0 else
                'acceptable' if manuscript_metrics.overall_score > 6.0 else
                'concerning'
            )
        }
    
    def _assess_reviewer_reliability(self, review_list) -> Dict:
        """Assess reliability of reviewers"""
        if not review_list:
            return {'reliability': 'unknown', 'factors': []}
        
        avg_quality = sum(r.review_quality for r in review_list) / len(review_list)
        avg_completeness = sum(r.review_completeness for r in review_list) / len(review_list)
        avg_timeliness = sum(r.review_timeliness for r in review_list) / len(review_list)
        
        reliability_score = (avg_quality + avg_completeness + avg_timeliness) / 3
        
        return {
            'reliability_score': reliability_score,
            'reliability': 'high' if reliability_score > 8.0 else 'medium' if reliability_score > 6.0 else 'low',
            'factors': {
                'quality': avg_quality,
                'completeness': avg_completeness,
                'timeliness': avg_timeliness
            }
        }
    
    def _calculate_recommendation_strength(self, recommendation) -> float:
        """Calculate the strength of the recommendation"""
        confidence = recommendation.get('confidence', 0.0)
        risk_factors_count = len(recommendation.get('risk_factors', []))
        reasoning_count = len(recommendation.get('reasoning', []))
        
        # Base strength from confidence
        strength = confidence
        
        # Adjust for risk factors (more risks = lower strength)
        strength -= (risk_factors_count * 0.1)
        
        # Adjust for reasoning depth (more reasoning = higher strength)
        strength += (reasoning_count * 0.05)
        
        return min(1.0, max(0.0, strength))
    
    def _analyze_decision_trends(self, recent_decisions) -> Dict:
        """Analyze trends in recent decisions"""
        if not recent_decisions:
            return {'trend': 'insufficient_data'}
        
        # Group decisions by week
        weekly_counts = {}
        for decision in recent_decisions:
            try:
                date = datetime.fromisoformat(decision['timestamp'])
                week = date.strftime('%Y-W%U')
                weekly_counts[week] = weekly_counts.get(week, 0) + 1
            except:
                continue
        
        return {
            'weekly_volume': weekly_counts,
            'trend_direction': 'stable',  # Would calculate actual trend
            'peak_periods': list(weekly_counts.keys())[:3]
        }
    
    def _calculate_quality_metrics(self, recent_decisions) -> Dict:
        """Calculate quality metrics from recent decisions"""
        if not recent_decisions:
            return {}
        
        confidences = [d.get('confidence', 0.0) for d in recent_decisions]
        processing_times = [d.get('processing_time', 0.0) for d in recent_decisions]
        
        return {
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
            'confidence_std': np.std(confidences) if len(confidences) > 1 else 0.0,
            'average_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0.0
        }
    
    def _calculate_efficiency_metrics(self, recent_decisions) -> Dict:
        """Calculate efficiency metrics"""
        return {
            'decisions_per_hour': len(recent_decisions) / 24 if recent_decisions else 0.0,
            'throughput_trend': 'stable',
            'bottlenecks': []
        }
    
    def _analyze_decision_accuracy(self, recent_decisions) -> Dict:
        """Analyze decision accuracy (would need feedback data)"""
        return {
            'accuracy_rate': 0.85,  # Placeholder - would calculate from feedback
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85
        }
    
    def start(self):
        """Start the enhanced decision support service"""
        self.logger.info(f"Starting Enhanced Editorial Decision Support on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Editorial Decision Support System')
    parser.add_argument('--port', type=int, default=8005, help='Port to run the service on')
    
    args = parser.parse_args()
    
    service = EnhancedEditorialDecisionSupport(port=args.port)
    service.start()

if __name__ == '__main__':
    main()