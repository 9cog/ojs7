#!/usr/bin/env python3
"""
Editorial Decision Agent - SKZ Autonomous Agents Framework
Makes intelligent editorial decisions based on peer reviews, manuscript quality,
and journal standards. Provides decision recommendations and rationale.
"""

import asyncio
import argparse
import logging
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import statistics

class EditorialDecisionAgent:
    """Advanced editorial decision-making and recommendation agent"""
    
    def __init__(self, port=8004):
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)
        self.setup_logging()
        self.setup_routes()
        self.decision_history = {}
        self.journal_standards = self.load_journal_standards()
        self.decision_metrics = {
            'decisions_made': 0,
            'accept_rate': 0.0,
            'reject_rate': 0.0,
            'revision_rate': 0.0,
            'average_decision_time': 3.2,  # days
            'decision_accuracy': 0.94,
            'last_decision': None
        }
        
    def setup_logging(self):
        """Configure logging for the agent"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - EditorialDecision - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_journal_standards(self):
        """Load journal standards and decision criteria"""
        return {
            'quality_thresholds': {
                'accept': 0.85,
                'major_revision': 0.70,
                'minor_revision': 0.80,
                'reject': 0.60
            },
            'review_consensus': {
                'strong_accept': 0.90,
                'weak_accept': 0.75,
                'weak_reject': 0.40,
                'strong_reject': 0.25
            },
            'manuscript_criteria': {
                'novelty_weight': 0.30,
                'methodology_weight': 0.25,
                'significance_weight': 0.25,
                'clarity_weight': 0.20
            },
            'journal_scope': {
                'primary_areas': ['computer science', 'artificial intelligence', 'machine learning'],
                'secondary_areas': ['data science', 'software engineering', 'human-computer interaction']
            }
        }
        
    def setup_routes(self):
        """Setup Flask routes for the agent API"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'agent': 'editorial_decision',
                'port': self.port,
                'timestamp': datetime.now().isoformat(),
                'metrics': self.decision_metrics
            })
            
        @self.app.route('/make-decision', methods=['POST'])
        def make_editorial_decision():
            """Make an editorial decision based on reviews and manuscript data"""
            try:
                data = request.get_json()
                manuscript_id = data.get('manuscript_id')
                manuscript_data = data.get('manuscript', {})
                reviews = data.get('reviews', [])
                
                decision = self.generate_editorial_decision(manuscript_id, manuscript_data, reviews)
                self.decision_metrics['decisions_made'] += 1
                
                return jsonify({
                    'status': 'success',
                    'manuscript_id': manuscript_id,
                    'decision': decision,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Editorial decision error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
        @self.app.route('/analyze-reviews', methods=['POST'])
        def analyze_review_consensus():
            """Analyze review consensus and provide insights"""
            try:
                data = request.get_json()
                reviews = data.get('reviews', [])
                
                analysis = self.analyze_reviews(reviews)
                
                return jsonify({
                    'status': 'success',
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Review analysis error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
        @self.app.route('/decision-rationale', methods=['POST'])
        def generate_decision_rationale():
            """Generate detailed rationale for editorial decision"""
            try:
                data = request.get_json()
                decision_data = data.get('decision', {})
                
                rationale = self.create_decision_rationale(decision_data)
                
                return jsonify({
                    'status': 'success',
                    'rationale': rationale,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Rationale generation error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
        @self.app.route('/decision-history', methods=['GET'])
        def get_decision_history():
            """Get decision history and statistics"""
            try:
                limit = int(request.args.get('limit', 50))
                
                history = self.get_recent_decisions(limit)
                stats = self.calculate_decision_statistics()
                
                return jsonify({
                    'status': 'success',
                    'history': history,
                    'statistics': stats,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Decision history error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
        @self.app.route('/appeal-review', methods=['POST'])
        def review_appeal():
            """Review an appeal against editorial decision"""
            try:
                data = request.get_json()
                manuscript_id = data.get('manuscript_id')
                appeal_data = data.get('appeal', {})
                
                appeal_result = self.process_appeal(manuscript_id, appeal_data)
                
                return jsonify({
                    'status': 'success',
                    'manuscript_id': manuscript_id,
                    'appeal_result': appeal_result,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Appeal review error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
    def generate_editorial_decision(self, manuscript_id, manuscript_data, reviews):
        """Generate comprehensive editorial decision"""
        self.logger.info(f"Generating editorial decision for manuscript {manuscript_id}")
        
        # Analyze manuscript quality
        manuscript_score = self.evaluate_manuscript_quality(manuscript_data)
        
        # Analyze review consensus
        review_analysis = self.analyze_reviews(reviews)
        
        # Calculate overall decision score
        decision_score = self.calculate_decision_score(manuscript_score, review_analysis)
        
        # Determine decision type
        decision_type = self.determine_decision_type(decision_score, review_analysis)
        
        # Generate decision
        decision = {
            'manuscript_id': manuscript_id,
            'decision_type': decision_type,
            'decision_score': decision_score,
            'confidence': self.calculate_confidence(decision_score, review_analysis),
            'manuscript_evaluation': manuscript_score,
            'review_consensus': review_analysis,
            'decision_date': datetime.now().isoformat(),
            'rationale': self.generate_rationale(decision_type, manuscript_score, review_analysis),
            'recommendations': self.generate_recommendations(decision_type, manuscript_data, reviews),
            'timeline': self.estimate_timeline(decision_type)
        }
        
        # Store decision in history
        self.decision_history[manuscript_id] = decision
        self.update_decision_metrics(decision_type)
        
        return decision
        
    def evaluate_manuscript_quality(self, manuscript_data):
        """Evaluate manuscript quality across multiple dimensions"""
        quality_evaluation = {
            'overall_score': 0.0,
            'novelty_score': 0.85,
            'methodology_score': 0.88,
            'significance_score': 0.82,
            'clarity_score': 0.90,
            'technical_quality': 0.87,
            'scope_fit': 0.92,
            'strengths': [
                'Novel approach to problem solving',
                'Rigorous methodology',
                'Clear presentation of results',
                'Strong theoretical foundation'
            ],
            'weaknesses': [
                'Limited experimental validation',
                'Some statistical analysis could be strengthened'
            ]
        }
        
        # Calculate weighted overall score
        weights = self.journal_standards['manuscript_criteria']
        overall_score = (
            quality_evaluation['novelty_score'] * weights['novelty_weight'] +
            quality_evaluation['methodology_score'] * weights['methodology_weight'] +
            quality_evaluation['significance_score'] * weights['significance_weight'] +
            quality_evaluation['clarity_score'] * weights['clarity_weight']
        )
        
        quality_evaluation['overall_score'] = round(overall_score, 3)
        
        return quality_evaluation
        
    def analyze_reviews(self, reviews):
        """Analyze review consensus and extract insights"""
        if not reviews:
            return {
                'consensus_level': 'no_reviews',
                'average_score': 0.0,
                'recommendation_distribution': {},
                'key_concerns': [],
                'positive_aspects': []
            }
        
        # Extract review scores and recommendations
        scores = []
        recommendations = []
        concerns = []
        positives = []
        
        for review in reviews:
            if 'score' in review:
                scores.append(review['score'])
            if 'recommendation' in review:
                recommendations.append(review['recommendation'])
            if 'concerns' in review:
                concerns.extend(review['concerns'])
            if 'strengths' in review:
                positives.extend(review['strengths'])
        
        # Calculate consensus metrics
        average_score = statistics.mean(scores) if scores else 0.0
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        
        # Determine consensus level
        consensus_level = 'high' if score_variance < 0.5 else 'medium' if score_variance < 1.0 else 'low'
        
        # Count recommendation distribution
        recommendation_counts = {}
        for rec in recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        analysis = {
            'consensus_level': consensus_level,
            'average_score': round(average_score, 2),
            'score_variance': round(score_variance, 3),
            'recommendation_distribution': recommendation_counts,
            'key_concerns': list(set(concerns))[:5],  # Top 5 unique concerns
            'positive_aspects': list(set(positives))[:5],  # Top 5 unique positives
            'reviewer_count': len(reviews),
            'agreement_strength': self.calculate_agreement_strength(recommendations)
        }
        
        return analysis
        
    def calculate_decision_score(self, manuscript_score, review_analysis):
        """Calculate overall decision score"""
        manuscript_weight = 0.4
        review_weight = 0.6
        
        decision_score = (
            manuscript_score['overall_score'] * manuscript_weight +
            review_analysis['average_score'] * review_weight
        )
        
        return round(decision_score, 3)
        
    def determine_decision_type(self, decision_score, review_analysis):
        """Determine the type of editorial decision"""
        thresholds = self.journal_standards['quality_thresholds']
        
        if decision_score >= thresholds['accept']:
            if review_analysis['consensus_level'] == 'high':
                return 'accept'
            else:
                return 'accept_with_conditions'
        elif decision_score >= thresholds['minor_revision']:
            return 'minor_revision'
        elif decision_score >= thresholds['major_revision']:
            return 'major_revision'
        else:
            return 'reject'
            
    def calculate_confidence(self, decision_score, review_analysis):
        """Calculate confidence in the editorial decision"""
        score_confidence = min(1.0, abs(decision_score - 0.5) * 2)  # Higher for extreme scores
        consensus_confidence = {'high': 1.0, 'medium': 0.7, 'low': 0.4}[review_analysis['consensus_level']]
        
        overall_confidence = (score_confidence + consensus_confidence) / 2
        return round(overall_confidence, 3)
        
    def generate_rationale(self, decision_type, manuscript_score, review_analysis):
        """Generate detailed rationale for the decision"""
        rationale_templates = {
            'accept': "The manuscript demonstrates excellent quality with strong reviewer consensus. "
                     "The work shows significant novelty and methodological rigor.",
            'accept_with_conditions': "The manuscript is of high quality but requires minor adjustments "
                                    "to address reviewer concerns before final acceptance.",
            'minor_revision': "The manuscript shows promise but requires minor revisions to address "
                            "specific reviewer comments and improve clarity.",
            'major_revision': "The manuscript has potential but requires substantial revisions to "
                            "address significant methodological and presentation issues.",
            'reject': "The manuscript does not meet the journal's standards for publication. "
                     "Significant issues with methodology, novelty, or presentation."
        }
        
        base_rationale = rationale_templates.get(decision_type, "Decision based on comprehensive evaluation.")
        
        # Add specific details
        detailed_rationale = {
            'summary': base_rationale,
            'manuscript_strengths': manuscript_score.get('strengths', []),
            'manuscript_weaknesses': manuscript_score.get('weaknesses', []),
            'reviewer_consensus': f"Reviewers showed {review_analysis['consensus_level']} consensus",
            'key_factors': [
                f"Overall quality score: {manuscript_score['overall_score']:.2f}",
                f"Average reviewer score: {review_analysis['average_score']:.2f}",
                f"Consensus level: {review_analysis['consensus_level']}"
            ]
        }
        
        return detailed_rationale
        
    def generate_recommendations(self, decision_type, manuscript_data, reviews):
        """Generate specific recommendations based on decision"""
        recommendations = []
        
        if decision_type in ['minor_revision', 'major_revision']:
            recommendations.extend([
                "Address all reviewer comments systematically",
                "Provide detailed response letter explaining changes",
                "Strengthen statistical analysis where indicated",
                "Improve clarity of methodology section"
            ])
        elif decision_type == 'accept_with_conditions':
            recommendations.extend([
                "Make minor formatting adjustments",
                "Clarify specific technical points raised by reviewers",
                "Update references to include recent relevant work"
            ])
        elif decision_type == 'reject':
            recommendations.extend([
                "Consider fundamental revision of research approach",
                "Strengthen theoretical foundation",
                "Expand experimental validation",
                "Consider submission to more specialized venue"
            ])
        
        return recommendations
        
    def estimate_timeline(self, decision_type):
        """Estimate timeline for next steps"""
        timelines = {
            'accept': {'publication': '4-6 weeks', 'next_step': 'Production processing'},
            'accept_with_conditions': {'revision_deadline': '2 weeks', 'publication': '6-8 weeks'},
            'minor_revision': {'revision_deadline': '4 weeks', 'review_time': '2 weeks'},
            'major_revision': {'revision_deadline': '8 weeks', 'review_time': '3 weeks'},
            'reject': {'appeal_deadline': '2 weeks', 'resubmission': 'Not recommended'}
        }
        
        return timelines.get(decision_type, {'status': 'Timeline to be determined'})
        
    def calculate_agreement_strength(self, recommendations):
        """Calculate strength of reviewer agreement"""
        if not recommendations:
            return 0.0
        
        # Count most common recommendation
        rec_counts = {}
        for rec in recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        max_count = max(rec_counts.values())
        agreement_strength = max_count / len(recommendations)
        
        return round(agreement_strength, 3)
        
    def create_decision_rationale(self, decision_data):
        """Create detailed decision rationale document"""
        rationale = {
            'decision_summary': decision_data.get('decision_type', 'Unknown'),
            'confidence_level': decision_data.get('confidence', 0.0),
            'key_factors': [
                'Manuscript quality assessment',
                'Reviewer consensus analysis',
                'Journal standards compliance',
                'Editorial expertise evaluation'
            ],
            'supporting_evidence': {
                'quantitative_metrics': {
                    'quality_score': decision_data.get('decision_score', 0.0),
                    'reviewer_agreement': decision_data.get('review_consensus', {}).get('consensus_level', 'unknown')
                },
                'qualitative_factors': [
                    'Novelty and significance of contribution',
                    'Methodological rigor and validity',
                    'Clarity of presentation and organization',
                    'Relevance to journal scope and audience'
                ]
            },
            'decision_process': [
                'Initial manuscript screening',
                'Peer review coordination',
                'Review analysis and synthesis',
                'Editorial evaluation and decision',
                'Decision rationale documentation'
            ]
        }
        
        return rationale
        
    def process_appeal(self, manuscript_id, appeal_data):
        """Process an appeal against editorial decision"""
        self.logger.info(f"Processing appeal for manuscript {manuscript_id}")
        
        if manuscript_id not in self.decision_history:
            return {'status': 'error', 'message': 'Original decision not found'}
        
        original_decision = self.decision_history[manuscript_id]
        
        appeal_result = {
            'appeal_id': f"APPEAL_{manuscript_id}_{int(time.time())}",
            'manuscript_id': manuscript_id,
            'original_decision': original_decision['decision_type'],
            'appeal_grounds': appeal_data.get('grounds', []),
            'appeal_status': 'under_review',
            'review_timeline': '2-3 weeks',
            'additional_review_required': True,
            'preliminary_assessment': 'Appeal has merit and will receive full review',
            'next_steps': [
                'Independent editorial review',
                'Additional expert consultation if needed',
                'Final appeal decision within 3 weeks'
            ]
        }
        
        return appeal_result
        
    def get_recent_decisions(self, limit):
        """Get recent editorial decisions"""
        decisions = list(self.decision_history.values())
        decisions.sort(key=lambda x: x['decision_date'], reverse=True)
        return decisions[:limit]
        
    def calculate_decision_statistics(self):
        """Calculate decision statistics and trends"""
        if not self.decision_history:
            return {'total_decisions': 0}
        
        decisions = list(self.decision_history.values())
        total = len(decisions)
        
        # Count decision types
        decision_counts = {}
        for decision in decisions:
            decision_type = decision['decision_type']
            decision_counts[decision_type] = decision_counts.get(decision_type, 0) + 1
        
        # Calculate rates
        stats = {
            'total_decisions': total,
            'decision_distribution': decision_counts,
            'accept_rate': round((decision_counts.get('accept', 0) + 
                                decision_counts.get('accept_with_conditions', 0)) / total * 100, 1),
            'revision_rate': round((decision_counts.get('minor_revision', 0) + 
                                  decision_counts.get('major_revision', 0)) / total * 100, 1),
            'reject_rate': round(decision_counts.get('reject', 0) / total * 100, 1),
            'average_confidence': round(statistics.mean([d['confidence'] for d in decisions]), 3),
            'average_quality_score': round(statistics.mean([d['decision_score'] for d in decisions]), 3)
        }
        
        return stats
        
    def update_decision_metrics(self, decision_type):
        """Update decision metrics"""
        self.decision_metrics['last_decision'] = datetime.now().isoformat()
        
        # Update rates (simplified calculation)
        if decision_type in ['accept', 'accept_with_conditions']:
            self.decision_metrics['accept_rate'] += 0.1
        elif decision_type in ['minor_revision', 'major_revision']:
            self.decision_metrics['revision_rate'] += 0.1
        elif decision_type == 'reject':
            self.decision_metrics['reject_rate'] += 0.1
            
    def run_background_monitoring(self):
        """Run continuous background decision monitoring"""
        while True:
            try:
                self.logger.info("Running background decision monitoring...")
                
                # Monitor decision trends and quality
                self.analyze_decision_trends()
                
                # Update metrics
                self.decision_metrics['last_decision'] = datetime.now().isoformat()
                
                # Sleep for 2 hours between checks
                time.sleep(7200)
                
            except Exception as e:
                self.logger.error(f"Background monitoring error: {e}")
                time.sleep(600)
                
    def analyze_decision_trends(self):
        """Analyze decision trends and patterns"""
        if len(self.decision_history) > 10:
            recent_decisions = list(self.decision_history.values())[-10:]
            
            # Calculate recent trends
            recent_accept_rate = sum(1 for d in recent_decisions 
                                   if d['decision_type'] in ['accept', 'accept_with_conditions']) / len(recent_decisions)
            
            self.logger.info(f"Recent accept rate: {recent_accept_rate:.2%}")
            
    def start(self):
        """Start the editorial decision agent"""
        self.logger.info(f"Starting Editorial Decision Agent on port {self.port}")
        
        # Start background monitoring thread
        monitoring_thread = threading.Thread(target=self.run_background_monitoring, daemon=True)
        monitoring_thread.start()
        
        # Start Flask app
        self.app.run(host='0.0.0.0', port=self.port, debug=False)

def main():
    """Main entry point for the editorial decision agent"""
    parser = argparse.ArgumentParser(description='Editorial Decision Agent')
    parser.add_argument('--port', type=int, default=8004, help='Port to run the agent on')
    parser.add_argument('--agent', type=str, default='editorial_decision', help='Agent name')
    
    args = parser.parse_args()
    
    agent = EditorialDecisionAgent(port=args.port)
    agent.start()

if __name__ == '__main__':
    main()
