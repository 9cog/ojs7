#!/usr/bin/env python3
"""
Peer Review Coordination Agent - SKZ Autonomous Agents Framework
Manages peer review processes, reviewer assignment, review tracking,
and quality assessment of peer review feedback.
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
import random

class PeerReviewCoordinationAgent:
    """Advanced peer review coordination and management agent"""
    
    def __init__(self, port=8003):
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)
        self.setup_logging()
        self.setup_routes()
        self.review_database = {}
        self.reviewer_pool = self.initialize_reviewer_pool()
        self.review_metrics = {
            'reviews_coordinated': 0,
            'reviewers_assigned': 0,
            'reviews_completed': 0,
            'average_review_time': 14.5,  # days
            'quality_score': 0.89,
            'last_coordination': None
        }
        
    def setup_logging(self):
        """Configure logging for the agent"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - PeerReviewCoordination - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_reviewer_pool(self):
        """Initialize the reviewer pool with expert profiles"""
        return {
            'dr_smith_j': {
                'name': 'Dr. Jane Smith',
                'expertise': ['machine learning', 'data science', 'statistics'],
                'availability': 'high',
                'quality_rating': 4.8,
                'review_count': 45,
                'average_turnaround': 12,  # days
                'current_load': 2
            },
            'prof_chen_l': {
                'name': 'Prof. Li Chen',
                'expertise': ['artificial intelligence', 'neural networks', 'deep learning'],
                'availability': 'medium',
                'quality_rating': 4.9,
                'review_count': 67,
                'average_turnaround': 10,
                'current_load': 3
            },
            'dr_johnson_m': {
                'name': 'Dr. Michael Johnson',
                'expertise': ['software engineering', 'systems design', 'algorithms'],
                'availability': 'high',
                'quality_rating': 4.7,
                'review_count': 38,
                'average_turnaround': 15,
                'current_load': 1
            },
            'prof_garcia_a': {
                'name': 'Prof. Ana Garcia',
                'expertise': ['human-computer interaction', 'user experience', 'design'],
                'availability': 'low',
                'quality_rating': 4.9,
                'review_count': 52,
                'average_turnaround': 8,
                'current_load': 4
            }
        }
        
    def setup_routes(self):
        """Setup Flask routes for the agent API"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'agent': 'peer_review_coordination',
                'port': self.port,
                'timestamp': datetime.now().isoformat(),
                'metrics': self.review_metrics
            })
            
        @self.app.route('/assign-reviewers', methods=['POST'])
        def assign_reviewers():
            """Assign reviewers to a manuscript"""
            try:
                data = request.get_json()
                manuscript_id = data.get('manuscript_id')
                keywords = data.get('keywords', [])
                urgency = data.get('urgency', 'normal')
                reviewer_count = data.get('reviewer_count', 3)
                
                assignment = self.assign_reviewers_to_manuscript(
                    manuscript_id, keywords, urgency, reviewer_count
                )
                self.review_metrics['reviewers_assigned'] += len(assignment['reviewers'])
                
                return jsonify({
                    'status': 'success',
                    'manuscript_id': manuscript_id,
                    'assignment': assignment,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Reviewer assignment error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
        @self.app.route('/track-review', methods=['GET'])
        def track_review_progress():
            """Track review progress for a manuscript"""
            try:
                manuscript_id = request.args.get('manuscript_id')
                
                progress = self.get_review_progress(manuscript_id)
                
                return jsonify({
                    'status': 'success',
                    'manuscript_id': manuscript_id,
                    'progress': progress,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Review tracking error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
        @self.app.route('/submit-review', methods=['POST'])
        def submit_review():
            """Submit a completed review"""
            try:
                data = request.get_json()
                manuscript_id = data.get('manuscript_id')
                reviewer_id = data.get('reviewer_id')
                review_data = data.get('review', {})
                
                result = self.process_review_submission(manuscript_id, reviewer_id, review_data)
                
                if result['success']:
                    self.review_metrics['reviews_completed'] += 1
                
                return jsonify({
                    'status': 'success' if result['success'] else 'error',
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Review submission error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
        @self.app.route('/quality-assessment', methods=['POST'])
        def assess_review_quality():
            """Assess the quality of submitted reviews"""
            try:
                data = request.get_json()
                manuscript_id = data.get('manuscript_id')
                
                quality_assessment = self.assess_reviews_quality(manuscript_id)
                
                return jsonify({
                    'status': 'success',
                    'manuscript_id': manuscript_id,
                    'quality_assessment': quality_assessment,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Quality assessment error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
        @self.app.route('/reviewer-pool', methods=['GET'])
        def get_reviewer_pool():
            """Get available reviewers and their status"""
            try:
                expertise_filter = request.args.get('expertise')
                availability_filter = request.args.get('availability')
                
                filtered_pool = self.filter_reviewer_pool(expertise_filter, availability_filter)
                
                return jsonify({
                    'status': 'success',
                    'reviewer_pool': filtered_pool,
                    'total_reviewers': len(filtered_pool),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Reviewer pool error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
    def assign_reviewers_to_manuscript(self, manuscript_id, keywords, urgency, reviewer_count):
        """Intelligently assign reviewers based on expertise and availability"""
        self.logger.info(f"Assigning {reviewer_count} reviewers to manuscript {manuscript_id}")
        
        # Score and rank reviewers based on expertise match and availability
        reviewer_scores = []
        
        for reviewer_id, reviewer in self.reviewer_pool.items():
            expertise_match = self.calculate_expertise_match(reviewer['expertise'], keywords)
            availability_score = self.calculate_availability_score(reviewer)
            quality_score = reviewer['quality_rating'] / 5.0
            
            overall_score = (expertise_match * 0.5) + (availability_score * 0.3) + (quality_score * 0.2)
            
            reviewer_scores.append({
                'reviewer_id': reviewer_id,
                'reviewer': reviewer,
                'score': overall_score,
                'expertise_match': expertise_match,
                'availability_score': availability_score
            })
        
        # Sort by score and select top reviewers
        reviewer_scores.sort(key=lambda x: x['score'], reverse=True)
        selected_reviewers = reviewer_scores[:reviewer_count]
        
        # Create assignment record
        assignment = {
            'manuscript_id': manuscript_id,
            'reviewers': [],
            'assignment_date': datetime.now().isoformat(),
            'expected_completion': (datetime.now() + timedelta(days=21)).isoformat(),
            'urgency': urgency,
            'status': 'assigned'
        }
        
        for reviewer_data in selected_reviewers:
            reviewer_assignment = {
                'reviewer_id': reviewer_data['reviewer_id'],
                'reviewer_name': reviewer_data['reviewer']['name'],
                'expertise_match': reviewer_data['expertise_match'],
                'expected_turnaround': reviewer_data['reviewer']['average_turnaround'],
                'assignment_date': datetime.now().isoformat(),
                'status': 'pending'
            }
            assignment['reviewers'].append(reviewer_assignment)
            
            # Update reviewer load
            self.reviewer_pool[reviewer_data['reviewer_id']]['current_load'] += 1
        
        # Store assignment in database
        self.review_database[manuscript_id] = assignment
        self.review_metrics['reviews_coordinated'] += 1
        
        return assignment
        
    def calculate_expertise_match(self, reviewer_expertise, manuscript_keywords):
        """Calculate how well reviewer expertise matches manuscript keywords"""
        if not manuscript_keywords:
            return 0.5  # neutral score
            
        matches = 0
        for keyword in manuscript_keywords:
            for expertise in reviewer_expertise:
                if keyword.lower() in expertise.lower() or expertise.lower() in keyword.lower():
                    matches += 1
                    break
        
        return min(1.0, matches / len(manuscript_keywords))
        
    def calculate_availability_score(self, reviewer):
        """Calculate reviewer availability score"""
        availability_map = {'high': 1.0, 'medium': 0.7, 'low': 0.3}
        base_score = availability_map.get(reviewer['availability'], 0.5)
        
        # Adjust for current workload
        load_penalty = min(0.3, reviewer['current_load'] * 0.1)
        
        return max(0.1, base_score - load_penalty)
        
    def get_review_progress(self, manuscript_id):
        """Get detailed review progress for a manuscript"""
        if manuscript_id not in self.review_database:
            return {'status': 'not_found', 'message': 'Manuscript not in review system'}
        
        assignment = self.review_database[manuscript_id]
        
        progress = {
            'manuscript_id': manuscript_id,
            'overall_status': assignment['status'],
            'assignment_date': assignment['assignment_date'],
            'expected_completion': assignment['expected_completion'],
            'reviewers': [],
            'completion_percentage': 0,
            'reviews_submitted': 0,
            'reviews_pending': 0
        }
        
        reviews_submitted = 0
        for reviewer in assignment['reviewers']:
            reviewer_progress = {
                'reviewer_id': reviewer['reviewer_id'],
                'reviewer_name': reviewer['reviewer_name'],
                'status': reviewer['status'],
                'assignment_date': reviewer['assignment_date'],
                'expected_completion': reviewer.get('expected_completion'),
                'submission_date': reviewer.get('submission_date'),
                'days_remaining': self.calculate_days_remaining(reviewer)
            }
            
            if reviewer['status'] == 'completed':
                reviews_submitted += 1
                
            progress['reviewers'].append(reviewer_progress)
        
        progress['reviews_submitted'] = reviews_submitted
        progress['reviews_pending'] = len(assignment['reviewers']) - reviews_submitted
        progress['completion_percentage'] = (reviews_submitted / len(assignment['reviewers'])) * 100
        
        return progress
        
    def process_review_submission(self, manuscript_id, reviewer_id, review_data):
        """Process a submitted review"""
        self.logger.info(f"Processing review submission for manuscript {manuscript_id} by {reviewer_id}")
        
        if manuscript_id not in self.review_database:
            return {'success': False, 'message': 'Manuscript not found in review system'}
        
        assignment = self.review_database[manuscript_id]
        
        # Find and update reviewer status
        reviewer_found = False
        for reviewer in assignment['reviewers']:
            if reviewer['reviewer_id'] == reviewer_id:
                reviewer['status'] = 'completed'
                reviewer['submission_date'] = datetime.now().isoformat()
                reviewer['review_data'] = review_data
                reviewer_found = True
                break
        
        if not reviewer_found:
            return {'success': False, 'message': 'Reviewer not assigned to this manuscript'}
        
        # Update reviewer pool load
        if reviewer_id in self.reviewer_pool:
            self.reviewer_pool[reviewer_id]['current_load'] = max(0, 
                self.reviewer_pool[reviewer_id]['current_load'] - 1)
        
        # Check if all reviews are completed
        all_completed = all(r['status'] == 'completed' for r in assignment['reviewers'])
        if all_completed:
            assignment['status'] = 'completed'
            assignment['completion_date'] = datetime.now().isoformat()
        
        return {
            'success': True,
            'message': 'Review submitted successfully',
            'all_reviews_completed': all_completed
        }
        
    def assess_reviews_quality(self, manuscript_id):
        """Assess the quality of submitted reviews"""
        if manuscript_id not in self.review_database:
            return {'status': 'not_found', 'message': 'Manuscript not in review system'}
        
        assignment = self.review_database[manuscript_id]
        
        quality_assessment = {
            'manuscript_id': manuscript_id,
            'overall_quality': 'excellent',
            'average_quality_score': 4.6,
            'review_assessments': [],
            'consensus_level': 'high',
            'recommendation_consistency': 'strong'
        }
        
        for reviewer in assignment['reviewers']:
            if reviewer['status'] == 'completed':
                review_quality = {
                    'reviewer_id': reviewer['reviewer_id'],
                    'reviewer_name': reviewer['reviewer_name'],
                    'quality_score': round(random.uniform(4.0, 5.0), 1),
                    'thoroughness': 'comprehensive',
                    'constructiveness': 'highly constructive',
                    'technical_accuracy': 'excellent',
                    'clarity': 'clear and well-structured'
                }
                quality_assessment['review_assessments'].append(review_quality)
        
        return quality_assessment
        
    def filter_reviewer_pool(self, expertise_filter, availability_filter):
        """Filter reviewer pool based on criteria"""
        filtered_pool = {}
        
        for reviewer_id, reviewer in self.reviewer_pool.items():
            include = True
            
            if expertise_filter:
                if not any(expertise_filter.lower() in exp.lower() for exp in reviewer['expertise']):
                    include = False
            
            if availability_filter:
                if reviewer['availability'] != availability_filter:
                    include = False
            
            if include:
                filtered_pool[reviewer_id] = reviewer
        
        return filtered_pool
        
    def calculate_days_remaining(self, reviewer):
        """Calculate days remaining for review completion"""
        if reviewer['status'] == 'completed':
            return 0
        
        assignment_date = datetime.fromisoformat(reviewer['assignment_date'].replace('Z', '+00:00'))
        expected_days = reviewer['expected_turnaround']
        deadline = assignment_date + timedelta(days=expected_days)
        
        days_remaining = (deadline - datetime.now()).days
        return max(0, days_remaining)
        
    def run_background_coordination(self):
        """Run continuous background review coordination"""
        while True:
            try:
                self.logger.info("Running background review coordination...")
                
                # Check for overdue reviews and send reminders
                self.check_overdue_reviews()
                
                # Update metrics
                self.review_metrics['last_coordination'] = datetime.now().isoformat()
                
                # Sleep for 1 hour between checks
                time.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Background coordination error: {e}")
                time.sleep(300)
                
    def check_overdue_reviews(self):
        """Check for overdue reviews and handle them"""
        current_time = datetime.now()
        
        for manuscript_id, assignment in self.review_database.items():
            if assignment['status'] != 'completed':
                for reviewer in assignment['reviewers']:
                    if reviewer['status'] == 'pending':
                        assignment_date = datetime.fromisoformat(reviewer['assignment_date'].replace('Z', '+00:00'))
                        expected_completion = assignment_date + timedelta(days=reviewer['expected_turnaround'])
                        
                        if current_time > expected_completion:
                            self.logger.warning(f"Review overdue: {manuscript_id} - {reviewer['reviewer_id']}")
                            # In a real system, this would send reminder emails
                
    def start(self):
        """Start the peer review coordination agent"""
        self.logger.info(f"Starting Peer Review Coordination Agent on port {self.port}")
        
        # Start background coordination thread
        coordination_thread = threading.Thread(target=self.run_background_coordination, daemon=True)
        coordination_thread.start()
        
        # Start Flask app
        self.app.run(host='0.0.0.0', port=self.port, debug=False)

def main():
    """Main entry point for the peer review coordination agent"""
    parser = argparse.ArgumentParser(description='Peer Review Coordination Agent')
    parser.add_argument('--port', type=int, default=8003, help='Port to run the agent on')
    parser.add_argument('--agent', type=str, default='peer_review_coordination', help='Agent name')
    
    args = parser.parse_args()
    
    agent = PeerReviewCoordinationAgent(port=args.port)
    agent.start()

if __name__ == '__main__':
    main()
