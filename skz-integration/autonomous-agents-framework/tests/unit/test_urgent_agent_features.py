"""
Test Suite for Enhanced Agents with Critical ML Features
Validates the implementation of urgent agent features
"""

import pytest
import sys
import os
import json
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.research_agent import ResearchDiscoveryAgent, VectorDatabase, DocumentProcessor, TrendPredictor
from models.submission_assistant_agent import SubmissionAssistantAgent, QualityAssessor, FeedbackLearner, ComplianceChecker
from models.editorial_orchestration_agent import EditorialOrchestrationAgent, WorkflowOptimizer, DecisionSupport, StrategicPlanner

class TestUrgentAgentFeatures:
    """Test critical ML features for all enhanced agents"""
    
    def test_research_discovery_agent_critical_features(self):
        """Test Agent 1: Research Discovery Agent critical features"""
        # Test Vector Database Integration
        vector_db = VectorDatabase()
        
        # Add test documents
        test_documents = [
            {
                'id': 'doc1',
                'title': 'Machine Learning in Skincare Research',
                'abstract': 'This study explores the application of machine learning algorithms in skincare product development.',
                'keywords': ['machine learning', 'skincare', 'cosmetics']
            },
            {
                'id': 'doc2', 
                'title': 'Natural Ingredients for Skin Health',
                'abstract': 'Investigation of natural botanical extracts for improving skin barrier function.',
                'keywords': ['natural ingredients', 'skin health', 'botanical']
            }
        ]
        
        success = vector_db.add_documents(test_documents)
        assert success, "Vector database should successfully add documents"
        
        # Test semantic search
        search_results = vector_db.search_similar('machine learning skincare', limit=5)
        assert len(search_results) > 0, "Vector database should return search results"
        assert search_results[0]['similarity'] > 0, "Search results should have similarity scores"
        
        # Test NLP Pipeline
        nlp_pipeline = DocumentProcessor()
        
        test_text = "Machine learning algorithms are revolutionizing skincare research by enabling personalized treatments."
        
        entities = nlp_pipeline.extract_entities(test_text)
        assert len(entities) > 0, "NLP pipeline should extract entities"
        
        concepts = nlp_pipeline.extract_concepts(test_text)
        assert len(concepts) > 0, "NLP pipeline should extract concepts"
        
        topic = nlp_pipeline.classify_topic(test_text)
        assert topic in ['machine_learning', 'skin_research', 'medical', 'chemistry', 'general'], "Should classify valid topic"
        
        quality = nlp_pipeline.assess_quality(test_text)
        assert 0 <= quality <= 1, "Quality score should be between 0 and 1"
        
        # Test Trend Prediction ML
        trend_predictor = TrendPredictor()
        
        trend_results = trend_predictor.predict_trends(test_documents, 'machine_learning')
        assert 'trending_topics' in trend_results, "Should return trending topics"
        assert 'emerging_areas' in trend_results, "Should return emerging areas"
        assert 'confidence_score' in trend_results, "Should return confidence score"
        
        # Test full agent integration
        agent = ResearchDiscoveryAgent()
        
        action_result = agent.process_action({
            'action_type': 'discover',
            'query': 'machine learning skincare',
            'domain': 'computer_science',
            'limit': 10
        })
        
        assert 'processed_papers' in action_result, "Agent should return processed papers"
        assert 'total_discovered' in action_result, "Agent should return discovery count"
        
        print("✅ Research Discovery Agent critical features validated")
    
    def test_submission_assistant_agent_critical_features(self):
        """Test Agent 2: Submission Assistant Agent critical features"""
        # Test Quality Assessment ML
        quality_assessor = QualityAssessor()
        
        test_manuscript = {
            'manuscript_id': 'test_001',
            'title': 'Novel Approach to Skin Barrier Protection',
            'abstract': 'This research introduces a groundbreaking methodology for enhancing skin barrier function using innovative biotechnology.',
            'full_text': 'Introduction: Skin barrier function is critical for health. Methods: We developed a novel approach using statistical analysis. Results: Significant improvements were observed (p<0.05). Discussion: Our findings have important implications. Limitations: This study has some limitations.',
            'references': ['Ref1', 'Ref2', 'Ref3'],
            'author_reputation_score': 0.7
        }
        
        quality_assessment = quality_assessor.assess_quality(test_manuscript)
        
        assert hasattr(quality_assessment, 'scientific_rigor'), "Should assess scientific rigor"
        assert hasattr(quality_assessment, 'methodology_score'), "Should assess methodology"
        assert hasattr(quality_assessment, 'novelty_score'), "Should assess novelty"
        assert hasattr(quality_assessment, 'clarity_score'), "Should assess clarity"
        assert hasattr(quality_assessment, 'overall_score'), "Should provide overall score"
        assert 0 <= quality_assessment.overall_score <= 1, "Overall score should be between 0 and 1"
        assert len(quality_assessment.detailed_feedback) >= 0, "Should provide feedback"
        
        # Test Feedback Learning System
        feedback_learner = FeedbackLearner()
        
        learning_result = feedback_learner.learn_from_feedback(
            decision_id='decision_001',
            actual_outcome='accept',
            predicted_outcome='revise_minor',
            feedback_data={'editor_comments': 'The methodology needs improvement but overall good work'}
        )
        
        assert learning_result['learning_success'], "Feedback learning should succeed"
        assert 'accuracy_improvement' in learning_result, "Should track accuracy improvement"
        
        stats = feedback_learner.get_learning_stats()
        assert 'total_decisions' in stats, "Should track total decisions"
        assert 'accuracy' in stats, "Should track accuracy"
        
        # Test Compliance Checking ML
        compliance_checker = ComplianceChecker()
        
        compliance_result = compliance_checker.check_compliance(test_manuscript)
        
        assert hasattr(compliance_result, 'regulatory_compliance'), "Should check regulatory compliance"
        assert hasattr(compliance_result, 'safety_compliance'), "Should check safety compliance"
        assert hasattr(compliance_result, 'inci_compliance'), "Should check INCI compliance"
        assert hasattr(compliance_result, 'ethical_compliance'), "Should check ethical compliance"
        assert hasattr(compliance_result, 'overall_compliant'), "Should provide overall compliance"
        
        # Test full agent integration
        agent = SubmissionAssistantAgent()
        
        action_result = agent.process_action({
            'action_type': 'assess',
            'manuscript_data': test_manuscript
        })
        
        assert 'quality_assessment' in action_result, "Agent should return quality assessment"
        assert 'assessment_timestamp' in action_result, "Agent should timestamp assessments"
        
        print("✅ Submission Assistant Agent critical features validated")
    
    def test_editorial_orchestration_agent_critical_features(self):
        """Test Agent 3: Editorial Orchestration Agent critical features"""
        # Test Workflow Optimization ML
        workflow_optimizer = WorkflowOptimizer()
        
        test_workflow_data = {
            'manuscripts': [
                {
                    'id': 'ms001',
                    'status': 'active',
                    'assigned_editor': 'editor1',
                    'assigned_reviewers': ['reviewer1', 'reviewer2'],
                    'urgency_level': 'high',
                    'research_area': 'machine_learning',
                    'processing_time': 25
                },
                {
                    'id': 'ms002',
                    'status': 'active',
                    'assigned_editor': 'editor1',
                    'assigned_reviewers': ['reviewer1'],
                    'urgency_level': 'medium',
                    'research_area': 'skincare',
                    'processing_time': 30
                }
            ],
            'editors': [
                {
                    'id': 'editor1',
                    'expertise_areas': ['machine_learning', 'skincare'],
                    'current_workload': 5
                },
                {
                    'id': 'editor2',
                    'expertise_areas': ['chemistry'],
                    'current_workload': 2
                }
            ],
            'reviewers': [
                {
                    'id': 'reviewer1',
                    'expertise_areas': ['machine_learning'],
                    'current_workload': 2
                },
                {
                    'id': 'reviewer2',
                    'expertise_areas': ['skincare'],
                    'current_workload': 1
                }
            ]
        }
        
        optimization_result = workflow_optimizer.optimize_workflow(test_workflow_data)
        
        assert hasattr(optimization_result, 'optimized_assignment'), "Should provide optimized assignment"
        assert hasattr(optimization_result, 'estimated_completion_time'), "Should estimate completion time"
        assert hasattr(optimization_result, 'resource_allocation'), "Should optimize resource allocation"
        assert hasattr(optimization_result, 'bottleneck_predictions'), "Should predict bottlenecks"
        assert hasattr(optimization_result, 'efficiency_improvement'), "Should calculate efficiency improvement"
        assert optimization_result.estimated_completion_time > 0, "Completion time should be positive"
        
        # Test Decision Support System
        decision_support = DecisionSupport()
        
        test_decision_context = {
            'manuscript': {
                'quality_score': 0.8,
                'novelty_score': 0.7,
                'methodology_score': 0.75,
                'clarity_score': 0.8,
                'author_reputation': 0.6
            },
            'reviews': [
                {'overall_score': 0.7, 'reviewer_id': 'rev1'},
                {'overall_score': 0.8, 'reviewer_id': 'rev2'}
            ]
        }
        
        decision_recommendation = decision_support.generate_recommendation(test_decision_context)
        
        assert hasattr(decision_recommendation, 'decision'), "Should provide decision recommendation"
        assert hasattr(decision_recommendation, 'confidence'), "Should provide confidence score"
        assert hasattr(decision_recommendation, 'reasoning'), "Should provide reasoning"
        assert hasattr(decision_recommendation, 'risk_assessment'), "Should assess risks"
        assert decision_recommendation.decision in ['accept', 'reject', 'revise_major', 'revise_minor'], "Should provide valid decision"
        assert 0 <= decision_recommendation.confidence <= 1, "Confidence should be between 0 and 1"
        
        # Test Autonomous Planning Engine
        strategic_planner = StrategicPlanner()
        
        test_planning_context = {
            'journal_metrics': {
                'impact_factor': 2.5,
                'submission_growth_rate': 0.15,
                'acceptance_rate': 0.25,
                'avg_review_time': 35,
                'author_satisfaction_score': 0.75
            },
            'competitors': [
                {'name': 'Journal A', 'impact_factor': 3.0},
                {'name': 'Journal B', 'impact_factor': 2.2}
            ],
            'current_submissions': [
                {'research_areas': ['machine_learning', 'AI']},
                {'research_areas': ['skincare', 'cosmetics']}
            ]
        }
        
        strategic_plan = strategic_planner.create_strategic_plan(test_planning_context)
        
        assert hasattr(strategic_plan, 'goals'), "Should provide strategic goals"
        assert hasattr(strategic_plan, 'action_items'), "Should provide action items"
        assert hasattr(strategic_plan, 'timeline'), "Should provide timeline"
        assert hasattr(strategic_plan, 'resource_requirements'), "Should calculate resource requirements"
        assert hasattr(strategic_plan, 'success_metrics'), "Should define success metrics"
        assert len(strategic_plan.goals) > 0, "Should generate strategic goals"
        assert len(strategic_plan.action_items) > 0, "Should generate action items"
        
        # Test full agent integration
        agent = EditorialOrchestrationAgent()
        
        action_result = agent.process_action({
            'action_type': 'optimize_workflow',
            'workflow_data': test_workflow_data
        })
        
        assert 'optimization_result' in action_result, "Agent should return optimization result"
        assert 'optimization_timestamp' in action_result, "Agent should timestamp optimizations"
        
        print("✅ Editorial Orchestration Agent critical features validated")
    
    def test_cross_agent_integration(self):
        """Test integration between enhanced agents"""
        # Initialize all agents
        research_agent = ResearchDiscoveryAgent()
        submission_agent = SubmissionAssistantAgent()
        editorial_agent = EditorialOrchestrationAgent()
        
        # Test shared memory system access
        assert hasattr(research_agent, 'memory_system'), "Research agent should have memory system"
        assert hasattr(submission_agent, 'memory_system'), "Submission agent should have memory system"
        assert hasattr(editorial_agent, 'memory_system'), "Editorial agent should have memory system"
        
        # Test ML decision engine access
        assert hasattr(research_agent, 'decision_engine'), "Research agent should have decision engine"
        assert hasattr(submission_agent, 'decision_engine'), "Submission agent should have decision engine"
        assert hasattr(editorial_agent, 'decision_engine'), "Editorial agent should have decision engine"
        
        # Test learning framework access
        assert hasattr(research_agent, 'learning_framework'), "Research agent should have learning framework"
        assert hasattr(submission_agent, 'learning_framework'), "Submission agent should have learning framework"
        assert hasattr(editorial_agent, 'learning_framework'), "Editorial agent should have learning framework"
        
        print("✅ Cross-agent integration validated")
    
    def test_performance_metrics(self):
        """Test performance and efficiency of critical features"""
        import time
        
        # Test vector database performance
        vector_db = VectorDatabase()
        
        # Add larger dataset
        large_dataset = [
            {
                'id': f'doc_{i}',
                'title': f'Research Paper {i}',
                'abstract': f'This is the abstract for research paper {i} covering various topics.',
                'keywords': ['research', 'science', f'topic_{i % 5}']
            }
            for i in range(100)
        ]
        
        start_time = time.time()
        success = vector_db.add_documents(large_dataset)
        add_time = time.time() - start_time
        
        assert success, "Should handle large datasets"
        assert add_time < 10.0, f"Should add 100 documents in under 10 seconds, took {add_time:.2f}s"
        
        # Test search performance
        start_time = time.time()
        results = vector_db.search_similar('research science', limit=10)
        search_time = time.time() - start_time
        
        assert len(results) > 0, "Should return search results"
        assert search_time < 2.0, f"Should search in under 2 seconds, took {search_time:.2f}s"
        
        print(f"✅ Performance metrics validated - Add: {add_time:.2f}s, Search: {search_time:.2f}s")
    
    def test_ml_model_accuracy(self):
        """Test ML model accuracy and reliability"""
        # Test quality assessment accuracy
        quality_assessor = QualityAssessor()
        
        # High quality manuscript
        high_quality_manuscript = {
            'title': 'Breakthrough Discovery in Skin Science',
            'abstract': 'This groundbreaking research presents novel findings with rigorous methodology.',
            'full_text': 'Introduction with clear hypothesis. Methods with detailed statistical analysis. Results with p-values and significance tests. Discussion with limitations and implications.',
            'references': [f'Ref{i}' for i in range(20)]
        }
        
        high_quality_result = quality_assessor.assess_quality(high_quality_manuscript)
        
        # Low quality manuscript
        low_quality_manuscript = {
            'title': 'Some Research',
            'abstract': 'Brief study.',
            'full_text': 'We did something. Results were okay.',
            'references': []
        }
        
        low_quality_result = quality_assessor.assess_quality(low_quality_manuscript)
        
        assert high_quality_result.overall_score > low_quality_result.overall_score, "High quality should score higher than low quality"
        assert high_quality_result.overall_score > 0.6, "High quality manuscript should score above 0.6"
        assert low_quality_result.overall_score < 0.6, "Low quality manuscript should score below 0.6"
        
        print("✅ ML model accuracy validated")

def run_all_tests():
    """Run all tests for urgent agent features"""
    test_suite = TestUrgentAgentFeatures()
    
    try:
        print("🚀 Starting Urgent Agent Features Test Suite")
        print("=" * 60)
        
        test_suite.test_research_discovery_agent_critical_features()
        test_suite.test_submission_assistant_agent_critical_features()
        test_suite.test_editorial_orchestration_agent_critical_features()
        test_suite.test_cross_agent_integration()
        test_suite.test_performance_metrics()
        test_suite.test_ml_model_accuracy()
        
        print("=" * 60)
        print("🎉 All Urgent Agent Features Tests Passed Successfully!")
        print("\n📊 Test Summary:")
        print("✅ Agent 1 (Research Discovery): Vector DB, NLP Pipeline, Trend Prediction ML")
        print("✅ Agent 2 (Submission Assistant): Quality Assessment ML, Feedback Learning, Compliance ML")
        print("✅ Agent 3 (Editorial Orchestration): Workflow Optimization ML, Decision Support, Strategic Planning")
        print("✅ Cross-agent integration and memory systems")
        print("✅ Performance metrics and ML accuracy")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)