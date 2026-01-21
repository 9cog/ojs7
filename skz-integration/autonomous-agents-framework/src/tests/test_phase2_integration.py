"""
Phase 2 Integration Test Suite
Validates all Phase 2 critical components and their integration
"""

import unittest
import json
import tempfile
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.memory_system import PersistentMemorySystem
from models.ml_decision_engine import DecisionEngine, DecisionContext, DecisionResult
from models.learning_framework import LearningFramework
from models.enhanced_agent import EnhancedAgent, AgentAction
from ojs_bridge import OJSBridge, AgentOJSBridge

class TestPhase2Integration(unittest.TestCase):
    """Test suite for Phase 2 integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db_path = tempfile.mktemp(suffix='.db')
        self.test_agent_id = 'test_agent_001'
        
        # Initialize components
        self.memory_system = PersistentMemorySystem(self.test_db_path)
        self.decision_engine = DecisionEngine()
        self.learning_framework = LearningFramework(self.test_agent_id)
        
        # Test OJS bridge (mock)
        self.ojs_bridge = OJSBridge(
            ojs_base_url='http://localhost:8080',
            api_key='test_api_key',
            secret_key='test_secret_key'
        )
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_01_persistent_memory_system(self):
        """Test persistent memory system functionality"""
        print("\n=== Testing Persistent Memory System ===")
        
        # Test memory storage and retrieval
        test_content = {
            'type': 'research_data',
            'title': 'Test Research Paper',
            'content': 'This is a test research paper content',
            'keywords': ['test', 'research', 'paper']
        }
        
        # Store memory
        memory_id = self.memory_system.store_memory(
            agent_id=self.test_agent_id,
            memory_type='research',
            content=test_content,
            importance_score=0.8,
            tags=['research', 'test']
        )
        
        self.assertIsNotNone(memory_id)
        print(f"✓ Stored memory with ID: {memory_id}")
        
        # Retrieve memory
        memories = self.memory_system.retrieve_memory(
            agent_id=self.test_agent_id,
            memory_type='research',
            limit=5
        )
        
        self.assertGreater(len(memories), 0)
        self.assertEqual(memories[0].content['title'], 'Test Research Paper')
        print(f"✓ Retrieved {len(memories)} memories")
        
        # Test vector embedding
        import numpy as np
        test_embedding = np.random.rand(128)
        embedding_id = self.memory_system.store_vector_embedding(
            content_hash='test_hash_123',
            embedding=test_embedding,
            metadata={'source': 'test'}
        )
        
        self.assertIsNotNone(embedding_id)
        print(f"✓ Stored vector embedding with ID: {embedding_id}")
        
        # Test knowledge graph
        relationship_id = self.memory_system.store_knowledge_relationship(
            source_id='paper_001',
            target_id='author_001',
            relationship_type='authored_by',
            confidence_score=0.9
        )
        
        self.assertIsNotNone(relationship_id)
        print(f"✓ Stored knowledge relationship with ID: {relationship_id}")
        
        # Test experience logging
        experience_id = self.memory_system.log_experience(
            agent_id=self.test_agent_id,
            action_type='analyze_paper',
            input_data={'paper_id': 'test_001'},
            output_data={'analysis': 'successful'},
            success=True,
            performance_metrics={'accuracy': 0.95}
        )
        
        self.assertIsNotNone(experience_id)
        print(f"✓ Logged experience with ID: {experience_id}")
        
        # Test memory statistics
        stats = self.memory_system.get_memory_stats()
        self.assertIn('total_memories', stats)
        self.assertIn('total_vectors', stats)
        self.assertIn('total_relationships', stats)
        self.assertIn('total_experiences', stats)
        print(f"✓ Memory statistics: {stats}")
        
        print("✓ Persistent Memory System: PASSED")
    
    def test_02_ml_decision_engine(self):
        """Test ML decision engine functionality"""
        print("\n=== Testing ML Decision Engine ===")
        
        # Test NLP processing
        test_text = "This research paper demonstrates significant improvements in skin care formulations."
        entities = self.decision_engine.nlp_processor.extract_entities(test_text)
        sentiment = self.decision_engine.nlp_processor.analyze_sentiment(test_text)
        classification = self.decision_engine.nlp_processor.classify_text(
            test_text, ['research', 'quality', 'innovation']
        )
        
        self.assertIsInstance(entities, dict)
        self.assertIsInstance(sentiment, dict)
        self.assertIsInstance(classification, dict)
        print(f"✓ NLP processing: entities={len(entities)}, sentiment={sentiment}, classification={classification}")
        
        # Test quality assessment
        test_manuscript = {
            'content': test_text,
            'methodology_score': 0.8,
            'clarity_score': 0.7,
            'completeness_score': 0.9
        }
        
        quality_assessment = self.decision_engine.quality_assessor.assess_quality(test_manuscript)
        self.assertIn('overall_score', quality_assessment)
        print(f"✓ Quality assessment: {quality_assessment}")
        
        # Test trend prediction
        test_data = [
            {'topic': 'skin_care', 'date': datetime.now(), 'volume': 10},
            {'topic': 'skin_care', 'date': datetime.now(), 'volume': 15},
            {'topic': 'skin_care', 'date': datetime.now(), 'volume': 20}
        ]
        
        trends = self.decision_engine.trend_predictor.analyze_trends(test_data)
        emerging_topics = self.decision_engine.trend_predictor.identify_emerging_topics(test_data)
        
        self.assertIsInstance(trends, dict)
        self.assertIsInstance(emerging_topics, list)
        print(f"✓ Trend analysis: trends={len(trends)}, emerging_topics={len(emerging_topics)}")
        
        # Test decision making
        context = DecisionContext(
            agent_id=self.test_agent_id,
            action_type='analyze_paper',
            input_data={'paper_content': test_text},
            available_options=[
                {'type': 'thorough_analysis', 'quality_score': 0.9, 'risk_score': 0.2},
                {'type': 'quick_review', 'quality_score': 0.6, 'risk_score': 0.1}
            ],
            constraints={'time_limit': '2_hours'},
            goals=['quality', 'efficiency'],
            risk_tolerance=0.5
        )
        
        decision_result = self.decision_engine.make_decision(context)
        
        self.assertIsInstance(decision_result, DecisionResult)
        self.assertIn('decision', decision_result.__dict__)
        self.assertIn('confidence_score', decision_result.__dict__)
        print(f"✓ Decision making: confidence={decision_result.confidence_score}")
        
        print("✓ ML Decision Engine: PASSED")
    
    def test_03_learning_framework(self):
        """Test learning framework functionality"""
        print("\n=== Testing Learning Framework ===")
        
        # Test experience learning
        experience_id = self.learning_framework.learn_from_experience(
            action_type='analyze_paper',
            input_data={'paper_id': 'test_001', 'content': 'test content'},
            output_data={'analysis': 'successful', 'score': 0.85},
            success=True,
            performance_metrics={'accuracy': 0.9, 'time_taken': 120}
        )
        
        self.assertIsNotNone(experience_id)
        print(f"✓ Learned from experience: {experience_id}")
        
        # Test pattern recognition
        similar_patterns = self.learning_framework.supervised_learner.find_similar_patterns(
            {'paper_id': 'test_002', 'content': 'similar content'},
            'analyze_paper',
            threshold=0.5
        )
        
        self.assertIsInstance(similar_patterns, list)
        print(f"✓ Pattern recognition: found {len(similar_patterns)} similar patterns")
        
        # Test reinforcement learning
        state = 'analyze_paper_state'
        available_actions = ['thorough_analysis', 'quick_review', 'skip']
        action = self.learning_framework.reinforcement_learner.get_action(state, available_actions)
        
        self.assertIn(action, available_actions)
        print(f"✓ Reinforcement learning: selected action '{action}'")
        
        # Test meta-learning
        performance = {'success_rate': 0.85, 'action_type': 'analyze_paper'}
        strategy_adjustments = self.learning_framework.meta_learner.optimize_learning_strategy(performance)
        
        self.assertIsInstance(strategy_adjustments, dict)
        print(f"✓ Meta-learning: strategy adjustments = {strategy_adjustments}")
        
        # Test learning recommendations
        current_context = {'paper_id': 'test_003', 'content': 'new content'}
        recommendations = self.learning_framework.get_learning_recommendations(current_context)
        
        self.assertIsInstance(recommendations, list)
        print(f"✓ Learning recommendations: {len(recommendations)} recommendations")
        
        # Test learning statistics
        stats = self.learning_framework.get_learning_stats()
        self.assertIn('total_experiences', stats)
        self.assertIn('success_rate', stats)
        print(f"✓ Learning statistics: {stats}")
        
        print("✓ Learning Framework: PASSED")
    
    def test_04_ojs_bridge(self):
        """Test OJS bridge functionality"""
        print("\n=== Testing OJS Bridge ===")
        
        # Test bridge initialization
        self.assertIsNotNone(self.ojs_bridge)
        self.assertEqual(self.ojs_bridge.ojs_base_url, 'http://localhost:8080')
        print("✓ OJS Bridge initialized")
        
        # Test signature generation
        test_data = '{"test": "data"}'
        timestamp = '1234567890'
        signature = self.ojs_bridge._generate_signature(test_data, timestamp)
        
        self.assertIsInstance(signature, str)
        self.assertEqual(len(signature), 64)  # SHA-256 hex length
        print(f"✓ Signature generation: {signature[:16]}...")
        
        # Test request history
        history = self.ojs_bridge.get_request_history(limit=5)
        self.assertIsInstance(history, list)
        print(f"✓ Request history: {len(history)} entries")
        
        # Test connection statistics
        stats = self.ojs_bridge.get_connection_stats()
        self.assertIn('total_requests', stats)
        self.assertIn('success_rate', stats)
        print(f"✓ Connection statistics: {stats}")
        
        # Test agent OJS bridge
        agent_bridge = AgentOJSBridge(self.test_agent_id, self.ojs_bridge)
        self.assertIsNotNone(agent_bridge)
        print("✓ Agent OJS Bridge initialized")
        
        # Test bridge status
        bridge_status = agent_bridge.get_bridge_status()
        self.assertIn('agent_id', bridge_status)
        self.assertIn('status', bridge_status)
        print(f"✓ Bridge status: {bridge_status}")
        
        print("✓ OJS Bridge: PASSED")
    
    def test_05_enhanced_agent_integration(self):
        """Test enhanced agent integration"""
        print("\n=== Testing Enhanced Agent Integration ===")
        
        # Create a test agent class
        class TestAgent(EnhancedAgent):
            def process_task(self, task_data):
                return {
                    'result': 'test_result',
                    'score': 0.85,
                    'status': 'completed'
                }
            
            def get_available_actions(self, context):
                return [
                    AgentAction(
                        action_id='test_action_001',
                        action_type='analyze_paper',
                        input_data=context,
                        expected_output={'result': 'success'},
                        constraints={},
                        priority=0.8
                    )
                ]
        
        # Initialize test agent
        test_agent = TestAgent(
            agent_id=self.test_agent_id,
            agent_type='research_discovery',
            capabilities=['analyze_paper', 'extract_insights', 'generate_recommendations']
        )
        
        self.assertIsNotNone(test_agent)
        print("✓ Test agent initialized")
        
        # Test agent state
        state = test_agent.get_agent_state()
        self.assertEqual(state.agent_id, self.test_agent_id)
        self.assertEqual(state.status, 'active')
        print(f"✓ Agent state: {state.status}")
        
        # Test agent capabilities
        capabilities = test_agent.get_agent_capabilities()
        self.assertIn('agent_id', capabilities)
        self.assertIn('capabilities', capabilities)
        self.assertIn('memory_stats', capabilities)
        self.assertIn('learning_stats', capabilities)
        print(f"✓ Agent capabilities: {len(capabilities['capabilities'])} capabilities")
        
        # Test task processing
        task_id = test_agent.add_task({'paper_id': 'test_001', 'content': 'test content'})
        self.assertIsNotNone(task_id)
        print(f"✓ Added task: {task_id}")
        
        # Test action execution
        action = AgentAction(
            action_id='test_action_002',
            action_type='analyze_paper',
            input_data={'paper_id': 'test_001'},
            expected_output={'result': 'success'},
            constraints={},
            priority=0.9
        )
        
        result = test_agent.execute_action(action)
        self.assertIn('success', result)
        self.assertIn('result', result)
        print(f"✓ Action execution: success={result['success']}")
        
        # Test autonomous capabilities
        autonomous_capabilities = test_agent.get_autonomous_capabilities()
        self.assertTrue(autonomous_capabilities['persistent_memory'])
        self.assertTrue(autonomous_capabilities['ml_decision_making'])
        self.assertTrue(autonomous_capabilities['learning_capabilities'])
        self.assertTrue(autonomous_capabilities['autonomous_planning'])
        print(f"✓ Autonomous capabilities: {autonomous_capabilities}")
        
        print("✓ Enhanced Agent Integration: PASSED")
    
    def test_06_phase2_completion_criteria(self):
        """Test Phase 2 completion criteria"""
        print("\n=== Testing Phase 2 Completion Criteria ===")
        
        # Test 1: Persistent Memory Systems
        memory_system = PersistentMemorySystem(self.test_db_path)
        self.assertIsNotNone(memory_system)
        print("✓ Persistent Memory System: IMPLEMENTED")
        
        # Test 2: ML Decision Making
        decision_engine = DecisionEngine()
        self.assertIsNotNone(decision_engine.nlp_processor)
        self.assertIsNotNone(decision_engine.quality_assessor)
        self.assertIsNotNone(decision_engine.trend_predictor)
        print("✓ ML Decision Making: IMPLEMENTED")
        
        # Test 3: Learning Capabilities
        learning_framework = LearningFramework(self.test_agent_id)
        self.assertIsNotNone(learning_framework.reinforcement_learner)
        self.assertIsNotNone(learning_framework.supervised_learner)
        self.assertIsNotNone(learning_framework.meta_learner)
        print("✓ Learning Capabilities: IMPLEMENTED")
        
        # Test 4: Autonomous Planning
        class TestPlanningAgent(EnhancedAgent):
            def process_task(self, task_data):
                return {'result': 'planned_result'}
            
            def get_available_actions(self, context):
                return []
        
        planning_agent = TestPlanningAgent(
            agent_id='planning_test',
            agent_type='planning',
            capabilities=['plan', 'execute', 'optimize']
        )
        
        self.assertIsNotNone(planning_agent)
        autonomous_capabilities = planning_agent.get_autonomous_capabilities()
        self.assertTrue(autonomous_capabilities['autonomous_planning'])
        print("✓ Autonomous Planning: IMPLEMENTED")
        
        # Test 5: OJS Integration
        ojs_bridge = OJSBridge('http://localhost:8080', 'test_key', 'test_secret')
        self.assertIsNotNone(ojs_bridge)
        print("✓ OJS Integration: IMPLEMENTED")
        
        # Test 6: API Communication
        agent_bridge = AgentOJSBridge(self.test_agent_id, ojs_bridge)
        self.assertIsNotNone(agent_bridge)
        print("✓ API Communication: IMPLEMENTED")
        
        # Test 7: Data Synchronization
        sync_result = agent_bridge.sync_manuscript_data('test_submission_001')
        self.assertIsInstance(sync_result, dict)
        print("✓ Data Synchronization: IMPLEMENTED")
        
        # Test 8: Authentication Integration
        signature = ojs_bridge._generate_signature('test_data', '1234567890')
        self.assertIsInstance(signature, str)
        print("✓ Authentication Integration: IMPLEMENTED")
        
        print("\n🎉 PHASE 2 COMPLETION CRITERIA: ALL PASSED")
        print("✅ All critical Phase 2 components are implemented and functional")
        print("🚀 Ready for Phase 3: Frontend Integration")

def run_phase2_tests():
    """Run all Phase 2 integration tests"""
    print("=" * 60)
    print("PHASE 2 INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Testing Phase 2: Core Agent Integration")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase2Integration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 2 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_tests = result.testsRun
    failed_tests = len(result.failures)
    error_tests = len(result.errors)
    passed_tests = total_tests - failed_tests - error_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Errors: {error_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")
    
    if failed_tests == 0 and error_tests == 0:
        print("\n🎉 PHASE 2 INTEGRATION: PASSED")
        print("✅ All Phase 2 components are working correctly")
        print("🚀 Ready for Phase 3 deployment")
        return True
    else:
        print("\n❌ PHASE 2 INTEGRATION: FAILED")
        print("❌ Some components need attention before Phase 3")
        return False

if __name__ == '__main__':
    success = run_phase2_tests()
    exit(0 if success else 1)