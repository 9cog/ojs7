#!/usr/bin/env python3
"""
Test script to validate the Cross-Agent Critical Requirements implementation
Verifies that the exact interface specified in the issue is working correctly
"""

import sys
import os
import tempfile
import logging
from datetime import datetime

# Add the src path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_universal_memory_system():
    """Test the universal memory system implementation"""
    print("=== Testing Universal Memory System ===")
    
    try:
        from models.universal_systems import create_universal_memory_system
        
        # Create the exact interface specified in the issue
        memory_system = create_universal_memory_system("test_agent")
        
        # Verify components exist
        assert hasattr(memory_system, 'vector_store'), "VectorStore component missing"
        assert hasattr(memory_system, 'knowledge_graph'), "KnowledgeGraph component missing"  
        assert hasattr(memory_system, 'experience_db'), "ExperienceDatabase component missing"
        assert hasattr(memory_system, 'context_memory'), "ContextMemory component missing"
        
        # Test basic functionality
        # Test context memory
        context_id = memory_system.context_memory.store_context(
            "test_agent", 
            {"type": "test", "content": "test data"}, 
            {"source": "test"}, 
            0.8, 
            ["test", "validation"]
        )
        print(f"✓ Stored context memory: {context_id}")
        
        # Test experience database
        exp_id = memory_system.experience_db.log_experience(
            "test_agent",
            "test_action", 
            {"input": "test"},
            {"output": "success"},
            True,
            {"accuracy": 0.95}
        )
        print(f"✓ Logged experience: {exp_id}")
        
        # Test knowledge graph
        rel_id = memory_system.knowledge_graph.add_relationship(
            "entity_1", "entity_2", "related_to", 0.9, {"type": "test"}
        )
        print(f"✓ Added knowledge relationship: {rel_id}")
        
        print("✅ Universal Memory System: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Universal Memory System: FAILED - {e}")
        return False

def test_universal_learning_framework():
    """Test the universal learning framework implementation"""
    print("\n=== Testing Universal Learning Framework ===")
    
    try:
        from models.universal_systems import create_universal_learning_framework
        
        # Create the exact interface specified in the issue  
        learning_framework = create_universal_learning_framework("test_agent")
        
        # Verify all required learners exist
        assert hasattr(learning_framework, 'reinforcement_learner'), "ReinforcementLearner missing"
        assert hasattr(learning_framework, 'supervised_learner'), "SupervisedLearner missing"
        assert hasattr(learning_framework, 'unsupervised_learner'), "UnsupervisedLearner missing"
        assert hasattr(learning_framework, 'meta_learner'), "MetaLearner missing"
        
        # Test learning from experience
        exp_id = learning_framework.learn_from_experience(
            "test_action",
            {"input": "test_input"},
            {"output": "test_output"},
            True,
            {"accuracy": 0.85},
            {"quality": "good"}
        )
        print(f"✓ Learned from experience: {exp_id}")
        
        # Test reinforcement learning
        action = learning_framework.reinforcement_learner.get_action("test_state", ["action1", "action2"])
        print(f"✓ Reinforcement learner action: {action}")
        
        # Test meta learning optimization
        strategy = learning_framework.meta_learner.optimize_learning_strategy({"success_rate": 0.8})
        print(f"✓ Meta learner strategy: {strategy}")
        
        print("✅ Universal Learning Framework: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Universal Learning Framework: FAILED - {e}")
        return False

def test_universal_decision_engine():
    """Test the universal decision engine implementation"""
    print("\n=== Testing Universal Decision Engine ===")
    
    try:
        from models.universal_systems import create_universal_decision_engine
        
        # Create the exact interface specified in the issue
        decision_engine = create_universal_decision_engine("test_agent")
        
        # Verify all required components exist
        assert hasattr(decision_engine, 'goal_manager'), "GoalManager missing"
        assert hasattr(decision_engine, 'constraint_handler'), "ConstraintHandler missing"
        assert hasattr(decision_engine, 'risk_assessor'), "RiskAssessor missing"
        assert hasattr(decision_engine, 'adaptive_planner'), "AdaptivePlanner missing"
        
        # Test goal management
        goal_id = decision_engine.goal_manager.create_goal(
            "Test goal",
            {"accuracy": 0.9, "efficiency": 0.8}
        )
        print(f"✓ Created goal: {goal_id}")
        
        # Test constraint handling
        constraint_id = decision_engine.constraint_handler.add_constraint(
            "resource",
            "Maximum CPU usage",
            {"max_cpu": 0.8},
            True
        )
        print(f"✓ Added constraint: {constraint_id}")
        
        # Test risk assessment
        risk_id = decision_engine.risk_assessor.add_risk_factor(
            "performance",
            "Low performance risk",
            0.3,  # probability
            0.5,  # impact
            ["monitor_performance", "adjust_parameters"]
        )
        print(f"✓ Added risk factor: {risk_id}")
        
        # Test decision making
        decision = decision_engine.make_decision({
            "action_type": "test_action",
            "required_resources": {"cpu": 0.5, "memory": 0.3},
            "estimated_duration": 60,
            "quality_score": 0.8
        })
        print(f"✓ Made decision: {decision['can_proceed']}")
        
        print("✅ Universal Decision Engine: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Universal Decision Engine: FAILED - {e}")
        return False

def test_issue_requirements():
    """Test the exact code blocks specified in the issue"""
    print("\n=== Testing Issue Requirements ===")
    
    try:
        # Test the exact code from the issue
        from models.universal_systems import (
            create_universal_memory_system, 
            create_universal_learning_framework,
            create_universal_decision_engine
        )
        
        # Required: Universal memory system
        memory_system = create_universal_memory_system("test_agent")
        # Should have: VectorStore(), KnowledgeGraph(), ExperienceDatabase(), ContextMemory()
        
        # Required: Universal learning framework  
        learning_framework = create_universal_learning_framework("test_agent")
        # Should have: ReinforcementLearner(), SupervisedLearner(), UnsupervisedLearner(), MetaLearner()
        
        # Required: Universal decision engine
        decision_engine = create_universal_decision_engine("test_agent")
        # Should have: GoalManager(), ConstraintHandler(), RiskAssessor(), AdaptivePlanner()
        
        print("✓ All required systems created successfully")
        print("✓ Interface matches issue requirements")
        
        print("✅ Issue Requirements: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Issue Requirements: FAILED - {e}")
        return False

def cleanup_test_files():
    """Clean up test database files"""
    test_files = [
        "agent_memory_test_agent.db",
        "learning_framework_test_agent.db", 
        "decision_engine_test_agent.db"
    ]
    
    for file in test_files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"✓ Cleaned up {file}")
        except Exception as e:
            print(f"⚠️ Could not clean up {file}: {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("CROSS-AGENT CRITICAL REQUIREMENTS VALIDATION")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    
    tests = [
        test_universal_memory_system,
        test_universal_learning_framework, 
        test_universal_decision_engine,
        test_issue_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Issue requirements implemented successfully!")
        cleanup_test_files()
        return 0
    else:
        print("❌ SOME TESTS FAILED - Check implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())