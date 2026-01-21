"""
Test Learning Framework Interface
Tests both the new injectable interface and backward compatibility
"""

import sys
import os
import traceback

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from models.learning_framework import (
    LearningFramework, 
    ReinforcementLearner, 
    SupervisedLearner, 
    UnsupervisedLearner, 
    MetaLearner
)

class TestLearningFrameworkInterface:
    """Test the Learning Framework interfaces"""
    
    def test_traditional_constructor(self):
        """Test backward compatibility - traditional constructor"""
        # This is how the framework was created before
        framework = LearningFramework(agent_id="test_agent", db_path=":memory:")
        
        # Verify all components are created
        assert framework.reinforcement_learner is not None
        assert framework.supervised_learner is not None  
        assert framework.unsupervised_learner is not None
        assert framework.meta_learner is not None
        
        # Verify agent_id is set correctly
        assert framework.agent_id == "test_agent"
        assert framework.reinforcement_learner.agent_id == "test_agent"
        assert framework.supervised_learner.agent_id == "test_agent"
        assert framework.unsupervised_learner.agent_id == "test_agent"
        assert framework.meta_learner.agent_id == "test_agent"
        print("✓ Traditional constructor test passed")
    
    def test_injectable_constructor_required_interface(self):
        """Test new interface - injectable learners as required by issue"""
        # This is the interface required by the issue
        reinforcement_learner = ReinforcementLearner("agent_001")
        supervised_learner = SupervisedLearner("agent_001")
        unsupervised_learner = UnsupervisedLearner("agent_001")
        meta_learner = MetaLearner("agent_001")
        
        framework = LearningFramework(
            reinforcement_learner=reinforcement_learner,
            supervised_learner=supervised_learner,
            unsupervised_learner=unsupervised_learner,
            meta_learner=meta_learner,
            db_path=":memory:"
        )
        
        # Verify injected components are used
        assert framework.reinforcement_learner is reinforcement_learner
        assert framework.supervised_learner is supervised_learner
        assert framework.unsupervised_learner is unsupervised_learner
        assert framework.meta_learner is meta_learner
        
        # Verify agent_id is derived correctly
        assert framework.agent_id == "agent_001"
        print("✓ Injectable constructor test passed")
    
    def test_partial_injection(self):
        """Test partial injection of learners"""
        custom_reinforcement = ReinforcementLearner("test_agent")
        
        framework = LearningFramework(
            agent_id="test_agent",
            reinforcement_learner=custom_reinforcement,
            db_path=":memory:"
        )
        
        # Verify mixed usage - some injected, some created
        assert framework.reinforcement_learner is custom_reinforcement
        assert framework.supervised_learner is not None
        assert framework.unsupervised_learner is not None
        assert framework.meta_learner is not None
        
        # Verify agent_id consistency
        assert framework.agent_id == "test_agent"
        assert framework.supervised_learner.agent_id == "test_agent"
        print("✓ Partial injection test passed")
    
    def test_framework_functionality(self):
        """Test that the framework functionality works with both interfaces"""
        # Test with traditional interface
        framework1 = LearningFramework(agent_id="test_agent_1", db_path=":memory:")
        experience_id1 = framework1.learn_from_experience(
            action_type="test_action",
            input_data={"key": "value"},
            output_data={"result": "success"},
            success=True
        )
        assert experience_id1.startswith("exp_test_agent_1")
        
        # Test with injectable interface
        framework2 = LearningFramework(
            reinforcement_learner=ReinforcementLearner("test_agent_2"),
            supervised_learner=SupervisedLearner("test_agent_2"),
            unsupervised_learner=UnsupervisedLearner("test_agent_2"),
            meta_learner=MetaLearner("test_agent_2"),
            db_path=":memory:"
        )
        experience_id2 = framework2.learn_from_experience(
            action_type="test_action",
            input_data={"key": "value"},
            output_data={"result": "success"},
            success=True
        )
        assert experience_id2.startswith("exp_test_agent_2")
        
        # Both should generate stats
        stats1 = framework1.get_learning_stats()
        stats2 = framework2.get_learning_stats()
        
        assert stats1['total_experiences'] == 1
        assert stats2['total_experiences'] == 1
        print("✓ Framework functionality test passed")
    
    def test_agent_id_derivation(self):
        """Test agent_id derivation when not explicitly provided"""
        reinforcement_learner = ReinforcementLearner("derived_agent")
        
        framework = LearningFramework(
            reinforcement_learner=reinforcement_learner,
            supervised_learner=SupervisedLearner("derived_agent"),
            unsupervised_learner=UnsupervisedLearner("derived_agent"),
            meta_learner=MetaLearner("derived_agent"),
            db_path=":memory:"
        )
        
        # Should derive agent_id from reinforcement learner
        assert framework.agent_id == "derived_agent"
        print("✓ Agent ID derivation test passed")
    
    def test_default_agent_fallback(self):
        """Test fallback to default agent when no agent_id available"""
        framework = LearningFramework(db_path=":memory:")
        
        # Should fallback to default_agent
        assert framework.agent_id == "default_agent"
        assert framework.reinforcement_learner.agent_id == "default_agent"
        print("✓ Default agent fallback test passed")

def run_tests():
    """Run all tests"""
    test_instance = TestLearningFrameworkInterface()
    tests = [
        test_instance.test_traditional_constructor,
        test_instance.test_injectable_constructor_required_interface,
        test_instance.test_partial_injection,
        test_instance.test_framework_functionality,
        test_instance.test_agent_id_derivation,
        test_instance.test_default_agent_fallback
    ]
    
    print("Running Learning Framework Interface Tests...")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1
    
    print("=" * 50)
    print(f"Tests completed: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)