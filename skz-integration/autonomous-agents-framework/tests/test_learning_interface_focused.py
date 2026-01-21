"""
Focused test for Learning Framework Interface 
Minimal test to verify the exact requirement implementation
"""

import sys
import os
import tempfile

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Use real numpy instead of mocking
import numpy as np

# Now import the learning framework 
from models.learning_framework import (
    LearningFramework, 
    ReinforcementLearner, 
    SupervisedLearner, 
    UnsupervisedLearner, 
    MetaLearner
)

def test_required_interface():
    """Test the exact interface required by the issue"""
    print("Testing required interface...")
    
    # Create temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # This is the exact interface required by the issue
        learning_framework = LearningFramework(
            reinforcement_learner=ReinforcementLearner("test_agent"),
            supervised_learner=SupervisedLearner("test_agent"),
            unsupervised_learner=UnsupervisedLearner("test_agent"),
            meta_learner=MetaLearner("test_agent"),
            db_path=db_path
        )
        
        # Verify the framework was created successfully
        assert learning_framework is not None
        assert learning_framework.reinforcement_learner is not None
        assert learning_framework.supervised_learner is not None
        assert learning_framework.unsupervised_learner is not None
        assert learning_framework.meta_learner is not None
        
        print("✓ Required interface test PASSED")
        
        # Test that the framework can learn from experience
        experience_id = learning_framework.learn_from_experience(
            action_type="test_action",
            input_data={"input": "test"},
            output_data={"output": "result"},
            success=True
        )
        
        assert experience_id is not None
        assert "test_agent" in experience_id
        
        print("✓ Learning functionality test PASSED")
        
        return True
        
    except Exception as e:
        print(f"✗ Required interface test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        try:
            os.unlink(db_path)
        except:
            pass

def test_backward_compatibility():
    """Test that the old interface still works"""
    print("Testing backward compatibility...")
    
    # Create temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # This is the old interface that should still work
        learning_framework = LearningFramework(
            agent_id="test_agent",
            db_path=db_path
        )
        
        # Verify the framework was created successfully
        assert learning_framework is not None
        assert learning_framework.reinforcement_learner is not None
        assert learning_framework.supervised_learner is not None
        assert learning_framework.unsupervised_learner is not None
        assert learning_framework.meta_learner is not None
        assert learning_framework.agent_id == "test_agent"
        
        print("✓ Backward compatibility test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        try:
            os.unlink(db_path)
        except:
            pass

def main():
    """Run all tests"""
    print("Learning Framework Interface Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 2
    
    # Test required interface
    if test_required_interface():
        tests_passed += 1
    
    # Test backward compatibility  
    if test_backward_compatibility():
        tests_passed += 1
    
    print("=" * 40)
    print(f"Tests: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests PASSED!")
        return True
    else:
        print("❌ Some tests FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)