#!/usr/bin/env python3
"""
Test script to validate the exact Decision Engine interface specified in the issue.

This tests the requirement:
```python
decision_engine = DecisionEngine(
    goal_manager=GoalManager(),
    constraint_handler=ConstraintHandler(),
    risk_assessor=RiskAssessor(),
    adaptive_planner=AdaptivePlanner()
)
```
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_exact_issue_interface():
    """Test the exact interface specified in the issue"""
    print("\n=== Testing Exact Issue Interface ===")
    
    try:
        from models.universal_systems import GoalManager, ConstraintHandler, RiskAssessor, AdaptivePlanner
        from models.decision_engine import DecisionEngine
        
        # Create the exact interface specified in the issue
        decision_engine = DecisionEngine(
            goal_manager=GoalManager(),
            constraint_handler=ConstraintHandler(),
            risk_assessor=RiskAssessor(),
            adaptive_planner=AdaptivePlanner()
        )
        
        print("✓ Created DecisionEngine with component parameters")
        
        # Verify all required components exist
        assert hasattr(decision_engine, 'goal_manager'), "GoalManager missing"
        assert hasattr(decision_engine, 'constraint_handler'), "ConstraintHandler missing"
        assert hasattr(decision_engine, 'risk_assessor'), "RiskAssessor missing"
        assert hasattr(decision_engine, 'adaptive_planner'), "AdaptivePlanner missing"
        
        print("✓ All required components present")
        
        # Test goal management functionality
        goal_id = decision_engine.goal_manager.create_goal(
            "Test goal from interface",
            {"accuracy": 0.9, "efficiency": 0.8}
        )
        print(f"✓ Created goal: {goal_id}")
        
        # Test constraint handling functionality
        constraint_id = decision_engine.constraint_handler.add_constraint(
            "resource",
            "Maximum CPU usage from interface",
            {"max_cpu": 0.8},
            True
        )
        print(f"✓ Added constraint: {constraint_id}")
        
        # Test risk assessment functionality
        risk_id = decision_engine.risk_assessor.add_risk_factor(
            "performance",
            "Low performance risk from interface",
            0.3,  # probability
            0.5,  # impact
            ["monitor_performance", "adjust_parameters"]
        )
        print(f"✓ Added risk factor: {risk_id}")
        
        # Test decision making
        decision = decision_engine.make_decision({
            "action_type": "test_action_from_interface",
            "required_resources": {"cpu": 0.5, "memory": 0.3},
            "estimated_duration": 60,
            "quality_score": 0.8
        })
        print(f"✓ Made decision: {decision['can_proceed']}")
        
        print("✅ Exact Issue Interface: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Exact Issue Interface: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_helper_function_interface():
    """Test the helper function that creates the same interface"""
    print("\n=== Testing Helper Function Interface ===")
    
    try:
        from models.universal_systems import create_decision_engine_with_components
        from models.universal_systems import GoalManager, ConstraintHandler, RiskAssessor, AdaptivePlanner
        
        # Test with default components
        decision_engine = create_decision_engine_with_components()
        print("✓ Created DecisionEngine with default components")
        
        # Test with custom components
        decision_engine_custom = create_decision_engine_with_components(
            goal_manager=GoalManager(),
            constraint_handler=ConstraintHandler(),
            risk_assessor=RiskAssessor(),
            adaptive_planner=AdaptivePlanner()
        )
        print("✓ Created DecisionEngine with custom components")
        
        # Verify all required components exist
        assert hasattr(decision_engine_custom, 'goal_manager'), "GoalManager missing"
        assert hasattr(decision_engine_custom, 'constraint_handler'), "ConstraintHandler missing"
        assert hasattr(decision_engine_custom, 'risk_assessor'), "RiskAssessor missing"
        assert hasattr(decision_engine_custom, 'adaptive_planner'), "AdaptivePlanner missing"
        
        print("✓ All required components present")
        
        print("✅ Helper Function Interface: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Helper Function Interface: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that the existing interface still works"""
    print("\n=== Testing Backward Compatibility ===")
    
    try:
        from models.universal_systems import create_universal_decision_engine
        
        # Test existing interface
        decision_engine = create_universal_decision_engine("test_agent")
        print("✓ Created DecisionEngine with existing interface")
        
        # Verify components
        assert hasattr(decision_engine, 'goal_manager'), "GoalManager missing"
        assert hasattr(decision_engine, 'constraint_handler'), "ConstraintHandler missing"
        assert hasattr(decision_engine, 'risk_assessor'), "RiskAssessor missing"
        assert hasattr(decision_engine, 'adaptive_planner'), "AdaptivePlanner missing"
        
        print("✓ All required components present")
        
        print("✅ Backward Compatibility: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Backward Compatibility: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("============================================================")
    print("DECISION ENGINE INTERFACE VALIDATION")
    print("============================================================")
    
    tests = [
        test_exact_issue_interface,
        test_helper_function_interface,
        test_backward_compatibility
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n============================================================")
    print("TEST RESULTS SUMMARY")
    print("============================================================")
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Issue requirements fully implemented!")
    else:
        print("❌ SOME TESTS FAILED - Check implementation")
        sys.exit(1)