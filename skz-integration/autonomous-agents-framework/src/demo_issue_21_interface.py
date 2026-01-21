#!/usr/bin/env python3
"""
Demo script showcasing the exact Decision Engine interface specified in Issue #21.

This demonstrates the requirement:
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

def demo_issue_interface():
    """Demonstrate the exact interface from the issue"""
    print("============================================================")
    print("DECISION ENGINE - ISSUE #21 INTERFACE DEMO")
    print("============================================================")
    print()
    
    from models.universal_systems import GoalManager, ConstraintHandler, RiskAssessor, AdaptivePlanner
    from models.decision_engine import DecisionEngine
    
    print("Creating Decision Engine with the exact interface from Issue #21:")
    print()
    print("```python")
    print("# Required: Universal decision engine")
    print("decision_engine = DecisionEngine(")
    print("    goal_manager=GoalManager(),")
    print("    constraint_handler=ConstraintHandler(),")
    print("    risk_assessor=RiskAssessor(),")
    print("    adaptive_planner=AdaptivePlanner()")
    print(")")
    print("```")
    print()
    
    # Create the exact interface specified in the issue
    decision_engine = DecisionEngine(
        goal_manager=GoalManager(),
        constraint_handler=ConstraintHandler(),
        risk_assessor=RiskAssessor(),
        adaptive_planner=AdaptivePlanner()
    )
    
    print("✅ Decision Engine created successfully!")
    print(f"   Agent ID: {decision_engine.agent_id}")
    print(f"   Components: goal_manager, constraint_handler, risk_assessor, adaptive_planner")
    print()
    
    print("🎯 Testing Goal Management:")
    goal_id = decision_engine.goal_manager.create_goal(
        "Optimize manuscript processing",
        {"accuracy": 0.95, "speed": 0.8, "quality": 0.9}
    )
    print(f"   ✓ Created goal: {goal_id}")
    print()
    
    print("🛡️ Testing Constraint Handling:")
    constraint_id = decision_engine.constraint_handler.add_constraint(
        "resource",
        "Maximum resource utilization",
        {"max_cpu": 0.7, "max_memory": 0.8},
        True
    )
    print(f"   ✓ Added constraint: {constraint_id}")
    print()
    
    print("⚠️ Testing Risk Assessment:")
    risk_id = decision_engine.risk_assessor.add_risk_factor(
        "operational",
        "System overload risk",
        0.2,  # probability
        0.6,  # impact
        ["scale_down_operations", "alert_administrators"]
    )
    print(f"   ✓ Added risk factor: {risk_id}")
    print()
    
    print("🧠 Testing Decision Making:")
    decision = decision_engine.make_decision({
        "action_type": "process_manuscript",
        "required_resources": {"cpu": 0.6, "memory": 0.5},
        "estimated_duration": 120,
        "priority": "high",
        "quality_score": 0.85
    })
    
    print(f"   ✓ Decision made: {'PROCEED' if decision['can_proceed'] else 'HALT'}")
    print(f"   ✓ Confidence: {decision['confidence_score']:.2f}")
    print(f"   ✓ Active goals: {len(decision['active_goals'])}")
    print(f"   ✓ Constraint violations: {len(decision['constraint_violations'])}")
    print()
    
    print("📊 System Status:")
    status = decision_engine.get_status()
    print(f"   ✓ Active goals: {status['active_goals_count']}")
    print(f"   ✓ Active constraints: {status['active_constraints_count']}")
    print(f"   ✓ All components: {', '.join(status['components_status'].keys())}")
    print()
    
    print("🎉 Issue #21 - Decision Engine: SUCCESSFULLY IMPLEMENTED")
    print("============================================================")

if __name__ == "__main__":
    demo_issue_interface()