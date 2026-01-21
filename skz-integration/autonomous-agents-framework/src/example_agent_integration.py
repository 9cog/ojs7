#!/usr/bin/env python3
"""
Example Agent Integration with Cross-Agent Critical Requirements
Shows how an existing agent can be updated to use the universal systems
"""

import sys
import os

# Add the src path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from models.universal_systems import (
    create_universal_memory_system,
    create_universal_learning_framework,
    create_universal_decision_engine
)

class ExampleAgent:
    """
    Example agent demonstrating integration with cross-agent critical requirements
    This shows the minimal changes needed to integrate existing agents
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Initialize universal systems as specified in the issue
        self.memory_system = create_universal_memory_system(agent_id)
        self.learning_framework = create_universal_learning_framework(agent_id)
        self.decision_engine = create_universal_decision_engine(agent_id)
        
        # Set up initial goals and constraints
        self._setup_agent()
        
        print(f"✓ Agent {agent_id} initialized with universal systems")
    
    def _setup_agent(self):
        """Set up initial goals and constraints for the agent"""
        
        # Create initial goal
        self.primary_goal_id = self.decision_engine.goal_manager.create_goal(
            "Provide autonomous academic publishing assistance",
            {"quality_score": 0.9, "efficiency": 0.85, "user_satisfaction": 0.9},
            priority="high"
        )
        
        # Add operational constraints
        self.decision_engine.constraint_handler.add_constraint(
            "quality",
            "Maintain minimum quality standards",
            {"min_quality": 0.8, "min_accuracy": 0.85},
            strict=True,
            priority="high"
        )
        
        self.decision_engine.constraint_handler.add_constraint(
            "resource",
            "Stay within resource limits",
            {"max_cpu": 0.7, "max_memory": 0.8, "max_network": 0.6},
            strict=True,
            priority="medium"
        )
        
        # Add common risk factors
        self.decision_engine.risk_assessor.add_risk_factor(
            "data_quality",
            "Risk of poor input data quality",
            0.3, 0.6,
            ["data_validation", "quality_checks", "fallback_data"],
            ["data_quality_metrics", "validation_results"]
        )
        
        self.decision_engine.risk_assessor.add_risk_factor(
            "system_load",
            "Risk of high system load affecting performance",
            0.2, 0.5,
            ["load_balancing", "resource_scaling", "request_throttling"],
            ["cpu_usage", "memory_usage", "response_time"]
        )
    
    def process_task(self, task_data: dict) -> dict:
        """
        Process a task using the universal systems
        This demonstrates the integration pattern for existing agent methods
        """
        
        # Step 1: Store context in memory system
        context_id = self.memory_system.context_memory.store_context(
            self.agent_id,
            {
                "task_type": task_data.get("type", "unknown"),
                "input_size": len(str(task_data)),
                "timestamp": task_data.get("timestamp"),
                "session_id": task_data.get("session_id")
            },
            {"source": "task_processing", "urgency": task_data.get("priority", "medium")},
            importance_score=0.7,
            tags=["task", "processing", task_data.get("type", "unknown")]
        )
        
        # Step 2: Make decision using decision engine
        decision_context = {
            "action_type": "process_task",
            "required_resources": {
                "cpu": 0.4,
                "memory": 0.3,
                "network": 0.2
            },
            "estimated_duration": task_data.get("estimated_duration", 60),
            "quality_score": 0.8,
            "task_complexity": task_data.get("complexity", "medium")
        }
        
        decision = self.decision_engine.make_decision(decision_context)
        
        if not decision["can_proceed"]:
            return {
                "success": False,
                "reason": "Decision engine blocked execution",
                "violations": decision["constraint_violations"],
                "recommendations": decision["recommendations"]
            }
        
        # Step 3: Execute task (simulated)
        print(f"  📋 Processing task: {task_data.get('type', 'unknown')}")
        print(f"  🎯 Decision confidence: {decision['confidence_score']:.2f}")
        
        # Simulate task execution
        import random
        execution_success = random.choice([True, True, True, False])  # 75% success rate
        quality_score = random.uniform(0.7, 0.95)
        
        result = {
            "success": execution_success,
            "quality_score": quality_score,
            "processing_time": random.randint(30, 120),
            "output_size": random.randint(100, 1000)
        }
        
        # Step 4: Learn from experience
        performance_metrics = {
            "execution_time": result["processing_time"],
            "quality_score": result["quality_score"],
            "resource_efficiency": random.uniform(0.7, 0.9)
        }
        
        feedback = {
            "decision_quality": "good" if execution_success else "needs_improvement",
            "system_performance": "stable",
            "output_quality": "high" if quality_score > 0.8 else "medium"
        }
        
        learning_exp_id = self.learning_framework.learn_from_experience(
            "process_task",
            task_data,
            result,
            execution_success,
            performance_metrics,
            feedback
        )
        
        # Step 5: Log experience in memory
        experience_id = self.memory_system.experience_db.log_experience(
            self.agent_id,
            "process_task",
            task_data,
            result,
            execution_success,
            performance_metrics
        )
        
        # Step 6: Update goal progress
        current_progress = min(1.0, random.uniform(0.1, 0.3))  # Incremental progress
        self.decision_engine.update_goal_progress(
            self.primary_goal_id, 
            current_progress,
            {
                "last_task_success": execution_success,
                "quality_trend": "improving" if quality_score > 0.8 else "stable",
                "efficiency_metrics": performance_metrics
            }
        )
        
        print(f"  📚 Learned from experience: {learning_exp_id}")
        print(f"  💾 Logged experience: {experience_id}")
        print(f"  📈 Updated goal progress: +{current_progress:.2f}")
        
        return {
            "success": execution_success,
            "result": result,
            "context_id": context_id,
            "experience_id": experience_id,
            "learning_exp_id": learning_exp_id,
            "decision": decision,
            "goal_progress": current_progress
        }
    
    def get_recommendations(self) -> dict:
        """Get recommendations from the learning system"""
        
        # Get learning insights
        learning_insights = self.learning_framework.meta_learner.get_learning_insights()
        
        # Get decision engine status
        decision_status = self.decision_engine.get_status()
        
        # Get recent experiences
        recent_experiences = self.memory_system.experience_db.get_experiences(
            self.agent_id, "process_task", limit=5
        )
        
        return {
            "learning_insights": learning_insights,
            "decision_status": decision_status,
            "recent_experiences_count": len(recent_experiences),
            "recommendations": [
                "Continue current approach if performance is stable",
                "Monitor resource usage to prevent constraint violations",
                "Collect more data for improved learning accuracy"
            ]
        }
    
    def cleanup(self):
        """Clean up agent resources"""
        print(f"🧹 Cleaning up agent {self.agent_id} resources...")
        
        # Clean up database files
        import os
        files_to_remove = [
            f"agent_memory_{self.agent_id}.db",
            f"learning_framework_{self.agent_id}.db",
            f"decision_engine_{self.agent_id}.db"
        ]
        
        for file in files_to_remove:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"  ✓ Removed {file}")
            except Exception as e:
                print(f"  ⚠️ Could not remove {file}: {e}")

def demonstrate_agent_integration():
    """Demonstrate how to integrate an existing agent with universal systems"""
    
    print("=" * 60)
    print("AGENT INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Create an example agent
    agent = ExampleAgent("research_discovery_agent")
    
    print(f"\n🤖 Created agent with universal systems:")
    print(f"  - Memory system: ✓")
    print(f"  - Learning framework: ✓") 
    print(f"  - Decision engine: ✓")
    
    # Simulate processing several tasks
    print(f"\n📋 Processing example tasks...")
    
    sample_tasks = [
        {
            "type": "research_analysis",
            "priority": "high",
            "complexity": "medium",
            "estimated_duration": 90,
            "session_id": "session_001",
            "data": {"query": "machine learning trends", "sources": ["pubmed", "arxiv"]}
        },
        {
            "type": "quality_assessment", 
            "priority": "medium",
            "complexity": "low",
            "estimated_duration": 45,
            "session_id": "session_001",
            "data": {"manuscript_id": "ms_001", "criteria": ["novelty", "methodology"]}
        },
        {
            "type": "trend_prediction",
            "priority": "high", 
            "complexity": "high",
            "estimated_duration": 120,
            "session_id": "session_002",
            "data": {"domain": "biotechnology", "timeframe": "2024-2025"}
        }
    ]
    
    results = []
    for i, task in enumerate(sample_tasks, 1):
        print(f"\n--- Task {i}: {task['type']} ---")
        result = agent.process_task(task)
        results.append(result)
        print(f"  ✓ Task completed: {'SUCCESS' if result['success'] else 'FAILED'}")
    
    # Get recommendations
    print(f"\n📊 Getting agent recommendations...")
    recommendations = agent.get_recommendations()
    
    print(f"  📈 Learning status: {recommendations['learning_insights']['status'] if 'status' in recommendations['learning_insights'] else 'active'}")
    print(f"  🎯 Active goals: {recommendations['decision_status']['active_goals_count']}")
    print(f"  📚 Recent experiences: {recommendations['recent_experiences_count']}")
    
    print(f"\n💡 Recommendations:")
    for rec in recommendations['recommendations']:
        print(f"  - {rec}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("INTEGRATION SUMMARY")
    print("=" * 60)
    
    successful_tasks = sum(1 for r in results if r['success'])
    print(f"📈 Tasks processed: {len(results)}")
    print(f"✅ Successful: {successful_tasks}")
    print(f"❌ Failed: {len(results) - successful_tasks}")
    print(f"📊 Success rate: {successful_tasks/len(results)*100:.1f}%")
    
    print(f"\n🔧 Integration benefits:")
    print(f"  ✓ Centralized memory management")
    print(f"  ✓ Continuous learning from experiences")
    print(f"  ✓ Intelligent decision making")
    print(f"  ✓ Goal tracking and progress monitoring")
    print(f"  ✓ Risk assessment and constraint handling")
    
    # Cleanup
    agent.cleanup()
    
    print(f"\n🎉 Agent integration demonstration complete!")

if __name__ == "__main__":
    demonstrate_agent_integration()