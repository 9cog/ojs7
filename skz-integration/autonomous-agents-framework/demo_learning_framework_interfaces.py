#!/usr/bin/env python3
"""
Demo: Learning Framework Interface Implementation
Shows both the new required interface and backward compatibility
"""

import sys
import os
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock numpy for demo
class MockNumpy:
    class ndarray:
        pass
    @staticmethod 
    def random():
        return 0.5
    @staticmethod
    def choice(options):
        return options[0] if options else None
    @staticmethod
    def mean(values):
        return sum(values) / len(values) if values else 0.0

sys.modules['numpy'] = MockNumpy()

# Import the learning framework
from models.learning_framework import (
    LearningFramework, 
    ReinforcementLearner, 
    SupervisedLearner, 
    UnsupervisedLearner, 
    MetaLearner
)

def demonstrate_required_interface():
    """Demonstrate the exact interface required by the issue"""
    print("🎯 DEMONSTRATING REQUIRED INTERFACE")
    print("=" * 50)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # This is the EXACT interface required by issue #20
        learning_framework = LearningFramework(
            reinforcement_learner=ReinforcementLearner("demo_agent"),
            supervised_learner=SupervisedLearner("demo_agent"),
            unsupervised_learner=UnsupervisedLearner("demo_agent"),
            meta_learner=MetaLearner("demo_agent"),
            db_path=db_path
        )
        
        print("✅ SUCCESS: Learning framework created with injectable learners")
        print(f"   Agent ID: {learning_framework.agent_id}")
        print(f"   Database: {learning_framework.db_path}")
        
        # Test functionality
        experience_id = learning_framework.learn_from_experience(
            action_type="manuscript_review",
            input_data={"manuscript_id": "demo_001", "reviewer": "expert_1"},
            output_data={"recommendation": "accept", "confidence": 0.95},
            success=True,
            performance_metrics={"processing_time": 2.5},
            feedback={"quality": "excellent"}
        )
        
        print(f"✅ SUCCESS: Learned from experience: {experience_id}")
        
        # Get stats
        stats = learning_framework.get_learning_stats()
        print(f"✅ SUCCESS: Framework stats: {stats['total_experiences']} experiences, {stats['success_rate']:.2f} success rate")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            os.unlink(db_path)
        except:
            pass

def demonstrate_backward_compatibility():
    """Demonstrate backward compatibility with existing interface"""
    print("\n🔄 DEMONSTRATING BACKWARD COMPATIBILITY")
    print("=" * 50)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # This is the existing interface that should still work
        learning_framework = LearningFramework(
            agent_id="legacy_agent",
            db_path=db_path
        )
        
        print("✅ SUCCESS: Learning framework created with traditional constructor")
        print(f"   Agent ID: {learning_framework.agent_id}")
        print(f"   Database: {learning_framework.db_path}")
        
        # Test functionality
        experience_id = learning_framework.learn_from_experience(
            action_type="peer_review_assignment",
            input_data={"manuscript_type": "research_article", "expertise_required": "machine_learning"},
            output_data={"assigned_reviewer": "dr_smith", "estimated_duration": "14_days"},
            success=True
        )
        
        print(f"✅ SUCCESS: Learned from experience: {experience_id}")
        
        # Get recommendations
        recommendations = learning_framework.get_learning_recommendations({
            "manuscript_type": "research_article",
            "expertise_required": "machine_learning"
        })
        
        print(f"✅ SUCCESS: Generated {len(recommendations)} recommendations")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            os.unlink(db_path)
        except:
            pass

def demonstrate_flexible_usage():
    """Demonstrate flexible usage patterns"""
    print("\n🔧 DEMONSTRATING FLEXIBLE USAGE")
    print("=" * 50)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # Partial injection - inject only some learners
        custom_rl = ReinforcementLearner("flexible_agent")
        
        learning_framework = LearningFramework(
            agent_id="flexible_agent",
            reinforcement_learner=custom_rl,  # Inject only RL learner
            db_path=db_path
        )
        
        print("✅ SUCCESS: Framework with partial injection")
        print(f"   Injected RL learner: {learning_framework.reinforcement_learner is custom_rl}")
        print(f"   Created SL learner: {learning_framework.supervised_learner is not None}")
        
        # Agent ID derivation
        auto_framework = LearningFramework(
            reinforcement_learner=ReinforcementLearner("auto_agent"),
            supervised_learner=SupervisedLearner("auto_agent"),
            unsupervised_learner=UnsupervisedLearner("auto_agent"),
            meta_learner=MetaLearner("auto_agent"),
            db_path=":memory:"
        )
        
        print(f"✅ SUCCESS: Auto-derived agent ID: {auto_framework.agent_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            os.unlink(db_path)
        except:
            pass

def main():
    """Run all demonstrations"""
    print("🚀 LEARNING FRAMEWORK INTERFACE DEMONSTRATION")
    print("=" * 60)
    print("Implementing issue #20: Universal Learning Framework")
    print("")
    
    success_count = 0
    total_demos = 3
    
    # Demonstrate required interface
    if demonstrate_required_interface():
        success_count += 1
    
    # Demonstrate backward compatibility  
    if demonstrate_backward_compatibility():
        success_count += 1
    
    # Demonstrate flexible usage
    if demonstrate_flexible_usage():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"DEMONSTRATION RESULTS: {success_count}/{total_demos} successful")
    
    if success_count == total_demos:
        print("🎉 ALL DEMONSTRATIONS SUCCESSFUL!")
        print("\nThe Learning Framework now supports:")
        print("  ✅ Required injectable interface (issue #20)")
        print("  ✅ Backward compatibility with existing code")
        print("  ✅ Flexible usage patterns")
        return True
    else:
        print("❌ SOME DEMONSTRATIONS FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)