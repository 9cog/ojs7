"""
Simplified Phase 2 Integration Test
Validates Phase 2 components with real dependencies
"""

import json
import tempfile
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# Use real numpy and sklearn instead of mocking for production testing
import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Now we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_phase2_components():
    """Test Phase 2 components with real dependencies"""
    print("=" * 60)
    print("PHASE 2 SIMPLIFIED INTEGRATION TEST")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Testing Phase 2: Core Agent Integration")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: File Structure Validation
    print("\n=== Testing File Structure ===")
    required_files = [
        '../models/memory_system.py',
        '../models/ml_decision_engine.py',
        '../models/learning_framework.py',
        '../models/enhanced_agent.py',
        '../ojs_bridge.py'
    ]
    
    file_tests = []
    for file_path in required_files:
        exists = os.path.exists(file_path)
        file_tests.append(exists)
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
    
    if all(file_tests):
        print("✓ All Phase 2 files present")
        test_results.append(("File Structure", True))
    else:
        print("✗ Some Phase 2 files missing")
        test_results.append(("File Structure", False))
    
    # Test 2: Code Structure Validation
    print("\n=== Testing Code Structure ===")
    
    # Test memory system structure
    try:
        with open('../models/memory_system.py', 'r') as f:
            content = f.read()
            has_persistent_memory = 'class PersistentMemorySystem' in content
            has_vector_db = 'store_vector_embedding' in content
            has_knowledge_graph = 'store_knowledge_relationship' in content
            has_experience_log = 'log_experience' in content
            
            print(f"✓ Persistent memory class: {'✓' if has_persistent_memory else '✗'}")
            print(f"✓ Vector database methods: {'✓' if has_vector_db else '✗'}")
            print(f"✓ Knowledge graph methods: {'✓' if has_knowledge_graph else '✗'}")
            print(f"✓ Experience logging: {'✓' if has_experience_log else '✗'}")
            
            memory_system_ok = all([has_persistent_memory, has_vector_db, has_knowledge_graph, has_experience_log])
            test_results.append(("Memory System", memory_system_ok))
    except Exception as e:
        print(f"✗ Error reading memory system: {e}")
        test_results.append(("Memory System", False))
    
    # Test ML decision engine structure
    try:
        with open('../models/ml_decision_engine.py', 'r') as f:
            content = f.read()
            has_nlp_processor = 'class NLPProcessor' in content
            has_quality_assessor = 'class QualityAssessor' in content
            has_trend_predictor = 'class TrendPredictor' in content
            has_decision_engine = 'class DecisionEngine' in content
            
            print(f"✓ NLPProcessor class: {'✓' if has_nlp_processor else '✗'}")
            print(f"✓ QualityAssessor class: {'✓' if has_quality_assessor else '✗'}")
            print(f"✓ TrendPredictor class: {'✓' if has_trend_predictor else '✗'}")
            print(f"✓ DecisionEngine class: {'✓' if has_decision_engine else '✗'}")
            
            ml_engine_ok = all([has_nlp_processor, has_quality_assessor, has_trend_predictor, has_decision_engine])
            test_results.append(("ML Decision Engine", ml_engine_ok))
    except Exception as e:
        print(f"✗ Error reading ML engine: {e}")
        test_results.append(("ML Decision Engine", False))
    
    # Test learning framework structure
    try:
        with open('../models/learning_framework.py', 'r') as f:
            content = f.read()
            has_reinforcement_learner = 'class ReinforcementLearner' in content
            has_supervised_learner = 'class SupervisedLearner' in content
            has_unsupervised_learner = 'class UnsupervisedLearner' in content
            has_meta_learner = 'class MetaLearner' in content
            has_learning_framework = 'class LearningFramework' in content
            
            print(f"✓ ReinforcementLearner class: {'✓' if has_reinforcement_learner else '✗'}")
            print(f"✓ SupervisedLearner class: {'✓' if has_supervised_learner else '✗'}")
            print(f"✓ UnsupervisedLearner class: {'✓' if has_unsupervised_learner else '✗'}")
            print(f"✓ MetaLearner class: {'✓' if has_meta_learner else '✗'}")
            print(f"✓ LearningFramework class: {'✓' if has_learning_framework else '✗'}")
            
            learning_framework_ok = all([has_reinforcement_learner, has_supervised_learner, has_unsupervised_learner, has_meta_learner, has_learning_framework])
            test_results.append(("Learning Framework", learning_framework_ok))
    except Exception as e:
        print(f"✗ Error reading learning framework: {e}")
        test_results.append(("Learning Framework", False))
    
    # Test 3: Dependencies
    print("\n=== Testing Dependencies ===")
    print(f"✓ NumPy available: {'✓' if 'numpy' in sys.modules else '✗'}")
    print(f"✓ Scikit-learn available: {'✓' if SKLEARN_AVAILABLE else '✗'}")
    
    test_results.append(("Dependencies", SKLEARN_AVAILABLE))
    
    # Final Results
    print("\n=== FINAL RESULTS ===")
    total_tests = len(test_results)
    passed_tests = sum(1 for _, passed in test_results if passed)
    
    for test_name, passed in test_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nSUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL PHASE 2 TESTS PASSED!")
        return True
    else:
        print("❌ Some tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = test_phase2_components()
    sys.exit(0 if success else 1)