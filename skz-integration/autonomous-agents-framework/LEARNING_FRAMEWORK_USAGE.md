# Learning Framework Interface Examples

This document shows how to use the Universal Learning Framework with the new injectable interface (issue #20) while maintaining backward compatibility.

## New Required Interface (Issue #20)

The Learning Framework now supports the exact interface required by the issue:

```python
from models.learning_framework import (
    LearningFramework, 
    ReinforcementLearner, 
    SupervisedLearner, 
    UnsupervisedLearner, 
    MetaLearner
)

# Required: Universal learning framework
learning_framework = LearningFramework(
    reinforcement_learner=ReinforcementLearner("agent_id"),
    supervised_learner=SupervisedLearner("agent_id"),
    unsupervised_learner=UnsupervisedLearner("agent_id"),
    meta_learner=MetaLearner("agent_id"),
    db_path="learning.db"  # Optional
)
```

## Backward Compatibility

Existing code continues to work unchanged:

```python
# Existing interface still works
learning_framework = LearningFramework(
    agent_id="my_agent",
    db_path="learning.db"
)
```

## Flexible Usage Patterns

### Partial Injection
```python
# Inject only specific learners, create others automatically
custom_rl = ReinforcementLearner("my_agent")
learning_framework = LearningFramework(
    agent_id="my_agent",
    reinforcement_learner=custom_rl  # Only RL learner injected
)
```

### Agent ID Auto-Derivation
```python
# Agent ID automatically derived from first learner
learning_framework = LearningFramework(
    reinforcement_learner=ReinforcementLearner("auto_agent"),
    supervised_learner=SupervisedLearner("auto_agent"),
    unsupervised_learner=UnsupervisedLearner("auto_agent"),
    meta_learner=MetaLearner("auto_agent")
)
# learning_framework.agent_id will be "auto_agent"
```

### Default Fallback
```python
# Falls back to "default_agent" if no agent_id provided
learning_framework = LearningFramework(db_path=":memory:")
# learning_framework.agent_id will be "default_agent"
```

## Universal Systems Integration

The existing `create_universal_learning_framework` function continues to work:

```python
from models.universal_systems import create_universal_learning_framework

# Creates framework with wrapper components
learning_framework = create_universal_learning_framework(
    agent_id="universal_agent",
    db_path="universal.db"
)

# Access learners directly or through wrappers
learning_framework.reinforcement_learner  # Direct access
learning_framework.reinforcement_learner_wrapper  # Wrapper access
```

## Usage in Academic Publishing Workflow

Example using the framework for manuscript processing:

```python
# Create learning framework for editorial agent
learning_framework = LearningFramework(
    reinforcement_learner=ReinforcementLearner("editorial_agent"),
    supervised_learner=SupervisedLearner("editorial_agent"),
    unsupervised_learner=UnsupervisedLearner("editorial_agent"),
    meta_learner=MetaLearner("editorial_agent"),
    db_path="editorial_learning.db"
)

# Learn from manuscript review experience
experience_id = learning_framework.learn_from_experience(
    action_type="manuscript_review",
    input_data={
        "manuscript_id": "ms_001",
        "subject_area": "machine_learning",
        "reviewer_expertise": ["deep_learning", "computer_vision"]
    },
    output_data={
        "review_decision": "accept",
        "confidence_score": 0.92,
        "review_duration_days": 12
    },
    success=True,
    performance_metrics={
        "reviewer_satisfaction": 4.8,
        "author_satisfaction": 4.5,
        "processing_efficiency": 0.95
    },
    feedback={
        "review_quality": "excellent",
        "timeliness": "on_schedule"
    }
)

# Get learning-based recommendations for new manuscript
recommendations = learning_framework.get_learning_recommendations({
    "manuscript_id": "ms_002",
    "subject_area": "machine_learning",
    "urgency": "high"
})

# Monitor learning progress
stats = learning_framework.get_learning_stats()
print(f"Learning progress: {stats['success_rate']:.2f} success rate over {stats['total_experiences']} experiences")
```

## Benefits of New Interface

1. **Dependency Injection**: Enables easier testing and customization
2. **Flexibility**: Supports partial injection and various usage patterns  
3. **Backward Compatibility**: Existing code continues to work
4. **Agent ID Management**: Intelligent derivation and fallback mechanisms
5. **Universal Integration**: Works seamlessly with universal systems

The implementation maintains the principle of minimal changes while providing maximum flexibility.