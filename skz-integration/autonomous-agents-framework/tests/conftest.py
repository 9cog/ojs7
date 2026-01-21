"""
Test configuration and fixtures for the SKZ Agents Framework
"""
import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock
from datetime import datetime
import tempfile
import os

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    mock_db = Mock()
    mock_db.execute = AsyncMock(return_value=Mock())
    mock_db.fetchone = AsyncMock(return_value=None)
    mock_db.fetchall = AsyncMock(return_value=[])
    mock_db.commit = AsyncMock()
    mock_db.rollback = AsyncMock()
    return mock_db

@pytest.fixture
def mock_redis():
    """Mock Redis connection for testing."""
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.exists = AsyncMock(return_value=False)
    return mock_redis

@pytest.fixture
def sample_manuscript_data():
    """Sample manuscript data for testing."""
    return {
        'id': 1,
        'title': 'Advanced Cosmetic Formulation with Novel Active Ingredients',
        'abstract': 'This study investigates the efficacy of novel active ingredients in cosmetic formulations for enhanced skin barrier function and anti-aging properties.',
        'authors': [
            {'name': 'Dr. Jane Smith', 'affiliation': 'University of Cosmetic Science'},
            {'name': 'Prof. John Doe', 'affiliation': 'Institute of Dermatology'}
        ],
        'keywords': ['cosmetics', 'skincare', 'active ingredients', 'anti-aging', 'skin barrier'],
        'content': 'Full manuscript content would be here...',
        'submission_date': '2024-01-15T10:00:00Z',
        'status': 'submitted',
        'journal_id': 'skin-zone-journal'
    }

@pytest.fixture
def sample_reviewer_data():
    """Sample reviewer data for testing."""
    return [
        {
            'id': 1,
            'name': 'Dr. Sarah Wilson',
            'expertise': ['cosmetic chemistry', 'formulation science'],
            'current_workload': 2,
            'max_workload': 5,
            'avg_review_time': 14,  # days
            'quality_score': 4.5
        },
        {
            'id': 2,
            'name': 'Prof. Michael Johnson',
            'expertise': ['dermatology', 'skin barrier function'],
            'current_workload': 1,
            'max_workload': 3,
            'avg_review_time': 10,
            'quality_score': 4.8
        }
    ]

@pytest.fixture
def sample_agent_memory():
    """Sample agent memory data for testing."""
    return {
        'agent_id': 'research_discovery_001',
        'memory_type': 'vector',
        'content': 'Previous research on retinol formulations showed improved stability with encapsulation technology.',
        'importance_score': 0.85,
        'created_at': '2024-01-10T15:30:00Z',
        'context': {
            'manuscript_id': 1,
            'research_domain': 'cosmetic_chemistry',
            'confidence': 0.92
        }
    }

@pytest.fixture
def sample_decision_data():
    """Sample agent decision data for testing."""
    return {
        'agent_id': 'editorial_orchestration_001',
        'decision_type': 'manuscript_assignment',
        'context_data': {
            'manuscript_id': 1,
            'available_reviewers': [1, 2, 3],
            'deadline': '2024-02-15T23:59:59Z'
        },
        'decision_result': {
            'assigned_reviewers': [1, 2],
            'priority': 'high',
            'estimated_completion': '2024-02-10T00:00:00Z'
        },
        'confidence_score': 0.89,
        'created_at': '2024-01-16T09:00:00Z'
    }

@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing."""
    config_data = {
        'database': {
            'host': 'localhost',
            'port': 3306,
            'name': 'test_ojs',
            'user': 'test_user',
            'password': 'test_password'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 1
        },
        'agents': {
            'research_discovery': {'port': 5001, 'max_workers': 4},
            'submission_assistant': {'port': 5002, 'max_workers': 4},
            'editorial_orchestration': {'port': 5003, 'max_workers': 2},
            'review_coordination': {'port': 5004, 'max_workers': 3},
            'content_quality': {'port': 5005, 'max_workers': 3},
            'publishing_production': {'port': 5006, 'max_workers': 2},
            'analytics_monitoring': {'port': 5007, 'max_workers': 2}
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    os.unlink(temp_file)

@pytest.fixture
def mock_ml_model():
    """Mock ML model for testing."""
    mock_model = Mock()
    mock_model.predict = Mock(return_value=[0.75])
    mock_model.predict_proba = Mock(return_value=[[0.25, 0.75]])
    mock_model.fit = Mock()
    mock_model.score = Mock(return_value=0.85)
    return mock_model

@pytest.fixture
def mock_ojs_api():
    """Mock OJS API responses for testing."""
    mock_api = Mock()
    mock_api.get_manuscript = AsyncMock(return_value={'id': 1, 'status': 'submitted'})
    mock_api.update_manuscript = AsyncMock(return_value={'success': True})
    mock_api.get_reviewers = AsyncMock(return_value=[{'id': 1, 'name': 'Test Reviewer'}])
    mock_api.assign_reviewer = AsyncMock(return_value={'success': True})
    return mock_api

class AsyncContextManagerMock:
    """Mock async context manager for testing."""
    def __init__(self, return_value):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, *args):
        pass

@pytest.fixture
def async_context_mock():
    """Factory for creating async context manager mocks."""
    return AsyncContextManagerMock
