"""
Test for Universal Memory System Interface
Tests the exact interface specified in Issue #19
"""

import pytest
import tempfile
import os
from src.models.memory_system import PersistentMemorySystem
from src.models.universal_systems import VectorStore, KnowledgeGraph, ExperienceDatabase, ContextMemory


class TestUniversalMemoryInterface:
    """Test the universal memory system interface as specified in the issue"""
    
    def test_universal_memory_system_constructor_interface(self):
        """Test the exact interface specified in Issue #19"""
        
        # Test the exact interface required by the issue
        memory_system = PersistentMemorySystem(
            vector_store=VectorStore(),
            knowledge_graph=KnowledgeGraph(),
            experience_db=ExperienceDatabase(),
            context_memory=ContextMemory()
        )
        
        # Verify the memory system was created correctly
        assert isinstance(memory_system, PersistentMemorySystem)
        assert hasattr(memory_system, 'vector_store')
        assert hasattr(memory_system, 'knowledge_graph')
        assert hasattr(memory_system, 'experience_db')
        assert hasattr(memory_system, 'context_memory')
        
        # Verify components are of correct types
        assert isinstance(memory_system.vector_store, VectorStore)
        assert isinstance(memory_system.knowledge_graph, KnowledgeGraph)
        assert isinstance(memory_system.experience_db, ExperienceDatabase)
        assert isinstance(memory_system.context_memory, ContextMemory)
    
    def test_component_functionality_works(self):
        """Test that components work correctly with the new interface"""
        
        # Create components and memory system as specified in issue
        memory_system = PersistentMemorySystem(
            vector_store=VectorStore(),
            knowledge_graph=KnowledgeGraph(),
            experience_db=ExperienceDatabase(),
            context_memory=ContextMemory()
        )
        
        # Test vector store functionality
        import numpy as np
        test_embedding = np.random.rand(128)
        vector_id = memory_system.vector_store.store_vector("test_hash", test_embedding)
        assert vector_id is not None
        
        # Test knowledge graph functionality
        rel_id = memory_system.knowledge_graph.add_relationship(
            "source_1", "target_1", "related_to", 0.9
        )
        assert rel_id is not None
        
        # Test experience database functionality
        exp_id = memory_system.experience_db.log_experience(
            "test_agent", "test_action", {"input": "test"}, {"output": "result"}, True
        )
        assert exp_id is not None
        
        # Test context memory functionality
        ctx_id = memory_system.context_memory.store_context(
            "test_agent", {"context": "test"}, importance_score=0.8
        )
        assert ctx_id is not None
    
    def test_backward_compatibility(self):
        """Test that existing db_path interface still works"""
        
        # Test that the old interface still works
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            # Old interface should still work
            memory_system = PersistentMemorySystem(db_path=db_path)
            assert isinstance(memory_system, PersistentMemorySystem)
            assert memory_system.db_path == db_path
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_mixed_parameters(self):
        """Test that mixing db_path with components works"""
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            # Test mixed parameters
            memory_system = PersistentMemorySystem(
                db_path=db_path,
                vector_store=VectorStore(),
                knowledge_graph=KnowledgeGraph(),
                experience_db=ExperienceDatabase(),
                context_memory=ContextMemory()
            )
            
            assert isinstance(memory_system, PersistentMemorySystem)
            assert memory_system.db_path == db_path
            assert hasattr(memory_system, 'vector_store')
            assert hasattr(memory_system, 'knowledge_graph')
            assert hasattr(memory_system, 'experience_db')
            assert hasattr(memory_system, 'context_memory')
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__])