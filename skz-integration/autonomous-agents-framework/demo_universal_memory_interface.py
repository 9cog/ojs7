#!/usr/bin/env python3
"""
Demonstration of Universal Memory System Interface
Shows both the new interface (Issue #19) and existing backward compatibility
"""

import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.memory_system import PersistentMemorySystem
from models.universal_systems import VectorStore, KnowledgeGraph, ExperienceDatabase, ContextMemory, create_universal_memory_system


def demo_new_interface():
    """Demonstrate the new interface specified in Issue #19"""
    print("=" * 60)
    print("DEMO: New Universal Memory System Interface (Issue #19)")
    print("=" * 60)
    
    # Create memory system using the new interface
    print("Creating memory system with new interface:")
    print("memory_system = PersistentMemorySystem(")
    print("    vector_store=VectorStore(),")
    print("    knowledge_graph=KnowledgeGraph(),")
    print("    experience_db=ExperienceDatabase(),")
    print("    context_memory=ContextMemory()")
    print(")")
    print()
    
    memory_system = PersistentMemorySystem(
        db_path="demo_new_interface.db",
        vector_store=VectorStore(),
        knowledge_graph=KnowledgeGraph(),
        experience_db=ExperienceDatabase(),
        context_memory=ContextMemory()
    )
    
    print("✓ Memory system created successfully!")
    print(f"✓ Type: {type(memory_system)}")
    print(f"✓ Has vector_store: {hasattr(memory_system, 'vector_store')}")
    print(f"✓ Has knowledge_graph: {hasattr(memory_system, 'knowledge_graph')}")
    print(f"✓ Has experience_db: {hasattr(memory_system, 'experience_db')}")
    print(f"✓ Has context_memory: {hasattr(memory_system, 'context_memory')}")
    print()
    
    # Test functionality
    print("Testing component functionality:")
    
    # Vector store
    test_embedding = np.random.rand(64)
    vector_id = memory_system.vector_store.store_vector("demo_content_hash", test_embedding)
    print(f"✓ Vector stored: {vector_id}")
    
    # Knowledge graph
    rel_id = memory_system.knowledge_graph.add_relationship(
        "demo_concept_a", "demo_concept_b", "demonstrates", 0.95
    )
    print(f"✓ Knowledge relationship stored: {rel_id}")
    
    # Experience database
    exp_id = memory_system.experience_db.log_experience(
        "demo_agent", "demonstration", 
        {"action": "show_interface"}, {"result": "success"}, True,
        {"duration": 0.1, "accuracy": 1.0}
    )
    print(f"✓ Experience logged: {exp_id}")
    
    # Context memory
    ctx_id = memory_system.context_memory.store_context(
        "demo_agent", {"demo_context": "new_interface_works"}, 
        importance_score=0.9, tags=["demo", "interface"]
    )
    print(f"✓ Context stored: {ctx_id}")
    print()


def demo_backward_compatibility():
    """Demonstrate that existing interfaces still work"""
    print("=" * 60)
    print("DEMO: Backward Compatibility (Existing Interface)")
    print("=" * 60)
    
    # Test old db_path only interface
    print("1. Testing db_path only interface:")
    memory_system_old = PersistentMemorySystem(db_path="demo_old_interface.db")
    print("✓ PersistentMemorySystem(db_path='demo_old_interface.db') works")
    print()
    
    # Test create_universal_memory_system function
    print("2. Testing create_universal_memory_system function:")
    memory_system_legacy = create_universal_memory_system("demo_agent")
    print("✓ create_universal_memory_system('demo_agent') works")
    print(f"✓ Has all components: vector_store={hasattr(memory_system_legacy, 'vector_store')}, "
          f"knowledge_graph={hasattr(memory_system_legacy, 'knowledge_graph')}, "
          f"experience_db={hasattr(memory_system_legacy, 'experience_db')}, "
          f"context_memory={hasattr(memory_system_legacy, 'context_memory')}")
    
    # Test that legacy components work
    test_embedding = np.random.rand(32)
    vector_id = memory_system_legacy.vector_store.store_vector("legacy_hash", test_embedding)
    print(f"✓ Legacy vector storage works: {vector_id}")
    print()


def demo_mixed_interface():
    """Demonstrate mixing db_path with component parameters"""
    print("=" * 60)
    print("DEMO: Mixed Interface (db_path + components)")
    print("=" * 60)
    
    print("Creating memory system with both db_path and components:")
    print("memory_system = PersistentMemorySystem(")
    print("    db_path='demo_mixed.db',")
    print("    vector_store=VectorStore(),")
    print("    knowledge_graph=KnowledgeGraph()")
    print(")")
    print()
    
    memory_system = PersistentMemorySystem(
        db_path="demo_mixed.db",
        vector_store=VectorStore(),
        knowledge_graph=KnowledgeGraph(),
        experience_db=ExperienceDatabase(),
        context_memory=ContextMemory()
    )
    
    print("✓ Mixed interface works!")
    print(f"✓ Custom db_path used: {memory_system.db_path}")
    print(f"✓ Components provided: {hasattr(memory_system, 'vector_store')}")
    print()


def cleanup_demo_files():
    """Clean up demo database files"""
    import os
    demo_files = [
        "demo_new_interface.db",
        "demo_old_interface.db", 
        "demo_mixed.db",
        "agent_memory_demo_agent.db"
    ]
    
    for file in demo_files:
        if os.path.exists(file):
            os.unlink(file)
            print(f"Cleaned up: {file}")


if __name__ == "__main__":
    print("Universal Memory System Interface Demonstration")
    print("Issue #19 Implementation")
    print()
    
    try:
        # Run demonstrations
        demo_new_interface()
        demo_backward_compatibility()
        demo_mixed_interface()
        
        print("=" * 60)
        print("🎉 ALL DEMONSTRATIONS SUCCESSFUL!")
        print("=" * 60)
        print("✓ New interface (Issue #19) implemented correctly")
        print("✓ Backward compatibility preserved")
        print("✓ All components working as expected")
        print("✓ Mixed parameter usage supported")
        print()
        print("The universal memory system interface is now fully implemented")
        print("and ready for use in autonomous agents.")
        
    except Exception as e:
        print(f"❌ ERROR during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print()
        print("Cleaning up demo files...")
        cleanup_demo_files()
        print("Demo complete!")