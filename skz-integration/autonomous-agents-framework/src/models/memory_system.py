"""
Persistent Memory System for Autonomous Agents
Phase 2 Critical Component - Provides persistent memory for all agents
"""

import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, asdict
import pickle
import threading
import logging

if TYPE_CHECKING:
    # Avoid circular imports by using TYPE_CHECKING
    from .universal_systems import VectorStore, KnowledgeGraph, ExperienceDatabase, ContextMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Represents a single memory entry"""
    id: str
    agent_id: str
    memory_type: str  # 'vector', 'knowledge', 'experience', 'context'
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    accessed_at: datetime
    importance_score: float
    tags: List[str]

@dataclass
class VectorEmbedding:
    """Represents a vector embedding for semantic search"""
    id: str
    content_hash: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    created_at: datetime

class PersistentMemorySystem:
    """
    Comprehensive persistent memory system for autonomous agents
    Provides vector database, knowledge graph, experience database, and context memory
    """
    
    def __init__(self, db_path: str = "agent_memory.db", 
                 vector_store: Optional['VectorStore'] = None,
                 knowledge_graph: Optional['KnowledgeGraph'] = None,
                 experience_db: Optional['ExperienceDatabase'] = None,
                 context_memory: Optional['ContextMemory'] = None):
        self.db_path = db_path
        self.lock = threading.RLock()
        self._init_database()
        
        # Set up component references if provided
        if vector_store is not None:
            self.vector_store = vector_store
            vector_store.memory_system = self
            
        if knowledge_graph is not None:
            self.knowledge_graph = knowledge_graph
            knowledge_graph.memory_system = self
            
        if experience_db is not None:
            self.experience_db = experience_db
            experience_db.memory_system = self
            
        if context_memory is not None:
            self.context_memory = context_memory
            context_memory.memory_system = self
        
    def _init_database(self):
        """Initialize the memory database with all required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    tags TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_embeddings (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT UNIQUE NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_graph (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experience_log (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    output_data TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_agent_type ON memory_entries(agent_id, memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_importance ON memory_entries(importance_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_hash ON vector_embeddings(content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experience_agent ON experience_log(agent_id)")
            
            conn.commit()
    
    def store_memory(self, agent_id: str, memory_type: str, content: Dict[str, Any], 
                    metadata: Dict[str, Any] = None, importance_score: float = 0.5, 
                    tags: List[str] = None) -> str:
        """
        Store a memory entry for an agent
        
        Args:
            agent_id: ID of the agent
            memory_type: Type of memory ('vector', 'knowledge', 'experience', 'context')
            content: The content to store
            metadata: Additional metadata
            importance_score: Importance score (0.0 to 1.0)
            tags: List of tags for categorization
            
        Returns:
            Memory entry ID
        """
        with self.lock:
            memory_id = f"{agent_id}_{memory_type}_{hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()[:8]}"
            
            entry = MemoryEntry(
                id=memory_id,
                agent_id=agent_id,
                memory_type=memory_type,
                content=content,
                metadata=metadata or {},
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                importance_score=max(0.0, min(1.0, importance_score)),
                tags=tags or []
            )
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memory_entries 
                    (id, agent_id, memory_type, content, metadata, created_at, accessed_at, importance_score, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.id,
                    entry.agent_id,
                    entry.memory_type,
                    json.dumps(entry.content),
                    json.dumps(entry.metadata),
                    entry.created_at.isoformat(),
                    entry.accessed_at.isoformat(),
                    entry.importance_score,
                    json.dumps(entry.tags)
                ))
                conn.commit()
            
            logger.info(f"Stored memory entry {memory_id} for agent {agent_id}")
            return memory_id
    
    def retrieve_memory(self, agent_id: str, memory_type: str = None, 
                       query: str = None, limit: int = 10, 
                       min_importance: float = 0.0) -> List[MemoryEntry]:
        """
        Retrieve memory entries for an agent
        
        Args:
            agent_id: ID of the agent
            memory_type: Optional filter by memory type
            query: Optional text query for semantic search
            limit: Maximum number of entries to return
            min_importance: Minimum importance score filter
            
        Returns:
            List of memory entries
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Build query
                sql = "SELECT * FROM memory_entries WHERE agent_id = ? AND importance_score >= ?"
                params = [agent_id, min_importance]
                
                if memory_type:
                    sql += " AND memory_type = ?"
                    params.append(memory_type)
                
                sql += " ORDER BY importance_score DESC, accessed_at DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                
                entries = []
                for row in rows:
                    entry = MemoryEntry(
                        id=row['id'],
                        agent_id=row['agent_id'],
                        memory_type=row['memory_type'],
                        content=json.loads(row['content']),
                        metadata=json.loads(row['metadata']),
                        created_at=datetime.fromisoformat(row['created_at']),
                        accessed_at=datetime.fromisoformat(row['accessed_at']),
                        importance_score=row['importance_score'],
                        tags=json.loads(row['tags'])
                    )
                    entries.append(entry)
                    
                    # Update accessed_at
                    conn.execute("""
                        UPDATE memory_entries SET accessed_at = ? WHERE id = ?
                    """, (datetime.now().isoformat(), entry.id))
                
                conn.commit()
                return entries
    
    def store_vector_embedding(self, content_hash: str, embedding: np.ndarray, 
                             metadata: Dict[str, Any] = None) -> str:
        """
        Store a vector embedding for semantic search
        
        Args:
            content_hash: Hash of the content
            embedding: Vector embedding
            metadata: Additional metadata
            
        Returns:
            Embedding ID
        """
        with self.lock:
            embedding_id = f"vec_{content_hash[:8]}"
            
            vector_entry = VectorEmbedding(
                id=embedding_id,
                content_hash=content_hash,
                embedding=embedding,
                metadata=metadata or {},
                created_at=datetime.now()
            )
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO vector_embeddings 
                    (id, content_hash, embedding, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    vector_entry.id,
                    vector_entry.content_hash,
                    pickle.dumps(vector_entry.embedding),
                    json.dumps(vector_entry.metadata),
                    vector_entry.created_at.isoformat()
                ))
                conn.commit()
            
            logger.info(f"Stored vector embedding {embedding_id}")
            return embedding_id
    
    def find_similar_vectors(self, query_embedding: np.ndarray, limit: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar vectors using cosine similarity
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            
        Returns:
            List of (embedding_id, similarity_score) tuples
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT id, embedding FROM vector_embeddings")
                rows = cursor.fetchall()
                
                similarities = []
                for row in rows:
                    stored_embedding = pickle.loads(row['embedding'])
                    similarity = np.dot(query_embedding, stored_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                    )
                    similarities.append((row['id'], float(similarity)))
                
                # Sort by similarity and return top results
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:limit]
    
    def store_knowledge_relationship(self, source_id: str, target_id: str, 
                                  relationship_type: str, confidence_score: float = 1.0,
                                  metadata: Dict[str, Any] = None) -> str:
        """
        Store a knowledge graph relationship
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Type of relationship
            confidence_score: Confidence in the relationship (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            Relationship ID
        """
        with self.lock:
            relationship_id = f"rel_{source_id}_{target_id}_{relationship_type}"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO knowledge_graph 
                    (id, source_id, target_id, relationship_type, confidence_score, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    relationship_id,
                    source_id,
                    target_id,
                    relationship_type,
                    confidence_score,
                    json.dumps(metadata or {}),
                    datetime.now().isoformat()
                ))
                conn.commit()
            
            logger.info(f"Stored knowledge relationship {relationship_id}")
            return relationship_id
    
    def log_experience(self, agent_id: str, action_type: str, input_data: Dict[str, Any],
                      output_data: Dict[str, Any], success: bool, 
                      performance_metrics: Dict[str, Any] = None) -> str:
        """
        Log an agent experience for learning
        
        Args:
            agent_id: ID of the agent
            action_type: Type of action performed
            input_data: Input data for the action
            output_data: Output data from the action
            success: Whether the action was successful
            performance_metrics: Performance metrics
            
        Returns:
            Experience log ID
        """
        with self.lock:
            import time
            import uuid
            experience_id = f"exp_{agent_id}_{action_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000000) % 1000000}_{uuid.uuid4().hex[:8]}"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO experience_log 
                    (id, agent_id, action_type, input_data, output_data, success, performance_metrics, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experience_id,
                    agent_id,
                    action_type,
                    json.dumps(input_data),
                    json.dumps(output_data),
                    success,
                    json.dumps(performance_metrics or {}),
                    datetime.now().isoformat()
                ))
                conn.commit()
            
            logger.info(f"Logged experience {experience_id} for agent {agent_id}")
            return experience_id
    
    def get_agent_experiences(self, agent_id: str, action_type: str = None, 
                             limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get agent experiences for learning
        
        Args:
            agent_id: ID of the agent
            action_type: Optional filter by action type
            limit: Maximum number of experiences to return
            
        Returns:
            List of experience dictionaries
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                sql = "SELECT * FROM experience_log WHERE agent_id = ?"
                params = [agent_id]
                
                if action_type:
                    sql += " AND action_type = ?"
                    params.append(action_type)
                
                sql += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                
                experiences = []
                for row in rows:
                    experience = {
                        'id': row['id'],
                        'agent_id': row['agent_id'],
                        'action_type': row['action_type'],
                        'input_data': json.loads(row['input_data']),
                        'output_data': json.loads(row['output_data']),
                        'success': bool(row['success']),
                        'performance_metrics': json.loads(row['performance_metrics']),
                        'created_at': datetime.fromisoformat(row['created_at'])
                    }
                    experiences.append(experience)
                
                return experiences
    
    def cleanup_old_memories(self, days_old: int = 30, min_importance: float = 0.3):
        """
        Clean up old, low-importance memories
        
        Args:
            days_old: Age threshold for cleanup
            min_importance: Minimum importance to keep
        """
        with self.lock:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            with sqlite3.connect(self.db_path) as conn:
                # Clean up old memory entries
                conn.execute("""
                    DELETE FROM memory_entries 
                    WHERE created_at < ? AND importance_score < ?
                """, (cutoff_date.isoformat(), min_importance))
                
                # Clean up old vector embeddings
                conn.execute("""
                    DELETE FROM vector_embeddings 
                    WHERE created_at < ?
                """, (cutoff_date.isoformat(),))
                
                # Clean up old experience logs (keep more recent ones)
                experience_cutoff = datetime.now() - timedelta(days=days_old * 2)
                conn.execute("""
                    DELETE FROM experience_log 
                    WHERE created_at < ?
                """, (experience_cutoff.isoformat(),))
                
                conn.commit()
            
            logger.info(f"Cleaned up memories older than {days_old} days")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                # Memory entries stats
                cursor = conn.execute("SELECT COUNT(*) FROM memory_entries")
                total_memories = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM vector_embeddings")
                total_vectors = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM knowledge_graph")
                total_relationships = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM experience_log")
                total_experiences = cursor.fetchone()[0]
                
                # Memory by type
                cursor = conn.execute("""
                    SELECT memory_type, COUNT(*) as count 
                    FROM memory_entries 
                    GROUP BY memory_type
                """)
                memory_by_type = dict(cursor.fetchall())
                
                return {
                    'total_memories': total_memories,
                    'total_vectors': total_vectors,
                    'total_relationships': total_relationships,
                    'total_experiences': total_experiences,
                    'memory_by_type': memory_by_type
                }