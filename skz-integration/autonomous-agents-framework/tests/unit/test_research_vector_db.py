"""
Unit tests for Research Vector Database System
"""
import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.models.research_vector_db import (
    ResearchVectorDatabase,
    Document,
    Query,
    SearchResult,
    TrendAnalysis,
    KnowledgeGraphNode,
    KnowledgeGraphEdge
)

@pytest.fixture
def temp_db_path():
    """Create temporary database path"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield os.path.join(temp_dir, "test_research_db")

@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        Document(
            document_id="doc1",
            title="Machine Learning in Cosmetics",
            content="Advanced machine learning techniques for cosmetic formulation analysis.",
            authors=["Dr. Smith", "Dr. Johnson"],
            keywords=["machine learning", "cosmetics", "formulation"],
            publication_date="2024-01-15",
            journal="Journal of Cosmetic Science",
            doi="10.1234/jcs.2024.001",
            abstract="This paper explores ML applications in cosmetic research.",
            document_type="research_paper",
            metadata={
                "impact_factor": 3.2,
                "citation_count": 15,
                "research_area": "cosmetic_science"
            }
        ),
        Document(
            document_id="doc2",
            title="Natural Ingredients in Skincare",
            content="Comprehensive analysis of natural ingredients and their efficacy in skincare products.",
            authors=["Dr. Brown", "Dr. Davis"],
            keywords=["natural ingredients", "skincare", "efficacy"],
            publication_date="2024-02-10",
            journal="International Journal of Dermatology",
            doi="10.1234/ijd.2024.002",
            abstract="Natural ingredients offer promising benefits for skin health.",
            document_type="research_paper",
            metadata={
                "impact_factor": 4.1,
                "citation_count": 28,
                "research_area": "dermatology"
            }
        )
    ]

@pytest.fixture
def research_db(temp_db_path):
    """Initialize research vector database"""
    config = {
        'chroma_db_path': temp_db_path,
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'collection_name': 'test_research_documents'
    }
    return ResearchVectorDatabase(config)

class TestResearchVectorDatabase:
    """Test ResearchVectorDatabase class"""

    @pytest.mark.asyncio
    async def test_initialization(self, research_db):
        """Test database initialization"""
        assert research_db.config is not None
        assert research_db.collection_name == 'test_research_documents'

    @pytest.mark.asyncio
    async def test_add_documents(self, research_db, sample_documents):
        """Test adding documents to database"""
        # Add documents
        success = await research_db.add_documents(sample_documents)
        assert success is True
        
        # Verify documents were added
        assert len(research_db.documents) == 2
        assert "doc1" in research_db.documents
        assert "doc2" in research_db.documents

    @pytest.mark.asyncio
    async def test_semantic_search(self, research_db, sample_documents):
        """Test semantic search functionality"""
        # Add documents first
        await research_db.add_documents(sample_documents)
        
        # Create search query
        query = Query(
            query_text="machine learning cosmetics",
            query_type="semantic_search",
            filters={"research_area": "cosmetic_science"},
            max_results=5
        )
        
        # Perform search
        results = await research_db.search_documents(query)
        
        # Verify results
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].similarity_score >= 0.0

    @pytest.mark.asyncio
    async def test_knowledge_graph_construction(self, research_db, sample_documents):
        """Test knowledge graph construction"""
        # Add documents first
        await research_db.add_documents(sample_documents)
        
        # Build knowledge graph
        success = await research_db.build_knowledge_graph()
        assert success is True
        
        # Verify knowledge graph components
        assert len(research_db.knowledge_graph['nodes']) > 0
        assert len(research_db.knowledge_graph['edges']) > 0
        
        # Check for author nodes
        author_nodes = [n for n in research_db.knowledge_graph['nodes'] if n.node_type == 'author']
        assert len(author_nodes) > 0
        
        # Check for keyword nodes
        keyword_nodes = [n for n in research_db.knowledge_graph['nodes'] if n.node_type == 'keyword']
        assert len(keyword_nodes) > 0

    @pytest.mark.asyncio
    async def test_trend_analysis(self, research_db, sample_documents):
        """Test trend analysis functionality"""
        # Add documents first
        await research_db.add_documents(sample_documents)
        
        # Perform trend analysis
        trends = await research_db.analyze_research_trends(["machine learning", "natural ingredients"])
        
        # Verify trend analysis
        assert isinstance(trends, TrendAnalysis)
        assert len(trends.trending_topics) > 0
        assert len(trends.emerging_areas) >= 0
        assert trends.analysis_period is not None

    @pytest.mark.asyncio
    async def test_research_gap_identification(self, research_db, sample_documents):
        """Test research gap identification"""
        # Add documents first
        await research_db.add_documents(sample_documents)
        
        # Identify research gaps
        gaps = await research_db.identify_research_gaps("cosmetic science")
        
        # Verify gap identification
        assert len(gaps) >= 0
        if gaps:
            assert "gap_description" in gaps[0]
            assert "opportunity_score" in gaps[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
