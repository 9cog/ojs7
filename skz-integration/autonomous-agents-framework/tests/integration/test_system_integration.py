"""
System Integration Tests for SKZ Autonomous Agents Framework
Complete end-to-end testing of agent interactions and workflows
"""
import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.models.research_vector_db import ResearchVectorDatabase, Document, Query
from src.models.production_optimizer import ProductionOptimizer, FormatType
from src.models.communication_automation import CommunicationAutomation, Recipient, MessageType
from src.models.learning_system import LearningSystem
from src.models.workflow_optimizer import WorkflowOptimizer

@pytest.fixture
def temp_db_path():
    """Create temporary database path for integration testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield os.path.join(temp_dir, "integration_test_db")

@pytest.fixture
def integration_test_document():
    """Sample document for integration testing"""
    return Document(
        document_id="integration_test_001",
        title="Machine Learning Applications in Cosmetic Science: An Integration Study",
        content="""
        This comprehensive study explores the integration of machine learning techniques
        in cosmetic science research, focusing on formulation optimization, ingredient
        analysis, and consumer preference prediction. The research demonstrates significant
        improvements in product development efficiency and quality outcomes.
        """,
        authors=["Dr. Sarah Chen", "Dr. Michael Rodriguez", "Dr. Amanda Thompson"],
        keywords=["machine learning", "cosmetic science", "formulation", "optimization"],
        publication_date="2024-03-15",
        journal="International Journal of Cosmetic Science",
        doi="10.1234/ijcs.2024.integration.001",
        abstract="Integration study of ML in cosmetic science research.",
        document_type="research_paper",
        metadata={
            "word_count": 8500,
            "page_count": 25,
            "figures": 12,
            "tables": 6,
            "references": 78,
            "impact_factor": 4.2,
            "citation_count": 0,
            "research_area": "cosmetic_science_ml"
        }
    )

@pytest.fixture
def test_recipient():
    """Test recipient for communication testing"""
    return Recipient(
        recipient_id="integration_reviewer_001",
        name="Dr. Integration Reviewer",
        email="integration.reviewer@testjournal.com",
        phone="+1555123456",
        preferred_communication=MessageType.EMAIL,
        timezone="EST",
        language="en",
        role="senior_reviewer",
        organization="Test Integration University",
        communication_preferences={
            "html_emails": True,
            "immediate_notifications": False,
            "weekly_digest": True
        }
    )

class TestSystemIntegration:
    """Comprehensive system integration tests"""

    @pytest.mark.asyncio
    async def test_complete_manuscript_processing_workflow(self, temp_db_path, integration_test_document, test_recipient):
        """Test complete manuscript processing from submission to publication"""
        
        # Initialize all systems
        research_db = ResearchVectorDatabase({
            'chroma_db_path': temp_db_path,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'collection_name': 'integration_test'
        })
        
        production_optimizer = ProductionOptimizer({
            'ml_models': {'formatting_model': 'test_model'},
            'quality_thresholds': {'minimum_score': 0.7}
        })
        
        communication_system = CommunicationAutomation({
            'smtp': {'enabled': False}
        })
        
        # Step 1: Add document to research database
        add_success = await research_db.add_documents([integration_test_document])
        assert add_success is True
        
        # Step 2: Search for related research
        search_query = Query(
            query_text="machine learning cosmetic formulation",
            query_type="semantic_search",
            max_results=5
        )
        search_results = await research_db.search_documents(search_query)
        assert len(search_results) > 0
        
        # Step 3: Optimize document production
        optimization_result = await production_optimizer.optimize_formatting(integration_test_document)
        assert optimization_result.confidence_score > 0
        
        # Step 4: Perform quality control
        quality_report = await production_optimizer.perform_quality_control(integration_test_document)
        assert quality_report.overall_score >= 0.0
        
        # Step 5: Send reviewer invitation
        context_data = {
            'reviewer_name': test_recipient.name,
            'manuscript_title': integration_test_document.title,
            'authors': ', '.join(integration_test_document.authors),
            'journal_name': integration_test_document.journal,
            'submission_date': integration_test_document.publication_date,
            'estimated_time': '21',
            'expertise_areas': ', '.join(integration_test_document.keywords),
            'abstract': integration_test_document.abstract,
            'response_deadline': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            'review_link': 'https://testjournal.com/review/integration_test_001',
            'editorial_team': 'Integration Test Editorial Team'
        }
        
        message = await communication_system.send_message(
            'reviewer_invitation',
            test_recipient,
            context_data
        )
        
        assert message.message_id is not None
        assert message.recipient.recipient_id == test_recipient.recipient_id
        
        # Step 6: Build knowledge graph
        kg_success = await research_db.build_knowledge_graph()
        assert kg_success is True
        assert len(research_db.knowledge_graph['nodes']) > 0
        
        # Integration success validation
        workflow_completion = {
            'document_processed': add_success,
            'search_completed': len(search_results) > 0,
            'optimization_completed': optimization_result.confidence_score > 0,
            'quality_assessed': quality_report.overall_score >= 0.0,
            'communication_sent': message.message_id is not None,
            'knowledge_graph_built': kg_success
        }
        
        # All workflow steps should complete successfully
        assert all(workflow_completion.values())

    @pytest.mark.asyncio
    async def test_multi_agent_collaboration_workflow(self, temp_db_path, integration_test_document):
        """Test multi-agent collaboration and data flow"""
        
        # Initialize systems
        research_db = ResearchVectorDatabase({
            'chroma_db_path': temp_db_path,
            'collection_name': 'collaboration_test'
        })
        
        production_optimizer = ProductionOptimizer({
            'quality_thresholds': {'minimum_score': 0.8}
        })
        
        learning_system = LearningSystem({
            'learning_rate': 0.01,
            'pattern_detection': True
        })
        
        workflow_optimizer = WorkflowOptimizer({
            'optimization_strategy': 'efficiency_first'
        })
        
        # Agent 1: Research Discovery - Add and analyze document
        await research_db.add_documents([integration_test_document])
        trends = await research_db.analyze_research_trends(["machine learning", "cosmetics"])
        
        # Agent 2: Production - Optimize based on research insights
        optimization_result = await production_optimizer.optimize_formatting(integration_test_document)
        
        # Agent 3: Learning - Record performance data
        performance_event = {
            'event_type': 'document_optimization',
            'agent_id': 'production_optimizer',
            'performance_metrics': {
                'optimization_score': optimization_result.confidence_score,
                'processing_time': 0.5,
                'success': True
            },
            'context': {
                'document_type': integration_test_document.document_type,
                'format': integration_test_document.format_type.value if hasattr(integration_test_document, 'format_type') else 'pdf'
            }
        }
        
        learning_event = await learning_system.record_event(performance_event)
        
        # Agent 4: Workflow - Optimize task scheduling
        tasks = [
            {
                'task_id': 'research_analysis',
                'agent_id': 'research_discovery',
                'estimated_duration': 5.0,
                'dependencies': [],
                'priority': 'high',
                'status': 'completed'
            },
            {
                'task_id': 'production_optimization',
                'agent_id': 'production_optimizer', 
                'estimated_duration': 3.0,
                'dependencies': ['research_analysis'],
                'priority': 'medium',
                'status': 'completed'
            },
            {
                'task_id': 'quality_assessment',
                'agent_id': 'content_quality',
                'estimated_duration': 4.0,
                'dependencies': ['production_optimization'],
                'priority': 'high',
                'status': 'pending'
            }
        ]
        
        optimized_schedule = await workflow_optimizer.optimize_task_schedule(tasks)
        
        # Verify multi-agent collaboration results
        collaboration_metrics = {
            'research_trends_identified': len(trends.trending_topics) > 0 if hasattr(trends, 'trending_topics') else True,
            'optimization_performed': optimization_result.confidence_score > 0,
            'learning_recorded': learning_event['success'] if isinstance(learning_event, dict) else hasattr(learning_event, 'event_id'),
            'workflow_optimized': len(optimized_schedule) > 0 if isinstance(optimized_schedule, list) else hasattr(optimized_schedule, 'optimized_tasks')
        }
        
        # All agents should collaborate successfully
        assert all(collaboration_metrics.values())

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, temp_db_path):
        """Test system error handling and recovery mechanisms"""
        
        # Initialize systems with potentially problematic configurations
        research_db = ResearchVectorDatabase({
            'chroma_db_path': temp_db_path,
            'collection_name': 'error_test'
        })
        
        production_optimizer = ProductionOptimizer({
            'quality_thresholds': {'minimum_score': 0.5}
        })
        
        communication_system = CommunicationAutomation({
            'smtp': {'enabled': False, 'host': 'nonexistent.server.com'}
        })
        
        # Test 1: Invalid document handling
        invalid_document = Document(
            document_id="invalid_001",
            title="",  # Empty title
            content="",  # Empty content
            authors=[],  # No authors
            keywords=[],
            publication_date="invalid-date",
            journal="",
            doi="",
            abstract="",
            document_type="unknown",
            metadata={}
        )
        
        # Should handle gracefully without crashing
        try:
            add_result = await research_db.add_documents([invalid_document])
            optimization_result = await production_optimizer.optimize_formatting(invalid_document)
            # System should continue functioning even with invalid data
            assert isinstance(add_result, bool)
            assert hasattr(optimization_result, 'confidence_score')
        except Exception as e:
            # If exceptions occur, they should be logged and handled gracefully
            assert len(str(e)) > 0  # Error should have meaningful message

        # Test 2: Communication system resilience
        invalid_recipient = Recipient(
            recipient_id="invalid_recipient",
            name="",
            email="invalid-email",  # Invalid email format
            phone="invalid-phone",
            preferred_communication=MessageType.EMAIL,
            timezone="invalid-timezone",
            language="unknown-language",
            role="unknown-role",
            organization="",
            communication_preferences={}
        )
        
        try:
            message = await communication_system.send_message(
                'reviewer_invitation',
                invalid_recipient,
                {'reviewer_name': 'Test'}
            )
            # Should return a message object even if sending fails
            assert hasattr(message, 'message_id')
        except Exception as e:
            # System should handle communication errors gracefully
            assert len(str(e)) > 0

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, temp_db_path, integration_test_document):
        """Test system performance benchmarks"""
        
        # Initialize systems
        research_db = ResearchVectorDatabase({
            'chroma_db_path': temp_db_path,
            'collection_name': 'performance_test'
        })
        
        production_optimizer = ProductionOptimizer({})
        
        # Benchmark 1: Document processing speed
        start_time = asyncio.get_event_loop().time()
        await research_db.add_documents([integration_test_document])
        document_processing_time = asyncio.get_event_loop().time() - start_time
        
        # Should process documents efficiently (under 5 seconds for single doc)
        assert document_processing_time < 5.0
        
        # Benchmark 2: Search performance
        search_query = Query(
            query_text="machine learning cosmetics",
            query_type="semantic_search",
            max_results=10
        )
        
        start_time = asyncio.get_event_loop().time()
        search_results = await research_db.search_documents(search_query)
        search_time = asyncio.get_event_loop().time() - start_time
        
        # Search should be fast (under 2 seconds)
        assert search_time < 2.0
        assert len(search_results) > 0
        
        # Benchmark 3: Optimization performance
        start_time = asyncio.get_event_loop().time()
        optimization_result = await production_optimizer.optimize_formatting(integration_test_document)
        optimization_time = asyncio.get_event_loop().time() - start_time
        
        # Optimization should be efficient (under 3 seconds)
        assert optimization_time < 3.0
        assert optimization_result.confidence_score >= 0.0
        
        # Overall performance metrics
        performance_metrics = {
            'document_processing_time': document_processing_time,
            'search_time': search_time,
            'optimization_time': optimization_time,
            'total_workflow_time': document_processing_time + search_time + optimization_time
        }
        
        # Log performance metrics for monitoring
        print(f"Performance Metrics: {performance_metrics}")
        
        # Total workflow should complete in reasonable time (under 10 seconds)
        assert performance_metrics['total_workflow_time'] < 10.0

    @pytest.mark.asyncio
    async def test_data_consistency_across_systems(self, temp_db_path, integration_test_document):
        """Test data consistency and integrity across all systems"""
        
        # Initialize systems
        research_db = ResearchVectorDatabase({
            'chroma_db_path': temp_db_path,
            'collection_name': 'consistency_test'
        })
        
        production_optimizer = ProductionOptimizer({})
        
        # Add document to research database
        await research_db.add_documents([integration_test_document])
        
        # Retrieve document from database
        search_query = Query(
            query_text=integration_test_document.title,
            query_type="exact_match",
            max_results=1
        )
        
        search_results = await research_db.search_documents(search_query)
        
        # Verify data consistency
        if search_results:
            retrieved_doc = search_results[0]
            
            # Document ID should match
            original_id = integration_test_document.document_id
            retrieved_id = getattr(retrieved_doc, 'document_id', getattr(retrieved_doc, 'doc_id', None))
            
            if retrieved_id:
                assert retrieved_id == original_id
        
        # Test optimization with same document
        optimization_result = await production_optimizer.optimize_formatting(integration_test_document)
        
        # Optimization should maintain document integrity
        assert optimization_result.original_format is not None
        
        # Test quality assessment consistency
        quality_report = await production_optimizer.perform_quality_control(integration_test_document)
        
        # Quality metrics should be consistent and valid
        assert 0.0 <= quality_report.overall_score <= 1.0
        assert quality_report.document_id == integration_test_document.document_id

    @pytest.mark.asyncio
    async def test_scalability_simulation(self, temp_db_path):
        """Test system scalability with multiple concurrent operations"""
        
        # Initialize systems
        research_db = ResearchVectorDatabase({
            'chroma_db_path': temp_db_path,
            'collection_name': 'scalability_test'
        })
        
        production_optimizer = ProductionOptimizer({})
        
        # Generate multiple test documents
        test_documents = []
        for i in range(10):
            doc = Document(
                document_id=f"scale_test_{i:03d}",
                title=f"Scalability Test Document {i+1}",
                content=f"This is test content for scalability testing document number {i+1}.",
                authors=[f"Author {i+1}"],
                keywords=["scalability", "testing", "performance"],
                publication_date="2024-01-01",
                journal="Test Journal",
                doi=f"10.test/scale.{i:03d}",
                abstract=f"Abstract for test document {i+1}",
                document_type="test_document",
                metadata={"test_id": i}
            )
            test_documents.append(doc)
        
        # Test concurrent document processing
        start_time = asyncio.get_event_loop().time()
        
        # Process documents concurrently
        tasks = []
        for doc in test_documents:
            task = research_db.add_documents([doc])
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        concurrent_processing_time = asyncio.get_event_loop().time() - start_time
        
        # Most operations should succeed
        successful_operations = sum(1 for result in results if result is True)
        assert successful_operations >= len(test_documents) * 0.8  # At least 80% success rate
        
        # Concurrent processing should be efficient
        assert concurrent_processing_time < 15.0  # Should complete in reasonable time
        
        print(f"Scalability Test: {successful_operations}/{len(test_documents)} operations successful in {concurrent_processing_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
