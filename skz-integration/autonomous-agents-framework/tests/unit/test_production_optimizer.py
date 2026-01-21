"""
Unit tests for Production Optimization System
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.models.production_optimizer import (
    ProductionOptimizer,
    Document,
    OptimizationResult,
    QualityCheck,
    QualityReport,
    PublicationPrediction,
    FormatType
)

@pytest.fixture
def sample_document():
    """Sample document for testing"""
    return Document(
        document_id="test_doc_001",
        title="Advanced Cosmetic Formulation Techniques",
        content="This document discusses various formulation techniques...",
        authors=["Dr. Alice Smith", "Dr. Bob Johnson"],
        format_type=FormatType.PDF,
        metadata={
            "word_count": 5000,
            "page_count": 15,
            "figures": 8,
            "tables": 3,
            "references": 45,
            "submission_date": "2024-01-15",
            "journal": "Journal of Cosmetic Science"
        }
    )

@pytest.fixture
def production_optimizer():
    """Initialize production optimizer"""
    config = {
        'ml_models': {
            'formatting_model': 'test_formatting_model',
            'quality_model': 'test_quality_model',
            'success_prediction_model': 'test_prediction_model'
        },
        'quality_thresholds': {
            'minimum_score': 0.7,
            'warning_threshold': 0.8,
            'excellent_threshold': 0.9
        },
        'format_rules': {
            'pdf': {'max_pages': 50, 'min_dpi': 300},
            'html': {'max_file_size': '10MB', 'responsive': True},
            'xml': {'schema_validation': True, 'encoding': 'UTF-8'}
        }
    }
    return ProductionOptimizer(config)

class TestProductionOptimizer:
    """Test ProductionOptimizer class"""

    def test_initialization(self, production_optimizer):
        """Test production optimizer initialization"""
        assert production_optimizer.config is not None
        assert 'ml_models' in production_optimizer.config
        assert 'quality_thresholds' in production_optimizer.config

    @pytest.mark.asyncio
    async def test_optimize_formatting(self, production_optimizer, sample_document):
        """Test document formatting optimization"""
        # Mock ML model prediction
        with patch.object(production_optimizer, '_predict_optimal_format', return_value={
            'recommended_format': FormatType.PDF,
            'confidence': 0.85,
            'optimization_rules': ['increase_figure_quality', 'standardize_citations']
        }):
            result = await production_optimizer.optimize_formatting(sample_document)
            
            assert isinstance(result, OptimizationResult)
            assert result.original_format == FormatType.PDF
            assert result.optimized_format in [FormatType.PDF, FormatType.HTML, FormatType.XML]
            assert result.confidence_score >= 0.0
            assert len(result.applied_optimizations) > 0

    @pytest.mark.asyncio
    async def test_quality_control_checks(self, production_optimizer, sample_document):
        """Test quality control functionality"""
        quality_report = await production_optimizer.perform_quality_control(sample_document)
        
        assert isinstance(quality_report, QualityReport)
        assert quality_report.overall_score >= 0.0
        assert quality_report.overall_score <= 1.0
        assert len(quality_report.checks) > 0
        
        # Check that all required quality checks are present
        check_types = [check.check_type for check in quality_report.checks]
        expected_checks = ['metadata_completeness', 'format_consistency', 'content_quality']
        
        for expected_check in expected_checks:
            assert expected_check in check_types

    @pytest.mark.asyncio
    async def test_metadata_completeness_check(self, production_optimizer, sample_document):
        """Test metadata completeness validation"""
        result = await production_optimizer._check_metadata_completeness(sample_document)
        
        assert isinstance(result, QualityCheck)
        assert result.check_type == 'metadata_completeness'
        assert result.score >= 0.0
        assert result.score <= 1.0
        assert len(result.details) > 0

    @pytest.mark.asyncio
    async def test_format_consistency_check(self, production_optimizer, sample_document):
        """Test format consistency validation"""
        result = await production_optimizer._check_format_consistency(sample_document)
        
        assert isinstance(result, QualityCheck)
        assert result.check_type == 'format_consistency'
        assert result.score >= 0.0
        assert result.score <= 1.0

    @pytest.mark.asyncio
    async def test_content_quality_check(self, production_optimizer, sample_document):
        """Test content quality assessment"""
        result = await production_optimizer._check_content_quality(sample_document)
        
        assert isinstance(result, QualityCheck)
        assert result.check_type == 'content_quality'
        assert result.score >= 0.0
        assert result.score <= 1.0

    @pytest.mark.asyncio
    async def test_publication_success_prediction(self, production_optimizer, sample_document):
        """Test publication success prediction"""
        # Mock quality report
        mock_quality_report = QualityReport(
            document_id=sample_document.document_id,
            overall_score=0.85,
            checks=[],
            recommendations=[],
            analysis_date=datetime.now().isoformat()
        )
        
        with patch.object(production_optimizer, 'perform_quality_control', return_value=mock_quality_report):
            prediction = await production_optimizer.predict_publication_success(sample_document)
            
            assert isinstance(prediction, PublicationPrediction)
            assert prediction.success_probability >= 0.0
            assert prediction.success_probability <= 1.0
            assert prediction.predicted_impact_score >= 0.0
            assert len(prediction.risk_factors) >= 0
            assert len(prediction.optimization_suggestions) >= 0

    @pytest.mark.asyncio
    async def test_format_specific_optimization(self, production_optimizer, sample_document):
        """Test format-specific optimization rules"""
        # Test PDF optimization
        pdf_optimizations = await production_optimizer._apply_format_specific_optimizations(
            sample_document, FormatType.PDF
        )
        assert len(pdf_optimizations) > 0
        
        # Test HTML optimization
        html_optimizations = await production_optimizer._apply_format_specific_optimizations(
            sample_document, FormatType.HTML
        )
        assert len(html_optimizations) > 0

    @pytest.mark.asyncio
    async def test_risk_factor_identification(self, production_optimizer, sample_document):
        """Test risk factor identification"""
        mock_quality_report = QualityReport(
            document_id=sample_document.document_id,
            overall_score=0.65,  # Below warning threshold
            checks=[
                QualityCheck(
                    check_type='metadata_completeness',
                    score=0.6,
                    passed=False,
                    details={'missing_fields': ['doi', 'keywords']},
                    recommendations=['Add DOI', 'Add keywords']
                )
            ],
            recommendations=[],
            analysis_date=datetime.now().isoformat()
        )
        
        risk_factors = await production_optimizer._identify_risk_factors(mock_quality_report)
        
        assert len(risk_factors) > 0
        assert any('metadata' in risk.lower() for risk in risk_factors)

    def test_format_type_enum(self):
        """Test FormatType enum"""
        assert FormatType.PDF.value == 'pdf'
        assert FormatType.HTML.value == 'html'
        assert FormatType.XML.value == 'xml'
        assert FormatType.EPUB.value == 'epub'

    @pytest.mark.asyncio
    async def test_optimization_with_insufficient_data(self, production_optimizer):
        """Test optimization with incomplete document data"""
        incomplete_document = Document(
            document_id="incomplete_001",
            title="Test Document",
            content="Minimal content",
            authors=[],
            format_type=FormatType.PDF,
            metadata={}
        )
        
        result = await production_optimizer.optimize_formatting(incomplete_document)
        
        # Should still return a result, even with limited data
        assert isinstance(result, OptimizationResult)
        assert result.confidence_score >= 0.0

    @pytest.mark.asyncio
    async def test_bulk_optimization(self, production_optimizer, sample_document):
        """Test bulk document optimization"""
        documents = [sample_document] * 3  # Simulate multiple documents
        
        results = await production_optimizer.optimize_bulk_documents(documents)
        
        assert len(results) == 3
        assert all(isinstance(result, OptimizationResult) for result in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
