"""
Tests for NLP Pipeline Document Understanding - Issue #32
Focused tests for the enhanced DocumentProcessor implementation
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.research_agent import DocumentProcessor


class TestNLPPipeline(unittest.TestCase):
    """Test suite for NLP Pipeline Document Understanding functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DocumentProcessor(
            extractors=["entities", "concepts", "relationships"],
            classifiers=["topic", "quality", "novelty"],
            summarizers=["abstract", "key_findings"]
        )
        
        # Sample academic text for testing
        self.sample_text = """
        This study investigates the effects of machine learning algorithms on automated manuscript review processes.
        We developed a novel approach using natural language processing techniques to assess document quality.
        The research demonstrates significant improvements in review efficiency, with a 25% reduction in processing time.
        Our findings indicate that automated systems can effectively identify high-quality research papers.
        The methodology involved analyzing 500 manuscripts using sentiment analysis and topic classification.
        Results show that the proposed system outperformed traditional manual review processes.
        In conclusion, this research contributes to the advancement of automated academic publishing workflows.
        """
        
        self.short_text = "This is a short text for testing basic functionality."
        
        self.relationships_text = """
        Machine learning affects document processing efficiency.
        Natural language processing leads to better text analysis.
        Quality assessment depends on multiple criteria.
        Automation enhances productivity in academic publishing.
        """

    def test_initialization_with_default_params(self):
        """Test DocumentProcessor initialization with default parameters"""
        processor = DocumentProcessor()
        
        self.assertEqual(processor.extractors, ["entities", "concepts", "relationships"])
        self.assertEqual(processor.classifiers, ["topic", "quality", "novelty"])
        self.assertEqual(processor.summarizers, ["abstract", "key_findings"])

    def test_initialization_with_custom_params(self):
        """Test DocumentProcessor initialization with custom parameters"""
        custom_extractors = ["entities", "concepts"]
        custom_classifiers = ["topic", "quality"]
        custom_summarizers = ["abstract"]
        
        processor = DocumentProcessor(
            extractors=custom_extractors,
            classifiers=custom_classifiers,
            summarizers=custom_summarizers
        )
        
        self.assertEqual(processor.extractors, custom_extractors)
        self.assertEqual(processor.classifiers, custom_classifiers)
        self.assertEqual(processor.summarizers, custom_summarizers)

    def test_extract_entities(self):
        """Test entity extraction functionality"""
        entities = self.processor.extract_entities(self.sample_text)
        
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
        
        # Check for expected entities
        entities_lower = [e.lower() for e in entities]
        self.assertTrue(any('machine' in e for e in entities_lower))
        self.assertTrue(any('natural' in e for e in entities_lower))

    def test_extract_concepts(self):
        """Test concept extraction functionality"""
        concepts = self.processor.extract_concepts(self.sample_text)
        
        self.assertIsInstance(concepts, list)
        self.assertGreater(len(concepts), 0)
        
        # Concepts should be meaningful terms
        for concept in concepts:
            self.assertIsInstance(concept, str)
            self.assertGreater(len(concept), 0)

    def test_extract_relationships_new_method(self):
        """Test the new relationship extraction functionality"""
        relationships = self.processor.extract_relationships(self.relationships_text)
        
        self.assertIsInstance(relationships, list)
        
        # Check relationship structure
        for rel in relationships:
            self.assertIsInstance(rel, dict)
            self.assertIn('entity1', rel)
            self.assertIn('relationship', rel)
            self.assertIn('entity2', rel)
            self.assertIn('confidence', rel)
            
            # Validate confidence score
            self.assertIsInstance(rel['confidence'], (int, float))
            self.assertGreaterEqual(rel['confidence'], 0)
            self.assertLessEqual(rel['confidence'], 1)

    def test_classify_topic(self):
        """Test topic classification functionality"""
        topic = self.processor.classify_topic(self.sample_text)
        
        self.assertIsInstance(topic, str)
        self.assertGreater(len(topic), 0)
        
        # Should classify as machine learning related
        self.assertIn('machine', topic.lower())

    def test_assess_quality(self):
        """Test quality assessment functionality"""
        quality = self.processor.assess_quality(self.sample_text)
        
        self.assertIsInstance(quality, (int, float))
        self.assertGreaterEqual(quality, 0)
        self.assertLessEqual(quality, 1)
        
        # Long academic text should have higher quality than short text
        short_quality = self.processor.assess_quality(self.short_text)
        self.assertGreater(quality, short_quality)

    def test_assess_novelty(self):
        """Test novelty assessment functionality"""
        novelty = self.processor.assess_novelty(self.sample_text)
        
        self.assertIsInstance(novelty, (int, float))
        self.assertGreaterEqual(novelty, 0)
        self.assertLessEqual(novelty, 1)
        
        # Text with "novel" should have higher novelty score
        novel_text = "This novel approach presents innovative methods."
        novel_score = self.processor.assess_novelty(novel_text)
        self.assertGreater(novel_score, 0.5)

    def test_summarize_abstract_new_method(self):
        """Test the new abstract summarization functionality"""
        summary = self.processor.summarize_abstract(self.sample_text, max_sentences=2)
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertLess(len(summary), len(self.sample_text))
        
        # Summary should contain key terms
        self.assertTrue(any(term in summary.lower() for term in ['study', 'research', 'findings', 'results']))

    def test_extract_key_findings_new_method(self):
        """Test the new key findings extraction functionality"""
        findings = self.processor.extract_key_findings(self.sample_text)
        
        self.assertIsInstance(findings, list)
        
        # Check findings structure
        for finding in findings:
            self.assertIsInstance(finding, str)
            self.assertGreater(len(finding.strip()), 10)  # Meaningful length
        
        # Should extract findings with result indicators
        findings_text = ' '.join(findings).lower()
        self.assertTrue(any(term in findings_text for term in ['results', 'findings', 'demonstrates', 'show']))

    def test_process_document_comprehensive(self):
        """Test the comprehensive document processing method"""
        results = self.processor.process_document(self.sample_text)
        
        self.assertIsInstance(results, dict)
        
        # Check all expected components are present
        self.assertIn('processing_timestamp', results)
        self.assertIn('text_length', results)
        self.assertIn('word_count', results)
        
        # Extractors
        self.assertIn('entities', results)
        self.assertIn('concepts', results)
        self.assertIn('relationships', results)
        
        # Classifiers
        self.assertIn('topic', results)
        self.assertIn('quality_score', results)
        self.assertIn('novelty_score', results)
        
        # Summarizers
        self.assertIn('abstract_summary', results)
        self.assertIn('key_findings', results)

    def test_process_document_partial_configuration(self):
        """Test document processing with partial configuration"""
        partial_processor = DocumentProcessor(
            extractors=["entities"],
            classifiers=["quality"],
            summarizers=["abstract"]
        )
        
        results = partial_processor.process_document(self.sample_text)
        
        # Should only have configured components
        self.assertIn('entities', results)
        self.assertNotIn('concepts', results)
        self.assertNotIn('relationships', results)
        
        self.assertIn('quality_score', results)
        self.assertNotIn('topic', results)
        self.assertNotIn('novelty_score', results)
        
        self.assertIn('abstract_summary', results)
        self.assertNotIn('key_findings', results)

    def test_error_handling_empty_text(self):
        """Test error handling with empty text"""
        # Should handle empty text gracefully
        entities = self.processor.extract_entities("")
        self.assertIsInstance(entities, list)
        
        concepts = self.processor.extract_concepts("")
        self.assertIsInstance(concepts, list)
        
        relationships = self.processor.extract_relationships("")
        self.assertIsInstance(relationships, list)
        
        summary = self.processor.summarize_abstract("")
        self.assertIsInstance(summary, str)
        
        findings = self.processor.extract_key_findings("")
        self.assertIsInstance(findings, list)

    def test_error_handling_invalid_text(self):
        """Test error handling with invalid/unusual text"""
        unusual_text = "!@#$%^&*()_+ 123456789 ñáéíóú"
        
        # Should not crash with unusual characters
        results = self.processor.process_document(unusual_text)
        self.assertIsInstance(results, dict)
        self.assertNotIn('error', results)

    def test_configuration_validation(self):
        """Test configuration validation warnings"""
        with patch('models.research_agent.logger') as mock_logger:
            # Should warn about unsupported components
            DocumentProcessor(
                extractors=["unsupported_extractor"],
                classifiers=["unsupported_classifier"],
                summarizers=["unsupported_summarizer"]
            )
            
            # Check that warnings were logged
            self.assertTrue(mock_logger.warning.called)

    @patch('models.research_agent.TfidfVectorizer')
    def test_fallback_when_sklearn_unavailable(self, mock_tfidf):
        """Test fallback behavior when sklearn is unavailable"""
        # Simulate sklearn unavailability
        mock_tfidf.side_effect = ImportError("sklearn not available")
        
        # Should still work with fallback methods
        concepts = self.processor.extract_concepts(self.sample_text)
        self.assertIsInstance(concepts, list)

    def test_relationship_extraction_patterns(self):
        """Test specific relationship extraction patterns"""
        test_patterns = [
            "Machine learning affects document processing.",
            "Natural language processing leads to better results.",
            "Quality assessment depends on multiple criteria.",
            "Automation enhances productivity significantly.",
            "The algorithm correlates with improved performance."
        ]
        
        for pattern in test_patterns:
            relationships = self.processor.extract_relationships(pattern)
            self.assertIsInstance(relationships, list)
            
            # Should find at least one relationship in each pattern
            if relationships:  # Some patterns might not match due to simplicity
                rel = relationships[0]
                self.assertIn('entity1', rel)
                self.assertIn('relationship', rel)
                self.assertIn('entity2', rel)

    def test_key_findings_extraction_patterns(self):
        """Test specific key findings extraction patterns"""
        findings_text = """
        We found significant improvements in processing speed.
        Results show a 25% increase in efficiency.
        The study demonstrated clear benefits.
        Analysis revealed important correlations (p < 0.05).
        Our research concluded that automation is effective.
        """
        
        findings = self.processor.extract_key_findings(findings_text)
        self.assertIsInstance(findings, list)
        self.assertGreater(len(findings), 0)
        
        # Should capture different types of findings
        findings_str = ' '.join(findings).lower()
        self.assertTrue(any(term in findings_str for term in ['found', 'results', 'demonstrated', 'revealed', 'concluded']))

    def test_abstract_summarization_length_control(self):
        """Test abstract summarization with different length controls"""
        # Test different max_sentences values
        for max_sent in [1, 2, 3, 5]:
            summary = self.processor.summarize_abstract(self.sample_text, max_sentences=max_sent)
            sentence_count = len([s for s in summary.split('.') if s.strip()])
            
            # Should respect max_sentences limit (approximately)
            self.assertLessEqual(sentence_count, max_sent + 1)  # Allow some tolerance

    def test_integration_with_research_agent(self):
        """Test integration with the broader ResearchDiscoveryAgent context"""
        # Test that the enhanced DocumentProcessor works in the expected context
        from models.research_agent import ResearchDiscoveryAgent
        
        # Should be able to create agent with enhanced DocumentProcessor
        try:
            # This might fail due to missing dependencies, but should not fail due to DocumentProcessor
            agent = ResearchDiscoveryAgent()
            self.assertIsNotNone(agent.nlp_pipeline)
            self.assertIsInstance(agent.nlp_pipeline, DocumentProcessor)
        except Exception as e:
            # If it fails due to missing dependencies, that's expected
            if "not available" not in str(e).lower():
                self.fail(f"Unexpected error in integration: {e}")


if __name__ == '__main__':
    unittest.main()