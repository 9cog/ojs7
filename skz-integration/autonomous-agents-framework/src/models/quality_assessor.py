"""
Quality Assessment ML Model for Submission Assistant Agent
Advanced manuscript quality evaluation using machine learning approaches
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import re
import pickle
from pathlib import Path
import hashlib

# NLP and ML imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    import spacy
    from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
except ImportError as e:
    logging.warning(f"ML dependencies not fully available: {e}")

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality assessment metrics structure"""
    scientific_rigor: float
    methodology_score: float
    novelty_score: float
    clarity_score: float
    statistical_validity: float
    overall_score: float
    confidence: float
    detailed_scores: Dict[str, float]
    recommendations: List[str]
    timestamp: str

@dataclass
class ManuscriptFeatures:
    """Extracted manuscript features for ML analysis"""
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    readability_score: float
    statistical_terms_count: int
    methodology_terms_count: int
    citation_count: int
    figure_count: int
    table_count: int
    abstract_length: int
    keywords_count: int
    reference_count: int

class QualityAssessor:
    """Advanced ML-based manuscript quality assessment system"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else Path("./models")
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.is_initialized = False
        
        # Quality assessment dimensions
        self.assessment_dimensions = [
            'scientific_rigor',
            'methodology',
            'novelty',
            'clarity',
            'statistical_validity'
        ]
        
        # Domain-specific term dictionaries
        self.statistical_terms = {
            'p-value', 'confidence interval', 'regression', 'anova', 'correlation',
            'standard deviation', 'mean', 'median', 'variance', 'hypothesis test',
            'significance', 'chi-square', 't-test', 'mann-whitney', 'kruskal-wallis'
        }
        
        self.methodology_terms = {
            'randomized', 'controlled', 'double-blind', 'placebo', 'protocol',
            'inclusion criteria', 'exclusion criteria', 'sample size', 'power analysis',
            'crossover', 'parallel group', 'dose-response', 'washout period'
        }
        
        self.cosmetic_science_terms = {
            'inci', 'formulation', 'stability', 'preservation', 'emulsification',
            'rheology', 'viscosity', 'ph', 'dermatological', 'patch test',
            'consumer testing', 'sensory evaluation', 'skin barrier', 'hydration'
        }
        
        # Initialize NLP pipeline
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not available, using basic text processing")
            self.nlp = None
    
    async def initialize_models(self, training_data_path: Optional[str] = None):
        """Initialize or load pre-trained ML models"""
        try:
            # Try to load existing models
            if self.model_path.exists():
                await self._load_models()
            else:
                # Create new models if training data is available
                if training_data_path:
                    await self._train_models(training_data_path)
                else:
                    # Initialize with default models
                    await self._initialize_default_models()
            
            self.is_initialized = True
            logger.info("Quality assessment models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize quality assessment models: {e}")
            # Fallback to rule-based system
            self.is_initialized = False
    
    async def assess_manuscript(self, manuscript_text: str, metadata: Optional[Dict] = None) -> QualityMetrics:
        """
        Comprehensive manuscript quality assessment
        """
        logger.info("Starting comprehensive manuscript quality assessment")
        
        try:
            # Extract features from manuscript
            features = await self._extract_manuscript_features(manuscript_text)
            
            # Initialize scores
            scores = {}
            detailed_scores = {}
            recommendations = []
            
            if self.is_initialized and self.models:
                # Use ML models for assessment
                scores = await self._ml_assessment(manuscript_text, features)
            else:
                # Fallback to rule-based assessment
                scores = await self._rule_based_assessment(manuscript_text, features)
            
            # Generate detailed analysis
            detailed_scores = await self._detailed_analysis(manuscript_text, features)
            
            # Generate improvement recommendations
            recommendations = await self._generate_recommendations(scores, features, detailed_scores)
            
            # Calculate overall score and confidence
            overall_score = sum(scores.values()) / len(scores)
            confidence = self._calculate_confidence(scores, features)
            
            return QualityMetrics(
                scientific_rigor=scores.get('scientific_rigor', 0.5),
                methodology_score=scores.get('methodology', 0.5),
                novelty_score=scores.get('novelty', 0.5),
                clarity_score=scores.get('clarity', 0.5),
                statistical_validity=scores.get('statistical_validity', 0.5),
                overall_score=overall_score,
                confidence=confidence,
                detailed_scores=detailed_scores,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in manuscript assessment: {e}")
            return self._create_error_metrics(str(e))
    
    async def _extract_manuscript_features(self, text: str) -> ManuscriptFeatures:
        """Extract comprehensive features from manuscript text"""
        
        # Basic text statistics
        word_count = len(text.split())
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        paragraphs = text.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Readability analysis
        try:
            readability_score = flesch_reading_ease(text)
        except:
            readability_score = 50.0  # Default moderate readability
        
        # Domain-specific term counting
        text_lower = text.lower()
        statistical_terms_count = sum(1 for term in self.statistical_terms if term in text_lower)
        methodology_terms_count = sum(1 for term in self.methodology_terms if term in text_lower)
        
        # Citation and reference analysis
        citation_count = len(re.findall(r'\[\d+\]|\(\d{4}\)', text))
        reference_count = len(re.findall(r'^References?:', text, re.MULTILINE | re.IGNORECASE))
        
        # Figure and table counting
        figure_count = len(re.findall(r'Figure \d+|Fig\. \d+', text, re.IGNORECASE))
        table_count = len(re.findall(r'Table \d+', text, re.IGNORECASE))
        
        # Abstract analysis
        abstract_match = re.search(r'Abstract:?\s*(.*?)(?:\n\n|\nKeywords|\nIntroduction)', text, re.DOTALL | re.IGNORECASE)
        abstract_length = len(abstract_match.group(1).split()) if abstract_match else 0
        
        # Keywords analysis
        keywords_match = re.search(r'Keywords:?\s*(.*?)(?:\n\n|\nIntroduction)', text, re.IGNORECASE)
        keywords_count = len(keywords_match.group(1).split(',')) if keywords_match else 0
        
        return ManuscriptFeatures(
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            avg_sentence_length=avg_sentence_length,
            readability_score=readability_score,
            statistical_terms_count=statistical_terms_count,
            methodology_terms_count=methodology_terms_count,
            citation_count=citation_count,
            figure_count=figure_count,
            table_count=table_count,
            abstract_length=abstract_length,
            keywords_count=keywords_count,
            reference_count=reference_count
        )
    
    async def _rule_based_assessment(self, text: str, features: ManuscriptFeatures) -> Dict[str, float]:
        """Rule-based quality assessment as fallback"""
        
        scores = {}
        
        # Scientific Rigor Assessment
        rigor_score = 0.5
        if features.statistical_terms_count >= 5:
            rigor_score += 0.2
        if features.citation_count >= 20:
            rigor_score += 0.2
        if features.methodology_terms_count >= 3:
            rigor_score += 0.1
        scores['scientific_rigor'] = min(1.0, rigor_score)
        
        # Methodology Assessment
        methodology_score = 0.4
        if features.methodology_terms_count >= 5:
            methodology_score += 0.3
        if 'control' in text.lower() or 'controlled' in text.lower():
            methodology_score += 0.2
        if features.statistical_terms_count >= 3:
            methodology_score += 0.1
        scores['methodology'] = min(1.0, methodology_score)
        
        # Novelty Assessment (simplified)
        novelty_score = 0.5
        if 'novel' in text.lower() or 'innovative' in text.lower():
            novelty_score += 0.2
        if features.word_count >= 5000:  # Comprehensive studies tend to be novel
            novelty_score += 0.1
        scores['novelty'] = min(1.0, novelty_score)
        
        # Clarity Assessment
        clarity_score = 0.5
        if 40 <= features.readability_score <= 70:  # Optimal readability range
            clarity_score += 0.2
        if features.figure_count >= 2:
            clarity_score += 0.1
        if features.table_count >= 1:
            clarity_score += 0.1
        if features.abstract_length >= 150:
            clarity_score += 0.1
        scores['clarity'] = min(1.0, clarity_score)
        
        # Statistical Validity Assessment
        validity_score = 0.4
        if features.statistical_terms_count >= 5:
            validity_score += 0.3
        if 'p-value' in text.lower() or 'confidence interval' in text.lower():
            validity_score += 0.2
        if features.word_count >= 3000:  # Adequate detail for statistical analysis
            validity_score += 0.1
        scores['statistical_validity'] = min(1.0, validity_score)
        
        return scores
    
    async def _detailed_analysis(self, text: str, features: ManuscriptFeatures) -> Dict[str, float]:
        """Generate detailed quality analysis scores"""
        
        detailed = {}
        
        # Abstract quality
        abstract_quality = 0.5
        if 150 <= features.abstract_length <= 300:
            abstract_quality = 0.8
        elif features.abstract_length > 0:
            abstract_quality = 0.6
        detailed['abstract_quality'] = abstract_quality
        
        # Reference adequacy
        ref_adequacy = min(1.0, features.citation_count / 30.0)  # Target ~30 references
        detailed['reference_adequacy'] = ref_adequacy
        
        # Structure completeness
        structure_score = 0.0
        required_sections = ['abstract', 'introduction', 'method', 'result', 'discussion', 'conclusion']
        text_lower = text.lower()
        
        for section in required_sections:
            if section in text_lower:
                structure_score += 1.0 / len(required_sections)
        
        detailed['structure_completeness'] = structure_score
        
        # Statistical rigor
        stat_rigor = min(1.0, features.statistical_terms_count / 10.0)
        detailed['statistical_rigor'] = stat_rigor
        
        # Visual aids adequacy
        visual_score = 0.0
        if features.figure_count > 0:
            visual_score += 0.5
        if features.table_count > 0:
            visual_score += 0.5
        detailed['visual_aids'] = visual_score
        
        # Word count appropriateness (for research articles)
        word_appropriateness = 1.0
        if features.word_count < 2000:
            word_appropriateness = 0.4
        elif features.word_count > 10000:
            word_appropriateness = 0.7
        detailed['word_count_appropriateness'] = word_appropriateness
        
        return detailed
    
    async def _generate_recommendations(self, scores: Dict[str, float], features: ManuscriptFeatures, detailed: Dict[str, float]) -> List[str]:
        """Generate intelligent improvement recommendations"""
        
        recommendations = []
        
        # Scientific rigor recommendations
        if scores.get('scientific_rigor', 0.5) < 0.7:
            recommendations.append("Strengthen scientific rigor by adding more statistical analyses and increasing citation coverage")
        
        # Methodology recommendations
        if scores.get('methodology', 0.5) < 0.7:
            recommendations.append("Improve methodology section with clearer experimental design and control descriptions")
        
        # Clarity recommendations
        if scores.get('clarity', 0.5) < 0.7:
            if features.readability_score < 30:
                recommendations.append("Improve text clarity - consider simplifying sentence structure")
            if features.figure_count == 0:
                recommendations.append("Add figures to improve visual communication of results")
        
        # Statistical validity recommendations
        if scores.get('statistical_validity', 0.5) < 0.7:
            recommendations.append("Strengthen statistical analysis with appropriate tests and effect size reporting")
        
        # Structure recommendations
        if detailed.get('structure_completeness', 0.5) < 0.8:
            recommendations.append("Ensure all required manuscript sections are present and clearly labeled")
        
        # Abstract recommendations
        if detailed.get('abstract_quality', 0.5) < 0.7:
            if features.abstract_length < 150:
                recommendations.append("Expand abstract to include more comprehensive study overview")
            elif features.abstract_length > 300:
                recommendations.append("Consider condensing abstract while maintaining key information")
        
        # Reference recommendations
        if detailed.get('reference_adequacy', 0.5) < 0.6:
            recommendations.append("Increase reference coverage to better support claims and provide context")
        
        return recommendations
    
    def _calculate_confidence(self, scores: Dict[str, float], features: ManuscriptFeatures) -> float:
        """Calculate confidence in the quality assessment"""
        
        # Base confidence
        confidence = 0.7
        
        # Increase confidence with more data
        if features.word_count >= 3000:
            confidence += 0.1
        if features.citation_count >= 15:
            confidence += 0.1
        if features.statistical_terms_count >= 5:
            confidence += 0.1
        
        # Decrease confidence for edge cases
        if features.word_count < 1000:
            confidence -= 0.2
        if features.citation_count < 5:
            confidence -= 0.1
        
        return min(1.0, max(0.3, confidence))
    
    def _create_error_metrics(self, error_message: str) -> QualityMetrics:
        """Create error metrics when assessment fails"""
        return QualityMetrics(
            scientific_rigor=0.0,
            methodology_score=0.0,
            novelty_score=0.0,
            clarity_score=0.0,
            statistical_validity=0.0,
            overall_score=0.0,
            confidence=0.0,
            detailed_scores={'error': 1.0},
            recommendations=[f"Assessment failed: {error_message}"],
            timestamp=datetime.now().isoformat()
        )
    
    async def batch_assessment(self, manuscripts: List[Dict[str, str]]) -> List[QualityMetrics]:
        """Assess multiple manuscripts in batch"""
        
        logger.info(f"Starting batch assessment of {len(manuscripts)} manuscripts")
        
        tasks = [
            self.assess_manuscript(manuscript['content'], manuscript.get('metadata'))
            for manuscript in manuscripts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error assessing manuscript {i}: {result}")
                final_results.append(self._create_error_metrics(str(result)))
            else:
                final_results.append(result)
        
        return final_results
    
    async def _initialize_default_models(self):
        """Initialize default ML models when no trained models available"""
        
        # Create simple default models for each dimension
        for dimension in self.assessment_dimensions:
            # Use logistic regression as default
            self.models[dimension] = LogisticRegression(random_state=42)
            self.vectorizers[dimension] = TfidfVectorizer(max_features=1000, stop_words='english')
            self.scalers[dimension] = StandardScaler()
        
        logger.info("Default models initialized (training required for full functionality)")


# Utility functions
async def quick_quality_check(manuscript_text: str) -> float:
    """Quick quality assessment returning single score"""
    assessor = QualityAssessor()
    await assessor.initialize_models()
    
    metrics = await assessor.assess_manuscript(manuscript_text)
    return metrics.overall_score

async def compare_manuscripts(manuscripts: List[str]) -> List[Tuple[int, float]]:
    """Compare multiple manuscripts, return ranked by quality"""
    assessor = QualityAssessor()
    await assessor.initialize_models()
    
    manuscript_dicts = [{'content': text} for text in manuscripts]
    results = await assessor.batch_assessment(manuscript_dicts)
    
    # Create ranked list
    ranked = [(i, result.overall_score) for i, result in enumerate(results)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    return ranked
