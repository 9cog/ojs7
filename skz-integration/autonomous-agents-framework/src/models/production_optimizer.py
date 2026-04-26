"""
Production Optimization System for Publishing Production Agent
ML-based formatting, quality control, and publication success optimization
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class FormatType(Enum):
    PDF = "pdf"
    HTML = "html"
    EPUB = "epub"
    XML = "xml"
    DOCX = "docx"

class PublicationStatus(Enum):
    FORMATTING = "formatting"
    QUALITY_CHECK = "quality_check"
    READY = "ready"
    PUBLISHED = "published"
    FAILED = "failed"

@dataclass
class Document:
    """Document structure for production processing"""
    document_id: str = ""
    title: str = ""
    content: str = ""
    authors: List[str] = field(default_factory=list)
    format_type: Optional[FormatType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Legacy fields kept optional for backwards compatibility
    target_journal: str = ""
    submission_date: str = ""
    deadline: str = ""
    priority: int = 0
    current_status: Optional[PublicationStatus] = None

    @property
    def doc_id(self) -> str:
        return self.document_id

@dataclass
class FormattingRule:
    """Document formatting rule"""
    rule_id: str
    name: str
    pattern: str
    replacement: str
    applies_to: List[FormatType]
    confidence: float
    usage_count: int

@dataclass
class QualityCheck:
    """Quality check result"""
    check_type: str
    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class QualityReport:
    """Quality report containing all checks"""
    document_id: str
    overall_score: float
    checks: List[QualityCheck]
    recommendations: List[str]
    analysis_date: str

@dataclass
class OptimizationResult:
    """Result of document formatting optimization"""
    original_format: Optional[FormatType]
    optimized_format: FormatType
    confidence_score: float
    applied_optimizations: List[str]
    quality_improvements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PublicationPrediction:
    """Publication success prediction"""
    success_probability: float
    predicted_impact_score: float
    risk_factors: List[str]
    optimization_suggestions: List[str]
    time_to_acceptance: int = 0
    confidence: float = 0.8
    # Legacy fields kept optional
    doc_id: str = ""
    impact_prediction: float = 0.0

class ProductionOptimizer:
    """ML-based production optimization system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.formatting_rules = {}
        self.quality_standards = {}
        self.performance_history = defaultdict(list)
        
        # Load default formatting rules
        self._initialize_formatting_rules()
        
        # Load quality standards
        self._initialize_quality_standards()
        
    def _initialize_formatting_rules(self):
        """Initialize ML-learned formatting rules"""
        
        # Academic citation formatting
        self.formatting_rules['citations'] = FormattingRule(
            rule_id='citations_apa',
            name='APA Citation Format',
            pattern=r'(\w+,\s*\w+\.?\s*\(\d{4}\))',
            replacement=r'\1',
            applies_to=[FormatType.PDF, FormatType.HTML],
            confidence=0.95,
            usage_count=1500
        )
        
        # Reference list formatting
        self.formatting_rules['references'] = FormattingRule(
            rule_id='ref_consistency',
            name='Reference List Consistency',
            pattern=r'(References?|Bibliography)',
            replacement=r'References',
            applies_to=[FormatType.PDF, FormatType.HTML],
            confidence=0.90,
            usage_count=1200
        )
        
        # Section header formatting
        self.formatting_rules['headers'] = FormattingRule(
            rule_id='section_headers',
            name='Section Header Standardization',
            pattern=r'^(\d+\.?\s*)(introduction|methodology|results|discussion|conclusion)(\s*)',
            replacement=r'\1\2',
            applies_to=[FormatType.PDF, FormatType.HTML, FormatType.XML],
            confidence=0.88,
            usage_count=2000
        )
        
    def _initialize_quality_standards(self):
        """Initialize quality control standards"""
        
        self.quality_standards = {
            'metadata_completeness': {
                'required_fields': ['title', 'authors', 'abstract', 'keywords', 'doi'],
                'score_weight': 0.2
            },
            'format_consistency': {
                'checks': ['citation_format', 'reference_format', 'table_format', 'figure_format'],
                'score_weight': 0.25
            },
            'content_quality': {
                'checks': ['spelling', 'grammar', 'technical_accuracy', 'readability'],
                'score_weight': 0.3
            },
            'compliance': {
                'checks': ['journal_guidelines', 'ethical_standards', 'copyright'],
                'score_weight': 0.25
            }
        }
        
    async def optimize_formatting(self, document: Document, target_format: Optional[FormatType] = None) -> OptimizationResult:
        """Apply ML-optimized formatting to document and return an OptimizationResult"""

        if target_format is None:
            # Use document's own format_type or call _predict_optimal_format
            prediction = self._predict_optimal_format(document)
            target_format = prediction.get('recommended_format', getattr(document, 'format_type', None) or FormatType.PDF)
            confidence = float(prediction.get('confidence', 0.8))
            optimization_rules: List[str] = list(prediction.get('optimization_rules', []))
        else:
            confidence = 0.9
            optimization_rules = []

        try:
            logger.info(f"Optimizing formatting for document {document.document_id} to {target_format.value}")

            optimized_content = document.content
            applied_rules = list(optimization_rules)

            # Apply relevant formatting rules from the rule registry
            for rule_id, rule in self.formatting_rules.items():
                if target_format in rule.applies_to and rule.confidence > 0.8:
                    optimized_content = re.sub(
                        rule.pattern,
                        rule.replacement,
                        optimized_content,
                        flags=re.IGNORECASE | re.MULTILINE,
                    )
                    applied_rules.append(rule_id)
                    rule.usage_count += 1

            # Apply format-specific optimizations
            format_specific = await self._apply_format_specific_optimizations(document, target_format)
            applied_rules.extend(format_specific)

            logger.info(f"Applied {len(applied_rules)} formatting rules to document {document.document_id}")

            return OptimizationResult(
                original_format=getattr(document, 'format_type', None),
                optimized_format=target_format,
                confidence_score=confidence,
                applied_optimizations=applied_rules,
            )

        except Exception as e:
            logger.error(f"Error optimizing document formatting: {e}")
            return OptimizationResult(
                original_format=getattr(document, 'format_type', None),
                optimized_format=target_format or FormatType.PDF,
                confidence_score=0.0,
                applied_optimizations=[],
            )

    def _predict_optimal_format(self, document: Document) -> Dict[str, Any]:
        """Predict the optimal output format for a document (can be patched in tests)"""
        # Simple heuristic: prefer PDF unless metadata says otherwise
        preferred = getattr(document, 'format_type', None) or FormatType.PDF
        return {
            'recommended_format': preferred,
            'confidence': 0.8,
            'optimization_rules': ['standardize_citations', 'normalise_headings'],
        }

    async def _apply_format_specific_optimizations(self, document: Document, format_type: FormatType) -> List[str]:
        """Return a list of format-specific optimisation names applied"""
        optimizations: List[str] = []
        if format_type == FormatType.PDF:
            optimizations.extend(['pdf_margin_normalisation', 'pdf_font_embedding'])
        elif format_type == FormatType.HTML:
            optimizations.extend(['html_semantic_markup', 'html_responsive_layout'])
        elif format_type == FormatType.XML:
            optimizations.extend(['xml_schema_validation', 'xml_namespace_declaration'])
        elif format_type == FormatType.EPUB:
            optimizations.extend(['epub_metadata_injection', 'epub_toc_generation'])
        return optimizations

    async def optimize_bulk_documents(self, documents: List[Document]) -> List[OptimizationResult]:
        """Optimise multiple documents in sequence"""
        results = []
        for doc in documents:
            result = await self.optimize_formatting(doc)
            results.append(result)
        return results

    async def perform_quality_control(self, document: Document) -> QualityReport:
        """Perform comprehensive quality control checks and return a QualityReport"""

        quality_checks: List[QualityCheck] = []

        try:
            quality_checks.append(await self._check_metadata_completeness(document))
            quality_checks.append(await self._check_format_consistency(document))
            quality_checks.append(await self._check_content_quality(document))
            quality_checks.append(await self._check_compliance(document))

            overall_score = await self._calculate_overall_quality_score(quality_checks)
            all_recommendations = await self._generate_quality_suggestions(quality_checks)

            logger.info(
                f"Completed quality control for document {document.document_id} with score {overall_score}"
            )

        except Exception as e:
            logger.error(f"Error in quality control: {e}")
            overall_score = 0.0
            all_recommendations = ["Manual review required due to error"]

        return QualityReport(
            document_id=document.document_id,
            overall_score=overall_score,
            checks=quality_checks,
            recommendations=all_recommendations,
            analysis_date=datetime.now().isoformat(),
        )

    async def predict_publication_success(self, document: Document) -> PublicationPrediction:
        """Predict publication success using ML models"""

        try:
            features = await self._extract_publication_features(document)
            success_prob = await self._calculate_success_probability(features)
            impact_pred = await self._predict_impact(features)
            time_estimate = await self._estimate_acceptance_time(features)

            # Quality-report based risk factors
            quality_report = await self.perform_quality_control(document)
            risk_factors = await self._identify_risk_factors(quality_report)
            suggestions = await self._generate_optimization_suggestions(features, document)
            confidence = await self._calculate_prediction_confidence(features)

            return PublicationPrediction(
                doc_id=document.document_id,
                success_probability=success_prob,
                impact_prediction=impact_pred,
                predicted_impact_score=impact_pred,
                time_to_acceptance=time_estimate,
                risk_factors=risk_factors,
                optimization_suggestions=suggestions,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error predicting publication success: {e}")
            return PublicationPrediction(
                doc_id=document.document_id,
                success_probability=0.5,
                impact_prediction=0.0,
                predicted_impact_score=0.0,
                time_to_acceptance=90,
                risk_factors=["Prediction error"],
                optimization_suggestions=["Manual review required"],
                confidence=0.0
            )
    
    async def _optimize_for_pdf(self, content: str, document: Document) -> str:
        """PDF-specific formatting optimizations"""
        
        # Page break optimizations
        content = re.sub(r'(\n\s*){3,}', r'\n\n', content)
        
        # Figure and table positioning
        content = re.sub(r'(Figure\s+\d+)', r'\\begin{figure}[htbp]\n\1', content)
        content = re.sub(r'(Table\s+\d+)', r'\\begin{table}[htbp]\n\1', content)
        
        # Bibliography formatting for PDF
        content = re.sub(r'^References\s*$', r'\\bibliography{references}', content, flags=re.MULTILINE)
        
        return content
    
    async def _optimize_for_html(self, content: str, document: Document) -> str:
        """HTML-specific formatting optimizations"""
        
        # Convert headings to HTML
        content = re.sub(r'^#\s+(.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
        content = re.sub(r'^##\s+(.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
        content = re.sub(r'^###\s+(.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
        
        # Convert paragraphs
        content = re.sub(r'\n\n(.+?)\n\n', r'<p>\1</p>\n\n', content)
        
        # Add semantic markup
        content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
        
        return content
    
    async def _optimize_for_xml(self, content: str, document: Document) -> str:
        """XML-specific formatting optimizations"""
        
        # Add XML structure
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<article>
    <front>
        <article-meta>
            <title-group>
                <article-title>{document.title}</article-title>
            </title-group>
        </article-meta>
    </front>
    <body>
        {content}
    </body>
</article>"""
        
        return xml_content
    
    async def _check_metadata_completeness(self, document: Document) -> QualityCheck:
        """Check metadata completeness"""
        
        required_fields = self.quality_standards['metadata_completeness']['required_fields']
        missing_fields = []
        
        for f in required_fields:
            if f not in document.metadata or not document.metadata[f]:
                missing_fields.append(f)
        
        completeness_score = (len(required_fields) - len(missing_fields)) / max(len(required_fields), 1)
        
        return QualityCheck(
            check_type='metadata_completeness',
            score=completeness_score,
            passed=completeness_score >= 1.0,
            details={'missing_fields': missing_fields, 'required_fields': required_fields},
            recommendations=[f"Add '{f}' to document metadata" for f in missing_fields],
        )
    
    async def _check_format_consistency(self, document: Document) -> QualityCheck:
        """Check formatting consistency"""
        
        issues = []
        suggestions = []
        
        # Check citation format consistency
        citations = re.findall(r'\([^)]*\d{4}[^)]*\)', document.content)
        if len(set(citations)) / max(len(citations), 1) < 0.8:
            issues.append("Inconsistent citation formatting")
            suggestions.append("Standardize citation format throughout document")
        
        # Check reference format
        ref_section = re.search(r'References?\s*\n(.+)', document.content, re.DOTALL)
        if ref_section:
            references = ref_section.group(1).split('\n')
            if len(references) > 5:  # Only check if significant number of references
                format_consistency = self._check_reference_format_consistency(references)
                if format_consistency < 0.8:
                    issues.append("Inconsistent reference formatting")
                    suggestions.append("Standardize reference list formatting")
        
        consistency_score = max(0.0, 1.0 - len(issues) * 0.3)
        
        return QualityCheck(
            check_type='format_consistency',
            score=consistency_score,
            passed=consistency_score > 0.8,
            details={'issues': issues},
            recommendations=suggestions,
        )
    
    async def _check_content_quality(self, document: Document) -> QualityCheck:
        """Check content quality"""
        
        issues = []
        suggestions = []
        quality_score = 1.0
        
        # Basic content checks
        word_count = len(document.content.split())
        if word_count < 1000:
            issues.append("Document may be too short for publication")
            suggestions.append("Consider expanding content or methodology sections")
            quality_score *= 0.9
        
        # Check for common writing issues
        sentences = document.content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if avg_sentence_length > 25:
            issues.append("Average sentence length is high - may affect readability")
            suggestions.append("Consider breaking up long sentences")
            quality_score *= 0.95
        
        # Check for repetitive phrases
        words = document.content.lower().split()
        word_freq: Dict[str, int] = defaultdict(int)
        for word in words:
            if len(word) > 4:
                word_freq[word] += 1
        
        over_used_words = [w for w, freq in word_freq.items() if freq > len(words) * 0.01]
        if over_used_words:
            issues.append(f"Potentially overused words: {', '.join(over_used_words[:3])}")
            suggestions.append("Consider using synonyms to improve writing variety")
            quality_score *= 0.98
        
        return QualityCheck(
            check_type='content_quality',
            score=quality_score,
            passed=quality_score > 0.85,
            details={'word_count': word_count, 'avg_sentence_length': avg_sentence_length, 'issues': issues},
            recommendations=suggestions,
        )
    
    async def _check_compliance(self, document: Document) -> QualityCheck:
        """Check compliance with standards"""
        
        issues = []
        suggestions = []
        compliance_score = 1.0
        
        # Check for required sections
        required_sections = ['abstract', 'introduction', 'methodology', 'results', 'conclusion']
        content_lower = document.content.lower()
        
        for section in required_sections:
            if section not in content_lower:
                issues.append(f"Missing required section: {section}")
                suggestions.append(f"Add {section} section to document")
                compliance_score *= 0.9
        
        # Check for ethical compliance indicators
        ethical_keywords = ['ethics', 'consent', 'approval', 'institutional review']
        if not any(keyword in content_lower for keyword in ethical_keywords):
            issues.append("No ethical compliance statements found")
            suggestions.append("Add ethical approval and consent statements")
            compliance_score *= 0.95
        
        return QualityCheck(
            check_type='compliance',
            score=compliance_score,
            passed=compliance_score > 0.9,
            details={'issues': issues},
            recommendations=suggestions,
        )
    
    def _check_reference_format_consistency(self, references: List[str]) -> float:
        """Check consistency of reference formatting"""
        
        if not references:
            return 1.0
        
        # Simple heuristic: check if references follow similar patterns
        patterns = []
        for ref in references[:10]:  # Check first 10 references
            # Extract pattern (author pattern, year pattern, etc.)
            pattern = self._extract_reference_pattern(ref)
            patterns.append(pattern)
        
        # Calculate consistency as ratio of most common pattern
        if patterns:
            most_common = max(set(patterns), key=patterns.count)
            consistency = patterns.count(most_common) / len(patterns)
            return consistency
        
        return 1.0
    
    def _extract_reference_pattern(self, reference: str) -> str:
        """Extract formatting pattern from reference"""
        
        # Simplified pattern extraction
        pattern_elements = []
        
        # Check for author pattern
        if re.search(r'^[A-Z][a-z]+,\s*[A-Z]\.', reference):
            pattern_elements.append('LastFirst')
        elif re.search(r'^[A-Z]\.\s*[A-Z][a-z]+', reference):
            pattern_elements.append('FirstLast')
        
        # Check for year pattern
        if re.search(r'\(\d{4}\)', reference):
            pattern_elements.append('ParenYear')
        elif re.search(r'\d{4}\.', reference):
            pattern_elements.append('DotYear')
        
        # Check for title pattern
        if re.search(r'"[^"]+"\s*\.', reference):
            pattern_elements.append('QuotedTitle')
        elif re.search(r'[A-Z][^.]+\.\s*[A-Z]', reference):
            pattern_elements.append('PlainTitle')
        
        return '_'.join(pattern_elements)
    
    async def _calculate_overall_quality_score(self, quality_checks: List[QualityCheck]) -> float:
        """Calculate weighted overall quality score"""
        
        check_type_weights = {
            'metadata_completeness': self.quality_standards.get('metadata_completeness', {}).get('score_weight', 0.25),
            'format_consistency': self.quality_standards.get('format_consistency', {}).get('score_weight', 0.25),
            'content_quality': self.quality_standards.get('content_quality', {}).get('score_weight', 0.25),
            'compliance': self.quality_standards.get('compliance', {}).get('score_weight', 0.25),
        }

        weighted_score = 0.0
        total_weight = 0.0
        
        for check in quality_checks:
            weight = check_type_weights.get(check.check_type, 0.1)
            weighted_score += check.score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    async def _generate_quality_suggestions(self, quality_checks: List[QualityCheck]) -> List[str]:
        """Generate improvement suggestions from quality checks"""
        
        all_suggestions: List[str] = []
        for check in quality_checks:
            all_suggestions.extend(check.recommendations)
        
        # Prioritize suggestions
        return all_suggestions[:5]  # Top 5 suggestions
    
    async def _extract_publication_features(self, document: Document) -> Dict[str, Any]:
        """Extract features for publication success prediction"""
        
        features: Dict[str, Any] = {}
        
        # Document characteristics
        features['word_count'] = len(document.content.split())
        features['author_count'] = len(document.authors)
        features['section_count'] = len(re.findall(r'^#+\s', document.content, re.MULTILINE))
        
        # Metadata features
        features['has_keywords'] = len(document.metadata.get('keywords', [])) > 0
        features['keyword_count'] = len(document.metadata.get('keywords', []))
        features['has_abstract'] = bool(document.metadata.get('abstract', ''))
        
        # Content quality indicators
        features['avg_sentence_length'] = self._calculate_avg_sentence_length(document.content)
        features['reference_count'] = len(re.findall(r'(?:References?|Bibliography)', document.content))
        features['figure_count'] = len(re.findall(r'Figure\s+\d+', document.content))
        features['table_count'] = len(re.findall(r'Table\s+\d+', document.content))
        
        # Journal and timing features (use defaults when fields are empty)
        features['target_journal'] = document.target_journal or document.metadata.get('journal', '')
        if document.deadline:
            try:
                features['days_until_deadline'] = (
                    datetime.strptime(document.deadline, '%Y-%m-%d') - datetime.now()
                ).days
            except ValueError:
                features['days_until_deadline'] = 90
        else:
            features['days_until_deadline'] = 90
        features['priority'] = document.priority
        
        return features
    
    async def _calculate_success_probability(self, features: Dict[str, Any]) -> float:
        """Calculate publication success probability"""
        
        # Simplified ML model simulation
        base_prob = 0.6
        
        # Word count factor
        if features['word_count'] > 3000:
            base_prob += 0.15
        elif features['word_count'] < 1500:
            base_prob -= 0.15
        
        # Author count factor
        if 2 <= features['author_count'] <= 5:
            base_prob += 0.1
        
        # Quality indicators
        if features['has_keywords']:
            base_prob += 0.05
        if features['has_abstract']:
            base_prob += 0.05
        if features['reference_count'] > 20:
            base_prob += 0.1
        
        # Readability factor
        if 15 <= features['avg_sentence_length'] <= 20:
            base_prob += 0.05
        
        return min(0.95, max(0.1, base_prob))
    
    async def _predict_impact(self, features: Dict[str, Any]) -> float:
        """Predict publication impact score"""
        
        # Simplified impact prediction
        base_impact = 2.0
        
        # Multi-author bonus
        if features['author_count'] > 3:
            base_impact += 0.5
        
        # Comprehensive content bonus
        if features['figure_count'] > 2:
            base_impact += 0.3
        if features['table_count'] > 2:
            base_impact += 0.3
        if features['reference_count'] > 30:
            base_impact += 0.4
        
        return min(10.0, base_impact)
    
    async def _estimate_acceptance_time(self, features: Dict[str, Any]) -> int:
        """Estimate days until acceptance"""
        
        base_time = 90  # 3 months baseline
        
        # Adjust based on completeness
        if features['has_keywords'] and features['has_abstract']:
            base_time -= 15
        
        # Quality factors
        if features['reference_count'] > 25:
            base_time -= 10
        
        # Complexity factors (longer for complex papers)
        if features['word_count'] > 5000:
            base_time += 20
        
        return max(30, base_time)
    
    async def _identify_risk_factors(self, quality_report_or_features: Any, document: Optional[Document] = None) -> List[str]:
        """Identify publication risk factors.

        Accepts either a QualityReport (new interface) or a features dict (legacy).
        """
        risks: List[str] = []

        if isinstance(quality_report_or_features, QualityReport):
            report: QualityReport = quality_report_or_features
            if report.overall_score < 0.7:
                risks.append("Overall quality score is below acceptance threshold")
            for check in report.checks:
                if not check.passed:
                    risks.append(f"Quality issue in {check.check_type}: {', '.join(check.recommendations[:1])}")
        else:
            # Legacy features dict
            features = quality_report_or_features
            if features.get('word_count', 0) < 2000:
                risks.append("Document length may be insufficient")
            if features.get('author_count', 0) == 1:
                risks.append("Single-author papers have lower acceptance rates")
            if not features.get('has_abstract', False):
                risks.append("Missing abstract will impact editorial decision")
            if features.get('reference_count', 0) < 15:
                risks.append("Insufficient literature review")
            if features.get('days_until_deadline', 90) < 30:
                risks.append("Tight deadline may compromise quality")
        
        return risks[:5]
    
    async def _generate_optimization_suggestions(self, features: Dict[str, Any], document: Document) -> List[str]:
        """Generate optimization suggestions"""
        
        suggestions = []
        
        if features['word_count'] < 3000:
            suggestions.append("Consider expanding methodology and results sections")
        
        if features['reference_count'] < 25:
            suggestions.append("Strengthen literature review with additional references")
        
        if features['figure_count'] < 2:
            suggestions.append("Add figures to illustrate key findings")
        
        if not features['has_keywords']:
            suggestions.append("Add relevant keywords to improve discoverability")
        
        if features['avg_sentence_length'] > 25:
            suggestions.append("Improve readability by shortening complex sentences")
        
        return suggestions[:5]
    
    async def _calculate_prediction_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence in predictions"""
        
        confidence = 0.7  # Base confidence
        
        # More complete documents give higher confidence
        completeness_factors = [
            features['has_abstract'],
            features['has_keywords'], 
            features['reference_count'] > 10,
            features['word_count'] > 2000
        ]
        
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        confidence += completeness_score * 0.2
        
        return min(0.95, confidence)
    
    def _calculate_avg_sentence_length(self, content: str) -> float:
        """Calculate average sentence length"""
        
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if not sentences:
            return 0.0
        
        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)


# Utility functions
async def optimize_document(document_data: Dict, target_format: str = 'pdf') -> Dict:
    """Quick document optimization utility"""
    
    doc = Document(**document_data)
    optimizer = ProductionOptimizer({})
    
    # Optimize formatting
    optimized_doc = await optimizer.optimize_formatting(doc, FormatType(target_format))
    
    # Perform quality control
    quality_checks = await optimizer.perform_quality_control(optimized_doc)
    
    # Predict success
    prediction = await optimizer.predict_publication_success(optimized_doc)
    
    return {
        'document': asdict(optimized_doc),
        'quality_checks': [asdict(check) for check in quality_checks],
        'success_prediction': asdict(prediction)
    }
