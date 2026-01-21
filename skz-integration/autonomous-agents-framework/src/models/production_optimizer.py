"""
Production Optimization System for Publishing Production Agent
ML-based formatting, quality control, and publication success optimization
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
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
    doc_id: str
    title: str
    authors: List[str]
    content: str
    metadata: Dict[str, Any]
    format_requirements: List[FormatType]
    target_journal: str
    submission_date: str
    deadline: str
    priority: int
    current_status: PublicationStatus

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
    check_id: str
    check_name: str
    status: str  # pass, fail, warning
    score: float
    issues: List[str]
    suggestions: List[str]
    auto_fixable: bool

@dataclass
class PublicationPrediction:
    """Publication success prediction"""
    doc_id: str
    success_probability: float
    impact_prediction: float
    time_to_acceptance: int
    risk_factors: List[str]
    optimization_suggestions: List[str]
    confidence: float

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
        
    async def optimize_formatting(self, document: Document, target_format: FormatType) -> Document:
        """Apply ML-optimized formatting to document"""
        
        try:
            logger.info(f"Optimizing formatting for document {document.doc_id} to {target_format.value}")
            
            optimized_content = document.content
            applied_rules = []
            
            # Apply relevant formatting rules
            for rule_id, rule in self.formatting_rules.items():
                if target_format in rule.applies_to:
                    
                    # Apply rule with confidence weighting
                    if rule.confidence > 0.8:
                        optimized_content = re.sub(
                            rule.pattern, 
                            rule.replacement, 
                            optimized_content,
                            flags=re.IGNORECASE | re.MULTILINE
                        )
                        applied_rules.append(rule_id)
                        
                        # Update usage statistics
                        rule.usage_count += 1
            
            # Apply format-specific optimizations
            if target_format == FormatType.PDF:
                optimized_content = await self._optimize_for_pdf(optimized_content, document)
            elif target_format == FormatType.HTML:
                optimized_content = await self._optimize_for_html(optimized_content, document)
            elif target_format == FormatType.XML:
                optimized_content = await self._optimize_for_xml(optimized_content, document)
            
            # Update document
            optimized_doc = document
            optimized_doc.content = optimized_content
            optimized_doc.metadata['applied_formatting_rules'] = applied_rules
            optimized_doc.metadata['last_formatted'] = datetime.now().isoformat()
            
            logger.info(f"Applied {len(applied_rules)} formatting rules to document {document.doc_id}")
            
            return optimized_doc
            
        except Exception as e:
            logger.error(f"Error optimizing document formatting: {e}")
            return document
    
    async def perform_quality_control(self, document: Document) -> List[QualityCheck]:
        """Perform comprehensive quality control checks"""
        
        quality_checks = []
        
        try:
            # Metadata completeness check
            metadata_check = await self._check_metadata_completeness(document)
            quality_checks.append(metadata_check)
            
            # Format consistency check
            format_check = await self._check_format_consistency(document)
            quality_checks.append(format_check)
            
            # Content quality check
            content_check = await self._check_content_quality(document)
            quality_checks.append(content_check)
            
            # Compliance check
            compliance_check = await self._check_compliance(document)
            quality_checks.append(compliance_check)
            
            # Calculate overall quality score
            overall_score = await self._calculate_overall_quality_score(quality_checks)
            
            # Add summary check
            summary_check = QualityCheck(
                check_id='overall_quality',
                check_name='Overall Quality Assessment',
                status='pass' if overall_score > 0.8 else 'warning' if overall_score > 0.6 else 'fail',
                score=overall_score,
                issues=[],
                suggestions=await self._generate_quality_suggestions(quality_checks),
                auto_fixable=False
            )
            quality_checks.append(summary_check)
            
            logger.info(f"Completed quality control for document {document.doc_id} with score {overall_score}")
            
        except Exception as e:
            logger.error(f"Error in quality control: {e}")
        
        return quality_checks
    
    async def predict_publication_success(self, document: Document) -> PublicationPrediction:
        """Predict publication success using ML models"""
        
        try:
            # Extract features for prediction
            features = await self._extract_publication_features(document)
            
            # Calculate success probability
            success_prob = await self._calculate_success_probability(features)
            
            # Predict impact
            impact_pred = await self._predict_impact(features)
            
            # Estimate time to acceptance
            time_estimate = await self._estimate_acceptance_time(features)
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(features, document)
            
            # Generate optimization suggestions
            suggestions = await self._generate_optimization_suggestions(features, document)
            
            # Calculate prediction confidence
            confidence = await self._calculate_prediction_confidence(features)
            
            return PublicationPrediction(
                doc_id=document.doc_id,
                success_probability=success_prob,
                impact_prediction=impact_pred,
                time_to_acceptance=time_estimate,
                risk_factors=risk_factors,
                optimization_suggestions=suggestions,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error predicting publication success: {e}")
            return PublicationPrediction(
                doc_id=document.doc_id,
                success_probability=0.5,
                impact_prediction=0.0,
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
        
        for field in required_fields:
            if field not in document.metadata or not document.metadata[field]:
                missing_fields.append(field)
        
        completeness_score = (len(required_fields) - len(missing_fields)) / len(required_fields)
        
        return QualityCheck(
            check_id='metadata_completeness',
            check_name='Metadata Completeness',
            status='pass' if completeness_score == 1.0 else 'warning' if completeness_score > 0.8 else 'fail',
            score=completeness_score,
            issues=[f"Missing required field: {field}" for field in missing_fields],
            suggestions=[f"Add {field} to document metadata" for field in missing_fields],
            auto_fixable=False
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
        
        consistency_score = 1.0 - (len(issues) * 0.3)
        consistency_score = max(0.0, consistency_score)
        
        return QualityCheck(
            check_id='format_consistency',
            check_name='Format Consistency',
            status='pass' if consistency_score > 0.8 else 'warning' if consistency_score > 0.6 else 'fail',
            score=consistency_score,
            issues=issues,
            suggestions=suggestions,
            auto_fixable=True
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
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        if avg_sentence_length > 25:
            issues.append("Average sentence length is high - may affect readability")
            suggestions.append("Consider breaking up long sentences")
            quality_score *= 0.95
        
        # Check for repetitive phrases
        words = document.content.lower().split()
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 4:  # Only check significant words
                word_freq[word] += 1
        
        over_used_words = [word for word, freq in word_freq.items() if freq > len(words) * 0.01]
        if over_used_words:
            issues.append(f"Potentially overused words: {', '.join(over_used_words[:3])}")
            suggestions.append("Consider using synonyms to improve writing variety")
            quality_score *= 0.98
        
        return QualityCheck(
            check_id='content_quality',
            check_name='Content Quality',
            status='pass' if quality_score > 0.85 else 'warning' if quality_score > 0.7 else 'fail',
            score=quality_score,
            issues=issues,
            suggestions=suggestions,
            auto_fixable=False
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
            check_id='compliance',
            check_name='Standards Compliance',
            status='pass' if compliance_score > 0.9 else 'warning' if compliance_score > 0.8 else 'fail',
            score=compliance_score,
            issues=issues,
            suggestions=suggestions,
            auto_fixable=False
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
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for check in quality_checks:
            if check.check_id in self.quality_standards:
                weight = self.quality_standards[check.check_id]['score_weight']
                weighted_score += check.score * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    async def _generate_quality_suggestions(self, quality_checks: List[QualityCheck]) -> List[str]:
        """Generate improvement suggestions from quality checks"""
        
        all_suggestions = []
        for check in quality_checks:
            all_suggestions.extend(check.suggestions)
        
        # Prioritize suggestions
        return all_suggestions[:5]  # Top 5 suggestions
    
    async def _extract_publication_features(self, document: Document) -> Dict[str, Any]:
        """Extract features for publication success prediction"""
        
        features = {}
        
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
        
        # Journal and timing features
        features['target_journal'] = document.target_journal
        features['days_until_deadline'] = (datetime.strptime(document.deadline, '%Y-%m-%d') - datetime.now()).days
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
    
    async def _identify_risk_factors(self, features: Dict[str, Any], document: Document) -> List[str]:
        """Identify publication risk factors"""
        
        risks = []
        
        if features['word_count'] < 2000:
            risks.append("Document length may be insufficient")
        
        if features['author_count'] == 1:
            risks.append("Single-author papers have lower acceptance rates")
        
        if not features['has_abstract']:
            risks.append("Missing abstract will impact editorial decision")
        
        if features['reference_count'] < 15:
            risks.append("Insufficient literature review")
        
        if features['days_until_deadline'] < 30:
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
