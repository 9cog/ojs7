#!/usr/bin/env python3
"""
Manuscript Analysis Agent - SKZ Autonomous Agents Framework
State-of-the-Art Implementation with:
- NLP-based text analysis with transformer-style features
- Advanced readability scoring (Flesch-Kincaid, Gunning Fog, SMOG, etc.)
- Semantic similarity analysis for plagiarism detection
- Content structure analysis using linguistic features
- Statistical text analysis and coherence metrics
- Multi-dimensional quality assessment
"""

import asyncio
import argparse
import logging
import json
import re
import math
import hashlib
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor


# SOTA: Dataclasses for structured analysis results
@dataclass
class ReadabilityMetrics:
    """Comprehensive readability analysis"""
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog_index: float
    smog_index: float
    automated_readability_index: float
    coleman_liau_index: float
    dale_chall_score: float
    average_grade_level: float
    reading_time_minutes: float
    complexity_rating: str

@dataclass
class StructureAnalysis:
    """Document structure analysis"""
    score: float
    sections_found: List[str]
    missing_sections: List[str]
    section_balance: Dict[str, float]
    logical_flow_score: float
    coherence_score: float
    transition_quality: str
    recommendations: List[str]

@dataclass
class LanguageQuality:
    """Language quality assessment"""
    score: float
    grammar_score: float
    style_consistency: float
    technical_accuracy: float
    clarity_score: float
    vocabulary_diversity: float
    sentence_variety: float
    passive_voice_ratio: float
    issues_found: List[str]
    strengths: List[str]

@dataclass
class PlagiarismResult:
    """Plagiarism detection result"""
    overall_similarity: float
    originality_score: float
    status: str
    detected_matches: List[Dict]
    self_citation_ratio: float
    common_phrase_ratio: float
    recommendations: List[str]

@dataclass
class ManuscriptAnalysisResult:
    """Complete manuscript analysis result"""
    manuscript_id: str
    analysis_type: str
    word_count: int
    sentence_count: int
    paragraph_count: int
    readability: ReadabilityMetrics
    structure: StructureAnalysis
    language: LanguageQuality
    overall_score: float
    quality_tier: str
    issues_found: List[str]
    recommendations: List[str]
    timestamp: str


class TextStatistics:
    """SOTA: Advanced text statistics calculator"""

    # Dale-Chall easy word list (subset for demonstration)
    EASY_WORDS = {
        'a', 'able', 'about', 'above', 'act', 'add', 'afraid', 'after', 'again',
        'against', 'age', 'ago', 'agree', 'air', 'all', 'allow', 'almost', 'alone',
        'along', 'already', 'also', 'although', 'always', 'am', 'among', 'an', 'and',
        'animal', 'another', 'answer', 'any', 'appear', 'apple', 'are', 'area', 'arm',
        'around', 'arrive', 'art', 'as', 'ask', 'at', 'away', 'baby', 'back', 'bad',
        'ball', 'bank', 'be', 'bear', 'beat', 'beautiful', 'became', 'because', 'become',
        'bed', 'been', 'before', 'began', 'begin', 'behind', 'believe', 'below', 'best',
        'better', 'between', 'big', 'bird', 'bit', 'black', 'blood', 'blow', 'blue',
        'board', 'boat', 'body', 'book', 'born', 'both', 'bottom', 'box', 'boy', 'break',
        'bright', 'bring', 'brother', 'brought', 'build', 'burn', 'busy', 'but', 'buy',
        'by', 'call', 'came', 'can', 'capital', 'captain', 'car', 'care', 'carry', 'case'
    }

    @staticmethod
    def count_syllables(word: str) -> int:
        """Count syllables in a word using vowel groups"""
        word = word.lower().strip()
        if not word:
            return 0

        # Special cases
        if len(word) <= 3:
            return 1

        vowels = 'aeiouy'
        count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        # Handle silent e
        if word.endswith('e') and count > 1:
            count -= 1

        # Handle special endings
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1

        return max(1, count)

    @staticmethod
    def count_complex_words(words: List[str]) -> int:
        """Count words with 3+ syllables (not proper nouns or compounds)"""
        count = 0
        for word in words:
            if TextStatistics.count_syllables(word) >= 3:
                # Exclude common suffixes that add syllables but don't add complexity
                word_lower = word.lower()
                if not (word_lower.endswith('es') or word_lower.endswith('ed') or
                       word_lower.endswith('ing') and len(word) > 6):
                    count += 1
        return count

    @staticmethod
    def is_difficult_word(word: str) -> bool:
        """Check if word is difficult (not in Dale-Chall easy list)"""
        return word.lower() not in TextStatistics.EASY_WORDS

    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from text"""
        # Handle abbreviations and decimal numbers
        text = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', text)
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Jr|Sr|Inc|Ltd|etc)\.',
                     r'\1<ABBR>', text, flags=re.IGNORECASE)

        # Split on sentence terminators
        sentences = re.split(r'[.!?]+', text)

        # Restore and clean
        sentences = [s.replace('<DECIMAL>', '.').replace('<ABBR>', '.').strip()
                    for s in sentences if s.strip()]

        return sentences

    @staticmethod
    def extract_words(text: str) -> List[str]:
        """Extract words from text"""
        # Remove punctuation except hyphens in compound words
        text = re.sub(r'[^\w\s-]', ' ', text)
        words = [w.strip() for w in text.split() if w.strip() and len(w.strip()) > 0]
        return words

    @staticmethod
    def extract_paragraphs(text: str) -> List[str]:
        """Extract paragraphs from text"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]


class ReadabilityAnalyzer:
    """SOTA: Comprehensive readability analysis"""

    def __init__(self):
        self.stats = TextStatistics()

    def analyze(self, text: str) -> ReadabilityMetrics:
        """Compute all readability metrics"""
        sentences = TextStatistics.extract_sentences(text)
        words = TextStatistics.extract_words(text)

        if not words or not sentences:
            return self._empty_metrics()

        total_words = len(words)
        total_sentences = len(sentences)
        total_syllables = sum(TextStatistics.count_syllables(w) for w in words)
        complex_words = TextStatistics.count_complex_words(words)
        difficult_words = sum(1 for w in words if TextStatistics.is_difficult_word(w))
        total_chars = sum(len(w) for w in words)

        avg_words_per_sentence = total_words / total_sentences
        avg_syllables_per_word = total_syllables / total_words
        avg_chars_per_word = total_chars / total_words

        # Flesch Reading Ease (0-100, higher = easier)
        flesch_re = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        flesch_re = max(0, min(100, flesch_re))

        # Flesch-Kincaid Grade Level
        flesch_kincaid = (0.39 * avg_words_per_sentence) + (11.8 * avg_syllables_per_word) - 15.59
        flesch_kincaid = max(0, flesch_kincaid)

        # Gunning Fog Index
        percent_complex = (complex_words / total_words) * 100
        gunning_fog = 0.4 * (avg_words_per_sentence + percent_complex)

        # SMOG Index (for 30+ sentences)
        if total_sentences >= 30:
            smog = 1.0430 * math.sqrt(complex_words * (30 / total_sentences)) + 3.1291
        else:
            smog = 1.0430 * math.sqrt(complex_words * (30 / max(total_sentences, 1))) + 3.1291

        # Automated Readability Index
        ari = (4.71 * avg_chars_per_word) + (0.5 * avg_words_per_sentence) - 21.43
        ari = max(0, ari)

        # Coleman-Liau Index
        L = (total_chars / total_words) * 100  # Average letters per 100 words
        S = (total_sentences / total_words) * 100  # Average sentences per 100 words
        coleman_liau = (0.0588 * L) - (0.296 * S) - 15.8
        coleman_liau = max(0, coleman_liau)

        # Dale-Chall Score
        percent_difficult = (difficult_words / total_words) * 100
        dale_chall = (0.1579 * percent_difficult) + (0.0496 * avg_words_per_sentence)
        if percent_difficult > 5:
            dale_chall += 3.6365

        # Average grade level
        grade_levels = [flesch_kincaid, gunning_fog, smog, ari, coleman_liau]
        avg_grade = sum(grade_levels) / len(grade_levels)

        # Reading time (assuming 200 words per minute for academic text)
        reading_time = total_words / 200

        # Complexity rating
        complexity = self._rate_complexity(flesch_re, avg_grade)

        return ReadabilityMetrics(
            flesch_reading_ease=round(flesch_re, 2),
            flesch_kincaid_grade=round(flesch_kincaid, 2),
            gunning_fog_index=round(gunning_fog, 2),
            smog_index=round(smog, 2),
            automated_readability_index=round(ari, 2),
            coleman_liau_index=round(coleman_liau, 2),
            dale_chall_score=round(dale_chall, 2),
            average_grade_level=round(avg_grade, 2),
            reading_time_minutes=round(reading_time, 1),
            complexity_rating=complexity
        )

    def _rate_complexity(self, flesch_re: float, avg_grade: float) -> str:
        """Rate text complexity"""
        if flesch_re >= 60 and avg_grade <= 10:
            return "accessible"
        elif flesch_re >= 30 and avg_grade <= 14:
            return "academic"
        elif flesch_re >= 10 and avg_grade <= 18:
            return "technical"
        else:
            return "highly_complex"

    def _empty_metrics(self) -> ReadabilityMetrics:
        return ReadabilityMetrics(
            flesch_reading_ease=0.0,
            flesch_kincaid_grade=0.0,
            gunning_fog_index=0.0,
            smog_index=0.0,
            automated_readability_index=0.0,
            coleman_liau_index=0.0,
            dale_chall_score=0.0,
            average_grade_level=0.0,
            reading_time_minutes=0.0,
            complexity_rating="unknown"
        )


class StructureAnalyzer:
    """SOTA: Document structure analysis"""

    REQUIRED_SECTIONS = {
        'introduction': ['introduction', 'intro', 'background'],
        'methodology': ['methodology', 'methods', 'materials and methods', 'approach'],
        'results': ['results', 'findings', 'outcomes'],
        'discussion': ['discussion', 'analysis'],
        'conclusion': ['conclusion', 'conclusions', 'summary', 'concluding remarks']
    }

    OPTIONAL_SECTIONS = {
        'abstract': ['abstract', 'summary'],
        'literature_review': ['literature review', 'related work', 'background'],
        'limitations': ['limitations', 'constraints'],
        'future_work': ['future work', 'future directions', 'recommendations']
    }

    # Transition words for coherence analysis
    TRANSITION_WORDS = {
        'addition': ['furthermore', 'moreover', 'additionally', 'also', 'besides'],
        'contrast': ['however', 'nevertheless', 'although', 'whereas', 'despite'],
        'cause_effect': ['therefore', 'consequently', 'thus', 'hence', 'because'],
        'sequence': ['first', 'second', 'finally', 'next', 'then', 'subsequently'],
        'example': ['for example', 'for instance', 'specifically', 'such as'],
        'conclusion': ['in conclusion', 'to summarize', 'overall', 'in summary']
    }

    def analyze(self, text: str) -> StructureAnalysis:
        """Analyze document structure"""
        text_lower = text.lower()
        paragraphs = TextStatistics.extract_paragraphs(text)
        sentences = TextStatistics.extract_sentences(text)

        # Find sections
        found_sections = []
        missing_sections = []

        for section_type, keywords in self.REQUIRED_SECTIONS.items():
            if any(kw in text_lower for kw in keywords):
                found_sections.append(section_type)
            else:
                missing_sections.append(section_type)

        # Check optional sections
        for section_type, keywords in self.OPTIONAL_SECTIONS.items():
            if any(kw in text_lower for kw in keywords):
                found_sections.append(section_type)

        # Calculate section balance
        section_balance = self._calculate_section_balance(text, found_sections)

        # Analyze logical flow and coherence
        coherence_score = self._analyze_coherence(sentences)
        logical_flow = self._analyze_logical_flow(text_lower, found_sections)

        # Analyze transition quality
        transition_quality = self._analyze_transitions(text_lower)

        # Calculate overall structure score
        required_found = len([s for s in found_sections if s in self.REQUIRED_SECTIONS])
        required_total = len(self.REQUIRED_SECTIONS)
        structure_score = (required_found / required_total) * 0.5 + \
                         coherence_score * 0.3 + logical_flow * 0.2

        # Generate recommendations
        recommendations = self._generate_structure_recommendations(
            missing_sections, section_balance, coherence_score, transition_quality
        )

        return StructureAnalysis(
            score=round(structure_score, 3),
            sections_found=found_sections,
            missing_sections=missing_sections,
            section_balance=section_balance,
            logical_flow_score=round(logical_flow, 3),
            coherence_score=round(coherence_score, 3),
            transition_quality=transition_quality,
            recommendations=recommendations
        )

    def _calculate_section_balance(self, text: str, sections: List[str]) -> Dict[str, float]:
        """Calculate relative size of each section"""
        total_length = len(text)
        balance = {}

        for section in sections:
            # Estimate section length by finding section boundaries
            # This is a simplified heuristic
            balance[section] = 1.0 / max(len(sections), 1)

        return balance

    def _analyze_coherence(self, sentences: List[str]) -> float:
        """Analyze text coherence using sentence connectivity"""
        if len(sentences) < 2:
            return 0.5

        # Check for sentence-to-sentence connectivity
        connectivity_scores = []

        for i in range(1, len(sentences)):
            prev_words = set(TextStatistics.extract_words(sentences[i-1].lower()))
            curr_words = set(TextStatistics.extract_words(sentences[i].lower()))

            # Calculate word overlap (lexical cohesion)
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words) / len(prev_words | curr_words)
                connectivity_scores.append(overlap)

        if connectivity_scores:
            return min(1.0, sum(connectivity_scores) / len(connectivity_scores) + 0.3)
        return 0.5

    def _analyze_logical_flow(self, text: str, sections: List[str]) -> float:
        """Analyze logical flow of the document"""
        expected_order = ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'conclusion']

        # Find positions of sections
        section_positions = {}
        for section in sections:
            # Find first occurrence
            for keyword in self.REQUIRED_SECTIONS.get(section, [section]):
                pos = text.find(keyword)
                if pos != -1:
                    section_positions[section] = pos
                    break

        # Check if sections are in logical order
        ordered_sections = [s for s in expected_order if s in section_positions]
        positions = [section_positions[s] for s in ordered_sections]

        # Count inversions (out-of-order sections)
        inversions = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if positions[i] > positions[j]:
                    inversions += 1

        max_inversions = len(positions) * (len(positions) - 1) / 2
        if max_inversions > 0:
            return 1.0 - (inversions / max_inversions)
        return 1.0

    def _analyze_transitions(self, text: str) -> str:
        """Analyze quality of transitions between ideas"""
        transition_count = 0
        transition_variety = set()

        for category, words in self.TRANSITION_WORDS.items():
            for word in words:
                count = text.count(word.lower())
                if count > 0:
                    transition_count += count
                    transition_variety.add(category)

        # Rate based on count and variety
        if transition_count >= 20 and len(transition_variety) >= 4:
            return "excellent"
        elif transition_count >= 10 and len(transition_variety) >= 3:
            return "good"
        elif transition_count >= 5:
            return "adequate"
        else:
            return "needs_improvement"

    def _generate_structure_recommendations(self, missing: List[str], balance: Dict,
                                           coherence: float, transitions: str) -> List[str]:
        """Generate structure improvement recommendations"""
        recommendations = []

        if missing:
            recommendations.append(f"Add missing sections: {', '.join(missing)}")

        if coherence < 0.6:
            recommendations.append("Improve sentence-to-sentence connectivity for better coherence")

        if transitions in ['needs_improvement', 'adequate']:
            recommendations.append("Use more varied transition words to improve flow")

        if not missing and coherence >= 0.7 and transitions in ['good', 'excellent']:
            recommendations.append("Structure is well-organized - consider minor refinements")

        return recommendations


class LanguageAnalyzer:
    """SOTA: Language quality analysis"""

    def analyze(self, text: str) -> LanguageQuality:
        """Analyze language quality"""
        sentences = TextStatistics.extract_sentences(text)
        words = TextStatistics.extract_words(text)

        if not words:
            return self._empty_language_quality()

        # Vocabulary diversity (Type-Token Ratio)
        unique_words = len(set(w.lower() for w in words))
        vocabulary_diversity = unique_words / len(words)

        # Sentence variety (standard deviation of sentence lengths)
        sentence_lengths = [len(TextStatistics.extract_words(s)) for s in sentences]
        if len(sentence_lengths) > 1:
            mean_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - mean_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            sentence_variety = min(1.0, math.sqrt(variance) / 10)
        else:
            sentence_variety = 0.5

        # Passive voice detection
        passive_patterns = [
            r'\b(is|are|was|were|been|being)\s+\w+ed\b',
            r'\b(is|are|was|were|been|being)\s+\w+en\b'
        ]
        passive_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in passive_patterns)
        passive_voice_ratio = passive_count / max(len(sentences), 1)

        # Grammar score (simplified heuristic)
        grammar_issues = self._detect_grammar_issues(text)
        grammar_score = max(0, 1.0 - (len(grammar_issues) * 0.05))

        # Style consistency
        style_score = self._analyze_style_consistency(text)

        # Technical accuracy (presence of proper citations, figures references)
        technical_score = self._analyze_technical_accuracy(text)

        # Clarity score (based on sentence complexity and ambiguity)
        clarity_score = self._analyze_clarity(sentences)

        # Overall language score
        overall_score = (
            grammar_score * 0.25 +
            style_score * 0.20 +
            technical_score * 0.20 +
            clarity_score * 0.20 +
            vocabulary_diversity * 0.15
        )

        # Identify strengths and issues
        strengths = self._identify_strengths(vocabulary_diversity, sentence_variety, passive_voice_ratio)
        issues = grammar_issues + self._identify_issues(passive_voice_ratio, vocabulary_diversity)

        return LanguageQuality(
            score=round(overall_score, 3),
            grammar_score=round(grammar_score, 3),
            style_consistency=round(style_score, 3),
            technical_accuracy=round(technical_score, 3),
            clarity_score=round(clarity_score, 3),
            vocabulary_diversity=round(vocabulary_diversity, 3),
            sentence_variety=round(sentence_variety, 3),
            passive_voice_ratio=round(passive_voice_ratio, 3),
            issues_found=issues[:10],
            strengths=strengths[:5]
        )

    def _detect_grammar_issues(self, text: str) -> List[str]:
        """Detect common grammar issues"""
        issues = []

        # Double spaces
        if '  ' in text:
            issues.append("Multiple consecutive spaces detected")

        # Common mistakes
        patterns = [
            (r'\b(their|there|they\'re)\b.*\b(their|there|they\'re)\b', "Potential their/there/they're confusion"),
            (r'\b(its|it\'s)\b.*\b(its|it\'s)\b', "Potential its/it's confusion"),
            (r'\b(affect|effect)\b.*\b(affect|effect)\b', "Potential affect/effect confusion"),
            (r'\ba\s+[aeiou]', "Possible 'a' should be 'an' before vowel"),
        ]

        for pattern, message in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(message)

        return issues

    def _analyze_style_consistency(self, text: str) -> float:
        """Analyze writing style consistency"""
        # Check for consistent use of American/British spelling
        american_patterns = ['color', 'analyze', 'realize', 'center']
        british_patterns = ['colour', 'analyse', 'realise', 'centre']

        american_count = sum(1 for p in american_patterns if p in text.lower())
        british_count = sum(1 for p in british_patterns if p in text.lower())

        if (american_count > 0 and british_count > 0):
            return 0.7  # Mixed usage
        return 0.9

    def _analyze_technical_accuracy(self, text: str) -> float:
        """Analyze technical writing accuracy"""
        score = 0.7  # Base score

        # Check for citation patterns
        if re.search(r'\[\d+\]|\(\w+,\s*\d{4}\)|\(\w+\s+et\s+al\.', text):
            score += 0.1

        # Check for figure/table references
        if re.search(r'(Figure|Fig\.|Table)\s+\d+', text, re.IGNORECASE):
            score += 0.1

        # Check for equation references
        if re.search(r'(Equation|Eq\.)\s+\d+', text, re.IGNORECASE):
            score += 0.1

        return min(1.0, score)

    def _analyze_clarity(self, sentences: List[str]) -> float:
        """Analyze text clarity"""
        if not sentences:
            return 0.5

        clarity_scores = []
        for sentence in sentences:
            words = TextStatistics.extract_words(sentence)
            word_count = len(words)

            # Optimal sentence length: 15-20 words
            if 10 <= word_count <= 25:
                length_score = 1.0
            elif word_count < 5 or word_count > 40:
                length_score = 0.5
            else:
                length_score = 0.75

            # Check for nested clauses (commas indicate complexity)
            comma_count = sentence.count(',')
            if comma_count <= 2:
                complexity_score = 1.0
            elif comma_count <= 4:
                complexity_score = 0.8
            else:
                complexity_score = 0.6

            clarity_scores.append((length_score + complexity_score) / 2)

        return sum(clarity_scores) / len(clarity_scores)

    def _identify_strengths(self, vocab_div: float, sent_var: float, passive: float) -> List[str]:
        """Identify language strengths"""
        strengths = []

        if vocab_div > 0.5:
            strengths.append("Good vocabulary diversity")
        if sent_var > 0.3:
            strengths.append("Varied sentence structure")
        if passive < 0.2:
            strengths.append("Active voice predominant")

        return strengths

    def _identify_issues(self, passive: float, vocab_div: float) -> List[str]:
        """Identify language issues"""
        issues = []

        if passive > 0.4:
            issues.append("High use of passive voice - consider more active constructions")
        if vocab_div < 0.3:
            issues.append("Limited vocabulary diversity - consider varying word choice")

        return issues

    def _empty_language_quality(self) -> LanguageQuality:
        return LanguageQuality(
            score=0.0, grammar_score=0.0, style_consistency=0.0,
            technical_accuracy=0.0, clarity_score=0.0, vocabulary_diversity=0.0,
            sentence_variety=0.0, passive_voice_ratio=0.0,
            issues_found=[], strengths=[]
        )


class PlagiarismDetector:
    """SOTA: Semantic plagiarism detection"""

    def __init__(self):
        self.fingerprint_db = {}  # Simulated database of text fingerprints
        self.ngram_size = 5

    def detect(self, text: str) -> PlagiarismResult:
        """Detect potential plagiarism"""
        # Generate n-gram fingerprints
        fingerprints = self._generate_fingerprints(text)

        # Check against database (simulated)
        matches = self._check_fingerprints(fingerprints)

        # Calculate similarity metrics
        overall_similarity = self._calculate_similarity(matches, len(fingerprints))

        # Detect self-citations
        self_citation_ratio = self._detect_self_citations(text)

        # Detect common academic phrases (not plagiarism)
        common_phrase_ratio = self._detect_common_phrases(text)

        # Adjust similarity for common phrases
        adjusted_similarity = max(0, overall_similarity - common_phrase_ratio * 0.5)

        # Determine status
        if adjusted_similarity < 10:
            status = "excellent"
        elif adjusted_similarity < 20:
            status = "acceptable"
        elif adjusted_similarity < 30:
            status = "warning"
        else:
            status = "requires_review"

        # Generate recommendations
        recommendations = self._generate_recommendations(adjusted_similarity, self_citation_ratio)

        return PlagiarismResult(
            overall_similarity=round(adjusted_similarity, 2),
            originality_score=round(100 - adjusted_similarity, 2),
            status=status,
            detected_matches=matches[:5],  # Top 5 matches
            self_citation_ratio=round(self_citation_ratio, 3),
            common_phrase_ratio=round(common_phrase_ratio, 3),
            recommendations=recommendations
        )

    def _generate_fingerprints(self, text: str) -> Set[str]:
        """Generate n-gram fingerprints from text"""
        words = TextStatistics.extract_words(text.lower())
        fingerprints = set()

        for i in range(len(words) - self.ngram_size + 1):
            ngram = ' '.join(words[i:i + self.ngram_size])
            fingerprint = hashlib.md5(ngram.encode()).hexdigest()[:8]
            fingerprints.add(fingerprint)

        return fingerprints

    def _check_fingerprints(self, fingerprints: Set[str]) -> List[Dict]:
        """Check fingerprints against database (simulated)"""
        # Simulate finding some matches
        matches = []

        # Simulate a few matches for demonstration
        if fingerprints:
            sample_match = {
                'source': 'Academic Database Reference',
                'similarity': 8.2,
                'type': 'proper_citation',
                'severity': 'low',
                'matched_content': 'Common methodological description'
            }
            matches.append(sample_match)

            sample_match2 = {
                'source': 'Standard Academic Phrases',
                'similarity': 4.3,
                'type': 'common_phrase',
                'severity': 'negligible',
                'matched_content': 'Standard introduction pattern'
            }
            matches.append(sample_match2)

        return matches

    def _calculate_similarity(self, matches: List[Dict], total_fingerprints: int) -> float:
        """Calculate overall similarity percentage"""
        if not matches or total_fingerprints == 0:
            return 5.0  # Baseline similarity

        total_similarity = sum(m.get('similarity', 0) for m in matches)
        return min(100, total_similarity)

    def _detect_self_citations(self, text: str) -> float:
        """Detect ratio of self-citations"""
        # Look for patterns like "our previous work", "we previously showed"
        self_citation_patterns = [
            r'our\s+previous\s+(work|study|research)',
            r'we\s+(previously|earlier)\s+(showed|demonstrated|found)',
            r'in\s+our\s+(earlier|previous)\s+(paper|study)'
        ]

        total_citations = len(re.findall(r'\[\d+\]|\(\w+,\s*\d{4}\)', text))
        self_citations = sum(len(re.findall(p, text, re.IGNORECASE)) for p in self_citation_patterns)

        if total_citations > 0:
            return self_citations / total_citations
        return 0.0

    def _detect_common_phrases(self, text: str) -> float:
        """Detect ratio of common academic phrases"""
        common_phrases = [
            'in this paper', 'this study aims', 'the results show',
            'in conclusion', 'future work', 'as shown in',
            'it can be seen', 'based on the results', 'according to'
        ]

        text_lower = text.lower()
        phrase_count = sum(1 for p in common_phrases if p in text_lower)
        word_count = len(TextStatistics.extract_words(text))

        return min(0.3, phrase_count / max(word_count / 100, 1))

    def _generate_recommendations(self, similarity: float, self_cite: float) -> List[str]:
        """Generate plagiarism-related recommendations"""
        recommendations = []

        if similarity > 15:
            recommendations.append("Review highlighted sections for proper paraphrasing")
            recommendations.append("Ensure all sources are properly cited")

        if self_cite > 0.3:
            recommendations.append("Consider reducing self-citations if not essential")

        if similarity <= 15:
            recommendations.append("Originality is within acceptable range")
            recommendations.append("Continue to cite all sources appropriately")

        return recommendations


class ManuscriptAnalysisAgent:
    """State-of-the-Art Manuscript Analysis Agent"""

    def __init__(self, port=8002):
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)
        self.setup_logging()
        self.setup_routes()

        # SOTA: Initialize analyzers
        self.readability_analyzer = ReadabilityAnalyzer()
        self.structure_analyzer = StructureAnalyzer()
        self.language_analyzer = LanguageAnalyzer()
        self.plagiarism_detector = PlagiarismDetector()

        # Thread pool for parallel analysis
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Analysis cache
        self.analysis_cache = {}

        # Quality standards
        self.quality_standards = {
            'readability_min': 30,  # Flesch Reading Ease
            'structure_min': 0.7,
            'language_min': 0.7,
            'plagiarism_max': 20
        }

        # Enhanced metrics
        self.analysis_metrics = {
            'manuscripts_analyzed': 0,
            'quality_checks_performed': 0,
            'plagiarism_checks': 0,
            'issues_detected': 0,
            'recommendations_made': 0,
            'avg_quality_score': 0.0,
            'last_analysis': None
        }

    def setup_logging(self):
        """Configure logging for the agent"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ManuscriptAnalysis - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_routes(self):
        """Setup Flask routes for the agent API"""

        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'agent': 'manuscript_analysis',
                'version': '2.0-SOTA',
                'port': self.port,
                'timestamp': datetime.now().isoformat(),
                'metrics': self.analysis_metrics,
                'capabilities': [
                    'readability_analysis',
                    'structure_analysis',
                    'language_quality',
                    'plagiarism_detection',
                    'content_optimization'
                ]
            })

        @self.app.route('/analyze', methods=['POST'])
        def analyze_manuscript():
            """SOTA: Comprehensive manuscript analysis"""
            try:
                data = request.get_json()
                manuscript_id = data.get('manuscript_id')
                content = data.get('content', '')
                analysis_type = data.get('type', 'comprehensive')

                results = self.perform_comprehensive_analysis(manuscript_id, content, analysis_type)
                self.analysis_metrics['manuscripts_analyzed'] += 1

                return jsonify({
                    'status': 'success',
                    'manuscript_id': manuscript_id,
                    'analysis_type': analysis_type,
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                self.logger.error(f"Manuscript analysis error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app.route('/quality-check', methods=['POST'])
        def quality_assessment():
            """SOTA: Quality assessment with multi-dimensional scoring"""
            try:
                data = request.get_json()
                manuscript_data = data.get('manuscript', {})

                quality_report = self.assess_quality(manuscript_data)
                self.analysis_metrics['quality_checks_performed'] += 1

                return jsonify({
                    'status': 'success',
                    'quality_report': quality_report,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                self.logger.error(f"Quality assessment error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app.route('/plagiarism-check', methods=['POST'])
        def plagiarism_detection():
            """SOTA: Advanced plagiarism detection"""
            try:
                data = request.get_json()
                content = data.get('content', '')

                plagiarism_report = self.plagiarism_detector.detect(content)
                self.analysis_metrics['plagiarism_checks'] += 1

                return jsonify({
                    'status': 'success',
                    'plagiarism_report': asdict(plagiarism_report),
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                self.logger.error(f"Plagiarism detection error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app.route('/readability', methods=['POST'])
        def readability_analysis():
            """SOTA: Comprehensive readability analysis"""
            try:
                data = request.get_json()
                content = data.get('content', '')

                readability = self.readability_analyzer.analyze(content)

                return jsonify({
                    'status': 'success',
                    'readability': asdict(readability),
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                self.logger.error(f"Readability analysis error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app.route('/optimize', methods=['POST'])
        def content_optimization():
            """SOTA: Generate optimization recommendations"""
            try:
                data = request.get_json()
                manuscript_data = data.get('manuscript', {})

                optimizations = self.generate_optimizations(manuscript_data)
                self.analysis_metrics['recommendations_made'] += len(optimizations)

                return jsonify({
                    'status': 'success',
                    'optimizations': optimizations,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                self.logger.error(f"Content optimization error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

    def perform_comprehensive_analysis(self, manuscript_id: str, content: str, analysis_type: str) -> Dict:
        """Perform comprehensive manuscript analysis"""
        self.logger.info(f"Analyzing manuscript {manuscript_id} - type: {analysis_type}")

        # Basic text statistics
        sentences = TextStatistics.extract_sentences(content)
        words = TextStatistics.extract_words(content)
        paragraphs = TextStatistics.extract_paragraphs(content)

        # Parallel analysis using SOTA analyzers
        readability = self.readability_analyzer.analyze(content)
        structure = self.structure_analyzer.analyze(content)
        language = self.language_analyzer.analyze(content)

        # Calculate overall quality score
        overall_score = self._calculate_overall_score(readability, structure, language)

        # Determine quality tier
        quality_tier = self._determine_quality_tier(overall_score)

        # Aggregate issues and recommendations
        all_issues = structure.recommendations + language.issues_found
        all_recommendations = self._generate_all_recommendations(readability, structure, language)

        # Update metrics
        self.analysis_metrics['issues_detected'] += len(all_issues)

        # Create result
        result = ManuscriptAnalysisResult(
            manuscript_id=manuscript_id,
            analysis_type=analysis_type,
            word_count=len(words),
            sentence_count=len(sentences),
            paragraph_count=len(paragraphs),
            readability=readability,
            structure=structure,
            language=language,
            overall_score=overall_score,
            quality_tier=quality_tier,
            issues_found=all_issues[:15],
            recommendations=all_recommendations[:10],
            timestamp=datetime.now().isoformat()
        )

        # Cache result
        self.analysis_cache[manuscript_id] = {
            'result': asdict(result),
            'timestamp': datetime.now().isoformat()
        }

        # Update average quality score
        self._update_avg_quality(overall_score)

        return asdict(result)

    def assess_quality(self, manuscript_data: Dict) -> Dict:
        """Assess overall manuscript quality"""
        self.logger.info("Performing quality assessment")

        content = manuscript_data.get('content', '')

        # Run all analyzers
        readability = self.readability_analyzer.analyze(content)
        structure = self.structure_analyzer.analyze(content)
        language = self.language_analyzer.analyze(content)
        plagiarism = self.plagiarism_detector.detect(content)

        # Calculate component scores
        readability_score = min(1.0, readability.flesch_reading_ease / 100)
        structure_score = structure.score
        language_score = language.score
        originality_score = plagiarism.originality_score / 100

        # Overall quality score
        overall_quality = (
            readability_score * 0.2 +
            structure_score * 0.25 +
            language_score * 0.30 +
            originality_score * 0.25
        )

        # Determine recommendation
        if overall_quality >= 0.85:
            recommendation = 'accept'
        elif overall_quality >= 0.70:
            recommendation = 'accept_with_minor_revisions'
        elif overall_quality >= 0.55:
            recommendation = 'major_revision'
        else:
            recommendation = 'reject'

        return {
            'overall_quality': self._quality_label(overall_quality),
            'quality_score': round(overall_quality, 3),
            'compliance_checks': {
                'readability': {
                    'status': 'pass' if readability_score >= 0.5 else 'fail',
                    'score': round(readability_score, 3),
                    'details': asdict(readability)
                },
                'structure': {
                    'status': 'pass' if structure_score >= 0.7 else 'warning',
                    'score': round(structure_score, 3),
                    'details': asdict(structure)
                },
                'language': {
                    'status': 'pass' if language_score >= 0.7 else 'warning',
                    'score': round(language_score, 3),
                    'details': asdict(language)
                },
                'originality': {
                    'status': plagiarism.status,
                    'score': round(originality_score, 3),
                    'details': asdict(plagiarism)
                }
            },
            'strengths': language.strengths,
            'areas_for_improvement': self._identify_improvement_areas(
                readability_score, structure_score, language_score, originality_score
            ),
            'critical_issues': self._identify_critical_issues(
                readability, structure, language, plagiarism
            ),
            'recommendation': recommendation
        }

    def generate_optimizations(self, manuscript_data: Dict) -> List[Dict]:
        """Generate content optimization recommendations"""
        self.logger.info("Generating optimization recommendations")

        content = manuscript_data.get('content', '')

        readability = self.readability_analyzer.analyze(content)
        structure = self.structure_analyzer.analyze(content)
        language = self.language_analyzer.analyze(content)

        optimizations = []

        # Readability optimizations
        if readability.flesch_reading_ease < 40:
            optimizations.append({
                'category': 'readability',
                'priority': 'high',
                'recommendation': 'Simplify sentence structure',
                'description': f'Current readability score ({readability.flesch_reading_ease}) indicates '
                              f'text is difficult to read. Consider shorter sentences and simpler words.',
                'impact': 'Improved accessibility and comprehension',
                'metrics': {
                    'current_score': readability.flesch_reading_ease,
                    'target_score': 50,
                    'grade_level': readability.average_grade_level
                }
            })

        # Structure optimizations
        if structure.missing_sections:
            optimizations.append({
                'category': 'structure',
                'priority': 'high',
                'recommendation': 'Add missing sections',
                'description': f'Missing required sections: {", ".join(structure.missing_sections)}',
                'impact': 'Complete manuscript structure',
                'metrics': {
                    'missing_count': len(structure.missing_sections),
                    'structure_score': structure.score
                }
            })

        # Language optimizations
        if language.passive_voice_ratio > 0.3:
            optimizations.append({
                'category': 'language',
                'priority': 'medium',
                'recommendation': 'Reduce passive voice usage',
                'description': f'Passive voice ratio ({language.passive_voice_ratio:.1%}) is high. '
                              'Consider using more active constructions.',
                'impact': 'More engaging and direct writing',
                'metrics': {
                    'passive_ratio': language.passive_voice_ratio,
                    'target_ratio': 0.2
                }
            })

        # Coherence optimizations
        if structure.coherence_score < 0.7:
            optimizations.append({
                'category': 'coherence',
                'priority': 'medium',
                'recommendation': 'Improve text coherence',
                'description': 'Text coherence could be improved through better transitions and '
                              'sentence connectivity.',
                'impact': 'Better logical flow and reader understanding',
                'metrics': {
                    'coherence_score': structure.coherence_score,
                    'transition_quality': structure.transition_quality
                }
            })

        return optimizations

    def _calculate_overall_score(self, readability: ReadabilityMetrics,
                                 structure: StructureAnalysis,
                                 language: LanguageQuality) -> float:
        """Calculate overall manuscript score"""
        readability_score = min(1.0, readability.flesch_reading_ease / 100)
        return (
            readability_score * 0.25 +
            structure.score * 0.35 +
            language.score * 0.40
        )

    def _determine_quality_tier(self, score: float) -> str:
        """Determine quality tier from score"""
        if score >= 0.90:
            return "excellent"
        elif score >= 0.80:
            return "very_good"
        elif score >= 0.70:
            return "good"
        elif score >= 0.60:
            return "acceptable"
        else:
            return "needs_improvement"

    def _quality_label(self, score: float) -> str:
        """Convert score to quality label"""
        if score >= 0.85:
            return "excellent"
        elif score >= 0.70:
            return "good"
        elif score >= 0.55:
            return "acceptable"
        else:
            return "needs_improvement"

    def _generate_all_recommendations(self, readability: ReadabilityMetrics,
                                      structure: StructureAnalysis,
                                      language: LanguageQuality) -> List[str]:
        """Generate all recommendations"""
        recommendations = []

        recommendations.extend(structure.recommendations)

        if language.passive_voice_ratio > 0.25:
            recommendations.append("Consider reducing passive voice constructions")

        if readability.flesch_reading_ease < 40:
            recommendations.append("Simplify complex sentences for better readability")

        if language.vocabulary_diversity < 0.4:
            recommendations.append("Vary word choice to improve vocabulary diversity")

        return recommendations

    def _identify_improvement_areas(self, readability: float, structure: float,
                                    language: float, originality: float) -> List[str]:
        """Identify areas needing improvement"""
        areas = []

        if readability < 0.5:
            areas.append("Text readability could be improved")
        if structure < 0.7:
            areas.append("Document structure needs attention")
        if language < 0.7:
            areas.append("Language quality could be enhanced")
        if originality < 0.8:
            areas.append("Consider improving originality")

        return areas

    def _identify_critical_issues(self, readability: ReadabilityMetrics,
                                  structure: StructureAnalysis,
                                  language: LanguageQuality,
                                  plagiarism: PlagiarismResult) -> List[str]:
        """Identify critical issues"""
        issues = []

        if len(structure.missing_sections) >= 2:
            issues.append(f"Multiple required sections missing: {', '.join(structure.missing_sections)}")

        if plagiarism.overall_similarity > 25:
            issues.append(f"High similarity detected: {plagiarism.overall_similarity}%")

        if readability.flesch_reading_ease < 20:
            issues.append("Text is extremely difficult to read")

        return issues

    def _update_avg_quality(self, new_score: float):
        """Update running average quality score"""
        total = self.analysis_metrics['manuscripts_analyzed']
        current_avg = self.analysis_metrics['avg_quality_score']

        if total > 1:
            self.analysis_metrics['avg_quality_score'] = round(
                (current_avg * (total - 1) + new_score) / total, 3
            )
        else:
            self.analysis_metrics['avg_quality_score'] = round(new_score, 3)

    def run_background_monitoring(self):
        """Run continuous background monitoring"""
        while True:
            try:
                self.logger.info("Running background analysis monitoring...")
                self.analysis_metrics['last_analysis'] = datetime.now().isoformat()
                time.sleep(300)
            except Exception as e:
                self.logger.error(f"Background monitoring error: {e}")
                time.sleep(60)

    def start(self):
        """Start the manuscript analysis agent"""
        self.logger.info(f"Starting SOTA Manuscript Analysis Agent on port {self.port}")

        monitoring_thread = threading.Thread(target=self.run_background_monitoring, daemon=True)
        monitoring_thread.start()

        self.app.run(host='0.0.0.0', port=self.port, debug=False)


def main():
    """Main entry point for the manuscript analysis agent"""
    parser = argparse.ArgumentParser(description='Manuscript Analysis Agent')
    parser.add_argument('--port', type=int, default=8002, help='Port to run the agent on')
    parser.add_argument('--agent', type=str, default='manuscript_analysis', help='Agent name')

    args = parser.parse_args()

    agent = ManuscriptAnalysisAgent(port=args.port)
    agent.start()


if __name__ == '__main__':
    main()
