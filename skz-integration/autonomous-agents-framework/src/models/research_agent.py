"""
Research Discovery Agent Implementation - Enhanced with Critical ML Features
Specialized agent for literature discovery, trend analysis, and research gap identification
Implements Phase 2 Critical Features: Vector DB Integration, NLP Pipeline, Trend Prediction ML
"""

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests
import pickle
import logging
import os

# Optional dependencies - handle gracefully if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from .enhanced_agent import EnhancedAgent
    from .memory_system import PersistentMemorySystem
    from .ml_decision_engine import DecisionEngine, DecisionContext
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Critical Feature 1: Vector Database Integration for Research Content"""
    
    def __init__(self, embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 storage_type: str = "chromadb", index_type: str = "hnsw"):
        self.embeddings_model = embeddings_model
        self.storage_type = storage_type
        self.index_type = index_type
        
        # Initialize sentence transformer
        self.sentence_transformer = None
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer(embeddings_model.replace("sentence-transformers/", ""))
            logger.info(f"Using sentence transformer: {embeddings_model}")
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to TF-IDF")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer {embeddings_model}: {e}")
        
        # Initialize storage backend
        self.chromadb_client = None
        self.chromadb_collection = None
        self.document_vectors = {}
        self.document_metadata = {}
        
        if storage_type == "chromadb" and self.sentence_transformer is not None:
            try:
                import chromadb
                self.chromadb_client = chromadb.PersistentClient(path="./research_vectordb")
                
                # Configure HNSW index if specified
                metadata = {}
                if index_type == "hnsw":
                    metadata["hnsw:space"] = "cosine"
                
                self.chromadb_collection = self.chromadb_client.get_or_create_collection(
                    name="research_documents",
                    metadata=metadata
                )
                logger.info(f"Using ChromaDB with {index_type} index")
            except ImportError:
                logger.warning("chromadb not available, falling back to in-memory storage")
                self.storage_type = "sklearn"
            except Exception as e:
                logger.warning(f"Could not initialize ChromaDB: {e}, falling back to in-memory storage")
                self.storage_type = "sklearn"
        
        # Fallback to TF-IDF if advanced options not available
        if self.sentence_transformer is None or self.storage_type == "sklearn":
            if SKLEARN_AVAILABLE:
                self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                self.is_fitted = False
                logger.info("Using TF-IDF vectorization as fallback")
            else:
                logger.warning("scikit-learn not available, using simple text storage")
                self.simple_storage = True
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add research documents to vector database"""
        try:
            texts = []
            for doc in documents:
                text = f"{doc.get('title', '')} {doc.get('abstract', '')} {' '.join(doc.get('keywords', []))}"
                texts.append(text)
                self.document_metadata[doc.get('id', len(texts))] = doc
            
            if self.chromadb_collection is not None and self.sentence_transformer is not None:
                # Use ChromaDB with sentence transformers
                vectors = self.sentence_transformer.encode(texts)
                ids = [str(doc.get('id', f"doc_{i}")) for i, doc in enumerate(documents)]
                metadatas = [doc for doc in documents]
                
                self.chromadb_collection.add(
                    embeddings=vectors.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added {len(documents)} documents to ChromaDB")
                
            elif self.sentence_transformer is not None:
                # Use in-memory storage with sentence transformers
                vectors = self.sentence_transformer.encode(texts)
                for i, doc in enumerate(documents):
                    doc_id = doc.get('id', len(self.document_vectors))
                    self.document_vectors[doc_id] = vectors[i]
                logger.info(f"Added {len(documents)} documents to in-memory vector storage")
                
            elif SKLEARN_AVAILABLE and hasattr(self, 'vectorizer'):
                # Use TF-IDF fallback
                if not self.is_fitted:
                    vectors = self.vectorizer.fit_transform(texts)
                    self.is_fitted = True
                else:
                    vectors = self.vectorizer.transform(texts)
                
                for i, doc in enumerate(documents):
                    doc_id = doc.get('id', len(self.document_vectors))
                    self.document_vectors[doc_id] = vectors[i]
                logger.info(f"Added {len(documents)} documents using TF-IDF")
                
            else:
                # Simple text storage fallback
                for i, doc in enumerate(documents):
                    doc_id = doc.get('id', len(self.document_vectors))
                    text = f"{doc.get('title', '')} {doc.get('abstract', '')} {' '.join(doc.get('keywords', []))}"
                    self.document_vectors[doc_id] = text  # Store text directly
                logger.info(f"Added {len(documents)} documents using simple text storage")
            
            return True
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {e}")
            return False
    
    def search_similar(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity"""
        try:
            if self.chromadb_collection is not None and self.sentence_transformer is not None:
                # Use ChromaDB for search
                query_embedding = self.sentence_transformer.encode([query])
                results = self.chromadb_collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=limit
                )
                
                similarities = []
                if results['distances'] and results['metadatas']:
                    for i, (distance, metadata) in enumerate(zip(results['distances'][0], results['metadatas'][0])):
                        # Convert distance to similarity (ChromaDB returns distances)
                        similarity = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)
                        similarities.append({
                            'doc_id': results['ids'][0][i] if results['ids'] else f"doc_{i}",
                            'similarity': float(similarity),
                            'metadata': metadata or {}
                        })
                return similarities
                
            elif self.sentence_transformer is not None and self.document_vectors:
                # Use in-memory storage with sentence transformers
                query_vector = self.sentence_transformer.encode([query])[0]
                similarities = []
                
                for doc_id, doc_vector in self.document_vectors.items():
                    # Compute cosine similarity
                    if NUMPY_AVAILABLE:
                        similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
                    else:
                        # Simple dot product approximation
                        similarity = sum(a * b for a, b in zip(query_vector, doc_vector)) / 1000.0
                    similarities.append({
                        'doc_id': doc_id,
                        'similarity': float(similarity),
                        'metadata': self.document_metadata.get(doc_id, {})
                    })
                
                # Sort by similarity and return top results
                similarities.sort(key=lambda x: x['similarity'], reverse=True)
                return similarities[:limit]
                
            elif hasattr(self, 'is_fitted') and self.is_fitted and self.document_vectors and SKLEARN_AVAILABLE:
                # Use TF-IDF fallback
                query_vector = self.vectorizer.transform([query])
                similarities = []
                
                for doc_id, doc_vector in self.document_vectors.items():
                    similarity = cosine_similarity(query_vector, doc_vector)[0][0]
                    similarities.append({
                        'doc_id': doc_id,
                        'similarity': similarity,
                        'metadata': self.document_metadata.get(doc_id, {})
                    })
                
                # Sort by similarity and return top results
                similarities.sort(key=lambda x: x['similarity'], reverse=True)
                return similarities[:limit]
                
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []

class DocumentProcessor:
    """Critical Feature 2: NLP Pipeline for Document Understanding"""
    
    def __init__(self, extractors: List[str] = None, classifiers: List[str] = None, summarizers: List[str] = None):
        self.extractors = extractors or ["entities", "concepts", "relationships"]
        self.classifiers = classifiers or ["topic", "quality", "novelty"]
        self.summarizers = summarizers or ["abstract", "key_findings"]
        self.topic_classifier = None
        self.quality_classifier = None
        
        # Validate that all required capabilities are supported
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate that all specified extractors, classifiers, and summarizers are supported"""
        supported_extractors = ["entities", "concepts", "relationships"]
        supported_classifiers = ["topic", "quality", "novelty"]
        supported_summarizers = ["abstract", "key_findings"]
        
        for extractor in self.extractors:
            if extractor not in supported_extractors:
                logger.warning(f"Unsupported extractor: {extractor}")
        
        for classifier in self.classifiers:
            if classifier not in supported_classifiers:
                logger.warning(f"Unsupported classifier: {classifier}")
        
        for summarizer in self.summarizers:
            if summarizer not in supported_summarizers:
                logger.warning(f"Unsupported summarizer: {summarizer}")
    
    def process_document(self, text: str) -> Dict[str, Any]:
        """Process a document with all configured extractors, classifiers, and summarizers"""
        try:
            results = {
                'processing_timestamp': datetime.now().isoformat(),
                'text_length': len(text),
                'word_count': len(text.split())
            }
            
            # Run extractors
            if "entities" in self.extractors:
                results['entities'] = self.extract_entities(text)
            if "concepts" in self.extractors:
                results['concepts'] = self.extract_concepts(text)
            if "relationships" in self.extractors:
                results['relationships'] = self.extract_relationships(text)
            
            # Run classifiers
            if "topic" in self.classifiers:
                results['topic'] = self.classify_topic(text)
            if "quality" in self.classifiers:
                results['quality_score'] = self.assess_quality(text)
            if "novelty" in self.classifiers:
                results['novelty_score'] = self.assess_novelty(text)
            
            # Run summarizers
            if "abstract" in self.summarizers:
                results['abstract_summary'] = self.summarize_abstract(text)
            if "key_findings" in self.summarizers:
                results['key_findings'] = self.extract_key_findings(text)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {"error": str(e)}
        
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        # Simple regex-based entity extraction for basic functionality
        entities = []
        
        # Extract potential research terms (capitalized words/phrases)
        research_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(research_terms)
        
        # Extract acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        entities.extend(acronyms)
        
        # Extract compound technical terms (like "machine learning")
        technical_terms = re.findall(r'\b(?:machine learning|natural language processing|artificial intelligence|deep learning|neural network|data science|computer science|information technology)\b', text, re.IGNORECASE)
        entities.extend(technical_terms)
        
        # Extract terms that appear with common research indicators
        research_indicators = re.findall(r'\b(?:study|research|analysis|investigation|method|approach|technique|algorithm|model|system|framework|methodology)\b', text, re.IGNORECASE)
        entities.extend(research_indicators)
        
        return list(set(entities))
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Use TF-IDF to identify important terms
        try:
            vectorizer = TfidfVectorizer(max_features=20, stop_words='english', 
                                       ngram_range=(1, 3))
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top concepts
            concept_scores = list(zip(feature_names, scores))
            concept_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [concept for concept, score in concept_scores[:10] if score > 0]
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []
    
    def classify_topic(self, text: str) -> str:
        """Classify document topic"""
        # Simple keyword-based topic classification
        topics = {
            'machine_learning': ['machine learning', 'neural network', 'deep learning', 'AI', 'algorithm'],
            'skin_research': ['skin', 'dermatology', 'cosmetic', 'skincare', 'topical'],
            'medical': ['medical', 'clinical', 'patient', 'treatment', 'therapy'],
            'chemistry': ['chemical', 'compound', 'synthesis', 'molecular', 'reaction']
        }
        
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in topics.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            topic_scores[topic] = score
        
        return max(topic_scores, key=topic_scores.get) if topic_scores else 'general'
    
    def assess_quality(self, text: str) -> float:
        """Assess document quality"""
        # Simple quality metrics
        quality_score = 0.5  # Base score
        
        # Length check
        word_count = len(text.split())
        if word_count > 100:
            quality_score += 0.1
        if word_count > 500:
            quality_score += 0.1
        
        # Reference indicators
        if re.search(r'\[\d+\]|\(\d{4}\)', text):
            quality_score += 0.15
        
        # Methodology indicators
        methodology_terms = ['method', 'analysis', 'experiment', 'study', 'research', 'data']
        method_score = sum(1 for term in methodology_terms if term in text.lower())
        quality_score += min(method_score * 0.05, 0.15)
        
        return min(quality_score, 1.0)
    
    def assess_novelty(self, text: str) -> float:
        """Assess document novelty"""
        # Simple novelty indicators
        novelty_score = 0.5  # Base score
        
        novelty_terms = ['novel', 'new', 'innovative', 'first', 'breakthrough', 'unprecedented']
        for term in novelty_terms:
            if term in text.lower():
                novelty_score += 0.1
        
        return min(novelty_score, 1.0)
    
    def extract_relationships(self, text: str) -> List[Dict[str, str]]:
        """Extract relationships between entities and concepts"""
        try:
            relationships = []
            
            # Extract entities and concepts first
            entities = self.extract_entities(text)
            concepts = self.extract_concepts(text)
            
            # Simple relationship patterns
            relationship_patterns = [
                (r'(\w+)\s+(?:affects|influences|impacts|modifies)\s+(\w+)', 'affects'),
                (r'(\w+)\s+(?:causes|leads to|results in)\s+(\w+)', 'causes'),
                (r'(\w+)\s+(?:correlates with|is related to)\s+(\w+)', 'correlates'),
                (r'(\w+)\s+(?:depends on|requires)\s+(\w+)', 'depends_on'),
                (r'(\w+)\s+(?:is part of|belongs to)\s+(\w+)', 'part_of'),
                (r'(\w+)\s+(?:enhances|improves|increases)\s+(\w+)', 'enhances')
            ]
            
            # Find relationship patterns in text
            text_lower = text.lower()
            for pattern, relation_type in relationship_patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    entity1, entity2 = match.groups()
                    if entity1 and entity2:
                        relationships.append({
                            'entity1': entity1.strip(),
                            'relationship': relation_type,
                            'entity2': entity2.strip(),
                            'confidence': 0.6
                        })
            
            # Extract co-occurrence relationships from entities and concepts
            all_terms = entities + concepts
            for i, term1 in enumerate(all_terms):
                for term2 in all_terms[i+1:]:
                    # Check if both terms appear in close proximity
                    if term1.lower() in text_lower and term2.lower() in text_lower:
                        # Simple proximity check
                        term1_pos = text_lower.find(term1.lower())
                        term2_pos = text_lower.find(term2.lower())
                        if abs(term1_pos - term2_pos) < 200:  # Within 200 characters
                            relationships.append({
                                'entity1': term1,
                                'relationship': 'co_occurs_with',
                                'entity2': term2,
                                'confidence': 0.4
                            })
            
            # Remove duplicates and limit results
            unique_relationships = []
            seen = set()
            for rel in relationships:
                rel_key = (rel['entity1'], rel['relationship'], rel['entity2'])
                if rel_key not in seen:
                    seen.add(rel_key)
                    unique_relationships.append(rel)
            
            return unique_relationships[:20]  # Return top 20 relationships
            
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []
    
    def summarize_abstract(self, text: str, max_sentences: int = 3) -> str:
        """Generate a concise abstract summary"""
        try:
            # Split text into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) <= max_sentences:
                return text.strip()
            
            # Score sentences based on importance indicators
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = 0.0
                sentence_lower = sentence.lower()
                
                # Higher score for sentences with research indicators
                research_terms = ['study', 'research', 'analysis', 'investigation', 'experiment']
                score += sum(0.2 for term in research_terms if term in sentence_lower)
                
                # Higher score for sentences with methodology terms
                method_terms = ['method', 'approach', 'technique', 'algorithm', 'model']
                score += sum(0.2 for term in method_terms if term in sentence_lower)
                
                # Higher score for sentences with result indicators
                result_terms = ['result', 'finding', 'conclusion', 'demonstrate', 'show']
                score += sum(0.3 for term in result_terms if term in sentence_lower)
                
                # Higher score for sentences with numerical data
                if re.search(r'\d+\.?\d*%|\d+\.?\d*\s*(significant|p\s*<|correlation)', sentence_lower):
                    score += 0.3
                
                # Position bias - earlier sentences often more important
                score += (len(sentences) - i) / len(sentences) * 0.1
                
                # Length bias - moderate length sentences preferred
                length_score = 1.0 - abs(len(sentence.split()) - 15) / 30.0
                score += max(0, length_score) * 0.1
                
                sentence_scores.append((score, sentence))
            
            # Select top sentences
            sentence_scores.sort(key=lambda x: x[0], reverse=True)
            selected_sentences = [sent for _, sent in sentence_scores[:max_sentences]]
            
            # Maintain original order
            summary_sentences = []
            for sentence in sentences:
                if sentence in selected_sentences:
                    summary_sentences.append(sentence)
            
            return '. '.join(summary_sentences) + '.'
            
        except Exception as e:
            logger.error(f"Error summarizing abstract: {e}")
            # Fallback to first few sentences
            sentences = text.split('.')[:max_sentences]
            return '. '.join(s.strip() for s in sentences if s.strip()) + '.'
    
    def extract_key_findings(self, text: str) -> List[str]:
        """Extract key research findings from text"""
        try:
            findings = []
            
            # Split text into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            # Finding indicator patterns
            finding_patterns = [
                r'(?:we found|we discovered|results show|findings indicate|data suggest)',
                r'(?:significant|notable|important|key)\s+(?:finding|result|conclusion)',
                r'(?:demonstrated|showed|revealed|indicated|confirmed)',
                r'(?:analysis revealed|study found|research shows|evidence suggests)',
                r'(?:conclude|conclusion|in conclusion)',
                r'(?:improvement|increase|decrease|reduction).*(?:\d+%|\d+\.\d+)',
                r'(?:correlation|association|relationship).*(?:significant|strong|weak)',
                r'(?:effective|ineffective|superior|inferior).*(?:compared to|than)'
            ]
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Check for finding indicators
                for pattern in finding_patterns:
                    if re.search(pattern, sentence_lower):
                        # Clean and validate the finding
                        if len(sentence.split()) >= 5:  # Minimum meaningful length
                            findings.append(sentence.strip())
                        break
            
            # Extract sentences with statistical significance
            stats_patterns = [
                r'p\s*[<>=]\s*0\.\d+',
                r'statistically significant',
                r'confidence interval',
                r'\d+\.?\d*%.*(?:improvement|increase|decrease|reduction)',
                r'(?:r|correlation)\s*=\s*0\.\d+'
            ]
            
            for sentence in sentences:
                if sentence.strip() not in findings:  # Avoid duplicates
                    for pattern in stats_patterns:
                        if re.search(pattern, sentence.lower()):
                            if len(sentence.split()) >= 5:
                                findings.append(sentence.strip())
                            break
            
            # Extract sentences with comparative language
            comparative_patterns = [
                r'(?:better|worse|higher|lower|faster|slower).*than',
                r'(?:compared to|in comparison|relative to)',
                r'(?:superior|inferior|outperformed|exceeded)'
            ]
            
            for sentence in sentences:
                if sentence.strip() not in findings:  # Avoid duplicates
                    for pattern in comparative_patterns:
                        if re.search(pattern, sentence.lower()):
                            if len(sentence.split()) >= 6:
                                findings.append(sentence.strip())
                            break
            
            # Remove duplicates while preserving order
            unique_findings = []
            seen = set()
            for finding in findings:
                finding_clean = finding.lower().strip()
                if finding_clean not in seen:
                    seen.add(finding_clean)
                    unique_findings.append(finding)
            
            return unique_findings[:10]  # Return top 10 findings
            
        except Exception as e:
            logger.error(f"Error extracting key findings: {e}")
            return []

class TrendPredictor:
    """Critical Feature 3: Trend Prediction ML Model"""
    
    def __init__(self, model_type: str = "transformer", features: List[str] = None, prediction_horizon: str = "6_months"):
        self.model_type = model_type
        self.features = features or ["citation_patterns", "keyword_evolution", "author_networks"]
        self.prediction_horizon = prediction_horizon
        self.trend_model = None
        self.keyword_trends = {}
        self.citation_patterns = {}
        self.author_network_data = {}
        
    def predict_trends(self, documents: List[Dict[str, Any]], research_area: str) -> Dict[str, Any]:
        """Predict research trends from document collection"""
        try:
            # Extract trend features
            features = self._extract_trend_features(documents)
            
            # Choose prediction method based on model type
            if self.model_type == "transformer":
                trending_topics, emerging_areas, growth_predictions = self._transformer_prediction(features, documents)
            else:
                # Legacy clustering approach
                if len(features) > 5:
                    trending_topics = self._identify_trending_topics(features)
                    emerging_areas = self._identify_emerging_areas(documents)
                    growth_predictions = self._predict_growth(documents)
                else:
                    trending_topics = []
                    emerging_areas = []
                    growth_predictions = {}
            
            return {
                'research_area': research_area,
                'prediction_horizon': self.prediction_horizon,
                'trending_topics': trending_topics,
                'emerging_areas': emerging_areas,
                'growth_predictions': growth_predictions,
                'confidence_score': 0.85 if self.model_type == "transformer" else 0.75,
                'model_type': self.model_type,
                'analysis_date': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error predicting trends: {e}")
            return {}
    
    def _transformer_prediction(self, features, documents: List[Dict[str, Any]]) -> tuple:
        """Transformer-based trend prediction"""
        try:
            if not features or len(features) == 0:
                return [], [], {}
            
            # Enhanced trending topic identification
            trending_topics = self._identify_trending_topics_transformer(features, documents)
            
            # Advanced emerging area detection
            emerging_areas = self._identify_emerging_areas_transformer(documents)
            
            # Sophisticated growth prediction
            growth_predictions = self._predict_growth_transformer(documents)
            
            return trending_topics, emerging_areas, growth_predictions
            
        except Exception as e:
            logger.error(f"Error in transformer prediction: {e}")
            # Fallback to clustering method
            if features and len(features) > 5:
                trending_topics = self._identify_trending_topics(features)
                emerging_areas = self._identify_emerging_areas(documents)
                growth_predictions = self._predict_growth(documents)
            else:
                trending_topics = []
                emerging_areas = []
                growth_predictions = {}
            return trending_topics, emerging_areas, growth_predictions
    
    def _identify_trending_topics_transformer(self, features, documents: List[Dict[str, Any]]) -> List[str]:
        """Advanced trending topic identification using transformer approach"""
        try:
            # Analyze citation patterns and keyword evolution
            topic_scores = {}
            
            for i, doc in enumerate(documents):
                keywords = doc.get('keywords', [])
                citation_count = doc.get('citation_count', 0)
                year = doc.get('year', 2020)
                
                # Calculate trending score based on multiple factors
                for keyword in keywords:
                    if keyword not in topic_scores:
                        topic_scores[keyword] = {
                            'citations': 0,
                            'recency_score': 0,
                            'frequency': 0,
                            'growth_score': 0
                        }
                    
                    topic_scores[keyword]['citations'] += citation_count
                    topic_scores[keyword]['recency_score'] += max(0, (year - 2020) / 4.0)  # 2020-2024 range
                    topic_scores[keyword]['frequency'] += 1
            
            # Calculate composite trending scores
            trending_topics = []
            for topic, scores in topic_scores.items():
                if scores['frequency'] >= 2:  # Must appear in multiple documents
                    composite_score = (
                        scores['citations'] * 0.4 +
                        scores['recency_score'] * 0.3 +
                        scores['frequency'] * 0.3
                    )
                    trending_topics.append((topic, composite_score))
            
            # Sort by score and return top topics
            trending_topics.sort(key=lambda x: x[1], reverse=True)
            return [topic for topic, score in trending_topics[:8]]  # Top 8 trending topics
            
        except Exception as e:
            logger.error(f"Error in transformer trending topics: {e}")
            return []
    
    def _identify_emerging_areas_transformer(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Advanced emerging area identification using transformer approach"""
        try:
            # Analyze keyword evolution and author networks
            recent_keywords = {}
            keyword_author_networks = {}
            
            for doc in documents:
                year = doc.get('year', 2020)
                if year >= 2023:  # Focus on recent publications
                    keywords = doc.get('keywords', [])
                    authors = doc.get('authors', [])
                    
                    for keyword in keywords:
                        if keyword not in recent_keywords:
                            recent_keywords[keyword] = {
                                'count': 0,
                                'unique_authors': set(),
                                'avg_citations': 0,
                                'total_citations': 0
                            }
                        
                        recent_keywords[keyword]['count'] += 1
                        recent_keywords[keyword]['unique_authors'].update(authors)
                        citations = doc.get('citation_count', 0)
                        recent_keywords[keyword]['total_citations'] += citations
                        recent_keywords[keyword]['avg_citations'] = (
                            recent_keywords[keyword]['total_citations'] / recent_keywords[keyword]['count']
                        )
            
            # Identify emerging areas (moderate frequency, high novelty)
            emerging_areas = []
            for keyword, data in recent_keywords.items():
                # Emerging criteria: 2-6 occurrences, multiple authors, decent citations
                if (2 <= data['count'] <= 6 and 
                    len(data['unique_authors']) >= 2 and
                    data['avg_citations'] >= 3):
                    emerging_areas.append(keyword)
            
            return emerging_areas[:6]  # Top 6 emerging areas
            
        except Exception as e:
            logger.error(f"Error in transformer emerging areas: {e}")
            return []
    
    def _predict_growth_transformer(self, documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """Advanced growth prediction using transformer approach"""
        try:
            # Analyze multi-dimensional growth patterns
            yearly_data = {}
            topic_growth = {}
            
            for doc in documents:
                year = doc.get('year', 2020)
                keywords = doc.get('keywords', [])
                citations = doc.get('citation_count', 0)
                
                if year not in yearly_data:
                    yearly_data[year] = {
                        'count': 0,
                        'total_citations': 0,
                        'unique_topics': set()
                    }
                
                yearly_data[year]['count'] += 1
                yearly_data[year]['total_citations'] += citations
                yearly_data[year]['unique_topics'].update(keywords)
                
                # Track topic-specific growth
                for keyword in keywords:
                    if keyword not in topic_growth:
                        topic_growth[keyword] = {}
                    if year not in topic_growth[keyword]:
                        topic_growth[keyword][year] = 0
                    topic_growth[keyword][year] += 1
            
            # Calculate overall growth rate
            if len(yearly_data) >= 2:
                years = sorted(yearly_data.keys())
                recent_years = years[-2:]
                
                if len(recent_years) == 2:
                    count_growth = (yearly_data[recent_years[1]]['count'] - yearly_data[recent_years[0]]['count']) / max(1, yearly_data[recent_years[0]]['count'])
                    citation_growth = (yearly_data[recent_years[1]]['total_citations'] - yearly_data[recent_years[0]]['total_citations']) / max(1, yearly_data[recent_years[0]]['total_citations'])
                    
                    # Calculate topic diversity growth
                    topic_diversity_growth = (len(yearly_data[recent_years[1]]['unique_topics']) - len(yearly_data[recent_years[0]]['unique_topics'])) / max(1, len(yearly_data[recent_years[0]]['unique_topics']))
                    
                    return {
                        'predicted_growth_rate': count_growth,
                        'citation_growth_rate': citation_growth,
                        'topic_diversity_growth': topic_diversity_growth,
                        'composite_growth_score': (count_growth + citation_growth + topic_diversity_growth) / 3.0
                    }
            
            return {'predicted_growth_rate': 0.0, 'citation_growth_rate': 0.0, 'topic_diversity_growth': 0.0, 'composite_growth_score': 0.0}
            
        except Exception as e:
            logger.error(f"Error in transformer growth prediction: {e}")
            return {'predicted_growth_rate': 0.0}
    
    def _extract_trend_features(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for trend analysis"""
        features = []
        
        if self.model_type == "transformer":
            # Enhanced feature extraction for transformer model
            for doc in documents:
                doc_features = self._extract_transformer_features(doc)
                features.append(doc_features)
        else:
            # Legacy clustering feature extraction
            for doc in documents:
                # Citation features
                citation_count = doc.get('citation_count', 0)
                
                # Publication recency
                pub_year = doc.get('year', 2020)
                recency = 2024 - pub_year
                
                # Keyword diversity
                keywords = doc.get('keywords', [])
                keyword_count = len(keywords)
                
                # Abstract length (complexity indicator)
                abstract = doc.get('abstract', '')
                abstract_length = len(abstract.split())
                
                features.append([citation_count, recency, keyword_count, abstract_length])
        
        return np.array(features) if features and NUMPY_AVAILABLE else features
    
    def _extract_transformer_features(self, doc: Dict[str, Any]) -> List[float]:
        """Extract advanced features for transformer-based analysis"""
        feature_vector = []
        
        # 1. Citation patterns features
        if "citation_patterns" in self.features:
            citation_count = doc.get('citation_count', 0)
            citation_growth = doc.get('citation_growth_rate', 0.0)
            citations_per_year = citation_count / max(1, (2024 - doc.get('year', 2020)))
            
            feature_vector.extend([citation_count, citation_growth, citations_per_year])
        
        # 2. Keyword evolution features
        if "keyword_evolution" in self.features:
            keywords = doc.get('keywords', [])
            keyword_count = len(keywords)
            keyword_diversity = len(set(keywords)) / max(1, keyword_count)
            emerging_keyword_score = self._calculate_keyword_novelty(keywords)
            
            feature_vector.extend([keyword_count, keyword_diversity, emerging_keyword_score])
        
        # 3. Author networks features  
        if "author_networks" in self.features:
            authors = doc.get('authors', [])
            author_count = len(authors)
            collaboration_score = self._calculate_collaboration_score(authors)
            author_expertise_score = self._calculate_author_expertise(authors)
            
            feature_vector.extend([author_count, collaboration_score, author_expertise_score])
        
        # Additional contextual features
        pub_year = doc.get('year', 2020)
        recency = 2024 - pub_year
        abstract_length = len(doc.get('abstract', '').split())
        
        feature_vector.extend([recency, abstract_length])
        
        return feature_vector
    
    def _calculate_keyword_novelty(self, keywords: List[str]) -> float:
        """Calculate novelty score of keywords based on historical trends"""
        if not keywords:
            return 0.0
        
        novelty_score = 0.0
        for keyword in keywords:
            # Higher score for less common keywords
            if keyword not in self.keyword_trends:
                novelty_score += 1.0  # New keyword gets max score
            else:
                # Score based on rarity (inverse frequency)
                frequency = self.keyword_trends[keyword]
                novelty_score += 1.0 / (1.0 + frequency)
        
        return novelty_score / len(keywords)
    
    def _calculate_collaboration_score(self, authors: List[str]) -> float:
        """Calculate collaboration score based on author co-occurrence"""
        if len(authors) <= 1:
            return 0.0
        
        # Simple collaboration score based on number of collaborators
        return min(1.0, (len(authors) - 1) / 10.0)  # Normalize to 0-1
    
    def _calculate_author_expertise(self, authors: List[str]) -> float:
        """Calculate average author expertise score"""
        if not authors:
            return 0.0
        
        total_expertise = 0.0
        for author in authors:
            # Simplified expertise score (could be enhanced with actual author data)
            if author in self.author_network_data:
                total_expertise += self.author_network_data[author].get('expertise_score', 0.5)
            else:
                total_expertise += 0.5  # Default moderate expertise
        
        return total_expertise / len(authors)
    
    def _identify_trending_topics(self, features: np.ndarray) -> List[str]:
        """Identify trending topics using clustering"""
        try:
            if len(features) < 3:
                return []
                
            # Simple K-means clustering
            n_clusters = min(3, len(features))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features)
            
            # Identify high-activity clusters (simplified)
            cluster_activity = {}
            for i, cluster in enumerate(clusters):
                if cluster not in cluster_activity:
                    cluster_activity[cluster] = []
                cluster_activity[cluster].append(features[i])
            
            trending_topics = []
            for cluster_id, cluster_features in cluster_activity.items():
                avg_citations = np.mean([f[0] for f in cluster_features])
                if avg_citations > 5:  # Threshold for trending
                    trending_topics.append(f"Trending Topic Cluster {cluster_id}")
            
            return trending_topics
        except Exception as e:
            logger.error(f"Error identifying trending topics: {e}")
            return []
    
    def _identify_emerging_areas(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Identify emerging research areas"""
        # Look for recent papers with novel keywords
        recent_keywords = {}
        
        for doc in documents:
            year = doc.get('year', 2020)
            if year >= 2023:  # Recent papers
                keywords = doc.get('keywords', [])
                for keyword in keywords:
                    if keyword not in recent_keywords:
                        recent_keywords[keyword] = 0
                    recent_keywords[keyword] += 1
        
        # Identify keywords with moderate frequency (emerging)
        emerging = [kw for kw, count in recent_keywords.items() if 2 <= count <= 5]
        return emerging[:5]  # Top 5 emerging areas
    
    def _predict_growth(self, documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """Predict research area growth"""
        # Simple growth prediction based on recent publication trends
        yearly_counts = {}
        
        for doc in documents:
            year = doc.get('year', 2020)
            if year not in yearly_counts:
                yearly_counts[year] = 0
            yearly_counts[year] += 1
        
        # Calculate growth rate
        if len(yearly_counts) >= 2:
            years = sorted(yearly_counts.keys())
            recent_years = years[-2:]
            if len(recent_years) == 2:
                growth_rate = (yearly_counts[recent_years[1]] - yearly_counts[recent_years[0]]) / yearly_counts[recent_years[0]]
                return {'predicted_growth_rate': growth_rate}
        
        return {'predicted_growth_rate': 0.0}

class GapAnalyzer:
    """Component for analyzing research gaps"""
    
    def __init__(self):
        self.gap_patterns = [
            'limited research',
            'insufficient data',
            'lack of studies',
            'rarely investigated',
            'understudied',
            'gaps in knowledge',
            'future work',
            'further research needed'
        ]
    
    def analyze_gaps(self, documents: List[Dict[str, Any]], domain: str) -> Dict[str, Any]:
        """Analyze research gaps in the given document collection"""
        try:
            # Extract all text content
            all_text = []
            for doc in documents:
                text = f"{doc.get('title', '')} {doc.get('abstract', '')}"
                all_text.append(text.lower())
            
            combined_text = ' '.join(all_text)
            
            # Look for gap indicators
            identified_gaps = []
            for pattern in self.gap_patterns:
                if pattern in combined_text:
                    # Extract surrounding context
                    import re
                    matches = re.finditer(rf'.{{0,50}}{re.escape(pattern)}.{{0,50}}', combined_text)
                    for match in matches:
                        context = match.group().strip()
                        identified_gaps.append({
                            'pattern': pattern,
                            'context': context,
                            'confidence': 0.7
                        })
            
            # Analyze topic coverage gaps
            topics_mentioned = self._extract_research_topics(all_text)
            coverage_gaps = self._identify_coverage_gaps(topics_mentioned, domain)
            
            return {
                'domain': domain,
                'textual_gaps': identified_gaps[:10],  # Top 10
                'coverage_gaps': coverage_gaps,
                'total_documents_analyzed': len(documents),
                'gap_analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in gap analysis: {e}")
            return {}
    
    def _extract_research_topics(self, texts: List[str]) -> List[str]:
        """Extract research topics from text collection"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        try:
            # Extract key topics using TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,  # Lower min_df to handle small collections
                max_df=0.95  # Exclude very common terms
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = tfidf_matrix.mean(axis=0).A1
            topic_scores = list(zip(feature_names, mean_scores))
            topic_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [topic for topic, score in topic_scores[:20] if score > 0.01]
        except Exception as e:
            logger.warning(f"Could not extract topics using TF-IDF: {e}")
            # Fallback to simple keyword extraction
            all_words = []
            for text in texts:
                words = text.lower().split()
                # Filter out common words and keep meaningful terms
                meaningful_words = [w for w in words if len(w) > 3 and w.isalpha()]
                all_words.extend(meaningful_words)
            
            # Count frequency and return top words
            from collections import Counter
            word_counts = Counter(all_words)
            return [word for word, count in word_counts.most_common(20)]
    
    def _identify_coverage_gaps(self, topics: List[str], domain: str) -> List[str]:
        """Identify gaps in topic coverage for the domain"""
        # Domain-specific expected topics
        expected_topics = {
            'machine_learning': ['algorithms', 'neural networks', 'optimization', 'evaluation', 'datasets'],
            'skin_research': ['dermatology', 'cosmetics', 'ingredients', 'testing', 'safety', 'efficacy'],
            'medical': ['clinical trials', 'patient outcomes', 'diagnosis', 'treatment', 'safety'],
            'chemistry': ['synthesis', 'analysis', 'properties', 'reactions', 'characterization']
        }
        
        domain_topics = expected_topics.get(domain, [])
        covered_topics = set(topic.lower() for topic in topics)
        
        gaps = []
        for expected in domain_topics:
            if not any(expected in covered for covered in covered_topics):
                gaps.append(expected)
        
        return gaps

class HypothesisGenerator:
    """Component for generating research hypotheses"""
    
    def __init__(self):
        self.hypothesis_templates = [
            "Based on {gap}, we hypothesize that {intervention} will result in {expected_outcome}.",
            "Given the limited research on {topic}, we propose that {approach} may lead to {benefit}.",
            "The gap in {area} suggests that investigating {method} could reveal {insight}.",
            "We hypothesize that combining {technique1} with {technique2} will improve {metric}.",
            "The absence of studies on {context} indicates that {investigation} may uncover {discovery}."
        ]
    
    def generate_hypotheses(self, gap_analysis: Dict[str, Any], research_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate research hypotheses based on identified gaps"""
        try:
            hypotheses = []
            domain = gap_analysis.get('domain', 'general')
            textual_gaps = gap_analysis.get('textual_gaps', [])
            coverage_gaps = gap_analysis.get('coverage_gaps', [])
            
            # Generate hypotheses from textual gaps
            for gap in textual_gaps[:5]:  # Top 5 gaps
                hypothesis = self._generate_hypothesis_from_gap(gap, domain)
                if hypothesis:
                    hypotheses.append(hypothesis)
            
            # Generate hypotheses from coverage gaps
            for gap_topic in coverage_gaps[:3]:  # Top 3 coverage gaps
                hypothesis = self._generate_hypothesis_from_coverage_gap(gap_topic, domain)
                if hypothesis:
                    hypotheses.append(hypothesis)
            
            # Generate cross-domain hypotheses if multiple domains
            if len(coverage_gaps) > 1:
                cross_hypothesis = self._generate_cross_domain_hypothesis(coverage_gaps, domain)
                if cross_hypothesis:
                    hypotheses.append(cross_hypothesis)
            
            return hypotheses
            
        except Exception as e:
            logger.error(f"Error generating hypotheses: {e}")
            return []
    
    def _generate_hypothesis_from_gap(self, gap: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Generate hypothesis from a specific research gap"""
        import random
        
        template = random.choice(self.hypothesis_templates)
        context = gap.get('context', '')
        
        # Extract key terms from context
        terms = context.split()
        key_terms = [term for term in terms if len(term) > 4 and term.isalpha()]
        
        if len(key_terms) >= 2:
            hypothesis_text = template.format(
                gap=gap.get('pattern', 'research limitation'),
                intervention=f"investigating {key_terms[0]}",
                expected_outcome=f"improved understanding of {key_terms[1]}",
                topic=key_terms[0] if key_terms else domain,
                approach=f"systematic analysis of {key_terms[0]}" if key_terms else "comprehensive review",
                benefit="enhanced knowledge",
                area=domain,
                method=f"{key_terms[0]} methodology" if key_terms else "novel approach",
                insight="significant findings",
                technique1=key_terms[0] if len(key_terms) > 0 else "method A",
                technique2=key_terms[1] if len(key_terms) > 1 else "method B",
                metric="performance",
                context=domain,
                investigation=f"study of {key_terms[0]}" if key_terms else "investigation",
                discovery="new insights"
            )
            
            return {
                'hypothesis': hypothesis_text,
                'confidence': gap.get('confidence', 0.7),
                'source_gap': gap,
                'research_priority': 'medium',
                'feasibility': 'high',
                'generated_at': datetime.now().isoformat()
            }
        
        return None
    
    def _generate_hypothesis_from_coverage_gap(self, gap_topic: str, domain: str) -> Dict[str, Any]:
        """Generate hypothesis from coverage gap"""
        hypothesis_text = f"Investigation of {gap_topic} in {domain} research may reveal significant insights that could advance the field and fill current knowledge gaps."
        
        return {
            'hypothesis': hypothesis_text,
            'confidence': 0.8,
            'source_gap': {'type': 'coverage_gap', 'topic': gap_topic},
            'research_priority': 'high',
            'feasibility': 'medium',
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_cross_domain_hypothesis(self, gaps: List[str], domain: str) -> Dict[str, Any]:
        """Generate cross-domain hypothesis"""
        if len(gaps) >= 2:
            hypothesis_text = f"Integrating {gaps[0]} and {gaps[1]} approaches in {domain} research may lead to novel methodologies and breakthrough discoveries."
            
            return {
                'hypothesis': hypothesis_text,
                'confidence': 0.75,
                'source_gap': {'type': 'cross_domain', 'topics': gaps[:2]},
                'research_priority': 'high',
                'feasibility': 'low',
                'generated_at': datetime.now().isoformat()
            }
        
        return None

class SystematicReviewPlanner:
    """Component for planning systematic literature reviews"""
    
    def __init__(self):
        self.review_stages = [
            'protocol_development',
            'search_strategy',
            'inclusion_criteria',
            'exclusion_criteria',
            'data_extraction',
            'quality_assessment',
            'synthesis_method',
            'reporting_guidelines'
        ]
    
    def plan_systematic_review(self, research_question: str, domain: str, scope: str = 'comprehensive') -> Dict[str, Any]:
        """Plan a systematic literature review"""
        try:
            review_plan = {
                'research_question': research_question,
                'domain': domain,
                'scope': scope,
                'planning_date': datetime.now().isoformat(),
                'estimated_duration': self._estimate_duration(scope),
                'stages': self._plan_review_stages(research_question, domain, scope),
                'search_strategy': self._develop_search_strategy(research_question, domain),
                'quality_criteria': self._define_quality_criteria(domain),
                'synthesis_approach': self._select_synthesis_approach(scope)
            }
            
            return review_plan
            
        except Exception as e:
            logger.error(f"Error planning systematic review: {e}")
            return {}
    
    def _estimate_duration(self, scope: str) -> str:
        """Estimate review duration based on scope"""
        duration_map = {
            'rapid': '4-6 weeks',
            'focused': '8-12 weeks', 
            'comprehensive': '16-24 weeks',
            'extensive': '24-36 weeks'
        }
        return duration_map.get(scope, '12-16 weeks')
    
    def _plan_review_stages(self, research_question: str, domain: str, scope: str) -> List[Dict[str, Any]]:
        """Plan the stages of the systematic review"""
        stages = []
        
        for stage in self.review_stages:
            stage_plan = {
                'stage': stage,
                'description': self._get_stage_description(stage),
                'estimated_time': self._get_stage_duration(stage, scope),
                'key_activities': self._get_stage_activities(stage, domain),
                'deliverables': self._get_stage_deliverables(stage)
            }
            stages.append(stage_plan)
        
        return stages
    
    def _develop_search_strategy(self, research_question: str, domain: str) -> Dict[str, Any]:
        """Develop search strategy for the review"""
        # Extract key terms from research question
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', research_question.lower())
        key_terms = [word for word in words if word not in ['the', 'and', 'for', 'with', 'what', 'how', 'does']]
        
        databases = {
            'machine_learning': ['IEEE Xplore', 'ACM Digital Library', 'arXiv', 'Google Scholar'],
            'skin_research': ['PubMed', 'Scopus', 'Web of Science', 'EMBASE'],
            'medical': ['PubMed', 'Cochrane Library', 'EMBASE', 'CINAHL'],
            'chemistry': ['Chemical Abstracts', 'Scopus', 'Web of Science', 'Reaxys']
        }
        
        return {
            'key_terms': key_terms[:10],
            'boolean_operators': ['AND', 'OR', 'NOT'],
            'databases': databases.get(domain, ['Google Scholar', 'Scopus', 'Web of Science']),
            'search_filters': {
                'publication_years': '2015-2024',
                'language': 'English',
                'document_types': ['journal articles', 'conference papers']
            },
            'synonyms': self._generate_synonyms(key_terms)
        }
    
    def _define_quality_criteria(self, domain: str) -> List[str]:
        """Define quality assessment criteria"""
        base_criteria = [
            'Clear research objectives',
            'Appropriate methodology',
            'Adequate sample size',
            'Valid outcome measures',
            'Appropriate statistical analysis',
            'Clear reporting of results',
            'Discussion of limitations'
        ]
        
        domain_specific = {
            'medical': ['Randomized controlled trial design', 'Patient informed consent', 'Ethical approval'],
            'machine_learning': ['Dataset description', 'Baseline comparisons', 'Reproducibility'],
            'skin_research': ['Safety assessment', 'Regulatory compliance', 'Clinical relevance']
        }
        
        criteria = base_criteria + domain_specific.get(domain, [])
        return criteria
    
    def _select_synthesis_approach(self, scope: str) -> str:
        """Select appropriate synthesis approach"""
        approaches = {
            'rapid': 'narrative synthesis',
            'focused': 'thematic analysis',
            'comprehensive': 'meta-analysis where appropriate, otherwise narrative synthesis',
            'extensive': 'mixed-methods synthesis with meta-analysis'
        }
        return approaches.get(scope, 'narrative synthesis')
    
    def _get_stage_description(self, stage: str) -> str:
        """Get description for review stage"""
        descriptions = {
            'protocol_development': 'Develop detailed review protocol with research questions and methodology',
            'search_strategy': 'Design comprehensive search strategy for literature identification',
            'inclusion_criteria': 'Define clear criteria for study inclusion',
            'exclusion_criteria': 'Define clear criteria for study exclusion',
            'data_extraction': 'Extract relevant data from included studies',
            'quality_assessment': 'Assess quality and bias risk of included studies',
            'synthesis_method': 'Synthesize findings using appropriate methods',
            'reporting_guidelines': 'Follow PRISMA or other relevant reporting guidelines'
        }
        return descriptions.get(stage, 'Stage description not available')
    
    def _get_stage_duration(self, stage: str, scope: str) -> str:
        """Get estimated duration for stage"""
        duration_multipliers = {'rapid': 0.5, 'focused': 0.75, 'comprehensive': 1.0, 'extensive': 1.5}
        multiplier = duration_multipliers.get(scope, 1.0)
        
        base_durations = {
            'protocol_development': 7,
            'search_strategy': 14,
            'inclusion_criteria': 3,
            'exclusion_criteria': 3,
            'data_extraction': 21,
            'quality_assessment': 14,
            'synthesis_method': 21,
            'reporting_guidelines': 7
        }
        
        days = int(base_durations.get(stage, 7) * multiplier)
        return f"{days} days"
    
    def _get_stage_activities(self, stage: str, domain: str) -> List[str]:
        """Get key activities for stage"""
        activities = {
            'protocol_development': ['Define research questions', 'Select methodology', 'Register protocol'],
            'search_strategy': ['Identify databases', 'Develop search terms', 'Test search strategy'],
            'inclusion_criteria': ['Define population', 'Define interventions', 'Define outcomes'],
            'exclusion_criteria': ['Define exclusion criteria', 'Create screening forms'],
            'data_extraction': ['Design extraction forms', 'Extract data', 'Verify extraction'],
            'quality_assessment': ['Select assessment tools', 'Assess study quality', 'Rate bias risk'],
            'synthesis_method': ['Analyze data', 'Synthesize findings', 'Assess heterogeneity'],
            'reporting_guidelines': ['Follow PRISMA', 'Write report', 'Peer review']
        }
        return activities.get(stage, ['Activity not defined'])
    
    def _get_stage_deliverables(self, stage: str) -> List[str]:
        """Get deliverables for stage"""
        deliverables = {
            'protocol_development': ['Review protocol', 'Registration confirmation'],
            'search_strategy': ['Search strategy document', 'Database search results'],
            'inclusion_criteria': ['Inclusion criteria document', 'Screening forms'],
            'exclusion_criteria': ['Exclusion criteria document'],
            'data_extraction': ['Data extraction forms', 'Extracted data'],
            'quality_assessment': ['Quality assessment results', 'Bias risk tables'],
            'synthesis_method': ['Analysis results', 'Synthesis report'],
            'reporting_guidelines': ['Final review report', 'Publication draft']
        }
        return deliverables.get(stage, ['Deliverable not defined'])
    
    def _generate_synonyms(self, terms: List[str]) -> Dict[str, List[str]]:
        """Generate synonyms for search terms"""
        # Simple synonym generation (could be enhanced with actual thesaurus)
        synonym_patterns = {
            'machine': ['artificial', 'automated', 'computational'],
            'learning': ['training', 'education', 'development'],
            'neural': ['artificial neural', 'deep', 'connectionist'],
            'network': ['net', 'system', 'architecture'],
            'skin': ['dermal', 'cutaneous', 'epidermal'],
            'cosmetic': ['beauty', 'aesthetic', 'personal care'],
            'treatment': ['therapy', 'intervention', 'procedure'],
            'analysis': ['examination', 'evaluation', 'assessment']
        }
        
        synonyms = {}
        for term in terms:
            if term in synonym_patterns:
                synonyms[term] = synonym_patterns[term]
            else:
                synonyms[term] = []  # No synonyms found
        
        return synonyms

class ResearchPlanner:
    """Critical Feature 4: Autonomous Research Planning Engine"""
    
    def __init__(self, gap_analyzer: GapAnalyzer = None, hypothesis_generator: HypothesisGenerator = None, 
                 review_planner: SystematicReviewPlanner = None):
        self.gap_analyzer = gap_analyzer or GapAnalyzer()
        self.hypothesis_generator = hypothesis_generator or HypothesisGenerator()
        self.review_planner = review_planner or SystematicReviewPlanner()
    
    def create_research_plan(self, research_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive autonomous research plan"""
        try:
            domain = research_context.get('domain', 'general')
            research_question = research_context.get('research_question', '')
            documents = research_context.get('documents', [])
            scope = research_context.get('scope', 'comprehensive')
            
            # Step 1: Analyze research gaps
            gap_analysis = self.gap_analyzer.analyze_gaps(documents, domain)
            
            # Step 2: Generate research hypotheses
            hypotheses = self.hypothesis_generator.generate_hypotheses(gap_analysis, research_context)
            
            # Step 3: Plan systematic review if needed
            review_plan = None
            if research_question:
                review_plan = self.review_planner.plan_systematic_review(research_question, domain, scope)
            
            # Step 4: Create integrated research plan
            research_plan = {
                'research_context': research_context,
                'gap_analysis': gap_analysis,
                'generated_hypotheses': hypotheses,
                'systematic_review_plan': review_plan,
                'research_priorities': self._prioritize_research_directions(hypotheses, gap_analysis),
                'recommended_next_steps': self._recommend_next_steps(hypotheses, gap_analysis),
                'plan_created_at': datetime.now().isoformat(),
                'plan_confidence': self._calculate_plan_confidence(gap_analysis, hypotheses)
            }
            
            return research_plan
            
        except Exception as e:
            logger.error(f"Error creating research plan: {e}")
            return {}
    
    def _prioritize_research_directions(self, hypotheses: List[Dict[str, Any]], gap_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize research directions based on hypotheses and gaps"""
        priorities = []
        
        # Score each hypothesis
        for hypothesis in hypotheses:
            priority_score = 0.0
            
            # Factor in confidence
            priority_score += hypothesis.get('confidence', 0.5) * 0.3
            
            # Factor in research priority
            priority_map = {'high': 0.9, 'medium': 0.6, 'low': 0.3}
            priority_score += priority_map.get(hypothesis.get('research_priority', 'medium'), 0.6) * 0.4
            
            # Factor in feasibility
            feasibility_map = {'high': 0.8, 'medium': 0.6, 'low': 0.3}
            priority_score += feasibility_map.get(hypothesis.get('feasibility', 'medium'), 0.6) * 0.3
            
            priorities.append({
                'hypothesis': hypothesis['hypothesis'],
                'priority_score': priority_score,
                'reasoning': f"Confidence: {hypothesis.get('confidence', 0.5)}, Priority: {hypothesis.get('research_priority', 'medium')}, Feasibility: {hypothesis.get('feasibility', 'medium')}",
                'recommended_approach': self._recommend_approach(hypothesis)
            })
        
        # Sort by priority score
        priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        return priorities
    
    def _recommend_next_steps(self, hypotheses: List[Dict[str, Any]], gap_analysis: Dict[str, Any]) -> List[str]:
        """Recommend concrete next steps for research"""
        next_steps = []
        
        if hypotheses:
            next_steps.append("1. Begin with the highest priority hypothesis identified in the research plan")
            next_steps.append("2. Conduct preliminary literature review to refine the research question")
            next_steps.append("3. Design pilot study to test feasibility of the proposed approach")
        
        if gap_analysis.get('coverage_gaps'):
            next_steps.append("4. Address identified coverage gaps through targeted literature search")
        
        if gap_analysis.get('textual_gaps'):
            next_steps.append("5. Investigate specific research limitations mentioned in existing literature")
        
        next_steps.extend([
            "6. Establish collaborations with experts in identified gap areas",
            "7. Prepare research proposal based on prioritized research directions",
            "8. Seek funding opportunities aligned with research priorities"
        ])
        
        return next_steps
    
    def _calculate_plan_confidence(self, gap_analysis: Dict[str, Any], hypotheses: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the research plan"""
        confidence_factors = []
        
        # Factor in number of documents analyzed
        doc_count = gap_analysis.get('total_documents_analyzed', 0)
        if doc_count > 50:
            confidence_factors.append(0.9)
        elif doc_count > 20:
            confidence_factors.append(0.7)
        elif doc_count > 5:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
        
        # Factor in number of gaps identified
        gap_count = len(gap_analysis.get('textual_gaps', []))
        if gap_count > 5:
            confidence_factors.append(0.8)
        elif gap_count > 2:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        # Factor in hypothesis quality
        if hypotheses:
            avg_hypothesis_confidence = sum(h.get('confidence', 0.5) for h in hypotheses) / len(hypotheses)
            confidence_factors.append(avg_hypothesis_confidence)
        else:
            confidence_factors.append(0.3)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _recommend_approach(self, hypothesis: Dict[str, Any]) -> str:
        """Recommend research approach for hypothesis"""
        feasibility = hypothesis.get('feasibility', 'medium')
        priority = hypothesis.get('research_priority', 'medium')
        
        if feasibility == 'high' and priority == 'high':
            return "Immediate investigation with full research study"
        elif feasibility == 'high':
            return "Conduct pilot study followed by full investigation"
        elif priority == 'high':
            return "Seek collaborations or resources to improve feasibility"
        else:
            return "Include in longer-term research planning"

class ResearchDiscoveryAgent(EnhancedAgent):
    """
    Enhanced Research Discovery Agent with Critical ML Features
    Implements Vector Database Integration, NLP Pipeline, and Trend Prediction ML
    """
    
    def __init__(self, agent_id: str = "research_discovery_001"):
        super().__init__(
            agent_id=agent_id,
            agent_type="research_discovery",
            capabilities=["literature_discovery", "trend_analysis", "gap_identification", "vector_search", "nlp_processing", "research_planning", "hypothesis_generation"]
        )
        
        # Critical Feature 1: Vector Database Integration
        self.vector_db = VectorDatabase(
            embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
            storage_type="sklearn",
            index_type="cosine"
        )
        
        # Critical Feature 2: NLP Pipeline for Document Understanding
        self.nlp_pipeline = DocumentProcessor(
            extractors=["entities", "concepts", "relationships"],
            classifiers=["topic", "quality", "novelty"],
            summarizers=["abstract", "key_findings"]
        )
        
        # Critical Feature 3: Trend Prediction ML Model
        self.trend_model = TrendPredictor(
            model_type="clustering",
            features=["citation_patterns", "keyword_evolution", "author_networks"],
            prediction_horizon="6_months"
        )
        
        # Critical Feature 4: Autonomous Research Planning Engine
        self.research_planner = ResearchPlanner(
            gap_analyzer=GapAnalyzer(),
            hypothesis_generator=HypothesisGenerator(),
            review_planner=SystematicReviewPlanner()
        )
        
        # Initialize OpenAI client if API key is available
        self.openai_client = None
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai.OpenAI()
        
        # Research databases and APIs
        self.databases = {
            "semantic_scholar": "https://api.semanticscholar.org/graph/v1",
            "crossref": "https://api.crossref.org/works",
            "arxiv": "http://export.arxiv.org/api/query"
        }
        
        # Cache for research results
        self.research_cache = {}
        
        logger.info(f"Research Discovery Agent initialized with critical ML features")
    
    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a research task - implementation of abstract method"""
        try:
            task_type = task_data.get('task_type', 'research')
            
            if task_type == 'literature_search':
                return self._perform_literature_search(task_data)
            elif task_type == 'trend_analysis':
                return self._perform_trend_analysis(task_data)
            elif task_type == 'gap_identification':
                return self._identify_research_gaps(task_data)
            elif task_type == 'research_planning':
                return self._autonomous_research_planning(task_data)
            else:
                # Default to using process_action method
                return self.process_action(task_data)
                
        except Exception as e:
            logger.error(f"Error processing research task: {e}")
            return {"error": str(e)}
    
    def get_available_actions(self, context: Dict[str, Any]) -> List:
        """Get available research actions for a given context - implementation of abstract method"""
        from .enhanced_agent import AgentAction
        
        available_actions = []
        
        # Literature discovery actions
        available_actions.append(AgentAction(
            action_id="discover_literature",
            action_type="discover",
            input_data={"query": "", "domain": "", "limit": 20},
            expected_output={"papers": [], "total_discovered": 0},
            constraints={"max_limit": 200, "min_query_length": 3},
            priority=0.8
        ))
        
        # Trend analysis actions
        available_actions.append(AgentAction(
            action_id="analyze_trends",
            action_type="analyze_trends", 
            input_data={"research_area": "", "time_window_months": 12},
            expected_output={"trend_predictions": {}, "confidence_score": 0.0},
            constraints={"max_time_window": 60, "min_time_window": 3},
            priority=0.7
        ))
        
        # Gap identification actions
        available_actions.append(AgentAction(
            action_id="identify_gaps",
            action_type="identify_gaps",
            input_data={"research_area": ""},
            expected_output={"conceptual_gaps": [], "methodological_gaps": []},
            constraints={"max_papers": 200},
            priority=0.9
        ))
        
        # Research planning actions
        available_actions.append(AgentAction(
            action_id="plan_research",
            action_type="plan_research",
            input_data={"domain": "", "research_question": "", "scope": "comprehensive"},
            expected_output={"research_plan": {}, "planning_confidence": 0.0},
            constraints={"max_scope": "extensive"},
            priority=0.95
        ))
        
        # Knowledge building actions
        available_actions.append(AgentAction(
            action_id="build_knowledge",
            action_type="build_knowledge",
            input_data={"domains": [], "papers_per_domain": 50},
            expected_output={"total_papers_added": 0, "knowledge_base_size": 0},
            constraints={"max_papers_per_domain": 100, "max_domains": 10},
            priority=0.6
        ))
        
        return available_actions
    
    def process_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process autonomous research actions using enhanced ML capabilities"""
        try:
            action_type = action_data.get('action_type', 'discover')
            
            if action_type == 'discover':
                return self._autonomous_research_discovery(action_data)
            elif action_type == 'analyze_trends':
                return self._autonomous_trend_analysis(action_data)
            elif action_type == 'identify_gaps':
                return self._autonomous_gap_identification(action_data)
            elif action_type == 'build_knowledge':
                return self._autonomous_knowledge_building(action_data)
            elif action_type == 'plan_research':
                return self._autonomous_research_planning(action_data)
            else:
                return {"error": f"Unknown action type: {action_type}"}
                
        except Exception as e:
            logger.error(f"Error processing research action: {e}")
            return {"error": str(e)}
    
    def _autonomous_research_discovery(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous research discovery using vector database and ML"""
        try:
            query = action_data.get('query', '')
            domain = action_data.get('domain', 'general')
            limit = action_data.get('limit', 20)
            
            # Step 1: Use vector database for semantic search if documents exist
            vector_results = self.vector_db.search_similar(query, limit)
            
            # Step 2: External literature search
            external_results = self._perform_literature_search({
                'query': query,
                'domain': domain,
                'limit': limit
            })
            
            # Step 3: Process with NLP pipeline
            processed_results = []
            for paper in external_results.get('papers', []):
                text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                
                # Extract entities and concepts
                entities = self.nlp_pipeline.extract_entities(text)
                concepts = self.nlp_pipeline.extract_concepts(text)
                topic = self.nlp_pipeline.classify_topic(text)
                quality = self.nlp_pipeline.assess_quality(text)
                novelty = self.nlp_pipeline.assess_novelty(text)
                
                enhanced_paper = {
                    **paper,
                    'extracted_entities': entities,
                    'key_concepts': concepts,
                    'classified_topic': topic,
                    'quality_score': quality,
                    'novelty_score': novelty
                }
                processed_results.append(enhanced_paper)
            
            # Step 4: Store in vector database for future searches
            if processed_results:
                self.vector_db.add_documents(processed_results)
            
            return {
                'query': query,
                'vector_results': len(vector_results),
                'external_results': len(processed_results),
                'processed_papers': processed_results,
                'total_discovered': len(vector_results) + len(processed_results),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in autonomous research discovery: {e}")
            return {"error": str(e)}
    
    def _autonomous_trend_analysis(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous trend analysis using ML prediction model"""
        try:
            research_area = action_data.get('research_area', 'general')
            time_window = action_data.get('time_window_months', 12)
            
            # Get recent papers for the research area
            search_results = self._perform_literature_search({
                'query': research_area,
                'domain': research_area,
                'limit': 100
            })
            
            # Use trend prediction ML model
            trend_predictions = self.trend_model.predict_trends(
                search_results.get('papers', []),
                research_area
            )
            
            # Store results in memory system
            self.memory_system.store_memory(
                agent_id=self.agent_id,
                memory_type='analysis',
                content={'trend_analysis': trend_predictions},
                importance_score=0.8,
                tags=['trend_analysis', research_area]
            )
            
            return {
                'research_area': research_area,
                'trend_predictions': trend_predictions,
                'papers_analyzed': len(search_results.get('papers', [])),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in autonomous trend analysis: {e}")
            return {"error": str(e)}
    
    def _autonomous_gap_identification(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous research gap identification using ML analysis"""
        try:
            research_area = action_data.get('research_area', 'general')
            
            # Get comprehensive literature for the area
            search_results = self._perform_literature_search({
                'query': research_area,
                'domain': research_area,
                'limit': 200
            })
            
            papers = search_results.get('papers', [])
            
            # Analyze for gaps using NLP pipeline
            all_concepts = []
            all_methodologies = []
            coverage_map = {}
            
            for paper in papers:
                text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                concepts = self.nlp_pipeline.extract_concepts(text)
                all_concepts.extend(concepts)
                
                # Extract methodology mentions
                methods = re.findall(r'\b(?:method|approach|technique|algorithm|model)\w*\b', text.lower())
                all_methodologies.extend(methods)
                
                # Build coverage map
                for concept in concepts:
                    if concept not in coverage_map:
                        coverage_map[concept] = 0
                    coverage_map[concept] += 1
            
            # Identify gaps (concepts with low coverage)
            concept_frequencies = {concept: count for concept, count in coverage_map.items()}
            low_coverage_concepts = [concept for concept, count in concept_frequencies.items() if count <= 2]
            
            # Identify methodology gaps
            methodology_freq = {}
            for method in all_methodologies:
                methodology_freq[method] = methodology_freq.get(method, 0) + 1
            
            underexplored_methods = [method for method, count in methodology_freq.items() if count <= 1]
            
            gaps = {
                'research_area': research_area,
                'conceptual_gaps': low_coverage_concepts[:10],
                'methodological_gaps': underexplored_methods[:10],
                'total_papers_analyzed': len(papers),
                'gap_confidence': 0.7,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Store in memory
            self.memory_system.store_memory(
                agent_id=self.agent_id,
                memory_type='analysis',
                content={'gap_analysis': gaps},
                importance_score=0.9,
                tags=['gap_identification', research_area]
            )
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error in autonomous gap identification: {e}")
            return {"error": str(e)}
    
    def _autonomous_knowledge_building(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous knowledge base building using vector database"""
        try:
            domains = action_data.get('domains', ['computer_science', 'medicine'])
            papers_per_domain = action_data.get('papers_per_domain', 50)
            
            total_added = 0
            
            for domain in domains:
                # Search for comprehensive literature in domain
                search_results = self._perform_literature_search({
                    'query': f'{domain} research',
                    'domain': domain,
                    'limit': papers_per_domain
                })
                
                papers = search_results.get('papers', [])
                
                # Process and add to vector database
                if papers:
                    processed_papers = []
                    for paper in papers:
                        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                        entities = self.nlp_pipeline.extract_entities(text)
                        concepts = self.nlp_pipeline.extract_concepts(text)
                        
                        enhanced_paper = {
                            **paper,
                            'id': f"{domain}_{len(processed_papers)}",
                            'extracted_entities': entities,
                            'key_concepts': concepts,
                            'domain': domain
                        }
                        processed_papers.append(enhanced_paper)
                    
                    # Add to vector database
                    success = self.vector_db.add_documents(processed_papers)
                    if success:
                        total_added += len(processed_papers)
            
            return {
                'domains_processed': domains,
                'total_papers_added': total_added,
                'knowledge_base_size': len(self.vector_db.document_vectors),
                'build_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in autonomous knowledge building: {e}")
            return {"error": str(e)}
    
    def _autonomous_research_planning(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous research planning using ResearchPlanner with gap analysis and hypothesis generation"""
        try:
            domain = action_data.get('domain', 'general')
            research_question = action_data.get('research_question', '')
            scope = action_data.get('scope', 'comprehensive')
            
            # Get relevant literature for the domain/question
            search_results = self._perform_literature_search({
                'query': research_question or domain,
                'domain': domain,
                'limit': 100
            })
            
            papers = search_results.get('papers', [])
            
            # Create research context
            research_context = {
                'domain': domain,
                'research_question': research_question,
                'documents': papers,
                'scope': scope,
                'requester': action_data.get('requester', 'autonomous'),
                'planning_date': datetime.now().isoformat()
            }
            
            # Use ResearchPlanner to create comprehensive plan
            research_plan = self.research_planner.create_research_plan(research_context)
            
            # Store plan in memory system
            self.memory_system.store_memory(
                agent_id=self.agent_id,
                memory_type='research_plan',
                content={'research_plan': research_plan},
                importance_score=0.9,
                tags=['research_planning', domain, 'autonomous']
            )
            
            return {
                'domain': domain,
                'research_question': research_question,
                'research_plan': research_plan,
                'papers_analyzed': len(papers),
                'planning_confidence': research_plan.get('plan_confidence', 0.5),
                'planning_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in autonomous research planning: {e}")
            return {"error": str(e)}
    
    def _process_message(self, message):
        """Process research-related messages"""
        try:
            content = json.loads(message.content) if isinstance(message.content, str) else message.content
            
            if message.message_type == MessageType.QUERY:
                response_data = self._handle_research_query(content)
            elif message.message_type == MessageType.COMMAND:
                response_data = self._handle_research_command(content)
            else:
                response_data = {"error": "Unsupported message type"}
            
            # Send response back
            self.send_message(
                recipient_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                content=response_data,
                correlation_id=message.id
            )
            
            # Update performance metrics
            metrics = self.get_performance_metrics()
            metrics['tasks_completed'] += 1
            self.update_performance_metrics(metrics)
            
        except Exception as e:
            error_response = {"error": str(e), "agent": self.name}
            self.send_message(
                recipient_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                content=error_response,
                correlation_id=message.id
            )
    
    def _handle_research_query(self, content: Dict) -> Dict:
        """Handle research query requests"""
        query_type = content.get('query_type')
        
        if query_type == 'literature_search':
            return self._perform_literature_search(content)
        elif query_type == 'trend_analysis':
            return self._perform_trend_analysis(content)
        elif query_type == 'gap_identification':
            return self._identify_research_gaps(content)
        elif query_type == 'impact_assessment':
            return self._assess_research_impact(content)
        else:
            return {"error": f"Unknown query type: {query_type}"}
    
    def _handle_research_command(self, content: Dict) -> Dict:
        """Handle research command requests"""
        command = content.get('command')
        
        if command == 'update_knowledge_base':
            return self._update_knowledge_base(content)
        elif command == 'generate_research_report':
            return self._generate_research_report(content)
        else:
            return {"error": f"Unknown command: {command}"}
    
    def _perform_literature_search(self, params: Dict) -> Dict:
        """Perform comprehensive literature search across multiple databases"""
        query = params.get('query', '')
        domain = params.get('domain', 'computer_science')
        limit = params.get('limit', 20)
        
        # Check cache first
        cache_key = f"search_{hash(query)}_{domain}_{limit}"
        if cache_key in self.research_cache:
            cached_result = self.research_cache[cache_key]
            if datetime.now() - cached_result['timestamp'] < timedelta(hours=1):
                return cached_result['data']
        
        results = {
            "query": query,
            "domain": domain,
            "papers": [],
            "total_found": 0,
            "search_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Search Semantic Scholar
            semantic_results = self._search_semantic_scholar(query, limit)
            results["papers"].extend(semantic_results)
            
            # Search arXiv for CS papers
            if domain in ['computer_science', 'physics', 'mathematics']:
                arxiv_results = self._search_arxiv(query, limit)
                results["papers"].extend(arxiv_results)
            
            # Remove duplicates and sort by relevance
            results["papers"] = self._deduplicate_papers(results["papers"])
            results["total_found"] = len(results["papers"])
            
            # Enhance with AI analysis if available
            if self.openai_client and results["papers"]:
                results["ai_analysis"] = self._analyze_papers_with_ai(results["papers"][:5])
            
            # Cache results
            self.research_cache[cache_key] = {
                'data': results,
                'timestamp': datetime.now()
            }
            
            return results
            
        except Exception as e:
            return {"error": f"Literature search failed: {str(e)}"}
    
    def _search_semantic_scholar(self, query: str, limit: int) -> List[Dict]:
        """Search Semantic Scholar API"""
        try:
            url = f"{self.databases['semantic_scholar']}/paper/search"
            params = {
                'query': query,
                'limit': limit,
                'fields': 'title,authors,year,abstract,citationCount,url,venue'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for paper in data.get('data', []):
                papers.append({
                    'title': paper.get('title', ''),
                    'authors': [author.get('name', '') for author in paper.get('authors', [])],
                    'year': paper.get('year'),
                    'abstract': paper.get('abstract', ''),
                    'citation_count': paper.get('citationCount', 0),
                    'url': paper.get('url', ''),
                    'venue': paper.get('venue', ''),
                    'source': 'semantic_scholar'
                })
            
            return papers
            
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return []
    
    def _search_arxiv(self, query: str, limit: int) -> List[Dict]:
        """Search arXiv API"""
        try:
            url = self.databases['arxiv']
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': limit,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                published = entry.find('{http://www.w3.org/2005/Atom}published').text
                
                authors = []
                for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                    name = author.find('{http://www.w3.org/2005/Atom}name').text
                    authors.append(name)
                
                papers.append({
                    'title': title,
                    'authors': authors,
                    'year': int(published[:4]),
                    'abstract': summary,
                    'citation_count': 0,  # arXiv doesn't provide citation counts
                    'url': entry.find('{http://www.w3.org/2005/Atom}id').text,
                    'venue': 'arXiv',
                    'source': 'arxiv'
                })
            
            return papers
            
        except Exception as e:
            print(f"arXiv search error: {e}")
            return []
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            # Simple deduplication based on title
            title_key = re.sub(r'[^\w\s]', '', title)
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        # Sort by citation count and year
        unique_papers.sort(key=lambda x: (x.get('citation_count', 0), x.get('year', 0)), reverse=True)
        return unique_papers
    
    def _analyze_papers_with_ai(self, papers: List[Dict]) -> Dict:
        """Use AI to analyze papers and extract insights"""
        if not self.openai_client:
            return {"error": "OpenAI client not available"}
        
        try:
            # Prepare paper summaries for AI analysis
            paper_summaries = []
            for paper in papers:
                summary = f"Title: {paper.get('title', '')}\n"
                summary += f"Authors: {', '.join(paper.get('authors', []))}\n"
                summary += f"Year: {paper.get('year', 'Unknown')}\n"
                summary += f"Abstract: {paper.get('abstract', '')[:500]}...\n"
                paper_summaries.append(summary)
            
            prompt = f"""
            Analyze the following research papers and provide insights:
            
            {chr(10).join(paper_summaries)}
            
            Please provide:
            1. Key themes and trends
            2. Research gaps identified
            3. Methodological approaches
            4. Potential future directions
            5. Overall assessment of the field
            
            Format your response as JSON with these keys: themes, gaps, methods, future_directions, assessment
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            # Parse AI response
            ai_content = response.choices[0].message.content
            try:
                return json.loads(ai_content)
            except json.JSONDecodeError:
                return {"ai_analysis": ai_content}
                
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}
    
    def _perform_trend_analysis(self, params: Dict) -> Dict:
        """Analyze research trends in a specific domain"""
        domain = params.get('domain', 'computer_science')
        time_range = params.get('time_range', 5)  # years
        
        # This would typically involve more sophisticated analysis
        # For now, return a simplified trend analysis
        return {
            "domain": domain,
            "time_range": time_range,
            "trends": [
                {
                    "topic": "Machine Learning",
                    "growth_rate": 0.25,
                    "paper_count": 1500,
                    "key_keywords": ["neural networks", "deep learning", "AI"]
                },
                {
                    "topic": "Natural Language Processing",
                    "growth_rate": 0.30,
                    "paper_count": 800,
                    "key_keywords": ["NLP", "transformers", "language models"]
                }
            ],
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _identify_research_gaps(self, params: Dict) -> Dict:
        """Identify potential research gaps in a domain"""
        domain = params.get('domain', 'computer_science')
        
        # Simplified gap identification
        return {
            "domain": domain,
            "identified_gaps": [
                {
                    "gap_area": "Explainable AI in Healthcare",
                    "description": "Limited research on interpretable ML models for medical diagnosis",
                    "opportunity_score": 0.85,
                    "related_keywords": ["explainable AI", "healthcare", "interpretability"]
                },
                {
                    "gap_area": "Sustainable Computing",
                    "description": "Insufficient focus on energy-efficient algorithms",
                    "opportunity_score": 0.75,
                    "related_keywords": ["green computing", "energy efficiency", "sustainability"]
                }
            ],
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _assess_research_impact(self, params: Dict) -> Dict:
        """Assess the potential impact of research topics"""
        topics = params.get('topics', [])
        
        impact_assessment = []
        for topic in topics:
            # Simplified impact assessment
            impact_assessment.append({
                "topic": topic,
                "impact_score": 0.8,  # Would be calculated based on various factors
                "factors": {
                    "novelty": 0.7,
                    "applicability": 0.9,
                    "market_potential": 0.8,
                    "academic_interest": 0.85
                },
                "recommendation": "High potential for significant impact"
            })
        
        return {
            "impact_assessments": impact_assessment,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _update_knowledge_base(self, params: Dict) -> Dict:
        """Update the agent's knowledge base with new information"""
        # This would typically update a vector database or knowledge graph
        return {
            "status": "knowledge_base_updated",
            "updated_entries": params.get('entries', 0),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_research_report(self, params: Dict) -> Dict:
        """Generate a comprehensive research report"""
        topic = params.get('topic', 'General Research')
        
        # This would generate a detailed report based on collected research
        return {
            "report_title": f"Research Report: {topic}",
            "generated_at": datetime.now().isoformat(),
            "sections": [
                {
                    "title": "Executive Summary",
                    "content": "Overview of current research landscape..."
                },
                {
                    "title": "Key Findings",
                    "content": "Major discoveries and trends identified..."
                },
                {
                    "title": "Research Gaps",
                    "content": "Areas requiring further investigation..."
                },
                {
                    "title": "Recommendations",
                    "content": "Suggested research directions..."
                }
            ]
        }

