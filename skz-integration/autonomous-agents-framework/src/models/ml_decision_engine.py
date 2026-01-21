"""
ML Decision Engine for Autonomous Agents
Phase 2 Critical Component - Provides ML capabilities for intelligent decision-making
"""

import json
import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DecisionContext:
    """Context for decision making"""
    agent_id: str
    action_type: str
    input_data: Dict[str, Any]
    available_options: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    goals: List[str]
    risk_tolerance: float

@dataclass
class DecisionResult:
    """Result of a decision"""
    decision: Dict[str, Any]
    confidence_score: float
    reasoning: str
    alternatives: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    performance_prediction: Dict[str, Any]

class NLPProcessor:
    """Natural Language Processing capabilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.entity_patterns = {
            'ingredient': r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+)*\b',
            'compound': r'\b[A-Z][a-z]+\d*\b',
            'methodology': r'\b(?:study|analysis|experiment|trial|test)\b',
            'author': r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b'
        }
        self.sentiment_keywords = {
            'positive': ['effective', 'significant', 'improved', 'successful', 'beneficial'],
            'negative': ['adverse', 'toxic', 'harmful', 'ineffective', 'risky'],
            'neutral': ['standard', 'normal', 'typical', 'conventional']
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = list(set(matches))
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        text_lower = text.lower()
        scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for sentiment, keywords in self.sentiment_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[sentiment] += 1
        
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        
        return scores
    
    def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text into categories using production ML models"""
        
        # PRODUCTION ENFORCEMENT: Require ML models in production
        if os.getenv('ENVIRONMENT', '').lower() == 'production':
            if not hasattr(self, 'bert_model') or self.bert_model is None:
                raise ValueError(
                    "PRODUCTION VIOLATION: BERT model required for production text classification. "
                    "Configure ML models properly. NEVER SACRIFICE QUALITY!!"
                )
            return self._classify_text_bert(text, categories)
        
        # Development/staging can use BERT if available, otherwise require explicit config
        if hasattr(self, 'bert_model') and self.bert_model is not None:
            return self._classify_text_bert(text, categories)
        else:
            # Development mode: require explicit acknowledgment of non-ML classification
            if not self.config.get('allow_keyword_classification_in_development', False):
                raise ValueError(
                    "ML models not available and keyword classification not explicitly enabled. "
                    "Set allow_keyword_classification_in_development=True in config for development use."
                )
            return self._classify_text_keywords(text, categories)
    
    def _classify_text_bert(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Production BERT-based text classification"""
        try:
            # Check if running in production mode and BERT is required
            if self.config.get('force_ml_models', False) and not hasattr(self, 'bert_model'):
                raise ValueError("BERT model required for production but not loaded")
            
            # Import required libraries (fail gracefully if not available)
            try:
                from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
                import torch
                import torch.nn.functional as F
                import numpy as np
            except ImportError as e:
                logger.error(f"Required ML libraries not installed: {e}")
                if self.config.get('force_ml_models', False):
                    raise ValueError("BERT libraries required for production but not available")
                return self._classify_text_keywords(text, categories)
            
            # Initialize BERT model if not already done
            if not hasattr(self, 'bert_tokenizer') or not hasattr(self, 'bert_model'):
                self._initialize_bert_model()
            
            # Tokenize input text
            inputs = self.bert_tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.config.get('max_sequence_length', 512),
                return_tensors='pt'
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            
            # Classification for each category
            category_scores = {}
            
            for category in categories:
                try:
                    # Use category-specific classifier if available
                    classifier_path = self.config.get('category_classifiers', {}).get(category)
                    
                    if classifier_path and os.path.exists(classifier_path):
                        classifier = self._load_category_classifier(classifier_path)
                        if classifier is not None and hasattr(classifier, "predict_proba"):
                            score = classifier.predict_proba(embeddings.numpy())[0][1]
                        else:
                            category_keywords = self._get_category_keywords(category)
                            score = self._calculate_semantic_similarity_bert(text, category_keywords)
                    else:
                        category_keywords = self._get_category_keywords(category)
                        score = self._calculate_semantic_similarity_bert(text, category_keywords)
                    
                    category_scores[category] = float(score)
                    
                except Exception as e:
                    logger.warning(f"Error classifying category {category}: {e}")
                    # Fallback to keyword-based scoring for this category
                    keyword_score = self._calculate_keyword_score(text, category)
                    category_scores[category] = keyword_score
            
            # Normalize scores to sum to 1.0
            total_score = sum(category_scores.values())
            if total_score > 0:
                category_scores = {k: v / total_score for k, v in category_scores.items()}
            
            logger.info(f"BERT classification completed for {len(categories)} categories")
            return category_scores
            
        except Exception as e:
            logger.error(f"BERT classification error: {e}")
            if self.config.get('force_ml_models', False):
                raise ValueError(f"BERT classification failed in production mode: {e}")
            return self._classify_text_keywords(text, categories)
    
    def _initialize_bert_model(self):
        """Initialize BERT model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            model_name = self.config.get('bert_model', 'sentence-transformers/allenai-specter')
            device = self.config.get('device', 'cpu')
            cache_dir = self.config.get('model_cache_dir')
            
            logger.info(f"Loading BERT model: {model_name}")
            
            # Load tokenizer and model
            self.bert_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            self.bert_model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Move to appropriate device
            if device == 'cuda' and torch.cuda.is_available():
                self.bert_model = self.bert_model.cuda()
                logger.info("BERT model loaded on GPU")
            else:
                self.bert_model = self.bert_model.cpu()
                logger.info("BERT model loaded on CPU")
            
            self.bert_model.eval()
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT model: {e}")
            if self.config.get('force_ml_models', False):
                raise ValueError(f"BERT model initialization failed in production mode: {e}")
    
    def _load_category_classifier(self, classifier_path: str):
        """Load a pre-trained category-specific classifier"""
        try:
            import pickle
            import joblib
            
            # Try different loading methods
            if classifier_path.endswith('.pkl'):
                with open(classifier_path, 'rb') as f:
                    return pickle.load(f)
            elif classifier_path.endswith('.joblib'):
                return joblib.load(classifier_path)
            else:
                logger.warning(f"Unknown classifier format: {classifier_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load classifier {classifier_path}: {e}")
            return None
    
    def _calculate_semantic_similarity_bert(self, text: str, keywords: List[str]) -> float:
        """Calculate semantic similarity between text and keywords using BERT"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import torch
            
            # Get embeddings for input text
            text_inputs = self.bert_tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                text_outputs = self.bert_model(**text_inputs)
                text_embedding = text_outputs.last_hidden_state.mean(dim=1).numpy()
            
            # Get embeddings for keywords
            keyword_text = ' '.join(keywords)
            keyword_inputs = self.bert_tokenizer(
                keyword_text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                keyword_outputs = self.bert_model(**keyword_inputs)
                keyword_embedding = keyword_outputs.last_hidden_state.mean(dim=1).numpy()
            
            # Calculate cosine similarity
            similarity = cosine_similarity(text_embedding, keyword_embedding)[0][0]
            
            # Normalize to 0-1 range
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.error(f"Semantic similarity calculation error: {e}")
            return 0.0
    
    def _calculate_keyword_score(self, text: str, category: str) -> float:
        """Calculate keyword-based score for a category (fallback method)"""
        category_keywords = self._get_category_keywords(category)
        text_lower = text.lower()
        
        # Count keyword matches
        matches = sum(1 for keyword in category_keywords if keyword.lower() in text_lower)
        
        # Calculate score based on match ratio
        if category_keywords:
            score = matches / len(category_keywords)
        else:
            score = 0.0
        
        return min(1.0, score)  # Cap at 1.0
    
    def _classify_text_keywords(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Keyword-based text classification (DEVELOPMENT/TESTING ONLY)"""
        
        # PRODUCTION ENFORCEMENT: This method should not be reached in production
        if os.getenv('ENVIRONMENT', '').lower() == 'production':
            raise ValueError(
                "PRODUCTION VIOLATION: Keyword-based classification should not be reached in production mode. "
                "Configure BERT models properly. NEVER SACRIFICE QUALITY!!"
            )
        
        # Development/testing keyword-based classification
        logger.warning("USING KEYWORD CLASSIFICATION - NOT SUITABLE FOR PRODUCTION ML")
        
        # Simple keyword-based classification
        category_scores = {}
        for category in categories:
            category_keywords = self._get_category_keywords(category)
            score = sum(1 for keyword in category_keywords if keyword.lower() in text.lower())
            category_scores[category] = score / len(category_keywords) if category_keywords else 0
        
        return category_scores
    
    def _get_category_keywords(self, category: str) -> List[str]:
        """Get keywords for a category"""
        category_keywords = {
            'research': ['study', 'analysis', 'investigation', 'research', 'experiment'],
            'quality': ['quality', 'standard', 'compliance', 'validation', 'assessment'],
            'safety': ['safety', 'toxicity', 'risk', 'hazard', 'compliance'],
            'innovation': ['novel', 'innovative', 'breakthrough', 'new', 'original'],
            'methodology': ['method', 'procedure', 'protocol', 'technique', 'approach']
        }
        return category_keywords.get(category, [])

class QualityAssessor:
    """Quality assessment using ML"""
    
    def __init__(self):
        self.quality_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = [
            'word_count', 'sentence_count', 'paragraph_count',
            'citation_count', 'figure_count', 'table_count',
            'methodology_score', 'clarity_score', 'completeness_score'
        ]
        self._load_or_initialize_model()
    
    def _initialize_dummy_model(self):
        """Initialize dummy model with some basic training data"""
        # Create some dummy training data
        X_dummy = np.array([
            [100, 5, 3, 2, 1, 0, 0.5, 0.5, 0.5],  # Low quality
            [500, 25, 10, 15, 5, 3, 0.8, 0.8, 0.8],  # High quality
            [300, 15, 7, 8, 3, 2, 0.6, 0.7, 0.6],  # Medium quality
            [800, 40, 15, 20, 8, 5, 0.9, 0.9, 0.8],  # High quality
            [150, 8, 4, 3, 1, 1, 0.4, 0.5, 0.4],  # Low quality
        ])
        y_dummy = np.array([0, 1, 0, 1, 0])  # 0 = low quality, 1 = high quality
        
        # Fit scaler and model
        self.scaler.fit(X_dummy)
        self.quality_model.fit(X_dummy, y_dummy)
        logger.info("Initialized dummy quality assessment model")
    
    def _load_or_initialize_model(self):
        """Load existing model or initialize new one"""
        try:
            with open('quality_model.pkl', 'rb') as f:
                self.quality_model = pickle.load(f)
            with open('quality_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Loaded existing quality assessment model")
        except FileNotFoundError:
            logger.info("Initializing new quality assessment model")
            self._initialize_dummy_model()
    
    def extract_features(self, manuscript: Dict[str, Any]) -> np.ndarray:
        """Extract features from manuscript"""
        features = []
        
        # Basic text features
        text = manuscript.get('content', '')
        features.append(len(text.split()))  # word_count
        features.append(len(text.split('.')))  # sentence_count
        features.append(len(text.split('\n\n')))  # paragraph_count
        
        # Citation and figure features
        features.append(len(re.findall(r'\[.*?\]', text)))  # citation_count
        features.append(len(re.findall(r'Figure|Fig\.', text)))  # figure_count
        features.append(len(re.findall(r'Table|Tab\.', text)))  # table_count
        
        # Quality scores
        features.append(manuscript.get('methodology_score', 0.5))
        features.append(manuscript.get('clarity_score', 0.5))
        features.append(manuscript.get('completeness_score', 0.5))
        
        return np.array(features).reshape(1, -1)
    
    def assess_quality(self, manuscript: Dict[str, Any]) -> Dict[str, Any]:
        """Assess manuscript quality using ML or basic approach"""
        
        # Check if production ML models are available
        if getattr(self, 'quality_ensemble', None) is not None:
            return self._assess_quality_ml(manuscript)
        else:
            return self._assess_quality_basic(manuscript)
    
    def _assess_quality_ml(self, manuscript: Dict[str, Any]) -> Dict[str, Any]:
        """Production ML-based quality assessment (REQUIRES ML SETUP)"""
        try:
            # TODO: Implement ensemble ML quality assessment
            # This would require:
            # 1. BERT-based content quality assessment
            # 2. Statistical feature quality analysis  
            # 3. Writing quality assessment using language models
            # 4. Novelty detection using citation analysis
            # 5. Ensemble prediction combining all models
            
            logger.warning("ML quality assessment not yet implemented, falling back to basic")
            return self._assess_quality_basic(manuscript)
            
        except Exception as e:
            logger.error(f"ML quality assessment error: {e}")
            return self._assess_quality_basic(manuscript)
    
    def _assess_quality_basic(self, manuscript: Dict[str, Any]) -> Dict[str, Any]:
        """Basic quality assessment (FALLBACK - REPLACE WITH ML)"""
        features = self.extract_features(manuscript)
        features_scaled = self.scaler.transform(features)
        
        # Predict quality score
        quality_score = self.quality_model.predict_proba(features_scaled)[0][1]
        
        # Generate quality breakdown
        breakdown = {
            'overall_score': float(quality_score),
            'methodology': manuscript.get('methodology_score', 0.5),
            'clarity': manuscript.get('clarity_score', 0.5),
            'completeness': manuscript.get('completeness_score', 0.5),
            'technical_rigor': min(quality_score * 1.2, 1.0),
            'innovation': min(quality_score * 0.8, 1.0)
        }
        
        return breakdown
    
    def train_model(self, training_data: List[Dict[str, Any]]):
        """Train the quality assessment model"""
        X = []
        y = []
        
        for item in training_data:
            features = self.extract_features(item['manuscript'])
            X.append(features.flatten())
            y.append(item['quality_label'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.quality_model.fit(X_scaled, y)
        
        # Save model
        with open('quality_model.pkl', 'wb') as f:
            pickle.dump(self.quality_model, f)
        with open('quality_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info("Quality assessment model trained and saved")

class TrendPredictor:
    """Trend prediction using ML"""
    
    def __init__(self):
        self.trend_data = {}
        self.prediction_window = 6  # months
        self.lock = threading.RLock()
    
    def analyze_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in research data"""
        with self.lock:
            # Extract time series data
            time_series = {}
            for item in data:
                topic = item.get('topic', 'general')
                date = item.get('date', datetime.now())
                month_key = date.strftime('%Y-%m')
                
                if topic not in time_series:
                    time_series[topic] = {}
                if month_key not in time_series[topic]:
                    time_series[topic][month_key] = 0
                time_series[topic][month_key] += 1
            
            # Calculate trend indicators
            trends = {}
            for topic, data_points in time_series.items():
                if len(data_points) >= 3:
                    # Simple linear trend calculation
                    months = sorted(data_points.keys())
                    values = [data_points[month] for month in months]
                    
                    # Calculate trend slope
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    
                    trends[topic] = {
                        'current_volume': values[-1] if values else 0,
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                        'trend_strength': abs(slope),
                        'prediction': self._predict_future_volume(values, slope)
                    }
            
            return trends
    
    def _predict_future_volume(self, values: List[int], slope: float) -> Dict[str, Any]:
        """Predict future volume based on trend"""
        if not values:
            return {'next_month': 0, 'next_quarter': 0}
        
        current = values[-1]
        next_month = max(0, current + slope)
        next_quarter = max(0, current + slope * 3)
        
        return {
            'next_month': int(next_month),
            'next_quarter': int(next_quarter),
            'confidence': min(0.9, 1.0 - abs(slope) / max(values))
        }
    
    def identify_emerging_topics(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify emerging research topics"""
        # Analyze recent vs historical data
        recent_cutoff = datetime.now() - timedelta(days=90)
        recent_data = [item for item in data if item.get('date', datetime.now()) > recent_cutoff]
        historical_data = [item for item in data if item.get('date', datetime.now()) <= recent_cutoff]
        
        # Count topics
        recent_topics = {}
        historical_topics = {}
        
        for item in recent_data:
            topic = item.get('topic', 'general')
            recent_topics[topic] = recent_topics.get(topic, 0) + 1
        
        for item in historical_data:
            topic = item.get('topic', 'general')
            historical_topics[topic] = historical_topics.get(topic, 0) + 1
        
        # Calculate growth rates
        emerging_topics = []
        for topic in recent_topics:
            recent_count = recent_topics[topic]
            historical_count = historical_topics.get(topic, 0)
            
            if historical_count > 0:
                growth_rate = (recent_count - historical_count) / historical_count
                if growth_rate > 0.5:  # 50% growth threshold
                    emerging_topics.append({
                        'topic': topic,
                        'growth_rate': growth_rate,
                        'recent_volume': recent_count,
                        'historical_volume': historical_count
                    })
        
        # Sort by growth rate
        emerging_topics.sort(key=lambda x: x['growth_rate'], reverse=True)
        return emerging_topics

class DecisionEngine:
    """Main decision engine for autonomous agents"""
    
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.quality_assessor = QualityAssessor()
        self.trend_predictor = TrendPredictor()
        self.lock = threading.RLock()
        
        # Decision rules and weights
        self.decision_rules = {
            'quality_threshold': 0.7,
            'risk_tolerance_default': 0.5,
            'confidence_threshold': 0.8
        }
    
    def make_decision(self, context: DecisionContext) -> DecisionResult:
        """Make an autonomous decision based on context"""
        with self.lock:
            # Analyze input data
            analysis = self._analyze_input(context.input_data)
            
            # Evaluate options
            option_scores = []
            for option in context.available_options:
                score = self._evaluate_option(option, context, analysis)
                option_scores.append((option, score))
            
            # Select best option
            option_scores.sort(key=lambda x: x[1]['total_score'], reverse=True)
            best_option, best_score = option_scores[0]
            
            # Generate alternatives
            alternatives = [{'option': opt, 'score': score} for opt, score in option_scores[1:3]]
            
            # Assess risks
            risk_assessment = self._assess_risks(best_option, context, analysis)
            
            # Predict performance
            performance_prediction = self._predict_performance(best_option, context, analysis)
            
            return DecisionResult(
                decision=best_option,
                confidence_score=best_score['confidence'],
                reasoning=best_score['reasoning'],
                alternatives=alternatives,
                risk_assessment=risk_assessment,
                performance_prediction=performance_prediction
            )
    
    def _analyze_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input data using NLP and ML"""
        analysis = {}
        
        # Text analysis if present
        if 'text' in input_data:
            text = input_data['text']
            analysis['entities'] = self.nlp_processor.extract_entities(text)
            analysis['sentiment'] = self.nlp_processor.analyze_sentiment(text)
            analysis['classification'] = self.nlp_processor.classify_text(text, 
                ['research', 'quality', 'safety', 'innovation', 'methodology'])
        
        # Quality assessment if manuscript data present
        if 'manuscript' in input_data:
            analysis['quality'] = self.quality_assessor.assess_quality(input_data['manuscript'])
        
        # Trend analysis if research data present
        if 'research_data' in input_data:
            analysis['trends'] = self.trend_predictor.analyze_trends(input_data['research_data'])
            analysis['emerging_topics'] = self.trend_predictor.identify_emerging_topics(input_data['research_data'])
        
        return analysis
    
    def _evaluate_option(self, option: Dict[str, Any], context: DecisionContext, 
                        analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a decision option"""
        scores = {}
        reasoning = []
        
        # Quality score
        if 'quality' in analysis and 'quality_score' in option:
            quality_score = option['quality_score']
            scores['quality'] = quality_score
            reasoning.append(f"Quality score: {quality_score:.2f}")
        
        # Risk score
        risk_score = option.get('risk_score', 0.5)
        scores['risk'] = 1.0 - risk_score  # Invert so lower risk = higher score
        reasoning.append(f"Risk score: {1.0 - risk_score:.2f}")
        
        # Efficiency score
        efficiency_score = option.get('efficiency_score', 0.5)
        scores['efficiency'] = efficiency_score
        reasoning.append(f"Efficiency score: {efficiency_score:.2f}")
        
        # Goal alignment score
        goal_alignment = self._calculate_goal_alignment(option, context.goals, analysis)
        scores['goal_alignment'] = goal_alignment
        reasoning.append(f"Goal alignment: {goal_alignment:.2f}")
        
        # Calculate total score
        weights = {'quality': 0.3, 'risk': 0.25, 'efficiency': 0.25, 'goal_alignment': 0.2}
        total_score = sum(scores.get(key, 0) * weights[key] for key in weights)
        
        # Calculate confidence
        confidence = min(1.0, total_score * 1.2)  # Boost confidence for high scores
        
        return {
            'total_score': total_score,
            'confidence': confidence,
            'component_scores': scores,
            'reasoning': '; '.join(reasoning)
        }
    
    def _calculate_goal_alignment(self, option: Dict[str, Any], goals: List[str], 
                                analysis: Dict[str, Any]) -> float:
        """Calculate how well an option aligns with goals"""
        alignment_score = 0.5  # Default neutral score
        
        # Check for goal-specific features in the option
        for goal in goals:
            if goal.lower() in ['quality', 'safety'] and 'quality_score' in option:
                alignment_score = max(alignment_score, option['quality_score'])
            elif goal.lower() in ['efficiency', 'speed'] and 'efficiency_score' in option:
                alignment_score = max(alignment_score, option['efficiency_score'])
            elif goal.lower() in ['innovation'] and 'innovation_score' in option:
                alignment_score = max(alignment_score, option['innovation_score'])
        
        return alignment_score
    
    def _assess_risks(self, decision: Dict[str, Any], context: DecisionContext, 
                     analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks of a decision"""
        risks = {
            'technical_risk': decision.get('technical_risk', 0.3),
            'operational_risk': decision.get('operational_risk', 0.2),
            'compliance_risk': decision.get('compliance_risk', 0.1),
            'reputation_risk': decision.get('reputation_risk', 0.2)
        }
        
        # Adjust based on context
        if context.risk_tolerance < 0.3:
            # Conservative approach - increase risk perception
            risks = {k: min(1.0, v * 1.5) for k, v in risks.items()}
        
        total_risk = sum(risks.values()) / len(risks)
        
        return {
            'individual_risks': risks,
            'total_risk': total_risk,
            'risk_level': 'high' if total_risk > 0.7 else 'medium' if total_risk > 0.4 else 'low'
        }
    
    def _predict_performance(self, decision: Dict[str, Any], context: DecisionContext, 
                           analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Predict performance of a decision"""
        base_performance = decision.get('expected_performance', 0.7)
        
        # Adjust based on analysis
        if 'quality' in analysis:
            quality_factor = analysis['quality']['overall_score']
            base_performance *= (0.8 + quality_factor * 0.4)  # Quality boost
        
        if 'sentiment' in analysis:
            sentiment_factor = analysis['sentiment']['positive']
            base_performance *= (0.9 + sentiment_factor * 0.2)  # Sentiment boost
        
        return {
            'expected_performance': min(1.0, base_performance),
            'confidence_interval': [max(0.0, base_performance - 0.1), min(1.0, base_performance + 0.1)],
            'success_probability': base_performance
        }
    
    def update_decision_rules(self, new_rules: Dict[str, Any]):
        """Update decision rules based on learning"""
        with self.lock:
            self.decision_rules.update(new_rules)
            logger.info(f"Updated decision rules: {new_rules}")
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision engine statistics"""
        return {
            'total_decisions': 0,  # Would track in production
            'average_confidence': 0.8,  # Would calculate in production
            'success_rate': 0.85,  # Would track in production
            'model_versions': {
                'quality_assessor': '1.0',
                'trend_predictor': '1.0',
                'nlp_processor': '1.0'
            }
        }
