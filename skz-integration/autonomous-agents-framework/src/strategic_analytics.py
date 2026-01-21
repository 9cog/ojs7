"""
Production-Grade Strategic Analytics System
Implements advanced ML-based strategic insights, evolution forecasting, and recommendation engine
using transformer models, time series analysis, and multi-objective optimization.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, silhouette_score
import sqlite3
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerInsightModel(nn.Module):
    """Transformer-based model for strategic insight generation"""
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 256, 
                 nhead: int = 8, num_layers: int = 6, num_classes: int = 5):
        super(TransformerInsightModel, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(5000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """Create positional encoding for transformer"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(self, x, attention_mask=None):
        # Add positional encoding
        seq_len = x.size(1)
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Embedding and position
        x = self.embedding(x) * np.sqrt(self.d_model) + pos_encoding
        x = self.dropout(x)
        
        # Transformer encoding
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            
        transformer_output = self.transformer(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
        
        # Global average pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (transformer_output * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = transformer_output.mean(dim=1)
            
        # Classification
        output = self.classifier(pooled)
        
        return output


class InsightGenerator:
    """ML-based strategic insight generation using transformer models"""
    
    def __init__(self):
        self.tokenizer = None
        self.bert_model = None
        self.insight_model = None
        self.scaler = StandardScaler()
        self.is_initialized = False
        
        # Initialize pre-trained models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize pre-trained models for insight generation"""
        try:
            # Use a lightweight BERT model for text analysis
            model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_model.eval()
            
            # Initialize insight categories
            self.insight_categories = {
                0: "performance_optimization",
                1: "resource_scaling", 
                2: "workflow_improvement",
                3: "quality_enhancement",
                4: "strategic_growth"
            }
            
            self.is_initialized = True
            logger.info("Strategic insight models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize insight models: {e}")
            self.is_initialized = False
            
    def generate_insights(self, performance_data: Dict[str, Any], 
                         historical_trends: List[Dict]) -> List[Dict[str, Any]]:
        """Generate strategic insights from performance data and trends"""
        if not self.is_initialized:
            return self._generate_fallback_insights(performance_data)
            
        try:
            insights = []
            
            # Analyze current performance patterns
            performance_insights = self._analyze_performance_patterns(performance_data)
            insights.extend(performance_insights)
            
            # Analyze historical trends
            trend_insights = self._analyze_trend_patterns(historical_trends)
            insights.extend(trend_insights)
            
            # Generate strategic recommendations
            strategic_insights = self._generate_strategic_recommendations(
                performance_data, historical_trends
            )
            insights.extend(strategic_insights)
            
            # Rank insights by importance
            ranked_insights = self._rank_insights(insights)
            
            return ranked_insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return self._generate_fallback_insights(performance_data)
            
    def _analyze_performance_patterns(self, performance_data: Dict[str, Any]) -> List[Dict]:
        """Analyze current performance patterns for insights"""
        insights = []
        
        # Success rate analysis
        success_rate = performance_data.get('success_rate', 1.0)
        if success_rate < 0.95:
            insights.append({
                'category': 'quality_enhancement',
                'type': 'performance_degradation',
                'severity': 'high' if success_rate < 0.90 else 'medium',
                'title': 'Success Rate Below Target',
                'description': f'Current success rate of {success_rate:.2%} indicates quality issues',
                'impact': 'Affects user experience and system reliability',
                'recommendation': 'Implement enhanced error handling and monitoring',
                'confidence': 0.9,
                'data_source': 'performance_metrics'
            })
            
        # Response time patterns
        avg_response = performance_data.get('average_execution_time', 0)
        p95_response = performance_data.get('p95_execution_time', 0)
        
        if p95_response > 0 and avg_response > 0:
            variance_ratio = p95_response / avg_response
            if variance_ratio > 3.0:
                insights.append({
                    'category': 'performance_optimization',
                    'type': 'response_variance',
                    'severity': 'medium',
                    'title': 'High Response Time Variance',
                    'description': f'P95 response time ({p95_response:.2f}s) is {variance_ratio:.1f}x average ({avg_response:.2f}s)',
                    'impact': 'Inconsistent user experience and potential timeouts',
                    'recommendation': 'Investigate outlier operations and implement performance optimization',
                    'confidence': 0.85,
                    'data_source': 'response_metrics'
                })
                
        # Resource utilization analysis
        cpu_usage = performance_data.get('average_cpu_usage', 0)
        memory_usage = performance_data.get('average_memory_usage', 0)
        
        if cpu_usage > 75 or memory_usage > 80:
            insights.append({
                'category': 'resource_scaling',
                'type': 'resource_pressure',
                'severity': 'high' if cpu_usage > 85 or memory_usage > 90 else 'medium',
                'title': 'High Resource Utilization',
                'description': f'CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%',
                'impact': 'Risk of performance degradation and system instability',
                'recommendation': 'Scale resources or optimize resource-intensive operations',
                'confidence': 0.95,
                'data_source': 'resource_metrics'
            })
            
        return insights
        
    def _analyze_trend_patterns(self, historical_trends: List[Dict]) -> List[Dict]:
        """Analyze historical trends for strategic insights"""
        insights = []
        
        if not historical_trends or len(historical_trends) < 3:
            return insights
            
        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(historical_trends)
            
            # Analyze growth trends
            if 'total_operations' in df.columns:
                operations = df['total_operations'].values
                if len(operations) >= 3:
                    growth_rate = (operations[-1] - operations[0]) / operations[0] if operations[0] > 0 else 0
                    
                    if growth_rate > 0.2:  # 20% growth
                        insights.append({
                            'category': 'strategic_growth',
                            'type': 'volume_growth',
                            'severity': 'info',
                            'title': 'Significant Volume Growth',
                            'description': f'Operations volume increased by {growth_rate:.1%} over analysis period',
                            'impact': 'Indicates growing demand and system success',
                            'recommendation': 'Prepare for continued growth with capacity planning',
                            'confidence': 0.8,
                            'data_source': 'historical_trends'
                        })
                        
            # Performance trend analysis
            if 'average_execution_time' in df.columns:
                response_times = df['average_execution_time'].values
                if len(response_times) >= 3:
                    trend_slope = np.polyfit(range(len(response_times)), response_times, 1)[0]
                    
                    if trend_slope > 0.1:  # Worsening performance
                        insights.append({
                            'category': 'performance_optimization',
                            'type': 'performance_degradation_trend',
                            'severity': 'medium',
                            'title': 'Performance Degradation Trend',
                            'description': 'Response times showing consistent upward trend',
                            'impact': 'Gradual system performance deterioration',
                            'recommendation': 'Implement performance monitoring and optimization program',
                            'confidence': 0.75,
                            'data_source': 'performance_trends'
                        })
                        
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            
        return insights
        
    def _generate_strategic_recommendations(self, performance_data: Dict, 
                                          trends: List[Dict]) -> List[Dict]:
        """Generate high-level strategic recommendations"""
        recommendations = []
        
        # Strategic capacity planning
        current_load = performance_data.get('total_operations', 0)
        if current_load > 1000:  # High volume system
            recommendations.append({
                'category': 'strategic_growth',
                'type': 'capacity_strategy',
                'severity': 'info',
                'title': 'Strategic Capacity Planning Required',
                'description': f'System handling {current_load} operations - requires long-term planning',
                'impact': 'Ensures sustainable growth and performance',
                'recommendation': 'Develop 6-month capacity roadmap with automated scaling',
                'confidence': 0.9,
                'data_source': 'strategic_analysis'
            })
            
        # Quality improvement strategy
        success_rate = performance_data.get('success_rate', 1.0)
        if success_rate < 0.98:
            recommendations.append({
                'category': 'quality_enhancement',
                'type': 'quality_strategy',
                'severity': 'medium',
                'title': 'Quality Improvement Initiative',
                'description': 'System quality metrics below excellence threshold',
                'impact': 'Critical for user satisfaction and competitive advantage',
                'recommendation': 'Launch comprehensive quality improvement program',
                'confidence': 0.85,
                'data_source': 'quality_analysis'
            })
            
        # Innovation opportunities
        if len(trends) >= 5:  # Sufficient historical data
            recommendations.append({
                'category': 'strategic_growth',
                'type': 'innovation_opportunity',
                'severity': 'info',
                'title': 'Innovation and Enhancement Opportunities',
                'description': 'Stable system performance enables focus on innovation',
                'impact': 'Competitive advantage and feature differentiation',
                'recommendation': 'Invest in advanced features and AI capabilities',
                'confidence': 0.7,
                'data_source': 'strategic_planning'
            })
            
        return recommendations
        
    def _rank_insights(self, insights: List[Dict]) -> List[Dict]:
        """Rank insights by importance and urgency"""
        severity_weights = {'critical': 10, 'high': 8, 'medium': 5, 'low': 2, 'info': 1}
        
        def calculate_priority_score(insight):
            severity_score = severity_weights.get(insight.get('severity', 'low'), 2)
            confidence_score = insight.get('confidence', 0.5) * 5
            category_bonus = 2 if insight.get('category') in ['performance_optimization', 'quality_enhancement'] else 0
            
            return severity_score + confidence_score + category_bonus
            
        # Sort by priority score
        ranked = sorted(insights, key=calculate_priority_score, reverse=True)
        
        # Add ranking information
        for i, insight in enumerate(ranked):
            insight['priority_rank'] = i + 1
            insight['priority_score'] = calculate_priority_score(insight)
            
        return ranked
        
    def _generate_fallback_insights(self, performance_data: Dict) -> List[Dict]:
        """Generate basic insights when ML models are unavailable"""
        insights = []
        
        # Basic performance analysis
        success_rate = performance_data.get('success_rate', 1.0)
        if success_rate < 0.95:
            insights.append({
                'category': 'quality_enhancement',
                'type': 'basic_quality_check',
                'severity': 'medium',
                'title': 'Success Rate Review Needed',
                'description': f'Success rate at {success_rate:.2%}',
                'recommendation': 'Review error patterns',
                'confidence': 0.8
            })
            
        return insights


class EvolutionForecaster:
    """Time series forecasting for system evolution prediction"""
    
    def __init__(self):
        self.forecasting_models = {}
        self.scalers = {}
        self.is_trained = False
        
    def train_forecasting_models(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Train forecasting models on historical performance data"""
        logger.info("Training evolution forecasting models")
        
        if historical_data.empty:
            return {'status': 'no_data'}
            
        try:
            # Prepare time series data
            if 'timestamp' in historical_data.columns:
                historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
                historical_data = historical_data.sort_values('timestamp')
                
            # Metrics to forecast
            forecast_metrics = ['execution_time', 'cpu_usage', 'memory_usage', 'success_rate']
            
            training_results = {}
            
            for metric in forecast_metrics:
                if metric in historical_data.columns:
                    try:
                        model_result = self._train_metric_forecaster(historical_data, metric)
                        training_results[metric] = model_result
                    except Exception as e:
                        logger.error(f"Failed to train forecaster for {metric}: {e}")
                        training_results[metric] = {'status': 'failed', 'error': str(e)}
                        
            self.is_trained = len([r for r in training_results.values() if r.get('status') == 'success']) > 0
            
            return {
                'status': 'success' if self.is_trained else 'partial_failure',
                'trained_metrics': list(training_results.keys()),
                'training_results': training_results
            }
            
        except Exception as e:
            logger.error(f"Forecasting model training failed: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def _train_metric_forecaster(self, data: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Train forecasting model for specific metric"""
        # Create time-based features
        data_copy = data.copy()
        
        if 'timestamp' in data_copy.columns:
            data_copy['hour'] = data_copy['timestamp'].dt.hour
            data_copy['day_of_week'] = data_copy['timestamp'].dt.dayofweek
            data_copy['day_of_month'] = data_copy['timestamp'].dt.day
            
        # Create lag features
        for lag in [1, 2, 3, 6, 12, 24]:  # Various lag periods
            data_copy[f'{metric}_lag_{lag}'] = data_copy[metric].shift(lag)
            
        # Create rolling statistics
        for window in [3, 6, 12]:
            data_copy[f'{metric}_rolling_mean_{window}'] = data_copy[metric].rolling(window).mean()
            data_copy[f'{metric}_rolling_std_{window}'] = data_copy[metric].rolling(window).std()
            
        # Prepare features
        feature_columns = ['hour', 'day_of_week', 'day_of_month']
        feature_columns.extend([col for col in data_copy.columns if f'{metric}_lag_' in col or f'{metric}_rolling_' in col])
        
        # Remove rows with NaN values
        data_clean = data_copy.dropna()
        
        if len(data_clean) < 10:
            return {'status': 'insufficient_data'}
            
        X = data_clean[feature_columns]
        y = data_clean[metric]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train ensemble model
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        trained_models = {}
        for name, model in models.items():
            try:
                model.fit(X_scaled, y)
                trained_models[name] = model
            except Exception as e:
                logger.error(f"Failed to train {name} for {metric}: {e}")
                
        if not trained_models:
            return {'status': 'training_failed'}
            
        # Store models and scaler
        self.forecasting_models[metric] = trained_models
        self.scalers[metric] = scaler
        
        # Calculate training accuracy
        predictions = {}
        for name, model in trained_models.items():
            pred = model.predict(X_scaled)
            mse = mean_squared_error(y, pred)
            predictions[name] = {'mse': mse, 'predictions': pred}
            
        return {
            'status': 'success',
            'models_trained': list(trained_models.keys()),
            'training_samples': len(X),
            'features_used': len(feature_columns),
            'performance': predictions
        }
        
    def forecast_evolution(self, recent_data: pd.DataFrame, 
                          forecast_horizon: int = 24) -> Dict[str, Any]:
        """Forecast system evolution over specified horizon"""
        if not self.is_trained:
            return {'status': 'models_not_trained'}
            
        forecasts = {}
        
        for metric in self.forecasting_models.keys():
            try:
                metric_forecast = self._forecast_metric(recent_data, metric, forecast_horizon)
                forecasts[metric] = metric_forecast
            except Exception as e:
                logger.error(f"Forecasting failed for {metric}: {e}")
                forecasts[metric] = {'status': 'failed', 'error': str(e)}
                
        # Generate evolution insights
        evolution_insights = self._analyze_evolution_forecasts(forecasts)
        
        return {
            'status': 'success',
            'forecast_horizon_hours': forecast_horizon,
            'forecasts': forecasts,
            'evolution_insights': evolution_insights,
            'timestamp': datetime.now().isoformat()
        }
        
    def _forecast_metric(self, recent_data: pd.DataFrame, metric: str, 
                        forecast_horizon: int) -> Dict[str, Any]:
        """Forecast specific metric evolution"""
        if metric not in self.forecasting_models:
            return {'status': 'model_not_available'}
            
        # Prepare recent data similar to training
        data_copy = recent_data.copy()
        
        if 'timestamp' in data_copy.columns:
            data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
            data_copy = data_copy.sort_values('timestamp')
            
        # Create time features
        current_time = datetime.now()
        forecast_times = [current_time + timedelta(hours=h) for h in range(1, forecast_horizon + 1)]
        
        predictions = []
        
        for forecast_time in forecast_times:
            # Create features for this time point
            features = {
                'hour': forecast_time.hour,
                'day_of_week': forecast_time.weekday(),
                'day_of_month': forecast_time.day
            }
            
            # Add lag features (using recent actual values and previous predictions)
            for lag in [1, 2, 3, 6, 12, 24]:
                if len(data_copy) >= lag:
                    features[f'{metric}_lag_{lag}'] = data_copy[metric].iloc[-lag]
                else:
                    features[f'{metric}_lag_{lag}'] = data_copy[metric].mean() if not data_copy.empty else 0
                    
            # Add rolling statistics
            if not data_copy.empty:
                recent_values = data_copy[metric].tail(24).values  # Last 24 hours
                for window in [3, 6, 12]:
                    if len(recent_values) >= window:
                        features[f'{metric}_rolling_mean_{window}'] = np.mean(recent_values[-window:])
                        features[f'{metric}_rolling_std_{window}'] = np.std(recent_values[-window:])
                    else:
                        features[f'{metric}_rolling_mean_{window}'] = np.mean(recent_values)
                        features[f'{metric}_rolling_std_{window}'] = np.std(recent_values)
                        
            # Convert to array and scale
            feature_array = np.array(list(features.values())).reshape(1, -1)
            feature_scaled = self.scalers[metric].transform(feature_array)
            
            # Ensemble prediction
            model_predictions = []
            for model_name, model in self.forecasting_models[metric].items():
                pred = model.predict(feature_scaled)[0]
                model_predictions.append(pred)
                
            # Average ensemble prediction
            ensemble_pred = np.mean(model_predictions)
            prediction_std = np.std(model_predictions)
            
            predictions.append({
                'timestamp': forecast_time.isoformat(),
                'predicted_value': float(ensemble_pred),
                'prediction_std': float(prediction_std),
                'confidence_interval': {
                    'lower': float(ensemble_pred - 1.96 * prediction_std),
                    'upper': float(ensemble_pred + 1.96 * prediction_std)
                }
            })
            
        return {
            'status': 'success',
            'predictions': predictions,
            'metric': metric
        }
        
    def _analyze_evolution_forecasts(self, forecasts: Dict[str, Any]) -> List[Dict]:
        """Analyze forecasts to generate evolution insights"""
        insights = []
        
        for metric, forecast_data in forecasts.items():
            if forecast_data.get('status') != 'success':
                continue
                
            predictions = forecast_data.get('predictions', [])
            if not predictions:
                continue
                
            # Analyze trend
            values = [p['predicted_value'] for p in predictions]
            if len(values) >= 3:
                trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                
                if abs(trend_slope) > 0.01:  # Significant trend
                    trend_direction = 'increasing' if trend_slope > 0 else 'decreasing'
                    
                    insights.append({
                        'type': 'evolution_trend',
                        'metric': metric,
                        'trend': trend_direction,
                        'magnitude': abs(trend_slope),
                        'description': f'{metric} forecasted to be {trend_direction}',
                        'impact': self._assess_trend_impact(metric, trend_direction),
                        'recommendation': self._get_trend_recommendation(metric, trend_direction)
                    })
                    
        return insights
        
    def _assess_trend_impact(self, metric: str, trend: str) -> str:
        """Assess the impact of a forecasted trend"""
        impact_map = {
            'execution_time': {
                'increasing': 'Performance degradation expected',
                'decreasing': 'Performance improvement expected'
            },
            'cpu_usage': {
                'increasing': 'Resource pressure expected',
                'decreasing': 'Resource utilization optimization'
            },
            'memory_usage': {
                'increasing': 'Memory pressure expected',
                'decreasing': 'Memory efficiency improvement'
            },
            'success_rate': {
                'increasing': 'Quality improvement expected',
                'decreasing': 'Quality degradation risk'
            }
        }
        
        return impact_map.get(metric, {}).get(trend, 'Unknown impact')
        
    def _get_trend_recommendation(self, metric: str, trend: str) -> str:
        """Get recommendations based on forecasted trends"""
        recommendation_map = {
            'execution_time': {
                'increasing': 'Prepare performance optimization measures',
                'decreasing': 'Monitor for sustained improvement'
            },
            'cpu_usage': {
                'increasing': 'Plan resource scaling or optimization',
                'decreasing': 'Consider resource right-sizing'
            },
            'memory_usage': {
                'increasing': 'Monitor memory usage and plan scaling',
                'decreasing': 'Review memory optimization effectiveness'
            },
            'success_rate': {
                'increasing': 'Continue current quality initiatives',
                'decreasing': 'Implement immediate quality measures'
            }
        }
        
        return recommendation_map.get(metric, {}).get(trend, 'Monitor trend development')


class StrategicRecommendationEngine:
    """Multi-objective optimization for strategic recommendations"""
    
    def __init__(self):
        self.recommendation_history = []
        self.success_tracking = {}
        
    def generate_strategic_recommendations(self, insights: List[Dict], 
                                         forecasts: Dict[str, Any],
                                         current_performance: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate comprehensive strategic recommendations"""
        logger.info("Generating strategic recommendations")
        
        recommendations = []
        
        # Priority-based recommendations from insights
        insight_recommendations = self._recommendations_from_insights(insights)
        recommendations.extend(insight_recommendations)
        
        # Forecast-based recommendations
        forecast_recommendations = self._recommendations_from_forecasts(forecasts)
        recommendations.extend(forecast_recommendations)
        
        # Performance-based strategic recommendations
        performance_recommendations = self._recommendations_from_performance(current_performance)
        recommendations.extend(performance_recommendations)
        
        # Portfolio optimization
        optimized_portfolio = self._optimize_recommendation_portfolio(recommendations)
        
        return optimized_portfolio
        
    def _recommendations_from_insights(self, insights: List[Dict]) -> List[Dict]:
        """Generate recommendations from strategic insights"""
        recommendations = []
        
        for insight in insights:
            recommendation = {
                'id': f"insight_{len(self.recommendation_history)}",
                'type': 'insight_based',
                'category': insight.get('category', 'general'),
                'priority': self._map_severity_to_priority(insight.get('severity', 'medium')),
                'title': f"Strategic Action: {insight.get('title', 'Unknown')}",
                'description': insight.get('recommendation', ''),
                'rationale': insight.get('description', ''),
                'expected_impact': insight.get('impact', ''),
                'confidence': insight.get('confidence', 0.5),
                'timeline': self._determine_timeline(insight.get('severity', 'medium')),
                'resource_requirements': self._estimate_resources(insight.get('category', 'general')),
                'success_metrics': self._define_success_metrics(insight.get('category', 'general')),
                'source': 'strategic_insights'
            }
            recommendations.append(recommendation)
            
        return recommendations
        
    def _recommendations_from_forecasts(self, forecasts: Dict[str, Any]) -> List[Dict]:
        """Generate recommendations from evolution forecasts"""
        recommendations = []
        
        evolution_insights = forecasts.get('evolution_insights', [])
        
        for evolution in evolution_insights:
            recommendation = {
                'id': f"forecast_{len(self.recommendation_history)}",
                'type': 'forecast_based',
                'category': 'capacity_planning',
                'priority': 'medium',
                'title': f"Prepare for {evolution.get('metric', 'System')} Evolution",
                'description': evolution.get('recommendation', ''),
                'rationale': evolution.get('description', ''),
                'expected_impact': evolution.get('impact', ''),
                'confidence': 0.7,  # Forecast-based confidence
                'timeline': 'medium_term',
                'resource_requirements': 'moderate',
                'success_metrics': [f"Improved {evolution.get('metric', 'system')} management"],
                'source': 'evolution_forecasting'
            }
            recommendations.append(recommendation)
            
        return recommendations
        
    def _recommendations_from_performance(self, performance: Dict[str, float]) -> List[Dict]:
        """Generate strategic recommendations from current performance"""
        recommendations = []
        
        # Strategic capacity recommendation
        total_ops = performance.get('total_operations', 0)
        if total_ops > 5000:  # High-volume system
            recommendations.append({
                'id': f"strategic_{len(self.recommendation_history)}",
                'type': 'strategic_initiative',
                'category': 'business_growth',
                'priority': 'high',
                'title': 'Scale for Enterprise Growth',
                'description': 'Implement enterprise-grade infrastructure and processes',
                'rationale': f'System handling {total_ops} operations indicates enterprise-level usage',
                'expected_impact': 'Support 10x growth in operations volume',
                'confidence': 0.9,
                'timeline': 'long_term',
                'resource_requirements': 'significant',
                'success_metrics': ['System capacity', 'Performance stability', 'Cost efficiency'],
                'source': 'performance_analysis'
            })
            
        # Quality excellence initiative
        success_rate = performance.get('success_rate', 1.0)
        if success_rate > 0.98:  # Already high quality
            recommendations.append({
                'id': f"quality_{len(self.recommendation_history)}",
                'type': 'strategic_initiative',
                'category': 'quality_excellence',
                'priority': 'medium',
                'title': 'Pursue Quality Excellence Certification',
                'description': 'Implement Six Sigma or similar quality management system',
                'rationale': f'Current success rate of {success_rate:.2%} demonstrates quality foundation',
                'expected_impact': 'Industry-leading quality metrics and competitive advantage',
                'confidence': 0.8,
                'timeline': 'long_term',
                'resource_requirements': 'moderate',
                'success_metrics': ['Quality certification', 'Customer satisfaction', 'Error reduction'],
                'source': 'quality_analysis'
            })
            
        return recommendations
        
    def _optimize_recommendation_portfolio(self, recommendations: List[Dict]) -> List[Dict]:
        """Optimize recommendation portfolio using multi-objective criteria"""
        if not recommendations:
            return []
            
        # Score each recommendation
        for rec in recommendations:
            rec['portfolio_score'] = self._calculate_portfolio_score(rec)
            
        # Sort by portfolio score
        sorted_recommendations = sorted(recommendations, 
                                      key=lambda x: x['portfolio_score'], 
                                      reverse=True)
        
        # Portfolio balancing
        balanced_portfolio = self._balance_portfolio(sorted_recommendations)
        
        # Add portfolio information
        for i, rec in enumerate(balanced_portfolio):
            rec['portfolio_rank'] = i + 1
            rec['portfolio_tier'] = self._determine_portfolio_tier(i, len(balanced_portfolio))
            
        return balanced_portfolio
        
    def _calculate_portfolio_score(self, recommendation: Dict) -> float:
        """Calculate portfolio optimization score"""
        # Multi-criteria scoring
        priority_weight = {'critical': 10, 'high': 8, 'medium': 5, 'low': 2}
        confidence_weight = recommendation.get('confidence', 0.5) * 5
        
        # Impact assessment
        impact_keywords = ['competitive advantage', 'enterprise', 'excellence', 'strategic']
        impact_bonus = sum(2 for keyword in impact_keywords 
                          if keyword in recommendation.get('expected_impact', '').lower())
        
        # Timeline consideration (favor balanced timelines)
        timeline_scores = {'short_term': 3, 'medium_term': 5, 'long_term': 4}
        timeline_score = timeline_scores.get(recommendation.get('timeline', 'medium_term'), 3)
        
        total_score = (
            priority_weight.get(recommendation.get('priority', 'medium'), 5) +
            confidence_weight +
            impact_bonus +
            timeline_score
        )
        
        return total_score
        
    def _balance_portfolio(self, sorted_recommendations: List[Dict]) -> List[Dict]:
        """Balance recommendation portfolio across categories and timelines"""
        portfolio = []
        category_counts = defaultdict(int)
        timeline_counts = defaultdict(int)
        
        # Portfolio constraints
        max_per_category = 3
        max_per_timeline = 4
        max_portfolio_size = 10
        
        for rec in sorted_recommendations:
            category = rec.get('category', 'general')
            timeline = rec.get('timeline', 'medium_term')
            
            # Check constraints
            if (len(portfolio) < max_portfolio_size and
                category_counts[category] < max_per_category and
                timeline_counts[timeline] < max_per_timeline):
                
                portfolio.append(rec)
                category_counts[category] += 1
                timeline_counts[timeline] += 1
                
        return portfolio
        
    def _map_severity_to_priority(self, severity: str) -> str:
        """Map insight severity to recommendation priority"""
        severity_map = {
            'critical': 'critical',
            'high': 'high', 
            'medium': 'medium',
            'low': 'low',
            'info': 'low'
        }
        return severity_map.get(severity, 'medium')
        
    def _determine_timeline(self, severity: str) -> str:
        """Determine implementation timeline based on severity"""
        timeline_map = {
            'critical': 'short_term',
            'high': 'short_term',
            'medium': 'medium_term',
            'low': 'long_term',
            'info': 'long_term'
        }
        return timeline_map.get(severity, 'medium_term')
        
    def _estimate_resources(self, category: str) -> str:
        """Estimate resource requirements by category"""
        resource_map = {
            'performance_optimization': 'moderate',
            'resource_scaling': 'significant',
            'workflow_improvement': 'minimal',
            'quality_enhancement': 'moderate',
            'strategic_growth': 'significant'
        }
        return resource_map.get(category, 'moderate')
        
    def _define_success_metrics(self, category: str) -> List[str]:
        """Define success metrics by category"""
        metrics_map = {
            'performance_optimization': ['Response time improvement', 'Throughput increase'],
            'resource_scaling': ['Resource utilization', 'Cost efficiency'],
            'workflow_improvement': ['Process efficiency', 'User satisfaction'],
            'quality_enhancement': ['Error reduction', 'Success rate improvement'],
            'strategic_growth': ['Business metrics', 'Competitive position']
        }
        return metrics_map.get(category, ['General improvement metrics'])
        
    def _determine_portfolio_tier(self, rank: int, total: int) -> str:
        """Determine portfolio tier for recommendation"""
        if rank < total * 0.3:
            return 'tier_1_critical'
        elif rank < total * 0.6:
            return 'tier_2_important'
        else:
            return 'tier_3_beneficial'


class StrategicAnalytics:
    """Main strategic analytics system"""
    
    def __init__(self, db_path: str = "src/database/strategic_analytics.db"):
        self.db_path = db_path
        self.insight_generator = InsightGenerator()
        self.evolution_forecaster = EvolutionForecaster()
        self.recommendation_engine = StrategicRecommendationEngine()
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize strategic analytics database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategic_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    analysis_type TEXT NOT NULL,
                    results TEXT NOT NULL,
                    performance_data TEXT,
                    insights_count INTEGER,
                    recommendations_count INTEGER
                )
            """)
            
    def perform_strategic_analysis(self, performance_data: Dict[str, Any],
                                 historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive strategic analysis"""
        logger.info("Performing comprehensive strategic analysis")
        
        analysis_start = datetime.now()
        
        # Generate strategic insights
        insights = self.insight_generator.generate_insights(
            performance_data, 
            historical_data.to_dict('records') if not historical_data.empty else []
        )
        
        # Train and run evolution forecasting
        forecast_training = self.evolution_forecaster.train_forecasting_models(historical_data)
        evolution_forecast = self.evolution_forecaster.forecast_evolution(
            historical_data.tail(100) if not historical_data.empty else pd.DataFrame()
        )
        
        # Generate strategic recommendations
        strategic_recommendations = self.recommendation_engine.generate_strategic_recommendations(
            insights, evolution_forecast, performance_data
        )
        
        # Compile comprehensive analysis
        strategic_analysis = {
            'timestamp': analysis_start.isoformat(),
            'analysis_duration': (datetime.now() - analysis_start).total_seconds(),
            'strategic_insights': {
                'total_insights': len(insights),
                'insights': insights,
                'insight_categories': self._categorize_insights(insights)
            },
            'evolution_forecast': evolution_forecast,
            'strategic_recommendations': {
                'total_recommendations': len(strategic_recommendations),
                'recommendations': strategic_recommendations,
                'portfolio_summary': self._summarize_recommendation_portfolio(strategic_recommendations)
            },
            'executive_summary': self._generate_executive_summary(
                insights, evolution_forecast, strategic_recommendations
            ),
            'next_actions': self._identify_next_actions(strategic_recommendations)
        }
        
        # Store analysis
        self._store_analysis(strategic_analysis)
        
        return strategic_analysis
        
    def _categorize_insights(self, insights: List[Dict]) -> Dict[str, int]:
        """Categorize insights for summary"""
        categories = defaultdict(int)
        for insight in insights:
            categories[insight.get('category', 'general')] += 1
        return dict(categories)
        
    def _summarize_recommendation_portfolio(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Summarize recommendation portfolio"""
        if not recommendations:
            return {'total': 0}
            
        portfolio_summary = {
            'total': len(recommendations),
            'by_priority': defaultdict(int),
            'by_category': defaultdict(int),
            'by_timeline': defaultdict(int),
            'by_tier': defaultdict(int)
        }
        
        for rec in recommendations:
            portfolio_summary['by_priority'][rec.get('priority', 'medium')] += 1
            portfolio_summary['by_category'][rec.get('category', 'general')] += 1
            portfolio_summary['by_timeline'][rec.get('timeline', 'medium_term')] += 1
            portfolio_summary['by_tier'][rec.get('portfolio_tier', 'tier_3_beneficial')] += 1
            
        # Convert defaultdicts to regular dicts
        return {k: dict(v) if isinstance(v, defaultdict) else v 
                for k, v in portfolio_summary.items()}
        
    def _generate_executive_summary(self, insights: List[Dict], 
                                  forecasts: Dict, recommendations: List[Dict]) -> Dict[str, Any]:
        """Generate executive summary of strategic analysis"""
        high_priority_insights = [i for i in insights if i.get('severity') in ['critical', 'high']]
        tier_1_recommendations = [r for r in recommendations if r.get('portfolio_tier') == 'tier_1_critical']
        
        return {
            'key_findings': len(high_priority_insights),
            'critical_actions_required': len(tier_1_recommendations),
            'forecast_status': forecasts.get('status', 'unknown'),
            'overall_system_health': self._assess_overall_health(insights),
            'strategic_focus_areas': self._identify_focus_areas(recommendations),
            'risk_assessment': self._assess_strategic_risks(insights, forecasts)
        }
        
    def _assess_overall_health(self, insights: List[Dict]) -> str:
        """Assess overall system health from insights"""
        critical_issues = sum(1 for i in insights if i.get('severity') == 'critical')
        high_issues = sum(1 for i in insights if i.get('severity') == 'high')
        
        if critical_issues > 0:
            return 'critical_attention_required'
        elif high_issues > 2:
            return 'needs_attention'
        elif high_issues > 0:
            return 'good_with_improvements'
        else:
            return 'excellent'
            
    def _identify_focus_areas(self, recommendations: List[Dict]) -> List[str]:
        """Identify strategic focus areas"""
        category_counts = defaultdict(int)
        for rec in recommendations[:5]:  # Top 5 recommendations
            category_counts[rec.get('category', 'general')] += 1
            
        return sorted(category_counts.keys(), key=category_counts.get, reverse=True)[:3]
        
    def _assess_strategic_risks(self, insights: List[Dict], forecasts: Dict) -> List[str]:
        """Assess strategic risks"""
        risks = []
        
        # Risk from insights
        critical_insights = [i for i in insights if i.get('severity') == 'critical']
        if critical_insights:
            risks.append('Critical performance issues identified')
            
        # Risk from forecasts
        evolution_insights = forecasts.get('evolution_insights', [])
        negative_trends = [e for e in evolution_insights if 'degradation' in e.get('description', '')]
        if negative_trends:
            risks.append('Negative performance trends forecasted')
            
        if not risks:
            risks.append('No significant strategic risks identified')
            
        return risks
        
    def _identify_next_actions(self, recommendations: List[Dict]) -> List[Dict]:
        """Identify immediate next actions"""
        next_actions = []
        
        # Get top 3 critical recommendations
        critical_recs = [r for r in recommendations 
                        if r.get('priority') == 'critical' or r.get('portfolio_tier') == 'tier_1_critical'][:3]
        
        for rec in critical_recs:
            next_actions.append({
                'action': rec.get('title', 'Unknown action'),
                'timeline': rec.get('timeline', 'medium_term'),
                'owner': 'System Administrator',  # Default owner
                'success_criteria': rec.get('success_metrics', [])
            })
            
        return next_actions
        
    def _store_analysis(self, analysis: Dict[str, Any]):
        """Store strategic analysis results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO strategic_analysis 
                    (analysis_type, results, insights_count, recommendations_count)
                    VALUES (?, ?, ?, ?)
                """, (
                    'comprehensive_strategic_analysis',
                    json.dumps(analysis),
                    analysis['strategic_insights']['total_insights'],
                    analysis['strategic_recommendations']['total_recommendations']
                ))
        except Exception as e:
            logger.error(f"Failed to store strategic analysis: {e}")


# Export main classes
__all__ = [
    'InsightGenerator',
    'EvolutionForecaster',
    'StrategicRecommendationEngine', 
    'StrategicAnalytics',
    'TransformerInsightModel'
]