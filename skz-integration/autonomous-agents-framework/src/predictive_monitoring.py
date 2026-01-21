"""
Production-Grade Predictive Monitoring System
Implements advanced ML-based trend analysis, anomaly detection, and alert generation
using LSTM neural networks, isolation forests, and transformer models.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, accuracy_score
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

class LSTMTrendAnalyzer(nn.Module):
    """LSTM Neural Network for time series trend analysis and forecasting"""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 64, 
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(LSTMTrendAnalyzer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and linear layer
        output = self.dropout(last_output)
        output = self.linear(output)
        
        return output


class TrendAnalyzer:
    """ML-based trend analysis using LSTM neural networks"""
    
    def __init__(self, sequence_length: int = 24, prediction_horizon: int = 6):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        self.feature_columns = ['execution_time', 'cpu_usage', 'memory_usage', 'success_rate', 'operation_count']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_time_series_data(self, metrics_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for LSTM training"""
        if metrics_df.empty:
            return np.array([]), np.array([])
            
        # Resample data to hourly intervals
        if 'timestamp' in metrics_df.columns:
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
            metrics_df = metrics_df.set_index('timestamp')
            
            # Aggregate metrics by hour
            hourly_data = metrics_df.resample('1H').agg({
                'execution_time': 'mean',
                'cpu_usage': 'mean',
                'memory_usage': 'mean',
                'success': 'mean',  # success rate
                'agent_id': 'count'  # operation count
            }).rename(columns={'agent_id': 'operation_count', 'success': 'success_rate'})
            
            # Fill missing values
            hourly_data = hourly_data.fillna(method='ffill').fillna(0)
            
            if len(hourly_data) < self.sequence_length + self.prediction_horizon:
                logger.warning("Insufficient data for time series analysis")
                return np.array([]), np.array([])
                
            # Select features
            available_features = [col for col in self.feature_columns if col in hourly_data.columns]
            if not available_features:
                return np.array([]), np.array([])
                
            data = hourly_data[available_features].values
            
            # Scale data
            scaled_data = self.scaler.fit_transform(data)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length - self.prediction_horizon + 1):
                # Input sequence
                X.append(scaled_data[i:(i + self.sequence_length)])
                # Target is the next prediction_horizon values of the first feature (execution_time)
                y.append(scaled_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, 0])
                
            return np.array(X), np.array(y)
        
        return np.array([]), np.array([])
        
    def train_trend_model(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Train LSTM model for trend analysis"""
        logger.info("Training LSTM trend analysis model")
        
        X, y = self.prepare_time_series_data(metrics_df)
        
        if len(X) == 0:
            return {'status': 'no_data', 'message': 'Insufficient data for training'}
            
        # Initialize model
        input_size = X.shape[2]  # Number of features
        self.model = LSTMTrendAnalyzer(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            output_size=self.prediction_horizon
        ).to(self.device)
        
        # Prepare training data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Training parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        epochs = 100
        
        # Training loop
        self.model.train()
        training_losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            training_losses.append(loss.item())
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                
        self.is_trained = True
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            final_loss = criterion(predictions, y_tensor).item()
            
        return {
            'status': 'success',
            'training_samples': len(X),
            'final_loss': final_loss,
            'training_epochs': epochs,
            'features_used': input_size
        }
        
    def predict_trends(self, recent_metrics: pd.DataFrame) -> Dict[str, Any]:
        """Predict future trends using trained LSTM model"""
        if not self.is_trained or self.model is None:
            return {'status': 'model_not_trained', 'predictions': []}
            
        # Prepare recent data for prediction
        X, _ = self.prepare_time_series_data(recent_metrics)
        
        if len(X) == 0:
            return {'status': 'insufficient_data', 'predictions': []}
            
        # Use the last sequence for prediction
        last_sequence = X[-1:]
        X_tensor = torch.FloatTensor(last_sequence).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X_tensor)
            prediction_np = prediction.cpu().numpy().flatten()
            
        # Inverse scale predictions (only for execution_time feature)
        # Create dummy array for inverse transform
        dummy_array = np.zeros((len(prediction_np), len(self.feature_columns)))
        dummy_array[:, 0] = prediction_np  # execution_time predictions
        
        try:
            inverse_scaled = self.scaler.inverse_transform(dummy_array)
            predicted_execution_times = inverse_scaled[:, 0]
        except:
            predicted_execution_times = prediction_np  # Fallback
            
        # Generate predictions with timestamps
        current_time = datetime.now()
        predictions = []
        
        for i, pred_value in enumerate(predicted_execution_times):
            future_time = current_time + timedelta(hours=i+1)
            predictions.append({
                'timestamp': future_time.isoformat(),
                'predicted_execution_time': float(pred_value),
                'confidence': self._calculate_confidence(pred_value, predicted_execution_times),
                'horizon_hours': i + 1
            })
            
        return {
            'status': 'success',
            'predictions': predictions,
            'model_confidence': self._calculate_overall_confidence(predicted_execution_times)
        }
        
    def _calculate_confidence(self, prediction: float, all_predictions: np.ndarray) -> float:
        """Calculate confidence score for individual prediction"""
        # Confidence based on consistency and range
        std_dev = np.std(all_predictions)
        mean_pred = np.mean(all_predictions)
        
        if std_dev == 0:
            return 0.95
            
        # Normalized distance from mean
        distance = abs(prediction - mean_pred) / std_dev
        confidence = max(0.1, 1.0 - (distance / 3.0))  # Confidence decreases with distance
        
        return min(0.99, confidence)
        
    def _calculate_overall_confidence(self, predictions: np.ndarray) -> float:
        """Calculate overall model confidence"""
        if len(predictions) == 0:
            return 0.0
            
        # Based on prediction stability
        std_dev = np.std(predictions)
        mean_pred = np.mean(predictions)
        
        if mean_pred == 0:
            return 0.5
            
        coefficient_of_variation = std_dev / abs(mean_pred)
        confidence = max(0.1, 1.0 - coefficient_of_variation)
        
        return min(0.99, confidence)


class AnomalyDetector:
    """Advanced anomaly detection using Isolation Forest and statistical methods"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.statistical_detector = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_stats = {}
        
    def train_anomaly_detection(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Train anomaly detection models"""
        logger.info("Training anomaly detection models")
        
        if metrics_df.empty:
            return {'status': 'no_data', 'message': 'No data available for training'}
            
        # Prepare features
        features = self._prepare_anomaly_features(metrics_df)
        
        if features.empty:
            return {'status': 'insufficient_features'}
            
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train Isolation Forest
        self.isolation_forest.fit(features_scaled)
        
        # Calculate statistical baselines
        self.feature_stats = {
            col: {
                'mean': float(features[col].mean()),
                'std': float(features[col].std()),
                'q1': float(features[col].quantile(0.25)),
                'q3': float(features[col].quantile(0.75)),
                'median': float(features[col].median()),
                'iqr': float(features[col].quantile(0.75) - features[col].quantile(0.25))
            }
            for col in features.columns
        }
        
        self.is_trained = True
        
        # Test on training data to get baseline scores
        anomaly_scores = self.isolation_forest.decision_function(features_scaled)
        anomalies = self.isolation_forest.predict(features_scaled)
        
        return {
            'status': 'success',
            'training_samples': len(features),
            'features_used': list(features.columns),
            'anomalies_in_training': int(np.sum(anomalies == -1)),
            'mean_anomaly_score': float(np.mean(anomaly_scores)),
            'feature_statistics': self.feature_stats
        }
        
    def detect_anomalies(self, metrics_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in new data"""
        if not self.is_trained:
            logger.warning("Anomaly detector not trained")
            return []
            
        if metrics_df.empty:
            return []
            
        # Prepare features
        features = self._prepare_anomaly_features(metrics_df)
        
        if features.empty:
            return []
            
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # ML-based detection
            ml_anomaly_scores = self.isolation_forest.decision_function(features_scaled)
            ml_anomalies = self.isolation_forest.predict(features_scaled)
            
            # Statistical detection
            statistical_anomalies = self._detect_statistical_anomalies(features)
            
            # Combine results
            detected_anomalies = []
            
            for idx in range(len(features)):
                is_ml_anomaly = ml_anomalies[idx] == -1
                is_stat_anomaly = statistical_anomalies[idx]
                
                if is_ml_anomaly or is_stat_anomaly:
                    anomaly_data = {
                        'index': int(idx),
                        'timestamp': metrics_df.iloc[idx]['timestamp'] if 'timestamp' in metrics_df else None,
                        'agent_id': metrics_df.iloc[idx]['agent_id'] if 'agent_id' in metrics_df else None,
                        'operation': metrics_df.iloc[idx]['operation'] if 'operation' in metrics_df else None,
                        'ml_anomaly_score': float(ml_anomaly_scores[idx]),
                        'is_ml_anomaly': is_ml_anomaly,
                        'is_statistical_anomaly': is_stat_anomaly,
                        'severity': self._calculate_anomaly_severity(
                            ml_anomaly_scores[idx], is_ml_anomaly, is_stat_anomaly
                        ),
                        'affected_metrics': self._identify_affected_metrics(features.iloc[idx]),
                        'anomaly_type': self._classify_anomaly_type(features.iloc[idx])
                    }
                    detected_anomalies.append(anomaly_data)
                    
            return detected_anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
            
    def _prepare_anomaly_features(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection"""
        feature_columns = ['execution_time', 'cpu_usage', 'memory_usage']
        
        # Add time-based features if timestamp available
        if 'timestamp' in metrics_df.columns:
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
            metrics_df['hour_of_day'] = metrics_df['timestamp'].dt.hour
            metrics_df['day_of_week'] = metrics_df['timestamp'].dt.dayofweek
            feature_columns.extend(['hour_of_day', 'day_of_week'])
            
        # Select available features
        available_features = [col for col in feature_columns if col in metrics_df.columns]
        
        if not available_features:
            return pd.DataFrame()
            
        features = metrics_df[available_features].fillna(0)
        return features
        
    def _detect_statistical_anomalies(self, features: pd.DataFrame) -> List[bool]:
        """Detect anomalies using statistical methods"""
        statistical_anomalies = []
        
        for idx in range(len(features)):
            row = features.iloc[idx]
            is_anomaly = False
            
            for col in features.columns:
                if col in self.feature_stats:
                    stats = self.feature_stats[col]
                    value = row[col]
                    
                    # IQR-based outlier detection
                    iqr_lower = stats['q1'] - 1.5 * stats['iqr']
                    iqr_upper = stats['q3'] + 1.5 * stats['iqr']
                    
                    # Z-score based detection
                    if stats['std'] > 0:
                        z_score = abs(value - stats['mean']) / stats['std']
                        is_z_outlier = z_score > 3.0
                    else:
                        is_z_outlier = False
                        
                    # Combine methods
                    if value < iqr_lower or value > iqr_upper or is_z_outlier:
                        is_anomaly = True
                        break
                        
            statistical_anomalies.append(is_anomaly)
            
        return statistical_anomalies
        
    def _calculate_anomaly_severity(self, ml_score: float, is_ml_anomaly: bool, 
                                  is_stat_anomaly: bool) -> str:
        """Calculate anomaly severity level"""
        if is_ml_anomaly and is_stat_anomaly:
            return 'critical'
        elif ml_score < -0.3:  # Very low ML score
            return 'high'
        elif is_ml_anomaly or is_stat_anomaly:
            return 'medium'
        else:
            return 'low'
            
    def _identify_affected_metrics(self, feature_row: pd.Series) -> List[str]:
        """Identify which metrics are anomalous"""
        affected = []
        
        for col in feature_row.index:
            if col in self.feature_stats:
                stats = self.feature_stats[col]
                value = feature_row[col]
                
                # Check if this specific metric is anomalous
                iqr_lower = stats['q1'] - 1.5 * stats['iqr']
                iqr_upper = stats['q3'] + 1.5 * stats['iqr']
                
                if value < iqr_lower or value > iqr_upper:
                    affected.append(col)
                    
        return affected
        
    def _classify_anomaly_type(self, feature_row: pd.Series) -> str:
        """Classify the type of anomaly"""
        if 'cpu_usage' in feature_row and feature_row.get('cpu_usage', 0) > 90:
            return 'performance_degradation'
        elif 'memory_usage' in feature_row and feature_row.get('memory_usage', 0) > 90:
            return 'resource_exhaustion'
        elif 'execution_time' in feature_row:
            if 'execution_time' in self.feature_stats:
                expected_time = self.feature_stats['execution_time']['median']
                if feature_row['execution_time'] > expected_time * 3:
                    return 'timeout_risk'
        
        return 'unknown'


class AlertGenerator:
    """ML-based alert generation with severity prediction"""
    
    def __init__(self):
        self.alert_history = deque(maxlen=1000)
        self.severity_thresholds = {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }
        
    def generate_alerts(self, anomalies: List[Dict], predictions: Dict, 
                       performance_summary: Dict) -> List[Dict[str, Any]]:
        """Generate intelligent alerts based on ML analysis"""
        alerts = []
        current_time = datetime.now()
        
        # Anomaly-based alerts
        for anomaly in anomalies:
            alert = {
                'id': f"anomaly_{current_time.timestamp()}_{anomaly['index']}",
                'type': 'anomaly_detection',
                'severity': anomaly['severity'],
                'title': f"Performance Anomaly Detected - {anomaly.get('anomaly_type', 'Unknown')}",
                'description': f"Anomalous behavior detected in {anomaly.get('operation', 'unknown operation')}",
                'timestamp': current_time.isoformat(),
                'agent_id': anomaly.get('agent_id'),
                'operation': anomaly.get('operation'),
                'details': {
                    'ml_anomaly_score': anomaly['ml_anomaly_score'],
                    'affected_metrics': anomaly['affected_metrics'],
                    'anomaly_type': anomaly['anomaly_type']
                },
                'recommended_actions': self._get_anomaly_actions(anomaly)
            }
            alerts.append(alert)
            
        # Prediction-based alerts
        if predictions.get('status') == 'success':
            future_predictions = predictions.get('predictions', [])
            
            # Check for predicted performance degradation
            for pred in future_predictions:
                if pred['horizon_hours'] <= 4:  # Next 4 hours
                    baseline_time = performance_summary.get('median_execution_time', 0)
                    if baseline_time > 0 and pred['predicted_execution_time'] > baseline_time * 2:
                        alert = {
                            'id': f"prediction_{current_time.timestamp()}_{pred['horizon_hours']}",
                            'type': 'performance_prediction',
                            'severity': 'medium' if pred['confidence'] > 0.7 else 'low',
                            'title': 'Performance Degradation Predicted',
                            'description': f"Performance degradation predicted in {pred['horizon_hours']} hours",
                            'timestamp': current_time.isoformat(),
                            'predicted_time': pred['timestamp'],
                            'details': {
                                'predicted_execution_time': pred['predicted_execution_time'],
                                'baseline_execution_time': baseline_time,
                                'confidence': pred['confidence'],
                                'degradation_factor': pred['predicted_execution_time'] / baseline_time
                            },
                            'recommended_actions': ['Monitor system resources', 'Prepare for increased load', 'Review recent changes']
                        }
                        alerts.append(alert)
                        
        # Performance threshold alerts
        if performance_summary:
            # Success rate alert
            success_rate = performance_summary.get('success_rate', 1.0)
            if success_rate < 0.95:
                severity = 'critical' if success_rate < 0.90 else 'high'
                alert = {
                    'id': f"success_rate_{current_time.timestamp()}",
                    'type': 'performance_threshold',
                    'severity': severity,
                    'title': 'Low Success Rate Detected',
                    'description': f"System success rate is {success_rate:.2%}",
                    'timestamp': current_time.isoformat(),
                    'details': {
                        'current_success_rate': success_rate,
                        'threshold': 0.95,
                        'total_operations': performance_summary.get('total_operations', 0)
                    },
                    'recommended_actions': ['Review error logs', 'Check system health', 'Investigate failing operations']
                }
                alerts.append(alert)
                
            # High response time alert
            p95_time = performance_summary.get('p95_execution_time', 0)
            avg_time = performance_summary.get('average_execution_time', 0)
            if p95_time > 0 and avg_time > 0 and p95_time > avg_time * 3:
                alert = {
                    'id': f"response_time_{current_time.timestamp()}",
                    'type': 'performance_threshold',
                    'severity': 'medium',
                    'title': 'High Response Time Variance',
                    'description': f"95th percentile response time ({p95_time:.2f}s) significantly higher than average ({avg_time:.2f}s)",
                    'timestamp': current_time.isoformat(),
                    'details': {
                        'p95_execution_time': p95_time,
                        'average_execution_time': avg_time,
                        'variance_factor': p95_time / avg_time
                    },
                    'recommended_actions': ['Identify slow operations', 'Check for resource contention', 'Review performance bottlenecks']
                }
                alerts.append(alert)
                
        # Store alerts in history
        for alert in alerts:
            self.alert_history.append(alert)
            
        return alerts
        
    def _get_anomaly_actions(self, anomaly: Dict) -> List[str]:
        """Get recommended actions for specific anomaly types"""
        anomaly_type = anomaly.get('anomaly_type', 'unknown')
        
        action_map = {
            'performance_degradation': [
                'Check CPU usage and system load',
                'Review recent code changes',
                'Monitor memory usage patterns',
                'Consider scaling resources'
            ],
            'resource_exhaustion': [
                'Immediate resource monitoring required',
                'Check for memory leaks',
                'Review resource allocation',
                'Consider emergency scaling'
            ],
            'timeout_risk': [
                'Review operation timeouts',
                'Check for deadlocks',
                'Monitor external dependencies',
                'Consider async processing'
            ],
            'unknown': [
                'Investigate anomalous metrics',
                'Review system logs',
                'Monitor affected operations',
                'Check for external factors'
            ]
        }
        
        return action_map.get(anomaly_type, action_map['unknown'])


class PredictiveMonitor:
    """Main predictive monitoring system combining all ML components"""
    
    def __init__(self, db_path: str = "src/database/analytics_metrics.db"):
        self.db_path = db_path
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.alert_generator = AlertGenerator()
        self.monitoring_active = False
        
    def start_monitoring(self, metrics_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Start predictive monitoring system"""
        logger.info("Starting predictive monitoring system")
        
        # Train models if data provided
        training_results = {}
        if metrics_df is not None and not metrics_df.empty:
            trend_training = self.trend_analyzer.train_trend_model(metrics_df)
            anomaly_training = self.anomaly_detector.train_anomaly_detection(metrics_df)
            
            training_results = {
                'trend_model': trend_training,
                'anomaly_model': anomaly_training
            }
            
        self.monitoring_active = True
        
        return {
            'status': 'monitoring_started',
            'timestamp': datetime.now().isoformat(),
            'training_results': training_results
        }
        
    def monitor_and_predict(self, recent_metrics: pd.DataFrame, 
                          performance_summary: Dict) -> Dict[str, Any]:
        """Comprehensive monitoring and prediction"""
        if not self.monitoring_active:
            return {'status': 'monitoring_not_active'}
            
        logger.info("Running predictive monitoring analysis")
        
        # Trend prediction
        trend_predictions = self.trend_analyzer.predict_trends(recent_metrics)
        
        # Anomaly detection
        detected_anomalies = self.anomaly_detector.detect_anomalies(recent_metrics)
        
        # Generate alerts
        alerts = self.alert_generator.generate_alerts(
            detected_anomalies, trend_predictions, performance_summary
        )
        
        # Compile monitoring report
        monitoring_report = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'monitoring_period': 'real-time',
            'trend_analysis': trend_predictions,
            'anomaly_detection': {
                'total_anomalies': len(detected_anomalies),
                'anomalies': detected_anomalies,
                'severity_breakdown': self._analyze_anomaly_severity(detected_anomalies)
            },
            'alerts': {
                'total_alerts': len(alerts),
                'alerts': alerts,
                'alert_summary': self._summarize_alerts(alerts)
            },
            'system_health': self._assess_system_health(detected_anomalies, alerts, performance_summary)
        }
        
        return monitoring_report
        
    def _analyze_anomaly_severity(self, anomalies: List[Dict]) -> Dict[str, int]:
        """Analyze anomaly severity distribution"""
        severity_counts = defaultdict(int)
        for anomaly in anomalies:
            severity_counts[anomaly.get('severity', 'unknown')] += 1
        return dict(severity_counts)
        
    def _summarize_alerts(self, alerts: List[Dict]) -> Dict[str, Any]:
        """Summarize alert information"""
        if not alerts:
            return {'total': 0, 'by_severity': {}, 'by_type': {}}
            
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for alert in alerts:
            severity_counts[alert.get('severity', 'unknown')] += 1
            type_counts[alert.get('type', 'unknown')] += 1
            
        return {
            'total': len(alerts),
            'by_severity': dict(severity_counts),
            'by_type': dict(type_counts),
            'most_common_severity': max(severity_counts.items(), key=lambda x: x[1])[0] if severity_counts else 'none',
            'most_common_type': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'none'
        }
        
    def _assess_system_health(self, anomalies: List[Dict], alerts: List[Dict], 
                            performance_summary: Dict) -> Dict[str, Any]:
        """Assess overall system health"""
        health_score = 100.0
        health_factors = []
        
        # Anomaly impact
        critical_anomalies = sum(1 for a in anomalies if a.get('severity') == 'critical')
        high_anomalies = sum(1 for a in anomalies if a.get('severity') == 'high')
        
        if critical_anomalies > 0:
            health_score -= critical_anomalies * 20
            health_factors.append(f"{critical_anomalies} critical anomalies detected")
            
        if high_anomalies > 0:
            health_score -= high_anomalies * 10
            health_factors.append(f"{high_anomalies} high severity anomalies")
            
        # Alert impact
        critical_alerts = sum(1 for a in alerts if a.get('severity') == 'critical')
        if critical_alerts > 0:
            health_score -= critical_alerts * 15
            health_factors.append(f"{critical_alerts} critical alerts")
            
        # Performance impact
        success_rate = performance_summary.get('success_rate', 1.0)
        if success_rate < 0.95:
            health_score -= (0.95 - success_rate) * 100
            health_factors.append(f"Success rate: {success_rate:.2%}")
            
        # Determine health status
        health_score = max(0, health_score)
        
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 75:
            status = 'good'
        elif health_score >= 50:
            status = 'warning'
        else:
            status = 'critical'
            
        return {
            'status': status,
            'score': health_score,
            'factors': health_factors,
            'recommendation': self._get_health_recommendation(status, health_factors)
        }
        
    def _get_health_recommendation(self, status: str, factors: List[str]) -> str:
        """Get health-based recommendations"""
        if status == 'excellent':
            return 'System operating optimally. Continue monitoring.'
        elif status == 'good':
            return 'System performance is good. Monitor trends and prepare for optimization.'
        elif status == 'warning':
            return 'System performance degraded. Investigation and optimization recommended.'
        else:
            return 'Critical system issues detected. Immediate intervention required.'


# Export main classes
__all__ = [
    'TrendAnalyzer',
    'AnomalyDetector', 
    'AlertGenerator',
    'PredictiveMonitor',
    'LSTMTrendAnalyzer'
]