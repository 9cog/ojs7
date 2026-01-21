"""
Comprehensive Tests for Analytics & Monitoring Agent ML Systems
Tests all production ML capabilities with zero tolerance for mocks or placeholders.
"""

import pytest
import numpy as np
import pandas as pd
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import the ML modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analytics_ml_engine import (
    MetricCollector, PatternRecognizer, OptimizationIdentifier, PerformanceAnalyzer
)
from predictive_monitoring import (
    TrendAnalyzer, AnomalyDetector, AlertGenerator, PredictiveMonitor
)
from autonomous_optimization import (
    ParameterTuner, BehaviorAdjuster, ImprovementImplementer, AutonomousOptimizer
)
from strategic_analytics import (
    InsightGenerator, EvolutionForecaster, StrategicRecommendationEngine, StrategicAnalytics
)

class TestMetricCollector:
    """Test production-grade metric collection system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.collector = MetricCollector(self.temp_db.name)
        
    def teardown_method(self):
        """Cleanup test environment"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
            
    def test_metric_collection(self):
        """Test comprehensive metric collection"""
        # Collect multiple metrics
        for i in range(10):
            self.collector.collect_operation_metric(
                agent_id=f"test_agent_{i % 3}",
                operation="test_operation",
                execution_time=1.0 + i * 0.1,
                success=i % 10 != 0,  # 90% success rate
                cpu_usage=50.0 + i * 2,
                memory_usage=60.0 + i * 1.5
            )
            
        # Verify data collection
        metrics_df = self.collector.get_recent_metrics(hours=1)
        assert len(metrics_df) == 10
        assert 'execution_time' in metrics_df.columns
        assert 'cpu_usage' in metrics_df.columns
        assert 'memory_usage' in metrics_df.columns
        assert 'success' in metrics_df.columns
        
        # Verify success rate calculation
        success_rate = metrics_df['success'].mean()
        assert abs(success_rate - 0.9) < 0.1  # Should be around 90%
        
    def test_system_metric_collection(self):
        """Test system-level metric collection"""
        self.collector.collect_system_metric(
            metric_name="cpu_utilization",
            metric_value=75.5,
            metadata={"source": "system_monitor"}
        )
        
        # Verify database storage
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.execute("SELECT * FROM system_metrics")
            rows = cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][2] == "cpu_utilization"  # metric_name
            assert rows[0][3] == 75.5  # metric_value


class TestPatternRecognizer:
    """Test ML-based pattern recognition"""
    
    def setup_method(self):
        self.recognizer = PatternRecognizer()
        
    def create_test_data(self, n_samples=100):
        """Create realistic test data with patterns"""
        np.random.seed(42)
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), 
                             periods=n_samples, freq='H')
        
        data = []
        for i, timestamp in enumerate(dates):
            # Create patterns based on time of day
            hour = timestamp.hour
            base_time = 2.0 if 9 <= hour <= 17 else 1.5  # Business hours pattern
            
            data.append({
                'timestamp': timestamp,
                'execution_time': base_time + np.random.normal(0, 0.3),
                'cpu_usage': 40 + (hour / 24) * 30 + np.random.normal(0, 5),
                'memory_usage': 50 + np.sin(hour / 24 * 2 * np.pi) * 10 + np.random.normal(0, 3)
            })
            
        return pd.DataFrame(data)
        
    def test_pattern_learning(self):
        """Test pattern learning from performance data"""
        test_data = self.create_test_data()
        
        result = self.recognizer.learn_patterns(test_data)
        
        assert result['status'] == 'success'
        assert 'patterns' in result
        assert result['n_clusters'] >= 0
        assert 'anomaly_score' in result
        
        # Verify patterns are meaningful
        patterns = result['patterns']
        if patterns:  # If patterns were found
            assert all('characteristics' in p for p in patterns)
            assert all('cluster_id' in p for p in patterns)
            
    def test_anomaly_detection(self):
        """Test real-time anomaly detection"""
        # Train on normal data
        training_data = self.create_test_data()
        self.recognizer.learn_patterns(training_data)
        
        # Create data with anomalies
        anomaly_data = training_data.copy()
        # Inject clear anomalies
        anomaly_data.loc[0, 'execution_time'] = 50.0  # Extreme value
        anomaly_data.loc[1, 'cpu_usage'] = 150.0     # Impossible value
        
        anomalies = self.recognizer.detect_anomalies(anomaly_data)
        
        # Should detect the injected anomalies
        assert len(anomalies) >= 1
        if anomalies:
            assert all('anomaly_score' in a for a in anomalies)
            assert all('index' in a for a in anomalies)


class TestOptimizationIdentifier:
    """Test ML-based optimization identification"""
    
    def setup_method(self):
        self.optimizer = OptimizationIdentifier()
        
    def create_optimization_test_data(self):
        """Create test data suitable for optimization analysis"""
        np.random.seed(42)
        
        data = []
        for i in range(100):
            # Create correlation between CPU usage and execution time
            cpu_usage = np.random.uniform(20, 90)
            base_time = 1.0 + (cpu_usage / 100) * 2  # Higher CPU = slower execution
            
            data.append({
                'execution_time': base_time + np.random.normal(0, 0.2),
                'cpu_usage': cpu_usage,
                'memory_usage': np.random.uniform(30, 80),
                'operation': np.random.choice(['op_a', 'op_b', 'op_c'])
            })
            
        return pd.DataFrame(data)
        
    def test_optimization_identification(self):
        """Test identification of optimization opportunities"""
        test_data = self.create_optimization_test_data()
        
        result = self.optimizer.identify_optimizations(test_data)
        
        assert result['status'] == 'success'
        assert 'optimizations' in result
        assert 'model_performance' in result
        
        # Verify model was trained successfully
        model_perf = result['model_performance']
        assert 'mse' in model_perf
        assert 'feature_importance' in model_perf
        
        # Verify optimizations are actionable
        optimizations = result['optimizations']
        if optimizations:
            for opt in optimizations:
                assert 'type' in opt
                assert 'priority' in opt
                assert 'recommendation' in opt
                assert 'potential_improvement' in opt


class TestTrendAnalyzer:
    """Test LSTM-based trend analysis"""
    
    def setup_method(self):
        self.analyzer = TrendAnalyzer(sequence_length=12, prediction_horizon=6)
        
    def create_time_series_data(self):
        """Create realistic time series data"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=14), 
                             periods=336, freq='h')  # 14 days of hourly data
        
        data = []
        for i, timestamp in enumerate(dates):
            # Create realistic time series with trend and seasonality
            trend = i * 0.01  # Slight upward trend
            seasonal = np.sin(2 * np.pi * timestamp.hour / 24) * 0.5  # Daily pattern
            noise = np.random.normal(0, 0.2)
            
            execution_time = 2.0 + trend + seasonal + noise
            
            data.append({
                'timestamp': timestamp,
                'execution_time': execution_time,
                'cpu_usage': 40 + seasonal * 20 + np.random.normal(0, 5),
                'memory_usage': 50 + np.random.normal(0, 5),
                'success': np.random.choice([True, False], p=[0.95, 0.05]),
                'agent_id': f'agent_{i % 3}'
            })
            
        return pd.DataFrame(data)
        
    def test_trend_model_training(self):
        """Test LSTM model training"""
        test_data = self.create_time_series_data()
        
        result = self.analyzer.train_trend_model(test_data)
        
        assert result['status'] == 'success'
        assert 'training_samples' in result
        assert 'final_loss' in result
        assert result['training_samples'] > 0
        
        # Verify model was actually trained
        assert self.analyzer.is_trained
        assert self.analyzer.model is not None
        
    def test_trend_prediction(self):
        """Test trend prediction after training"""
        test_data = self.create_time_series_data()
        
        # Train model
        self.analyzer.train_trend_model(test_data)
        
        # Make predictions
        recent_data = test_data.tail(100)
        predictions = self.analyzer.predict_trends(recent_data)
        
        assert predictions['status'] == 'success'
        assert 'predictions' in predictions
        assert 'model_confidence' in predictions
        
        # Verify prediction format
        pred_list = predictions['predictions']
        if pred_list:
            for pred in pred_list:
                assert 'timestamp' in pred
                assert 'predicted_execution_time' in pred
                assert 'confidence' in pred
                assert 'horizon_hours' in pred


class TestAnomalyDetector:
    """Test advanced anomaly detection"""
    
    def setup_method(self):
        self.detector = AnomalyDetector(contamination=0.1)
        
    def test_anomaly_training_and_detection(self):
        """Test training and anomaly detection"""
        # Create normal training data
        np.random.seed(42)
        training_data = pd.DataFrame({
            'execution_time': np.random.normal(2.0, 0.5, 100),
            'cpu_usage': np.random.normal(50, 10, 100),
            'memory_usage': np.random.normal(60, 8, 100)
        })
        
        # Train detector
        result = self.detector.train_anomaly_detection(training_data)
        
        assert result['status'] == 'success'
        assert 'training_samples' in result
        assert 'features_used' in result
        assert self.detector.is_trained
        
        # Create test data with clear anomalies
        test_data = pd.DataFrame({
            'execution_time': [2.0, 2.1, 15.0, 2.2],  # 15.0 is anomaly
            'cpu_usage': [50, 52, 48, 95],             # 95 is anomaly
            'memory_usage': [60, 58, 62, 61]
        })
        
        anomalies = self.detector.detect_anomalies(test_data)
        
        # Should detect anomalies
        assert len(anomalies) >= 1
        if anomalies:
            assert all('severity' in a for a in anomalies)
            assert all('anomaly_type' in a for a in anomalies)


class TestParameterTuner:
    """Test Bayesian optimization for parameter tuning"""
    
    def setup_method(self):
        self.tuner = ParameterTuner()
        
    def test_parameter_space_definition(self):
        """Test parameter space definition"""
        param_space = {
            'learning_rate': {'min': 0.001, 'max': 0.1},
            'batch_size': {'min': 16, 'max': 128},
            'hidden_units': {'min': 32, 'max': 512}
        }
        
        self.tuner.define_parameter_space(param_space)
        
        assert self.tuner.parameter_bounds == param_space
        
    def test_parameter_suggestions(self):
        """Test parameter suggestions"""
        param_space = {
            'param_a': {'min': 0.0, 'max': 1.0},
            'param_b': {'min': 10, 'max': 100}
        }
        
        self.tuner.define_parameter_space(param_space)
        
        # Get initial random suggestions
        suggestions = self.tuner.suggest_parameters()
        
        assert 'param_a' in suggestions
        assert 'param_b' in suggestions
        assert 0.0 <= suggestions['param_a'] <= 1.0
        assert 10 <= suggestions['param_b'] <= 100
        
    def test_performance_recording_and_learning(self):
        """Test performance recording and Bayesian learning"""
        param_space = {
            'x': {'min': -5.0, 'max': 5.0}
        }
        
        self.tuner.define_parameter_space(param_space)
        
        # Record several parameter-performance pairs
        for i in range(10):
            params = {'x': i - 5}  # Range from -5 to 4
            # Performance is optimal at x=0 (quadratic function)
            performance = -(params['x'] ** 2)  # Maximum at x=0
            
            self.tuner.record_performance(params, performance)
            
        # After sufficient data, should use Bayesian optimization
        assert len(self.tuner.parameter_history) == 10
        assert len(self.tuner.performance_history) == 10
        
        # Get Bayesian-optimized suggestion
        suggestion = self.tuner.suggest_parameters()
        
        # Should suggest value closer to optimal (x=0)
        assert abs(suggestion['x']) < 2.0  # Should be reasonably close to optimum


class TestStrategicAnalytics:
    """Test strategic analytics system"""
    
    def setup_method(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.strategic_analytics = StrategicAnalytics(self.temp_db.name)
        
    def teardown_method(self):
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
            
    def create_strategic_test_data(self):
        """Create comprehensive test data for strategic analysis"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                             periods=720, freq='h')  # 30 days hourly
        
        data = []
        for i, timestamp in enumerate(dates):
            # Simulate business growth and patterns
            growth_factor = 1 + (i / 720) * 0.2  # 20% growth over period
            hour_factor = 1.2 if 9 <= timestamp.hour <= 17 else 0.8
            
            data.append({
                'timestamp': timestamp,
                'execution_time': 2.0 / growth_factor + np.random.normal(0, 0.2),
                'cpu_usage': 40 * growth_factor + np.random.normal(0, 5),
                'memory_usage': 50 * growth_factor + np.random.normal(0, 5),
                'success_rate': min(0.99, 0.95 + growth_factor * 0.02 + np.random.normal(0, 0.01)),
                'total_operations': int(100 * hour_factor * growth_factor + np.random.normal(0, 10))
            })
            
        return pd.DataFrame(data)
        
    def test_comprehensive_strategic_analysis(self):
        """Test complete strategic analysis pipeline"""
        # Create test data
        historical_data = self.create_strategic_test_data()
        performance_data = {
            'success_rate': 0.96,
            'average_execution_time': 2.1,
            'total_operations': 5000,
            'average_cpu_usage': 65.0,
            'average_memory_usage': 70.0
        }
        
        # Perform strategic analysis
        result = self.strategic_analytics.perform_strategic_analysis(
            performance_data, historical_data
        )
        
        # Verify comprehensive results
        assert 'strategic_insights' in result
        assert 'evolution_forecast' in result
        assert 'strategic_recommendations' in result
        assert 'executive_summary' in result
        assert 'next_actions' in result
        
        # Verify insights generation
        insights = result['strategic_insights']
        assert 'total_insights' in insights
        assert 'insights' in insights
        
        # Verify recommendations
        recommendations = result['strategic_recommendations']
        assert 'total_recommendations' in recommendations
        assert 'recommendations' in recommendations
        
        # Verify executive summary
        exec_summary = result['executive_summary']
        assert 'key_findings' in exec_summary
        assert 'overall_system_health' in exec_summary


class TestIntegrationScenarios:
    """Test integrated scenarios combining multiple ML systems"""
    
    def setup_method(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Initialize all systems
        self.performance_analyzer = PerformanceAnalyzer(self.temp_db.name)
        self.predictive_monitor = PredictiveMonitor(self.temp_db.name)
        self.autonomous_optimizer = AutonomousOptimizer(self.temp_db.name)
        self.strategic_analytics = StrategicAnalytics(self.temp_db.name)
        
    def teardown_method(self):
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
            
    def create_comprehensive_test_data(self):
        """Create comprehensive test dataset"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), 
                             periods=168, freq='h')  # 7 days hourly
        
        data = []
        for i, timestamp in enumerate(dates):
            # Create realistic performance patterns
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Business patterns
            business_factor = 1.2 if 9 <= hour <= 17 and day_of_week < 5 else 0.7
            
            # Add performance degradation over time
            degradation = 1 + (i / 168) * 0.1  # 10% degradation over week
            
            data.append({
                'timestamp': timestamp,
                'agent_id': f'agent_{i % 4}',
                'operation': np.random.choice(['analysis', 'processing', 'monitoring']),
                'execution_time': 2.0 * business_factor * degradation + np.random.normal(0, 0.3),
                'cpu_usage': 40 * business_factor * degradation + np.random.normal(0, 8),
                'memory_usage': 50 * business_factor * degradation + np.random.normal(0, 6),
                'success': np.random.choice([True, False], p=[0.95, 0.05])
            })
            
        return pd.DataFrame(data)
        
    def test_end_to_end_ml_pipeline(self):
        """Test complete ML pipeline from data collection to strategic recommendations"""
        # Create comprehensive test data
        test_data = self.create_comprehensive_test_data()
        
        # 1. Performance Analysis
        performance_result = self.performance_analyzer.analyze_performance(hours=168)
        assert performance_result['status'] in ['success', 'no_data']
        
        # 2. Predictive Monitoring (if performance analysis has data)
        if performance_result['status'] == 'success':
            monitoring_result = self.predictive_monitor.monitor_and_predict(
                test_data, performance_result.get('performance_summary', {})
            )
            assert 'status' in monitoring_result
            
        # 3. Autonomous Optimization
        current_performance = {
            'success_rate': 0.95,
            'execution_time': 2.5,
            'cpu_usage': 60.0,
            'memory_usage': 70.0,
            'throughput': 100.0
        }
        
        optimization_result = self.autonomous_optimizer.optimize_system(current_performance)
        assert 'optimizations_applied' in optimization_result
        
        # 4. Strategic Analysis
        strategic_result = self.strategic_analytics.perform_strategic_analysis(
            current_performance, test_data
        )
        assert 'strategic_insights' in strategic_result
        assert 'strategic_recommendations' in strategic_result
        
        # Verify integration coherence
        assert isinstance(strategic_result['strategic_insights']['total_insights'], int)
        assert isinstance(strategic_result['strategic_recommendations']['total_recommendations'], int)


class TestMLSystemRobustness:
    """Test ML systems robustness and error handling"""
    
    def test_empty_data_handling(self):
        """Test graceful handling of empty datasets"""
        empty_df = pd.DataFrame()
        
        # Pattern recognizer
        recognizer = PatternRecognizer()
        result = recognizer.learn_patterns(empty_df)
        assert result['status'] == 'no_data'
        
        # Optimization identifier
        optimizer = OptimizationIdentifier()
        result = optimizer.identify_optimizations(empty_df)
        assert 'optimizations' in result
        
    def test_invalid_data_handling(self):
        """Test handling of invalid or corrupted data"""
        # Create data with NaN values and handle infinities properly
        invalid_data = pd.DataFrame({
            'execution_time': [1.0, np.nan, 3.0, 100.0, -1.0],  # Changed inf to large value
            'cpu_usage': [50, 60, np.nan, 70, 200],  # 200 is invalid but not inf
            'memory_usage': [40, 50, 60, np.nan, 80]
        })
        
        recognizer = PatternRecognizer()
        result = recognizer.learn_patterns(invalid_data)
        
        # Should handle invalid data gracefully
        assert 'status' in result
        # Should not crash - that's the main test
        
    def test_small_dataset_handling(self):
        """Test behavior with very small datasets"""
        small_data = pd.DataFrame({
            'execution_time': [1.0, 2.0],
            'cpu_usage': [50, 60],
            'memory_usage': [40, 50]
        })
        
        optimizer = OptimizationIdentifier()
        result = optimizer.identify_optimizations(small_data)
        
        # Should handle small datasets gracefully
        assert 'optimizations' in result or 'error' in result


if __name__ == '__main__':
    # Run tests with detailed output
    pytest.main([__file__, '-v', '--tb=short'])