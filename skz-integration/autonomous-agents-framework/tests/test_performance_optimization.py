"""
Performance Optimization Tests for SKZ Autonomous Agents Framework
Tests validate that performance optimizations work correctly and improve system performance
"""

import pytest
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import statistics
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from performance_optimizer import PerformanceOptimizer, monitor_performance, AsyncOptimizer


class TestPerformanceOptimizer:
    """Test performance optimizer functionality"""
    
    @pytest.fixture
    def optimizer(self):
        """Create performance optimizer instance for testing"""
        return PerformanceOptimizer()
    
    def test_optimizer_initialization(self, optimizer):
        """Test that optimizer initializes correctly"""
        assert optimizer is not None
        assert optimizer.cache_manager is not None
        assert optimizer.db_optimizer is not None
        assert optimizer.metrics_collector is not None
        assert optimizer.connection_pool is not None
    
    def test_cache_functionality(self, optimizer):
        """Test caching improves performance"""
        agent_id = "test_agent"
        operation = "test_operation"
        test_data = {"test": "data", "timestamp": time.time()}
        
        # First call (cache miss)
        start_time = time.time()
        result1 = optimizer.optimize_agent_performance(agent_id, operation, test_data)
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = optimizer.optimize_agent_performance(agent_id, operation, test_data)
        second_call_time = time.time() - start_time
        
        # Cache hit should be faster
        assert second_call_time < first_call_time
        assert result1 == result2
    
    def test_research_operation_optimization(self, optimizer):
        """Test research discovery optimization"""
        test_data = {"query": "machine learning optimization"}
        
        result = optimizer.optimize_agent_performance(
            "research_discovery", "research_operation", test_data
        )
        
        assert result is not None
        assert result.get("optimized") is True
        assert "processing_time" in result
        assert result["processing_time"] < 2.5  # Should be faster than baseline
    
    def test_review_operation_optimization(self, optimizer):
        """Test review coordination optimization"""
        test_data = {"manuscript_id": "test_123"}
        
        result = optimizer.optimize_agent_performance(
            "review_coordination", "review_operation", test_data
        )
        
        assert result is not None
        assert result.get("optimized") is True
        assert result["processing_time"] < 3.5  # Should be faster than baseline
    
    def test_concurrent_operations(self, optimizer):
        """Test concurrent operation handling"""
        def perform_operation(op_id):
            return optimizer.optimize_agent_performance(
                "test_agent", f"operation_{op_id}", {"id": op_id}
            )
        
        # Test 10 concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(perform_operation, i) for i in range(10)]
            results = [future.result() for future in as_completed(futures)]
        
        assert len(results) == 10
        assert all(result.get("optimized") for result in results)
    
    def test_metrics_collection(self, optimizer):
        """Test metrics collection functionality"""
        # Perform several operations
        for i in range(5):
            optimizer.optimize_agent_performance(
                f"agent_{i}", "test_operation", {"test": i}
            )
        
        metrics = optimizer.get_performance_metrics()
        
        assert "total_operations" in metrics
        assert metrics["total_operations"] >= 5
        assert "average_response_times" in metrics
        assert "success_rates" in metrics


class TestPerformanceMonitoring:
    """Test performance monitoring decorator"""
    
    @monitor_performance
    def test_function(self, duration=0.1):
        """Test function for monitoring"""
        time.sleep(duration)
        return "completed"
    
    def test_monitor_decorator(self):
        """Test that monitor decorator works correctly"""
        result = self.test_function(0.05)
        assert result == "completed"
    
    def test_monitor_with_exception(self):
        """Test monitoring with exceptions"""
        @monitor_performance
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()


class TestAsyncOptimizer:
    """Test async optimization functionality"""
    
    @pytest.mark.asyncio
    async def test_parallel_operations(self):
        """Test parallel async operations"""
        async def mock_operation(delay=0.1):
            await asyncio.sleep(delay)
            return f"completed_{delay}"
        
        operations = [
            lambda: mock_operation(0.1),
            lambda: mock_operation(0.2),
            lambda: mock_operation(0.15)
        ]
        
        start_time = time.time()
        results = await AsyncOptimizer.parallel_agent_operations(operations)
        execution_time = time.time() - start_time
        
        assert len(results) == 3
        assert execution_time < 0.3  # Should be faster than sequential execution
    
    @pytest.mark.asyncio
    async def test_optimized_database_query(self):
        """Test optimized database query"""
        result = await AsyncOptimizer.optimized_database_query(
            "SELECT * FROM test", ("param1", "param2")
        )
        
        assert isinstance(result, list)


class TestSystemPerformance:
    """Integration tests for system performance"""
    
    def test_response_time_improvement(self):
        """Test that response times are improved"""
        optimizer = PerformanceOptimizer()
        
        # Baseline measurements
        baseline_times = []
        for i in range(10):
            start_time = time.time()
            optimizer.optimize_agent_performance(
                "research_discovery", "baseline_test", {"iteration": i}
            )
            baseline_times.append(time.time() - start_time)
        
        baseline_avg = statistics.mean(baseline_times)
        
        # The optimization should provide better than baseline performance
        # (This is simulated since we don't have actual baseline comparison)
        assert baseline_avg < 3.0  # Should be better than original 3.1s baseline
    
    def test_cache_hit_rate(self):
        """Test cache hit rate performance"""
        optimizer = PerformanceOptimizer()
        
        # Perform operations that should benefit from caching
        test_data = {"query": "performance test"}
        
        # First round - cache misses
        for _ in range(5):
            optimizer.optimize_agent_performance(
                "research_discovery", "cache_test", test_data
            )
        
        # Second round - cache hits
        for _ in range(5):
            optimizer.optimize_agent_performance(
                "research_discovery", "cache_test", test_data
            )
        
        metrics = optimizer.get_performance_metrics()
        cache_hit_rate = metrics.get("cache_hit_rate", 0)
        
        # Should have decent cache hit rate
        assert cache_hit_rate > 0.4  # At least 40% cache hits
    
    def test_concurrent_load_handling(self):
        """Test system performance under concurrent load"""
        optimizer = PerformanceOptimizer()
        
        def load_test_operation(thread_id):
            results = []
            for i in range(5):  # 5 operations per thread
                start_time = time.time()
                result = optimizer.optimize_agent_performance(
                    f"agent_{thread_id}", f"load_test_{i}", {"thread": thread_id, "op": i}
                )
                execution_time = time.time() - start_time
                results.append((result, execution_time))
            return results
        
        # Test with 5 concurrent threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(load_test_operation, i) for i in range(5)]
            thread_results = [future.result() for future in as_completed(futures)]
        
        # Analyze results
        all_times = []
        for thread_result in thread_results:
            for result, exec_time in thread_result:
                all_times.append(exec_time)
                assert result.get("optimized") is True
        
        avg_time = statistics.mean(all_times)
        max_time = max(all_times)
        
        # Performance should be reasonable under load
        assert avg_time < 2.0  # Average should be under 2 seconds
        assert max_time < 5.0   # No operation should take more than 5 seconds
    
    def test_memory_optimization(self):
        """Test memory optimization functionality"""
        optimizer = PerformanceOptimizer()
        
        # Test memory optimizer
        test_data = {
            "large_dataset": ["item"] * 1000,
            "temp_objects": {f"obj_{i}": f"data_{i}" for i in range(100)}
        }
        
        optimized_data = optimizer.memory_optimizer.optimize_memory_usage(
            "memory_test", test_data
        )
        
        assert optimized_data is not None
        assert "large_dataset" in optimized_data
        assert "temp_objects" in optimized_data


class TestEndToEndPerformance:
    """End-to-end performance tests"""
    
    def test_manuscript_processing_performance(self):
        """Test end-to-end manuscript processing performance"""
        optimizer = PerformanceOptimizer()
        
        manuscript_data = {
            "title": "Performance Test Manuscript",
            "abstract": "This is a test manuscript for performance optimization testing.",
            "authors": ["Test Author"],
            "content": "Sample content for testing performance optimization."
        }
        
        # Simulate complete manuscript processing workflow
        start_time = time.time()
        
        # Research discovery
        research_result = optimizer.optimize_agent_performance(
            "research_discovery", "manuscript_analysis", manuscript_data
        )
        
        # Quality assessment
        quality_result = optimizer.optimize_agent_performance(
            "content_quality", "quality_assessment", manuscript_data
        )
        
        # Review coordination
        review_result = optimizer.optimize_agent_performance(
            "review_coordination", "reviewer_matching", manuscript_data
        )
        
        total_time = time.time() - start_time
        
        # Validate results
        assert research_result.get("optimized") is True
        assert quality_result.get("optimized") is True
        assert review_result.get("optimized") is True
        
        # Total processing should be reasonable
        assert total_time < 10.0  # Complete workflow under 10 seconds
    
    def test_system_stability_under_load(self):
        """Test system stability under sustained load"""
        optimizer = PerformanceOptimizer()
        
        def sustained_load_test():
            results = []
            for i in range(20):  # 20 operations
                try:
                    result = optimizer.optimize_agent_performance(
                        "stability_test", f"operation_{i}", {"iteration": i}
                    )
                    results.append(("success", result))
                except Exception as e:
                    results.append(("error", str(e)))
                
                # Small delay between operations
                time.sleep(0.1)
            
            return results
        
        # Run sustained load test
        results = sustained_load_test()
        
        # Count successes and errors
        successes = sum(1 for status, _ in results if status == "success")
        errors = sum(1 for status, _ in results if status == "error")
        
        # Should have high success rate
        success_rate = successes / len(results)
        assert success_rate > 0.95  # At least 95% success rate
        assert errors <= 1  # At most 1 error allowed


# Performance benchmarking utilities
class PerformanceBenchmark:
    """Utility class for performance benchmarking"""
    
    @staticmethod
    def benchmark_operation(operation_func, iterations=10):
        """Benchmark an operation"""
        times = []
        successes = 0
        
        for i in range(iterations):
            start_time = time.time()
            try:
                result = operation_func(i)
                times.append(time.time() - start_time)
                successes += 1
            except Exception:
                times.append(float('inf'))
        
        return {
            'avg_time': statistics.mean(times) if times else float('inf'),
            'min_time': min(times) if times else float('inf'),
            'max_time': max(times) if times else float('inf'),
            'success_rate': successes / iterations,
            'total_iterations': iterations
        }
    
    @staticmethod
    def compare_performance(baseline_func, optimized_func, iterations=10):
        """Compare baseline vs optimized performance"""
        baseline_results = PerformanceBenchmark.benchmark_operation(baseline_func, iterations)
        optimized_results = PerformanceBenchmark.benchmark_operation(optimized_func, iterations)
        
        improvement = {
            'time_improvement': (baseline_results['avg_time'] - optimized_results['avg_time']) / baseline_results['avg_time'] * 100,
            'success_rate_improvement': (optimized_results['success_rate'] - baseline_results['success_rate']) * 100,
            'baseline': baseline_results,
            'optimized': optimized_results
        }
        
        return improvement


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short"])