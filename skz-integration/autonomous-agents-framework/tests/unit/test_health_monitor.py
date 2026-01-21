"""
Unit tests for the Health Monitor system
Tests the comprehensive health monitoring functionality
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import json
from datetime import datetime

# Import the health monitor classes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from health_monitor import HealthMonitor, HealthStatus, quick_health_check


class TestHealthStatus:
    """Test the HealthStatus dataclass"""
    
    def test_health_status_creation(self):
        """Test creating a HealthStatus instance"""
        status = HealthStatus(
            service='test_service',
            status='healthy',
            response_time=0.123,
            timestamp='2024-01-01T00:00:00Z',
            details={'test': 'data'}
        )
        
        assert status.service == 'test_service'
        assert status.status == 'healthy'
        assert status.response_time == 0.123
        assert status.timestamp == '2024-01-01T00:00:00Z'
        assert status.details == {'test': 'data'}
        assert status.error is None
    
    def test_health_status_with_error(self):
        """Test creating a HealthStatus instance with error"""
        status = HealthStatus(
            service='test_service',
            status='unhealthy',
            response_time=10.0,
            timestamp='2024-01-01T00:00:00Z',
            details={},
            error='Connection timeout'
        )
        
        assert status.service == 'test_service'
        assert status.status == 'unhealthy'
        assert status.error == 'Connection timeout'


class TestHealthMonitor:
    """Test the HealthMonitor class"""
    
    def setup_method(self):
        """Setup test configuration"""
        self.config = {
            'database': {
                'host': 'localhost',
                'port': 3306,
                'user': 'test_user',
                'password': 'test_password',
                'name': 'test_db'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            }
        }
        self.monitor = HealthMonitor(self.config)
    
    def test_health_monitor_initialization(self):
        """Test HealthMonitor initialization"""
        assert self.monitor.config == self.config
        assert 'api_gateway' in self.monitor.services
        assert 'research_agent' in self.monitor.services
        assert self.monitor.services['api_gateway'] == 'http://localhost:5000/health'
    
    @pytest.mark.asyncio
    async def test_initialize_connections_success(self):
        """Test successful connection initialization"""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis_class.return_value = mock_redis
            
            await self.monitor.initialize_connections()
            
            assert self.monitor.redis_client is not None
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_connections_failure(self):
        """Test connection initialization failure"""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis_class.side_effect = Exception('Connection failed')
            
            await self.monitor.initialize_connections()
            
            assert self.monitor.redis_client is None
    
    @pytest.mark.asyncio
    async def test_check_service_health_success(self, async_context_mock):
        """Test successful service health check"""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'status': 'healthy'})
        
        mock_session = Mock()
        mock_session.get = Mock(return_value=async_context_mock(mock_response))
        
        with patch('aiohttp.ClientSession', return_value=async_context_mock(mock_session)):
            result = await self.monitor.check_service_health('test_service', 'http://localhost:5000/health')
            
            assert result.service == 'test_service'
            assert result.status == 'healthy'
            assert result.error is None
            assert result.details['http_status'] == 200
    
    @pytest.mark.asyncio
    async def test_check_service_health_failure(self, async_context_mock):
        """Test failed service health check"""
        mock_response = Mock()
        mock_response.status = 503
        
        mock_session = Mock()
        mock_session.get = Mock(return_value=async_context_mock(mock_response))
        
        with patch('aiohttp.ClientSession', return_value=async_context_mock(mock_session)):
            result = await self.monitor.check_service_health('test_service', 'http://localhost:5000/health')
            
            assert result.service == 'test_service'
            assert result.status == 'unhealthy'
            assert result.error == 'HTTP 503'
            assert result.details['http_status'] == 503
    
    @pytest.mark.asyncio
    async def test_check_service_health_timeout(self):
        """Test service health check timeout"""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session.get.side_effect = asyncio.TimeoutError()
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await self.monitor.check_service_health('test_service', 'http://localhost:5000/health')
            
            assert result.service == 'test_service'
            assert result.status == 'unhealthy'
            assert result.error == 'Timeout after 10 seconds'
    
    def test_check_database_health_success(self):
        """Test successful database health check"""
        with patch('mysql.connector.connect') as mock_connect:
            mock_connection = Mock()
            mock_connection.is_connected.return_value = True
            mock_cursor = Mock()
            mock_cursor.fetchone.return_value = ['8.0.28']
            mock_connection.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_connection
            
            result = self.monitor.check_database_health()
            
            assert result.service == 'mysql_database'
            assert result.status == 'healthy'
            assert result.details['version'] == '8.0.28'
            assert result.details['connection_successful'] is True
    
    def test_check_database_health_failure(self):
        """Test failed database health check"""
        with patch('mysql.connector.connect') as mock_connect:
            mock_connect.side_effect = Exception('Connection refused')
            
            result = self.monitor.check_database_health()
            
            assert result.service == 'mysql_database'
            assert result.status == 'unhealthy'
            assert 'Connection refused' in result.error
    
    def test_check_redis_health_success(self):
        """Test successful Redis health check"""
        mock_redis = Mock()
        mock_redis.set.return_value = True
        mock_redis.get.return_value = 'test_value'
        mock_redis.delete.return_value = 1
        mock_redis.info.return_value = {
            'redis_version': '6.2.6',
            'used_memory_human': '1.2M',
            'connected_clients': 3
        }
        
        self.monitor.redis_client = mock_redis
        
        result = self.monitor.check_redis_health()
        
        assert result.service == 'redis'
        assert result.status == 'healthy'
        assert result.details['version'] == '6.2.6'
        assert result.details['used_memory'] == '1.2M'
        assert result.details['connected_clients'] == 3
    
    def test_check_redis_health_failure(self):
        """Test failed Redis health check"""
        mock_redis = Mock()
        mock_redis.set.side_effect = Exception('Redis connection failed')
        
        self.monitor.redis_client = mock_redis
        
        result = self.monitor.check_redis_health()
        
        assert result.service == 'redis'
        assert result.status == 'unhealthy'
        assert 'Redis connection failed' in result.error
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_system_resources_healthy(self, mock_disk, mock_memory, mock_cpu):
        """Test system resources check - healthy status"""
        mock_cpu.return_value = 45.5
        
        mock_memory_info = Mock()
        mock_memory_info.percent = 60.2
        mock_memory_info.available = 8 * (1024**3)  # 8GB
        mock_memory.return_value = mock_memory_info
        
        mock_disk_info = Mock()
        mock_disk_info.percent = 75.0
        mock_disk_info.free = 100 * (1024**3)  # 100GB
        mock_disk.return_value = mock_disk_info
        
        result = self.monitor.check_system_resources()
        
        assert result.service == 'system_resources'
        assert result.status == 'healthy'
        assert result.details['cpu_percent'] == 45.5
        assert result.details['memory_percent'] == 60.2
        assert result.details['disk_percent'] == 75.0
        assert len(result.details['warnings']) == 0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_system_resources_degraded(self, mock_disk, mock_memory, mock_cpu):
        """Test system resources check - degraded status"""
        mock_cpu.return_value = 85.0  # High CPU usage
        
        mock_memory_info = Mock()
        mock_memory_info.percent = 90.0  # High memory usage
        mock_memory_info.available = 1 * (1024**3)  # 1GB
        mock_memory.return_value = mock_memory_info
        
        mock_disk_info = Mock()
        mock_disk_info.percent = 95.0  # High disk usage
        mock_disk_info.free = 5 * (1024**3)  # 5GB
        mock_disk.return_value = mock_disk_info
        
        result = self.monitor.check_system_resources()
        
        assert result.service == 'system_resources'
        assert result.status == 'degraded'
        assert len(result.details['warnings']) == 3
        assert any('High CPU usage' in warning for warning in result.details['warnings'])
        assert any('High memory usage' in warning for warning in result.details['warnings'])
        assert any('High disk usage' in warning for warning in result.details['warnings'])
    
    @pytest.mark.asyncio
    async def test_get_health_summary_all_healthy(self):
        """Test health summary with all services healthy"""
        # Mock all health check methods
        healthy_status = HealthStatus(
            service='test_service',
            status='healthy',
            response_time=0.1,
            timestamp='2024-01-01T00:00:00Z',
            details={}
        )
        
        with patch.object(self.monitor, 'check_all_services', return_value={
            'api_gateway': healthy_status,
            'research_agent': healthy_status,
            'mysql_database': healthy_status,
            'redis': healthy_status,
            'system_resources': healthy_status
        }):
            summary = await self.monitor.get_health_summary()
            
            assert summary['overall_status'] == 'healthy'
            assert summary['summary']['total_services'] == 5
            assert summary['summary']['healthy'] == 5
            assert summary['summary']['degraded'] == 0
            assert summary['summary']['unhealthy'] == 0
    
    @pytest.mark.asyncio
    async def test_get_health_summary_mixed_status(self):
        """Test health summary with mixed service status"""
        healthy_status = HealthStatus('svc1', 'healthy', 0.1, '2024-01-01T00:00:00Z', {})
        degraded_status = HealthStatus('svc2', 'degraded', 0.2, '2024-01-01T00:00:00Z', {})
        unhealthy_status = HealthStatus('svc3', 'unhealthy', 1.0, '2024-01-01T00:00:00Z', {})
        
        with patch.object(self.monitor, 'check_all_services', return_value={
            'service1': healthy_status,
            'service2': degraded_status,
            'service3': unhealthy_status
        }):
            summary = await self.monitor.get_health_summary()
            
            assert summary['overall_status'] == 'unhealthy'  # Worst status wins
            assert summary['summary']['total_services'] == 3
            assert summary['summary']['healthy'] == 1
            assert summary['summary']['degraded'] == 1
            assert summary['summary']['unhealthy'] == 1


class TestUtilityFunctions:
    """Test utility functions"""
    
    @pytest.mark.asyncio
    async def test_quick_health_check(self):
        """Test quick health check utility function"""
        config = {'test': 'config'}
        
        with patch('health_monitor.HealthMonitor') as mock_monitor_class:
            mock_monitor = Mock()
            mock_monitor.initialize_connections = AsyncMock()
            mock_monitor.get_health_summary = AsyncMock(return_value={'status': 'healthy'})
            mock_monitor_class.return_value = mock_monitor
            
            result = await quick_health_check(config)
            
            mock_monitor_class.assert_called_once_with(config)
            mock_monitor.initialize_connections.assert_called_once()
            mock_monitor.get_health_summary.assert_called_once()
            assert result == {'status': 'healthy'}
    
    def test_create_health_monitor_with_config(self):
        """Test creating health monitor with config file"""
        with patch('builtins.open') as mock_open:
            with patch('json.load') as mock_json_load:
                test_config = {'test': 'config'}
                mock_json_load.return_value = test_config
                
                from health_monitor import create_health_monitor
                monitor = create_health_monitor('test_config.json')
                
                mock_open.assert_called_once_with('test_config.json', 'r')
                assert monitor.config == test_config
    
    def test_create_health_monitor_without_config(self):
        """Test creating health monitor with default config"""
        from health_monitor import create_health_monitor
        monitor = create_health_monitor()
        
        assert 'database' in monitor.config
        assert 'redis' in monitor.config
        assert monitor.config['database']['host'] == 'localhost'
        assert monitor.config['redis']['port'] == 6379
