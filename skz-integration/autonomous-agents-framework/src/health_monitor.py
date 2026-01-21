"""
Health Monitoring System for SKZ Autonomous Agents Framework
Provides comprehensive health checks for all system components
"""
import asyncio
import aiohttp
import json
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import redis
import mysql.connector
from mysql.connector import Error

logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    """Health status data structure"""
    service: str
    status: str  # 'healthy', 'unhealthy', 'degraded'
    response_time: float
    timestamp: str
    details: Dict[str, Any]
    error: Optional[str] = None

class HealthMonitor:
    """Comprehensive health monitoring system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services = {
            'api_gateway': f"http://localhost:5000/health",
            'research_agent': f"http://localhost:5001/health",
            'submission_agent': f"http://localhost:5002/health",
            'editorial_agent': f"http://localhost:5003/health",
            'review_agent': f"http://localhost:5004/health",
            'quality_agent': f"http://localhost:5005/health",
            'publishing_agent': f"http://localhost:5006/health",
            'analytics_agent': f"http://localhost:5007/health"
        }
        self.redis_client = None
        self.mysql_connection = None
        
    async def initialize_connections(self):
        """Initialize external service connections"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=self.config.get('redis', {}).get('host', 'localhost'),
                port=self.config.get('redis', {}).get('port', 6379),
                db=self.config.get('redis', {}).get('db', 0),
                decode_responses=True
            )
            
            # Test Redis connection
            self.redis_client.ping()
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            self.redis_client = None
    
    async def check_service_health(self, service_name: str, url: str) -> HealthStatus:
        """Check health of individual service"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        try:
                            response_data = await response.json()
                        except:
                            response_data = {}
                        
                        return HealthStatus(
                            service=service_name,
                            status='healthy',
                            response_time=response_time,
                            timestamp=datetime.now().isoformat(),
                            details={
                                'http_status': response.status,
                                'response_data': response_data
                            }
                        )
                    else:
                        return HealthStatus(
                            service=service_name,
                            status='unhealthy',
                            response_time=response_time,
                            timestamp=datetime.now().isoformat(),
                            details={'http_status': response.status},
                            error=f"HTTP {response.status}"
                        )
                        
        except asyncio.TimeoutError:
            return HealthStatus(
                service=service_name,
                status='unhealthy',
                response_time=10.0,
                timestamp=datetime.now().isoformat(),
                details={},
                error="Timeout after 10 seconds"
            )
        except Exception as e:
            return HealthStatus(
                service=service_name,
                status='unhealthy',
                response_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                details={},
                error=str(e)
            )
    
    def check_database_health(self) -> HealthStatus:
        """Check MySQL database health"""
        start_time = time.time()
        
        try:
            db_config = self.config.get('database', {})
            connection = mysql.connector.connect(
                host=db_config.get('host', 'localhost'),
                port=db_config.get('port', 3306),
                user=db_config.get('user', 'root'),
                password=db_config.get('password', ''),
                database=db_config.get('name', 'ojs'),
                connection_timeout=5
            )
            
            if connection.is_connected():
                cursor = connection.cursor()
                cursor.execute("SELECT VERSION()")
                version = cursor.fetchone()
                cursor.close()
                connection.close()
                
                response_time = time.time() - start_time
                
                return HealthStatus(
                    service='mysql_database',
                    status='healthy',
                    response_time=response_time,
                    timestamp=datetime.now().isoformat(),
                    details={
                        'version': version[0] if version else 'unknown',
                        'connection_successful': True
                    }
                )
                
        except Error as e:
            return HealthStatus(
                service='mysql_database',
                status='unhealthy',
                response_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                details={},
                error=f"MySQL Error: {e}"
            )
        except Exception as e:
            return HealthStatus(
                service='mysql_database',
                status='unhealthy',
                response_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                details={},
                error=str(e)
            )
    
    async def check_redis_health(self) -> HealthStatus:
        """Check Redis health"""
        start_time = time.time()
        
        try:
            if self.redis_client is None:
                await self.initialize_connections()
            
            if self.redis_client:
                # Test basic operations
                test_key = f"health_check_{int(time.time())}"
                self.redis_client.set(test_key, "test_value", ex=60)
                retrieved_value = self.redis_client.get(test_key)
                self.redis_client.delete(test_key)
                
                if retrieved_value == "test_value":
                    response_time = time.time() - start_time
                    
                    # Get Redis info
                    info = self.redis_client.info()
                    
                    return HealthStatus(
                        service='redis',
                        status='healthy',
                        response_time=response_time,
                        timestamp=datetime.now().isoformat(),
                        details={
                            'version': info.get('redis_version', 'unknown'),
                            'used_memory': info.get('used_memory_human', 'unknown'),
                            'connected_clients': info.get('connected_clients', 0)
                        }
                    )
                else:
                    return HealthStatus(
                        service='redis',
                        status='unhealthy',
                        response_time=time.time() - start_time,
                        timestamp=datetime.now().isoformat(),
                        details={},
                        error="Redis operation test failed"
                    )
            else:
                return HealthStatus(
                    service='redis',
                    status='unhealthy',
                    response_time=time.time() - start_time,
                    timestamp=datetime.now().isoformat(),
                    details={},
                    error="Redis connection not initialized"
                )
                
        except Exception as e:
            return HealthStatus(
                service='redis',
                status='unhealthy',
                response_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                details={},
                error=str(e)
            )
    
    def check_system_resources(self) -> HealthStatus:
        """Check system resource utilization"""
        start_time = time.time()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on thresholds
            status = 'healthy'
            warnings = []
            
            if cpu_percent > 80:
                status = 'degraded'
                warnings.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 85:
                status = 'degraded'
                warnings.append(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                status = 'degraded'
                warnings.append(f"High disk usage: {disk.percent}%")
            
            response_time = time.time() - start_time
            
            return HealthStatus(
                service='system_resources',
                status=status,
                response_time=response_time,
                timestamp=datetime.now().isoformat(),
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': round(memory.available / (1024**3), 2),
                    'disk_percent': disk.percent,
                    'disk_free_gb': round(disk.free / (1024**3), 2),
                    'warnings': warnings
                }
            )
            
        except Exception as e:
            return HealthStatus(
                service='system_resources',
                status='unhealthy',
                response_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                details={},
                error=str(e)
            )
    
    async def check_all_services(self) -> Dict[str, HealthStatus]:
        """Check health of all services"""
        results = {}
        
        # Check agent services
        service_tasks = [
            self.check_service_health(service, url) 
            for service, url in self.services.items()
        ]
        
        service_results = await asyncio.gather(*service_tasks, return_exceptions=True)
        
        for i, (service, _) in enumerate(self.services.items()):
            result = service_results[i]
            if isinstance(result, Exception):
                results[service] = HealthStatus(
                    service=service,
                    status='unhealthy',
                    response_time=0.0,
                    timestamp=datetime.now().isoformat(),
                    details={},
                    error=str(result)
                )
            else:
                results[service] = result
        
        # Check infrastructure services
        results['mysql_database'] = self.check_database_health()
        results['redis'] = self.check_redis_health()
        results['system_resources'] = self.check_system_resources()
        
        return results
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        health_results = await self.check_all_services()
        
        # Calculate summary metrics
        total_services = len(health_results)
        healthy_services = sum(1 for result in health_results.values() if result.status == 'healthy')
        degraded_services = sum(1 for result in health_results.values() if result.status == 'degraded')
        unhealthy_services = sum(1 for result in health_results.values() if result.status == 'unhealthy')
        
        overall_status = 'healthy'
        if unhealthy_services > 0:
            overall_status = 'unhealthy'
        elif degraded_services > 0:
            overall_status = 'degraded'
        
        avg_response_time = sum(result.response_time for result in health_results.values()) / total_services
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_services': total_services,
                'healthy': healthy_services,
                'degraded': degraded_services,
                'unhealthy': unhealthy_services,
                'avg_response_time': round(avg_response_time, 3)
            },
            'services': {
                service: asdict(status) for service, status in health_results.items()
            }
        }
    
    async def continuous_monitoring(self, interval_seconds: int = 60):
        """Run continuous health monitoring"""
        logger.info(f"Starting continuous health monitoring (interval: {interval_seconds}s)")
        
        while True:
            try:
                health_summary = await self.get_health_summary()
                
                # Log health status
                if health_summary['overall_status'] != 'healthy':
                    logger.warning(f"System health degraded: {health_summary['overall_status']}")
                    logger.warning(f"Summary: {health_summary['summary']}")
                else:
                    logger.info(f"System health: {health_summary['overall_status']}")
                
                # Store health data in Redis if available
                if self.redis_client:
                    try:
                        health_key = f"health_status:{int(time.time())}"
                        self.redis_client.setex(health_key, 3600, json.dumps(health_summary))
                    except Exception as e:
                        logger.error(f"Failed to store health data in Redis: {e}")
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(interval_seconds)

# Utility functions for quick health checks
async def quick_health_check(config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a quick health check of all services"""
    monitor = HealthMonitor(config)
    await monitor.initialize_connections()
    return await monitor.get_health_summary()

def create_health_monitor(config_path: str = None) -> HealthMonitor:
    """Create a health monitor instance with configuration"""
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'database': {
                'host': 'localhost',
                'port': 3306,
                'user': 'root',
                'password': '',
                'name': 'ojs'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            }
        }
    
    return HealthMonitor(config)
