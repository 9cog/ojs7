"""
Health Check API Routes for SKZ Autonomous Agents Framework
Provides RESTful endpoints for system health monitoring
"""
from flask import Blueprint, jsonify, request
from datetime import datetime
import json
import logging
import asyncio
from functools import wraps
import os
import sys

# Add the parent directory to the path to import health_monitor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from health_monitor import HealthMonitor, quick_health_check

logger = logging.getLogger(__name__)

# Create Blueprint for health routes
health_bp = Blueprint('health', __name__, url_prefix='/api/v1/health')

# Global health monitor instance
health_monitor = None

def async_route(f):
    """Decorator to handle async route functions"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

def init_health_monitor(config):
    """Initialize the global health monitor"""
    global health_monitor
    health_monitor = HealthMonitor(config)
    return health_monitor

@health_bp.route('/', methods=['GET'])
@async_route
async def basic_health_check():
    """
    Basic health check endpoint
    Returns simple status for load balancers and monitoring systems
    """
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'skz-agents-framework',
            'version': '1.0.0'
        }), 200
    except Exception as e:
        logger.error(f"Basic health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

@health_bp.route('/detailed', methods=['GET'])
@async_route
async def detailed_health_check():
    """
    Detailed health check endpoint
    Returns comprehensive health status of all system components
    """
    try:
        global health_monitor
        if health_monitor is None:
            # Use default configuration if health monitor not initialized
            config = {
                'database': {
                    'host': os.getenv('DATABASE_HOST', 'localhost'),
                    'port': int(os.getenv('DATABASE_PORT', 3306)),
                    'user': os.getenv('DATABASE_USER', 'root'),
                    'password': os.getenv('DATABASE_PASSWORD', ''),
                    'name': os.getenv('DATABASE_NAME', 'ojs')
                },
                'redis': {
                    'host': os.getenv('REDIS_HOST', 'localhost'),
                    'port': int(os.getenv('REDIS_PORT', 6379)),
                    'db': int(os.getenv('REDIS_DB', 0))
                }
            }
            health_monitor = HealthMonitor(config)
            await health_monitor.initialize_connections()
        
        health_summary = await health_monitor.get_health_summary()
        
        # Determine HTTP status code based on overall health
        status_code = 200
        if health_summary['overall_status'] == 'unhealthy':
            status_code = 503  # Service Unavailable
        elif health_summary['overall_status'] == 'degraded':
            status_code = 206  # Partial Content
        
        return jsonify(health_summary), status_code
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'overall_status': 'unhealthy'
        }), 500

@health_bp.route('/services', methods=['GET'])
@async_route
async def services_health():
    """
    Get health status of individual agent services
    """
    try:
        global health_monitor
        if health_monitor is None:
            return jsonify({'error': 'Health monitor not initialized'}), 500
        
        health_results = await health_monitor.check_all_services()
        
        # Filter to only agent services
        agent_services = {
            service: status for service, status in health_results.items()
            if 'agent' in service or service == 'api_gateway'
        }
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'services': {
                service: {
                    'status': status.status,
                    'response_time': status.response_time,
                    'error': status.error
                } for service, status in agent_services.items()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Services health check failed: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@health_bp.route('/infrastructure', methods=['GET'])
@async_route
async def infrastructure_health():
    """
    Get health status of infrastructure components (database, redis, system)
    """
    try:
        global health_monitor
        if health_monitor is None:
            return jsonify({'error': 'Health monitor not initialized'}), 500
        
        health_results = await health_monitor.check_all_services()
        
        # Filter to only infrastructure services
        infrastructure_services = {
            service: status for service, status in health_results.items()
            if service in ['mysql_database', 'redis', 'system_resources']
        }
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'infrastructure': {
                service: {
                    'status': status.status,
                    'response_time': status.response_time,
                    'details': status.details,
                    'error': status.error
                } for service, status in infrastructure_services.items()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Infrastructure health check failed: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@health_bp.route('/metrics', methods=['GET'])
@async_route
async def health_metrics():
    """
    Get health metrics in Prometheus format for monitoring systems
    """
    try:
        global health_monitor
        if health_monitor is None:
            return jsonify({'error': 'Health monitor not initialized'}), 500
        
        health_results = await health_monitor.check_all_services()
        
        # Generate Prometheus-style metrics
        metrics = []
        timestamp = int(datetime.now().timestamp() * 1000)
        
        for service, status in health_results.items():
            # Service health status (1 = healthy, 0.5 = degraded, 0 = unhealthy)
            health_value = 1 if status.status == 'healthy' else (0.5 if status.status == 'degraded' else 0)
            metrics.append(f'skz_service_health{{service="{service}"}} {health_value} {timestamp}')
            
            # Service response time
            if status.response_time is not None:
                metrics.append(f'skz_service_response_time{{service="{service}"}} {status.response_time} {timestamp}')
        
        return '\n'.join(metrics), 200, {'Content-Type': 'text/plain; charset=utf-8'}
        
    except Exception as e:
        logger.error(f"Health metrics failed: {e}")
        return f'# Error generating metrics: {e}', 500, {'Content-Type': 'text/plain; charset=utf-8'}

@health_bp.route('/ready', methods=['GET'])
@async_route
async def readiness_check():
    """
    Kubernetes readiness probe endpoint
    Returns 200 if the service is ready to receive traffic
    """
    try:
        global health_monitor
        if health_monitor is None:
            return jsonify({'ready': False, 'reason': 'Health monitor not initialized'}), 503
        
        # Check critical services for readiness
        health_results = await health_monitor.check_all_services()
        
        critical_services = ['mysql_database', 'redis']
        ready = all(
            health_results.get(service, {}).status in ['healthy', 'degraded']
            for service in critical_services
        )
        
        if ready:
            return jsonify({
                'ready': True,
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            failed_services = [
                service for service in critical_services
                if health_results.get(service, {}).status == 'unhealthy'
            ]
            return jsonify({
                'ready': False,
                'reason': f'Critical services unhealthy: {failed_services}',
                'timestamp': datetime.now().isoformat()
            }), 503
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({
            'ready': False,
            'reason': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@health_bp.route('/live', methods=['GET'])
def liveness_check():
    """
    Kubernetes liveness probe endpoint
    Returns 200 if the service is alive (simple check)
    """
    try:
        return jsonify({
            'alive': True,
            'timestamp': datetime.now().isoformat(),
            'pid': os.getpid()
        }), 200
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return jsonify({
            'alive': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@health_bp.route('/version', methods=['GET'])
def version_info():
    """
    Get version and build information
    """
    try:
        # Try to read version from environment or file
        version = os.getenv('SKZ_VERSION', '1.0.0')
        build_date = os.getenv('SKZ_BUILD_DATE', 'unknown')
        commit_hash = os.getenv('SKZ_COMMIT_HASH', 'unknown')
        
        return jsonify({
            'version': version,
            'build_date': build_date,
            'commit_hash': commit_hash,
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Version info failed: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@health_bp.route('/config', methods=['GET'])
def config_status():
    """
    Get configuration status (non-sensitive information only)
    """
    try:
        config_status = {
            'database_configured': bool(os.getenv('DATABASE_HOST')),
            'redis_configured': bool(os.getenv('REDIS_HOST')),
            'ojs_api_configured': bool(os.getenv('OJS_API_KEY')),
            'agents_configured': {
                'research_agent': bool(os.getenv('RESEARCH_AGENT_PORT')),
                'submission_agent': bool(os.getenv('SUBMISSION_AGENT_PORT')),
                'editorial_agent': bool(os.getenv('EDITORIAL_AGENT_PORT')),
                'review_agent': bool(os.getenv('REVIEW_AGENT_PORT')),
                'quality_agent': bool(os.getenv('QUALITY_AGENT_PORT')),
                'publishing_agent': bool(os.getenv('PUBLISHING_AGENT_PORT')),
                'analytics_agent': bool(os.getenv('ANALYTICS_AGENT_PORT'))
            },
            'external_apis': {
                'inci_api': bool(os.getenv('INCI_API_KEY')),
                'uspto_api': bool(os.getenv('USPTO_API_KEY')),
                'pubmed_api': bool(os.getenv('PUBMED_API_KEY'))
            }
        }
        
        return jsonify({
            'configuration': config_status,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Config status failed: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Error handlers for the health blueprint
@health_bp.errorhandler(404)
def health_not_found(e):
    return jsonify({
        'error': 'Health endpoint not found',
        'available_endpoints': [
            '/api/v1/health/',
            '/api/v1/health/detailed',
            '/api/v1/health/services',
            '/api/v1/health/infrastructure',
            '/api/v1/health/metrics',
            '/api/v1/health/ready',
            '/api/v1/health/live',
            '/api/v1/health/version',
            '/api/v1/health/config'
        ],
        'timestamp': datetime.now().isoformat()
    }), 404

@health_bp.errorhandler(500)
def health_server_error(e):
    return jsonify({
        'error': 'Internal server error in health check',
        'timestamp': datetime.now().isoformat()
    }), 500
