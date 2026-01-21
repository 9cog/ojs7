"""
API Gateway for Autonomous Agents Framework
Handles routing and load balancing for all agent microservices
Enhanced with JWT authentication and authorization
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import json
import os
import sys
import logging
from datetime import datetime
import uuid

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from auth_service import (
    auth_service, 
    authorization_service, 
    require_auth, 
    require_permission,
    require_api_auth,
    get_current_user
)

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service registry - maps service names to endpoints
SERVICE_REGISTRY = {
    'research-discovery': os.getenv('RESEARCH_DISCOVERY_URL', 'http://localhost:5001'),
    'submission-assistant': os.getenv('SUBMISSION_ASSISTANT_URL', 'http://localhost:5002'),
    'editorial-orchestration': os.getenv('EDITORIAL_ORCHESTRATION_URL', 'http://localhost:5003'),
    'review-coordination': os.getenv('REVIEW_COORDINATION_URL', 'http://localhost:5004'),
    'content-quality': os.getenv('CONTENT_QUALITY_URL', 'http://localhost:5005'),
    'publishing-production': os.getenv('PUBLISHING_PRODUCTION_URL', 'http://localhost:5006'),
    'analytics-monitoring': os.getenv('ANALYTICS_MONITORING_URL', 'http://localhost:5007'),
}

@app.route('/api/v1/auth/login', methods=['POST'])
def login():
    """Login endpoint for generating JWT tokens"""
    try:
        data = request.get_json()
        
        # This would normally validate against OJS user database
        # For now, we'll accept basic credentials for testing
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        # Mock user validation (replace with actual OJS integration)
        user_data = {
            'user_id': 1,
            'username': username,
            'email': f'{username}@example.com',
            'roles': ['ROLE_ID_MANAGER'],  # Default role for testing
            'context_id': 1
        }
        
        # Generate JWT token
        token = auth_service.generate_token(user_data)
        
        return jsonify({
            'token': token,
            'user': user_data,
            'expires_in': 86400  # 24 hours
        })
        
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/v1/auth/validate', methods=['POST'])
@require_auth
def validate_token():
    """Validate JWT token endpoint"""
    user_data = get_current_user()
    return jsonify({
        'valid': True,
        'user': user_data
    })

@app.route('/api/v1/auth/permissions')
@require_auth
def get_permissions():
    """Get user permissions"""
    user_data = get_current_user()
    user_roles = user_data.get('roles', [])
    permissions = authorization_service.get_user_permissions(user_roles)
    
    return jsonify({
        'roles': user_roles,
        'permissions': permissions
    })

@app.route('/health')
def health_check():
    """Gateway health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'api-gateway',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'auth_enabled': True
    })

@app.route('/api/v1/services')
@require_auth
@require_permission('agents:view')
def list_services():
    """List all available services and their status"""
    services = []
    for service_name, service_url in SERVICE_REGISTRY.items():
        try:
            # Check service health
            response = requests.get(f"{service_url}/health", timeout=5)
            status = 'healthy' if response.status_code == 200 else 'unhealthy'
        except Exception as e:
            status = 'unavailable'
            logger.error(f"Service {service_name} health check failed: {e}")
        
        services.append({
            'name': service_name,
            'url': service_url,
            'status': status
        })
    
    return jsonify({
        'services': services,
        'total_count': len(services),
        'healthy_count': len([s for s in services if s['status'] == 'healthy'])
    })

@app.route('/api/v1/agents', methods=['GET'])
@require_auth
@require_permission('agents:view')
def list_agents():
    """List all agents from all services"""
    all_agents = []
    
    for service_name, service_url in SERVICE_REGISTRY.items():
        try:
            response = requests.get(f"{service_url}/agent", timeout=10)
            if response.status_code == 200:
                agent_data = response.json()
                agent_data['service'] = service_name
                all_agents.append(agent_data)
        except Exception as e:
            logger.error(f"Failed to get agent data from {service_name}: {e}")
    
    return jsonify({
        'agents': all_agents,
        'total_count': len(all_agents)
    })

@app.route('/api/v1/agents/<service_name>', methods=['GET'])
@require_auth
@require_permission('agents:view')
def get_agent(service_name):
    """Get specific agent details"""
    if service_name not in SERVICE_REGISTRY:
        return jsonify({'error': 'Service not found'}), 404
    
    try:
        service_url = SERVICE_REGISTRY[service_name]
        response = requests.get(f"{service_url}/agent", timeout=10)
        
        if response.status_code == 200:
            agent_data = response.json()
            agent_data['service'] = service_name
            return jsonify(agent_data)
        else:
            return jsonify({'error': 'Service unavailable'}), 503
    except Exception as e:
        logger.error(f"Failed to get agent data from {service_name}: {e}")
        return jsonify({'error': 'Service error'}), 500

@app.route('/api/v1/agents/<service_name>/action', methods=['POST'])
@require_auth
@require_permission('agents:execute')
def trigger_agent_action(service_name):
    """Trigger an action for a specific agent"""
    if service_name not in SERVICE_REGISTRY:
        return jsonify({'error': 'Service not found'}), 404
    
    try:
        service_url = SERVICE_REGISTRY[service_name]
        
        # Forward the request to the appropriate service with authentication context
        user_data = get_current_user()
        request_payload = request.get_json() or {}
        request_payload['_auth'] = {
            'user_id': user_data.get('user_id'),
            'username': user_data.get('username'),
            'roles': user_data.get('roles', []),
            'context_id': user_data.get('context_id')
        }
        
        response = requests.post(
            f"{service_url}/action",
            json=request_payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        # Add request tracking
        result = response.json() if response.status_code == 200 else {'error': 'Service error'}
        result['gateway_request_id'] = str(uuid.uuid4())
        result['gateway_timestamp'] = datetime.now().isoformat()
        
        return jsonify(result), response.status_code
        
    except Exception as e:
        logger.error(f"Failed to trigger action on {service_name}: {e}")
        return jsonify({'error': 'Service error', 'details': str(e)}), 500

@app.route('/api/v1/agents/<service_name>/<path:endpoint>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_to_service(service_name, endpoint):
    """Generic proxy to forward requests to services"""
    if service_name not in SERVICE_REGISTRY:
        return jsonify({'error': 'Service not found'}), 404
    
    try:
        service_url = SERVICE_REGISTRY[service_name]
        target_url = f"{service_url}/{endpoint}"
        
        # Forward the request
        response = requests.request(
            method=request.method,
            url=target_url,
            headers={k: v for k, v in request.headers if k.lower() != 'host'},
            data=request.get_data(),
            params=request.args,
            timeout=30
        )
        
        # Create response
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for (name, value) in response.raw.headers.items()
                  if name.lower() not in excluded_headers]
        
        return Response(response.content, response.status_code, headers)
        
    except Exception as e:
        logger.error(f"Failed to proxy request to {service_name}: {e}")
        return jsonify({'error': 'Service error', 'details': str(e)}), 500

@app.route('/api/v1/dashboard')
@require_auth
@require_permission('analytics:view')
def dashboard_data():
    """Aggregate dashboard data from all services"""
    dashboard_data = {
        'summary': {
            'total_services': len(SERVICE_REGISTRY),
            'healthy_services': 0,
            'total_agents': 0,
            'timestamp': datetime.now().isoformat()
        },
        'services': [],
        'system_metrics': {}
    }
    
    for service_name, service_url in SERVICE_REGISTRY.items():
        try:
            # Get service health
            health_response = requests.get(f"{service_url}/health", timeout=5)
            is_healthy = health_response.status_code == 200
            
            if is_healthy:
                dashboard_data['summary']['healthy_services'] += 1
                dashboard_data['summary']['total_agents'] += 1
                
                # Get agent data
                agent_response = requests.get(f"{service_url}/agent", timeout=5)
                if agent_response.status_code == 200:
                    agent_data = agent_response.json()
                    dashboard_data['services'].append({
                        'name': service_name,
                        'status': 'healthy',
                        'agent_data': agent_data
                    })
                    
        except Exception as e:
            logger.error(f"Failed to get dashboard data from {service_name}: {e}")
            dashboard_data['services'].append({
                'name': service_name,
                'status': 'unavailable',
                'error': str(e)
            })
    
    return jsonify(dashboard_data)

@app.route('/api/v1/metrics')
@require_auth
@require_permission('analytics:view')
def metrics():
    """Aggregate metrics from all services"""
    metrics_data = {
        'gateway': {
            'service_count': len(SERVICE_REGISTRY),
            'timestamp': datetime.now().isoformat()
        },
        'services': {}
    }
    
    for service_name, service_url in SERVICE_REGISTRY.items():
        try:
            response = requests.get(f"{service_url}/metrics", timeout=5)
            if response.status_code == 200:
                metrics_data['services'][service_name] = response.json()
        except Exception as e:
            logger.error(f"Failed to get metrics from {service_name}: {e}")
            metrics_data['services'][service_name] = {'error': str(e)}
    
    return jsonify(metrics_data)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)