"""
API Gateway for Autonomous Agents Framework - Local Test Configuration
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import json
import os
import logging
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service registry - local testing configuration
SERVICE_REGISTRY = {
    'research-discovery': 'http://localhost:5001',
    'submission-assistant': 'http://localhost:5002',
    'editorial-orchestration': 'http://localhost:5003',
    'review-coordination': 'http://localhost:5004',
    'content-quality': 'http://localhost:5005',
    'publishing-production': 'http://localhost:5006',
    'analytics-monitoring': 'http://localhost:5007',
}

@app.route('/')
def home():
    """API Gateway dashboard"""
    return jsonify({
        'service': 'Autonomous Agents API Gateway',
        'version': '1.0.0',
        'status': 'running',
        'registered_services': len(SERVICE_REGISTRY),
        'endpoints': {
            'health': '/health',
            'services': '/api/v1/services',
            'agents': '/api/v1/agents',
            'dashboard': '/api/v1/dashboard',
            'metrics': '/api/v1/metrics'
        }
    })

@app.route('/health')
def health_check():
    """Gateway health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'api-gateway',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/v1/services')
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
            logger.debug(f"Service {service_name} health check failed: {e}")
        
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
def list_agents():
    """List all agents from all services"""
    all_agents = []
    
    for service_name, service_url in SERVICE_REGISTRY.items():
        try:
            response = requests.get(f"{service_url}/agent", timeout=10)
            if response.status_code == 200:
                agent_data = response.json()
                agent_data['service'] = service_name
                agent_data['service_url'] = service_url
                all_agents.append(agent_data)
        except Exception as e:
            logger.debug(f"Failed to get agent data from {service_name}: {e}")
    
    return jsonify({
        'agents': all_agents,
        'total_count': len(all_agents)
    })

@app.route('/api/v1/agents/<service_name>', methods=['GET'])
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
            agent_data['service_url'] = service_url
            return jsonify(agent_data)
        else:
            return jsonify({'error': 'Service unavailable'}), 503
    except Exception as e:
        logger.error(f"Failed to get agent data from {service_name}: {e}")
        return jsonify({'error': 'Service error'}), 500

@app.route('/api/v1/agents/<service_name>/action', methods=['POST'])
def trigger_agent_action(service_name):
    """Trigger an action for a specific agent"""
    if service_name not in SERVICE_REGISTRY:
        return jsonify({'error': 'Service not found'}), 404
    
    try:
        service_url = SERVICE_REGISTRY[service_name]
        
        # Forward the request to the appropriate service
        response = requests.post(
            f"{service_url}/action",
            json=request.get_json(),
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        # Add request tracking
        result = response.json() if response.status_code == 200 else {'error': 'Service error'}
        result['gateway_request_id'] = str(uuid.uuid4())
        result['gateway_timestamp'] = datetime.now().isoformat()
        result['service'] = service_name
        
        return jsonify(result), response.status_code
        
    except Exception as e:
        logger.error(f"Failed to trigger action on {service_name}: {e}")
        return jsonify({'error': 'Service error', 'details': str(e)}), 500

@app.route('/api/v1/dashboard')
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
            else:
                dashboard_data['services'].append({
                    'name': service_name,
                    'status': 'unhealthy'
                })
                    
        except Exception as e:
            logger.debug(f"Failed to get dashboard data from {service_name}: {e}")
            dashboard_data['services'].append({
                'name': service_name,
                'status': 'unavailable',
                'error': str(e)
            })
    
    return jsonify(dashboard_data)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"🤖 Starting Autonomous Agents API Gateway on port {port}")
    print(f"📊 Dashboard: http://localhost:{port}")
    print(f"🔍 Services: http://localhost:{port}/api/v1/services")
    print(f"🤖 Agents: http://localhost:{port}/api/v1/agents")
    app.run(host='0.0.0.0', port=port, debug=False)