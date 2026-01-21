"""
Base Agent Class for Microservices
Provides common functionality for all agent services
Enhanced with authentication and authorization
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import time
import uuid
import os
import sys
from datetime import datetime
from abc import ABC, abstractmethod

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

try:
    from auth_service import (
        auth_service, 
        authorization_service, 
        require_auth, 
        require_permission,
        get_current_user
    )
    AUTH_AVAILABLE = True
except ImportError:
    # Fallback if auth service is not available
    AUTH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Authentication service not available - running without auth")
    
    def require_auth(f):
        return f
    
    def require_permission(permission):
        def decorator(f):
            return f
        return decorator
    
    def get_current_user():
        return None

class BaseAgent(ABC):
    """Base class for all agent microservices"""
    
    def __init__(self, agent_name, agent_type, port=5000):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(agent_name)
        
        # Initialize metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'start_time': datetime.now().isoformat()
        }
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup common routes for all agents"""
        
        @self.app.route('/health')
        def health_check():
            return jsonify({
                'status': 'healthy',
                'service': self.agent_name,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'auth_enabled': AUTH_AVAILABLE
            })
        
        @self.app.route('/agent')
        def get_agent_info():
            return jsonify(self.get_agent_data())
        
        @self.app.route('/action', methods=['POST'])
        def trigger_action():
            start_time = time.time()
            self.metrics['total_requests'] += 1
            
            try:
                data = request.get_json() or {}
                
                # Extract authentication context from request
                auth_context = data.pop('_auth', None)
                
                # Set authentication context for the agent
                if auth_context:
                    request.auth_context = auth_context
                
                result = self.process_action(data)
                
                # Calculate response time
                response_time = time.time() - start_time
                self._update_metrics(response_time, success=True)
                
                return jsonify({
                    'action_id': str(uuid.uuid4()),
                    'agent_type': self.agent_type,
                    'success': True,
                    'processing_time': response_time,
                    'timestamp': datetime.now().isoformat(),
                    'result': result,
                    'auth_user': auth_context.get('username') if auth_context else None
                })
                
            except Exception as e:
                response_time = time.time() - start_time
                self._update_metrics(response_time, success=False)
                
                self.logger.error(f"Action processing failed: {e}")
                return jsonify({
                    'action_id': str(uuid.uuid4()),
                    'agent_type': self.agent_type,
                    'success': False,
                    'processing_time': response_time,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }), 500
        
        @self.app.route('/metrics')
        def get_metrics():
            return jsonify(self.metrics)
    
    def _update_metrics(self, response_time, success=True):
        """Update service metrics"""
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Update average response time
        total_requests = self.metrics['total_requests']
        current_avg = self.metrics['avg_response_time']
        self.metrics['avg_response_time'] = (current_avg * (total_requests - 1) + response_time) / total_requests
    
    @abstractmethod
    def get_agent_data(self):
        """Return agent-specific data - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def process_action(self, data):
        """Process agent-specific actions - must be implemented by subclasses"""
        pass
    
    def get_auth_context(self):
        """Get authentication context from current request"""
        return getattr(request, 'auth_context', None)
    
    def is_authorized(self, permission: str) -> bool:
        """Check if current user has required permission"""
        if not AUTH_AVAILABLE:
            return True  # Allow all operations if auth is disabled
        
        auth_context = self.get_auth_context()
        if not auth_context:
            return False
        
        user_roles = auth_context.get('roles', [])
        return authorization_service.check_permission(user_roles, permission)
    
    def require_permission_check(self, permission: str):
        """Check permission and raise exception if not authorized"""
        if not self.is_authorized(permission):
            raise PermissionError(f"Permission required: {permission}")
    
    def run(self):
        """Start the agent service"""
        self.logger.info(f"Starting {self.agent_name} on port {self.port}")
        self.logger.info(f"Authentication enabled: {AUTH_AVAILABLE}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False)