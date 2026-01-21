"""
Base Agent Class for Microservices
Provides common functionality for all agent services
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import time
import uuid
from datetime import datetime
import os
from abc import ABC, abstractmethod

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
                'version': '1.0.0'
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
                    'result': result
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
    
    def run(self):
        """Start the agent service"""
        self.logger.info(f"Starting {self.agent_name} on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False)