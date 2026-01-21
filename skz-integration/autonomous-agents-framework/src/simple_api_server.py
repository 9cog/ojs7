#!/usr/bin/env python3
"""
Simple API Server for SKZ Agents
Minimal HTTP server for OJS-Python agent communication without external dependencies
"""

import json
import uuid
import hashlib
import hmac
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import sys
import os

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

try:
    from providers.factory import get_ml_engine, get_comm_automation, get_data_sync  # type: ignore
    from providers.migrations import run_guarded_migrations  # type: ignore
except Exception:
    get_ml_engine = None  # type: ignore
    get_comm_automation = None  # type: ignore
    get_data_sync = None  # type: ignore
    run_guarded_migrations = None  # type: ignore

RUNTIME_CONTEXT = {}

try:
    from models.enhanced_agent import EnhancedAgent
    from ojs_bridge import OJSBridge, AgentOJSBridge
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import enhanced agents: {e}")
    AGENTS_AVAILABLE = False

class AgentAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for agent API endpoints"""
    
    def __init__(self, *args, **kwargs):
        if not RUNTIME_CONTEXT:
            try:
                if run_guarded_migrations:
                    run_guarded_migrations(os.getenv("POSTGRES_DSN"))
                RUNTIME_CONTEXT["ml_engine"] = get_ml_engine({}) if get_ml_engine else None
                RUNTIME_CONTEXT["comm"] = get_comm_automation({}) if get_comm_automation else None
                RUNTIME_CONTEXT["data_sync"] = get_data_sync(None) if get_data_sync else None
            except Exception:
                pass

        # Initialize agent system
        self.agents = self._initialize_agents()
        super().__init__(*args, **kwargs)
    
    def _initialize_agents(self):
        """Initialize the 7 autonomous agents"""
        agents = {}
        
        if AGENTS_AVAILABLE:
            agent_configs = [
                ('research_discovery', 'Research Discovery Agent'),
                ('submission_assistant', 'Submission Assistant Agent'),
                ('editorial_orchestration', 'Editorial Orchestration Agent'),
                ('review_coordination', 'Review Coordination Agent'),
                ('content_quality', 'Content Quality Agent'),
                ('publishing_production', 'Publishing Production Agent'),
                ('analytics_monitoring', 'Analytics & Monitoring Agent')
            ]
            
            for agent_id, agent_name in agent_configs:
                try:
                    agent = EnhancedAgent(agent_id, agent_name)
                    agents[agent_id] = {
                        'agent': agent,
                        'name': agent_name,
                        'status': 'active',
                        'last_activity': datetime.now(),
                        'request_count': 0
                    }
                except Exception as e:
                    print(f"Warning: Could not initialize {agent_id}: {e}")
                    # Fallback to mock agent
                    agents[agent_id] = self._create_mock_agent(agent_id, agent_name)
        else:
            # Create mock agents for testing
            agent_configs = [
                ('research_discovery', 'Research Discovery Agent'),
                ('submission_assistant', 'Submission Assistant Agent'),
                ('editorial_orchestration', 'Editorial Orchestration Agent'),
                ('review_coordination', 'Review Coordination Agent'),
                ('content_quality', 'Content Quality Agent'),
                ('publishing_production', 'Publishing Production Agent'),
                ('analytics_monitoring', 'Analytics & Monitoring Agent')
            ]
            
            for agent_id, agent_name in agent_configs:
                agents[agent_id] = self._create_mock_agent(agent_id, agent_name)
        
        return agents
    
    def _create_mock_agent(self, agent_id, agent_name):
        """Create a mock agent for testing purposes"""
        return {
            'agent': None,
            'name': agent_name,
            'status': 'active',
            'last_activity': datetime.now(),
            'request_count': 0,
            'capabilities': self._get_agent_capabilities(agent_id),
            'performance': {
                'success_rate': 0.95,
                'avg_response_time': 2.3,
                'total_actions': 100
            }
        }
    
    def _get_agent_capabilities(self, agent_id):
        """Get capabilities for each agent type"""
        capabilities_map = {
            'research_discovery': ['literature_search', 'gap_analysis', 'trend_identification'],
            'submission_assistant': ['format_checking', 'quality_assessment', 'compliance_validation'],
            'editorial_orchestration': ['workflow_management', 'decision_support', 'deadline_tracking'],
            'review_coordination': ['reviewer_matching', 'review_tracking', 'quality_assessment'],
            'content_quality': ['scientific_validation', 'safety_assessment', 'standards_enforcement'],
            'publishing_production': ['content_formatting', 'visual_generation', 'distribution'],
            'analytics_monitoring': ['performance_analytics', 'trend_forecasting', 'strategic_insights']
        }
        return capabilities_map.get(agent_id, ['general_processing'])
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path_parts = parsed_path.path.strip('/').split('/')
        
        if not path_parts or path_parts[0] == '':
            self._send_response(200, {'status': 'API Server Running', 'timestamp': datetime.now().isoformat()})
            return
        
        if path_parts[0] == 'status':
            self._handle_status_request()
        elif path_parts[0] == 'agents':
            if len(path_parts) == 1:
                self._handle_agents_list()
            elif len(path_parts) == 2:
                self._handle_agent_status(path_parts[1])
            else:
                self._send_error(404, 'Not Found')
        elif path_parts[0] == 'api':
            if len(path_parts) >= 2 and path_parts[1] == 'v1':
                self._handle_api_v1(path_parts[2:])
            else:
                self._send_error(404, 'API version not specified')
        else:
            self._send_error(404, 'Not Found')
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path_parts = parsed_path.path.strip('/').split('/')
        
        # Read request data
        content_length = int(self.headers.get('Content-Length', 0))
        request_data = {}
        
        if content_length > 0:
            try:
                body = self.rfile.read(content_length).decode('utf-8')
                request_data = json.loads(body) if body else {}
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._send_error(400, f'Invalid JSON: {e}')
                return
        
        # Verify authentication if API key is provided
        if not self._verify_authentication(request_data):
            self._send_error(401, 'Authentication failed')
            return
        
        if path_parts[0] == 'api' and len(path_parts) >= 2 and path_parts[1] == 'v1':
            self._handle_api_v1_post(path_parts[2:], request_data)
        elif len(path_parts) >= 2:
            # Direct agent call: /agent_name/action
            agent_name = path_parts[0]
            action = path_parts[1] if len(path_parts) > 1 else 'process'
            self._handle_agent_request(agent_name, action, request_data)
        else:
            self._send_error(404, 'Not Found')
    
    def _verify_authentication(self, request_data):
        """Verify HMAC authentication (simplified for testing)"""
        # For testing purposes, we'll accept any request
        # In production, implement proper HMAC verification
        return True
    
    def _handle_status_request(self):
        """Handle system status request"""
        status = {
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'agents_count': len(self.agents),
            'agents_status': {
                name: info['status'] for name, info in self.agents.items()
            },
            'version': '1.0.0',
            'capabilities': ['agent_communication', 'oj_integration', 'workflow_automation']
        }
        self._send_response(200, status)
    
    def _handle_agents_list(self):
        """Handle agents list request"""
        agents_info = {}
        for agent_id, info in self.agents.items():
            agents_info[agent_id] = {
                'name': info['name'],
                'status': info['status'],
                'last_activity': info['last_activity'].isoformat(),
                'request_count': info['request_count'],
                'capabilities': info.get('capabilities', []),
                'performance': info.get('performance', {})
            }
        
        self._send_response(200, {'agents': agents_info})
    
    def _handle_agent_status(self, agent_id):
        """Handle individual agent status request"""
        if agent_id not in self.agents:
            self._send_error(404, f'Agent {agent_id} not found')
            return
        
        agent_info = self.agents[agent_id]
        status = {
            'agent_id': agent_id,
            'name': agent_info['name'],
            'status': agent_info['status'],
            'last_activity': agent_info['last_activity'].isoformat(),
            'request_count': agent_info['request_count'],
            'capabilities': agent_info.get('capabilities', []),
            'performance': agent_info.get('performance', {})
        }
        
        self._send_response(200, status)
    
    def _handle_api_v1(self, path_parts):
        """Handle API v1 GET requests"""
        if not path_parts:
            self._send_response(200, {
                'api_version': 'v1',
                'endpoints': [
                    '/status',
                    '/agents',
                    '/agents/{agent_id}',
                    '/agents/{agent_id}/{action}'
                ]
            })
        else:
            self._send_error(404, 'Endpoint not found')
    
    def _handle_api_v1_post(self, path_parts, request_data):
        """Handle API v1 POST requests"""
        if not path_parts:
            self._send_error(400, 'No endpoint specified')
            return
        
        if path_parts[0] == 'agents':
            if len(path_parts) >= 3:
                agent_id = path_parts[1]
                action = path_parts[2]
                self._handle_agent_request(agent_id, action, request_data)
            else:
                self._send_error(400, 'Agent and action must be specified')
        else:
            self._send_error(404, 'Endpoint not found')
    
    def _handle_agent_request(self, agent_id, action, request_data):
        """Handle agent processing request"""
        if agent_id not in self.agents:
            self._send_error(404, f'Agent {agent_id} not found')
            return
        
        agent_info = self.agents[agent_id]
        
        # Update agent activity
        agent_info['last_activity'] = datetime.now()
        agent_info['request_count'] += 1
        
        # Process request
        try:
            if agent_info['agent'] and AGENTS_AVAILABLE:
                # Use real agent
                result = self._process_with_real_agent(agent_info['agent'], action, request_data)
            else:
                # Use mock processing
                result = self._process_with_mock_agent(agent_id, action, request_data)
            
            self._send_response(200, {
                'success': True,
                'agent_id': agent_id,
                'action': action,
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'processing_time': 0.5  # Mock processing time
            })
            
        except Exception as e:
            self._send_error(500, f'Agent processing error: {str(e)}')
    
    def _process_with_real_agent(self, agent, action, request_data):
        """Process request with real enhanced agent"""
        # This would use the actual agent's process method
        # For now, return a success response
        return {
            'action_taken': action,
            'data_processed': True,
            'recommendations': ['Use real agent processing'],
            'confidence': 0.95
        }
    
    def _process_with_mock_agent(self, agent_id, action, request_data):
        """Process request with mock agent for testing"""
        # Generate mock responses based on agent type and action
        mock_responses = {
            'research_discovery': {
                'analyze': {
                    'research_gaps': ['gap1', 'gap2'],
                    'trends': ['emerging_trend_1', 'emerging_trend_2'],
                    'innovation_score': 0.85,
                    'market_relevance': 0.78
                },
                'search': {
                    'papers_found': 25,
                    'relevant_papers': 18,
                    'key_insights': ['insight1', 'insight2']
                }
            },
            'submission_assistant': {
                'process': {
                    'quality_score': 0.87,
                    'format_issues': [],
                    'suggestions': ['improve_abstract', 'add_keywords'],
                    'compliance_status': 'passed'
                },
                'validate': {
                    'validation_passed': True,
                    'issues_found': 0,
                    'recommendations': []
                }
            },
            'editorial_orchestration': {
                'coordinate': {
                    'workflow_status': 'in_progress',
                    'next_steps': ['assign_reviewers', 'set_deadlines'],
                    'timeline_estimate': '2_weeks'
                }
            },
            'review_coordination': {
                'coordinate': {
                    'reviewers_matched': 3,
                    'matching_confidence': 0.92,
                    'estimated_completion': '10_days'
                }
            },
            'content_quality': {
                'validate': {
                    'quality_score': 0.89,
                    'safety_assessment': 'safe',
                    'compliance_issues': []
                }
            },
            'publishing_production': {
                'produce': {
                    'formatting_applied': True,
                    'visual_elements_generated': 2,
                    'distribution_ready': True
                }
            },
            'analytics_monitoring': {
                'analyze': {
                    'performance_metrics': {
                        'success_rate': 0.94,
                        'avg_processing_time': 2.1,
                        'total_submissions': 156
                    },
                    'trends': ['increasing_quality', 'faster_processing']
                }
            }
        }
        
        agent_responses = mock_responses.get(agent_id, {})
        default_response = {
            'status': 'processed',
            'action': action,
            'data_received': bool(request_data),
            'timestamp': datetime.now().isoformat()
        }
        
        return agent_responses.get(action, default_response)
    
    def _send_response(self, code, data):
        """Send JSON response"""
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-API-Key, X-Timestamp, X-Signature')
        self.end_headers()
        
        response_json = json.dumps(data, indent=2)
        self.wfile.write(response_json.encode('utf-8'))
    
    def _send_error(self, code, message):
        """Send error response"""
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        error_response = {
            'error': message,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }
        
        response_json = json.dumps(error_response, indent=2)
        self.wfile.write(response_json.encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-API-Key, X-Timestamp, X-Signature')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Custom log message format"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {format % args}")

def run_server(port=5000, host='localhost'):
    """Run the API server"""
    server_address = (host, port)
    httpd = HTTPServer(server_address, AgentAPIHandler)
    
    print(f"""
=== SKZ Agents API Server ===
🚀 Server starting on http://{host}:{port}
📡 API Endpoints:
   GET  /status              - System status
   GET  /agents              - List all agents
   GET  /agents/[agent_id]   - Agent status
   POST /agents/[agent_id]/[action] - Call agent
   POST /api/v1/agents/[agent_id]/[action] - API v1 endpoint

🤖 Available Agents:
   - research_discovery      - Research Discovery Agent
   - submission_assistant    - Submission Assistant Agent  
   - editorial_orchestration - Editorial Orchestration Agent
   - review_coordination     - Review Coordination Agent
   - content_quality         - Content Quality Agent
   - publishing_production   - Publishing Production Agent
   - analytics_monitoring    - Analytics & Monitoring Agent

🔧 Test the API:
   curl http://{host}:{port}/status
   curl http://{host}:{port}/agents
   
Press Ctrl+C to stop the server
""")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        httpd.server_close()

if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    port = 5000
    host = 'localhost'
    
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        host = sys.argv[2]
    
    run_server(port, host)
