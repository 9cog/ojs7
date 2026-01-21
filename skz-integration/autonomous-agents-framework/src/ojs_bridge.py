"""
OJS Bridge for Autonomous Agents
Phase 2 Critical Component - Enables communication between OJS and Python agents
Enhanced with JWT authentication and secure API communication
"""

import json
import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
from dataclasses import dataclass
import hashlib
import hmac
import os
import sys

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'microservices', 'shared'))

try:
    from auth_service import auth_service
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OJSRequest:
    """Represents a request to OJS"""
    endpoint: str
    method: str
    data: Dict[str, Any]
    headers: Dict[str, str]
    timestamp: datetime

@dataclass
class OJSResponse:
    """Represents a response from OJS"""
    status_code: int
    data: Dict[str, Any]
    headers: Dict[str, str]
    timestamp: datetime

class OJSBridge:
    """
    Bridge for communication between autonomous agents and OJS
    Handles authentication, data synchronization, and API communication
    Enhanced with JWT token management
    """
    
    def __init__(self, ojs_base_url: str, api_key: str, secret_key: str, jwt_token: Optional[str] = None):
        self.ojs_base_url = ojs_base_url.rstrip('/')
        self.api_key = api_key
        self.secret_key = secret_key
        self.jwt_token = jwt_token
        self.session = requests.Session()
        self.lock = threading.RLock()
        
        # Configure session
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SKZ-Agent-Bridge/1.0'
        })
        
        # Add JWT token if available
        if self.jwt_token:
            self.session.headers.update({
                'Authorization': f'Bearer {self.jwt_token}'
            })
        
        # Request tracking
        self.request_history = []
        self.response_cache = {}
        
        logger.info(f"Initialized OJS bridge for {ojs_base_url} (Auth: {bool(jwt_token)})")
    
    def set_jwt_token(self, token: str):
        """Set JWT token for authenticated requests"""
        self.jwt_token = token
        self.session.headers.update({
            'Authorization': f'Bearer {token}'
        })
        logger.info("JWT token updated for OJS bridge")
    
    def authenticate_with_ojs(self, username: str, password: str) -> bool:
        """Authenticate with OJS and get JWT token"""
        try:
            auth_url = f"{self.ojs_base_url}/api/v1/auth/login"
            response = requests.post(auth_url, json={
                'username': username,
                'password': password
            })
            
            if response.status_code == 200:
                auth_data = response.json()
                self.set_jwt_token(auth_data['token'])
                logger.info(f"Successfully authenticated with OJS as {username}")
                return True
            else:
                logger.error(f"OJS authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"OJS authentication error: {e}")
            return False
    
    def _generate_signature(self, data: str, timestamp: str) -> str:
        """Generate HMAC signature for authentication"""
        message = f"{timestamp}:{data}"
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_request(self, endpoint: str, method: str = 'GET', 
                     data: Optional[Dict[str, Any]] = None, 
                     headers: Optional[Dict[str, str]] = None) -> OJSResponse:
        """Make authenticated request to OJS"""
        with self.lock:
            url = f"{self.ojs_base_url}{endpoint}"
            timestamp = str(int(datetime.now().timestamp()))
            
            # Prepare request data
            request_data = data or {}
            data_str = json.dumps(request_data, sort_keys=True)
            
            # Generate signature
            signature = self._generate_signature(data_str, timestamp)
            
            # Prepare headers
            request_headers = {
                'X-API-Key': self.api_key,
                'X-Timestamp': timestamp,
                'X-Signature': signature,
                **(headers or {})
            }
            
            # Add JWT token if available
            if self.jwt_token:
                request_headers['Authorization'] = f'Bearer {self.jwt_token}'
            
            # Create request
            request = OJSRequest(
                endpoint=endpoint,
                method=method,
                data=request_data,
                headers=request_headers,
                timestamp=datetime.now()
            )
            
            try:
                # Make request
                if method.upper() == 'GET':
                    response = self.session.get(url, headers=request_headers, params=request_data)
                elif method.upper() == 'POST':
                    response = self.session.post(url, headers=request_headers, json=request_data)
                elif method.upper() == 'PUT':
                    response = self.session.put(url, headers=request_headers, json=request_data)
                elif method.upper() == 'DELETE':
                    response = self.session.delete(url, headers=request_headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Parse response
                response_data = {}
                try:
                    response_data = response.json()
                except json.JSONDecodeError:
                    response_data = {'raw_content': response.text}
                
                ojs_response = OJSResponse(
                    status_code=response.status_code,
                    data=response_data,
                    headers=dict(response.headers),
                    timestamp=datetime.now()
                )
                
                # Store in history
                self.request_history.append({
                    'request': request,
                    'response': ojs_response
                })
                
                # Keep only recent history
                if len(self.request_history) > 100:
                    self.request_history = self.request_history[-50:]
                
                logger.info(f"OJS request to {endpoint}: {response.status_code}")
                return ojs_response
                
            except requests.RequestException as e:
                logger.error(f"OJS request failed: {str(e)}")
                return OJSResponse(
                    status_code=500,
                    data={'error': str(e)},
                    headers={},
                    timestamp=datetime.now()
                )
    
    def get_manuscripts(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get manuscripts from OJS"""
        endpoint = '/api/v1/submissions'
        response = self._make_request(endpoint, 'GET', data=filters or {})
        
        if response.status_code == 200:
            return response.data.get('submissions', [])
        else:
            logger.error(f"Failed to get manuscripts: {response.data}")
            return []
    
    def get_manuscript(self, submission_id: str) -> Optional[Dict[str, Any]]:
        """Get specific manuscript from OJS"""
        endpoint = f'/api/v1/submissions/{submission_id}'
        response = self._make_request(endpoint, 'GET')
        
        if response.status_code == 200:
            return response.data
        else:
            logger.error(f"Failed to get manuscript {submission_id}: {response.data}")
            return None
    
    def update_manuscript(self, submission_id: str, updates: Dict[str, Any]) -> bool:
        """Update manuscript in OJS"""
        endpoint = f'/api/v1/submissions/{submission_id}'
        response = self._make_request(endpoint, 'PUT', data=updates)
        
        success = response.status_code in [200, 201]
        if not success:
            logger.error(f"Failed to update manuscript {submission_id}: {response.data}")
        
        return success
    
    def get_reviewers(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get reviewers from OJS"""
        endpoint = '/api/v1/reviewers'
        response = self._make_request(endpoint, 'GET', data=filters or {})
        
        if response.status_code == 200:
            return response.data.get('reviewers', [])
        else:
            logger.error(f"Failed to get reviewers: {response.data}")
            return []
    
    def assign_reviewer(self, submission_id: str, reviewer_id: str, 
                       assignment_data: Optional[Dict[str, Any]] = None) -> bool:
        """Assign reviewer to manuscript"""
        endpoint = f'/api/v1/submissions/{submission_id}/reviewers'
        data = {
            'reviewer_id': reviewer_id,
            **(assignment_data or {})
        }
        response = self._make_request(endpoint, 'POST', data=data)
        
        success = response.status_code in [200, 201]
        if not success:
            logger.error(f"Failed to assign reviewer: {response.data}")
        
        return success
    
    def get_editorial_decisions(self, submission_id: str) -> List[Dict[str, Any]]:
        """Get editorial decisions for a manuscript"""
        endpoint = f'/api/v1/submissions/{submission_id}/decisions'
        response = self._make_request(endpoint, 'GET')
        
        if response.status_code == 200:
            return response.data.get('decisions', [])
        else:
            logger.error(f"Failed to get editorial decisions: {response.data}")
            return []
    
    def create_editorial_decision(self, submission_id: str, decision_data: Dict[str, Any]) -> bool:
        """Create editorial decision"""
        endpoint = f'/api/v1/submissions/{submission_id}/decisions'
        response = self._make_request(endpoint, 'POST', data=decision_data)
        
        success = response.status_code in [200, 201]
        if not success:
            logger.error(f"Failed to create editorial decision: {response.data}")
        
        return success
    
    def get_publication_data(self, submission_id: str) -> Optional[Dict[str, Any]]:
        """Get publication data for a manuscript"""
        endpoint = f'/api/v1/submissions/{submission_id}/publication'
        response = self._make_request(endpoint, 'GET')
        
        if response.status_code == 200:
            return response.data
        else:
            logger.error(f"Failed to get publication data: {response.data}")
            return None
    
    def update_publication_data(self, submission_id: str, publication_data: Dict[str, Any]) -> bool:
        """Update publication data"""
        endpoint = f'/api/v1/submissions/{submission_id}/publication'
        response = self._make_request(endpoint, 'PUT', data=publication_data)
        
        success = response.status_code in [200, 201]
        if not success:
            logger.error(f"Failed to update publication data: {response.data}")
        
        return success
    
    def get_analytics_data(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get analytics data from OJS"""
        endpoint = '/api/v1/analytics'
        response = self._make_request(endpoint, 'GET', data=filters or {})
        
        if response.status_code == 200:
            return response.data
        else:
            logger.error(f"Failed to get analytics data: {response.data}")
            return {}
    
    def send_agent_result(self, agent_id: str, result_data: Dict[str, Any]) -> bool:
        """Send agent result back to OJS"""
        endpoint = '/api/v1/agent-results'
        data = {
            'agent_id': agent_id,
            'result': result_data,
            'timestamp': datetime.now().isoformat()
        }
        response = self._make_request(endpoint, 'POST', data=data)
        
        success = response.status_code in [200, 201]
        if not success:
            logger.error(f"Failed to send agent result: {response.data}")
        
        return success
    
    def register_webhook(self, event_type: str, callback_url: str) -> bool:
        """Register webhook for OJS events"""
        endpoint = '/api/v1/webhooks'
        data = {
            'event_type': event_type,
            'callback_url': callback_url,
            'active': True
        }
        response = self._make_request(endpoint, 'POST', data=data)
        
        success = response.status_code in [200, 201]
        if not success:
            logger.error(f"Failed to register webhook: {response.data}")
        
        return success
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get OJS system status"""
        endpoint = '/api/v1/status'
        response = self._make_request(endpoint, 'GET')
        
        if response.status_code == 200:
            return response.data
        else:
            logger.error(f"Failed to get system status: {response.data}")
            return {'status': 'unknown', 'error': response.data.get('error', 'Unknown error')}
    
    def get_request_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent request history"""
        with self.lock:
            history = []
            for entry in self.request_history[-limit:]:
                history.append({
                    'endpoint': entry['request'].endpoint,
                    'method': entry['request'].method,
                    'status_code': entry['response'].status_code,
                    'timestamp': entry['request'].timestamp.isoformat(),
                    'success': entry['response'].status_code < 400
                })
            return history
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        with self.lock:
            total_requests = len(self.request_history)
            successful_requests = sum(1 for entry in self.request_history 
                                   if entry['response'].status_code < 400)
            
            return {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'success_rate': successful_requests / total_requests if total_requests > 0 else 0.0,
                'last_request': self.request_history[-1]['request'].timestamp.isoformat() if self.request_history else None,
                'ojs_base_url': self.ojs_base_url
            }

class AgentOJSBridge:
    """
    Higher-level bridge for agent-OJS communication
    Provides agent-specific methods and data synchronization
    """
    
    def __init__(self, agent_id: str, ojs_bridge: OJSBridge):
        self.agent_id = agent_id
        self.ojs_bridge = ojs_bridge
        self.sync_interval = 300  # 5 minutes
        self.last_sync = datetime.now()
        
        logger.info(f"Initialized agent OJS bridge for {agent_id}")
    
    def sync_manuscript_data(self, submission_id: str) -> Dict[str, Any]:
        """Synchronize manuscript data between agent and OJS"""
        # Get current manuscript data from OJS
        manuscript = self.ojs_bridge.get_manuscript(submission_id)
        if not manuscript:
            return {'success': False, 'error': 'Manuscript not found'}
        
        # Get agent's analysis of the manuscript
        agent_analysis = self._analyze_manuscript(manuscript)
        
        # Send agent results back to OJS
        success = self.ojs_bridge.send_agent_result(self.agent_id, {
            'submission_id': submission_id,
            'analysis': agent_analysis,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'success': success,
            'manuscript': manuscript,
            'agent_analysis': agent_analysis
        }
    
    def _analyze_manuscript(self, manuscript: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze manuscript based on agent type"""
        analysis = {
            'agent_id': self.agent_id,
            'analysis_type': 'manuscript_review',
            'timestamp': datetime.now().isoformat(),
            'findings': {}
        }
        
        # Agent-specific analysis
        if 'research' in self.agent_id.lower():
            analysis['findings'] = self._research_analysis(manuscript)
        elif 'quality' in self.agent_id.lower():
            analysis['findings'] = self._quality_analysis(manuscript)
        elif 'coordination' in self.agent_id.lower():
            analysis['findings'] = self._coordination_analysis(manuscript)
        else:
            analysis['findings'] = self._general_analysis(manuscript)
        
        return analysis
    
    def _research_analysis(self, manuscript: Dict[str, Any]) -> Dict[str, Any]:
        """Research discovery analysis"""
        return {
            'research_gaps': ['gap1', 'gap2'],
            'trends': ['trend1', 'trend2'],
            'innovation_score': 0.75,
            'market_relevance': 0.8
        }
    
    def _quality_analysis(self, manuscript: Dict[str, Any]) -> Dict[str, Any]:
        """Quality assessment analysis"""
        return {
            'quality_score': 0.85,
            'compliance_issues': [],
            'improvement_suggestions': ['suggestion1', 'suggestion2'],
            'safety_assessment': 'safe'
        }
    
    def _coordination_analysis(self, manuscript: Dict[str, Any]) -> Dict[str, Any]:
        """Coordination analysis"""
        return {
            'workflow_status': 'in_progress',
            'next_steps': ['step1', 'step2'],
            'timeline_estimate': '2_weeks',
            'resource_requirements': ['reviewer1', 'reviewer2']
        }
    
    def _general_analysis(self, manuscript: Dict[str, Any]) -> Dict[str, Any]:
        """General manuscript analysis"""
        return {
            'overall_score': 0.7,
            'key_findings': ['finding1', 'finding2'],
            'recommendations': ['rec1', 'rec2']
        }
    
    def process_ojs_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process OJS event and return agent response"""
        event_type = event_data.get('event_type', 'unknown')
        
        if event_type == 'submission_created':
            return self._handle_submission_created(event_data)
        elif event_type == 'review_assigned':
            return self._handle_review_assigned(event_data)
        elif event_type == 'decision_made':
            return self._handle_decision_made(event_data)
        else:
            return {'success': False, 'error': f'Unknown event type: {event_type}'}
    
    def _handle_submission_created(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle submission created event"""
        submission_id = event_data.get('submission_id')
        if not submission_id:
            return {'success': False, 'error': 'No submission ID provided'}
        
        # Sync manuscript data
        sync_result = self.sync_manuscript_data(submission_id)
        
        return {
            'success': sync_result['success'],
            'action_taken': 'manuscript_analysis',
            'submission_id': submission_id,
            'analysis': sync_result.get('agent_analysis', {})
        }
    
    def _handle_review_assigned(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle review assigned event"""
        submission_id = event_data.get('submission_id')
        reviewer_id = event_data.get('reviewer_id')
        
        return {
            'success': True,
            'action_taken': 'review_coordination',
            'submission_id': submission_id,
            'reviewer_id': reviewer_id,
            'coordination_plan': {
                'timeline': '2_weeks',
                'milestones': ['milestone1', 'milestone2']
            }
        }
    
    def _handle_decision_made(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle decision made event"""
        submission_id = event_data.get('submission_id')
        decision = event_data.get('decision')
        
        return {
            'success': True,
            'action_taken': 'decision_processing',
            'submission_id': submission_id,
            'decision': decision,
            'next_steps': ['step1', 'step2']
        }
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get bridge status and statistics"""
        return {
            'agent_id': self.agent_id,
            'ojs_connection': self.ojs_bridge.get_connection_stats(),
            'last_sync': self.last_sync.isoformat(),
            'sync_interval': self.sync_interval,
            'status': 'active'
        }
