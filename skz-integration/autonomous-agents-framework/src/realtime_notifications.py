"""
Real-time Notifications Service
Provides WebSocket-based real-time updates for the SKZ Autonomous Agents Framework
"""

from flask_socketio import SocketIO, emit, join_room, leave_room
from flask import request
import json
from datetime import datetime
from typing import Dict, List, Any
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeNotificationService:
    def __init__(self, app=None, cors_allowed_origins="*"):
        self.socketio = None
        self.active_connections = {}
        self.agent_status_cache = {}
        self.notification_queue = []
        self.is_running = False
        
        if app:
            self.init_app(app, cors_allowed_origins)
    
    def init_app(self, app, cors_allowed_origins="*"):
        """Initialize SocketIO with the Flask app"""
        self.socketio = SocketIO(
            app, 
            cors_allowed_origins=cors_allowed_origins,
            async_mode='eventlet',
            logger=True,
            engineio_logger=True
        )
        self.setup_event_handlers()
        self.start_background_tasks()
    
    def setup_event_handlers(self):
        """Set up WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect(auth):
            """Handle client connection"""
            client_id = request.sid
            self.active_connections[client_id] = {
                'connected_at': datetime.now().isoformat(),
                'rooms': ['general'],
                'user_agent': request.headers.get('User-Agent', 'Unknown')
            }
            
            # Join general room for broadcast messages
            join_room('general')
            
            logger.info(f"Client {client_id} connected. Total active: {len(self.active_connections)}")
            
            # Send current agent status to new client
            emit('agent_status_update', self.get_current_agent_status())
            
            # Send connection confirmation
            emit('connection_confirmed', {
                'client_id': client_id,
                'timestamp': datetime.now().isoformat(),
                'active_agents': len(self.agent_status_cache)
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            client_id = request.sid
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            
            logger.info(f"Client {client_id} disconnected. Total active: {len(self.active_connections)}")
        
        @self.socketio.on('subscribe_agent')
        def handle_agent_subscription(data):
            """Handle agent-specific subscription"""
            client_id = request.sid
            agent_id = data.get('agent_id')
            
            if agent_id:
                room_name = f"agent_{agent_id}"
                join_room(room_name)
                
                if client_id in self.active_connections:
                    if room_name not in self.active_connections[client_id]['rooms']:
                        self.active_connections[client_id]['rooms'].append(room_name)
                
                emit('subscription_confirmed', {
                    'agent_id': agent_id,
                    'room': room_name,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Client {client_id} subscribed to agent {agent_id}")
        
        @self.socketio.on('unsubscribe_agent')
        def handle_agent_unsubscription(data):
            """Handle agent-specific unsubscription"""
            client_id = request.sid
            agent_id = data.get('agent_id')
            
            if agent_id:
                room_name = f"agent_{agent_id}"
                leave_room(room_name)
                
                if client_id in self.active_connections:
                    if room_name in self.active_connections[client_id]['rooms']:
                        self.active_connections[client_id]['rooms'].remove(room_name)
                
                emit('unsubscription_confirmed', {
                    'agent_id': agent_id,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Client {client_id} unsubscribed from agent {agent_id}")
        
        @self.socketio.on('ping')
        def handle_ping():
            """Handle ping for connection testing"""
            emit('pong', {'timestamp': datetime.now().isoformat()})
    
    def start_background_tasks(self):
        """Start background tasks for real-time updates"""
        if not self.is_running:
            self.is_running = True
            self.socketio.start_background_task(self.periodic_updates)
            self.socketio.start_background_task(self.process_notification_queue)
    
    def periodic_updates(self):
        """Periodic task to send agent status updates"""
        while self.is_running:
            try:
                # Simulate agent status updates (replace with actual agent polling)
                updated_status = self.poll_agent_statuses()
                
                if updated_status:
                    self.socketio.emit('agent_status_update', updated_status, room='general')
                
                self.socketio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error in periodic updates: {e}")
                self.socketio.sleep(10)
    
    def process_notification_queue(self):
        """Process queued notifications"""
        while self.is_running:
            try:
                if self.notification_queue:
                    notification = self.notification_queue.pop(0)
                    self.send_notification(notification)
                
                self.socketio.sleep(1)  # Check queue every second
            except Exception as e:
                logger.error(f"Error processing notification queue: {e}")
                self.socketio.sleep(5)
    
    def poll_agent_statuses(self) -> Dict[str, Any]:
        """Poll current agent statuses (simulate for now)"""
        import random
        
        agents = [
            'research_discovery',
            'submission_assistant', 
            'editorial_orchestration',
            'review_coordination',
            'content_quality',
            'publishing_production',
            'analytics_monitoring'
        ]
        
        updated_agents = {}
        
        for agent_id in agents:
            # Simulate status changes
            if random.random() < 0.3:  # 30% chance of status update
                status = {
                    'id': agent_id,
                    'status': random.choice(['active', 'processing', 'idle']),
                    'cpu_usage': random.uniform(10, 90),
                    'memory_usage': random.uniform(20, 80),
                    'active_tasks': random.randint(0, 5),
                    'last_action': datetime.now().isoformat(),
                    'performance': {
                        'success_rate': random.uniform(0.85, 0.99),
                        'avg_response_time': random.uniform(0.5, 3.0),
                        'total_actions': random.randint(50, 300)
                    }
                }
                updated_agents[agent_id] = status
                self.agent_status_cache[agent_id] = status
        
        return updated_agents if updated_agents else None
    
    def get_current_agent_status(self) -> Dict[str, Any]:
        """Get current cached agent statuses"""
        if not self.agent_status_cache:
            # Initialize with default statuses
            self.poll_agent_statuses()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'agents': self.agent_status_cache,
            'total_agents': len(self.agent_status_cache)
        }
    
    def notify_agent_status_change(self, agent_id: str, status: Dict[str, Any]):
        """Notify clients of agent status change"""
        self.agent_status_cache[agent_id] = status
        
        notification = {
            'type': 'agent_status_change',
            'agent_id': agent_id,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        self.queue_notification(notification)
    
    def notify_workflow_event(self, event_type: str, data: Dict[str, Any]):
        """Notify clients of workflow events"""
        notification = {
            'type': 'workflow_event',
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        self.queue_notification(notification)
    
    def notify_manuscript_update(self, manuscript_id: str, update_type: str, data: Dict[str, Any]):
        """Notify clients of manuscript updates"""
        notification = {
            'type': 'manuscript_update',
            'manuscript_id': manuscript_id,
            'update_type': update_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        self.queue_notification(notification)
    
    def queue_notification(self, notification: Dict[str, Any]):
        """Add notification to processing queue"""
        self.notification_queue.append(notification)
    
    def send_notification(self, notification: Dict[str, Any]):
        """Send notification to appropriate clients"""
        notification_type = notification.get('type')
        
        if notification_type == 'agent_status_change':
            agent_id = notification.get('agent_id')
            # Send to general room and agent-specific room
            self.socketio.emit('notification', notification, room='general')
            self.socketio.emit('notification', notification, room=f'agent_{agent_id}')
        
        elif notification_type in ['workflow_event', 'manuscript_update']:
            # Send to general room
            self.socketio.emit('notification', notification, room='general')
        
        logger.info(f"Sent notification: {notification_type}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics"""
        return {
            'active_connections': len(self.active_connections),
            'total_notifications_queued': len(self.notification_queue),
            'active_agents': len(self.agent_status_cache),
            'rooms': list(set(
                room for conn in self.active_connections.values() 
                for room in conn.get('rooms', [])
            ))
        }

# Global instance
realtime_service = RealtimeNotificationService()