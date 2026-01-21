"""
Enhanced Main Application with Real-time Updates
Extends the existing Flask app with WebSocket-based real-time notifications
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import uuid
from datetime import datetime, timedelta
import random
import time
import os
import sys

# Import the real-time notification service
from realtime_notifications import RealtimeNotificationService

app = Flask(__name__)
CORS(app)

# Initialize real-time notifications
realtime_service = RealtimeNotificationService()
realtime_service.init_app(app)

# In-memory storage for demo purposes (inherited from original)
agents_data = {
    'research_discovery': {
        'id': 'agent_research_discovery',
        'name': 'Research Discovery Agent',
        'status': 'active',
        'capabilities': ['literature_search', 'gap_analysis', 'trend_identification'],
        'performance': {'success_rate': 0.95, 'avg_response_time': 2.3, 'total_actions': 156}
    },
    'submission_assistant': {
        'id': 'agent_submission_assistant', 
        'name': 'Submission Assistant Agent',
        'status': 'active',
        'capabilities': ['format_checking', 'venue_recommendation', 'compliance_validation'],
        'performance': {'success_rate': 0.98, 'avg_response_time': 1.8, 'total_actions': 203}
    },
    'editorial_orchestration': {
        'id': 'agent_editorial_orchestration',
        'name': 'Editorial Orchestration Agent', 
        'status': 'active',
        'capabilities': ['workflow_management', 'decision_support', 'deadline_tracking'],
        'performance': {'success_rate': 0.92, 'avg_response_time': 3.1, 'total_actions': 89}
    },
    'review_coordination': {
        'id': 'agent_review_coordination',
        'name': 'Review Coordination Agent',
        'status': 'active', 
        'capabilities': ['reviewer_matching', 'review_tracking', 'quality_assessment'],
        'performance': {'success_rate': 0.88, 'avg_response_time': 4.2, 'total_actions': 134}
    },
    'content_quality': {
        'id': 'agent_content_quality',
        'name': 'Content Quality Agent',
        'status': 'active',
        'capabilities': ['plagiarism_detection', 'scientific_validation', 'formatting_check'],
        'performance': {'success_rate': 0.91, 'avg_response_time': 2.8, 'total_actions': 178}
    },
    'publishing_production': {
        'id': 'agent_publishing_production',
        'name': 'Publishing Production Agent',
        'status': 'active',
        'capabilities': ['layout_formatting', 'metadata_extraction', 'publication_scheduling'],
        'performance': {'success_rate': 0.96, 'avg_response_time': 3.5, 'total_actions': 92}
    },
    'analytics_monitoring': {
        'id': 'agent_analytics_monitoring',
        'name': 'Analytics & Monitoring Agent',
        'status': 'active',
        'capabilities': ['performance_tracking', 'anomaly_detection', 'reporting'],
        'performance': {'success_rate': 0.97, 'avg_response_time': 1.2, 'total_actions': 245}
    }
}

# Workflow state storage
workflow_states = {}
manuscript_updates = []

# Enhanced API Routes

@app.route('/api/v1/agents', methods=['GET'])
def get_agents():
    """Get all agents with real-time status"""
    return jsonify({
        'status': 'success',
        'data': agents_data,
        'timestamp': datetime.now().isoformat(),
        'realtime_enabled': True
    })

@app.route('/api/v1/agents/<agent_type>', methods=['GET'])
def get_agent(agent_type):
    """Get specific agent details"""
    if agent_type in agents_data:
        return jsonify({
            'status': 'success',
            'data': agents_data[agent_type],
            'timestamp': datetime.now().isoformat()
        })
    return jsonify({'status': 'error', 'message': 'Agent not found'}), 404

@app.route('/api/v1/agents/<agent_type>/action', methods=['POST'])
def trigger_agent_action(agent_type):
    """Trigger agent action and notify via WebSocket"""
    if agent_type not in agents_data:
        return jsonify({'status': 'error', 'message': 'Agent not found'}), 404
    
    action_data = request.get_json() or {}
    action_id = str(uuid.uuid4())
    
    # Simulate action processing
    result = {
        'action_id': action_id,
        'agent_type': agent_type,
        'action': action_data.get('action', 'unknown'),
        'status': 'completed',
        'result': f'Action {action_data.get("action", "unknown")} completed successfully',
        'timestamp': datetime.now().isoformat(),
        'execution_time': random.uniform(0.5, 3.0)
    }
    
    # Update agent performance
    agents_data[agent_type]['performance']['total_actions'] += 1
    agents_data[agent_type]['performance']['avg_response_time'] = result['execution_time']
    
    # Notify via WebSocket
    realtime_service.notify_agent_status_change(agent_type, {
        'id': agent_type,
        'status': 'active',
        'last_action': result,
        'performance': agents_data[agent_type]['performance']
    })
    
    return jsonify({
        'status': 'success',
        'data': result
    })

@app.route('/api/v1/workflow/<workflow_id>/update', methods=['POST'])
def update_workflow(workflow_id):
    """Update workflow status and notify clients"""
    update_data = request.get_json() or {}
    
    workflow_update = {
        'workflow_id': workflow_id,
        'stage': update_data.get('stage', 'unknown'),
        'status': update_data.get('status', 'in_progress'),
        'assignee': update_data.get('assignee'),
        'timestamp': datetime.now().isoformat(),
        'details': update_data.get('details', {})
    }
    
    workflow_states[workflow_id] = workflow_update
    
    # Notify via WebSocket
    realtime_service.notify_workflow_event('workflow_update', workflow_update)
    
    return jsonify({
        'status': 'success',
        'data': workflow_update
    })

@app.route('/api/v1/manuscripts/<manuscript_id>/update', methods=['POST'])
def update_manuscript(manuscript_id):
    """Update manuscript status and notify clients"""
    update_data = request.get_json() or {}
    update_type = update_data.get('update_type', 'status_change')
    
    manuscript_update = {
        'manuscript_id': manuscript_id,
        'update_type': update_type,
        'status': update_data.get('status'),
        'message': update_data.get('message', ''),
        'agent_id': update_data.get('agent_id'),
        'timestamp': datetime.now().isoformat(),
        'metadata': update_data.get('metadata', {})
    }
    
    manuscript_updates.append(manuscript_update)
    
    # Keep only last 100 updates
    if len(manuscript_updates) > 100:
        manuscript_updates.pop(0)
    
    # Notify via WebSocket
    realtime_service.notify_manuscript_update(manuscript_id, update_type, manuscript_update)
    
    return jsonify({
        'status': 'success',
        'data': manuscript_update
    })

@app.route('/api/v1/realtime/status', methods=['GET'])
def get_realtime_status():
    """Get real-time service status"""
    return jsonify({
        'status': 'success',
        'data': realtime_service.get_connection_stats(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/v1/notifications/test', methods=['POST'])
def test_notification():
    """Test notification system"""
    notification_data = request.get_json() or {}
    
    if notification_data.get('type') == 'agent_status':
        agent_id = notification_data.get('agent_id', 'research_discovery')
        realtime_service.notify_agent_status_change(agent_id, {
            'id': agent_id,
            'status': 'test_notification',
            'message': 'This is a test notification',
            'timestamp': datetime.now().isoformat()
        })
    
    elif notification_data.get('type') == 'workflow':
        realtime_service.notify_workflow_event('test_event', {
            'message': 'Test workflow notification',
            'timestamp': datetime.now().isoformat()
        })
    
    elif notification_data.get('type') == 'manuscript':
        manuscript_id = notification_data.get('manuscript_id', 'test_manuscript')
        realtime_service.notify_manuscript_update(manuscript_id, 'test_update', {
            'message': 'Test manuscript notification',
            'timestamp': datetime.now().isoformat()
        })
    
    return jsonify({
        'status': 'success',
        'message': 'Test notification sent',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/v1/notifications/from-ojs', methods=['POST'])
def receive_ojs_notification():
    """Receive notifications from OJS system"""
    try:
        notification_data = request.get_json()
        
        if not notification_data:
            return jsonify({'status': 'error', 'message': 'No notification data provided'}), 400
        
        # Process notification based on type
        notification_type = notification_data.get('type')
        
        if notification_type == 'agent_status_change':
            agent_id = notification_data.get('agent_id')
            status = notification_data.get('status', {})
            realtime_service.notify_agent_status_change(agent_id, status)
            
        elif notification_type == 'workflow_event':
            event_type = notification_data.get('event_type', 'ojs_event')
            data = notification_data.get('data', {})
            realtime_service.notify_workflow_event(event_type, data)
            
        elif notification_type == 'manuscript_update':
            manuscript_id = notification_data.get('manuscript_id', 'unknown')
            update_type = notification_data.get('update_type', 'update')
            data = notification_data.get('data', {})
            realtime_service.notify_manuscript_update(manuscript_id, update_type, data)
        
        else:
            # Generic notification handling
            realtime_service.notify_workflow_event('ojs_notification', notification_data)
        
        return jsonify({
            'status': 'success',
            'message': 'OJS notification processed',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Enhanced dashboard with real-time features
@app.route('/')
def dashboard():
    """Enhanced dashboard with real-time capabilities"""
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SKZ Autonomous Agents - Real-time Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: rgba(255,255,255,0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        .connection-status {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 15px;
        }
        .connected {
            background: #e8f5e8;
            color: #2d5a2d;
        }
        .disconnected {
            background: #ffeaea;
            color: #8b0000;
        }
        .real-time-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #4CAF50;
            border-radius: 50%;
            margin-right: 5px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .agents-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .agent-card {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .agent-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        }
        .agent-card.updated {
            animation: highlight 1s ease-in-out;
        }
        @keyframes highlight {
            0% { background: rgba(255,255,255,0.95); }
            50% { background: rgba(255,255,0,0.3); }
            100% { background: rgba(255,255,255,0.95); }
        }
        .agent-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        .agent-icon {
            font-size: 24px;
            margin-right: 12px;
        }
        .agent-name {
            flex: 1;
            font-weight: bold;
            font-size: 16px;
        }
        .agent-status {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-active { background: #e8f5e8; color: #2d5a2d; }
        .status-processing { background: #fff3cd; color: #856404; }
        .status-idle { background: #d1ecf1; color: #0c5460; }
        .status-error { background: #f8d7da; color: #721c24; }
        .performance {
            display: flex;
            justify-content: space-between;
            margin-top: 15px;
        }
        .metric {
            text-align: center;
        }
        .metric-value {
            font-size: 18px;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            font-size: 11px;
            color: #666;
            margin-top: 2px;
        }
        .notifications {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .notification {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            animation: slideIn 0.5s ease-in-out;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-100%); }
            to { opacity: 1; transform: translateX(0); }
        }
        .notification.agent_status { border-left-color: #28a745; background: #e8f5e8; }
        .notification.workflow_event { border-left-color: #17a2b8; background: #d1ecf1; }
        .notification.manuscript_update { border-left-color: #ffc107; background: #fff3cd; }
        .test-buttons {
            margin-top: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s;
        }
        .btn-primary { background: #667eea; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-info { background: #17a2b8; color: white; }
        .btn-warning { background: #ffc107; color: #212529; }
        .btn:hover { opacity: 0.8; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 SKZ Autonomous Agents Framework - Real-time Dashboard</h1>
            <span class="real-time-indicator"></span>
            <span>Live Updates Enabled</span>
            <span id="connectionStatus" class="connection-status disconnected">Disconnected</span>
            <div style="margin-top: 10px; font-size: 14px;">
                Active Connections: <span id="activeConnections">0</span> | 
                Total Agents: <span id="totalAgents">7</span> |
                Last Update: <span id="lastUpdate">Never</span>
            </div>
        </div>
        
        <div class="agents-grid" id="agentsGrid">
            <!-- Agent cards will be populated by JavaScript -->
        </div>
        
        <div class="notifications">
            <h2>📢 Real-time Notifications</h2>
            <div id="notifications"></div>
            
            <div class="test-buttons">
                <button class="btn btn-primary" onclick="testAgentNotification()">Test Agent Status</button>
                <button class="btn btn-info" onclick="testWorkflowNotification()">Test Workflow Event</button>
                <button class="btn btn-warning" onclick="testManuscriptNotification()">Test Manuscript Update</button>
                <button class="btn btn-success" onclick="clearNotifications()">Clear Notifications</button>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();
        let agentData = {};
        
        // Connection event handlers
        socket.on('connect', function() {
            document.getElementById('connectionStatus').textContent = 'Connected';
            document.getElementById('connectionStatus').className = 'connection-status connected';
            addNotification('system', 'Connected to real-time service', 'success');
        });
        
        socket.on('disconnect', function() {
            document.getElementById('connectionStatus').textContent = 'Disconnected';
            document.getElementById('connectionStatus').className = 'connection-status disconnected';
            addNotification('system', 'Disconnected from real-time service', 'error');
        });
        
        socket.on('connection_confirmed', function(data) {
            document.getElementById('activeConnections').textContent = '1';
            document.getElementById('totalAgents').textContent = data.active_agents;
            addNotification('system', `Connection confirmed. Client ID: ${data.client_id}`, 'info');
        });
        
        // Agent status updates
        socket.on('agent_status_update', function(data) {
            if (data.agents) {
                Object.assign(agentData, data.agents);
                updateAgentsDisplay();
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
            }
        });
        
        // Real-time notifications
        socket.on('notification', function(data) {
            addNotification(data.type, JSON.stringify(data, null, 2), data.type);
            
            if (data.type === 'agent_status_change' && data.agent_id) {
                highlightAgentCard(data.agent_id);
            }
        });
        
        // Initialize with current agent data
        fetch('/api/v1/agents')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    agentData = data.data;
                    updateAgentsDisplay();
                }
            });
        
        function updateAgentsDisplay() {
            const grid = document.getElementById('agentsGrid');
            grid.innerHTML = '';
            
            Object.entries(agentData).forEach(([agentType, agent]) => {
                const card = createAgentCard(agentType, agent);
                grid.appendChild(card);
            });
        }
        
        function createAgentCard(agentType, agent) {
            const card = document.createElement('div');
            card.className = 'agent-card';
            card.id = `agent-${agentType}`;
            
            const icons = {
                'research_discovery': '🔍',
                'submission_assistant': '📝',
                'editorial_orchestration': '🎭',
                'review_coordination': '👥',
                'content_quality': '✅',
                'publishing_production': '📚',
                'analytics_monitoring': '📊'
            };
            
            const statusClass = `status-${agent.status || 'active'}`;
            
            card.innerHTML = `
                <div class="agent-header">
                    <div class="agent-icon">${icons[agentType] || '🤖'}</div>
                    <div class="agent-name">${agent.name}</div>
                    <div class="agent-status ${statusClass}">${agent.status || 'Active'}</div>
                </div>
                <div class="performance">
                    <div class="metric">
                        <div class="metric-value">${Math.round((agent.performance?.success_rate || 0.9) * 100)}%</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${(agent.performance?.avg_response_time || 2.0).toFixed(1)}s</div>
                        <div class="metric-label">Avg Response</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${agent.performance?.total_actions || 0}</div>
                        <div class="metric-label">Total Actions</div>
                    </div>
                </div>
            `;
            
            return card;
        }
        
        function highlightAgentCard(agentId) {
            const card = document.getElementById(`agent-${agentId}`);
            if (card) {
                card.classList.add('updated');
                setTimeout(() => card.classList.remove('updated'), 1000);
            }
        }
        
        function addNotification(type, message, level = 'info') {
            const notifications = document.getElementById('notifications');
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            
            const timestamp = new Date().toLocaleTimeString();
            notification.innerHTML = `
                <strong>${type.toUpperCase()}</strong> [${timestamp}]<br>
                <pre style="font-size: 12px; margin-top: 5px; white-space: pre-wrap;">${message}</pre>
            `;
            
            notifications.insertBefore(notification, notifications.firstChild);
            
            // Keep only last 10 notifications
            const notificationElements = notifications.children;
            if (notificationElements.length > 10) {
                notifications.removeChild(notificationElements[notificationElements.length - 1]);
            }
        }
        
        // Test functions
        function testAgentNotification() {
            fetch('/api/v1/notifications/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: 'agent_status', agent_id: 'research_discovery' })
            });
        }
        
        function testWorkflowNotification() {
            fetch('/api/v1/notifications/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: 'workflow' })
            });
        }
        
        function testManuscriptNotification() {
            fetch('/api/v1/notifications/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: 'manuscript', manuscript_id: 'test_manuscript_001' })
            });
        }
        
        function clearNotifications() {
            document.getElementById('notifications').innerHTML = '';
        }
    </script>
</body>
</html>
    ''')

if __name__ == '__main__':
    print("🚀 Starting Enhanced SKZ Autonomous Agents Framework with Real-time Updates...")
    print("📊 Dashboard: http://localhost:5000")
    print("🔌 API: http://localhost:5000/api/v1/")
    print("⚡ Real-time: WebSocket enabled")
    
    realtime_service.socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=True,
        allow_unsafe_werkzeug=True
    )