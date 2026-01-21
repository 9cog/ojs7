# Real-time Updates and Notifications Implementation

## Overview

This implementation adds comprehensive real-time updates and notifications to the SKZ Autonomous Agents Framework, extending the existing OJS notification system with WebSocket-based live updates.

## Features Implemented

### 1. WebSocket Server (Flask-SocketIO)
- **File**: `skz-integration/autonomous-agents-framework/src/main_realtime.py`
- **Service**: `skz-integration/autonomous-agents-framework/src/realtime_notifications.py`

**Capabilities:**
- Real-time agent status broadcasting
- Workflow event notifications
- Manuscript update notifications
- Client connection management
- Room-based subscriptions
- Automatic agent status polling

**API Endpoints:**
- `GET /api/v1/realtime/status` - Connection statistics
- `POST /api/v1/notifications/test` - Test notifications
- `POST /api/v1/notifications/from-ojs` - OJS integration endpoint

### 2. React Dashboard Enhancement
- **Main Component**: `skz-integration/workflow-visualization-dashboard/src/EnhancedApp.jsx`
- **WebSocket Hook**: `skz-integration/workflow-visualization-dashboard/src/hooks/useWebSocket.ts`

**Real-time Components:**
- **RealtimeStatus**: Connection status and statistics
- **RealtimeAgentCard**: Live agent monitoring with subscription support
- **NotificationsPanel**: Real-time notifications display and testing

**Features:**
- Live agent status updates
- CPU and memory usage monitoring  
- Performance metrics tracking
- Subscription-based agent filtering
- Interactive notification testing
- Connection status indicators

### 3. OJS Integration Bridge
- **File**: `lib/pkp/classes/notification/SKZRealtimeNotificationBridge.inc.php`

**Integration Points:**
- Extends existing OJS NotificationDAO
- Maps OJS notification types to real-time events
- Provides JavaScript integration for OJS interface
- Automatic status indicators in OJS navigation
- Browser notification support

**OJS Event Mapping:**
- `NOTIFICATION_TYPE_EDITOR_ASSIGNMENT_SUBMISSION` → `workflow_event`
- `NOTIFICATION_TYPE_EDITOR_DECISION_ACCEPT` → `manuscript_update` + agent status
- `NOTIFICATION_TYPE_REVIEWER_COMMENT` → `workflow_event` + agent status
- `NOTIFICATION_TYPE_SUBMISSION_SUBMITTED` → `manuscript_update` + agent status
- `NOTIFICATION_TYPE_PUBLISHED_ISSUE` → `workflow_event` + agent status

## System Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────────────┐
│   React         │◄─────────────── │  Flask-SocketIO Server  │
│   Dashboard     │                 │  (Port 5000)            │
└─────────────────┘                 └─────────────────────────┘
                                               ▲
                                               │ HTTP
                                               │
┌─────────────────┐    PHP Bridge   ┌─────────┴─────────┐
│   OJS Core      │────────────────►│  Real-time        │
│   System        │    Notifications │  Service API      │
└─────────────────┘                 └───────────────────┘
```

## Real-time Notification Types

### Agent Status Changes
```json
{
  "type": "agent_status_change",
  "agent_id": "research_discovery",
  "status": {
    "status": "active",
    "cpu_usage": 45.2,
    "memory_usage": 62.1,
    "active_tasks": 3,
    "performance": {
      "success_rate": 0.94,
      "avg_response_time": 2.3,
      "total_actions": 156
    }
  },
  "timestamp": "2025-08-07T10:02:09.155193"
}
```

### Workflow Events
```json
{
  "type": "workflow_event", 
  "event_type": "manuscript_submitted",
  "data": {
    "submissionId": 123,
    "title": "Cosmetic Science Research",
    "stage": "submission"
  },
  "timestamp": "2025-08-07T10:02:22.975198"
}
```

### Manuscript Updates
```json
{
  "type": "manuscript_update",
  "manuscript_id": "MS-2024-156", 
  "update_type": "status_change",
  "data": {
    "status": "under_review",
    "assignee": "editor@example.com"
  },
  "timestamp": "2025-08-07T10:02:35.123456"
}
```

## Installation & Setup

### 1. Backend Dependencies
```bash
cd skz-integration/autonomous-agents-framework
pip install -r requirements.txt
```

### 2. Frontend Dependencies  
```bash
cd skz-integration/workflow-visualization-dashboard
npm install --legacy-peer-deps
```

### 3. Start Services

**WebSocket Server:**
```bash
cd skz-integration/autonomous-agents-framework/src
python main_realtime.py
# Server starts on http://localhost:5000
```

**React Dashboard:**
```bash
cd skz-integration/workflow-visualization-dashboard
npm run build
npm run preview
# Dashboard available on http://localhost:4173
```

### 4. OJS Configuration

Add to OJS `config.inc.php`:
```php
[skz]
realtime_service_url = "http://localhost:5000"
realtime_notifications_enabled = true
```

Include in OJS templates:
```php
<?php
global $skzRealtimeBridge;
echo $skzRealtimeBridge->includeRealtimeJavaScript();
?>
```

## Usage Examples

### Testing Real-time Notifications

**Agent Status Update:**
```bash
curl -X POST http://localhost:5000/api/v1/notifications/test \
  -H "Content-Type: application/json" \
  -d '{"type":"agent_status","agent_id":"research_discovery"}'
```

**OJS Workflow Event:**
```bash
curl -X POST http://localhost:5000/api/v1/notifications/from-ojs \
  -H "Content-Type: application/json" \
  -d '{
    "type": "workflow_event",
    "event_type": "manuscript_submitted", 
    "data": {"submissionId": 123, "title": "Test Manuscript"}
  }'
```

### WebSocket Client Integration

```javascript
import { useWebSocket } from './hooks/useWebSocket';

function MyComponent() {
  const {
    connected,
    agentStatuses,
    notifications,
    subscribeToAgent,
    sendTestNotification
  } = useWebSocket('http://localhost:5000');

  return (
    <div>
      <p>Status: {connected ? 'Connected' : 'Disconnected'}</p>
      <button onClick={() => subscribeToAgent('research_discovery')}>
        Subscribe to Research Agent
      </button>
    </div>
  );
}
```

## Performance & Scalability

- **Connection Pooling**: Supports multiple concurrent WebSocket connections
- **Room-based Broadcasting**: Efficient targeted notifications
- **Automatic Cleanup**: Connection management and memory cleanup  
- **Error Handling**: Graceful degradation when service unavailable
- **Backoff Strategy**: Exponential backoff for reconnection attempts

## Security Considerations

- **CORS Configuration**: Configurable allowed origins
- **Rate Limiting**: Built-in request throttling
- **Input Validation**: JSON schema validation for all endpoints
- **Error Handling**: No sensitive data in error responses
- **Connection Limits**: Configurable maximum connections

## Monitoring & Debugging

**Connection Statistics:**
```bash
curl http://localhost:5000/api/v1/realtime/status
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "active_connections": 3,
    "total_notifications_queued": 0, 
    "active_agents": 7,
    "rooms": ["general", "agent_research_discovery"]
  },
  "timestamp": "2025-08-07T10:01:53.779429"
}
```

**WebSocket Events in Browser:**
- Open browser DevTools Console
- Watch for Socket.IO connection logs
- Monitor real-time event messages

## Screenshots

### Real-time Dashboard
![Real-time Dashboard](https://github.com/user-attachments/assets/e715dc7d-d4d4-4e90-bf0b-7fb74bbb7edf)

### Agent Status Monitoring
![Agent Dashboard](https://github.com/user-attachments/assets/2f7cc572-f6dd-4c15-946e-625d6dbc0bac)

### Workflow Simulation
![Workflow Simulation](https://github.com/user-attachments/assets/9726021a-b593-4992-94de-5c49a371704c)

## Integration with Existing System

The implementation seamlessly extends existing infrastructure:

- **OJS Notifications**: Enhances without replacing existing notification system
- **Agent Framework**: Adds real-time layer to existing Flask services
- **React Dashboard**: Extends existing UI with new real-time components
- **Database**: No schema changes required
- **Configuration**: Minimal config additions needed

## Future Enhancements

- **Authentication**: JWT-based WebSocket authentication
- **Persistence**: Redis for notification queue persistence  
- **Clustering**: Multi-instance WebSocket support
- **Analytics**: Real-time performance dashboards
- **Mobile**: React Native components for mobile apps

---

**Implementation Status**: ✅ Complete and Operational
**Testing**: ✅ All endpoints and features tested
**Documentation**: ✅ Complete with examples
**Integration**: ✅ OJS bridge implemented and functional