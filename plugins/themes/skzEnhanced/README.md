# SKZ Enhanced Theme - Agent Interface Integration

## Overview

The SKZ Enhanced Theme provides comprehensive integration of autonomous agent interfaces into the Open Journal Systems (OJS) workflow. This theme extends the default OJS theme with specialized components for displaying and controlling the 7 autonomous agents of the SKZ framework.

## Features

### 🎯 Core Components

1. **Agent Status Bar** - Header-integrated status display for all 7 agents
2. **Workflow Agent Controls** - Interactive control panels in editorial workflow pages
3. **Submission Agent Status** - Sidebar widgets showing agent activity for specific submissions
4. **Real-time Notifications** - WebSocket-based live updates and notifications

### 🤖 The 7 Autonomous Agents

1. **Research Discovery Agent** - INCI database mining and patent analysis
2. **Submission Assistant Agent** - Quality assessment and INCI verification
3. **Editorial Orchestration Agent** - Workflow coordination and decision making
4. **Review Coordination Agent** - Reviewer matching and workload management
5. **Content Quality Agent** - Scientific validation and safety assessment
6. **Publishing Production Agent** - Content formatting and distribution
7. **Analytics & Monitoring Agent** - Performance analytics and optimization

### 🎨 Theme Options

The theme provides three visual styles:

- **Professional** - Clean, corporate styling with blue/gray color scheme
- **Modern** - Contemporary design with gradient colors and rounded corners
- **Minimal** - Simple, clean interface with minimal visual elements

### ⚡ Real-time Features

- **WebSocket Integration** - Live agent status updates
- **Polling Fallback** - Automatic fallback when WebSocket unavailable
- **Progress Tracking** - Real-time workflow progress visualization
- **Agent Recommendations** - Dynamic recommendation system

## Installation

### Prerequisites

- OJS 3.3.0 or later
- SKZ Agents plugin enabled
- Modern web browser with JavaScript enabled

### Theme Activation

1. **Copy Theme Files**
   ```bash
   # Theme files are already in place at:
   # /plugins/themes/skzEnhanced/
   ```

2. **Enable Theme in OJS Admin**
   - Go to Settings > Website > Appearance > Theme
   - Select "SKZ Enhanced Theme" from the dropdown
   - Configure theme options as needed
   - Save settings

3. **Configure Agent Integration**
   - Ensure SKZ Agents plugin is enabled
   - Configure API endpoints in plugin settings
   - Test agent connectivity

## Theme Structure

```
plugins/themes/skzEnhanced/
├── SKZEnhancedThemePlugin.inc.php  # Main theme plugin class
├── version.xml                      # Plugin version information
├── settings.xml                     # Default settings
├── index.php                        # Plugin entry point
├── locale/en_US/locale.xml         # Localization strings
├── styles/                          # CSS/LESS stylesheets
│   ├── skz-agent-interface.less    # Core agent UI styles
│   ├── skz-status-indicators.less  # Status indicators and badges
│   └── skz-workflow-integration.less # Workflow integration styles
├── js/                             # JavaScript functionality
│   ├── skz-agent-ui.js            # Main agent interface controller
│   ├── skz-status-monitor.js      # Real-time monitoring
│   └── skz-workflow-integration.js # Workflow page integration
└── templates/components/           # Template components
    ├── agent-status-bar.tpl       # Header status bar
    ├── workflow-agent-controls.tpl # Workflow controls
    ├── submission-agent-status.tpl # Submission status widget
    └── agent-notifications.tpl    # Notification system
```

## Configuration Options

### Theme Settings

Access via **Settings > Website > Appearance > Theme Options**:

- **Agent Status Bar** - Show/hide the header status bar
- **Interface Theme** - Choose visual style (Professional/Modern/Minimal)
- **Real-time Updates** - Enable/disable live WebSocket updates
- **Workflow Controls** - Show/hide agent controls in workflow pages

### CSS Variables

The theme uses CSS custom properties for easy customization:

```css
:root {
  --skz-agent-primary: #1e40af;      /* Primary brand color */
  --skz-agent-secondary: #6b7280;    /* Secondary text color */
  --skz-agent-success: #059669;      /* Success state color */
  --skz-agent-warning: #d97706;      /* Warning state color */
  --skz-agent-error: #dc2626;        /* Error state color */
  --skz-agent-background: #f8fafc;   /* Background color */
  --skz-agent-border: #e5e7eb;       /* Border color */
  --skz-agent-text: #374151;         /* Text color */
  --skz-agent-border-radius: 6px;    /* Border radius */
  --skz-agent-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); /* Box shadow */
}
```

## Component Documentation

### Agent Status Bar

**Location**: Header area below navigation
**Template**: `templates/components/agent-status-bar.tpl`
**Functionality**:
- Displays system operational status
- Shows individual agent status indicators
- Provides quick access to dashboard
- Responsive design with mobile toggle

**Usage**:
```php
// Automatically inserted via hook:
// Templates::Common::Header::navbar
```

### Workflow Agent Controls

**Location**: Editorial workflow pages
**Template**: `templates/components/workflow-agent-controls.tpl`
**Functionality**:
- Shows active agents for current submission
- Displays AI processing progress
- Provides agent control buttons (view, pause, configure)
- Shows agent recommendations

**Usage**:
```php
// Automatically inserted via hook:
// Templates::Workflow::index
```

### Submission Agent Status

**Location**: Workflow sidebar
**Template**: `templates/components/submission-agent-status.tpl`
**Functionality**:
- Submission-specific agent status
- Workflow progress tracking
- Recent agent activities timeline
- Action buttons for refresh and reports

### Real-time Notifications

**Location**: Fixed position, top-right
**Template**: `templates/components/agent-notifications.tpl`
**Functionality**:
- WebSocket-based live notifications
- Agent status change alerts
- Workflow progress updates
- Automatic notification management

## JavaScript API

### SKZ.AgentUI

Main agent interface controller:

```javascript
// Initialize agent UI
SKZ.AgentUI.init();

// Load agent data
SKZ.AgentUI.loadAgentData();

// Select specific agent
SKZ.AgentUI.selectAgent('research_discovery');

// Check connection status
var isConnected = SKZ.AgentUI.isConnected();
```

### SKZ.StatusMonitor

Real-time monitoring:

```javascript
// Initialize status monitoring
SKZ.StatusMonitor.init();

// Check WebSocket connection
var connected = SKZ.StatusMonitor.isConnected();

// Get performance metrics
var metrics = SKZ.StatusMonitor.getMetrics();
```

### SKZ.WorkflowIntegration

Workflow integration:

```javascript
// Initialize workflow integration
SKZ.WorkflowIntegration.init();

// Get current submission ID
var submissionId = SKZ.WorkflowIntegration.getCurrentSubmissionId();

// Get workflow stage
var stage = SKZ.WorkflowIntegration.getWorkflowStage();
```

## Custom Events

The theme triggers several custom events for integration:

```javascript
// Agent data updated
$(document).on('skz:agentDataUpdated', function(event, data) {
    console.log('Agent data updated:', data);
});

// Agent selected
$(document).on('skz:agentSelected', function(event, agentId) {
    console.log('Agent selected:', agentId);
});

// Agent status changed
$(document).on('skz:agentStatusChanged', function(event, agentId, status, data) {
    console.log('Agent status changed:', agentId, status);
});

// Workflow progress updated
$(document).on('skz:workflowProgressUpdated', function(event, submissionId, progress, stage) {
    console.log('Workflow progress:', progress, '%');
});

// Connection status events
$(document).on('skz:connectionEstablished', function() {
    console.log('WebSocket connected');
});

$(document).on('skz:connectionLost', function() {
    console.log('WebSocket disconnected');
});
```

## Responsive Design

The theme is fully responsive and adapts to different screen sizes:

### Desktop (>1024px)
- Full agent status bar in header
- Expanded workflow controls
- Sidebar status widgets
- Full notification system

### Tablet (768px - 1024px)
- Condensed status bar
- Collapsible workflow controls
- Responsive agent grid
- Adapted notifications

### Mobile (<768px)
- Collapsible status bar with toggle button
- Single-column agent layout
- Touch-optimized controls
- Full-width notifications

## Accessibility

The theme follows WCAG 2.1 guidelines:

- **Keyboard Navigation** - All controls accessible via keyboard
- **Screen Reader Support** - Proper ARIA labels and descriptions
- **Color Contrast** - High contrast ratios for text and backgrounds
- **Focus Indicators** - Clear focus indicators for all interactive elements

## Performance Considerations

### Optimization Features

- **Lazy Loading** - Components load only when needed
- **Efficient Updates** - Minimal DOM manipulation
- **Caching** - Intelligent caching of agent data
- **Debounced Events** - Rate-limited API calls

### Resource Usage

- **JavaScript Bundle** - ~80KB minified
- **CSS Bundle** - ~25KB minified
- **WebSocket** - Minimal bandwidth usage
- **API Calls** - Optimized polling frequency

## Troubleshooting

### Common Issues

1. **Agent Status Not Loading**
   - Check SKZ Agents plugin is enabled
   - Verify API endpoints are configured correctly
   - Check browser console for JavaScript errors

2. **WebSocket Connection Failed**
   - Ensure WebSocket server is running on port 5000
   - Check firewall and proxy settings
   - Theme will fallback to polling automatically

3. **Theme Not Appearing**
   - Verify theme files are in correct directory
   - Check OJS cache is cleared
   - Ensure theme is selected in admin settings

### Debug Mode

Enable debug logging:

```javascript
// In browser console
window.SKZ_DEBUG = true;
```

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Security

The theme implements several security measures:

- **CSRF Protection** - All API calls include CSRF tokens
- **Input Validation** - Client-side input validation
- **XSS Prevention** - Proper output escaping
- **Secure WebSocket** - WSS in production environments

## License

This theme is distributed under the GNU General Public License v3, consistent with OJS licensing.

## Support

For technical support and issues:

1. Check this documentation
2. Review browser console for errors
3. Check OJS error logs
4. Submit issues to the project repository

## Version History

### v1.0.0 (2024-08-05)
- Initial release
- Complete agent interface integration
- Real-time monitoring system
- Responsive design implementation
- Accessibility compliance