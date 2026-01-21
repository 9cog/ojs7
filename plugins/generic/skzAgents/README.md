# SKZ Agents Plugin for OJS

## Overview

The SKZ Agents Plugin integrates the SKZ autonomous agents framework with Open Journal Systems (OJS) to provide intelligent academic publishing workflow automation. This plugin implements the complete framework for managing 7 specialized AI agents that enhance every aspect of the publishing process.

## Features

### 7 Autonomous Agents

1. **Research Discovery Agent** - INCI database mining, patent analysis, trend identification
2. **Submission Assistant Agent** - Quality assessment, safety compliance, statistical review
3. **Editorial Orchestration Agent** - Workflow coordination, decision making, conflict resolution
4. **Review Coordination Agent** - Reviewer matching, workload management, quality monitoring
5. **Content Quality Agent** - Scientific validation, safety assessment, standards enforcement
6. **Publishing Production Agent** - Content formatting, visual generation, multi-channel distribution
7. **Analytics & Monitoring Agent** - Performance analytics, trend forecasting, strategic insights

### Plugin Capabilities

- **Settings Management** - Comprehensive configuration interface with validation
- **Database Integration** - Automated schema installation and data management
- **Performance Monitoring** - Real-time metrics and analytics dashboard
- **Agent Communication** - Robust API bridge with error handling and logging
- **Workflow Integration** - Seamless hooks into OJS publishing workflows
- **Security** - API authentication, role-based access control, audit logging

## Installation

### Prerequisites

- OJS 3.x installation
- MySQL/MariaDB database
- PHP 7.4+ with cURL extension
- SKZ autonomous agents framework (included in `/skz-integration/`)

### Setup Steps

1. **Enable the Plugin**
   - Go to OJS Admin → Settings → Website → Plugins
   - Find "SKZ Autonomous Agents" plugin
   - Click "Enable"

2. **Configure Settings**
   - Click "Settings" button next to the plugin
   - Configure API settings:
     - Agent Framework Base URL (default: `http://localhost:5000/api`)
     - API Key (generate a secure key)
     - Timeout settings
   - Enable desired features:
     - Auto submission processing
     - Auto reviewer assignment  
     - Auto quality checks
     - Performance monitoring

3. **Start Agent Framework**
   ```bash
   cd skz-integration/autonomous-agents-framework
   pip install -r requirements.txt
   python src/main.py
   ```

4. **Test Connectivity**
   - Go to plugin settings and verify connection status
   - Check the agent dashboard at `/skzAgents/dashboard`

## Configuration

### API Settings

| Setting | Description | Default |
|---------|-------------|---------|
| Agent Base URL | SKZ agents framework API endpoint | `http://localhost:5000/api` |
| API Key | Authentication key for agent communication | (required) |
| Timeout | Request timeout in seconds | 30 |

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| Auto Submission Processing | Enable automatic submission quality assessment | Disabled |
| Auto Reviewer Assignment | Enable automatic reviewer matching and assignment | Disabled |
| Auto Quality Checks | Enable automatic content quality validation | Enabled |
| Performance Monitoring | Enable agent performance tracking | Enabled |

### Performance Settings

| Setting | Description | Default |
|---------|-------------|---------|
| Max Concurrent Requests | Maximum simultaneous agent requests | 10 |
| Cache TTL | Cache time-to-live in seconds | 300 |

## Database Schema

The plugin automatically creates these tables when enabled:

### `skz_agent_states`
Stores current state of each agent for submissions.

### `skz_agent_communications`
Logs all communication between OJS and agents for auditing and performance monitoring.

## Usage

### Workflow Integration

The plugin automatically integrates with OJS workflows through hooks:

- **Submission Hook** - `submissionsubmitform::execute`
- **File Upload Hook** - `submissionfilesuploadform::execute`
- **Editorial Hook** - `editoraction::execute`
- **Review Hook** - `reviewassignmentform::execute`
- **Copyediting Hook** - `copyeditingform::execute`
- **Production Hook** - `publicationform::execute`

### Agent Dashboard

Access the agent management dashboard at `/skzAgents/dashboard` to:

- Monitor agent status and health
- View performance metrics
- Review communication logs
- Trigger manual agent actions
- Analyze workflow summaries

### API Endpoints

The plugin provides AJAX endpoints for real-time interaction:

- `POST /skzAgents/status` - Get agent status
- `POST /skzAgents/metrics` - Get performance metrics
- `POST /skzAgents/communications` - Get communication logs
- `POST /skzAgents/testConnection` - Test agent connectivity

## Development

### File Structure

```
plugins/generic/skzAgents/
├── SKZAgentsPlugin.inc.php           # Main plugin class
├── version.xml                       # Plugin version info
├── classes/
│   ├── SKZAgentBridge.inc.php       # API communication bridge
│   ├── SKZDAO.inc.php               # Database operations
│   └── SKZAgentsSettingsForm.inc.php # Settings form
├── pages/
│   └── SKZAgentsHandler.inc.php     # Page handlers and AJAX endpoints
├── templates/
│   └── settings.tpl                 # Settings template
├── locale/en_US/
│   └── locale.xml                   # Localization strings
└── schema.sql                       # Database schema
```

### Extending the Plugin

To add new agent integrations:

1. Add hook registration in `_registerAgentHooks()`
2. Implement handler method in main plugin class
3. Update bridge communication methods if needed
4. Add locale strings for new features

### Testing

Run the included test script to verify plugin integrity:

```bash
cd /path/to/ojs
php /path/to/test_skz_plugin.php
```

## Security Considerations

- **API Authentication** - Secure API keys for agent communication
- **Role-based Access** - Plugin functions restricted to appropriate user roles
- **Data Encryption** - Sensitive data encrypted in transit and at rest
- **Audit Logging** - Complete audit trail of all agent actions
- **Input Validation** - All user inputs validated and sanitized

## Performance Monitoring

The plugin includes comprehensive performance monitoring:

- **Response Times** - Track agent API response times
- **Success Rates** - Monitor agent reliability and error rates
- **Resource Usage** - Track bandwidth and processing metrics
- **Workflow Efficiency** - Measure publishing workflow improvements

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify agent framework is running
   - Check API URL and key configuration
   - Confirm network connectivity

2. **Database Errors**
   - Ensure database permissions are correct
   - Verify schema was installed properly
   - Check MySQL/MariaDB compatibility

3. **Plugin Not Appearing**
   - Check file permissions
   - Verify PHP syntax with `php -l`
   - Review OJS error logs

### Logging

Agent communication is logged to:
- Database: `skz_agent_communications` table
- PHP Error Log: Agent errors and warnings
- OJS Activity Log: Plugin actions and events

## Support

For issues and questions:

1. Check the plugin dashboard for agent status
2. Review communication logs for error details
3. Verify SKZ integration framework documentation
4. Check OJS error logs for detailed error information

## License

This plugin is part of the SKZ autonomous agents framework integration and follows the same licensing terms as the parent project.