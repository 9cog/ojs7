# SKZ API Gateway Configuration Documentation

## Overview

The SKZ API Gateway provides a secure, scalable interface between the Open Journal Systems (OJS) PHP application and the SKZ autonomous agents framework (Python Flask). This gateway handles routing, authentication, rate limiting, request/response transformation, and monitoring for all agent communications.

## Architecture

```
OJS (PHP) <-> SKZ API Gateway <-> SKZ Autonomous Agents Framework (Python Flask)
```

### Components

1. **SKZAPIGateway** - Core gateway functionality with security and routing
2. **SKZAPIRouter** - Request routing and endpoint management
3. **SKZAgentsHandler** - OJS page handler for API endpoints
4. **Configuration Files** - Gateway and agent configuration management

## Configuration Files

### 1. SKZ Agents Configuration (`skz-integration/config/skz-agents.conf`)

Main configuration file for the SKZ integration:

```ini
[skz]
# Basic settings
enabled = true
agent_base_url = "http://localhost:5000/api"
api_key = "your-secure-api-key-here"

# API Gateway settings
gateway_enabled = true
gateway_base_path = "/index.php/context/skzAgents/api"
gateway_timeout = 45
gateway_max_retries = 3

# Security settings
api_gateway_auth_required = true
rate_limit_enabled = true
rate_limit_requests_per_minute = 100
webhook_signature_validation = true

# Performance settings
performance_monitoring = true
request_transformation_enabled = true
```

### 2. API Gateway Configuration (`skz-integration/config/api-gateway.yml`)

Detailed gateway configuration in YAML format:

```yaml
api_gateway:
  enabled: true
  base_path: "/index.php/context/skzAgents/api"
  security:
    authentication_required: true
    rate_limiting:
      enabled: true
      requests_per_minute: 100
  performance:
    timeout: 45
    max_retries: 3
```

## API Endpoints

### Base URL Structure
```
https://your-ojs-domain.com/index.php/context/skzAgents/api/
```

### Supported Endpoints

#### 1. Agent Status
- **URL**: `/api/status`
- **Method**: GET
- **Description**: Get status of all agents or specific agent
- **Example**: `/api/status` or `/api/status/research-discovery`

#### 2. Agent Actions
- **URL**: `/api/{agent-name}/{action}`
- **Method**: POST
- **Description**: Execute specific action on an agent
- **Example**: `/api/research-discovery/literature_search`

#### 3. Webhook Registration
- **URL**: `/api/webhook/register`
- **Method**: POST
- **Description**: Register webhook for agent events
- **Payload**: 
  ```json
  {
    "event": "agent.task.completed",
    "callback_url": "https://your-domain.com/webhook"
  }
  ```

#### 4. Webhook Callbacks
- **URL**: `/webhook/{event-type}`
- **Method**: POST
- **Description**: Receive callbacks from agents
- **Headers**: `X-SKZ-Signature` for verification

## Agent Endpoints Configuration

### The 7 Autonomous Agents

#### 1. Research Discovery Agent
- **Path**: `/agents/research-discovery`
- **Actions**: 
  - `literature_search` - Search academic literature
  - `gap_analysis` - Identify research gaps
  - `trend_identification` - Analyze research trends
  - `inci_analysis` - Analyze cosmetic ingredients
  - `patent_analysis` - Analyze patent landscapes
  - `regulatory_monitoring` - Monitor regulatory changes

#### 2. Submission Assistant Agent
- **Path**: `/agents/submission-assistant`
- **Actions**:
  - `format_check` - Check manuscript formatting
  - `venue_recommendation` - Recommend publication venues
  - `compliance_validation` - Validate submission compliance
  - `quality_assessment` - Assess manuscript quality
  - `safety_compliance` - Check safety compliance
  - `statistical_review` - Review statistical methods

#### 3. Editorial Orchestration Agent
- **Path**: `/agents/editorial-orchestration`
- **Actions**:
  - `workflow_management` - Manage editorial workflows
  - `decision_support` - Support editorial decisions
  - `deadline_tracking` - Track submission deadlines
  - `conflict_resolution` - Resolve editorial conflicts
  - `strategic_planning` - Strategic editorial planning
  - `resource_allocation` - Allocate editorial resources

#### 4. Review Coordination Agent
- **Path**: `/agents/review-coordination`
- **Actions**:
  - `reviewer_matching` - Match reviewers to submissions
  - `review_tracking` - Track review progress
  - `quality_assessment` - Assess review quality
  - `workload_management` - Manage reviewer workloads
  - `expert_network` - Manage expert networks
  - `timeline_optimization` - Optimize review timelines

#### 5. Content Quality Agent
- **Path**: `/agents/content-quality`
- **Actions**:
  - `scientific_validation` - Validate scientific content
  - `safety_assessment` - Assess content safety
  - `standards_enforcement` - Enforce quality standards
  - `regulatory_compliance` - Check regulatory compliance
  - `data_integrity` - Verify data integrity
  - `methodology_review` - Review research methodologies

#### 6. Publishing Production Agent
- **Path**: `/agents/publishing-production`
- **Actions**:
  - `content_formatting` - Format content for publication
  - `visual_generation` - Generate visualizations
  - `distribution_preparation` - Prepare for distribution
  - `regulatory_reporting` - Generate regulatory reports
  - `metadata_generation` - Generate publication metadata
  - `quality_control` - Perform quality control

#### 7. Analytics & Monitoring Agent
- **Path**: `/agents/analytics-monitoring`
- **Actions**:
  - `performance_analytics` - Analyze system performance
  - `trend_forecasting` - Forecast publication trends
  - `strategic_insights` - Generate strategic insights
  - `continuous_learning` - Continuous system learning
  - `system_health` - Monitor system health
  - `predictive_analysis` - Perform predictive analysis

## Security Features

### Authentication
- API key validation for all requests
- Role-based access control through OJS permissions
- Webhook signature validation using HMAC-SHA256

### Rate Limiting
- Configurable rate limits per agent/action
- Burst protection with exponential backoff
- IP-based and user-based limiting

### Request Validation
- Input sanitization and validation
- Schema validation for request/response data
- Maximum request size limits

### Error Handling
- Circuit breaker pattern for failed services
- Automatic retry with exponential backoff
- Graceful degradation and fallback responses

## Monitoring and Logging

### Performance Metrics
- Request/response times
- Success/failure rates
- Agent availability
- Rate limit violations

### Logging
- Request/response logging
- Error tracking and alerting
- Security event logging
- Performance threshold alerts

### Health Checks
- Agent availability monitoring
- Gateway health status
- Database connectivity checks
- External service dependencies

## Installation and Setup

### 1. Prerequisites
- OJS 3.x installation
- PHP 7.4+ with cURL extension
- MySQL/MariaDB database
- SKZ autonomous agents framework

### 2. Configuration Steps

1. **Enable the Plugin**:
   - Go to OJS Admin → Settings → Website → Plugins
   - Find "SKZ Autonomous Agents" plugin
   - Click "Enable"

2. **Configure API Settings**:
   - Click "Settings" next to the plugin
   - Set Agent Framework Base URL: `http://localhost:5000/api`
   - Generate and set a secure API key
   - Configure timeout and retry settings

3. **Configure Gateway Settings**:
   - Enable API Gateway: Yes
   - Set Gateway Base Path: `/index.php/context/skzAgents/api`
   - Configure rate limiting and security options
   - Set webhook secret for secure callbacks

4. **Test Configuration**:
   - Run the test script: `php plugins/generic/skzAgents/test_gateway_config.php`
   - Verify all components are properly configured
   - Test API endpoints for connectivity

### 3. Starting the Framework
```bash
cd skz-integration/autonomous-agents-framework
pip install -r requirements.txt
python3 src/main.py
```

### 4. Testing API Endpoints
```bash
# Test gateway status
curl -X GET "http://your-ojs-domain.com/index.php/context/skzAgents/api/status" \
  -H "X-API-Key: your-api-key"

# Test agent action
curl -X POST "http://your-ojs-domain.com/index.php/context/skzAgents/api/research-discovery/literature_search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "cosmetic science", "filters": {"year": "2023"}}'
```

## Troubleshooting

### Common Issues

1. **Connection Refused**:
   - Verify SKZ agents framework is running
   - Check firewall settings
   - Confirm API base URL configuration

2. **Authentication Errors**:
   - Verify API key configuration
   - Check user permissions in OJS
   - Confirm webhook signature settings

3. **Rate Limit Exceeded**:
   - Review rate limit configuration
   - Implement proper retry logic
   - Consider increasing limits for high-volume operations

4. **Timeout Errors**:
   - Increase gateway timeout settings
   - Check agent framework performance
   - Optimize request payload size

### Debug Mode
Enable debug logging by setting:
```ini
[skz]
gateway_debug_mode = true
verbose_logging = true
```

## Performance Optimization

### Best Practices
1. Configure appropriate timeout values
2. Use connection pooling for high-volume requests
3. Implement caching for frequently accessed data
4. Monitor and tune rate limits based on usage patterns
5. Use webhooks for asynchronous operations

### Scaling Considerations
1. Load balancing for multiple agent instances
2. Database connection pooling
3. Redis caching for session management
4. Horizontal scaling of agent framework

## Version Compatibility

- **OJS**: 3.x series
- **PHP**: 7.4+
- **Python**: 3.8+
- **Flask**: 3.x
- **Database**: MySQL 5.7+ or MariaDB 10.3+

## Support and Maintenance

### Regular Maintenance
1. Monitor API performance metrics
2. Review and rotate API keys regularly
3. Update rate limits based on usage patterns
4. Review security logs for anomalies
5. Keep framework dependencies updated

### Monitoring Checklist
- [ ] Agent availability and response times
- [ ] API gateway error rates
- [ ] Database performance
- [ ] Security event logs
- [ ] Resource utilization metrics