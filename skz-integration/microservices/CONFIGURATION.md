# Microservices Configuration for Autonomous Agents Framework

## Quick Start Guide

### 🚀 Local Development Setup

1. **Prerequisites**
   ```bash
   python3 --version  # Should be 3.11+
   pip install Flask flask-cors requests
   ```

2. **Start All Services**
   ```bash
   cd skz-integration/microservices
   python3 test_microservices.py
   ```

3. **Access Points**
   - API Gateway: http://localhost:5000
   - Services Status: http://localhost:5000/api/v1/services
   - All Agents: http://localhost:5000/api/v1/agents

### 🐳 Docker Deployment

1. **Prerequisites**
   ```bash
   docker --version
   docker-compose --version
   ```

2. **Deploy with Docker**
   ```bash
   cd skz-integration/microservices
   ./deploy.sh
   ```

3. **Management Commands**
   ```bash
   ./manage.sh start     # Start all services
   ./manage.sh stop      # Stop all services  
   ./manage.sh status    # Check status
   ./manage.sh health    # Health check
   ./manage.sh logs      # View logs
   ```

## 🏗️ Architecture Overview

### Service Map
```
API Gateway (5000)
├── Research Discovery Agent (5001)
├── Submission Assistant Agent (5002)
├── Editorial Orchestration Agent (5003)
├── Review Coordination Agent (5004)
├── Content Quality Agent (5005)
├── Publishing Production Agent (5006)
└── Analytics Monitoring Agent (5007)
```

### Service Communication
- **API Gateway**: Central entry point, handles routing and load balancing
- **Service Discovery**: Automatic registration and health monitoring
- **HTTP APIs**: RESTful communication between services
- **Health Checks**: Built-in monitoring endpoints

## 🔌 API Reference

### Gateway Endpoints
```bash
GET  /health                           # Gateway health
GET  /api/v1/services                  # List all services
GET  /api/v1/agents                    # List all agents
GET  /api/v1/agents/{service}          # Get specific agent
POST /api/v1/agents/{service}/action   # Trigger agent action
GET  /api/v1/dashboard                 # Dashboard data
```

### Agent Endpoints (Each Service)
```bash
GET  /health       # Service health
GET  /agent        # Agent information
POST /action       # Trigger agent action
GET  /metrics      # Service metrics
```

### Example Usage
```bash
# Get all agents
curl http://localhost:5000/api/v1/agents

# Trigger research discovery
curl -X POST http://localhost:5000/api/v1/agents/research-discovery/action \
  -H "Content-Type: application/json" \
  -d '{"action":"literature_search","parameters":{"query":"AI publishing"}}'

# Check service health
curl http://localhost:5000/api/v1/services
```

## 🔧 Configuration

### Environment Variables
```bash
# API Gateway
API_GATEWAY_PORT=5000

# Service URLs (Docker)
RESEARCH_DISCOVERY_URL=http://research-discovery:5001
SUBMISSION_ASSISTANT_URL=http://submission-assistant:5002
# ... etc

# Service URLs (Local)
RESEARCH_DISCOVERY_URL=http://localhost:5001
SUBMISSION_ASSISTANT_URL=http://localhost:5002
# ... etc
```

### Service Configuration
Each service supports:
- **Port Configuration**: Environment variable `PORT`
- **Debug Mode**: Environment variable `DEBUG`
- **Service Name**: Environment variable `SERVICE_NAME`

## 🧪 Testing

### Automated Testing
```bash
# Comprehensive test suite
python3 test_microservices.py

# Individual service test
curl http://localhost:5001/health
```

### Manual Testing
```bash
# Start individual service
cd research-discovery && python3 app.py

# Test health
curl http://localhost:5001/health

# Test agent
curl http://localhost:5001/agent

# Test action
curl -X POST http://localhost:5001/action \
  -H "Content-Type: application/json" \
  -d '{"action":"test"}'
```

## 📊 Monitoring

### Health Checks
- **Service Level**: Each service exposes `/health`
- **Gateway Level**: Aggregated health at `/api/v1/services`
- **Automated**: Health checks run automatically

### Metrics
- **Response Times**: Tracked per service
- **Success Rates**: Calculated per agent
- **Request Counts**: Total actions per service
- **Availability**: Service uptime monitoring

### Logs
```bash
# View all logs
./manage.sh logs

# View specific service logs
./manage.sh logs research-discovery

# Docker logs
docker-compose logs -f
```

## 🚦 Service Management

### Starting Services
```bash
# All services
./manage.sh start

# Individual service (Docker)
docker-compose up research-discovery

# Individual service (Local)
cd research-discovery && python3 app.py
```

### Stopping Services
```bash
# All services
./manage.sh stop

# Individual service
docker-compose stop research-discovery
```

### Scaling Services
```bash
# Scale specific service
./manage.sh scale research-discovery 3

# Or with docker-compose
docker-compose up --scale research-discovery=3
```

## 🔒 Security

### Authentication
- API key support (configurable)
- JWT token validation (optional)
- CORS policy configuration

### Network Security
- Service-to-service communication
- Health check endpoint protection
- Request timeout configuration

## 🔄 Integration with OJS

### API Bridge
The microservices can be integrated with OJS through:
- **HTTP API calls** from OJS to gateway
- **Webhook notifications** from OJS to agents
- **Database synchronization** for shared data

### Example Integration
```php
// In OJS plugin
$gateway_url = 'http://localhost:5000';
$response = $http_client->post("$gateway_url/api/v1/agents/research-discovery/action", [
    'action' => 'literature_search',
    'parameters' => ['query' => $manuscript_title]
]);
```

## 🐛 Troubleshooting

### Common Issues

1. **Service Not Starting**
   ```bash
   # Check if port is available
   netstat -tulpn | grep :5001
   
   # Check service logs
   ./manage.sh logs research-discovery
   ```

2. **Service Discovery Failing**
   ```bash
   # Check service health
   curl http://localhost:5001/health
   
   # Check gateway configuration
   curl http://localhost:5000/api/v1/services
   ```

3. **Docker Issues**
   ```bash
   # Rebuild services
   ./manage.sh build
   
   # Clean and restart
   ./manage.sh clean
   ./deploy.sh
   ```

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
python3 app.py

# Or in Docker
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up
```

## 📈 Performance

### Optimization Tips
- **Caching**: Enable Redis for response caching
- **Load Balancing**: Scale services horizontally
- **Resource Limits**: Set memory/CPU limits in Docker
- **Connection Pooling**: Configure HTTP client pools

### Benchmarking
```bash
# Load test gateway
ab -n 1000 -c 10 http://localhost:5000/api/v1/agents

# Monitor resource usage
docker stats

# Check response times
curl -w "%{time_total}" http://localhost:5000/health
```

This microservices architecture provides a robust, scalable foundation for the autonomous agents framework, ready for production deployment and integration with the existing OJS system.