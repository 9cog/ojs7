# Autonomous Agents Microservices Architecture

This directory contains the microservices implementation of the SKZ autonomous agents framework.

## Architecture Overview

The framework is deployed as 8 independent microservices:

1. **API Gateway** (`api-gateway/`) - Central entry point and routing
2. **Research Discovery Agent** (`research-discovery/`) - Literature search and analysis
3. **Submission Assistant Agent** (`submission-assistant/`) - Manuscript submission support
4. **Editorial Orchestration Agent** (`editorial-orchestration/`) - Workflow management
5. **Review Coordination Agent** (`review-coordination/`) - Peer review coordination
6. **Content Quality Agent** (`content-quality/`) - Quality assessment
7. **Publishing Production Agent** (`publishing-production/`) - Publication management
8. **Analytics Monitoring Agent** (`analytics-monitoring/`) - System monitoring

## Deployment

### Using Docker Compose (Development)
```bash
cd microservices
docker-compose up --build
```

### Individual Service Development
```bash
cd microservices/<service-name>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Service Communication

Services communicate through HTTP APIs routed via the API Gateway. Each service exposes:
- Health check endpoint (`/health`)
- Service-specific functionality endpoints
- Metrics endpoint (`/metrics`)

## Configuration

Configuration is managed through environment variables and shared configuration files.
See `config/` directory for service configurations.

## Monitoring

All services include built-in health checks and metrics collection for monitoring and observability.