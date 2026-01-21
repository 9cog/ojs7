# API Documentation Structure

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Core OJS APIs](#core-ojs-apis)
4. [Agent Framework APIs](#agent-framework-apis)
5. [Individual Agent APIs](#individual-agent-apis)
6. [WebSocket APIs](#websocket-apis)
7. [Integration APIs](#integration-apis)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)
10. [SDKs and Examples](#sdks-and-examples)

## Overview

The Enhanced OJS with SKZ integration provides a comprehensive RESTful API ecosystem that includes:

- **OJS Core APIs**: Traditional journal management functionality
- **Agent Framework APIs**: Autonomous agent coordination and management
- **Individual Agent APIs**: Specialized endpoints for each of the 7 agents
- **Real-time APIs**: WebSocket connections for live updates
- **Integration APIs**: External system integration points

### Base URLs

```
Production:  https://api.yourjournal.com/v1
Staging:     https://staging-api.yourjournal.com/v1
Development: https://dev-api.yourjournal.com/v1
```

### API Versioning

All APIs are versioned using URL path versioning:
- Current version: `v1`
- Beta features: `v1-beta`
- Legacy support: `v0` (deprecated)

## Authentication

### JWT Token Authentication

```http
Authorization: Bearer <jwt_token>
```

### API Key Authentication

```http
X-API-Key: <your_api_key>
```

### OAuth 2.0 (for third-party integrations)

```
Authorization Code Flow:
GET /oauth/authorize?client_id=CLIENT_ID&response_type=code&scope=SCOPE&redirect_uri=REDIRECT_URI

Token Exchange:
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&code=CODE&client_id=CLIENT_ID&client_secret=CLIENT_SECRET
```

## Core OJS APIs

### Manuscript Management

#### Submit Manuscript
```http
POST /api/v1/submissions
Content-Type: multipart/form-data
Authorization: Bearer <token>

{
  "title": "Research Title",
  "abstract": "Abstract text",
  "authors": [
    {
      "firstName": "John",
      "lastName": "Doe",
      "email": "john.doe@example.com",
      "affiliation": "University Name"
    }
  ],
  "files": ["manuscript.pdf", "supplementary.zip"],
  "keywords": ["cosmetic science", "INCI", "safety"],
  "sectionId": 1
}
```

#### Get Submission Status
```http
GET /api/v1/submissions/{id}/status
Authorization: Bearer <token>

Response:
{
  "submissionId": 123,
  "status": "under_review",
  "currentStage": "external_review",
  "lastUpdated": "2024-01-15T10:30:00Z",
  "agentProcessing": {
    "researchDiscovery": "completed",
    "submissionAssistant": "completed", 
    "editorialOrchestration": "in_progress",
    "reviewCoordination": "pending"
  }
}
```

### Editorial Workflow

#### Get Editorial Dashboard
```http
GET /api/v1/editorial/dashboard
Authorization: Bearer <token>

Response:
{
  "submissions": {
    "new": 15,
    "under_review": 23,
    "revision_required": 8,
    "accepted": 12,
    "rejected": 5
  },
  "agents": {
    "active": 7,
    "performance": {
      "efficiency": 94.2,
      "accuracy": 98.7,
      "speed": "65% faster"
    }
  },
  "alerts": [
    {
      "type": "warning",
      "message": "High review workload detected",
      "agentId": "review-coordination",
      "timestamp": "2024-01-15T09:15:00Z"
    }
  ]
}
```

## Agent Framework APIs

### Agent Management

#### List All Agents
```http
GET /api/v1/agents
Authorization: Bearer <token>

Response:
{
  "agents": [
    {
      "id": "research-discovery",
      "name": "Research Discovery Agent",
      "status": "active",
      "version": "2.1.0",
      "performance": {
        "successRate": 96.8,
        "avgResponseTime": "1.2s",
        "processedToday": 45
      }
    },
    {
      "id": "submission-assistant",
      "name": "Submission Assistant Agent",
      "status": "active",
      "version": "2.0.3",
      "performance": {
        "successRate": 98.1,
        "avgResponseTime": "0.8s",
        "processedToday": 38
      }
    }
  ]
}
```

#### Get Agent Health
```http
GET /api/v1/agents/{agentId}/health
Authorization: Bearer <token>

Response:
{
  "agentId": "research-discovery",
  "status": "healthy",
  "uptime": "7d 14h 32m",
  "memoryUsage": "45%",
  "cpuUsage": "23%",
  "lastHeartbeat": "2024-01-15T10:29:58Z",
  "dependencies": {
    "inci_database": "connected",
    "patent_api": "connected",
    "redis_cache": "connected"
  }
}
```

### Agent Orchestration

#### Trigger Workflow
```http
POST /api/v1/orchestration/workflows
Authorization: Bearer <token>
Content-Type: application/json

{
  "workflowType": "manuscript_processing",
  "submissionId": 123,
  "priority": "normal",
  "agents": ["research-discovery", "submission-assistant", "content-quality"],
  "parameters": {
    "fastTrack": false,
    "qualityThreshold": 0.95
  }
}

Response:
{
  "workflowId": "wf_abc123",
  "status": "initiated",
  "estimatedDuration": "2-4 hours",
  "agents": [
    {
      "agentId": "research-discovery",
      "status": "queued",
      "estimatedStart": "2024-01-15T10:35:00Z"
    }
  ]
}
```

## Individual Agent APIs

### Research Discovery Agent

#### Analyze INCI Components
```http
POST /api/v1/agents/research-discovery/analyze-inci
Authorization: Bearer <token>
Content-Type: application/json

{
  "ingredients": [
    "Aqua",
    "Glycerin", 
    "Hyaluronic Acid",
    "Retinol"
  ],
  "analysisType": "comprehensive",
  "includePatents": true,
  "includeTrends": true
}

Response:
{
  "analysisId": "rd_789xyz",
  "results": {
    "ingredients": [
      {
        "name": "Hyaluronic Acid",
        "inciName": "Sodium Hyaluronate",
        "safety": {
          "rating": "safe",
          "concentration": "0.1-2.0%",
          "restrictions": "none"
        },
        "patents": {
          "active": 23,
          "recent": 5,
          "expiring": 2
        },
        "trends": {
          "popularity": "increasing",
          "marketGrowth": "15.2%"
        }
      }
    ],
    "interactions": [],
    "recommendations": [
      "Consider novel delivery systems for enhanced efficacy"
    ]
  }
}
```

### Submission Assistant Agent

#### Quality Assessment
```http
POST /api/v1/agents/submission-assistant/assess-quality
Authorization: Bearer <token>
Content-Type: application/json

{
  "submissionId": 123,
  "assessmentType": "comprehensive",
  "includeStatistical": true,
  "includeSafety": true,
  "includeCompliance": true
}

Response:
{
  "assessmentId": "sa_456def",
  "overallScore": 87.5,
  "assessments": {
    "technical": {
      "score": 92,
      "issues": [],
      "recommendations": ["Consider expanding methodology section"]
    },
    "safety": {
      "score": 95,
      "compliance": "full",
      "warnings": []
    },
    "statistical": {
      "score": 78,
      "issues": ["Small sample size", "Missing power analysis"],
      "recommendations": ["Increase sample size to 50+"]
    }
  }
}
```

### Editorial Orchestration Agent

#### Get Workflow Status
```http
GET /api/v1/agents/editorial-orchestration/workflows/{workflowId}
Authorization: Bearer <token>

Response:
{
  "workflowId": "wf_abc123",
  "submissionId": 123,
  "status": "in_progress",
  "currentStage": "external_review",
  "progress": 65,
  "stages": [
    {
      "name": "initial_review",
      "status": "completed",
      "completedAt": "2024-01-15T11:00:00Z",
      "agent": "submission-assistant"
    },
    {
      "name": "external_review",
      "status": "in_progress",
      "startedAt": "2024-01-15T11:30:00Z",
      "agent": "review-coordination",
      "estimatedCompletion": "2024-01-22T17:00:00Z"
    }
  ]
}
```

### Review Coordination Agent

#### Assign Reviewers
```http
POST /api/v1/agents/review-coordination/assign-reviewers
Authorization: Bearer <token>
Content-Type: application/json

{
  "submissionId": 123,
  "reviewerCount": 3,
  "criteria": {
    "expertise": ["cosmetic chemistry", "dermatology"],
    "excludeAuthors": true,
    "preferredRegions": ["North America", "Europe"],
    "maxWorkload": 5
  }
}

Response:
{
  "assignmentId": "rc_321ghi",
  "reviewers": [
    {
      "reviewerId": 456,
      "name": "Dr. Sarah Johnson",
      "expertise": ["cosmetic chemistry", "skin care"],
      "currentWorkload": 3,
      "estimatedCompletion": "2024-01-29T17:00:00Z"
    }
  ],
  "timeline": {
    "invitationsSent": "2024-01-15T12:00:00Z",
    "expectedAcceptance": "2024-01-17T17:00:00Z",
    "reviewDeadline": "2024-01-29T17:00:00Z"
  }
}
```

### Content Quality Agent

#### Validate Scientific Content
```http
POST /api/v1/agents/content-quality/validate-content
Authorization: Bearer <token>
Content-Type: application/json

{
  "submissionId": 123,
  "validationType": "scientific",
  "sections": ["methodology", "results", "discussion"],
  "criteria": {
    "statisticalRigor": true,
    "methodologySound": true,
    "conclusionsSupported": true
  }
}

Response:
{
  "validationId": "cq_654jkl",
  "overallScore": 91.2,
  "validations": {
    "methodology": {
      "score": 88,
      "issues": ["Control group size could be larger"],
      "recommendations": ["Consider stratified randomization"]
    },
    "results": {
      "score": 95,
      "issues": [],
      "recommendations": []
    },
    "discussion": {
      "score": 90,
      "issues": ["Limited discussion of limitations"],
      "recommendations": ["Expand limitations section"]
    }
  }
}
```

### Publishing Production Agent

#### Format for Publication
```http
POST /api/v1/agents/publishing-production/format-publication
Authorization: Bearer <token>
Content-Type: application/json

{
  "submissionId": 123,
  "outputFormats": ["html", "pdf", "epub"],
  "includeVisuals": true,
  "generateMetadata": true,
  "optimizeForChannels": ["web", "print", "mobile"]
}

Response:
{
  "productionId": "pp_987mno",
  "status": "in_progress",
  "outputs": [
    {
      "format": "html",
      "status": "completed",
      "url": "/publications/123/formats/html",
      "size": "2.3MB"
    },
    {
      "format": "pdf",
      "status": "in_progress",
      "progress": 75
    }
  ],
  "metadata": {
    "doi": "10.1234/journal.2024.123",
    "keywords": ["cosmetic science", "INCI", "safety"],
    "citations": 15
  }
}
```

### Analytics & Monitoring Agent

#### Get Performance Analytics
```http
GET /api/v1/agents/analytics-monitoring/performance
Authorization: Bearer <token>
Parameters: ?timeframe=7d&metrics=efficiency,accuracy,speed

Response:
{
  "timeframe": "7d",
  "overview": {
    "totalSubmissions": 156,
    "processedSubmissions": 152,
    "avgProcessingTime": "3.2 hours",
    "efficiencyImprovement": "65%"
  },
  "agentPerformance": [
    {
      "agentId": "research-discovery",
      "metrics": {
        "efficiency": 96.8,
        "accuracy": 98.2,
        "avgResponseTime": "1.2s",
        "processedCount": 145
      }
    }
  ],
  "trends": {
    "submissionVolume": "increasing",
    "processingSpeed": "stable",
    "qualityScore": "improving"
  }
}
```

## WebSocket APIs

### Real-time Agent Status

#### Connect to Agent Status Stream
```javascript
const ws = new WebSocket('wss://api.yourjournal.com/ws/v1/agents/status');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Agent status update:', data);
};

// Example message:
{
  "type": "agent_status_update",
  "agentId": "research-discovery",
  "status": "processing",
  "submissionId": 123,
  "progress": 45,
  "timestamp": "2024-01-15T12:30:00Z"
}
```

### Workflow Progress Stream

#### Connect to Workflow Updates
```javascript
const ws = new WebSocket('wss://api.yourjournal.com/ws/v1/workflows/wf_abc123');

// Example messages:
{
  "type": "workflow_progress",
  "workflowId": "wf_abc123",
  "stage": "external_review", 
  "progress": 65,
  "estimatedCompletion": "2024-01-22T17:00:00Z"
}
```

## Integration APIs

### External System Integration

#### INCI Database Integration
```http
GET /api/v1/integrations/inci/search
Authorization: Bearer <token>
Parameters: ?q=hyaluronic&limit=10&offset=0

Response:
{
  "results": [
    {
      "inciName": "Sodium Hyaluronate",
      "commonName": "Hyaluronic Acid",
      "function": "Humectant",
      "safetyRating": "A",
      "restrictions": "None"
    }
  ],
  "total": 1,
  "page": 1
}
```

#### Patent Database Integration
```http
GET /api/v1/integrations/patents/search
Authorization: Bearer <token>
Parameters: ?keywords=retinol+delivery&status=active&limit=20

Response:
{
  "patents": [
    {
      "patentNumber": "US10123456",
      "title": "Novel Retinol Delivery System",
      "status": "active",
      "expiryDate": "2025-12-31",
      "relevanceScore": 0.95
    }
  ],
  "total": 23,
  "page": 1
}
```

## Error Handling

### Standard Error Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid submission data provided",
    "details": {
      "field": "authors",
      "issue": "At least one author is required"
    },
    "timestamp": "2024-01-15T12:30:00Z",
    "requestId": "req_123abc"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_TOKEN` | 401 | Authentication token is invalid or expired |
| `INSUFFICIENT_PERMISSIONS` | 403 | User lacks required permissions |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource does not exist |
| `VALIDATION_ERROR` | 400 | Request data validation failed |
| `AGENT_UNAVAILABLE` | 503 | Requested agent is temporarily unavailable |
| `RATE_LIMIT_EXCEEDED` | 429 | API rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error occurred |

## Rate Limiting

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642261800
X-RateLimit-Window: 3600
```

### Rate Limits by Endpoint Category

| Category | Limit | Window |
|----------|-------|--------|
| Authentication | 10 requests | 1 minute |
| Agent Management | 100 requests | 1 hour |
| Submissions | 50 requests | 1 hour |
| Analytics | 200 requests | 1 hour |
| WebSocket Connections | 5 connections | Per user |

## SDKs and Examples

### Python SDK Example

```python
from ojs_skz_sdk import OJSClient

client = OJSClient(
    base_url='https://api.yourjournal.com/v1',
    api_key='your_api_key'
)

# Submit manuscript
submission = client.submissions.create({
    'title': 'Novel Cosmetic Formulation',
    'authors': [{'firstName': 'John', 'lastName': 'Doe'}],
    'files': ['manuscript.pdf']
})

# Monitor agent processing
status = client.agents.get_workflow_status(submission.workflow_id)
print(f"Current stage: {status.current_stage}")

# Get analytics
analytics = client.analytics.get_performance(timeframe='7d')
print(f"Efficiency improvement: {analytics.efficiency_improvement}")
```

### JavaScript SDK Example

```javascript
import { OJSClient } from 'ojs-skz-sdk';

const client = new OJSClient({
  baseURL: 'https://api.yourjournal.com/v1',
  apiKey: 'your_api_key'
});

// Submit manuscript
const submission = await client.submissions.create({
  title: 'Novel Cosmetic Formulation',
  authors: [{ firstName: 'John', lastName: 'Doe' }],
  files: ['manuscript.pdf']
});

// Real-time workflow monitoring
client.workflows.subscribe(submission.workflowId, (update) => {
  console.log('Workflow update:', update);
});
```

### cURL Examples

#### Basic Submission
```bash
curl -X POST https://api.yourjournal.com/v1/submissions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Research Title",
    "authors": [{"firstName": "John", "lastName": "Doe"}]
  }'
```

#### Agent Health Check
```bash
curl -X GET https://api.yourjournal.com/v1/agents/research-discovery/health \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

This comprehensive API documentation provides complete coverage of the Enhanced OJS with SKZ integration API ecosystem, enabling developers to build powerful integrations and applications on top of the autonomous academic publishing platform.