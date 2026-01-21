# OJS Web Interface API Specification

## Overview

This document defines the API endpoints and data structures required for the React-based web interface to communicate with the existing OJS backend and new chatbot services.

## Base Configuration

### API Base URLs
- **OJS Backend**: `/api/v1/`
- **Chatbot Service**: `/api/chatbot/v1/`
- **WebSocket**: `ws://localhost:8080/ws`

### Authentication
- **Method**: JWT Bearer Token
- **Header**: `Authorization: Bearer <token>`
- **Refresh**: Automatic token refresh before expiration

## Core API Endpoints

### Authentication & User Management

#### POST /api/v1/auth/login
```json
{
  "username": "string",
  "password": "string",
  "remember": "boolean"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "token": "jwt_token_string",
    "refresh_token": "refresh_token_string",
    "user": {
      "id": "integer",
      "username": "string",
      "email": "string",
      "roles": ["author", "editor", "reviewer"],
      "profile": {
        "firstName": "string",
        "lastName": "string",
        "affiliation": "string",
        "orcid": "string"
      }
    }
  }
}
```

#### GET /api/v1/user/profile
**Response:**
```json
{
  "success": true,
  "data": {
    "id": "integer",
    "username": "string",
    "email": "string",
    "profile": {
      "firstName": "string",
      "lastName": "string",
      "affiliation": "string",
      "biography": "string",
      "orcid": "string",
      "avatar": "string"
    },
    "preferences": {
      "language": "string",
      "timezone": "string",
      "notifications": {
        "email": "boolean",
        "browser": "boolean"
      }
    }
  }
}
```

### Dashboard Data

#### GET /api/v1/dashboard/author
**Response:**
```json
{
  "success": true,
  "data": {
    "overview": {
      "totalSubmissions": "integer",
      "activeSubmissions": "integer",
      "acceptedSubmissions": "integer",
      "rejectedSubmissions": "integer"
    },
    "recentSubmissions": [
      {
        "id": "integer",
        "title": "string",
        "status": "string",
        "stage": "string",
        "submittedDate": "ISO_date",
        "lastActivity": "ISO_date"
      }
    ],
    "recentActivity": [
      {
        "type": "string",
        "message": "string",
        "date": "ISO_date",
        "submissionId": "integer"
      }
    ],
    "upcomingDeadlines": [
      {
        "type": "string",
        "description": "string",
        "dueDate": "ISO_date",
        "submissionId": "integer"
      }
    ]
  }
}
```

#### GET /api/v1/dashboard/editor
**Response:**
```json
{
  "success": true,
  "data": {
    "overview": {
      "pendingSubmissions": "integer",
      "inReview": "integer",
      "awaitingDecision": "integer",
      "overdueReviews": "integer"
    },
    "submissionsByStage": {
      "submission": "integer",
      "review": "integer",
      "copyediting": "integer",
      "production": "integer"
    },
    "recentSubmissions": [
      {
        "id": "integer",
        "title": "string",
        "authors": ["string"],
        "stage": "string",
        "daysInStage": "integer",
        "assignedEditor": "string"
      }
    ],
    "reviewerPerformance": {
      "averageReviewTime": "integer",
      "onTimeReviews": "float",
      "overdueReviews": "integer"
    }
  }
}
```

### Submission Management

#### GET /api/v1/submissions
**Query Parameters:**
- `page`: integer (default: 1)
- `limit`: integer (default: 20)
- `status`: string (optional)
- `stage`: string (optional)
- `search`: string (optional)

**Response:**
```json
{
  "success": true,
  "data": {
    "submissions": [
      {
        "id": "integer",
        "title": "string",
        "abstract": "string",
        "authors": [
          {
            "id": "integer",
            "firstName": "string",
            "lastName": "string",
            "email": "string",
            "affiliation": "string",
            "isPrimary": "boolean"
          }
        ],
        "status": "string",
        "stage": "string",
        "submittedDate": "ISO_date",
        "lastModified": "ISO_date",
        "section": {
          "id": "integer",
          "title": "string"
        },
        "keywords": ["string"],
        "files": [
          {
            "id": "integer",
            "name": "string",
            "type": "string",
            "size": "integer",
            "uploadDate": "ISO_date"
          }
        ]
      }
    ],
    "pagination": {
      "currentPage": "integer",
      "totalPages": "integer",
      "totalItems": "integer",
      "itemsPerPage": "integer"
    }
  }
}
```

#### POST /api/v1/submissions
**Request:**
```json
{
  "title": "string",
  "abstract": "string",
  "sectionId": "integer",
  "language": "string",
  "keywords": ["string"],
  "authors": [
    {
      "firstName": "string",
      "lastName": "string",
      "email": "string",
      "affiliation": "string",
      "isPrimary": "boolean",
      "biography": "string"
    }
  ],
  "metadata": {
    "prefix": "string",
    "subtitle": "string",
    "coverage": "string",
    "type": "string",
    "source": "string",
    "rights": "string"
  }
}
```

#### GET /api/v1/submissions/{id}
**Response:**
```json
{
  "success": true,
  "data": {
    "id": "integer",
    "title": "string",
    "abstract": "string",
    "status": "string",
    "stage": "string",
    "currentRound": "integer",
    "submittedDate": "ISO_date",
    "lastModified": "ISO_date",
    "authors": [...],
    "files": [...],
    "reviews": [
      {
        "id": "integer",
        "reviewerId": "integer",
        "reviewerName": "string",
        "recommendation": "string",
        "comments": "string",
        "dateCompleted": "ISO_date",
        "round": "integer"
      }
    ],
    "editorial": {
      "assignedEditor": {
        "id": "integer",
        "name": "string",
        "email": "string"
      },
      "decisions": [
        {
          "decision": "string",
          "date": "ISO_date",
          "editorId": "integer",
          "comments": "string"
        }
      ]
    }
  }
}
```

### Review Management

#### GET /api/v1/reviews/assignments
**Response:**
```json
{
  "success": true,
  "data": {
    "pending": [
      {
        "id": "integer",
        "submissionId": "integer",
        "submissionTitle": "string",
        "authors": ["string"],
        "dueDate": "ISO_date",
        "assignedDate": "ISO_date",
        "round": "integer",
        "reviewMethod": "string"
      }
    ],
    "completed": [...],
    "declined": [...]
  }
}
```

#### POST /api/v1/reviews/{id}/submit
**Request:**
```json
{
  "recommendation": "string",
  "comments": "string",
  "confidentialComments": "string",
  "reviewFiles": ["integer"]
}
```

### Issue Management

#### GET /api/v1/issues
**Response:**
```json
{
  "success": true,
  "data": {
    "current": {
      "id": "integer",
      "volume": "integer",
      "number": "integer",
      "year": "integer",
      "title": "string",
      "publishedDate": "ISO_date",
      "articles": [
        {
          "id": "integer",
          "title": "string",
          "authors": ["string"],
          "pages": "string",
          "doi": "string"
        }
      ]
    },
    "upcoming": [...],
    "archive": [...]
  }
}
```

## Chatbot API Endpoints

### Chat Session Management

#### POST /api/chatbot/v1/sessions
**Request:**
```json
{
  "userId": "integer",
  "context": {
    "page": "string",
    "submissionId": "integer",
    "userRole": "string"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "sessionId": "string",
    "agentType": "string",
    "welcomeMessage": "string"
  }
}
```

#### POST /api/chatbot/v1/sessions/{sessionId}/messages
**Request:**
```json
{
  "message": "string",
  "messageType": "text",
  "attachments": ["string"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "messageId": "string",
    "response": {
      "text": "string",
      "type": "text",
      "actions": [
        {
          "type": "button",
          "label": "string",
          "action": "string",
          "data": {}
        }
      ],
      "suggestions": ["string"]
    },
    "agentType": "string",
    "confidence": "float"
  }
}
```

### Agent-Specific Endpoints

#### POST /api/chatbot/v1/agents/submission/validate
**Request:**
```json
{
  "submissionData": {
    "title": "string",
    "abstract": "string",
    "authors": [...],
    "files": [...]
  },
  "journalId": "integer"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "isValid": "boolean",
    "issues": [
      {
        "field": "string",
        "severity": "error|warning|info",
        "message": "string",
        "suggestion": "string"
      }
    ],
    "completeness": "float"
  }
}
```

#### POST /api/chatbot/v1/agents/editorial/recommend-reviewers
**Request:**
```json
{
  "submissionId": "integer",
  "keywords": ["string"],
  "excludeReviewers": ["integer"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "reviewerId": "integer",
        "name": "string",
        "affiliation": "string",
        "expertise": ["string"],
        "matchScore": "float",
        "availability": "string",
        "recentReviews": "integer"
      }
    ]
  }
}
```

## WebSocket Events

### Real-time Notifications

#### Connection
```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
ws.send(JSON.stringify({
  type: 'authenticate',
  token: 'jwt_token'
}));
```

#### Event Types

**Submission Status Update:**
```json
{
  "type": "submission_status_update",
  "data": {
    "submissionId": "integer",
    "newStatus": "string",
    "message": "string",
    "timestamp": "ISO_date"
  }
}
```

**New Review Assignment:**
```json
{
  "type": "review_assignment",
  "data": {
    "reviewId": "integer",
    "submissionId": "integer",
    "submissionTitle": "string",
    "dueDate": "ISO_date"
  }
}
```

**Chat Message:**
```json
{
  "type": "chat_message",
  "data": {
    "sessionId": "string",
    "message": "string",
    "sender": "user|agent",
    "timestamp": "ISO_date"
  }
}
```

## Error Handling

### Standard Error Response
```json
{
  "success": false,
  "error": {
    "code": "string",
    "message": "string",
    "details": {},
    "timestamp": "ISO_date"
  }
}
```

### Common Error Codes
- `AUTH_REQUIRED`: Authentication required
- `AUTH_INVALID`: Invalid authentication token
- `PERMISSION_DENIED`: Insufficient permissions
- `VALIDATION_ERROR`: Request validation failed
- `NOT_FOUND`: Resource not found
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `SERVER_ERROR`: Internal server error

## Rate Limiting

### Limits
- **API Requests**: 1000 requests per hour per user
- **Chatbot Messages**: 100 messages per hour per user
- **File Uploads**: 10 uploads per minute per user

### Headers
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp

## Data Validation

### Request Validation
- All requests validated against JSON schemas
- Required fields enforced
- Data type validation
- Length and format constraints

### Response Validation
- Consistent response structure
- Proper HTTP status codes
- Error message standardization
- Data sanitization

## Security Considerations

### API Security
- HTTPS enforcement
- CORS configuration
- Input sanitization
- SQL injection prevention
- XSS protection

### Authentication Security
- JWT token expiration
- Refresh token rotation
- Session management
- Password hashing (bcrypt)
- Multi-factor authentication support

### Chatbot Security
- Input validation and sanitization
- Rate limiting for AI requests
- Conversation logging and monitoring
- Privacy-preserving data handling
- Prompt injection prevention

## Performance Optimization

### Caching Strategy
- Response caching for static data
- Redis for session storage
- CDN for file delivery
- Database query optimization

### Pagination
- Consistent pagination across endpoints
- Configurable page sizes
- Cursor-based pagination for large datasets
- Total count optimization

### File Handling
- Chunked file uploads
- File type validation
- Size limitations
- Virus scanning integration
- Cloud storage integration

This API specification provides a comprehensive foundation for building the React-based web interface while maintaining compatibility with the existing OJS backend and enabling advanced chatbot functionality.

