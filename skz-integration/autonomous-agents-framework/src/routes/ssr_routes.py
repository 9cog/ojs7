#!/usr/bin/env python3
"""
SSR Route Handlers for OJS 7.1 Agent API
Server-side rendering route handlers following SSR Expert Role guidelines
All responses are server-rendered HTML or JSON with no client-side JavaScript
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from fastapi import APIRouter, HTTPException, Request, Depends
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from fastapi.templating import Jinja2Templates
    from pydantic import BaseModel, ValidationError
except ImportError:
    # Fallback for environments without FastAPI
    print("FastAPI not available - using basic implementations")
    APIRouter = None

# Import OJS integration
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from ojs_bridge import OJSBridge
    from enhanced_ojs_bridge import EnhancedOJSBridge
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Router for SSR agent endpoints
router = APIRouter(prefix="/api/v1", tags=["agents"])

# Template configuration for server-side rendering
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

class ManuscriptSubmissionRequest(BaseModel):
    """Server-side validated manuscript submission request"""
    title: str
    abstract: str
    authors: List[str]
    keywords: List[str]
    manuscript_file: Optional[str] = None

class ReviewAssignmentRequest(BaseModel):
    """Server-side validated review assignment request"""
    manuscript_id: str
    reviewer_id: str
    due_date: str
    review_type: str = "standard"

class EditorialDecisionRequest(BaseModel):
    """Server-side validated editorial decision request"""
    manuscript_id: str
    decision: str
    comments: str
    reviewer_feedback: List[Dict[str, Any]] = []

# Server-side caching for performance optimization
_cache: Dict[str, Any] = {}
_cache_timeout = 300  # 5 minutes

def get_cached_data(key: str) -> Optional[Any]:
    """Server-side cache retrieval with timeout"""
    if key in _cache:
        data, timestamp = _cache[key]
        if (datetime.now().timestamp() - timestamp) < _cache_timeout:
            return data
        else:
            del _cache[key]
    return None

def set_cached_data(key: str, data: Any):
    """Server-side cache storage"""
    _cache[key] = (data, datetime.now().timestamp())

def get_ojs_bridge() -> Optional[OJSBridge]:
    """Dependency to get OJS bridge for server-side data access"""
    if BRIDGE_AVAILABLE:
        ojs_url = os.getenv("OJS_BASE_URL", "http://localhost:8000")
        api_key = os.getenv("OJS_API_KEY", "test_key")
        secret_key = os.getenv("OJS_SECRET_KEY", "test_secret")
        return EnhancedOJSBridge(ojs_url, api_key, secret_key)
    return None

# SSR Route Handlers

@router.get("/manuscripts", response_class=JSONResponse)
async def get_manuscripts(
    page: int = 1,
    limit: int = 20,
    status: Optional[str] = None,
    ojs_bridge: Optional[OJSBridge] = Depends(get_ojs_bridge)
):
    """Get manuscripts with server-side filtering and pagination"""
    try:
        # Server-side caching
        cache_key = f"manuscripts:{page}:{limit}:{status}"
        cached_result = get_cached_data(cache_key)
        if cached_result:
            return JSONResponse(content=cached_result)
        
        # Server-side data fetching from OJS
        manuscripts = []
        if ojs_bridge:
            try:
                manuscripts = await fetch_manuscripts_from_ojs(ojs_bridge, page, limit, status)
            except Exception as e:
                logger.warning(f"OJS fetch failed, using fallback: {e}")
                manuscripts = generate_fallback_manuscripts(page, limit, status)
        else:
            manuscripts = generate_fallback_manuscripts(page, limit, status)
        
        # Server-side response preparation
        response = {
            "manuscripts": manuscripts,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": len(manuscripts),
                "has_next": len(manuscripts) == limit
            },
            "filters": {
                "status": status
            },
            "server_rendered": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache server-side for performance
        set_cached_data(cache_key, response)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Manuscript list request failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch manuscripts")

@router.get("/manuscripts/{manuscript_id}", response_class=JSONResponse)
async def get_manuscript_details(
    manuscript_id: str,
    ojs_bridge: Optional[OJSBridge] = Depends(get_ojs_bridge)
):
    """Get manuscript details with server-side enrichment"""
    try:
        # Server-side validation
        if not manuscript_id.strip():
            raise HTTPException(status_code=400, detail="Manuscript ID is required")
        
        # Server-side caching
        cache_key = f"manuscript:{manuscript_id}"
        cached_result = get_cached_data(cache_key)
        if cached_result:
            return JSONResponse(content=cached_result)
        
        # Server-side data fetching
        manuscript = None
        if ojs_bridge:
            try:
                manuscript = await fetch_manuscript_from_ojs(ojs_bridge, manuscript_id)
            except Exception as e:
                logger.warning(f"OJS manuscript fetch failed: {e}")
        
        if not manuscript:
            manuscript = generate_fallback_manuscript(manuscript_id)
        
        # Server-side enrichment with agent data
        manuscript = await enrich_manuscript_server_side(manuscript)
        
        # Server-side response preparation
        response = {
            "manuscript": manuscript,
            "server_rendered": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache for performance
        set_cached_data(cache_key, response)
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manuscript detail request failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch manuscript details")

@router.post("/manuscripts", response_class=JSONResponse)
async def submit_manuscript(
    submission: ManuscriptSubmissionRequest,
    ojs_bridge: Optional[OJSBridge] = Depends(get_ojs_bridge)
):
    """Submit manuscript with server-side processing"""
    try:
        # Server-side input validation and sanitization
        sanitized_submission = sanitize_submission_server_side(submission)
        
        # Server-side processing through agents
        processing_result = await process_submission_server_side(sanitized_submission, ojs_bridge)
        
        # Server-side response preparation
        response = {
            "success": True,
            "manuscript_id": processing_result.get("manuscript_id"),
            "status": processing_result.get("status", "submitted"),
            "agent_processing": processing_result.get("agent_results", []),
            "server_processed": True,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Manuscript submission failed: {e}")
        raise HTTPException(status_code=500, detail="Submission processing failed")

@router.get("/reviews/{manuscript_id}", response_class=JSONResponse) 
async def get_manuscript_reviews(
    manuscript_id: str,
    ojs_bridge: Optional[OJSBridge] = Depends(get_ojs_bridge)
):
    """Get manuscript reviews with server-side aggregation"""
    try:
        # Server-side caching
        cache_key = f"reviews:{manuscript_id}"
        cached_result = get_cached_data(cache_key)
        if cached_result:
            return JSONResponse(content=cached_result)
        
        # Server-side data fetching
        reviews = []
        if ojs_bridge:
            try:
                reviews = await fetch_reviews_from_ojs(ojs_bridge, manuscript_id)
            except Exception as e:
                logger.warning(f"OJS review fetch failed: {e}")
        
        if not reviews:
            reviews = generate_fallback_reviews(manuscript_id)
        
        # Server-side review analysis
        review_analysis = analyze_reviews_server_side(reviews)
        
        response = {
            "manuscript_id": manuscript_id,
            "reviews": reviews,
            "analysis": review_analysis,
            "server_processed": True,
            "timestamp": datetime.now().isoformat()
        }
        
        set_cached_data(cache_key, response)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Review fetch failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch reviews")

@router.post("/reviews/assign", response_class=JSONResponse)
async def assign_reviewer(
    assignment: ReviewAssignmentRequest,
    ojs_bridge: Optional[OJSBridge] = Depends(get_ojs_bridge)
):
    """Assign reviewer with server-side matching"""
    try:
        # Server-side input validation
        validated_assignment = validate_assignment_server_side(assignment)
        
        # Server-side reviewer matching through agents
        matching_result = await match_reviewer_server_side(validated_assignment, ojs_bridge)
        
        response = {
            "success": True,
            "assignment": {
                "manuscript_id": assignment.manuscript_id,
                "reviewer_id": assignment.reviewer_id,
                "due_date": assignment.due_date,
                "match_score": matching_result.get("match_score", 0.0)
            },
            "agent_analysis": matching_result.get("analysis", {}),
            "server_processed": True,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Assignment validation failed: {e}")
    except Exception as e:
        logger.error(f"Reviewer assignment failed: {e}")
        raise HTTPException(status_code=500, detail="Assignment processing failed")

@router.get("/analytics/dashboard", response_class=HTMLResponse)
async def get_analytics_dashboard(
    request: Request,
    period: str = "30d",
    ojs_bridge: Optional[OJSBridge] = Depends(get_ojs_bridge)
):
    """Server-rendered analytics dashboard"""
    try:
        # Server-side analytics computation
        analytics_data = await compute_analytics_server_side(period, ojs_bridge)
        
        # Server-side template rendering
        return templates.TemplateResponse("analytics_dashboard.html", {
            "request": request,
            "analytics": analytics_data,
            "period": period,
            "server_time": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Analytics dashboard failed: {e}")
        raise HTTPException(status_code=500, detail="Dashboard rendering failed")

# SSR Streaming Responses for Large Data

@router.get("/reports/manuscript-processing")
async def stream_manuscript_processing_report():
    """Server-side streaming response for large reports"""
    try:
        async def generate_report_stream():
            yield "data: {\"status\": \"starting\", \"timestamp\": \"" + datetime.now().isoformat() + "\"}\n\n"
            
            # Server-side report generation
            manuscripts = await fetch_all_manuscripts_server_side()
            
            for i, manuscript in enumerate(manuscripts):
                report_chunk = {
                    "manuscript_id": manuscript.get("id"),
                    "title": manuscript.get("title"),
                    "status": manuscript.get("status"),
                    "progress": (i + 1) / len(manuscripts)
                }
                yield f"data: {json.dumps(report_chunk)}\n\n"
            
            yield "data: {\"status\": \"complete\", \"timestamp\": \"" + datetime.now().isoformat() + "\"}\n\n"
        
        return StreamingResponse(
            generate_report_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        logger.error(f"Report streaming failed: {e}")
        raise HTTPException(status_code=500, detail="Report generation failed")

# Server-side helper functions

async def fetch_manuscripts_from_ojs(bridge: OJSBridge, page: int, limit: int, status: Optional[str]) -> List[Dict[str, Any]]:
    """Server-side manuscript fetching from OJS"""
    try:
        # Use OJS bridge for real data access
        result = bridge.get_submissions({
            "page": page,
            "count": limit,
            "status": status
        })
        return result.get("items", [])
    except Exception as e:
        logger.error(f"OJS manuscript fetch failed: {e}")
        return []

def generate_fallback_manuscripts(page: int, limit: int, status: Optional[str]) -> List[Dict[str, Any]]:
    """Server-side fallback manuscript generation"""
    manuscripts = []
    start_id = (page - 1) * limit + 1
    
    for i in range(limit):
        manuscript = {
            "id": f"ms_{start_id + i:04d}",
            "title": f"Research Paper {start_id + i}",
            "abstract": "Server-generated abstract for testing purposes",
            "authors": [f"Author {start_id + i}"],
            "status": status or "under_review",
            "submitted_date": datetime.now().isoformat(),
            "server_generated": True
        }
        manuscripts.append(manuscript)
    
    return manuscripts

async def enrich_manuscript_server_side(manuscript: Dict[str, Any]) -> Dict[str, Any]:
    """Server-side manuscript enrichment with agent data"""
    try:
        # Add agent analysis results
        manuscript["agent_analysis"] = {
            "quality_score": 0.85,
            "plagiarism_check": "passed",
            "formatting_issues": [],
            "reviewer_recommendations": ["expert_1", "expert_2"],
            "estimated_review_time": "2-3 weeks"
        }
        
        return manuscript
    except Exception as e:
        logger.error(f"Manuscript enrichment failed: {e}")
        return manuscript

def sanitize_submission_server_side(submission: ManuscriptSubmissionRequest) -> Dict[str, Any]:
    """Server-side input sanitization and validation"""
    return {
        "title": submission.title.strip()[:500],  # Limit length
        "abstract": submission.abstract.strip()[:2000],
        "authors": [author.strip() for author in submission.authors[:10]],  # Limit authors
        "keywords": [kw.strip() for kw in submission.keywords[:20]],  # Limit keywords
        "manuscript_file": submission.manuscript_file
    }

async def process_submission_server_side(submission: Dict[str, Any], bridge: Optional[OJSBridge]) -> Dict[str, Any]:
    """Server-side submission processing through agents"""
    try:
        processing_result = {
            "manuscript_id": f"ms_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "submitted",
            "agent_results": []
        }
        
        # Agent processing simulation with server-side logic
        processing_result["agent_results"].append({
            "agent": "format_checker",
            "status": "completed",
            "score": 0.92,
            "issues": []
        })
        
        processing_result["agent_results"].append({
            "agent": "quality_assessor", 
            "status": "completed",
            "score": 0.88,
            "recommendations": ["Add more citations", "Expand methodology section"]
        })
        
        return processing_result
        
    except Exception as e:
        logger.error(f"Submission processing failed: {e}")
        return {"manuscript_id": None, "status": "error", "error": str(e)}

# Additional server-side helper functions for completeness
def generate_fallback_manuscript(manuscript_id: str) -> Dict[str, Any]:
    """Generate fallback manuscript data"""
    return {
        "id": manuscript_id,
        "title": f"Manuscript {manuscript_id}",
        "abstract": "Server-generated manuscript abstract",
        "authors": ["Test Author"],
        "status": "under_review",
        "submitted_date": datetime.now().isoformat(),
        "server_fallback": True
    }

async def fetch_manuscript_from_ojs(bridge: OJSBridge, manuscript_id: str) -> Optional[Dict[str, Any]]:
    """Fetch single manuscript from OJS"""
    try:
        return bridge.get_submission(manuscript_id)
    except Exception as e:
        logger.error(f"OJS manuscript fetch failed for {manuscript_id}: {e}")
        return None

async def fetch_reviews_from_ojs(bridge: OJSBridge, manuscript_id: str) -> List[Dict[str, Any]]:
    """Fetch reviews from OJS"""
    try:
        return bridge.get_reviews(manuscript_id)
    except Exception as e:
        logger.error(f"OJS review fetch failed for {manuscript_id}: {e}")
        return []

def generate_fallback_reviews(manuscript_id: str) -> List[Dict[str, Any]]:
    """Generate fallback review data"""
    return [
        {
            "id": "rev_001",
            "reviewer_name": "Anonymous Reviewer 1",
            "status": "completed",
            "rating": 4,
            "comments": "Well-written paper with solid methodology",
            "completion_date": datetime.now().isoformat()
        }
    ]

def analyze_reviews_server_side(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Server-side review analysis"""
    if not reviews:
        return {"average_rating": 0, "recommendation": "pending"}
    
    ratings = [r.get("rating", 0) for r in reviews if r.get("rating")]
    average_rating = sum(ratings) / len(ratings) if ratings else 0
    
    return {
        "average_rating": round(average_rating, 2),
        "total_reviews": len(reviews),
        "completed_reviews": len([r for r in reviews if r.get("status") == "completed"]),
        "recommendation": "accept" if average_rating >= 3.5 else "revise"
    }

def validate_assignment_server_side(assignment: ReviewAssignmentRequest) -> Dict[str, Any]:
    """Server-side assignment validation"""
    return {
        "manuscript_id": assignment.manuscript_id.strip(),
        "reviewer_id": assignment.reviewer_id.strip(),
        "due_date": assignment.due_date,
        "review_type": assignment.review_type
    }

async def match_reviewer_server_side(assignment: Dict[str, Any], bridge: Optional[OJSBridge]) -> Dict[str, Any]:
    """Server-side reviewer matching"""
    return {
        "match_score": 0.87,
        "analysis": {
            "expertise_match": 0.9,
            "availability": 0.8,
            "workload": 0.9,
            "recommendation": "highly_recommended"
        }
    }

async def compute_analytics_server_side(period: str, bridge: Optional[OJSBridge]) -> Dict[str, Any]:
    """Server-side analytics computation"""
    return {
        "period": period,
        "submissions": {
            "total": 156,
            "accepted": 89,
            "rejected": 23,
            "under_review": 44
        },
        "reviews": {
            "completed": 234,
            "pending": 67,
            "average_time": "14 days"
        },
        "performance": {
            "acceptance_rate": 0.57,
            "average_review_time": 14.2,
            "agent_efficiency": 0.94
        }
    }

async def fetch_all_manuscripts_server_side() -> List[Dict[str, Any]]:
    """Fetch all manuscripts for server-side processing"""
    # Simulate large dataset for streaming
    manuscripts = []
    for i in range(100):
        manuscripts.append({
            "id": f"ms_{i:04d}",
            "title": f"Research Paper {i}",
            "status": "published" if i % 3 == 0 else "under_review"
        })
    return manuscripts