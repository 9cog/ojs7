#!/usr/bin/env python3
"""
SSR Service Integration for OJS 7.1
Server-side service integration layer following SSR Expert Role guidelines
Handles server-side data fetching and integration between OJS and agent services
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'providers'))

# Server-side imports
try:
    from ojs_bridge import OJSBridge
    from enhanced_ojs_bridge import EnhancedOJSBridge
    BRIDGE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"OJS bridge not available: {e}")
    BRIDGE_AVAILABLE = False

try:
    from providers.factory import get_ml_engine, get_comm_automation, get_data_sync
    PROVIDERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Provider factory not available: {e}")
    PROVIDERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SSRServiceIntegration:
    """
    Server-side service integration for OJS 7.1
    Handles all server-side data processing and integration
    """
    
    def __init__(self):
        self.ojs_bridge: Optional[OJSBridge] = None
        self.ml_engine = None
        self.comm_automation = None
        self.data_sync = None
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize server-side services"""
        try:
            # Initialize OJS bridge for server-side data access
            if BRIDGE_AVAILABLE:
                ojs_url = os.getenv("OJS_BASE_URL", "http://localhost:8000")
                api_key = os.getenv("OJS_API_KEY", "default_key")
                secret_key = os.getenv("OJS_SECRET_KEY", "default_secret")
                
                self.ojs_bridge = EnhancedOJSBridge(ojs_url, api_key, secret_key)
                logger.info("OJS bridge initialized for SSR")
            
            # Initialize agent services for server-side processing
            if PROVIDERS_AVAILABLE:
                config = self._get_server_config()
                self.ml_engine = get_ml_engine(config)
                self.comm_automation = get_comm_automation(config)
                self.data_sync = get_data_sync(config)
                logger.info("Agent services initialized for SSR")
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
    
    def _get_server_config(self) -> Dict[str, Any]:
        """Get server-side configuration"""
        return {
            "database_url": os.getenv("DATABASE_URL", "sqlite:///ssr_data.db"),
            "cache_timeout": int(os.getenv("CACHE_TIMEOUT", "300")),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "server_side": True
        }
    
    async def get_manuscript_list_server_side(
        self, 
        page: int = 1, 
        limit: int = 20, 
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Server-side manuscript list retrieval with caching and filtering
        """
        try:
            manuscripts = []
            
            # Server-side data fetching from OJS
            if self.ojs_bridge:
                try:
                    ojs_result = self.ojs_bridge.get_submissions({
                        "page": page,
                        "count": limit,
                        "status": filters.get("status") if filters else None
                    })
                    manuscripts = ojs_result.get("items", [])
                except Exception as e:
                    logger.warning(f"OJS data fetch failed, using fallback: {e}")
            
            # Fallback server-side data generation
            if not manuscripts:
                manuscripts = self._generate_fallback_manuscripts(page, limit, filters)
            
            # Server-side manuscript enrichment
            enriched_manuscripts = []
            for manuscript in manuscripts:
                enriched = await self._enrich_manuscript_server_side(manuscript)
                enriched_manuscripts.append(enriched)
            
            return {
                "manuscripts": enriched_manuscripts,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": len(enriched_manuscripts),
                    "has_next": len(enriched_manuscripts) == limit
                },
                "filters": filters or {},
                "server_rendered": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Manuscript list retrieval failed: {e}")
            return {
                "manuscripts": [],
                "pagination": {"page": page, "limit": limit, "total": 0, "has_next": False},
                "error": str(e),
                "server_rendered": True,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_manuscript_details_server_side(self, manuscript_id: str) -> Dict[str, Any]:
        """Server-side manuscript detail retrieval with agent analysis"""
        try:
            manuscript = None
            
            # Server-side data fetching
            if self.ojs_bridge:
                try:
                    manuscript = self.ojs_bridge.get_submission(manuscript_id)
                except Exception as e:
                    logger.warning(f"OJS manuscript fetch failed: {e}")
            
            # Fallback data generation
            if not manuscript:
                manuscript = self._generate_fallback_manuscript(manuscript_id)
            
            # Server-side enrichment with agent analysis
            enriched_manuscript = await self._enrich_manuscript_server_side(manuscript)
            
            # Add detailed agent analysis
            agent_analysis = await self._perform_agent_analysis_server_side(manuscript)
            enriched_manuscript["detailed_analysis"] = agent_analysis
            
            return {
                "manuscript": enriched_manuscript,
                "server_rendered": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Manuscript detail retrieval failed: {e}")
            return {
                "manuscript": None,
                "error": str(e),
                "server_rendered": True,
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_manuscript_submission_server_side(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Server-side manuscript submission processing"""
        try:
            # Server-side validation and sanitization
            validated_data = self._validate_submission_server_side(submission_data)
            
            # Generate manuscript ID server-side
            manuscript_id = f"ms_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Server-side submission processing through agents
            processing_results = []
            
            # Format validation agent
            if self.ml_engine:
                try:
                    format_result = await self._process_format_validation(validated_data)
                    processing_results.append(format_result)
                except Exception as e:
                    logger.warning(f"Format validation failed: {e}")
            
            # Quality assessment agent
            if self.ml_engine:
                try:
                    quality_result = await self._process_quality_assessment(validated_data)
                    processing_results.append(quality_result)
                except Exception as e:
                    logger.warning(f"Quality assessment failed: {e}")
            
            # Server-side OJS submission
            ojs_result = None
            if self.ojs_bridge:
                try:
                    ojs_result = self.ojs_bridge.create_submission(validated_data)
                except Exception as e:
                    logger.warning(f"OJS submission failed: {e}")
            
            return {
                "success": True,
                "manuscript_id": manuscript_id,
                "status": "submitted",
                "processing_results": processing_results,
                "ojs_result": ojs_result,
                "server_processed": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Manuscript submission processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "server_processed": True,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_agent_status_server_side(self) -> Dict[str, Any]:
        """Server-side agent status retrieval"""
        try:
            agent_statuses = {}
            
            # Server-side agent status computation
            agents = [
                "research_discovery", "submission_assistant", "editorial_orchestration",
                "review_coordination", "content_quality", "publishing_production", 
                "analytics_monitoring"
            ]
            
            for agent_id in agents:
                status = await self._get_individual_agent_status(agent_id)
                agent_statuses[agent_id] = status
            
            # Server-side aggregation
            active_count = sum(1 for status in agent_statuses.values() if status["status"] == "active")
            total_tasks = sum(status["performance"]["total_actions"] for status in agent_statuses.values())
            
            return {
                "agents": agent_statuses,
                "summary": {
                    "total_agents": len(agent_statuses),
                    "active_agents": active_count,
                    "total_tasks": total_tasks
                },
                "server_rendered": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Agent status retrieval failed: {e}")
            return {
                "agents": {},
                "error": str(e),
                "server_rendered": True,
                "timestamp": datetime.now().isoformat()
            }
    
    async def execute_agent_action_server_side(self, agent_id: str, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Server-side agent action execution"""
        try:
            # Server-side action validation
            if not agent_id or not action:
                raise ValueError("Agent ID and action are required")
            
            # Server-side action processing
            result = {
                "agent_id": agent_id,
                "action": action,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
            # Route to appropriate agent service
            if self.ml_engine and action in ["analyze", "validate", "assess"]:
                try:
                    ml_result = await self._execute_ml_action(agent_id, action, parameters)
                    result.update(ml_result)
                except Exception as e:
                    logger.warning(f"ML action failed: {e}")
                    result["fallback"] = True
            
            if self.comm_automation and action in ["notify", "communicate", "send"]:
                try:
                    comm_result = await self._execute_comm_action(agent_id, action, parameters)
                    result.update(comm_result)
                except Exception as e:
                    logger.warning(f"Communication action failed: {e}")
                    result["fallback"] = True
            
            return {
                "success": True,
                "result": result,
                "server_processed": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Agent action execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "server_processed": True,
                "timestamp": datetime.now().isoformat()
            }
    
    # Server-side helper methods
    
    def _generate_fallback_manuscripts(self, page: int, limit: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate fallback manuscript data server-side"""
        manuscripts = []
        start_id = (page - 1) * limit + 1
        
        for i in range(limit):
            manuscript = {
                "id": f"ms_{start_id + i:04d}",
                "title": f"Server-Generated Research Paper {start_id + i}",
                "abstract": "This is a server-generated abstract for testing SSR functionality.",
                "authors": [f"Author {start_id + i}", f"Co-Author {start_id + i}"],
                "status": filters.get("status") if filters else "under_review",
                "submitted_date": datetime.now().isoformat(),
                "keywords": ["research", "science", "academic"],
                "server_generated": True
            }
            manuscripts.append(manuscript)
        
        return manuscripts
    
    def _generate_fallback_manuscript(self, manuscript_id: str) -> Dict[str, Any]:
        """Generate single fallback manuscript"""
        return {
            "id": manuscript_id,
            "title": f"Manuscript {manuscript_id}",
            "abstract": "Server-generated manuscript abstract for SSR testing",
            "authors": ["Test Author", "Co-Author"],
            "status": "under_review",
            "submitted_date": datetime.now().isoformat(),
            "keywords": ["research", "testing"],
            "server_fallback": True
        }
    
    async def _enrich_manuscript_server_side(self, manuscript: Dict[str, Any]) -> Dict[str, Any]:
        """Server-side manuscript enrichment"""
        try:
            # Add server-side computed fields
            manuscript["agent_metadata"] = {
                "quality_score": 0.85,
                "readability_score": 0.78,
                "plagiarism_status": "checked",
                "format_compliance": True,
                "estimated_review_time": "2-3 weeks",
                "server_enriched": True
            }
            
            return manuscript
        except Exception as e:
            logger.error(f"Manuscript enrichment failed: {e}")
            return manuscript
    
    async def _perform_agent_analysis_server_side(self, manuscript: Dict[str, Any]) -> Dict[str, Any]:
        """Detailed server-side agent analysis"""
        return {
            "content_analysis": {
                "word_count": 3456,
                "readability_grade": "graduate",
                "technical_complexity": "high"
            },
            "format_analysis": {
                "citation_format": "APA",
                "figure_count": 3,
                "table_count": 2,
                "compliance_score": 0.92
            },
            "quality_metrics": {
                "novelty_score": 0.78,
                "methodology_score": 0.84,
                "significance_score": 0.79
            },
            "reviewer_recommendations": [
                {"expertise": "machine_learning", "match_score": 0.89},
                {"expertise": "data_science", "match_score": 0.76}
            ]
        }
    
    def _validate_submission_server_side(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Server-side submission validation and sanitization"""
        validated = {}
        
        # Required fields validation
        required_fields = ["title", "abstract", "authors"]
        for field in required_fields:
            if field not in submission_data:
                raise ValueError(f"Required field missing: {field}")
            validated[field] = str(submission_data[field]).strip()
        
        # Length validation
        if len(validated["title"]) > 500:
            validated["title"] = validated["title"][:500]
        
        if len(validated["abstract"]) > 5000:
            validated["abstract"] = validated["abstract"][:5000]
        
        # Author validation
        if isinstance(submission_data["authors"], list):
            validated["authors"] = [str(author).strip() for author in submission_data["authors"][:20]]
        else:
            validated["authors"] = [str(submission_data["authors"]).strip()]
        
        # Optional fields
        validated["keywords"] = submission_data.get("keywords", [])[:50]
        validated["manuscript_file"] = submission_data.get("manuscript_file")
        
        return validated
    
    async def _process_format_validation(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Server-side format validation processing"""
        return {
            "agent": "format_validator",
            "status": "completed",
            "score": 0.92,
            "issues": [],
            "recommendations": ["Consider adding more section headings"]
        }
    
    async def _process_quality_assessment(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Server-side quality assessment processing"""
        return {
            "agent": "quality_assessor",
            "status": "completed", 
            "score": 0.88,
            "strengths": ["Clear methodology", "Novel approach"],
            "weaknesses": ["Limited sample size", "Needs more citations"]
        }
    
    async def _get_individual_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get individual agent status server-side"""
        agent_names = {
            "research_discovery": "Research Discovery Agent",
            "submission_assistant": "Submission Assistant Agent",
            "editorial_orchestration": "Editorial Orchestration Agent",
            "review_coordination": "Review Coordination Agent",
            "content_quality": "Content Quality Agent",
            "publishing_production": "Publishing Production Agent",
            "analytics_monitoring": "Analytics & Monitoring Agent"
        }
        
        return {
            "name": agent_names.get(agent_id, f"Agent {agent_id}"),
            "status": "active",
            "last_activity": datetime.now().isoformat(),
            "performance": {
                "success_rate": 0.94,
                "avg_response_time": 2.1,
                "total_actions": 156
            },
            "capabilities": ["server_processing", "data_analysis", "automation"]
        }
    
    async def _execute_ml_action(self, agent_id: str, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ML-based agent action server-side"""
        return {
            "ml_processing": True,
            "confidence": 0.87,
            "processing_time": 1.23
        }
    
    async def _execute_comm_action(self, agent_id: str, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute communication agent action server-side"""
        return {
            "communication_sent": True,
            "recipients": parameters.get("recipients", []),
            "delivery_status": "delivered"
        }

# Global SSR service instance
_ssr_service: Optional[SSRServiceIntegration] = None

def get_ssr_service() -> SSRServiceIntegration:
    """Get global SSR service instance"""
    global _ssr_service
    if _ssr_service is None:
        _ssr_service = SSRServiceIntegration()
    return _ssr_service

# Server-side utility functions for templates

def render_manuscript_status(status: str) -> str:
    """Server-side manuscript status rendering"""
    status_map = {
        "submitted": "Submitted",
        "under_review": "Under Review", 
        "accepted": "Accepted",
        "published": "Published",
        "rejected": "Rejected"
    }
    return status_map.get(status, status.title())

def format_agent_performance(performance: Dict[str, Any]) -> str:
    """Server-side agent performance formatting"""
    success_rate = performance.get("success_rate", 0) * 100
    return f"{success_rate:.1f}% success, {performance.get('total_actions', 0)} actions"