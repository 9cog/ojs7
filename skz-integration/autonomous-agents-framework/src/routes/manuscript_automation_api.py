"""
API Routes for Manuscript Processing Automation
Provides RESTful endpoints for manuscript automation control and monitoring
"""
from flask import Blueprint, request, jsonify, current_app
import asyncio
import logging
from typing import Dict, Any
import json
from datetime import datetime

from src.models.manuscript_processing_automation import (
    ManuscriptProcessingAutomation, 
    AutomationPriority,
    ManuscriptStatus
)

logger = logging.getLogger(__name__)

# Create blueprint for manuscript automation API
manuscript_automation_bp = Blueprint('manuscript_automation', __name__, url_prefix='/api/v1/automation')

# Global automation instance - would be configured through dependency injection in production
automation_instance = None

def init_automation(config: Dict[str, Any]):
    """Initialize automation instance with configuration"""
    global automation_instance
    automation_instance = ManuscriptProcessingAutomation(config)
    logger.info("Manuscript Processing Automation initialized")

@manuscript_automation_bp.route('/submit', methods=['POST'])
def submit_manuscript():
    """
    Submit a manuscript for automated processing
    
    Expected JSON payload:
    {
        "id": "manuscript_123",
        "title": "Novel Cosmetic Formulation Study",
        "authors": [{"name": "Dr. Smith", "email": "smith@example.com"}],
        "abstract": "This study investigates...",
        "keywords": ["cosmetics", "formulation", "safety"],
        "research_type": "experimental",
        "field_of_study": "cosmetic_science",
        "file_paths": ["/uploads/manuscript.pdf"],
        "priority": 2,
        "special_requirements": ["inci_verification"]
    }
    
    Returns:
    {
        "success": true,
        "workflow_id": "uuid-string",
        "status": "received",
        "estimated_completion": "2024-01-15T18:30:00Z"
    }
    """
    try:
        if not automation_instance:
            return jsonify({
                'success': False,
                'error': 'Automation system not initialized'
            }), 500
        
        manuscript_data = request.get_json()
        if not manuscript_data:
            return jsonify({
                'success': False,
                'error': 'No manuscript data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['title', 'authors', 'abstract']
        for field in required_fields:
            if field not in manuscript_data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Submit manuscript for automation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            workflow_id = loop.run_until_complete(
                automation_instance.submit_manuscript_for_automation(manuscript_data)
            )
        finally:
            loop.close()
        
        # Get workflow status for response
        workflow_status = automation_instance.get_workflow_status(workflow_id)
        
        return jsonify({
            'success': True,
            'workflow_id': workflow_id,
            'status': workflow_status['status'],
            'estimated_completion': workflow_status['estimated_completion'],
            'current_stage': workflow_status['current_stage'],
            'progress_percentage': workflow_status['progress_percentage']
        }), 200
        
    except Exception as e:
        logger.error(f"Error submitting manuscript for automation: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@manuscript_automation_bp.route('/status/<workflow_id>', methods=['GET'])
def get_workflow_status(workflow_id: str):
    """
    Get current status of a manuscript processing workflow
    
    Returns:
    {
        "success": true,
        "workflow_id": "uuid-string",
        "manuscript_id": "manuscript_123",
        "status": "processing",
        "current_stage": "quality_assessment",
        "progress_percentage": 45.0,
        "estimated_completion": "2024-01-15T18:30:00Z",
        "tasks": [...]
    }
    """
    try:
        if not automation_instance:
            return jsonify({
                'success': False,
                'error': 'Automation system not initialized'
            }), 500
        
        workflow_status = automation_instance.get_workflow_status(workflow_id)
        
        if not workflow_status:
            return jsonify({
                'success': False,
                'error': f'Workflow not found: {workflow_id}'
            }), 404
        
        return jsonify({
            'success': True,
            **workflow_status
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting workflow status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@manuscript_automation_bp.route('/metrics', methods=['GET'])
def get_automation_metrics():
    """
    Get automation system performance metrics
    
    Returns:
    {
        "success": true,
        "performance_metrics": {
            "total_processed": 150,
            "success_rate": 0.94,
            "average_processing_time": 120.5,
            "automation_efficiency": 0.87,
            "error_rate": 0.06
        },
        "active_workflows": 8,
        "queue_length": 3,
        "completed_workflows": 142
    }
    """
    try:
        if not automation_instance:
            return jsonify({
                'success': False,
                'error': 'Automation system not initialized'
            }), 500
        
        metrics = automation_instance.get_automation_metrics()
        
        return jsonify({
            'success': True,
            **metrics
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting automation metrics: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@manuscript_automation_bp.route('/workflows', methods=['GET'])
def list_workflows():
    """
    List all workflows with optional filtering
    
    Query parameters:
    - status: filter by status (received, processing, completed, error)
    - limit: maximum number of workflows to return (default: 50)
    - offset: number of workflows to skip (default: 0)
    
    Returns:
    {
        "success": true,
        "workflows": [...],
        "total_count": 145,
        "page_info": {
            "limit": 50,
            "offset": 0,
            "has_more": true
        }
    }
    """
    try:
        if not automation_instance:
            return jsonify({
                'success': False,
                'error': 'Automation system not initialized'
            }), 500
        
        # Get query parameters
        status_filter = request.args.get('status')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        # Get all workflows
        all_workflows = []
        all_workflows.extend(automation_instance.active_workflows.values())
        all_workflows.extend(automation_instance.completed_workflows.values())
        
        # Apply status filter
        if status_filter:
            all_workflows = [w for w in all_workflows if w.status.value == status_filter]
        
        # Sort by created_at (newest first)
        all_workflows.sort(key=lambda w: w.created_at, reverse=True)
        
        # Apply pagination
        total_count = len(all_workflows)
        workflows_page = all_workflows[offset:offset + limit]
        
        # Convert to response format
        workflows_data = []
        for workflow in workflows_page:
            workflows_data.append({
                'workflow_id': workflow.workflow_id,
                'manuscript_id': workflow.manuscript_id,
                'title': workflow.manuscript_metadata.title,
                'status': workflow.status.value,
                'current_stage': workflow.current_stage.value,
                'progress_percentage': workflow.progress_percentage,
                'created_at': workflow.created_at,
                'updated_at': workflow.updated_at
            })
        
        return jsonify({
            'success': True,
            'workflows': workflows_data,
            'total_count': total_count,
            'page_info': {
                'limit': limit,
                'offset': offset,
                'has_more': (offset + limit) < total_count
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing workflows: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@manuscript_automation_bp.route('/config', methods=['GET'])
def get_automation_config():
    """
    Get current automation system configuration
    
    Returns:
    {
        "success": true,
        "config": {
            "agent_endpoints": {...},
            "routing_rules": {...},
            "performance_thresholds": {...}
        }
    }
    """
    try:
        if not automation_instance:
            return jsonify({
                'success': False,
                'error': 'Automation system not initialized'
            }), 500
        
        return jsonify({
            'success': True,
            'config': {
                'agent_endpoints': automation_instance.agent_endpoints,
                'routing_rules': automation_instance.routing_rules,
                'performance_metrics': automation_instance.performance_metrics
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting automation config: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@manuscript_automation_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for automation system
    
    Returns:
    {
        "success": true,
        "status": "healthy",
        "timestamp": "2024-01-15T12:30:45Z",
        "system_info": {
            "automation_initialized": true,
            "active_workflows": 8,
            "agent_endpoints_available": 7
        }
    }
    """
    try:
        system_healthy = True
        system_info = {
            'automation_initialized': automation_instance is not None,
            'active_workflows': len(automation_instance.active_workflows) if automation_instance else 0,
            'agent_endpoints_available': len(automation_instance.agent_endpoints) if automation_instance else 0
        }
        
        if not automation_instance:
            system_healthy = False
        
        return jsonify({
            'success': True,
            'status': 'healthy' if system_healthy else 'degraded',
            'timestamp': datetime.utcnow().isoformat(),
            'system_info': system_info
        }), 200 if system_healthy else 503
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 500

# Error handlers for the blueprint
@manuscript_automation_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request',
        'message': str(error)
    }), 400

@manuscript_automation_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Not found',
        'message': str(error)
    }), 404

@manuscript_automation_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': str(error)
    }), 500