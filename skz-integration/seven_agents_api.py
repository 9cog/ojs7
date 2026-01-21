from flask import Blueprint, request, jsonify
from src.models.seven_agents import (
    SevenAgentSystem, AgentCoordinator, WorkflowTask, 
    AgentType, WorkflowStatus, PriorityLevel
)
from src.models.user import db
from datetime import datetime
import json

seven_agents_bp = Blueprint('seven_agents', __name__)

# Initialize the agent coordinator (will be done in app context)
coordinator = None

def init_coordinator():
    """Initialize the agent coordinator within app context"""
    global coordinator
    if coordinator is None:
        coordinator = AgentCoordinator()
    return coordinator

@seven_agents_bp.route('/api/seven-agents/status', methods=['GET'])
def get_system_status():
    """Get comprehensive system status for all 7 agents"""
    try:
        coordinator = init_coordinator()
        status = coordinator.get_system_status()
        return jsonify({
            "success": True,
            "data": status
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@seven_agents_bp.route('/api/seven-agents/agents', methods=['GET'])
def get_all_agents():
    """Get information about all 7 agents"""
    try:
        agents = SevenAgentSystem.query.all()
        agents_data = [agent.to_dict() for agent in agents]
        
        return jsonify({
            "success": True,
            "data": {
                "agents": agents_data,
                "total_count": len(agents_data)
            }
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@seven_agents_bp.route('/api/seven-agents/agents/<agent_type>', methods=['GET'])
def get_agent_by_type(agent_type):
    """Get specific agent information by type"""
    try:
        # Convert string to AgentType enum
        try:
            agent_enum = AgentType(agent_type)
        except ValueError:
            return jsonify({
                "success": False,
                "error": f"Invalid agent type: {agent_type}"
            }), 400
        
        agent = SevenAgentSystem.query.filter_by(agent_type=agent_enum).first()
        if not agent:
            return jsonify({
                "success": False,
                "error": f"Agent not found: {agent_type}"
            }), 404
        
        return jsonify({
            "success": True,
            "data": agent.to_dict()
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@seven_agents_bp.route('/api/seven-agents/agents/<agent_type>/execute', methods=['POST'])
def execute_agent_action(agent_type):
    """Execute a specific action for an agent"""
    try:
        # Convert string to AgentType enum
        try:
            agent_enum = AgentType(agent_type)
        except ValueError:
            return jsonify({
                "success": False,
                "error": f"Invalid agent type: {agent_type}"
            }), 400
        
        agent = SevenAgentSystem.query.filter_by(agent_type=agent_enum).first()
        if not agent:
            return jsonify({
                "success": False,
                "error": f"Agent not found: {agent_type}"
            }), 404
        
        data = request.get_json() or {}
        action_type = data.get('action_type', 'default_action')
        action_data = data.get('data', {})
        
        result = agent.execute_action(action_type, action_data)
        
        # Update database
        db.session.commit()
        
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@seven_agents_bp.route('/api/seven-agents/workflows/execute', methods=['POST'])
def execute_workflow():
    """Execute a complete workflow involving multiple agents"""
    try:
        coordinator = init_coordinator()
        data = request.get_json() or {}
        workflow_type = data.get('workflow_type')
        workflow_data = data.get('data', {})
        
        if not workflow_type:
            return jsonify({
                "success": False,
                "error": "workflow_type is required"
            }), 400
        
        coordinator = init_coordinator()
        result = coordinator.execute_workflow(workflow_type, workflow_data)
        
        # Update database
        db.session.commit()
        
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@seven_agents_bp.route('/api/seven-agents/workflows/manuscript-processing', methods=['POST'])
def execute_manuscript_workflow():
    """Execute manuscript processing workflow"""
    try:
        data = request.get_json() or {}
        
        # Add some default manuscript data if not provided
        manuscript_data = data.get('manuscript', {
            "title": "Novel Anti-Aging Properties of Bakuchiol in Cosmetic Formulations",
            "authors": ["Dr. Sarah Chen", "Prof. Michael Rodriguez"],
            "abstract": "This study investigates the anti-aging properties of Bakuchiol...",
            "keywords": ["bakuchiol", "anti-aging", "cosmetic", "natural", "retinol alternative"]
        })
        
        workflow_data = {
            "manuscript": manuscript_data,
            "submission_type": data.get('submission_type', 'research_article'),
            "priority": data.get('priority', 'standard')
        }
        
        coordinator = init_coordinator()
        result = coordinator.execute_workflow("manuscript_processing", workflow_data)
        
        # Update database
        db.session.commit()
        
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@seven_agents_bp.route('/api/seven-agents/workflows/research-discovery', methods=['POST'])
def execute_research_discovery_workflow():
    """Execute research discovery workflow"""
    try:
        data = request.get_json() or {}
        
        workflow_data = {
            "search_criteria": data.get('search_criteria', {
                "categories": ["anti-aging", "moisturizing", "sun-protection"],
                "safety_threshold": 0.8,
                "novelty_threshold": 0.7
            }),
            "databases": data.get('databases', ["inci", "patents", "pubmed"]),
            "time_range": data.get('time_range', "last_6_months")
        }
        
        coordinator = init_coordinator()
        result = coordinator.execute_workflow("research_discovery", workflow_data)
        
        # Update database
        db.session.commit()
        
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@seven_agents_bp.route('/api/seven-agents/workflows/publication-production', methods=['POST'])
def execute_publication_workflow():
    """Execute publication production workflow"""
    try:
        data = request.get_json() or {}
        
        workflow_data = {
            "manuscript_id": data.get('manuscript_id', 'MS_2024_001'),
            "publication_type": data.get('publication_type', 'standard_article'),
            "distribution_channels": data.get('distribution_channels', [
                "academic_databases", "industry_platforms", "social_media"
            ]),
            "target_audience": data.get('target_audience', [
                "cosmetic_chemists", "dermatologists", "regulatory_professionals"
            ])
        }
        
        coordinator = init_coordinator()
        result = coordinator.execute_workflow("publication_production", workflow_data)
        
        # Update database
        db.session.commit()
        
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@seven_agents_bp.route('/api/seven-agents/performance', methods=['GET'])
def get_performance_metrics():
    """Get comprehensive performance metrics for all agents"""
    try:
        agents = SevenAgentSystem.query.all()
        performance_data = {}
        
        for agent in agents:
            metrics = agent.get_performance_metrics()
            performance_data[agent.agent_type.value] = metrics
        
        # Calculate system-wide metrics
        total_actions = sum(m["total_actions"] for m in performance_data.values())
        total_success = sum(m["successful_actions"] for m in performance_data.values())
        overall_success_rate = (total_success / total_actions) if total_actions > 0 else 0
        
        avg_response_time = sum(m["avg_response_time"] for m in performance_data.values()) / len(performance_data)
        
        return jsonify({
            "success": True,
            "data": {
                "individual_performance": performance_data,
                "system_metrics": {
                    "total_actions": total_actions,
                    "overall_success_rate": overall_success_rate,
                    "average_response_time": avg_response_time,
                    "active_agents": len([a for a in agents if a.status == "active"])
                },
                "last_updated": datetime.utcnow().isoformat()
            }
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@seven_agents_bp.route('/api/seven-agents/interactions', methods=['GET'])
def get_agent_interactions():
    """Get agent interaction patterns and coordination data"""
    try:
        agents = SevenAgentSystem.query.all()
        interaction_data = {}
        
        for agent in agents:
            interaction_data[agent.agent_type.value] = {
                "coordination_rules": agent.coordination_rules,
                "interaction_patterns": agent.interaction_patterns,
                "capabilities": agent.capabilities
            }
        
        return jsonify({
            "success": True,
            "data": {
                "agent_interactions": interaction_data,
                "coordination_overview": {
                    "hierarchical_agents": [
                        "editorial_orchestration",
                        "content_quality", 
                        "analytics_monitoring"
                    ],
                    "distributed_agents": [
                        "research_discovery",
                        "submission_assistant",
                        "review_coordination",
                        "publishing_production"
                    ],
                    "coordination_frequency": "continuous",
                    "feedback_loops": "enabled"
                }
            }
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@seven_agents_bp.route('/api/seven-agents/simulation', methods=['POST'])
def run_simulation():
    """Run a comprehensive simulation of the 7-agent system"""
    try:
        data = request.get_json() or {}
        simulation_type = data.get('simulation_type', 'full_workflow')
        duration = data.get('duration', 30)  # simulation duration in days
        
        simulation_results = {
            "simulation_type": simulation_type,
            "duration_days": duration,
            "start_time": datetime.utcnow().isoformat(),
            "results": {}
        }
        
        if simulation_type == 'full_workflow':
            # Simulate manuscript processing workflow
            coordinator = init_coordinator()
            manuscript_result = coordinator.execute_workflow("manuscript_processing", {
                "manuscript": {
                    "title": "Comprehensive Study of Novel Peptide Complex for Anti-Aging Applications",
                    "category": "anti-aging",
                    "innovation_score": 0.85
                }
            })
            simulation_results["results"]["manuscript_processing"] = manuscript_result
            
            # Simulate research discovery workflow
            discovery_result = coordinator.execute_workflow("research_discovery", {
                "search_criteria": {"categories": ["peptides", "anti-aging", "clinical-studies"]}
            })
            simulation_results["results"]["research_discovery"] = discovery_result
            
            # Simulate publication workflow
            publication_result = coordinator.execute_workflow("publication_production", {
                "manuscript_id": "SIM_MS_001",
                "publication_type": "feature_article"
            })
            simulation_results["results"]["publication_production"] = publication_result
        
        # Get final system status
        final_status = coordinator.get_system_status()
        simulation_results["final_system_status"] = final_status
        simulation_results["end_time"] = datetime.utcnow().isoformat()
        
        # Update database
        db.session.commit()
        
        return jsonify({
            "success": True,
            "data": simulation_results
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@seven_agents_bp.route('/api/seven-agents/cognitive-architecture', methods=['GET'])
def get_cognitive_architecture():
    """Get cognitive architecture information"""
    try:
        architecture_data = {
            "architecture_type": "hybrid_cognitive_system",
            "design_principles": {
                "hierarchical_priority_management": {
                    "description": "Central coordination for priority decisions",
                    "agents": ["editorial_orchestration", "content_quality", "analytics_monitoring"],
                    "benefits": ["efficient_execution", "consistent_standards", "strategic_alignment"]
                },
                "distributed_innovation_networks": {
                    "description": "Autonomous exploration and collaborative enhancement",
                    "agents": ["research_discovery", "submission_assistant", "review_coordination", "publishing_production"],
                    "benefits": ["innovation_fostering", "adaptive_learning", "creative_exploration"]
                }
            },
            "balance_mechanisms": {
                "priority_vs_novelty": {
                    "coordinator": "editorial_orchestration",
                    "innovator": "research_discovery",
                    "balance_point": "strategic_innovation"
                },
                "efficiency_vs_creativity": {
                    "coordinator": "content_quality",
                    "innovator": "submission_assistant",
                    "balance_point": "quality_enhancement"
                },
                "standards_vs_adaptation": {
                    "coordinator": "analytics_monitoring",
                    "innovator": "review_coordination",
                    "balance_point": "continuous_improvement"
                }
            },
            "feedback_loops": {
                "performance_optimization": {
                    "source": "analytics_monitoring",
                    "targets": "all_agents",
                    "frequency": "continuous"
                },
                "quality_improvement": {
                    "source": "content_quality",
                    "targets": ["submission_assistant", "review_coordination"],
                    "frequency": "per_manuscript"
                },
                "innovation_guidance": {
                    "source": "research_discovery",
                    "targets": ["editorial_orchestration", "submission_assistant"],
                    "frequency": "trend_based"
                }
            },
            "adaptive_learning": {
                "pattern_recognition": "enabled",
                "process_optimization": "continuous",
                "strategic_evolution": "quarterly_reviews"
            }
        }
        
        return jsonify({
            "success": True,
            "data": architecture_data
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@seven_agents_bp.route('/api/seven-agents/health', methods=['GET'])
def health_check():
    """Health check endpoint for the 7-agent system"""
    try:
        # Check database connectivity
        agent_count = SevenAgentSystem.query.count()
        
        # Check agent coordinator
        coordinator = init_coordinator()
        system_status = coordinator.get_system_status()
        
        health_data = {
            "status": "healthy",
            "database_connection": "ok",
            "agent_count": agent_count,
            "coordinator_status": "operational",
            "system_metrics": {
                "total_agents": system_status["total_agents"],
                "active_agents": system_status["active_agents"],
                "overall_success_rate": system_status["overall_success_rate"]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return jsonify({
            "success": True,
            "data": health_data
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "unhealthy"
        }), 500

