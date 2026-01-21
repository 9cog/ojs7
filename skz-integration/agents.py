"""
API Routes for Autonomous Agents Framework
Provides REST endpoints for agent management and interaction
"""

from flask import Blueprint, request, jsonify
from src.models.agent import Agent, Message, WorkflowInstance, AgentCapability, MessageType, agent_registry, db
from src.models.research_agent import ResearchDiscoveryAgent
from src.models.editorial_agent import EditorialOrchestrationAgent
import json
from datetime import datetime, timedelta

agents_bp = Blueprint('agents', __name__)

@agents_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Autonomous Agents Framework"
    })

@agents_bp.route('/agents', methods=['GET'])
def list_agents():
    """List all registered agents"""
    try:
        agents = Agent.query.all()
        return jsonify({
            "agents": [agent.to_dict() for agent in agents],
            "total_count": len(agents),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/agents', methods=['POST'])
def create_agent():
    """Create a new agent instance"""
    try:
        data = request.get_json()
        agent_type = data.get('agent_type')
        name = data.get('name', f"{agent_type}_agent")
        
        # Create specific agent type
        if agent_type == 'research_discovery':
            agent = ResearchDiscoveryAgent(name=name)
        elif agent_type == 'editorial_orchestration':
            agent = EditorialOrchestrationAgent(name=name)
        else:
            # Create generic agent
            capabilities_str = data.get('capabilities', [])
            capabilities = [AgentCapability(cap) for cap in capabilities_str if cap in [c.value for c in AgentCapability]]
            agent = Agent(
                name=name,
                agent_type=agent_type,
                capabilities=capabilities,
                arena_context=data.get('arena_context', {})
            )
        
        # Save to database
        db.session.add(agent)
        db.session.commit()
        
        # Register with agent registry
        agent_registry.register_agent(agent)
        
        return jsonify({
            "message": "Agent created successfully",
            "agent": agent.to_dict()
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/agents/<agent_id>', methods=['GET'])
def get_agent(agent_id):
    """Get specific agent details"""
    try:
        agent = Agent.query.get(agent_id)
        if not agent:
            return jsonify({"error": "Agent not found"}), 404
        
        return jsonify({
            "agent": agent.to_dict(),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/agents/<agent_id>', methods=['DELETE'])
def delete_agent(agent_id):
    """Delete an agent"""
    try:
        agent = Agent.query.get(agent_id)
        if not agent:
            return jsonify({"error": "Agent not found"}), 404
        
        # Unregister from agent registry
        agent_registry.unregister_agent(agent_id)
        
        # Delete from database
        db.session.delete(agent)
        db.session.commit()
        
        return jsonify({
            "message": "Agent deleted successfully",
            "agent_id": agent_id
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/agents/<agent_id>/status', methods=['GET'])
def get_agent_status(agent_id):
    """Get agent status and performance metrics"""
    try:
        agent = Agent.query.get(agent_id)
        if not agent:
            return jsonify({"error": "Agent not found"}), 404
        
        # Get runtime status from registry
        runtime_agent = agent_registry.get_agent(agent_id)
        runtime_status = runtime_agent.status.value if runtime_agent else "offline"
        
        return jsonify({
            "agent_id": agent_id,
            "name": agent.name,
            "status": runtime_status,
            "performance_metrics": agent.get_performance_metrics(),
            "last_heartbeat": agent.last_heartbeat.isoformat(),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/agents/capabilities/<capability>', methods=['GET'])
def get_agents_by_capability(capability):
    """Get agents with specific capability"""
    try:
        # Validate capability
        try:
            cap_enum = AgentCapability(capability)
        except ValueError:
            return jsonify({"error": f"Invalid capability: {capability}"}), 400
        
        agents = agent_registry.get_agents_by_capability(cap_enum)
        
        return jsonify({
            "capability": capability,
            "agents": [agent.to_dict() for agent in agents],
            "count": len(agents),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/messages', methods=['POST'])
def send_message():
    """Send message to an agent"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['recipient_id', 'message_type', 'content']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Create message
        message = Message(
            sender_id=data.get('sender_id', 'api_client'),
            recipient_id=data['recipient_id'],
            message_type=MessageType(data['message_type']),
            content=json.dumps(data['content']) if isinstance(data['content'], dict) else data['content'],
            correlation_id=data.get('correlation_id')
        )
        
        # Save to database
        db.session.add(message)
        db.session.commit()
        
        # Route message to agent
        agent_registry.route_message(message)
        
        return jsonify({
            "message": "Message sent successfully",
            "message_id": message.id,
            "timestamp": datetime.now().isoformat()
        }), 201
        
    except ValueError as e:
        return jsonify({"error": f"Invalid message type: {data.get('message_type')}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/messages/<message_id>', methods=['GET'])
def get_message(message_id):
    """Get message details"""
    try:
        message = Message.query.get(message_id)
        if not message:
            return jsonify({"error": "Message not found"}), 404
        
        return jsonify({
            "message": message.to_dict(),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/messages', methods=['GET'])
def list_messages():
    """List messages with optional filtering"""
    try:
        # Get query parameters
        sender_id = request.args.get('sender_id')
        recipient_id = request.args.get('recipient_id')
        message_type = request.args.get('message_type')
        limit = int(request.args.get('limit', 50))
        
        # Build query
        query = Message.query
        
        if sender_id:
            query = query.filter(Message.sender_id == sender_id)
        if recipient_id:
            query = query.filter(Message.recipient_id == recipient_id)
        if message_type:
            query = query.filter(Message.message_type == MessageType(message_type))
        
        # Order by creation time and limit
        messages = query.order_by(Message.created_at.desc()).limit(limit).all()
        
        return jsonify({
            "messages": [message.to_dict() for message in messages],
            "count": len(messages),
            "filters": {
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "message_type": message_type,
                "limit": limit
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except ValueError as e:
        return jsonify({"error": f"Invalid message type: {message_type}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/workflows', methods=['POST'])
def create_workflow():
    """Create a new workflow instance"""
    try:
        data = request.get_json()
        
        workflow = WorkflowInstance(
            workflow_type=data.get('workflow_type', 'generic'),
            context_data=json.dumps(data.get('context_data', {})),
            current_step=data.get('current_step', 'start')
        )
        
        db.session.add(workflow)
        db.session.commit()
        
        return jsonify({
            "message": "Workflow created successfully",
            "workflow": workflow.to_dict()
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/workflows/<workflow_id>', methods=['GET'])
def get_workflow(workflow_id):
    """Get workflow details"""
    try:
        workflow = WorkflowInstance.query.get(workflow_id)
        if not workflow:
            return jsonify({"error": "Workflow not found"}), 404
        
        return jsonify({
            "workflow": workflow.to_dict(),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/workflows', methods=['GET'])
def list_workflows():
    """List workflow instances"""
    try:
        status = request.args.get('status')
        workflow_type = request.args.get('workflow_type')
        limit = int(request.args.get('limit', 50))
        
        query = WorkflowInstance.query
        
        if status:
            query = query.filter(WorkflowInstance.status == status)
        if workflow_type:
            query = query.filter(WorkflowInstance.workflow_type == workflow_type)
        
        workflows = query.order_by(WorkflowInstance.created_at.desc()).limit(limit).all()
        
        return jsonify({
            "workflows": [workflow.to_dict() for workflow in workflows],
            "count": len(workflows),
            "filters": {
                "status": status,
                "workflow_type": workflow_type,
                "limit": limit
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/workflows/<workflow_id>', methods=['PUT'])
def update_workflow(workflow_id):
    """Update workflow status and context"""
    try:
        workflow = WorkflowInstance.query.get(workflow_id)
        if not workflow:
            return jsonify({"error": "Workflow not found"}), 404
        
        data = request.get_json()
        
        if 'status' in data:
            workflow.status = data['status']
        if 'current_step' in data:
            workflow.current_step = data['current_step']
        if 'context_data' in data:
            workflow.update_context_data(data['context_data'])
        
        if data.get('status') == 'completed':
            workflow.completed_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            "message": "Workflow updated successfully",
            "workflow": workflow.to_dict()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Research Agent specific endpoints
@agents_bp.route('/research/search', methods=['POST'])
def research_literature_search():
    """Perform literature search using research agents"""
    try:
        data = request.get_json()
        
        # Find available research agents
        research_agents = agent_registry.get_agents_by_capability(AgentCapability.RESEARCH_DISCOVERY)
        if not research_agents:
            return jsonify({"error": "No research agents available"}), 503
        
        # Use the first available research agent
        agent = research_agents[0]
        
        # Create and send query message
        message = Message(
            sender_id='api_client',
            recipient_id=agent.id,
            message_type=MessageType.QUERY,
            content=json.dumps({
                'query_type': 'literature_search',
                'query': data.get('query', ''),
                'domain': data.get('domain', 'computer_science'),
                'limit': data.get('limit', 20)
            })
        )
        
        db.session.add(message)
        db.session.commit()
        
        # Route message to agent
        agent_registry.route_message(message)
        
        return jsonify({
            "message": "Literature search initiated",
            "message_id": message.id,
            "agent_id": agent.id,
            "timestamp": datetime.now().isoformat()
        }), 202
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/research/trends', methods=['POST'])
def research_trend_analysis():
    """Perform trend analysis using research agents"""
    try:
        data = request.get_json()
        
        research_agents = agent_registry.get_agents_by_capability(AgentCapability.RESEARCH_DISCOVERY)
        if not research_agents:
            return jsonify({"error": "No research agents available"}), 503
        
        agent = research_agents[0]
        
        message = Message(
            sender_id='api_client',
            recipient_id=agent.id,
            message_type=MessageType.QUERY,
            content=json.dumps({
                'query_type': 'trend_analysis',
                'domain': data.get('domain', 'computer_science'),
                'time_range': data.get('time_range', 5)
            })
        )
        
        db.session.add(message)
        db.session.commit()
        agent_registry.route_message(message)
        
        return jsonify({
            "message": "Trend analysis initiated",
            "message_id": message.id,
            "agent_id": agent.id,
            "timestamp": datetime.now().isoformat()
        }), 202
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Editorial Agent specific endpoints
@agents_bp.route('/editorial/triage', methods=['POST'])
def editorial_manuscript_triage():
    """Perform manuscript triage using editorial agents"""
    try:
        data = request.get_json()
        
        editorial_agents = agent_registry.get_agents_by_capability(AgentCapability.EDITORIAL_ORCHESTRATION)
        if not editorial_agents:
            return jsonify({"error": "No editorial agents available"}), 503
        
        agent = editorial_agents[0]
        
        message = Message(
            sender_id='api_client',
            recipient_id=agent.id,
            message_type=MessageType.QUERY,
            content=json.dumps({
                'query_type': 'manuscript_triage',
                'manuscript_id': data.get('manuscript_id'),
                'manuscript_data': data.get('manuscript_data', {})
            })
        )
        
        db.session.add(message)
        db.session.commit()
        agent_registry.route_message(message)
        
        return jsonify({
            "message": "Manuscript triage initiated",
            "message_id": message.id,
            "agent_id": agent.id,
            "timestamp": datetime.now().isoformat()
        }), 202
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/editorial/decision-support', methods=['POST'])
def editorial_decision_support():
    """Get editorial decision support"""
    try:
        data = request.get_json()
        
        editorial_agents = agent_registry.get_agents_by_capability(AgentCapability.EDITORIAL_ORCHESTRATION)
        if not editorial_agents:
            return jsonify({"error": "No editorial agents available"}), 503
        
        agent = editorial_agents[0]
        
        message = Message(
            sender_id='api_client',
            recipient_id=agent.id,
            message_type=MessageType.QUERY,
            content=json.dumps({
                'query_type': 'decision_support',
                'manuscript_id': data.get('manuscript_id'),
                'reviews': data.get('reviews', []),
                'manuscript_data': data.get('manuscript_data', {})
            })
        )
        
        db.session.add(message)
        db.session.commit()
        agent_registry.route_message(message)
        
        return jsonify({
            "message": "Decision support analysis initiated",
            "message_id": message.id,
            "agent_id": agent.id,
            "timestamp": datetime.now().isoformat()
        }), 202
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/system/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics"""
    try:
        # Get agent statistics
        total_agents = Agent.query.count()
        active_agents = len(agent_registry.get_all_agents())
        
        # Get message statistics
        total_messages = Message.query.count()
        recent_messages = Message.query.filter(
            Message.created_at >= datetime.now() - timedelta(hours=24)
        ).count()
        
        # Get workflow statistics
        total_workflows = WorkflowInstance.query.count()
        active_workflows = WorkflowInstance.query.filter(
            WorkflowInstance.status == 'active'
        ).count()
        
        return jsonify({
            "system_stats": {
                "agents": {
                    "total": total_agents,
                    "active": active_agents,
                    "offline": total_agents - active_agents
                },
                "messages": {
                    "total": total_messages,
                    "last_24h": recent_messages
                },
                "workflows": {
                    "total": total_workflows,
                    "active": active_workflows
                }
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

