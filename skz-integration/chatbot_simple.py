from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import json
import uuid
from datetime import datetime

chatbot_bp = Blueprint('chatbot', __name__)

# In-memory storage for chat sessions
chat_sessions = {}

# Agent configurations
AGENTS = {
    'submission': {
        'name': 'Submission Assistant',
        'system_prompt': 'You are a helpful AI assistant specialized in academic journal submissions.',
        'capabilities': ['manuscript_formatting', 'submission_guidance', 'requirement_validation']
    },
    'editorial': {
        'name': 'Editorial Support',
        'system_prompt': 'You are an AI assistant that helps journal editors with workflow management.',
        'capabilities': ['reviewer_matching', 'workflow_management', 'deadline_tracking']
    },
    'review': {
        'name': 'Review Facilitator',
        'system_prompt': 'You are an AI assistant that supports peer reviewers in conducting reviews.',
        'capabilities': ['review_guidance', 'quality_assessment', 'feedback_templates']
    },
    'support': {
        'name': 'Technical Support',
        'system_prompt': 'You are a technical support AI assistant for the Open Journal Systems platform.',
        'capabilities': ['system_help', 'troubleshooting', 'feature_explanation']
    }
}

# Sample responses for demonstration
SAMPLE_RESPONSES = {
    'submission': [
        "To submit a new manuscript, click the 'New Submission' button on your dashboard. You'll need to provide a title, abstract, author information, and upload your manuscript file.",
        "For manuscript formatting, please use double-spaced text, 12-point Times New Roman font, and include line numbers. Check our submission guidelines for specific requirements.",
        "The typical review process takes 4-6 weeks. You'll receive email notifications at each stage of the review process."
    ],
    'editorial': [
        "To assign reviewers, go to the submission details page and click 'Assign Reviewers'. You can search by expertise area or browse our reviewer database.",
        "The editorial workflow includes initial screening, peer review assignment, review collection, and final decision. Each stage has configurable deadlines.",
        "You can set automatic reminders for reviewers through the system settings. We recommend sending reminders 1 week before the deadline."
    ],
    'review': [
        "A good peer review should assess the novelty, methodology, clarity, and significance of the research. Provide constructive feedback to help improve the manuscript.",
        "Please complete your review within the specified deadline. If you need an extension, contact the editor as soon as possible.",
        "Use our review template to ensure you cover all important aspects: summary, strengths, weaknesses, and recommendation."
    ],
    'support': [
        "To reset your password, click 'Forgot Password' on the login page and follow the instructions sent to your email.",
        "You can find all your submissions in the 'My Submissions' section of your dashboard. Use the filters to find specific submissions.",
        "To update your profile, click on your name in the top-right corner and select 'Profile Settings'."
    ]
}

@chatbot_bp.route('/sessions', methods=['POST'])
@cross_origin()
def create_session():
    """Create a new chat session"""
    try:
        data = request.get_json()
        user_id = data.get('userId')
        context = data.get('context', {})
        
        # Determine agent type based on context
        page = context.get('page', 'dashboard')
        user_role = context.get('userRole', 'author')
        
        # Simple agent selection logic
        if 'submission' in page.lower():
            agent_type = 'submission'
        elif user_role == 'editor':
            agent_type = 'editorial'
        elif 'review' in page.lower():
            agent_type = 'review'
        else:
            agent_type = 'support'
        
        session_id = str(uuid.uuid4())
        agent = AGENTS[agent_type]
        
        # Initialize session
        chat_sessions[session_id] = {
            'session_id': session_id,
            'user_id': user_id,
            'agent_type': agent_type,
            'context': context,
            'messages': [],
            'created_at': datetime.now().isoformat()
        }
        
        welcome_message = f"Hello! I'm {agent['name']}, your AI assistant. How can I help you today?"
        
        return jsonify({
            'success': True,
            'data': {
                'sessionId': session_id,
                'agentType': agent_type,
                'welcomeMessage': welcome_message
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'SESSION_CREATE_ERROR',
                'message': str(e)
            }
        }), 500

@chatbot_bp.route('/sessions/<session_id>/messages', methods=['POST'])
@cross_origin()
def send_message(session_id):
    """Send a message to the chatbot"""
    try:
        if session_id not in chat_sessions:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'SESSION_NOT_FOUND',
                    'message': 'Chat session not found'
                }
            }), 404
        
        data = request.get_json()
        user_message = data.get('message', '')
        
        session_data = chat_sessions[session_id]
        agent_type = session_data['agent_type']
        agent = AGENTS[agent_type]
        
        # Add user message to session
        session_data['messages'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Get a sample response based on agent type
        import random
        responses = SAMPLE_RESPONSES.get(agent_type, SAMPLE_RESPONSES['support'])
        ai_response = random.choice(responses)
        confidence = 0.85
        
        # Add AI response to session
        session_data['messages'].append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate suggestions based on agent type
        suggestions = get_suggestions(agent_type)
        
        # Generate action buttons based on context
        actions = get_actions(agent_type, user_message, session_data['context'])
        
        message_id = str(uuid.uuid4())
        
        return jsonify({
            'success': True,
            'data': {
                'messageId': message_id,
                'response': {
                    'text': ai_response,
                    'type': 'text',
                    'actions': actions,
                    'suggestions': suggestions
                },
                'agentType': agent_type,
                'confidence': confidence
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'MESSAGE_SEND_ERROR',
                'message': str(e)
            }
        }), 500

@chatbot_bp.route('/sessions/<session_id>', methods=['GET'])
@cross_origin()
def get_session(session_id):
    """Get chat session details"""
    try:
        if session_id not in chat_sessions:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'SESSION_NOT_FOUND',
                    'message': 'Chat session not found'
                }
            }), 404
        
        session_data = chat_sessions[session_id]
        
        return jsonify({
            'success': True,
            'data': session_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'SESSION_GET_ERROR',
                'message': str(e)
            }
        }), 500

def get_suggestions(agent_type):
    """Generate contextual suggestions based on agent type"""
    suggestions_map = {
        'submission': [
            "How do I format my manuscript?",
            "What are the submission requirements?",
            "How long does the review process take?",
            "Can you help me write an abstract?"
        ],
        'editorial': [
            "How do I assign reviewers?",
            "What's the typical review timeline?",
            "How do I handle reviewer conflicts?",
            "Can you suggest deadline reminders?"
        ],
        'review': [
            "What should I look for in a review?",
            "How do I provide constructive feedback?",
            "What's the review template?",
            "How do I handle conflicts of interest?"
        ],
        'support': [
            "How do I reset my password?",
            "Where can I find my submissions?",
            "How do I update my profile?",
            "What are the system requirements?"
        ]
    }
    
    return suggestions_map.get(agent_type, [])

def get_actions(agent_type, user_message, context):
    """Generate action buttons based on agent type and context"""
    actions = []
    
    if agent_type == 'submission':
        actions = [
            {
                'type': 'button',
                'label': 'Start New Submission',
                'action': 'navigate',
                'data': {'url': '/submissions/new'}
            },
            {
                'type': 'button',
                'label': 'View Guidelines',
                'action': 'open_modal',
                'data': {'modal': 'submission_guidelines'}
            }
        ]
    elif agent_type == 'editorial':
        actions = [
            {
                'type': 'button',
                'label': 'Assign Reviewers',
                'action': 'navigate',
                'data': {'url': '/editorial/assign-reviewers'}
            },
            {
                'type': 'button',
                'label': 'View Workflow',
                'action': 'navigate',
                'data': {'url': '/editorial/workflow'}
            }
        ]
    
    return actions

