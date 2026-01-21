from flask import Blueprint, request, jsonify, session
from flask_cors import cross_origin
import openai
import json
import uuid
from datetime import datetime
import os

chatbot_bp = Blueprint('chatbot', __name__)

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

# In-memory storage for chat sessions (in production, use a database)
chat_sessions = {}

# Agent configurations
AGENTS = {
    'submission': {
        'name': 'Submission Assistant',
        'system_prompt': """You are a helpful AI assistant specialized in academic journal submissions. 
        You help authors with manuscript preparation, formatting guidelines, submission requirements, 
        and navigating the peer review process. You provide clear, actionable advice and can guide 
        users through step-by-step processes. Always be encouraging and professional.""",
        'capabilities': ['manuscript_formatting', 'submission_guidance', 'requirement_validation']
    },
    'editorial': {
        'name': 'Editorial Support',
        'system_prompt': """You are an AI assistant that helps journal editors with workflow management, 
        reviewer assignments, and editorial decisions. You understand the peer review process, can 
        suggest reviewers based on expertise, and help with deadline management. You maintain 
        professional standards and editorial ethics.""",
        'capabilities': ['reviewer_matching', 'workflow_management', 'deadline_tracking']
    },
    'review': {
        'name': 'Review Facilitator',
        'system_prompt': """You are an AI assistant that supports peer reviewers in conducting 
        thorough and constructive reviews. You help with review criteria, provide templates, 
        and guide reviewers through best practices. You emphasize constructive feedback and 
        maintain academic integrity.""",
        'capabilities': ['review_guidance', 'quality_assessment', 'feedback_templates']
    },
    'support': {
        'name': 'Technical Support',
        'system_prompt': """You are a technical support AI assistant for the Open Journal Systems 
        platform. You help users with system features, troubleshoot issues, explain functionality, 
        and provide guidance on using the platform effectively. You are patient and provide 
        clear, step-by-step instructions.""",
        'capabilities': ['system_help', 'troubleshooting', 'feature_explanation']
    }
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
        message_type = data.get('messageType', 'text')
        
        session_data = chat_sessions[session_id]
        agent_type = session_data['agent_type']
        agent = AGENTS[agent_type]
        
        # Add user message to session
        session_data['messages'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Prepare messages for OpenAI
        messages = [
            {'role': 'system', 'content': agent['system_prompt']}
        ]
        
        # Add conversation history (last 10 messages to manage token limits)
        for msg in session_data['messages'][-10:]:
            messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
        
        # Get AI response
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            confidence = 0.9  # Placeholder confidence score
            
        except Exception as openai_error:
            # Fallback response if OpenAI fails
            ai_response = f"I apologize, but I'm experiencing technical difficulties. As {agent['name']}, I'm here to help with your questions. Could you please try rephrasing your question?"
            confidence = 0.5
        
        # Add AI response to session
        session_data['messages'].append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate suggestions based on agent type
        suggestions = get_suggestions(agent_type, user_message)
        
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

@chatbot_bp.route('/agents/submission/validate', methods=['POST'])
@cross_origin()
def validate_submission():
    """Validate submission data"""
    try:
        data = request.get_json()
        submission_data = data.get('submissionData', {})
        journal_id = data.get('journalId')
        
        # Simple validation logic (in production, this would be more sophisticated)
        issues = []
        
        if not submission_data.get('title'):
            issues.append({
                'field': 'title',
                'severity': 'error',
                'message': 'Title is required',
                'suggestion': 'Please provide a descriptive title for your manuscript'
            })
        
        if not submission_data.get('abstract'):
            issues.append({
                'field': 'abstract',
                'severity': 'error',
                'message': 'Abstract is required',
                'suggestion': 'Please provide an abstract summarizing your research'
            })
        
        if not submission_data.get('authors'):
            issues.append({
                'field': 'authors',
                'severity': 'error',
                'message': 'At least one author is required',
                'suggestion': 'Please add author information'
            })
        
        # Calculate completeness
        required_fields = ['title', 'abstract', 'authors', 'files']
        completed_fields = sum(1 for field in required_fields if submission_data.get(field))
        completeness = (completed_fields / len(required_fields)) * 100
        
        return jsonify({
            'success': True,
            'data': {
                'isValid': len([i for i in issues if i['severity'] == 'error']) == 0,
                'issues': issues,
                'completeness': completeness
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'VALIDATION_ERROR',
                'message': str(e)
            }
        }), 500

@chatbot_bp.route('/agents/editorial/recommend-reviewers', methods=['POST'])
@cross_origin()
def recommend_reviewers():
    """Recommend reviewers for a submission"""
    try:
        data = request.get_json()
        submission_id = data.get('submissionId')
        keywords = data.get('keywords', [])
        exclude_reviewers = data.get('excludeReviewers', [])
        
        # Mock reviewer recommendations (in production, this would query a database)
        mock_reviewers = [
            {
                'reviewerId': 1,
                'name': 'Dr. Alice Johnson',
                'affiliation': 'MIT Computer Science',
                'expertise': ['machine learning', 'natural language processing'],
                'matchScore': 0.95,
                'availability': 'Available',
                'recentReviews': 3
            },
            {
                'reviewerId': 2,
                'name': 'Prof. Bob Chen',
                'affiliation': 'Stanford AI Lab',
                'expertise': ['artificial intelligence', 'deep learning'],
                'matchScore': 0.88,
                'availability': 'Busy',
                'recentReviews': 5
            },
            {
                'reviewerId': 3,
                'name': 'Dr. Carol Williams',
                'affiliation': 'Oxford University',
                'expertise': ['computational linguistics', 'AI ethics'],
                'matchScore': 0.82,
                'availability': 'Available',
                'recentReviews': 2
            }
        ]
        
        # Filter out excluded reviewers
        recommendations = [r for r in mock_reviewers if r['reviewerId'] not in exclude_reviewers]
        
        return jsonify({
            'success': True,
            'data': {
                'recommendations': recommendations
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'REVIEWER_RECOMMENDATION_ERROR',
                'message': str(e)
            }
        }), 500

def get_suggestions(agent_type, user_message):
    """Generate contextual suggestions based on agent type and user message"""
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

