import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.models.user import db
from src.models.chatbot import ChatSession, ChatMessage, AgentKnowledge, UserFeedback
from src.routes.user import user_bp
from src.routes.chatbot import chatbot_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# Enable CORS for all routes
CORS(app, origins="*")

# Register blueprints
app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(chatbot_bp, url_prefix='/api/chatbot/v1')

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()
    
    # Initialize knowledge base with sample data
    if AgentKnowledge.query.count() == 0:
        sample_knowledge = [
            {
                'agent_type': 'submission',
                'category': 'formatting',
                'question': 'How should I format my manuscript?',
                'answer': 'Manuscripts should be formatted in double-spaced, 12-point Times New Roman font with 1-inch margins. Include a title page, abstract, main text, references, and any figures or tables.',
                'keywords': ['format', 'manuscript', 'style', 'guidelines']
            },
            {
                'agent_type': 'submission',
                'category': 'requirements',
                'question': 'What files do I need to submit?',
                'answer': 'You need to submit your main manuscript file (Word or PDF), any supplementary materials, figures (high resolution), and a cover letter. Some journals also require author agreements.',
                'keywords': ['files', 'requirements', 'submission', 'documents']
            },
            {
                'agent_type': 'editorial',
                'category': 'workflow',
                'question': 'How do I assign reviewers?',
                'answer': 'Go to the submission details page, click "Assign Reviewers", search for experts in the field, and send invitations. Consider reviewer workload and potential conflicts of interest.',
                'keywords': ['reviewers', 'assignment', 'workflow', 'editorial']
            },
            {
                'agent_type': 'review',
                'category': 'guidelines',
                'question': 'What should I include in my review?',
                'answer': 'A good review should assess the novelty, methodology, clarity, and significance of the work. Provide constructive feedback, suggest improvements, and make a clear recommendation.',
                'keywords': ['review', 'feedback', 'assessment', 'guidelines']
            }
        ]
        
        for kb_item in sample_knowledge:
            knowledge = AgentKnowledge(
                agent_type=kb_item['agent_type'],
                category=kb_item['category'],
                question=kb_item['question'],
                answer=kb_item['answer']
            )
            knowledge.set_keywords(kb_item['keywords'])
            db.session.add(knowledge)
        
        db.session.commit()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404

@app.route('/health')
def health_check():
    return {'status': 'healthy', 'service': 'OJS Chatbot Service'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
