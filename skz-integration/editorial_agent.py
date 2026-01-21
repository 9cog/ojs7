"""
Editorial Orchestration Agent Implementation
Specialized agent for editorial workflow management, decision support, and coordination
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from src.models.agent import Agent, AgentCapability, MessageType, db, WorkflowInstance
import openai
import os

class EditorialOrchestrationAgent(Agent):
    """
    Editorial Orchestration Agent for comprehensive editorial workflow management
    Implements intelligent decision support and workflow optimization
    """
    
    def __init__(self, name: str = "Editorial Orchestration Agent"):
        super().__init__(
            name=name,
            agent_type="editorial_orchestration",
            capabilities=[AgentCapability.EDITORIAL_ORCHESTRATION],
            arena_context={
                "workflow_types": ["submission_triage", "review_coordination", "decision_making"],
                "decision_criteria": ["quality", "novelty", "relevance", "methodology"],
                "stakeholders": ["authors", "reviewers", "editors", "publishers"]
            }
        )
        
        # Initialize OpenAI client if available
        self.openai_client = None
        if os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai.OpenAI()
        
        # Editorial decision rules and criteria
        self.decision_criteria = {
            "quality_threshold": 0.7,
            "novelty_threshold": 0.6,
            "methodology_threshold": 0.8,
            "relevance_threshold": 0.7
        }
        
        # Workflow templates
        self.workflow_templates = {
            "standard_review": {
                "steps": ["triage", "reviewer_assignment", "review_collection", "decision"],
                "timeouts": {"triage": 2, "reviewer_assignment": 3, "review_collection": 21, "decision": 7}
            },
            "fast_track": {
                "steps": ["triage", "expedited_review", "decision"],
                "timeouts": {"triage": 1, "expedited_review": 7, "decision": 2}
            }
        }
        
        # Active workflows
        self.active_workflows = {}
    
    def _process_message(self, message):
        """Process editorial workflow messages"""
        try:
            content = json.loads(message.content) if isinstance(message.content, str) else message.content
            
            if message.message_type == MessageType.QUERY:
                response_data = self._handle_editorial_query(content)
            elif message.message_type == MessageType.COMMAND:
                response_data = self._handle_editorial_command(content)
            elif message.message_type == MessageType.EVENT:
                response_data = self._handle_editorial_event(content)
            else:
                response_data = {"error": "Unsupported message type"}
            
            # Send response back
            self.send_message(
                recipient_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                content=response_data,
                correlation_id=message.id
            )
            
            # Update performance metrics
            metrics = self.get_performance_metrics()
            metrics['tasks_completed'] += 1
            self.update_performance_metrics(metrics)
            
        except Exception as e:
            error_response = {"error": str(e), "agent": self.name}
            self.send_message(
                recipient_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                content=error_response,
                correlation_id=message.id
            )
    
    def _handle_editorial_query(self, content: Dict) -> Dict:
        """Handle editorial query requests"""
        query_type = content.get('query_type')
        
        if query_type == 'manuscript_triage':
            return self._perform_manuscript_triage(content)
        elif query_type == 'editor_assignment':
            return self._suggest_editor_assignment(content)
        elif query_type == 'decision_support':
            return self._provide_decision_support(content)
        elif query_type == 'workflow_status':
            return self._get_workflow_status(content)
        else:
            return {"error": f"Unknown query type: {query_type}"}
    
    def _handle_editorial_command(self, content: Dict) -> Dict:
        """Handle editorial command requests"""
        command = content.get('command')
        
        if command == 'start_workflow':
            return self._start_editorial_workflow(content)
        elif command == 'update_workflow':
            return self._update_workflow_status(content)
        elif command == 'make_decision':
            return self._make_editorial_decision(content)
        elif command == 'assign_reviewer':
            return self._assign_reviewer(content)
        else:
            return {"error": f"Unknown command: {command}"}
    
    def _handle_editorial_event(self, content: Dict) -> Dict:
        """Handle editorial event notifications"""
        event_type = content.get('event_type')
        
        if event_type == 'submission_received':
            return self._handle_submission_received(content)
        elif event_type == 'review_completed':
            return self._handle_review_completed(content)
        elif event_type == 'deadline_approaching':
            return self._handle_deadline_approaching(content)
        else:
            return {"error": f"Unknown event type: {event_type}"}
    
    def _perform_manuscript_triage(self, params: Dict) -> Dict:
        """Perform intelligent manuscript triage"""
        manuscript_id = params.get('manuscript_id')
        manuscript_data = params.get('manuscript_data', {})
        
        # Extract manuscript features for triage
        title = manuscript_data.get('title', '')
        abstract = manuscript_data.get('abstract', '')
        keywords = manuscript_data.get('keywords', [])
        author_info = manuscript_data.get('authors', [])
        
        # Perform basic quality checks
        quality_score = self._assess_manuscript_quality(manuscript_data)
        
        # Determine triage decision
        triage_result = {
            "manuscript_id": manuscript_id,
            "triage_decision": "accept_for_review",  # accept_for_review, desk_reject, request_revision
            "quality_score": quality_score,
            "priority": "normal",  # high, normal, low
            "suggested_track": "standard_review",
            "estimated_review_time": 21,  # days
            "triage_timestamp": datetime.now().isoformat()
        }
        
        # Use AI for enhanced triage if available
        if self.openai_client and abstract:
            ai_assessment = self._ai_triage_assessment(title, abstract, keywords)
            triage_result["ai_assessment"] = ai_assessment
        
        # Apply decision rules
        if quality_score < self.decision_criteria["quality_threshold"]:
            triage_result["triage_decision"] = "desk_reject"
            triage_result["rejection_reason"] = "Quality below threshold"
        elif quality_score > 0.9:
            triage_result["priority"] = "high"
            triage_result["suggested_track"] = "fast_track"
        
        return triage_result
    
    def _assess_manuscript_quality(self, manuscript_data: Dict) -> float:
        """Assess manuscript quality based on various factors"""
        quality_factors = {
            "title_quality": 0.0,
            "abstract_quality": 0.0,
            "methodology_clarity": 0.0,
            "novelty_indicator": 0.0
        }
        
        # Title quality assessment
        title = manuscript_data.get('title', '')
        if title:
            quality_factors["title_quality"] = min(len(title.split()) / 15.0, 1.0)  # Optimal title length
        
        # Abstract quality assessment
        abstract = manuscript_data.get('abstract', '')
        if abstract:
            word_count = len(abstract.split())
            quality_factors["abstract_quality"] = min(word_count / 250.0, 1.0)  # Optimal abstract length
        
        # Simple methodology check
        methodology_keywords = ['method', 'approach', 'algorithm', 'experiment', 'analysis']
        if any(keyword in abstract.lower() for keyword in methodology_keywords):
            quality_factors["methodology_clarity"] = 0.8
        
        # Novelty indicator (simplified)
        novelty_keywords = ['novel', 'new', 'innovative', 'first', 'breakthrough']
        if any(keyword in abstract.lower() for keyword in novelty_keywords):
            quality_factors["novelty_indicator"] = 0.7
        
        # Calculate overall quality score
        return sum(quality_factors.values()) / len(quality_factors)
    
    def _ai_triage_assessment(self, title: str, abstract: str, keywords: List[str]) -> Dict:
        """Use AI to provide triage assessment"""
        try:
            prompt = f"""
            Assess this manuscript for academic publication:
            
            Title: {title}
            Abstract: {abstract}
            Keywords: {', '.join(keywords)}
            
            Provide assessment on:
            1. Scientific quality (0-1 score)
            2. Novelty (0-1 score)
            3. Clarity (0-1 score)
            4. Significance (0-1 score)
            5. Recommendation (accept_for_review, desk_reject, request_revision)
            6. Key strengths (list)
            7. Key concerns (list)
            
            Format as JSON with these keys: quality, novelty, clarity, significance, recommendation, strengths, concerns
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            ai_content = response.choices[0].message.content
            try:
                return json.loads(ai_content)
            except json.JSONDecodeError:
                return {"ai_assessment": ai_content}
                
        except Exception as e:
            return {"error": f"AI assessment failed: {str(e)}"}
    
    def _suggest_editor_assignment(self, params: Dict) -> Dict:
        """Suggest appropriate editor for manuscript"""
        manuscript_data = params.get('manuscript_data', {})
        available_editors = params.get('available_editors', [])
        
        # Extract manuscript characteristics
        keywords = manuscript_data.get('keywords', [])
        subject_area = manuscript_data.get('subject_area', '')
        
        # Simple editor matching based on expertise
        editor_scores = []
        for editor in available_editors:
            score = self._calculate_editor_match_score(editor, keywords, subject_area)
            editor_scores.append({
                "editor_id": editor.get('id'),
                "editor_name": editor.get('name'),
                "match_score": score,
                "expertise_areas": editor.get('expertise', []),
                "current_workload": editor.get('workload', 0)
            })
        
        # Sort by match score and workload
        editor_scores.sort(key=lambda x: (x['match_score'], -x['current_workload']), reverse=True)
        
        return {
            "manuscript_id": params.get('manuscript_id'),
            "recommended_editors": editor_scores[:3],  # Top 3 recommendations
            "assignment_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_editor_match_score(self, editor: Dict, keywords: List[str], subject_area: str) -> float:
        """Calculate editor-manuscript match score"""
        editor_expertise = editor.get('expertise', [])
        
        # Keyword overlap score
        keyword_overlap = len(set(keywords) & set(editor_expertise)) / max(len(keywords), 1)
        
        # Subject area match
        subject_match = 1.0 if subject_area in editor_expertise else 0.5
        
        # Workload penalty
        workload = editor.get('workload', 0)
        workload_penalty = max(0, (workload - 5) * 0.1)  # Penalty for high workload
        
        return (keyword_overlap * 0.6 + subject_match * 0.4) - workload_penalty
    
    def _provide_decision_support(self, params: Dict) -> Dict:
        """Provide decision support based on reviews and manuscript data"""
        manuscript_id = params.get('manuscript_id')
        reviews = params.get('reviews', [])
        manuscript_data = params.get('manuscript_data', {})
        
        # Analyze reviews
        review_analysis = self._analyze_reviews(reviews)
        
        # Generate decision recommendation
        decision_recommendation = self._generate_decision_recommendation(review_analysis, manuscript_data)
        
        return {
            "manuscript_id": manuscript_id,
            "review_analysis": review_analysis,
            "decision_recommendation": decision_recommendation,
            "confidence_score": decision_recommendation.get('confidence', 0.0),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _analyze_reviews(self, reviews: List[Dict]) -> Dict:
        """Analyze reviewer feedback and scores"""
        if not reviews:
            return {"error": "No reviews available"}
        
        # Extract scores and comments
        scores = []
        sentiments = []
        key_points = []
        
        for review in reviews:
            if 'score' in review:
                scores.append(review['score'])
            
            # Simple sentiment analysis of comments
            comments = review.get('comments', '')
            sentiment = self._analyze_sentiment(comments)
            sentiments.append(sentiment)
            
            # Extract key points (simplified)
            if comments:
                key_points.extend(self._extract_key_points(comments))
        
        return {
            "average_score": sum(scores) / len(scores) if scores else 0,
            "score_variance": self._calculate_variance(scores) if len(scores) > 1 else 0,
            "sentiment_distribution": {
                "positive": sentiments.count('positive'),
                "neutral": sentiments.count('neutral'),
                "negative": sentiments.count('negative')
            },
            "key_concerns": [point for point in key_points if 'concern' in point.lower()],
            "key_strengths": [point for point in key_points if 'strength' in point.lower()]
        }
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis"""
        positive_words = ['excellent', 'good', 'strong', 'clear', 'innovative', 'significant']
        negative_words = ['poor', 'weak', 'unclear', 'limited', 'insufficient', 'flawed']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from review text"""
        # Simplified key point extraction
        sentences = text.split('.')
        key_points = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and any(keyword in sentence.lower() 
                                        for keyword in ['concern', 'strength', 'issue', 'problem', 'good', 'excellent']):
                key_points.append(sentence)
        
        return key_points[:5]  # Limit to top 5 key points
    
    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance of scores"""
        if len(scores) < 2:
            return 0
        
        mean = sum(scores) / len(scores)
        return sum((score - mean) ** 2 for score in scores) / len(scores)
    
    def _generate_decision_recommendation(self, review_analysis: Dict, manuscript_data: Dict) -> Dict:
        """Generate editorial decision recommendation"""
        avg_score = review_analysis.get('average_score', 0)
        score_variance = review_analysis.get('score_variance', 0)
        sentiment_dist = review_analysis.get('sentiment_distribution', {})
        
        # Decision logic
        if avg_score >= 4.0 and sentiment_dist.get('positive', 0) >= sentiment_dist.get('negative', 0):
            decision = "accept"
            confidence = 0.9 if score_variance < 0.5 else 0.7
        elif avg_score >= 3.0 and score_variance < 1.0:
            decision = "minor_revision"
            confidence = 0.8
        elif avg_score >= 2.5:
            decision = "major_revision"
            confidence = 0.7
        else:
            decision = "reject"
            confidence = 0.8
        
        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": self._generate_decision_reasoning(review_analysis),
            "next_steps": self._suggest_next_steps(decision)
        }
    
    def _generate_decision_reasoning(self, review_analysis: Dict) -> str:
        """Generate reasoning for decision"""
        avg_score = review_analysis.get('average_score', 0)
        concerns = review_analysis.get('key_concerns', [])
        strengths = review_analysis.get('key_strengths', [])
        
        reasoning = f"Based on average review score of {avg_score:.1f}. "
        
        if strengths:
            reasoning += f"Key strengths include: {', '.join(strengths[:2])}. "
        
        if concerns:
            reasoning += f"Main concerns: {', '.join(concerns[:2])}."
        
        return reasoning
    
    def _suggest_next_steps(self, decision: str) -> List[str]:
        """Suggest next steps based on decision"""
        next_steps = {
            "accept": [
                "Send acceptance letter to authors",
                "Begin production process",
                "Schedule publication"
            ],
            "minor_revision": [
                "Send revision request to authors",
                "Set revision deadline (30 days)",
                "Prepare reviewer feedback summary"
            ],
            "major_revision": [
                "Send detailed revision request",
                "Set extended revision deadline (60 days)",
                "Consider additional reviewer if needed"
            ],
            "reject": [
                "Send rejection letter with feedback",
                "Suggest alternative venues if appropriate",
                "Close manuscript record"
            ]
        }
        
        return next_steps.get(decision, ["Review decision and take appropriate action"])
    
    def _start_editorial_workflow(self, params: Dict) -> Dict:
        """Start a new editorial workflow"""
        manuscript_id = params.get('manuscript_id')
        workflow_type = params.get('workflow_type', 'standard_review')
        
        # Create workflow instance
        workflow = WorkflowInstance(
            workflow_type=workflow_type,
            context_data=json.dumps({
                "manuscript_id": manuscript_id,
                "started_by": self.id,
                "workflow_template": self.workflow_templates.get(workflow_type, {})
            }),
            current_step="triage"
        )
        
        db.session.add(workflow)
        db.session.commit()
        
        # Track active workflow
        self.active_workflows[workflow.id] = workflow
        
        return {
            "workflow_id": workflow.id,
            "manuscript_id": manuscript_id,
            "workflow_type": workflow_type,
            "current_step": workflow.current_step,
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_workflow_status(self, params: Dict) -> Dict:
        """Get status of editorial workflow"""
        workflow_id = params.get('workflow_id')
        
        workflow = WorkflowInstance.query.get(workflow_id)
        if not workflow:
            return {"error": "Workflow not found"}
        
        return workflow.to_dict()
    
    def _update_workflow_status(self, params: Dict) -> Dict:
        """Update workflow status and progress"""
        workflow_id = params.get('workflow_id')
        new_step = params.get('new_step')
        update_data = params.get('update_data', {})
        
        workflow = WorkflowInstance.query.get(workflow_id)
        if not workflow:
            return {"error": "Workflow not found"}
        
        workflow.current_step = new_step
        workflow.update_context_data(update_data)
        
        db.session.commit()
        
        return {
            "workflow_id": workflow_id,
            "updated_step": new_step,
            "status": "updated",
            "timestamp": datetime.now().isoformat()
        }
    
    def _make_editorial_decision(self, params: Dict) -> Dict:
        """Make final editorial decision"""
        manuscript_id = params.get('manuscript_id')
        decision = params.get('decision')
        reasoning = params.get('reasoning', '')
        
        # Record decision
        decision_record = {
            "manuscript_id": manuscript_id,
            "decision": decision,
            "reasoning": reasoning,
            "decided_by": self.id,
            "decision_timestamp": datetime.now().isoformat()
        }
        
        return decision_record
    
    def _assign_reviewer(self, params: Dict) -> Dict:
        """Assign reviewer to manuscript"""
        manuscript_id = params.get('manuscript_id')
        reviewer_id = params.get('reviewer_id')
        deadline = params.get('deadline')
        
        assignment_record = {
            "manuscript_id": manuscript_id,
            "reviewer_id": reviewer_id,
            "assigned_by": self.id,
            "deadline": deadline,
            "assignment_timestamp": datetime.now().isoformat(),
            "status": "assigned"
        }
        
        return assignment_record
    
    def _handle_submission_received(self, content: Dict) -> Dict:
        """Handle new submission event"""
        manuscript_id = content.get('manuscript_id')
        
        # Automatically start triage workflow
        workflow_result = self._start_editorial_workflow({
            "manuscript_id": manuscript_id,
            "workflow_type": "standard_review"
        })
        
        return {
            "event_handled": "submission_received",
            "manuscript_id": manuscript_id,
            "workflow_started": workflow_result,
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_review_completed(self, content: Dict) -> Dict:
        """Handle review completion event"""
        manuscript_id = content.get('manuscript_id')
        reviewer_id = content.get('reviewer_id')
        
        return {
            "event_handled": "review_completed",
            "manuscript_id": manuscript_id,
            "reviewer_id": reviewer_id,
            "next_action": "check_all_reviews_complete",
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_deadline_approaching(self, content: Dict) -> Dict:
        """Handle deadline approaching event"""
        workflow_id = content.get('workflow_id')
        deadline_type = content.get('deadline_type')
        
        return {
            "event_handled": "deadline_approaching",
            "workflow_id": workflow_id,
            "deadline_type": deadline_type,
            "action_required": "send_reminder",
            "timestamp": datetime.now().isoformat()
        }

