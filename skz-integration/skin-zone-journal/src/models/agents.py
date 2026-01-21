from src.models.user import db
from datetime import datetime
import json
import random
import time

class Agent(db.Model):
    """Base model for autonomous agents"""
    __tablename__ = 'agents'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    agent_type = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    
    # Status and performance
    status = db.Column(db.String(50), default='active')  # active, inactive, maintenance
    success_rate = db.Column(db.Float, default=0.0)  # 0-1 scale
    avg_response_time = db.Column(db.Float, default=0.0)  # seconds
    total_actions = db.Column(db.Integer, default=0)
    
    # Configuration
    capabilities = db.Column(db.Text, nullable=True)  # JSON array of capabilities
    parameters = db.Column(db.Text, nullable=True)  # JSON object of configuration
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    actions = db.relationship('AgentAction', back_populates='agent', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Agent {self.name} ({self.agent_type})>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'agent_type': self.agent_type,
            'description': self.description,
            'status': self.status,
            'success_rate': self.success_rate,
            'avg_response_time': self.avg_response_time,
            'total_actions': self.total_actions,
            'capabilities': json.loads(self.capabilities) if self.capabilities else [],
            'parameters': json.loads(self.parameters) if self.parameters else {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_active': self.last_active.isoformat() if self.last_active else None
        }
    
    def set_capabilities(self, capabilities_list):
        """Set capabilities as JSON string"""
        self.capabilities = json.dumps(capabilities_list)
    
    def set_parameters(self, parameters_dict):
        """Set parameters as JSON string"""
        self.parameters = json.dumps(parameters_dict)
    
    def update_performance(self, success, response_time):
        """Update agent performance metrics"""
        self.total_actions += 1
        
        # Update success rate (exponential moving average)
        alpha = 0.1  # learning rate
        if self.total_actions == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            current_success = 1.0 if success else 0.0
            self.success_rate = alpha * current_success + (1 - alpha) * self.success_rate
        
        # Update average response time (exponential moving average)
        if self.total_actions == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = alpha * response_time + (1 - alpha) * self.avg_response_time
        
        self.last_active = datetime.utcnow()

class AgentAction(db.Model):
    """Model for tracking agent actions and results"""
    __tablename__ = 'agent_actions'
    
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey('agents.id'), nullable=False)
    
    # Action details
    action_type = db.Column(db.String(100), nullable=False)
    input_data = db.Column(db.Text, nullable=True)  # JSON input
    output_data = db.Column(db.Text, nullable=True)  # JSON output
    
    # Performance metrics
    success = db.Column(db.Boolean, default=False)
    response_time = db.Column(db.Float, nullable=True)  # seconds
    confidence_score = db.Column(db.Float, nullable=True)  # 0-1 scale
    
    # Error handling
    error_message = db.Column(db.Text, nullable=True)
    retry_count = db.Column(db.Integer, default=0)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    agent = db.relationship('Agent', back_populates='actions')
    
    def __repr__(self):
        return f'<AgentAction {self.agent.name} - {self.action_type}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'action_type': self.action_type,
            'input_data': json.loads(self.input_data) if self.input_data else {},
            'output_data': json.loads(self.output_data) if self.output_data else {},
            'success': self.success,
            'response_time': self.response_time,
            'confidence_score': self.confidence_score,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class IngredientIntelligenceAgent:
    """Specialized agent for ingredient analysis and safety assessment"""
    
    def __init__(self, agent_model):
        self.agent = agent_model
        self.inci_database = self._load_inci_database()
    
    def _load_inci_database(self):
        """Load INCI database (simplified for demo)"""
        return {
            'Retinyl Palmitate': {
                'cas_number': '79-81-2',
                'molecular_formula': 'C36H60O2',
                'category': 'active',
                'safety_score': 0.85,
                'max_concentration': 0.5,
                'applications': ['anti-aging', 'night-care'],
                'restrictions': ['pregnancy', 'breastfeeding']
            },
            'Hyaluronic Acid': {
                'cas_number': '9067-32-7',
                'molecular_formula': '(C14H21NO11)n',
                'category': 'moisturizer',
                'safety_score': 0.98,
                'max_concentration': 2.0,
                'applications': ['hydration', 'anti-aging', 'sensitive-skin'],
                'restrictions': []
            },
            'Salicylic Acid': {
                'cas_number': '69-72-7',
                'molecular_formula': 'C7H6O3',
                'category': 'active',
                'safety_score': 0.75,
                'max_concentration': 2.0,
                'applications': ['acne-treatment', 'exfoliation'],
                'restrictions': ['pregnancy', 'sensitive-skin']
            },
            'Niacinamide': {
                'cas_number': '98-92-0',
                'molecular_formula': 'C6H6N2O',
                'category': 'active',
                'safety_score': 0.92,
                'max_concentration': 10.0,
                'applications': ['brightening', 'pore-minimizing', 'oil-control'],
                'restrictions': []
            },
            'Vitamin C': {
                'cas_number': '50-81-7',
                'molecular_formula': 'C6H8O6',
                'category': 'active',
                'safety_score': 0.88,
                'max_concentration': 20.0,
                'applications': ['brightening', 'antioxidant', 'anti-aging'],
                'restrictions': ['light-sensitive', 'pH-dependent']
            }
        }
    
    def analyze_ingredient(self, inci_name):
        """Analyze ingredient safety and properties"""
        start_time = time.time()
        
        try:
            # Simulate analysis process
            time.sleep(random.uniform(1.0, 3.0))  # Simulate processing time
            
            if inci_name in self.inci_database:
                ingredient_data = self.inci_database[inci_name]
                
                result = {
                    'inci_name': inci_name,
                    'analysis_complete': True,
                    'safety_assessment': {
                        'safety_score': ingredient_data['safety_score'],
                        'risk_level': self._calculate_risk_level(ingredient_data['safety_score']),
                        'regulatory_status': 'approved',
                        'max_concentration': ingredient_data['max_concentration']
                    },
                    'properties': {
                        'cas_number': ingredient_data['cas_number'],
                        'molecular_formula': ingredient_data['molecular_formula'],
                        'category': ingredient_data['category']
                    },
                    'usage_recommendations': {
                        'applications': ingredient_data['applications'],
                        'restrictions': ingredient_data['restrictions']
                    },
                    'confidence_score': 0.95
                }
                success = True
            else:
                result = {
                    'inci_name': inci_name,
                    'analysis_complete': False,
                    'error': 'Ingredient not found in database',
                    'confidence_score': 0.0
                }
                success = False
            
            response_time = time.time() - start_time
            self.agent.update_performance(success, response_time)
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.agent.update_performance(False, response_time)
            return {
                'inci_name': inci_name,
                'analysis_complete': False,
                'error': str(e),
                'confidence_score': 0.0
            }
    
    def _calculate_risk_level(self, safety_score):
        """Calculate risk level based on safety score"""
        if safety_score >= 0.9:
            return 'low'
        elif safety_score >= 0.7:
            return 'medium'
        else:
            return 'high'

class FormulationScienceAgent:
    """Specialized agent for formulation analysis and optimization"""
    
    def __init__(self, agent_model):
        self.agent = agent_model
        self.compatibility_matrix = self._load_compatibility_matrix()
    
    def _load_compatibility_matrix(self):
        """Load ingredient compatibility data"""
        return {
            ('Retinyl Palmitate', 'Vitamin C'): 0.3,  # Low compatibility
            ('Retinyl Palmitate', 'Niacinamide'): 0.8,  # High compatibility
            ('Salicylic Acid', 'Niacinamide'): 0.9,  # Very high compatibility
            ('Hyaluronic Acid', 'Vitamin C'): 0.95,  # Excellent compatibility
            ('Hyaluronic Acid', 'Niacinamide'): 0.92,  # Excellent compatibility
        }
    
    def analyze_formulation(self, ingredient_list):
        """Analyze formulation compatibility and stability"""
        start_time = time.time()
        
        try:
            # Simulate formulation analysis
            time.sleep(random.uniform(2.0, 5.0))
            
            compatibility_scores = []
            stability_factors = []
            
            # Analyze pairwise compatibility
            for i, ing1 in enumerate(ingredient_list):
                for j, ing2 in enumerate(ingredient_list[i+1:], i+1):
                    pair = (ing1['inci_name'], ing2['inci_name'])
                    reverse_pair = (ing2['inci_name'], ing1['inci_name'])
                    
                    if pair in self.compatibility_matrix:
                        score = self.compatibility_matrix[pair]
                    elif reverse_pair in self.compatibility_matrix:
                        score = self.compatibility_matrix[reverse_pair]
                    else:
                        score = 0.7  # Default compatibility
                    
                    compatibility_scores.append(score)
            
            # Calculate overall scores
            overall_compatibility = sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0.8
            stability_score = min(0.95, overall_compatibility + random.uniform(-0.1, 0.1))
            efficacy_score = min(0.95, overall_compatibility + random.uniform(-0.05, 0.15))
            
            result = {
                'formulation_analysis': {
                    'compatibility_score': round(overall_compatibility, 3),
                    'stability_score': round(stability_score, 3),
                    'efficacy_score': round(efficacy_score, 3),
                    'overall_score': round((overall_compatibility + stability_score + efficacy_score) / 3, 3)
                },
                'recommendations': self._generate_recommendations(overall_compatibility),
                'optimization_suggestions': self._generate_optimization_suggestions(ingredient_list),
                'confidence_score': 0.88
            }
            
            success = True
            response_time = time.time() - start_time
            self.agent.update_performance(success, response_time)
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.agent.update_performance(False, response_time)
            return {
                'error': str(e),
                'confidence_score': 0.0
            }
    
    def _generate_recommendations(self, compatibility_score):
        """Generate formulation recommendations"""
        if compatibility_score >= 0.9:
            return [
                "Excellent formulation compatibility",
                "Proceed with stability testing",
                "Consider clinical efficacy studies"
            ]
        elif compatibility_score >= 0.7:
            return [
                "Good formulation compatibility",
                "Monitor for potential interactions",
                "Conduct accelerated stability testing"
            ]
        else:
            return [
                "Low formulation compatibility detected",
                "Review ingredient interactions",
                "Consider alternative ingredients or concentrations"
            ]
    
    def _generate_optimization_suggestions(self, ingredient_list):
        """Generate optimization suggestions"""
        suggestions = []
        
        # Check for high-concentration actives
        for ingredient in ingredient_list:
            if ingredient.get('concentration', 0) > 5.0 and ingredient.get('category') == 'active':
                suggestions.append(f"Consider reducing {ingredient['inci_name']} concentration for better tolerance")
        
        # Suggest synergistic combinations
        inci_names = [ing['inci_name'] for ing in ingredient_list]
        if 'Niacinamide' in inci_names and 'Hyaluronic Acid' not in inci_names:
            suggestions.append("Consider adding Hyaluronic Acid for enhanced hydration")
        
        if not suggestions:
            suggestions.append("Formulation appears well-optimized")
        
        return suggestions

class ClinicalEvidenceAgent:
    """Specialized agent for clinical study design and analysis"""
    
    def __init__(self, agent_model):
        self.agent = agent_model
    
    def design_clinical_study(self, study_parameters):
        """Design clinical study protocol"""
        start_time = time.time()
        
        try:
            # Simulate study design process
            time.sleep(random.uniform(3.0, 6.0))
            
            study_type = study_parameters.get('study_type', 'efficacy')
            primary_endpoint = study_parameters.get('primary_endpoint', 'skin improvement')
            
            # Calculate sample size
            sample_size = self._calculate_sample_size(study_type)
            
            # Generate study protocol
            protocol = {
                'study_design': {
                    'type': study_type,
                    'design': 'randomized controlled trial',
                    'duration': self._determine_study_duration(study_type),
                    'sample_size': sample_size,
                    'primary_endpoint': primary_endpoint
                },
                'methodology': {
                    'inclusion_criteria': self._generate_inclusion_criteria(study_type),
                    'exclusion_criteria': self._generate_exclusion_criteria(study_type),
                    'assessment_methods': self._generate_assessment_methods(study_type),
                    'statistical_plan': self._generate_statistical_plan(sample_size)
                },
                'timeline': {
                    'recruitment': '4 weeks',
                    'treatment': f"{self._determine_study_duration(study_type)} weeks",
                    'follow_up': '2 weeks',
                    'analysis': '4 weeks'
                },
                'confidence_score': 0.91
            }
            
            success = True
            response_time = time.time() - start_time
            self.agent.update_performance(success, response_time)
            
            return protocol
            
        except Exception as e:
            response_time = time.time() - start_time
            self.agent.update_performance(False, response_time)
            return {
                'error': str(e),
                'confidence_score': 0.0
            }
    
    def _calculate_sample_size(self, study_type):
        """Calculate appropriate sample size"""
        base_sizes = {
            'efficacy': 60,
            'safety': 30,
            'consumer': 100,
            'bioavailability': 24
        }
        return base_sizes.get(study_type, 50)
    
    def _determine_study_duration(self, study_type):
        """Determine study duration in weeks"""
        durations = {
            'efficacy': 12,
            'safety': 4,
            'consumer': 2,
            'bioavailability': 1
        }
        return durations.get(study_type, 8)
    
    def _generate_inclusion_criteria(self, study_type):
        """Generate inclusion criteria"""
        base_criteria = [
            "Age 18-65 years",
            "Healthy skin condition",
            "Willing to provide informed consent"
        ]
        
        if study_type == 'efficacy':
            base_criteria.extend([
                "Visible signs of target skin concern",
                "No active skin conditions"
            ])
        elif study_type == 'safety':
            base_criteria.extend([
                "No known allergies to cosmetic ingredients",
                "No current use of similar products"
            ])
        
        return base_criteria
    
    def _generate_exclusion_criteria(self, study_type):
        """Generate exclusion criteria"""
        return [
            "Pregnancy or breastfeeding",
            "Active skin disease",
            "Use of investigational products within 30 days",
            "Known allergies to study ingredients"
        ]
    
    def _generate_assessment_methods(self, study_type):
        """Generate assessment methods"""
        methods = {
            'efficacy': [
                "Clinical grading by dermatologist",
                "Instrumental measurements (hydration, elasticity)",
                "Digital photography",
                "Subject self-assessment questionnaire"
            ],
            'safety': [
                "Visual assessment for irritation",
                "Patch testing",
                "Subject-reported adverse events",
                "Dermatologist evaluation"
            ],
            'consumer': [
                "Product preference questionnaire",
                "Sensory evaluation",
                "Purchase intent survey",
                "Usage diary"
            ]
        }
        return methods.get(study_type, methods['efficacy'])
    
    def _generate_statistical_plan(self, sample_size):
        """Generate statistical analysis plan"""
        return {
            'primary_analysis': 'Paired t-test for before/after comparison',
            'secondary_analysis': 'ANOVA for group comparisons',
            'significance_level': 0.05,
            'power': 0.80,
            'sample_size': sample_size,
            'missing_data': 'Last observation carried forward'
        }

