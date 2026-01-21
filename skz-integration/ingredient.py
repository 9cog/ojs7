from src.models.user import db
from datetime import datetime
import json

class Ingredient(db.Model):
    """Model for cosmetic ingredients with INCI names and safety data"""
    __tablename__ = 'ingredients'
    
    id = db.Column(db.Integer, primary_key=True)
    inci_name = db.Column(db.String(200), unique=True, nullable=False, index=True)
    cas_number = db.Column(db.String(50), unique=True, nullable=True)
    molecular_formula = db.Column(db.String(100), nullable=True)
    molecular_weight = db.Column(db.Float, nullable=True)
    
    # Functional categories
    category = db.Column(db.String(100), nullable=False)  # active, emulsifier, preservative, etc.
    subcategory = db.Column(db.String(100), nullable=True)
    
    # Safety data
    safety_score = db.Column(db.Float, default=0.0)  # 0-1 scale
    regulatory_status = db.Column(db.String(50), default='unknown')  # approved, restricted, banned
    max_concentration = db.Column(db.Float, nullable=True)  # maximum allowed concentration %
    
    # Usage information
    applications = db.Column(db.Text, nullable=True)  # JSON array of applications
    restrictions = db.Column(db.Text, nullable=True)  # JSON array of restrictions
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    formulations = db.relationship('FormulationIngredient', back_populates='ingredient', cascade='all, delete-orphan')
    safety_assessments = db.relationship('SafetyAssessment', back_populates='ingredient', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Ingredient {self.inci_name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'inci_name': self.inci_name,
            'cas_number': self.cas_number,
            'molecular_formula': self.molecular_formula,
            'molecular_weight': self.molecular_weight,
            'category': self.category,
            'subcategory': self.subcategory,
            'safety_score': self.safety_score,
            'regulatory_status': self.regulatory_status,
            'max_concentration': self.max_concentration,
            'applications': json.loads(self.applications) if self.applications else [],
            'restrictions': json.loads(self.restrictions) if self.restrictions else [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def set_applications(self, applications_list):
        """Set applications as JSON string"""
        self.applications = json.dumps(applications_list)
    
    def set_restrictions(self, restrictions_list):
        """Set restrictions as JSON string"""
        self.restrictions = json.dumps(restrictions_list)

class Formulation(db.Model):
    """Model for cosmetic formulations"""
    __tablename__ = 'formulations'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    
    # Formulation properties
    product_type = db.Column(db.String(100), nullable=False)  # cream, serum, cleanser, etc.
    target_skin_type = db.Column(db.String(100), nullable=True)  # dry, oily, sensitive, etc.
    target_concerns = db.Column(db.Text, nullable=True)  # JSON array of skin concerns
    
    # Performance metrics
    stability_score = db.Column(db.Float, default=0.0)  # 0-1 scale
    efficacy_score = db.Column(db.Float, default=0.0)  # 0-1 scale
    cost_per_unit = db.Column(db.Float, nullable=True)
    
    # Status
    status = db.Column(db.String(50), default='draft')  # draft, testing, approved, discontinued
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = db.Column(db.String(200), nullable=True)  # Store username as string instead of foreign key
    
    # Relationships
    ingredients = db.relationship('FormulationIngredient', back_populates='formulation', cascade='all, delete-orphan')
    clinical_studies = db.relationship('ClinicalStudy', back_populates='formulation', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Formulation {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'product_type': self.product_type,
            'target_skin_type': self.target_skin_type,
            'target_concerns': json.loads(self.target_concerns) if self.target_concerns else [],
            'stability_score': self.stability_score,
            'efficacy_score': self.efficacy_score,
            'cost_per_unit': self.cost_per_unit,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'ingredients': [fi.to_dict() for fi in self.ingredients]
        }

class FormulationIngredient(db.Model):
    """Association table for formulation ingredients with concentrations"""
    __tablename__ = 'formulation_ingredients'
    
    id = db.Column(db.Integer, primary_key=True)
    formulation_id = db.Column(db.Integer, db.ForeignKey('formulations.id'), nullable=False)
    ingredient_id = db.Column(db.Integer, db.ForeignKey('ingredients.id'), nullable=False)
    
    concentration = db.Column(db.Float, nullable=False)  # percentage
    function = db.Column(db.String(100), nullable=True)  # role in formulation
    phase = db.Column(db.String(50), nullable=True)  # water, oil, active, etc.
    
    # Relationships
    formulation = db.relationship('Formulation', back_populates='ingredients')
    ingredient = db.relationship('Ingredient', back_populates='formulations')
    
    def __repr__(self):
        return f'<FormulationIngredient {self.ingredient.inci_name} @ {self.concentration}%>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'ingredient': self.ingredient.to_dict() if self.ingredient else None,
            'concentration': self.concentration,
            'function': self.function,
            'phase': self.phase
        }

class SafetyAssessment(db.Model):
    """Model for ingredient safety assessments"""
    __tablename__ = 'safety_assessments'
    
    id = db.Column(db.Integer, primary_key=True)
    ingredient_id = db.Column(db.Integer, db.ForeignKey('ingredients.id'), nullable=False)
    
    # Assessment details
    assessment_type = db.Column(db.String(100), nullable=False)  # toxicology, clinical, regulatory
    methodology = db.Column(db.String(200), nullable=True)
    
    # Results
    safety_score = db.Column(db.Float, nullable=False)  # 0-1 scale
    risk_level = db.Column(db.String(50), nullable=False)  # low, medium, high
    findings = db.Column(db.Text, nullable=True)
    recommendations = db.Column(db.Text, nullable=True)
    
    # Study information
    study_reference = db.Column(db.String(500), nullable=True)
    study_date = db.Column(db.Date, nullable=True)
    assessor = db.Column(db.String(200), nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    ingredient = db.relationship('Ingredient', back_populates='safety_assessments')
    
    def __repr__(self):
        return f'<SafetyAssessment {self.ingredient.inci_name} - {self.assessment_type}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'ingredient_id': self.ingredient_id,
            'assessment_type': self.assessment_type,
            'methodology': self.methodology,
            'safety_score': self.safety_score,
            'risk_level': self.risk_level,
            'findings': self.findings,
            'recommendations': self.recommendations,
            'study_reference': self.study_reference,
            'study_date': self.study_date.isoformat() if self.study_date else None,
            'assessor': self.assessor,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class ClinicalStudy(db.Model):
    """Model for clinical studies and efficacy testing"""
    __tablename__ = 'clinical_studies'
    
    id = db.Column(db.Integer, primary_key=True)
    formulation_id = db.Column(db.Integer, db.ForeignKey('formulations.id'), nullable=True)
    
    # Study details
    title = db.Column(db.String(500), nullable=False)
    study_type = db.Column(db.String(100), nullable=False)  # efficacy, safety, consumer
    methodology = db.Column(db.Text, nullable=True)
    
    # Participants
    participant_count = db.Column(db.Integer, nullable=True)
    demographics = db.Column(db.Text, nullable=True)  # JSON object
    
    # Results
    primary_endpoint = db.Column(db.String(200), nullable=True)
    results_summary = db.Column(db.Text, nullable=True)
    statistical_significance = db.Column(db.Boolean, default=False)
    effect_size = db.Column(db.Float, nullable=True)
    
    # Study metadata
    study_duration = db.Column(db.Integer, nullable=True)  # days
    study_start_date = db.Column(db.Date, nullable=True)
    study_end_date = db.Column(db.Date, nullable=True)
    principal_investigator = db.Column(db.String(200), nullable=True)
    
    # Publication
    publication_status = db.Column(db.String(50), default='unpublished')
    doi = db.Column(db.String(200), nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    formulation = db.relationship('Formulation', back_populates='clinical_studies')
    
    def __repr__(self):
        return f'<ClinicalStudy {self.title}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'formulation_id': self.formulation_id,
            'title': self.title,
            'study_type': self.study_type,
            'methodology': self.methodology,
            'participant_count': self.participant_count,
            'demographics': json.loads(self.demographics) if self.demographics else {},
            'primary_endpoint': self.primary_endpoint,
            'results_summary': self.results_summary,
            'statistical_significance': self.statistical_significance,
            'effect_size': self.effect_size,
            'study_duration': self.study_duration,
            'study_start_date': self.study_start_date.isoformat() if self.study_start_date else None,
            'study_end_date': self.study_end_date.isoformat() if self.study_end_date else None,
            'principal_investigator': self.principal_investigator,
            'publication_status': self.publication_status,
            'doi': self.doi,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

