from flask import Blueprint, request, jsonify
from datetime import datetime
import json
import time

from src.models.user import db
from src.models.ingredient import Ingredient, Formulation, FormulationIngredient, SafetyAssessment, ClinicalStudy
from src.models.agents import Agent, AgentAction, IngredientIntelligenceAgent, FormulationScienceAgent, ClinicalEvidenceAgent

skin_zone_bp = Blueprint('skin_zone', __name__)

# Initialize specialized agents
def get_or_create_agents():
    """Get or create specialized agents for Skin Zone"""
    agents = {}
    
    # Ingredient Intelligence Agent
    iia = Agent.query.filter_by(agent_type='ingredient_intelligence').first()
    if not iia:
        iia = Agent(
            name='Ingredient Intelligence Agent',
            agent_type='ingredient_intelligence',
            description='Specialized agent for ingredient analysis and safety assessment',
            status='active',
            success_rate=0.95,
            avg_response_time=2.3,
            total_actions=156
        )
        iia.set_capabilities([
            'INCI Database Integration',
            'Chemical Structure Analysis',
            'Safety Profile Assessment',
            'Regulatory Status Tracking',
            'Literature Mining'
        ])
        db.session.add(iia)
    
    # Formulation Science Agent
    fsa = Agent.query.filter_by(agent_type='formulation_science').first()
    if not fsa:
        fsa = Agent(
            name='Formulation Science Agent',
            agent_type='formulation_science',
            description='Specialized agent for formulation analysis and optimization',
            status='active',
            success_rate=0.88,
            avg_response_time=4.1,
            total_actions=92
        )
        fsa.set_capabilities([
            'Compatibility Matrix Analysis',
            'Stability Modeling',
            'Delivery System Optimization',
            'Sensory Prediction',
            'Cost Optimization'
        ])
        db.session.add(fsa)
    
    # Clinical Evidence Agent
    cea = Agent.query.filter_by(agent_type='clinical_evidence').first()
    if not cea:
        cea = Agent(
            name='Clinical Evidence Agent',
            agent_type='clinical_evidence',
            description='Specialized agent for clinical study design and analysis',
            status='active',
            success_rate=0.91,
            avg_response_time=5.2,
            total_actions=67
        )
        cea.set_capabilities([
            'Study Design Optimization',
            'Data Quality Assessment',
            'Meta-Analysis Automation',
            'Regulatory Compliance',
            'Biostatistics'
        ])
        db.session.add(cea)
    
    db.session.commit()
    
    agents['ingredient_intelligence'] = IngredientIntelligenceAgent(iia)
    agents['formulation_science'] = FormulationScienceAgent(fsa)
    agents['clinical_evidence'] = ClinicalEvidenceAgent(cea)
    
    return agents

@skin_zone_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'Skin Zone Journal Backend',
        'version': '1.0.0'
    })

@skin_zone_bp.route('/agents', methods=['GET'])
def list_agents():
    """List all available agents"""
    agents = Agent.query.all()
    return jsonify({
        'agents': [agent.to_dict() for agent in agents],
        'total_count': len(agents),
        'active_count': len([a for a in agents if a.status == 'active'])
    })

@skin_zone_bp.route('/agents/<agent_type>', methods=['GET'])
def get_agent(agent_type):
    """Get specific agent details"""
    agent = Agent.query.filter_by(agent_type=agent_type).first()
    if not agent:
        return jsonify({'error': 'Agent not found'}), 404
    
    return jsonify(agent.to_dict())

@skin_zone_bp.route('/agents/<agent_type>/action', methods=['POST'])
def trigger_agent_action(agent_type):
    """Trigger an agent action"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    agents = get_or_create_agents()
    
    if agent_type not in agents:
        return jsonify({'error': 'Agent type not found'}), 404
    
    agent = agents[agent_type]
    action_type = data.get('action', 'unknown')
    
    # Create action record
    action_record = AgentAction(
        agent_id=agent.agent.id,
        action_type=action_type,
        input_data=json.dumps(data)
    )
    db.session.add(action_record)
    db.session.commit()
    
    try:
        # Execute agent-specific actions
        if agent_type == 'ingredient_intelligence':
            result = handle_ingredient_intelligence_action(agent, data)
        elif agent_type == 'formulation_science':
            result = handle_formulation_science_action(agent, data)
        elif agent_type == 'clinical_evidence':
            result = handle_clinical_evidence_action(agent, data)
        else:
            result = {'error': 'Unknown agent type'}
        
        # Update action record
        action_record.output_data = json.dumps(result)
        action_record.success = 'error' not in result
        action_record.confidence_score = result.get('confidence_score', 0.0)
        action_record.completed_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify(result)
        
    except Exception as e:
        action_record.error_message = str(e)
        action_record.success = False
        action_record.completed_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500

def handle_ingredient_intelligence_action(agent, data):
    """Handle Ingredient Intelligence Agent actions"""
    action = data.get('action', '')
    
    if action == 'analyze_ingredient':
        inci_name = data.get('inci_name', '')
        if not inci_name:
            return {'error': 'INCI name required'}
        
        result = agent.analyze_ingredient(inci_name)
        
        # Store ingredient in database if analysis successful
        if result.get('analysis_complete', False):
            ingredient = Ingredient.query.filter_by(inci_name=inci_name).first()
            if not ingredient:
                ingredient = Ingredient(
                    inci_name=inci_name,
                    cas_number=result['properties']['cas_number'],
                    molecular_formula=result['properties']['molecular_formula'],
                    category=result['properties']['category'],
                    safety_score=result['safety_assessment']['safety_score'],
                    regulatory_status=result['safety_assessment']['regulatory_status'],
                    max_concentration=result['safety_assessment']['max_concentration']
                )
                ingredient.set_applications(result['usage_recommendations']['applications'])
                ingredient.set_restrictions(result['usage_recommendations']['restrictions'])
                db.session.add(ingredient)
                db.session.commit()
        
        return result
    
    elif action == 'batch_analyze':
        ingredient_list = data.get('ingredients', [])
        results = []
        
        for inci_name in ingredient_list:
            result = agent.analyze_ingredient(inci_name)
            results.append(result)
        
        return {
            'batch_analysis': results,
            'total_analyzed': len(results),
            'successful_analyses': len([r for r in results if r.get('analysis_complete', False)])
        }
    
    else:
        return {'error': f'Unknown action: {action}'}

def handle_formulation_science_action(agent, data):
    """Handle Formulation Science Agent actions"""
    action = data.get('action', '')
    
    if action == 'analyze_formulation':
        ingredients = data.get('ingredients', [])
        if not ingredients:
            return {'error': 'Ingredient list required'}
        
        result = agent.analyze_formulation(ingredients)
        
        # Store formulation in database if provided
        formulation_name = data.get('formulation_name')
        if formulation_name and 'error' not in result:
            formulation = Formulation(
                name=formulation_name,
                description=data.get('description', ''),
                product_type=data.get('product_type', 'unknown'),
                stability_score=result['formulation_analysis']['stability_score'],
                efficacy_score=result['formulation_analysis']['efficacy_score']
            )
            db.session.add(formulation)
            db.session.flush()  # Get the ID
            
            # Add ingredients to formulation
            for ing_data in ingredients:
                ingredient = Ingredient.query.filter_by(inci_name=ing_data['inci_name']).first()
                if ingredient:
                    form_ing = FormulationIngredient(
                        formulation_id=formulation.id,
                        ingredient_id=ingredient.id,
                        concentration=ing_data.get('concentration', 0.0),
                        function=ing_data.get('function', ''),
                        phase=ing_data.get('phase', '')
                    )
                    db.session.add(form_ing)
            
            db.session.commit()
        
        return result
    
    elif action == 'optimize_formulation':
        ingredients = data.get('ingredients', [])
        target_properties = data.get('target_properties', {})
        
        # Simulate optimization process
        time.sleep(2.0)
        
        result = agent.analyze_formulation(ingredients)
        result['optimization'] = {
            'original_score': result['formulation_analysis']['overall_score'],
            'optimized_score': min(0.95, result['formulation_analysis']['overall_score'] + 0.1),
            'improvements': [
                'Adjusted active ingredient concentrations',
                'Optimized pH buffering system',
                'Enhanced stability with antioxidants'
            ]
        }
        
        return result
    
    else:
        return {'error': f'Unknown action: {action}'}

def handle_clinical_evidence_action(agent, data):
    """Handle Clinical Evidence Agent actions"""
    action = data.get('action', '')
    
    if action == 'design_study':
        study_parameters = data.get('parameters', {})
        if not study_parameters:
            return {'error': 'Study parameters required'}
        
        result = agent.design_clinical_study(study_parameters)
        
        # Store study design in database if provided
        if 'error' not in result and data.get('save_study', False):
            study = ClinicalStudy(
                title=study_parameters.get('title', 'Untitled Study'),
                study_type=study_parameters.get('study_type', 'efficacy'),
                methodology=json.dumps(result['methodology']),
                participant_count=result['study_design']['sample_size'],
                study_duration=result['study_design']['duration'] * 7,  # Convert weeks to days
                primary_endpoint=result['study_design']['primary_endpoint']
            )
            db.session.add(study)
            db.session.commit()
        
        return result
    
    elif action == 'analyze_study_data':
        study_data = data.get('study_data', {})
        
        # Simulate data analysis
        time.sleep(3.0)
        
        return {
            'analysis_results': {
                'primary_endpoint_met': True,
                'statistical_significance': True,
                'p_value': 0.023,
                'effect_size': 0.65,
                'confidence_interval': [0.45, 0.85]
            },
            'interpretation': {
                'clinical_significance': 'Moderate improvement observed',
                'safety_profile': 'No adverse events reported',
                'recommendations': [
                    'Results support product efficacy claims',
                    'Consider larger confirmatory study',
                    'Monitor long-term safety in post-market surveillance'
                ]
            },
            'confidence_score': 0.89
        }
    
    else:
        return {'error': f'Unknown action: {action}'}

@skin_zone_bp.route('/ingredients', methods=['GET'])
def list_ingredients():
    """List all ingredients in the database"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    category = request.args.get('category', '')
    
    query = Ingredient.query
    if category:
        query = query.filter_by(category=category)
    
    ingredients = query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'ingredients': [ing.to_dict() for ing in ingredients.items],
        'total': ingredients.total,
        'pages': ingredients.pages,
        'current_page': page,
        'per_page': per_page
    })

@skin_zone_bp.route('/ingredients/<int:ingredient_id>', methods=['GET'])
def get_ingredient(ingredient_id):
    """Get specific ingredient details"""
    ingredient = Ingredient.query.get_or_404(ingredient_id)
    return jsonify(ingredient.to_dict())

@skin_zone_bp.route('/formulations', methods=['GET'])
def list_formulations():
    """List all formulations"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    formulations = Formulation.query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'formulations': [form.to_dict() for form in formulations.items],
        'total': formulations.total,
        'pages': formulations.pages,
        'current_page': page,
        'per_page': per_page
    })

@skin_zone_bp.route('/formulations/<int:formulation_id>', methods=['GET'])
def get_formulation(formulation_id):
    """Get specific formulation details"""
    formulation = Formulation.query.get_or_404(formulation_id)
    return jsonify(formulation.to_dict())

@skin_zone_bp.route('/clinical-studies', methods=['GET'])
def list_clinical_studies():
    """List all clinical studies"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    studies = ClinicalStudy.query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'studies': [study.to_dict() for study in studies.items],
        'total': studies.total,
        'pages': studies.pages,
        'current_page': page,
        'per_page': per_page
    })

@skin_zone_bp.route('/analytics/dashboard', methods=['GET'])
def get_dashboard_analytics():
    """Get dashboard analytics data"""
    # Agent performance metrics
    agents = Agent.query.all()
    agent_metrics = []
    
    for agent in agents:
        agent_metrics.append({
            'name': agent.name,
            'type': agent.agent_type,
            'success_rate': agent.success_rate,
            'avg_response_time': agent.avg_response_time,
            'total_actions': agent.total_actions,
            'status': agent.status
        })
    
    # Database statistics
    ingredient_count = Ingredient.query.count()
    formulation_count = Formulation.query.count()
    study_count = ClinicalStudy.query.count()
    
    # Recent activity
    recent_actions = AgentAction.query.order_by(AgentAction.created_at.desc()).limit(10).all()
    
    return jsonify({
        'overview': {
            'total_ingredients': ingredient_count,
            'total_formulations': formulation_count,
            'total_studies': study_count,
            'active_agents': len([a for a in agents if a.status == 'active'])
        },
        'agent_performance': agent_metrics,
        'recent_activity': [action.to_dict() for action in recent_actions],
        'system_health': {
            'database_status': 'healthy',
            'api_status': 'operational',
            'agent_status': 'all_active'
        }
    })

@skin_zone_bp.route('/search/ingredients', methods=['GET'])
def search_ingredients():
    """Search ingredients by name or properties"""
    query = request.args.get('q', '')
    category = request.args.get('category', '')
    safety_threshold = request.args.get('safety_threshold', 0.0, type=float)
    
    ingredients_query = Ingredient.query
    
    if query:
        ingredients_query = ingredients_query.filter(
            Ingredient.inci_name.ilike(f'%{query}%')
        )
    
    if category:
        ingredients_query = ingredients_query.filter_by(category=category)
    
    if safety_threshold > 0:
        ingredients_query = ingredients_query.filter(
            Ingredient.safety_score >= safety_threshold
        )
    
    ingredients = ingredients_query.limit(50).all()
    
    return jsonify({
        'results': [ing.to_dict() for ing in ingredients],
        'count': len(ingredients),
        'query': query,
        'filters': {
            'category': category,
            'safety_threshold': safety_threshold
        }
    })

@skin_zone_bp.route('/recommendations/ingredients', methods=['POST'])
def get_ingredient_recommendations():
    """Get ingredient recommendations based on criteria"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No criteria provided'}), 400
    
    target_function = data.get('function', '')
    skin_type = data.get('skin_type', '')
    concerns = data.get('concerns', [])
    
    # Simple recommendation logic (would be more sophisticated in production)
    recommendations = []
    
    if 'anti-aging' in concerns:
        recommendations.extend([
            'Retinyl Palmitate',
            'Hyaluronic Acid',
            'Vitamin C'
        ])
    
    if 'acne' in concerns:
        recommendations.extend([
            'Salicylic Acid',
            'Niacinamide'
        ])
    
    if 'hydration' in concerns:
        recommendations.extend([
            'Hyaluronic Acid',
            'Niacinamide'
        ])
    
    # Remove duplicates and get ingredient details
    unique_recommendations = list(set(recommendations))
    ingredient_details = []
    
    for inci_name in unique_recommendations:
        ingredient = Ingredient.query.filter_by(inci_name=inci_name).first()
        if ingredient:
            ingredient_details.append(ingredient.to_dict())
    
    return jsonify({
        'recommendations': ingredient_details,
        'criteria': {
            'function': target_function,
            'skin_type': skin_type,
            'concerns': concerns
        },
        'count': len(ingredient_details)
    })

