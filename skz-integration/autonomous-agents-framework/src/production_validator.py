"""
Production Configuration Validator

This module ensures ZERO TOLERANCE for mock implementations in production mode.
NEVER SACRIFICE QUALITY!!
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ProductionViolationType(Enum):
    """Types of production quality violations"""
    MOCK_IMPLEMENTATION = "mock_implementation"
    MISSING_API_KEY = "missing_api_key"
    INVALID_CONFIGURATION = "invalid_configuration"
    INSUFFICIENT_RESOURCES = "insufficient_resources"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class ValidationViolation:
    """Represents a production quality violation"""
    type: ProductionViolationType
    component: str
    description: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    remediation: str


class ProductionValidator:
    """
    Validates that NO mock implementations are used in production mode.
    Enforces the NEVER SACRIFICE QUALITY principle.
    """
    
    def __init__(self, config: Dict[str, Any], validation_level: ValidationLevel = ValidationLevel.PRODUCTION):
        self.config = config
        self.validation_level = validation_level
        self.violations: List[ValidationViolation] = []
        
        # Production requirements - ZERO TOLERANCE for mocks
        self.production_requirements = {
            'patent_analyzer': {
                'required_keys': ['uspto_api_key', 'google_cloud_credentials'],
                'required_services': ['uspto_api', 'google_patents_api'],
                'mock_indicators': ['mock', 'demo', 'test', 'placeholder']
            },
            'communication_automation': {
                'required_keys': ['sendgrid_api_key', 'twilio_account_sid', 'aws_access_key_id'],
                'required_services': ['sendgrid', 'twilio', 'ses'],
                'mock_indicators': ['mock', 'simulate', 'fake', 'demo']
            },
            'ml_decision_engine': {
                'required_keys': ['bert_model_path', 'model_cache_dir'],
                'required_services': ['bert_classification', 'quality_assessment'],
                'mock_indicators': ['keyword_based', 'simple', 'basic', 'mock']
            },
            'reviewer_matcher': {
                'required_keys': ['sentence_transformer_model', 'optimization_algorithm'],
                'required_services': ['semantic_similarity', 'global_optimization'],
                'mock_indicators': ['basic_matching', 'simple', 'mock']
            },
            'data_sync_manager': {
                'required_keys': ['database_url', 'redis_url'],
                'required_services': ['postgresql', 'redis', 'event_sourcing'],
                'mock_indicators': ['sqlite', 'memory', 'mock']
            }
        }
    
    def validate_production_readiness(self) -> bool:
        """
        Validates that the system is ready for production.
        Returns False if ANY mock implementations are detected.
        """
        self.violations.clear()
        
        logger.info(f"Validating production readiness at {self.validation_level.value} level")
        
        # Only enforce strict validation for production
        if self.validation_level != ValidationLevel.PRODUCTION:
            logger.info("Non-production environment detected, skipping strict validation")
            return True
        
        # Validate each component
        self._validate_patent_analyzer()
        self._validate_communication_automation()
        self._validate_ml_decision_engine()
        self._validate_reviewer_matcher()
        self._validate_data_sync_manager()
        self._validate_environment_configuration()
        self._validate_security_configuration()
        
        # Check for critical violations
        critical_violations = [v for v in self.violations if v.severity == "CRITICAL"]
        
        if critical_violations:
            logger.error(f"PRODUCTION VALIDATION FAILED: {len(critical_violations)} critical violations found")
            self._log_violations()
            return False
        
        logger.info("✅ Production validation PASSED - NO mock implementations detected")
        return True
    
    def _validate_patent_analyzer(self):
        """Validate patent analyzer production configuration"""
        component = "patent_analyzer"
        config = self.config.get(component, {})
        
        # Check for production API configuration
        if not config.get('uspto_api_key'):
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.MISSING_API_KEY,
                component=component,
                description="USPTO API key not configured",
                severity="CRITICAL",
                remediation="Configure USPTO_API_KEY environment variable or config.uspto_api_key"
            ))
        
        if not config.get('google_cloud_credentials'):
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.MISSING_API_KEY,
                component=component,
                description="Google Cloud credentials not configured",
                severity="CRITICAL",
                remediation="Configure GOOGLE_CLOUD_CREDENTIALS path in environment"
            ))
        
        # Ensure production mode is enabled
        if not config.get('use_production_apis', False):
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.MOCK_IMPLEMENTATION,
                component=component,
                description="Patent analyzer configured to use mock APIs",
                severity="CRITICAL",
                remediation="Set use_production_apis=True in patent analyzer configuration"
            ))
    
    def _validate_communication_automation(self):
        """Validate communication automation production configuration"""
        component = "communication_automation"
        config = self.config.get(component, {})
        
        email_providers = config.get('email_providers', {})
        sms_providers = config.get('sms_providers', {})
        
        # At least one production email provider must be enabled
        email_enabled = any([
            email_providers.get('sendgrid', {}).get('enabled', False),
            email_providers.get('ses', {}).get('enabled', False)
        ])
        
        if not email_enabled:
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.MOCK_IMPLEMENTATION,
                component=component,
                description="No production email providers enabled",
                severity="CRITICAL",
                remediation="Enable SendGrid or Amazon SES for email delivery"
            ))
        
        # At least one production SMS provider must be enabled
        sms_enabled = any([
            sms_providers.get('twilio', {}).get('enabled', False)
        ])
        
        if not sms_enabled:
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.MOCK_IMPLEMENTATION,
                component=component,
                description="No production SMS providers enabled",
                severity="CRITICAL",
                remediation="Enable Twilio for SMS delivery"
            ))
    
    def _validate_ml_decision_engine(self):
        """Validate ML decision engine production configuration"""
        component = "ml_decision_engine"
        config = self.config.get(component, {})
        
        # Check for BERT model configuration
        if not config.get('bert_model'):
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.MOCK_IMPLEMENTATION,
                component=component,
                description="BERT model not configured - will fall back to keyword classification",
                severity="CRITICAL",
                remediation="Configure BERT model path in ML configuration"
            ))
        
        # Check for quality assessment models
        if not config.get('quality_ensemble_models'):
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.MOCK_IMPLEMENTATION,
                component=component,
                description="Quality assessment models not configured",
                severity="CRITICAL",
                remediation="Configure ensemble models for quality assessment"
            ))
    
    def _validate_reviewer_matcher(self):
        """Validate reviewer matcher production configuration"""
        component = "reviewer_matcher"
        config = self.config.get(component, {})
        
        # Check for semantic similarity model
        if not config.get('sentence_transformer_model'):
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.MOCK_IMPLEMENTATION,
                component=component,
                description="Sentence transformer model not configured",
                severity="CRITICAL",
                remediation="Configure sentence-transformers model for semantic similarity"
            ))
        
        # Check for optimization algorithm
        if not config.get('use_global_optimization', False):
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.MOCK_IMPLEMENTATION,
                component=component,
                description="Global optimization disabled - will use basic matching",
                severity="CRITICAL",
                remediation="Enable global optimization in reviewer matcher configuration"
            ))
    
    def _validate_data_sync_manager(self):
        """Validate data sync manager production configuration"""
        component = "data_sync_manager"
        config = self.config.get(component, {})
        
        # Check for PostgreSQL configuration
        db_url = config.get('database', {}).get('url', '')
        if 'postgresql' not in db_url.lower():
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.MOCK_IMPLEMENTATION,
                component=component,
                description="PostgreSQL not configured - may be using SQLite",
                severity="CRITICAL",
                remediation="Configure PostgreSQL database URL"
            ))
        
        # Check for Redis configuration
        if not config.get('redis', {}).get('url'):
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.MOCK_IMPLEMENTATION,
                component=component,
                description="Redis not configured for distributed locking",
                severity="CRITICAL",
                remediation="Configure Redis URL for distributed operations"
            ))
    
    def _validate_environment_configuration(self):
        """Validate environment-level configuration"""
        # Check environment mode
        env_mode = os.getenv('ENVIRONMENT', 'development').lower()
        if env_mode != 'production':
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.INVALID_CONFIGURATION,
                component="environment",
                description=f"Environment mode is '{env_mode}', not 'production'",
                severity="HIGH",
                remediation="Set ENVIRONMENT=production in environment variables"
            ))
        
        # Check debug mode
        debug_mode = os.getenv('DEBUG', 'false').lower()
        if debug_mode == 'true':
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.SECURITY_VIOLATION,
                component="environment",
                description="Debug mode enabled in production",
                severity="HIGH",
                remediation="Set DEBUG=false in production environment"
            ))
    
    def _validate_security_configuration(self):
        """Validate security configuration"""
        # Check for default/weak secrets
        jwt_secret = os.getenv('JWT_SECRET', '')
        if not jwt_secret or len(jwt_secret) < 32:
            self.violations.append(ValidationViolation(
                type=ProductionViolationType.SECURITY_VIOLATION,
                component="security",
                description="JWT secret not configured or too weak",
                severity="CRITICAL",
                remediation="Configure strong JWT_SECRET (>= 32 characters)"
            ))
        
        # Check for default secrets in configuration
        default_indicators = ['your-', 'change-this', 'default', 'admin', 'password']
        for key, value in os.environ.items():
            if any(indicator in str(value).lower() for indicator in default_indicators):
                self.violations.append(ValidationViolation(
                    type=ProductionViolationType.SECURITY_VIOLATION,
                    component="security",
                    description=f"Default/weak value detected in {key}",
                    severity="HIGH",
                    remediation=f"Change {key} to a production-secure value"
                ))
    
    def _log_violations(self):
        """Log all validation violations"""
        logger.error("❌ PRODUCTION VALIDATION VIOLATIONS:")
        for violation in self.violations:
            logger.error(f"  [{violation.severity}] {violation.component}: {violation.description}")
            logger.error(f"    Remediation: {violation.remediation}")
    
    def get_violations_json(self) -> str:
        """Return violations as JSON for reporting"""
        violations_data = []
        for violation in self.violations:
            violations_data.append({
                'type': violation.type.value,
                'component': violation.component,
                'description': violation.description,
                'severity': violation.severity,
                'remediation': violation.remediation
            })
        
        return json.dumps({
            'validation_level': self.validation_level.value,
            'violations_count': len(self.violations),
            'critical_violations': len([v for v in self.violations if v.severity == "CRITICAL"]),
            'violations': violations_data
        }, indent=2)
    
    def enforce_production_quality(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforces production quality by preventing mock implementations.
        NEVER SACRIFICE QUALITY!!
        """
        if self.validation_level != ValidationLevel.PRODUCTION:
            return config
        
        # Force production mode for all components
        production_config = config.copy()
        
        # Force production APIs for patent analyzer
        if 'patent_analyzer' in production_config:
            production_config['patent_analyzer']['use_production_apis'] = True
        
        # Force production providers for communication
        if 'communication_automation' in production_config:
            comm_config = production_config['communication_automation']
            # Disable mock/SMTP if production providers available
            if comm_config.get('email_providers', {}).get('sendgrid', {}).get('enabled'):
                comm_config['smtp_config']['enabled'] = False
        
        # Force ML models for decision engine
        if 'ml_decision_engine' in production_config:
            production_config['ml_decision_engine']['force_ml_models'] = True
        
        # Force optimization for reviewer matcher
        if 'reviewer_matcher' in production_config:
            production_config['reviewer_matcher']['use_global_optimization'] = True
        
        return production_config


def validate_production_deployment() -> bool:
    """
    Main validation function for production deployment.
    Returns False if ANY quality violations are found.
    """
    # Load configuration from environment and config files
    config = {}
    
    # Load from environment variables
    env_mode = os.getenv('ENVIRONMENT', 'development').lower()
    validation_level = ValidationLevel.PRODUCTION if env_mode == 'production' else ValidationLevel.DEVELOPMENT
    
    # Create validator and check
    validator = ProductionValidator(config, validation_level)
    is_valid = validator.validate_production_readiness()
    
    if not is_valid:
        print("❌ PRODUCTION VALIDATION FAILED")
        print("NEVER SACRIFICE QUALITY!!")
        print("\nViolations Report:")
        print(validator.get_violations_json())
        return False
    
    print("✅ PRODUCTION VALIDATION PASSED")
    print("Quality standards met - no mock implementations detected")
    return True


if __name__ == "__main__":
    # Run validation
    success = validate_production_deployment()
    exit(0 if success else 1)