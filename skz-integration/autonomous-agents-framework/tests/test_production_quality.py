"""
Production Quality Tests - Ensure ZERO mock implementations in production

This test suite validates the NEVER SACRIFICE QUALITY principle by ensuring
no mock or placeholder implementations are used when ENVIRONMENT=production.
"""

import os
import pytest
import asyncio
import logging
from unittest.mock import patch, MagicMock
import sys
import tempfile

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from production_validator import ProductionValidator, ValidationLevel, validate_production_deployment


class TestProductionQualityEnforcement:
    """Test that production mode prevents any mock usage"""
    
    def setup_method(self):
        """Set up test environment"""
        # Store original environment
        self.original_env = os.getenv('ENVIRONMENT')
        
        # Set production environment for testing
        os.environ['ENVIRONMENT'] = 'production'
        
        # Configure logging
        logging.basicConfig(level=logging.DEBUG)
    
    def teardown_method(self):
        """Restore original environment"""
        if self.original_env:
            os.environ['ENVIRONMENT'] = self.original_env
        elif 'ENVIRONMENT' in os.environ:
            del os.environ['ENVIRONMENT']
    
    def test_production_validator_creation(self):
        """Test that production validator can be created"""
        config = {}
        validator = ProductionValidator(config, ValidationLevel.PRODUCTION)
        assert validator is not None
        assert validator.validation_level == ValidationLevel.PRODUCTION
    
    def test_production_validation_fails_without_config(self):
        """Test that production validation fails when required config is missing"""
        config = {}
        validator = ProductionValidator(config, ValidationLevel.PRODUCTION)
        
        # Should fail due to missing configuration
        is_valid = validator.validate_production_readiness()
        assert not is_valid
        assert len(validator.violations) > 0
        
        # Check that critical violations are found
        critical_violations = [v for v in validator.violations if v.severity == "CRITICAL"]
        assert len(critical_violations) > 0
    
    def test_communication_automation_blocks_mock_email(self):
        """Test that email mock is blocked in production"""
        from models.communication_automation import CommunicationAutomation, CommunicationMessage, MessageRecipient
        
        # Create communication automation instance
        config = {}
        comm = CommunicationAutomation(config)
        
        # Create test message
        recipient = MessageRecipient(
            email="test@example.com",
            name="Test User",
            user_type="author"
        )
        
        message = CommunicationMessage(
            message_id="test-123",
            recipient=recipient,
            subject="Test Message",
            body="Test body",
            message_type="email",
            priority="normal",
            sender_agent="test_agent"
        )
        
        # Should raise ValueError in production mode
        with pytest.raises(ValueError) as exc_info:
            asyncio.run(comm._send_email_mock(message))
        
        assert "PRODUCTION VIOLATION" in str(exc_info.value)
        assert "NEVER SACRIFICE QUALITY" in str(exc_info.value)
    
    def test_communication_automation_blocks_mock_sms(self):
        """Test that SMS mock is blocked in production"""
        from models.communication_automation import CommunicationAutomation, CommunicationMessage, MessageRecipient
        
        config = {}
        comm = CommunicationAutomation(config)
        
        recipient = MessageRecipient(
            phone="+1234567890",
            name="Test User",
            user_type="author"
        )
        
        message = CommunicationMessage(
            message_id="test-123",
            recipient=recipient,
            subject="Test SMS",
            body="Test SMS body",
            message_type="sms",
            priority="normal",
            sender_agent="test_agent"
        )
        
        # Should raise ValueError in production mode
        with pytest.raises(ValueError) as exc_info:
            asyncio.run(comm._send_sms_mock(message))
        
        assert "PRODUCTION VIOLATION" in str(exc_info.value)
        assert "NEVER SACRIFICE QUALITY" in str(exc_info.value)
    
    def test_patent_analyzer_blocks_mock_uspto(self):
        """Test that USPTO mock is blocked in production"""
        from models.patent_analyzer import PatentAnalyzer
        
        config = {'use_production_apis': False}  # Force mock usage
        analyzer = PatentAnalyzer(config)
        
        # Should raise ValueError in production mode
        with pytest.raises(ValueError) as exc_info:
            asyncio.run(analyzer._search_uspto_mock("test query", None, 10))
        
        assert "PRODUCTION VIOLATION" in str(exc_info.value)
        assert "NEVER SACRIFICE QUALITY" in str(exc_info.value)
    
    def test_patent_analyzer_blocks_mock_google_patents(self):
        """Test that Google Patents mock is blocked in production"""
        from models.patent_analyzer import PatentAnalyzer
        
        config = {'use_production_apis': False}
        analyzer = PatentAnalyzer(config)
        
        # Should raise ValueError in production mode
        with pytest.raises(ValueError) as exc_info:
            asyncio.run(analyzer._search_google_patents_mock("test query", None, 10))
        
        assert "PRODUCTION VIOLATION" in str(exc_info.value)
        assert "NEVER SACRIFICE QUALITY" in str(exc_info.value)
    
    def test_ml_decision_engine_blocks_keyword_classification(self):
        """Test that keyword classification is blocked in production when ML is required"""
        from models.ml_decision_engine import MLDecisionEngine
        
        config = {'force_ml_models': True}  # Force production ML models
        engine = MLDecisionEngine(config)
        
        # Should raise ValueError in production mode with force_ml_models
        with pytest.raises(ValueError) as exc_info:
            engine._classify_text_keywords("test text", ["category1", "category2"])
        
        assert "PRODUCTION VIOLATION" in str(exc_info.value)
        assert "NEVER SACRIFICE QUALITY" in str(exc_info.value)
    
    def test_production_config_enforcement(self):
        """Test that production configuration is enforced"""
        config = {
            'patent_analyzer': {'use_production_apis': False},
            'communication_automation': {
                'email_providers': {'sendgrid': {'enabled': False}},
                'sms_providers': {'twilio': {'enabled': False}}
            },
            'ml_decision_engine': {'force_ml_models': False},
            'reviewer_matcher': {'use_global_optimization': False}
        }
        
        validator = ProductionValidator(config, ValidationLevel.PRODUCTION)
        enforced_config = validator.enforce_production_quality(config)
        
        # Should force production settings
        assert enforced_config['patent_analyzer']['use_production_apis'] == True
        assert enforced_config['ml_decision_engine']['force_ml_models'] == True
        assert enforced_config['reviewer_matcher']['use_global_optimization'] == True
    
    def test_development_mode_allows_mocks(self):
        """Test that development mode allows mock implementations"""
        # Temporarily set to development mode
        os.environ['ENVIRONMENT'] = 'development'
        
        try:
            from models.communication_automation import CommunicationAutomation, CommunicationMessage, MessageRecipient
            
            config = {}
            comm = CommunicationAutomation(config)
            
            recipient = MessageRecipient(
                email="test@example.com",
                name="Test User",
                user_type="author"
            )
            
            message = CommunicationMessage(
                message_id="test-123",
                recipient=recipient,
                subject="Test Message",
                body="Test body",
                message_type="email",
                priority="normal",
                sender_agent="test_agent"
            )
            
            # Should work fine in development mode
            result = asyncio.run(comm._send_email_mock(message))
            assert result == True
            
        finally:
            # Restore production mode
            os.environ['ENVIRONMENT'] = 'production'
    
    def test_production_deployment_validation(self):
        """Test the main production deployment validation function"""
        # Should fail in production mode without proper configuration
        is_valid = validate_production_deployment()
        assert not is_valid
    
    def test_violation_reporting(self):
        """Test that violations are properly reported"""
        config = {}
        validator = ProductionValidator(config, ValidationLevel.PRODUCTION)
        validator.validate_production_readiness()
        
        # Should have violations
        assert len(validator.violations) > 0
        
        # Test JSON reporting
        violations_json = validator.get_violations_json()
        assert violations_json is not None
        assert "violations_count" in violations_json
        assert "critical_violations" in violations_json
    
    def test_security_configuration_validation(self):
        """Test that security configuration is validated"""
        # Set weak JWT secret
        os.environ['JWT_SECRET'] = 'weak'
        
        try:
            config = {}
            validator = ProductionValidator(config, ValidationLevel.PRODUCTION)
            validator._validate_security_configuration()
            
            # Should have security violations
            security_violations = [v for v in validator.violations 
                                 if v.component == "security"]
            assert len(security_violations) > 0
            
        finally:
            if 'JWT_SECRET' in os.environ:
                del os.environ['JWT_SECRET']
    
    def test_environment_configuration_validation(self):
        """Test that environment configuration is validated"""
        # Set debug mode in production
        os.environ['DEBUG'] = 'true'
        
        try:
            config = {}
            validator = ProductionValidator(config, ValidationLevel.PRODUCTION)
            validator._validate_environment_configuration()
            
            # Should have environment violations
            env_violations = [v for v in validator.violations 
                            if v.component == "environment"]
            assert len(env_violations) > 0
            
        finally:
            if 'DEBUG' in os.environ:
                del os.environ['DEBUG']


class TestProductionConfigurationValidation:
    """Test specific configuration validation scenarios"""
    
    def test_patent_analyzer_production_config(self):
        """Test patent analyzer production configuration validation"""
        # Missing API keys
        config = {
            'patent_analyzer': {
                'use_production_apis': True,
                # Missing uspto_api_key and google_cloud_credentials
            }
        }
        
        validator = ProductionValidator(config, ValidationLevel.PRODUCTION)
        validator._validate_patent_analyzer()
        
        violations = [v for v in validator.violations if v.component == "patent_analyzer"]
        assert len(violations) >= 2  # Should have violations for missing API keys
    
    def test_communication_automation_production_config(self):
        """Test communication automation production configuration validation"""
        # No production providers enabled
        config = {
            'communication_automation': {
                'email_providers': {
                    'sendgrid': {'enabled': False},
                    'ses': {'enabled': False}
                },
                'sms_providers': {
                    'twilio': {'enabled': False}
                }
            }
        }
        
        validator = ProductionValidator(config, ValidationLevel.PRODUCTION)
        validator._validate_communication_automation()
        
        violations = [v for v in validator.violations 
                     if v.component == "communication_automation"]
        assert len(violations) >= 2  # Should have violations for email and SMS
    
    def test_ml_decision_engine_production_config(self):
        """Test ML decision engine production configuration validation"""
        # Missing BERT models
        config = {
            'ml_decision_engine': {
                # Missing bert_model and quality_ensemble_models
            }
        }
        
        validator = ProductionValidator(config, ValidationLevel.PRODUCTION)
        validator._validate_ml_decision_engine()
        
        violations = [v for v in validator.violations 
                     if v.component == "ml_decision_engine"]
        assert len(violations) >= 1  # Should have violations for missing models


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])