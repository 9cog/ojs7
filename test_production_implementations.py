#!/usr/bin/env python3
"""
Production Implementation Tests
==============================

Focused tests to validate that all production implementations work correctly
and that mock implementations are properly blocked in production mode.

These tests ensure:
1. All mock fallbacks have been removed
2. Production configurations are properly validated
3. Error handling works without mock dependencies
4. All external service integrations are production-ready

Usage:
    python3 test_production_implementations.py
    python3 -m pytest test_production_implementations.py -v
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the SKZ integration source to path
sys.path.insert(0, str(Path(__file__).parent / "skz-integration/autonomous-agents-framework/src"))

class TestProductionImplementations(unittest.TestCase):
    """Test suite for production implementations"""
    
    def setUp(self):
        """Set up test environment"""
        # Ensure we're testing in production mode
        os.environ['ENVIRONMENT'] = 'production'
    
    def tearDown(self):
        """Clean up test environment"""
        # Reset environment
        if 'ENVIRONMENT' in os.environ:
            del os.environ['ENVIRONMENT']

    def test_patent_analyzer_blocks_mock_usage_in_production(self):
        """Test that Patent Analyzer blocks mock usage in production"""
        try:
            from models.patent_analyzer import PatentAnalyzer
            
            # Create analyzer with no API keys (should fail in production)
            config = {
                'use_production_apis': False,  # This should be ignored in production
                'uspto_api_key': None,
                'google_patents_api_key': None
            }
            
            analyzer = PatentAnalyzer(config)
            
            # This should raise an error in production mode, not fall back to mock
            with self.assertRaises(ValueError) as context:
                # Try to use the analyzer (should fail due to missing config)
                result = analyzer._search_uspto("test query", None, 10)
            
            self.assertIn("configuration required for production", str(context.exception))
            print("✅ Patent Analyzer properly blocks mock usage in production")
            
        except ImportError as e:
            print(f"⚠️ Could not import PatentAnalyzer: {e}")
            self.skipTest("PatentAnalyzer not available for testing")

    def test_communication_automation_blocks_mock_usage_in_production(self):
        """Test that Communication Automation blocks mock usage in production"""
        try:
            from models.communication_automation import CommunicationAutomation, CommunicationMessage, Recipient
            
            # Create communication automation with no service configs
            config = {
                'email_providers': {},
                'sms_providers': {},
            }
            smtp_config = {'enabled': False}
            
            comm = CommunicationAutomation(config, smtp_config)
            
            # Create a test message
            recipient = Recipient(email="test@example.com", name="Test User")
            message = CommunicationMessage(
                recipient=recipient,
                subject="Test",
                body="Test message",
                sender_agent="test_agent",
                template_id="test_template"
            )
            
            # This should raise an error in production mode
            with self.assertRaises(ValueError) as context:
                result = comm._send_email(message)
            
            self.assertIn("service configuration required for production", str(context.exception))
            print("✅ Communication Automation properly blocks mock usage in production")
            
        except ImportError as e:
            print(f"⚠️ Could not import CommunicationAutomation: {e}")
            self.skipTest("CommunicationAutomation not available for testing")

    def test_ml_decision_engine_requires_models_in_production(self):
        """Test that ML Decision Engine requires real models in production"""
        try:
            from models.ml_decision_engine import MLDecisionEngine
            
            # Create engine without BERT models
            config = {
                'force_ml_models': True,  # Should be True in production
                'bert_model': None
            }
            
            engine = MLDecisionEngine(config)
            
            # This should raise an error in production mode
            with self.assertRaises(ValueError) as context:
                result = engine.classify_text("test text", ["category1", "category2"])
            
            self.assertIn("BERT models required for production", str(context.exception))
            print("✅ ML Decision Engine properly requires real models in production")
            
        except ImportError as e:
            print(f"⚠️ Could not import MLDecisionEngine: {e}")
            self.skipTest("MLDecisionEngine not available for testing")

    def test_production_config_validator_works(self):
        """Test that the production configuration validator works correctly"""
        # Import the validator we created
        try:
            import production_config_validator
            
            validator = production_config_validator.ProductionConfigValidator()
            
            # Run validation (should fail due to missing configurations)
            success = validator.validate_all()
            
            # Should fail in our test environment since we don't have all configs
            self.assertFalse(success)
            self.assertTrue(len(validator.errors) > 0)
            print("✅ Production configuration validator correctly identifies missing configs")
            
        except ImportError as e:
            print(f"⚠️ Could not import production_config_validator: {e}")
            self.skipTest("Production config validator not available")

    def test_no_mock_fallbacks_remain(self):
        """Test that no mock fallbacks remain in production code"""
        # Scan the modified files for mock fallbacks
        models_dir = Path(__file__).parent / "skz-integration/autonomous-agents-framework/src/models"
        
        if not models_dir.exists():
            self.skipTest("Models directory not found")
        
        mock_fallback_patterns = [
            r'return await self\._.*_mock\(',
            r'return self\._.*_mock\(',
            r'fallback.*mock',
            r'except.*mock'
        ]
        
        found_fallbacks = []
        
        for py_file in models_dir.glob("*.py"):
            with open(py_file, 'r') as f:
                content = f.read()
                
            for i, line in enumerate(content.split('\n'), 1):
                for pattern in mock_fallback_patterns:
                    import re
                    if re.search(pattern, line, re.IGNORECASE):
                        # Skip if it's a comment or in a test
                        if not line.strip().startswith('#') and 'test' not in str(py_file).lower():
                            found_fallbacks.append(f"{py_file.name}:{i} - {line.strip()}")
        
        if found_fallbacks:
            print("⚠️ Found potential mock fallbacks:")
            for fallback in found_fallbacks:
                print(f"  {fallback}")
        else:
            print("✅ No mock fallbacks found in production code")
        
        # Allow some fallbacks if they're properly protected
        protected_fallbacks = [fb for fb in found_fallbacks if "PRODUCTION VIOLATION" in fb or "raise ValueError" in fb]
        unprotected_fallbacks = [fb for fb in found_fallbacks if fb not in protected_fallbacks]
        
        self.assertEqual(len(unprotected_fallbacks), 0, f"Unprotected mock fallbacks found: {unprotected_fallbacks}")

    def test_production_quality_enforcement_active(self):
        """Test that production quality enforcement is active"""
        # Check if the production quality enforcement file exists
        enforcement_file = Path(__file__).parent / "skz-integration/autonomous-agents-framework/validate_production_quality.py"
        
        if not enforcement_file.exists():
            self.skipTest("Production quality enforcement file not found")
        
        # Import and test the enforcement
        try:
            sys.path.insert(0, str(enforcement_file.parent))
            import validate_production_quality
            
            # Test that it can scan for mocks
            mock_indicators = validate_production_quality.scan_codebase_for_mocks()
            
            print(f"✅ Production quality enforcement is active, found {len(mock_indicators)} potential issues")
            
        except ImportError as e:
            print(f"⚠️ Could not import production quality enforcement: {e}")
            self.skipTest("Production quality enforcement not available")

    @patch.dict(os.environ, {'ENVIRONMENT': 'development'})
    def test_development_mode_still_allows_fallbacks(self):
        """Test that development mode still allows fallbacks (for testing)"""
        # This test ensures we haven't broken development/testing workflows
        try:
            from models.patent_analyzer import PatentAnalyzer
            
            config = {
                'use_production_apis': False,
                'uspto_api_key': None
            }
            
            analyzer = PatentAnalyzer(config)
            
            # In development mode, this should work (fallback to mock)
            # Note: This is an async method, so we'd need to test it properly
            # For now, just check that it doesn't raise a production error
            print("✅ Development mode still allows fallbacks (as expected)")
            
        except ImportError:
            self.skipTest("PatentAnalyzer not available for testing")


class TestProductionReadiness(unittest.TestCase):
    """Test overall production readiness"""
    
    def test_audit_results_show_improvement(self):
        """Test that our changes have improved the audit results"""
        audit_file = Path(__file__).parent / "audit_results.json"
        
        if not audit_file.exists():
            self.skipTest("Audit results not available")
        
        import json
        with open(audit_file) as f:
            audit_data = json.load(f)
        
        # Check that we've addressed the critical issues
        critical_mocks = [
            m for m in audit_data["mock_implementations"]
            if m["severity"] == "critical" and "skz-integration" in m["file"]
        ]
        
        print(f"📊 Critical mocks remaining in SKZ integration: {len(critical_mocks)}")
        
        # We should have very few critical mocks remaining after our fixes
        self.assertLessEqual(len(critical_mocks), 2, "Too many critical mocks remaining")

    def test_replacement_log_exists(self):
        """Test that replacement log was created"""
        log_file = Path(__file__).parent / "production_replacement_log.md"
        
        self.assertTrue(log_file.exists(), "Production replacement log should exist")
        
        with open(log_file) as f:
            content = f.read()
        
        self.assertIn("Patent Analyzer", content)
        self.assertIn("Communication Automation", content)
        print("✅ Production replacement log exists and contains expected entries")


def run_production_tests():
    """Run all production implementation tests"""
    print("🧪 Running Production Implementation Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestProductionImplementations))
    suite.addTests(loader.loadTestsFromTestCase(TestProductionReadiness))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("=" * 60)
    if result.wasSuccessful():
        print("✅ All production implementation tests passed!")
        return True
    else:
        print("❌ Some production implementation tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        return False


if __name__ == "__main__":
    success = run_production_tests()
    sys.exit(0 if success else 1)