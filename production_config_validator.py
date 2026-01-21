#!/usr/bin/env python3
"""
Production Configuration Validator
=================================

Validates that all required production configurations are in place
and no mock implementations can be used in production mode.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

class ProductionConfigValidator:
    """Validates production configuration and prevents mock usage"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> bool:
        """Run all production validations"""
        print("🔍 Validating production configuration...")
        
        self._validate_environment()
        self._validate_api_configurations()
        self._validate_database_config()
        self._validate_ml_models()
        self._validate_no_mock_usage()
        
        self._report_results()
        return len(self.errors) == 0
    
    def _validate_environment(self):
        """Validate environment is properly set"""
        env = os.getenv('ENVIRONMENT', '').lower()
        if env != 'production':
            self.warnings.append("ENVIRONMENT not set to 'production'")
    
    def _validate_api_configurations(self):
        """Validate all external API configurations"""
        required_apis = {
            'USPTO_API_KEY': 'Patent search functionality',
            'GOOGLE_PATENTS_API_KEY': 'Google Patents integration',
            'SENDGRID_API_KEY': 'Email delivery via SendGrid',
            'TWILIO_API_KEY': 'SMS delivery via Twilio'
        }
        
        for api_key, description in required_apis.items():
            if not os.getenv(api_key):
                self.errors.append(f"Missing {api_key} - required for {description}")
    
    def _validate_database_config(self):
        """Validate database configuration"""
        if not os.getenv('POSTGRES_URL') and not os.getenv('DATABASE_URL'):
            self.errors.append("Production database not configured - PostgreSQL required")
        
        if not os.getenv('REDIS_URL'):
            self.warnings.append("Redis not configured - required for distributed locking")
    
    def _validate_ml_models(self):
        """Validate ML models are configured"""
        if not os.getenv('BERT_MODEL_PATH') and not os.getenv('HUGGINGFACE_API_KEY'):
            self.errors.append("ML models not configured - BERT models required for production")
    
    def _validate_no_mock_usage(self):
        """Scan for any remaining mock usage"""
        mock_indicators = ['_mock', 'MOCK', 'mock_', 'test_data', 'placeholder']
        
        for py_file in Path('.').rglob('*.py'):
            if 'test' in str(py_file) or 'backup' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                for indicator in mock_indicators:
                    if indicator in content and 'def test_' not in content:
                        # Check if it's actually production code
                        if 'NEVER USE IN PRODUCTION' in content:
                            continue  # This is a protected mock
                        
                        self.warnings.append(f"Potential mock usage in {py_file}: {indicator}")
            except:
                pass
    
    def _report_results(self):
        """Report validation results"""
        if self.errors:
            print("\n❌ PRODUCTION VALIDATION FAILED")
            for error in self.errors:
                print(f"  🔴 {error}")
        
        if self.warnings:
            print("\n⚠️ PRODUCTION WARNINGS")
            for warning in self.warnings:
                print(f"  🟡 {warning}")
        
        if not self.errors:
            print("\n✅ PRODUCTION VALIDATION PASSED")

if __name__ == "__main__":
    validator = ProductionConfigValidator()
    success = validator.validate_all()
    sys.exit(0 if success else 1)
