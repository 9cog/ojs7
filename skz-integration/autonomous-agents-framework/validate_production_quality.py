#!/usr/bin/env python3
"""
Production Quality Gate - Startup Validation

This script MUST be run before any production deployment to ensure
ZERO TOLERANCE for mock implementations.

NEVER SACRIFICE QUALITY!!
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from production_validator import ProductionValidator, ValidationLevel, validate_production_deployment
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Cannot import production validator: {e}")
    print("Ensure you are running from the correct directory and dependencies are installed.")
    sys.exit(1)


def check_environment():
    """Check that environment is properly configured for production validation"""
    env_mode = os.getenv('ENVIRONMENT', 'development').lower()
    
    print(f"🔍 Environment Mode: {env_mode}")
    
    if env_mode == 'production':
        print("✅ Production mode detected - enforcing ZERO TOLERANCE for mocks")
        return True
    else:
        print("ℹ️  Development/Staging mode - allowing mock implementations")
        return False


def load_configuration() -> Dict[str, Any]:
    """Load configuration from various sources"""
    config = {}
    
    # Load from environment variables
    config['environment'] = {
        'mode': os.getenv('ENVIRONMENT', 'development'),
        'debug': os.getenv('DEBUG', 'false').lower() == 'true',
        'jwt_secret': os.getenv('JWT_SECRET', ''),
    }
    
    # Patent analyzer configuration
    config['patent_analyzer'] = {
        'uspto_api_key': os.getenv('USPTO_API_KEY'),
        'google_cloud_credentials': os.getenv('GOOGLE_CLOUD_CREDENTIALS'),
        'use_production_apis': os.getenv('USE_PRODUCTION_APIS', 'false').lower() == 'true'
    }
    
    # Communication configuration
    config['communication_automation'] = {
        'email_providers': {
            'sendgrid': {
                'enabled': os.getenv('SENDGRID_API_KEY') is not None,
                'api_key': os.getenv('SENDGRID_API_KEY')
            },
            'ses': {
                'enabled': all([
                    os.getenv('AWS_ACCESS_KEY_ID'),
                    os.getenv('AWS_SECRET_ACCESS_KEY')
                ]),
                'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
                'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY')
            }
        },
        'sms_providers': {
            'twilio': {
                'enabled': all([
                    os.getenv('TWILIO_ACCOUNT_SID'),
                    os.getenv('TWILIO_AUTH_TOKEN')
                ]),
                'account_sid': os.getenv('TWILIO_ACCOUNT_SID'),
                'auth_token': os.getenv('TWILIO_AUTH_TOKEN')
            }
        }
    }
    
    # ML configuration
    config['ml_decision_engine'] = {
        'bert_model': os.getenv('BERT_MODEL_PATH'),
        'model_cache_dir': os.getenv('MODEL_CACHE_DIR'),
        'quality_ensemble_models': os.getenv('QUALITY_ENSEMBLE_MODELS', '').split(',') if os.getenv('QUALITY_ENSEMBLE_MODELS') else [],
        'force_ml_models': os.getenv('FORCE_ML_MODELS', 'false').lower() == 'true'
    }
    
    # Reviewer matcher configuration
    config['reviewer_matcher'] = {
        'sentence_transformer_model': os.getenv('SENTENCE_TRANSFORMER_MODEL'),
        'use_global_optimization': os.getenv('USE_GLOBAL_OPTIMIZATION', 'false').lower() == 'true'
    }
    
    # Data sync manager configuration
    config['data_sync_manager'] = {
        'database': {
            'url': os.getenv('DATABASE_URL', '')
        },
        'redis': {
            'url': os.getenv('REDIS_URL', '')
        }
    }
    
    return config


def print_validation_summary(validator: ProductionValidator):
    """Print a comprehensive validation summary"""
    violations = validator.violations
    
    print("\n" + "="*80)
    print("📋 PRODUCTION QUALITY VALIDATION REPORT")
    print("="*80)
    
    print(f"Validation Level: {validator.validation_level.value}")
    print(f"Total Violations: {len(violations)}")
    
    # Count by severity
    critical = len([v for v in violations if v.severity == "CRITICAL"])
    high = len([v for v in violations if v.severity == "HIGH"])
    medium = len([v for v in violations if v.severity == "MEDIUM"])
    low = len([v for v in violations if v.severity == "LOW"])
    
    print(f"Critical: {critical}, High: {high}, Medium: {medium}, Low: {low}")
    
    if violations:
        print("\n🚨 VIOLATIONS FOUND:")
        print("-" * 80)
        
        for violation in violations:
            severity_icon = {
                "CRITICAL": "🔴",
                "HIGH": "🟠", 
                "MEDIUM": "🟡",
                "LOW": "🔵"
            }.get(violation.severity, "⚪")
            
            print(f"{severity_icon} [{violation.severity}] {violation.component}")
            print(f"   Description: {violation.description}")
            print(f"   Remediation: {violation.remediation}")
            print()
    
    print("="*80)
    
    if critical > 0:
        print("❌ PRODUCTION DEPLOYMENT BLOCKED")
        print("NEVER SACRIFICE QUALITY!!")
        print("Fix all CRITICAL violations before proceeding.")
    else:
        print("✅ PRODUCTION QUALITY VALIDATION PASSED")
        print("No critical violations found - deployment approved.")


def check_file_mock_indicators(file_path: Path) -> List[str]:
    """Check a Python file for mock implementation indicators"""
    mock_indicators = [
        'mock', 'Mock', 'MOCK',
        'simulate', 'Simulate', 'SIMULATE',
        'fake', 'Fake', 'FAKE',
        'demo', 'Demo', 'DEMO',
        'test', 'Test', 'TEST',
        'placeholder', 'Placeholder', 'PLACEHOLDER',
        'TODO', 'FIXME', 'XXX',
        'not implemented', 'Not implemented', 'NOT IMPLEMENTED'
    ]
    
    found_indicators = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            for indicator in mock_indicators:
                if indicator in content:
                    # Count occurrences
                    count = content.count(indicator)
                    found_indicators.append(f"{indicator} ({count} times)")
    
    except Exception as e:
        found_indicators.append(f"Error reading file: {e}")
    
    return found_indicators


def scan_codebase_for_mocks():
    """Scan the codebase for potential mock implementations"""
    print("\n🔍 SCANNING CODEBASE FOR MOCK IMPLEMENTATIONS...")
    print("-" * 60)
    
    src_dir = Path(__file__).parent / 'src'
    
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return
    
    mock_files = []
    
    for py_file in src_dir.rglob('*.py'):
        if py_file.name.startswith('test_'):
            continue  # Skip test files
        
        indicators = check_file_mock_indicators(py_file)
        if indicators:
            relative_path = py_file.relative_to(src_dir)
            mock_files.append((relative_path, indicators))
    
    if mock_files:
        print(f"⚠️  Found {len(mock_files)} files with mock indicators:")
        for file_path, indicators in mock_files:
            print(f"  📄 {file_path}")
            for indicator in indicators[:3]:  # Show first 3 indicators
                print(f"     - {indicator}")
            if len(indicators) > 3:
                print(f"     ... and {len(indicators) - 3} more")
    else:
        print("✅ No obvious mock indicators found in codebase")


def create_quality_report(validator: ProductionValidator):
    """Create a detailed quality report file"""
    report_data = {
        'validation_timestamp': __import__('datetime').datetime.now().isoformat(),
        'validation_level': validator.validation_level.value,
        'environment': os.getenv('ENVIRONMENT', 'unknown'),
        'total_violations': len(validator.violations),
        'critical_violations': len([v for v in validator.violations if v.severity == "CRITICAL"]),
        'violations': []
    }
    
    for violation in validator.violations:
        report_data['violations'].append({
            'type': violation.type.value,
            'component': violation.component,
            'description': violation.description,
            'severity': violation.severity,
            'remediation': violation.remediation
        })
    
    report_file = Path('production_quality_report.json')
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"📊 Quality report saved to: {report_file}")


def main():
    """Main validation routine"""
    print("🛡️  PRODUCTION QUALITY GATE - STARTUP VALIDATION")
    print("NEVER SACRIFICE QUALITY!!")
    print("=" * 60)
    
    # Check environment
    is_production = check_environment()
    
    # Load configuration
    print("\n📋 Loading configuration...")
    config = load_configuration()
    
    # Determine validation level
    validation_level = ValidationLevel.PRODUCTION if is_production else ValidationLevel.DEVELOPMENT
    
    # Create validator
    print(f"\n🔧 Creating validator (Level: {validation_level.value})...")
    validator = ProductionValidator(config, validation_level)
    
    # Run validation
    print("\n🔍 Running production readiness validation...")
    is_valid = validator.validate_production_readiness()
    
    # Print summary
    print_validation_summary(validator)
    
    # Scan codebase
    scan_codebase_for_mocks()
    
    # Create report
    create_quality_report(validator)
    
    # Final decision
    print("\n" + "="*80)
    if is_production and not is_valid:
        print("🚫 PRODUCTION DEPLOYMENT REJECTED")
        print("Critical quality violations prevent production deployment.")
        print("NEVER SACRIFICE QUALITY!!")
        print("\nRemediation required:")
        critical_violations = [v for v in validator.violations if v.severity == "CRITICAL"]
        for i, violation in enumerate(critical_violations[:5], 1):  # Show first 5
            print(f"{i}. {violation.remediation}")
        
        if len(critical_violations) > 5:
            print(f"... and {len(critical_violations) - 5} more critical issues")
        
        sys.exit(1)
    elif is_production and is_valid:
        print("✅ PRODUCTION DEPLOYMENT APPROVED")
        print("All quality checks passed - system ready for production.")
        sys.exit(0)
    else:
        print("ℹ️  DEVELOPMENT MODE - Quality checks informational only")
        if not is_valid:
            print("⚠️  Issues found that would block production deployment")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)