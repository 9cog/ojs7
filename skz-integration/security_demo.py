#!/usr/bin/env python3
"""
Quick Security Demo
Demonstrates the key security features of the SKZ security framework
"""

import os
import sys
import json
from pathlib import Path

# Add current directory for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from security_audit_system import SecurityAuditSystem
    from security_hardening_manager import SecurityHardeningManager
    from security_monitoring_system import SecurityEvent, SecurityAlertManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def demo_security_audit(project_root):
    """Demonstrate security audit capabilities"""
    print("🔍 Security Audit Demo")
    print("=" * 30)
    
    # Create a small test environment
    test_dir = Path(project_root) / "security_demo_test"
    test_dir.mkdir(exist_ok=True)
    
    # Create test files with vulnerabilities
    test_files = {
        'vulnerable.php': '''<?php
$user_id = $_GET['id'];
$query = "SELECT * FROM users WHERE id = " . $user_id;
mysql_query($query);
echo $_POST['message'];
?>''',
        'insecure.py': '''
import os
API_KEY = "sk-1234567890abcdef1234567890abcdef"
password = "hardcoded_password123"
''',
        'config.inc.php': '''<?php
installed = On
base_url = "http://example.com"
encryption = md5
?>'''
    }
    
    for filename, content in test_files.items():
        (test_dir / filename).write_text(content)
    
    # Run audit on test directory
    audit_system = SecurityAuditSystem(str(test_dir))
    
    print("Scanning for vulnerabilities...")
    vulnerabilities = audit_system.scan_code_vulnerabilities()
    
    print(f"\n📊 Vulnerability Summary:")
    total_vulns = sum(len(vulns) for vulns in vulnerabilities.values())
    print(f"Total vulnerabilities found: {total_vulns}")
    
    for vuln_type, vulns in vulnerabilities.items():
        if vulns:
            print(f"  🔴 {vuln_type}: {len(vulns)} issues")
            for vuln in vulns[:2]:  # Show first 2
                print(f"    - {vuln['file']} (line {vuln['line']}): {vuln['severity']}")
    
    # Check configuration security
    print(f"\n⚙️  Configuration Issues:")
    config_issues = audit_system.check_configuration_security()
    if config_issues:
        for category, issues in config_issues.items():
            print(f"  🟡 {category}: {len(issues)} issues")
    else:
        print("  ✅ No configuration issues found")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    
    return total_vulns


def demo_security_hardening(project_root):
    """Demonstrate security hardening capabilities"""
    print("\n🛡️  Security Hardening Demo")
    print("=" * 35)
    
    hardening_manager = SecurityHardeningManager(project_root)
    
    print("Generating security configurations...")
    
    # Generate PHP security config
    php_config = hardening_manager._generate_php_ini_security_config({
        'display_errors': 'Off',
        'expose_php': 'Off',
        'session.cookie_httponly': 'On'
    })
    print(f"✅ PHP security configuration generated ({len(php_config)} chars)")
    
    # Generate Apache security config
    apache_results = hardening_manager._apply_apache_hardening()
    print(f"✅ Apache security configuration: {apache_results['applied_headers']} headers, {apache_results['applied_rules']} rules")
    
    # Generate SKZ agents security config
    skz_results = hardening_manager.apply_skz_agents_security_hardening()
    print(f"✅ SKZ agents security configuration: {skz_results['applied_categories']} categories")
    
    return True


def demo_security_monitoring():
    """Demonstrate security monitoring capabilities"""
    print("\n👁️  Security Monitoring Demo")
    print("=" * 35)
    
    # Create security events
    events = [
        SecurityEvent('sql_injection_attempt', 'critical', 'web_app', 
                     'SQL injection detected in login form'),
        SecurityEvent('authentication_failure', 'medium', 'auth_system', 
                     'Multiple failed login attempts'),
        SecurityEvent('suspicious_user_agent', 'low', 'web_server', 
                     'Suspicious scanning tool detected')
    ]
    
    print("Sample security events:")
    for event in events:
        severity_emoji = "🔴" if event.severity == 'critical' else "🟡" if event.severity == 'medium' else "🔵"
        print(f"  {severity_emoji} {event.event_type}: {event.description}")
    
    # Test alert manager
    alert_config = {
        'email': {'enabled': False},
        'webhook': {'enabled': False},
        'log': {'enabled': True, 'file': 'demo_alerts.log'},
        'console': {'enabled': True}
    }
    
    alert_manager = SecurityAlertManager(alert_config)
    print(f"\n✅ Alert manager configured with {len(alert_manager.alert_handlers)} handlers")
    
    # Test log patterns
    test_log = "192.168.1.1 - - [10/Oct/2023:13:55:36] \"GET /index.php?id=1' OR 1=1-- HTTP/1.1\" 200"
    print(f"\n🔍 Testing log pattern detection:")
    print(f"Sample log: {test_log[:50]}...")
    
    # Import the monitoring module and test pattern detection
    from security_monitoring_system import SecurityMonitor
    
    # Create a minimal config to avoid the setup issues
    monitor_config = {
        'log_monitoring': {'enabled': False},
        'file_integrity': {'enabled': False},
        'process_monitoring': {'enabled': False},
        'api_monitoring': {'enabled': False}
    }
    
    monitor = SecurityMonitor('.', config_file=None)
    monitor.config.update(monitor_config)
    
    # Test pattern detection
    sql_pattern = monitor.security_patterns.get('sql_injection_attempt')
    sql_pattern_found = sql_pattern.search(test_log) if sql_pattern else False
    print(f"SQL injection pattern detected: {'✅ Yes' if sql_pattern_found else '❌ No'}")
    
    return True


def main():
    """Run security demos"""
    print("🔒 SKZ Security Framework Demo")
    print("=" * 50)
    print("Demonstrating comprehensive security capabilities for OJS with SKZ integration\n")
    
    project_root = "."
    
    # Demo 1: Security Audit
    vuln_count = demo_security_audit(project_root)
    
    # Demo 2: Security Hardening  
    hardening_success = demo_security_hardening(project_root)
    
    # Demo 3: Security Monitoring
    monitoring_success = demo_security_monitoring()
    
    # Summary
    print("\n🎯 Security Framework Summary")
    print("=" * 40)
    print("✅ Security Audit System: Comprehensive vulnerability scanning")
    print("✅ Security Hardening Manager: Automated security configuration")
    print("✅ Security Monitoring System: Real-time threat detection")
    print("✅ Security Integration Manager: Orchestrated security operations")
    print("✅ Comprehensive Test Suite: Validation and verification")
    
    print(f"\n📊 Demo Results:")
    print(f"  - Vulnerabilities detected: {vuln_count}")
    print(f"  - Hardening configurations: {'✅ Generated' if hardening_success else '❌ Failed'}")
    print(f"  - Monitoring capabilities: {'✅ Demonstrated' if monitoring_success else '❌ Failed'}")
    
    print(f"\n🚀 Next Steps:")
    print("  1. Run full security audit: python3 skz-integration/security_integration_manager.py --audit")
    print("  2. Apply security hardening: python3 skz-integration/security_integration_manager.py --harden")
    print("  3. Start security monitoring: python3 skz-integration/security_integration_manager.py --monitor")
    print("  4. Generate security report: python3 skz-integration/security_integration_manager.py --report")
    
    print(f"\n📚 Documentation:")
    print("  - Security Policy: SECURITY.md")
    print("  - Security Architecture: Comprehensive multi-layer framework")
    print("  - Compliance Support: OWASP Top 10, ISO 27001, GDPR")
    print("  - Incident Response: Automated alerting and response procedures")


if __name__ == "__main__":
    main()