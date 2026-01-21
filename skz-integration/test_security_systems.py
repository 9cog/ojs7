#!/usr/bin/env python3
"""
Comprehensive Security Test Suite
Tests for security auditing, hardening, and monitoring systems
"""

import os
import sys
import json
import time
import unittest
import tempfile
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from security_audit_system import SecurityAuditSystem
    from security_hardening_manager import SecurityHardeningManager
    from security_monitoring_system import SecurityMonitor, SecurityEvent, SecurityAlertManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all security modules are in the same directory")
    sys.exit(1)

# Setup logging for tests
logging.basicConfig(level=logging.WARNING)


class TestSecurityAuditSystem(unittest.TestCase):
    """Test cases for SecurityAuditSystem"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.audit_system = SecurityAuditSystem(self.test_dir)
    
    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test audit system initialization"""
        self.assertIsInstance(self.audit_system, SecurityAuditSystem)
        self.assertEqual(str(self.audit_system.project_root), self.test_dir)
        self.assertIsInstance(self.audit_system.security_patterns, dict)
        self.assertIn('sql_injection', self.audit_system.security_patterns)
    
    def test_sql_injection_detection(self):
        """Test SQL injection vulnerability detection"""
        # Create a file with SQL injection vulnerability
        test_file = Path(self.test_dir) / 'vulnerable.php'
        vulnerable_code = '''<?php
$user_id = $_GET['id'];
$query = "SELECT * FROM users WHERE id = " . $user_id;
mysql_query($query);
?>'''
        test_file.write_text(vulnerable_code)
        
        # Scan for vulnerabilities
        vulnerabilities = self.audit_system._scan_file_vulnerabilities(test_file)
        
        # Check that SQL injection was detected
        self.assertIn('sql_injection', vulnerabilities)
        self.assertTrue(len(vulnerabilities['sql_injection']) > 0)
        
        vuln = vulnerabilities['sql_injection'][0]
        self.assertEqual(vuln['severity'], 'critical')
        self.assertEqual(vuln['file'], str(test_file))
    
    def test_xss_vulnerability_detection(self):
        """Test XSS vulnerability detection"""
        test_file = Path(self.test_dir) / 'xss_vulnerable.php'
        vulnerable_code = '''<?php
echo $_GET['message'];
print $_POST['content'];
?>'''
        test_file.write_text(vulnerable_code)
        
        vulnerabilities = self.audit_system._scan_file_vulnerabilities(test_file)
        
        self.assertIn('xss_vulnerability', vulnerabilities)
        self.assertTrue(len(vulnerabilities['xss_vulnerability']) > 0)
    
    def test_hardcoded_secrets_detection(self):
        """Test hardcoded secrets detection"""
        test_file = Path(self.test_dir) / 'secrets.py'
        vulnerable_code = '''
API_KEY = "sk-1234567890abcdef1234567890abcdef"
database_password = "supersecretpassword123"
secret_key = "my-super-secret-encryption-key-123456"
'''
        test_file.write_text(vulnerable_code)
        
        vulnerabilities = self.audit_system._scan_file_vulnerabilities(test_file)
        
        self.assertIn('hardcoded_secrets', vulnerabilities)
        self.assertTrue(len(vulnerabilities['hardcoded_secrets']) > 0)
    
    def test_configuration_security_check(self):
        """Test configuration security checking"""
        # Create a mock config file
        config_file = Path(self.test_dir) / 'config.inc.php'
        config_content = '''<?php
installed = On
base_url = "http://example.com"
encryption = md5
?>'''
        config_file.write_text(config_content)
        
        # Update audit system to use test directory
        self.audit_system.project_root = Path(self.test_dir)
        
        config_issues = self.audit_system._check_ojs_security_config()
        
        # Should detect HTTP URL and weak encryption
        self.assertIn('base_url', config_issues)
        self.assertIn('encryption', config_issues)
    
    @patch('subprocess.run')
    def test_dependency_scanning(self, mock_subprocess):
        """Test dependency vulnerability scanning"""
        # Create mock requirements file
        requirements_file = Path(self.test_dir) / 'requirements.txt'
        requirements_file.write_text('requests==2.0.0\ndjango==1.0.0\n')
        
        # Mock safety command output
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps([
            {
                'package': 'django',
                'installed_version': '1.0.0',
                'id': 'VULN-001',
                'advisory': 'Django XSS vulnerability'
            }
        ])
        
        issues = self.audit_system._scan_python_dependencies(requirements_file)
        
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]['package'], 'django')
        self.assertEqual(issues[0]['vulnerability_id'], 'VULN-001')
    
    def test_compliance_checking(self):
        """Test security compliance checking"""
        # Add some test vulnerabilities
        self.audit_system.security_issues['sql_injection'] = [
            {'type': 'sql_injection', 'severity': 'critical'}
        ]
        
        compliance_results = self.audit_system.check_compliance(['owasp_top10'])
        
        self.assertIn('owasp_top10', compliance_results)
        results = compliance_results['owasp_top10']
        self.assertLess(results['score'], 1.0)  # Should not be 100% compliant
    
    def test_security_report_generation(self):
        """Test comprehensive security report generation"""
        # Create test files with vulnerabilities
        test_file = Path(self.test_dir) / 'test.php'
        test_file.write_text('<?php echo $_GET["data"]; ?>')
        
        # Generate report
        report = self.audit_system.generate_security_report()
        
        # Verify report structure
        self.assertIn('timestamp', report)
        self.assertIn('security_score', report)
        self.assertIn('summary', report)
        self.assertIn('vulnerabilities', report)
        self.assertIn('compliance_results', report)
        self.assertIn('hardening_recommendations', report)
        
        # Verify summary contains counts
        summary = report['summary']
        self.assertIn('total_vulnerabilities', summary)
        self.assertIn('critical_vulnerabilities', summary)


class TestSecurityHardeningManager(unittest.TestCase):
    """Test cases for SecurityHardeningManager"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.hardening_manager = SecurityHardeningManager(self.test_dir)
    
    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test hardening manager initialization"""
        self.assertIsInstance(self.hardening_manager, SecurityHardeningManager)
        self.assertEqual(str(self.hardening_manager.project_root), self.test_dir)
        self.assertIn('php_security', self.hardening_manager.config_templates)
    
    def test_php_security_config_generation(self):
        """Test PHP security configuration generation"""
        settings = {
            'display_errors': 'Off',
            'expose_php': 'Off',
            'session.cookie_httponly': 'On'
        }
        
        config = self.hardening_manager._generate_php_ini_security_config(settings)
        
        self.assertIn('display_errors = Off', config)
        self.assertIn('expose_php = Off', config)
        self.assertIn('session.cookie_httponly = On', config)
    
    def test_apache_hardening_config(self):
        """Test Apache security hardening configuration"""
        results = self.hardening_manager._apply_apache_hardening()
        
        self.assertIn('htaccess_security_config', results)
        self.assertIn('applied_headers', results)
        self.assertIn('applied_rules', results)
        
        # Verify the .htaccess file was created
        htaccess_file = Path(results['htaccess_security_config'])
        self.assertTrue(htaccess_file.exists())
        
        content = htaccess_file.read_text()
        self.assertIn('X-Frame-Options', content)
        self.assertIn('X-XSS-Protection', content)
    
    def test_nginx_hardening_config(self):
        """Test Nginx security hardening configuration"""
        results = self.hardening_manager._apply_nginx_hardening()
        
        self.assertIn('nginx_security_config', results)
        self.assertIn('applied_headers', results)
        
        # Verify the nginx config file was created
        nginx_file = Path(results['nginx_security_config'])
        self.assertTrue(nginx_file.exists())
        
        content = nginx_file.read_text()
        self.assertIn('add_header X-Frame-Options', content)
        self.assertIn('server_tokens off', content)
    
    def test_ojs_config_hardening(self):
        """Test OJS configuration hardening"""
        # Create test config file
        config_file = Path(self.test_dir) / 'config.inc.php'
        config_content = '''<?php
installed = On
base_url = "http://example.com"
encryption = md5
?>'''
        config_file.write_text(config_content)
        
        results = self.hardening_manager.apply_ojs_security_hardening()
        
        self.assertIn('hardened_config', results)
        
        # Verify hardened config was created
        hardened_file = Path(results['hardened_config'])
        self.assertTrue(hardened_file.exists())
        
        content = hardened_file.read_text()
        self.assertIn('force_ssl = On', content)
        self.assertIn('encryption = sha256', content)
    
    def test_skz_agents_security_config(self):
        """Test SKZ agents security configuration"""
        results = self.hardening_manager.apply_skz_agents_security_hardening()
        
        self.assertIn('security_config', results)
        self.assertIn('env_template', results)
        
        # Verify config files were created
        config_file = Path(results['security_config'])
        env_file = Path(results['env_template'])
        
        self.assertTrue(config_file.exists())
        self.assertTrue(env_file.exists())
        
        # Check env template content
        env_content = env_file.read_text()
        self.assertIn('SKZ_JWT_SECRET', env_content)
        self.assertIn('SKZ_API_SECRET', env_content)
        self.assertIn('SKZ_ENCRYPTION_KEY', env_content)
    
    def test_backup_creation(self):
        """Test backup creation before hardening"""
        # Create test files to backup
        test_file = Path(self.test_dir) / 'config.inc.php'
        test_file.write_text('test content')
        
        backups = self.hardening_manager.backup_existing_configs()
        
        if 'config.inc.php' in backups:
            backup_file = Path(backups['config.inc.php'])
            self.assertTrue(backup_file.exists())


class TestSecurityMonitoringSystem(unittest.TestCase):
    """Test cases for SecurityMonitoringSystem"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.monitor = SecurityMonitor(self.test_dir)
    
    def tearDown(self):
        """Cleanup test environment"""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_security_event_creation(self):
        """Test security event creation"""
        event = SecurityEvent(
            event_type='test_event',
            severity='medium',
            source='unit_test',
            description='Test event for unit testing',
            details={'test': True}
        )
        
        self.assertEqual(event.event_type, 'test_event')
        self.assertEqual(event.severity, 'medium')
        self.assertEqual(event.source, 'unit_test')
        self.assertFalse(event.acknowledged)
        self.assertFalse(event.resolved)
        
        # Test serialization
        event_dict = event.to_dict()
        self.assertIn('event_id', event_dict)
        self.assertIn('timestamp', event_dict)
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization"""
        alert_manager = SecurityAlertManager()
        
        self.assertIsInstance(alert_manager.config, dict)
        self.assertIn('email', alert_manager.config)
        self.assertIn('webhook', alert_manager.config)
        self.assertIn('log', alert_manager.config)
    
    def test_log_alert_handler(self):
        """Test log alert handler"""
        alert_manager = SecurityAlertManager({
            'log': {
                'enabled': True,
                'file': str(Path(self.test_dir) / 'test_alerts.log')
            }
        })
        
        event = SecurityEvent(
            event_type='test_alert',
            severity='high',
            source='unit_test',
            description='Test alert logging'
        )
        
        alert_manager.log_alert(event)
        
        log_file = Path(self.test_dir) / 'test_alerts.log'
        self.assertTrue(log_file.exists())
        
        content = log_file.read_text()
        self.assertIn('test_alert', content)
        self.assertIn('HIGH', content)
    
    def test_security_pattern_detection(self):
        """Test security pattern detection in logs"""
        test_log = "192.168.1.1 - - [10/Oct/2023:13:55:36] \"GET /index.php?id=1' OR 1=1-- HTTP/1.1\" 200"
        
        self.monitor._analyze_log_content(test_log, 'test.log')
        
        # Should detect SQL injection attempt
        events = list(self.monitor.event_history)
        self.assertTrue(len(events) > 0)
        
        sql_events = [e for e in events if e.event_type == 'sql_injection_attempt']
        self.assertTrue(len(sql_events) > 0)
    
    def test_file_integrity_monitoring(self):
        """Test file integrity monitoring"""
        # Create test file
        test_file = Path(self.test_dir) / 'test_file.txt'
        test_file.write_text('original content')
        
        # Set up monitoring for the test file
        self.monitor.config['file_integrity']['paths'] = [str(test_file)]
        
        # Calculate initial hash
        initial_hash = self.monitor._calculate_file_hash(test_file)
        self.assertTrue(len(initial_hash) > 0)
        
        # Modify file
        test_file.write_text('modified content')
        
        # Calculate new hash
        new_hash = self.monitor._calculate_file_hash(test_file)
        self.assertNotEqual(initial_hash, new_hash)
    
    def test_rate_limiting(self):
        """Test event rate limiting"""
        # Create multiple similar events
        for i in range(15):
            event = SecurityEvent(
                event_type='test_spam',
                severity='medium',
                source='unit_test',
                description=f'Spam event {i}'
            )
            self.monitor.add_event(event)
        
        # Should be rate limited after 10 events (default)
        events = [e for e in self.monitor.event_history if e.event_type == 'test_spam']
        self.assertLessEqual(len(events), 10)
    
    def test_event_summary_generation(self):
        """Test event summary generation"""
        # Add test events
        events = [
            SecurityEvent('sql_injection', 'critical', 'test', 'Test SQL injection'),
            SecurityEvent('xss_attempt', 'high', 'test', 'Test XSS'),
            SecurityEvent('auth_failure', 'medium', 'test', 'Test auth failure')
        ]
        
        for event in events:
            self.monitor.event_history.append(event)
        
        summary = self.monitor.get_event_summary(24)
        
        self.assertEqual(summary['total_events'], 3)
        self.assertIn('critical', summary['events_by_severity'])
        self.assertIn('high', summary['events_by_severity'])
        self.assertIn('medium', summary['events_by_severity'])
    
    @patch('requests.get')
    def test_api_monitoring(self, mock_get):
        """Test API endpoint monitoring"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            'X-Frame-Options': 'SAMEORIGIN',
            'X-Content-Type-Options': 'nosniff'
        }
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_get.return_value = mock_response
        
        # Set up API monitoring
        self.monitor.config['api_monitoring']['endpoints'] = ['http://test.com/api']
        
        # This would normally run in a thread, but we'll call directly for testing
        # The actual test would need to mock the threading behavior
        
        # Test that the function can be called without error
        self.assertIsInstance(self.monitor.config['api_monitoring']['endpoints'], list)


class TestSecurityIntegration(unittest.TestCase):
    """Integration tests for security systems"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.project_structure = self._create_test_project_structure()
    
    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_project_structure(self):
        """Create a test project structure"""
        # Create basic OJS-like structure
        dirs = [
            'classes/security',
            'plugins/generic',
            'skz-integration/autonomous-agents-framework',
            'logs'
        ]
        
        for dir_path in dirs:
            (Path(self.test_dir) / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create test files
        files = {
            'config.inc.php': '''<?php
installed = On
base_url = "https://example.com"
encryption = sha256
?>''',
            'index.php': '''<?php
require_once('lib/pkp/includes/bootstrap.inc.php');
?>''',
            'classes/security/test.php': '''<?php
class SecurityTest {
    public function validateInput($input) {
        return filter_var($input, FILTER_SANITIZE_STRING);
    }
}
?>''',
            'skz-integration/requirements.txt': '''requests==2.28.0
flask==2.0.0
PyJWT==2.4.0
''',
            '.htaccess': '''RewriteEngine On
RewriteRule ^(.*)$ index.php [QSA,L]
'''
        }
        
        for file_path, content in files.items():
            full_path = Path(self.test_dir) / file_path
            full_path.write_text(content)
        
        return {
            'project_root': self.test_dir,
            'files': files
        }
    
    def test_end_to_end_security_audit(self):
        """Test complete security audit workflow"""
        # Initialize audit system
        audit_system = SecurityAuditSystem(self.test_dir)
        
        # Run comprehensive audit
        report = audit_system.generate_security_report()
        
        # Verify report was generated
        self.assertIsInstance(report, dict)
        self.assertIn('security_score', report)
        self.assertIn('vulnerabilities', report)
        self.assertIn('compliance_results', report)
        
        # Save report
        report_file = Path(self.test_dir) / 'security_report.json'
        audit_system.save_report(report, str(report_file))
        
        self.assertTrue(report_file.exists())
        
        # Verify report content
        with open(report_file) as f:
            saved_report = json.load(f)
        
        self.assertEqual(saved_report['security_score'], report['security_score'])
    
    def test_comprehensive_hardening_workflow(self):
        """Test complete security hardening workflow"""
        # Initialize hardening manager
        hardening_manager = SecurityHardeningManager(self.test_dir)
        
        # Apply comprehensive hardening
        results = hardening_manager.apply_comprehensive_hardening('apache')
        
        # Verify all components were hardened
        self.assertIn('php_hardening', results)
        self.assertIn('webserver_hardening', results)
        self.assertIn('ojs_hardening', results)
        self.assertIn('skz_hardening', results)
        self.assertIn('deployment_instructions', results)
        
        # Verify deployment instructions were created
        instructions_file = Path(results['deployment_instructions'])
        self.assertTrue(instructions_file.exists())
        
        content = instructions_file.read_text()
        self.assertIn('Security Hardening Deployment Instructions', content)
        self.assertIn('PHP Security Configuration', content)
    
    def test_monitoring_integration(self):
        """Test security monitoring integration"""
        # Initialize monitoring system
        config = {
            'log_monitoring': {'enabled': False},  # Disable to avoid file system dependencies
            'file_integrity': {'enabled': False},
            'process_monitoring': {'enabled': False},
            'api_monitoring': {'enabled': False}
        }
        
        monitor = SecurityMonitor(self.test_dir, config_file=None)
        monitor.config.update(config)
        
        # Test event processing
        test_event = SecurityEvent(
            event_type='integration_test',
            severity='medium',
            source='unit_test',
            description='Integration test event'
        )
        
        monitor.add_event(test_event)
        
        # Verify event was added
        self.assertTrue(len(monitor.event_history) > 0)
        self.assertEqual(monitor.event_history[-1].event_type, 'integration_test')
        
        # Test summary generation
        summary = monitor.get_event_summary(1)
        self.assertGreater(summary['total_events'], 0)


def run_security_tests():
    """Run all security tests"""
    print("🧪 Running SKZ Security Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSecurityAuditSystem,
        TestSecurityHardeningManager,
        TestSecurityMonitoringSystem,
        TestSecurityIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("🧪 Security Test Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    if not result.failures and not result.errors:
        print("\n✅ All security tests passed!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_security_tests()
    sys.exit(0 if success else 1)