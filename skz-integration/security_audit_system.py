#!/usr/bin/env python3
"""
Comprehensive Security Audit and Hardening System
Advanced security scanning, vulnerability assessment, and hardening for SKZ-enhanced OJS
"""

import os
import re
import json
import hashlib
import hmac
import subprocess
import ssl
import socket
import requests
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityAuditSystem:
    """
    Comprehensive security audit system for OJS with SKZ integration
    """
    
    def __init__(self, project_root: str = ".", config_file: str = None):
        self.project_root = Path(project_root)
        self.config = self._load_config(config_file)
        self.vulnerabilities = []
        self.security_issues = defaultdict(list)
        self.hardening_recommendations = []
        self.compliance_status = {}
        
        # Security scan patterns
        self.security_patterns = {
            'sql_injection': [
                re.compile(r'(?i)\b(SELECT|INSERT|UPDATE|DELETE)\b.*\$_(?:GET|POST|REQUEST|COOKIE)', re.IGNORECASE),
                re.compile(r'(?i)mysql_query\s*\(\s*["\'].*\$', re.IGNORECASE),
                re.compile(r'(?i)mysqli_query\s*\(\s*\$\w+\s*,\s*["\'].*\$', re.IGNORECASE),
                re.compile(r'(?i)query.*\$_(?:GET|POST|REQUEST)', re.IGNORECASE),
            ],
            'xss_vulnerability': [
                re.compile(r'echo\s+\$_(?:GET|POST|REQUEST)', re.IGNORECASE),
                re.compile(r'print\s+\$_(?:GET|POST|REQUEST)', re.IGNORECASE),
                re.compile(r'innerHTML\s*=\s*.*\$_(?:GET|POST)', re.IGNORECASE),
            ],
            'csrf_vulnerability': [
                re.compile(r'<form[^>]*method\s*=\s*["\']post["\'][^>]*>', re.IGNORECASE),
                re.compile(r'\$_POST\[.*\].*without.*token', re.IGNORECASE),
            ],
            'insecure_random': [
                re.compile(r'\brand\s*\(\s*\)', re.IGNORECASE),
                re.compile(r'\bmt_rand\s*\(\s*\)', re.IGNORECASE),
                re.compile(r'Math\.random\s*\(\s*\)', re.IGNORECASE),
            ],
            'hardcoded_secrets': [
                re.compile(r'(?i)(password|pwd|secret|key|token)\s*[=:]\s*["\'][^"\']{8,}["\']'),
                re.compile(r'(?i)(api[_-]?key|auth[_-]?token)\s*[=:]\s*["\'][^"\']{16,}["\']'),
                re.compile(r'(?i)(secret[_-]?key|private[_-]?key)\s*[=:]\s*["\'][^"\']{20,}["\']'),
            ],
            'directory_traversal': [
                re.compile(r'(?:include|require)(?:_once)?\s*\(\s*\$_(?:GET|POST|REQUEST)', re.IGNORECASE),
                re.compile(r'file_get_contents\s*\(\s*\$_(?:GET|POST|REQUEST)', re.IGNORECASE),
                re.compile(r'fopen\s*\(\s*\$_(?:GET|POST|REQUEST)', re.IGNORECASE),
            ],
            'weak_encryption': [
                re.compile(r'\bmd5\s*\(', re.IGNORECASE),
                re.compile(r'\bsha1\s*\(', re.IGNORECASE),
                re.compile(r'mcrypt_encrypt\s*\(\s*MCRYPT_DES', re.IGNORECASE),
            ],
            'insecure_communication': [
                re.compile(r'curl_setopt\s*\([^,]+,\s*CURLOPT_SSL_VERIFYPEER\s*,\s*false\s*\)', re.IGNORECASE),
                re.compile(r'stream_context_create\s*\([^)]*verify_peer[^)]*false', re.IGNORECASE),
            ]
        }
        
        # Security configuration checks
        self.config_checks = {
            'php_security': {
                'display_errors': 'Off',
                'expose_php': 'Off',
                'allow_url_include': 'Off',
                'allow_url_fopen': 'Off',
                'session.cookie_httponly': 'On',
                'session.cookie_secure': 'On',
                'session.use_strict_mode': 'On'
            },
            'ojs_security': {
                'disable_path_info': 'On',
                'encryption': 'sha1',  # Should be upgraded
                'session_check_ip': 'On',
                'restful_urls': 'On'
            }
        }
        
        # Compliance frameworks
        self.compliance_frameworks = {
            'owasp_top10': {
                'A01_injection': ['sql_injection', 'directory_traversal'],
                'A02_broken_authentication': ['weak_encryption', 'insecure_random'],
                'A03_sensitive_data_exposure': ['hardcoded_secrets', 'insecure_communication'],
                'A04_xxe': ['xml_external_entity'],
                'A05_broken_access_control': ['csrf_vulnerability'],
                'A06_security_misconfiguration': ['php_security', 'ojs_security'],
                'A07_xss': ['xss_vulnerability'],
                'A08_insecure_deserialization': ['insecure_deserialization'],
                'A09_known_vulnerabilities': ['dependency_vulnerabilities'],
                'A10_insufficient_logging': ['logging_security']
            }
        }
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load security audit configuration"""
        default_config = {
            'scan_extensions': ['.php', '.js', '.py', '.yaml', '.yml', '.json', '.xml'],
            'exclude_dirs': ['.git', 'node_modules', '__pycache__', 'vendor', '.env'],
            'severity_levels': ['critical', 'high', 'medium', 'low', 'info'],
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'enable_network_scan': False,
            'enable_dependency_scan': True,
            'compliance_frameworks': ['owasp_top10']
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
        
        return default_config
    
    def scan_code_vulnerabilities(self) -> Dict[str, List[Dict[str, Any]]]:
        """Scan codebase for security vulnerabilities"""
        logger.info("Starting code vulnerability scan...")
        
        vulnerabilities = defaultdict(list)
        scanned_files = 0
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.config['exclude_dirs']]
            
            for file in files:
                file_path = Path(root) / file
                
                # Check file extension
                if file_path.suffix not in self.config['scan_extensions']:
                    continue
                
                # Check file size
                try:
                    if file_path.stat().st_size > self.config['max_file_size']:
                        continue
                except OSError:
                    continue
                
                # Scan file for vulnerabilities
                file_vulnerabilities = self._scan_file_vulnerabilities(file_path)
                for vuln_type, vulns in file_vulnerabilities.items():
                    vulnerabilities[vuln_type].extend(vulns)
                
                scanned_files += 1
        
        logger.info(f"Scanned {scanned_files} files for vulnerabilities")
        return dict(vulnerabilities)
    
    def _scan_file_vulnerabilities(self, file_path: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Scan individual file for security vulnerabilities"""
        vulnerabilities = defaultdict(list)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                # Check each vulnerability pattern
                for vuln_type, patterns in self.security_patterns.items():
                    for pattern in patterns:
                        for i, line in enumerate(lines, 1):
                            matches = pattern.findall(line)
                            if matches:
                                vulnerability = {
                                    'type': vuln_type,
                                    'file': str(file_path),
                                    'line': i,
                                    'code': line.strip(),
                                    'severity': self._get_vulnerability_severity(vuln_type),
                                    'description': self._get_vulnerability_description(vuln_type),
                                    'remediation': self._get_vulnerability_remediation(vuln_type)
                                }
                                vulnerabilities[vuln_type].append(vulnerability)
        
        except Exception as e:
            logger.warning(f"Failed to scan file {file_path}: {e}")
        
        return dict(vulnerabilities)
    
    def _get_vulnerability_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type"""
        severity_map = {
            'sql_injection': 'critical',
            'xss_vulnerability': 'high',
            'csrf_vulnerability': 'high',
            'directory_traversal': 'critical',
            'hardcoded_secrets': 'high',
            'weak_encryption': 'medium',
            'insecure_random': 'medium',
            'insecure_communication': 'high'
        }
        return severity_map.get(vuln_type, 'medium')
    
    def _get_vulnerability_description(self, vuln_type: str) -> str:
        """Get description for vulnerability type"""
        descriptions = {
            'sql_injection': 'Potential SQL injection vulnerability detected',
            'xss_vulnerability': 'Cross-site scripting (XSS) vulnerability detected',
            'csrf_vulnerability': 'Cross-site request forgery (CSRF) protection missing',
            'directory_traversal': 'Directory traversal vulnerability detected',
            'hardcoded_secrets': 'Hardcoded secrets or credentials detected',
            'weak_encryption': 'Weak encryption algorithm detected',
            'insecure_random': 'Insecure random number generation detected',
            'insecure_communication': 'Insecure communication configuration detected'
        }
        return descriptions.get(vuln_type, f'Security issue of type {vuln_type} detected')
    
    def _get_vulnerability_remediation(self, vuln_type: str) -> str:
        """Get remediation advice for vulnerability type"""
        remediations = {
            'sql_injection': 'Use prepared statements or parameterized queries',
            'xss_vulnerability': 'Sanitize and escape user input before output',
            'csrf_vulnerability': 'Implement CSRF tokens for all state-changing operations',
            'directory_traversal': 'Validate and sanitize file paths, use whitelists',
            'hardcoded_secrets': 'Use environment variables or secure key management',
            'weak_encryption': 'Use strong encryption algorithms (SHA-256, bcrypt)',
            'insecure_random': 'Use cryptographically secure random functions',
            'insecure_communication': 'Enable SSL/TLS verification and use HTTPS'
        }
        return remediations.get(vuln_type, 'Review and fix security issue')
    
    def check_configuration_security(self) -> Dict[str, Dict[str, Any]]:
        """Check security configuration settings"""
        logger.info("Checking configuration security...")
        
        config_issues = {}
        
        # Check PHP configuration
        php_config = self._check_php_security_config()
        if php_config:
            config_issues['php_security'] = php_config
        
        # Check OJS configuration
        ojs_config = self._check_ojs_security_config()
        if ojs_config:
            config_issues['ojs_security'] = ojs_config
        
        # Check web server configuration
        webserver_config = self._check_webserver_security_config()
        if webserver_config:
            config_issues['webserver_security'] = webserver_config
        
        return config_issues
    
    def _check_php_security_config(self) -> Dict[str, Any]:
        """Check PHP security configuration"""
        issues = {}
        
        try:
            # Try to get PHP configuration
            result = subprocess.run(['php', '-i'], capture_output=True, text=True)
            if result.returncode == 0:
                php_info = result.stdout
                
                for setting, expected in self.config_checks['php_security'].items():
                    pattern = re.compile(rf'{setting}\s*=>\s*([^\n]+)', re.IGNORECASE)
                    match = pattern.search(php_info)
                    
                    if match:
                        actual = match.group(1).strip()
                        if actual.lower() != expected.lower():
                            issues[setting] = {
                                'expected': expected,
                                'actual': actual,
                                'severity': 'medium',
                                'description': f'PHP setting {setting} should be {expected}'
                            }
        
        except Exception as e:
            logger.warning(f"Failed to check PHP configuration: {e}")
        
        return issues
    
    def _check_ojs_security_config(self) -> Dict[str, Any]:
        """Check OJS security configuration"""
        issues = {}
        
        config_file = self.project_root / 'config.inc.php'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # Check for security-related settings
                security_settings = {
                    'installed': r'installed\s*=\s*([^\n]+)',
                    'base_url': r'base_url\s*=\s*"([^"]*)"',
                    'encryption': r'encryption\s*=\s*([^\n]+)',
                    'session_check_ip': r'session_check_ip\s*=\s*([^\n]+)'
                }
                
                for setting, pattern in security_settings.items():
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        
                        # Check for specific security issues
                        if setting == 'base_url' and value.startswith('http://'):
                            issues[setting] = {
                                'value': value,
                                'severity': 'high',
                                'description': 'Base URL should use HTTPS in production'
                            }
                        elif setting == 'encryption' and value in ['md5', 'sha1']:
                            issues[setting] = {
                                'value': value,
                                'severity': 'medium',
                                'description': 'Weak encryption algorithm detected'
                            }
            
            except Exception as e:
                logger.warning(f"Failed to check OJS configuration: {e}")
        
        return issues
    
    def _check_webserver_security_config(self) -> Dict[str, Any]:
        """Check web server security configuration"""
        issues = {}
        
        # Check for common security headers
        try:
            # This would typically check .htaccess or server configuration
            htaccess_file = self.project_root / '.htaccess'
            if htaccess_file.exists():
                with open(htaccess_file, 'r') as f:
                    content = f.read()
                
                security_headers = [
                    'X-Frame-Options',
                    'X-Content-Type-Options',
                    'X-XSS-Protection',
                    'Strict-Transport-Security',
                    'Content-Security-Policy'
                ]
                
                missing_headers = []
                for header in security_headers:
                    if header not in content:
                        missing_headers.append(header)
                
                if missing_headers:
                    issues['missing_security_headers'] = {
                        'headers': missing_headers,
                        'severity': 'medium',
                        'description': 'Missing security headers in web server configuration'
                    }
        
        except Exception as e:
            logger.warning(f"Failed to check web server configuration: {e}")
        
        return issues
    
    def scan_dependencies(self) -> Dict[str, List[Dict[str, Any]]]:
        """Scan dependencies for known vulnerabilities"""
        logger.info("Scanning dependencies for vulnerabilities...")
        
        dependency_issues = defaultdict(list)
        
        # Scan Python dependencies
        requirements_files = list(self.project_root.rglob('requirements*.txt'))
        for req_file in requirements_files:
            python_issues = self._scan_python_dependencies(req_file)
            dependency_issues['python'].extend(python_issues)
        
        # Scan PHP dependencies
        composer_files = list(self.project_root.rglob('composer.json'))
        for composer_file in composer_files:
            php_issues = self._scan_php_dependencies(composer_file)
            dependency_issues['php'].extend(php_issues)
        
        # Scan JavaScript dependencies
        package_files = list(self.project_root.rglob('package.json'))
        for package_file in package_files:
            js_issues = self._scan_js_dependencies(package_file)
            dependency_issues['javascript'].extend(js_issues)
        
        return dict(dependency_issues)
    
    def _scan_python_dependencies(self, requirements_file: Path) -> List[Dict[str, Any]]:
        """Scan Python dependencies for vulnerabilities"""
        issues = []
        
        try:
            # Use safety to check for known vulnerabilities
            result = subprocess.run(
                ['safety', 'check', '-r', str(requirements_file), '--json'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                vulnerabilities = json.loads(result.stdout)
                for vuln in vulnerabilities:
                    issues.append({
                        'package': vuln.get('package'),
                        'version': vuln.get('installed_version'),
                        'vulnerability_id': vuln.get('id'),
                        'severity': 'high',
                        'description': vuln.get('advisory'),
                        'file': str(requirements_file)
                    })
        
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            # safety command not available or failed
            logger.warning(f"Could not scan Python dependencies in {requirements_file}")
        
        return issues
    
    def _scan_php_dependencies(self, composer_file: Path) -> List[Dict[str, Any]]:
        """Scan PHP dependencies for vulnerabilities"""
        issues = []
        
        try:
            # Use composer audit if available
            composer_dir = composer_file.parent
            result = subprocess.run(
                ['composer', 'audit', '--format=json'],
                cwd=composer_dir, capture_output=True, text=True
            )
            
            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                for package, vulnerabilities in audit_data.get('advisories', {}).items():
                    for vuln in vulnerabilities:
                        issues.append({
                            'package': package,
                            'vulnerability_id': vuln.get('id'),
                            'severity': 'high',
                            'description': vuln.get('title'),
                            'file': str(composer_file)
                        })
        
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Could not scan PHP dependencies in {composer_file}")
        
        return issues
    
    def _scan_js_dependencies(self, package_file: Path) -> List[Dict[str, Any]]:
        """Scan JavaScript dependencies for vulnerabilities"""
        issues = []
        
        try:
            # Use npm audit if available
            package_dir = package_file.parent
            result = subprocess.run(
                ['npm', 'audit', '--json'],
                cwd=package_dir, capture_output=True, text=True
            )
            
            if result.returncode != 0:  # npm audit returns non-zero when vulnerabilities found
                audit_data = json.loads(result.stdout)
                for vuln_id, vuln in audit_data.get('vulnerabilities', {}).items():
                    issues.append({
                        'package': vuln.get('name'),
                        'vulnerability_id': vuln_id,
                        'severity': vuln.get('severity', 'medium'),
                        'description': vuln.get('title'),
                        'file': str(package_file)
                    })
        
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Could not scan JavaScript dependencies in {package_file}")
        
        return issues
    
    def check_compliance(self, frameworks: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Check compliance with security frameworks"""
        logger.info("Checking security compliance...")
        
        if frameworks is None:
            frameworks = self.config['compliance_frameworks']
        
        compliance_results = {}
        
        for framework in frameworks:
            if framework in self.compliance_frameworks:
                framework_results = self._check_framework_compliance(framework)
                compliance_results[framework] = framework_results
        
        return compliance_results
    
    def _check_framework_compliance(self, framework: str) -> Dict[str, Any]:
        """Check compliance with specific framework"""
        framework_config = self.compliance_frameworks[framework]
        compliance_score = 0
        total_checks = len(framework_config)
        results = {}
        
        for check_name, vulnerability_types in framework_config.items():
            # Check if vulnerabilities of these types were found
            found_vulns = []
            for vuln_type in vulnerability_types:
                if vuln_type in self.security_issues:
                    found_vulns.extend(self.security_issues[vuln_type])
            
            if not found_vulns:
                compliance_score += 1
                results[check_name] = {
                    'compliant': True,
                    'vulnerabilities': 0
                }
            else:
                results[check_name] = {
                    'compliant': False,
                    'vulnerabilities': len(found_vulns),
                    'details': found_vulns[:5]  # First 5 vulnerabilities
                }
        
        return {
            'score': compliance_score / total_checks,
            'total_checks': total_checks,
            'passed_checks': compliance_score,
            'results': results
        }
    
    def generate_hardening_recommendations(self) -> List[Dict[str, Any]]:
        """Generate security hardening recommendations"""
        logger.info("Generating security hardening recommendations...")
        
        recommendations = []
        
        # Recommendations based on found vulnerabilities
        for vuln_type, vulns in self.security_issues.items():
            if vulns:
                recommendations.append({
                    'category': 'vulnerability_remediation',
                    'priority': self._get_vulnerability_severity(vuln_type),
                    'title': f'Fix {vuln_type} vulnerabilities',
                    'description': f'Found {len(vulns)} instances of {vuln_type}',
                    'action': self._get_vulnerability_remediation(vuln_type),
                    'affected_files': list(set(v['file'] for v in vulns))
                })
        
        # General hardening recommendations
        general_recommendations = [
            {
                'category': 'authentication',
                'priority': 'high',
                'title': 'Implement multi-factor authentication',
                'description': 'Add MFA for admin accounts',
                'action': 'Configure TOTP or other MFA methods for admin users'
            },
            {
                'category': 'encryption',
                'priority': 'high',
                'title': 'Enforce HTTPS everywhere',
                'description': 'Ensure all communications use SSL/TLS',
                'action': 'Configure HTTPS redirects and HSTS headers'
            },
            {
                'category': 'monitoring',
                'priority': 'medium',
                'title': 'Implement security monitoring',
                'description': 'Add real-time security event monitoring',
                'action': 'Set up log aggregation and security alerting'
            },
            {
                'category': 'backup',
                'priority': 'medium',
                'title': 'Secure backup procedures',
                'description': 'Implement encrypted, regular backups',
                'action': 'Set up automated encrypted backups with offsite storage'
            }
        ]
        
        recommendations.extend(general_recommendations)
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security audit report"""
        logger.info("Generating comprehensive security report...")
        
        # Run all security checks
        code_vulnerabilities = self.scan_code_vulnerabilities()
        self.security_issues.update(code_vulnerabilities)
        
        config_issues = self.check_configuration_security()
        dependency_issues = self.scan_dependencies()
        compliance_results = self.check_compliance()
        hardening_recommendations = self.generate_hardening_recommendations()
        
        # Calculate security score
        total_issues = sum(len(issues) for issues in self.security_issues.values())
        total_config_issues = sum(len(issues) for issues in config_issues.values())
        total_dependency_issues = sum(len(issues) for issues in dependency_issues.values())
        
        security_score = max(0, 100 - (total_issues * 5) - (total_config_issues * 3) - (total_dependency_issues * 2))
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'security_score': security_score,
            'summary': {
                'total_vulnerabilities': total_issues,
                'critical_vulnerabilities': sum(1 for issues in self.security_issues.values() 
                                              for issue in issues if issue.get('severity') == 'critical'),
                'high_vulnerabilities': sum(1 for issues in self.security_issues.values() 
                                          for issue in issues if issue.get('severity') == 'high'),
                'configuration_issues': total_config_issues,
                'dependency_issues': total_dependency_issues
            },
            'vulnerabilities': dict(self.security_issues),
            'configuration_issues': config_issues,
            'dependency_issues': dependency_issues,
            'compliance_results': compliance_results,
            'hardening_recommendations': hardening_recommendations,
            'scan_metadata': {
                'scan_duration': 'calculated_at_runtime',
                'files_scanned': 'calculated_at_runtime',
                'frameworks_checked': list(compliance_results.keys())
            }
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: str = "security_audit_report.json") -> None:
        """Save security audit report to file"""
        output_path = self.project_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Security report saved to {output_path}")
    
    def print_security_summary(self, report: Dict[str, Any]) -> None:
        """Print security audit summary"""
        print("\n" + "="*80)
        print("🔒 SECURITY AUDIT SUMMARY")
        print("="*80)
        
        # Security score
        score = report['security_score']
        score_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        print(f"{score_emoji} Security Score: {score}/100")
        
        # Summary statistics
        summary = report['summary']
        print(f"\n📊 Vulnerability Summary:")
        print(f"   🔴 Critical: {summary['critical_vulnerabilities']}")
        print(f"   🟠 High: {summary['high_vulnerabilities']}")
        print(f"   📝 Total: {summary['total_vulnerabilities']}")
        print(f"   ⚙️  Configuration Issues: {summary['configuration_issues']}")
        print(f"   📦 Dependency Issues: {summary['dependency_issues']}")
        
        # Compliance results
        if report['compliance_results']:
            print(f"\n🏆 Compliance Results:")
            for framework, results in report['compliance_results'].items():
                score = results['score'] * 100
                print(f"   {framework.upper()}: {score:.1f}% ({results['passed_checks']}/{results['total_checks']})")
        
        # Top recommendations
        if report['hardening_recommendations']:
            print(f"\n🛡️  Top Security Recommendations:")
            for i, rec in enumerate(report['hardening_recommendations'][:5], 1):
                priority_emoji = "🔴" if rec['priority'] == 'critical' else "🟠" if rec['priority'] == 'high' else "🟡"
                print(f"   {i}. {priority_emoji} {rec['title']}")


def main():
    """Main entry point for security audit system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Security Audit and Hardening System")
    parser.add_argument('--project-root', '-p', default='.', help='Project root directory')
    parser.add_argument('--config', '-c', help='Configuration file')
    parser.add_argument('--output', '-o', default='security_audit_report.json', help='Output report file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    parser.add_argument('--compliance', nargs='+', default=['owasp_top10'], help='Compliance frameworks to check')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🔒 SKZ Security Audit and Hardening System")
        print("=" * 60)
    
    # Initialize security audit system
    security_audit = SecurityAuditSystem(args.project_root, args.config)
    
    # Generate comprehensive security report
    report = security_audit.generate_security_report()
    security_audit.save_report(report, args.output)
    
    if not args.quiet:
        security_audit.print_security_summary(report)
        print("\n✅ Security audit completed successfully!")


if __name__ == "__main__":
    main()