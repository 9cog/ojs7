#!/usr/bin/env python3
"""
Security Hardening Configuration Manager
Applies security hardening configurations across OJS and SKZ components
"""

import os
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SecurityHardeningManager:
    """
    Manages security hardening configurations for OJS and SKZ integration
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "security_backups"
        self.config_templates = self._load_hardening_templates()
    
    def _load_hardening_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load security hardening configuration templates"""
        return {
            'php_security': {
                'ini_settings': {
                    'display_errors': 'Off',
                    'display_startup_errors': 'Off',
                    'log_errors': 'On',
                    'error_log': '/var/log/php_errors.log',
                    'expose_php': 'Off',
                    'allow_url_fopen': 'Off',
                    'allow_url_include': 'Off',
                    'session.cookie_httponly': 'On',
                    'session.cookie_secure': 'On',
                    'session.use_strict_mode': 'On',
                    'session.cookie_samesite': 'Strict',
                    'max_execution_time': '30',
                    'max_input_time': '60',
                    'memory_limit': '256M',
                    'post_max_size': '20M',
                    'upload_max_filesize': '20M',
                    'max_file_uploads': '20'
                }
            },
            'apache_security': {
                'headers': [
                    'Header always set X-Frame-Options "SAMEORIGIN"',
                    'Header always set X-Content-Type-Options "nosniff"',
                    'Header always set X-XSS-Protection "1; mode=block"',
                    'Header always set Referrer-Policy "strict-origin-when-cross-origin"',
                    'Header always set Permissions-Policy "geolocation=(), microphone=(), camera=()"',
                    'Header always set Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"'
                ],
                'security_rules': [
                    'ServerTokens Prod',
                    'ServerSignature Off',
                    'TraceEnable Off',
                    'Options -Indexes -FollowSymLinks',
                    'AllowOverride None'
                ]
            },
            'nginx_security': {
                'headers': {
                    'X-Frame-Options': 'SAMEORIGIN',
                    'X-Content-Type-Options': 'nosniff',
                    'X-XSS-Protection': '1; mode=block',
                    'Referrer-Policy': 'strict-origin-when-cross-origin',
                    'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
                    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload'
                },
                'security_config': [
                    'server_tokens off;',
                    'client_max_body_size 20M;',
                    'client_body_timeout 10s;',
                    'client_header_timeout 10s;',
                    'keepalive_timeout 5s 5s;',
                    'send_timeout 10s;'
                ]
            },
            'ojs_security': {
                'config_settings': {
                    'force_ssl': 'On',
                    'encryption': 'sha256',
                    'session_check_ip': 'On',
                    'session_lifetime': '30',
                    'disable_path_info': 'On',
                    'restful_urls': 'On',
                    'force_login_ssl': 'On',
                    'allow_protocol_relative_urls': 'Off'
                },
                'database_security': {
                    'driver': 'mysqli',
                    'charset': 'utf8mb4',
                    'collation': 'utf8mb4_unicode_ci'
                }
            },
            'skz_agents_security': {
                'authentication': {
                    'jwt_algorithm': 'HS256',
                    'jwt_expiry_hours': 24,
                    'require_https': True,
                    'enable_rate_limiting': True,
                    'max_requests_per_minute': 100
                },
                'api_security': {
                    'require_signature': True,
                    'signature_algorithm': 'sha256',
                    'timestamp_tolerance_seconds': 300,
                    'enable_cors': False,
                    'allowed_origins': []
                },
                'encryption': {
                    'algorithm': 'AES-256-GCM',
                    'key_rotation_days': 90,
                    'secure_random_source': '/dev/urandom'
                }
            }
        }
    
    def backup_existing_configs(self) -> Dict[str, str]:
        """Backup existing configuration files before hardening"""
        logger.info("Backing up existing configuration files...")
        
        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_paths = {}
        config_files = [
            'config.inc.php',
            '.htaccess',
            'nginx.conf',
            'php.ini'
        ]
        
        for config_file in config_files:
            source_path = self.project_root / config_file
            if source_path.exists():
                backup_path = self.backup_dir / f"{config_file}.backup_{timestamp}"
                shutil.copy2(source_path, backup_path)
                backup_paths[config_file] = str(backup_path)
                logger.info(f"Backed up {config_file} to {backup_path}")
        
        return backup_paths
    
    def apply_php_security_hardening(self) -> Dict[str, Any]:
        """Apply PHP security hardening"""
        logger.info("Applying PHP security hardening...")
        
        php_config = self.config_templates['php_security']
        applied_settings = {}
        
        # Generate php.ini security configuration
        php_ini_content = self._generate_php_ini_security_config(php_config['ini_settings'])
        
        # Write to security-specific PHP configuration file
        security_php_ini = self.project_root / 'php_security.ini'
        with open(security_php_ini, 'w') as f:
            f.write(php_ini_content)
        
        applied_settings['php_security_config'] = str(security_php_ini)
        
        # Generate .htaccess PHP settings
        htaccess_php_content = self._generate_htaccess_php_config(php_config['ini_settings'])
        applied_settings['htaccess_php_settings'] = htaccess_php_content
        
        return applied_settings
    
    def _generate_php_ini_security_config(self, settings: Dict[str, str]) -> str:
        """Generate PHP INI security configuration"""
        content = [
            "; PHP Security Hardening Configuration",
            "; Generated by SKZ Security Hardening Manager",
            "; Include this file in your main php.ini or use php_admin_value in web server config",
            ""
        ]
        
        for setting, value in settings.items():
            content.append(f"{setting} = {value}")
        
        return "\n".join(content)
    
    def _generate_htaccess_php_config(self, settings: Dict[str, str]) -> str:
        """Generate .htaccess PHP configuration"""
        content = [
            "# PHP Security Settings via .htaccess",
            "# Generated by SKZ Security Hardening Manager",
            ""
        ]
        
        for setting, value in settings.items():
            content.append(f"php_admin_value {setting} {value}")
        
        return "\n".join(content)
    
    def apply_webserver_security_hardening(self, server_type: str = 'apache') -> Dict[str, Any]:
        """Apply web server security hardening"""
        logger.info(f"Applying {server_type} security hardening...")
        
        applied_configs = {}
        
        if server_type.lower() == 'apache':
            applied_configs = self._apply_apache_hardening()
        elif server_type.lower() == 'nginx':
            applied_configs = self._apply_nginx_hardening()
        
        return applied_configs
    
    def _apply_apache_hardening(self) -> Dict[str, Any]:
        """Apply Apache security hardening"""
        apache_config = self.config_templates['apache_security']
        
        # Generate .htaccess security configuration
        htaccess_content = [
            "# Apache Security Hardening Configuration",
            "# Generated by SKZ Security Hardening Manager",
            "",
            "# Security Headers"
        ]
        
        htaccess_content.extend(apache_config['headers'])
        htaccess_content.extend([
            "",
            "# Security Rules"
        ])
        htaccess_content.extend(apache_config['security_rules'])
        
        htaccess_content.extend([
            "",
            "# Block sensitive files",
            "<Files ~ \"\\.(htaccess|htpasswd|ini|log|sh|inc|bak)$\">",
            "    Require all denied",
            "</Files>",
            "",
            "# Block directory browsing",
            "Options -Indexes",
            "",
            "# Disable server signature",
            "ServerSignature Off",
            "",
            "# Protect against clickjacking",
            "Header always append X-Frame-Options SAMEORIGIN",
            "",
            "# MIME type sniffing security protection",
            "Header always set X-Content-Type-Options nosniff",
            "",
            "# Enable XSS filtering",
            "Header always set X-XSS-Protection \"1; mode=block\"",
            "",
            "# HTTPS redirect (uncomment for production)",
            "# RewriteEngine On",
            "# RewriteCond %{HTTPS} off",
            "# RewriteRule ^(.*)$ https://%{HTTP_HOST}%{REQUEST_URI} [L,R=301]"
        ])
        
        # Write hardened .htaccess
        htaccess_path = self.project_root / '.htaccess.security'
        with open(htaccess_path, 'w') as f:
            f.write("\n".join(htaccess_content))
        
        return {
            'htaccess_security_config': str(htaccess_path),
            'applied_headers': len(apache_config['headers']),
            'applied_rules': len(apache_config['security_rules'])
        }
    
    def _apply_nginx_hardening(self) -> Dict[str, Any]:
        """Apply Nginx security hardening"""
        nginx_config = self.config_templates['nginx_security']
        
        # Generate nginx security configuration
        nginx_content = [
            "# Nginx Security Hardening Configuration",
            "# Generated by SKZ Security Hardening Manager",
            "# Include this in your nginx server block",
            ""
        ]
        
        # Add security headers
        nginx_content.append("# Security Headers")
        for header, value in nginx_config['headers'].items():
            nginx_content.append(f"add_header {header} \"{value}\" always;")
        
        nginx_content.extend([
            "",
            "# Security Configuration"
        ])
        nginx_content.extend(nginx_config['security_config'])
        
        nginx_content.extend([
            "",
            "# Block sensitive files",
            "location ~ /\\. {",
            "    deny all;",
            "    access_log off;",
            "    log_not_found off;",
            "}",
            "",
            "location ~ \\.(ini|log|conf)$ {",
            "    deny all;",
            "    access_log off;",
            "    log_not_found off;",
            "}",
            "",
            "# HTTPS redirect (uncomment for production)",
            "# if ($scheme != \"https\") {",
            "#     return 301 https://$host$request_uri;",
            "# }"
        ])
        
        # Write hardened nginx config
        nginx_path = self.project_root / 'nginx_security.conf'
        with open(nginx_path, 'w') as f:
            f.write("\n".join(nginx_content))
        
        return {
            'nginx_security_config': str(nginx_path),
            'applied_headers': len(nginx_config['headers']),
            'applied_configs': len(nginx_config['security_config'])
        }
    
    def apply_ojs_security_hardening(self) -> Dict[str, Any]:
        """Apply OJS-specific security hardening"""
        logger.info("Applying OJS security hardening...")
        
        ojs_config = self.config_templates['ojs_security']
        
        # Read existing config.inc.php
        config_file = self.project_root / 'config.inc.php'
        if not config_file.exists():
            logger.error("OJS config.inc.php not found")
            return {}
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Apply security settings
        hardened_content = self._apply_ojs_config_settings(content, ojs_config['config_settings'])
        
        # Write hardened configuration
        hardened_config_file = self.project_root / 'config.inc.php.hardened'
        with open(hardened_config_file, 'w') as f:
            f.write(hardened_content)
        
        return {
            'hardened_config': str(hardened_config_file),
            'applied_settings': len(ojs_config['config_settings'])
        }
    
    def _apply_ojs_config_settings(self, content: str, settings: Dict[str, str]) -> str:
        """Apply OJS configuration settings"""
        import re
        
        lines = content.split('\n')
        modified_lines = []
        applied_settings = set()
        
        for line in lines:
            modified = False
            
            for setting, value in settings.items():
                # Check if this line contains the setting
                pattern = rf'^(\s*;?\s*{re.escape(setting)}\s*=\s*)[^;\n]*'
                match = re.match(pattern, line)
                
                if match:
                    # Replace the setting
                    new_line = f"{setting} = {value}"
                    modified_lines.append(new_line)
                    applied_settings.add(setting)
                    modified = True
                    break
            
            if not modified:
                modified_lines.append(line)
        
        # Add any settings that weren't found in the file
        missing_settings = set(settings.keys()) - applied_settings
        if missing_settings:
            modified_lines.append("")
            modified_lines.append("; Security hardening settings added by SKZ Security Manager")
            for setting in missing_settings:
                modified_lines.append(f"{setting} = {settings[setting]}")
        
        return '\n'.join(modified_lines)
    
    def apply_skz_agents_security_hardening(self) -> Dict[str, Any]:
        """Apply SKZ agents security hardening"""
        logger.info("Applying SKZ agents security hardening...")
        
        skz_config = self.config_templates['skz_agents_security']
        
        # Generate security configuration file
        security_config = {
            'security': skz_config
        }
        
        config_file = self.project_root / 'skz-integration' / 'security_config.yaml'
        config_file.parent.mkdir(exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(security_config, f, default_flow_style=False, indent=2)
        
        # Generate environment variables template
        env_template = self._generate_security_env_template(skz_config)
        env_file = self.project_root / 'skz-integration' / '.env.security'
        with open(env_file, 'w') as f:
            f.write(env_template)
        
        return {
            'security_config': str(config_file),
            'env_template': str(env_file),
            'applied_categories': len(skz_config)
        }
    
    def _generate_security_env_template(self, config: Dict[str, Any]) -> str:
        """Generate security environment variables template"""
        content = [
            "# SKZ Agents Security Environment Variables",
            "# Generated by SKZ Security Hardening Manager",
            "# Copy to .env and set appropriate values",
            ""
        ]
        
        # Authentication settings
        content.extend([
            "# Authentication Settings",
            "SKZ_JWT_SECRET=change-me-to-a-strong-secret-key",
            "SKZ_JWT_ALGORITHM=HS256",
            f"SKZ_JWT_EXPIRY_HOURS={config['authentication']['jwt_expiry_hours']}",
            f"SKZ_REQUIRE_HTTPS={str(config['authentication']['require_https']).lower()}",
            f"SKZ_RATE_LIMIT_ENABLED={str(config['authentication']['enable_rate_limiting']).lower()}",
            f"SKZ_MAX_REQUESTS_PER_MINUTE={config['authentication']['max_requests_per_minute']}",
            ""
        ])
        
        # API Security settings
        content.extend([
            "# API Security Settings",
            "SKZ_API_SECRET=change-me-to-a-strong-api-secret",
            f"SKZ_REQUIRE_SIGNATURE={str(config['api_security']['require_signature']).lower()}",
            f"SKZ_SIGNATURE_ALGORITHM={config['api_security']['signature_algorithm']}",
            f"SKZ_TIMESTAMP_TOLERANCE={config['api_security']['timestamp_tolerance_seconds']}",
            f"SKZ_ENABLE_CORS={str(config['api_security']['enable_cors']).lower()}",
            "SKZ_ALLOWED_ORIGINS=https://your-domain.com",
            ""
        ])
        
        # Encryption settings
        content.extend([
            "# Encryption Settings",
            "SKZ_ENCRYPTION_KEY=generate-a-32-byte-encryption-key",
            f"SKZ_ENCRYPTION_ALGORITHM={config['encryption']['algorithm']}",
            f"SKZ_KEY_ROTATION_DAYS={config['encryption']['key_rotation_days']}",
            ""
        ])
        
        return "\n".join(content)
    
    def apply_comprehensive_hardening(self, server_type: str = 'apache') -> Dict[str, Any]:
        """Apply comprehensive security hardening across all components"""
        logger.info("Applying comprehensive security hardening...")
        
        # Backup existing configurations
        backup_info = self.backup_existing_configs()
        
        # Apply hardening to all components
        results = {
            'backups': backup_info,
            'php_hardening': self.apply_php_security_hardening(),
            'webserver_hardening': self.apply_webserver_security_hardening(server_type),
            'ojs_hardening': self.apply_ojs_security_hardening(),
            'skz_hardening': self.apply_skz_agents_security_hardening()
        }
        
        # Generate deployment instructions
        instructions = self._generate_deployment_instructions(results)
        instructions_file = self.project_root / 'SECURITY_DEPLOYMENT_INSTRUCTIONS.md'
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        results['deployment_instructions'] = str(instructions_file)
        
        return results
    
    def _generate_deployment_instructions(self, results: Dict[str, Any]) -> str:
        """Generate deployment instructions for hardened configurations"""
        content = [
            "# Security Hardening Deployment Instructions",
            "",
            "This document provides step-by-step instructions for deploying the security hardening configurations generated by the SKZ Security Hardening Manager.",
            "",
            "## Prerequisites",
            "",
            "- Administrative access to web server",
            "- PHP configuration access",
            "- Database access (for OJS configuration)",
            "- SSL/TLS certificate configured",
            "",
            "## Deployment Steps",
            "",
            "### 1. PHP Security Configuration",
            "",
            f"Apply PHP security settings from: `{results['php_hardening'].get('php_security_config', 'Not generated')}`",
            "",
            "**Option A: Include in main php.ini**",
            "```bash",
            "# Append to main php.ini",
            "cat php_security.ini >> /etc/php/php.ini",
            "```",
            "",
            "**Option B: Use in web server configuration**",
            "For Apache, add to virtual host:",
            "```apache",
            "php_admin_value include_path \"/path/to/php_security.ini\"",
            "```",
            "",
            "### 2. Web Server Security Configuration",
            ""
        ]
        
        if 'htaccess_security_config' in results.get('webserver_hardening', {}):
            content.extend([
                "**Apache Configuration:**",
                f"Replace or merge with existing .htaccess: `{results['webserver_hardening']['htaccess_security_config']}`",
                "",
                "```bash",
                "# Backup existing .htaccess",
                "mv .htaccess .htaccess.backup",
                "# Deploy new security configuration",
                "mv .htaccess.security .htaccess",
                "```",
                ""
            ])
        
        if 'nginx_security_config' in results.get('webserver_hardening', {}):
            content.extend([
                "**Nginx Configuration:**",
                f"Include security configuration: `{results['webserver_hardening']['nginx_security_config']}`",
                "",
                "```bash",
                "# Include in nginx server block",
                "include /path/to/nginx_security.conf;",
                "```",
                ""
            ])
        
        content.extend([
            "### 3. OJS Security Configuration",
            "",
            f"Review and apply OJS hardened configuration: `{results['ojs_hardening'].get('hardened_config', 'Not generated')}`",
            "",
            "```bash",
            "# Backup existing configuration",
            "cp config.inc.php config.inc.php.backup",
            "# Review changes and apply",
            "diff config.inc.php config.inc.php.hardened",
            "cp config.inc.php.hardened config.inc.php",
            "```",
            "",
            "### 4. SKZ Agents Security Configuration",
            "",
            f"Configure SKZ agents security: `{results['skz_hardening'].get('security_config', 'Not generated')}`",
            "",
            "```bash",
            "# Copy environment template",
            f"cp {results['skz_hardening'].get('env_template', '.env.security')} .env",
            "# Edit .env with your specific values",
            "nano .env",
            "```",
            "",
            "**Required Environment Variables:**",
            "- `SKZ_JWT_SECRET`: Strong secret for JWT tokens",
            "- `SKZ_API_SECRET`: Strong secret for API authentication",
            "- `SKZ_ENCRYPTION_KEY`: 32-byte encryption key",
            "",
            "### 5. SSL/TLS Configuration",
            "",
            "Ensure SSL/TLS is properly configured:",
            "",
            "```bash",
            "# Test SSL configuration",
            "openssl s_client -connect yourdomain.com:443 -servername yourdomain.com",
            "",
            "# Verify security headers",
            "curl -I https://yourdomain.com",
            "```",
            "",
            "### 6. Restart Services",
            "",
            "After applying configurations, restart all services:",
            "",
            "```bash",
            "# Restart web server",
            "sudo systemctl restart apache2  # or nginx",
            "",
            "# Restart PHP-FPM (if used)",
            "sudo systemctl restart php-fpm",
            "",
            "# Restart SKZ agents",
            "cd skz-integration && ./restart_agents.sh",
            "```",
            "",
            "### 7. Verification",
            "",
            "Verify the security hardening is working:",
            "",
            "```bash",
            "# Run security audit",
            "python3 skz-integration/security_audit_system.py",
            "",
            "# Test security headers",
            "curl -I https://yourdomain.com | grep -E \"X-Frame-Options|X-XSS-Protection|Strict-Transport-Security\"",
            "",
            "# Test SKZ agents authentication",
            "python3 skz-integration/microservices/test_auth.py",
            "```",
            "",
            "## Security Monitoring",
            "",
            "Set up ongoing security monitoring:",
            "",
            "1. **Log Monitoring**: Configure log aggregation for security events",
            "2. **Automated Scanning**: Schedule regular security audits",
            "3. **Alert System**: Set up alerts for security violations",
            "4. **Backup Verification**: Regularly test backup and recovery procedures",
            "",
            "## Rollback Procedures",
            "",
            "If issues occur, rollback using the backed-up configurations:",
            "",
            "```bash",
            "# Restore from backups"
        ])
        
        for config_file, backup_path in results.get('backups', {}).items():
            content.append(f"cp {backup_path} {config_file}")
        
        content.extend([
            "```",
            "",
            "## Support",
            "",
            "For issues with security hardening:",
            "",
            "1. Check the security audit report for specific recommendations",
            "2. Review server logs for configuration errors",
            "3. Test individual components after each configuration change",
            "4. Consult the SKZ Security Documentation for advanced configurations",
            "",
            "---",
            "",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "By: SKZ Security Hardening Manager"
        ])
        
        return "\n".join(content)


# Import datetime for deployment instructions
from datetime import datetime


def main():
    """Main entry point for security hardening manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Security Hardening Configuration Manager")
    parser.add_argument('--project-root', '-p', default='.', help='Project root directory')
    parser.add_argument('--server-type', '-s', choices=['apache', 'nginx'], default='apache', help='Web server type')
    parser.add_argument('--component', '-c', choices=['php', 'webserver', 'ojs', 'skz', 'all'], default='all', help='Component to harden')
    parser.add_argument('--backup', '-b', action='store_true', help='Create backups before applying changes')
    
    args = parser.parse_args()
    
    print("🛡️  SKZ Security Hardening Manager")
    print("=" * 50)
    
    # Initialize hardening manager
    hardening_manager = SecurityHardeningManager(args.project_root)
    
    if args.backup:
        print("📦 Creating configuration backups...")
        backups = hardening_manager.backup_existing_configs()
        for config, backup_path in backups.items():
            print(f"   ✅ {config} backed up to {backup_path}")
    
    # Apply hardening based on component selection
    if args.component == 'all':
        print("🔒 Applying comprehensive security hardening...")
        results = hardening_manager.apply_comprehensive_hardening(args.server_type)
        
        print("\n✅ Security hardening completed!")
        print(f"   📝 Deployment instructions: {results['deployment_instructions']}")
        
    elif args.component == 'php':
        print("🐘 Applying PHP security hardening...")
        results = hardening_manager.apply_php_security_hardening()
        print(f"   ✅ PHP security config: {results['php_security_config']}")
        
    elif args.component == 'webserver':
        print(f"🌐 Applying {args.server_type} security hardening...")
        results = hardening_manager.apply_webserver_security_hardening(args.server_type)
        print(f"   ✅ Configuration applied: {len(results)} items")
        
    elif args.component == 'ojs':
        print("📚 Applying OJS security hardening...")
        results = hardening_manager.apply_ojs_security_hardening()
        print(f"   ✅ OJS hardened config: {results['hardened_config']}")
        
    elif args.component == 'skz':
        print("🤖 Applying SKZ agents security hardening...")
        results = hardening_manager.apply_skz_agents_security_hardening()
        print(f"   ✅ SKZ security config: {results['security_config']}")
    
    print("\n🎯 Next steps:")
    print("   1. Review generated configuration files")
    print("   2. Follow deployment instructions")
    print("   3. Test security configurations")
    print("   4. Run security audit to verify")


if __name__ == "__main__":
    main()