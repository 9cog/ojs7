#!/usr/bin/env python3
"""
Real-time Security Monitoring System
Monitors security events and provides real-time alerts for OJS and SKZ integration
"""

import os
import json
import time
import queue
import threading
import logging
import hashlib
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Fallback for systems without full email support
    MimeText = None
    MimeMultipart = None
from pathlib import Path
from collections import defaultdict, deque
import re
import socket
import subprocess

logger = logging.getLogger(__name__)

class SecurityEvent:
    """Represents a security event"""
    
    def __init__(self, event_type: str, severity: str, source: str, 
                 description: str, details: Dict[str, Any] = None):
        self.event_id = hashlib.md5(f"{time.time()}{event_type}{source}".encode()).hexdigest()[:8]
        self.event_type = event_type
        self.severity = severity
        self.source = source
        self.description = description
        self.details = details or {}
        self.timestamp = datetime.now()
        self.acknowledged = False
        self.resolved = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'severity': self.severity,
            'source': self.source,
            'description': self.description,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved
        }


class SecurityAlertManager:
    """Manages security alerts and notifications"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = self._default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        self.alert_handlers = []
        self.setup_alert_handlers()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'email': {
                'enabled': False,
                'smtp_server': 'localhost',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_address': 'security@localhost',
                'to_addresses': ['admin@localhost']
            },
            'webhook': {
                'enabled': False,
                'url': '',
                'method': 'POST',
                'headers': {}
            },
            'log': {
                'enabled': True,
                'file': 'security_alerts.log',
                'level': 'INFO'
            },
            'console': {
                'enabled': True,
                'level': 'WARNING'
            }
        }
    
    def setup_alert_handlers(self):
        """Setup alert handlers based on configuration"""
        if self.config['email']['enabled']:
            self.alert_handlers.append(self.send_email_alert)
        
        if self.config['webhook']['enabled']:
            self.alert_handlers.append(self.send_webhook_alert)
        
        if self.config['log']['enabled']:
            self.alert_handlers.append(self.log_alert)
        
        if self.config['console']['enabled']:
            self.alert_handlers.append(self.console_alert)
    
    def send_alert(self, event: SecurityEvent):
        """Send alert using all configured handlers"""
        for handler in self.alert_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Failed to send alert via {handler.__name__}: {e}")
    
    def send_email_alert(self, event: SecurityEvent):
        """Send email alert"""
        if not MimeText or not MimeMultipart:
            logger.warning("Email functionality not available - skipping email alert")
            return
            
        if event.severity not in ['critical', 'high']:
            return
        
        config = self.config['email']
        
        subject = f"[SECURITY ALERT] {event.severity.upper()}: {event.event_type}"
        
        body = f"""
Security Alert

Event ID: {event.event_id}
Type: {event.event_type}
Severity: {event.severity}
Source: {event.source}
Time: {event.timestamp}

Description:
{event.description}

Details:
{json.dumps(event.details, indent=2)}

This is an automated security alert from the SKZ Security Monitoring System.
        """.strip()
        
        msg = MimeMultipart()
        msg['From'] = config['from_address']
        msg['To'] = ', '.join(config['to_addresses'])
        msg['Subject'] = subject
        msg.attach(MimeText(body, 'plain'))
        
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            if config['username'] and config['password']:
                server.starttls()
                server.login(config['username'], config['password'])
            server.send_message(msg)
    
    def send_webhook_alert(self, event: SecurityEvent):
        """Send webhook alert"""
        import requests
        
        config = self.config['webhook']
        
        payload = {
            'alert_type': 'security_event',
            'event': event.to_dict()
        }
        
        response = requests.request(
            config['method'],
            config['url'],
            json=payload,
            headers=config['headers'],
            timeout=10
        )
        response.raise_for_status()
    
    def log_alert(self, event: SecurityEvent):
        """Log alert to file"""
        log_file = Path(self.config['log']['file'])
        log_entry = f"{event.timestamp.isoformat()} | {event.severity.upper()} | {event.event_type} | {event.source} | {event.description}\n"
        
        with open(log_file, 'a') as f:
            f.write(log_entry)
    
    def console_alert(self, event: SecurityEvent):
        """Print alert to console"""
        severity_colors = {
            'critical': '\033[91m',  # Red
            'high': '\033[93m',      # Yellow
            'medium': '\033[94m',    # Blue
            'low': '\033[92m',       # Green
            'info': '\033[96m'       # Cyan
        }
        reset_color = '\033[0m'
        
        color = severity_colors.get(event.severity, '')
        
        print(f"{color}[SECURITY ALERT] {event.timestamp} | {event.severity.upper()} | {event.event_type}{reset_color}")
        print(f"  Source: {event.source}")
        print(f"  Description: {event.description}")
        if event.details:
            print(f"  Details: {json.dumps(event.details, indent=2)}")


class SecurityMonitor:
    """Real-time security monitoring system"""
    
    def __init__(self, project_root: str = ".", config_file: str = None):
        self.project_root = Path(project_root)
        self.config = self._load_config(config_file)
        self.alert_manager = SecurityAlertManager(self.config.get('alerts', {}))
        self.event_queue = queue.Queue()
        self.monitoring_threads = []
        self.running = False
        self.event_history = deque(maxlen=1000)
        self.rate_limiters = defaultdict(lambda: deque(maxlen=100))
        
        # Security patterns for log monitoring
        self.security_patterns = {
            'sql_injection_attempt': re.compile(r'(?i)(union\s+select|or\s+1\s*=\s*1|drop\s+table)', re.IGNORECASE),
            'xss_attempt': re.compile(r'(?i)(<script|javascript:|onload=|onerror=)', re.IGNORECASE),
            'directory_traversal': re.compile(r'\.\.[\\/]', re.IGNORECASE),
            'authentication_failure': re.compile(r'(?i)(login\s+failed|authentication\s+failed|invalid\s+credentials)', re.IGNORECASE),
            'privilege_escalation': re.compile(r'(?i)(sudo|su\s+root|chmod\s+777)', re.IGNORECASE),
            'suspicious_user_agent': re.compile(r'(?i)(sqlmap|nikto|nessus|burp|acunetix)', re.IGNORECASE)
        }
        
        # Initialize monitoring components
        self.setup_monitoring()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            'log_monitoring': {
                'enabled': True,
                'files': [
                    '/var/log/apache2/access.log',
                    '/var/log/nginx/access.log',
                    '/var/log/php_errors.log',
                    'logs/ojs.log'
                ],
                'check_interval': 5
            },
            'file_integrity': {
                'enabled': True,
                'paths': [
                    'config.inc.php',
                    'index.php',
                    'classes/',
                    'plugins/',
                    'skz-integration/'
                ],
                'check_interval': 300
            },
            'network_monitoring': {
                'enabled': False,
                'interfaces': ['eth0'],
                'suspicious_ports': [22, 3389, 5900]
            },
            'process_monitoring': {
                'enabled': True,
                'check_interval': 60,
                'suspicious_processes': [
                    'nc', 'netcat', 'nmap', 'sqlmap', 'nikto'
                ]
            },
            'api_monitoring': {
                'enabled': True,
                'endpoints': [
                    'http://localhost:5000/api/v1/health',
                    'http://localhost:5001/health'
                ],
                'check_interval': 30
            },
            'alerts': {
                'rate_limiting': {
                    'enabled': True,
                    'max_events_per_minute': 10
                }
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
        
        return default_config
    
    def setup_monitoring(self):
        """Setup monitoring components"""
        logger.info("Setting up security monitoring components...")
        
        if self.config['log_monitoring']['enabled']:
            self.monitoring_threads.append(
                threading.Thread(target=self.monitor_logs, daemon=True)
            )
        
        if self.config['file_integrity']['enabled']:
            self.monitoring_threads.append(
                threading.Thread(target=self.monitor_file_integrity, daemon=True)
            )
        
        if self.config['process_monitoring']['enabled']:
            self.monitoring_threads.append(
                threading.Thread(target=self.monitor_processes, daemon=True)
            )
        
        if self.config['api_monitoring']['enabled']:
            self.monitoring_threads.append(
                threading.Thread(target=self.monitor_api_endpoints, daemon=True)
            )
        
        # Event processing thread
        self.monitoring_threads.append(
            threading.Thread(target=self.process_events, daemon=True)
        )
    
    def start_monitoring(self):
        """Start all monitoring threads"""
        logger.info("Starting security monitoring...")
        self.running = True
        
        for thread in self.monitoring_threads:
            thread.start()
        
        logger.info(f"Started {len(self.monitoring_threads)} monitoring threads")
    
    def stop_monitoring(self):
        """Stop all monitoring"""
        logger.info("Stopping security monitoring...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.monitoring_threads:
            thread.join(timeout=5)
    
    def add_event(self, event: SecurityEvent):
        """Add security event to queue"""
        if self._should_rate_limit(event):
            return
        
        self.event_queue.put(event)
        self.event_history.append(event)
    
    def _should_rate_limit(self, event: SecurityEvent) -> bool:
        """Check if event should be rate limited"""
        if not self.config['alerts'].get('rate_limiting', {}).get('enabled', True):
            return False
        
        max_events = self.config['alerts']['rate_limiting'].get('max_events_per_minute', 10)
        rate_limiter = self.rate_limiters[f"{event.event_type}_{event.source}"]
        
        now = time.time()
        # Remove events older than 1 minute
        while rate_limiter and rate_limiter[0] < now - 60:
            rate_limiter.popleft()
        
        if len(rate_limiter) >= max_events:
            return True
        
        rate_limiter.append(now)
        return False
    
    def process_events(self):
        """Process security events from queue"""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1)
                self.alert_manager.send_alert(event)
                logger.debug(f"Processed security event: {event.event_id}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing security event: {e}")
    
    def monitor_logs(self):
        """Monitor log files for security events"""
        log_files = {}
        
        # Initialize log file positions
        for log_file in self.config['log_monitoring']['files']:
            log_path = Path(log_file)
            if log_path.exists():
                log_files[log_file] = log_path.stat().st_size
        
        while self.running:
            try:
                for log_file, last_position in list(log_files.items()):
                    log_path = Path(log_file)
                    
                    if not log_path.exists():
                        continue
                    
                    current_size = log_path.stat().st_size
                    
                    if current_size > last_position:
                        # Read new content
                        with open(log_path, 'r', errors='ignore') as f:
                            f.seek(last_position)
                            new_content = f.read()
                        
                        # Check for security patterns
                        self._analyze_log_content(new_content, log_file)
                        log_files[log_file] = current_size
                
                time.sleep(self.config['log_monitoring']['check_interval'])
                
            except Exception as e:
                logger.error(f"Error monitoring logs: {e}")
                time.sleep(10)
    
    def _analyze_log_content(self, content: str, source_file: str):
        """Analyze log content for security patterns"""
        lines = content.split('\n')
        
        for line in lines:
            if not line.strip():
                continue
            
            for pattern_name, pattern in self.security_patterns.items():
                if pattern.search(line):
                    event = SecurityEvent(
                        event_type=pattern_name,
                        severity=self._get_pattern_severity(pattern_name),
                        source=f"log_monitor:{source_file}",
                        description=f"Security pattern detected in log file: {pattern_name}",
                        details={
                            'log_file': source_file,
                            'log_line': line.strip(),
                            'pattern': pattern_name
                        }
                    )
                    self.add_event(event)
    
    def _get_pattern_severity(self, pattern_name: str) -> str:
        """Get severity level for security pattern"""
        severity_map = {
            'sql_injection_attempt': 'critical',
            'xss_attempt': 'high',
            'directory_traversal': 'high',
            'authentication_failure': 'medium',
            'privilege_escalation': 'critical',
            'suspicious_user_agent': 'medium'
        }
        return severity_map.get(pattern_name, 'medium')
    
    def monitor_file_integrity(self):
        """Monitor file integrity for changes"""
        file_hashes = {}
        
        # Calculate initial hashes
        for path in self.config['file_integrity']['paths']:
            file_path = self.project_root / path
            if file_path.exists():
                if file_path.is_file():
                    file_hashes[str(file_path)] = self._calculate_file_hash(file_path)
                elif file_path.is_dir():
                    for file in file_path.rglob('*'):
                        if file.is_file():
                            file_hashes[str(file)] = self._calculate_file_hash(file)
        
        while self.running:
            try:
                for file_path, old_hash in list(file_hashes.items()):
                    path = Path(file_path)
                    
                    if not path.exists():
                        # File deleted
                        event = SecurityEvent(
                            event_type='file_deleted',
                            severity='medium',
                            source='file_integrity_monitor',
                            description=f"Monitored file deleted: {file_path}",
                            details={'file_path': file_path}
                        )
                        self.add_event(event)
                        del file_hashes[file_path]
                        continue
                    
                    new_hash = self._calculate_file_hash(path)
                    if new_hash != old_hash:
                        # File modified
                        event = SecurityEvent(
                            event_type='file_modified',
                            severity='medium',
                            source='file_integrity_monitor',
                            description=f"Monitored file modified: {file_path}",
                            details={
                                'file_path': file_path,
                                'old_hash': old_hash,
                                'new_hash': new_hash
                            }
                        )
                        self.add_event(event)
                        file_hashes[file_path] = new_hash
                
                time.sleep(self.config['file_integrity']['check_interval'])
                
            except Exception as e:
                logger.error(f"Error monitoring file integrity: {e}")
                time.sleep(30)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        except Exception:
            return ""
        return hash_sha256.hexdigest()
    
    def monitor_processes(self):
        """Monitor running processes for suspicious activity"""
        while self.running:
            try:
                # Get running processes
                result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                if result.returncode == 0:
                    processes = result.stdout.split('\n')
                    
                    for process_line in processes[1:]:  # Skip header
                        if not process_line.strip():
                            continue
                        
                        for suspicious_proc in self.config['process_monitoring']['suspicious_processes']:
                            if suspicious_proc in process_line:
                                event = SecurityEvent(
                                    event_type='suspicious_process',
                                    severity='high',
                                    source='process_monitor',
                                    description=f"Suspicious process detected: {suspicious_proc}",
                                    details={
                                        'process_line': process_line.strip(),
                                        'suspicious_pattern': suspicious_proc
                                    }
                                )
                                self.add_event(event)
                
                time.sleep(self.config['process_monitoring']['check_interval'])
                
            except Exception as e:
                logger.error(f"Error monitoring processes: {e}")
                time.sleep(30)
    
    def monitor_api_endpoints(self):
        """Monitor API endpoints for availability and security"""
        while self.running:
            try:
                for endpoint in self.config['api_monitoring']['endpoints']:
                    try:
                        import requests
                        response = requests.get(endpoint, timeout=10)
                        
                        if response.status_code != 200:
                            event = SecurityEvent(
                                event_type='api_endpoint_down',
                                severity='medium',
                                source='api_monitor',
                                description=f"API endpoint returned non-200 status: {endpoint}",
                                details={
                                    'endpoint': endpoint,
                                    'status_code': response.status_code,
                                    'response_time': response.elapsed.total_seconds()
                                }
                            )
                            self.add_event(event)
                        
                        # Check for security headers
                        missing_headers = []
                        security_headers = [
                            'X-Frame-Options',
                            'X-Content-Type-Options',
                            'X-XSS-Protection'
                        ]
                        
                        for header in security_headers:
                            if header not in response.headers:
                                missing_headers.append(header)
                        
                        if missing_headers:
                            event = SecurityEvent(
                                event_type='missing_security_headers',
                                severity='low',
                                source='api_monitor',
                                description=f"Missing security headers on endpoint: {endpoint}",
                                details={
                                    'endpoint': endpoint,
                                    'missing_headers': missing_headers
                                }
                            )
                            self.add_event(event)
                    
                    except requests.exceptions.RequestException as e:
                        event = SecurityEvent(
                            event_type='api_endpoint_unreachable',
                            severity='high',
                            source='api_monitor',
                            description=f"API endpoint unreachable: {endpoint}",
                            details={
                                'endpoint': endpoint,
                                'error': str(e)
                            }
                        )
                        self.add_event(event)
                
                time.sleep(self.config['api_monitoring']['check_interval'])
                
            except Exception as e:
                logger.error(f"Error monitoring API endpoints: {e}")
                time.sleep(30)
    
    def get_event_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of security events from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.event_history if e.timestamp >= cutoff_time]
        
        summary = {
            'total_events': len(recent_events),
            'events_by_severity': defaultdict(int),
            'events_by_type': defaultdict(int),
            'events_by_source': defaultdict(int),
            'critical_events': [],
            'timeframe_hours': hours
        }
        
        for event in recent_events:
            summary['events_by_severity'][event.severity] += 1
            summary['events_by_type'][event.event_type] += 1
            summary['events_by_source'][event.source] += 1
            
            if event.severity == 'critical':
                summary['critical_events'].append(event.to_dict())
        
        return summary
    
    def export_events(self, output_file: str, hours: int = 24):
        """Export security events to file"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [e.to_dict() for e in self.event_history if e.timestamp >= cutoff_time]
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'timeframe_hours': hours,
            'total_events': len(recent_events),
            'events': recent_events
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


def main():
    """Main entry point for security monitoring system"""
    import argparse
    import signal
    
    parser = argparse.ArgumentParser(description="Real-time Security Monitoring System")
    parser.add_argument('--project-root', '-p', default='.', help='Project root directory')
    parser.add_argument('--config', '-c', help='Configuration file')
    parser.add_argument('--daemon', '-d', action='store_true', help='Run as daemon')
    parser.add_argument('--export', '-e', help='Export events to file')
    parser.add_argument('--summary', '-s', action='store_true', help='Show event summary')
    parser.add_argument('--hours', type=int, default=24, help='Hours for summary/export')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize security monitor
    monitor = SecurityMonitor(args.project_root, args.config)
    
    if args.summary:
        summary = monitor.get_event_summary(args.hours)
        print(json.dumps(summary, indent=2, default=str))
        return
    
    if args.export:
        monitor.export_events(args.export, args.hours)
        print(f"Events exported to {args.export}")
        return
    
    print("🔒 SKZ Security Monitoring System")
    print("=" * 50)
    print(f"Monitoring project: {args.project_root}")
    print("Press Ctrl+C to stop monitoring")
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print("\nStopping security monitor...")
        monitor.stop_monitoring()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start monitoring
    monitor.start_monitoring()
    
    if args.daemon:
        # Run as daemon
        while True:
            time.sleep(60)
    else:
        # Interactive mode - show live events
        try:
            while True:
                summary = monitor.get_event_summary(1)  # Last hour
                if summary['total_events'] > 0:
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Events in last hour: {summary['total_events']}")
                    for severity, count in summary['events_by_severity'].items():
                        print(f"  {severity}: {count}")
                
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nStopping security monitor...")
            monitor.stop_monitoring()


if __name__ == "__main__":
    main()