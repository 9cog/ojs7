#!/usr/bin/env python3
"""
SKZ Security Integration Manager
Orchestrates security auditing, hardening, and monitoring for OJS with SKZ agents
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from security_audit_system import SecurityAuditSystem
    from security_hardening_manager import SecurityHardeningManager
    from security_monitoring_system import SecurityMonitor
    from test_security_systems import run_security_tests
except ImportError as e:
    print(f"Error importing security modules: {e}")
    print("Make sure all security system files are in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SKZSecurityManager:
    """
    Main security manager that orchestrates all security operations
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / "security_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize security components
        self.audit_system = SecurityAuditSystem(str(self.project_root))
        self.hardening_manager = SecurityHardeningManager(str(self.project_root))
        self.monitor = None  # Initialized on demand
        
        self.security_state = {
            'last_audit': None,
            'last_hardening': None,
            'monitoring_active': False,
            'security_score': 0,
            'compliance_status': {}
        }
    
    def run_comprehensive_security_audit(self, save_results: bool = True) -> Dict[str, Any]:
        """Run comprehensive security audit"""
        logger.info("Starting comprehensive security audit...")
        
        try:
            # Generate security audit report
            audit_report = self.audit_system.generate_security_report()
            
            if save_results:
                # Save audit report
                audit_file = self.results_dir / f"security_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.audit_system.save_report(audit_report, str(audit_file))
                logger.info(f"Audit report saved to {audit_file}")
            
            # Update security state
            self.security_state['last_audit'] = datetime.now().isoformat()
            self.security_state['security_score'] = audit_report['security_score']
            self.security_state['compliance_status'] = audit_report.get('compliance_results', {})
            
            logger.info(f"Security audit completed. Score: {audit_report['security_score']}/100")
            return audit_report
            
        except Exception as e:
            logger.error(f"Security audit failed: {e}")
            raise
    
    def apply_security_hardening(self, server_type: str = 'apache', create_backup: bool = True) -> Dict[str, Any]:
        """Apply comprehensive security hardening"""
        logger.info("Starting security hardening process...")
        
        try:
            # Create backups if requested
            if create_backup:
                logger.info("Creating configuration backups...")
                backups = self.hardening_manager.backup_existing_configs()
                logger.info(f"Created {len(backups)} configuration backups")
            
            # Apply comprehensive hardening
            hardening_results = self.hardening_manager.apply_comprehensive_hardening(server_type)
            
            # Save hardening results
            results_file = self.results_dir / f"security_hardening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(hardening_results, f, indent=2, default=str)
            
            # Update security state
            self.security_state['last_hardening'] = datetime.now().isoformat()
            
            logger.info("Security hardening completed successfully")
            return hardening_results
            
        except Exception as e:
            logger.error(f"Security hardening failed: {e}")
            raise
    
    def start_security_monitoring(self, config_file: Optional[str] = None) -> None:
        """Start real-time security monitoring"""
        logger.info("Starting security monitoring...")
        
        try:
            self.monitor = SecurityMonitor(str(self.project_root), config_file)
            self.monitor.start_monitoring()
            
            self.security_state['monitoring_active'] = True
            logger.info("Security monitoring started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start security monitoring: {e}")
            raise
    
    def stop_security_monitoring(self) -> None:
        """Stop security monitoring"""
        if self.monitor:
            logger.info("Stopping security monitoring...")
            self.monitor.stop_monitoring()
            self.security_state['monitoring_active'] = False
            logger.info("Security monitoring stopped")
    
    def run_security_tests(self) -> bool:
        """Run comprehensive security test suite"""
        logger.info("Running security test suite...")
        
        try:
            success = run_security_tests()
            logger.info(f"Security tests completed. Success: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Security tests failed: {e}")
            return False
    
    def generate_security_dashboard(self) -> Dict[str, Any]:
        """Generate security dashboard data"""
        logger.info("Generating security dashboard...")
        
        # Get latest audit results
        audit_files = list(self.results_dir.glob("security_audit_*.json"))
        latest_audit = None
        
        if audit_files:
            latest_audit_file = max(audit_files, key=lambda x: x.stat().st_mtime)
            with open(latest_audit_file) as f:
                latest_audit = json.load(f)
        
        # Get monitoring summary if available
        monitoring_summary = {}
        if self.monitor:
            monitoring_summary = self.monitor.get_event_summary(24)
        
        # Generate dashboard data
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'security_state': self.security_state,
            'latest_audit': {
                'timestamp': latest_audit.get('timestamp') if latest_audit else None,
                'security_score': latest_audit.get('security_score', 0) if latest_audit else 0,
                'total_vulnerabilities': latest_audit.get('summary', {}).get('total_vulnerabilities', 0) if latest_audit else 0,
                'critical_vulnerabilities': latest_audit.get('summary', {}).get('critical_vulnerabilities', 0) if latest_audit else 0,
                'compliance_scores': {
                    framework: results.get('score', 0) * 100
                    for framework, results in (latest_audit.get('compliance_results', {}).items() if latest_audit else [])
                }
            },
            'monitoring_summary': monitoring_summary,
            'security_trends': self._calculate_security_trends(),
            'recommendations': self._generate_security_recommendations(latest_audit)
        }
        
        # Save dashboard data
        dashboard_file = self.results_dir / "security_dashboard.json"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard, f, indent=2, default=str)
        
        return dashboard
    
    def _calculate_security_trends(self) -> Dict[str, Any]:
        """Calculate security trends from historical data"""
        audit_files = list(self.results_dir.glob("security_audit_*.json"))
        
        if len(audit_files) < 2:
            return {'trend': 'insufficient_data', 'change': 0}
        
        # Sort files by modification time
        audit_files.sort(key=lambda x: x.stat().st_mtime)
        
        try:
            # Get latest two audit reports
            with open(audit_files[-1]) as f:
                latest = json.load(f)
            with open(audit_files[-2]) as f:
                previous = json.load(f)
            
            latest_score = latest.get('security_score', 0)
            previous_score = previous.get('security_score', 0)
            
            change = latest_score - previous_score
            trend = 'improving' if change > 5 else 'declining' if change < -5 else 'stable'
            
            return {
                'trend': trend,
                'change': change,
                'latest_score': latest_score,
                'previous_score': previous_score
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate security trends: {e}")
            return {'trend': 'error', 'change': 0}
    
    def _generate_security_recommendations(self, audit_report: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate security recommendations based on audit results"""
        recommendations = []
        
        if not audit_report:
            recommendations.append({
                'priority': 'high',
                'category': 'audit',
                'title': 'Run Security Audit',
                'description': 'No recent security audit found. Run comprehensive audit to assess current security posture.',
                'action': 'python3 skz-integration/security_integration_manager.py --audit'
            })
            return recommendations
        
        security_score = audit_report.get('security_score', 0)
        
        # Score-based recommendations
        if security_score < 60:
            recommendations.append({
                'priority': 'critical',
                'category': 'hardening',
                'title': 'Critical Security Hardening Required',
                'description': f'Security score is {security_score}/100. Immediate hardening required.',
                'action': 'python3 skz-integration/security_integration_manager.py --harden'
            })
        elif security_score < 80:
            recommendations.append({
                'priority': 'high',
                'category': 'hardening',
                'title': 'Security Improvements Needed',
                'description': f'Security score is {security_score}/100. Apply security hardening.',
                'action': 'python3 skz-integration/security_integration_manager.py --harden'
            })
        
        # Vulnerability-based recommendations
        summary = audit_report.get('summary', {})
        critical_vulns = summary.get('critical_vulnerabilities', 0)
        total_vulns = summary.get('total_vulnerabilities', 0)
        
        if critical_vulns > 0:
            recommendations.append({
                'priority': 'critical',
                'category': 'vulnerabilities',
                'title': 'Fix Critical Vulnerabilities',
                'description': f'{critical_vulns} critical vulnerabilities found. Immediate remediation required.',
                'action': 'Review audit report and apply fixes for critical vulnerabilities'
            })
        
        if total_vulns > 10:
            recommendations.append({
                'priority': 'medium',
                'category': 'vulnerabilities',
                'title': 'Address Security Vulnerabilities',
                'description': f'{total_vulns} total vulnerabilities found. Plan remediation activities.',
                'action': 'Review audit report and create remediation plan'
            })
        
        # Compliance-based recommendations
        compliance_results = audit_report.get('compliance_results', {})
        for framework, results in compliance_results.items():
            score = results.get('score', 0) * 100
            if score < 80:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'compliance',
                    'title': f'Improve {framework.upper()} Compliance',
                    'description': f'{framework.upper()} compliance is {score:.1f}%. Review failed checks.',
                    'action': f'Address compliance gaps identified in {framework} assessment'
                })
        
        # Monitoring recommendation
        if not self.security_state.get('monitoring_active', False):
            recommendations.append({
                'priority': 'medium',
                'category': 'monitoring',
                'title': 'Enable Security Monitoring',
                'description': 'Real-time security monitoring is not active. Enable for continuous protection.',
                'action': 'python3 skz-integration/security_integration_manager.py --monitor'
            })
        
        return recommendations
    
    def create_security_report(self, output_file: str = None) -> str:
        """Create comprehensive security report"""
        logger.info("Creating comprehensive security report...")
        
        # Generate dashboard data
        dashboard = self.generate_security_dashboard()
        
        # Create markdown report
        report_content = self._generate_markdown_report(dashboard)
        
        # Save report
        if not output_file:
            output_file = self.results_dir / f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Security report saved to {output_file}")
        return str(output_file)
    
    def _generate_markdown_report(self, dashboard: Dict[str, Any]) -> str:
        """Generate markdown security report"""
        content = [
            "# SKZ Security Assessment Report",
            "",
            f"**Generated**: {dashboard['timestamp']}",
            f"**Project**: {self.project_root}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Security score
        latest_audit = dashboard['latest_audit']
        score = latest_audit['security_score']
        score_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        
        content.extend([
            f"**Overall Security Score**: {score_emoji} {score}/100",
            "",
            f"- **Total Vulnerabilities**: {latest_audit['total_vulnerabilities']}",
            f"- **Critical Vulnerabilities**: {latest_audit['critical_vulnerabilities']}",
            f"- **Monitoring Status**: {'🟢 Active' if dashboard['security_state']['monitoring_active'] else '🔴 Inactive'}",
            ""
        ])
        
        # Compliance scores
        if latest_audit['compliance_scores']:
            content.extend([
                "## Compliance Status",
                ""
            ])
            
            for framework, score in latest_audit['compliance_scores'].items():
                compliance_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                content.append(f"- **{framework.upper()}**: {compliance_emoji} {score:.1f}%")
            
            content.append("")
        
        # Security trends
        trends = dashboard['security_trends']
        if trends['trend'] != 'insufficient_data':
            content.extend([
                "## Security Trends",
                "",
                f"**Trend**: {trends['trend'].title()}",
                f"**Change**: {trends['change']:+.1f} points",
                ""
            ])
        
        # Monitoring summary
        monitoring = dashboard['monitoring_summary']
        if monitoring:
            content.extend([
                "## Security Events (Last 24 Hours)",
                "",
                f"- **Total Events**: {monitoring.get('total_events', 0)}",
                f"- **Critical Events**: {monitoring.get('events_by_severity', {}).get('critical', 0)}",
                f"- **High Severity Events**: {monitoring.get('events_by_severity', {}).get('high', 0)}",
                ""
            ])
        
        # Recommendations
        recommendations = dashboard['recommendations']
        if recommendations:
            content.extend([
                "## Security Recommendations",
                ""
            ])
            
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = "🔴" if rec['priority'] == 'critical' else "🟠" if rec['priority'] == 'high' else "🟡"
                content.extend([
                    f"### {i}. {priority_emoji} {rec['title']}",
                    "",
                    f"**Priority**: {rec['priority'].title()}",
                    f"**Category**: {rec['category'].title()}",
                    "",
                    f"{rec['description']}",
                    "",
                    f"**Action**: `{rec['action']}`",
                    ""
                ])
        
        content.extend([
            "## Next Steps",
            "",
            "1. Review and address critical and high-priority recommendations",
            "2. Schedule regular security audits (monthly recommended)",
            "3. Ensure security monitoring is active and configured",
            "4. Plan remediation activities for identified vulnerabilities",
            "5. Review and update security configurations regularly",
            "",
            "---",
            "",
            "*This report was generated automatically by the SKZ Security Integration Manager.*"
        ])
        
        return "\n".join(content)


def main():
    """Main entry point for security integration manager"""
    parser = argparse.ArgumentParser(description="SKZ Security Integration Manager")
    parser.add_argument('--project-root', '-p', default='.', help='Project root directory')
    parser.add_argument('--audit', '-a', action='store_true', help='Run security audit')
    parser.add_argument('--harden', action='store_true', help='Apply security hardening')
    parser.add_argument('--monitor', '-m', action='store_true', help='Start security monitoring')
    parser.add_argument('--test', '-t', action='store_true', help='Run security tests')
    parser.add_argument('--dashboard', '-d', action='store_true', help='Generate security dashboard')
    parser.add_argument('--report', '-r', action='store_true', help='Create security report')
    parser.add_argument('--all', action='store_true', help='Run all security operations')
    parser.add_argument('--server-type', choices=['apache', 'nginx'], default='apache', help='Web server type for hardening')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation during hardening')
    parser.add_argument('--config', '-c', help='Configuration file for monitoring')
    parser.add_argument('--output', '-o', help='Output file for reports')
    
    args = parser.parse_args()
    
    print("🔒 SKZ Security Integration Manager")
    print("=" * 60)
    
    # Initialize security manager
    security_manager = SKZSecurityManager(args.project_root)
    
    try:
        # Run operations based on arguments
        if args.all or args.audit:
            print("\n🔍 Running Security Audit...")
            audit_report = security_manager.run_comprehensive_security_audit()
            print(f"✅ Security audit completed. Score: {audit_report['security_score']}/100")
        
        if args.all or args.test:
            print("\n🧪 Running Security Tests...")
            test_success = security_manager.run_security_tests()
            if test_success:
                print("✅ All security tests passed")
            else:
                print("❌ Some security tests failed")
        
        if args.all or args.harden:
            print("\n🛡️  Applying Security Hardening...")
            hardening_results = security_manager.apply_security_hardening(
                server_type=args.server_type,
                create_backup=not args.no_backup
            )
            print("✅ Security hardening completed")
            print(f"📝 Deployment instructions: {hardening_results['deployment_instructions']}")
        
        if args.monitor:
            print("\n👁️  Starting Security Monitoring...")
            security_manager.start_security_monitoring(args.config)
            print("✅ Security monitoring started")
            print("Press Ctrl+C to stop monitoring")
            
            try:
                while True:
                    time.sleep(60)
                    # Show periodic status
                    if security_manager.monitor:
                        summary = security_manager.monitor.get_event_summary(1)
                        if summary['total_events'] > 0:
                            print(f"📊 Events in last hour: {summary['total_events']}")
            except KeyboardInterrupt:
                print("\n🛑 Stopping security monitoring...")
                security_manager.stop_security_monitoring()
        
        if args.all or args.dashboard:
            print("\n📊 Generating Security Dashboard...")
            dashboard = security_manager.generate_security_dashboard()
            print(f"✅ Security dashboard generated: {security_manager.results_dir}/security_dashboard.json")
        
        if args.all or args.report:
            print("\n📄 Creating Security Report...")
            report_file = security_manager.create_security_report(args.output)
            print(f"✅ Security report created: {report_file}")
        
        # Show summary if multiple operations
        if args.all:
            print("\n🎯 Security Integration Summary:")
            print("   ✅ Security audit completed")
            print("   ✅ Security tests executed")
            print("   ✅ Security hardening applied")
            print("   ✅ Security dashboard generated")
            print("   ✅ Security report created")
            print("\n📚 Next steps:")
            print("   1. Review security report for recommendations")
            print("   2. Follow deployment instructions for hardening")
            print("   3. Start security monitoring for ongoing protection")
            print("   4. Schedule regular security audits")
    
    except KeyboardInterrupt:
        print("\n🛑 Operation interrupted by user")
        if security_manager.monitor:
            security_manager.stop_security_monitoring()
    
    except Exception as e:
        logger.error(f"Security operation failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    print("\n✅ Security integration completed successfully!")


if __name__ == "__main__":
    main()