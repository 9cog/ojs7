#!/usr/bin/env python3
"""
Feature & Documentation Verification Audit System
Comprehensive audit tool for autonomous academic publishing platform
"""

import os
import re
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
from datetime import datetime

class FeatureAuditSystem:
    """
    Recursive cognitive tracking system for feature development roadmap generation
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.features = defaultdict(dict)
        self.documentation_files = []
        self.code_files = []
        self.test_files = []
        self.config_files = []
        
        # Supported file extensions for different categories
        self.doc_extensions = {'.md', '.txt', '.rst', '.doc', '.docx'}
        self.code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.php', '.go', '.rs', '.java'}
        self.test_extensions = {'.test.py', '.test.js', '.spec.py', '.spec.js', '_test.py', '_spec.py'}
        self.config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
        
        # Feature pattern matchers
        self.feature_patterns = {
            'function': re.compile(r'def\s+(\w+)\s*\(|function\s+(\w+)\s*\(|class\s+(\w+)'),
            'endpoint': re.compile(r'@app\.route\s*\(\s*[\'"]([^\'"]+)[\'"]|router\.\w+\s*\(\s*[\'"]([^\'"]+)[\'"]'),
            'component': re.compile(r'(?:class|function|const)\s+(\w+)(?:Component|Widget|View|Page)'),
            'agent': re.compile(r'(?:class|def)\s+(\w+)(?:Agent|Bot|Assistant)'),
            'workflow': re.compile(r'(?:workflow|process|pipeline|stage)[\s_-]*(\w+)', re.IGNORECASE),
            'api': re.compile(r'(?:GET|POST|PUT|DELETE|PATCH)\s+([/\w\-\{\}]+)', re.IGNORECASE),
            'database': re.compile(r'(?:CREATE TABLE|class\s+\w+.*Model|db\.Model).*?(\w+)', re.IGNORECASE),
        }
        
        # Documentation feature patterns
        self.doc_feature_patterns = {
            'heading': re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE),
            'bullet': re.compile(r'^\s*[-*+]\s+(.+)$', re.MULTILINE),
            'numbered': re.compile(r'^\s*\d+\.\s+(.+)$', re.MULTILINE),
            'todo': re.compile(r'(?:TODO|FIXME|BUG|FEATURE):\s*(.+)', re.IGNORECASE),
            'mention': re.compile(r'`([^`]+)`|"([^"]+)"|\'([^\']+)\''),
        }

    def step1_enumerate_documents(self) -> List[str]:
        """Step 1: Enumerate all documentation and code files"""
        print("🔍 Step 1: Enumerating documents and code files...")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip common ignore directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'dist', 'build'}]
            
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()
                
                if ext in self.doc_extensions:
                    self.documentation_files.append(str(file_path))
                elif ext in self.code_extensions or any(pattern in file for pattern in self.test_extensions):
                    if any(pattern in file for pattern in self.test_extensions):
                        self.test_files.append(str(file_path))
                    else:
                        self.code_files.append(str(file_path))
                elif ext in self.config_extensions:
                    self.config_files.append(str(file_path))
        
        print(f"   📄 Documentation files: {len(self.documentation_files)}")
        print(f"   💻 Code files: {len(self.code_files)}")
        print(f"   🧪 Test files: {len(self.test_files)}")
        print(f"   ⚙️  Config files: {len(self.config_files)}")
        
        return self.documentation_files + self.code_files
    
    def step2_extract_features(self) -> Dict[str, Set[str]]:
        """Step 2: Extract features from documentation and code"""
        print("\n🔬 Step 2: Extracting features from all files...")
        
        doc_features = set()
        code_features = set()
        
        # Extract from documentation
        for doc_file in self.documentation_files:
            try:
                with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    features = self._extract_doc_features(content)
                    doc_features.update(features)
                    for feature in features:
                        self.features[feature]['documented'] = True
                        self.features[feature]['doc_files'] = self.features[feature].get('doc_files', [])
                        self.features[feature]['doc_files'].append(doc_file)
            except Exception as e:
                print(f"   ⚠️  Error reading {doc_file}: {e}")
        
        # Extract from code
        for code_file in self.code_files:
            try:
                with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    features = self._extract_code_features(content, code_file)
                    code_features.update(features)
                    for feature in features:
                        self.features[feature]['implemented'] = True
                        self.features[feature]['code_files'] = self.features[feature].get('code_files', [])
                        self.features[feature]['code_files'].append(code_file)
            except Exception as e:
                print(f"   ⚠️  Error reading {code_file}: {e}")
        
        print(f"   📝 Documentation features: {len(doc_features)}")
        print(f"   🛠️  Code features: {len(code_features)}")
        print(f"   🔗 Total unique features: {len(self.features)}")
        
        return {'documentation': doc_features, 'code': code_features}
    
    def _extract_doc_features(self, content: str) -> Set[str]:
        """Extract features from documentation content"""
        features = set()
        
        for pattern_name, pattern in self.doc_feature_patterns.items():
            matches = pattern.findall(content)
            for match in matches:
                if isinstance(match, tuple):
                    match = next(m for m in match if m)
                
                # Clean and normalize feature names
                feature = self._normalize_feature_name(match.strip())
                if feature and len(feature) > 2:
                    features.add(feature)
        
        return features
    
    def _extract_code_features(self, content: str, file_path: str) -> Set[str]:
        """Extract features from code content"""
        features = set()
        
        for pattern_name, pattern in self.feature_patterns.items():
            matches = pattern.findall(content)
            for match in matches:
                if isinstance(match, tuple):
                    match = next(m for m in match if m)
                
                feature = self._normalize_feature_name(match.strip())
                if feature and len(feature) > 2:
                    features.add(feature)
        
        return features
    
    def _normalize_feature_name(self, name: str) -> str:
        """Normalize feature names for consistent matching"""
        # Remove special characters, convert to lowercase
        name = re.sub(r'[^\w\s-]', '', name).strip().lower()
        # Replace multiple spaces with single space
        name = re.sub(r'\s+', ' ', name)
        return name
    
    def step3_verify_implementation_status(self) -> Dict[str, Dict[str, bool]]:
        """Step 3: Verify implementation status of features"""
        print("\n✅ Step 3: Verifying implementation status...")
        
        status_report = {}
        
        for feature, data in self.features.items():
            status_report[feature] = {
                'documented': data.get('documented', False),
                'implemented': data.get('implemented', False),
                'tested': self._check_if_tested(feature),
                'deployed': self._check_if_deployed(feature)
            }
        
        # Calculate summary statistics
        total_features = len(status_report)
        documented = sum(1 for s in status_report.values() if s['documented'])
        implemented = sum(1 for s in status_report.values() if s['implemented'])
        tested = sum(1 for s in status_report.values() if s['tested'])
        deployed = sum(1 for s in status_report.values() if s['deployed'])
        
        print(f"   📊 Status Summary:")
        print(f"      Total Features: {total_features}")
        print(f"      Documented: {documented} ({documented/total_features*100:.1f}%)")
        print(f"      Implemented: {implemented} ({implemented/total_features*100:.1f}%)")
        print(f"      Tested: {tested} ({tested/total_features*100:.1f}%)")
        print(f"      Deployed: {deployed} ({deployed/total_features*100:.1f}%)")
        
        return status_report
    
    def _check_if_tested(self, feature: str) -> bool:
        """Check if a feature has associated tests"""
        feature_lower = feature.lower()
        
        for test_file in self.test_files:
            try:
                with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    if feature_lower in content:
                        return True
            except:
                continue
        
        return False
    
    def _check_if_deployed(self, feature: str) -> bool:
        """Check if a feature is deployed (based on config files or deployment scripts)"""
        feature_lower = feature.lower()
        
        # Check in config files
        for config_file in self.config_files:
            try:
                with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    if feature_lower in content:
                        return True
            except:
                continue
        
        # Check for deployment indicators
        deployment_indicators = ['dockerfile', 'docker-compose', 'deploy', 'prod', 'production']
        for indicator in deployment_indicators:
            if any(indicator in str(f).lower() for f in Path(self.project_root).rglob('*')):
                return True
        
        return False
    
    def step4_execute_tests(self) -> Dict[str, Any]:
        """Step 4: Execute existing tests and collect results"""
        print("\n🧪 Step 4: Executing tests...")
        
        test_results = {
            'total_tests': len(self.test_files),
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        if not self.test_files:
            print("   ℹ️  No test files found")
            return test_results
        
        # Try to run Python tests
        for test_file in self.test_files:
            if test_file.endswith('.py'):
                try:
                    result = subprocess.run(
                        ['python', '-m', 'pytest', test_file, '-v'],
                        capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0:
                        test_results['passed'] += 1
                    else:
                        test_results['failed'] += 1
                        test_results['errors'].append(f"{test_file}: {result.stderr}")
                except subprocess.TimeoutExpired:
                    test_results['failed'] += 1
                    test_results['errors'].append(f"{test_file}: Test timeout")
                except Exception as e:
                    test_results['failed'] += 1
                    test_results['errors'].append(f"{test_file}: {str(e)}")
        
        print(f"   ✅ Passed: {test_results['passed']}")
        print(f"   ❌ Failed: {test_results['failed']}")
        print(f"   ⏭️  Skipped: {test_results['skipped']}")
        
        return test_results
    
    def step5_generate_completion_matrix(self, status_report: Dict[str, Dict[str, bool]]) -> str:
        """Step 5: Generate completion matrix table"""
        print("\n📊 Step 5: Generating completion matrix...")
        
        matrix = []
        matrix.append("| Feature | Documented | Implemented | Tested | Deployed | Completion |")
        matrix.append("|---------|------------|-------------|--------|----------|------------|")
        
        for feature, status in sorted(status_report.items()):
            doc_status = "✅" if status['documented'] else "❌"
            impl_status = "✅" if status['implemented'] else "❌"
            test_status = "✅" if status['tested'] else "❌"
            deploy_status = "✅" if status['deployed'] else "❌"
            
            completion = sum(status.values()) / 4 * 100
            completion_str = f"{completion:.0f}%"
            
            matrix.append(f"| {feature[:30]} | {doc_status} | {impl_status} | {test_status} | {deploy_status} | {completion_str} |")
        
        matrix_table = "\n".join(matrix)
        print(f"   📋 Generated matrix with {len(status_report)} features")
        
        return matrix_table
    
    def step6_synthesize_roadmap(self, status_report: Dict[str, Dict[str, bool]]) -> List[Dict[str, Any]]:
        """Step 6: Synthesize development roadmap with actionable items"""
        print("\n🗺️  Step 6: Synthesizing development roadmap...")
        
        roadmap_tasks = []
        
        for feature, status in status_report.items():
            tasks = []
            
            if not status['documented']:
                tasks.append({
                    'type': 'documentation',
                    'priority': 'medium',
                    'description': f"Document feature: {feature}",
                    'estimated_hours': 2
                })
            
            if not status['implemented']:
                tasks.append({
                    'type': 'implementation',
                    'priority': 'high',
                    'description': f"Implement feature: {feature}",
                    'estimated_hours': 8
                })
            
            if not status['tested']:
                tasks.append({
                    'type': 'testing',
                    'priority': 'high',
                    'description': f"Add tests for feature: {feature}",
                    'estimated_hours': 4
                })
            
            if not status['deployed']:
                tasks.append({
                    'type': 'deployment',
                    'priority': 'medium',
                    'description': f"Deploy feature: {feature}",
                    'estimated_hours': 2
                })
            
            roadmap_tasks.extend(tasks)
        
        # Sort by priority and type
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        roadmap_tasks.sort(key=lambda x: (priority_order[x['priority']], x['type']))
        
        print(f"   📝 Generated {len(roadmap_tasks)} actionable tasks")
        
        return roadmap_tasks
    
    def step7_assign_agents(self, roadmap_tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Step 7: Assign tasks to autonomous agents"""
        print("\n🤖 Step 7: Assigning tasks to autonomous agents...")
        
        agent_assignments = {
            'documentation_lead_agent': [],
            'system_integration_agent': [],
            'project_architect_agent': [],
            'analytics_monitoring_agent': []
        }
        
        for task in roadmap_tasks:
            if task['type'] == 'documentation':
                agent_assignments['documentation_lead_agent'].append(task)
            elif task['type'] in ['implementation', 'testing']:
                agent_assignments['system_integration_agent'].append(task)
            elif task['type'] == 'deployment':
                agent_assignments['project_architect_agent'].append(task)
            else:
                agent_assignments['analytics_monitoring_agent'].append(task)
        
        for agent, tasks in agent_assignments.items():
            print(f"   🔧 {agent}: {len(tasks)} tasks")
        
        return agent_assignments
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        print("\n📋 Generating comprehensive audit report...")
        
        # Execute all steps
        files = self.step1_enumerate_documents()
        features = self.step2_extract_features()
        status_report = self.step3_verify_implementation_status()
        test_results = self.step4_execute_tests()
        completion_matrix = self.step5_generate_completion_matrix(status_report)
        roadmap_tasks = self.step6_synthesize_roadmap(status_report)
        agent_assignments = self.step7_assign_agents(roadmap_tasks)
        
        # Compile comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'summary': {
                'total_files': len(files),
                'documentation_files': len(self.documentation_files),
                'code_files': len(self.code_files),
                'test_files': len(self.test_files),
                'total_features': len(self.features),
                'completion_rate': sum(sum(s.values()) for s in status_report.values()) / (len(status_report) * 4) * 100 if status_report else 0
            },
            'features': dict(self.features),
            'status_report': status_report,
            'completion_matrix': completion_matrix,
            'test_results': test_results,
            'roadmap_tasks': roadmap_tasks,
            'agent_assignments': agent_assignments
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: str = "audit_report.json") -> None:
        """Save audit report to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"📄 Report saved to {output_file}")
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print summary of audit results"""
        print("\n" + "="*80)
        print("📊 AUDIT SUMMARY")
        print("="*80)
        
        summary = report['summary']
        print(f"📁 Total Files Analyzed: {summary['total_files']}")
        print(f"📄 Documentation Files: {summary['documentation_files']}")
        print(f"💻 Code Files: {summary['code_files']}")
        print(f"🧪 Test Files: {summary['test_files']}")
        print(f"🔍 Features Identified: {summary['total_features']}")
        print(f"📈 Overall Completion: {summary['completion_rate']:.1f}%")
        
        print(f"\n🗺️  Roadmap Tasks: {len(report['roadmap_tasks'])}")
        for agent, tasks in report['agent_assignments'].items():
            print(f"   🤖 {agent.replace('_', ' ').title()}: {len(tasks)} tasks")


def main():
    """Main entry point for the audit system"""
    parser = argparse.ArgumentParser(description="Feature & Documentation Verification Audit System")
    parser.add_argument('--project-root', '-p', default='.', help='Project root directory')
    parser.add_argument('--output', '-o', default='audit_report.json', help='Output report file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode - minimal output')
    parser.add_argument('--matrix-only', '-m', action='store_true', help='Only generate completion matrix')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🚀 Feature & Documentation Verification Audit System")
        print("=" * 60)
    
    # Initialize audit system
    audit_system = FeatureAuditSystem(args.project_root)
    
    if args.matrix_only:
        # Quick matrix generation
        audit_system.step1_enumerate_documents()
        audit_system.step2_extract_features()
        status_report = audit_system.step3_verify_implementation_status()
        matrix = audit_system.step5_generate_completion_matrix(status_report)
        print("\n" + matrix)
    else:
        # Full audit
        report = audit_system.generate_comprehensive_report()
        audit_system.save_report(report, args.output)
        
        if not args.quiet:
            audit_system.print_summary(report)
    
    if not args.quiet:
        print("\n✅ Audit completed successfully!")


if __name__ == "__main__":
    main()