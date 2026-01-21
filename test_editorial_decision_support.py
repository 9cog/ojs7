#!/usr/bin/env python3
"""
Test script for Editorial Decision Support Systems
Validates the integration between OJS and SKZ decision engines
"""

import json
import requests
import time
import sys
import os
from datetime import datetime
import subprocess

def test_decision_engine_availability():
    """Test if the decision engine services are available"""
    print("Testing Decision Engine Availability...")
    
    services = [
        {'name': 'Editorial Decision Agent', 'url': 'http://localhost:8004/health'},
        {'name': 'Enhanced Decision Support', 'url': 'http://localhost:8005/health'}
    ]
    
    results = {}
    for service in services:
        try:
            response = requests.get(service['url'], timeout=5)
            if response.status_code == 200:
                print(f"✓ {service['name']}: Available")
                results[service['name']] = True
            else:
                print(f"✗ {service['name']}: HTTP {response.status_code}")
                results[service['name']] = False
        except requests.exceptions.RequestException as e:
            print(f"✗ {service['name']}: Connection failed - {e}")
            results[service['name']] = False
    
    return results

def test_decision_recommendation():
    """Test decision recommendation generation"""
    print("\nTesting Decision Recommendation Generation...")
    
    # Sample manuscript data
    test_data = {
        'submission_id': 'TEST-MS-001',
        'manuscript_data': {
            'title': 'Test Manuscript for Editorial Decision Support',
            'abstract': 'This is a test manuscript to validate the editorial decision support system integration.',
            'technical_quality': 7.8,
            'novelty_score': 8.2,
            'significance_score': 7.5,
            'clarity_score': 8.0,
            'completeness_score': 8.5,
            'ethical_compliance': 9.0,
            'statistical_rigor': 7.6,
            'literature_coverage': 8.1,
            'methodology_score': 7.9,
            'reproducibility_score': 7.4,
            'overall_score': 7.8
        },
        'reviews_data': [
            {
                'reviewer_id': 'REV-001',
                'recommendation': 'minor_revisions',
                'confidence': 0.85,
                'review_quality': 8.5,
                'technical_comments': ['Strong methodology', 'Clear presentation'],
                'major_issues': [],
                'minor_issues': ['Minor formatting issues', 'Some references need updates'],
                'completeness': 0.9,
                'timeliness': 0.8
            },
            {
                'reviewer_id': 'REV-002', 
                'recommendation': 'accept',
                'confidence': 0.9,
                'review_quality': 9.0,
                'technical_comments': ['Excellent contribution', 'Novel approach'],
                'major_issues': [],
                'minor_issues': ['Very minor typos'],
                'completeness': 0.95,
                'timeliness': 0.9
            }
        ],
        'context_data': {
            'journal_standards': {'quality_bar': 7.5},
            'acceptance_rate': 0.25,
            'workload': 15,
            'special_issue': False,
            'urgency': 'medium',
            'expertise_match': 0.9,
            'scope_alignment': 0.95
        }
    }
    
    try:
        # Test enhanced decision support
        response = requests.post(
            'http://localhost:8005/api/v1/decision/recommend',
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Decision recommendation generated successfully")
            
            if 'recommendation' in result:
                rec = result['recommendation']
                print(f"  Recommended Decision: {rec.get('recommended_decision', 'N/A')}")
                print(f"  Confidence: {rec.get('confidence', 0.0):.2%}")
                print(f"  Processing Time: {result.get('processing_time', 0.0):.2f}s")
                
                if 'reasoning' in rec:
                    print(f"  Reasoning: {', '.join(rec['reasoning'][:3])}")
                
                return True
            else:
                print("✗ No recommendation in response")
                return False
        else:
            print(f"✗ HTTP {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return False

def test_decision_analytics():
    """Test decision analytics and statistics"""
    print("\nTesting Decision Analytics...")
    
    try:
        # Test statistics endpoint
        response = requests.get('http://localhost:8005/api/v1/decision/statistics', timeout=10)
        
        if response.status_code == 200:
            stats = response.json()
            print("✓ Decision statistics retrieved successfully")
            
            if 'statistics' in stats:
                s = stats['statistics']
                print(f"  Total Decisions: {s.get('total_decisions', 0)}")
                print(f"  Average Confidence: {s.get('average_confidence', 0.0):.2%}")
                print(f"  Accept Rate: {s.get('accept_rate', 0.0):.1f}%")
                print(f"  Revision Rate: {s.get('revision_rate', 0.0):.1f}%")
                
                return True
            else:
                print("✗ No statistics in response")
                return False
        else:
            print(f"✗ HTTP {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return False

def test_ojs_plugin_files():
    """Test if OJS plugin files are properly installed"""
    print("\nTesting OJS Plugin Files...")
    
    plugin_path = '/home/runner/work/oj7/oj7/plugins/generic/skzEditorialDecisionSupport'
    required_files = [
        'SkzEditorialDecisionSupportPlugin.inc.php',
        'index.php',
        'version.xml',
        'js/SkzDecisionSupport.js',
        'css/skz-decision-support.css'
    ]
    
    all_present = True
    for file_path in required_files:
        full_path = os.path.join(plugin_path, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_present = False
    
    return all_present

def test_dashboard_component():
    """Test if dashboard component is properly integrated"""
    print("\nTesting Dashboard Component...")
    
    component_path = '/home/runner/work/oj7/oj7/skz-integration/workflow-visualization-dashboard/src/components/EditorialDecisionAnalytics.jsx'
    
    if os.path.exists(component_path):
        print("✓ EditorialDecisionAnalytics.jsx component present")
        
        # Check if component contains required functionality
        with open(component_path, 'r') as f:
            content = f.read()
            
        required_features = [
            'EditorialDecisionAnalytics',
            'fetchDecisionAnalytics',
            'decisionStats',
            'performanceMetrics',
            'Brain'
        ]
        
        features_present = True
        for feature in required_features:
            if feature in content:
                print(f"✓ Feature: {feature}")
            else:
                print(f"✗ Feature: {feature} (missing)")
                features_present = False
        
        return features_present
    else:
        print("✗ EditorialDecisionAnalytics.jsx component missing")
        return False

def test_database_integration():
    """Test database integration for decision audit"""
    print("\nTesting Database Integration...")
    
    try:
        import sqlite3
        
        # Check if audit database can be created/accessed
        db_path = '/tmp/test_editorial_decisions.db'
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Test table creation
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_decisions (
                id INTEGER PRIMARY KEY,
                submission_id TEXT,
                decision_type TEXT,
                timestamp TEXT
            )
        ''')
        
        # Test data insertion
        cursor.execute('''
            INSERT INTO test_decisions (submission_id, decision_type, timestamp)
            VALUES (?, ?, ?)
        ''', ('TEST-001', 'accept', datetime.now().isoformat()))
        
        # Test data retrieval
        cursor.execute('SELECT COUNT(*) FROM test_decisions')
        count = cursor.fetchone()[0]
        
        conn.commit()
        conn.close()
        
        # Clean up test database
        os.remove(db_path)
        
        print("✓ Database operations successful")
        print(f"  Test records: {count}")
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("=" * 60)
    print("EDITORIAL DECISION SUPPORT SYSTEM - INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("Service Availability", test_decision_engine_availability),
        ("Decision Recommendation", test_decision_recommendation), 
        ("Decision Analytics", test_decision_analytics),
        ("OJS Plugin Files", test_ojs_plugin_files),
        ("Dashboard Component", test_dashboard_component),
        ("Database Integration", test_database_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"TEST: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        icon = "✓" if result else "✗"
        print(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 All tests passed! Editorial Decision Support System is ready.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the implementation.")
        return False

def start_test_services():
    """Start required services for testing"""
    print("Starting test services...")
    
    try:
        # Start enhanced decision support service
        subprocess.Popen([
            'python3', 
            '/home/runner/work/oj7/oj7/skz-integration/enhanced_decision_support.py',
            '--port', '8005'
        ])
        
        print("Services started. Waiting for initialization...")
        time.sleep(3)
        return True
        
    except Exception as e:
        print(f"Failed to start services: {e}")
        return False

def main():
    """Main test execution"""
    if len(sys.argv) > 1 and sys.argv[1] == '--start-services':
        if not start_test_services():
            sys.exit(1)
        time.sleep(2)
    
    success = run_comprehensive_test()
    
    if success:
        print("\n✅ Editorial Decision Support System integration complete!")
    else:
        print("\n❌ Integration issues detected. Please review failed tests.")
        sys.exit(1)

if __name__ == '__main__':
    main()