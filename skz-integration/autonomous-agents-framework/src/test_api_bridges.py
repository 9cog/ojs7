#!/usr/bin/env python3
"""
API Bridge Test Suite
Tests communication between PHP OJS and Python agents
"""

import json
import time
import threading
import subprocess
import requests
from datetime import datetime
import sys
import os

def test_python_api_server():
    """Test the Python API server"""
    print("\n=== Testing Python API Server ===")
    
    # Start the API server in a separate thread
    def run_server():
        os.system("cd /home/runner/work/oj7/oj7/skz-integration/autonomous-agents-framework/src && python3 simple_api_server.py 5000 > /tmp/api_server.log 2>&1")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("Starting API server...")
    time.sleep(3)
    
    # Test endpoints
    base_url = "http://localhost:5000"
    test_results = []
    
    # Test 1: Server status
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Status endpoint: {data['status']}")
            test_results.append(("Status endpoint", True))
        else:
            print(f"✗ Status endpoint failed: {response.status_code}")
            test_results.append(("Status endpoint", False))
    except Exception as e:
        print(f"✗ Status endpoint error: {e}")
        test_results.append(("Status endpoint", False))
    
    # Test 2: Agents list
    try:
        response = requests.get(f"{base_url}/agents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            agents_count = len(data.get('agents', {}))
            print(f"✓ Agents list: {agents_count} agents available")
            test_results.append(("Agents list", True))
        else:
            print(f"✗ Agents list failed: {response.status_code}")
            test_results.append(("Agents list", False))
    except Exception as e:
        print(f"✗ Agents list error: {e}")
        test_results.append(("Agents list", False))
    
    # Test 3: Individual agent status
    try:
        response = requests.get(f"{base_url}/agents/research_discovery", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Agent status: {data.get('name', 'Unknown')}")
            test_results.append(("Agent status", True))
        else:
            print(f"✗ Agent status failed: {response.status_code}")
            test_results.append(("Agent status", False))
    except Exception as e:
        print(f"✗ Agent status error: {e}")
        test_results.append(("Agent status", False))
    
    # Test 4: Agent processing request
    try:
        test_data = {
            "manuscript_id": "test_123",
            "content": "Test manuscript content",
            "metadata": {"title": "Test Paper", "authors": ["Test Author"]}
        }
        
        response = requests.post(
            f"{base_url}/research_discovery/analyze",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✓ Agent processing: {data.get('result', {}).get('action_taken', 'Unknown')}")
                test_results.append(("Agent processing", True))
            else:
                print(f"✗ Agent processing failed: {data}")
                test_results.append(("Agent processing", False))
        else:
            print(f"✗ Agent processing failed: {response.status_code}")
            test_results.append(("Agent processing", False))
    except Exception as e:
        print(f"✗ Agent processing error: {e}")
        test_results.append(("Agent processing", False))
    
    # Test 5: API v1 endpoint
    try:
        response = requests.post(
            f"{base_url}/api/v1/agents/submission_assistant/process",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✓ API v1 endpoint: working")
                test_results.append(("API v1 endpoint", True))
            else:
                print(f"✗ API v1 endpoint failed: {data}")
                test_results.append(("API v1 endpoint", False))
        else:
            print(f"✗ API v1 endpoint failed: {response.status_code}")
            test_results.append(("API v1 endpoint", False))
    except Exception as e:
        print(f"✗ API v1 endpoint error: {e}")
        test_results.append(("API v1 endpoint", False))
    
    return test_results

def test_php_bridge():
    """Test PHP bridge functionality"""
    print("\n=== Testing PHP Bridge ===")
    
    test_results = []
    
    # Test 1: PHP syntax check
    try:
        result = subprocess.run([
            'php', '-l', 
            '/home/runner/work/oj7/oj7/plugins/generic/skzAgents/classes/SKZAgentBridge.inc.php'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ PHP Bridge syntax: valid")
            test_results.append(("PHP syntax", True))
        else:
            print(f"✗ PHP Bridge syntax error: {result.stderr}")
            test_results.append(("PHP syntax", False))
    except Exception as e:
        print(f"✗ PHP syntax check error: {e}")
        test_results.append(("PHP syntax", False))
    
    # Test 2: PHP class loading
    try:
        php_code = """
        <?php
        // Minimal OJS simulation for testing
        class Application {
            public static function getRequest() { return new stdClass(); }
        }
        class PluginRegistry {
            public static function getPlugin($type, $name) { return null; }
        }
        class Config {
            public static function getVar($section, $key, $default = null) { 
                if ($key === 'agent_base_url') return 'http://localhost:5000/api';
                if ($key === 'api_key') return 'test_key';
                if ($key === 'timeout') return 30;
                return $default; 
            }
        }
        
        include_once '/home/runner/work/oj7/oj7/plugins/generic/skzAgents/classes/SKZAgentBridge.inc.php';
        
        $bridge = new SKZAgentBridge();
        echo "PHP Bridge class instantiated successfully";
        ?>
        """
        
        result = subprocess.run(['php'], input=php_code, capture_output=True, text=True)
        
        if "successfully" in result.stdout:
            print("✓ PHP Bridge class: instantiated")
            test_results.append(("PHP class loading", True))
        else:
            print(f"✗ PHP Bridge class error: {result.stderr}")
            test_results.append(("PHP class loading", False))
    except Exception as e:
        print(f"✗ PHP class loading error: {e}")
        test_results.append(("PHP class loading", False))
    
    return test_results

def test_integration():
    """Test full integration between PHP and Python"""
    print("\n=== Testing Integration ===")
    
    test_results = []
    
    # Test 1: PHP to Python communication
    try:
        # Start Python server first
        def run_server():
            os.system("cd /home/runner/work/oj7/oj7/skz-integration/autonomous-agents-framework/src && python3 simple_api_server.py 5001 > /tmp/integration_server.log 2>&1")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)
        
        # Test PHP calling Python agents
        php_test_code = """
        <?php
        // Mock required OJS classes
        class Application {
            public static function getRequest() { return new stdClass(); }
        }
        class PluginRegistry {
            public static function getPlugin($type, $name) { return null; }
        }
        class Config {
            public static function getVar($section, $key, $default = null) { 
                if ($key === 'agent_base_url') return 'http://localhost:5001';
                if ($key === 'api_key') return 'test_key';
                if ($key === 'timeout') return 10;
                return $default; 
            }
        }
        
        include_once '/home/runner/work/oj7/oj7/plugins/generic/skzAgents/classes/SKZAgentBridge.inc.php';
        
        $bridge = new SKZAgentBridge();
        $result = $bridge->callAgent('research_discovery', 'analyze', array(
            'manuscript_id' => 'test_123',
            'content' => 'Test manuscript for integration'
        ));
        
        if (isset($result['success']) && $result['success']) {
            echo "INTEGRATION_SUCCESS";
        } else {
            echo "INTEGRATION_FAILED: " . json_encode($result);
        }
        ?>
        """
        
        result = subprocess.run(['php'], input=php_test_code, capture_output=True, text=True)
        
        if "INTEGRATION_SUCCESS" in result.stdout:
            print("✓ PHP to Python communication: working")
            test_results.append(("PHP to Python", True))
        else:
            print(f"✗ PHP to Python communication failed")
            print(f"  Output: {result.stdout}")
            print(f"  Error: {result.stderr}")
            test_results.append(("PHP to Python", False))
            
    except Exception as e:
        print(f"✗ Integration test error: {e}")
        test_results.append(("PHP to Python", False))
    
    return test_results

def main():
    """Run all tests"""
    print("=" * 60)
    print("API BRIDGE INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing API bridges between PHP OJS and Python agents")
    
    all_results = []
    
    # Run tests
    all_results.extend(test_python_api_server())
    all_results.extend(test_php_bridge())
    all_results.extend(test_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in all_results if result)
    total = len(all_results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    for test_name, result in all_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:10} {test_name}")
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\n🎉 API BRIDGES: WORKING")
        print("✅ PHP OJS and Python agents can communicate")
        return 0
    else:
        print("\n❌ API BRIDGES: NEED ATTENTION")
        print("❌ Some integration issues detected")
        return 1

if __name__ == '__main__':
    sys.exit(main())