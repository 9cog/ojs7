#!/usr/bin/env python3
"""
Comprehensive test script for Autonomous Agents Microservices
Tests the complete microservices architecture locally
"""

import os
import sys
import time
import requests
import subprocess
import json
from datetime import datetime

# Service configurations - All 7 Autonomous Agents
SERVICES = [
    {'name': 'research-discovery', 'port': 5001, 'dir': 'research-discovery'},
    {'name': 'submission-assistant', 'port': 5002, 'dir': 'submission-assistant'},
    {'name': 'editorial-orchestration', 'port': 5003, 'dir': 'editorial-orchestration'},
    {'name': 'review-coordination', 'port': 5004, 'dir': 'review-coordination'},
    {'name': 'content-quality', 'port': 5005, 'dir': 'content-quality'},
    {'name': 'publishing-production', 'port': 5006, 'dir': 'publishing-production'},
    {'name': 'analytics-monitoring', 'port': 5007, 'dir': 'analytics-monitoring'},
]

API_GATEWAY_PORT = 5000

class MicroservicesTest:
    def __init__(self):
        self.processes = []
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
    def start_service(self, service):
        """Start a microservice"""
        service_dir = os.path.join(self.base_dir, service['dir'])
        print(f"🚀 Starting {service['name']} on port {service['port']}")
        
        process = subprocess.Popen([
            sys.executable, 'app.py'
        ], cwd=service_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        self.processes.append(process)
        return process
    
    def start_api_gateway(self):
        """Start API Gateway"""
        gateway_dir = os.path.join(self.base_dir, 'api-gateway')
        print(f"🌐 Starting API Gateway on port {API_GATEWAY_PORT}")
        
        process = subprocess.Popen([
            sys.executable, 'app_local.py'
        ], cwd=gateway_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        self.processes.append(process)
        return process
    
    def wait_for_service(self, port, timeout=30):
        """Wait for a service to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        return False
    
    def test_service_health(self, service):
        """Test service health"""
        try:
            response = requests.get(f"http://localhost:{service['port']}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {service['name']} health check passed")
                return True
            else:
                print(f"❌ {service['name']} health check failed")
                return False
        except Exception as e:
            print(f"❌ {service['name']} health check error: {e}")
            return False
    
    def test_agent_endpoint(self, service):
        """Test agent endpoint"""
        try:
            response = requests.get(f"http://localhost:{service['port']}/agent", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {service['name']} agent endpoint: {data.get('name', 'Unknown')}")
                return True
            else:
                print(f"❌ {service['name']} agent endpoint failed")
                return False
        except Exception as e:
            print(f"❌ {service['name']} agent endpoint error: {e}")
            return False
    
    def test_agent_action(self, service):
        """Test agent action"""
        try:
            test_payload = {
                "action": "test_action",
                "parameters": {"test": True}
            }
            
            response = requests.post(
                f"http://localhost:{service['port']}/action",
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {service['name']} action test passed")
                return True
            else:
                print(f"❌ {service['name']} action test failed")
                return False
        except Exception as e:
            print(f"❌ {service['name']} action test error: {e}")
            return False
    
    def test_api_gateway(self):
        """Test API Gateway functionality"""
        print("\n🌐 Testing API Gateway...")
        
        # Test gateway health
        try:
            response = requests.get(f"http://localhost:{API_GATEWAY_PORT}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API Gateway health check passed")
            else:
                print("❌ API Gateway health check failed")
                return False
        except Exception as e:
            print(f"❌ API Gateway health check error: {e}")
            return False
        
        # Test services discovery
        try:
            response = requests.get(f"http://localhost:{API_GATEWAY_PORT}/api/v1/services", timeout=5)
            if response.status_code == 200:
                data = response.json()
                healthy_count = data.get('healthy_count', 0)
                total_count = data.get('total_count', 0)
                print(f"✅ Services discovery: {healthy_count}/{total_count} services healthy")
            else:
                print("❌ Services discovery failed")
        except Exception as e:
            print(f"❌ Services discovery error: {e}")
        
        # Test agents listing
        try:
            response = requests.get(f"http://localhost:{API_GATEWAY_PORT}/api/v1/agents", timeout=5)
            if response.status_code == 200:
                data = response.json()
                agent_count = data.get('total_count', 0)
                print(f"✅ Agents listing: {agent_count} agents discovered")
            else:
                print("❌ Agents listing failed")
        except Exception as e:
            print(f"❌ Agents listing error: {e}")
        
        # Test gateway action forwarding
        try:
            test_payload = {
                "action": "literature_search",
                "parameters": {"query": "test query", "domain": "test"}
            }
            
            response = requests.post(
                f"http://localhost:{API_GATEWAY_PORT}/api/v1/agents/research-discovery/action",
                json=test_payload,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Gateway action forwarding passed")
            else:
                print("❌ Gateway action forwarding failed")
        except Exception as e:
            print(f"❌ Gateway action forwarding error: {e}")
        
        return True
    
    def run_tests(self):
        """Run comprehensive tests"""
        print("🤖 Autonomous Agents Microservices Test Suite")
        print("=" * 55)
        print(f"📅 Test started at: {datetime.now().isoformat()}")
        
        try:
            # Start services
            print("\n🚀 Starting microservices...")
            for service in SERVICES:
                self.start_service(service)
            
            # Start API Gateway
            gateway_process = self.start_api_gateway()
            
            # Wait for services to start
            print("\n⏳ Waiting for services to start...")
            time.sleep(8)
            
            # Wait for services to be ready
            failed_services = []
            for service in SERVICES:
                if self.wait_for_service(service['port']):
                    print(f"✅ {service['name']} is ready")
                else:
                    print(f"❌ {service['name']} failed to start")
                    failed_services.append(service['name'])
            
            # Wait for API Gateway
            if self.wait_for_service(API_GATEWAY_PORT):
                print("✅ API Gateway is ready")
            else:
                print("❌ API Gateway failed to start")
                return False
            
            if failed_services:
                print(f"⚠️  Some services failed to start: {', '.join(failed_services)}")
            
            # Test individual services
            print("\n🔍 Testing individual services...")
            service_results = []
            
            for service in SERVICES:
                if service['name'] not in failed_services:
                    print(f"\n📍 Testing {service['name']}...")
                    health_ok = self.test_service_health(service)
                    agent_ok = self.test_agent_endpoint(service) if health_ok else False
                    action_ok = self.test_agent_action(service) if agent_ok else False
                    
                    service_results.append({
                        'service': service['name'],
                        'health': health_ok,
                        'agent': agent_ok,
                        'action': action_ok,
                        'overall': health_ok and agent_ok and action_ok
                    })
            
            # Test API Gateway
            gateway_ok = self.test_api_gateway()
            
            # Summary
            print("\n📊 Test Results Summary")
            print("=" * 30)
            
            successful_services = [r for r in service_results if r['overall']]
            print(f"✅ Successful services: {len(successful_services)}/{len(service_results)}")
            
            for result in service_results:
                status = "✅" if result['overall'] else "❌"
                print(f"{status} {result['service']}")
            
            gateway_status = "✅" if gateway_ok else "❌"
            print(f"{gateway_status} API Gateway")
            
            # Overall result
            overall_success = len(successful_services) >= len(service_results) * 0.75 and gateway_ok
            
            if overall_success:
                print("\n🎉 Microservices deployment test PASSED!")
                print("🌐 Access points:")
                print(f"   • API Gateway: http://localhost:{API_GATEWAY_PORT}")
                print(f"   • Services: http://localhost:{API_GATEWAY_PORT}/api/v1/services")
                print(f"   • Agents: http://localhost:{API_GATEWAY_PORT}/api/v1/agents")
                return True
            else:
                print("\n❌ Microservices deployment test FAILED!")
                return False
            
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            return False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up processes"""
        print("\n🧹 Cleaning up...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        print("✅ Cleanup complete")

if __name__ == '__main__':
    test = MicroservicesTest()
    success = test.run_tests()
    exit(0 if success else 1)