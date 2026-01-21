#!/usr/bin/env python3
"""
Autonomous Agents Microservices Demo
Demonstrates the functionality of the deployed microservices architecture
"""

import requests
import json
import time
from datetime import datetime

API_GATEWAY = "http://localhost:5000"

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🤖 {title}")
    print(f"{'='*60}")

def print_json(data, title="Response"):
    print(f"\n📄 {title}:")
    print(json.dumps(data, indent=2))

def demo_gateway_health():
    """Demo: Gateway Health Check"""
    print_header("Gateway Health Check")
    
    try:
        response = requests.get(f"{API_GATEWAY}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ API Gateway is healthy!")
            print_json(data)
        else:
            print("❌ Gateway health check failed")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_service_discovery():
    """Demo: Service Discovery"""
    print_header("Service Discovery")
    
    try:
        response = requests.get(f"{API_GATEWAY}/api/v1/services", timeout=5)
        if response.status_code == 200:
            data = response.json()
            healthy_count = data.get('healthy_count', 0)
            total_count = data.get('total_count', 0)
            
            print(f"🔍 Services Status: {healthy_count}/{total_count} healthy")
            
            for service in data.get('services', []):
                status_icon = "✅" if service['status'] == 'healthy' else "❌"
                print(f"{status_icon} {service['name']} - {service['status']}")
        else:
            print("❌ Service discovery failed")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_agents_listing():
    """Demo: List All Agents"""
    print_header("Agents Discovery")
    
    try:
        response = requests.get(f"{API_GATEWAY}/api/v1/agents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            agent_count = data.get('total_count', 0)
            
            print(f"🤖 Found {agent_count} active agents:")
            
            for agent in data.get('agents', []):
                print(f"\n🔹 {agent.get('name', 'Unknown Agent')}")
                print(f"   Type: {agent.get('type', 'unknown')}")
                print(f"   Service: {agent.get('service', 'unknown')}")
                print(f"   Capabilities: {', '.join(agent.get('capabilities', []))}")
                
                performance = agent.get('performance', {})
                if performance:
                    print(f"   Success Rate: {performance.get('success_rate', 0)*100:.1f}%")
                    print(f"   Avg Response: {performance.get('avg_response_time', 0):.1f}s")
        else:
            print("❌ Agents listing failed")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_research_discovery():
    """Demo: Research Discovery Agent Action"""
    print_header("Research Discovery Agent Demo")
    
    try:
        payload = {
            "action": "literature_search",
            "parameters": {
                "query": "autonomous agents in academic publishing",
                "domain": "computer_science"
            }
        }
        
        print("📤 Triggering literature search...")
        print_json(payload, "Request")
        
        response = requests.post(
            f"{API_GATEWAY}/api/v1/agents/research-discovery/action",
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Research Discovery completed!")
            
            result = data.get('result', {})
            print(f"\n📊 Search Results:")
            print(f"   Papers Found: {result.get('papers_found', 0)}")
            print(f"   Relevant Papers: {result.get('relevant_papers', 0)}")
            print(f"   Quality Score: {result.get('search_quality_score', 0):.2f}")
            print(f"   Processing Time: {data.get('processing_time', 0):.2f}s")
            
            if 'recommendations' in result:
                print("\n💡 Recommendations:")
                for rec in result['recommendations']:
                    print(f"   • {rec}")
        else:
            print("❌ Research Discovery failed")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_submission_assistant():
    """Demo: Submission Assistant Agent Action"""
    print_header("Submission Assistant Agent Demo")
    
    try:
        payload = {
            "action": "venue_recommendation",
            "parameters": {
                "field": "artificial_intelligence",
                "impact_preference": "high"
            }
        }
        
        print("📤 Requesting venue recommendations...")
        print_json(payload, "Request")
        
        response = requests.post(
            f"{API_GATEWAY}/api/v1/agents/submission-assistant/action",
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Venue recommendations completed!")
            
            result = data.get('result', {})
            
            if 'recommended_venues' in result:
                print("\n🎯 Recommended Venues:")
                for venue in result['recommended_venues']:
                    print(f"   • {venue.get('name', 'Unknown')} (Match: {venue.get('match_score', 0):.2f})")
            
            print(f"\n📊 Success Probability: {result.get('success_probability', 0):.2f}")
            print(f"Processing Time: {data.get('processing_time', 0):.2f}s")
        else:
            print("❌ Submission Assistant failed")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_content_quality():
    """Demo: Content Quality Agent Action"""
    print_header("Content Quality Agent Demo")
    
    try:
        payload = {
            "action": "quality_assessment",
            "parameters": {
                "text": "Sample manuscript content for quality assessment"
            }
        }
        
        print("📤 Requesting quality assessment...")
        
        response = requests.post(
            f"{API_GATEWAY}/api/v1/agents/content-quality/action",
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Quality assessment completed!")
            
            result = data.get('result', {})
            
            print(f"\n📊 Quality Scores:")
            print(f"   Overall Quality: {result.get('quality_score', 0):.1f}/10")
            print(f"   Novelty: {result.get('novelty_score', 0):.1f}/10")
            print(f"   Clarity: {result.get('clarity_score', 0):.1f}/10")
            print(f"   Significance: {result.get('significance_score', 0):.1f}/10")
            
            if 'improvement_suggestions' in result:
                print("\n💡 Improvement Suggestions:")
                for suggestion in result['improvement_suggestions'][:3]:
                    print(f"   • {suggestion}")
        else:
            print("❌ Content Quality assessment failed")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_analytics_monitoring():
    """Demo: Analytics Monitoring Agent Action"""
    print_header("Analytics Monitoring Agent Demo")
    
    try:
        payload = {
            "action": "system_status",
            "parameters": {}
        }
        
        print("📤 Requesting system analytics...")
        
        response = requests.post(
            f"{API_GATEWAY}/api/v1/agents/analytics-monitoring/action",
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ System analytics completed!")
            
            result = data.get('result', {})
            
            print(f"\n📊 System Health: {result.get('system_health', 'unknown')}")
            
            metrics = result.get('performance_metrics', {})
            if metrics:
                print(f"\n📈 Performance Metrics:")
                for key, value in metrics.items():
                    print(f"   {key.replace('_', ' ').title()}: {value}")
            
            if 'recommendations' in result:
                print("\n💡 System Recommendations:")
                for rec in result['recommendations'][:3]:
                    print(f"   • {rec}")
        else:
            print("❌ Analytics monitoring failed")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run the complete demo"""
    print_header("Autonomous Agents Microservices Demo")
    print("This demo showcases the functionality of the deployed microservices")
    print(f"Started at: {datetime.now().isoformat()}")
    
    # Check if services are running
    try:
        response = requests.get(f"{API_GATEWAY}/health", timeout=5)
        if response.status_code != 200:
            print("\n❌ API Gateway is not running!")
            print("Please start the microservices first:")
            print("   python3 test_microservices.py")
            print("   # or")
            print("   ./deploy.sh")
            return
    except:
        print("\n❌ Cannot connect to API Gateway!")
        print("Please start the microservices first:")
        print("   python3 test_microservices.py")
        return
    
    # Run demos
    demo_gateway_health()
    time.sleep(1)
    
    demo_service_discovery()
    time.sleep(1)
    
    demo_agents_listing()
    time.sleep(1)
    
    demo_research_discovery()
    time.sleep(1)
    
    demo_submission_assistant()
    time.sleep(1)
    
    demo_content_quality()
    time.sleep(1)
    
    demo_analytics_monitoring()
    
    print_header("Demo Complete")
    print("🎉 All microservices demonstrated successfully!")
    print(f"\n🌐 Access the API Gateway at: {API_GATEWAY}")
    print("   • Services Status: /api/v1/services")
    print("   • All Agents: /api/v1/agents")
    print("   • Dashboard: /api/v1/dashboard")

if __name__ == '__main__':
    main()