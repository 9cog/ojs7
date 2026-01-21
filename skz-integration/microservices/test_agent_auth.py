#!/usr/bin/env python3
"""
Test script for authenticated agent actions
"""

import requests
import json
import time

# Test URLs
API_BASE = "http://localhost:5000"
AGENT_BASE = "http://localhost:5001"

def get_auth_token():
    """Get authentication token"""
    print("Getting authentication token...")
    response = requests.post(f"{API_BASE}/api/v1/auth/login", json={
        'username': 'testuser',
        'password': 'testpass'
    })
    
    if response.status_code == 200:
        data = response.json()
        token = data.get('token')
        print(f"✓ Token received: {token[:20]}...")
        return token
    else:
        print(f"✗ Failed to get token: {response.status_code}")
        return None

def test_agent_direct():
    """Test agent directly (should work without auth)"""
    print("\nTesting direct agent access...")
    
    try:
        # Test health endpoint
        response = requests.get(f"{AGENT_BASE}/health")
        print(f"Agent health: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Auth enabled: {data.get('auth_enabled')}")
        
        # Test agent info
        response = requests.get(f"{AGENT_BASE}/agent")
        print(f"Agent info: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Agent type: {data.get('type')}")
            print(f"  Auth required: {data.get('auth_required')}")
        
        # Test action without auth context
        response = requests.post(f"{AGENT_BASE}/action", json={
            'action_type': 'search_inci',
            'search_terms': ['hyaluronic acid', 'peptides']
        })
        print(f"Action without auth: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Success: {data.get('success')}")
            print(f"  Auth user: {data.get('auth_user')}")
        
    except Exception as e:
        print(f"Direct agent test failed: {e}")

def test_agent_via_gateway(token):
    """Test agent through API Gateway with authentication"""
    print("\nTesting agent via API Gateway...")
    
    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        # Test agent info via gateway
        response = requests.get(f"{API_BASE}/api/v1/agents/research-discovery", headers=headers)
        print(f"Gateway agent info: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Agent name: {data.get('name')}")
            print(f"  Service: {data.get('service')}")
        
        # Test authenticated action via gateway
        response = requests.post(f"{API_BASE}/api/v1/agents/research-discovery/action", 
                               headers=headers,
                               json={
                                   'action_type': 'analyze_submission',
                                   'submission_id': 'test-123',
                                   'manuscript_data': {
                                       'title': 'Novel Peptide Formulation Study',
                                       'abstract': 'Testing anti-aging properties...'
                                   }
                               })
        print(f"Authenticated action: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Success: {data.get('success')}")
            print(f"  Processing time: {data.get('processing_time'):.3f}s")
            print(f"  Auth user: {data.get('auth_user')}")
            
            # Show analysis results
            result = data.get('result', {})
            if result.get('analysis'):
                analysis = result['analysis']
                print(f"  Research gaps: {len(analysis.get('research_gaps_identified', []))}")
                print(f"  Processed by: {analysis.get('processed_by')}")
        
    except Exception as e:
        print(f"Gateway agent test failed: {e}")

def test_unauthorized_gateway_access():
    """Test accessing gateway without proper authentication"""
    print("\nTesting unauthorized gateway access...")
    
    try:
        # Test without token
        response = requests.get(f"{API_BASE}/api/v1/agents/research-discovery")
        print(f"No token: {response.status_code}")
        if response.status_code == 401:
            print("  ✓ Correctly rejected")
        
        # Test with invalid token
        headers = {'Authorization': 'Bearer invalid-token'}
        response = requests.get(f"{API_BASE}/api/v1/agents/research-discovery", headers=headers)
        print(f"Invalid token: {response.status_code}")
        if response.status_code == 401:
            print("  ✓ Correctly rejected")
        
        # Test action without token
        response = requests.post(f"{API_BASE}/api/v1/agents/research-discovery/action", 
                               json={'action_type': 'analyze_submission'})
        print(f"Action without token: {response.status_code}")
        if response.status_code == 401:
            print("  ✓ Correctly rejected")
        
    except Exception as e:
        print(f"Unauthorized test failed: {e}")

def test_permission_based_actions(token):
    """Test different actions requiring different permissions"""
    print("\nTesting permission-based actions...")
    
    headers = {'Authorization': f'Bearer {token}'}
    
    # Get user permissions
    response = requests.get(f"{API_BASE}/api/v1/auth/permissions", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"User permissions: {data.get('permissions', [])}")
    
    # Test different action types
    actions = [
        {
            'action_type': 'search_inci',
            'search_terms': ['retinol', 'vitamin c'],
            'description': 'INCI database search'
        },
        {
            'action_type': 'analyze_patents',
            'technology_area': 'anti-aging',
            'description': 'Patent landscape analysis'
        },
        {
            'action_type': 'identify_trends',
            'market_segment': 'skincare',
            'description': 'Market trend identification'
        }
    ]
    
    for action in actions:
        try:
            response = requests.post(f"{API_BASE}/api/v1/agents/research-discovery/action", 
                                   headers=headers,
                                   json=action)
            print(f"{action['description']}: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('result', {})
                print(f"  ✓ Success - {result.get('status', 'unknown')}")
            elif response.status_code == 403:
                print(f"  ✗ Permission denied")
            else:
                print(f"  ✗ Error: {response.text}")
                
        except Exception as e:
            print(f"  ✗ Exception: {e}")

def main():
    """Run comprehensive authentication tests"""
    print("=== SKZ Agents Authentication Integration Test ===")
    
    # Test direct agent access
    test_agent_direct()
    
    # Test unauthorized gateway access
    test_unauthorized_gateway_access()
    
    # Get authentication token
    token = get_auth_token()
    if not token:
        print("Cannot continue without authentication token")
        return
    
    # Test agent via authenticated gateway
    test_agent_via_gateway(token)
    
    # Test permission-based actions
    test_permission_based_actions(token)
    
    print("\n=== Authentication Integration Tests Complete ===")

if __name__ == "__main__":
    main()