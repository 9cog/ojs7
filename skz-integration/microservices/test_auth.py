#!/usr/bin/env python3
"""
Test script for authentication and authorization system
"""

import requests
import json
import time
import sys
import os

# Test API Gateway URL
API_BASE = "http://localhost:5000"

def test_health():
    """Test health endpoint (no auth required)"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Status: {data.get('status')}")
            print(f"  Auth enabled: {data.get('auth_enabled')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_login():
    """Test login endpoint"""
    print("\nTesting login endpoint...")
    try:
        response = requests.post(f"{API_BASE}/api/v1/auth/login", json={
            'username': 'testuser',
            'password': 'testpass'
        })
        print(f"Login: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Token received: {data.get('token')[:20]}...")
            print(f"  User: {data.get('user', {}).get('username')}")
            return data.get('token')
        else:
            print(f"  Error: {response.text}")
        return None
    except Exception as e:
        print(f"Login failed: {e}")
        return None

def test_protected_endpoint(token):
    """Test protected endpoint with token"""
    print("\nTesting protected endpoints...")
    headers = {'Authorization': f'Bearer {token}'}
    
    # Test services endpoint
    try:
        response = requests.get(f"{API_BASE}/api/v1/services", headers=headers)
        print(f"Services endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Services found: {len(data.get('services', []))}")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"Services endpoint failed: {e}")
    
    # Test permissions endpoint
    try:
        response = requests.get(f"{API_BASE}/api/v1/auth/permissions", headers=headers)
        print(f"Permissions endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Roles: {data.get('roles')}")
            print(f"  Permissions: {data.get('permissions')}")
    except Exception as e:
        print(f"Permissions endpoint failed: {e}")

def test_unauthorized_access():
    """Test accessing protected endpoint without token"""
    print("\nTesting unauthorized access...")
    try:
        response = requests.get(f"{API_BASE}/api/v1/services")
        print(f"Unauthorized services access: {response.status_code}")
        if response.status_code == 401:
            print("  Correctly rejected unauthorized request")
        else:
            print(f"  Unexpected response: {response.text}")
    except Exception as e:
        print(f"Unauthorized test failed: {e}")

def main():
    """Run all authentication tests"""
    print("=== SKZ Agents Authentication Test ===")
    
    # Test health endpoint
    if not test_health():
        print("Health check failed - API Gateway may not be running")
        sys.exit(1)
    
    # Test unauthorized access
    test_unauthorized_access()
    
    # Test login
    token = test_login()
    if not token:
        print("Login failed - cannot continue with authenticated tests")
        sys.exit(1)
    
    # Test protected endpoints
    test_protected_endpoint(token)
    
    print("\n=== Authentication Tests Complete ===")

if __name__ == "__main__":
    main()