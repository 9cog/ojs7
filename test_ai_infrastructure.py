#!/usr/bin/env python3
"""
AI Infrastructure Integration Test
Tests the AI inference enforcement infrastructure without requiring models
"""

import os
import sys
import subprocess
from pathlib import Path

def test_file_structure():
    """Test that all required AI infrastructure files exist"""
    print("🧪 Testing AI infrastructure file structure...")
    
    required_files = [
        ".github/copilot-instructions.md",
        ".github/pull_request_template.md", 
        ".github/copilot-prompt-seed.md",
        ".github/workflows/ai-validation.yml",
        "skz-integration/autonomous-agents-framework/src/models/production_ai_engine.py",
        "skz-integration/autonomous-agents-framework/.env.ai.template",
        "skz-integration/ai-inference-node/package.json",
        "skz-integration/ai-inference-node/ai-engine.js"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            return False
    
    return True

def test_copilot_instructions():
    """Test that copilot instructions contain AI requirements"""
    print("\n🧪 Testing Copilot instructions content...")
    
    instructions_path = ".github/copilot-instructions.md"
    
    if not os.path.exists(instructions_path):
        print(f"❌ Instructions file not found: {instructions_path}")
        return False
    
    with open(instructions_path, 'r') as f:
        content = f.read()
    
    required_keywords = [
        "AI Inference Engine Requirements",
        "llama.cpp",
        "node-llama-cpp", 
        "NEVER SACRIFICE QUALITY",
        "PRODUCTION-GRADE AI INFERENCE MANDATORY"
    ]
    
    for keyword in required_keywords:
        if keyword in content:
            print(f"✅ Found: {keyword}")
        else:
            print(f"❌ Missing keyword: {keyword}")
            return False
    
    return True

def test_pr_template():
    """Test that PR template contains AI compliance checklist"""
    print("\n🧪 Testing PR template content...")
    
    template_path = ".github/pull_request_template.md"
    
    if not os.path.exists(template_path):
        print(f"❌ PR template not found: {template_path}")
        return False
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    required_elements = [
        "AI Engine Implementation Checklist",
        "llama.cpp",
        "No mock/stub/placeholder logic",
        "AI Model Configuration",
        "Production Quality Verification"
    ]
    
    for element in required_elements:
        if element in content:
            print(f"✅ Found: {element}")
        else:
            print(f"❌ Missing element: {element}")
            return False
    
    return True

def test_github_action():
    """Test that GitHub Action exists and has proper structure"""
    print("\n🧪 Testing GitHub Action structure...")
    
    action_path = ".github/workflows/ai-validation.yml"
    
    if not os.path.exists(action_path):
        print(f"❌ GitHub Action not found: {action_path}")
        return False
    
    with open(action_path, 'r') as f:
        content = f.read()
    
    required_components = [
        "name: AI Engine Enforcement",
        "check-ai-engine:",
        "Search for prohibited mock/stub usage",
        "Verify AI model integration",
        "AI Enforcement Summary"
    ]
    
    for component in required_components:
        if component in content:
            print(f"✅ Found: {component}")
        else:
            print(f"❌ Missing component: {component}")
            return False
    
    return True

def test_ai_dependencies():
    """Test that AI dependencies are included in requirements"""
    print("\n🧪 Testing AI dependencies...")
    
    requirements_path = "skz-integration/autonomous-agents-framework/requirements.txt"
    
    if not os.path.exists(requirements_path):
        print(f"❌ Requirements file not found: {requirements_path}")
        return False
    
    with open(requirements_path, 'r') as f:
        content = f.read()
    
    required_deps = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "llama-cpp-python>=0.2.20",
        "onnxruntime>=1.15.0"
    ]
    
    for dep in required_deps:
        if dep in content:
            print(f"✅ Found dependency: {dep}")
        else:
            print(f"❌ Missing dependency: {dep}")
            return False
    
    return True

def test_node_integration():
    """Test Node.js integration structure"""
    print("\n🧪 Testing Node.js integration...")
    
    package_path = "skz-integration/ai-inference-node/package.json"
    
    if not os.path.exists(package_path):
        print(f"❌ Node.js package.json not found: {package_path}")
        return False
    
    engine_path = "skz-integration/ai-inference-node/ai-engine.js"
    
    if not os.path.exists(engine_path):
        print(f"❌ Node.js AI engine not found: {engine_path}")
        return False
    
    with open(engine_path, 'r') as f:
        content = f.read()
    
    required_elements = [
        "node-llama-cpp",
        "ProductionAIEngine",
        "PRODUCTION VIOLATION",
        "Zero tolerance for mock implementations"
    ]
    
    for element in required_elements:
        if element in content:
            print(f"✅ Found: {element}")
        else:
            print(f"❌ Missing: {element}")
            return False
    
    return True

def main():
    """Run all integration tests"""
    print("🚀 AI Inference Enforcement Infrastructure Test")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_copilot_instructions,
        test_pr_template, 
        test_github_action,
        test_ai_dependencies,
        test_node_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED\n")
            else:
                failed += 1
                print("❌ FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ ERROR: {e}\n")
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All AI inference enforcement infrastructure tests PASSED!")
        print("✅ Ready for production AI model deployment")
        return 0
    else:
        print("❌ Some tests failed - review infrastructure setup")
        return 1

if __name__ == "__main__":
    sys.exit(main())