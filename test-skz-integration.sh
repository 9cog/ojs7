#!/bin/bash

# SKZ Dashboard Integration Test Script
# This script validates that the React dashboard is properly integrated with OJS

set -e

echo "🧪 Starting SKZ Dashboard Integration Tests..."

# Test 1: Check if dashboard files exist
echo "📁 Test 1: Checking dashboard deployment..."
if [ -f "public/skz-dashboard/index.html" ]; then
    echo "✅ Dashboard HTML file exists"
else
    echo "❌ Dashboard HTML file missing"
    exit 1
fi

if [ -d "public/skz-dashboard/assets" ]; then
    echo "✅ Dashboard assets directory exists"
else
    echo "❌ Dashboard assets directory missing"
    exit 1
fi

# Test 2: Check OJS handler exists
echo "🔧 Test 2: Checking OJS integration files..."
if [ -f "pages/skzDashboard/SkzDashboardHandler.inc.php" ]; then
    echo "✅ OJS Dashboard handler exists"
else
    echo "❌ OJS Dashboard handler missing"
    exit 1
fi

if [ -f "templates/skzDashboard/index.tpl" ]; then
    echo "✅ OJS Dashboard template exists"
else
    echo "❌ OJS Dashboard template missing"
    exit 1
fi

# Test 3: Check navigation integration
echo "🧭 Test 3: Checking navigation integration..."
if grep -q "skzDashboard" templates/frontend/components/primaryNavMenu.tpl; then
    echo "✅ Navigation menu includes SKZ Dashboard"
else
    echo "❌ Navigation menu missing SKZ Dashboard"
    exit 1
fi

# Test 4: Validate dashboard HTML structure
echo "📄 Test 4: Validating dashboard HTML..."
if grep -q "skz-dashboard-root" public/skz-dashboard/index.html; then
    echo "✅ Dashboard has correct mount point"
else
    echo "❌ Dashboard missing mount point"
    exit 1
fi

# Test 5: Check asset references
echo "🎨 Test 5: Checking asset references..."
if grep -q "./assets/" public/skz-dashboard/index.html; then
    echo "✅ Dashboard uses relative asset paths"
else
    echo "❌ Dashboard has incorrect asset paths"
    exit 1
fi

# Test 6: Check deployment script
echo "📦 Test 6: Checking deployment script..."
if [ -x "deploy-skz-dashboard.sh" ]; then
    echo "✅ Deployment script is executable"
else
    echo "❌ Deployment script missing or not executable"
    exit 1
fi

# Test 7: Check documentation
echo "📚 Test 7: Checking documentation..."
if [ -f "SKZ_DASHBOARD_INTEGRATION.md" ]; then
    echo "✅ Integration documentation exists"
else
    echo "❌ Integration documentation missing"
    exit 1
fi

# Test 8: Validate React app mount point
echo "⚛️ Test 8: Checking React integration..."
if grep -q "skz-dashboard-root" skz-integration/workflow-visualization-dashboard/src/main.jsx; then
    echo "✅ React app configured for OJS integration"
else
    echo "❌ React app not configured for OJS integration"
    exit 1
fi

echo ""
echo "🎉 All integration tests passed!"
echo ""
echo "🚀 SKZ Dashboard Integration Status:"
echo "   ✅ React dashboard built and deployed"
echo "   ✅ OJS handlers and templates configured"
echo "   ✅ Navigation menu integration complete"
echo "   ✅ Asset paths properly configured"
echo "   ✅ Documentation and deployment scripts ready"
echo ""
echo "🔗 Access dashboard at: [ojs-url]/index.php/[journal]/skzDashboard"
echo "📖 See SKZ_DASHBOARD_INTEGRATION.md for detailed usage"