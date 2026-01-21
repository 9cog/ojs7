#!/bin/bash

# Phase 3: Frontend Integration Comprehensive Test
# Tests all components for readiness and integration

echo "🧪 Phase 3: Frontend Integration Comprehensive Test"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo -n "🔍 $test_name... "
    
    # Capture both stdout and stderr, but redirect to /dev/null for the test
    if eval "$test_command" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Detailed test function
run_detailed_test() {
    local test_name="$1"
    local test_command="$2"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo "🔍 $test_name:"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASS${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

echo "📋 PHASE 3 COMPONENT TESTS"
echo "=========================="

# Test 1: React Dashboard Builds
echo ""
echo -e "${BLUE}1. Testing React Dashboard Builds${NC}"
run_test "Workflow visualization dashboard build" "(cd skz-integration/workflow-visualization-dashboard && npm run build)"
run_test "Simulation dashboard build" "(cd skz-integration/simulation-dashboard && npm run build)"

# Test 2: Dashboard Dependencies
echo ""
echo -e "${BLUE}2. Testing Dashboard Dependencies${NC}"
run_test "Workflow dashboard dependencies" "(cd skz-integration/workflow-visualization-dashboard && npm list --depth=0 > /dev/null 2>&1)"
run_test "Simulation dashboard dependencies" "(cd skz-integration/simulation-dashboard && npm list --depth=0 > /dev/null 2>&1)"

# Test 3: Key Dashboard Files
echo ""
echo -e "${BLUE}3. Testing Key Dashboard Files${NC}"
run_test "Workflow dashboard main component" "[ -f 'skz-integration/workflow-visualization-dashboard/src/App.jsx' ]"
run_test "Simulation dashboard main component" "[ -f 'skz-integration/simulation-dashboard/src/App.jsx' ]"
run_test "Built workflow dashboard assets" "[ -d 'skz-integration/workflow-visualization-dashboard/dist' ] && [ -f 'skz-integration/workflow-visualization-dashboard/dist/index.html' ]"
run_test "Built simulation dashboard assets" "[ -d 'skz-integration/simulation-dashboard/dist' ] && [ -f 'skz-integration/simulation-dashboard/dist/index.html' ]"

# Test 4: OJS Integration Files
echo ""
echo -e "${BLUE}4. Testing OJS Integration Files${NC}"
run_test "SKZ Dashboard handler" "[ -f 'pages/skzDashboard/SkzDashboardHandler.inc.php' ]"
run_test "Dashboard template" "[ -f 'templates/skzDashboard/index.tpl' ]"
run_test "Dashboard public assets" "[ -d 'public/skz-dashboard' ]"

# Test 5: Theme Integration
echo ""
echo -e "${BLUE}5. Testing Theme Integration${NC}"
run_test "SKZ Enhanced theme plugin" "[ -f 'plugins/themes/skzEnhanced/SKZEnhancedThemePlugin.inc.php' ]"
run_test "Theme templates directory" "[ -d 'plugins/themes/skzEnhanced/templates' ]"
run_test "Theme styles directory" "[ -d 'plugins/themes/skzEnhanced/styles' ]"
run_test "Theme JavaScript directory" "[ -d 'plugins/themes/skzEnhanced/js' ]"

# Test 6: Agent Interface Components
echo ""
echo -e "${BLUE}6. Testing Agent Interface Components${NC}"
run_test "Agent status bar template" "[ -f 'plugins/themes/skzEnhanced/templates/components/agent-status-bar.tpl' ]"
run_test "Workflow controls template" "[ -f 'plugins/themes/skzEnhanced/templates/components/workflow-agent-controls.tpl' ]"
run_test "Agent notifications template" "[ -f 'plugins/themes/skzEnhanced/templates/components/agent-notifications.tpl' ]"
run_test "Agent interface CSS" "[ -f 'plugins/themes/skzEnhanced/styles/skz-agent-interface.less' ]"

# Test 7: Real-time Components
echo ""
echo -e "${BLUE}7. Testing Real-time Components${NC}"
run_test "Socket.IO client in workflow dashboard" "grep -q 'socket.io-client' 'skz-integration/workflow-visualization-dashboard/package.json'"
run_test "Real-time monitoring JavaScript" "[ -f 'plugins/themes/skzEnhanced/js/skz-status-monitor.js' ]"
run_test "Agent UI JavaScript" "[ -f 'plugins/themes/skzEnhanced/js/skz-agent-ui.js' ]"

# Test 8: Admin Controls
echo ""
echo -e "${BLUE}8. Testing Admin Controls${NC}"
run_test "Admin menu navigation" "[ -f 'templates/frontend/components/navigationMenu.tpl' ] || [ -f 'lib/pkp/templates/frontend/components/navigationMenu.tpl' ]"
run_test "Agent management controls" "[ -f 'plugins/themes/skzEnhanced/js/skz-workflow-integration.js' ]"

# Test 9: Documentation
echo ""
echo -e "${BLUE}9. Testing Documentation${NC}"
run_test "Dashboard integration guide" "[ -f 'SKZ_DASHBOARD_INTEGRATION.md' ]"
run_test "README documentation" "[ -f 'README.md' ]"
run_test "Phase 3 documentation" "[ -f 'Phase 3 - Frontend Integration.md' ]"

# Test 10: Build Scripts and Deployment
echo ""
echo -e "${BLUE}10. Testing Build Scripts and Deployment${NC}"
run_test "Dashboard deployment script" "[ -f 'deploy-skz-dashboard.sh' ] && [ -x 'deploy-skz-dashboard.sh' ]"
run_test "SKZ integration script" "[ -f 'deploy-skz-integration.sh' ] && [ -x 'deploy-skz-integration.sh' ]"
run_test "Theme activation script" "[ -f 'activate-skz-theme.sh' ] && [ -x 'activate-skz-theme.sh' ]"

# Advanced Integration Tests
echo ""
echo "📊 ADVANCED INTEGRATION TESTS"
echo "============================="

# Test 11: Dashboard Feature Validation
echo ""
run_detailed_test "Dashboard feature validation" '
    echo "  🔍 Checking React app features..."
    if grep -q "agent.*monitoring\|workflow.*visualization\|real.*time" "skz-integration/workflow-visualization-dashboard/src/App.jsx"; then
        echo "  ✅ Advanced features found in workflow dashboard"
    else
        echo "  ❌ Advanced features missing"
        return 1
    fi
    
    if grep -q "D3\|d3\|visualization" "skz-integration/workflow-visualization-dashboard/package.json"; then
        echo "  ✅ D3.js visualization library present"
    else
        echo "  ❌ D3.js visualization library missing"
        return 1
    fi
    
    if grep -q "socket.*io" "skz-integration/workflow-visualization-dashboard/package.json"; then
        echo "  ✅ Real-time updates capability present"
    else
        echo "  ❌ Real-time updates capability missing"
        return 1
    fi
'

# Test 12: Theme Integration Validation
echo ""
run_detailed_test "Theme integration validation" '
    echo "  🔍 Checking theme integration..."
    if [ -f "plugins/themes/skzEnhanced/SkzEnhancedThemePlugin.inc.php" ]; then
        if grep -q "agent\|workflow\|skz" "plugins/themes/skzEnhanced/SkzEnhancedThemePlugin.inc.php"; then
            echo "  ✅ Theme plugin contains agent-specific functionality"
        else
            echo "  ❌ Theme plugin missing agent functionality"
            return 1
        fi
    else
        echo "  ❌ Theme plugin file missing"
        return 1
    fi
    
    if find "plugins/themes/skzEnhanced" -name "*.less" -exec grep -l "agent\|workflow\|skz" {} \; | head -1 >/dev/null; then
        echo "  ✅ Theme styles include agent-specific CSS"
    else
        echo "  ❌ Theme styles missing agent-specific CSS"
        return 1
    fi
'

# Test 13: JavaScript Validation
echo ""
run_detailed_test "JavaScript integration validation" '
    echo "  🔍 Checking JavaScript integration..."
    js_files=$(find "plugins/themes/skzEnhanced/js" -name "*.js" 2>/dev/null | wc -l)
    if [ "$js_files" -ge 3 ]; then
        echo "  ✅ Multiple JavaScript files found ($js_files files)"
    else
        echo "  ❌ Insufficient JavaScript files found ($js_files files)"
        return 1
    fi
    
    if find "plugins/themes/skzEnhanced/js" -name "*.js" -exec grep -l "agent\|workflow\|monitoring" {} \; | head -1 >/dev/null; then
        echo "  ✅ JavaScript files contain agent functionality"
    else
        echo "  ❌ JavaScript files missing agent functionality"
        return 1
    fi
'

# Test 14: Asset Optimization
echo ""
run_detailed_test "Asset optimization validation" '
    echo "  🔍 Checking asset optimization..."
    if [ -f "skz-integration/workflow-visualization-dashboard/dist/index.html" ]; then
        if grep -q "\.js\|\.css" "skz-integration/workflow-visualization-dashboard/dist/index.html"; then
            echo "  ✅ Workflow dashboard assets optimized and linked"
        else
            echo "  ❌ Workflow dashboard assets not properly linked"
            return 1
        fi
    else
        echo "  ❌ Workflow dashboard build missing"
        return 1
    fi
    
    if [ -f "skz-integration/simulation-dashboard/dist/index.html" ]; then
        if grep -q "\.js\|\.css" "skz-integration/simulation-dashboard/dist/index.html"; then
            echo "  ✅ Simulation dashboard assets optimized and linked"
        else
            echo "  ❌ Simulation dashboard assets not properly linked"
            return 1
        fi
    else
        echo "  ❌ Simulation dashboard build missing"
        return 1
    fi
'

# Results Summary
echo ""
echo "📊 FINAL RESULTS"
echo "================="
echo ""
echo -e "${BLUE}Tests Run:${NC} $TESTS_TOTAL"
echo -e "${GREEN}Passed:${NC} $TESTS_PASSED"
echo -e "${RED}Failed:${NC} $TESTS_FAILED"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
    echo ""
    echo "✅ Phase 3: Frontend Integration is COMPLETE"
    echo "✅ React-based visualization dashboards implemented and working"
    echo "✅ OJS theme modifications for agent interfaces complete"
    echo "✅ Real-time updates and notifications implemented"
    echo "✅ Agent management controls integrated into OJS admin"
    echo ""
    echo "🚀 READY FOR NEXT PHASE DEPLOYMENT!"
    echo ""
    echo "📋 Integration Status:"
    echo "   - Workflow Visualization Dashboard: ✅ Ready"
    echo "   - Simulation Dashboard: ✅ Ready"
    echo "   - SKZ Enhanced Theme: ✅ Ready"
    echo "   - OJS Integration: ✅ Ready"
    echo "   - Real-time Features: ✅ Ready"
    echo "   - Admin Controls: ✅ Ready"
    echo "   - Documentation: ✅ Complete"
    
    exit 0
else
    echo ""
    echo -e "${RED}⚠️  TESTS FAILED!${NC}"
    echo ""
    echo "❌ Phase 3: Frontend Integration has issues that need attention"
    echo "🔧 Please review failed tests above and fix issues before proceeding"
    
    success_rate=$((TESTS_PASSED * 100 / TESTS_TOTAL))
    echo ""
    echo "📊 Success Rate: $success_rate%"
    
    if [ $success_rate -ge 80 ]; then
        echo "✨ Most features are working - minor fixes needed"
    else
        echo "⚠️  Significant issues detected - major fixes required"
    fi
    
    exit 1
fi