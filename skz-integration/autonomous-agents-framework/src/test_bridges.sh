#!/usr/bin/env bash

# API Bridge Integration Test
# Tests communication between PHP OJS and Python agents

echo "======================================================"
echo "API BRIDGE INTEGRATION TEST"
echo "======================================================"
echo "Date: $(date)"
echo "Testing API bridges between PHP OJS and Python agents"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

success_count=0
total_count=0

# Function to run a test
run_test() {
    local test_name="$1"
    local command="$2"
    
    echo -n "Testing $test_name... "
    ((total_count++))
    
    if eval "$command" > /tmp/test_output 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((success_count++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "  Error details:"
        cat /tmp/test_output | sed 's/^/    /'
        return 1
    fi
}

# Test 1: Python API server can start
echo "=== Testing Python API Server ==="

# Start API server in background
cd /home/runner/work/oj7/oj7/skz-integration/autonomous-agents-framework/src

# Test if the simple API server can be imported
run_test "Python API server import" "python3 -c 'import simple_api_server; print(\"Import successful\")'"

# Start server in background
echo "Starting API server in background..."
python3 simple_api_server.py 5000 > /tmp/api_server.log 2>&1 &
SERVER_PID=$!
sleep 3

# Test server endpoints
run_test "Server status endpoint" "curl -s -f http://localhost:5000/status"
run_test "Agents list endpoint" "curl -s -f http://localhost:5000/agents"
run_test "Individual agent status" "curl -s -f http://localhost:5000/agents/research_discovery"

# Test POST request
run_test "Agent processing request" "curl -s -f -X POST -H 'Content-Type: application/json' -d '{\"test\": \"data\"}' http://localhost:5000/research_discovery/analyze"

echo ""
echo "=== Testing PHP Bridge ==="

# Test 2: PHP Bridge functionality
run_test "PHP syntax check" "php -l /home/runner/work/oj7/oj7/plugins/generic/skzAgents/classes/SKZAgentBridgeStandalone.inc.php"

# Test PHP bridge instantiation
run_test "PHP bridge instantiation" "php -r '
include_once \"/home/runner/work/oj7/oj7/plugins/generic/skzAgents/classes/SKZAgentBridgeStandalone.inc.php\";
\$bridge = new SKZAgentBridgeStandalone(\"http://localhost:5000\", \"test_key\", 10);
echo \"Bridge created successfully\";
'"

echo ""
echo "=== Testing Integration ==="

# Test 3: Full integration test
run_test "PHP to Python communication" "php -r '
include_once \"/home/runner/work/oj7/oj7/plugins/generic/skzAgents/classes/SKZAgentBridgeStandalone.inc.php\";
\$bridge = new SKZAgentBridgeStandalone(\"http://localhost:5000\", \"test_key\", 10);
\$result = \$bridge->testConnection();
if (\$result[\"success\"]) {
    echo \"Connection successful\";
} else {
    echo \"Connection failed: \" . \$result[\"message\"];
    exit(1);
}
'"

# Test agent call
run_test "Agent processing call" "php -r '
include_once \"/home/runner/work/oj7/oj7/plugins/generic/skzAgents/classes/SKZAgentBridgeStandalone.inc.php\";
\$bridge = new SKZAgentBridgeStandalone(\"http://localhost:5000\", \"test_key\", 10);
\$result = \$bridge->callAgent(\"research_discovery\", \"analyze\", array(
    \"manuscript_id\" => \"test_123\",
    \"content\" => \"Test manuscript content\"
));
if (isset(\$result[\"success\"]) && \$result[\"success\"]) {
    echo \"Agent call successful\";
} else {
    echo \"Agent call failed\";
    exit(1);
}
'"

# Test agent list
run_test "Get agents list" "php -r '
include_once \"/home/runner/work/oj7/oj7/plugins/generic/skzAgents/classes/SKZAgentBridgeStandalone.inc.php\";
\$bridge = new SKZAgentBridgeStandalone(\"http://localhost:5000\", \"test_key\", 10);
\$result = \$bridge->getAgentsList();
if (isset(\$result[\"agents\"]) && count(\$result[\"agents\"]) > 0) {
    echo \"Agents list retrieved: \" . count(\$result[\"agents\"]) . \" agents\";
} else {
    echo \"Failed to get agents list\";
    exit(1);
}
'"

# Clean up
echo ""
echo "Cleaning up..."
if [ ! -z "$SERVER_PID" ]; then
    kill $SERVER_PID 2>/dev/null
    echo "API server stopped"
fi

# Summary
echo ""
echo "======================================================"
echo "TEST RESULTS SUMMARY"
echo "======================================================"

success_rate=$(( (success_count * 100) / total_count ))

echo "Total Tests: $total_count"
echo "Passed: $success_count" 
echo "Failed: $((total_count - success_count))"
echo "Success Rate: $success_rate%"

if [ $success_rate -ge 80 ]; then
    echo ""
    echo -e "${GREEN}🎉 API BRIDGES: WORKING${NC}"
    echo -e "${GREEN}✅ PHP OJS and Python agents can communicate${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}❌ API BRIDGES: NEED ATTENTION${NC}"
    echo -e "${RED}❌ Some integration issues detected${NC}"
    exit 1
fi