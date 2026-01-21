#!/bin/bash

# SKZ Agents Phase 1 Test Runner
# Runs all tests to validate Phase 1 completion

echo "=== SKZ Agents Phase 1 Test Runner ==="
echo ""

# Change to OJS root directory
cd "$(dirname "$0")/../../../.."

echo "Running Phase 1 Integration Tests..."
echo "===================================="

# Run the comprehensive integration test
php plugins/generic/skzAgents/tests/Phase1IntegrationTest.php
exit_code=$?

echo ""
echo "Running original gateway configuration test..."
echo "=============================================="

# Run the existing gateway config test
php plugins/generic/skzAgents/test_gateway_config.php

echo ""
if [ $exit_code -eq 0 ]; then
    echo "🎉 ALL PHASE 1 TESTS PASSED!"
    echo "✅ Phase 1: Foundation Setup is COMPLETE"
    echo "🚀 Ready to proceed to Phase 2: Core Agent Integration"
else
    echo "❌ PHASE 1 TESTS FAILED"
    echo "🛠️ Please address the issues above before proceeding"
fi

echo ""
echo "Phase 1 Testing Complete"
echo "========================"

exit $exit_code