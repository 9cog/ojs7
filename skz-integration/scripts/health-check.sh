#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python3 "${ROOT_DIR}/skz-integration/health_check.py" || true
python3 "${ROOT_DIR}/skz-integration/scripts/run-migrations.py" || true
python3 "${ROOT_DIR}/skz-integration/scripts/smoke_providers.py" || true

#!/bin/bash

# SKZ Integration Health Check Script

echo "🏥 SKZ Integration Health Check"
echo "==============================="

# Check OJS core system
echo "🔍 Checking OJS Core System..."
if curl -f -s http://localhost:8000 > /dev/null; then
    echo "✅ OJS Core: Running"
else
    echo "❌ OJS Core: Not responding"
fi

# Check agent framework
echo "🔍 Checking Agent Framework..."
if curl -f -s http://localhost:5000/api/v1/agents > /dev/null; then
    echo "✅ Agent Framework: Running"
else
    echo "❌ Agent Framework: Not responding"
fi

# Check skin zone journal (if running)
echo "🔍 Checking Skin Zone Journal..."
if curl -f -s http://localhost:5001/api/status > /dev/null 2>&1; then
    echo "✅ Skin Zone Journal: Running"
else
    echo "⚠️ Skin Zone Journal: Not running (may not be started)"
fi

# Check workflow visualization dashboard
echo "🔍 Checking Workflow Dashboard..."
if [ -d "skz-integration/workflow-visualization-dashboard/dist" ]; then
    echo "✅ Workflow Dashboard: Built"
else
    echo "❌ Workflow Dashboard: Not built"
fi

# Check simulation dashboard
echo "🔍 Checking Simulation Dashboard..."
if [ -d "skz-integration/simulation-dashboard/dist" ]; then
    echo "✅ Simulation Dashboard: Built"
else
    echo "❌ Simulation Dashboard: Not built"
fi

# Check Python virtual environments
echo "🔍 Checking Python Environments..."
if [ -d "skz-integration/autonomous-agents-framework/venv" ]; then
    echo "✅ Agent Framework venv: Created"
else
    echo "❌ Agent Framework venv: Missing"
fi

if [ -d "skz-integration/skin-zone-journal/venv" ]; then
    echo "✅ Skin Zone Journal venv: Created"
else
    echo "❌ Skin Zone Journal venv: Missing"
fi

# Check Composer dependencies
echo "🔍 Checking Composer Dependencies..."
if [ -d "lib/pkp/lib/vendor" ]; then
    echo "✅ Composer Dependencies: Installed"
else
    echo "❌ Composer Dependencies: Missing (run composer install)"
fi

echo "==============================="
echo "🏥 Health check complete"
