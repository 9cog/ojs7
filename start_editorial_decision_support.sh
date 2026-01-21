#!/bin/bash

# Editorial Decision Support System Startup Script
# Starts all required services for the SKZ decision support system

set -e

echo "=========================================="
echo "SKZ Editorial Decision Support System"
echo "Starting Services..."
echo "=========================================="

# Navigate to the base directory
cd /home/runner/work/oj7/oj7

# Check if Python virtual environment exists for SKZ agents
if [ ! -d "skz-integration/autonomous-agents-framework/venv" ]; then
    echo "Creating Python virtual environment..."
    cd skz-integration/autonomous-agents-framework
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install flask flask-cors numpy scikit-learn requests
    cd ../..
fi

# Function to start a service
start_service() {
    local name=$1
    local script=$2
    local port=$3
    local log_file=$4
    
    echo "Starting $name on port $port..."
    
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: Python3 not found"
        return 1
    fi
    
    # Kill existing process on port if any
    if lsof -ti:$port > /dev/null 2>&1; then
        echo "  Stopping existing service on port $port..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # Start the service
    cd skz-integration
    source autonomous-agents-framework/venv/bin/activate 2>/dev/null || true
    
    nohup python3 "$script" --port $port > "$log_file" 2>&1 &
    local pid=$!
    
    echo "  Started with PID: $pid"
    echo "  Log file: $log_file"
    
    # Wait a bit and check if process is still running
    sleep 3
    if kill -0 $pid 2>/dev/null; then
        echo "  ✓ Service running successfully"
        return 0
    else
        echo "  ✗ Service failed to start"
        return 1
    fi
}

# Function to check service health
check_service() {
    local name=$1
    local url=$2
    local max_attempts=10
    local attempt=1
    
    echo "Checking $name health..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "  ✓ $name is healthy"
            return 0
        fi
        
        echo "  Attempt $attempt/$max_attempts - waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "  ✗ $name health check failed"
    return 1
}

# Create logs directory
mkdir -p logs

# Start Enhanced Decision Support Service
start_service \
    "Enhanced Decision Support" \
    "enhanced_decision_support.py" \
    "8005" \
    "logs/enhanced_decision_support.log"

enhanced_status=$?

# Start Original Editorial Decision Agent (if available)
if [ -f "skz-integration/autonomous-agents-framework/agents/editorial_decision_agent.py" ]; then
    start_service \
        "Editorial Decision Agent" \
        "autonomous-agents-framework/agents/editorial_decision_agent.py" \
        "8004" \
        "logs/editorial_decision_agent.log"
    agent_status=$?
else
    echo "Editorial Decision Agent not found, skipping..."
    agent_status=0
fi

# Wait for services to fully initialize
echo ""
echo "Waiting for services to initialize..."
sleep 5

# Health checks
echo ""
echo "Performing health checks..."

check_service "Enhanced Decision Support" "http://localhost:8005/health"
enhanced_health=$?

if [ -f "skz-integration/autonomous-agents-framework/agents/editorial_decision_agent.py" ]; then
    check_service "Editorial Decision Agent" "http://localhost:8004/health"
    agent_health=$?
else
    agent_health=0
fi

# Summary
echo ""
echo "=========================================="
echo "SERVICE STARTUP SUMMARY"
echo "=========================================="

if [ $enhanced_status -eq 0 ] && [ $enhanced_health -eq 0 ]; then
    echo "✓ Enhanced Decision Support: Running on port 8005"
else
    echo "✗ Enhanced Decision Support: Failed to start or unhealthy"
fi

if [ -f "skz-integration/autonomous-agents-framework/agents/editorial_decision_agent.py" ]; then
    if [ $agent_status -eq 0 ] && [ $agent_health -eq 0 ]; then
        echo "✓ Editorial Decision Agent: Running on port 8004"
    else
        echo "✗ Editorial Decision Agent: Failed to start or unhealthy"
    fi
else
    echo "- Editorial Decision Agent: Not available"
fi

# Show running processes
echo ""
echo "Running processes:"
ps aux | grep -E "(enhanced_decision_support|editorial_decision_agent)" | grep -v grep || echo "No decision support processes found"

# Show logs location
echo ""
echo "Service logs available in: logs/"
ls -la logs/ 2>/dev/null || true

if [ $enhanced_status -eq 0 ] && [ $enhanced_health -eq 0 ]; then
    echo ""
    echo "🎉 Editorial Decision Support System is ready!"
    echo "   Enhanced API: http://localhost:8005"
    echo "   Health Check: http://localhost:8005/health"
    echo ""
    echo "To test the system, run:"
    echo "   python3 test_editorial_decision_support.py"
    exit 0
else
    echo ""
    echo "❌ Failed to start Editorial Decision Support System"
    echo "   Check logs in logs/ directory for details"
    exit 1
fi