#!/bin/bash

# Autonomous Agents Microservices Deployment Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICROSERVICES_DIR="$SCRIPT_DIR"

echo "🤖 Starting Autonomous Agents Microservices Deployment"
echo "========================================================"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Prerequisites met"

# Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker daemon is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker daemon is running"

# Navigate to microservices directory
cd "$MICROSERVICES_DIR"

echo "📁 Working directory: $(pwd)"

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Build and start services
echo "🏗️  Building and starting microservices..."
docker-compose up --build -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Health check function
check_service_health() {
    local service_name=$1
    local port=$2
    local max_retries=30
    local retry=0
    
    echo -n "🔍 Checking $service_name health..."
    
    while [ $retry -lt $max_retries ]; do
        if curl -s "http://localhost:$port/health" >/dev/null 2>&1; then
            echo " ✅ Healthy"
            return 0
        fi
        
        echo -n "."
        sleep 2
        retry=$((retry + 1))
    done
    
    echo " ❌ Failed to start"
    return 1
}

# Check service health
echo "🏥 Performing health checks..."

SERVICES=(
    "API Gateway:5000"
    "Research Discovery:5001"
    "Submission Assistant:5002"
    "Editorial Orchestration:5003"
    "Review Coordination:5004"
    "Content Quality:5005"
    "Publishing Production:5006"
    "Analytics Monitoring:5007"
)

failed_services=()

for service in "${SERVICES[@]}"; do
    name=$(echo "$service" | cut -d: -f1)
    port=$(echo "$service" | cut -d: -f2)
    
    if ! check_service_health "$name" "$port"; then
        failed_services+=("$name")
    fi
done

# Report results
echo ""
echo "📊 Deployment Summary"
echo "===================="

if [ ${#failed_services[@]} -eq 0 ]; then
    echo "🎉 All services started successfully!"
    echo ""
    echo "🌐 Access Points:"
    echo "   • API Gateway:              http://localhost:5000"
    echo "   • Service Dashboard:        http://localhost:5000/api/v1/services"
    echo "   • Agents Overview:          http://localhost:5000/api/v1/agents"
    echo "   • System Metrics:           http://localhost:5000/api/v1/metrics"
    echo ""
    echo "🔧 Individual Services:"
    echo "   • Research Discovery:       http://localhost:5001"
    echo "   • Submission Assistant:     http://localhost:5002"
    echo "   • Editorial Orchestration:  http://localhost:5003"
    echo "   • Review Coordination:      http://localhost:5004"
    echo "   • Content Quality:          http://localhost:5005"
    echo "   • Publishing Production:    http://localhost:5006"
    echo "   • Analytics Monitoring:     http://localhost:5007"
    echo ""
    echo "🚀 Ready for action!"
    
    # Quick functionality test
    echo ""
    echo "🧪 Running quick functionality test..."
    if curl -s "http://localhost:5000/api/v1/agents" >/dev/null; then
        agent_count=$(curl -s "http://localhost:5000/api/v1/agents" | python3 -c "import sys, json; print(json.load(sys.stdin)['total_count'])" 2>/dev/null || echo "unknown")
        echo "✅ API Gateway functional - $agent_count agents detected"
    else
        echo "⚠️  API Gateway test failed"
    fi
    
else
    echo "❌ Some services failed to start:"
    for service in "${failed_services[@]}"; do
        echo "   • $service"
    done
    echo ""
    echo "📝 Check logs with: docker-compose logs <service-name>"
    echo "🔧 Try restarting with: docker-compose restart <service-name>"
fi

echo ""
echo "📖 Management Commands:"
echo "   • View logs:        docker-compose logs -f"
echo "   • Stop services:    docker-compose down"
echo "   • Restart services: docker-compose restart"
echo "   • View status:      docker-compose ps"

echo ""
echo "🤖 Autonomous Agents Microservices Deployment Complete!"