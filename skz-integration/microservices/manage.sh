#!/bin/bash

# Autonomous Agents Microservices Management Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    echo "🤖 Autonomous Agents Microservices Management"
    echo "============================================="
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  start      Start all microservices"
    echo "  stop       Stop all microservices"
    echo "  restart    Restart all microservices"
    echo "  status     Show status of all services"
    echo "  logs       Show logs from all services"
    echo "  logs <svc> Show logs from specific service"
    echo "  build      Rebuild all services"
    echo "  clean      Stop and remove all containers/volumes"
    echo "  health     Check health of all services"
    echo "  scale <svc> <n>  Scale service to n replicas"
    echo ""
    echo "Services:"
    echo "  api-gateway, research-discovery, submission-assistant"
    echo "  editorial-orchestration, review-coordination, content-quality"
    echo "  publishing-production, analytics-monitoring"
}

check_service_health() {
    local service_name=$1
    local port=$2
    
    if curl -s "http://localhost:$port/health" >/dev/null 2>&1; then
        echo "✅ $service_name (port $port)"
        return 0
    else
        echo "❌ $service_name (port $port)"
        return 1
    fi
}

case "${1:-}" in
    start)
        echo "🚀 Starting microservices..."
        docker-compose up -d
        echo "✅ Services started"
        ;;
    
    stop)
        echo "🛑 Stopping microservices..."
        docker-compose down
        echo "✅ Services stopped"
        ;;
    
    restart)
        echo "🔄 Restarting microservices..."
        docker-compose restart
        echo "✅ Services restarted"
        ;;
    
    status)
        echo "📊 Service Status:"
        docker-compose ps
        ;;
    
    logs)
        if [ -n "${2:-}" ]; then
            echo "📝 Showing logs for $2..."
            docker-compose logs -f "$2"
        else
            echo "📝 Showing logs for all services..."
            docker-compose logs -f
        fi
        ;;
    
    build)
        echo "🏗️ Rebuilding services..."
        docker-compose build --no-cache
        echo "✅ Build complete"
        ;;
    
    clean)
        echo "🧹 Cleaning up containers and volumes..."
        docker-compose down --volumes --remove-orphans
        docker system prune -f
        echo "✅ Cleanup complete"
        ;;
    
    health)
        echo "🏥 Health Check Results:"
        echo "========================"
        
        services=(
            "API Gateway:5000"
            "Research Discovery:5001" 
            "Submission Assistant:5002"
            "Editorial Orchestration:5003"
            "Review Coordination:5004"
            "Content Quality:5005"
            "Publishing Production:5006"
            "Analytics Monitoring:5007"
        )
        
        healthy_count=0
        total_count=${#services[@]}
        
        for service in "${services[@]}"; do
            name=$(echo "$service" | cut -d: -f1)
            port=$(echo "$service" | cut -d: -f2)
            
            if check_service_health "$name" "$port"; then
                ((healthy_count++))
            fi
        done
        
        echo ""
        echo "📈 Summary: $healthy_count/$total_count services healthy"
        
        if [ $healthy_count -eq $total_count ]; then
            echo "🎉 All services are healthy!"
        else
            echo "⚠️  Some services need attention"
        fi
        ;;
    
    scale)
        if [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
            echo "❌ Usage: $0 scale <service> <replicas>"
            exit 1
        fi
        
        echo "📈 Scaling $2 to $3 replicas..."
        docker-compose up -d --scale "$2=$3"
        echo "✅ Scaling complete"
        ;;
    
    *)
        usage
        exit 1
        ;;
esac