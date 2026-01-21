#!/usr/bin/env python3
"""
Health Check Script for SKZ Agents Framework
Verifies all agents and services are running properly
"""

import os
import sys
import time
import requests
import logging
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('health_check.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_agent_health(agent_name, port):
    """Check health of individual agent"""
    logger = logging.getLogger(__name__)
    
    try:
        url = f"http://localhost:{port}/health"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            logger.info(f"HEALTHY: {agent_name} (port {port})")
            return True
        else:
            logger.warning(f"UNHEALTHY: {agent_name} (port {port}) - Status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error(f"OFFLINE: {agent_name} (port {port}) - Connection refused")
        return False
    except requests.exceptions.Timeout:
        logger.error(f"TIMEOUT: {agent_name} (port {port}) - Request timeout")
        return False
    except Exception as e:
        logger.error(f"ERROR: {agent_name} (port {port}) - {e}")
        return False

def check_system_resources():
    """Check system resources"""
    logger = logging.getLogger(__name__)
    
    try:
        import psutil
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"CPU Usage: {cpu_percent}%")
        
        # Check memory usage
        memory = psutil.virtual_memory()
        logger.info(f"Memory Usage: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        logger.info(f"Disk Usage: {disk.percent}% ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)")
        
        return {
            'cpu': cpu_percent,
            'memory': memory.percent,
            'disk': disk.percent
        }
        
    except ImportError:
        logger.warning("psutil not available - skipping system resource check")
        return None
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return None

def main():
    """Main health check function"""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("SKZ AGENTS FRAMEWORK - HEALTH CHECK")
    logger.info("=" * 60)
    
    # Define agents to check
    agents = [
        {'name': 'research_discovery', 'port': 8001},
        {'name': 'manuscript_analysis', 'port': 8002},
        {'name': 'peer_review_coordination', 'port': 8003},
        {'name': 'editorial_decision', 'port': 8004},
        {'name': 'publication_formatting', 'port': 8005},
        {'name': 'quality_assurance', 'port': 8006},
        {'name': 'workflow_orchestration', 'port': 8007}
    ]
    
    healthy_agents = 0
    total_agents = len(agents)
    
    logger.info("Checking agent health...")
    
    for agent in agents:
        if check_agent_health(agent['name'], agent['port']):
            healthy_agents += 1
    
    # Check system resources
    logger.info("\nChecking system resources...")
    resources = check_system_resources()
    
    # Calculate health score
    health_score = (healthy_agents / total_agents) * 100
    
    logger.info("=" * 60)
    logger.info("HEALTH CHECK SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Healthy Agents: {healthy_agents}/{total_agents}")
    logger.info(f"Health Score: {health_score:.1f}%")
    
    if resources:
        logger.info(f"System CPU: {resources['cpu']}%")
        logger.info(f"System Memory: {resources['memory']}%")
        logger.info(f"System Disk: {resources['disk']}%")
    
    # Determine overall health status
    if health_score >= 80:
        status = "EXCELLENT"
        logger.info(f"Overall Status: {status}")
        return 0
    elif health_score >= 60:
        status = "GOOD"
        logger.info(f"Overall Status: {status}")
        return 0
    elif health_score >= 40:
        status = "DEGRADED"
        logger.warning(f"Overall Status: {status}")
        return 1
    else:
        status = "CRITICAL"
        logger.error(f"Overall Status: {status}")
        return 2
    
    # Save health report
    with open('health_report.txt', 'w') as f:
        f.write(f"Health Check Report - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Healthy Agents: {healthy_agents}/{total_agents}\n")
        f.write(f"Health Score: {health_score:.1f}%\n")
        f.write(f"Overall Status: {status}\n")
        if resources:
            f.write(f"CPU: {resources['cpu']}%\n")
            f.write(f"Memory: {resources['memory']}%\n")
            f.write(f"Disk: {resources['disk']}%\n")

if __name__ == "__main__":
    sys.exit(main())
