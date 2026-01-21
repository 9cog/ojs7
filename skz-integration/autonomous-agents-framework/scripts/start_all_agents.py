#!/usr/bin/env python3
"""
Start All SKZ Agents Script
Launches all autonomous agents in the SKZ framework
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('agent_startup.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def start_agent_service(agent_name, port, script_path):
    """Start an individual agent service"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting {agent_name} on port {port}...")
        
        # Start the agent process
        process = subprocess.Popen([
            sys.executable, script_path,
            '--port', str(port),
            '--agent', agent_name
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info(f"SUCCESS: {agent_name} started successfully on port {port}")
            return True, process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"FAILED: {agent_name} failed to start: {stderr.decode()}")
            return False, None
            
    except Exception as e:
        logger.error(f"ERROR: Failed to start {agent_name}: {e}")
        return False, None

def main():
    """Main function to start all agents"""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("SKZ AGENTS FRAMEWORK - STARTING ALL AGENTS")
    logger.info("=" * 60)
    
    # Define agent configurations
    agents = [
        {
            'name': 'research_discovery',
            'port': 8001,
            'script': 'agents/research_discovery_agent.py'
        },
        {
            'name': 'manuscript_analysis', 
            'port': 8002,
            'script': 'agents/manuscript_analysis_agent.py'
        },
        {
            'name': 'peer_review_coordination',
            'port': 8003,
            'script': 'agents/peer_review_agent.py'
        },
        {
            'name': 'editorial_decision',
            'port': 8004,
            'script': 'agents/editorial_decision_agent.py'
        },
        {
            'name': 'publication_formatting',
            'port': 8005,
            'script': 'agents/publication_formatting_agent.py'
        },
        {
            'name': 'quality_assurance',
            'port': 8006,
            'script': 'agents/quality_assurance_agent.py'
        },
        {
            'name': 'workflow_orchestration',
            'port': 8007,
            'script': 'agents/workflow_orchestration_agent.py'
        }
    ]
    
    started_agents = []
    failed_agents = []
    
    for agent in agents:
        success, process = start_agent_service(
            agent['name'], 
            agent['port'], 
            agent['script']
        )
        
        if success:
            started_agents.append({
                'name': agent['name'],
                'port': agent['port'],
                'process': process
            })
        else:
            failed_agents.append(agent['name'])
    
    # Summary
    logger.info("=" * 60)
    logger.info("AGENT STARTUP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Successfully started: {len(started_agents)} agents")
    logger.info(f"Failed to start: {len(failed_agents)} agents")
    
    if started_agents:
        logger.info("\nRunning Agents:")
        for agent in started_agents:
            logger.info(f"  - {agent['name']} (port {agent['port']})")
    
    if failed_agents:
        logger.warning(f"\nFailed Agents: {', '.join(failed_agents)}")
    
    # Create status file
    status = {
        'timestamp': time.time(),
        'started_agents': len(started_agents),
        'failed_agents': len(failed_agents),
        'total_agents': len(agents),
        'success_rate': len(started_agents) / len(agents) * 100
    }
    
    with open('agent_status.txt', 'w') as f:
        f.write(f"Agent Status - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Started: {status['started_agents']}/{status['total_agents']}\n")
        f.write(f"Success Rate: {status['success_rate']:.1f}%\n")
    
    logger.info(f"\nOverall Success Rate: {status['success_rate']:.1f}%")
    
    if status['success_rate'] >= 70:
        logger.info("DEPLOYMENT STATUS: SUCCESSFUL")
        return 0
    else:
        logger.error("DEPLOYMENT STATUS: FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
