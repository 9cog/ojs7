"""
Workflow Optimization System for SKZ Autonomous Agents
Advanced workflow coordination, optimization, and performance monitoring
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import networkx as nx
from collections import deque, defaultdict
import heapq

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5

@dataclass
class WorkflowTask:
    """Individual workflow task"""
    task_id: str
    task_name: str
    agent_id: str
    dependencies: List[str]
    estimated_duration: int  # minutes
    priority: TaskPriority
    deadline: Optional[str]
    resource_requirements: Dict[str, Any]
    input_data: Dict[str, Any]
    output_schema: Dict[str, Any]
    retry_count: int
    max_retries: int
    status: WorkflowStatus
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]

@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    workflow_id: str
    workflow_name: str
    description: str
    tasks: List[WorkflowTask]
    global_timeout: int  # minutes
    parallel_limit: int
    retry_policy: Dict[str, Any]
    success_criteria: Dict[str, Any]
    failure_handling: Dict[str, Any]
    created_at: str

@dataclass
class AgentResource:
    """Agent resource availability"""
    agent_id: str
    agent_type: str
    current_load: int
    max_capacity: int
    avg_task_time: float
    success_rate: float
    last_active: str
    capabilities: List[str]
    status: str

@dataclass
class OptimizationResult:
    """Workflow optimization result"""
    workflow_id: str
    optimized_schedule: List[Dict[str, Any]]
    estimated_completion_time: int
    resource_utilization: Dict[str, float]
    critical_path: List[str]
    bottlenecks: List[str]
    optimization_score: float
    recommendations: List[str]
    timestamp: str

class WorkflowOptimizer:
    """Advanced workflow optimization and coordination system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_workflows = {}
        self.agent_resources = {}
        self.task_queue = []
        self.performance_history = defaultdict(list)
        
        # Optimization parameters
        self.optimization_weights = {
            'completion_time': 0.4,
            'resource_efficiency': 0.3,
            'success_probability': 0.2,
            'cost_optimization': 0.1
        }
        
    async def optimize_workflow(self, workflow: WorkflowDefinition, available_agents: List[AgentResource]) -> OptimizationResult:
        """Optimize workflow execution strategy"""
        
        logger.info(f"Optimizing workflow {workflow.workflow_id}")
        
        try:
            # Build dependency graph
            dependency_graph = await self._build_dependency_graph(workflow.tasks)
            
            # Calculate critical path
            critical_path = await self._calculate_critical_path(dependency_graph, workflow.tasks)
            
            # Identify bottlenecks
            bottlenecks = await self._identify_bottlenecks(workflow.tasks, available_agents)
            
            # Generate optimized schedule
            schedule = await self._generate_optimal_schedule(workflow, available_agents, dependency_graph)
            
            # Calculate metrics
            completion_time = await self._estimate_completion_time(schedule, workflow.tasks)
            resource_utilization = await self._calculate_resource_utilization(schedule, available_agents)
            
            # Generate recommendations
            recommendations = await self._generate_optimization_recommendations(
                workflow, schedule, bottlenecks, resource_utilization
            )
            
            # Calculate optimization score
            optimization_score = await self._calculate_optimization_score(
                completion_time, resource_utilization, len(bottlenecks)
            )
            
            return OptimizationResult(
                workflow_id=workflow.workflow_id,
                optimized_schedule=schedule,
                estimated_completion_time=completion_time,
                resource_utilization=resource_utilization,
                critical_path=critical_path,
                bottlenecks=bottlenecks,
                optimization_score=optimization_score,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error optimizing workflow: {e}")
            return OptimizationResult("", [], 0, {}, [], [], 0.0, [], datetime.now().isoformat())
    
    async def _build_dependency_graph(self, tasks: List[WorkflowTask]) -> nx.DiGraph:
        """Build task dependency graph"""
        
        graph = nx.DiGraph()
        
        # Add all tasks as nodes
        for task in tasks:
            graph.add_node(task.task_id, task_data=task)
        
        # Add dependency edges
        for task in tasks:
            for dependency in task.dependencies:
                if dependency in [t.task_id for t in tasks]:
                    graph.add_edge(dependency, task.task_id)
        
        return graph
    
    async def _calculate_critical_path(self, graph: nx.DiGraph, tasks: List[WorkflowTask]) -> List[str]:
        """Calculate critical path through workflow"""
        
        try:
            # Create task duration mapping
            task_durations = {task.task_id: task.estimated_duration for task in tasks}
            
            # Calculate longest path (critical path)
            if nx.is_directed_acyclic_graph(graph):
                # Topological sort to find valid execution order
                topo_order = list(nx.topological_sort(graph))
                
                # Calculate earliest start times
                earliest_start = {task_id: 0 for task_id in topo_order}
                
                for task_id in topo_order:
                    for predecessor in graph.predecessors(task_id):
                        earliest_start[task_id] = max(
                            earliest_start[task_id],
                            earliest_start[predecessor] + task_durations[predecessor]
                        )
                
                # Find critical path
                max_completion = max(
                    earliest_start[task_id] + task_durations[task_id] 
                    for task_id in topo_order
                )
                
                critical_tasks = []
                for task_id in topo_order:
                    if earliest_start[task_id] + task_durations[task_id] == max_completion:
                        critical_tasks.append(task_id)
                
                return critical_tasks
            else:
                logger.warning("Workflow contains cycles - using topological order")
                return list(nx.topological_sort(graph))
                
        except Exception as e:
            logger.error(f"Error calculating critical path: {e}")
            return [task.task_id for task in tasks]
    
    async def _identify_bottlenecks(self, tasks: List[WorkflowTask], agents: List[AgentResource]) -> List[str]:
        """Identify potential workflow bottlenecks"""
        
        bottlenecks = []
        
        # Resource capacity bottlenecks
        agent_capacity = {agent.agent_id: agent.max_capacity - agent.current_load for agent in agents}
        
        for task in tasks:
            required_agent = task.agent_id
            if required_agent in agent_capacity:
                if agent_capacity[required_agent] < 1:
                    bottlenecks.append(f"Agent capacity: {required_agent}")
                    
        # Task dependency bottlenecks
        dependency_counts = defaultdict(int)
        for task in tasks:
            dependency_counts[len(task.dependencies)] += 1
        
        # Tasks with many dependencies can be bottlenecks
        high_dependency_tasks = [task.task_id for task in tasks if len(task.dependencies) > 3]
        for task_id in high_dependency_tasks:
            bottlenecks.append(f"High dependency task: {task_id}")
        
        # Duration-based bottlenecks
        avg_duration = sum(task.estimated_duration for task in tasks) / len(tasks)
        long_tasks = [task.task_id for task in tasks if task.estimated_duration > avg_duration * 2]
        for task_id in long_tasks:
            bottlenecks.append(f"Long duration task: {task_id}")
        
        return bottlenecks[:5]  # Top 5 bottlenecks
    
    async def _generate_optimal_schedule(self, 
                                       workflow: WorkflowDefinition,
                                       agents: List[AgentResource],
                                       dependency_graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """Generate optimal task execution schedule"""
        
        schedule = []
        
        try:
            # Get topological order for dependency constraints
            execution_order = list(nx.topological_sort(dependency_graph))
            
            # Create agent availability tracking
            agent_availability = {}
            for agent in agents:
                agent_availability[agent.agent_id] = {
                    'next_free_time': 0,
                    'current_tasks': [],
                    'capacity_remaining': agent.max_capacity - agent.current_load
                }
            
            # Schedule tasks in dependency order
            scheduled_tasks = {}
            current_time = 0
            
            for task_id in execution_order:
                task = next(t for t in workflow.tasks if t.task_id == task_id)
                
                # Find best agent for this task
                best_agent = await self._find_best_agent(task, agents, agent_availability)
                
                if best_agent:
                    # Calculate start time based on dependencies and agent availability
                    dependency_completion_time = 0
                    for dep_id in task.dependencies:
                        if dep_id in scheduled_tasks:
                            dependency_completion_time = max(
                                dependency_completion_time,
                                scheduled_tasks[dep_id]['end_time']
                            )
                    
                    start_time = max(
                        dependency_completion_time,
                        agent_availability[best_agent.agent_id]['next_free_time']
                    )
                    
                    end_time = start_time + task.estimated_duration
                    
                    # Schedule the task
                    schedule_entry = {
                        'task_id': task.task_id,
                        'agent_id': best_agent.agent_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': task.estimated_duration,
                        'priority': task.priority.value,
                        'dependencies_met': all(dep in scheduled_tasks for dep in task.dependencies)
                    }
                    
                    schedule.append(schedule_entry)
                    scheduled_tasks[task_id] = schedule_entry
                    
                    # Update agent availability
                    agent_availability[best_agent.agent_id]['next_free_time'] = end_time
                    agent_availability[best_agent.agent_id]['capacity_remaining'] -= 1
                    
                else:
                    logger.warning(f"No suitable agent found for task {task_id}")
            
            # Sort schedule by start time
            schedule.sort(key=lambda x: x['start_time'])
            
        except Exception as e:
            logger.error(f"Error generating schedule: {e}")
        
        return schedule
    
    async def _find_best_agent(self, task: WorkflowTask, agents: List[AgentResource], availability: Dict) -> Optional[AgentResource]:
        """Find best agent for a task"""
        
        suitable_agents = []
        
        for agent in agents:
            # Check if agent can handle this task type
            if (task.agent_id == agent.agent_id or 
                task.agent_id in agent.capabilities or
                agent.agent_type in task.resource_requirements.get('agent_types', [])):
                
                # Check capacity
                if availability[agent.agent_id]['capacity_remaining'] > 0:
                    
                    # Calculate suitability score
                    score = 0.0
                    
                    # Success rate factor
                    score += agent.success_rate * 0.4
                    
                    # Load factor (prefer less loaded agents)
                    load_factor = 1.0 - (agent.current_load / agent.max_capacity)
                    score += load_factor * 0.3
                    
                    # Speed factor
                    if agent.avg_task_time > 0:
                        speed_factor = min(1.0, task.estimated_duration / agent.avg_task_time)
                        score += speed_factor * 0.3
                    
                    suitable_agents.append((agent, score))
        
        # Return best agent
        if suitable_agents:
            suitable_agents.sort(key=lambda x: x[1], reverse=True)
            return suitable_agents[0][0]
        
        return None
    
    async def _estimate_completion_time(self, schedule: List[Dict[str, Any]], tasks: List[WorkflowTask]) -> int:
        """Estimate total workflow completion time"""
        
        if not schedule:
            return sum(task.estimated_duration for task in tasks)
        
        return max(entry['end_time'] for entry in schedule)
    
    async def _calculate_resource_utilization(self, schedule: List[Dict[str, Any]], agents: List[AgentResource]) -> Dict[str, float]:
        """Calculate resource utilization metrics"""
        
        utilization = {}
        
        if not schedule:
            return {agent.agent_id: 0.0 for agent in agents}
        
        total_time = max(entry['end_time'] for entry in schedule)
        
        for agent in agents:
            agent_tasks = [entry for entry in schedule if entry['agent_id'] == agent.agent_id]
            
            if agent_tasks:
                busy_time = sum(entry['duration'] for entry in agent_tasks)
                utilization[agent.agent_id] = busy_time / total_time if total_time > 0 else 0.0
            else:
                utilization[agent.agent_id] = 0.0
        
        return utilization
    
    async def _generate_optimization_recommendations(self, 
                                                   workflow: WorkflowDefinition,
                                                   schedule: List[Dict[str, Any]],
                                                   bottlenecks: List[str],
                                                   utilization: Dict[str, float]) -> List[str]:
        """Generate workflow optimization recommendations"""
        
        recommendations = []
        
        # Utilization-based recommendations
        avg_utilization = sum(utilization.values()) / len(utilization) if utilization else 0
        
        if avg_utilization < 0.6:
            recommendations.append("Consider reducing agent pool or increasing parallel task execution")
        elif avg_utilization > 0.9:
            recommendations.append("Consider adding more agent resources to improve throughput")
        
        # Bottleneck-based recommendations
        if bottlenecks:
            recommendations.append(f"Address {len(bottlenecks)} identified bottlenecks to improve flow")
        
        # Task distribution recommendations
        agent_task_counts = defaultdict(int)
        for entry in schedule:
            agent_task_counts[entry['agent_id']] += 1
        
        if agent_task_counts:
            max_tasks = max(agent_task_counts.values())
            min_tasks = min(agent_task_counts.values())
            
            if max_tasks > min_tasks * 2:
                recommendations.append("Rebalance task distribution across agents")
        
        # Critical path recommendations
        critical_tasks = [entry for entry in schedule if entry.get('on_critical_path', False)]
        if len(critical_tasks) > len(schedule) * 0.3:
            recommendations.append("Consider parallelizing critical path tasks where possible")
        
        # Priority-based recommendations
        high_priority_delayed = any(
            entry['start_time'] > entry['priority'] * 10 
            for entry in schedule 
            if entry['priority'] >= TaskPriority.HIGH.value
        )
        
        if high_priority_delayed:
            recommendations.append("Prioritize high-priority tasks to reduce delays")
        
        return recommendations[:5]  # Top 5 recommendations
    
    async def _calculate_optimization_score(self, 
                                          completion_time: int,
                                          utilization: Dict[str, float],
                                          bottleneck_count: int) -> float:
        """Calculate overall optimization score"""
        
        # Time efficiency (lower completion time is better)
        time_score = max(0.0, 1.0 - (completion_time / (24 * 60)))  # Normalize by 24 hours
        
        # Resource efficiency (balanced utilization is better)
        if utilization:
            avg_util = sum(utilization.values()) / len(utilization)
            util_variance = sum((u - avg_util) ** 2 for u in utilization.values()) / len(utilization)
            resource_score = avg_util * (1.0 - util_variance)
        else:
            resource_score = 0.0
        
        # Bottleneck penalty
        bottleneck_score = max(0.0, 1.0 - (bottleneck_count / 10.0))
        
        # Weighted final score
        final_score = (
            time_score * self.optimization_weights['completion_time'] +
            resource_score * self.optimization_weights['resource_efficiency'] +
            bottleneck_score * 0.3
        )
        
        return min(1.0, final_score)
    
    async def monitor_workflow_execution(self, workflow_id: str) -> Dict[str, Any]:
        """Monitor active workflow execution"""
        
        if workflow_id not in self.active_workflows:
            return {'error': 'Workflow not found'}
        
        workflow_data = self.active_workflows[workflow_id]
        
        # Calculate progress metrics
        total_tasks = len(workflow_data['tasks'])
        completed_tasks = len([t for t in workflow_data['tasks'] if t['status'] == WorkflowStatus.COMPLETED])
        running_tasks = len([t for t in workflow_data['tasks'] if t['status'] == WorkflowStatus.RUNNING])
        failed_tasks = len([t for t in workflow_data['tasks'] if t['status'] == WorkflowStatus.FAILED])
        
        progress_percentage = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        # Calculate time metrics
        start_time = workflow_data.get('start_time')
        if start_time:
            elapsed_time = (datetime.now() - datetime.fromisoformat(start_time)).total_seconds() / 60
        else:
            elapsed_time = 0
        
        return {
            'workflow_id': workflow_id,
            'status': workflow_data.get('status', 'unknown'),
            'progress_percentage': progress_percentage,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'running_tasks': running_tasks,
            'failed_tasks': failed_tasks,
            'elapsed_time_minutes': elapsed_time,
            'estimated_remaining_time': workflow_data.get('estimated_completion', 0) - elapsed_time,
            'current_bottlenecks': await self._identify_runtime_bottlenecks(workflow_data),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _identify_runtime_bottlenecks(self, workflow_data: Dict[str, Any]) -> List[str]:
        """Identify runtime bottlenecks in active workflow"""
        
        bottlenecks = []
        
        # Find stuck tasks (running too long)
        current_time = datetime.now()
        for task in workflow_data.get('tasks', []):
            if task.get('status') == WorkflowStatus.RUNNING:
                start_time = task.get('started_at')
                if start_time:
                    runtime = (current_time - datetime.fromisoformat(start_time)).total_seconds() / 60
                    expected_duration = task.get('estimated_duration', 30)
                    
                    if runtime > expected_duration * 1.5:
                        bottlenecks.append(f"Task {task['task_id']} running longer than expected")
        
        # Find dependency blockers
        pending_tasks = [t for t in workflow_data.get('tasks', []) if t.get('status') == WorkflowStatus.PENDING]
        for task in pending_tasks:
            unmet_deps = []
            for dep_id in task.get('dependencies', []):
                dep_task = next((t for t in workflow_data['tasks'] if t['task_id'] == dep_id), None)
                if dep_task and dep_task.get('status') != WorkflowStatus.COMPLETED:
                    unmet_deps.append(dep_id)
            
            if unmet_deps:
                bottlenecks.append(f"Task {task['task_id']} blocked by dependencies: {unmet_deps}")
        
        return bottlenecks


# Utility functions
async def create_simple_workflow(task_configs: List[Dict[str, Any]], workflow_name: str = "Simple Workflow") -> WorkflowDefinition:
    """Create a simple workflow from task configurations"""
    
    tasks = []
    for i, config in enumerate(task_configs):
        task = WorkflowTask(
            task_id=f"task_{i+1}",
            task_name=config.get('name', f'Task {i+1}'),
            agent_id=config.get('agent_id', 'default_agent'),
            dependencies=config.get('dependencies', []),
            estimated_duration=config.get('duration', 30),
            priority=TaskPriority(config.get('priority', 2)),
            deadline=config.get('deadline'),
            resource_requirements=config.get('resources', {}),
            input_data=config.get('input', {}),
            output_schema=config.get('output_schema', {}),
            retry_count=0,
            max_retries=config.get('max_retries', 2),
            status=WorkflowStatus.PENDING,
            created_at=datetime.now().isoformat(),
            started_at=None,
            completed_at=None
        )
        tasks.append(task)
    
    return WorkflowDefinition(
        workflow_id=f"workflow_{datetime.now().timestamp()}",
        workflow_name=workflow_name,
        description="Auto-generated workflow",
        tasks=tasks,
        global_timeout=480,  # 8 hours
        parallel_limit=5,
        retry_policy={'max_retries': 2, 'backoff_factor': 2},
        success_criteria={'completion_rate': 0.95},
        failure_handling={'stop_on_critical_failure': True},
        created_at=datetime.now().isoformat()
    )
