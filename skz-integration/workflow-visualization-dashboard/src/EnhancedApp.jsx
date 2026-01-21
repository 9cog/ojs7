import React, { useState, useEffect, useMemo } from 'react';
import * as d3 from 'd3';
import * as anime from 'animejs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Badge } from './components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Progress } from './components/ui/progress';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Zap, 
  Users, 
  FileText, 
  CheckCircle, 
  TrendingUp,
  Network,
  Brain,
  Activity,
  BarChart3,
  Monitor,
  Globe,
  GitBranch,
  Bell
} from 'lucide-react';
import './App.css';

// Import enhanced real-time components
import { useWebSocket } from './hooks/useWebSocket';
import { RealtimeStatus } from './components/realtime/RealtimeStatus';
import { NotificationsPanel } from './components/realtime/NotificationsPanel';
import { RealtimeAgentCard } from './components/realtime/RealtimeAgentCard';

// Import existing components
import OJSIntegration from './components/OJSIntegration';
import AgentMonitoring from './components/AgentMonitoring';
import WorkflowVisualization from './components/WorkflowVisualization';

// Import process flow diagrams
import agent1Image from './assets/agent1_research_discovery.png';
import agent2Image from './assets/agent2_submission_assistant.png';
import agent3Image from './assets/agent3_editorial_orchestration.png';
import agentInteractionImage from './assets/agent_interaction_overview.png';
import cognitiveArchitectureImage from './assets/cognitive_architecture_diagram.png';

const App = () => {
  const [activeTab, setActiveTab] = useState('realtime');
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [simulationProgress, setSimulationProgress] = useState(0);
  const [subscribedAgents, setSubscribedAgents] = useState(new Set());
  const [lastUpdate, setLastUpdate] = useState(null);

  // Initialize WebSocket connection for real-time updates
  const {
    socket,
    connected,
    agentStatuses,
    notifications,
    connectionStats,
    subscribeToAgent,
    unsubscribeFromAgent,
    sendTestNotification,
    clearNotifications,
    error
  } = useWebSocket('http://localhost:5000');

  // Enhanced agent data with fallback to static data
  const agentData = useMemo(() => {
    const staticAgents = {
      research_discovery: {
        name: 'Research Discovery Agent',
        type: 'Cosmetic Intelligence',
        icon: <Zap className="w-6 h-6" />,
        color: '#e1f5fe',
        borderColor: '#01579b',
        capabilities: ['INCI Database Mining', 'Patent Analysis', 'Trend Identification'],
        currentTask: 'Analyzing emerging peptide compounds'
      },
      submission_assistant: {
        name: 'Submission Assistant Agent',
        type: 'Manuscript Intelligence',
        icon: <FileText className="w-6 h-6" />,
        color: '#f3e5f5',
        borderColor: '#4a148c',
        capabilities: ['Quality Assessment', 'INCI Verification', 'Safety Compliance'],
        currentTask: 'Processing manuscript MS-2024-156'
      },
      editorial_orchestration: {
        name: 'Editorial Orchestration Agent',
        type: 'Publication Intelligence',
        icon: <Users className="w-6 h-6" />,
        color: '#fff3e0',
        borderColor: '#e65100',
        capabilities: ['Workflow Coordination', 'Decision Making', 'Resource Allocation'],
        currentTask: 'Coordinating 12 active reviews'
      },
      review_coordination: {
        name: 'Review Coordination Agent',
        type: 'Peer Review Intelligence',
        icon: <CheckCircle className="w-6 h-6" />,
        color: '#e8f5e8',
        borderColor: '#1b5e20',
        capabilities: ['Reviewer Matching', 'Expertise Assessment', 'Workload Management'],
        currentTask: 'Matching reviewers for anti-aging study'
      },
      content_quality: {
        name: 'Content Quality Agent',
        type: 'Scientific Intelligence',
        icon: <Activity className="w-6 h-6" />,
        color: '#fff8e1',
        borderColor: '#f57f17',
        capabilities: ['Scientific Validation', 'Safety Assessment', 'Regulatory Compliance'],
        currentTask: 'Final quality check for 3 manuscripts'
      },
      publishing_production: {
        name: 'Publishing Production Agent',
        type: 'Content Intelligence',
        icon: <TrendingUp className="w-6 h-6" />,
        color: '#fce4ec',
        borderColor: '#880e4f',
        capabilities: ['Content Formatting', 'Visual Generation', 'Multi-Channel Distribution'],
        currentTask: 'Producing feature article on retinol alternatives'
      },
      analytics_monitoring: {
        name: 'Analytics & Monitoring Agent',
        type: 'Intelligence Intelligence',
        icon: <BarChart3 className="w-6 h-6" />,
        color: '#f1f8e9',
        borderColor: '#33691e',
        capabilities: ['Performance Analytics', 'Trend Forecasting', 'Optimization Recommendations'],
        currentTask: 'Generating weekly performance report'
      }
    };

    // Merge real-time data with static data
    return Object.keys(staticAgents).map(agentType => ({
      id: agentType,
      ...staticAgents[agentType],
      status: agentStatuses[agentType]?.status || 'active',
      performance: {
        efficiency: agentStatuses[agentType]?.performance?.success_rate || 0.9,
        accuracy: Math.random() * 0.1 + 0.85,
        totalActions: agentStatuses[agentType]?.performance?.total_actions || Math.floor(Math.random() * 1000) + 500,
        successRate: agentStatuses[agentType]?.performance?.success_rate || 0.9
      },
      cpu_usage: agentStatuses[agentType]?.cpu_usage,
      memory_usage: agentStatuses[agentType]?.memory_usage,
      active_tasks: agentStatuses[agentType]?.active_tasks,
      last_action: agentStatuses[agentType]?.last_action,
      lastUpdate: agentStatuses[agentType]?.last_action
    }));
  }, [agentStatuses]);

  // Update last update timestamp when agent statuses change
  useEffect(() => {
    if (Object.keys(agentStatuses).length > 0) {
      setLastUpdate(new Date().toLocaleTimeString());
    }
  }, [agentStatuses]);

  // Agent subscription handlers
  const handleSubscribeAgent = (agentId) => {
    subscribeToAgent(agentId);
    setSubscribedAgents(prev => new Set([...prev, agentId]));
  };

  const handleUnsubscribeAgent = (agentId) => {
    unsubscribeFromAgent(agentId);
    setSubscribedAgents(prev => {
      const newSet = new Set(prev);
      newSet.delete(agentId);
      return newSet;
    });
  };

  // Workflow simulation data
  const workflowSteps = [
    { id: 1, name: 'Research Discovery', agent: 'research_discovery', duration: 2000, status: 'completed' },
    { id: 2, name: 'Manuscript Submission', agent: 'submission_assistant', duration: 3000, status: 'completed' },
    { id: 3, name: 'Editorial Review', agent: 'editorial_orchestration', duration: 1500, status: 'in_progress' },
    { id: 4, name: 'Peer Review Assignment', agent: 'review_coordination', duration: 2500, status: 'pending' },
    { id: 5, name: 'Quality Assessment', agent: 'content_quality', duration: 2000, status: 'pending' },
    { id: 6, name: 'Publication Production', agent: 'publishing_production', duration: 3500, status: 'pending' },
    { id: 7, name: 'Performance Analysis', agent: 'analytics_monitoring', duration: 1000, status: 'continuous' }
  ];

  const runSimulation = () => {
    setSimulationRunning(true);
    setSimulationProgress(0);

    // Animate progress
    anime({
      targets: { progress: 0 },
      progress: 100,
      duration: 10000,
      easing: 'easeInOutQuad',
      update: function(anim) {
        setSimulationProgress(Math.round(anim.animatables[0].target.progress));
      },
      complete: function() {
        setSimulationRunning(false);
      }
    });
  };

  const resetSimulation = () => {
    setSimulationRunning(false);
    setSimulationProgress(0);
  };

  // Calculate system statistics from real-time data
  const systemStats = useMemo(() => {
    const totalActions = agentData.reduce((sum, agent) => sum + (agent.performance?.totalActions || 0), 0);
    const avgSuccessRate = agentData.reduce((sum, agent) => sum + (agent.performance?.successRate || 0), 0) / agentData.length;
    const avgResponseTime = Object.values(agentStatuses).reduce((sum, status) => sum + (status.performance?.avg_response_time || 0), 0) / Math.max(Object.keys(agentStatuses).length, 1);
    const activeWorkflows = connectionStats?.active_agents || 0;

    return {
      totalActions,
      avgSuccessRate,
      avgResponseTime,
      activeWorkflows
    };
  }, [agentData, agentStatuses, connectionStats]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-3">
              <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Skin Zone Journal</h1>
                <p className="text-sm text-gray-600">Real-time Autonomous Agents Workflow</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant={connected ? "default" : "secondary"} className={connected ? "bg-green-50 text-green-700 border-green-200" : ""}>
                <Activity className="w-3 h-3 mr-1" />
                {connected ? 'Live Updates' : 'Offline'}
              </Badge>
              <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                {agentData.length} Agents Active
              </Badge>
              <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-200">
                <Bell className="w-3 h-3 mr-1" />
                {notifications.length} Notifications
              </Badge>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-7">
            <TabsTrigger value="realtime">Real-time</TabsTrigger>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="agents">Agents</TabsTrigger>
            <TabsTrigger value="workflows">Workflows</TabsTrigger>
            <TabsTrigger value="network">Network</TabsTrigger>
            <TabsTrigger value="cognitive">Cognitive</TabsTrigger>
            <TabsTrigger value="diagrams">Diagrams</TabsTrigger>
          </TabsList>

          {/* Real-time Tab */}
          <TabsContent value="realtime" className="space-y-6">
            {/* Real-time Status */}
            <RealtimeStatus 
              connected={connected}
              connectionStats={connectionStats}
              error={error}
              lastUpdate={lastUpdate}
            />

            {/* Real-time Agent Cards */}
            <div>
              <h2 className="text-xl font-semibold mb-4">Live Agent Status</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {agentData.map((agent) => (
                  <RealtimeAgentCard
                    key={agent.id}
                    agentType={agent.id}
                    agentStatus={{
                      id: agent.id,
                      status: agent.status,
                      cpu_usage: agent.cpu_usage,
                      memory_usage: agent.memory_usage,
                      active_tasks: agent.active_tasks,
                      last_action: agent.last_action,
                      performance: {
                        success_rate: agent.performance.successRate,
                        avg_response_time: agentStatuses[agent.id]?.performance?.avg_response_time || 2.0,
                        total_actions: agent.performance.totalActions
                      }
                    }}
                    onSubscribe={handleSubscribeAgent}
                    onUnsubscribe={handleUnsubscribeAgent}
                    isSubscribed={subscribedAgents.has(agent.id)}
                    lastUpdated={agent.lastUpdate}
                  />
                ))}
              </div>
            </div>

            {/* Real-time Notifications */}
            <NotificationsPanel
              notifications={notifications}
              onClear={clearNotifications}
              onTestNotification={sendTestNotification}
            />
          </TabsContent>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Actions</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemStats.totalActions.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground">
                    {connected ? 'Live data' : '+12% from last week'}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
                  <CheckCircle className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{Math.round(systemStats.avgSuccessRate * 100)}%</div>
                  <p className="text-xs text-muted-foreground">
                    {connected ? 'Real-time average' : '+2.1% from last week'}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Avg Response Time</CardTitle>
                  <Zap className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemStats.avgResponseTime.toFixed(1)}s</div>
                  <p className="text-xs text-muted-foreground">
                    {connected ? 'Live monitoring' : '-0.3s from last week'}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
                  <Network className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{systemStats.activeWorkflows}</div>
                  <p className="text-xs text-muted-foreground">
                    {connected ? 'Connected now' : '+5 from yesterday'}
                  </p>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Real-time System Status</CardTitle>
                <CardDescription>
                  Live monitoring of the 7 autonomous agents with WebSocket updates
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-3 text-orange-600">Hierarchical Management</h4>
                    <div className="space-y-2">
                      {['editorial_orchestration', 'content_quality', 'analytics_monitoring'].map(agentId => {
                        const agent = agentData.find(a => a.id === agentId);
                        const isActive = agentStatuses[agentId]?.status === 'active';
                        return (
                          <div key={agentId} className="flex items-center space-x-2">
                            <div className={`w-3 h-3 rounded-full ${isActive ? 'bg-green-400 animate-pulse' : 'bg-orange-200'}`}></div>
                            <span className="text-sm">{agent?.name}</span>
                            <Badge variant="outline" className="text-xs">
                              {agentStatuses[agentId]?.status || 'unknown'}
                            </Badge>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-3 text-blue-600">Distributed Networks</h4>
                    <div className="space-y-2">
                      {['research_discovery', 'submission_assistant', 'review_coordination', 'publishing_production'].map(agentId => {
                        const agent = agentData.find(a => a.id === agentId);
                        const isActive = agentStatuses[agentId]?.status === 'active';
                        return (
                          <div key={agentId} className="flex items-center space-x-2">
                            <div className={`w-3 h-3 rounded-full ${isActive ? 'bg-green-400 animate-pulse' : 'bg-blue-200'}`}></div>
                            <span className="text-sm">{agent?.name}</span>
                            <Badge variant="outline" className="text-xs">
                              {agentStatuses[agentId]?.status || 'unknown'}
                            </Badge>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Agents Tab - Enhanced with real-time data */}
          <TabsContent value="agents" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              {agentData.map((agent) => (
                <Card key={agent.id} className="hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div 
                          className="p-2 rounded-lg"
                          style={{ backgroundColor: agent.color, borderColor: agent.borderColor, borderWidth: '2px' }}
                        >
                          {agent.icon}
                        </div>
                        <div>
                          <CardTitle className="text-lg">{agent.name}</CardTitle>
                          <CardDescription>{agent.type}</CardDescription>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant={agent.status === 'active' ? 'default' : 'secondary'}>
                          {agent.status}
                        </Badge>
                        {connected && (
                          <div className={`w-2 h-2 rounded-full ${agent.status === 'active' ? 'bg-green-400 animate-pulse' : 'bg-gray-300'}`} />
                        )}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Efficiency</span>
                        <span>{Math.round(agent.performance.efficiency * 100)}%</span>
                      </div>
                      <Progress value={agent.performance.efficiency * 100} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Accuracy</span>
                        <span>{Math.round(agent.performance.accuracy * 100)}%</span>
                      </div>
                      <Progress value={agent.performance.accuracy * 100} className="h-2" />
                    </div>
                    
                    {/* Real-time metrics */}
                    {connected && (agent.cpu_usage || agent.memory_usage) && (
                      <div className="border-t pt-3">
                        <p className="text-sm font-medium text-gray-700 mb-2">Live System Metrics</p>
                        {agent.cpu_usage && (
                          <div className="mb-2">
                            <div className="flex justify-between text-sm mb-1">
                              <span>CPU Usage</span>
                              <span>{agent.cpu_usage.toFixed(1)}%</span>
                            </div>
                            <Progress value={agent.cpu_usage} className="h-2" />
                          </div>
                        )}
                        {agent.memory_usage && (
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span>Memory Usage</span>
                              <span>{agent.memory_usage.toFixed(1)}%</span>
                            </div>
                            <Progress value={agent.memory_usage} className="h-2" />
                          </div>
                        )}
                      </div>
                    )}
                    
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Total Actions</span>
                        <div className="font-semibold">{agent.performance.totalActions.toLocaleString()}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Success Rate</span>
                        <div className="font-semibold">{Math.round(agent.performance.successRate * 100)}%</div>
                      </div>
                    </div>
                    <div>
                      <span className="text-sm text-muted-foreground">Current Task</span>
                      <div className="text-sm font-medium">{agent.currentTask}</div>
                      {agent.lastUpdate && (
                        <div className="text-xs text-muted-foreground mt-1">
                          Last update: {typeof agent.lastUpdate === 'string' ? agent.lastUpdate : new Date(agent.lastUpdate).toLocaleTimeString()}
                        </div>
                      )}
                    </div>
                    <div>
                      <span className="text-sm text-muted-foreground mb-2 block">Key Capabilities</span>
                      <div className="flex flex-wrap gap-1">
                        {agent.capabilities.slice(0, 3).map((capability, index) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {capability}
                          </Badge>
                        ))}
                        {agent.capabilities.length > 3 && (
                          <Badge variant="outline" className="text-xs">
                            +{agent.capabilities.length - 3} more
                          </Badge>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Workflows Tab */}
          <TabsContent value="workflows" className="space-y-6">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Workflow Simulation</CardTitle>
                    <CardDescription>
                      Simulate the complete manuscript processing workflow with real-time updates
                    </CardDescription>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button 
                      onClick={runSimulation} 
                      disabled={simulationRunning}
                      className="flex items-center space-x-2"
                    >
                      {simulationRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      <span>{simulationRunning ? 'Running...' : 'Run Simulation'}</span>
                    </Button>
                    <Button variant="outline" onClick={resetSimulation}>
                      <RotateCcw className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Simulation Progress</span>
                      <span>{simulationProgress}%</span>
                    </div>
                    <Progress value={simulationProgress} className="h-3" />
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {workflowSteps.map((step) => {
                      const agent = agentData.find(a => a.id === step.agent);
                      const isActive = agentStatuses[step.agent]?.status === 'active';
                      return (
                        <div 
                          key={step.id}
                          className={`workflow-step-${step.id} p-4 rounded-lg border transition-all duration-300 ${isActive ? 'ring-2 ring-green-400' : ''}`}
                          style={{ 
                            backgroundColor: agent?.color || '#f0f0f0',
                            borderColor: agent?.borderColor || '#ccc'
                          }}
                        >
                          <div className="flex items-center space-x-2 mb-2">
                            {agent?.icon}
                            <span className="font-semibold text-sm">{step.name}</span>
                            {connected && isActive && (
                              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                            )}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Agent: {agent?.name}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Duration: {step.duration}ms
                          </div>
                          <div className="flex items-center space-x-2 mt-2">
                            <Badge 
                              variant={step.status === 'completed' ? 'default' : 
                                     step.status === 'in_progress' ? 'secondary' : 'outline'}
                              className="text-xs"
                            >
                              {step.status.replace('_', ' ')}
                            </Badge>
                            {connected && (
                              <Badge variant="outline" className="text-xs">
                                {agentStatuses[step.agent]?.status || 'offline'}
                              </Badge>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Other existing tabs remain the same */}
          <TabsContent value="network" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Agent Interaction Network</CardTitle>
                <CardDescription>
                  Real-time visualization of agent communication and coordination patterns
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-20">
                  <Network className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                  <p className="text-gray-600">Interactive network visualization</p>
                  <p className="text-sm text-gray-500">Shows real-time agent interactions when connected</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="cognitive" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Cognitive Architecture</CardTitle>
                <CardDescription>
                  Hierarchical priority management balanced with distributed innovation networks
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-20">
                  <Brain className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                  <p className="text-gray-600">Cognitive architecture visualization</p>
                  <p className="text-sm text-gray-500">Real-time cognitive balance monitoring</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="diagrams" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Research Discovery Agent</CardTitle>
                  <CardDescription>Cosmetic intelligence and ingredient discovery workflow</CardDescription>
                </CardHeader>
                <CardContent>
                  <img src={agent1Image} alt="Research Discovery Agent Process Flow" className="w-full rounded-lg" />
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Submission Assistant Agent</CardTitle>
                  <CardDescription>Manuscript processing and quality assessment workflow</CardDescription>
                </CardHeader>
                <CardContent>
                  <img src={agent2Image} alt="Submission Assistant Agent Process Flow" className="w-full rounded-lg" />
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Editorial Orchestration Agent</CardTitle>
                  <CardDescription>Central coordination and editorial management workflow</CardDescription>
                </CardHeader>
                <CardContent>
                  <img src={agent3Image} alt="Editorial Orchestration Agent Process Flow" className="w-full rounded-lg" />
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Agent Interaction Overview</CardTitle>
                  <CardDescription>Complete system interaction and communication patterns</CardDescription>
                </CardHeader>
                <CardContent>
                  <img src={agentInteractionImage} alt="Agent Interaction Overview" className="w-full rounded-lg" />
                </CardContent>
              </Card>
            </div>
            
            <Card>
              <CardHeader>
                <CardTitle>Cognitive Architecture Diagram</CardTitle>
                <CardDescription>
                  Comprehensive view of the hierarchical and distributed cognitive balance
                </CardDescription>
              </CardHeader>
              <CardContent>
                <img src={cognitiveArchitectureImage} alt="Cognitive Architecture Diagram" className="w-full rounded-lg max-w-4xl mx-auto" />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default App;