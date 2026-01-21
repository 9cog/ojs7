import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { 
  Play, 
  Pause, 
  Square, 
  Settings, 
  Activity, 
  Zap, 
  Brain,
  CheckCircle,
  AlertTriangle,
  Clock,
  TrendingUp,
  BarChart3
} from 'lucide-react';
import * as d3 from 'd3';

/**
 * Real-time Agent Monitoring Component
 * Advanced monitoring and control interface for SKZ autonomous agents
 */
const AgentMonitoring = ({ agentData, onAgentAction }) => {
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [realTimeData, setRealTimeData] = useState({});
  const [agentLogs, setAgentLogs] = useState({});
  const performanceRef = useRef(null);

  // Real-time data simulation
  useEffect(() => {
    const interval = setInterval(() => {
      updateRealTimeMetrics();
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Initialize performance visualization
  useEffect(() => {
    if (selectedAgent && performanceRef.current) {
      createPerformanceVisualization();
    }
  }, [selectedAgent]);

  const updateRealTimeMetrics = () => {
    const newData = {};
    agentData.forEach(agent => {
      newData[agent.id] = {
        cpuUsage: Math.random() * 100,
        memoryUsage: Math.random() * 100,
        activeConnections: Math.floor(Math.random() * 50) + 1,
        responseTime: Math.random() * 500 + 50,
        throughput: Math.floor(Math.random() * 100) + 20,
        errorRate: Math.random() * 5,
        lastActivity: new Date(),
        queueSize: Math.floor(Math.random() * 20)
      };
    });
    setRealTimeData(newData);
  };

  const createPerformanceVisualization = () => {
    const container = d3.select(performanceRef.current);
    container.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 400 - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;

    const svg = container
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom);

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Generate sample performance data
    const performanceData = Array.from({ length: 20 }, (_, i) => ({
      time: i,
      efficiency: 0.7 + Math.random() * 0.3,
      accuracy: 0.8 + Math.random() * 0.2
    }));

    const xScale = d3.scaleLinear()
      .domain([0, 19])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    const line = d3.line()
      .x(d => xScale(d.time))
      .y(d => yScale(d.efficiency))
      .curve(d3.curveMonotoneX);

    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append('g')
      .call(d3.axisLeft(yScale));

    // Add efficiency line
    g.append('path')
      .datum(performanceData)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Add accuracy line
    const accuracyLine = d3.line()
      .x(d => xScale(d.time))
      .y(d => yScale(d.accuracy))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(performanceData)
      .attr('fill', 'none')
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2)
      .attr('d', accuracyLine);
  };

  const handleAgentAction = (agentId, action) => {
    if (onAgentAction) {
      onAgentAction(agentId, action);
    }
    
    // Add to agent logs
    const newLog = {
      timestamp: new Date(),
      action: action,
      status: 'success',
      message: `Agent ${action} command executed successfully`
    };

    setAgentLogs(prev => ({
      ...prev,
      [agentId]: [...(prev[agentId] || []), newLog].slice(-10) // Keep last 10 logs
    }));
  };

  const getAgentStatusColor = (status) => {
    const colors = {
      'active': 'bg-green-100 text-green-800 border-green-200',
      'idle': 'bg-yellow-100 text-yellow-800 border-yellow-200',
      'error': 'bg-red-100 text-red-800 border-red-200',
      'maintenance': 'bg-blue-100 text-blue-800 border-blue-200'
    };
    return colors[status] || colors.idle;
  };

  const getHealthScore = (agent) => {
    const metrics = realTimeData[agent.id];
    if (!metrics) return 85;
    
    const cpuScore = Math.max(0, 100 - metrics.cpuUsage);
    const memoryScore = Math.max(0, 100 - metrics.memoryUsage);
    const errorScore = Math.max(0, 100 - metrics.errorRate * 20);
    const responseScore = Math.max(0, 100 - metrics.responseTime / 5);
    
    return Math.round((cpuScore + memoryScore + errorScore + responseScore) / 4);
  };

  return (
    <div className="space-y-6">
      {/* Agent Grid Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {agentData.map((agent) => {
          const metrics = realTimeData[agent.id] || {};
          const healthScore = getHealthScore(agent);
          
          return (
            <Card 
              key={agent.id} 
              className={`cursor-pointer transition-all hover:shadow-lg ${
                selectedAgent?.id === agent.id ? 'ring-2 ring-blue-500' : ''
              }`}
              onClick={() => setSelectedAgent(agent)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {agent.icon}
                    <div>
                      <CardTitle className="text-sm">{agent.name}</CardTitle>
                      <CardDescription className="text-xs">{agent.type}</CardDescription>
                    </div>
                  </div>
                  <Badge className={getAgentStatusColor(agent.status)}>
                    {agent.status}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-3">
                  {/* Health Score */}
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Health Score</span>
                      <span className={`font-medium ${
                        healthScore >= 80 ? 'text-green-600' : 
                        healthScore >= 60 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {healthScore}%
                      </span>
                    </div>
                    <Progress value={healthScore} className="h-2" />
                  </div>

                  {/* Real-time Metrics */}
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-500">CPU:</span>
                      <span className="ml-1 font-medium">{metrics.cpuUsage?.toFixed(1) || '0'}%</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Memory:</span>
                      <span className="ml-1 font-medium">{metrics.memoryUsage?.toFixed(1) || '0'}%</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Queue:</span>
                      <span className="ml-1 font-medium">{metrics.queueSize || 0}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Response:</span>
                      <span className="ml-1 font-medium">{metrics.responseTime?.toFixed(0) || '0'}ms</span>
                    </div>
                  </div>

                  {/* Current Task */}
                  <div className="text-xs">
                    <span className="text-gray-500">Current:</span>
                    <p className="font-medium text-gray-800 truncate" title={agent.currentTask}>
                      {agent.currentTask}
                    </p>
                  </div>

                  {/* Quick Actions */}
                  <div className="flex gap-1">
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="flex-1 text-xs h-7"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAgentAction(agent.id, 'start');
                      }}
                    >
                      <Play className="w-3 h-3 mr-1" />
                      Start
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="flex-1 text-xs h-7"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAgentAction(agent.id, 'pause');
                      }}
                    >
                      <Pause className="w-3 h-3 mr-1" />
                      Pause
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="text-xs h-7 px-2"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAgentAction(agent.id, 'configure');
                      }}
                    >
                      <Settings className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Detailed Agent View */}
      {selectedAgent && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Agent Details */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {selectedAgent.icon}
                {selectedAgent.name} - Detailed View
              </CardTitle>
              <CardDescription>
                Comprehensive monitoring and control interface
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Performance Metrics */}
                <div>
                  <h4 className="font-medium mb-2">Performance Metrics</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-sm text-gray-500">Efficiency</span>
                      <div className="text-2xl font-bold text-blue-600">
                        {(selectedAgent.performance.efficiency * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <span className="text-sm text-gray-500">Accuracy</span>
                      <div className="text-2xl font-bold text-green-600">
                        {(selectedAgent.performance.accuracy * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <span className="text-sm text-gray-500">Total Actions</span>
                      <div className="text-2xl font-bold text-purple-600">
                        {selectedAgent.performance.totalActions.toLocaleString()}
                      </div>
                    </div>
                    <div>
                      <span className="text-sm text-gray-500">Success Rate</span>
                      <div className="text-2xl font-bold text-orange-600">
                        {(selectedAgent.performance.successRate * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>

                {/* Capabilities */}
                <div>
                  <h4 className="font-medium mb-2">Capabilities</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedAgent.capabilities.map((capability) => (
                      <Badge key={capability} variant="outline" className="text-xs">
                        {capability}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Control Actions */}
                <div>
                  <h4 className="font-medium mb-2">Agent Controls</h4>
                  <div className="flex gap-2">
                    <Button 
                      size="sm" 
                      onClick={() => handleAgentAction(selectedAgent.id, 'start')}
                      className="flex items-center gap-1"
                    >
                      <Play className="w-4 h-4" />
                      Start
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleAgentAction(selectedAgent.id, 'pause')}
                      className="flex items-center gap-1"
                    >
                      <Pause className="w-4 h-4" />
                      Pause
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleAgentAction(selectedAgent.id, 'stop')}
                      className="flex items-center gap-1"
                    >
                      <Square className="w-4 h-4" />
                      Stop
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleAgentAction(selectedAgent.id, 'restart')}
                      className="flex items-center gap-1"
                    >
                      <Activity className="w-4 h-4" />
                      Restart
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Performance Chart & Logs */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                Performance Trends & Activity Logs
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Performance Chart */}
                <div>
                  <h4 className="font-medium mb-2">Performance Trends</h4>
                  <div ref={performanceRef} className="w-full"></div>
                </div>

                {/* Activity Logs */}
                <div>
                  <h4 className="font-medium mb-2">Recent Activity</h4>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {(agentLogs[selectedAgent.id] || []).reverse().map((log, index) => (
                      <div key={index} className="flex items-center gap-2 text-sm p-2 bg-gray-50 rounded">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span className="text-gray-500">
                          {log.timestamp.toLocaleTimeString()}
                        </span>
                        <span>{log.message}</span>
                      </div>
                    ))}
                    {(!agentLogs[selectedAgent.id] || agentLogs[selectedAgent.id].length === 0) && (
                      <div className="text-center py-4 text-gray-500 text-sm">
                        No recent activity logs
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};

export default AgentMonitoring;
