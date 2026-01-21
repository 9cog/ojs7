import React, { useState, useEffect, useRef } from 'react';
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
  GitBranch
} from 'lucide-react';
import './App.css';

// Import our new enhanced components
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
  const [activeTab, setActiveTab] = useState('overview');
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [simulationProgress, setSimulationProgress] = useState(0);
  const networkRef = useRef(null);
  const cognitiveRef = useRef(null);

  // Agent data with enhanced metrics
  const agentData = [
    {
      id: 'research_discovery',
      name: 'Research Discovery Agent',
      type: 'Cosmetic Intelligence',
      icon: <Zap className="w-6 h-6" />,
      color: '#e1f5fe',
      borderColor: '#01579b',
      status: 'active',
      performance: {
        efficiency: 0.91,
        accuracy: 0.88,
        totalActions: 1247,
        successRate: 0.94
      },
      capabilities: [
        'INCI Database Mining',
        'Patent Analysis',
        'Trend Identification',
        'Regulatory Monitoring',
        'Market Analysis'
      ],
      currentTask: 'Analyzing emerging peptide compounds',
      lastUpdate: '2 minutes ago'
    },
    {
      id: 'submission_assistant',
      name: 'Submission Assistant Agent',
      type: 'Manuscript Intelligence',
      icon: <FileText className="w-6 h-6" />,
      color: '#f3e5f5',
      borderColor: '#4a148c',
      status: 'active',
      performance: {
        efficiency: 0.85,
        accuracy: 0.92,
        totalActions: 892,
        successRate: 0.89
      },
      capabilities: [
        'Quality Assessment',
        'INCI Verification',
        'Safety Compliance',
        'Statistical Review',
        'Enhancement Suggestions'
      ],
      currentTask: 'Processing manuscript MS-2024-156',
      lastUpdate: '5 minutes ago'
    },
    {
      id: 'editorial_orchestration',
      name: 'Editorial Orchestration Agent',
      type: 'Publication Intelligence',
      icon: <Users className="w-6 h-6" />,
      color: '#fff3e0',
      borderColor: '#e65100',
      status: 'active',
      performance: {
        efficiency: 0.89,
        accuracy: 0.94,
        totalActions: 634,
        successRate: 0.96
      },
      capabilities: [
        'Workflow Coordination',
        'Decision Making',
        'Resource Allocation',
        'Conflict Resolution',
        'Strategic Planning'
      ],
      currentTask: 'Coordinating 12 active reviews',
      lastUpdate: '1 minute ago'
    },
    {
      id: 'review_coordination',
      name: 'Review Coordination Agent',
      type: 'Peer Review Intelligence',
      icon: <CheckCircle className="w-6 h-6" />,
      color: '#e8f5e8',
      borderColor: '#1b5e20',
      status: 'active',
      performance: {
        efficiency: 0.87,
        accuracy: 0.91,
        totalActions: 445,
        successRate: 0.93
      },
      capabilities: [
        'Reviewer Matching',
        'Expertise Assessment',
        'Workload Management',
        'Quality Monitoring',
        'Consensus Building'
      ],
      currentTask: 'Matching reviewers for anti-aging study',
      lastUpdate: '3 minutes ago'
    },
    {
      id: 'content_quality',
      name: 'Content Quality Agent',
      type: 'Scientific Intelligence',
      icon: <Activity className="w-6 h-6" />,
      color: '#fff8e1',
      borderColor: '#f57f17',
      status: 'active',
      performance: {
        efficiency: 0.93,
        accuracy: 0.96,
        totalActions: 378,
        successRate: 0.98
      },
      capabilities: [
        'Scientific Validation',
        'Safety Assessment',
        'Regulatory Compliance',
        'Methodology Review',
        'Standards Enforcement'
      ],
      currentTask: 'Final quality check for 3 manuscripts',
      lastUpdate: '4 minutes ago'
    },
    {
      id: 'publishing_production',
      name: 'Publishing Production Agent',
      type: 'Content Intelligence',
      icon: <TrendingUp className="w-6 h-6" />,
      color: '#fce4ec',
      borderColor: '#880e4f',
      status: 'active',
      performance: {
        efficiency: 0.88,
        accuracy: 0.89,
        totalActions: 267,
        successRate: 0.91
      },
      capabilities: [
        'Content Formatting',
        'Visual Generation',
        'Multi-Channel Distribution',
        'Regulatory Reporting',
        'Industry Briefing'
      ],
      currentTask: 'Producing feature article on retinol alternatives',
      lastUpdate: '6 minutes ago'
    },
    {
      id: 'analytics_monitoring',
      name: 'Analytics & Monitoring Agent',
      type: 'Intelligence Intelligence',
      icon: <BarChart3 className="w-6 h-6" />,
      color: '#f1f8e9',
      borderColor: '#33691e',
      status: 'active',
      performance: {
        efficiency: 0.95,
        accuracy: 0.93,
        totalActions: 1856,
        successRate: 0.97
      },
      capabilities: [
        'Performance Analytics',
        'Trend Forecasting',
        'Optimization Recommendations',
        'Strategic Insights',
        'Continuous Learning'
      ],
      currentTask: 'Generating weekly performance report',
      lastUpdate: '30 seconds ago'
    }
  ];

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

  // Initialize D3 network visualization
  useEffect(() => {
    if (activeTab === 'network' && networkRef.current) {
      createNetworkVisualization();
    }
  }, [activeTab]);

  // Initialize cognitive architecture visualization
  useEffect(() => {
    if (activeTab === 'cognitive' && cognitiveRef.current) {
      createCognitiveVisualization();
    }
  }, [activeTab]);

  const createNetworkVisualization = () => {
    const container = d3.select(networkRef.current);
    container.selectAll('*').remove();

    const width = 800;
    const height = 600;
    const svg = container.append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .style('background', 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)');

    // Create nodes data
    const nodes = agentData.map((agent, i) => ({
      id: agent.id,
      name: agent.name,
      type: agent.type,
      color: agent.color,
      borderColor: agent.borderColor,
      x: width / 2 + Math.cos(i * 2 * Math.PI / agentData.length) * 200,
      y: height / 2 + Math.sin(i * 2 * Math.PI / agentData.length) * 200,
      performance: agent.performance.efficiency
    }));

    // Create links data (agent interactions)
    const links = [
      { source: 'research_discovery', target: 'submission_assistant', strength: 0.9 },
      { source: 'submission_assistant', target: 'editorial_orchestration', strength: 0.8 },
      { source: 'editorial_orchestration', target: 'review_coordination', strength: 0.85 },
      { source: 'review_coordination', target: 'content_quality', strength: 0.7 },
      { source: 'content_quality', target: 'publishing_production', strength: 0.9 },
      { source: 'publishing_production', target: 'analytics_monitoring', strength: 0.8 },
      { source: 'analytics_monitoring', target: 'editorial_orchestration', strength: 0.75 }
    ];

    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(150))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2));

    // Create links
    const link = svg.append('g')
      .selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', d => Math.sqrt(d.strength * 10));

    // Create nodes
    const node = svg.append('g')
      .selectAll('circle')
      .data(nodes)
      .enter().append('circle')
      .attr('r', d => 20 + d.performance * 20)
      .attr('fill', d => d.color)
      .attr('stroke', d => d.borderColor)
      .attr('stroke-width', 3)
      .style('cursor', 'pointer')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    // Add labels
    const labels = svg.append('g')
      .selectAll('text')
      .data(nodes)
      .enter().append('text')
      .text(d => d.name.split(' ')[0])
      .attr('font-size', 12)
      .attr('font-weight', 'bold')
      .attr('text-anchor', 'middle')
      .attr('dy', 4)
      .style('pointer-events', 'none');

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);

      labels
        .attr('x', d => d.x)
        .attr('y', d => d.y);
    });

    // Drag functions
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // Add interaction animations
    node.on('mouseover', function(event, d) {
      d3.select(this)
        .transition()
        .duration(200)
        .attr('r', d => 25 + d.performance * 25);
    })
    .on('mouseout', function(event, d) {
      d3.select(this)
        .transition()
        .duration(200)
        .attr('r', d => 20 + d.performance * 20);
    });
  };

  const createCognitiveVisualization = () => {
    const container = d3.select(cognitiveRef.current);
    container.selectAll('*').remove();

    const width = 800;
    const height = 600;
    const svg = container.append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .style('background', 'linear-gradient(45deg, #667eea 0%, #764ba2 100%)');

    // Hierarchical agents (top layer)
    const hierarchicalAgents = [
      { name: 'Editorial Orchestration', x: width/2, y: 150, color: '#ffecb3' },
      { name: 'Content Quality', x: width/2 - 150, y: 250, color: '#fff3e0' },
      { name: 'Analytics & Monitoring', x: width/2 + 150, y: 250, color: '#fff8e1' }
    ];

    // Distributed agents (bottom layer)
    const distributedAgents = [
      { name: 'Research Discovery', x: width/2 - 200, y: 450, color: '#e1f5fe' },
      { name: 'Submission Assistant', x: width/2 - 67, y: 450, color: '#f3e5f5' },
      { name: 'Review Coordination', x: width/2 + 67, y: 450, color: '#e8f5e8' },
      { name: 'Publishing Production', x: width/2 + 200, y: 450, color: '#fce4ec' }
    ];

    // Create hierarchical layer
    const hierarchicalGroup = svg.append('g').attr('class', 'hierarchical');
    hierarchicalGroup.selectAll('circle')
      .data(hierarchicalAgents)
      .enter().append('circle')
      .attr('cx', d => d.x)
      .attr('cy', d => d.y)
      .attr('r', 40)
      .attr('fill', d => d.color)
      .attr('stroke', '#f57c00')
      .attr('stroke-width', 4)
      .style('opacity', 0.9);

    hierarchicalGroup.selectAll('text')
      .data(hierarchicalAgents)
      .enter().append('text')
      .attr('x', d => d.x)
      .attr('y', d => d.y)
      .attr('text-anchor', 'middle')
      .attr('dy', 4)
      .attr('font-size', 10)
      .attr('font-weight', 'bold')
      .attr('fill', '#333')
      .text(d => d.name.split(' ')[0]);

    // Create distributed layer
    const distributedGroup = svg.append('g').attr('class', 'distributed');
    distributedGroup.selectAll('circle')
      .data(distributedAgents)
      .enter().append('circle')
      .attr('cx', d => d.x)
      .attr('cy', d => d.y)
      .attr('r', 35)
      .attr('fill', d => d.color)
      .attr('stroke', '#0277bd')
      .attr('stroke-width', 3)
      .style('opacity', 0.8);

    distributedGroup.selectAll('text')
      .data(distributedAgents)
      .enter().append('text')
      .attr('x', d => d.x)
      .attr('y', d => d.y)
      .attr('text-anchor', 'middle')
      .attr('dy', 4)
      .attr('font-size', 9)
      .attr('font-weight', 'bold')
      .attr('fill', '#333')
      .text(d => d.name.split(' ')[0]);

    // Add balance indicators
    svg.append('text')
      .attr('x', width/2)
      .attr('y', 50)
      .attr('text-anchor', 'middle')
      .attr('font-size', 18)
      .attr('font-weight', 'bold')
      .attr('fill', 'white')
      .text('Cognitive Architecture Balance');

    svg.append('text')
      .attr('x', width/2)
      .attr('y', 100)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('fill', 'white')
      .text('Hierarchical Priority Management ↑');

    svg.append('text')
      .attr('x', width/2)
      .attr('y', 520)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('fill', 'white')
      .text('↓ Distributed Innovation Networks');
  };

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

    // Animate workflow steps
    workflowSteps.forEach((step, index) => {
      setTimeout(() => {
        anime({
          targets: `.workflow-step-${step.id}`,
          scale: [1, 1.1, 1],
          backgroundColor: ['#f0f0f0', '#4ade80', '#f0f0f0'],
          duration: 1000,
          easing: 'easeInOutQuad'
        });
      }, index * 1000);
    });
  };

  const resetSimulation = () => {
    setSimulationRunning(false);
    setSimulationProgress(0);
  };

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
                <p className="text-sm text-gray-600">7 Autonomous Agents Workflow Visualization</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                <Activity className="w-3 h-3 mr-1" />
                System Operational
              </Badge>
              <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                7 Agents Active
              </Badge>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-6">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="agents">Agents</TabsTrigger>
            <TabsTrigger value="workflows">Workflows</TabsTrigger>
            <TabsTrigger value="network">Network</TabsTrigger>
            <TabsTrigger value="cognitive">Cognitive</TabsTrigger>
            <TabsTrigger value="diagrams">Diagrams</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Actions</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">5,719</div>
                  <p className="text-xs text-muted-foreground">+12% from last week</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
                  <CheckCircle className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">94.2%</div>
                  <p className="text-xs text-muted-foreground">+2.1% from last week</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Avg Response Time</CardTitle>
                  <Zap className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">1.2s</div>
                  <p className="text-xs text-muted-foreground">-0.3s from last week</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Workflows</CardTitle>
                  <Network className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">23</div>
                  <p className="text-xs text-muted-foreground">+5 from yesterday</p>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>System Architecture Overview</CardTitle>
                <CardDescription>
                  Hierarchical priority management balanced with distributed innovation networks
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-3 text-orange-600">Hierarchical Structure</h4>
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-orange-200 rounded-full"></div>
                        <span className="text-sm">Editorial Orchestration Agent</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-orange-200 rounded-full"></div>
                        <span className="text-sm">Content Quality Agent</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-orange-200 rounded-full"></div>
                        <span className="text-sm">Analytics & Monitoring Agent</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-3 text-blue-600">Distributed Networks</h4>
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-blue-200 rounded-full"></div>
                        <span className="text-sm">Research Discovery Agent</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-blue-200 rounded-full"></div>
                        <span className="text-sm">Submission Assistant Agent</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-blue-200 rounded-full"></div>
                        <span className="text-sm">Review Coordination Agent</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-blue-200 rounded-full"></div>
                        <span className="text-sm">Publishing Production Agent</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Agents Tab */}
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
                      <Badge variant={agent.status === 'active' ? 'default' : 'secondary'}>
                        {agent.status}
                      </Badge>
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
                      <div className="text-xs text-muted-foreground mt-1">{agent.lastUpdate}</div>
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
                      Simulate the complete manuscript processing workflow
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
                      return (
                        <div 
                          key={step.id}
                          className={`workflow-step-${step.id} p-4 rounded-lg border transition-all duration-300`}
                          style={{ 
                            backgroundColor: agent?.color || '#f0f0f0',
                            borderColor: agent?.borderColor || '#ccc'
                          }}
                        >
                          <div className="flex items-center space-x-2 mb-2">
                            {agent?.icon}
                            <span className="font-semibold text-sm">{step.name}</span>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Agent: {agent?.name}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Duration: {step.duration}ms
                          </div>
                          <Badge 
                            variant={step.status === 'completed' ? 'default' : 
                                   step.status === 'in_progress' ? 'secondary' : 'outline'}
                            className="mt-2 text-xs"
                          >
                            {step.status.replace('_', ' ')}
                          </Badge>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Network Tab */}
          <TabsContent value="network" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Agent Interaction Network</CardTitle>
                <CardDescription>
                  Interactive visualization of agent communication and coordination patterns
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div ref={networkRef} className="w-full flex justify-center"></div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Cognitive Tab */}
          <TabsContent value="cognitive" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Cognitive Architecture</CardTitle>
                <CardDescription>
                  Hierarchical priority management balanced with distributed innovation networks
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div ref={cognitiveRef} className="w-full flex justify-center"></div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Diagrams Tab */}
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

