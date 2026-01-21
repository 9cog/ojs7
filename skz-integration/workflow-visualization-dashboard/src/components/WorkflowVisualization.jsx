import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  FastForward,
  ArrowRight,
  CheckCircle,
  Clock,
  AlertCircle,
  Zap,
  Activity
} from 'lucide-react';
import * as d3 from 'd3';
import * as anime from 'animejs';

/**
 * Advanced Workflow Visualization Component
 * Interactive manuscript processing workflow with real-time animations
 */
const WorkflowVisualization = ({ manuscriptData, agentData, onWorkflowAction }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [workflowProgress, setWorkflowProgress] = useState(0);
  const [selectedManuscript, setSelectedManuscript] = useState(null);
  const workflowRef = useRef(null);
  const animationRef = useRef(null);

  // Workflow steps definition
  const workflowSteps = [
    {
      id: 'submission',
      name: 'Manuscript Submission',
      agent: 'submission_assistant',
      icon: '📝',
      duration: 2000,
      description: 'Initial manuscript processing and validation',
      tasks: ['Format validation', 'Metadata extraction', 'Initial quality check']
    },
    {
      id: 'research_analysis',
      name: 'Research Discovery',
      agent: 'research_discovery',
      icon: '🔍',
      duration: 3000,
      description: 'Literature review and novelty assessment',
      tasks: ['Literature search', 'Novelty analysis', 'Trend identification']
    },
    {
      id: 'editorial_review',
      name: 'Editorial Orchestration',
      agent: 'editorial_orchestration',
      icon: '🎭',
      duration: 1500,
      description: 'Editorial decision making and coordination',
      tasks: ['Editor assignment', 'Priority assessment', 'Workflow planning']
    },
    {
      id: 'peer_review',
      name: 'Peer Review Coordination',
      agent: 'review_coordination',
      icon: '👥',
      duration: 4000,
      description: 'Reviewer matching and review management',
      tasks: ['Reviewer matching', 'Review assignment', 'Progress tracking']
    },
    {
      id: 'quality_assessment',
      name: 'Content Quality Control',
      agent: 'content_quality',
      icon: '✅',
      duration: 2500,
      description: 'Scientific validation and quality assurance',
      tasks: ['Scientific validation', 'Safety assessment', 'Standards compliance']
    },
    {
      id: 'production',
      name: 'Publishing Production',
      agent: 'publishing_production',
      icon: '🚀',
      duration: 3500,
      description: 'Content formatting and publication preparation',
      tasks: ['Content formatting', 'Visual generation', 'Distribution prep']
    },
    {
      id: 'analytics',
      name: 'Analytics & Monitoring',
      agent: 'analytics_monitoring',
      icon: '📊',
      duration: 1000,
      description: 'Performance analysis and insights generation',
      tasks: ['Performance tracking', 'Impact analysis', 'Optimization insights']
    }
  ];

  // Initialize workflow visualization
  useEffect(() => {
    if (workflowRef.current) {
      createWorkflowVisualization();
    }
  }, [selectedManuscript]);

  // Animation control
  useEffect(() => {
    if (isPlaying) {
      startWorkflowAnimation();
    } else {
      pauseWorkflowAnimation();
    }

    return () => {
      if (animationRef.current) {
        animationRef.current.pause();
      }
    };
  }, [isPlaying]);

  const createWorkflowVisualization = () => {
    const container = d3.select(workflowRef.current);
    container.selectAll('*').remove();

    const width = 1000;
    const height = 400;
    const stepWidth = 120;
    const stepHeight = 80;
    const stepSpacing = 20;

    const svg = container
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .style('background', 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)')
      .style('border-radius', '8px');

    // Create workflow steps
    const stepGroups = svg.selectAll('.step-group')
      .data(workflowSteps)
      .enter()
      .append('g')
      .attr('class', 'step-group')
      .attr('transform', (d, i) => {
        const x = 50 + i * (stepWidth + stepSpacing);
        const y = height / 2 - stepHeight / 2;
        return `translate(${x}, ${y})`;
      });

    // Step rectangles
    stepGroups
      .append('rect')
      .attr('class', 'step-rect')
      .attr('width', stepWidth)
      .attr('height', stepHeight)
      .attr('rx', 8)
      .attr('fill', (d, i) => i <= currentStep ? '#3b82f6' : '#e2e8f0')
      .attr('stroke', (d, i) => i === currentStep ? '#1d4ed8' : '#cbd5e1')
      .attr('stroke-width', (d, i) => i === currentStep ? 3 : 1)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        setCurrentStep(workflowSteps.indexOf(d));
      });

    // Step icons
    stepGroups
      .append('text')
      .attr('class', 'step-icon')
      .attr('x', stepWidth / 2)
      .attr('y', 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '24px')
      .text(d => d.icon);

    // Step names
    stepGroups
      .append('text')
      .attr('class', 'step-name')
      .attr('x', stepWidth / 2)
      .attr('y', 45)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .attr('fill', (d, i) => i <= currentStep ? 'white' : '#64748b')
      .text(d => d.name.split(' ')[0]);

    stepGroups
      .append('text')
      .attr('class', 'step-name-2')
      .attr('x', stepWidth / 2)
      .attr('y', 57)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .attr('fill', (d, i) => i <= currentStep ? 'white' : '#64748b')
      .text(d => d.name.split(' ').slice(1).join(' '));

    // Connection arrows
    for (let i = 0; i < workflowSteps.length - 1; i++) {
      const startX = 50 + i * (stepWidth + stepSpacing) + stepWidth;
      const endX = 50 + (i + 1) * (stepWidth + stepSpacing);
      const y = height / 2;

      svg.append('path')
        .attr('class', 'connection-arrow')
        .attr('d', `M ${startX + 5} ${y} L ${endX - 5} ${y}`)
        .attr('stroke', i < currentStep ? '#3b82f6' : '#cbd5e1')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#arrowhead)');
    }

    // Add arrowhead marker
    svg.append('defs')
      .append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M 0,-5 L 10,0 L 0,5')
      .attr('fill', '#3b82f6');

    // Progress indicator
    svg.append('rect')
      .attr('class', 'progress-bg')
      .attr('x', 50)
      .attr('y', height - 30)
      .attr('width', workflowSteps.length * (stepWidth + stepSpacing) - stepSpacing)
      .attr('height', 4)
      .attr('fill', '#e2e8f0')
      .attr('rx', 2);

    svg.append('rect')
      .attr('class', 'progress-fill')
      .attr('x', 50)
      .attr('y', height - 30)
      .attr('width', (workflowProgress / 100) * (workflowSteps.length * (stepWidth + stepSpacing) - stepSpacing))
      .attr('height', 4)
      .attr('fill', '#3b82f6')
      .attr('rx', 2);
  };

  const startWorkflowAnimation = () => {
    if (currentStep >= workflowSteps.length - 1) {
      setCurrentStep(0);
      setWorkflowProgress(0);
    }

    const animateStep = (stepIndex) => {
      if (stepIndex >= workflowSteps.length) {
        setIsPlaying(false);
        return;
      }

      setCurrentStep(stepIndex);
      
      // Animate progress
      anime({
        targets: { progress: workflowProgress },
        progress: ((stepIndex + 1) / workflowSteps.length) * 100,
        duration: workflowSteps[stepIndex].duration,
        easing: 'easeInOutQuad',
        update: function(anim) {
          setWorkflowProgress(anim.animatables[0].target.progress);
        },
        complete: function() {
          if (isPlaying) {
            setTimeout(() => animateStep(stepIndex + 1), 500);
          }
        }
      });
    };

    animateStep(currentStep);
  };

  const pauseWorkflowAnimation = () => {
    if (animationRef.current) {
      animationRef.current.pause();
    }
  };

  const resetWorkflow = () => {
    setIsPlaying(false);
    setCurrentStep(0);
    setWorkflowProgress(0);
    if (workflowRef.current) {
      createWorkflowVisualization();
    }
  };

  const getStepStatus = (stepIndex) => {
    if (stepIndex < currentStep) return 'completed';
    if (stepIndex === currentStep) return 'active';
    return 'pending';
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'active': return <Activity className="w-4 h-4 text-blue-500 animate-pulse" />;
      case 'pending': return <Clock className="w-4 h-4 text-gray-400" />;
      default: return <AlertCircle className="w-4 h-4 text-red-500" />;
    }
  };

  const getStatusColor = (status) => {
    const colors = {
      'completed': 'bg-green-100 text-green-800 border-green-200',
      'active': 'bg-blue-100 text-blue-800 border-blue-200',
      'pending': 'bg-gray-100 text-gray-600 border-gray-200',
      'error': 'bg-red-100 text-red-800 border-red-200'
    };
    return colors[status] || colors.pending;
  };

  return (
    <div className="space-y-6">
      {/* Workflow Controls */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5" />
                Manuscript Processing Workflow
              </CardTitle>
              <CardDescription>
                Interactive visualization of the autonomous agent workflow
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                variant={isPlaying ? "default" : "outline"}
                size="sm"
                onClick={() => setIsPlaying(!isPlaying)}
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-4 h-4 mr-2" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Play
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={resetWorkflow}
              >
                <RotateCcw className="w-4 h-4 mr-2" />
                Reset
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setCurrentStep(Math.min(currentStep + 1, workflowSteps.length - 1));
                  setWorkflowProgress(((currentStep + 1) / workflowSteps.length) * 100);
                }}
              >
                <FastForward className="w-4 h-4 mr-2" />
                Next Step
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {/* Progress Overview */}
          <div className="mb-4">
            <div className="flex justify-between text-sm mb-2">
              <span>Overall Progress</span>
              <span>{Math.round(workflowProgress)}%</span>
            </div>
            <Progress value={workflowProgress} className="h-3" />
          </div>

          {/* Workflow Visualization */}
          <div ref={workflowRef} className="w-full overflow-x-auto"></div>
        </CardContent>
      </Card>

      {/* Current Step Details */}
      {workflowSteps[currentStep] && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="text-2xl">{workflowSteps[currentStep].icon}</span>
              {workflowSteps[currentStep].name}
              <Badge className={getStatusColor(getStepStatus(currentStep))}>
                {getStatusIcon(getStepStatus(currentStep))}
                <span className="ml-1">{getStepStatus(currentStep)}</span>
              </Badge>
            </CardTitle>
            <CardDescription>
              {workflowSteps[currentStep].description}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Step Tasks */}
              <div>
                <h4 className="font-medium mb-3">Current Tasks</h4>
                <div className="space-y-2">
                  {workflowSteps[currentStep].tasks.map((task, index) => (
                    <div key={index} className="flex items-center gap-2 text-sm">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span>{task}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Agent Information */}
              <div>
                <h4 className="font-medium mb-3">Responsible Agent</h4>
                {agentData && (
                  <div className="space-y-2">
                    {(() => {
                      const agent = agentData.find(a => a.id === workflowSteps[currentStep].agent);
                      if (!agent) return <div>Agent not found</div>;
                      
                      return (
                        <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                          {agent.icon}
                          <div className="flex-1">
                            <div className="font-medium text-sm">{agent.name}</div>
                            <div className="text-xs text-gray-500">{agent.type}</div>
                            <div className="text-xs text-gray-600 mt-1">
                              Efficiency: {(agent.performance.efficiency * 100).toFixed(1)}% | 
                              Success Rate: {(agent.performance.successRate * 100).toFixed(1)}%
                            </div>
                          </div>
                          <Badge className={getStatusColor(agent.status)}>
                            {agent.status}
                          </Badge>
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step Timeline */}
      <Card>
        <CardHeader>
          <CardTitle>Workflow Timeline</CardTitle>
          <CardDescription>
            Detailed view of all workflow steps and their status
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {workflowSteps.map((step, index) => (
              <div 
                key={step.id} 
                className={`flex items-center gap-4 p-3 rounded-lg border transition-all cursor-pointer ${
                  index === currentStep 
                    ? 'bg-blue-50 border-blue-200' 
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
                onClick={() => setCurrentStep(index)}
              >
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <span className="text-xl">{step.icon}</span>
                  <div className="min-w-0 flex-1">
                    <div className="font-medium text-sm">{step.name}</div>
                    <div className="text-xs text-gray-500 truncate">{step.description}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge className={getStatusColor(getStepStatus(index))}>
                    {getStatusIcon(getStepStatus(index))}
                    <span className="ml-1">{getStepStatus(index)}</span>
                  </Badge>
                  {index < workflowSteps.length - 1 && (
                    <ArrowRight className="w-4 h-4 text-gray-400" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default WorkflowVisualization;
