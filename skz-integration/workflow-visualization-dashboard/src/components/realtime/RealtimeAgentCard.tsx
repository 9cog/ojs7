import React, { useEffect, useState } from 'react';
import { Activity, Zap, Clock, TrendingUp, Eye, EyeOff } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';

export interface AgentStatus {
  id: string;
  status: string;
  cpu_usage?: number;
  memory_usage?: number;
  active_tasks?: number;
  last_action?: string;
  performance?: {
    success_rate: number;
    avg_response_time: number;
    total_actions: number;
  };
}

interface AgentCardProps {
  agentType: string;
  agentStatus: AgentStatus;
  onSubscribe: (agentId: string) => void;
  onUnsubscribe: (agentId: string) => void;
  isSubscribed: boolean;
  lastUpdated?: string;
}

const AGENT_CONFIG = {
  research_discovery: {
    name: 'Research Discovery Agent',
    icon: '🔍',
    color: 'blue',
    description: 'Literature search and gap analysis'
  },
  submission_assistant: {
    name: 'Submission Assistant Agent',
    icon: '📝',
    color: 'green',
    description: 'Format checking and compliance'
  },
  editorial_orchestration: {
    name: 'Editorial Orchestration Agent',
    icon: '🎭',
    color: 'purple',
    description: 'Workflow management and decisions'
  },
  review_coordination: {
    name: 'Review Coordination Agent',
    icon: '👥',
    color: 'orange',
    description: 'Reviewer matching and tracking'
  },
  content_quality: {
    name: 'Content Quality Agent',
    icon: '✅',
    color: 'red',
    description: 'Validation and quality assurance'
  },
  publishing_production: {
    name: 'Publishing Production Agent',
    icon: '📚',
    color: 'indigo',
    description: 'Layout and publication scheduling'
  },
  analytics_monitoring: {
    name: 'Analytics & Monitoring Agent',
    icon: '📊',
    color: 'yellow',
    description: 'Performance tracking and reporting'
  }
} as const;

export const RealtimeAgentCard: React.FC<AgentCardProps> = ({
  agentType,
  agentStatus,
  onSubscribe,
  onUnsubscribe,
  isSubscribed,
  lastUpdated
}) => {
  const [isHighlighted, setIsHighlighted] = useState(false);
  const config = AGENT_CONFIG[agentType as keyof typeof AGENT_CONFIG];

  // Highlight animation when status updates
  useEffect(() => {
    if (lastUpdated) {
      setIsHighlighted(true);
      const timer = setTimeout(() => setIsHighlighted(false), 1000);
      return () => clearTimeout(timer);
    }
  }, [lastUpdated]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'processing':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'idle':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'error':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  if (!config) {
    return null;
  }

  return (
    <Card className={`transition-all duration-500 ${isHighlighted ? 'ring-2 ring-yellow-400 bg-yellow-50' : 'hover:shadow-lg'}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-2xl">{config.icon}</span>
            <div>
              <CardTitle className="text-sm">{config.name}</CardTitle>
              <CardDescription className="text-xs">{config.description}</CardDescription>
            </div>
          </div>
          <Button
            size="sm"
            variant={isSubscribed ? "default" : "outline"}
            onClick={() => isSubscribed ? onUnsubscribe(agentType) : onSubscribe(agentType)}
          >
            {isSubscribed ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
          </Button>
        </div>
        <div className="flex items-center justify-between">
          <Badge className={`${getStatusColor(agentStatus.status)} border`}>
            {agentStatus.status || 'Unknown'}
          </Badge>
          {lastUpdated && (
            <span className="text-xs text-gray-500">
              Updated: {formatTimestamp(lastUpdated)}
            </span>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Performance Metrics */}
        {agentStatus.performance && (
          <div className="grid grid-cols-3 gap-3 text-center">
            <div>
              <p className="text-lg font-bold text-green-600">
                {Math.round((agentStatus.performance.success_rate || 0) * 100)}%
              </p>
              <p className="text-xs text-gray-600">Success Rate</p>
            </div>
            <div>
              <p className="text-lg font-bold text-blue-600">
                {(agentStatus.performance.avg_response_time || 0).toFixed(1)}s
              </p>
              <p className="text-xs text-gray-600">Avg Response</p>
            </div>
            <div>
              <p className="text-lg font-bold text-purple-600">
                {agentStatus.performance.total_actions || 0}
              </p>
              <p className="text-xs text-gray-600">Total Actions</p>
            </div>
          </div>
        )}

        {/* System Metrics */}
        <div className="space-y-2">
          {typeof agentStatus.cpu_usage === 'number' && (
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span>CPU Usage</span>
                <span>{agentStatus.cpu_usage.toFixed(1)}%</span>
              </div>
              <Progress value={agentStatus.cpu_usage} className="h-2" />
            </div>
          )}

          {typeof agentStatus.memory_usage === 'number' && (
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span>Memory Usage</span>
                <span>{agentStatus.memory_usage.toFixed(1)}%</span>
              </div>
              <Progress value={agentStatus.memory_usage} className="h-2" />
            </div>
          )}

          {typeof agentStatus.active_tasks === 'number' && (
            <div className="flex items-center justify-between text-xs">
              <span className="flex items-center space-x-1">
                <Activity className="w-3 h-3" />
                <span>Active Tasks</span>
              </span>
              <Badge variant="outline">
                {agentStatus.active_tasks}
              </Badge>
            </div>
          )}
        </div>

        {/* Last Action */}
        {agentStatus.last_action && (
          <div className="text-xs text-gray-600">
            <div className="flex items-center space-x-1 mb-1">
              <Clock className="w-3 h-3" />
              <span>Last Action:</span>
            </div>
            <p className="text-gray-800 truncate">
              {typeof agentStatus.last_action === 'string' 
                ? agentStatus.last_action 
                : formatTimestamp(agentStatus.last_action)}
            </p>
          </div>
        )}

        {/* Real-time Indicators */}
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isSubscribed ? 'bg-green-400 animate-pulse' : 'bg-gray-300'}`} />
            <span className="text-gray-600">
              {isSubscribed ? 'Live updates' : 'Click eye to subscribe'}
            </span>
          </div>
          {isHighlighted && (
            <Badge variant="outline" className="animate-pulse">
              Updated
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  );
};