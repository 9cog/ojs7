import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { 
  RefreshCw, 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  Users, 
  FileText,
  Zap,
  Activity,
  TrendingUp,
  Database
} from 'lucide-react';

/**
 * Enhanced OJS Integration Component
 * Real-time connection and data synchronization with Open Journal Systems
 */
const OJSIntegration = ({ onDataUpdate }) => {
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [ojsData, setOjsData] = useState(null);
  const [lastSync, setLastSync] = useState(null);
  const [syncProgress, setSyncProgress] = useState(0);
  const [activeManuscripts, setActiveManuscripts] = useState([]);
  const [systemHealth, setSystemHealth] = useState({});

  // OJS API Configuration
  const OJS_CONFIG = {
    baseUrl: process.env.REACT_APP_OJS_BASE_URL || 'http://localhost:8080',
    apiKey: process.env.REACT_APP_OJS_API_KEY || 'demo_api_key',
    endpoints: {
      manuscripts: '/api/v1/submissions',
      users: '/api/v1/users',
      reviews: '/api/v1/reviews',
      agents: '/api/v1/agents',
      health: '/api/v1/health'
    }
  };

  // Initialize OJS connection
  useEffect(() => {
    initializeOJSConnection();
    const interval = setInterval(syncWithOJS, 30000); // Sync every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const initializeOJSConnection = async () => {
    try {
      setConnectionStatus('connecting');
      
      // Test OJS connection
      const healthResponse = await fetch(`${OJS_CONFIG.baseUrl}${OJS_CONFIG.endpoints.health}`, {
        headers: {
          'Authorization': `Bearer ${OJS_CONFIG.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      if (healthResponse.ok) {
        const healthData = await healthResponse.json();
        setSystemHealth(healthData);
        setConnectionStatus('connected');
        await syncWithOJS();
      } else {
        throw new Error('OJS connection failed');
      }
    } catch (error) {
      console.error('OJS connection error:', error);
      setConnectionStatus('error');
      // Use mock data for demonstration
      loadMockData();
    }
  };

  const syncWithOJS = async () => {
    try {
      setSyncProgress(0);
      
      // Sync manuscripts
      setSyncProgress(25);
      const manuscriptsResponse = await fetch(`${OJS_CONFIG.baseUrl}${OJS_CONFIG.endpoints.manuscripts}`, {
        headers: {
          'Authorization': `Bearer ${OJS_CONFIG.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      if (manuscriptsResponse.ok) {
        const manuscripts = await manuscriptsResponse.json();
        setActiveManuscripts(manuscripts.items || []);
      }

      // Sync agent status
      setSyncProgress(50);
      const agentsResponse = await fetch(`${OJS_CONFIG.baseUrl}${OJS_CONFIG.endpoints.agents}`, {
        headers: {
          'Authorization': `Bearer ${OJS_CONFIG.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      if (agentsResponse.ok) {
        const agentData = await agentsResponse.json();
        if (onDataUpdate) {
          onDataUpdate(agentData);
        }
      }

      setSyncProgress(100);
      setLastSync(new Date());
      
      setTimeout(() => setSyncProgress(0), 2000);
    } catch (error) {
      console.error('Sync error:', error);
      loadMockData();
    }
  };

  const loadMockData = () => {
    // Mock data for demonstration when OJS is not available
    const mockManuscripts = [
      {
        id: 'MS-2024-001',
        title: 'Advanced Peptide Formulations in Anti-Aging Cosmetics',
        status: 'under_review',
        submissionDate: '2024-01-15',
        currentStage: 'peer_review',
        assignedAgents: ['research_discovery', 'content_quality'],
        progress: 65
      },
      {
        id: 'MS-2024-002',
        title: 'Sustainable Packaging Solutions for Cosmetic Products',
        status: 'revision_required',
        submissionDate: '2024-01-12',
        currentStage: 'author_revision',
        assignedAgents: ['submission_assistant', 'editorial_orchestration'],
        progress: 45
      },
      {
        id: 'MS-2024-003',
        title: 'Microbiome-Friendly Skincare: Clinical Efficacy Study',
        status: 'accepted',
        submissionDate: '2024-01-08',
        currentStage: 'production',
        assignedAgents: ['publishing_production', 'analytics_monitoring'],
        progress: 90
      }
    ];

    setActiveManuscripts(mockManuscripts);
    setConnectionStatus('mock');
    setLastSync(new Date());
  };

  const getStatusColor = (status) => {
    const colors = {
      'connected': 'bg-green-100 text-green-800 border-green-200',
      'connecting': 'bg-yellow-100 text-yellow-800 border-yellow-200',
      'error': 'bg-red-100 text-red-800 border-red-200',
      'mock': 'bg-blue-100 text-blue-800 border-blue-200'
    };
    return colors[status] || colors.error;
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'connected': return <CheckCircle className="w-4 h-4" />;
      case 'connecting': return <RefreshCw className="w-4 h-4 animate-spin" />;
      case 'error': return <AlertCircle className="w-4 h-4" />;
      case 'mock': return <Database className="w-4 h-4" />;
      default: return <AlertCircle className="w-4 h-4" />;
    }
  };

  const getManuscriptStatusColor = (status) => {
    const colors = {
      'under_review': 'bg-blue-100 text-blue-800',
      'revision_required': 'bg-yellow-100 text-yellow-800',
      'accepted': 'bg-green-100 text-green-800',
      'rejected': 'bg-red-100 text-red-800',
      'published': 'bg-purple-100 text-purple-800'
    };
    return colors[status] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                OJS Integration Status
              </CardTitle>
              <CardDescription>
                Real-time connection to Open Journal Systems
              </CardDescription>
            </div>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={syncWithOJS}
              disabled={connectionStatus === 'connecting'}
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${connectionStatus === 'connecting' ? 'animate-spin' : ''}`} />
              Sync Now
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-4">
            <Badge className={getStatusColor(connectionStatus)}>
              {getStatusIcon(connectionStatus)}
              <span className="ml-2 capitalize">
                {connectionStatus === 'mock' ? 'Demo Mode' : connectionStatus}
              </span>
            </Badge>
            {lastSync && (
              <span className="text-sm text-gray-500 flex items-center gap-1">
                <Clock className="w-4 h-4" />
                Last sync: {lastSync.toLocaleTimeString()}
              </span>
            )}
          </div>

          {syncProgress > 0 && (
            <div className="mb-4">
              <div className="flex justify-between text-sm mb-1">
                <span>Synchronizing data...</span>
                <span>{syncProgress}%</span>
              </div>
              <Progress value={syncProgress} className="h-2" />
            </div>
          )}

          {connectionStatus === 'error' && (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Unable to connect to OJS. Running in demo mode with mock data.
                Please check your OJS configuration and network connection.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Active Manuscripts */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5" />
            Active Manuscripts ({activeManuscripts.length})
          </CardTitle>
          <CardDescription>
            Real-time manuscript processing status
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {activeManuscripts.map((manuscript) => (
              <div key={manuscript.id} className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <h4 className="font-medium text-sm mb-1">{manuscript.title}</h4>
                    <p className="text-xs text-gray-500 mb-2">ID: {manuscript.id}</p>
                    <div className="flex items-center gap-2 mb-2">
                      <Badge className={getManuscriptStatusColor(manuscript.status)}>
                        {manuscript.status.replace('_', ' ')}
                      </Badge>
                      <span className="text-xs text-gray-500">
                        Stage: {manuscript.currentStage?.replace('_', ' ')}
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium mb-1">{manuscript.progress}%</div>
                    <Progress value={manuscript.progress} className="w-20 h-2" />
                  </div>
                </div>
                
                {manuscript.assignedAgents && manuscript.assignedAgents.length > 0 && (
                  <div className="flex items-center gap-1 flex-wrap">
                    <Users className="w-3 h-3 text-gray-400" />
                    <span className="text-xs text-gray-500 mr-2">Agents:</span>
                    {manuscript.assignedAgents.map((agentId) => (
                      <Badge key={agentId} variant="outline" className="text-xs">
                        {agentId.replace('_', ' ')}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {activeManuscripts.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No active manuscripts found</p>
                <p className="text-sm">Manuscripts will appear here when submitted to OJS</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* System Health Metrics */}
      {Object.keys(systemHealth).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              System Health
            </CardTitle>
            <CardDescription>
              OJS and agent system performance metrics
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(systemHealth).map(([key, value]) => (
                <div key={key} className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {typeof value === 'number' ? value.toFixed(1) : value}
                  </div>
                  <div className="text-sm text-gray-500 capitalize">
                    {key.replace('_', ' ')}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default OJSIntegration;
