import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line
} from 'recharts';
import {
  Brain,
  TrendingUp,
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle,
  RotateCcw,
  Activity
} from 'lucide-react';

const EditorialDecisionAnalytics = () => {
  const [decisionStats, setDecisionStats] = useState(null);
  const [recentDecisions, setRecentDecisions] = useState([]);
  const [performanceMetrics, setPerformanceMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDecisionAnalytics();
  }, []);

  const fetchDecisionAnalytics = async () => {
    try {
      setLoading(true);
      
      // Fetch from SKZ decision engine
      const [statsResponse, decisionsResponse, metricsResponse] = await Promise.all([
        fetch('/api/v1/agents/editorial-decision/statistics'),
        fetch('/api/v1/agents/editorial-decision/recent-decisions?limit=20'),
        fetch('/api/v1/agents/editorial-decision/performance-metrics')
      ]);

      if (statsResponse.ok) {
        const stats = await statsResponse.json();
        setDecisionStats(stats);
      }

      if (decisionsResponse.ok) {
        const decisions = await decisionsResponse.json();
        setRecentDecisions(decisions.decisions || []);
      }

      if (metricsResponse.ok) {
        const metrics = await metricsResponse.json();
        setPerformanceMetrics(metrics);
      }

    } catch (error) {
      console.error('Error fetching decision analytics:', error);
      // Set mock data for development
      setMockData();
    } finally {
      setLoading(false);
    }
  };

  const setMockData = () => {
    setDecisionStats({
      total_decisions: 156,
      accept_rate: 23.1,
      revision_rate: 45.5,
      reject_rate: 31.4,
      average_confidence: 0.847,
      decision_distribution: {
        accept: 36,
        minor_revision: 42,
        major_revision: 29,
        reject: 49
      }
    });

    setPerformanceMetrics({
      accuracy: 0.894,
      avg_processing_time: 2.3,
      confidence_calibration: 0.856,
      reviewer_agreement: 0.78
    });

    setRecentDecisions([
      {
        manuscript_id: 'MS-2024-001',
        title: 'Machine Learning Approaches in Academic Publishing',
        decision_type: 'minor_revision',
        confidence: 0.89,
        timestamp: '2024-08-07T10:30:00Z',
        reasoning: ['High quality methodology', 'Minor presentation issues']
      },
      {
        manuscript_id: 'MS-2024-002',
        title: 'Autonomous Systems for Editorial Workflows',
        decision_type: 'accept',
        confidence: 0.95,
        timestamp: '2024-08-07T09:15:00Z',
        reasoning: ['Excellent contribution', 'Strong reviewer consensus']
      }
    ]);
  };

  const getDecisionIcon = (type) => {
    switch (type) {
      case 'accept':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'reject':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'minor_revision':
      case 'major_revision':
        return <RotateCcw className="h-4 w-4 text-yellow-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-blue-500" />;
    }
  };

  const getDecisionColor = (type) => {
    switch (type) {
      case 'accept':
        return 'bg-green-100 text-green-800';
      case 'reject':
        return 'bg-red-100 text-red-800';
      case 'minor_revision':
        return 'bg-yellow-100 text-yellow-800';
      case 'major_revision':
        return 'bg-orange-100 text-orange-800';
      default:
        return 'bg-blue-100 text-blue-800';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return 'text-green-600';
    if (confidence > 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const pieChartColors = ['#22c55e', '#eab308', '#f97316', '#ef4444'];

  const pieChartData = decisionStats?.decision_distribution ? [
    { name: 'Accept', value: decisionStats.decision_distribution.accept, color: '#22c55e' },
    { name: 'Minor Revision', value: decisionStats.decision_distribution.minor_revision, color: '#eab308' },
    { name: 'Major Revision', value: decisionStats.decision_distribution.major_revision, color: '#f97316' },
    { name: 'Reject', value: decisionStats.decision_distribution.reject, color: '#ef4444' }
  ] : [];

  const performanceData = performanceMetrics ? [
    { metric: 'Accuracy', value: performanceMetrics.accuracy * 100 },
    { metric: 'Confidence', value: performanceMetrics.confidence_calibration * 100 },
    { metric: 'Agreement', value: performanceMetrics.reviewer_agreement * 100 }
  ] : [];

  if (loading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Editorial Decision Analytics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <span className="ml-2">Loading decision analytics...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Brain className="h-6 w-6 text-blue-600" />
            Editorial Decision Analytics
          </h2>
          <p className="text-muted-foreground">
            AI-powered editorial decision insights and performance metrics
          </p>
        </div>
        <Button onClick={fetchDecisionAnalytics} variant="outline" size="sm">
          <Activity className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Decisions</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{decisionStats?.total_decisions || 0}</div>
            <p className="text-xs text-muted-foreground">
              +12% from last month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Accept Rate</CardTitle>
            <CheckCircle className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {decisionStats?.accept_rate?.toFixed(1) || 0}%
            </div>
            <p className="text-xs text-muted-foreground">
              Industry average: 25%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Confidence</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(decisionStats?.average_confidence * 100)?.toFixed(1) || 0}%
            </div>
            <Progress 
              value={(decisionStats?.average_confidence * 100) || 0} 
              className="mt-2" 
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Processing Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {performanceMetrics?.avg_processing_time?.toFixed(1) || 0}s
            </div>
            <p className="text-xs text-muted-foreground">
              -0.8s from last month
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="decisions">Recent Decisions</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            {/* Decision Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Decision Distribution</CardTitle>
                <CardDescription>
                  Breakdown of editorial decisions made
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={pieChartData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {pieChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Performance Metrics */}
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
                <CardDescription>
                  System accuracy and reliability indicators
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Value']} />
                    <Bar dataKey="value" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="decisions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Editorial Decisions</CardTitle>
              <CardDescription>
                Latest AI-assisted editorial decisions with confidence scores
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentDecisions.map((decision, index) => (
                  <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      {getDecisionIcon(decision.decision_type)}
                      <div className="space-y-1">
                        <p className="text-sm font-medium leading-none">
                          {decision.title || `Manuscript ${decision.manuscript_id}`}
                        </p>
                        <div className="flex items-center gap-2">
                          <Badge 
                            variant="secondary" 
                            className={getDecisionColor(decision.decision_type)}
                          >
                            {decision.decision_type.replace('_', ' ')}
                          </Badge>
                          <span className={`text-sm font-medium ${getConfidenceColor(decision.confidence)}`}>
                            {(decision.confidence * 100).toFixed(0)}% confidence
                          </span>
                        </div>
                        {decision.reasoning && (
                          <p className="text-xs text-muted-foreground">
                            {decision.reasoning.join(', ')}
                          </p>
                        )}
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-muted-foreground">
                        {new Date(decision.timestamp).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>System Performance</CardTitle>
                <CardDescription>
                  Decision engine performance over time
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Decision Accuracy</span>
                    <span className="text-sm text-muted-foreground">
                      {(performanceMetrics?.accuracy * 100)?.toFixed(1) || 0}%
                    </span>
                  </div>
                  <Progress value={(performanceMetrics?.accuracy * 100) || 0} />
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Reviewer Agreement</span>
                    <span className="text-sm text-muted-foreground">
                      {(performanceMetrics?.reviewer_agreement * 100)?.toFixed(1) || 0}%
                    </span>
                  </div>
                  <Progress value={(performanceMetrics?.reviewer_agreement * 100) || 0} />
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Confidence Calibration</span>
                    <span className="text-sm text-muted-foreground">
                      {(performanceMetrics?.confidence_calibration * 100)?.toFixed(1) || 0}%
                    </span>
                  </div>
                  <Progress value={(performanceMetrics?.confidence_calibration * 100) || 0} />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Decision Trends</CardTitle>
                <CardDescription>
                  Editorial decision patterns and trends
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="text-center py-8 text-muted-foreground">
                    <TrendingUp className="h-8 w-8 mx-auto mb-2" />
                    <p>Trend analysis coming soon</p>
                    <p className="text-xs">Historical data collection in progress</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default EditorialDecisionAnalytics;