import React, { useState, useEffect } from 'react';
import './App.css';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import { 
  Activity, Users, FileText, Clock, TrendingUp, Award, 
  Brain, Zap, Target, CheckCircle, XCircle, AlertCircle 
} from 'lucide-react';
import simulationData from './assets/simulation_results.json';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [data, setData] = useState(null);

  useEffect(() => {
    setData(simulationData);
  }, []);

  if (!data) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-indigo-600 mx-auto"></div>
          <p className="mt-4 text-lg text-gray-600">Loading simulation data...</p>
        </div>
      </div>
    );
  }

  const MetricCard = ({ title, value, icon: Icon, color, subtitle }) => (
    <div className="bg-white rounded-xl shadow-lg p-6 border-l-4" style={{ borderLeftColor: color }}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-3xl font-bold text-gray-900">{value}</p>
          {subtitle && <p className="text-sm text-gray-500 mt-1">{subtitle}</p>}
        </div>
        <div className="p-3 rounded-full" style={{ backgroundColor: color + '20' }}>
          <Icon className="h-8 w-8" style={{ color }} />
        </div>
      </div>
    </div>
  );

  const TabButton = ({ id, label, isActive, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`px-6 py-3 font-medium rounded-lg transition-all duration-200 ${
        isActive 
          ? 'bg-indigo-600 text-white shadow-lg' 
          : 'bg-white text-gray-600 hover:bg-gray-50 hover:text-indigo-600'
      }`}
    >
      {label}
    </button>
  );

  // Prepare data for charts
  const agentPerformanceData = Object.entries(data.agent_performance).map(([agentId, performance]) => ({
    name: agentId.replace('agent_', '').replace('_', ' '),
    successRate: (performance.success_rate * 100).toFixed(1),
    efficiency: performance.efficiency_score.toFixed(2),
    actions: performance.total_actions,
    avgTime: performance.avg_processing_time.toFixed(1)
  }));

  const domainDistributionData = Object.entries(data.domain_distribution).map(([domain, count]) => ({
    name: domain.replace('_', ' ').toUpperCase(),
    value: count
  }));

  const statusDistributionData = Object.entries(data.manuscript_distribution).map(([status, count]) => ({
    name: status.replace('_', ' ').toUpperCase(),
    value: count
  }));

  const venuePerformanceData = Object.entries(data.venue_performance).map(([venueId, venue]) => ({
    name: venue.name,
    submissions: venue.submissions,
    acceptances: venue.acceptances,
    acceptanceRate: (venue.acceptance_rate * 100).toFixed(1),
    avgQuality: venue.avg_quality.toFixed(1)
  }));

  const radarData = agentPerformanceData.map(agent => ({
    agent: agent.name,
    efficiency: parseFloat(agent.efficiency) * 20, // Scale for radar
    successRate: parseFloat(agent.successRate),
    productivity: agent.actions * 10, // Scale for radar
  }));

  const OverviewTab = () => (
    <div className="space-y-8">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Manuscripts"
          value={data.simulation_summary.total_manuscripts}
          icon={FileText}
          color="#3B82F6"
          subtitle="Processed in simulation"
        />
        <MetricCard
          title="Acceptance Rate"
          value={`${(data.simulation_summary.acceptance_rate * 100).toFixed(1)}%`}
          icon={CheckCircle}
          color="#10B981"
          subtitle="Overall success rate"
        />
        <MetricCard
          title="Avg Review Time"
          value={`${data.simulation_summary.avg_review_time_days.toFixed(0)} days`}
          icon={Clock}
          color="#F59E0B"
          subtitle="Time to complete review"
        />
        <MetricCard
          title="Revision Cycles"
          value={data.simulation_summary.avg_revision_cycles.toFixed(1)}
          icon={TrendingUp}
          color="#8B5CF6"
          subtitle="Average per manuscript"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Domain Distribution */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Research Domain Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={domainDistributionData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {domainDistributionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Manuscript Status */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Manuscript Status Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={statusDistributionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#3B82F6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* System Health Indicators */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-6">System Health Indicators</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
              <CheckCircle className="h-8 w-8 text-green-600" />
            </div>
            <h4 className="text-lg font-semibold text-gray-900">Quality Improvement</h4>
            <p className="text-3xl font-bold text-green-600">
              {(data.quality_metrics.quality_improvement_rate * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-gray-500">Rate of improvement</p>
          </div>
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
              <Users className="h-8 w-8 text-blue-600" />
            </div>
            <h4 className="text-lg font-semibold text-gray-900">Collaboration Success</h4>
            <p className="text-3xl font-bold text-blue-600">
              {(data.quality_metrics.collaboration_success_rate * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-gray-500">Successful collaborations</p>
          </div>
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-purple-100 rounded-full mb-4">
              <Target className="h-8 w-8 text-purple-600" />
            </div>
            <h4 className="text-lg font-semibold text-gray-900">Venue Match Accuracy</h4>
            <p className="text-3xl font-bold text-purple-600">
              {(data.quality_metrics.venue_match_accuracy * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-gray-500">Accurate recommendations</p>
          </div>
        </div>
      </div>
    </div>
  );

  const AgentsTab = () => (
    <div className="space-y-8">
      {/* Agent Performance Overview */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-6">Agent Performance Overview</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={agentPerformanceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="successRate" fill="#10B981" name="Success Rate (%)" />
            <Bar dataKey="efficiency" fill="#3B82F6" name="Efficiency Score" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Agent Radar Chart */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-6">Agent Capability Radar</h3>
        <ResponsiveContainer width="100%" height={400}>
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="agent" />
            <PolarRadiusAxis angle={90} domain={[0, 100]} />
            <Radar
              name="Efficiency"
              dataKey="efficiency"
              stroke="#3B82F6"
              fill="#3B82F6"
              fillOpacity={0.3}
            />
            <Radar
              name="Success Rate"
              dataKey="successRate"
              stroke="#10B981"
              fill="#10B981"
              fillOpacity={0.3}
            />
            <Radar
              name="Productivity"
              dataKey="productivity"
              stroke="#F59E0B"
              fill="#F59E0B"
              fillOpacity={0.3}
            />
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Individual Agent Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agentPerformanceData.map((agent, index) => (
          <div key={agent.name} className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-indigo-500">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-gray-900 capitalize">{agent.name}</h4>
              <Brain className="h-6 w-6 text-indigo-600" />
            </div>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Success Rate:</span>
                <span className="font-semibold text-green-600">{agent.successRate}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Efficiency:</span>
                <span className="font-semibold text-blue-600">{agent.efficiency}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Total Actions:</span>
                <span className="font-semibold text-gray-900">{agent.actions}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Avg Time:</span>
                <span className="font-semibold text-orange-600">{agent.avgTime}s</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const VenuesTab = () => (
    <div className="space-y-8">
      {/* Venue Performance Chart */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-6">Venue Performance Analysis</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={venuePerformanceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="submissions" fill="#3B82F6" name="Submissions" />
            <Bar dataKey="acceptances" fill="#10B981" name="Acceptances" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Venue Details Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {venuePerformanceData.map((venue, index) => (
          <div key={venue.name} className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-gray-900">{venue.name}</h4>
              <Award className="h-6 w-6 text-yellow-600" />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-600">{venue.submissions}</p>
                <p className="text-sm text-gray-600">Submissions</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-green-600">{venue.acceptances}</p>
                <p className="text-sm text-gray-600">Acceptances</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-purple-600">{venue.acceptanceRate}%</p>
                <p className="text-sm text-gray-600">Accept Rate</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-orange-600">{venue.avgQuality}</p>
                <p className="text-sm text-gray-600">Avg Quality</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const InsightsTab = () => (
    <div className="space-y-8">
      {/* System Recommendations */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-6">System Recommendations</h3>
        <div className="space-y-4">
          {data.system_performance.recommendations.length > 0 ? (
            data.system_performance.recommendations.map((rec, index) => (
              <div key={index} className="flex items-start space-x-3 p-4 bg-blue-50 rounded-lg">
                <AlertCircle className="h-5 w-5 text-blue-600 mt-0.5" />
                <p className="text-gray-700">{rec}</p>
              </div>
            ))
          ) : (
            <div className="flex items-center space-x-3 p-4 bg-green-50 rounded-lg">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <p className="text-gray-700">System is performing optimally. No recommendations at this time.</p>
            </div>
          )}
        </div>
      </div>

      {/* Bottlenecks Analysis */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-6">System Bottlenecks</h3>
        <div className="space-y-4">
          {data.system_performance.bottlenecks.length > 0 ? (
            data.system_performance.bottlenecks.map((bottleneck, index) => (
              <div key={index} className="flex items-start space-x-3 p-4 bg-red-50 rounded-lg">
                <XCircle className="h-5 w-5 text-red-600 mt-0.5" />
                <p className="text-gray-700">{bottleneck}</p>
              </div>
            ))
          ) : (
            <div className="flex items-center space-x-3 p-4 bg-green-50 rounded-lg">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <p className="text-gray-700">No bottlenecks detected. System is running smoothly.</p>
            </div>
          )}
        </div>
      </div>

      {/* Performance Insights */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-6">Performance Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-gray-800">Strengths</h4>
            <ul className="space-y-2">
              <li className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <span className="text-gray-700">High agent success rates (100%)</span>
              </li>
              <li className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <span className="text-gray-700">Efficient processing times</span>
              </li>
              <li className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <span className="text-gray-700">Robust agent coordination</span>
              </li>
            </ul>
          </div>
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-gray-800">Areas for Improvement</h4>
            <ul className="space-y-2">
              <li className="flex items-center space-x-2">
                <AlertCircle className="h-4 w-4 text-yellow-600" />
                <span className="text-gray-700">Manuscript quality scoring</span>
              </li>
              <li className="flex items-center space-x-2">
                <AlertCircle className="h-4 w-4 text-yellow-600" />
                <span className="text-gray-700">Venue matching algorithms</span>
              </li>
              <li className="flex items-center space-x-2">
                <AlertCircle className="h-4 w-4 text-yellow-600" />
                <span className="text-gray-700">Review process optimization</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-indigo-600 rounded-lg">
                <Activity className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Autonomous Agents Simulation</h1>
                <p className="text-sm text-gray-600">Academic Publishing Workflow Analysis</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">Simulation Duration</p>
                <p className="text-xs text-gray-600">{data.simulation_summary.duration}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex space-x-4 mb-8">
          <TabButton
            id="overview"
            label="Overview"
            isActive={activeTab === 'overview'}
            onClick={setActiveTab}
          />
          <TabButton
            id="agents"
            label="Agent Performance"
            isActive={activeTab === 'agents'}
            onClick={setActiveTab}
          />
          <TabButton
            id="venues"
            label="Venue Analysis"
            isActive={activeTab === 'venues'}
            onClick={setActiveTab}
          />
          <TabButton
            id="insights"
            label="Insights & Recommendations"
            isActive={activeTab === 'insights'}
            onClick={setActiveTab}
          />
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'agents' && <AgentsTab />}
        {activeTab === 'venues' && <VenuesTab />}
        {activeTab === 'insights' && <InsightsTab />}
      </div>

      {/* Footer */}
      <div className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-600">
            <p>Autonomous Academic Publishing Agents Framework - Simulation Dashboard</p>
            <p className="mt-1">Powered by AI-driven workflow optimization</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

