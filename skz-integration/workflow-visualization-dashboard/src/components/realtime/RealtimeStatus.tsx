import React from 'react';
import { AlertCircle, Wifi, WifiOff, Users, Activity, Clock } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';

interface ConnectionStats {
  active_connections: number;
  total_notifications_queued: number;
  active_agents: number;
  rooms: string[];
}

interface RealtimeStatusProps {
  connected: boolean;
  connectionStats: ConnectionStats | null;
  error: string | null;
  lastUpdate?: string;
}

export const RealtimeStatus: React.FC<RealtimeStatusProps> = ({
  connected,
  connectionStats,
  error,
  lastUpdate
}) => {
  const getStatusColor = () => {
    if (error) return 'destructive';
    if (connected) return 'default';
    return 'secondary';
  };

  const getStatusText = () => {
    if (error) return 'Error';
    if (connected) return 'Connected';
    return 'Disconnected';
  };

  return (
    <div className="space-y-4">
      {/* Connection Status */}
      <div className="flex items-center justify-between p-4 bg-white rounded-lg border shadow-sm">
        <div className="flex items-center space-x-3">
          <div className="relative">
            {connected ? (
              <Wifi className="w-5 h-5 text-green-600" />
            ) : (
              <WifiOff className="w-5 h-5 text-red-500" />
            )}
            {connected && (
              <div className="absolute -top-1 -right-1 w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            )}
          </div>
          <div>
            <h3 className="font-semibold text-sm">Real-time Status</h3>
            <p className="text-xs text-gray-600">
              {connected ? 'Live updates enabled' : 'Connection lost'}
            </p>
          </div>
        </div>
        <Badge variant={getStatusColor() as any}>
          {getStatusText()}
        </Badge>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Connection Error: {error}
          </AlertDescription>
        </Alert>
      )}

      {/* Connection Stats */}
      {connected && connectionStats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center space-x-2 p-3 bg-blue-50 rounded-lg">
            <Users className="w-4 h-4 text-blue-600" />
            <div>
              <p className="text-sm font-medium text-blue-900">
                {connectionStats.active_connections}
              </p>
              <p className="text-xs text-blue-700">Active Connections</p>
            </div>
          </div>

          <div className="flex items-center space-x-2 p-3 bg-green-50 rounded-lg">
            <Activity className="w-4 h-4 text-green-600" />
            <div>
              <p className="text-sm font-medium text-green-900">
                {connectionStats.active_agents}
              </p>
              <p className="text-xs text-green-700">Active Agents</p>
            </div>
          </div>

          <div className="flex items-center space-x-2 p-3 bg-yellow-50 rounded-lg">
            <Clock className="w-4 h-4 text-yellow-600" />
            <div>
              <p className="text-sm font-medium text-yellow-900">
                {connectionStats.total_notifications_queued}
              </p>
              <p className="text-xs text-yellow-700">Queued Notifications</p>
            </div>
          </div>
        </div>
      )}

      {/* Last Update */}
      {lastUpdate && (
        <div className="text-xs text-gray-500 text-center">
          Last updated: {lastUpdate}
        </div>
      )}
    </div>
  );
};