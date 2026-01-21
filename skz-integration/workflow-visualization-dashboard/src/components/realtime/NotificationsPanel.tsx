import React, { useState } from 'react';
import { Bell, X, Clock, User, FileText, Activity, AlertTriangle, Info, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';

export interface RealtimeNotification {
  type: 'agent_status_change' | 'workflow_event' | 'manuscript_update';
  timestamp: string;
  agent_id?: string;
  status?: any;
  data?: any;
}

interface NotificationsPanelProps {
  notifications: RealtimeNotification[];
  onClear: () => void;
  onTestNotification: (type: string, data?: any) => void;
}

export const NotificationsPanel: React.FC<NotificationsPanelProps> = ({
  notifications,
  onClear,
  onTestNotification
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'agent_status_change':
        return <Activity className="w-4 h-4 text-blue-600" />;
      case 'workflow_event':
        return <FileText className="w-4 h-4 text-green-600" />;
      case 'manuscript_update':
        return <User className="w-4 h-4 text-purple-600" />;
      default:
        return <Info className="w-4 h-4 text-gray-600" />;
    }
  };

  const getNotificationColor = (type: string) => {
    switch (type) {
      case 'agent_status_change':
        return 'bg-blue-50 border-blue-200';
      case 'workflow_event':
        return 'bg-green-50 border-green-200';
      case 'manuscript_update':
        return 'bg-purple-50 border-purple-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  const formatNotificationContent = (notification: RealtimeNotification) => {
    switch (notification.type) {
      case 'agent_status_change':
        return `Agent ${notification.agent_id} status changed to ${notification.status?.status || 'unknown'}`;
      case 'workflow_event':
        return notification.data?.message || 'Workflow event occurred';
      case 'manuscript_update':
        return `Manuscript update: ${notification.data?.update_type || 'Status changed'}`;
      default:
        return JSON.stringify(notification.data || notification, null, 2);
    }
  };

  return (
    <div className="bg-white rounded-lg border shadow-sm">
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-2">
          <Bell className="w-5 h-5 text-gray-600" />
          <h2 className="font-semibold">Real-time Notifications</h2>
          <Badge variant="outline">
            {notifications.length}
          </Badge>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? 'Collapse' : 'Expand'}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClear}
            disabled={notifications.length === 0}
          >
            Clear All
          </Button>
        </div>
      </div>

      {isExpanded && (
        <div className="p-4">
          {/* Test Buttons */}
          <div className="mb-4 space-y-2">
            <p className="text-sm font-medium text-gray-700">Test Notifications:</p>
            <div className="flex flex-wrap gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => onTestNotification('agent_status', { agent_id: 'research_discovery' })}
              >
                Test Agent Status
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => onTestNotification('workflow')}
              >
                Test Workflow
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => onTestNotification('manuscript', { manuscript_id: 'test_001' })}
              >
                Test Manuscript
              </Button>
            </div>
          </div>

          {/* Notifications List */}
          <ScrollArea className="h-96">
            {notifications.length === 0 ? (
              <div className="flex items-center justify-center h-32 text-gray-500">
                <div className="text-center">
                  <Bell className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                  <p className="text-sm">No notifications yet</p>
                  <p className="text-xs">Try the test buttons above</p>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                {notifications.map((notification, index) => (
                  <div
                    key={index}
                    className={`p-3 rounded-lg border ${getNotificationColor(notification.type)} transition-all duration-300 hover:shadow-sm`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-2 flex-1">
                        {getNotificationIcon(notification.type)}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center space-x-2 mb-1">
                            <Badge variant="outline" className="text-xs">
                              {notification.type.replace('_', ' ')}
                            </Badge>
                            <div className="flex items-center space-x-1 text-xs text-gray-500">
                              <Clock className="w-3 h-3" />
                              {formatTimestamp(notification.timestamp)}
                            </div>
                          </div>
                          <p className="text-sm text-gray-800 break-words">
                            {formatNotificationContent(notification)}
                          </p>
                          {notification.data && Object.keys(notification.data).length > 1 && (
                            <details className="mt-2">
                              <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                                Show details
                              </summary>
                              <pre className="mt-1 text-xs bg-gray-100 p-2 rounded text-gray-700 overflow-x-auto">
                                {JSON.stringify(notification.data, null, 2)}
                              </pre>
                            </details>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>

          {notifications.length > 0 && (
            <div className="mt-4 pt-4 border-t text-center">
              <p className="text-xs text-gray-500">
                Showing last {Math.min(notifications.length, 50)} notifications
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};