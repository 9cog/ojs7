import { useState, useEffect, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';

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

export interface RealtimeNotification {
  type: 'agent_status_change' | 'workflow_event' | 'manuscript_update';
  timestamp: string;
  agent_id?: string;
  status?: AgentStatus;
  data?: any;
}

export interface ConnectionStats {
  active_connections: number;
  total_notifications_queued: number;
  active_agents: number;
  rooms: string[];
}

interface UseWebSocketReturn {
  socket: Socket | null;
  connected: boolean;
  agentStatuses: Record<string, AgentStatus>;
  notifications: RealtimeNotification[];
  connectionStats: ConnectionStats | null;
  subscribeToAgent: (agentId: string) => void;
  unsubscribeFromAgent: (agentId: string) => void;
  sendTestNotification: (type: string, data?: any) => void;
  clearNotifications: () => void;
  error: string | null;
}

export const useWebSocket = (serverUrl: string = 'http://localhost:5000'): UseWebSocketReturn => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connected, setConnected] = useState<boolean>(false);
  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentStatus>>({});
  const [notifications, setNotifications] = useState<RealtimeNotification[]>([]);
  const [connectionStats, setConnectionStats] = useState<ConnectionStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const subscribedAgents = useRef<Set<string>>(new Set());

  const addNotification = useCallback((notification: RealtimeNotification) => {
    setNotifications(prev => {
      const newNotifications = [notification, ...prev];
      // Keep only last 50 notifications
      return newNotifications.slice(0, 50);
    });
  }, []);

  const clearNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  useEffect(() => {
    const socketInstance = io(serverUrl, {
      transports: ['websocket', 'polling'],
      timeout: 20000,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    setSocket(socketInstance);

    // Connection event handlers
    socketInstance.on('connect', () => {
      console.log('Connected to WebSocket server');
      setConnected(true);
      setError(null);
      addNotification({
        type: 'workflow_event',
        timestamp: new Date().toISOString(),
        data: { message: 'Connected to real-time service' }
      });
    });

    socketInstance.on('disconnect', (reason) => {
      console.log('Disconnected from WebSocket server:', reason);
      setConnected(false);
      addNotification({
        type: 'workflow_event',
        timestamp: new Date().toISOString(),
        data: { message: `Disconnected: ${reason}` }
      });
    });

    socketInstance.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      setError(error.message);
      setConnected(false);
    });

    // Real-time data handlers
    socketInstance.on('connection_confirmed', (data) => {
      console.log('Connection confirmed:', data);
      setConnectionStats(prevStats => ({
        ...prevStats,
        active_connections: 1,
        active_agents: data.active_agents || 0,
        total_notifications_queued: 0,
        rooms: ['general']
      }));
    });

    socketInstance.on('agent_status_update', (data) => {
      console.log('Agent status update:', data);
      if (data.agents) {
        setAgentStatuses(prev => ({
          ...prev,
          ...data.agents
        }));
      }
    });

    socketInstance.on('notification', (notification: RealtimeNotification) => {
      console.log('Received notification:', notification);
      addNotification(notification);
      
      // Update agent status if it's an agent status change
      if (notification.type === 'agent_status_change' && notification.agent_id && notification.status) {
        setAgentStatuses(prev => ({
          ...prev,
          [notification.agent_id!]: notification.status!
        }));
      }
    });

    socketInstance.on('subscription_confirmed', (data) => {
      console.log('Subscription confirmed:', data);
      addNotification({
        type: 'workflow_event',
        timestamp: new Date().toISOString(),
        data: { message: `Subscribed to agent ${data.agent_id}` }
      });
    });

    socketInstance.on('unsubscription_confirmed', (data) => {
      console.log('Unsubscription confirmed:', data);
      addNotification({
        type: 'workflow_event',
        timestamp: new Date().toISOString(),
        data: { message: `Unsubscribed from agent ${data.agent_id}` }
      });
    });

    socketInstance.on('pong', (data) => {
      console.log('Pong received:', data);
    });

    // Cleanup function
    return () => {
      console.log('Cleaning up WebSocket connection');
      socketInstance.disconnect();
    };
  }, [serverUrl, addNotification]);

  const subscribeToAgent = useCallback((agentId: string) => {
    if (socket && connected && !subscribedAgents.current.has(agentId)) {
      socket.emit('subscribe_agent', { agent_id: agentId });
      subscribedAgents.current.add(agentId);
    }
  }, [socket, connected]);

  const unsubscribeFromAgent = useCallback((agentId: string) => {
    if (socket && connected && subscribedAgents.current.has(agentId)) {
      socket.emit('unsubscribe_agent', { agent_id: agentId });
      subscribedAgents.current.delete(agentId);
    }
  }, [socket, connected]);

  const sendTestNotification = useCallback(async (type: string, data?: any) => {
    try {
      const response = await fetch(`${serverUrl}/api/v1/notifications/test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ type, ...data }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('Test notification sent:', result);
    } catch (error) {
      console.error('Error sending test notification:', error);
      setError(error instanceof Error ? error.message : 'Unknown error');
    }
  }, [serverUrl]);

  // Fetch connection stats periodically
  useEffect(() => {
    if (!connected) return;

    const fetchStats = async () => {
      try {
        const response = await fetch(`${serverUrl}/api/v1/realtime/status`);
        if (response.ok) {
          const result = await response.json();
          if (result.status === 'success') {
            setConnectionStats(result.data);
          }
        }
      } catch (error) {
        console.error('Error fetching connection stats:', error);
      }
    };

    const interval = setInterval(fetchStats, 10000); // Every 10 seconds
    fetchStats(); // Initial fetch

    return () => clearInterval(interval);
  }, [connected, serverUrl]);

  // Ping server periodically to maintain connection
  useEffect(() => {
    if (!socket || !connected) return;

    const pingInterval = setInterval(() => {
      socket.emit('ping');
    }, 30000); // Every 30 seconds

    return () => clearInterval(pingInterval);
  }, [socket, connected]);

  return {
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
  };
};