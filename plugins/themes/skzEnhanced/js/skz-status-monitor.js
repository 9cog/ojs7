/**
 * SKZ Status Monitor JavaScript
 * Real-time monitoring and status updates for agents
 */

(function($) {
    'use strict';
    
    // Extend SKZ namespace
    window.SKZ = window.SKZ || {};
    
    /**
     * SKZ Status Monitor
     */
    SKZ.StatusMonitor = {
        
        // Configuration
        config: {
            wsUrl: 'ws://localhost:5000/ws/status',
            pollInterval: 15000, // 15 seconds fallback polling
            reconnectDelay: 5000,
            maxReconnectAttempts: 10,
            heartbeatInterval: 30000
        },
        
        // State
        state: {
            websocket: null,
            isConnected: false,
            reconnectAttempts: 0,
            lastHeartbeat: null,
            pollTimer: null,
            metrics: {
                totalRequests: 0,
                successfulRequests: 0,
                failedRequests: 0,
                averageResponseTime: 0
            }
        },
        
        /**
         * Initialize status monitoring
         */
        init: function() {
            console.log('Initializing SKZ Status Monitor...');
            
            this.setupEventListeners();
            this.initializeConnection();
            this.startHeartbeat();
            
            console.log('SKZ Status Monitor initialized');
        },
        
        /**
         * Setup event listeners
         */
        setupEventListeners: function() {
            var self = this;
            
            // Listen for agent data updates
            $(document).on('skz:agentDataUpdated', function(event, data) {
                self.processStatusUpdate(data);
            });
            
            // Listen for page visibility changes
            $(document).on('visibilitychange', function() {
                if (document.hidden) {
                    self.pauseMonitoring();
                } else {
                    self.resumeMonitoring();
                }
            });
            
            // Listen for window unload
            $(window).on('beforeunload', function() {
                self.cleanup();
            });
        },
        
        /**
         * Initialize WebSocket connection
         */
        initializeConnection: function() {
            if (!window.WebSocket) {
                console.warn('WebSocket not supported, falling back to polling');
                this.startPolling();
                return;
            }
            
            this.connectWebSocket();
        },
        
        /**
         * Connect to WebSocket
         */
        connectWebSocket: function() {
            var self = this;
            
            try {
                this.state.websocket = new WebSocket(this.config.wsUrl);
                
                this.state.websocket.onopen = function() {
                    console.log('WebSocket connected to SKZ Status Monitor');
                    self.state.isConnected = true;
                    self.state.reconnectAttempts = 0;
                    self.onConnectionEstablished();
                };
                
                this.state.websocket.onmessage = function(event) {
                    self.handleWebSocketMessage(event);
                };
                
                this.state.websocket.onclose = function(event) {
                    console.log('WebSocket connection closed:', event.code, event.reason);
                    self.state.isConnected = false;
                    self.onConnectionLost();
                };
                
                this.state.websocket.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    self.handleConnectionError();
                };
                
            } catch (error) {
                console.error('Failed to create WebSocket connection:', error);
                this.startPolling();
            }
        },
        
        /**
         * Handle WebSocket message
         */
        handleWebSocketMessage: function(event) {
            try {
                var data = JSON.parse(event.data);
                this.processRealTimeUpdate(data);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        },
        
        /**
         * Process real-time update
         */
        processRealTimeUpdate: function(data) {
            switch (data.type) {
                case 'status_update':
                    this.updateAgentStatus(data.payload);
                    break;
                case 'workflow_progress':
                    this.updateWorkflowProgress(data.payload);
                    break;
                case 'agent_action':
                    this.handleAgentAction(data.payload);
                    break;
                case 'system_alert':
                    this.handleSystemAlert(data.payload);
                    break;
                case 'heartbeat':
                    this.handleHeartbeat(data.payload);
                    break;
                default:
                    console.log('Unknown message type:', data.type);
            }
        },
        
        /**
         * Update agent status
         */
        updateAgentStatus: function(payload) {
            var agentId = payload.agentId;
            var status = payload.status;
            var timestamp = payload.timestamp;
            
            // Update status indicators
            this.updateStatusIndicators(agentId, status);
            
            // Update agent cards
            this.updateAgentCards(agentId, status);
            
            // Show notification if status changed significantly
            if (status === 'error' || status === 'warning') {
                this.showStatusNotification(agentId, status, payload.message);
            }
            
            // Update last activity timestamp
            this.updateLastActivity(agentId, timestamp);
            
            // Trigger custom event
            $(document).trigger('skz:agentStatusChanged', [agentId, status, payload]);
        },
        
        /**
         * Update workflow progress
         */
        updateWorkflowProgress: function(payload) {
            var submissionId = payload.submissionId;
            var progress = payload.progress;
            var stage = payload.stage;
            
            // Update progress bars
            $('.skz-progress-fill').each(function() {
                $(this).css('width', progress + '%');
            });
            
            $('.skz-progress-percentage').text(progress + '%');
            
            // Update stage indicators
            if (stage) {
                $('.skz-stage-badge').removeClass('submission editorial_review peer_review copyediting production published')
                                    .addClass(stage)
                                    .text(stage.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()));
            }
            
            // Trigger custom event
            $(document).trigger('skz:workflowProgressUpdated', [submissionId, progress, stage]);
        },
        
        /**
         * Handle agent action
         */
        handleAgentAction: function(payload) {
            var agentId = payload.agentId;
            var action = payload.action;
            var result = payload.result;
            var timestamp = payload.timestamp;
            
            // Add to activity timeline
            this.addToActivityTimeline(agentId, action, timestamp);
            
            // Update metrics
            this.updateMetrics(result);
            
            // Show notification for important actions
            if (payload.notify) {
                this.showActionNotification(agentId, action, result);
            }
            
            // Trigger custom event
            $(document).trigger('skz:agentActionCompleted', [agentId, action, result]);
        },
        
        /**
         * Handle system alert
         */
        handleSystemAlert: function(payload) {
            var level = payload.level; // info, warning, error, critical
            var message = payload.message;
            var source = payload.source;
            
            // Show appropriate notification
            this.showSystemAlert(level, message, source);
            
            // Update system status if critical
            if (level === 'critical' || level === 'error') {
                this.updateSystemStatus('error');
            }
            
            // Trigger custom event
            $(document).trigger('skz:systemAlert', [level, message, source]);
        },
        
        /**
         * Handle heartbeat
         */
        handleHeartbeat: function(payload) {
            this.state.lastHeartbeat = new Date();
            
            // Update connection status
            if (!this.state.isConnected) {
                this.state.isConnected = true;
                this.onConnectionRestored();
            }
        },
        
        /**
         * Update status indicators
         */
        updateStatusIndicators: function(agentId, status) {
            var $indicators = $('[data-agent-id="' + agentId + '"]');
            
            $indicators.each(function() {
                var $indicator = $(this);
                var $status = $indicator.find('.skz-agent-status, .skz-agent-status-indicator');
                
                $status.removeClass('active warning error inactive')
                       .addClass(status);
                
                // Add animation for status changes
                $indicator.addClass('status-updating');
                setTimeout(function() {
                    $indicator.removeClass('status-updating');
                }, 500);
            });
        },
        
        /**
         * Update agent cards
         */
        updateAgentCards: function(agentId, status) {
            var $cards = $('.skz-agent-card[data-agent-id="' + agentId + '"]');
            
            $cards.removeClass('active warning error inactive')
                  .addClass(status);
        },
        
        /**
         * Add to activity timeline
         */
        addToActivityTimeline: function(agentId, action, timestamp) {
            var $timeline = $('.skz-actions-timeline');
            if (!$timeline.length) return;
            
            var timeString = new Date(timestamp).toLocaleTimeString('en-US', {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit'
            });
            
            var $newItem = $('<div class="skz-timeline-item">')
                .html(`
                    <div class="skz-timeline-marker ${agentId}"></div>
                    <div class="skz-timeline-content">
                        <span class="skz-action-text">${action}</span>
                        <span class="skz-action-time">${timeString}</span>
                    </div>
                `);
            
            $timeline.prepend($newItem);
            
            // Limit timeline items
            var $items = $timeline.find('.skz-timeline-item');
            if ($items.length > 5) {
                $items.slice(5).remove();
            }
            
            // Animate new item
            $newItem.css('opacity', 0).animate({ opacity: 1 }, 300);
        },
        
        /**
         * Update metrics
         */
        updateMetrics: function(result) {
            this.state.metrics.totalRequests++;
            
            if (result.success) {
                this.state.metrics.successfulRequests++;
            } else {
                this.state.metrics.failedRequests++;
            }
            
            if (result.responseTime) {
                var total = this.state.metrics.totalRequests;
                var current = this.state.metrics.averageResponseTime;
                this.state.metrics.averageResponseTime = 
                    ((current * (total - 1)) + result.responseTime) / total;
            }
            
            // Update success rate display
            var successRate = (this.state.metrics.successfulRequests / this.state.metrics.totalRequests) * 100;
            $('.skz-stat:contains("%")').find('span').text(successRate.toFixed(1) + '%');
        },
        
        /**
         * Update last activity
         */
        updateLastActivity: function(agentId, timestamp) {
            var timeString = new Date(timestamp).toLocaleTimeString('en-US', {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit'
            });
            
            $('#lastUpdateTime').text(timeString);
        },
        
        /**
         * Show status notification
         */
        showStatusNotification: function(agentId, status, message) {
            var agent = SKZ.AgentUI.getAgent(agentId);
            var agentName = agent ? agent.name : agentId;
            
            var type = status === 'error' ? 'error' : 'warning';
            var title = agentName + ' Status Change';
            
            if (window.skzNotifications) {
                window.skzNotifications.showNotification(type, title, message);
            }
        },
        
        /**
         * Show action notification
         */
        showActionNotification: function(agentId, action, result) {
            var agent = SKZ.AgentUI.getAgent(agentId);
            var agentName = agent ? agent.name : agentId;
            
            var type = result.success ? 'success' : 'error';
            var title = agentName + ' Action';
            var message = action + (result.success ? ' completed successfully' : ' failed');
            
            if (window.skzNotifications) {
                window.skzNotifications.showNotification(type, title, message);
            }
        },
        
        /**
         * Show system alert
         */
        showSystemAlert: function(level, message, source) {
            var type = level === 'critical' || level === 'error' ? 'error' : 
                      level === 'warning' ? 'warning' : 'info';
            
            var title = 'System Alert';
            if (source) {
                title += ' (' + source + ')';
            }
            
            if (window.skzNotifications) {
                window.skzNotifications.showNotification(type, title, message);
            }
        },
        
        /**
         * Update system status
         */
        updateSystemStatus: function(status) {
            var $statusIndicator = $('.skz-status-indicator');
            var $statusText = $('.skz-status-text');
            
            $statusIndicator.removeClass('active warning error')
                           .addClass(status === 'operational' ? 'active' : status);
            
            var statusText = status === 'operational' ? 'All Systems Operational' :
                            status === 'warning' ? 'System Warning' : 'System Error';
            $statusText.text(statusText);
        },
        
        /**
         * Start heartbeat
         */
        startHeartbeat: function() {
            var self = this;
            
            setInterval(function() {
                if (self.state.websocket && self.state.isConnected) {
                    try {
                        self.state.websocket.send(JSON.stringify({
                            type: 'heartbeat',
                            timestamp: new Date().toISOString()
                        }));
                    } catch (error) {
                        console.error('Failed to send heartbeat:', error);
                    }
                }
            }, this.config.heartbeatInterval);
        },
        
        /**
         * Start polling fallback
         */
        startPolling: function() {
            var self = this;
            
            this.state.pollTimer = setInterval(function() {
                if (SKZ.AgentUI) {
                    SKZ.AgentUI.loadAgentData();
                }
            }, this.config.pollInterval);
            
            console.log('Started polling fallback');
        },
        
        /**
         * Stop polling
         */
        stopPolling: function() {
            if (this.state.pollTimer) {
                clearInterval(this.state.pollTimer);
                this.state.pollTimer = null;
            }
        },
        
        /**
         * On connection established
         */
        onConnectionEstablished: function() {
            this.stopPolling();
            this.showConnectionStatus('connected');
            $(document).trigger('skz:connectionEstablished');
        },
        
        /**
         * On connection lost
         */
        onConnectionLost: function() {
            this.state.isConnected = false;
            this.showConnectionStatus('disconnected');
            this.attemptReconnect();
            $(document).trigger('skz:connectionLost');
        },
        
        /**
         * On connection restored
         */
        onConnectionRestored: function() {
            this.showConnectionStatus('reconnected');
            $(document).trigger('skz:connectionRestored');
        },
        
        /**
         * Show connection status
         */
        showConnectionStatus: function(status) {
            var message, type;
            
            switch (status) {
                case 'connected':
                    message = 'Connected to real-time monitoring';
                    type = 'success';
                    break;
                case 'disconnected':
                    message = 'Lost connection to monitoring service';
                    type = 'warning';
                    break;
                case 'reconnected':
                    message = 'Reconnected to monitoring service';
                    type = 'success';
                    break;
            }
            
            if (window.skzNotifications && message) {
                window.skzNotifications.showNotification(type, 'Connection Status', message);
            }
        },
        
        /**
         * Attempt reconnect
         */
        attemptReconnect: function() {
            if (this.state.reconnectAttempts >= this.config.maxReconnectAttempts) {
                console.log('Max reconnection attempts reached, falling back to polling');
                this.startPolling();
                return;
            }
            
            this.state.reconnectAttempts++;
            
            var self = this;
            setTimeout(function() {
                console.log('Attempting to reconnect... (attempt ' + 
                           self.state.reconnectAttempts + ')');
                self.connectWebSocket();
            }, this.config.reconnectDelay);
        },
        
        /**
         * Handle connection error
         */
        handleConnectionError: function() {
            console.error('WebSocket connection error');
            this.state.isConnected = false;
        },
        
        /**
         * Process status update
         */
        processStatusUpdate: function(data) {
            // This is called when agent data is updated via other means
            // We can use this to sync our state
            if (data.lastUpdate) {
                this.state.lastUpdate = new Date(data.lastUpdate);
            }
        },
        
        /**
         * Pause monitoring
         */
        pauseMonitoring: function() {
            console.log('Pausing monitoring (page hidden)');
            // Reduce activity when page is not visible
        },
        
        /**
         * Resume monitoring
         */
        resumeMonitoring: function() {
            console.log('Resuming monitoring (page visible)');
            // Resume full activity when page becomes visible
            if (SKZ.AgentUI) {
                SKZ.AgentUI.loadAgentData();
            }
        },
        
        /**
         * Cleanup
         */
        cleanup: function() {
            if (this.state.websocket) {
                this.state.websocket.close();
            }
            
            this.stopPolling();
            
            console.log('SKZ Status Monitor cleaned up');
        },
        
        /**
         * Get connection status
         */
        isConnected: function() {
            return this.state.isConnected;
        },
        
        /**
         * Get metrics
         */
        getMetrics: function() {
            return this.state.metrics;
        }
    };
    
    // Initialize when document is ready
    $(document).ready(function() {
        // Small delay to ensure AgentUI is initialized first
        setTimeout(function() {
            SKZ.StatusMonitor.init();
        }, 100);
    });
    
})(jQuery);