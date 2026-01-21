<?php

/**
 * @file SKZRealtimeNotificationBridge.inc.php
 * 
 * Bridge between OJS notification system and SKZ real-time notifications
 * Extends existing OJS notification infrastructure with WebSocket capabilities
 */

import('lib.pkp.classes.notification.NotificationDAO');

class SKZRealtimeNotificationBridge {
    
    private $realtimeServiceUrl;
    private $enabled;
    
    public function __construct() {
        $this->realtimeServiceUrl = Config::getVar('skz', 'realtime_service_url', 'http://localhost:5000');
        $this->enabled = Config::getVar('skz', 'realtime_notifications_enabled', true);
    }
    
    /**
     * Send real-time notification for agent status change
     */
    public function notifyAgentStatusChange($agentId, $status, $metadata = array()) {
        if (!$this->enabled) {
            return false;
        }
        
        $notification = array(
            'type' => 'agent_status_change',
            'agent_id' => $agentId,
            'status' => $status,
            'metadata' => $metadata,
            'timestamp' => date('c'),
            'source' => 'ojs'
        );
        
        return $this->sendToRealtimeService($notification);
    }
    
    /**
     * Send real-time notification for workflow events
     */
    public function notifyWorkflowEvent($eventType, $data = array()) {
        if (!$this->enabled) {
            return false;
        }
        
        $notification = array(
            'type' => 'workflow_event',
            'event_type' => $eventType,
            'data' => $data,
            'timestamp' => date('c'),
            'source' => 'ojs'
        );
        
        return $this->sendToRealtimeService($notification);
    }
    
    /**
     * Send real-time notification for manuscript updates
     */
    public function notifyManuscriptUpdate($manuscriptId, $updateType, $data = array()) {
        if (!$this->enabled) {
            return false;
        }
        
        $notification = array(
            'type' => 'manuscript_update',
            'manuscript_id' => $manuscriptId,
            'update_type' => $updateType,
            'data' => $data,
            'timestamp' => date('c'),
            'source' => 'ojs'
        );
        
        return $this->sendToRealtimeService($notification);
    }
    
    /**
     * Extend OJS notification system to include real-time notifications
     */
    public function enhanceOJSNotification($userId, $notificationType, $contextId = null, $data = array()) {
        // Send traditional OJS notification first
        $notificationDao = DAORegistry::getDAO('NotificationDAO');
        $notification = $notificationDao->newDataObject();
        $notification->setUserId($userId);
        $notification->setType($notificationType);
        $notification->setContextId($contextId);
        $notification->setLevel(NOTIFICATION_LEVEL_NORMAL);
        $notificationDao->insertObject($notification);
        
        // Map OJS notification types to real-time notifications
        $this->mapOJSNotificationToRealtime($notificationType, $data);
        
        return $notification;
    }
    
    /**
     * Map OJS notification types to real-time notifications
     */
    private function mapOJSNotificationToRealtime($ojsNotificationType, $data = array()) {
        switch ($ojsNotificationType) {
            case NOTIFICATION_TYPE_EDITOR_ASSIGNMENT_SUBMISSION:
                $this->notifyWorkflowEvent('editor_assigned', $data);
                break;
                
            case NOTIFICATION_TYPE_EDITOR_DECISION_ACCEPT:
                $this->notifyManuscriptUpdate($data['submissionId'] ?? 0, 'accepted', $data);
                $this->notifyAgentStatusChange('editorial_orchestration', array(
                    'status' => 'active',
                    'last_action' => 'Manuscript accepted',
                    'performance' => array('total_actions' => 1)
                ));
                break;
                
            case NOTIFICATION_TYPE_EDITOR_DECISION_DECLINE:
                $this->notifyManuscriptUpdate($data['submissionId'] ?? 0, 'declined', $data);
                break;
                
            case NOTIFICATION_TYPE_REVIEWER_COMMENT:
                $this->notifyWorkflowEvent('review_submitted', $data);
                $this->notifyAgentStatusChange('review_coordination', array(
                    'status' => 'processing',
                    'last_action' => 'Review received',
                    'active_tasks' => 1
                ));
                break;
                
            case NOTIFICATION_TYPE_SUBMISSION_SUBMITTED:
                $this->notifyManuscriptUpdate($data['submissionId'] ?? 0, 'submitted', $data);
                $this->notifyAgentStatusChange('submission_assistant', array(
                    'status' => 'processing',
                    'last_action' => 'New submission processing',
                    'active_tasks' => 1
                ));
                break;
                
            case NOTIFICATION_TYPE_PUBLISHED_ISSUE:
                $this->notifyWorkflowEvent('issue_published', $data);
                $this->notifyAgentStatusChange('publishing_production', array(
                    'status' => 'active',
                    'last_action' => 'Issue published',
                    'performance' => array('total_actions' => 1)
                ));
                break;
                
            default:
                // Generic workflow event for unmapped notifications
                $this->notifyWorkflowEvent('ojs_notification', array_merge($data, array(
                    'ojs_type' => $ojsNotificationType
                )));
                break;
        }
    }
    
    /**
     * Send notification to real-time service via HTTP
     */
    private function sendToRealtimeService($notification) {
        try {
            $url = $this->realtimeServiceUrl . '/api/v1/notifications/from-ojs';
            
            $postData = json_encode($notification);
            
            $options = array(
                'http' => array(
                    'header'  => "Content-type: application/json\r\n",
                    'method'  => 'POST',
                    'content' => $postData,
                    'timeout' => 5 // 5 second timeout
                )
            );
            
            $context = stream_context_create($options);
            $result = file_get_contents($url, false, $context);
            
            if ($result === FALSE) {
                error_log("SKZ Real-time Notification: Failed to send notification to service");
                return false;
            }
            
            $response = json_decode($result, true);
            return isset($response['status']) && $response['status'] === 'success';
            
        } catch (Exception $e) {
            error_log("SKZ Real-time Notification Error: " . $e->getMessage());
            return false;
        }
    }
    
    /**
     * Get real-time service status
     */
    public function getRealtimeServiceStatus() {
        try {
            $url = $this->realtimeServiceUrl . '/api/v1/realtime/status';
            
            $options = array(
                'http' => array(
                    'method'  => 'GET',
                    'timeout' => 3
                )
            );
            
            $context = stream_context_create($options);
            $result = file_get_contents($url, false, $context);
            
            if ($result === FALSE) {
                return array('status' => 'error', 'message' => 'Service unavailable');
            }
            
            return json_decode($result, true);
            
        } catch (Exception $e) {
            return array('status' => 'error', 'message' => $e->getMessage());
        }
    }
    
    /**
     * Initialize real-time notifications JavaScript in OJS interface
     */
    public function includeRealtimeJavaScript() {
        if (!$this->enabled) {
            return '';
        }
        
        $serviceUrl = json_encode($this->realtimeServiceUrl);
        
        return "
        <script src='https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js'></script>
        <script>
        // Initialize OJS Real-time Notifications
        (function() {
            if (typeof io === 'undefined') {
                console.warn('Socket.IO not loaded, real-time notifications disabled');
                return;
            }
            
            const socket = io($serviceUrl);
            let notificationCount = 0;
            
            socket.on('connect', function() {
                console.log('Connected to SKZ real-time service');
                showOJSNotification('Connected to real-time updates', 'success');
            });
            
            socket.on('disconnect', function() {
                console.log('Disconnected from SKZ real-time service');
                showOJSNotification('Lost real-time connection', 'warning');
            });
            
            socket.on('notification', function(data) {
                console.log('Real-time notification:', data);
                handleRealtimeNotification(data);
            });
            
            function handleRealtimeNotification(notification) {
                notificationCount++;
                
                // Update notification counter if exists
                const counter = document.getElementById('skz-notification-counter');
                if (counter) {
                    counter.textContent = notificationCount;
                    counter.style.display = notificationCount > 0 ? 'inline' : 'none';
                }
                
                // Show notification based on type
                let message = '';
                let type = 'info';
                
                switch (notification.type) {
                    case 'agent_status_change':
                        message = 'Agent ' + notification.agent_id + ' status updated';
                        type = 'info';
                        break;
                    case 'workflow_event':
                        message = 'Workflow event: ' + notification.event_type;
                        type = 'info';
                        break;
                    case 'manuscript_update':
                        message = 'Manuscript ' + notification.manuscript_id + ' updated';
                        type = 'success';
                        break;
                    default:
                        message = 'Real-time update received';
                        break;
                }
                
                showOJSNotification(message, type);
                
                // Trigger custom event for other parts of OJS to listen to
                document.dispatchEvent(new CustomEvent('skz-realtime-notification', {
                    detail: notification
                }));
            }
            
            function showOJSNotification(message, type) {
                // Try to use existing OJS notification system
                if (typeof $.pkp !== 'undefined' && $.pkp.classes && $.pkp.classes.notification) {
                    // Use PKP notification helper
                    const notification = {
                        content: message,
                        level: type === 'error' ? 'error' : 'normal'
                    };
                    
                    // Trigger notify event
                    $('body').trigger('notifyUser', [notification]);
                } else {
                    // Fallback to browser notification
                    if (Notification.permission === 'granted') {
                        new Notification('SKZ Agent Update', {
                            body: message,
                            icon: '/favicon.ico'
                        });
                    } else {
                        console.log('SKZ Notification:', message);
                    }
                }
            }
            
            // Request notification permission
            if ('Notification' in window && Notification.permission === 'default') {
                Notification.requestPermission();
            }
            
            // Add notification indicator to navigation if not exists
            function addNotificationIndicator() {
                const nav = document.querySelector('.pkp_navigation_primary');
                if (nav && !document.getElementById('skz-notification-indicator')) {
                    const indicator = document.createElement('div');
                    indicator.id = 'skz-notification-indicator';
                    indicator.innerHTML = `
                        <div style='position: relative; display: inline-block; margin-left: 10px;'>
                            <span style='font-size: 12px; color: #666;'>Live Updates</span>
                            <span id='skz-notification-counter' style='display: none; position: absolute; top: -8px; right: -8px; background: #e74c3c; color: white; border-radius: 50%; width: 16px; height: 16px; text-align: center; font-size: 10px; line-height: 16px;'>0</span>
                            <div id='skz-connection-status' style='width: 8px; height: 8px; background: #e74c3c; border-radius: 50%; margin-top: 2px;'></div>
                        </div>
                    `;
                    nav.appendChild(indicator);
                }
            }
            
            // Update connection status indicator
            socket.on('connect', function() {
                const status = document.getElementById('skz-connection-status');
                if (status) {
                    status.style.background = '#27ae60';
                }
            });
            
            socket.on('disconnect', function() {
                const status = document.getElementById('skz-connection-status');
                if (status) {
                    status.style.background = '#e74c3c';
                }
            });
            
            // Initialize when DOM is ready
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', addNotificationIndicator);
            } else {
                addNotificationIndicator();
            }
            
        })();
        </script>
        ";
    }
}

// Global instance
global $skzRealtimeBridge;
$skzRealtimeBridge = new SKZRealtimeNotificationBridge();