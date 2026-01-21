/**
 * SKZ Agent UI JavaScript
 * Core JavaScript functionality for agent interface components
 */

(function($) {
    'use strict';
    
    // Global SKZ namespace
    window.SKZ = window.SKZ || {};
    
    /**
     * SKZ Agent UI Controller
     */
    SKZ.AgentUI = {
        
        // Configuration
        config: {
            apiBaseUrl: '/index.php/journal/skzAgents/api',
            refreshInterval: 30000, // 30 seconds
            animationDuration: 300,
            maxRetries: 3
        },
        
        // State
        state: {
            agents: [],
            systemStatus: 'unknown',
            lastUpdate: null,
            isConnected: false,
            retryCount: 0
        },
        
        /**
         * Initialize the agent UI
         */
        init: function() {
            console.log('Initializing SKZ Agent UI...');
            
            this.bindEvents();
            this.loadAgentData();
            this.startStatusMonitoring();
            this.initializeTooltips();
            
            console.log('SKZ Agent UI initialized');
        },
        
        /**
         * Bind UI events
         */
        bindEvents: function() {
            var self = this;
            
            // Agent indicator hover events
            $(document).on('mouseenter', '.skz-agent-indicator', function() {
                self.showAgentTooltip($(this));
            });
            
            $(document).on('mouseleave', '.skz-agent-indicator', function() {
                self.hideAgentTooltip($(this));
            });
            
            // Agent card clicks
            $(document).on('click', '.skz-agent-card', function() {
                var agentId = $(this).data('agent-id');
                self.selectAgent(agentId);
            });
            
            // Status bar clicks
            $(document).on('click', '.skz-agent-status-bar', function() {
                self.toggleStatusDetails();
            });
            
            // Refresh button
            $(document).on('click', '[data-skz-action="refresh"]', function(e) {
                e.preventDefault();
                self.refreshAgentData();
            });
            
            // Dashboard link
            $(document).on('click', '[data-skz-action="dashboard"]', function(e) {
                e.preventDefault();
                self.openDashboard();
            });
            
            // Window resize handler
            $(window).on('resize', function() {
                self.handleResize();
            });
        },
        
        /**
         * Load agent data from API
         */
        loadAgentData: function() {
            var self = this;
            
            $.ajax({
                url: self.config.apiBaseUrl + '/status',
                method: 'GET',
                timeout: 10000,
                success: function(data) {
                    self.updateAgentData(data);
                    self.state.isConnected = true;
                    self.state.retryCount = 0;
                },
                error: function(xhr, status, error) {
                    console.error('Failed to load agent data:', error);
                    self.handleConnectionError();
                }
            });
        },
        
        /**
         * Update agent data in UI
         */
        updateAgentData: function(data) {
            if (!data || !data.agents) {
                console.warn('Invalid agent data received');
                return;
            }
            
            this.state.agents = data.agents;
            this.state.systemStatus = data.systemStatus || 'unknown';
            this.state.lastUpdate = new Date();
            
            this.updateStatusBar(data);
            this.updateAgentIndicators(data.agents);
            this.updateWorkflowControls(data);
            
            // Trigger custom event
            $(document).trigger('skz:agentDataUpdated', [data]);
        },
        
        /**
         * Update status bar
         */
        updateStatusBar: function(data) {
            var $statusBar = $('.skz-agent-status-bar');
            if (!$statusBar.length) return;
            
            var $indicator = $statusBar.find('.skz-status-indicator');
            var $text = $statusBar.find('.skz-status-text');
            var $count = $statusBar.find('.skz-status-count');
            var $stats = $statusBar.find('.skz-quick-stats');
            
            // Update status indicator
            $indicator.removeClass('active warning error')
                      .addClass(data.systemStatus === 'operational' ? 'active' : 'warning');
            
            // Update status text
            var statusText = data.systemStatus === 'operational' ? 
                           'All Systems Operational' : 'System Warning';
            $text.text(statusText);
            
            // Update agent count
            $count.text(data.agents.length + ' agents');
            
            // Update quick stats
            if (data.totalActions && data.successRate) {
                $stats.find('.skz-stat:first span').text(data.successRate + '%');
                $stats.find('.skz-stat:last span').text(data.totalActions);
            }
        },
        
        /**
         * Update agent indicators
         */
        updateAgentIndicators: function(agents) {
            var $indicators = $('.skz-agent-indicators');
            if (!$indicators.length) return;
            
            agents.forEach(function(agent) {
                var $indicator = $indicators.find('[data-agent-id="' + agent.id + '"]');
                if ($indicator.length) {
                    var $status = $indicator.find('.skz-agent-status');
                    $status.removeClass('active warning error inactive')
                           .addClass(agent.status);
                    
                    // Update tooltip data
                    $indicator.attr('title', agent.name + ' - ' + agent.status);
                }
            });
        },
        
        /**
         * Update workflow controls
         */
        updateWorkflowControls: function(data) {
            var $controls = $('.skz-workflow-agent-controls');
            if (!$controls.length) return;
            
            // Update progress if available
            if (data.workflowProgress !== undefined) {
                var $progressFill = $controls.find('.skz-progress-fill');
                var $progressText = $controls.find('#workflowProgress');
                
                $progressFill.css('width', data.workflowProgress + '%');
                $progressText.text(data.workflowProgress + '%');
            }
            
            // Update agent cards status
            data.agents.forEach(function(agent) {
                var $card = $controls.find('[data-agent-id="' + agent.id + '"]');
                if ($card.length) {
                    var $statusIndicator = $card.find('.skz-agent-status-indicator');
                    $statusIndicator.removeClass('active warning error inactive')
                                   .addClass(agent.status);
                    
                    // Update card class
                    $card.removeClass('active warning error inactive')
                         .addClass(agent.status);
                }
            });
        },
        
        /**
         * Show agent tooltip
         */
        showAgentTooltip: function($indicator) {
            var agentName = $indicator.data('agent-name');
            var agentId = $indicator.data('agent-id');
            
            if (!agentName) return;
            
            var $tooltip = $('<div class="skz-agent-tooltip">')
                .text(agentName)
                .css({
                    position: 'absolute',
                    bottom: '-30px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    background: 'rgba(0, 0, 0, 0.8)',
                    color: 'white',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    fontSize: '11px',
                    whiteSpace: 'nowrap',
                    zIndex: 1000,
                    pointerEvents: 'none'
                });
            
            $indicator.css('position', 'relative').append($tooltip);
            
            setTimeout(function() {
                $tooltip.css('opacity', '1');
            }, 50);
        },
        
        /**
         * Hide agent tooltip
         */
        hideAgentTooltip: function($indicator) {
            $indicator.find('.skz-agent-tooltip').remove();
        },
        
        /**
         * Select an agent
         */
        selectAgent: function(agentId) {
            console.log('Selecting agent:', agentId);
            
            // Remove previous selection
            $('.skz-agent-card').removeClass('selected');
            $('.skz-agent-indicator').removeClass('selected');
            
            // Add selection
            $('[data-agent-id="' + agentId + '"]').addClass('selected');
            
            // Load agent details
            this.loadAgentDetails(agentId);
            
            // Trigger custom event
            $(document).trigger('skz:agentSelected', [agentId]);
        },
        
        /**
         * Load agent details
         */
        loadAgentDetails: function(agentId) {
            var self = this;
            
            $.ajax({
                url: self.config.apiBaseUrl + '/agents/' + agentId,
                method: 'GET',
                success: function(data) {
                    self.showAgentDetails(data);
                },
                error: function(xhr, status, error) {
                    console.error('Failed to load agent details:', error);
                    self.showNotification('error', 'Error', 'Failed to load agent details');
                }
            });
        },
        
        /**
         * Show agent details
         */
        showAgentDetails: function(agentData) {
            // This would typically open a modal or sidebar with agent details
            console.log('Agent details:', agentData);
            
            // For now, just show a notification
            this.showNotification('info', 'Agent Details', 
                'Details for ' + agentData.name + ' are now available');
        },
        
        /**
         * Toggle status details
         */
        toggleStatusDetails: function() {
            var $details = $('.skz-status-details');
            
            if ($details.length) {
                $details.slideUp(this.config.animationDuration, function() {
                    $(this).remove();
                });
            } else {
                this.showStatusDetails();
            }
        },
        
        /**
         * Show status details
         */
        showStatusDetails: function() {
            var $statusBar = $('.skz-agent-status-bar');
            var self = this;
            
            var $details = $('<div class="skz-status-details">')
                .css({
                    background: 'white',
                    border: '1px solid #e5e7eb',
                    borderTop: 'none',
                    padding: '12px 16px',
                    display: 'none'
                })
                .html(this.getStatusDetailsHtml());
            
            $statusBar.after($details);
            $details.slideDown(this.config.animationDuration);
        },
        
        /**
         * Get status details HTML
         */
        getStatusDetailsHtml: function() {
            var html = '<div class="skz-status-grid">';
            
            this.state.agents.forEach(function(agent) {
                html += '<div class="skz-status-item">';
                html += '<i class="' + agent.icon + '" style="color: ' + agent.color + '"></i>';
                html += '<span>' + agent.name + '</span>';
                html += '<span class="skz-status-badge ' + agent.status + '">' + agent.status + '</span>';
                html += '</div>';
            });
            
            html += '</div>';
            
            if (this.state.lastUpdate) {
                html += '<div class="skz-status-footer">';
                html += 'Last updated: ' + this.state.lastUpdate.toLocaleTimeString();
                html += '</div>';
            }
            
            return html;
        },
        
        /**
         * Refresh agent data
         */
        refreshAgentData: function() {
            var $refreshBtn = $('[data-skz-action="refresh"]');
            
            // Show loading state
            $refreshBtn.prop('disabled', true)
                      .find('i').addClass('fa-spin');
            
            var self = this;
            
            setTimeout(function() {
                self.loadAgentData();
                
                // Reset button state
                $refreshBtn.prop('disabled', false)
                          .find('i').removeClass('fa-spin');
                
                self.showNotification('success', 'Refreshed', 'Agent data has been updated');
            }, 1000);
        },
        
        /**
         * Open dashboard
         */
        openDashboard: function() {
            var dashboardUrl = '/index.php/journal/skzAgents/dashboard';
            window.open(dashboardUrl, '_blank', 'width=1200,height=800');
        },
        
        /**
         * Handle connection error
         */
        handleConnectionError: function() {
            this.state.isConnected = false;
            this.state.retryCount++;
            
            if (this.state.retryCount < this.config.maxRetries) {
                console.log('Retrying connection in 5 seconds...');
                setTimeout(() => {
                    this.loadAgentData();
                }, 5000);
            } else {
                this.showNotification('error', 'Connection Error', 
                    'Unable to connect to SKZ agents. Please check your connection.');
            }
        },
        
        /**
         * Start status monitoring
         */
        startStatusMonitoring: function() {
            var self = this;
            
            setInterval(function() {
                if (self.state.isConnected) {
                    self.loadAgentData();
                }
            }, this.config.refreshInterval);
        },
        
        /**
         * Initialize tooltips
         */
        initializeTooltips: function() {
            // Initialize any tooltips if using a tooltip library
            if (typeof $.fn.tooltip === 'function') {
                $('[data-toggle="tooltip"]').tooltip();
            }
        },
        
        /**
         * Handle window resize
         */
        handleResize: function() {
            // Handle responsive layout changes
            if (window.innerWidth <= 768) {
                $('.skz-agent-status-bar').addClass('mobile');
            } else {
                $('.skz-agent-status-bar').removeClass('mobile');
            }
        },
        
        /**
         * Show notification
         */
        showNotification: function(type, title, message) {
            // This would integrate with the notification system
            if (window.skzNotifications) {
                window.skzNotifications.showNotification(type, title, message);
            } else {
                console.log('Notification:', type, title, message);
            }
        },
        
        /**
         * Get agent by ID
         */
        getAgent: function(agentId) {
            return this.state.agents.find(function(agent) {
                return agent.id === agentId;
            });
        },
        
        /**
         * Get system status
         */
        getSystemStatus: function() {
            return this.state.systemStatus;
        },
        
        /**
         * Check if connected
         */
        isConnected: function() {
            return this.state.isConnected;
        }
    };
    
    // Initialize when document is ready
    $(document).ready(function() {
        SKZ.AgentUI.init();
    });
    
})(jQuery);