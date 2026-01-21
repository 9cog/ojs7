/**
 * SKZ Workflow Integration JavaScript
 * JavaScript for workflow page integration and controls
 */

(function($) {
    'use strict';
    
    // Extend SKZ namespace
    window.SKZ = window.SKZ || {};
    
    /**
     * SKZ Workflow Integration
     */
    SKZ.WorkflowIntegration = {
        
        // Configuration
        config: {
            apiBaseUrl: '/index.php/journal/skzAgents/api',
            refreshInterval: 30000,
            animationDuration: 300,
            autoSaveDelay: 2000
        },
        
        // State
        state: {
            currentSubmissionId: null,
            activeWorkflow: null,
            agentRecommendations: [],
            workflowStage: null,
            autoSaveTimer: null
        },
        
        /**
         * Initialize workflow integration
         */
        init: function() {
            console.log('Initializing SKZ Workflow Integration...');
            
            this.detectWorkflowContext();
            this.bindWorkflowEvents();
            this.loadWorkflowData();
            this.initializeAgentControls();
            
            console.log('SKZ Workflow Integration initialized');
        },
        
        /**
         * Detect workflow context
         */
        detectWorkflowContext: function() {
            // Try to get submission ID from URL or page data
            var urlParams = new URLSearchParams(window.location.search);
            this.state.currentSubmissionId = urlParams.get('submissionId') || 
                                           urlParams.get('id') ||
                                           $('input[name="submissionId"]').val() ||
                                           $('[data-submission-id]').data('submission-id');
            
            // Detect workflow stage
            this.state.workflowStage = this.detectWorkflowStage();
            
            console.log('Workflow context:', {
                submissionId: this.state.currentSubmissionId,
                stage: this.state.workflowStage
            });
        },
        
        /**
         * Detect current workflow stage
         */
        detectWorkflowStage: function() {
            var path = window.location.pathname;
            var params = new URLSearchParams(window.location.search);
            
            if (path.includes('/submission/') || params.get('op') === 'submission') {
                return 'submission';
            } else if (path.includes('/workflow/') || params.get('op') === 'workflow') {
                var stageId = params.get('stageId');
                switch (stageId) {
                    case '1': return 'submission';
                    case '3': return 'external_review';
                    case '4': return 'editorial_review';
                    case '5': return 'copyediting';
                    case '6': return 'production';
                    default: return 'editorial_review';
                }
            } else if (path.includes('/authorDashboard/')) {
                return 'author_dashboard';
            }
            
            return 'unknown';
        },
        
        /**
         * Bind workflow events
         */
        bindWorkflowEvents: function() {
            var self = this;
            
            // Agent control buttons
            $(document).on('click', '[data-agent-action]', function(e) {
                e.preventDefault();
                var action = $(this).data('agent-action');
                var agentId = $(this).data('agent-id');
                self.handleAgentAction(action, agentId, $(this));
            });
            
            // Workflow control toggle
            $(document).on('click', '.skz-controls-toggle', function() {
                self.toggleWorkflowControls();
            });
            
            // Agent recommendation actions
            $(document).on('click', '.skz-recommendation-action', function(e) {
                e.preventDefault();
                var recommendationId = $(this).data('recommendation-id');
                var action = $(this).data('action');
                self.handleRecommendationAction(recommendationId, action);
            });
            
            // Listen for form submissions to trigger agent analysis
            $(document).on('submit', 'form[id*="submission"], form[id*="workflow"]', function() {
                self.onWorkflowFormSubmit($(this));
            });
            
            // Listen for file uploads
            $(document).on('change', 'input[type="file"]', function() {
                if (this.files.length > 0) {
                    self.onFileUpload($(this));
                }
            });
            
            // Listen for agent status updates
            $(document).on('skz:agentStatusChanged', function(event, agentId, status, data) {
                self.handleAgentStatusChange(agentId, status, data);
            });
            
            // Listen for workflow progress updates
            $(document).on('skz:workflowProgressUpdated', function(event, submissionId, progress, stage) {
                if (submissionId == self.state.currentSubmissionId) {
                    self.updateWorkflowProgress(progress, stage);
                }
            });
        },
        
        /**
         * Load workflow data
         */
        loadWorkflowData: function() {
            if (!this.state.currentSubmissionId) {
                console.log('No submission ID found, skipping workflow data load');
                return;
            }
            
            var self = this;
            
            $.ajax({
                url: this.config.apiBaseUrl + '/workflow/' + this.state.currentSubmissionId,
                method: 'GET',
                success: function(data) {
                    self.updateWorkflowData(data);
                },
                error: function(xhr, status, error) {
                    console.error('Failed to load workflow data:', error);
                }
            });
        },
        
        /**
         * Update workflow data
         */
        updateWorkflowData: function(data) {
            this.state.activeWorkflow = data;
            
            // Update progress
            if (data.progress !== undefined) {
                this.updateWorkflowProgress(data.progress, data.stage);
            }
            
            // Update recommendations
            if (data.recommendations) {
                this.updateRecommendations(data.recommendations);
            }
            
            // Update agent status
            if (data.agentStatus) {
                this.updateAgentStatus(data.agentStatus);
            }
            
            // Trigger custom event
            $(document).trigger('skz:workflowDataUpdated', [data]);
        },
        
        /**
         * Initialize agent controls
         */
        initializeAgentControls: function() {
            // Set up agent control interactions
            this.setupAgentCards();
            this.setupRecommendationCards();
            this.setupProgressTracking();
        },
        
        /**
         * Setup agent cards
         */
        setupAgentCards: function() {
            $('.skz-agent-card').each(function() {
                var $card = $(this);
                var agentId = $card.data('agent-id');
                
                // Add hover effects
                $card.hover(
                    function() { $(this).addClass('hover'); },
                    function() { $(this).removeClass('hover'); }
                );
                
                // Add click handler for selection
                $card.on('click', function() {
                    $('.skz-agent-card').removeClass('selected');
                    $(this).addClass('selected');
                    
                    SKZ.WorkflowIntegration.selectAgent(agentId);
                });
            });
        },
        
        /**
         * Setup recommendation cards
         */
        setupRecommendationCards: function() {
            $('.skz-recommendation').each(function() {
                var $rec = $(this);
                
                // Add expand/collapse for long recommendations
                var $content = $rec.find('.skz-rec-content p');
                if ($content.text().length > 100) {
                    $content.addClass('expandable');
                    
                    var $toggle = $('<button class="skz-expand-btn">Show more</button>');
                    $rec.append($toggle);
                    
                    $toggle.on('click', function() {
                        if ($content.hasClass('expanded')) {
                            $content.removeClass('expanded');
                            $(this).text('Show more');
                        } else {
                            $content.addClass('expanded');
                            $(this).text('Show less');
                        }
                    });
                }
            });
        },
        
        /**
         * Setup progress tracking
         */
        setupProgressTracking: function() {
            // Animate progress bars
            $('.skz-progress-fill').each(function() {
                var $fill = $(this);
                var targetWidth = $fill.css('width');
                
                $fill.css('width', '0%').animate({
                    width: targetWidth
                }, 1000);
            });
        },
        
        /**
         * Handle agent action
         */
        handleAgentAction: function(action, agentId, $button) {
            console.log('Agent action:', action, agentId);
            
            var self = this;
            var originalHtml = $button.html();
            var originalDisabled = $button.prop('disabled');
            
            // Show loading state
            $button.prop('disabled', true)
                   .html('<i class="fas fa-spinner fa-spin"></i> Processing...');
            
            switch (action) {
                case 'view':
                    this.viewAgentDetails(agentId);
                    break;
                case 'pause':
                case 'resume':
                    this.toggleAgent(agentId);
                    break;
                case 'configure':
                    this.configureAgent(agentId);
                    break;
                case 'analyze':
                    this.runAgentAnalysis(agentId);
                    break;
                case 'refresh':
                    this.refreshAgentData(agentId);
                    break;
                default:
                    console.warn('Unknown agent action:', action);
            }
            
            // Reset button state after delay
            setTimeout(function() {
                $button.prop('disabled', originalDisabled)
                       .html(originalHtml);
            }, 2000);
        },
        
        /**
         * View agent details
         */
        viewAgentDetails: function(agentId) {
            var self = this;
            
            $.ajax({
                url: this.config.apiBaseUrl + '/agents/' + agentId + '/details',
                method: 'GET',
                data: { submissionId: this.state.currentSubmissionId },
                success: function(data) {
                    self.showAgentDetailsModal(data);
                },
                error: function(xhr, status, error) {
                    self.showError('Failed to load agent details: ' + error);
                }
            });
        },
        
        /**
         * Toggle agent (pause/resume)
         */
        toggleAgent: function(agentId) {
            var self = this;
            
            $.ajax({
                url: this.config.apiBaseUrl + '/agents/' + agentId + '/toggle',
                method: 'POST',
                data: { submissionId: this.state.currentSubmissionId },
                success: function(data) {
                    self.showSuccess('Agent ' + (data.paused ? 'paused' : 'resumed'));
                    self.updateAgentCard(agentId, data);
                },
                error: function(xhr, status, error) {
                    self.showError('Failed to toggle agent: ' + error);
                }
            });
        },
        
        /**
         * Configure agent
         */
        configureAgent: function(agentId) {
            // This would open a configuration modal or page
            var configUrl = this.config.apiBaseUrl.replace('/api', '') + '/configure/' + agentId;
            window.open(configUrl, 'agentConfig', 'width=800,height=600');
        },
        
        /**
         * Run agent analysis
         */
        runAgentAnalysis: function(agentId) {
            var self = this;
            
            $.ajax({
                url: this.config.apiBaseUrl + '/agents/' + agentId + '/analyze',
                method: 'POST',
                data: { 
                    submissionId: this.state.currentSubmissionId,
                    stage: this.state.workflowStage
                },
                success: function(data) {
                    self.showSuccess('Analysis started');
                    self.handleAnalysisResult(agentId, data);
                },
                error: function(xhr, status, error) {
                    self.showError('Failed to start analysis: ' + error);
                }
            });
        },
        
        /**
         * Refresh agent data
         */
        refreshAgentData: function(agentId) {
            var self = this;
            
            setTimeout(function() {
                self.loadWorkflowData();
                self.showSuccess('Agent data refreshed');
            }, 1000);
        },
        
        /**
         * Handle analysis result
         */
        handleAnalysisResult: function(agentId, result) {
            if (result.recommendations) {
                this.addRecommendations(result.recommendations);
            }
            
            if (result.progress) {
                this.updateWorkflowProgress(result.progress);
            }
            
            if (result.notifications) {
                this.showAnalysisNotifications(result.notifications);
            }
        },
        
        /**
         * Select agent
         */
        selectAgent: function(agentId) {
            console.log('Selected agent:', agentId);
            
            // Update UI to show selected state
            $('.skz-agent-card').removeClass('selected');
            $('[data-agent-id="' + agentId + '"]').addClass('selected');
            
            // Load agent-specific recommendations or data
            this.loadAgentRecommendations(agentId);
            
            // Trigger custom event
            $(document).trigger('skz:agentSelected', [agentId]);
        },
        
        /**
         * Load agent recommendations
         */
        loadAgentRecommendations: function(agentId) {
            var self = this;
            
            $.ajax({
                url: this.config.apiBaseUrl + '/agents/' + agentId + '/recommendations',
                method: 'GET',
                data: { submissionId: this.state.currentSubmissionId },
                success: function(data) {
                    self.showAgentRecommendations(agentId, data.recommendations);
                },
                error: function(xhr, status, error) {
                    console.error('Failed to load agent recommendations:', error);
                }
            });
        },
        
        /**
         * Show agent recommendations
         */
        showAgentRecommendations: function(agentId, recommendations) {
            var $container = $('.skz-recommendations-content');
            if (!$container.length) return;
            
            $container.empty();
            
            recommendations.forEach(function(rec) {
                var $recElement = $('<div class="skz-recommendation">')
                    .html(`
                        <div class="skz-rec-icon">
                            <i class="${rec.icon || 'fas fa-lightbulb'}"></i>
                        </div>
                        <div class="skz-rec-content">
                            <strong>${rec.title}</strong>
                            <p>${rec.description}</p>
                        </div>
                        ${rec.actionable ? '<div class="skz-rec-actions"></div>' : ''}
                    `);
                
                $container.append($recElement);
            });
        },
        
        /**
         * Toggle workflow controls
         */
        toggleWorkflowControls: function() {
            var $content = $('.skz-controls-content');
            var $toggle = $('.skz-controls-toggle i');
            
            if ($content.is(':visible')) {
                $content.slideUp(this.config.animationDuration);
                $toggle.removeClass('fa-chevron-down').addClass('fa-chevron-right');
            } else {
                $content.slideDown(this.config.animationDuration);
                $toggle.removeClass('fa-chevron-right').addClass('fa-chevron-down');
            }
        },
        
        /**
         * Update workflow progress
         */
        updateWorkflowProgress: function(progress, stage) {
            $('.skz-progress-fill').animate({
                width: progress + '%'
            }, this.config.animationDuration);
            
            $('.skz-progress-percentage').text(progress + '%');
            
            if (stage) {
                this.state.workflowStage = stage;
                this.updateStageIndicators(stage);
            }
        },
        
        /**
         * Update stage indicators
         */
        updateStageIndicators: function(stage) {
            $('.skz-stage-badge').removeClass('submission editorial_review peer_review copyediting production published')
                                  .addClass(stage)
                                  .text(stage.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()));
        },
        
        /**
         * Update recommendations
         */
        updateRecommendations: function(recommendations) {
            this.state.agentRecommendations = recommendations;
            
            var $container = $('.skz-recommendations-content');
            if (!$container.length) return;
            
            $container.empty();
            
            recommendations.forEach(function(rec, index) {
                var $recElement = $('<div class="skz-recommendation">')
                    .data('recommendation-id', index)
                    .html(`
                        <div class="skz-rec-icon">
                            <i class="${rec.type === 'success' ? 'fas fa-check-circle text-success' : 
                                       rec.type === 'warning' ? 'fas fa-exclamation-triangle text-warning' : 
                                       'fas fa-info-circle text-info'}"></i>
                        </div>
                        <div class="skz-rec-content">
                            <strong>${rec.title}</strong>
                            <p>${rec.message}</p>
                        </div>
                    `);
                
                $container.append($recElement);
            });
        },
        
        /**
         * Update agent status
         */
        updateAgentStatus: function(agentStatus) {
            Object.keys(agentStatus).forEach(function(agentId) {
                var status = agentStatus[agentId];
                SKZ.WorkflowIntegration.updateAgentCard(agentId, status);
            });
        },
        
        /**
         * Update agent card
         */
        updateAgentCard: function(agentId, data) {
            var $card = $('[data-agent-id="' + agentId + '"]');
            if (!$card.length) return;
            
            // Update status indicator
            var $statusIndicator = $card.find('.skz-agent-status-indicator');
            $statusIndicator.removeClass('active warning error inactive')
                           .addClass(data.status);
            
            // Update card class
            $card.removeClass('active warning error inactive')
                 .addClass(data.status);
            
            // Update last activity if provided
            if (data.lastActivity) {
                $card.find('.skz-last-activity').text(data.lastActivity);
            }
        },
        
        /**
         * Handle agent status change
         */
        handleAgentStatusChange: function(agentId, status, data) {
            this.updateAgentCard(agentId, { status: status });
            
            // Show notification for significant status changes
            if (status === 'error') {
                this.showError('Agent ' + agentId + ' encountered an error');
            } else if (status === 'warning') {
                this.showWarning('Agent ' + agentId + ' is showing warnings');
            }
        },
        
        /**
         * Handle workflow form submit
         */
        onWorkflowFormSubmit: function($form) {
            console.log('Workflow form submitted:', $form.attr('id'));
            
            // Trigger agent analysis based on form type
            var formId = $form.attr('id');
            if (formId && formId.includes('submission')) {
                this.triggerSubmissionAnalysis();
            } else if (formId && formId.includes('review')) {
                this.triggerReviewAnalysis();
            }
        },
        
        /**
         * Handle file upload
         */
        onFileUpload: function($input) {
            console.log('File uploaded:', $input[0].files[0].name);
            
            // Trigger content analysis
            this.triggerContentAnalysis($input[0].files[0]);
        },
        
        /**
         * Trigger submission analysis
         */
        triggerSubmissionAnalysis: function() {
            if (!this.state.currentSubmissionId) return;
            
            console.log('Triggering submission analysis...');
            this.runAgentAnalysis('submission_assistant');
        },
        
        /**
         * Trigger review analysis
         */
        triggerReviewAnalysis: function() {
            if (!this.state.currentSubmissionId) return;
            
            console.log('Triggering review analysis...');
            this.runAgentAnalysis('review_coordination');
        },
        
        /**
         * Trigger content analysis
         */
        triggerContentAnalysis: function(file) {
            if (!this.state.currentSubmissionId) return;
            
            console.log('Triggering content analysis for:', file.name);
            this.runAgentAnalysis('content_quality');
        },
        
        /**
         * Show success message
         */
        showSuccess: function(message) {
            if (window.skzNotifications) {
                window.skzNotifications.showNotification('success', 'Success', message);
            } else {
                console.log('Success:', message);
            }
        },
        
        /**
         * Show error message
         */
        showError: function(message) {
            if (window.skzNotifications) {
                window.skzNotifications.showNotification('error', 'Error', message);
            } else {
                console.error('Error:', message);
            }
        },
        
        /**
         * Show warning message
         */
        showWarning: function(message) {
            if (window.skzNotifications) {
                window.skzNotifications.showNotification('warning', 'Warning', message);
            } else {
                console.warn('Warning:', message);
            }
        },
        
        /**
         * Get current submission ID
         */
        getCurrentSubmissionId: function() {
            return this.state.currentSubmissionId;
        },
        
        /**
         * Get workflow stage
         */
        getWorkflowStage: function() {
            return this.state.workflowStage;
        }
    };
    
    // Initialize when document is ready
    $(document).ready(function() {
        // Small delay to ensure other SKZ components are initialized first
        setTimeout(function() {
            SKZ.WorkflowIntegration.init();
        }, 200);
    });
    
})(jQuery);