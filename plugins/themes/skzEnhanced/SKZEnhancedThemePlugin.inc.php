<?php

/**
 * @file plugins/themes/skzEnhanced/SKZEnhancedThemePlugin.inc.php
 *
 * SKZ Enhanced Theme Plugin - Theme modifications for agent interfaces
 * Extends the default OJS theme with autonomous agents interface components
 */

import('plugins.themes.default.DefaultThemePlugin');

class SKZEnhancedThemePlugin extends DefaultThemePlugin {
    
    /**
     * Initialize the theme's styles, scripts and hooks for SKZ agent integration
     */
    public function init() {
        // Initialize parent theme first
        parent::init();
        
        // Add SKZ-specific theme options
        $this->addSKZThemeOptions();
        
        // Add SKZ agent interface styles
        $this->addStyle('skzAgentInterface', 'styles/skz-agent-interface.less');
        $this->addStyle('skzStatusIndicators', 'styles/skz-status-indicators.less');
        $this->addStyle('skzWorkflowIntegration', 'styles/skz-workflow-integration.less');
        
        // Add agent interface JavaScript
        $this->addScript('skzAgentUI', 'js/skz-agent-ui.js');
        $this->addScript('skzStatusMonitor', 'js/skz-status-monitor.js');
        $this->addScript('skzWorkflowIntegration', 'js/skz-workflow-integration.js');
        
        // Register agent interface hooks
        $this->registerAgentInterfaceHooks();
        
        // Load agent status data for frontend
        $this->loadAgentStatusData();
    }
    
    /**
     * Add SKZ-specific theme configuration options
     */
    private function addSKZThemeOptions() {
        // Agent status display options
        $this->addOption('showAgentStatusBar', 'FieldOptions', [
            'label' => __('plugins.themes.skzEnhanced.option.showAgentStatusBar.label'),
            'description' => __('plugins.themes.skzEnhanced.option.showAgentStatusBar.description'),
            'options' => [
                [
                    'value' => true,
                    'label' => __('plugins.themes.skzEnhanced.option.showAgentStatusBar.enable'),
                ],
            ],
            'default' => true,
        ]);
        
        // Agent interface theme
        $this->addOption('agentInterfaceTheme', 'FieldOptions', [
            'type' => 'radio',
            'label' => __('plugins.themes.skzEnhanced.option.agentTheme.label'),
            'description' => __('plugins.themes.skzEnhanced.option.agentTheme.description'),
            'options' => [
                [
                    'value' => 'professional',
                    'label' => __('plugins.themes.skzEnhanced.option.agentTheme.professional'),
                ],
                [
                    'value' => 'modern',
                    'label' => __('plugins.themes.skzEnhanced.option.agentTheme.modern'),
                ],
                [
                    'value' => 'minimal',
                    'label' => __('plugins.themes.skzEnhanced.option.agentTheme.minimal'),
                ],
            ],
            'default' => 'professional',
        ]);
        
        // Real-time updates
        $this->addOption('enableRealTimeUpdates', 'FieldOptions', [
            'label' => __('plugins.themes.skzEnhanced.option.realTimeUpdates.label'),
            'description' => __('plugins.themes.skzEnhanced.option.realTimeUpdates.description'),
            'options' => [
                [
                    'value' => true,
                    'label' => __('plugins.themes.skzEnhanced.option.realTimeUpdates.enable'),
                ],
            ],
            'default' => true,
        ]);
        
        // Agent controls in workflow
        $this->addOption('showAgentControlsInWorkflow', 'FieldOptions', [
            'label' => __('plugins.themes.skzEnhanced.option.workflowControls.label'),
            'description' => __('plugins.themes.skzEnhanced.option.workflowControls.description'),
            'options' => [
                [
                    'value' => true,
                    'label' => __('plugins.themes.skzEnhanced.option.workflowControls.enable'),
                ],
            ],
            'default' => true,
        ]);
    }
    
    /**
     * Register hooks for agent interface integration
     */
    private function registerAgentInterfaceHooks() {
        // Add agent status bar to header
        HookRegistry::register('Templates::Common::Header::navbar', array($this, 'addAgentStatusBar'));
        
        // Add agent controls to submission workflow
        HookRegistry::register('Templates::Workflow::index', array($this, 'addWorkflowAgentControls'));
        
        // Add agent status to editorial workflow
        HookRegistry::register('Templates::Workflow::submission', array($this, 'addSubmissionAgentStatus'));
        
        // Add agent notifications
        HookRegistry::register('Templates::Common::footer', array($this, 'addAgentNotifications'));
        
        // Inject agent interface CSS variables based on theme options
        HookRegistry::register('TemplateManager::display', array($this, 'injectAgentThemeVariables'));
    }
    
    /**
     * Add agent status bar to the header
     */
    public function addAgentStatusBar($hookName, $params) {
        if (!$this->getOption('showAgentStatusBar')) {
            return false;
        }
        
        $smarty = $params[1];
        $output = $params[2];
        
        $agentStatusBar = $smarty->fetch($this->getTemplateResource('components/agent-status-bar.tpl'));
        $output = str_replace('</nav>', '</nav>' . $agentStatusBar, $output);
        $params[2] = $output;
        
        return false;
    }
    
    /**
     * Add agent controls to workflow pages
     */
    public function addWorkflowAgentControls($hookName, $params) {
        if (!$this->getOption('showAgentControlsInWorkflow')) {
            return false;
        }
        
        $smarty = $params[1];
        $output = $params[2];
        
        // Get current submission ID for agent context
        $request = Application::get()->getRequest();
        $submissionId = $request->getUserVar('submissionId');
        
        $smarty->assign('submissionId', $submissionId);
        $agentControls = $smarty->fetch($this->getTemplateResource('components/workflow-agent-controls.tpl'));
        
        // Insert agent controls into workflow interface
        $output = str_replace('<div class="pkp_workflow">', 
                             '<div class="pkp_workflow">' . $agentControls, $output);
        $params[2] = $output;
        
        return false;
    }
    
    /**
     * Add agent status information to submission pages
     */
    public function addSubmissionAgentStatus($hookName, $params) {
        $smarty = $params[1];
        $output = $params[2];
        
        $request = Application::get()->getRequest();
        $submissionId = $request->getUserVar('submissionId');
        
        if ($submissionId) {
            // Load agent status for this submission
            $agentStatus = $this->getSubmissionAgentStatus($submissionId);
            $smarty->assign('submissionAgentStatus', $agentStatus);
            
            $agentStatusWidget = $smarty->fetch($this->getTemplateResource('components/submission-agent-status.tpl'));
            
            // Insert agent status widget
            $output = str_replace('<div class="pkp_workflow_sidebar">', 
                                 '<div class="pkp_workflow_sidebar">' . $agentStatusWidget, $output);
            $params[2] = $output;
        }
        
        return false;
    }
    
    /**
     * Add agent notification system to footer
     */
    public function addAgentNotifications($hookName, $params) {
        if (!$this->getOption('enableRealTimeUpdates')) {
            return false;
        }
        
        $smarty = $params[1];
        $output = $params[2];
        
        $notificationSystem = $smarty->fetch($this->getTemplateResource('components/agent-notifications.tpl'));
        $output = str_replace('</body>', $notificationSystem . '</body>', $output);
        $params[2] = $output;
        
        return false;
    }
    
    /**
     * Inject agent theme variables into CSS
     */
    public function injectAgentThemeVariables($hookName, $params) {
        $templateMgr = $params[0];
        
        // Define theme-specific CSS variables
        $agentTheme = $this->getOption('agentInterfaceTheme');
        $themeVariables = $this->getAgentThemeVariables($agentTheme);
        
        // Add inline styles for agent theme variables
        $this->addStyle(
            'agentThemeVariables',
            ':root { ' . implode('; ', $themeVariables) . '; }',
            ['inline' => true]
        );
        
        return false;
    }
    
    /**
     * Get CSS variables for agent theme
     */
    private function getAgentThemeVariables($theme) {
        $variables = [];
        
        switch ($theme) {
            case 'professional':
                $variables = [
                    '--skz-agent-primary: #1e40af',
                    '--skz-agent-secondary: #6b7280',
                    '--skz-agent-success: #059669',
                    '--skz-agent-warning: #d97706',
                    '--skz-agent-error: #dc2626',
                    '--skz-agent-background: #f8fafc',
                    '--skz-agent-border: #e5e7eb',
                    '--skz-agent-text: #374151',
                    '--skz-agent-border-radius: 6px',
                    '--skz-agent-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1)',
                ];
                break;
                
            case 'modern':
                $variables = [
                    '--skz-agent-primary: #6366f1',
                    '--skz-agent-secondary: #8b5cf6',
                    '--skz-agent-success: #10b981',
                    '--skz-agent-warning: #f59e0b',
                    '--skz-agent-error: #ef4444',
                    '--skz-agent-background: #fafbfc',
                    '--skz-agent-border: #e2e8f0',
                    '--skz-agent-text: #1e293b',
                    '--skz-agent-border-radius: 12px',
                    '--skz-agent-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                ];
                break;
                
            case 'minimal':
                $variables = [
                    '--skz-agent-primary: #000000',
                    '--skz-agent-secondary: #666666',
                    '--skz-agent-success: #22c55e',
                    '--skz-agent-warning: #eab308',
                    '--skz-agent-error: #ef4444',
                    '--skz-agent-background: #ffffff',
                    '--skz-agent-border: #d1d5db',
                    '--skz-agent-text: #111827',
                    '--skz-agent-border-radius: 4px',
                    '--skz-agent-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05)',
                ];
                break;
        }
        
        return $variables;
    }
    
    /**
     * Load agent status data for frontend use
     */
    private function loadAgentStatusData() {
        // Check if SKZ Agents plugin is enabled
        $pluginRegistry = PluginRegistry::getRegistry();
        $skzPlugin = $pluginRegistry->getPlugin('generic', 'skzAgents');
        
        if ($skzPlugin && $skzPlugin->getEnabled()) {
            // Load basic agent status data
            $agentData = $this->getBasicAgentStatus();
            
            // Make data available to templates
            $templateMgr = TemplateManager::getManager();
            $templateMgr->assign('skzAgentData', $agentData);
            $templateMgr->assign('skzAgentApiUrl', $skzPlugin->getPluginSetting(0, 'agentBaseUrl'));
        }
    }
    
    /**
     * Get basic agent status information
     */
    private function getBasicAgentStatus() {
        return [
            'agents' => [
                [
                    'id' => 'research_discovery',
                    'name' => 'Research Discovery',
                    'status' => 'active',
                    'icon' => 'fas fa-search',
                    'color' => '#3b82f6'
                ],
                [
                    'id' => 'submission_assistant',
                    'name' => 'Submission Assistant',
                    'status' => 'active',
                    'icon' => 'fas fa-file-text',
                    'color' => '#8b5cf6'
                ],
                [
                    'id' => 'editorial_orchestration',
                    'name' => 'Editorial Orchestration',
                    'status' => 'active',
                    'icon' => 'fas fa-users',
                    'color' => '#f59e0b'
                ],
                [
                    'id' => 'review_coordination',
                    'name' => 'Review Coordination',
                    'status' => 'active',
                    'icon' => 'fas fa-check-circle',
                    'color' => '#10b981'
                ],
                [
                    'id' => 'content_quality',
                    'name' => 'Content Quality',
                    'status' => 'active',
                    'icon' => 'fas fa-shield-alt',
                    'color' => '#f97316'
                ],
                [
                    'id' => 'publishing_production',
                    'name' => 'Publishing Production',
                    'status' => 'active',
                    'icon' => 'fas fa-print',
                    'color' => '#ec4899'
                ],
                [
                    'id' => 'analytics_monitoring',
                    'name' => 'Analytics & Monitoring',
                    'status' => 'active',
                    'icon' => 'fas fa-chart-bar',
                    'color' => '#84cc16'
                ]
            ],
            'systemStatus' => 'operational',
            'lastUpdate' => date('Y-m-d H:i:s'),
            'totalActions' => 5719,
            'successRate' => 94.2
        ];
    }
    
    /**
     * Get agent status for specific submission
     */
    private function getSubmissionAgentStatus($submissionId) {
        // This would typically query the SKZ agent database
        // For now, return mock data
        return [
            'submissionId' => $submissionId,
            'currentStage' => 'editorial_review',
            'activeAgents' => ['editorial_orchestration', 'content_quality'],
            'completedActions' => 12,
            'pendingActions' => 3,
            'lastAgentActivity' => date('Y-m-d H:i:s', strtotime('-5 minutes')),
            'workflowProgress' => 65
        ];
    }
    
    /**
     * Get the display name of this plugin
     */
    function getDisplayName() {
        return __('plugins.themes.skzEnhanced.name');
    }
    
    /**
     * Get the description of this plugin
     */
    function getDescription() {
        return __('plugins.themes.skzEnhanced.description');
    }
}