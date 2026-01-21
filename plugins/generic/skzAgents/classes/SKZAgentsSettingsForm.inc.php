<?php

/**
 * @file plugins/generic/skzAgents/classes/SKZAgentsSettingsForm.inc.php
 *
 * SKZ Agents Settings Form - Provides configuration interface for the plugin
 */

import('lib.pkp.classes.form.Form');

class SKZAgentsSettingsForm extends Form {
    
    /** @var SKZAgentsPlugin The plugin associated with this form */
    private $plugin;
    
    /** @var int Context ID */
    private $contextId;
    
    /**
     * Constructor
     * @param SKZAgentsPlugin $plugin Plugin object
     * @param int $contextId Context ID
     */
    public function __construct($plugin, $contextId) {
        $this->plugin = $plugin;
        $this->contextId = $contextId;
        
        parent::__construct($plugin->getTemplateResource('settings.tpl'));
        
        // Add form validation
        $this->addCheck(new FormValidator($this, 'agentBaseUrl', 'required', 'plugins.generic.skzAgents.settings.agentBaseUrl.required'));
        $this->addCheck(new FormValidatorUrl($this, 'agentBaseUrl', 'required', 'plugins.generic.skzAgents.settings.agentBaseUrl.invalid'));
        $this->addCheck(new FormValidator($this, 'apiKey', 'required', 'plugins.generic.skzAgents.settings.apiKey.required'));
        $this->addCheck(new FormValidatorPost($this));
        $this->addCheck(new FormValidatorCSRF($this));
    }
    
    /**
     * Initialize form data from plugin settings
     */
    public function initData() {
        $this->setData('agentBaseUrl', $this->plugin->getSetting($this->contextId, 'agentBaseUrl'));
        $this->setData('apiKey', $this->plugin->getSetting($this->contextId, 'apiKey'));
        $this->setData('timeout', $this->plugin->getSetting($this->contextId, 'timeout'));
        $this->setData('enableAutoSubmissionProcessing', $this->plugin->getSetting($this->contextId, 'enableAutoSubmissionProcessing'));
        $this->setData('enableAutoReviewerAssignment', $this->plugin->getSetting($this->contextId, 'enableAutoReviewerAssignment'));
        $this->setData('enableAutoQualityChecks', $this->plugin->getSetting($this->contextId, 'enableAutoQualityChecks'));
        $this->setData('enablePerformanceMonitoring', $this->plugin->getSetting($this->contextId, 'enablePerformanceMonitoring'));
        $this->setData('maxConcurrentRequests', $this->plugin->getSetting($this->contextId, 'maxConcurrentRequests'));
        $this->setData('cacheTtl', $this->plugin->getSetting($this->contextId, 'cacheTtl'));
    }
    
    /**
     * Assign form data to user-submitted data
     */
    public function readInputData() {
        $this->readUserVars(array(
            'agentBaseUrl',
            'apiKey', 
            'timeout',
            'enableAutoSubmissionProcessing',
            'enableAutoReviewerAssignment', 
            'enableAutoQualityChecks',
            'enablePerformanceMonitoring',
            'maxConcurrentRequests',
            'cacheTtl'
        ));
    }
    
    /**
     * Fetch the form
     * @param PKPRequest $request
     * @param string $template
     * @param bool $display
     * @return string|null
     */
    public function fetch($request, $template = null, $display = false) {
        $templateMgr = TemplateManager::getManager($request);
        $templateMgr->assign('pluginName', $this->plugin->getName());
        $templateMgr->assign('agentStatus', $this->_getAgentStatus());
        $templateMgr->assign('availableAgents', $this->_getAvailableAgents());
        
        return parent::fetch($request, $template, $display);
    }
    
    /**
     * Save plugin settings
     */
    public function execute(...$functionArgs) {
        $this->plugin->updateSetting($this->contextId, 'agentBaseUrl', trim($this->getData('agentBaseUrl'), "\"\';"), 'string');
        $this->plugin->updateSetting($this->contextId, 'apiKey', trim($this->getData('apiKey'), "\"\';"), 'string');
        $this->plugin->updateSetting($this->contextId, 'timeout', (int)$this->getData('timeout'), 'int');
        $this->plugin->updateSetting($this->contextId, 'enableAutoSubmissionProcessing', $this->getData('enableAutoSubmissionProcessing'), 'bool');
        $this->plugin->updateSetting($this->contextId, 'enableAutoReviewerAssignment', $this->getData('enableAutoReviewerAssignment'), 'bool');
        $this->plugin->updateSetting($this->contextId, 'enableAutoQualityChecks', $this->getData('enableAutoQualityChecks'), 'bool');
        $this->plugin->updateSetting($this->contextId, 'enablePerformanceMonitoring', $this->getData('enablePerformanceMonitoring'), 'bool');
        $this->plugin->updateSetting($this->contextId, 'maxConcurrentRequests', (int)$this->getData('maxConcurrentRequests'), 'int');
        $this->plugin->updateSetting($this->contextId, 'cacheTtl', (int)$this->getData('cacheTtl'), 'int');
        
        // Test connection to agent framework
        $this->_testAgentConnection();
        
        parent::execute(...$functionArgs);
    }
    
    /**
     * Get current agent status for display
     * @return array Agent status information
     */
    private function _getAgentStatus() {
        if (!$this->plugin->getEnabled($this->contextId)) {
            return array('status' => 'disabled');
        }
        
        try {
            import('plugins.generic.skzAgents.classes.SKZAgentBridge');
            $bridge = new SKZAgentBridge();
            return $bridge->getAgentStatus();
        } catch (Exception $e) {
            return array('status' => 'error', 'message' => $e->getMessage());
        }
    }
    
    /**
     * Get list of available agents
     * @return array Available agents information
     */
    private function _getAvailableAgents() {
        return array(
            'research-discovery' => array(
                'name' => __('plugins.generic.skzAgents.agents.researchDiscovery'),
                'description' => 'INCI database mining, patent analysis, trend identification',
                'endpoint' => '/api/research-discovery'
            ),
            'submission-assistant' => array(
                'name' => __('plugins.generic.skzAgents.agents.submissionAssistant'),
                'description' => 'Quality assessment, safety compliance, statistical review',
                'endpoint' => '/api/submission-assistant'
            ),
            'editorial-orchestration' => array(
                'name' => __('plugins.generic.skzAgents.agents.editorialOrchestration'),
                'description' => 'Workflow coordination, decision making, conflict resolution',
                'endpoint' => '/api/editorial-orchestration'
            ),
            'review-coordination' => array(
                'name' => __('plugins.generic.skzAgents.agents.reviewCoordination'),
                'description' => 'Reviewer matching, workload management, quality monitoring',
                'endpoint' => '/api/review-coordination'
            ),
            'content-quality' => array(
                'name' => __('plugins.generic.skzAgents.agents.contentQuality'),
                'description' => 'Scientific validation, safety assessment, standards enforcement',
                'endpoint' => '/api/content-quality'
            ),
            'publishing-production' => array(
                'name' => __('plugins.generic.skzAgents.agents.publishingProduction'),
                'description' => 'Content formatting, visual generation, multi-channel distribution',
                'endpoint' => '/api/publishing-production'
            ),
            'analytics-monitoring' => array(
                'name' => __('plugins.generic.skzAgents.agents.analyticsMonitoring'),
                'description' => 'Performance analytics, trend forecasting, strategic insights',
                'endpoint' => '/api/analytics-monitoring'
            )
        );
    }
    
    /**
     * Test connection to agent framework
     * @return bool Connection success
     */
    private function _testAgentConnection() {
        try {
            import('plugins.generic.skzAgents.classes.SKZAgentBridge');
            $bridge = new SKZAgentBridge();
            $status = $bridge->getAgentStatus();
            
            if (isset($status['error'])) {
                import('classes.notification.NotificationManager');
                $notificationMgr = new NotificationManager();
                $notificationMgr->createTrivialNotification(
                    Application::getRequest()->getUser()->getId(),
                    NOTIFICATION_TYPE_WARNING,
                    array('contents' => __('plugins.generic.skzAgents.errors.connectionFailed') . ': ' . $status['error'])
                );
                return false;
            }
            
            return true;
        } catch (Exception $e) {
            import('classes.notification.NotificationManager');
            $notificationMgr = new NotificationManager();
            $notificationMgr->createTrivialNotification(
                Application::getRequest()->getUser()->getId(),
                NOTIFICATION_TYPE_WARNING,
                array('contents' => __('plugins.generic.skzAgents.errors.connectionFailed') . ': ' . $e->getMessage())
            );
            return false;
        }
    }
}

?>