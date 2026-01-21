<?php

/**
 * @file plugins/generic/skzAgents/controllers/SKZAgentManagementHandler.inc.php
 *
 * Agent Management AJAX Handler - Handles AJAX requests for agent management
 */

import('lib.pkp.classes.handler.Handler');

class SKZAgentManagementHandler extends Handler {
    
    /** @var SKZAgentsPlugin Plugin object */
    private $plugin;
    
    /**
     * Constructor
     */
    public function __construct() {
        parent::__construct();
        $this->plugin = PluginRegistry::getPlugin('generic', 'skzAgents');
        $this->addRoleAssignment(
            array(ROLE_ID_MANAGER, ROLE_ID_SUB_EDITOR, ROLE_ID_ASSISTANT),
            array('getStatus', 'startAgent', 'stopAgent', 'configure')
        );
    }
    
    /**
     * @copydoc PKPHandler::authorize()
     */
    public function authorize($request, &$args, $roleAssignments) {
        import('lib.pkp.classes.security.authorization.ContextAccessPolicy');
        $this->addPolicy(new ContextAccessPolicy($request, $roleAssignments));
        
        return parent::authorize($request, $args, $roleAssignments);
    }
    
    /**
     * Get status of all agents
     * @param array $args
     * @param PKPRequest $request
     */
    public function getStatus($args, $request) {
        $context = $request->getContext();
        
        if (!$this->plugin || !$this->plugin->getEnabled($context->getId())) {
            return $this->sendJsonResponse(array(
                'success' => false,
                'error' => __('plugins.generic.skzAgents.errors.pluginDisabled')
            ));
        }
        
        try {
            $agentStatus = $this->fetchAgentStatus();
            return $this->sendJsonResponse(array(
                'success' => true,
                'data' => $agentStatus
            ));
        } catch (Exception $e) {
            return $this->sendJsonResponse(array(
                'success' => false,
                'error' => $e->getMessage()
            ));
        }
    }
    
    /**
     * Start an agent
     * @param array $args
     * @param PKPRequest $request
     */
    public function startAgent($args, $request) {
        $context = $request->getContext();
        $agentId = $request->getUserVar('agentId');
        
        if (!$this->plugin || !$this->plugin->getEnabled($context->getId())) {
            return $this->sendJsonResponse(array(
                'success' => false,
                'error' => __('plugins.generic.skzAgents.errors.pluginDisabled')
            ));
        }
        
        if (empty($agentId)) {
            return $this->sendJsonResponse(array(
                'success' => false,
                'error' => 'Agent ID is required'
            ));
        }
        
        try {
            $result = $this->callAgentAPI('POST', "/agents/{$agentId}/start");
            return $this->sendJsonResponse(array(
                'success' => true,
                'data' => $result
            ));
        } catch (Exception $e) {
            return $this->sendJsonResponse(array(
                'success' => false,
                'error' => $e->getMessage()
            ));
        }
    }
    
    /**
     * Stop an agent
     * @param array $args
     * @param PKPRequest $request
     */
    public function stopAgent($args, $request) {
        $context = $request->getContext();
        $agentId = $request->getUserVar('agentId');
        
        if (!$this->plugin || !$this->plugin->getEnabled($context->getId())) {
            return $this->sendJsonResponse(array(
                'success' => false,
                'error' => __('plugins.generic.skzAgents.errors.pluginDisabled')
            ));
        }
        
        if (empty($agentId)) {
            return $this->sendJsonResponse(array(
                'success' => false,
                'error' => 'Agent ID is required'
            ));
        }
        
        try {
            $result = $this->callAgentAPI('POST', "/agents/{$agentId}/stop");
            return $this->sendJsonResponse(array(
                'success' => true,
                'data' => $result
            ));
        } catch (Exception $e) {
            return $this->sendJsonResponse(array(
                'success' => false,
                'error' => $e->getMessage()
            ));
        }
    }
    
    /**
     * Configure an agent (redirect to configuration interface)
     * @param array $args
     * @param PKPRequest $request
     */
    public function configure($args, $request) {
        $context = $request->getContext();
        $agentId = $request->getUserVar('agentId');
        
        if (!$this->plugin || !$this->plugin->getEnabled($context->getId())) {
            $templateMgr = TemplateManager::getManager($request);
            $templateMgr->assign('errorMessage', __('plugins.generic.skzAgents.errors.pluginDisabled'));
            return $templateMgr->display($this->plugin->getTemplateResource('error.tpl'));
        }
        
        $templateMgr = TemplateManager::getManager($request);
        $templateMgr->assign(array(
            'pageTitle' => __('plugins.generic.skzAgents.management.configure') . ' - ' . $agentId,
            'agentId' => $agentId,
            'backUrl' => $request->getRequestedUrl()
        ));
        
        return $templateMgr->display($this->plugin->getTemplateResource('agentConfig.tpl'));
    }
    
    /**
     * Fetch current status of all agents from the API
     * @return array Agent status data
     */
    private function fetchAgentStatus() {
        $agents = [
            'research_discovery' => 'Research Discovery Agent',
            'submission_assistant' => 'Submission Assistant Agent',
            'editorial_orchestration' => 'Editorial Orchestration Agent',
            'peer_review' => 'Peer Review Agent',
            'quality_assurance' => 'Quality Assurance Agent',
            'publication_formatting' => 'Publication Formatting Agent',
            'workflow_orchestration' => 'Workflow Orchestration Agent',
        ];
        
        $status = array();
        foreach ($agents as $agentId => $agentName) {
            try {
                $agentStatus = $this->callAgentAPI('GET', "/agents/{$agentId}/status");
                $status[$agentId] = array(
                    'name' => $agentName,
                    'status' => $agentStatus['status'] ?? 'unknown',
                    'lastActivity' => $agentStatus['lastActivity'] ?? null,
                    'processedTasks' => $agentStatus['processedTasks'] ?? 0,
                    'configuration' => $agentStatus['configuration'] ?? array()
                );
            } catch (Exception $e) {
                // If agent is not reachable, mark as stopped
                $status[$agentId] = array(
                    'name' => $agentName,
                    'status' => 'stopped',
                    'lastActivity' => null,
                    'processedTasks' => 0,
                    'configuration' => array()
                );
            }
        }
        
        return $status;
    }
    
    /**
     * Make API call to agent framework
     * @param string $method HTTP method (GET, POST, etc.)
     * @param string $endpoint API endpoint
     * @param array $data Optional data to send
     * @return array Response data
     * @throws Exception If API call fails
     */
    private function callAgentAPI($method, $endpoint, $data = null) {
        // Get API configuration from plugin settings
        $context = Application::get()->getRequest()->getContext();
        $baseUrl = $this->plugin->getSetting($context->getId(), 'agentBaseUrl') ?: 'http://localhost:5000';
        $apiKey = $this->plugin->getSetting($context->getId(), 'apiKey') ?: '';
        
        $url = rtrim($baseUrl, '/') . '/api/v1' . $endpoint;
        
        // Initialize cURL
        $ch = curl_init();
        curl_setopt_array($ch, array(
            CURLOPT_URL => $url,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => 30,
            CURLOPT_CUSTOMREQUEST => $method,
            CURLOPT_HTTPHEADER => array(
                'Content-Type: application/json',
                'Authorization: Bearer ' . $apiKey
            )
        ));
        
        if ($data && in_array($method, array('POST', 'PUT', 'PATCH'))) {
            curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
        }
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);
        curl_close($ch);
        
        if ($error) {
            throw new Exception("API call failed: " . $error);
        }
        
        if ($httpCode >= 400) {
            throw new Exception("API call failed with HTTP {$httpCode}");
        }
        
        $result = json_decode($response, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new Exception("Invalid JSON response from API");
        }
        
        return $result;
    }
    
    /**
     * Send JSON response
     * @param array $data Response data
     */
    private function sendJsonResponse($data) {
        header('Content-Type: application/json');
        echo json_encode($data);
        exit;
    }
}

?>