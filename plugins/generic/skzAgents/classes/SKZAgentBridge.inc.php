<?php

/**
 * @file plugins/generic/skzAgents/classes/SKZAgentBridge.inc.php
 *
 * SKZ Agent Bridge - Handles communication between OJS and SKZ autonomous agents
 */

class SKZAgentBridge {
    
    /** @var string Base URL for agent framework API */
    private $agentBaseUrl;
    
    /** @var string API key for authentication */
    private $apiKey;
    
    /** @var int Request timeout in seconds */
    private $timeout;
    
    /**
     * Constructor
     */
    public function __construct() {
        $plugin = PluginRegistry::getPlugin('generic', 'skzAgents');
        $context = Application::getRequest()->getContext();
        $contextId = $context ? $context->getId() : 0;
        
        $this->agentBaseUrl = $plugin ? 
            $plugin->getSetting($contextId, 'agentBaseUrl') : 
            Config::getVar('skz', 'agent_base_url', 'http://localhost:5000/api');
            
        $this->apiKey = $plugin ? 
            $plugin->getSetting($contextId, 'apiKey') : 
            Config::getVar('skz', 'api_key', '');
            
        $this->timeout = $plugin ? 
            (int)$plugin->getSetting($contextId, 'timeout') : 
            (int)Config::getVar('skz', 'timeout', 30);
    }
    
    /**
     * Call an autonomous agent with specified action and data
     * 
     * @param string $agentName Name of the agent to call
     * @param string $action Action to perform
     * @param array $data Data to send to the agent
     * @return array Agent response
     */
    public function callAgent($agentName, $action, $data) {
        $url = $this->agentBaseUrl . '/' . $agentName . '/' . $action;
        
        $payload = array(
            'data' => $data,
            'timestamp' => time(),
            'ojs_version' => Application::getVersion(),
            'context' => $this->_getContextInfo()
        );
        
        $response = $this->_makeRequest($url, $payload);
        
        // Log agent communication
        $this->_logAgentCommunication($agentName, $action, $payload, $response);
        
        return $response;
    }
    
    /**
     * Get agent status for all agents
     * 
     * @return array Status information for all agents
     */
    public function getAgentStatus() {
        $url = $this->agentBaseUrl . '/status';
        return $this->_makeRequest($url, array(), 'GET');
    }
    
    /**
     * Get agent performance metrics
     * 
     * @param string $agentName Optional agent name for specific metrics
     * @return array Performance metrics
     */
    public function getAgentMetrics($agentName = null) {
        $url = $this->agentBaseUrl . '/metrics';
        if ($agentName) {
            $url .= '/' . $agentName;
        }
        return $this->_makeRequest($url, array(), 'GET');
    }
    
    /**
     * Register a webhook for agent notifications
     * 
     * @param string $event Event name to listen for
     * @param string $callbackUrl URL to call when event occurs
     * @return array Registration response
     */
    public function registerWebhook($event, $callbackUrl) {
        $url = $this->agentBaseUrl . '/webhooks/register';
        
        $payload = array(
            'event' => $event,
            'callback_url' => $callbackUrl,
            'ojs_instance' => $this->_getInstanceIdentifier()
        );
        
        return $this->_makeRequest($url, $payload);
    }
    
    /**
     * Send agent results back to OJS workflow
     * 
     * @param string $workflowType Type of workflow
     * @param int $submissionId Submission ID
     * @param array $results Agent processing results
     * @return bool Success status
     */
    public function sendResultsToWorkflow($workflowType, $submissionId, $results) {
        $submissionDao = Application::getSubmissionDAO();
        $submission = $submissionDao->getById($submissionId);
        
        if (!$submission) {
            return false;
        }
        
        // Store agent results using DAO
        $skzDAO = DAORegistry::getDAO('SKZDAO');
        if ($skzDAO) {
            $agentId = $workflowType . '_agent_' . $submissionId;
            $skzDAO->createAgentState($agentId, $results, $submissionId);
        }
        
        // Trigger workflow events based on results
        $this->_processWorkflowResults($workflowType, $submission, $results);
        
        return true;
    }
    
    /**
     * Make HTTP request to agent framework
     * 
     * @param string $url Request URL
     * @param array $data Request data
     * @param string $method HTTP method
     * @return array Response data
     */
    private function _makeRequest($url, $data = array(), $method = 'POST') {
        $curl = curl_init();
        
        $headers = array(
            'Content-Type: application/json',
            'X-API-Key: ' . $this->apiKey,
            'User-Agent: OJS-SKZ-Bridge/1.0'
        );
        
        curl_setopt_array($curl, array(
            CURLOPT_URL => $url,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => $this->timeout,
            CURLOPT_HTTPHEADER => $headers,
            CURLOPT_SSL_VERIFYPEER => false, // TODO: Enable in production
            CURLOPT_CUSTOMREQUEST => $method
        ));
        
        if ($method === 'POST' || $method === 'PUT') {
            curl_setopt($curl, CURLOPT_POSTFIELDS, json_encode($data));
        }
        
        $response = curl_exec($curl);
        $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
        $error = curl_error($curl);
        
        curl_close($curl);
        
        if ($error) {
            error_log("SKZ Agent Bridge Error: " . $error);
            return array('error' => $error);
        }
        
        if ($httpCode >= 400) {
            error_log("SKZ Agent Bridge HTTP Error: " . $httpCode);
            return array('error' => 'HTTP Error: ' . $httpCode);
        }
        
        $decodedResponse = json_decode($response, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            error_log("SKZ Agent Bridge JSON Error: " . json_last_error_msg());
            return array('error' => 'Invalid JSON response');
        }
        
        return $decodedResponse;
    }
    
    /**
     * Get current OJS context information
     * 
     * @return array Context information
     */
    private function _getContextInfo() {
        $request = Application::getRequest();
        $context = $request->getContext();
        
        return array(
            'context_id' => $context ? $context->getId() : null,
            'context_name' => $context ? $context->getLocalizedName() : null,
            'user_id' => $request->getUser() ? $request->getUser()->getId() : null,
            'session_id' => session_id()
        );
    }
    
    /**
     * Get unique instance identifier for this OJS installation
     * 
     * @return string Instance identifier
     */
    private function _getInstanceIdentifier() {
        return md5(Config::getVar('general', 'base_url') . Config::getVar('database', 'name'));
    }
    
    /**
     * Log agent communication for audit purposes
     * 
     * @param string $agentName Agent name
     * @param string $action Action performed
     * @param array $request Request data
     * @param array $response Response data
     */
    private function _logAgentCommunication($agentName, $action, $request, $response) {
        $logData = array(
            'agent_name' => $agentName,
            'action' => $action,
            'timestamp' => date('Y-m-d H:i:s'),
            'request_size' => strlen(json_encode($request)),
            'response_size' => strlen(json_encode($response)),
            'success' => !isset($response['error'])
        );
        
        // Store in database for analytics using DAO
        $skzDAO = DAORegistry::getDAO('SKZDAO');
        if ($skzDAO) {
            $skzDAO->logAgentCommunication(
                'ojs',
                $agentName,
                $action,
                array(
                    'request' => $request,
                    'response' => $response,
                    'metadata' => $logData
                )
            );
        }
        
        // Also log to error log if there was an error
        if (isset($response['error'])) {
            error_log("SKZ Agent Error: {$agentName}/{$action} - " . $response['error']);
        }
    }
    
    /**
     * Apply formatted content from production agent
     */
    private function _applyFormattedContent($submission, $content) {
        // Implementation for applying formatted content
    }
}

?>