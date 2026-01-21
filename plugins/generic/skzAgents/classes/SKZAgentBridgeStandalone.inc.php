<?php

/**
 * @file plugins/generic/skzAgents/classes/SKZAgentBridgeStandalone.inc.php
 *
 * Standalone SKZ Agent Bridge - Independent communication with Python agents
 * Can be used for testing and minimal OJS integration
 */

class SKZAgentBridgeStandalone {
    
    /** @var string Base URL for agent framework API */
    private $agentBaseUrl;
    
    /** @var string API key for authentication */
    private $apiKey;
    
    /** @var int Request timeout in seconds */
    private $timeout;
    
    /** @var array Request history for debugging */
    private $requestHistory = array();
    
    /**
     * Constructor
     * @param string $agentBaseUrl Base URL for agent API
     * @param string $apiKey API key for authentication
     * @param int $timeout Request timeout in seconds
     */
    public function __construct($agentBaseUrl = null, $apiKey = null, $timeout = 30) {
        $this->agentBaseUrl = $agentBaseUrl ?: 'http://localhost:5000';
        $this->apiKey = $apiKey ?: 'test_key';
        $this->timeout = $timeout;
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
            'php_version' => PHP_VERSION,
            'context' => $this->_getContextInfo()
        );
        
        $response = $this->_makeRequest($url, $payload);
        
        // Log communication
        $this->_logCommunication($agentName, $action, $payload, $response);
        
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
     * Get list of available agents
     * 
     * @return array List of agents
     */
    public function getAgentsList() {
        $url = $this->agentBaseUrl . '/agents';
        return $this->_makeRequest($url, array(), 'GET');
    }
    
    /**
     * Get specific agent status
     * 
     * @param string $agentName Name of the agent
     * @return array Agent status
     */
    public function getSpecificAgentStatus($agentName) {
        $url = $this->agentBaseUrl . '/agents/' . $agentName;
        return $this->_makeRequest($url, array(), 'GET');
    }
    
    /**
     * Test connection to agent framework
     * 
     * @return array Connection test result
     */
    public function testConnection() {
        try {
            $response = $this->getAgentStatus();
            
            if (isset($response['status'])) {
                return array(
                    'success' => true,
                    'message' => 'Connection successful',
                    'agent_status' => $response
                );
            } else {
                return array(
                    'success' => false,
                    'message' => 'Invalid response from agent framework',
                    'response' => $response
                );
            }
        } catch (Exception $e) {
            return array(
                'success' => false,
                'message' => 'Connection failed: ' . $e->getMessage()
            );
        }
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
            'User-Agent: PHP-SKZ-Bridge/1.0'
        );
        
        curl_setopt_array($curl, array(
            CURLOPT_URL => $url,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => $this->timeout,
            CURLOPT_HTTPHEADER => $headers,
            CURLOPT_SSL_VERIFYPEER => false,
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
            return array('error' => 'cURL Error: ' . $error);
        }
        
        if ($httpCode >= 400) {
            return array('error' => 'HTTP Error: ' . $httpCode, 'response' => $response);
        }
        
        $decodedResponse = json_decode($response, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            return array('error' => 'Invalid JSON response', 'raw_response' => $response);
        }
        
        return $decodedResponse;
    }
    
    /**
     * Get basic context information
     * 
     * @return array Context information
     */
    private function _getContextInfo() {
        return array(
            'php_version' => PHP_VERSION,
            'timestamp' => date('Y-m-d H:i:s'),
            'server_name' => isset($_SERVER['SERVER_NAME']) ? $_SERVER['SERVER_NAME'] : 'cli',
            'user_agent' => isset($_SERVER['HTTP_USER_AGENT']) ? $_SERVER['HTTP_USER_AGENT'] : 'php-cli'
        );
    }
    
    /**
     * Log communication for debugging
     * 
     * @param string $agentName Agent name
     * @param string $action Action performed
     * @param array $request Request data
     * @param array $response Response data
     */
    private function _logCommunication($agentName, $action, $request, $response) {
        $logEntry = array(
            'timestamp' => date('Y-m-d H:i:s'),
            'agent_name' => $agentName,
            'action' => $action,
            'request_size' => strlen(json_encode($request)),
            'response_size' => strlen(json_encode($response)),
            'success' => !isset($response['error'])
        );
        
        $this->requestHistory[] = $logEntry;
        
        // Keep only the last 50 entries
        if (count($this->requestHistory) > 50) {
            $this->requestHistory = array_slice($this->requestHistory, -25);
        }
    }
    
    /**
     * Get request history for debugging
     * 
     * @param int $limit Number of entries to return
     * @return array Request history
     */
    public function getRequestHistory($limit = 10) {
        return array_slice($this->requestHistory, -$limit);
    }
    
    /**
     * Get connection statistics
     * 
     * @return array Connection statistics
     */
    public function getConnectionStats() {
        $totalRequests = count($this->requestHistory);
        $successfulRequests = 0;
        
        foreach ($this->requestHistory as $entry) {
            if ($entry['success']) {
                $successfulRequests++;
            }
        }
        
        return array(
            'total_requests' => $totalRequests,
            'successful_requests' => $successfulRequests,
            'success_rate' => $totalRequests > 0 ? ($successfulRequests / $totalRequests) : 0,
            'agent_base_url' => $this->agentBaseUrl,
            'last_request' => $totalRequests > 0 ? end($this->requestHistory)['timestamp'] : null
        );
    }
    
    /**
     * Set configuration
     * 
     * @param string $key Configuration key
     * @param mixed $value Configuration value
     */
    public function setConfig($key, $value) {
        switch ($key) {
            case 'agentBaseUrl':
                $this->agentBaseUrl = $value;
                break;
            case 'apiKey':
                $this->apiKey = $value;
                break;
            case 'timeout':
                $this->timeout = (int)$value;
                break;
        }
    }
    
    /**
     * Get configuration
     * 
     * @param string $key Configuration key
     * @return mixed Configuration value
     */
    public function getConfig($key) {
        switch ($key) {
            case 'agentBaseUrl':
                return $this->agentBaseUrl;
            case 'apiKey':
                return $this->apiKey;
            case 'timeout':
                return $this->timeout;
            default:
                return null;
        }
    }
}

?>