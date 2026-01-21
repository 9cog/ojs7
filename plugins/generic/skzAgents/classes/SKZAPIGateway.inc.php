<?php

/**
 * @file plugins/generic/skzAgents/classes/SKZAPIGateway.inc.php
 *
 * SKZ API Gateway - Central routing and security layer for agent communications
 */

class SKZAPIGateway {
    
    /** @var array Agent endpoint mappings */
    private $agentEndpoints = array(
        'research-discovery' => array(
            'actions' => array('literature_search', 'gap_analysis', 'trend_identification', 'inci_analysis'),
            'path' => '/agents/research-discovery'
        ),
        'submission-assistant' => array(
            'actions' => array('format_check', 'venue_recommendation', 'compliance_validation', 'quality_assessment'),
            'path' => '/agents/submission-assistant'
        ),
        'editorial-orchestration' => array(
            'actions' => array('workflow_management', 'decision_support', 'deadline_tracking', 'conflict_resolution'),
            'path' => '/agents/editorial-orchestration'
        ),
        'review-coordination' => array(
            'actions' => array('reviewer_matching', 'review_tracking', 'quality_assessment', 'workload_management'),
            'path' => '/agents/review-coordination'
        ),
        'content-quality' => array(
            'actions' => array('scientific_validation', 'safety_assessment', 'standards_enforcement', 'regulatory_compliance'),
            'path' => '/agents/content-quality'
        ),
        'publishing-production' => array(
            'actions' => array('content_formatting', 'visual_generation', 'distribution_preparation', 'regulatory_reporting'),
            'path' => '/agents/publishing-production'
        ),
        'analytics-monitoring' => array(
            'actions' => array('performance_analytics', 'trend_forecasting', 'strategic_insights', 'continuous_learning'),
            'path' => '/agents/analytics-monitoring'
        )
    );
    
    /** @var SKZAgentBridge */
    private $bridge;
    
    /** @var array Configuration settings */
    private $config;
    
    /** @var array Rate limiting storage */
    private static $rateLimitCache = array();
    
    /**
     * Constructor
     */
    public function __construct() {
        import('plugins.generic.skzAgents.classes.SKZAgentBridge');
        $this->bridge = new SKZAgentBridge();
        $this->loadConfiguration();
    }
    
    /**
     * Route agent request through gateway
     * 
     * @param string $agentName Name of the target agent
     * @param string $action Action to perform
     * @param array $data Request data
     * @param array $options Request options (auth, timeout, etc.)
     * @return array Gateway response
     */
    public function routeAgentRequest($agentName, $action, $data, $options = array()) {
        // Start request timing
        $startTime = microtime(true);
        
        try {
            // Validate agent and action
            if (!$this->validateAgentAction($agentName, $action)) {
                return $this->createErrorResponse('Invalid agent or action', 400);
            }
            
            // Check rate limiting
            if (!$this->checkRateLimit($agentName, $action)) {
                return $this->createErrorResponse('Rate limit exceeded', 429);
            }
            
            // Authenticate request
            if (!$this->authenticateRequest($options)) {
                return $this->createErrorResponse('Authentication failed', 401);
            }
            
            // Transform request data
            $transformedData = $this->transformRequest($agentName, $action, $data);
            
            // Call agent through bridge
            $response = $this->bridge->callAgent($agentName, $action, $transformedData);
            
            // Transform response data
            $transformedResponse = $this->transformResponse($agentName, $action, $response);
            
            // Log successful request
            $this->logGatewayRequest($agentName, $action, $transformedData, $transformedResponse, microtime(true) - $startTime);
            
            return $transformedResponse;
            
        } catch (Exception $e) {
            // Log error
            $this->logGatewayError($agentName, $action, $e->getMessage(), microtime(true) - $startTime);
            
            return $this->createErrorResponse('Gateway error: ' . $e->getMessage(), 500);
        }
    }
    
    /**
     * Get agent status through gateway
     * 
     * @param string $agentName Optional specific agent name
     * @return array Agent status information
     */
    public function getAgentStatus($agentName = null) {
        try {
            if ($agentName && !$this->validateAgent($agentName)) {
                return $this->createErrorResponse('Invalid agent name', 400);
            }
            
            $status = $this->bridge->getAgentStatus();
            
            // Filter by agent if specified
            if ($agentName && isset($status['agents'])) {
                $status['agents'] = array_filter($status['agents'], function($agent) use ($agentName) {
                    return $agent['name'] === $agentName;
                });
            }
            
            return $this->createSuccessResponse($status);
            
        } catch (Exception $e) {
            return $this->createErrorResponse('Status check failed: ' . $e->getMessage(), 500);
        }
    }
    
    /**
     * Register webhook through gateway
     * 
     * @param string $event Event name
     * @param string $callbackUrl Callback URL
     * @param array $options Registration options
     * @return array Registration response
     */
    public function registerWebhook($event, $callbackUrl, $options = array()) {
        try {
            // Validate webhook configuration
            if (!$this->validateWebhookConfig($event, $callbackUrl)) {
                return $this->createErrorResponse('Invalid webhook configuration', 400);
            }
            
            $response = $this->bridge->registerWebhook($event, $callbackUrl);
            
            return $this->createSuccessResponse($response);
            
        } catch (Exception $e) {
            return $this->createErrorResponse('Webhook registration failed: ' . $e->getMessage(), 500);
        }
    }
    
    /**
     * Validate agent and action combination
     * 
     * @param string $agentName Agent name
     * @param string $action Action name
     * @return bool Validation result
     */
    private function validateAgentAction($agentName, $action) {
        if (!isset($this->agentEndpoints[$agentName])) {
            return false;
        }
        
        if (!in_array($action, $this->agentEndpoints[$agentName]['actions'])) {
            return false;
        }
        
        return true;
    }
    
    /**
     * Validate agent name
     * 
     * @param string $agentName Agent name
     * @return bool Validation result
     */
    private function validateAgent($agentName) {
        return isset($this->agentEndpoints[$agentName]);
    }
    
    /**
     * Check rate limiting
     * 
     * @param string $agentName Agent name
     * @param string $action Action name
     * @return bool Rate limit check result
     */
    private function checkRateLimit($agentName, $action) {
        if (!$this->config['rate_limit_enabled']) {
            return true;
        }
        
        $key = $agentName . ':' . $action;
        $now = time();
        $window = 60; // 1 minute window
        $limit = $this->config['rate_limit_requests_per_minute'];
        
        // Clean old entries
        if (isset(self::$rateLimitCache[$key])) {
            self::$rateLimitCache[$key] = array_filter(self::$rateLimitCache[$key], function($timestamp) use ($now, $window) {
                return ($now - $timestamp) < $window;
            });
        } else {
            self::$rateLimitCache[$key] = array();
        }
        
        // Check limit
        if (count(self::$rateLimitCache[$key]) >= $limit) {
            return false;
        }
        
        // Add current request
        self::$rateLimitCache[$key][] = $now;
        
        return true;
    }
    
    /**
     * Authenticate request
     * 
     * @param array $options Request options
     * @return bool Authentication result
     */
    private function authenticateRequest($options) {
        // For now, rely on the bridge's built-in authentication
        // In future versions, could add additional layers like JWT, OAuth, etc.
        return true;
    }
    
    /**
     * Transform request data for agent
     * 
     * @param string $agentName Agent name
     * @param string $action Action name
     * @param array $data Original data
     * @return array Transformed data
     */
    private function transformRequest($agentName, $action, $data) {
        // Add gateway metadata
        $transformed = array(
            'payload' => $data,
            'gateway_metadata' => array(
                'ojs_version' => Application::getVersion(),
                'plugin_version' => $this->getPluginVersion(),
                'request_id' => uniqid('skz_', true),
                'timestamp' => time(),
                'source_agent' => $agentName,
                'action' => $action
            )
        );
        
        // Add context information
        $request = Application::getRequest();
        $context = $request->getContext();
        if ($context) {
            $transformed['gateway_metadata']['context'] = array(
                'id' => $context->getId(),
                'name' => $context->getLocalizedName(),
                'url' => $context->getPath()
            );
        }
        
        return $transformed;
    }
    
    /**
     * Transform response data from agent
     * 
     * @param string $agentName Agent name
     * @param string $action Action name
     * @param array $response Original response
     * @return array Transformed response
     */
    private function transformResponse($agentName, $action, $response) {
        // Add gateway response metadata
        return array(
            'status' => 'success',
            'agent' => $agentName,
            'action' => $action,
            'data' => $response,
            'gateway_info' => array(
                'processed_at' => time(),
                'response_id' => uniqid('skz_resp_', true)
            )
        );
    }
    
    /**
     * Create error response
     * 
     * @param string $message Error message
     * @param int $code Error code
     * @return array Error response
     */
    private function createErrorResponse($message, $code) {
        return array(
            'status' => 'error',
            'error' => array(
                'message' => $message,
                'code' => $code,
                'timestamp' => time()
            )
        );
    }
    
    /**
     * Create success response
     * 
     * @param array $data Response data
     * @return array Success response
     */
    private function createSuccessResponse($data) {
        return array(
            'status' => 'success',
            'data' => $data,
            'timestamp' => time()
        );
    }
    
    /**
     * Validate webhook configuration
     * 
     * @param string $event Event name
     * @param string $callbackUrl Callback URL
     * @return bool Validation result
     */
    private function validateWebhookConfig($event, $callbackUrl) {
        // Validate URL
        if (!filter_var($callbackUrl, FILTER_VALIDATE_URL)) {
            return false;
        }
        
        // Validate event name
        $validEvents = array(
            'agent.status.changed',
            'agent.task.completed',
            'agent.task.failed',
            'agent.communication.received',
            'workflow.status.updated'
        );
        
        return in_array($event, $validEvents);
    }
    
    /**
     * Load gateway configuration
     */
    private function loadConfiguration() {
        $plugin = PluginRegistry::getPlugin('generic', 'skzAgents');
        $context = Application::getRequest()->getContext();
        $contextId = $context ? $context->getId() : 0;
        
        $this->config = array(
            'rate_limit_enabled' => $plugin ? 
                (bool)$plugin->getSetting($contextId, 'rateLimitEnabled') : 
                (bool)Config::getVar('skz', 'rate_limit_enabled', true),
            
            'rate_limit_requests_per_minute' => $plugin ? 
                (int)$plugin->getSetting($contextId, 'rateLimitRequestsPerMinute') : 
                (int)Config::getVar('skz', 'rate_limit_requests_per_minute', 100),
            
            'webhook_enabled' => $plugin ? 
                (bool)$plugin->getSetting($contextId, 'webhookEnabled') : 
                (bool)Config::getVar('skz', 'webhook_enabled', true),
            
            'performance_monitoring' => $plugin ? 
                (bool)$plugin->getSetting($contextId, 'performanceMonitoring') : 
                (bool)Config::getVar('skz', 'performance_monitoring', true)
        );
    }
    
    /**
     * Log gateway request
     * 
     * @param string $agentName Agent name
     * @param string $action Action name
     * @param array $request Request data
     * @param array $response Response data
     * @param float $duration Request duration
     */
    private function logGatewayRequest($agentName, $action, $request, $response, $duration) {
        if (!$this->config['performance_monitoring']) {
            return;
        }
        
        $logData = array(
            'type' => 'gateway_request',
            'agent' => $agentName,
            'action' => $action,
            'duration' => $duration,
            'success' => true,
            'timestamp' => time()
        );
        
        error_log('SKZ Gateway: ' . json_encode($logData));
    }
    
    /**
     * Log gateway error
     * 
     * @param string $agentName Agent name
     * @param string $action Action name
     * @param string $error Error message
     * @param float $duration Request duration
     */
    private function logGatewayError($agentName, $action, $error, $duration) {
        $logData = array(
            'type' => 'gateway_error',
            'agent' => $agentName,
            'action' => $action,
            'error' => $error,
            'duration' => $duration,
            'success' => false,
            'timestamp' => time()
        );
        
        error_log('SKZ Gateway Error: ' . json_encode($logData));
    }
    
    /**
     * Get plugin version
     * 
     * @return string Plugin version
     */
    private function getPluginVersion() {
        $plugin = PluginRegistry::getPlugin('generic', 'skzAgents');
        return $plugin ? $plugin->getCurrentVersion()->getVersionString() : '1.0.0';
    }
}