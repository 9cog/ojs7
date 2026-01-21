<?php

/**
 * @file plugins/generic/skzAgents/pages/SKZAgentsHandler.inc.php
 *
 * SKZ Agents Handler - Handles agent management pages and AJAX requests
 */

import('classes.handler.Handler');

class SKZAgentsHandler extends Handler {
    
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
            array('dashboard', 'status', 'metrics', 'communications', 'settings', 'api', 'webhook')
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
     * Display the agent dashboard
     * @param array $args
     * @param PKPRequest $request
     */
    public function dashboard($args, $request) {
        $templateMgr = TemplateManager::getManager($request);
        $context = $request->getContext();
        
        if (!$this->plugin->getEnabled($context->getId())) {
            $templateMgr->assign('errorMessage', __('plugins.generic.skzAgents.errors.pluginDisabled'));
            return $templateMgr->display($this->plugin->getTemplateResource('error.tpl'));
        }
        
        // Get agent status
        import('plugins.generic.skzAgents.classes.SKZAgentBridge');
        $bridge = new SKZAgentBridge();
        $agentStatus = $bridge->getAgentStatus();
        
        // Get recent communications
        import('plugins.generic.skzAgents.classes.SKZDAO');
        $skzDAO = new SKZDAO();
        $recentCommunications = $skzDAO->getAgentCommunications(null, 10);
        
        // Get performance metrics
        $performanceMetrics = $skzDAO->getAgentPerformanceMetrics();
        
        $templateMgr->assign(array(
            'pageTitle' => __('plugins.generic.skzAgents.dashboard.title'),
            'agentStatus' => $agentStatus,
            'recentCommunications' => $recentCommunications,
            'performanceMetrics' => $performanceMetrics,
            'pluginName' => $this->plugin->getName()
        ));
        
        return $templateMgr->display($this->plugin->getTemplateResource('dashboard.tpl'));
    }
    
    /**
     * Get agent status (AJAX endpoint)
     * @param array $args
     * @param PKPRequest $request
     */
    public function status($args, $request) {
        if (!$request->isPost()) {
            header('HTTP/1.0 405 Method Not Allowed');
            return;
        }
        
        $context = $request->getContext();
        if (!$this->plugin->getEnabled($context->getId())) {
            header('HTTP/1.0 403 Forbidden');
            echo json_encode(array('error' => 'Plugin disabled'));
            return;
        }
        
        try {
            import('plugins.generic.skzAgents.classes.SKZAgentBridge');
            $bridge = new SKZAgentBridge();
            $status = $bridge->getAgentStatus();
            
            header('Content-Type: application/json');
            echo json_encode($status);
        } catch (Exception $e) {
            header('HTTP/1.0 500 Internal Server Error');
            echo json_encode(array('error' => $e->getMessage()));
        }
    }
    
    /**
     * Get agent performance metrics (AJAX endpoint)
     * @param array $args
     * @param PKPRequest $request
     */
    public function metrics($args, $request) {
        if (!$request->isPost()) {
            header('HTTP/1.0 405 Method Not Allowed');
            return;
        }
        
        $context = $request->getContext();
        if (!$this->plugin->getEnabled($context->getId())) {
            header('HTTP/1.0 403 Forbidden');
            echo json_encode(array('error' => 'Plugin disabled'));
            return;
        }
        
        $agentName = $request->getUserVar('agentName');
        $startDate = $request->getUserVar('startDate');
        $endDate = $request->getUserVar('endDate');
        
        try {
            import('plugins.generic.skzAgents.classes.SKZDAO');
            $skzDAO = new SKZDAO();
            $metrics = $skzDAO->getAgentPerformanceMetrics($agentName, $startDate, $endDate);
            
            header('Content-Type: application/json');
            echo json_encode(array(
                'success' => true,
                'metrics' => $metrics
            ));
        } catch (Exception $e) {
            header('HTTP/1.0 500 Internal Server Error');
            echo json_encode(array('error' => $e->getMessage()));
        }
    }
    
    /**
     * Get agent communications log (AJAX endpoint)
     * @param array $args
     * @param PKPRequest $request
     */
    public function communications($args, $request) {
        if (!$request->isPost()) {
            header('HTTP/1.0 405 Method Not Allowed');
            return;
        }
        
        $context = $request->getContext();
        if (!$this->plugin->getEnabled($context->getId())) {
            header('HTTP/1.0 403 Forbidden');
            echo json_encode(array('error' => 'Plugin disabled'));
            return;
        }
        
        $agentName = $request->getUserVar('agentName');
        $limit = (int)$request->getUserVar('limit') ?: 50;
        $offset = (int)$request->getUserVar('offset') ?: 0;
        
        try {
            import('plugins.generic.skzAgents.classes.SKZDAO');
            $skzDAO = new SKZDAO();
            $communications = $skzDAO->getAgentCommunications($agentName, $limit, $offset);
            
            header('Content-Type: application/json');
            echo json_encode(array(
                'success' => true,
                'communications' => $communications,
                'limit' => $limit,
                'offset' => $offset
            ));
        } catch (Exception $e) {
            header('HTTP/1.0 500 Internal Server Error');
            echo json_encode(array('error' => $e->getMessage()));
        }
    }
    
    /**
     * Test agent connection (AJAX endpoint)
     * @param array $args
     * @param PKPRequest $request
     */
    public function testConnection($args, $request) {
        if (!$request->isPost()) {
            header('HTTP/1.0 405 Method Not Allowed');
            return;
        }
        
        $context = $request->getContext();
        if (!$this->plugin->getEnabled($context->getId())) {
            header('HTTP/1.0 403 Forbidden');
            echo json_encode(array('error' => 'Plugin disabled'));
            return;
        }
        
        $agentBaseUrl = $request->getUserVar('agentBaseUrl');
        $apiKey = $request->getUserVar('apiKey');
        
        if (!$agentBaseUrl || !$apiKey) {
            header('HTTP/1.0 400 Bad Request');
            echo json_encode(array('error' => 'Missing required parameters'));
            return;
        }
        
        try {
            // Test connection with provided parameters
            $curl = curl_init();
            curl_setopt_array($curl, array(
                CURLOPT_URL => rtrim($agentBaseUrl, '/') . '/status',
                CURLOPT_RETURNTRANSFER => true,
                CURLOPT_TIMEOUT => 10,
                CURLOPT_HTTPHEADER => array(
                    'Content-Type: application/json',
                    'X-API-Key: ' . $apiKey
                ),
                CURLOPT_SSL_VERIFYPEER => false
            ));
            
            $response = curl_exec($curl);
            $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
            $error = curl_error($curl);
            curl_close($curl);
            
            if ($error) {
                throw new Exception($error);
            }
            
            if ($httpCode >= 400) {
                throw new Exception("HTTP Error: {$httpCode}");
            }
            
            $decodedResponse = json_decode($response, true);
            if (json_last_error() !== JSON_ERROR_NONE) {
                throw new Exception("Invalid JSON response");
            }
            
            header('Content-Type: application/json');
            echo json_encode(array(
                'success' => true,
                'message' => 'Connection successful',
                'agentStatus' => $decodedResponse
            ));
            
        } catch (Exception $e) {
            header('HTTP/1.0 500 Internal Server Error');
            echo json_encode(array(
                'success' => false,
                'error' => $e->getMessage()
            ));
        }
    }
    
    /**
     * API endpoint for agent requests
     * @param array $args Request arguments (agent/action)
     * @param PKPRequest $request
     */
    public function api($args, $request) {
        $context = $request->getContext();
        if (!$this->plugin->getEnabled($context->getId())) {
            header('HTTP/1.0 403 Forbidden');
            echo json_encode(array('error' => 'Plugin disabled'));
            return;
        }
        
        try {
            import('plugins.generic.skzAgents.classes.SKZAPIRouter');
            $router = new SKZAPIRouter();
            
            // Handle different API endpoints
            if (empty($args)) {
                // General status endpoint
                $response = $router->getStatus($request, $args);
            } elseif ($args[0] === 'webhook') {
                if (count($args) > 1 && $args[1] === 'register') {
                    // Register webhook endpoint
                    $response = $router->registerWebhook($request, array_slice($args, 2));
                } else {
                    // Handle incoming webhook
                    $response = $router->handleWebhook($request, array_slice($args, 1));
                }
            } else {
                // Agent request endpoint
                $response = $router->handleRequest($request, $args);
            }
            
            header('Content-Type: application/json');
            echo json_encode($response);
            
        } catch (Exception $e) {
            header('HTTP/1.0 500 Internal Server Error');
            echo json_encode(array(
                'status' => 'error',
                'error' => array(
                    'message' => $e->getMessage(),
                    'code' => 500
                )
            ));
        }
    }
    
    /**
     * Webhook endpoint for agent callbacks
     * @param array $args
     * @param PKPRequest $request
     */
    public function webhook($args, $request) {
        try {
            import('plugins.generic.skzAgents.classes.SKZAPIRouter');
            $router = new SKZAPIRouter();
            
            $response = $router->handleWebhook($request, $args);
            
            header('Content-Type: application/json');
            echo json_encode($response);
            
        } catch (Exception $e) {
            header('HTTP/1.0 500 Internal Server Error');
            echo json_encode(array(
                'status' => 'error',
                'error' => array(
                    'message' => $e->getMessage(),
                    'code' => 500
                )
            ));
        }
    }
}

?>