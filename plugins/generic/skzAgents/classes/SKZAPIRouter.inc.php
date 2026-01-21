<?php

/**
 * @file plugins/generic/skzAgents/classes/SKZAPIRouter.inc.php
 *
 * SKZ API Router - Handles routing and request processing for agent API calls
 */

class SKZAPIRouter {
    
    /** @var SKZAPIGateway */
    private $gateway;
    
    /** @var array Cached route configurations */
    private $routes = array();
    
    /**
     * Constructor
     */
    public function __construct() {
        import('plugins.generic.skzAgents.classes.SKZAPIGateway');
        $this->gateway = new SKZAPIGateway();
        $this->initializeRoutes();
    }
    
    /**
     * Handle API request
     * 
     * @param PKPRequest $request OJS request object
     * @param array $args Request arguments
     * @return array API response
     */
    public function handleRequest($request, $args) {
        try {
            // Parse request path
            $path = $this->parseRequestPath($args);
            
            // Find matching route
            $route = $this->findRoute($path['agent'], $path['action']);
            if (!$route) {
                return $this->createErrorResponse('Route not found', 404);
            }
            
            // Extract request data
            $data = $this->extractRequestData($request);
            
            // Validate request data
            if (!$this->validateRequestData($data, $route)) {
                return $this->createErrorResponse('Invalid request data', 400);
            }
            
            // Route through gateway
            $response = $this->gateway->routeAgentRequest(
                $path['agent'],
                $path['action'],
                $data,
                $this->getRequestOptions($request)
            );
            
            return $response;
            
        } catch (Exception $e) {
            return $this->createErrorResponse('Router error: ' . $e->getMessage(), 500);
        }
    }
    
    /**
     * Handle webhook request from agents
     * 
     * @param PKPRequest $request OJS request object
     * @param array $args Request arguments
     * @return array Webhook response
     */
    public function handleWebhook($request, $args) {
        try {
            // Verify webhook signature
            if (!$this->verifyWebhookSignature($request)) {
                return $this->createErrorResponse('Invalid webhook signature', 401);
            }
            
            // Extract webhook data
            $data = $this->extractRequestData($request);
            
            // Process webhook event
            $result = $this->processWebhookEvent($data);
            
            return $this->createSuccessResponse($result);
            
        } catch (Exception $e) {
            return $this->createErrorResponse('Webhook error: ' . $e->getMessage(), 500);
        }
    }
    
    /**
     * Get agent status
     * 
     * @param PKPRequest $request OJS request object
     * @param array $args Request arguments
     * @return array Status response
     */
    public function getStatus($request, $args) {
        try {
            $agentName = isset($args[0]) ? $args[0] : null;
            
            $status = $this->gateway->getAgentStatus($agentName);
            
            return $status;
            
        } catch (Exception $e) {
            return $this->createErrorResponse('Status error: ' . $e->getMessage(), 500);
        }
    }
    
    /**
     * Register webhook
     * 
     * @param PKPRequest $request OJS request object
     * @param array $args Request arguments
     * @return array Registration response
     */
    public function registerWebhook($request, $args) {
        try {
            $data = $this->extractRequestData($request);
            
            if (!isset($data['event']) || !isset($data['callback_url'])) {
                return $this->createErrorResponse('Missing required fields: event, callback_url', 400);
            }
            
            $response = $this->gateway->registerWebhook(
                $data['event'],
                $data['callback_url'],
                $data
            );
            
            return $response;
            
        } catch (Exception $e) {
            return $this->createErrorResponse('Webhook registration error: ' . $e->getMessage(), 500);
        }
    }
    
    /**
     * Parse request path to extract agent and action
     * 
     * @param array $args Request arguments
     * @return array Parsed path components
     */
    private function parseRequestPath($args) {
        $agent = isset($args[0]) ? $args[0] : null;
        $action = isset($args[1]) ? $args[1] : 'default';
        
        return array(
            'agent' => $agent,
            'action' => $action
        );
    }
    
    /**
     * Find route configuration
     * 
     * @param string $agent Agent name
     * @param string $action Action name
     * @return array|null Route configuration
     */
    private function findRoute($agent, $action) {
        $routeKey = $agent . '.' . $action;
        
        if (isset($this->routes[$routeKey])) {
            return $this->routes[$routeKey];
        }
        
        // Check for wildcard route
        $wildcardKey = $agent . '.*';
        if (isset($this->routes[$wildcardKey])) {
            return $this->routes[$wildcardKey];
        }
        
        return null;
    }
    
    /**
     * Extract request data from OJS request
     * 
     * @param PKPRequest $request OJS request object
     * @return array Request data
     */
    private function extractRequestData($request) {
        $data = array();
        
        // Get JSON data from request body
        $inputJSON = file_get_contents('php://input');
        if ($inputJSON) {
            $jsonData = json_decode($inputJSON, true);
            if ($jsonData !== null) {
                $data = array_merge($data, $jsonData);
            }
        }
        
        // Get POST data
        $postData = $request->getUserVars();
        if ($postData) {
            $data = array_merge($data, $postData);
        }
        
        return $data;
    }
    
    /**
     * Validate request data against route requirements
     * 
     * @param array $data Request data
     * @param array $route Route configuration
     * @return bool Validation result
     */
    private function validateRequestData($data, $route) {
        // Check required fields
        if (isset($route['required_fields'])) {
            foreach ($route['required_fields'] as $field) {
                if (!isset($data[$field])) {
                    return false;
                }
            }
        }
        
        // Check data types
        if (isset($route['field_types'])) {
            foreach ($route['field_types'] as $field => $type) {
                if (isset($data[$field])) {
                    switch ($type) {
                        case 'integer':
                            if (!is_numeric($data[$field])) {
                                return false;
                            }
                            break;
                        case 'array':
                            if (!is_array($data[$field])) {
                                return false;
                            }
                            break;
                        case 'string':
                            if (!is_string($data[$field])) {
                                return false;
                            }
                            break;
                    }
                }
            }
        }
        
        return true;
    }
    
    /**
     * Get request options for gateway
     * 
     * @param PKPRequest $request OJS request object
     * @return array Request options
     */
    private function getRequestOptions($request) {
        $user = $request->getUser();
        
        return array(
            'user_id' => $user ? $user->getId() : null,
            'user_roles' => $user ? $this->getUserRoles($user) : array(),
            'context_id' => $request->getContext() ? $request->getContext()->getId() : null,
            'ip_address' => $request->getRemoteAddr(),
            'user_agent' => isset($_SERVER['HTTP_USER_AGENT']) ? $_SERVER['HTTP_USER_AGENT'] : ''
        );
    }
    
    /**
     * Get user roles
     * 
     * @param User $user User object
     * @return array User roles
     */
    private function getUserRoles($user) {
        $roles = array();
        $userGroupDao = DAORegistry::getDAO('UserGroupDAO');
        $userGroups = $userGroupDao->getByUserId($user->getId());
        
        while ($userGroup = $userGroups->next()) {
            $roles[] = $userGroup->getRoleId();
        }
        
        return $roles;
    }
    
    /**
     * Verify webhook signature
     * 
     * @param PKPRequest $request OJS request object
     * @return bool Verification result
     */
    private function verifyWebhookSignature($request) {
        // Get signature from header
        $signature = isset($_SERVER['HTTP_X_SKZ_SIGNATURE']) ? $_SERVER['HTTP_X_SKZ_SIGNATURE'] : '';
        
        if (!$signature) {
            return false;
        }
        
        // Get webhook secret
        $plugin = PluginRegistry::getPlugin('generic', 'skzAgents');
        $context = $request->getContext();
        $contextId = $context ? $context->getId() : 0;
        $secret = $plugin ? $plugin->getSetting($contextId, 'webhookSecret') : '';
        
        if (!$secret) {
            return false;
        }
        
        // Calculate expected signature
        $payload = file_get_contents('php://input');
        $expectedSignature = 'sha256=' . hash_hmac('sha256', $payload, $secret);
        
        // Compare signatures
        return hash_equals($expectedSignature, $signature);
    }
    
    /**
     * Process webhook event
     * 
     * @param array $data Webhook data
     * @return array Processing result
     */
    private function processWebhookEvent($data) {
        $event = isset($data['event']) ? $data['event'] : '';
        $payload = isset($data['payload']) ? $data['payload'] : array();
        
        switch ($event) {
            case 'agent.status.changed':
                return $this->handleAgentStatusChange($payload);
                
            case 'agent.task.completed':
                return $this->handleAgentTaskCompleted($payload);
                
            case 'agent.task.failed':
                return $this->handleAgentTaskFailed($payload);
                
            case 'workflow.status.updated':
                return $this->handleWorkflowStatusUpdate($payload);
                
            default:
                return array('message' => 'Event processed');
        }
    }
    
    /**
     * Handle agent status change event
     * 
     * @param array $payload Event payload
     * @return array Processing result
     */
    private function handleAgentStatusChange($payload) {
        // Update agent status in database
        import('plugins.generic.skzAgents.classes.SKZDAO');
        $skzDAO = new SKZDAO();
        
        if (isset($payload['agent_id']) && isset($payload['status'])) {
            $skzDAO->updateAgentStatus($payload['agent_id'], $payload['status']);
        }
        
        return array('message' => 'Agent status updated');
    }
    
    /**
     * Handle agent task completed event
     * 
     * @param array $payload Event payload
     * @return array Processing result
     */
    private function handleAgentTaskCompleted($payload) {
        // Process completed task results
        if (isset($payload['task_id']) && isset($payload['results'])) {
            import('plugins.generic.skzAgents.classes.SKZDAO');
            $skzDAO = new SKZDAO();
            $skzDAO->storeTaskResults($payload['task_id'], $payload['results']);
            
            // Trigger workflow events if needed
            if (isset($payload['workflow_id'])) {
                $this->triggerWorkflowEvent($payload['workflow_id'], 'task_completed', $payload);
            }
        }
        
        return array('message' => 'Task completion processed');
    }
    
    /**
     * Handle agent task failed event
     * 
     * @param array $payload Event payload
     * @return array Processing result
     */
    private function handleAgentTaskFailed($payload) {
        // Log task failure
        if (isset($payload['task_id']) && isset($payload['error'])) {
            error_log('SKZ Agent Task Failed: ' . $payload['task_id'] . ' - ' . $payload['error']);
            
            // Store failure information
            import('plugins.generic.skzAgents.classes.SKZDAO');
            $skzDAO = new SKZDAO();
            $skzDAO->storeTaskFailure($payload['task_id'], $payload['error']);
        }
        
        return array('message' => 'Task failure processed');
    }
    
    /**
     * Handle workflow status update event
     * 
     * @param array $payload Event payload
     * @return array Processing result
     */
    private function handleWorkflowStatusUpdate($payload) {
        // Update workflow status
        if (isset($payload['workflow_id']) && isset($payload['status'])) {
            import('plugins.generic.skzAgents.classes.SKZDAO');
            $skzDAO = new SKZDAO();
            $skzDAO->updateWorkflowStatus($payload['workflow_id'], $payload['status']);
        }
        
        return array('message' => 'Workflow status updated');
    }
    
    /**
     * Trigger workflow event
     * 
     * @param string $workflowId Workflow ID
     * @param string $event Event name
     * @param array $data Event data
     */
    private function triggerWorkflowEvent($workflowId, $event, $data) {
        // Trigger OJS workflow hooks based on agent events
        HookRegistry::call('SKZAgents::workflowEvent', array($workflowId, $event, $data));
    }
    
    /**
     * Initialize route configurations
     */
    private function initializeRoutes() {
        $this->routes = array(
            // Research Discovery Agent routes
            'research-discovery.literature_search' => array(
                'required_fields' => array('query', 'filters'),
                'field_types' => array('query' => 'string', 'filters' => 'array')
            ),
            'research-discovery.gap_analysis' => array(
                'required_fields' => array('manuscript_content'),
                'field_types' => array('manuscript_content' => 'string')
            ),
            'research-discovery.trend_identification' => array(
                'required_fields' => array('domain', 'timeframe'),
                'field_types' => array('domain' => 'string', 'timeframe' => 'string')
            ),
            'research-discovery.inci_analysis' => array(
                'required_fields' => array('ingredients'),
                'field_types' => array('ingredients' => 'array')
            ),
            
            // Submission Assistant Agent routes
            'submission-assistant.format_check' => array(
                'required_fields' => array('manuscript_file'),
                'field_types' => array('manuscript_file' => 'string')
            ),
            'submission-assistant.venue_recommendation' => array(
                'required_fields' => array('manuscript_content', 'research_area'),
                'field_types' => array('manuscript_content' => 'string', 'research_area' => 'string')
            ),
            'submission-assistant.compliance_validation' => array(
                'required_fields' => array('submission_data'),
                'field_types' => array('submission_data' => 'array')
            ),
            'submission-assistant.quality_assessment' => array(
                'required_fields' => array('manuscript_content'),
                'field_types' => array('manuscript_content' => 'string')
            ),
            
            // Editorial Orchestration Agent routes
            'editorial-orchestration.workflow_management' => array(
                'required_fields' => array('submission_id', 'current_stage'),
                'field_types' => array('submission_id' => 'integer', 'current_stage' => 'string')
            ),
            'editorial-orchestration.decision_support' => array(
                'required_fields' => array('submission_id', 'review_data'),
                'field_types' => array('submission_id' => 'integer', 'review_data' => 'array')
            ),
            'editorial-orchestration.deadline_tracking' => array(
                'required_fields' => array('submission_id'),
                'field_types' => array('submission_id' => 'integer')
            ),
            'editorial-orchestration.conflict_resolution' => array(
                'required_fields' => array('conflict_data'),
                'field_types' => array('conflict_data' => 'array')
            ),
            
            // Review Coordination Agent routes
            'review-coordination.reviewer_matching' => array(
                'required_fields' => array('submission_id', 'manuscript_keywords'),
                'field_types' => array('submission_id' => 'integer', 'manuscript_keywords' => 'array')
            ),
            'review-coordination.review_tracking' => array(
                'required_fields' => array('submission_id'),
                'field_types' => array('submission_id' => 'integer')
            ),
            'review-coordination.quality_assessment' => array(
                'required_fields' => array('review_id', 'review_content'),
                'field_types' => array('review_id' => 'integer', 'review_content' => 'string')
            ),
            'review-coordination.workload_management' => array(
                'required_fields' => array('reviewer_id'),
                'field_types' => array('reviewer_id' => 'integer')
            ),
            
            // Content Quality Agent routes
            'content-quality.scientific_validation' => array(
                'required_fields' => array('manuscript_content'),
                'field_types' => array('manuscript_content' => 'string')
            ),
            'content-quality.safety_assessment' => array(
                'required_fields' => array('content_data'),
                'field_types' => array('content_data' => 'array')
            ),
            'content-quality.standards_enforcement' => array(
                'required_fields' => array('submission_id', 'standards'),
                'field_types' => array('submission_id' => 'integer', 'standards' => 'array')
            ),
            'content-quality.regulatory_compliance' => array(
                'required_fields' => array('content_data', 'regulations'),
                'field_types' => array('content_data' => 'array', 'regulations' => 'array')
            ),
            
            // Publishing Production Agent routes
            'publishing-production.content_formatting' => array(
                'required_fields' => array('content', 'format_type'),
                'field_types' => array('content' => 'string', 'format_type' => 'string')
            ),
            'publishing-production.visual_generation' => array(
                'required_fields' => array('data', 'chart_type'),
                'field_types' => array('data' => 'array', 'chart_type' => 'string')
            ),
            'publishing-production.distribution_preparation' => array(
                'required_fields' => array('submission_id', 'channels'),
                'field_types' => array('submission_id' => 'integer', 'channels' => 'array')
            ),
            'publishing-production.regulatory_reporting' => array(
                'required_fields' => array('publication_data'),
                'field_types' => array('publication_data' => 'array')
            ),
            
            // Analytics & Monitoring Agent routes
            'analytics-monitoring.performance_analytics' => array(
                'required_fields' => array('metrics_type', 'time_range'),
                'field_types' => array('metrics_type' => 'string', 'time_range' => 'string')
            ),
            'analytics-monitoring.trend_forecasting' => array(
                'required_fields' => array('data_points'),
                'field_types' => array('data_points' => 'array')
            ),
            'analytics-monitoring.strategic_insights' => array(
                'required_fields' => array('analysis_scope'),
                'field_types' => array('analysis_scope' => 'string')
            ),
            'analytics-monitoring.continuous_learning' => array(
                'required_fields' => array('feedback_data'),
                'field_types' => array('feedback_data' => 'array')
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
}