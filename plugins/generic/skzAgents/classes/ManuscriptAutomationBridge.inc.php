<?php

/**
 * @file plugins/generic/skzAgents/classes/ManuscriptAutomationBridge.inc.php
 *
 * SKZ Manuscript Processing Automation Bridge for OJS
 * Integrates automated manuscript processing with OJS submission workflow
 */

class ManuscriptAutomationBridge {
    
    /** @var string Base URL for automation API */
    private $automationApiUrl;
    
    /** @var string API key for authentication */
    private $apiKey;
    
    /** @var int Request timeout in seconds */
    private $timeout;
    
    public function __construct() {
        $this->automationApiUrl = Config::getVar('skz', 'automation_api_url', 'http://localhost:5000/api/v1/automation');
        $this->apiKey = Config::getVar('skz', 'api_key', '');
        $this->timeout = Config::getVar('skz', 'timeout', 30);
    }
    
    /**
     * Submit manuscript for automated processing
     * 
     * @param Submission $submission OJS submission object
     * @return array Automation response with workflow_id
     */
    public function submitManuscriptForAutomation($submission) {
        try {
            // Extract manuscript data from OJS submission
            $manuscriptData = $this->extractManuscriptData($submission);
            
            // Send to automation API
            $response = $this->makeApiRequest('/submit', 'POST', $manuscriptData);
            
            if ($response && isset($response['success']) && $response['success']) {
                // Store workflow ID in submission settings
                $this->storeWorkflowId($submission, $response['workflow_id']);
                
                // Log automation start
                $this->logAutomationEvent($submission->getId(), 'automation_started', [
                    'workflow_id' => $response['workflow_id'],
                    'status' => $response['status'],
                    'estimated_completion' => $response['estimated_completion']
                ]);
                
                return $response;
            } else {
                throw new Exception('Automation API returned error: ' . json_encode($response));
            }
            
        } catch (Exception $e) {
            error_log("Manuscript automation submission failed: " . $e->getMessage());
            return [
                'success' => false,
                'error' => $e->getMessage()
            ];
        }
    }
    
    /**
     * Get automation workflow status
     * 
     * @param string $workflowId Automation workflow ID
     * @return array Workflow status information
     */
    public function getWorkflowStatus($workflowId) {
        try {
            $response = $this->makeApiRequest("/status/{$workflowId}", 'GET');
            
            if ($response && isset($response['success']) && $response['success']) {
                return $response;
            } else {
                throw new Exception('Failed to get workflow status: ' . json_encode($response));
            }
            
        } catch (Exception $e) {
            error_log("Failed to get workflow status: " . $e->getMessage());
            return [
                'success' => false,
                'error' => $e->getMessage()
            ];
        }
    }
    
    /**
     * Get automation system metrics
     * 
     * @return array System performance metrics
     */
    public function getAutomationMetrics() {
        try {
            $response = $this->makeApiRequest('/metrics', 'GET');
            
            if ($response && isset($response['success']) && $response['success']) {
                return $response;
            } else {
                throw new Exception('Failed to get automation metrics: ' . json_encode($response));
            }
            
        } catch (Exception $e) {
            error_log("Failed to get automation metrics: " . $e->getMessage());
            return [
                'success' => false,
                'error' => $e->getMessage()
            ];
        }
    }
    
    /**
     * Check automation system health
     * 
     * @return boolean True if system is healthy
     */
    public function checkAutomationHealth() {
        try {
            $response = $this->makeApiRequest('/health', 'GET');
            
            return $response && 
                   isset($response['success']) && 
                   $response['success'] && 
                   isset($response['status']) && 
                   $response['status'] === 'healthy';
                   
        } catch (Exception $e) {
            error_log("Automation health check failed: " . $e->getMessage());
            return false;
        }
    }
    
    /**
     * Extract manuscript data from OJS submission for automation
     * 
     * @param Submission $submission OJS submission object
     * @return array Manuscript data for automation
     */
    private function extractManuscriptData($submission) {
        $publication = $submission->getCurrentPublication();
        $context = Application::get()->getRequest()->getContext();
        
        // Get authors
        $authors = [];
        $authorDao = DAORegistry::getDAO('AuthorDAO');
        $submissionAuthors = $authorDao->getBySubmissionId($submission->getId());
        
        while ($author = $submissionAuthors->next()) {
            $authors[] = [
                'name' => $author->getFullName(),
                'email' => $author->getEmail(),
                'orcid' => $author->getOrcid(),
                'affiliation' => $author->getAffiliation(null)
            ];
        }
        
        // Get keywords
        $keywords = [];
        $keywordString = $publication->getLocalizedData('keywords');
        if (!empty($keywordString)) {
            $keywords = array_map('trim', explode(',', $keywordString));
        }
        
        // Get submission files
        $filePaths = [];
        $submissionFileDao = DAORegistry::getDAO('SubmissionFileDAO');
        $submissionFiles = $submissionFileDao->getBySubmissionId($submission->getId());
        
        foreach ($submissionFiles as $submissionFile) {
            $filePaths[] = $submissionFile->getFilePath();
        }
        
        // Determine field of study and research type from context/metadata
        $fieldOfStudy = 'cosmetic_science'; // Default for SKZ
        $researchType = 'experimental'; // Default
        
        // Try to extract from submission metadata or keywords
        if (!empty($keywords)) {
            $cosmeticKeywords = ['cosmetics', 'formulation', 'skincare', 'beauty', 'dermatology'];
            $clinicalKeywords = ['clinical', 'trial', 'safety', 'efficacy'];
            
            $keywordsLower = array_map('strtolower', $keywords);
            
            if (array_intersect($keywordsLower, $clinicalKeywords)) {
                $researchType = 'clinical';
                $fieldOfStudy = 'clinical_research';
            } elseif (array_intersect($keywordsLower, $cosmeticKeywords)) {
                $fieldOfStudy = 'cosmetic_science';
            }
        }
        
        // Determine priority based on submission context
        $priority = 2; // Normal priority
        
        // Higher priority for special issues or invited submissions
        if ($submission->getData('submissionStage') == WORKFLOW_STAGE_ID_EXTERNAL_REVIEW) {
            $priority = 3; // High priority if already in review
        }
        
        // Special requirements based on field and keywords
        $specialRequirements = [];
        if ($fieldOfStudy === 'cosmetic_science') {
            $specialRequirements[] = 'inci_verification';
            $specialRequirements[] = 'safety_assessment';
        }
        if ($researchType === 'clinical') {
            $specialRequirements[] = 'ethics_validation';
            $specialRequirements[] = 'regulatory_check';
        }
        
        return [
            'id' => 'ojs_submission_' . $submission->getId(),
            'ojs_submission_id' => $submission->getId(),
            'title' => $publication->getLocalizedData('title'),
            'authors' => $authors,
            'abstract' => strip_tags($publication->getLocalizedData('abstract')),
            'keywords' => $keywords,
            'research_type' => $researchType,
            'field_of_study' => $fieldOfStudy,
            'file_paths' => $filePaths,
            'priority' => $priority,
            'special_requirements' => $specialRequirements,
            'submission_date' => $submission->getDateSubmitted(),
            'journal_context' => [
                'journal_id' => $context->getId(),
                'journal_name' => $context->getLocalizedName(),
                'journal_issn' => $context->getData('printIssn') ?: $context->getData('onlineIssn')
            ]
        ];
    }
    
    /**
     * Store workflow ID in submission settings
     * 
     * @param Submission $submission OJS submission object
     * @param string $workflowId Automation workflow ID
     */
    private function storeWorkflowId($submission, $workflowId) {
        $submission->setData('skz_automation_workflow_id', $workflowId);
        $submissionDao = DAORegistry::getDAO('SubmissionDAO');
        $submissionDao->updateObject($submission);
    }
    
    /**
     * Get stored workflow ID for submission
     * 
     * @param Submission $submission OJS submission object
     * @return string|null Workflow ID or null if not found
     */
    public function getStoredWorkflowId($submission) {
        return $submission->getData('skz_automation_workflow_id');
    }
    
    /**
     * Make API request to automation system
     * 
     * @param string $endpoint API endpoint
     * @param string $method HTTP method (GET, POST, etc.)
     * @param array $data Request data for POST requests
     * @return array API response
     */
    private function makeApiRequest($endpoint, $method = 'GET', $data = null) {
        $url = $this->automationApiUrl . $endpoint;
        
        $ch = curl_init();
        
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, $this->timeout);
        curl_setopt($ch, CURLOPT_CUSTOMREQUEST, $method);
        
        // Set headers
        $headers = [
            'Content-Type: application/json',
            'Accept: application/json'
        ];
        
        if (!empty($this->apiKey)) {
            $headers[] = 'Authorization: Bearer ' . $this->apiKey;
        }
        
        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
        
        // Add data for POST requests
        if ($method === 'POST' && $data) {
            curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
        }
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);
        curl_close($ch);
        
        if ($error) {
            throw new Exception("CURL error: " . $error);
        }
        
        if ($httpCode >= 400) {
            throw new Exception("HTTP error: " . $httpCode . " - " . $response);
        }
        
        $decodedResponse = json_decode($response, true);
        if ($decodedResponse === null) {
            throw new Exception("Invalid JSON response: " . $response);
        }
        
        return $decodedResponse;
    }
    
    /**
     * Log automation event
     * 
     * @param int $submissionId Submission ID
     * @param string $eventType Event type
     * @param array $eventData Event data
     */
    private function logAutomationEvent($submissionId, $eventType, $eventData) {
        // Create a submission event log entry
        $submissionEventLogDao = DAORegistry::getDAO('SubmissionEventLogDAO');
        $entry = $submissionEventLogDao->newDataObject();
        
        $entry->setSubmissionId($submissionId);
        $entry->setUserId(null); // System-generated event
        $entry->setDateLogged(Core::getCurrentDate());
        $entry->setEventType(SUBMISSION_LOG_TYPE_DEFAULT);
        $entry->setAssocType(ASSOC_TYPE_SUBMISSION);
        $entry->setAssocId($submissionId);
        $entry->setMessage('skz.automation.' . $eventType);
        
        // Store event data as parameters
        if ($eventData) {
            foreach ($eventData as $key => $value) {
                $entry->setData($key, $value);
            }
        }
        
        $submissionEventLogDao->insertObject($entry);
    }
    
    /**
     * Update submission based on automation results
     * 
     * @param Submission $submission OJS submission object
     * @param array $automationResults Results from automation workflow
     */
    public function updateSubmissionFromAutomation($submission, $automationResults) {
        try {
            // Process automation results and update submission
            if (isset($automationResults['editorial_decision'])) {
                $this->processEditorialDecision($submission, $automationResults['editorial_decision']);
            }
            
            if (isset($automationResults['quality_assessment'])) {
                $this->storeQualityAssessment($submission, $automationResults['quality_assessment']);
            }
            
            if (isset($automationResults['review_recommendations'])) {
                $this->processReviewRecommendations($submission, $automationResults['review_recommendations']);
            }
            
            // Log the update
            $this->logAutomationEvent($submission->getId(), 'results_processed', $automationResults);
            
        } catch (Exception $e) {
            error_log("Error updating submission from automation: " . $e->getMessage());
        }
    }
    
    /**
     * Process editorial decision from automation
     * 
     * @param Submission $submission OJS submission object
     * @param array $decision Editorial decision data
     */
    private function processEditorialDecision($submission, $decision) {
        // Store decision data in submission settings
        $submission->setData('skz_automation_editorial_decision', json_encode($decision));
        
        // Update submission based on decision
        if (isset($decision['recommendation'])) {
            switch ($decision['recommendation']) {
                case 'accept':
                case 'minor_revision':
                case 'major_revision':
                case 'reject':
                    $submission->setData('skz_automation_recommendation', $decision['recommendation']);
                    break;
            }
        }
        
        $submissionDao = DAORegistry::getDAO('SubmissionDAO');
        $submissionDao->updateObject($submission);
    }
    
    /**
     * Store quality assessment results
     * 
     * @param Submission $submission OJS submission object
     * @param array $assessment Quality assessment data
     */
    private function storeQualityAssessment($submission, $assessment) {
        $submission->setData('skz_automation_quality_assessment', json_encode($assessment));
        
        $submissionDao = DAORegistry::getDAO('SubmissionDAO');
        $submissionDao->updateObject($submission);
    }
    
    /**
     * Process review recommendations from automation
     * 
     * @param Submission $submission OJS submission object
     * @param array $recommendations Review recommendations
     */
    private function processReviewRecommendations($submission, $recommendations) {
        $submission->setData('skz_automation_review_recommendations', json_encode($recommendations));
        
        $submissionDao = DAORegistry::getDAO('SubmissionDAO');
        $submissionDao->updateObject($submission);
    }
}