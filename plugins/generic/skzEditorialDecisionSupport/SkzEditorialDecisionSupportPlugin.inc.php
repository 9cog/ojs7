<?php

/**
 * @file plugins/generic/skzEditorialDecisionSupport/SkzEditorialDecisionSupportPlugin.inc.php
 *
 * SKZ Editorial Decision Support Plugin
 * Integrates SKZ autonomous agents decision engines with OJS editorial workflow
 */

import('lib.pkp.classes.plugins.GenericPlugin');

class SkzEditorialDecisionSupportPlugin extends GenericPlugin {

	/**
	 * @copydoc Plugin::register()
	 */
	function register($category, $path, $mainContextId = null) {
		$success = parent::register($category, $path, $mainContextId);
		if (!Config::getVar('general', 'installed') || defined('RUNNING_UPGRADE')) return $success;

		if ($success && $this->getEnabled($mainContextId)) {
			// Register hooks for editorial workflow
			HookRegistry::register('EditorAction::recordDecision', array($this, 'recordEditorialDecision'));
			HookRegistry::register('ReviewAssignmentDAO::getReviewsForSubmission', array($this, 'enhanceReviewData'));
			HookRegistry::register('WorkflowHandler::fetchTab', array($this, 'addDecisionSupportTab'));
			
			// Add JavaScript and CSS resources
			HookRegistry::register('TemplateManager::display', array($this, 'addAssets'));
		}
		return $success;
	}

	/**
	 * @copydoc Plugin::getDisplayName()
	 */
	function getDisplayName() {
		return __('plugins.generic.skzEditorialDecisionSupport.displayName');
	}

	/**
	 * @copydoc Plugin::getDescription()
	 */
	function getDescription() {
		return __('plugins.generic.skzEditorialDecisionSupport.description');
	}

	/**
	 * Record editorial decision and send to SKZ decision engine for learning
	 */
	function recordEditorialDecision($hookName, $params) {
		$submission = $params[0];
		$decision = $params[1];
		$reviewAssignments = $params[2];

		// Gather manuscript and review data
		$manuscriptData = $this->prepareManuscriptData($submission);
		$reviewData = $this->prepareReviewData($reviewAssignments);

		// Send to SKZ decision engine for analysis and learning
		$this->sendToDecisionEngine($submission->getId(), $manuscriptData, $reviewData, $decision);

		return false; // Continue with normal processing
	}

	/**
	 * Enhance review data with SKZ analysis
	 */
	function enhanceReviewData($hookName, $params) {
		$reviewAssignments = &$params[1];
		
		foreach ($reviewAssignments as &$reviewAssignment) {
			// Add SKZ quality assessment
			$reviewAssignment->setData('skzQualityScore', $this->getReviewQualityScore($reviewAssignment));
		}

		return false;
	}

	/**
	 * Add decision support tab to editorial workflow
	 */
	function addDecisionSupportTab($hookName, $params) {
		$templateMgr = &$params[0];
		$template = &$params[1];

		if (strpos($template, 'workflow/') === 0) {
			// Get submission from template variables
			$submission = $templateMgr->getTemplateVars('submission');
			
			if ($submission) {
				// Get SKZ decision recommendation
				$recommendation = $this->getDecisionRecommendation($submission);
				$templateMgr->assign('skzDecisionRecommendation', $recommendation);
			}
		}

		return false;
	}

	/**
	 * Add JavaScript and CSS assets
	 */
	function addAssets($hookName, $params) {
		$templateMgr = &$params[0];
		$template = &$params[1];

		// Add assets only on workflow pages
		if (strpos($template, 'workflow/') === 0) {
			$request = Application::get()->getRequest();
			$templateMgr->addJavaScript(
				'skzDecisionSupport',
				$request->getBaseUrl() . '/' . $this->getPluginPath() . '/js/SkzDecisionSupport.js',
				array('contexts' => 'backend')
			);
			$templateMgr->addStyleSheet(
				'skzDecisionSupport',
				$request->getBaseUrl() . '/' . $this->getPluginPath() . '/css/skz-decision-support.css',
				array('contexts' => 'backend')
			);
		}

		return false;
	}

	/**
	 * Prepare manuscript data for SKZ analysis
	 */
	private function prepareManuscriptData($submission) {
		return array(
			'manuscript_id' => $submission->getId(),
			'title' => $submission->getLocalizedTitle(),
			'abstract' => $submission->getLocalizedAbstract(),
			'content' => $this->extractManuscriptContent($submission),
			'keywords' => $submission->getLocalizedSubject(),
			'methodology_score' => 0.7, // Would be calculated from content analysis
			'clarity_score' => 0.8,
			'completeness_score' => 0.9,
			'submission_date' => $submission->getDateSubmitted()
		);
	}

	/**
	 * Prepare review data for SKZ analysis  
	 */
	private function prepareReviewData($reviewAssignments) {
		$reviewData = array();
		
		foreach ($reviewAssignments as $reviewAssignment) {
			if ($reviewAssignment->getDateCompleted()) {
				$reviewData[] = array(
					'reviewer_id' => $reviewAssignment->getReviewerId(),
					'recommendation' => $this->mapRecommendation($reviewAssignment->getRecommendation()),
					'confidence' => 0.8, // Would be extracted from review
					'review_quality' => $this->assessReviewQuality($reviewAssignment),
					'technical_comments' => $this->extractTechnicalComments($reviewAssignment),
					'major_issues' => $this->extractMajorIssues($reviewAssignment),
					'minor_issues' => $this->extractMinorIssues($reviewAssignment),
					'review_completeness' => 0.9,
					'review_timeliness' => $this->calculateTimeliness($reviewAssignment)
				);
			}
		}

		return $reviewData;
	}

	/**
	 * Send data to SKZ decision engine
	 */
	private function sendToDecisionEngine($manuscriptId, $manuscriptData, $reviewData, $decision) {
		$skzApiUrl = Config::getVar('skz', 'decision_engine_url', 'http://localhost:8004');
		
		$postData = array(
			'manuscript_id' => $manuscriptId,
			'manuscript' => $manuscriptData,
			'reviews' => $reviewData,
			'editor_decision' => $decision
		);

		// Make HTTP request to SKZ decision engine
		$ch = curl_init();
		curl_setopt($ch, CURLOPT_URL, $skzApiUrl . '/make-decision');
		curl_setopt($ch, CURLOPT_POST, true);
		curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($postData));
		curl_setopt($ch, CURLOPT_HTTPHEADER, array(
			'Content-Type: application/json'
		));
		curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
		curl_setopt($ch, CURLOPT_TIMEOUT, 30);

		$response = curl_exec($ch);
		$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
		curl_close($ch);

		if ($httpCode == 200) {
			$result = json_decode($response, true);
			error_log("SKZ Decision Engine Response: " . print_r($result, true));
			return $result;
		} else {
			error_log("SKZ Decision Engine Error: HTTP $httpCode");
			return false;
		}
	}

	/**
	 * Get decision recommendation from SKZ engine
	 */
	private function getDecisionRecommendation($submission) {
		$manuscriptData = $this->prepareManuscriptData($submission);
		
		// Get current reviews
		$reviewAssignmentDao = DAORegistry::getDAO('ReviewAssignmentDAO');
		$reviewAssignments = $reviewAssignmentDao->getBySubmissionId($submission->getId());
		$reviewData = $this->prepareReviewData($reviewAssignments);

		$skzApiUrl = Config::getVar('skz', 'decision_engine_url', 'http://localhost:8004');
		
		$postData = array(
			'manuscript_id' => $submission->getId(),
			'manuscript' => $manuscriptData,
			'reviews' => $reviewData
		);

		$ch = curl_init();
		curl_setopt($ch, CURLOPT_URL, $skzApiUrl . '/make-decision');
		curl_setopt($ch, CURLOPT_POST, true);
		curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($postData));
		curl_setopt($ch, CURLOPT_HTTPHEADER, array(
			'Content-Type: application/json'
		));
		curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
		curl_setopt($ch, CURLOPT_TIMEOUT, 10);

		$response = curl_exec($ch);
		$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
		curl_close($ch);

		if ($httpCode == 200) {
			return json_decode($response, true);
		} else {
			// Return default recommendation if service unavailable
			return array(
				'status' => 'service_unavailable',
				'decision' => array(
					'recommended_decision' => 'review',
					'confidence' => 0.0,
					'reasoning' => array('Decision engine unavailable'),
					'alternatives' => array()
				)
			);
		}
	}

	/**
	 * Extract manuscript content from submission files
	 */
	private function extractManuscriptContent($submission) {
		// This would extract text content from submission files
		// For now, return abstract as content sample
		return $submission->getLocalizedAbstract() ?: '';
	}

	/**
	 * Map OJS recommendation to SKZ format
	 */
	private function mapRecommendation($ojsRecommendation) {
		$mapping = array(
			SUBMISSION_REVIEWER_RECOMMENDATION_ACCEPT => 'accept',
			SUBMISSION_REVIEWER_RECOMMENDATION_PENDING_REVISIONS => 'minor_revisions',
			SUBMISSION_REVIEWER_RECOMMENDATION_RESUBMIT_HERE => 'major_revisions',
			SUBMISSION_REVIEWER_RECOMMENDATION_DECLINE => 'reject'
		);

		return $mapping[$ojsRecommendation] ?? 'review';
	}

	/**
	 * Assess review quality score
	 */
	private function assessReviewQuality($reviewAssignment) {
		// Basic quality assessment based on completion and timeliness
		$quality = 0.7; // Base quality
		
		if ($reviewAssignment->getDateCompleted()) {
			$quality += 0.2; // Completed reviews get bonus
		}
		
		$timeliness = $this->calculateTimeliness($reviewAssignment);
		$quality += $timeliness * 0.1; // Timely reviews get bonus

		return min(1.0, $quality);
	}

	/**
	 * Calculate review timeliness score
	 */
	private function calculateTimeliness($reviewAssignment) {
		if (!$reviewAssignment->getDateCompleted() || !$reviewAssignment->getDateDue()) {
			return 0.5; // Default score
		}

		$completed = strtotime($reviewAssignment->getDateCompleted());
		$due = strtotime($reviewAssignment->getDateDue());
		
		if ($completed <= $due) {
			return 1.0; // On time or early
		} else {
			$daysLate = ($completed - $due) / (24 * 3600);
			return max(0.1, 1.0 - ($daysLate / 14)); // Penalty for lateness
		}
	}

	/**
	 * Get review quality score for display
	 */
	private function getReviewQualityScore($reviewAssignment) {
		return $this->assessReviewQuality($reviewAssignment);
	}

	/**
	 * Extract technical comments (placeholder)
	 */
	private function extractTechnicalComments($reviewAssignment) {
		// Would extract technical comments from review text
		return array('Technical analysis needed');
	}

	/**
	 * Extract major issues (placeholder)
	 */
	private function extractMajorIssues($reviewAssignment) {
		// Would extract major issues from review text  
		return array('Methodology concerns');
	}

	/**
	 * Extract minor issues (placeholder)
	 */
	private function extractMinorIssues($reviewAssignment) {
		// Would extract minor issues from review text
		return array('Minor formatting issues');
	}

	/**
	 * @copydoc Plugin::getPluginPath()
	 */
	function getPluginPath() {
		return parent::getPluginPath() . DIRECTORY_SEPARATOR;
	}

	/**
	 * @copydoc Plugin::getTemplatePath()
	 */
	function getTemplatePath($inCore = false) {
		return parent::getTemplatePath($inCore) . 'templates' . DIRECTORY_SEPARATOR;
	}
}

?>