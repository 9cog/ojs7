<?php

/**
 * @file classes/workflow/EditorDecisionActionsManager.inc.php
 *
 * Copyright (c) 2014-2021 Simon Fraser University
 * Copyright (c) 2003-2021 John Willinsky
 * Distributed under the GNU GPL v3. For full terms see the file docs/COPYING.
 *
 * @class EditorDecisionActionsManager
 * @ingroup classes_workflow
 *
 * @brief Wrapper class for create and assign editor decisions actions to template manager.
 */

// Submission stage decision actions.
define('SUBMISSION_EDITOR_DECISION_EXTERNAL_REVIEW', 8);

// Submission and review stages decision actions.
define('SUBMISSION_EDITOR_DECISION_ACCEPT', 1);
define('SUBMISSION_EDITOR_DECISION_DECLINE', 4);

// Review stage decisions actions.
define('SUBMISSION_EDITOR_DECISION_PENDING_REVISIONS', 2);
define('SUBMISSION_EDITOR_DECISION_RESUBMIT', 3);
define('SUBMISSION_EDITOR_DECISION_NEW_ROUND', 16);

// Editorial stage decision actions.
define('SUBMISSION_EDITOR_DECISION_SEND_TO_PRODUCTION', 7);

import('lib.pkp.classes.workflow.PKPEditorDecisionActionsManager');

class EditorDecisionActionsManager extends PKPEditorDecisionActionsManager {

	/**
	 * Get decision actions labels.
	 * @param $context Context
	 * @param $stageId int
	 * @param $decisions array
	 * @return array
	 */
	public function getActionLabels($context, $submission, $stageId, $decisions) {
		$allDecisionsData =
			$this->_submissionStageDecisions($submission, $stageId) +
			$this->_externalReviewStageDecisions($context, $submission) +
			$this->_editorialStageDecisions();

		$actionLabels = array();
		foreach($decisions as $decision) {
			if ($allDecisionsData[$decision]['title']) {