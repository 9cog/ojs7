<?php

/**
 * @file classes/submission/reviewAssignment/ReviewAssignment.inc.php
 *
 * Copyright (c) 2014-2021 Simon Fraser University
 * Copyright (c) 2000-2021 John Willinsky
 * Distributed under the GNU GPL v3. For full terms see the file docs/COPYING.
 *
 * @class ReviewAssignment
 * @ingroup submission
 * @see ReviewAssignmentDAO
 *
 * @brief Describes review assignment properties.
 */

define('SUBMISSION_REVIEWER_RECOMMENDATION_ACCEPT', 1);
define('SUBMISSION_REVIEWER_RECOMMENDATION_PENDING_REVISIONS', 2);
define('SUBMISSION_REVIEWER_RECOMMENDATION_RESUBMIT_HERE', 3);
define('SUBMISSION_REVIEWER_RECOMMENDATION_RESUBMIT_ELSEWHERE', 4);
define('SUBMISSION_REVIEWER_RECOMMENDATION_DECLINE', 5);
define('SUBMISSION_REVIEWER_RECOMMENDATION_SEE_COMMENTS', 6);

define('SUBMISSION_REVIEWER_RATING_VERY_GOOD', 5);
define('SUBMISSION_REVIEWER_RATING_GOOD', 4);
define('SUBMISSION_REVIEWER_RATING_AVERAGE', 3);
define('SUBMISSION_REVIEWER_RATING_POOR', 2);
define('SUBMISSION_REVIEWER_RATING_VERY_POOR', 1);

define('SUBMISSION_REVIEW_METHOD_ANONYMOUS', 1);
define('SUBMISSION_REVIEW_METHOD_DOUBLEANONYMOUS', 2);
define('SUBMISSION_REVIEW_METHOD_OPEN', 3);

// A review is "unconsidered" when it is confirmed by an editor and then that
// confirmation is later revoked.
define('REVIEW_ASSIGNMENT_NOT_UNCONSIDERED', 0); // Has never been unconsidered
define('REVIEW_ASSIGNMENT_UNCONSIDERED', 1); // Has been unconsindered and is awaiting re-confirmation by an editor
define('REVIEW_ASSIGNMENT_UNCONSIDERED_READ', 2); // Has been reconfirmed by an editor

define('REVIEW_ASSIGNMENT_STATUS_AWAITING_RESPONSE', 0); // request has been sent but reviewer has not responded
define('REVIEW_ASSIGNMENT_STATUS_DECLINED', 1); // reviewer declined review request
define('REVIEW_ASSIGNMENT_STATUS_RESPONSE_OVERDUE', 4); // review not responded within due date
define('REVIEW_ASSIGNMENT_STATUS_ACCEPTED', 5); // reviewer has agreed to the review
define('REVIEW_ASSIGNMENT_STATUS_REVIEW_OVERDUE', 6); // review not submitted within due date
define('REVIEW_ASSIGNMENT_STATUS_RECEIVED', 7); // review has been submitted
define('REVIEW_ASSIGNMENT_STATUS_COMPLETE', 8); // review has been confirmed by an editor
define('REVIEW_ASSIGNMENT_STATUS_THANKED', 9); // reviewer has been thanked
define('REVIEW_ASSIGNMENT_STATUS_CANCELLED', 10); // reviewer cancelled review request

class ReviewAssignment extends DataObject {

	//
	// Get/set methods
	//

	/**
	 * Get ID of review assignment's submission.
	 * @return int
	 */
	function getSubmissionId() {
		return $this->getData('submissionId');
	}

	/**
	 * Set ID of review assignment's submission
	 * @param $submissionId int
	 */
	function setSubmissionId($submissionId) {
		$this->setData('submissionId', $submissionId);
	}

	/**
	 * Get ID of reviewer.
	 * @return int
	 */
	function getReviewerId() {
		return $this->getData('reviewerId');
	}

	/**