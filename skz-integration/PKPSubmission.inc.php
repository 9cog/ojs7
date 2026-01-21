<?php

/**
 * @defgroup submission Submission
 * The abstract concept of a submission is implemented here, and extended
 * in each application with the specifics of that content model, i.e.
 * Articles in OJS, Papers in OCS, and Monographs in OMP.
 */

/**
 * @file classes/submission/PKPSubmission.inc.php
 *
 * Copyright (c) 2014-2021 Simon Fraser University
 * Copyright (c) 2000-2021 John Willinsky
 * Distributed under the GNU GPL v3. For full terms see the file docs/COPYING.
 *
 * @class PKPSubmission
 * @ingroup submission
 * @see PKPSubmissionDAO
 *
 * @brief The Submission class implements the abstract data model of a
 * scholarly submission.
 */

// Submission status constants
define('STATUS_QUEUED', 1);
define('STATUS_PUBLISHED', 3);
define('STATUS_DECLINED', 4);
define('STATUS_SCHEDULED', 5);

// License settings (internal use only)
define ('PERMISSIONS_FIELD_LICENSE_URL', 1);
define ('PERMISSIONS_FIELD_COPYRIGHT_HOLDER', 2);
define ('PERMISSIONS_FIELD_COPYRIGHT_YEAR', 3);

abstract class PKPSubmission extends DataObject {
	/**
	 * Constructor.
	 */
	function __construct() {
		// Switch on meta-data adapter support.
		$this->setHasLoadableAdapters(true);

		parent::__construct();
	}

	/**
	 * Return the "best" article ID -- If a urlPath is set,
	 * use it; otherwise use the internal article Id.
	 * @return string
	 * @deprecated 3.2.0.0
	 */
	function getBestId() {
		$currentPublication = $this->getCurrentPublication();
		if (!$currentPublication) return $this->getId();
		if ($currentPublication->getData('urlPath')) {
			return $currentPublication->getData('urlPath');
		}
		return $this->getId();
	}

	/**
	 * Get the current publication
	 *
	 * Uses the `currentPublicationId` to get the current
	 * Publication object from the submission's list of
	 * publications.
	 *
	 * @return Publication|null
	 */
	public function getCurrentPublication() {
		$publicationId = $this->getData('currentPublicationId');
		$publications = $this->getData('publications');
		if (!$publicationId || empty($publications)) {
			return null;
		}
		foreach ($publications as $publication) {
			if ($publication->getId() === $publicationId) {
				return $publication;
			}
		}
	}

	/**
	 * Get the latest publication
	 *
	 * Returns the most recently created publication by ID
	 *
	 * @return Publication|null
	 */
	public function getLatestPublication() {
		$publications = $this->getData('publications');
		if (empty($publications)) {
			return null;
		}
		return array_reduce($publications, function($a, $b) {
			return $a && $a->getId() > $b->getId() ? $a : $b;
		});
	}
