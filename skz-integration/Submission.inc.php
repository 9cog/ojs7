<?php

/**
 * @defgroup submission Submission
 * Articles, OMP's extension of the generic Submission class in lib-pkp, are
 * implemented here.
 */

/**
 * @file classes/submission/Submission.inc.php
 *
 * Copyright (c) 2014-2021 Simon Fraser University
 * Copyright (c) 2003-2021 John Willinsky
 * Distributed under the GNU GPL v3. For full terms see the file docs/COPYING.
 *
 * @class Submission
 * @ingroup submission
 * @see SubmissionDAO
 *
 * @brief Article class.
 */

// Author display in ToC
define ('AUTHOR_TOC_DEFAULT', 0);
define ('AUTHOR_TOC_HIDE', 1);
define ('AUTHOR_TOC_SHOW', 2);

// Article access constants -- see Publication::getData('accessStatus')
define('ARTICLE_ACCESS_ISSUE_DEFAULT', 0);
define('ARTICLE_ACCESS_OPEN', 1);

import('lib.pkp.classes.submission.PKPSubmission');

class Submission extends PKPSubmission {

	//
	// Get/set methods
	//

	/**
	 * Get the value of a license field from the containing context.
	 * @param $locale string Locale code
	 * @param $field PERMISSIONS_FIELD_...
	 * @param $publication Publication
	 * @return string|array|null
	 */
	function _getContextLicenseFieldValue($locale, $field, $publication = null) {
		$context = Services::get('context')->get($this->getData('contextId'));
		$fieldValue = null; // Scrutinizer
		switch ($field) {
			case PERMISSIONS_FIELD_LICENSE_URL:
				$fieldValue = $context->getData('licenseUrl');
				break;
			case PERMISSIONS_FIELD_COPYRIGHT_HOLDER:
				switch($context->getData('copyrightHolderType')) {
					case 'author':
						$fieldValue = array($context->getPrimaryLocale() => $this->getAuthorString());
						break;
					case 'context':
					case null:
						$fieldValue = $context->getName(null);
						break;
					default:
						$fieldValue = $context->getData('copyrightHolderOther');
						break;
				}
				break;
			case PERMISSIONS_FIELD_COPYRIGHT_YEAR:
				// Default copyright year to current year
				$fieldValue = date('Y');

				// Override based on context settings
				if (!$publication) {
					$publication = $this->getCurrentPublication();
				}
				if ($publication) {
					switch($context->getData('copyrightYearBasis')) {
						case 'submission':
							// override to the submission's year if published as you go
							$fieldValue = date('Y', strtotime($publication->getData('datePublished')));