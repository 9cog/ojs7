<?php

/**
 * @defgroup journal Journal
 * Extensions to the pkp-lib "context" concept to specialize it for use in OJS
 * in representing Journal objects and journal-specific concerns.
 */

/**
 * @file classes/journal/Journal.inc.php
 *
 * Copyright (c) 2014-2021 Simon Fraser University
 * Copyright (c) 2003-2021 John Willinsky
 * Distributed under the GNU GPL v3. For full terms see the file docs/COPYING.
 *
 * @class Journal
 * @ingroup journal
 * @see JournalDAO
 *
 * @brief Describes basic journal properties.
 */


define('PUBLISHING_MODE_OPEN', 0);
define('PUBLISHING_MODE_SUBSCRIPTION', 1);
define('PUBLISHING_MODE_NONE', 2);

import('lib.pkp.classes.context.Context');

class Journal extends Context {

	/**
	 * Get "localized" journal page title (if applicable).
	 * @return string|null
	 * @deprecated 3.3.0, use getLocalizedData() instead
	 */
	function getLocalizedPageHeaderTitle() {
		$titleArray = $this->getData('name');
		$title = null;

		foreach (array(AppLocale::getLocale(), AppLocale::getPrimaryLocale()) as $locale) {
			if (isset($titleArray[$locale])) return $titleArray[$locale];
		}
		return null;
	}

	/**
	 * Get "localized" journal page logo (if applicable).
	 * @return array|null
	 * @deprecated 3.3.0, use getLocalizedData() instead