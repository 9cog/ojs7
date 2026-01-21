<?php

/**
 * @file pages/skzDashboard/SkzDashboardHandler.inc.php
 *
 * Copyright (c) 2014-2021 Simon Fraser University
 * Copyright (c) 2003-2021 John Willinsky
 * Distributed under the GNU GPL v3. For full terms see the file docs/COPYING.
 *
 * @class SkzDashboardHandler
 * @ingroup pages_skzDashboard
 *
 * @brief Handle SKZ visualization dashboard requests.
 */

import('lib.pkp.classes.handler.Handler');

class SkzDashboardHandler extends Handler {
	
	/**
	 * Constructor
	 */
	function __construct() {
		parent::__construct();
		$this->addRoleAssignment(
			array(ROLE_ID_SITE_ADMIN, ROLE_ID_MANAGER, ROLE_ID_SUB_EDITOR, ROLE_ID_ASSISTANT, ROLE_ID_REVIEWER, ROLE_ID_AUTHOR),
			array('index', 'workflow', 'agents', 'analytics')
		);
	}

	/**
	 * @copydoc PKPHandler::authorize()
	 */
	function authorize($request, &$args, $roleAssignments) {
		import('lib.pkp.classes.security.authorization.ContextRequiredPolicy');
		$this->addPolicy(new ContextRequiredPolicy($request));
		return parent::authorize($request, $args, $roleAssignments);
	}

	/**
	 * Display the main SKZ dashboard
	 * @param $args array
	 * @param $request PKPRequest
	 */
	function index($args, $request) {
		$this->setupTemplate($request);
		$templateMgr = TemplateManager::getManager($request);
		
		// Set page title
		$templateMgr->assign('pageTitle', 'plugins.generic.skz.dashboard.title');
		
		// Get the built React app path
		$dashboardPath = $request->getBaseUrl() . '/public/skz-dashboard/';
		$templateMgr->assign('dashboardPath', $dashboardPath);
		
		// Add the dashboard scripts and styles
		$templateMgr->addStyleSheet(
			'skzDashboardCSS',
			$dashboardPath . 'assets/index-BGF7fWty.css',
			array('contexts' => 'frontend')
		);
		
		$templateMgr->addJavaScript(
			'skzDashboardJS',
			$dashboardPath . 'assets/index-BtgDUcXP.js',
			array('contexts' => 'frontend', 'type' => 'module')
		);
		
		return $templateMgr->display('skzDashboard/index.tpl');
	}
	
	/**
	 * Display workflow visualization
	 * @param $args array
	 * @param $request PKPRequest
	 */
	function workflow($args, $request) {
		return $this->index($args, $request);
	}
	
	/**
	 * Display agents status
	 * @param $args array
	 * @param $request PKPRequest
	 */
	function agents($args, $request) {
		return $this->index($args, $request);
	}
	
	/**
	 * Display analytics dashboard
	 * @param $args array
	 * @param $request PKPRequest
	 */
	function analytics($args, $request) {
		return $this->index($args, $request);
	}
}

?>