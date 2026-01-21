{**
 * plugins/generic/skzAgents/templates/settings.tpl
 *
 * SKZ Agents Plugin settings form template
 *}
<script>
	$(function() {ldelim}
		// @{$smarty.ldelim}jsliteral{$smarty.rdelim}
		$('#skzAgentsSettings').pkpHandler('$.pkp.controllers.form.AjaxFormHandler');
		// {$smarty.ldelim}/jsliteral{$smarty.rdelim}
	{rdelim});
</script>

<form class="pkp_form" id="skzAgentsSettings" method="post" action="{url router=$smarty.const.ROUTE_COMPONENT op="manage" category="generic" plugin=$pluginName verb="settings" save=true}">
	{csrf}
	{include file="controllers/notification/inPlaceNotification.tpl" notificationId="skzAgentsSettingsNotification"}

	<div id="description">{translate key="plugins.generic.skzAgents.description"}</div>

	{fbvFormArea id="skzAgentsApiSettings" title="plugins.generic.skzAgents.manager.apiSettings"}
		{fbvFormSection}
			{fbvElement type="text" id="agentBaseUrl" value=$agentBaseUrl label="plugins.generic.skzAgents.settings.agentBaseUrl" maxlength="255" size=$fbvStyles.size.LARGE required=true}
			<span class="description">{translate key="plugins.generic.skzAgents.settings.agentBaseUrl.description"}</span>
		{/fbvFormSection}
		{fbvFormSection}
			{fbvElement type="text" id="apiKey" value=$apiKey label="plugins.generic.skzAgents.settings.apiKey" maxlength="255" size=$fbvStyles.size.LARGE required=true}
			<span class="description">{translate key="plugins.generic.skzAgents.settings.apiKey.description"}</span>
		{/fbvFormSection}
		{fbvFormSection}
			{fbvElement type="text" id="timeout" value=$timeout label="plugins.generic.skzAgents.settings.timeout" maxlength="10" size=$fbvStyles.size.SMALL}
			<span class="description">{translate key="plugins.generic.skzAgents.settings.timeout.description"}</span>
		{/fbvFormSection}
	{/fbvFormArea}

	{fbvFormArea id="skzAgentsFeatureSettings" title="plugins.generic.skzAgents.manager.agentSettings"}
		{fbvFormSection list=true}
			{fbvElement type="checkbox" id="enableAutoSubmissionProcessing" checked=$enableAutoSubmissionProcessing label="plugins.generic.skzAgents.settings.enableAutoSubmissionProcessing"}
			{fbvElement type="checkbox" id="enableAutoReviewerAssignment" checked=$enableAutoReviewerAssignment label="plugins.generic.skzAgents.settings.enableAutoReviewerAssignment"}
			{fbvElement type="checkbox" id="enableAutoQualityChecks" checked=$enableAutoQualityChecks label="plugins.generic.skzAgents.settings.enableAutoQualityChecks"}
			{fbvElement type="checkbox" id="enablePerformanceMonitoring" checked=$enablePerformanceMonitoring label="plugins.generic.skzAgents.settings.enablePerformanceMonitoring"}
		{/fbvFormSection}
	{/fbvFormArea}

	{fbvFormArea id="skzAgentsPerformanceSettings" title="plugins.generic.skzAgents.manager.performanceSettings"}
		{fbvFormSection}
			{fbvElement type="text" id="maxConcurrentRequests" value=$maxConcurrentRequests label="plugins.generic.skzAgents.settings.maxConcurrentRequests" maxlength="10" size=$fbvStyles.size.SMALL}
		{/fbvFormSection}
		{fbvFormSection}
			{fbvElement type="text" id="cacheTtl" value=$cacheTtl label="plugins.generic.skzAgents.settings.cacheTtl" maxlength="10" size=$fbvStyles.size.SMALL}
		{/fbvFormSection}
	{/fbvFormArea}

	{if $agentStatus}
		{fbvFormArea id="skzAgentsStatus" title="plugins.generic.skzAgents.dashboard.status"}
			{if $agentStatus.status == 'error'}
				<div class="notification notification-warning">
					{translate key="plugins.generic.skzAgents.errors.connectionFailed"}: {$agentStatus.message}
				</div>
			{elseif $agentStatus.status == 'disabled'}
				<div class="notification notification-info">
					{translate key="plugins.generic.skzAgents.errors.pluginDisabled"}
				</div>
			{else}
				<div class="notification notification-success">
					Connected to agent framework
				</div>
				{if $availableAgents}
					<h4>Available Agents:</h4>
					<ul>
						{foreach from=$availableAgents key=agentKey item=agent}
							<li><strong>{$agent.name}</strong> - {$agent.description}</li>
						{/foreach}
					</ul>
				{/if}
			{/if}
		{/fbvFormArea}
	{/if}

	{fbvFormButtons}
</form>