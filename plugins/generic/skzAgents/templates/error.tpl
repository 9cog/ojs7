{**
 * plugins/generic/skzAgents/templates/error.tpl
 *
 * Error display template for SKZ Agents
 *}
{extends file="layouts/backend.tpl"}

{block name="page"}
	<div class="app__page">
		<h1>{translate key="plugins.generic.skzAgents.displayName"}</h1>
		
		{if $errorMessage}
			<div class="pkp_notification_error">
				{$errorMessage}
			</div>
		{else}
			<div class="pkp_notification_error">
				{translate key="plugins.generic.skzAgents.errors.unknown"}
			</div>
		{/if}
		
		<p>
			<a href="{url page="management" op="settings"}" class="pkp_button pkp_button_primary">
				{translate key="navigation.settings"}
			</a>
		</p>
	</div>
{/block}