{**
 * plugins/generic/skzAgents/templates/agentConfig.tpl
 *
 * Agent configuration template for SKZ Agents
 *}
{extends file="layouts/backend.tpl"}

{block name="page"}
	<div class="app__page">
		<h1>{$pageTitle}</h1>
		
		<div class="agent-configuration">
			<p class="agent-info">
				{translate key="plugins.generic.skzAgents.management.configureDescription"} <strong>{$agentId}</strong>
			</p>
			
			<div class="configuration-placeholder">
				<h2>{translate key="plugins.generic.skzAgents.management.configurationOptions"}</h2>
				<p>{translate key="plugins.generic.skzAgents.management.configurationPlaceholder"}</p>
				
				{* This is a placeholder for future agent-specific configuration forms *}
				<div class="configuration-form">
					<form method="post" action="{url component="plugins.generic.skzAgents.controllers.SKZAgentManagementHandler" op="saveConfig"}">
						{csrf}
						<input type="hidden" name="agentId" value="{$agentId|escape}" />
						
						<fieldset>
							<legend>{translate key="plugins.generic.skzAgents.management.generalSettings"}</legend>
							
							<div class="form-group">
								<label for="enabled">{translate key="plugins.generic.skzAgents.management.enabled"}</label>
								<input type="checkbox" id="enabled" name="enabled" checked />
							</div>
							
							<div class="form-group">
								<label for="priority">{translate key="plugins.generic.skzAgents.management.priority"}</label>
								<select id="priority" name="priority">
									<option value="low">{translate key="plugins.generic.skzAgents.management.priorityLow"}</option>
									<option value="normal" selected>{translate key="plugins.generic.skzAgents.management.priorityNormal"}</option>
									<option value="high">{translate key="plugins.generic.skzAgents.management.priorityHigh"}</option>
								</select>
							</div>
							
							<div class="form-group">
								<label for="maxConcurrent">{translate key="plugins.generic.skzAgents.management.maxConcurrentTasks"}</label>
								<input type="number" id="maxConcurrent" name="maxConcurrent" value="5" min="1" max="50" />
							</div>
						</fieldset>
						
						<div class="form-actions">
							<button type="submit" class="pkp_button pkp_button_primary">
								{translate key="common.save"}
							</button>
							<a href="{url page="management" op="settings" path="agents"}" class="pkp_button pkp_button_secondary">
								{translate key="common.back"}
							</a>
						</div>
					</form>
				</div>
			</div>
		</div>
	</div>
	
	<style>
		.agent-configuration {
			max-width: 800px;
			margin: 0 auto;
		}
		
		.configuration-placeholder {
			background: #f8f9fa;
			border: 1px solid #dee2e6;
			border-radius: 8px;
			padding: 2em;
			margin: 1em 0;
		}
		
		.configuration-form fieldset {
			border: 1px solid #dee2e6;
			border-radius: 4px;
			padding: 1em;
			margin: 1em 0;
		}
		
		.configuration-form legend {
			font-weight: bold;
			padding: 0 0.5em;
		}
		
		.form-group {
			margin: 1em 0;
		}
		
		.form-group label {
			display: block;
			margin-bottom: 0.5em;
			font-weight: 600;
		}
		
		.form-group input,
		.form-group select {
			width: 100%;
			max-width: 300px;
			padding: 0.5em;
			border: 1px solid #ced4da;
			border-radius: 4px;
		}
		
		.form-actions {
			margin-top: 2em;
			display: flex;
			gap: 1em;
		}
	</style>
{/block}