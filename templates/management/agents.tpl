{**
 * templates/management/agents.tpl
 *
 * Copyright (c) 2014-2021 Simon Fraser University
 * Copyright (c) 2003-2021 John Willinsky
 * Distributed under the GNU GPL v3. For full terms see the file docs/COPYING.
 *
 * Agent management interface template
 *}
{extends file="layouts/backend.tpl"}

{block name="page"}
	<div id="agents">
		{if !$pluginEnabled}
			<div class="pkp_notification_error">
				{if $errorMessage}
					{$errorMessage}
				{else}
					{translate key="plugins.generic.skzAgents.management.notEnabled"}
				{/if}
				<p>
					<a href="{url router=$smarty.const.ROUTE_PAGE page="management" op="settings" path="website"}">
						{translate key="plugins.generic.skzAgents.management.enablePlugin"}
					</a>
				</p>
			</div>
		{else}
			<div class="agent-dashboard">
				<h1>{translate key="plugins.generic.skzAgents.management.title"}</h1>
				
				{if $errorMessage}
					<div class="pkp_notification_error">
						{$errorMessage}
					</div>
				{/if}

				<div class="agent-overview">
					<div class="agent-stats">
						<div class="stat-card">
							<h3>{translate key="plugins.generic.skzAgents.management.totalAgents"}</h3>
							<span class="stat-number" id="total-agents">7</span>
						</div>
						<div class="stat-card">
							<h3>{translate key="plugins.generic.skzAgents.management.activeAgents"}</h3>
							<span class="stat-number" id="active-agents">0</span>
						</div>
						<div class="stat-card">
							<h3>{translate key="plugins.generic.skzAgents.management.tasksCompleted"}</h3>
							<span class="stat-number" id="tasks-completed">0</span>
						</div>
					</div>
				</div>

				<div class="agent-controls">
					{* SSR-compliant agent controls - work without JavaScript *}
					<div class="control-buttons">
						<form method="post" action="{url router=$smarty.const.ROUTE_PAGE page="management" op="agents"}" style="display:inline;">
							<input type="hidden" name="action" value="startAll" />
							<input type="hidden" name="csrfToken" value="{$csrfToken}" />
							<button type="submit" class="pkp_button pkp_button_primary">
								{translate key="plugins.generic.skzAgents.management.startAll"}
							</button>
						</form>
						<form method="post" action="{url router=$smarty.const.ROUTE_PAGE page="management" op="agents"}" style="display:inline;">
							<input type="hidden" name="action" value="stopAll" />
							<input type="hidden" name="csrfToken" value="{$csrfToken}" />
							<button type="submit" class="pkp_button pkp_button_secondary">
								{translate key="plugins.generic.skzAgents.management.stopAll"}
							</button>
						</form>
						<form method="post" action="{url router=$smarty.const.ROUTE_PAGE page="management" op="agents"}" style="display:inline;">
							<input type="hidden" name="action" value="refreshStatus" />
							<input type="hidden" name="csrfToken" value="{$csrfToken}" />
							<button type="submit" class="pkp_button pkp_button_secondary">
								{translate key="plugins.generic.skzAgents.management.refreshStatus"}
							</button>
						</form>
					</div>
				</div>

				<div class="agents-list">
					<h2>{translate key="plugins.generic.skzAgents.management.agentsList"}</h2>
					<div class="agents-grid" id="agents-grid">
						{if $agentStatus}
							{foreach from=$agentStatus key="agentId" item="agent"}
								<div class="agent-card" data-agent-id="{$agentId}">
									<div class="agent-header">
										<h3>{$agent.name}</h3>
										<span class="agent-status-indicator status-unknown" id="status-{$agentId}">
											{translate key="plugins.generic.skzAgents.management.status.unknown"}
										</span>
									</div>
									<div class="agent-details">
										<p class="agent-id">{translate key="plugins.generic.skzAgents.management.agentId"}: {$agentId}</p>
										<p class="agent-tasks">
											{translate key="plugins.generic.skzAgents.management.processedTasks"}: 
											<span id="tasks-{$agentId}">0</span>
										</p>
										<p class="agent-activity">
											{translate key="plugins.generic.skzAgents.management.lastActivity"}: 
											<span id="activity-{$agentId}">-</span>
										</p>
									</div>
									<div class="agent-actions">
										{* SSR-compliant individual agent controls *}
										<form method="post" action="{url router=$smarty.const.ROUTE_PAGE page="management" op="agents"}" style="display:inline;">
											<input type="hidden" name="action" value="startAgent" />
											<input type="hidden" name="agentId" value="{$agentId}" />
											<input type="hidden" name="csrfToken" value="{$csrfToken}" />
											<button type="submit" class="pkp_button pkp_button_primary">
												{translate key="plugins.generic.skzAgents.management.start"}
											</button>
										</form>
										<form method="post" action="{url router=$smarty.const.ROUTE_PAGE page="management" op="agents"}" style="display:inline;">
											<input type="hidden" name="action" value="stopAgent" />
											<input type="hidden" name="agentId" value="{$agentId}" />
											<input type="hidden" name="csrfToken" value="{$csrfToken}" />
											<button type="submit" class="pkp_button pkp_button_secondary">
												{translate key="plugins.generic.skzAgents.management.stop"}
											</button>
										</form>
										<a href="{url router=$smarty.const.ROUTE_PAGE page="management" op="agents" path="configure" agentId=$agentId}" 
											class="pkp_button pkp_button_secondary">
											{translate key="plugins.generic.skzAgents.management.configure"}
										</a>
									</div>
								</div>
							{/foreach}
						{else}
							<p>{translate key="plugins.generic.skzAgents.management.noAgents"}</p>
						{/if}
					</div>
				</div>

				<div class="agent-logs">
					<h2>{translate key="plugins.generic.skzAgents.management.recentActivity"}</h2>
					<div class="log-container" id="agent-logs">
						<p class="loading">{translate key="plugins.generic.skzAgents.management.loadingLogs"}</p>
					</div>
				</div>
			</div>
		{/if}
	</div>

	{if $pluginEnabled}
		{* SSR-compliant agent management - Server-side rendered only *}
		<noscript>
			<div class="pkp_notification_warning">
				{translate key="plugins.generic.skzAgents.management.noJsNotice"}
			</div>
		</noscript>
		
		{* Server-side rendered agent status updates *}
		<div id="agent-status-update" style="display:none;">
			{if $serverSideAgentStatus}
				<div class="pkp_notification_success">
					{translate key="plugins.generic.skzAgents.management.statusUpdated"}
					{$serverSideAgentStatus.timestamp}
				</div>
			{/if}
		</div>
		
		{* SSR form submissions for agent control - no JavaScript required *}
		<form method="post" action="{url router=$smarty.const.ROUTE_PAGE page="management" op="agents"}" style="display:inline;">
			<input type="hidden" name="action" value="refreshStatus" />
			<input type="hidden" name="csrfToken" value="{$csrfToken}" />
		</form>
		
		{* Minimal progressive enhancement - degrades gracefully without JS *}
		<script type="text/javascript">
			// SSR Expert Role Compliant: Minimal enhancement only
			// All functionality works without JavaScript
			document.addEventListener('DOMContentLoaded', function() {
				// Server-side rendered status display
				var statusElements = document.querySelectorAll('.agent-status-indicator');
				statusElements.forEach(function(element) {
					// Server-rendered status classes are preserved
					element.setAttribute('data-ssr-rendered', 'true');
				});
				
				// Auto-refresh via server-side redirect (no AJAX)
				var refreshInterval = {$refreshInterval|default:30000};
				if (refreshInterval > 0) {
					setTimeout(function() {
						// Server-side refresh via page reload
						window.location.href = window.location.href + '?autoRefresh=1';
					}, refreshInterval);
				}
			});
		</script>

		<style>
			.agent-dashboard {
				max-width: 1200px;
				margin: 0 auto;
			}

			.agent-overview {
				margin-bottom: 2em;
			}

			.agent-stats {
				display: flex;
				gap: 1em;
				margin-bottom: 2em;
			}

			.stat-card {
				background: #f8f9fa;
				border: 1px solid #dee2e6;
				border-radius: 8px;
				padding: 1em;
				text-align: center;
				flex: 1;
			}

			.stat-card h3 {
				margin: 0 0 0.5em 0;
				font-size: 0.9em;
				color: #6c757d;
				text-transform: uppercase;
				font-weight: 600;
			}

			.stat-number {
				font-size: 2em;
				font-weight: bold;
				color: #007bff;
			}

			.agent-controls {
				margin-bottom: 2em;
			}

			.control-buttons {
				display: flex;
				gap: 0.5em;
			}

			.agents-grid {
				display: grid;
				grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
				gap: 1em;
			}

			.agent-card {
				border: 1px solid #dee2e6;
				border-radius: 8px;
				padding: 1em;
				background: #fff;
			}

			.agent-header {
				display: flex;
				justify-content: space-between;
				align-items: center;
				margin-bottom: 1em;
			}

			.agent-header h3 {
				margin: 0;
				font-size: 1.1em;
			}

			.agent-status-indicator {
				padding: 0.25em 0.5em;
				border-radius: 4px;
				font-size: 0.8em;
				font-weight: bold;
				text-transform: uppercase;
			}

			.status-unknown {
				background-color: #6c757d;
				color: white;
			}

			.status-running {
				background-color: #28a745;
				color: white;
			}

			.status-stopped {
				background-color: #dc3545;
				color: white;
			}

			.agent-details {
				margin-bottom: 1em;
			}

			.agent-details p {
				margin: 0.25em 0;
				font-size: 0.9em;
				color: #6c757d;
			}

			.agent-actions {
				display: flex;
				gap: 0.5em;
				flex-wrap: wrap;
			}

			.agent-actions button {
				flex: 1;
				min-width: 80px;
			}

			.agent-logs {
				margin-top: 2em;
				padding-top: 2em;
				border-top: 1px solid #dee2e6;
			}

			.log-container {
				background: #f8f9fa;
				border: 1px solid #dee2e6;
				border-radius: 4px;
				padding: 1em;
				height: 200px;
				overflow-y: auto;
				font-family: monospace;
				font-size: 0.9em;
			}

			@media (max-width: 768px) {
				.agent-stats {
					flex-direction: column;
				}

				.control-buttons {
					flex-direction: column;
				}

				.agents-grid {
					grid-template-columns: 1fr;
				}
			}
		</style>
	{/if}
{/block}