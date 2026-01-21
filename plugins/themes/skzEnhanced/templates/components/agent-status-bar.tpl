{**
 * templates/components/agent-status-bar.tpl
 *
 * Agent status bar component for header integration
 *}

<div class="skz-agent-status-bar" id="skzAgentStatusBar">
	<div class="skz-status-container">
		{if $skzAgentData}
			<div class="skz-status-summary">
				<div class="skz-status-indicator {if $skzAgentData.systemStatus == 'operational'}active{else}warning{/if}">
					<i class="fas fa-circle"></i>
				</div>
				<span class="skz-status-text">
					{if $skzAgentData.systemStatus == 'operational'}
						{translate key="plugins.themes.skzEnhanced.agentStatus.operational"}
					{else}
						{translate key="plugins.themes.skzEnhanced.agentStatus.warning"}
					{/if}
				</span>
				<span class="skz-status-count">{$skzAgentData.agents|@count} agents</span>
			</div>
			
			<div class="skz-agent-indicators">
				{foreach from=$skzAgentData.agents item=agent}
					<div class="skz-agent-indicator" 
						 data-agent-id="{$agent.id}"
						 data-agent-name="{$agent.name|escape}"
						 title="{$agent.name|escape} - {$agent.status|escape}">
						<i class="{$agent.icon} skz-agent-icon" style="color: {$agent.color}"></i>
						<div class="skz-agent-status {$agent.status}"></div>
					</div>
				{/foreach}
			</div>
			
			<div class="skz-quick-stats">
				<span class="skz-stat">
					<i class="fas fa-check-circle"></i>
					{$skzAgentData.successRate}%
				</span>
				<span class="skz-stat">
					<i class="fas fa-bolt"></i>
					{$skzAgentData.totalActions}
				</span>
			</div>
			
			<div class="skz-controls">
				<button class="skz-dashboard-btn" onclick="window.open('{url page="skzAgents" op="dashboard"}', '_blank')">
					<i class="fas fa-tachometer-alt"></i>
					Dashboard
				</button>
			</div>
		{else}
			<div class="skz-status-unavailable">
				<i class="fas fa-exclamation-triangle"></i>
				<span>SKZ Agents Unavailable</span>
			</div>
		{/if}
	</div>
</div>

{* Status bar toggle for mobile *}
<button class="skz-status-toggle" id="skzStatusToggle" onclick="toggleAgentStatusBar()">
	<i class="fas fa-robot"></i>
</button>

<script>
{literal}
function toggleAgentStatusBar() {
	const statusBar = document.getElementById('skzAgentStatusBar');
	const toggle = document.getElementById('skzStatusToggle');
	
	if (statusBar.classList.contains('collapsed')) {
		statusBar.classList.remove('collapsed');
		toggle.classList.remove('collapsed');
	} else {
		statusBar.classList.add('collapsed');
		toggle.classList.add('collapsed');
	}
}

// Auto-hide on mobile
if (window.innerWidth <= 768) {
	document.getElementById('skzAgentStatusBar').classList.add('collapsed');
	document.getElementById('skzStatusToggle').classList.add('collapsed');
}
{/literal}
</script>