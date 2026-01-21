{**
 * templates/components/submission-agent-status.tpl
 *
 * Agent status widget for submission workflow sidebar
 *}

<div class="skz-submission-agent-status" id="skzSubmissionStatus">
	<div class="skz-status-widget">
		<div class="skz-widget-header">
			<h4 class="skz-widget-title">
				<i class="fas fa-robot"></i>
				Agent Status
			</h4>
			<div class="skz-last-update">
				Updated: <span id="lastUpdateTime">{$submissionAgentStatus.lastAgentActivity|date_format:"%H:%M"}</span>
			</div>
		</div>
		
		{if $submissionAgentStatus}
			<div class="skz-widget-content">
				<div class="skz-current-stage">
					<label class="skz-label">Current Stage:</label>
					<span class="skz-stage-badge {$submissionAgentStatus.currentStage}">
						{$submissionAgentStatus.currentStage|replace:"_":" "|ucwords}
					</span>
				</div>
				
				<div class="skz-workflow-progress">
					<label class="skz-label">Workflow Progress:</label>
					<div class="skz-progress-container">
						<div class="skz-progress-bar">
							<div class="skz-progress-fill" style="width: {$submissionAgentStatus.workflowProgress}%"></div>
						</div>
						<span class="skz-progress-text">{$submissionAgentStatus.workflowProgress}%</span>
					</div>
				</div>
				
				<div class="skz-agent-activities">
					<label class="skz-label">Agent Activities:</label>
					<div class="skz-activity-summary">
						<div class="skz-activity-item">
							<i class="fas fa-check-circle text-success"></i>
							<span>Completed: {$submissionAgentStatus.completedActions}</span>
						</div>
						<div class="skz-activity-item">
							<i class="fas fa-clock text-warning"></i>
							<span>Pending: {$submissionAgentStatus.pendingActions}</span>
						</div>
					</div>
				</div>
				
				<div class="skz-active-agents-list">
					<label class="skz-label">Active Agents:</label>
					<div class="skz-agents-list">
						{foreach from=$submissionAgentStatus.activeAgents item=agentId}
							{foreach from=$skzAgentData.agents item=agent}
								{if $agent.id == $agentId}
									<div class="skz-active-agent">
										<i class="{$agent.icon}" style="color: {$agent.color}"></i>
										<span class="skz-agent-name">{$agent.name}</span>
										<div class="skz-agent-pulse"></div>
									</div>
								{/if}
							{/foreach}
						{/foreach}
					</div>
				</div>
				
				<div class="skz-recent-actions">
					<label class="skz-label">Recent Actions:</label>
					<div class="skz-actions-timeline">
						<div class="skz-timeline-item">
							<div class="skz-timeline-marker research_discovery"></div>
							<div class="skz-timeline-content">
								<span class="skz-action-text">Research Discovery analyzed INCI database</span>
								<span class="skz-action-time">5 min ago</span>
							</div>
						</div>
						<div class="skz-timeline-item">
							<div class="skz-timeline-marker content_quality"></div>
							<div class="skz-timeline-content">
								<span class="skz-action-text">Content Quality verified methodology</span>
								<span class="skz-action-time">12 min ago</span>
							</div>
						</div>
						<div class="skz-timeline-item">
							<div class="skz-timeline-marker editorial_orchestration"></div>
							<div class="skz-timeline-content">
								<span class="skz-action-text">Editorial Orchestration updated workflow</span>
								<span class="skz-action-time">20 min ago</span>
							</div>
						</div>
					</div>
				</div>
				
				<div class="skz-widget-actions">
					<button class="skz-widget-btn skz-btn-primary" onclick="refreshAgentStatus()">
						<i class="fas fa-sync-alt"></i>
						Refresh
					</button>
					<button class="skz-widget-btn skz-btn-secondary" onclick="viewFullReport()">
						<i class="fas fa-chart-line"></i>
						Full Report
					</button>
				</div>
			</div>
		{else}
			<div class="skz-widget-unavailable">
				<i class="fas fa-info-circle"></i>
				<p>No agent data available for this submission.</p>
			</div>
		{/if}
	</div>
</div>

<script>
{literal}
function refreshAgentStatus() {
	// Refresh agent status data
	const submissionId = {/literal}{$submissionId|default:0}{literal};
	console.log('Refreshing status for submission:', submissionId);
	
	// Show loading indicator
	const refreshBtn = event.target;
	const originalHTML = refreshBtn.innerHTML;
	refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
	refreshBtn.disabled = true;
	
	// Simulate API call
	setTimeout(() => {
		refreshBtn.innerHTML = originalHTML;
		refreshBtn.disabled = false;
		updateLastUpdateTime();
	}, 2000);
}

function viewFullReport() {
	// Open full agent report
	const submissionId = {/literal}{$submissionId|default:0}{literal};
	window.open(`/index.php/journal/skzAgents/report/${submissionId}`, '_blank');
}

function updateLastUpdateTime() {
	const now = new Date();
	const timeString = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
	document.getElementById('lastUpdateTime').textContent = timeString;
}

// Auto-refresh every 30 seconds if real-time updates are enabled
{/literal}{if $enableRealTimeUpdates}{literal}
setInterval(updateLastUpdateTime, 30000);
{/literal}{/if}{literal}
{/literal}
</script>