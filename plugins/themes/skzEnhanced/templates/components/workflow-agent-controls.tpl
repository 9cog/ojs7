{**
 * templates/components/workflow-agent-controls.tpl
 *
 * Agent controls for workflow integration
 *}

<div class="skz-workflow-agent-controls" id="skzWorkflowControls">
	<div class="skz-controls-header">
		<h3 class="skz-controls-title">
			<i class="fas fa-robot"></i>
			Autonomous Agents
		</h3>
		<button class="skz-controls-toggle" onclick="toggleWorkflowControls()" id="controlsToggle">
			<i class="fas fa-chevron-down"></i>
		</button>
	</div>
	
	<div class="skz-controls-content" id="controlsContent">
		{if $skzAgentData && $submissionId}
			<div class="skz-workflow-progress">
				<div class="skz-progress-header">
					<span class="skz-progress-label">AI Processing Progress</span>
					<span class="skz-progress-percentage" id="workflowProgress">65%</span>
				</div>
				<div class="skz-progress-bar">
					<div class="skz-progress-fill" style="width: 65%"></div>
				</div>
			</div>
			
			<div class="skz-active-agents">
				<h4 class="skz-section-title">Active Agents</h4>
				<div class="skz-agent-grid">
					{foreach from=$skzAgentData.agents item=agent}
						<div class="skz-agent-card {$agent.status}" data-agent-id="{$agent.id}">
							<div class="skz-agent-header">
								<i class="{$agent.icon}" style="color: {$agent.color}"></i>
								<span class="skz-agent-name">{$agent.name}</span>
								<div class="skz-agent-status-indicator {$agent.status}"></div>
							</div>
							<div class="skz-agent-actions">
								<button class="skz-action-btn skz-btn-view" 
										onclick="viewAgentDetails('{$agent.id}')"
										title="{translate key="plugins.themes.skzEnhanced.agent.viewDetails"}">
									<i class="fas fa-eye"></i>
								</button>
								<button class="skz-action-btn skz-btn-pause" 
										onclick="toggleAgent('{$agent.id}')"
										title="{translate key="plugins.themes.skzEnhanced.agent.pauseAgent"}">
									<i class="fas fa-pause"></i>
								</button>
								<button class="skz-action-btn skz-btn-config" 
										onclick="configureAgent('{$agent.id}')"
										title="{translate key="plugins.themes.skzEnhanced.agent.configure"}">
									<i class="fas fa-cog"></i>
								</button>
							</div>
						</div>
					{/foreach}
				</div>
			</div>
			
			<div class="skz-recommendations" id="agentRecommendations">
				<h4 class="skz-section-title">
					<i class="fas fa-lightbulb"></i>
					{translate key="plugins.themes.skzEnhanced.workflow.agentRecommendations"}
				</h4>
				<div class="skz-recommendations-content">
					<div class="skz-recommendation">
						<div class="skz-rec-icon">
							<i class="fas fa-check-circle text-success"></i>
						</div>
						<div class="skz-rec-content">
							<strong>Quality Review Complete</strong>
							<p>Content Quality Agent has validated manuscript formatting and structure.</p>
						</div>
					</div>
					<div class="skz-recommendation">
						<div class="skz-rec-icon">
							<i class="fas fa-clock text-warning"></i>
						</div>
						<div class="skz-rec-content">
							<strong>Reviewer Assignment Pending</strong>
							<p>Review Coordination Agent suggests 3 potential reviewers based on expertise matching.</p>
						</div>
					</div>
				</div>
			</div>
			
			<div class="skz-workflow-actions">
				<button class="skz-action-button skz-btn-primary" onclick="runAgentAnalysis()">
					<i class="fas fa-brain"></i>
					Run AI Analysis
				</button>
				<button class="skz-action-button skz-btn-secondary" onclick="openDashboard()">
					<i class="fas fa-tachometer-alt"></i>
					View Dashboard
				</button>
			</div>
		{else}
			<div class="skz-controls-unavailable">
				<i class="fas fa-exclamation-triangle"></i>
				<p>Agent controls are not available for this submission.</p>
			</div>
		{/if}
	</div>
</div>

<script>
{literal}
function toggleWorkflowControls() {
	const content = document.getElementById('controlsContent');
	const toggle = document.getElementById('controlsToggle');
	
	if (content.style.display === 'none') {
		content.style.display = 'block';
		toggle.innerHTML = '<i class="fas fa-chevron-down"></i>';
	} else {
		content.style.display = 'none';
		toggle.innerHTML = '<i class="fas fa-chevron-right"></i>';
	}
}

function viewAgentDetails(agentId) {
	// Open agent details modal or page
	console.log('Viewing details for agent:', agentId);
	// Implementation would open a modal or navigate to agent details
}

function toggleAgent(agentId) {
	// Toggle agent pause/resume
	console.log('Toggling agent:', agentId);
	// Implementation would call agent API to pause/resume
}

function configureAgent(agentId) {
	// Open agent configuration
	console.log('Configuring agent:', agentId);
	// Implementation would open agent configuration interface
}

function runAgentAnalysis() {
	// Trigger agent analysis for current submission
	console.log('Running agent analysis');
	// Implementation would call agent API to run analysis
}

function openDashboard() {
	window.open('/index.php/journal/skzAgents/dashboard', '_blank');
}
{/literal}
</script>