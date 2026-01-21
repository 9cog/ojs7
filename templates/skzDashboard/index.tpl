{**
 * templates/skzDashboard/index.tpl
 *
 * Copyright (c) 2014-2021 Simon Fraser University
 * Copyright (c) 2003-2021 John Willinsky
 * Distributed under the GNU GPL v3. For full terms see the file docs/COPYING.
 *
 * SKZ Visualization Dashboard template
 *}
{strip}
{assign var="pageTitleTranslated" value="SKZ Autonomous Agents Dashboard"}
{include file="frontend/components/header.tpl" pageTitle=$pageTitleTranslated}
{/strip}

<div class="page page_skz_dashboard">
	<nav class="cmp_breadcrumbs" role="navigation" aria-label="You are here:">
		<ol>
			<li>
				<a href="{url page="index"}">
					{translate key="navigation.home"}
				</a>
				<span class="separator">{translate key="navigation.breadcrumbSeparator"}</span>
			</li>
			<li>
				<span aria-current="page">
					{translate key="SKZ Dashboard"}
				</span>
			</li>
		</ol>
	</nav>

	<div class="page-header">
		<h1>{translate key="SKZ Autonomous Agents Dashboard"}</h1>
		<div class="page-header-description">
			<p>Real-time visualization and monitoring of the 7 autonomous agents workflow system</p>
		</div>
	</div>

	{* React Dashboard Container *}
	<div id="skz-dashboard-root" class="skz-dashboard-container">
		<div class="loading-placeholder" style="text-align: center; padding: 60px 20px;">
			<div style="display: inline-block; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #f9f9f9;">
				<h3 style="margin: 0 0 10px 0; color: #333;">Loading SKZ Dashboard...</h3>
				<p style="margin: 0; color: #666;">Initializing autonomous agents visualization</p>
				<div style="margin-top: 15px;">
					<div style="width: 200px; height: 4px; background: #e0e0e0; border-radius: 2px; overflow: hidden; margin: 0 auto;">
						<div class="loading-bar" style="width: 0%; height: 100%; background: linear-gradient(90deg, #4f46e5, #7c3aed); animation: loading 2s ease-in-out infinite;"></div>
					</div>
				</div>
			</div>
		</div>
	</div>

	{* Dashboard initialization script *}
	<script type="module">
		{literal}
		// Ensure the React app mounts to our container
		window.addEventListener('DOMContentLoaded', function() {
			const container = document.getElementById('skz-dashboard-root');
			if (container) {
				// Remove loading placeholder when React app loads
				const observer = new MutationObserver(function(mutations) {
					mutations.forEach(function(mutation) {
						if (mutation.addedNodes.length > 0) {
							const loadingPlaceholder = container.querySelector('.loading-placeholder');
							if (loadingPlaceholder && container.children.length > 1) {
								loadingPlaceholder.style.display = 'none';
							}
						}
					});
				});
				observer.observe(container, { childList: true, subtree: true });
			}
		});
		{/literal}
	</script>

	<style>
		{literal}
		.skz-dashboard-container {
			min-height: 600px;
			width: 100%;
			border: 1px solid #e5e7eb;
			border-radius: 8px;
			background: white;
			overflow: hidden;
		}

		@keyframes loading {
			0% { width: 0%; }
			50% { width: 70%; }
			100% { width: 100%; }
		}

		/* Responsive adjustments */
		@media (max-width: 768px) {
			.skz-dashboard-container {
				border-radius: 0;
				border-left: none;
				border-right: none;
			}
		}
		{/literal}
	</style>
</div>

{include file="frontend/components/footer.tpl"}