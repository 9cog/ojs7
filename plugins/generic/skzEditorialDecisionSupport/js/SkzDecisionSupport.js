/**
 * SKZ Editorial Decision Support - Frontend JavaScript
 * Provides real-time decision support interface for editors
 */

class SKZDecisionSupport {
    constructor() {
        this.apiUrl = '/plugins/generic/skzEditorialDecisionSupport/api';
        this.initialized = false;
        this.currentSubmission = null;
        this.decisionRecommendation = null;
        
        this.init();
    }
    
    init() {
        if (this.initialized) return;
        
        console.log('Initializing SKZ Editorial Decision Support');
        
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupInterface());
        } else {
            this.setupInterface();
        }
        
        this.initialized = true;
    }
    
    setupInterface() {
        // Find workflow tabs and add decision support
        const workflowTabs = document.querySelector('.pkp_tabs');
        if (workflowTabs) {
            this.addDecisionSupportTab(workflowTabs);
        }
        
        // Add decision widgets to editorial decisions area
        const decisionsArea = document.querySelector('.pkp_editorial_decisions');
        if (decisionsArea) {
            this.addDecisionWidget(decisionsArea);
        }
        
        // Set up event listeners
        this.setupEventListeners();
    }
    
    addDecisionSupportTab(tabContainer) {
        // Create decision support tab
        const tabsList = tabContainer.querySelector('ul');
        if (!tabsList) return;
        
        const tabItem = document.createElement('li');
        tabItem.innerHTML = `
            <a href="#skz-decision-support" data-tab="skz-decision-support">
                <span class="fa fa-brain" aria-hidden="true"></span>
                Decision Support
            </a>
        `;
        
        // Add tab content panel
        const tabContent = document.createElement('div');
        tabContent.id = 'skz-decision-support';
        tabContent.className = 'pkp_tab_content';
        tabContent.innerHTML = this.getDecisionSupportHTML();
        
        tabsList.appendChild(tabItem);
        tabContainer.appendChild(tabContent);
        
        // Make tab clickable
        tabItem.querySelector('a').addEventListener('click', (e) => {
            e.preventDefault();
            this.showDecisionSupportTab();
        });
    }
    
    addDecisionWidget(decisionsArea) {
        const widget = document.createElement('div');
        widget.className = 'skz-decision-widget';
        widget.innerHTML = `
            <div class="skz-decision-summary">
                <h3><i class="fa fa-brain"></i> AI Decision Recommendation</h3>
                <div class="skz-recommendation-content">
                    <div class="loading">Loading recommendation...</div>
                </div>
            </div>
        `;
        
        decisionsArea.insertBefore(widget, decisionsArea.firstChild);
        
        // Load recommendation
        this.loadDecisionRecommendation();
    }
    
    getDecisionSupportHTML() {
        return `
            <div class="skz-decision-support-content">
                <div class="skz-section">
                    <h3>Manuscript Analysis</h3>
                    <div class="skz-manuscript-metrics">
                        <div class="metric-item">
                            <label>Quality Score:</label>
                            <div class="score-bar"><div class="score-fill" data-score="0"></div></div>
                            <span class="score-value">--</span>
                        </div>
                        <div class="metric-item">
                            <label>Novelty Score:</label>
                            <div class="score-bar"><div class="score-fill" data-score="0"></div></div>
                            <span class="score-value">--</span>
                        </div>
                        <div class="metric-item">
                            <label>Methodology Score:</label>
                            <div class="score-bar"><div class="score-fill" data-score="0"></div></div>
                            <span class="score-value">--</span>
                        </div>
                    </div>
                </div>
                
                <div class="skz-section">
                    <h3>Review Analysis</h3>
                    <div class="skz-review-consensus">
                        <div class="consensus-item">
                            <label>Reviewer Consensus:</label>
                            <span class="consensus-level">--</span>
                        </div>
                        <div class="consensus-item">
                            <label>Average Score:</label>
                            <span class="average-score">--</span>
                        </div>
                    </div>
                </div>
                
                <div class="skz-section">
                    <h3>AI Recommendation</h3>
                    <div class="skz-recommendation">
                        <div class="recommendation-main">
                            <div class="decision-type">--</div>
                            <div class="confidence-score">Confidence: <span>--</span></div>
                        </div>
                        <div class="recommendation-reasoning">
                            <h4>Reasoning:</h4>
                            <ul class="reasoning-list"></ul>
                        </div>
                        <div class="alternative-decisions">
                            <h4>Alternative Decisions:</h4>
                            <ul class="alternatives-list"></ul>
                        </div>
                    </div>
                </div>
                
                <div class="skz-section">
                    <h3>Decision History</h3>
                    <div class="skz-history">
                        <button class="btn btn-secondary" onclick="skzDecisionSupport.showDecisionHistory()">
                            View Decision History
                        </button>
                    </div>
                </div>
            </div>
        `;
    }
    
    setupEventListeners() {
        // Listen for form submissions to capture editorial decisions
        document.addEventListener('submit', (e) => {
            if (e.target.classList.contains('pkp_form_submit')) {
                this.onEditorialDecisionSubmit(e);
            }
        });
        
        // Refresh recommendation when review data changes
        document.addEventListener('reviewUpdated', () => {
            this.loadDecisionRecommendation();
        });
    }
    
    async loadDecisionRecommendation() {
        try {
            const submissionId = this.getSubmissionId();
            if (!submissionId) return;
            
            const response = await fetch(`${this.apiUrl}/recommendation/${submissionId}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.decisionRecommendation = data.decision;
                this.updateDecisionWidget(data.decision);
                this.updateDecisionSupportTab(data.decision);
            } else {
                this.showError('Failed to load decision recommendation');
            }
        } catch (error) {
            console.error('Error loading decision recommendation:', error);
            this.showServiceUnavailable();
        }
    }
    
    updateDecisionWidget(decision) {
        const widget = document.querySelector('.skz-recommendation-content');
        if (!widget) return;
        
        const confidencePercent = Math.round(decision.confidence * 100);
        const confidenceClass = decision.confidence > 0.8 ? 'high' : 
                               decision.confidence > 0.6 ? 'medium' : 'low';
        
        widget.innerHTML = `
            <div class="recommendation-summary ${confidenceClass}">
                <div class="decision-type">
                    <strong>Recommended:</strong> ${this.formatDecisionType(decision.recommended_decision)}
                </div>
                <div class="confidence-score">
                    Confidence: ${confidencePercent}%
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                    </div>
                </div>
                <div class="reasoning-preview">
                    ${decision.reasoning.slice(0, 2).join('; ')}
                </div>
                <button class="btn btn-sm btn-secondary" onclick="skzDecisionSupport.showDecisionSupportTab()">
                    View Full Analysis
                </button>
            </div>
        `;
    }
    
    updateDecisionSupportTab(decision) {
        // Update manuscript metrics
        const metrics = decision.manuscript_evaluation || {};
        this.updateMetrics('.skz-manuscript-metrics .metric-item', {
            'Quality Score': metrics.overall_score || 0,
            'Novelty Score': metrics.novelty_score || 0,
            'Methodology Score': metrics.methodology_score || 0
        });
        
        // Update review consensus
        const consensus = decision.review_consensus || {};
        this.updateElement('.consensus-level', consensus.consensus_level || 'Unknown');
        this.updateElement('.average-score', consensus.average_score || 'N/A');
        
        // Update recommendation
        this.updateElement('.decision-type', this.formatDecisionType(decision.recommended_decision));
        this.updateElement('.confidence-score span', Math.round(decision.confidence * 100) + '%');
        
        // Update reasoning
        const reasoningList = document.querySelector('.reasoning-list');
        if (reasoningList && decision.reasoning) {
            reasoningList.innerHTML = decision.reasoning
                .map(reason => `<li>${reason}</li>`)
                .join('');
        }
        
        // Update alternatives
        const alternativesList = document.querySelector('.alternatives-list');
        if (alternativesList && decision.alternative_decisions) {
            alternativesList.innerHTML = decision.alternative_decisions
                .map(([type, confidence]) => 
                    `<li>${this.formatDecisionType(type)} (${Math.round(confidence * 100)}%)</li>`
                )
                .join('');
        }
    }
    
    updateMetrics(selector, metrics) {
        const metricItems = document.querySelectorAll(selector);
        metricItems.forEach(item => {
            const label = item.querySelector('label').textContent.replace(':', '');
            const score = metrics[label];
            if (score !== undefined) {
                const scoreFill = item.querySelector('.score-fill');
                const scoreValue = item.querySelector('.score-value');
                
                scoreFill.style.width = (score * 100) + '%';
                scoreFill.dataset.score = score;
                scoreValue.textContent = score.toFixed(2);
            }
        });
    }
    
    updateElement(selector, content) {
        const element = document.querySelector(selector);
        if (element) {
            element.textContent = content;
        }
    }
    
    formatDecisionType(type) {
        const typeMap = {
            'accept': 'Accept',
            'revise': 'Request Revisions',
            'reject': 'Reject',
            'desk_reject': 'Desk Reject',
            'review': 'Send for Review'
        };
        return typeMap[type] || type;
    }
    
    showDecisionSupportTab() {
        // Activate the decision support tab
        const tab = document.querySelector('[data-tab="skz-decision-support"]');
        const tabContent = document.querySelector('#skz-decision-support');
        
        if (tab && tabContent) {
            // Deactivate other tabs
            document.querySelectorAll('.pkp_tabs a').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.pkp_tab_content').forEach(c => c.classList.remove('active'));
            
            // Activate this tab
            tab.classList.add('active');
            tabContent.classList.add('active');
            
            // Refresh data if needed
            if (!this.decisionRecommendation) {
                this.loadDecisionRecommendation();
            }
        }
    }
    
    showDecisionHistory() {
        const submissionId = this.getSubmissionId();
        const historyWindow = window.open(
            `/plugins/generic/skzEditorialDecisionSupport/history/${submissionId}`,
            'decision-history',
            'width=800,height=600,scrollbars=yes'
        );
    }
    
    onEditorialDecisionSubmit(event) {
        // Record the decision for learning purposes
        const formData = new FormData(event.target);
        const decision = formData.get('decision');
        
        if (decision) {
            this.recordDecision(decision);
        }
    }
    
    async recordDecision(decision) {
        try {
            const submissionId = this.getSubmissionId();
            if (!submissionId) return;
            
            await fetch(`${this.apiUrl}/record-decision`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    submission_id: submissionId,
                    decision: decision,
                    timestamp: new Date().toISOString()
                })
            });
        } catch (error) {
            console.error('Error recording decision:', error);
        }
    }
    
    getSubmissionId() {
        // Extract submission ID from URL or page data
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('submissionId') || 
               document.querySelector('[data-submission-id]')?.dataset.submissionId;
    }
    
    showError(message) {
        const widget = document.querySelector('.skz-recommendation-content');
        if (widget) {
            widget.innerHTML = `<div class="error-message">${message}</div>`;
        }
    }
    
    showServiceUnavailable() {
        const widget = document.querySelector('.skz-recommendation-content');
        if (widget) {
            widget.innerHTML = `
                <div class="service-unavailable">
                    <p>Decision support service is currently unavailable.</p>
                    <p>Editorial decisions can still be made normally.</p>
                </div>
            `;
        }
    }
}

// Initialize the decision support system
let skzDecisionSupport = null;

// Wait for page load
document.addEventListener('DOMContentLoaded', () => {
    skzDecisionSupport = new SKZDecisionSupport();
});

// Also initialize if script loads after DOM is ready
if (document.readyState !== 'loading') {
    skzDecisionSupport = new SKZDecisionSupport();
}