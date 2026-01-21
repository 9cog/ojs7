{**
 * templates/components/agent-notifications.tpl
 *
 * Real-time agent notification system
 *}

<div class="skz-notification-system" id="skzNotificationSystem">
	<div class="skz-notifications-container" id="notificationsContainer">
		<!-- Notifications will be inserted here dynamically -->
	</div>
</div>

<script>
{literal}
class SKZNotificationSystem {
	constructor() {
		this.container = document.getElementById('notificationsContainer');
		this.notifications = [];
		this.maxNotifications = 5;
		this.autoHideDelay = 5000;
		
		// Initialize WebSocket connection if real-time updates are enabled
		{/literal}{if $enableRealTimeUpdates}{literal}
		this.initWebSocket();
		{/literal}{/if}{literal}
		
		// Demo notifications for testing
		this.showDemoNotifications();
	}
	
	initWebSocket() {
		// WebSocket connection to agent framework
		try {
			const wsUrl = 'ws://localhost:5000/ws/notifications';
			this.ws = new WebSocket(wsUrl);
			
			this.ws.onopen = () => {
				console.log('SKZ WebSocket connected');
				this.showNotification('success', 'Connected to SKZ Agents', 'Real-time updates enabled');
			};
			
			this.ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				this.handleAgentNotification(data);
			};
			
			this.ws.onclose = () => {
				console.log('SKZ WebSocket disconnected');
				// Attempt to reconnect after 5 seconds
				setTimeout(() => this.initWebSocket(), 5000);
			};
			
			this.ws.onerror = (error) => {
				console.error('SKZ WebSocket error:', error);
			};
		} catch (error) {
			console.log('WebSocket not available, using polling');
			this.initPolling();
		}
	}
	
	initPolling() {
		// Fallback to polling if WebSocket is not available
		setInterval(() => {
			this.pollForUpdates();
		}, 10000);
	}
	
	async pollForUpdates() {
		try {
			const response = await fetch('/index.php/journal/skzAgents/api/notifications');
			if (response.ok) {
				const notifications = await response.json();
				notifications.forEach(notification => {
					this.handleAgentNotification(notification);
				});
			}
		} catch (error) {
			console.error('Error polling for notifications:', error);
		}
	}
	
	handleAgentNotification(data) {
		const { type, title, message, agentId, timestamp } = data;
		
		// Determine notification type and icon
		let notificationType = 'info';
		let icon = 'fas fa-info-circle';
		
		switch (type) {
			case 'agent_completed':
				notificationType = 'success';
				icon = 'fas fa-check-circle';
				break;
			case 'agent_error':
				notificationType = 'error';
				icon = 'fas fa-exclamation-circle';
				break;
			case 'agent_warning':
				notificationType = 'warning';
				icon = 'fas fa-exclamation-triangle';
				break;
			case 'workflow_update':
				notificationType = 'info';
				icon = 'fas fa-sync-alt';
				break;
		}
		
		this.showNotification(notificationType, title, message, icon);
	}
	
	showNotification(type, title, message, icon = null) {
		const id = 'notif-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
		
		const notification = {
			id: id,
			type: type,
			title: title,
			message: message,
			icon: icon || this.getDefaultIcon(type),
			timestamp: new Date()
		};
		
		this.notifications.unshift(notification);
		
		// Remove oldest notifications if we exceed the maximum
		if (this.notifications.length > this.maxNotifications) {
			const removed = this.notifications.splice(this.maxNotifications);
			removed.forEach(notif => {
				const element = document.getElementById(notif.id);
				if (element) {
					element.remove();
				}
			});
		}
		
		this.renderNotification(notification);
		
		// Auto-hide after delay
		setTimeout(() => {
			this.hideNotification(id);
		}, this.autoHideDelay);
	}
	
	renderNotification(notification) {
		const notificationEl = document.createElement('div');
		notificationEl.id = notification.id;
		notificationEl.className = `skz-notification skz-notification-${notification.type}`;
		
		notificationEl.innerHTML = `
			<div class="skz-notification-content">
				<div class="skz-notification-icon">
					<i class="${notification.icon}"></i>
				</div>
				<div class="skz-notification-text">
					<div class="skz-notification-title">${notification.title}</div>
					<div class="skz-notification-message">${notification.message}</div>
				</div>
				<button class="skz-notification-close" onclick="skzNotifications.hideNotification('${notification.id}')">
					<i class="fas fa-times"></i>
				</button>
			</div>
			<div class="skz-notification-progress"></div>
		`;
		
		this.container.insertBefore(notificationEl, this.container.firstChild);
		
		// Trigger animation
		setTimeout(() => {
			notificationEl.classList.add('skz-notification-show');
		}, 100);
	}
	
	hideNotification(id) {
		const notification = document.getElementById(id);
		if (notification) {
			notification.classList.add('skz-notification-hide');
			setTimeout(() => {
				notification.remove();
			}, 300);
		}
		
		// Remove from notifications array
		this.notifications = this.notifications.filter(n => n.id !== id);
	}
	
	getDefaultIcon(type) {
		const icons = {
			success: 'fas fa-check-circle',
			error: 'fas fa-exclamation-circle',
			warning: 'fas fa-exclamation-triangle',
			info: 'fas fa-info-circle'
		};
		return icons[type] || icons.info;
	}
	
	showDemoNotifications() {
		// Show some demo notifications for testing
		setTimeout(() => {
			this.showNotification('success', 'Research Discovery Complete', 'Found 15 relevant research papers and 3 patent matches');
		}, 2000);
		
		setTimeout(() => {
			this.showNotification('info', 'Review Assignment', 'Editorial Orchestration Agent suggests Dr. Smith for peer review');
		}, 5000);
		
		setTimeout(() => {
			this.showNotification('warning', 'Quality Check', 'Content Quality Agent detected potential formatting issues');
		}, 8000);
	}
}

// Initialize notification system when page loads
document.addEventListener('DOMContentLoaded', function() {
	window.skzNotifications = new SKZNotificationSystem();
});
{/literal}
</script>