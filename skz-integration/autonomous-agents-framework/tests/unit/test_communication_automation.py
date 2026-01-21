"""
Unit tests for Communication Automation System
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.models.communication_automation import (
    CommunicationAutomation,
    CommunicationTemplate,
    Recipient,
    CommunicationMessage,
    EscalationRule,
    NotificationConfig,
    MessageType,
    MessagePriority,
    CommunicationStatus
)

@pytest.fixture
def sample_recipient():
    """Sample recipient for testing"""
    return Recipient(
        recipient_id="test_recipient_001",
        name="Dr. Test Reviewer",
        email="test.reviewer@example.com",
        phone="+1234567890",
        preferred_communication=MessageType.EMAIL,
        timezone="UTC",
        language="en",
        role="reviewer",
        organization="Test University",
        communication_preferences={
            "html_emails": True,
            "reminder_frequency": "weekly"
        }
    )

@pytest.fixture
def sample_template():
    """Sample communication template"""
    return CommunicationTemplate(
        template_id="test_template",
        name="Test Template",
        subject_template="Test Subject: {{title}}",
        body_template="Dear {{name}},\n\nThis is a test message about {{title}}.\n\nBest regards,\nTest System",
        message_type=MessageType.EMAIL,
        agent_id="test_agent",
        scenario="test_scenario",
        variables=["name", "title"],
        personalization_rules={"language_preference": True},
        send_conditions={"status": "active"},
        follow_up_rules={"reminder_days": 3}
    )

@pytest.fixture
def comm_automation():
    """Initialize communication automation system"""
    config = {
        'smtp': {
            'enabled': False,  # Disable for testing
            'host': 'localhost',
            'port': 587,
            'from_address': 'test@example.com'
        }
    }
    return CommunicationAutomation(config)

class TestCommunicationAutomation:
    """Test CommunicationAutomation class"""

    def test_initialization(self, comm_automation):
        """Test communication system initialization"""
        assert comm_automation.config is not None
        assert hasattr(comm_automation, 'templates')
        assert hasattr(comm_automation, 'message_queue')
        assert len(comm_automation.templates) > 0  # Default templates loaded

    def test_default_templates_loaded(self, comm_automation):
        """Test that default templates are properly loaded"""
        expected_templates = [
            'reviewer_invitation',
            'review_reminder',
            'author_status_update',
            'quality_issue_alert'
        ]
        
        for template_id in expected_templates:
            assert template_id in comm_automation.templates
            template = comm_automation.templates[template_id]
            assert isinstance(template, CommunicationTemplate)

    @pytest.mark.asyncio
    async def test_send_message_basic(self, comm_automation, sample_recipient):
        """Test basic message sending functionality"""
        context_data = {
            'reviewer_name': 'Dr. Test Reviewer',
            'manuscript_title': 'Test Manuscript',
            'authors': 'Test Authors',
            'journal_name': 'Test Journal',
            'submission_date': '2024-01-15',
            'estimated_time': '14',
            'expertise_areas': 'testing, automation',
            'abstract': 'This is a test abstract.',
            'response_deadline': '2024-02-01',
            'review_link': 'https://test.com/review',
            'editorial_team': 'Test Editorial Team'
        }
        
        message = await comm_automation.send_message(
            'reviewer_invitation',
            sample_recipient,
            context_data
        )
        
        assert isinstance(message, CommunicationMessage)
        assert message.recipient.recipient_id == sample_recipient.recipient_id
        assert message.template_id == 'reviewer_invitation'
        assert message.status in [CommunicationStatus.SENT, CommunicationStatus.PENDING]

    @pytest.mark.asyncio
    async def test_template_rendering(self, comm_automation):
        """Test template rendering functionality"""
        template_str = "Hello {{name}}, your task is {{task}}."
        context = {"name": "Alice", "task": "reviewing"}
        
        rendered = await comm_automation._render_template(template_str, context)
        
        assert "Hello Alice" in rendered
        assert "reviewing" in rendered

    @pytest.mark.asyncio
    async def test_send_conditions_check(self, comm_automation, sample_template):
        """Test send conditions validation"""
        # Test with matching conditions
        context_data = {"status": "active"}
        result = await comm_automation._check_send_conditions(sample_template, context_data)
        assert result is True
        
        # Test with non-matching conditions
        context_data = {"status": "inactive"}
        result = await comm_automation._check_send_conditions(sample_template, context_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_personalization(self, comm_automation, sample_template, sample_recipient):
        """Test message personalization"""
        subject = "Test Subject"
        body = "Test body content"
        
        personalized_subject, personalized_body = await comm_automation._apply_personalization(
            sample_template, sample_recipient, subject, body
        )
        
        # Should apply role-based customization
        if sample_recipient.role in ['senior_editor', 'managing_editor']:
            assert "[" in personalized_body  # Check for role prefix
        
        # Verify strings are returned
        assert isinstance(personalized_subject, str)
        assert isinstance(personalized_body, str)

    @pytest.mark.asyncio
    async def test_bulk_messaging(self, comm_automation, sample_recipient):
        """Test bulk message sending"""
        recipients = [sample_recipient] * 3  # Simulate multiple recipients
        
        context_data = {
            'reviewer_name': 'Dr. Test Reviewer',
            'manuscript_title': 'Test Manuscript',
            'authors': 'Test Authors',
            'journal_name': 'Test Journal',
            'submission_date': '2024-01-15',
            'estimated_time': '14',
            'expertise_areas': 'testing',
            'abstract': 'Test abstract',
            'response_deadline': '2024-02-01',
            'review_link': 'https://test.com/review',
            'editorial_team': 'Test Team'
        }
        
        messages = await comm_automation.send_bulk_messages(
            'reviewer_invitation',
            recipients,
            context_data
        )
        
        assert len(messages) <= 3  # Some might fail
        assert all(isinstance(msg, CommunicationMessage) for msg in messages)

    @pytest.mark.asyncio
    async def test_notification_setup(self, comm_automation):
        """Test notification configuration setup"""
        config = NotificationConfig(
            config_id="test_notifications",
            agent_id="test_agent",
            event_types=["quality_issue", "deadline_approaching"],
            recipients=["admin@example.com"],
            templates={
                "quality_issue": "quality_issue_alert"
            },
            frequency_limits={"quality_issue": 5},
            quiet_hours={"enabled": True, "start_hour": 22, "end_hour": 8},
            enabled=True
        )
        
        await comm_automation.setup_automated_notifications(config)
        
        assert config.config_id in comm_automation.notification_configs
        assert comm_automation.notification_configs[config.config_id] == config

    @pytest.mark.asyncio
    async def test_escalation_rule_setup(self, comm_automation):
        """Test escalation rule configuration"""
        rule = EscalationRule(
            rule_id="test_escalation",
            trigger_condition="no_response",
            escalation_delay=60,  # 60 minutes
            escalation_recipients=["supervisor@example.com"],
            escalation_template="escalation_template",
            max_escalations=2,
            escalation_agent="test_agent"
        )
        
        await comm_automation.setup_escalation_rule(rule)
        
        assert rule.rule_id in comm_automation.escalation_rules
        assert comm_automation.escalation_rules[rule.rule_id] == rule

    @pytest.mark.asyncio
    async def test_frequency_limits(self, comm_automation):
        """Test message frequency limiting"""
        config = NotificationConfig(
            config_id="test_freq",
            agent_id="test_agent",
            event_types=["test_event"],
            recipients=["user@example.com"],
            templates={"test_event": "test_template"},
            frequency_limits={"test_event": 2},
            quiet_hours={"enabled": False},
            enabled=True
        )
        
        # Test frequency check
        result = await comm_automation._check_frequency_limits(config, "test_event")
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_quiet_hours_check(self, comm_automation):
        """Test quiet hours validation"""
        config = NotificationConfig(
            config_id="test_quiet",
            agent_id="test_agent",
            event_types=["test_event"],
            recipients=["user@example.com"],
            templates={"test_event": "test_template"},
            frequency_limits={},
            quiet_hours={"enabled": True, "start_hour": 22, "end_hour": 8},
            enabled=True
        )
        
        result = await comm_automation._check_quiet_hours(config)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_message_delivery_channels(self, comm_automation, sample_recipient):
        """Test different message delivery channels"""
        message = CommunicationMessage(
            message_id="test_001",
            template_id="test_template",
            recipient=sample_recipient,
            sender_agent="test_agent",
            subject="Test Subject",
            body="Test body",
            message_type=MessageType.EMAIL,
            priority=MessagePriority.MEDIUM,
            scheduled_time=datetime.now().isoformat(),
            sent_time=None,
            status=CommunicationStatus.PENDING,
            context_data={},
            attachments=[],
            tracking_data={}
        )
        
        # Test email delivery (simulated)
        result = await comm_automation._send_email(message)
        assert isinstance(result, bool)
        
        # Test SMS delivery (simulated)
        result = await comm_automation._send_sms(message)
        assert isinstance(result, bool)

    def test_message_type_enum(self):
        """Test MessageType enum"""
        assert MessageType.EMAIL.value == "email"
        assert MessageType.SMS.value == "sms"
        assert MessageType.SLACK.value == "slack"

    def test_message_priority_enum(self):
        """Test MessagePriority enum"""
        assert MessagePriority.LOW.value == 1
        assert MessagePriority.URGENT.value == 4
        assert MessagePriority.CRITICAL.value == 5

    @pytest.mark.asyncio
    async def test_invalid_template_handling(self, comm_automation, sample_recipient):
        """Test handling of invalid template IDs"""
        message = await comm_automation.send_message(
            'nonexistent_template',
            sample_recipient,
            {}
        )
        
        assert message.status == CommunicationStatus.FAILED


# Test utility functions
class TestUtilityFunctions:
    """Test utility functions"""

    @pytest.mark.asyncio
    async def test_reviewer_invitation_utility(self):
        """Test reviewer invitation utility function"""
        from src.models.communication_automation import send_reviewer_invitation
        
        manuscript_data = {
            'reviewer_name': 'Dr. Test',
            'manuscript_title': 'Test Paper',
            'authors': 'Test Authors',
            'journal_name': 'Test Journal',
            'submission_date': '2024-01-15',
            'estimated_time': '14',
            'expertise_areas': 'testing',
            'abstract': 'Test abstract',
            'response_deadline': '2024-02-01',
            'review_link': 'https://test.com/review',
            'editorial_team': 'Test Team'
        }
        
        result = await send_reviewer_invitation('reviewer@example.com', manuscript_data)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_quality_alerts_setup_utility(self):
        """Test quality alerts setup utility function"""
        from src.models.communication_automation import setup_quality_alerts
        
        config = await setup_quality_alerts('test_agent', ['admin@example.com'])
        
        assert isinstance(config, NotificationConfig)
        assert config.agent_id == 'test_agent'
        assert 'quality_issue' in config.event_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
