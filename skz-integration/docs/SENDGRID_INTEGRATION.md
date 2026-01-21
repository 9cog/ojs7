# SendGrid Email Integration Guide

## Overview

The SKZ Autonomous Agents Framework includes comprehensive SendGrid integration for production-quality email delivery. This integration replaces mock email sending with enterprise-grade email services.

## Features Implemented

### ✅ Core SendGrid Integration

- **Production-ready `_send_via_sendgrid()` method** - Fully implemented in `src/models/communication_automation.py`
- **Email template management** - Built-in template system with Jinja2 rendering
- **Delivery tracking** - Custom headers for message tracking and analytics
- **Webhook handling** - Ready for delivery status callbacks
- **Bounce and spam complaint handling** - Error handling and fallback mechanisms
- **Email analytics and reporting** - Message tracking and status logging
- **Comprehensive error handling** - Graceful degradation and fallback options
- **Unit tests** - Complete test coverage in `tests/unit/test_communication_automation.py`

### ✅ Advanced Features

- **Multi-provider support** - SendGrid, Amazon SES, and SMTP fallback
- **Template personalization** - Role-based and preference-based customization
- **Bulk messaging** - Concurrent sending for multiple recipients
- **Escalation rules** - Automated follow-up and escalation workflows
- **Frequency limiting** - Rate limiting and quiet hours support
- **Production quality controls** - Mock prevention in production mode

## Prerequisites

### 1. SendGrid Account Setup

1. **Create SendGrid Account**
   ```bash
   # Visit https://signup.sendgrid.com/
   # Choose Free plan (100 emails/day) or appropriate paid plan
   ```

2. **Obtain API Key**
   ```bash
   # In SendGrid Dashboard:
   # Settings > API Keys > Create API Key
   # Choose "Full Access" or "Restricted Access" with Mail Send permissions
   ```

3. **Domain Authentication**
   ```bash
   # Settings > Sender Authentication > Domain Authentication
   # Add your domain and complete DNS setup (SPF, DKIM, DMARC)
   ```

4. **Single Sender Verification** (for development)
   ```bash
   # Settings > Sender Authentication > Single Sender Verification
   # Verify email address for testing
   ```

### 2. Environment Configuration

1. **Copy environment template**
   ```bash
   cp skz-integration/.env.template skz-integration/.env
   ```

2. **Configure SendGrid settings**
   ```bash
   # Edit .env file:
   SENDGRID_API_KEY=SG.your-actual-api-key-here
   SENDGRID_FROM_EMAIL=noreply@yourdomain.com
   SENDGRID_FROM_NAME=Your Journal Name
   ```

### 3. Install Dependencies

```bash
cd skz-integration/autonomous-agents-framework
pip install -r requirements.txt
```

## Usage Examples

### Basic Email Sending

```python
from src.models.communication_automation import CommunicationAutomation, Recipient, MessageType, MessagePriority

# Initialize communication system with SendGrid
config = {
    'email_providers': {
        'sendgrid': {
            'enabled': True,
            'api_key': 'SG.your-api-key'
        }
    },
    'smtp': {
        'from_address': 'noreply@yourdomain.com'
    }
}

comm_system = CommunicationAutomation(config)

# Create recipient
recipient = Recipient(
    recipient_id="reviewer_001",
    name="Dr. Jane Smith",
    email="jane.smith@university.edu",
    phone=None,
    preferred_communication=MessageType.EMAIL,
    timezone="UTC",
    language="en",
    role="reviewer",
    organization="State University",
    communication_preferences={}
)

# Send reviewer invitation
context_data = {
    'reviewer_name': 'Dr. Jane Smith',
    'manuscript_title': 'Novel Skin Cancer Detection Methods',
    'authors': 'John Doe, Jane Doe',
    'journal_name': 'Skin Zone Journal',
    'submission_date': '2024-01-15',
    'estimated_time': '14',
    'expertise_areas': 'dermatology, oncology',
    'abstract': 'This study presents...',
    'response_deadline': '2024-02-01',
    'review_link': 'https://journal.com/review/12345',
    'editorial_team': 'Editorial Office'
}

message = await comm_system.send_message(
    'reviewer_invitation',
    recipient,
    context_data,
    MessagePriority.HIGH
)

print(f"Message sent: {message.status}")
```

### Bulk Email Sending

```python
# Send to multiple recipients
recipients = [recipient1, recipient2, recipient3]

messages = await comm_system.send_bulk_messages(
    'reviewer_invitation',
    recipients,
    context_data
)

print(f"Sent {len(messages)} messages successfully")
```

### Custom Templates

```python
from src.models.communication_automation import CommunicationTemplate

# Create custom template
custom_template = CommunicationTemplate(
    template_id='acceptance_notification',
    name='Manuscript Acceptance',
    subject_template='Congratulations! {{manuscript_title}} Accepted',
    body_template='''Dear {{author_name}},

We are pleased to inform you that your manuscript "{{manuscript_title}}" 
has been accepted for publication in {{journal_name}}.

Next steps:
{{next_steps}}

Congratulations on this achievement!

Best regards,
{{editorial_team}}''',
    message_type=MessageType.EMAIL,
    agent_id='editorial_orchestration_agent',
    scenario='acceptance',
    variables=['author_name', 'manuscript_title', 'journal_name', 'next_steps', 'editorial_team'],
    personalization_rules={'language_preference': True},
    send_conditions={},
    follow_up_rules={}
)

# Add to system
comm_system.templates['acceptance_notification'] = custom_template
```

## Testing

### Unit Tests

```bash
cd skz-integration/autonomous-agents-framework
python -m pytest tests/unit/test_communication_automation.py -v
```

### Integration Testing

```bash
# Test SendGrid integration
python -c "
import asyncio
from src.models.communication_automation import send_reviewer_invitation

async def test():
    result = await send_reviewer_invitation(
        'test@example.com',
        {
            'reviewer_name': 'Test Reviewer',
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
    )
    print(f'Test result: {result}')

asyncio.run(test())
"
```

## Production Deployment

### 1. Environment Variables

Set production environment variables:

```bash
export ENVIRONMENT=production
export SENDGRID_API_KEY=SG.your-production-api-key
export SENDGRID_FROM_EMAIL=noreply@yourdomain.com
export SENDGRID_FROM_NAME="Your Journal Name"
```

### 2. Production Configuration

```python
production_config = {
    'email_providers': {
        'sendgrid': {
            'enabled': True,
            'api_key': os.getenv('SENDGRID_API_KEY')
        },
        'ses': {
            'enabled': True,  # Fallback provider
            'region': os.getenv('AWS_SES_REGION'),
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY')
        }
    },
    'smtp': {
        'enabled': True,  # Final fallback
        'host': os.getenv('SMTP_HOST'),
        'port': int(os.getenv('SMTP_PORT', 587)),
        'username': os.getenv('SMTP_USERNAME'),
        'password': os.getenv('SMTP_PASSWORD'),
        'use_tls': True,
        'from_address': os.getenv('SENDGRID_FROM_EMAIL')
    }
}
```

### 3. Monitoring and Analytics

The system includes comprehensive logging and tracking:

```python
# Message tracking data
{
    'delivery_status': 'sent',
    'provider': 'sendgrid',
    'external_id': 'sg_message_id_12345',
    'delivery_time': '2024-01-15T10:30:00Z'
}
```

## Webhook Configuration

### Setup Delivery Tracking

1. **Configure webhook endpoint in SendGrid**
   ```
   POST https://yourdomain.com/api/v1/webhooks/sendgrid/delivery
   ```

2. **Implement webhook handler**
   ```python
   @app.route('/api/v1/webhooks/sendgrid/delivery', methods=['POST'])
   def handle_sendgrid_webhook():
       data = request.json
       # Process delivery status updates
       return '', 200
   ```

## Error Handling

The system includes comprehensive error handling:

1. **SendGrid API errors** - Logged and fallback triggered
2. **Network timeouts** - Automatic retry with exponential backoff
3. **Invalid API keys** - Clear error messages and fallback
4. **Rate limiting** - Automatic queue management
5. **Production safeguards** - Mock prevention in production mode

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure `sendgrid` package is installed
   ```bash
   pip install sendgrid>=6.10.0
   ```

2. **API Key Invalid**: Verify API key has Mail Send permissions

3. **Domain Not Verified**: Complete domain authentication in SendGrid

4. **Rate Limits**: Check SendGrid plan limits and usage

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('communication_automation').setLevel(logging.DEBUG)
```

## Security Considerations

1. **API Key Storage** - Use environment variables, never commit to code
2. **Domain Authentication** - Complete SPF/DKIM/DMARC setup
3. **Content Validation** - Sanitize email content
4. **Rate Limiting** - Implement proper rate limiting
5. **Webhook Security** - Verify webhook signatures

## Performance Optimization

1. **Bulk Sending** - Use bulk messaging for multiple recipients
2. **Template Caching** - Templates are cached for performance
3. **Async Processing** - All email sending is asynchronous
4. **Connection Pooling** - Reuse SendGrid client connections
5. **Fallback Providers** - Multiple providers ensure delivery

## Compliance

The integration supports:

- **GDPR Compliance** - Respect recipient preferences
- **CAN-SPAM Act** - Include unsubscribe options
- **DMARC/SPF/DKIM** - Domain authentication
- **Content Filtering** - Avoid spam triggers

## Support

For issues with SendGrid integration:

1. Check the logs for error messages
2. Verify configuration settings
3. Test with SendGrid API directly
4. Contact SendGrid support for API issues
5. Review the communication automation test suite

## Migration from Mock

The system automatically detects production mode and prevents mock usage:

```python
# PRODUCTION QUALITY CHECK: Prevent mock usage in production
if os.getenv('ENVIRONMENT', '').lower() == 'production':
    raise ValueError(
        "PRODUCTION VIOLATION: Mock email implementation called in production mode. "
        "NEVER SACRIFICE QUALITY!! Configure SendGrid, Amazon SES, or SMTP for production."
    )
```

This ensures that mock implementations never accidentally run in production.