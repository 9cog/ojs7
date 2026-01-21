# Production Configuration Template for SKZ Autonomous Agents

# This file provides configuration templates for production deployment
# Copy this file to config_production.py and fill in your actual credentials

## Patent Analysis Configuration
PATENT_CONFIG = {
    # USPTO API Configuration
    'uspto_api_key': 'YOUR_USPTO_API_KEY_HERE',
    'use_production_apis': False,  # Set to True when APIs are configured
    
    # Google Patents API Configuration  
    'google_cloud_credentials': 'path/to/google-credentials.json',
    
    # Caching Configuration
    'redis_url': 'redis://localhost:6379/0',
    'cache_ttl': 3600,  # 1 hour
    'max_results_per_query': 100,
    'rate_limit_delay': 1.0,  # seconds
    'api_timeout': 30,  # seconds
}

## Communication Configuration
EMAIL_CONFIG = {
    'email_providers': {
        'sendgrid': {
            'api_key': 'YOUR_SENDGRID_API_KEY_HERE',
            'enabled': False,  # Set to True when configured
            'rate_limit': 100,  # per minute
            'webhook_url': 'https://yourdomain.com/webhooks/sendgrid'
        },
        'ses': {
            'aws_access_key_id': 'YOUR_AWS_ACCESS_KEY_ID',
            'aws_secret_access_key': 'YOUR_AWS_SECRET_ACCESS_KEY',
            'region': 'us-east-1',
            'enabled': False,  # Set to True when configured
            'rate_limit': 200  # per second
        }
    },
    'sms_providers': {
        'twilio': {
            'account_sid': 'YOUR_TWILIO_ACCOUNT_SID',
            'auth_token': 'YOUR_TWILIO_AUTH_TOKEN',
            'from_number': '+1234567890',
            'webhook_url': 'https://yourdomain.com/webhooks/twilio',
            'enabled': False  # Set to True when configured
        }
    },
    'smtp_config': {
        'enabled': True,
        'host': 'localhost',
        'port': 587,
        'use_tls': True,
        'from_address': 'noreply@yourdomain.com',
        'username': 'your_smtp_username',
        'password': 'your_smtp_password'
    },
    'webhook_base_url': 'https://yourdomain.com'
}

## ML Configuration
ML_CONFIG = {
    # BERT Model Configuration
    'bert_model': 'sentence-transformers/allenai-specter',  # Scientific papers
    'device': 'cpu',  # Change to 'cuda' if GPU available
    'max_sequence_length': 512,
    'batch_size': 32,
    'model_cache_dir': '/models/cache',
    'onnx_model_path': '/models/production/text_classifier.onnx',
    
    # Domain-specific classifiers
    'cosmetic_chemistry_classifier': '/models/cosmetic_chemistry_bert.pkl',
    'dermatology_classifier': '/models/dermatology_bert.pkl',
    'toxicology_classifier': '/models/toxicology_bert.pkl',
    
    # Quality assessment models
    'quality_ensemble_models': [
        '/models/content_quality_bert.pkl',
        '/models/statistical_quality_rf.pkl',
        '/models/writing_quality_gpt.pkl'
    ],
    
    # Performance settings
    'inference_timeout': 10.0,  # seconds
    'max_concurrent_requests': 100,
    'model_memory_limit': '8GB'
}

## Reviewer Matching Configuration
REVIEWER_MATCHING_CONFIG = {
    'sentence_transformer_model': 'sentence-transformers/allenai-specter',
    'expertise_model_path': '/models/reviewer_expertise_classifier.pkl',
    'workload_model_path': '/models/reviewer_workload_predictor.pkl', 
    'quality_model_path': '/models/review_quality_predictor.pkl',
    
    # Optimization settings
    'use_global_optimization': True,
    'optimization_algorithm': 'hungarian',  # 'hungarian', 'genetic', 'greedy'
    'max_optimization_time': 30,  # seconds
    
    # Scoring weights
    'expertise_weight': 0.35,
    'workload_weight': 0.25,
    'quality_weight': 0.20,
    'availability_weight': 0.20,
    
    # Performance settings
    'embedding_cache_size': 10000,
    'batch_embedding_size': 32,
    'max_concurrent_matches': 100
}

## Database Configuration
PRODUCTION_DB_CONFIG = {
    'database': {
        'url': 'postgresql://user:password@localhost:5432/skz_production',
        'pool_size': 20,
        'max_overflow': 30,
        'pool_timeout': 30,
        'pool_recycle': 3600,
        'isolation_level': 'READ_COMMITTED'
    },
    'redis': {
        'url': 'redis://localhost:6379/0',
        'max_connections': 100,
        'retry_on_timeout': True,
        'socket_timeout': 5
    },
    'event_store': {
        'url': 'postgresql://user:password@localhost:5432/skz_events',
        'retention_days': 365,
        'compression': True,
        'replication': True
    },
    'message_broker': {
        'url': 'amqp://user:password@localhost:5672/',
        'exchange': 'skz_events',
        'routing_key': 'sync.events'
    }
}

## Environment Variables Template
"""
# Add these to your environment or .env file:

# Patent Analysis
export USPTO_API_KEY="your_uspto_api_key"
export GOOGLE_CLOUD_CREDENTIALS="/path/to/google-credentials.json"

# Communication 
export SENDGRID_API_KEY="your_sendgrid_api_key"
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export TWILIO_ACCOUNT_SID="your_twilio_account_sid"  
export TWILIO_AUTH_TOKEN="your_twilio_auth_token"
export TWILIO_FROM_NUMBER="+1234567890"

# Database
export PRODUCTION_DB_URL="postgresql://user:password@localhost:5432/skz_production"
export REDIS_URL="redis://localhost:6379/0"

# ML Models
export MODEL_CACHE_DIR="/models/cache"
export BERT_MODEL_PATH="/models/bert-base-uncased"

# Webhooks
export WEBHOOK_BASE_URL="https://yourdomain.com"
export SENDGRID_WEBHOOK_URL="https://yourdomain.com/webhooks/sendgrid"
export TWILIO_WEBHOOK_URL="https://yourdomain.com/webhooks/twilio"
"""

## Production Dependencies
PRODUCTION_REQUIREMENTS = """
# Add these to requirements.txt for production deployment:

# Patent Analysis
aiohttp>=3.8.0
google-cloud-patents>=1.0.0
redis>=4.0.0
asyncio-throttle>=1.0.0

# Communication
sendgrid>=6.9.0
twilio>=8.0.0
boto3>=1.26.0
aiofiles>=0.8.0
pydantic>=1.10.0
celery>=5.2.0

# ML and NLP
torch>=1.13.0
transformers>=4.21.0
sentence-transformers>=2.2.0
onnxruntime>=1.12.0
mlflow>=2.0.0
scikit-learn>=1.1.0
numpy>=1.21.0
pandas>=1.5.0
spacy>=3.4.0
nltk>=3.7
textstat>=0.7.0

# Reviewer Matching
scipy>=1.9.0
networkx>=2.8.0
ortools>=9.4.0

# Database and Infrastructure
asyncpg>=0.27.0
aioredis>=2.0.1
sqlalchemy[asyncio]>=1.4.0
alembic>=1.8.0
kombu>=5.2.0
structlog>=22.1.0
"""

# Instructions for Production Setup
SETUP_INSTRUCTIONS = """
## Production Setup Instructions

### 1. Install Dependencies
pip install -r requirements_production.txt

### 2. Configure Environment Variables
Copy the environment variables template above to your .env file or system environment

### 3. Set up External Services
- Create USPTO developer account and obtain API key
- Set up Google Cloud project and enable Patents API
- Create SendGrid account for email delivery
- Set up Twilio account for SMS delivery
- Configure Amazon SES as email fallback
- Set up PostgreSQL database for production
- Configure Redis for caching and distributed locking

### 4. Configure Production Settings
- Copy this file to config_production.py
- Fill in all API keys and credentials
- Set enabled=True for configured services
- Update database URLs and connection settings

### 5. Initialize Database
- Run database migrations: alembic upgrade head
- Set up event sourcing tables
- Configure database replication and backups

### 6. Deploy ML Models
- Download pre-trained BERT models
- Set up model serving infrastructure
- Configure ONNX runtime for production inference
- Set up GPU instances if using CUDA

### 7. Test Configuration
- Run health checks on all services
- Test API integrations with small requests
- Verify database connectivity and transactions
- Test message delivery through all providers

### 8. Monitoring and Alerts
- Set up monitoring for all external APIs
- Configure alerts for service failures
- Set up performance monitoring
- Configure log aggregation and analysis
"""