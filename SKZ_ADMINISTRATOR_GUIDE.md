# SKZ Administrator Guide
## Enhanced Open Journal Systems with Autonomous Agents

**Version:** 5.0.0  
**Target Audience:** System Administrators, Journal Managers, Technical Staff  
**Last Updated:** August 9, 2025

---

## 🎯 Overview for Administrators

### What You Need to Know

As an administrator of SKZ Enhanced OJS, you're managing a sophisticated system that combines traditional Open Journal Systems functionality with 7 autonomous AI agents. This guide provides everything you need to successfully configure, maintain, and optimize the system.

### Key Responsibilities

| Area | Description | Complexity |
|------|-------------|------------|
| **System Configuration** | Initial setup and ongoing configuration | Medium |
| **Agent Management** | Monitoring and maintaining autonomous agents | High |
| **Performance Optimization** | System tuning and resource management | High |
| **User Management** | User accounts, roles, and permissions | Low |
| **Security Management** | Security policies and compliance | Medium |
| **Troubleshooting** | Issue resolution and system maintenance | High |

---

## 🚀 Initial System Setup

### Prerequisites Verification

Before beginning setup, verify all requirements are met:

```bash
# Check system requirements
php --version           # Should be 7.4+ (recommended: 8.1+)
python3 --version       # Should be 3.11+
node --version          # Should be 18+
mysql --version         # Should be 5.7+ or 8.0+
composer --version      # Should be 2.0+
npm --version           # Should be 8+

# Check system resources
free -h                 # Should have 4GB+ RAM (recommended: 8GB+)
df -h                   # Should have 20GB+ free space
nproc                   # Should have 2+ CPU cores (recommended: 4+)
```

### Installation Process

#### 1. Core System Setup

```bash
# Clone the repository
git clone https://github.com/EchoCog/oj7.git
cd oj7

# Initialize submodules
git submodule update --init --recursive

# Copy configuration template
cp config.TEMPLATE.inc.php config.inc.php
```

#### 2. PHP Dependencies (Critical: 35+ minutes)

```bash
# Install OJS core dependencies (NEVER CANCEL - takes 35+ minutes)
composer --working-dir=lib/pkp install --no-dev

# When prompted about composer-patches, answer 'y'
# Install plugin dependencies
composer --working-dir=plugins/paymethod/paypal install
composer --working-dir=plugins/generic/citationStyleLanguage install
```

**Critical Notes:**
- The composer installation takes 35+ minutes - this is normal, never cancel
- You will be prompted to trust "cweagans/composer-patches" - always answer "y"
- Set timeout to 60+ minutes to avoid cancellation

#### 3. Python Agent Framework Setup

```bash
# Set up autonomous agents framework
cd skz-integration/autonomous-agents-framework
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set up skin zone journal backend
cd ../skin-zone-journal
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 4. Node.js Dashboard Setup

```bash
# Set up workflow visualization dashboard
cd skz-integration/workflow-visualization-dashboard
npm install --legacy-peer-deps    # REQUIRED: --legacy-peer-deps flag
npm run build

# Set up simulation dashboard
cd ../simulation-dashboard
npm install --legacy-peer-deps    # REQUIRED: --legacy-peer-deps flag
npm run build
```

**Critical Note:** Always use `--legacy-peer-deps` flag due to date-fns version conflicts.

### Database Configuration

#### 1. Create Database and User

```sql
-- MySQL setup
CREATE DATABASE ojs_skz CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'ojs_user'@'localhost' IDENTIFIED BY 'secure_password';
GRANT ALL PRIVILEGES ON ojs_skz.* TO 'ojs_user'@'localhost';
FLUSH PRIVILEGES;
```

#### 2. Configure OJS Database Connection

Edit `config.inc.php`:

```php
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Database Settings
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

[database]
driver = mysqli
host = localhost
username = ojs_user
password = secure_password
name = ojs_skz

; Database connection charset
charset = utf8mb4
```

#### 3. SKZ Agent Database Schema

```bash
# Apply SKZ-specific database schema
mysql -u ojs_user -p ojs_skz < skz-integration/schema/skz-agents.sql
```

### Initial Configuration

#### 1. Basic OJS Configuration

```php
# Edit config.inc.php for production

; Basic configuration
base_url[index] = https://your-journal-domain.com
base_url[path] = https://your-journal-domain.com
disable_path_info = Off

; Security settings
allowed_hosts = "your-journal-domain.com"
force_ssl = On
force_login_ssl = On

; Session settings
session_cookie_name = OJSSKZ
session_check_ip = On

; File upload settings
max_file_tree_depth = 10
umask = 0022
```

#### 2. SKZ Agent Configuration

Create `skz-integration/config/.env`:

```bash
# Agent Framework Configuration
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your_secret_key_here

# Database Configuration
DATABASE_URL=mysql://ojs_user:secure_password@localhost/ojs_skz

# Agent Communication
AGENT_TIMEOUT=30
MAX_CONCURRENT_REQUESTS=10
AGENT_MEMORY_LIMIT=1GB

# External Integrations
INCI_DATABASE_API_KEY=your_inci_api_key
PATENT_API_KEY=your_patent_api_key
REGULATORY_API_KEY=your_regulatory_api_key

# Performance Settings
CACHE_TTL=3600
MAX_WORKERS=4
WORKER_MEMORY_LIMIT=512MB
```

---

## 🤖 Agent Management

### Understanding Agent Architecture

```
Agent Hierarchy and Communication:

Editorial Orchestration Agent (Central Coordinator)
├── Research Discovery Agent
├── Submission Assistant Agent
├── Review Coordination Agent
├── Content Quality Agent
├── Publishing Production Agent
└── Analytics & Monitoring Agent

Data Flow:
OJS Core ↔ API Gateway ↔ Agent Framework ↔ Individual Agents
```

### Starting and Stopping Agents

#### Starting All Agents

```bash
#!/bin/bash
# start-agents.sh

echo "Starting SKZ Autonomous Agents..."

# Start agent framework
cd skz-integration/autonomous-agents-framework
source venv/bin/activate
nohup python src/main.py > ../agent_framework.log 2>&1 &
echo $! > ../agent_framework.pid

# Start skin zone journal
cd ../skin-zone-journal
source venv/bin/activate
nohup python src/main.py > ../skin_zone_journal.log 2>&1 &
echo $! > ../skin_zone_journal.pid

echo "Agents started. Check logs for status."
```

#### Stopping All Agents

```bash
#!/bin/bash
# stop-agents.sh

echo "Stopping SKZ Autonomous Agents..."

# Kill agent processes
if [ -f skz-integration/agent_framework.pid ]; then
    kill $(cat skz-integration/agent_framework.pid)
    rm skz-integration/agent_framework.pid
fi

if [ -f skz-integration/skin_zone_journal.pid ]; then
    kill $(cat skz-integration/skin_zone_journal.pid)
    rm skz-integration/skin_zone_journal.pid
fi

# Force kill any remaining python agent processes
pkill -f "python.*main.py"

echo "Agents stopped."
```

### Agent Health Monitoring

#### Health Check Script

Create `scripts/health-check.sh`:

```bash
#!/bin/bash
# Health check for all SKZ components

echo "SKZ Health Check Report"
echo "======================"
echo "Timestamp: $(date)"

# Check OJS Core
echo "1. OJS Core Status:"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000 | grep -q "200"; then
    echo "   ✅ OJS is responding"
else
    echo "   ❌ OJS is not responding"
fi

# Check Agent Framework
echo "2. Agent Framework Status:"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/status | grep -q "200"; then
    echo "   ✅ Agent Framework is active"
    # Get detailed agent status
    curl -s http://localhost:5000/api/agents/status | jq '.'
else
    echo "   ❌ Agent Framework is not responding"
fi

# Check Skin Zone Journal
echo "3. Skin Zone Journal Status:"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/api/status | grep -q "200"; then
    echo "   ✅ Skin Zone Journal is active"
else
    echo "   ❌ Skin Zone Journal is not responding"
fi

# Check Database
echo "4. Database Status:"
if mysql -u ojs_user -p${DB_PASSWORD} -e "SELECT 1" ojs_skz > /dev/null 2>&1; then
    echo "   ✅ Database is accessible"
else
    echo "   ❌ Database connection failed"
fi

# Check System Resources
echo "5. System Resources:"
echo "   Memory Usage: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "   Disk Usage: $(df -h / | tail -1 | awk '{print $5 " of " $2}')"
echo "   CPU Load: $(uptime | awk -F'load average:' '{print $2}')"

echo "Health check complete."
```

#### Automated Monitoring Setup

```bash
# Add to crontab for automated monitoring
crontab -e

# Add this line for 5-minute health checks
*/5 * * * * /path/to/scripts/health-check.sh >> /var/log/skz-health.log 2>&1
```

### Agent Performance Tuning

#### Memory Optimization

```python
# In agent configuration files
import psutil
import gc

# Memory management settings
MAX_MEMORY_USAGE = 1024 * 1024 * 1024  # 1GB
MEMORY_CHECK_INTERVAL = 300  # 5 minutes

def optimize_memory():
    current_memory = psutil.virtual_memory().used
    if current_memory > MAX_MEMORY_USAGE:
        gc.collect()
        # Additional cleanup operations
```

#### Performance Monitoring

```bash
# Monitor agent performance
watch -n 5 'ps aux | grep python | grep main.py'

# Monitor memory usage
watch -n 5 'free -h'

# Monitor database connections
watch -n 5 'mysqladmin processlist | grep ojs_user'
```

### Agent Configuration Management

#### Individual Agent Settings

Each agent can be configured in `skz-integration/autonomous-agents-framework/src/config/`:

```python
# research_discovery_config.py
RESEARCH_DISCOVERY_CONFIG = {
    'inci_database_url': 'https://api.inci.org',
    'patent_api_endpoint': 'https://api.patents.org',
    'update_frequency': 3600,  # 1 hour
    'max_search_results': 100,
    'cache_duration': 86400,   # 24 hours
}

# submission_assistant_config.py
SUBMISSION_ASSISTANT_CONFIG = {
    'quality_threshold': 0.8,
    'auto_validation': True,
    'inci_verification': True,
    'safety_check': True,
    'statistical_review': True,
}

# editorial_orchestration_config.py
EDITORIAL_ORCHESTRATION_CONFIG = {
    'max_concurrent_workflows': 50,
    'decision_timeout': 86400,  # 24 hours
    'priority_scoring': True,
    'auto_assignment': True,
}
```

---

## 📊 System Monitoring and Analytics

### Performance Metrics Dashboard

Access the comprehensive analytics at:
- **Main Dashboard:** `http://localhost:5000/api/analytics/dashboard`
- **Agent Performance:** `http://localhost:5000/api/agents/performance`
- **System Metrics:** `http://localhost:5000/api/system/metrics`

### Key Performance Indicators

#### System-Level KPIs

```bash
# Monitor critical metrics
curl http://localhost:5000/api/metrics | jq '.'

# Expected output structure:
{
  "system": {
    "uptime": "99.95%",
    "response_time": "1.2s",
    "throughput": "150 requests/hour",
    "error_rate": "0.1%"
  },
  "agents": {
    "total_active": 7,
    "average_response_time": "800ms",
    "success_rate": "94.2%",
    "workload_distribution": "balanced"
  }
}
```

#### Agent-Specific KPIs

| Agent | Key Metrics | Target Values |
|-------|-------------|---------------|
| **Research Discovery** | Papers found, API calls, cache hits | 100+ papers/day, <2s response |
| **Submission Assistant** | Validations, quality scores, improvements | 95%+ accuracy, <5s validation |
| **Editorial Orchestration** | Workflows managed, decisions, conflicts | 100% automation, <1s coordination |
| **Review Coordination** | Matches made, assignments, quality | 90%+ match accuracy, <3s assignment |
| **Content Quality** | Validations, safety checks, compliance | 98%+ accuracy, <10s validation |
| **Publishing Production** | Formats generated, distributions, metadata | 100% automation, <30s processing |
| **Analytics & Monitoring** | Reports generated, insights, optimizations | Real-time updates, <1s queries |

### Log Management

#### Log Locations

```bash
# OJS Core Logs
/var/log/apache2/error.log          # Web server errors
/var/log/mysql/error.log            # Database errors
cache/logs/                         # OJS application logs

# Agent Framework Logs
skz-integration/agent_framework.log      # Main agent framework
skz-integration/skin_zone_journal.log   # Skin zone journal
skz-integration/agent_startup.log       # Agent startup logs
skz-integration/health_check.log        # Health check logs
```

#### Log Rotation Setup

```bash
# Create logrotate configuration
sudo tee /etc/logrotate.d/skz << EOF
/path/to/skz-integration/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload skz-agents
    endscript
}
EOF
```

### Performance Optimization

#### Database Optimization

```sql
-- MySQL optimization for SKZ
-- Add to my.cnf
[mysqld]
innodb_buffer_pool_size = 2G
query_cache_size = 256M
max_connections = 200
innodb_log_file_size = 256M
innodb_flush_log_at_trx_commit = 2

-- Useful indexes for SKZ
CREATE INDEX idx_agent_states_status ON agent_states(status, updated_at);
CREATE INDEX idx_submissions_workflow ON submissions(status, date_submitted);
CREATE INDEX idx_agent_communications ON agent_communications(agent_id, timestamp);
```

#### PHP Optimization

```php
# PHP optimization in php.ini
memory_limit = 2G
max_execution_time = 300
max_input_vars = 3000
post_max_size = 100M
upload_max_filesize = 100M

# OpCache settings
opcache.enable = 1
opcache.memory_consumption = 256
opcache.max_accelerated_files = 10000
opcache.validate_timestamps = 0  # Production only
```

#### Agent Performance Tuning

```python
# Agent optimization settings
PERFORMANCE_CONFIG = {
    'max_workers': 4,
    'worker_memory_limit': '512MB',
    'request_timeout': 30,
    'connection_pool_size': 10,
    'cache_size': 1000,
    'batch_processing': True,
    'async_operations': True,
}
```

---

## 🔐 Security and Compliance

### Security Configuration

#### SSL/TLS Setup

```apache
# Apache SSL configuration
<VirtualHost *:443>
    ServerName your-journal-domain.com
    DocumentRoot /path/to/oj7
    
    SSLEngine on
    SSLCertificateFile /path/to/certificate.crt
    SSLCertificateKeyFile /path/to/private.key
    SSLCertificateChainFile /path/to/ca-bundle.crt
    
    # Security headers
    Header always set Strict-Transport-Security "max-age=31536000; includeSubDomains"
    Header always set X-Content-Type-Options nosniff
    Header always set X-Frame-Options DENY
    Header always set X-XSS-Protection "1; mode=block"
</VirtualHost>
```

#### Firewall Configuration

```bash
# UFW firewall setup
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow required ports
sudo ufw allow 22/tcp     # SSH
sudo ufw allow 80/tcp     # HTTP
sudo ufw allow 443/tcp    # HTTPS
sudo ufw allow 3306/tcp   # MySQL (if needed)

# Internal agent communication (restrict to localhost)
sudo ufw allow from 127.0.0.1 to any port 5000
sudo ufw allow from 127.0.0.1 to any port 5001
```

#### Agent Security

```python
# Agent authentication configuration
SECURITY_CONFIG = {
    'jwt_secret_key': 'your-super-secret-jwt-key',
    'jwt_expiration': 3600,  # 1 hour
    'api_rate_limiting': True,
    'rate_limit_per_hour': 1000,
    'require_https': True,
    'validate_origins': True,
    'allowed_origins': ['https://your-journal-domain.com'],
}
```

### User Management

#### Role-Based Access Control

```php
# Configure user roles in OJS
// config.inc.php
$roles = array(
    'admin' => array('*'),
    'manager' => array('journal.management.*', 'user.management.*'),
    'editor' => array('editorial.*', 'review.*'),
    'reviewer' => array('review.assign.*', 'review.complete.*'),
    'author' => array('submission.*', 'revision.*'),
);
```

#### SKZ-Specific Permissions

```python
# Agent access permissions
AGENT_PERMISSIONS = {
    'research_discovery': ['external.api.access', 'database.read'],
    'submission_assistant': ['submission.read', 'submission.validate'],
    'editorial_orchestration': ['workflow.manage', 'decision.suggest'],
    'review_coordination': ['reviewer.assign', 'review.coordinate'],
    'content_quality': ['content.validate', 'safety.assess'],
    'publishing_production': ['content.format', 'distribution.manage'],
    'analytics_monitoring': ['metrics.read', 'analytics.generate'],
}
```

### Backup and Recovery

#### Automated Backup Script

```bash
#!/bin/bash
# backup-skz.sh

BACKUP_DIR="/backup/skz"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="skz_backup_$DATE"

echo "Starting SKZ backup: $BACKUP_NAME"

# Create backup directory
mkdir -p $BACKUP_DIR/$BACKUP_NAME

# Database backup
mysqldump -u root -p$MYSQL_ROOT_PASSWORD ojs_skz > $BACKUP_DIR/$BACKUP_NAME/database.sql

# File system backup
tar -czf $BACKUP_DIR/$BACKUP_NAME/files.tar.gz \
    files/ public/ plugins/ cache/ \
    skz-integration/ config.inc.php

# Agent data backup
cp -r skz-integration/autonomous-agents-framework/*.db $BACKUP_DIR/$BACKUP_NAME/

# Configuration backup
cp -r skz-integration/config/ $BACKUP_DIR/$BACKUP_NAME/config/

# Compress entire backup
cd $BACKUP_DIR
tar -czf $BACKUP_NAME.tar.gz $BACKUP_NAME/
rm -rf $BACKUP_NAME/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "skz_backup_*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
```

#### Recovery Procedures

```bash
#!/bin/bash
# restore-skz.sh

BACKUP_FILE=$1
RESTORE_DIR="/tmp/skz_restore"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

echo "Restoring from: $BACKUP_FILE"

# Stop services
systemctl stop apache2
./scripts/stop-agents.sh

# Extract backup
mkdir -p $RESTORE_DIR
tar -xzf $BACKUP_FILE -C $RESTORE_DIR

# Restore database
mysql -u root -p$MYSQL_ROOT_PASSWORD ojs_skz < $RESTORE_DIR/*/database.sql

# Restore files
tar -xzf $RESTORE_DIR/*/files.tar.gz -C /

# Restore agent data
cp $RESTORE_DIR/*/*.db skz-integration/autonomous-agents-framework/

# Restore configuration
cp -r $RESTORE_DIR/*/config/* skz-integration/config/

# Start services
systemctl start apache2
./scripts/start-agents.sh

echo "Restore completed"
```

### Compliance Management

#### GDPR Compliance

```php
# GDPR configuration in config.inc.php
[gdpr]
enabled = On
consent_required = On
data_retention_days = 2555  # 7 years
anonymization_enabled = On
audit_trail = On
```

#### Data Protection

```python
# Data protection settings for agents
DATA_PROTECTION = {
    'encryption_at_rest': True,
    'encryption_in_transit': True,
    'data_anonymization': True,
    'audit_logging': True,
    'retention_policy': '7_years',
    'gdpr_compliance': True,
}
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Agent Communication Failures

**Symptoms:** Agents not responding, API timeouts, workflow interruptions

**Diagnostics:**
```bash
# Check agent processes
ps aux | grep python | grep main.py

# Check port availability
netstat -tlnp | grep -E ":5000|:5001"

# Test agent endpoints
curl -v http://localhost:5000/api/status
curl -v http://localhost:5001/api/status
```

**Solutions:**
```bash
# Restart agents
./scripts/stop-agents.sh
./scripts/start-agents.sh

# Check agent logs
tail -f skz-integration/agent_framework.log
tail -f skz-integration/skin_zone_journal.log
```

#### Database Performance Issues

**Symptoms:** Slow queries, connection timeouts, high CPU usage

**Diagnostics:**
```sql
-- Check slow queries
SHOW PROCESSLIST;
SHOW STATUS LIKE 'Slow_queries';

-- Check table sizes
SELECT table_name, ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'Size in MB'
FROM information_schema.tables 
WHERE table_schema = 'ojs_skz'
ORDER BY (data_length + index_length) DESC;
```

**Solutions:**
```sql
-- Optimize tables
OPTIMIZE TABLE submissions;
OPTIMIZE TABLE agent_states;
OPTIMIZE TABLE agent_communications;

-- Add missing indexes
CREATE INDEX idx_performance ON submissions(status, date_submitted);
```

#### Memory Issues

**Symptoms:** Out of memory errors, system crashes, poor performance

**Diagnostics:**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check swap usage
swapon -s
```

**Solutions:**
```bash
# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Optimize agent memory usage
export PYTHONOPTIMIZE=1
export MALLOC_TRIM_THRESHOLD_=100000
```

### Emergency Procedures

#### System Recovery

```bash
#!/bin/bash
# emergency-recovery.sh

echo "SKZ Emergency Recovery Procedure"
echo "==============================="

# 1. Stop all services
echo "1. Stopping all services..."
systemctl stop apache2
./scripts/stop-agents.sh

# 2. Check system resources
echo "2. System resource check..."
free -h
df -h

# 3. Clear temporary files
echo "3. Clearing temporary files..."
rm -rf cache/t_*
rm -rf /tmp/skz_*

# 4. Reset agent databases (if corrupted)
echo "4. Checking agent databases..."
for db in skz-integration/autonomous-agents-framework/*.db; do
    if ! sqlite3 "$db" "PRAGMA integrity_check;" | grep -q "ok"; then
        echo "Corrupted database found: $db"
        mv "$db" "$db.corrupted"
        echo "Database backed up and will be recreated"
    fi
done

# 5. Restart services
echo "5. Restarting services..."
systemctl start apache2
./scripts/start-agents.sh

# 6. Health check
echo "6. Running health check..."
./scripts/health-check.sh

echo "Emergency recovery completed"
```

#### Data Recovery

```bash
#!/bin/bash
# data-recovery.sh

RECOVERY_DATE=$1
if [ -z "$RECOVERY_DATE" ]; then
    echo "Usage: $0 YYYYMMDD"
    exit 1
fi

echo "Data recovery for date: $RECOVERY_DATE"

# Find backup file
BACKUP_FILE="/backup/skz/skz_backup_${RECOVERY_DATE}_*.tar.gz"
if [ ! -f $BACKUP_FILE ]; then
    echo "Backup file not found for date: $RECOVERY_DATE"
    exit 1
fi

# Confirm recovery
read -p "This will restore data from $RECOVERY_DATE. Continue? (y/N): " confirm
if [ "$confirm" != "y" ]; then
    echo "Recovery cancelled"
    exit 0
fi

# Execute recovery
./scripts/restore-skz.sh $BACKUP_FILE
```

---

## 📚 Advanced Administration

### Custom Agent Development

#### Creating Custom Agents

```python
# custom_agent_template.py
from src.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.agent_name = "Custom Agent"
        self.agent_type = "custom"
    
    async def process_request(self, request):
        """
        Implement custom agent logic here
        """
        result = await self.custom_processing(request)
        return result
    
    async def custom_processing(self, request):
        """
        Custom processing implementation
        """
        # Your custom logic here
        return {"status": "completed", "data": "processed"}
```

#### Registering Custom Agents

```python
# In src/main.py
from agents.custom_agent import CustomAgent

# Register custom agent
agent_registry.register_agent(
    'custom_agent',
    CustomAgent,
    config=custom_agent_config
)
```

### Integration Development

#### External API Integration

```python
# external_api_integration.py
import aiohttp
import asyncio

class ExternalAPIIntegration:
    def __init__(self, api_config):
        self.api_key = api_config['api_key']
        self.base_url = api_config['base_url']
        self.session = aiohttp.ClientSession()
    
    async def fetch_data(self, endpoint, params=None):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        async with self.session.get(
            f"{self.base_url}/{endpoint}",
            headers=headers,
            params=params
        ) as response:
            return await response.json()
```

#### Database Extension

```sql
-- Custom database schema extensions
CREATE TABLE custom_agent_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    data_content JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_agent_data (agent_id, data_type),
    INDEX idx_created_at (created_at)
);

CREATE TABLE integration_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    integration_name VARCHAR(100) NOT NULL,
    operation VARCHAR(50) NOT NULL,
    status ENUM('success', 'failure', 'pending') NOT NULL,
    request_data JSON,
    response_data JSON,
    execution_time DECIMAL(10,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_integration_status (integration_name, status),
    INDEX idx_created_at (created_at)
);
```

### System Scaling

#### Load Balancing Setup

```nginx
# nginx load balancer configuration
upstream skz_backend {
    server 127.0.0.1:8000 weight=3;
    server 127.0.0.1:8001 weight=2;
    server 127.0.0.1:8002 weight=1;
}

upstream skz_agents {
    server 127.0.0.1:5000;
    server 127.0.0.1:5010;
    server 127.0.0.1:5020;
}

server {
    listen 443 ssl http2;
    server_name your-journal-domain.com;
    
    location / {
        proxy_pass http://skz_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /api/agents/ {
        proxy_pass http://skz_agents;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Database Clustering

```bash
# MySQL cluster setup for high availability
# Master-Slave replication configuration

# Master server configuration (my.cnf)
[mysqld]
server-id = 1
log-bin = mysql-bin
binlog-format = ROW
gtid-mode = ON
enforce-gtid-consistency = ON

# Slave server configuration (my.cnf)
[mysqld]
server-id = 2
read-only = 1
gtid-mode = ON
enforce-gtid-consistency = ON
```

---

## 📞 Support and Maintenance

### Regular Maintenance Tasks

#### Daily Tasks

```bash
#!/bin/bash
# daily-maintenance.sh

# Health check
./scripts/health-check.sh

# Log rotation check
logrotate -f /etc/logrotate.d/skz

# Database optimization
mysql -u root -p$MYSQL_ROOT_PASSWORD -e "
ANALYZE TABLE submissions;
ANALYZE TABLE agent_states;
"

# Clear temporary files
find /tmp -name "skz_*" -mtime +1 -delete
```

#### Weekly Tasks

```bash
#!/bin/bash
# weekly-maintenance.sh

# Full system backup
./scripts/backup-skz.sh

# Database integrity check
mysql -u root -p$MYSQL_ROOT_PASSWORD -e "
CHECK TABLE submissions;
CHECK TABLE agent_states;
CHECK TABLE agent_communications;
"

# Performance analysis
./scripts/performance-report.sh
```

#### Monthly Tasks

```bash
#!/bin/bash
# monthly-maintenance.sh

# Security updates
apt update && apt upgrade -y

# Certificate renewal (if using Let's Encrypt)
certbot renew

# Performance optimization
mysql -u root -p$MYSQL_ROOT_PASSWORD -e "
OPTIMIZE TABLE submissions;
OPTIMIZE TABLE agent_states;
OPTIMIZE TABLE agent_communications;
"

# Cleanup old logs
find /var/log -name "*.log.*" -mtime +90 -delete
```

### Support Contacts

#### Internal Support

- **Level 1:** System health checks and basic troubleshooting
- **Level 2:** Agent configuration and performance optimization
- **Level 3:** Custom development and advanced troubleshooting

#### External Support

- **GitHub Issues:** Bug reports and feature requests
- **Documentation:** Comprehensive guides and references
- **Community Forums:** User community support
- **Professional Support:** Enterprise support services

---

**Document Version:** 5.0.0  
**Last Updated:** August 9, 2025  
**Maintained by:** SKZ Technical Team  
**Next Review:** November 9, 2025