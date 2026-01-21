#!/bin/bash

# SKZ Integration Deployment Script
# Deploys the SKZ autonomous agents framework integration with OJS

set -e

echo "🚀 Starting SKZ Integration Deployment..."

# Check if we're in the correct directory
if [ ! -f "index.php" ] || [ ! -d "classes" ]; then
    echo "❌ Error: This script must be run from the OJS root directory"
    exit 1
fi

# Check if SKZ integration directory exists
if [ ! -d "skz-integration" ]; then
    echo "❌ Error: SKZ integration directory not found"
    exit 1
fi

echo "✅ OJS directory structure verified"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo "🔍 Checking dependencies..."

if ! command_exists python3; then
    echo "❌ Error: Python 3 is required but not installed"
    exit 1
fi

if ! command_exists node; then
    echo "❌ Error: Node.js is required but not installed"
    exit 1
fi

if ! command_exists npm; then
    echo "❌ Error: npm is required but not installed"
    exit 1
fi

echo "✅ Dependencies verified"

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p logs/skz-agents
mkdir -p cache/skz-agents
mkdir -p files/skz-agents

# Set up Python virtual environment for agent framework
echo "🐍 Setting up Python environment for agent framework..."
cd skz-integration/autonomous-agents-framework

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt

# Set up skin zone journal backend
echo "📚 Setting up Skin Zone Journal backend..."
cd ../skin-zone-journal

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt

# Set up workflow visualization dashboard
echo "📊 Setting up workflow visualization dashboard..."
cd ../workflow-visualization-dashboard

if [ ! -d "node_modules" ]; then
    npm install
fi

npm run build

# Set up simulation dashboard
echo "🎮 Setting up simulation dashboard..."
cd ../simulation-dashboard

if [ ! -d "node_modules" ]; then
    npm install
fi

npm run build

cd ../../..

# Copy configuration template
echo "⚙️ Setting up configuration..."
if [ ! -f "config/skz-agents.conf" ]; then
    cp skz-integration/config/skz-agents.conf config/skz-agents.conf
    echo "📝 Configuration template copied to config/skz-agents.conf"
    echo "🔧 Please edit config/skz-agents.conf with your specific settings"
fi

# Set up database schema for agent integration
echo "🗄️ Setting up database schema..."
if [ -f "skz-integration/schema/skz-agents.sql" ]; then
    echo "📝 Database schema found. Run the following SQL manually:"
    echo "   mysql -u [username] -p [database] < skz-integration/schema/skz-agents.sql"
else
    echo "⚠️ No database schema found. Creating basic schema..."
    cat > skz-integration/schema/skz-agents.sql << 'EOF'
-- SKZ Agents Integration Database Schema

CREATE TABLE IF NOT EXISTS skz_agent_states (
    agent_id VARCHAR(50) PRIMARY KEY,
    state_data JSON,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    submission_id INT,
    INDEX idx_submission (submission_id),
    INDEX idx_agent (agent_id),
    INDEX idx_updated (last_updated)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS skz_agent_communications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    agent_name VARCHAR(50) NOT NULL,
    action VARCHAR(100) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    request_size INT DEFAULT 0,
    response_size INT DEFAULT 0,
    success BOOLEAN DEFAULT FALSE,
    INDEX idx_agent_name (agent_name),
    INDEX idx_timestamp (timestamp),
    INDEX idx_success (success)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS skz_workflow_automation (
    id INT AUTO_INCREMENT PRIMARY KEY,
    submission_id INT NOT NULL,
    workflow_type VARCHAR(50) NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    automation_data JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_submission (submission_id),
    INDEX idx_workflow (workflow_type),
    INDEX idx_agent (agent_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
EOF
fi

# Create systemd services for production deployment
echo "🔧 Creating systemd services..."
mkdir -p skz-integration/systemd

cat > skz-integration/systemd/skz-agents-framework.service << 'EOF'
[Unit]
Description=SKZ Autonomous Agents Framework
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/ojs/skz-integration/autonomous-agents-framework
Environment=PATH=/path/to/ojs/skz-integration/autonomous-agents-framework/venv/bin
ExecStart=/path/to/ojs/skz-integration/autonomous-agents-framework/venv/bin/python src/main.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

cat > skz-integration/systemd/skz-skin-zone-journal.service << 'EOF'
[Unit]
Description=SKZ Skin Zone Journal Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/ojs/skz-integration/skin-zone-journal
Environment=PATH=/path/to/ojs/skz-integration/skin-zone-journal/venv/bin
ExecStart=/path/to/ojs/skz-integration/skin-zone-journal/venv/bin/python src/main.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Create nginx configuration for frontend services
echo "🌐 Creating nginx configuration..."
cat > skz-integration/nginx/skz-agents.conf << 'EOF'
# SKZ Agents Frontend Configuration

# Workflow Visualization Dashboard
location /skz/dashboard/ {
    alias /path/to/ojs/skz-integration/workflow-visualization-dashboard/dist/;
    try_files $uri $uri/ /index.html;
}

# Simulation Dashboard
location /skz/simulation/ {
    alias /path/to/ojs/skz-integration/simulation-dashboard/dist/;
    try_files $uri $uri/ /index.html;
}

# Agent Framework API Proxy
location /api/agents/ {
    proxy_pass http://localhost:5000/api/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

# Skin Zone Journal API Proxy
location /api/skin-zone/ {
    proxy_pass http://localhost:5001/api/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
EOF

# Create docker-compose for easy deployment
echo "🐳 Creating Docker configuration..."
cat > skz-integration/docker-compose.yml << 'EOF'
version: '3.8'

services:
  skz-agents-framework:
    build:
      context: ./autonomous-agents-framework
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=mysql://user:pass@db:3306/ojs
    volumes:
      - ./logs:/app/logs
    depends_on:
      - redis
      - db

  skz-skin-zone-journal:
    build:
      context: ./skin-zone-journal
      dockerfile: Dockerfile
    ports:
      - "5001:5001"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=mysql://user:pass@db:3306/ojs
    volumes:
      - ./logs:/app/logs
    depends_on:
      - redis
      - db

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: ojs
      MYSQL_USER: ojs
      MYSQL_PASSWORD: ojspassword
    volumes:
      - db_data:/var/lib/mysql
      - ./schema:/docker-entrypoint-initdb.d

volumes:
  redis_data:
  db_data:
EOF

# Create health check script
echo "🏥 Creating health check script..."
cat > skz-integration/scripts/health-check.sh << 'EOF'
#!/bin/bash

# SKZ Integration Health Check Script

echo "🏥 SKZ Integration Health Check"
echo "==============================="

# Check agent framework
echo "🔍 Checking Agent Framework..."
if curl -f -s http://localhost:5000/api/status > /dev/null; then
    echo "✅ Agent Framework: Running"
else
    echo "❌ Agent Framework: Not responding"
fi

# Check skin zone journal
echo "🔍 Checking Skin Zone Journal..."
if curl -f -s http://localhost:5001/api/status > /dev/null; then
    echo "✅ Skin Zone Journal: Running"
else
    echo "❌ Skin Zone Journal: Not responding"
fi

# Check database connectivity
echo "🔍 Checking Database..."
# Add database check here

# Check Redis (if enabled)
echo "🔍 Checking Redis..."
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis: Running"
else
    echo "⚠️ Redis: Not running (optional)"
fi

echo "==============================="
echo "🏥 Health check complete"
EOF

chmod +x skz-integration/scripts/health-check.sh

# Create monitoring script
echo "📊 Creating monitoring script..."
cat > skz-integration/scripts/monitor.sh << 'EOF'
#!/bin/bash

# SKZ Integration Monitoring Script

echo "📊 SKZ Integration Monitoring"
echo "============================="

# Agent performance metrics
echo "🤖 Agent Performance:"
curl -s http://localhost:5000/api/metrics | jq '.'

# System resources
echo ""
echo "💾 System Resources:"
echo "Memory usage: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "Disk usage: $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 ")"}')"
echo "CPU load: $(uptime | awk -F'load average:' '{print $2}')"

# Log analysis
echo ""
echo "📝 Recent Errors:"
tail -n 10 logs/skz-agents/error.log 2>/dev/null || echo "No errors found"

echo "============================="
EOF

chmod +x skz-integration/scripts/monitor.sh

# Set permissions
echo "🔒 Setting permissions..."
chown -R www-data:www-data skz-integration/ 2>/dev/null || echo "⚠️ Could not set ownership (run as root for production)"
chmod -R 755 skz-integration/

echo ""
echo "🎉 SKZ Integration Deployment Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Edit config/skz-agents.conf with your specific settings"
echo "2. Run the database schema: mysql -u [user] -p [db] < skz-integration/schema/skz-agents.sql"
echo "3. Enable the SKZ Agents plugin in OJS admin panel"
echo "4. Start the agent services:"
echo "   cd skz-integration/autonomous-agents-framework && source venv/bin/activate && python src/main.py"
echo "   cd skz-integration/skin-zone-journal && source venv/bin/activate && python src/main.py"
echo ""
echo "🔧 For production deployment:"
echo "1. Copy systemd services to /etc/systemd/system/"
echo "2. Update nginx configuration"
echo "3. Use Docker Compose for containerized deployment"
echo ""
echo "🏥 Health Check: ./skz-integration/scripts/health-check.sh"
echo "📊 Monitoring: ./skz-integration/scripts/monitor.sh"