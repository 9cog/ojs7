# SKZ Integration Troubleshooting Guide
## Enhanced Open Journal Systems with Autonomous Agents

**Version:** 5.0.0  
**Last Updated:** August 9, 2025  
**Applies to:** Phase 5 - Testing and Optimization

---

## 🎯 Quick Troubleshooting Index

| Issue Category | Quick Link | Common Resolution Time |
|----------------|------------|------------------------|
| [Installation Issues](#-installation-issues) | Setup and deployment problems | 5-30 minutes |
| [Agent Communication](#-agent-communication-issues) | Agent connectivity and API issues | 2-15 minutes |
| [Performance Issues](#-performance-issues) | System performance and optimization | 10-60 minutes |
| [Database Issues](#-database-issues) | Data persistence and migration | 5-45 minutes |
| [Authentication Issues](#-authentication-issues) | Login and access control | 2-10 minutes |
| [Frontend Issues](#-frontend-issues) | UI and user experience problems | 5-20 minutes |

---

## 🔧 Installation Issues

### Issue: Composer Installation Takes Too Long or Fails

**Symptoms:**
- Composer hangs during `composer install` 
- Installation takes over 60 minutes
- Memory errors during installation
- Plugin trust prompts

**Solutions:**

1. **Trust Composer Plugins (Most Common)**
   ```bash
   # When prompted: "Do you trust cweagans/composer-patches?"
   # Answer: y
   ```

2. **Increase Memory Limit**
   ```bash
   php -d memory_limit=2G composer --working-dir=lib/pkp install --no-dev
   ```

3. **Use Composer with Timeout**
   ```bash
   composer config --global process-timeout 3600
   composer --working-dir=lib/pkp install --no-dev
   ```

4. **Clear Composer Cache**
   ```bash
   composer clear-cache
   composer --working-dir=lib/pkp install --no-dev
   ```

**Expected Installation Time:** 35+ minutes (normal, never cancel)

### Issue: Python Virtual Environment Setup Fails

**Symptoms:**
- `python3 -m venv venv` fails
- Permission errors
- Module not found errors

**Solutions:**

1. **Check Python Version**
   ```bash
   python3 --version  # Should be 3.11+
   ```

2. **Install Python Virtual Environment**
   ```bash
   # Ubuntu/Debian
   sudo apt install python3-venv
   
   # CentOS/RHEL
   sudo yum install python3-venv
   ```

3. **Use Alternative Virtual Environment**
   ```bash
   pip install virtualenv
   virtualenv venv
   ```

### Issue: npm Install Fails with ERESOLVE Errors

**Symptoms:**
- npm install fails with dependency conflicts
- ERESOLVE errors for react-day-picker or date-fns
- Peer dependency warnings

**Solutions:**

1. **Always Use Legacy Peer Deps (Required)**
   ```bash
   npm install --legacy-peer-deps
   ```

2. **Clear npm Cache**
   ```bash
   npm cache clean --force
   npm install --legacy-peer-deps
   ```

3. **Delete node_modules and Reinstall**
   ```bash
   rm -rf node_modules package-lock.json
   npm install --legacy-peer-deps
   ```

### Issue: Git Submodule Issues

**Symptoms:**
- Submodules not initialized
- Missing PKP library files
- Empty lib/pkp directory

**Solutions:**

1. **Initialize Submodules**
   ```bash
   git submodule update --init --recursive
   ```

2. **Force Submodule Update**
   ```bash
   git submodule update --force --recursive
   ```

3. **Reset Submodules**
   ```bash
   git submodule deinit --all -f
   git submodule update --init --recursive
   ```

---

## 🤖 Agent Communication Issues

### Issue: Agents Not Responding or Offline

**Symptoms:**
- Agent status shows as "Offline"
- API calls return 500 errors
- Agent communication timeouts

**Solutions:**

1. **Check Agent Status**
   ```bash
   curl http://localhost:5000/api/status
   curl http://localhost:5001/api/status
   ```

2. **Restart Agent Services**
   ```bash
   # Stop all agent processes
   pkill -f "python.*main.py"
   
   # Restart autonomous agents framework
   cd skz-integration/autonomous-agents-framework
   source venv/bin/activate
   python src/main.py &
   
   # Restart skin zone journal
   cd ../skin-zone-journal
   source venv/bin/activate
   python src/main.py &
   ```

3. **Check Port Conflicts**
   ```bash
   netstat -tlnp | grep :5000
   netstat -tlnp | grep :5001
   ```

4. **Verify Agent Configuration**
   ```bash
   # Check agent configuration files
   cat skz-integration/autonomous-agents-framework/.env
   cat skz-integration/skin-zone-journal/.env
   ```

### Issue: Agent Performance Degradation

**Symptoms:**
- Agent response times > 5 seconds
- High CPU or memory usage
- Agent timeout errors

**Solutions:**

1. **Check Resource Usage**
   ```bash
   top -p $(pgrep -f "python.*main.py")
   ```

2. **Restart Agents with Memory Optimization**
   ```bash
   export PYTHONOPTIMIZE=1
   python src/main.py
   ```

3. **Clear Agent Caches**
   ```bash
   # Clear SQLite databases
   rm -f *.db
   # Restart agents
   ```

### Issue: API Gateway Connection Failures

**Symptoms:**
- 502 Bad Gateway errors
- Connection refused errors
- API endpoints not accessible

**Solutions:**

1. **Verify API Gateway Status**
   ```bash
   curl -I http://localhost:5000/api/health
   ```

2. **Check Flask Application Logs**
   ```bash
   # Check agent logs
   tail -f agent_startup.log
   tail -f agent_status.txt
   ```

3. **Restart with Debug Mode**
   ```bash
   export FLASK_ENV=development
   export FLASK_DEBUG=1
   python src/main.py
   ```

---

## ⚡ Performance Issues

### Issue: Slow Page Load Times

**Symptoms:**
- Page load times > 5 seconds
- High server response times
- Timeout errors

**Solutions:**

1. **Enable PHP OpCache**
   ```php
   ; Add to php.ini
   opcache.enable=1
   opcache.memory_consumption=256
   opcache.max_accelerated_files=10000
   ```

2. **Optimize Database Queries**
   ```bash
   # Check MySQL slow query log
   tail -f /var/log/mysql/slow.log
   ```

3. **Enable Caching**
   ```php
   // In config.inc.php
   enable_cdn = On
   baseurl[index] = https://your-cdn-domain.com
   ```

### Issue: High Memory Usage

**Symptoms:**
- Memory usage > 4GB
- Out of memory errors
- Server crashes

**Solutions:**

1. **Optimize PHP Memory Settings**
   ```php
   ; In php.ini
   memory_limit = 2G
   max_execution_time = 300
   ```

2. **Enable Agent Memory Optimization**
   ```bash
   export PYTHONOPTIMIZE=1
   export MALLOC_TRIM_THRESHOLD_=100000
   ```

3. **Configure Agent Pool Sizes**
   ```python
   # In agent configuration
   MAX_WORKERS = 4
   MEMORY_LIMIT_PER_WORKER = "512MB"
   ```

### Issue: Database Performance Issues

**Symptoms:**
- Slow database queries
- Database connection timeouts
- High database CPU usage

**Solutions:**

1. **Optimize MySQL Configuration**
   ```sql
   -- Add to my.cnf
   innodb_buffer_pool_size = 1G
   query_cache_size = 256M
   max_connections = 200
   ```

2. **Add Database Indexes**
   ```sql
   -- Common performance indexes
   CREATE INDEX idx_agent_status ON agent_states(status);
   CREATE INDEX idx_submission_date ON submissions(date_submitted);
   ```

3. **Monitor Database Performance**
   ```bash
   # MySQL performance monitoring
   mysqladmin -u root -p processlist
   mysqladmin -u root -p status
   ```

---

## 🗄️ Database Issues

### Issue: Database Connection Failures

**Symptoms:**
- "Database connection failed" errors
- Unable to connect to MySQL
- Connection timeout errors

**Solutions:**

1. **Verify Database Service**
   ```bash
   sudo systemctl status mysql
   sudo systemctl start mysql
   ```

2. **Check Database Configuration**
   ```php
   // In config.inc.php
   driver = mysqli
   host = localhost
   username = ojs_user
   password = your_password
   name = ojs_database
   ```

3. **Test Database Connection**
   ```bash
   mysql -u ojs_user -p -h localhost ojs_database
   ```

4. **Create Database User and Permissions**
   ```sql
   CREATE USER 'ojs_user'@'localhost' IDENTIFIED BY 'your_password';
   GRANT ALL PRIVILEGES ON ojs_database.* TO 'ojs_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

### Issue: Agent State Database Corruption

**Symptoms:**
- Agents reset to default state
- "Database is locked" errors
- Inconsistent agent behavior

**Solutions:**

1. **Check SQLite Database Integrity**
   ```bash
   sqlite3 learning_framework.db "PRAGMA integrity_check;"
   ```

2. **Backup and Repair Database**
   ```bash
   # Backup current database
   cp learning_framework.db learning_framework.db.backup
   
   # Repair database
   sqlite3 learning_framework.db ".recover" | sqlite3 learning_framework_new.db
   mv learning_framework_new.db learning_framework.db
   ```

3. **Reset Agent Databases (Last Resort)**
   ```bash
   # Stop agents first
   pkill -f "python.*main.py"
   
   # Remove corrupted databases
   rm -f *.db
   
   # Restart agents (will recreate databases)
   python src/main.py
   ```

### Issue: Data Migration Problems

**Symptoms:**
- Migration scripts fail
- Data inconsistencies
- Version compatibility issues

**Solutions:**

1. **Run Migration Scripts Manually**
   ```bash
   php tools/upgrade.php upgrade
   ```

2. **Check Migration Status**
   ```sql
   SELECT * FROM versions ORDER BY date_installed DESC;
   ```

3. **Force Migration Reset**
   ```bash
   # Backup database first
   mysqldump -u root -p ojs_database > backup.sql
   
   # Run forced migration
   php tools/upgrade.php upgrade --force
   ```

---

## 🔐 Authentication Issues

### Issue: Unable to Login to OJS Admin

**Symptoms:**
- "Invalid username or password" errors
- Admin account locked
- Session timeouts

**Solutions:**

1. **Reset Admin Password**
   ```bash
   php tools/resetPassword.php admin new_password
   ```

2. **Create New Admin User**
   ```bash
   php tools/createUser.php admin admin@example.com password
   ```

3. **Check Session Configuration**
   ```php
   // In config.inc.php
   session_cookie_name = OJSSID
   session_cookie_path = /
   session_cookie_domain = 
   ```

### Issue: Agent API Authentication Failures

**Symptoms:**
- 401 Unauthorized errors
- API key validation failures
- Token expiration errors

**Solutions:**

1. **Verify API Keys**
   ```bash
   # Check API key configuration
   cat skz-integration/config/.env
   ```

2. **Regenerate API Tokens**
   ```bash
   # Generate new JWT tokens
   python scripts/generate_api_tokens.py
   ```

3. **Test API Authentication**
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" \
        http://localhost:5000/api/status
   ```

---

## 🎨 Frontend Issues

### Issue: Dashboard Not Loading

**Symptoms:**
- Blank dashboard pages
- JavaScript errors in console
- React component failures

**Solutions:**

1. **Check Build Status**
   ```bash
   cd skz-integration/workflow-visualization-dashboard
   npm run build
   ```

2. **Clear Browser Cache**
   ```bash
   # Or use browser dev tools
   # Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
   ```

3. **Check Console Errors**
   ```javascript
   // Open browser dev tools (F12)
   // Check Console tab for errors
   // Look for network errors in Network tab
   ```

4. **Rebuild with Dependencies**
   ```bash
   rm -rf node_modules package-lock.json
   npm install --legacy-peer-deps
   npm run build
   ```

### Issue: Theme Integration Problems

**Symptoms:**
- Styling issues
- Layout problems
- Theme not applying

**Solutions:**

1. **Clear Template Cache**
   ```bash
   rm -rf cache/t_*
   ```

2. **Rebuild Theme Assets**
   ```bash
   npm run build:themes
   ```

3. **Check Theme Configuration**
   ```php
   // In config.inc.php
   theme = skzEnhanced
   ```

---

## 🩺 Health Check Procedures

### System Health Check Script

```bash
#!/bin/bash
echo "SKZ Integration Health Check"
echo "============================"

# Check OJS Core
echo "1. Checking OJS Core..."
if curl -s http://localhost:8000 > /dev/null; then
    echo "   ✅ OJS Core is running"
else
    echo "   ❌ OJS Core is not responding"
fi

# Check Agent Framework
echo "2. Checking Agent Framework..."
if curl -s http://localhost:5000/api/status > /dev/null; then
    echo "   ✅ Agent Framework is running"
else
    echo "   ❌ Agent Framework is not responding"
fi

# Check Skin Zone Journal
echo "3. Checking Skin Zone Journal..."
if curl -s http://localhost:5001/api/status > /dev/null; then
    echo "   ✅ Skin Zone Journal is running"
else
    echo "   ❌ Skin Zone Journal is not responding"
fi

# Check Database
echo "4. Checking Database..."
if mysql -u ojs_user -p${DB_PASSWORD} -e "SELECT 1" > /dev/null 2>&1; then
    echo "   ✅ Database is accessible"
else
    echo "   ❌ Database connection failed"
fi

# Check Agent Communication
echo "5. Checking Agent Communication..."
AGENT_STATUS=$(curl -s http://localhost:5000/api/agents/status | jq -r '.status')
if [ "$AGENT_STATUS" = "active" ]; then
    echo "   ✅ Agents are communicating"
else
    echo "   ❌ Agent communication issues"
fi

echo "Health check complete."
```

### Performance Monitoring Commands

```bash
# Monitor system resources
htop

# Monitor agent processes
ps aux | grep python | grep main.py

# Monitor database performance
mysqladmin -u root -p processlist

# Monitor network connections
netstat -tlnp | grep -E ":5000|:5001|:8000"

# Check disk usage
df -h

# Check memory usage
free -h
```

---

## 📞 Getting Additional Help

### Escalation Path

1. **Level 1:** Check this troubleshooting guide
2. **Level 2:** Review specific component documentation
3. **Level 3:** Check GitHub issues and discussions
4. **Level 4:** Contact development team

### Useful Commands for Support

```bash
# System information
uname -a
php --version
python3 --version
node --version
mysql --version

# Generate system report
./skz-integration/scripts/generate-support-report.sh

# Export logs
tar -czf support-logs.tar.gz \
    agent_startup.log \
    health_check.log \
    /var/log/apache2/error.log \
    /var/log/mysql/error.log
```

### Documentation Links

- **Main Documentation:** [PHASE5_DOCUMENTATION_INDEX.md](PHASE5_DOCUMENTATION_INDEX.md)
- **API Documentation:** [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)
- **Agent Specifications:** [skz-integration/docs/agent-specifications/](skz-integration/docs/agent-specifications/)
- **Deployment Guide:** [skz-integration/deployment_guide.md](skz-integration/deployment_guide.md)

---

**Document Version:** 5.0.0  
**Last Updated:** August 9, 2025  
**Maintained by:** SKZ Support Team