# SKZ Integration - Quick Start Guide

## 🚀 Getting Started with SKZ Autonomous Agents

This guide provides the essential steps to deploy and configure the SKZ autonomous agents integration with your OJS installation.

### Prerequisites Verification

```bash
# Check required dependencies
python3 --version  # Should be 3.11+
node --version     # Should be 18+
npm --version      # Should be 8+
php --version      # Should be 7.4+
mysql --version    # Should be 5.7+ or 8.0+
```

### Quick Deployment

1. **Run the automated deployment script:**
   ```bash
   cd /path/to/your/ojs
   chmod +x deploy-skz-integration.sh
   ./deploy-skz-integration.sh
   ```

2. **Configure the integration:**
   ```bash
   # Edit the configuration file
   nano config/skz-agents.conf
   
   # Set your specific values:
   # - agent_base_url (Flask API endpoint)
   # - api_key (secure API key)
   # - database settings
   ```

3. **Set up the database:**
   ```bash
   mysql -u [username] -p [database_name] < skz-integration/schema/skz-agents.sql
   ```

4. **Enable the plugin:**
   - Go to OJS Admin → Settings → Website → Plugins
   - Find "SKZ Autonomous Agents" plugin
   - Click "Enable"

### Starting the Agent Services

1. **Start the Autonomous Agents Framework:**
   ```bash
   cd skz-integration/autonomous-agents-framework
   source venv/bin/activate
   python src/main.py
   # Service will run on http://localhost:5000
   ```

2. **Start the Skin Zone Journal Backend:**
   ```bash
   cd skz-integration/skin-zone-journal
   source venv/bin/activate
   python src/main.py
   # Service will run on http://localhost:5001
   ```

3. **Serve the Frontend Dashboards:**
   ```bash
   # Workflow Visualization Dashboard
   cd skz-integration/workflow-visualization-dashboard
   npm run build
   # Serve via nginx or directly: npm run preview
   
   # Simulation Dashboard
   cd skz-integration/simulation-dashboard
   npm run build
   # Serve via nginx or directly: npm run preview
   ```

### Health Checks

```bash
# Quick health check
./skz-integration/scripts/health-check.sh

# Monitor performance
./skz-integration/scripts/monitor.sh

# Check agent status via API
curl http://localhost:5000/api/status
curl http://localhost:5001/api/status
```

### Agent Endpoints

| Agent | Endpoint | Function |
|-------|----------|----------|
| Research Discovery | `/api/research-discovery/analyze` | INCI analysis, patent research |
| Submission Assistant | `/api/submission-assistant/process` | Quality assessment, compliance |
| Editorial Orchestration | `/api/editorial-orchestration/coordinate` | Workflow management |
| Review Coordination | `/api/review-coordination/coordinate` | Reviewer matching |
| Content Quality | `/api/content-quality/validate` | Scientific validation |
| Publishing Production | `/api/publishing-production/produce` | Content formatting |
| Analytics Monitoring | `/api/analytics-monitoring/analyze` | Performance insights |

### Frontend Access Points

- **OJS Integration**: Your OJS installation with agent features enabled
- **Workflow Dashboard**: `http://your-domain/skz/dashboard/`
- **Simulation Dashboard**: `http://your-domain/skz/simulation/`
- **Agent APIs**: `http://your-domain/api/agents/`

### Configuration Options

Key settings in `config/skz-agents.conf`:

```ini
[skz]
enabled = true
agent_base_url = "http://localhost:5000/api"
api_key = "your-secure-api-key"

# Feature flags for gradual rollout
feature_auto_submission_processing = false  # Start with false
feature_auto_reviewer_assignment = false    # Enable gradually
feature_auto_quality_checks = true          # Safe to enable
```

### Troubleshooting

**Common Issues:**

1. **Agents not responding:**
   ```bash
   # Check if services are running
   ps aux | grep python
   netstat -tlnp | grep :5000
   netstat -tlnp | grep :5001
   ```

2. **Database connection errors:**
   ```bash
   # Verify database credentials in config/skz-agents.conf
   mysql -u [user] -p [database] -e "SHOW TABLES LIKE 'skz_%';"
   ```

3. **Plugin not appearing:**
   ```bash
   # Check OJS plugin directory permissions
   ls -la plugins/generic/skzAgents/
   # Verify plugin registration in OJS
   ```

### Performance Optimization

1. **Enable Redis caching:**
   ```ini
   [skz]
   use_redis_cache = true
   redis_host = "127.0.0.1"
   redis_port = 6379
   ```

2. **Adjust concurrent requests:**
   ```ini
   [skz]
   max_concurrent_requests = 10
   cache_ttl = 300
   ```

3. **Monitor performance:**
   ```bash
   # Check agent metrics
   curl http://localhost:5000/api/metrics
   
   # View database performance
   mysql -e "SELECT * FROM skz_agent_performance_summary LIMIT 10;"
   ```

### Security Considerations

1. **API Security:**
   - Generate strong API keys
   - Enable SSL/TLS in production
   - Set up rate limiting

2. **Database Security:**
   - Use dedicated database user with minimal privileges
   - Enable query logging for auditing
   - Regular backups of agent data

3. **Network Security:**
   - Firewall rules for agent services
   - VPN or private network for API communication
   - Regular security updates

### Production Deployment

For production environments:

1. **Use Docker Compose:**
   ```bash
   cd skz-integration
   docker-compose up -d
   ```

2. **Set up systemd services:**
   ```bash
   sudo cp skz-integration/systemd/*.service /etc/systemd/system/
   sudo systemctl enable skz-agents-framework
   sudo systemctl enable skz-skin-zone-journal
   sudo systemctl start skz-agents-framework
   sudo systemctl start skz-skin-zone-journal
   ```

3. **Configure nginx:**
   ```bash
   sudo cp skz-integration/nginx/skz-agents.conf /etc/nginx/sites-available/
   sudo ln -s /etc/nginx/sites-available/skz-agents.conf /etc/nginx/sites-enabled/
   sudo nginx -t && sudo systemctl reload nginx
   ```

### Support and Documentation

- **Integration Strategy**: `SKZ_INTEGRATION_STRATEGY.md`
- **API Documentation**: `skz-integration/docs/`
- **Agent Specifications**: `skz-integration/docs/agent-specifications/`
- **Workflow Diagrams**: `skz-integration/docs/workflow-diagrams/`

### Next Steps

1. **Gradual Feature Enablement:**
   - Start with quality checks and monitoring
   - Gradually enable workflow automation
   - Monitor performance and adjust settings

2. **User Training:**
   - Train editors on new agent features
   - Provide user guides for submission enhancements
   - Set up feedback channels for improvements

3. **Monitoring and Optimization:**
   - Regular performance reviews
   - Agent effectiveness analysis
   - Continuous improvement based on metrics

---

🎉 **Congratulations!** Your OJS installation is now enhanced with autonomous agents for intelligent academic publishing workflow automation.

For additional support, refer to the comprehensive documentation in the `skz-integration/docs/` directory.