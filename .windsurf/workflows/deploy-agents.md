---
description: Deploy SKZ agents to production environment
---

1. Run pre-deployment checks
python skz-integration/autonomous-agents-framework/scripts/deploy_production.py --check

// turbo
2. Start all agent services
python skz-integration/autonomous-agents-framework/scripts/start_all_agents.py

3. Verify deployment health
python skz-integration/autonomous-agents-framework/scripts/health_check.py

4. Commit deployment status
python auto_commit.py