---
description: Test and deploy OJS plugin integration
---

1. Test PHP bridge connection
php plugins/generic/skzAgents/tests/test_bridge.php

// turbo
2. Start Python API server
cd skz-integration/autonomous-agents-framework/src && python simple_api_server.py 5000

3. Run integration validation
bash validate_implementation.sh

4. Deploy to OJS
php plugins/generic/skzAgents/deploy.php