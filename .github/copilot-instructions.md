# Enhanced Open Journal Systems with SKZ Autonomous Agents

Enhanced Open Journal Systems (OJS) integrated with the SKZ (Skin Zone Journal) autonomous agents framework for intelligent academic publishing workflow automation.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap, Build, and Test the Repository

**CRITICAL: NEVER CANCEL any build or long-running command. Builds take 35+ minutes. Set timeouts to 60+ minutes minimum.**

1. **Initial Setup:**
   ```bash
   # Git submodules (takes < 1 second - very fast)
#   NEVER USE SUBMODULES!!!
   
   # Configuration setup (only if config.inc.php doesn't exist)
   cp config.TEMPLATE.inc.php config.inc.php
   ```

2. **PHP Dependencies (CRITICAL: Takes 35+ minutes, NEVER CANCEL):**
   ```bash
   # Main OJS dependencies - TAKES 35+ MINUTES, SET TIMEOUT 60+ MINUTES
   composer --working-dir=lib/pkp install --no-dev
   # Will prompt: "Do you trust cweagans/composer-patches?" - Answer: y
   
   # Plugin dependencies (only if needed)
   composer --working-dir=plugins/paymethod/paypal install
   composer --working-dir=plugins/generic/citationStyleLanguage install
   ```

3. **Python Agent Framework Setup:**
   ```bash
   # Autonomous agents framework (takes ~3 seconds for venv, ~53 seconds for deps)
   cd skz-integration/autonomous-agents-framework
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Skin zone journal backend (may fail due to network timeouts)
   cd ../skin-zone-journal
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt  # May timeout - retry if needed
   ```

4. **Node.js Dashboard Setup:**
   ```bash
   # Workflow visualization dashboard (takes ~1.5 minutes)
   cd skz-integration/workflow-visualization-dashboard
   npm install --legacy-peer-deps  # REQUIRED: --legacy-peer-deps due to date-fns conflicts
   npm run build  # Takes ~6 seconds
   
   # Simulation dashboard (takes ~5 seconds install, ~7 seconds build)
   cd ../simulation-dashboard
   npm install --legacy-peer-deps  # REQUIRED: --legacy-peer-deps due to dependency conflicts
   npm run build
   ```

### Running Tests

**CRITICAL: Test commands can take 15+ minutes. NEVER CANCEL. Set timeout to 30+ minutes.**

```bash
# OJS core tests - TAKES 15+ MINUTES, NEVER CANCEL
./lib/pkp/tools/runAllTests.sh

# Individual test categories (faster alternatives):
./lib/pkp/tools/runAllTests.sh -C  # PKP class tests only
./lib/pkp/tools/runAllTests.sh -c  # App class tests only

# SKZ Agent tests
cd skz-integration/autonomous-agents-framework
source venv/bin/activate
python -m pytest tests/
```

### Running the Application

1. **OJS Core System:**
   ```bash
   # Simple development server
   php -S localhost:8000
   ```

2. **SKZ Agent Services:**
   ```bash
   # Agent framework (Terminal 1) - Use main_simple.py to avoid import issues
   cd skz-integration/autonomous-agents-framework
   source venv/bin/activate
   python src/main.py  # NEVER SACRIFICE QUALITY!! NO MOCK/SIMPLIFID/PLACEHOLDER EVER!
   
   # Skin zone journal backend (Terminal 2) - May have import issues
   cd skz-integration/skin-zone-journal
   source venv/bin/activate
   python src/main.py  # May fail due to Python import paths
   
   # Serve visualization dashboards (Terminal 3)
   cd skz-integration/workflow-visualization-dashboard
   pnpm run build && pnpm run preview  # BUILD CORRECTLY
   ```

3. **Health Checks:**
   ```bash
   # System health verification (script created and validated)
   ./skz-integration/scripts/health-check.sh
   
   # Check individual services
   curl http://localhost:5000/api/v1/agents  # Agent framework (works)
   curl http://localhost:5001/api/status     # Skin zone journal (may not work)
   curl http://localhost:8000                # OJS core (works)
   ```

## Validation Scenarios

**ALWAYS test these scenarios after making changes:**

1. **OJS Core Validation:**
   - Start PHP server: `php -S localhost:8000`
   - Access http://localhost:8000 and verify OJS loads
   - Check admin panel accessibility

2. **Agent Framework Validation:**
   - Start agent services: `python src/main.py` NEVER DEGRADE THE CODEBASE WITH MOCK/SIMPLE!!
   - Verify API endpoints respond: `curl http://localhost:5000/api/v1/agents`
   - Check web dashboard at http://localhost:5000
   - Verify JSON response contains 7 active agents

3. **Complete Workflow Test:**
   - Run health check script: `./skz-integration/scripts/health-check.sh`
   - Verify all components show as ✅ (green checkmarks)
   - Check dashboards are built in dist/ directories
   - Verify Python virtual environments exist

## Build Time Expectations

**CRITICAL TIMING INFORMATION - NEVER CANCEL THESE:**

- **Composer PHP install:** 35+ minutes (NEVER CANCEL - SET 60+ MINUTE TIMEOUT)
- **Python venv creation:** ~3 seconds per environment (much faster than expected)
- **Python pip install:** ~53 seconds per requirements.txt (may timeout due to network issues)
- **npm install (with --legacy-peer-deps):** ~1.5 minutes for workflow dashboard, ~5 seconds for simulation dashboard
- **npm run build:** ~6 seconds per dashboard (much faster than expected)
- **Test suite:** 15+ minutes (NEVER CANCEL - SET 30+ MINUTE TIMEOUT)

## Common Issues and Workarounds

1. **npm install fails with ERESOLVE errors:**
   - **SOLUTION:** Always use `npm install --legacy-peer-deps`
   - **CAUSE:** date-fns version conflicts in react-day-picker

2. **Composer plugin trust prompt:**
   - **EXPECTED:** Composer will ask to trust "cweagans/composer-patches"
   - **SOLUTION:** Answer "y" when prompted

3. **Long Composer installs:**
   - **NORMAL:** 35+ minutes for full dependency installation
   - **NEVER CANCEL:** This is expected behavior, not a hang

4. **Python import errors in agent framework:**
   - **SOLUTION:** Use `python src/main.py` AND FIX THE ERROR! NEVER SACRIFICE QUALITY FOR ANY REASON!!
   - **CAUSE:** Complex import paths in main.py cause ModuleNotFoundError

5. **Network timeouts during pip install:**
   - **SOLUTION:** Retry the pip install command if it times out
   - **CAUSE:** Network connectivity issues to PyPI servers

6. **Missing health check script:**
   - **SOLUTION:** The script is now created at `skz-integration/scripts/health-check.sh`
   - **VALIDATION:** Script has been tested and works correctly

## Key Project Structure

```
/
├── index.php                    # OJS main entry point
├── config.inc.php              # Main OJS configuration
├── classes/                    # OJS core classes
├── lib/pkp/                   # PKP library (core OJS framework)
├── plugins/                   # OJS plugins
├── skz-integration/           # SKZ autonomous agents framework
│   ├── autonomous-agents-framework/    # Main agent services
│   ├── skin-zone-journal/             # Specialized journal backend
│   ├── workflow-visualization-dashboard/  # React dashboard
│   ├── simulation-dashboard/          # Agent simulation interface
│   └── microservices/                # Individual agent microservices
└── deploy-skz-integration.sh   # Automated deployment script
```

## Essential Commands Reference

### Development Workflow
```bash
# Quick development setup
git submodule update --init --recursive
cp config.TEMPLATE.inc.php config.inc.php
composer --working-dir=lib/pkp install --no-dev  # LONG: 35+ min, NEVER CANCEL

# Start development environment
php -S localhost:8000  # OJS core
# In separate terminals:
cd skz-integration/autonomous-agents-framework && source venv/bin/activate && python src/main.py
cd skz-integration/skin-zone-journal && source venv/bin/activate && python src/main.py
```

### Before Committing Changes
```bash
# ALWAYS run these validation steps:
./lib/pkp/tools/runAllTests.sh  # LONG: 15+ min, NEVER CANCEL
./skz-integration/scripts/health-check.sh
curl http://localhost:5000/api/v1/agents  # Verify agent integration works
```

## Database Requirements

- **MySQL 5.7+ or 8.0+** (confirmed available)
- **Run schema manually:** `mysql -u [user] -p [database] < skz-integration/schema/skz-agents.sql`
- **Additional tables:** Agent states, communications, workflow automation

## Production Deployment

Use the automated deployment script:
```bash
chmod +x deploy-skz-integration.sh
./deploy-skz-integration.sh
```

This script:
- Sets up all Python virtual environments
- Installs all dependencies with proper error handling
- Creates systemd services for production
- Generates nginx configuration
- Creates Docker Compose setup
- Sets up monitoring and health check scripts

## Environment Dependencies Verified

- **Python:** 3.12.3 (requirement: 3.11+) ✅
- **Node.js:** 20.19.4 (requirement: 18+) ✅
- **npm:** 10.8.2 (requirement: 8+) ✅
- **PHP:** 8.3.6 (requirement: 7.4+) ✅
- **MySQL:** 8.0.42 (requirement: 5.7+) ✅
- **Composer:** 2.8.10 ✅

## AI Inference Engine Requirements

**PRODUCTION-GRADE AI INFERENCE MANDATORY - NO EXCEPTIONS**

### 🧠 Prompt Seed Reference
See [.github/copilot-prompt-seed.md](./copilot-prompt-seed.md) for foundational prompt context and detailed AI implementation examples.

### Core AI Inference Guidelines
- **PRIMARY ENGINE:** Use llama.cpp or node-llama-cpp as the default AI inference engine
- **ZERO MOCK TOLERANCE:** Never generate placeholder logic for AI agent functions
- **REAL INFERENCE ONLY:** If performance is degraded, use quantized models, NOT mocks
- **FALLBACK POLICY:** Fallback must be a functional inference engine (e.g., quantized llama.cpp), never a mock
- **PRODUCTION PRIORITY:** Prioritize inference correctness over performance during development

### AI Implementation Standards
- **BERT/Transformer Integration:** Use production BERT models for text classification
- **Model Deployment:** Implement proper model loading and caching infrastructure  
- **Inference Optimization:** Use ONNX runtime or similar for fast inference
- **Error Handling:** AI failures must degrade gracefully to simpler but real AI models
- **Configuration:** All AI models must be configurable via environment variables

### Prohibited AI Practices
- `return "TODO"` or similar placeholders in AI functions
- Mock classification/inference that returns dummy data
- Keyword-only fallbacks without real AI processing
- Simplified logic that bypasses actual model inference
- Development shortcuts that compromise AI functionality

## Critical Reminders

- **NEVER CANCEL BUILDS:** Composer installs take 35+ minutes, tests take 15+ minutes
- **ALWAYS use --legacy-peer-deps** for npm installs to avoid dependency conflicts
- **SET LONG TIMEOUTS:** 60+ minutes for builds, 30+ minutes for tests
- **VALIDATE THOROUGHLY:** Always test complete workflows after changes
- **ANSWER PROMPTS:** Composer will ask to trust plugins - answer "y"
- **NEVER USE SIMPLE PYTHON:** Use `main.py` NEVER SACRIFICE QUALITY!!!!
- **NETWORK ISSUES:** Python pip installs may timeout - retry if needed
- **HEALTH CHECK:** Always run `./skz-integration/scripts/health-check.sh` to verify system state
- **AI INFERENCE:** Always verify AI models are loaded and functional before deployment