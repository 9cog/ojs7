# SKZ Architecture Quick Reference Diagrams

## 🏗️ High-Level Architecture Overview

```mermaid
flowchart TB
    subgraph "SKZ Autonomous Agents Framework"
        subgraph "Development Track"
            AAF["🔧 autonomous-agents-framework/"]
            AAF --> DEV_AGENTS["7 Standalone Agents<br/>Ports 8001-8007"]
        end
        
        subgraph "Production Track"
            MS["🚀 microservices/"]
            MS --> GATEWAY["API Gateway :5000"]
            GATEWAY --> PROD_SERVICES["7 Containerized Services"]
        end
    end
    
    OJS["📚 Open Journal Systems"] --> AAF
    OJS --> GATEWAY
    
    style AAF fill:#e1f5fe,stroke:#01579b
    style MS fill:#f3e5f5,stroke:#4a148c
    style GATEWAY fill:#fff3e0,stroke:#e65100
```

## 🔄 Agent Mapping Between Architectures

```mermaid
graph LR
    subgraph "Autonomous Agents Framework"
        A1["research_discovery_agent.py<br/>:8001"]
        A2["manuscript_analysis_agent.py<br/>:8002"]
        A3["peer_review_agent.py<br/>:8003"]
        A4["editorial_decision_agent.py<br/>:8004"]
        A5["publication_formatting_agent.py<br/>:8005"]
        A6["quality_assurance_agent.py<br/>:8006"]
        A7["workflow_orchestration_agent.py<br/>:8007"]
    end
    
    subgraph "Microservices Architecture"
        M1["research-discovery/"]
        M2["content-quality/"]
        M3["review-coordination/"]
        M4["editorial-orchestration/"]
        M5["publishing-production/"]
        M6["submission-assistant/"]
        M7["analytics-monitoring/"]
    end
    
    A1 -.-> M1
    A2 -.-> M2
    A3 -.-> M3
    A4 -.-> M4
    A5 -.-> M5
    A6 -.-> M6
    A7 -.-> M7
    
    style A1 fill:#e8f5e8
    style A2 fill:#e8f5e8
    style A3 fill:#e8f5e8
    style A4 fill:#e8f5e8
    style A5 fill:#e8f5e8
    style A6 fill:#e8f5e8
    style A7 fill:#e8f5e8
    
    style M1 fill:#f0e8ff
    style M2 fill:#f0e8ff
    style M3 fill:#f0e8ff
    style M4 fill:#f0e8ff
    style M5 fill:#f0e8ff
    style M6 fill:#f0e8ff
    style M7 fill:#f0e8ff
```

## 📊 Deployment Comparison

```mermaid
graph TB
    subgraph "Development Deployment"
        DEV_START["python scripts/start_all_agents.py"]
        DEV_START --> DEV_AGENTS["7 Flask Apps<br/>Direct HTTP APIs"]
        DEV_AGENTS --> DEV_READY["✅ Ready in 30 seconds"]
    end
    
    subgraph "Production Deployment"
        PROD_START["docker-compose up --build"]
        PROD_START --> PROD_CONTAINERS["7 Docker Containers<br/>+ API Gateway"]
        PROD_CONTAINERS --> PROD_READY["✅ Ready in 2-3 minutes"]
    end
    
    style DEV_START fill:#e8f5e8
    style DEV_READY fill:#c8e6c9
    style PROD_START fill:#f0e8ff
    style PROD_READY fill:#e1bee7
```

## 🎯 Use Case Decision Tree

```mermaid
flowchart TD
    START["Need SKZ Agents?"] --> PURPOSE{"What's your purpose?"}
    
    PURPOSE -->|Development & Testing| DEV_PATH["Use Autonomous Agents Framework"]
    PURPOSE -->|Production Deployment| PROD_PATH["Use Microservices Architecture"]
    
    DEV_PATH --> DEV_BENEFITS["✅ Fast iteration<br/>✅ Easy debugging<br/>✅ Direct access<br/>✅ Simple deployment"]
    
    PROD_PATH --> SCALE{"High volume?"}
    SCALE -->|Yes| PROD_BENEFITS["✅ Auto-scaling<br/>✅ Load balancing<br/>✅ Service discovery<br/>✅ Enterprise monitoring"]
    SCALE -->|No| CONSIDER["Consider starting with<br/>Autonomous Agents Framework"]
    
    style DEV_PATH fill:#e8f5e8
    style PROD_PATH fill:#f0e8ff
    style DEV_BENEFITS fill:#c8e6c9
    style PROD_BENEFITS fill:#e1bee7
```

## 🔧 Current Status Dashboard

```mermaid
graph LR
    subgraph "✅ DEPLOYED & OPERATIONAL"
        AAF_STATUS["Autonomous Agents Framework<br/>🟢 All 7 agents running<br/>🟢 Ports 8001-8007 active<br/>🟢 100% success rate"]
    end
    
    subgraph "🔄 READY FOR DEPLOYMENT"
        MS_STATUS["Microservices Architecture<br/>🟡 Container definitions ready<br/>🟡 API Gateway configured<br/>🟡 Awaiting production decision"]
    end
    
    OJS_INTEGRATION["OJS Integration"] --> AAF_STATUS
    FUTURE_SCALE["Future Scaling"] --> MS_STATUS
    
    style AAF_STATUS fill:#c8e6c9
    style MS_STATUS fill:#fff3e0
```

## 📈 Evolution Path

```mermaid
timeline
    title SKZ Architecture Evolution
    
    section Phase 1 : Development
        Current Status : Autonomous Agents Framework
                      : 7 agents deployed
                      : Direct HTTP APIs
                      : Development & testing ready
    
    section Phase 2 : Hybrid
        API Gateway    : Introduce central routing
                      : Gradual containerization
                      : Load balancer integration
    
    section Phase 3 : Production
        Microservices  : Full container orchestration
                      : Kubernetes deployment
                      : Auto-scaling & service mesh
```
