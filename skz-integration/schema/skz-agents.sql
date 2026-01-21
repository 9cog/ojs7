-- SKZ Agents Integration Database Schema
-- Adds tables for autonomous agent state management and communication logging

-- Agent state tracking table
CREATE TABLE IF NOT EXISTS skz_agent_states (
    agent_id VARCHAR(50) PRIMARY KEY,
    state_data JSON,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    submission_id INT,
    context_id INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_submission (submission_id),
    INDEX idx_context (context_id),
    INDEX idx_agent (agent_id),
    INDEX idx_updated (last_updated)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Agent communication logging table
CREATE TABLE IF NOT EXISTS skz_agent_communications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    agent_name VARCHAR(50) NOT NULL,
    action VARCHAR(100) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    request_size INT DEFAULT 0,
    response_size INT DEFAULT 0,
    success BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    user_id INT,
    submission_id INT,
    context_id INT,
    execution_time_ms INT DEFAULT 0,
    INDEX idx_agent_name (agent_name),
    INDEX idx_timestamp (timestamp),
    INDEX idx_success (success),
    INDEX idx_user (user_id),
    INDEX idx_submission (submission_id),
    INDEX idx_context (context_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Workflow automation tracking table
CREATE TABLE IF NOT EXISTS skz_workflow_automation (
    id INT AUTO_INCREMENT PRIMARY KEY,
    submission_id INT NOT NULL,
    workflow_type VARCHAR(50) NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    automation_data JSON,
    status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    completed_at DATETIME NULL,
    user_id INT,
    context_id INT,
    INDEX idx_submission (submission_id),
    INDEX idx_workflow (workflow_type),
    INDEX idx_agent (agent_name),
    INDEX idx_status (status),
    INDEX idx_created (created_at),
    INDEX idx_user (user_id),
    INDEX idx_context (context_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Agent performance metrics table
CREATE TABLE IF NOT EXISTS skz_agent_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    agent_name VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    metric_type ENUM('efficiency', 'accuracy', 'response_time', 'success_rate') NOT NULL,
    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    context_id INT,
    INDEX idx_agent (agent_name),
    INDEX idx_metric (metric_name),
    INDEX idx_type (metric_type),
    INDEX idx_recorded (recorded_at),
    INDEX idx_context (context_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Agent configuration settings table
CREATE TABLE IF NOT EXISTS skz_agent_settings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    agent_name VARCHAR(50) NOT NULL,
    setting_name VARCHAR(100) NOT NULL,
    setting_value TEXT,
    setting_type ENUM('string', 'integer', 'boolean', 'json') DEFAULT 'string',
    context_id INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_agent_setting_context (agent_name, setting_name, context_id),
    INDEX idx_agent (agent_name),
    INDEX idx_setting (setting_name),
    INDEX idx_context (context_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Agent workflow rules table
CREATE TABLE IF NOT EXISTS skz_workflow_rules (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL,
    workflow_type VARCHAR(50) NOT NULL,
    conditions JSON NOT NULL,
    actions JSON NOT NULL,
    priority INT DEFAULT 0,
    enabled BOOLEAN DEFAULT TRUE,
    context_id INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_workflow (workflow_type),
    INDEX idx_priority (priority),
    INDEX idx_enabled (enabled),
    INDEX idx_context (context_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert default agent settings
INSERT IGNORE INTO skz_agent_settings (agent_name, setting_name, setting_value, setting_type) VALUES
('research-discovery', 'enabled', 'true', 'boolean'),
('research-discovery', 'inci_database_enabled', 'true', 'boolean'),
('research-discovery', 'patent_analysis_enabled', 'true', 'boolean'),
('research-discovery', 'trend_identification_enabled', 'true', 'boolean'),
('research-discovery', 'regulatory_monitoring_enabled', 'true', 'boolean'),

('submission-assistant', 'enabled', 'true', 'boolean'),
('submission-assistant', 'quality_threshold', '0.7', 'string'),
('submission-assistant', 'safety_compliance_required', 'true', 'boolean'),
('submission-assistant', 'statistical_review_enabled', 'true', 'boolean'),
('submission-assistant', 'auto_enhancement_suggestions', 'true', 'boolean'),

('editorial-orchestration', 'enabled', 'true', 'boolean'),
('editorial-orchestration', 'auto_workflow_coordination', 'false', 'boolean'),
('editorial-orchestration', 'auto_decision_making', 'false', 'boolean'),
('editorial-orchestration', 'conflict_resolution_enabled', 'true', 'boolean'),
('editorial-orchestration', 'strategic_planning_enabled', 'true', 'boolean'),

('review-coordination', 'enabled', 'true', 'boolean'),
('review-coordination', 'auto_reviewer_matching', 'false', 'boolean'),
('review-coordination', 'workload_balancing', 'true', 'boolean'),
('review-coordination', 'quality_monitoring', 'true', 'boolean'),
('review-coordination', 'expert_network_enabled', 'true', 'boolean'),

('content-quality', 'enabled', 'true', 'boolean'),
('content-quality', 'scientific_validation', 'true', 'boolean'),
('content-quality', 'safety_assessment', 'true', 'boolean'),
('content-quality', 'standards_enforcement', 'true', 'boolean'),
('content-quality', 'regulatory_compliance', 'true', 'boolean'),

('publishing-production', 'enabled', 'true', 'boolean'),
('publishing-production', 'auto_formatting', 'false', 'boolean'),
('publishing-production', 'visual_generation', 'true', 'boolean'),
('publishing-production', 'multi_channel_distribution', 'false', 'boolean'),
('publishing-production', 'regulatory_reporting', 'true', 'boolean'),

('analytics-monitoring', 'enabled', 'true', 'boolean'),
('analytics-monitoring', 'performance_analytics', 'true', 'boolean'),
('analytics-monitoring', 'trend_forecasting', 'true', 'boolean'),
('analytics-monitoring', 'strategic_insights', 'true', 'boolean'),
('analytics-monitoring', 'continuous_learning', 'true', 'boolean');

-- Insert default workflow rules
INSERT IGNORE INTO skz_workflow_rules (rule_name, workflow_type, conditions, actions, priority) VALUES
('Auto Quality Check on Submission', 'submission', 
 '{"trigger": "submission_complete", "file_types": ["pdf", "doc", "docx"]}',
 '{"agents": ["content-quality"], "auto_approve": false, "notify_editor": true}', 
 10),

('Research Discovery on Upload', 'submission',
 '{"trigger": "file_upload", "section": "research"}',
 '{"agents": ["research-discovery"], "extract_keywords": true, "check_patents": true}',
 5),

('Reviewer Matching for Reviews', 'review',
 '{"trigger": "review_assignment_needed", "min_reviewers": 2}',
 '{"agents": ["review-coordination"], "suggest_reviewers": true, "check_availability": true}',
 8),

('Production Formatting on Accept', 'production',
 '{"trigger": "manuscript_accepted", "status": "accepted"}',
 '{"agents": ["publishing-production"], "format_content": true, "generate_visuals": false}',
 7);

-- Create views for reporting and analytics
CREATE OR REPLACE VIEW skz_agent_performance_summary AS
SELECT 
    agent_name,
    COUNT(*) as total_communications,
    AVG(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100 as success_rate,
    AVG(execution_time_ms) as avg_execution_time,
    AVG(request_size) as avg_request_size,
    AVG(response_size) as avg_response_size,
    DATE(timestamp) as date
FROM skz_agent_communications 
GROUP BY agent_name, DATE(timestamp)
ORDER BY date DESC, agent_name;

CREATE OR REPLACE VIEW skz_workflow_efficiency AS
SELECT 
    workflow_type,
    agent_name,
    COUNT(*) as total_workflows,
    AVG(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100 as completion_rate,
    AVG(TIMESTAMPDIFF(MINUTE, created_at, completed_at)) as avg_completion_time_minutes,
    DATE(created_at) as date
FROM skz_workflow_automation 
WHERE completed_at IS NOT NULL
GROUP BY workflow_type, agent_name, DATE(created_at)
ORDER BY date DESC, workflow_type, agent_name;

-- Add indexes for better performance
CREATE INDEX idx_communications_composite ON skz_agent_communications (agent_name, timestamp, success);
CREATE INDEX idx_workflow_composite ON skz_workflow_automation (workflow_type, status, created_at);
CREATE INDEX idx_metrics_composite ON skz_agent_metrics (agent_name, metric_type, recorded_at);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS skz_schema_version (
    version VARCHAR(20) PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    description TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO skz_schema_version (version, description) VALUES 
('1.0.0', 'Initial SKZ Agents Integration schema with core tables for agent state management, communication logging, workflow automation, metrics, settings, and rules.');

-- Trigger to update metrics when communications are logged
DELIMITER //
CREATE TRIGGER update_agent_metrics_on_communication
    AFTER INSERT ON skz_agent_communications
    FOR EACH ROW
BEGIN
    -- Update success rate metric
    INSERT INTO skz_agent_metrics (agent_name, metric_name, metric_value, metric_type)
    VALUES (NEW.agent_name, 'latest_success_rate', 
            CASE WHEN NEW.success = 1 THEN 100.0 ELSE 0.0 END, 
            'success_rate')
    ON DUPLICATE KEY UPDATE 
        metric_value = CASE WHEN NEW.success = 1 THEN 100.0 ELSE 0.0 END,
        recorded_at = NOW();
        
    -- Update response time metric
    IF NEW.execution_time_ms > 0 THEN
        INSERT INTO skz_agent_metrics (agent_name, metric_name, metric_value, metric_type)
        VALUES (NEW.agent_name, 'latest_response_time', NEW.execution_time_ms, 'response_time')
        ON DUPLICATE KEY UPDATE 
            metric_value = NEW.execution_time_ms,
            recorded_at = NOW();
    END IF;
END//
DELIMITER ;

COMMIT;