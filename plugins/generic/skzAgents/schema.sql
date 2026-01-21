-- SKZ Agents Plugin Database Schema
-- This file contains the database tables needed for the SKZ Agents plugin

-- Table for storing agent states
CREATE TABLE IF NOT EXISTS skz_agent_states (
    agent_id VARCHAR(50) PRIMARY KEY,
    state_data JSON,
    last_updated DATETIME,
    submission_id INT,
    INDEX idx_submission_id (submission_id),
    INDEX idx_last_updated (last_updated)
);

-- Table for storing agent communication logs
CREATE TABLE IF NOT EXISTS skz_agent_communications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    agent_from VARCHAR(50),
    agent_to VARCHAR(50),
    agent_name VARCHAR(50),
    action VARCHAR(50),
    message_type VARCHAR(50),
    payload JSON,
    timestamp DATETIME,
    success BOOLEAN DEFAULT TRUE,
    response_time FLOAT DEFAULT 0,
    request_size INT DEFAULT 0,
    response_size INT DEFAULT 0,
    INDEX idx_agent_name (agent_name),
    INDEX idx_timestamp (timestamp),
    INDEX idx_success (success)
);

-- Table for storing workflow automation data
CREATE TABLE IF NOT EXISTS skz_workflow_automation (
    id INT AUTO_INCREMENT PRIMARY KEY,
    submission_id INT NOT NULL,
    workflow_type VARCHAR(50) NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    automation_data JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',
    INDEX idx_submission (submission_id),
    INDEX idx_workflow (workflow_type),
    INDEX idx_agent (agent_name),
    INDEX idx_status (status),
    INDEX idx_created (created_at)
);

-- Initial data for testing
INSERT IGNORE INTO skz_agent_states (agent_id, state_data, last_updated) VALUES 
('test_agent', '{"status": "initialized", "version": "1.0.0"}', NOW());