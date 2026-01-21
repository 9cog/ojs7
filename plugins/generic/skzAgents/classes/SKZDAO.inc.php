<?php

/**
 * @file plugins/generic/skzAgents/classes/SKZDAO.inc.php
 *
 * SKZ DAO - Data Access Object for SKZ agent framework data
 */

import('lib.pkp.classes.db.DAO');

class SKZDAO extends DAO {
    
    /**
     * Create agent state record
     * @param string $agentId Agent identifier
     * @param array $stateData Agent state data
     * @param int $submissionId Submission ID (optional)
     * @return bool Success status
     */
    public function createAgentState($agentId, $stateData, $submissionId = null) {
        $result = $this->update(
            'INSERT INTO skz_agent_states (agent_id, state_data, last_updated, submission_id) VALUES (?, ?, ?, ?)',
            array(
                $agentId,
                json_encode($stateData),
                Core::getCurrentDate(),
                $submissionId
            )
        );
        
        return $result !== false;
    }
    
    /**
     * Update agent state record
     * @param string $agentId Agent identifier
     * @param array $stateData Agent state data
     * @param int $submissionId Submission ID (optional)
     * @return bool Success status
     */
    public function updateAgentState($agentId, $stateData, $submissionId = null) {
        $sql = 'UPDATE skz_agent_states SET state_data = ?, last_updated = ?';
        $params = array(json_encode($stateData), Core::getCurrentDate());
        
        if ($submissionId !== null) {
            $sql .= ', submission_id = ?';
            $params[] = $submissionId;
        }
        
        $sql .= ' WHERE agent_id = ?';
        $params[] = $agentId;
        
        $result = $this->update($sql, $params);
        return $result !== false;
    }
    
    /**
     * Get agent state by agent ID
     * @param string $agentId Agent identifier
     * @return array|null Agent state data
     */
    public function getAgentState($agentId) {
        $result = $this->retrieve(
            'SELECT * FROM skz_agent_states WHERE agent_id = ?',
            array($agentId)
        );
        
        if ($result->RecordCount() == 0) {
            $result->Close();
            return null;
        }
        
        $row = $result->getRowAssoc(false);
        $result->Close();
        
        $row['state_data'] = json_decode($row['state_data'], true);
        return $row;
    }
    
    /**
     * Get agent states by submission ID
     * @param int $submissionId Submission ID
     * @return array Agent states
     */
    public function getAgentStatesBySubmission($submissionId) {
        $result = $this->retrieve(
            'SELECT * FROM skz_agent_states WHERE submission_id = ? ORDER BY last_updated DESC',
            array($submissionId)
        );
        
        $states = array();
        while (!$result->EOF) {
            $row = $result->getRowAssoc(false);
            $row['state_data'] = json_decode($row['state_data'], true);
            $states[] = $row;
            $result->MoveNext();
        }
        
        $result->Close();
        return $states;
    }
    
    /**
     * Delete agent state
     * @param string $agentId Agent identifier
     * @return bool Success status
     */
    public function deleteAgentState($agentId) {
        $result = $this->update(
            'DELETE FROM skz_agent_states WHERE agent_id = ?',
            array($agentId)
        );
        
        return $result !== false;
    }
    
    /**
     * Log agent communication
     * @param string $agentFrom Source agent
     * @param string $agentTo Target agent
     * @param string $messageType Message type
     * @param array $payload Message payload
     * @return bool Success status
     */
    public function logAgentCommunication($agentFrom, $agentTo, $messageType, $payload) {
        $result = $this->update(
            'INSERT INTO skz_agent_communications (agent_from, agent_to, message_type, payload, timestamp) VALUES (?, ?, ?, ?, ?)',
            array(
                $agentFrom,
                $agentTo,
                $messageType,
                json_encode($payload),
                Core::getCurrentDate()
            )
        );
        
        return $result !== false;
    }
    
    /**
     * Get agent communication logs
     * @param string $agentName Optional agent name filter
     * @param int $limit Result limit
     * @param int $offset Result offset
     * @return array Communication logs
     */
    public function getAgentCommunications($agentName = null, $limit = 50, $offset = 0) {
        $sql = 'SELECT * FROM skz_agent_communications';
        $params = array();
        
        if ($agentName) {
            $sql .= ' WHERE agent_from = ? OR agent_to = ?';
            $params = array($agentName, $agentName);
        }
        
        $sql .= ' ORDER BY timestamp DESC LIMIT ? OFFSET ?';
        $params[] = $limit;
        $params[] = $offset;
        
        $result = $this->retrieve($sql, $params);
        
        $communications = array();
        while (!$result->EOF) {
            $row = $result->getRowAssoc(false);
            $row['payload'] = json_decode($row['payload'], true);
            $communications[] = $row;
            $result->MoveNext();
        }
        
        $result->Close();
        return $communications;
    }
    
    /**
     * Get agent performance metrics
     * @param string $agentName Optional agent name filter
     * @param string $startDate Start date for metrics (YYYY-MM-DD)
     * @param string $endDate End date for metrics (YYYY-MM-DD)
     * @return array Performance metrics
     */
    public function getAgentPerformanceMetrics($agentName = null, $startDate = null, $endDate = null) {
        $sql = 'SELECT 
                    agent_name,
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests,
                    AVG(response_time) as avg_response_time,
                    MAX(response_time) as max_response_time,
                    MIN(response_time) as min_response_time,
                    AVG(request_size) as avg_request_size,
                    AVG(response_size) as avg_response_size
                FROM skz_agent_communications 
                WHERE 1=1';
        
        $params = array();
        
        if ($agentName) {
            $sql .= ' AND agent_name = ?';
            $params[] = $agentName;
        }
        
        if ($startDate) {
            $sql .= ' AND timestamp >= ?';
            $params[] = $startDate . ' 00:00:00';
        }
        
        if ($endDate) {
            $sql .= ' AND timestamp <= ?';
            $params[] = $endDate . ' 23:59:59';
        }
        
        $sql .= ' GROUP BY agent_name';
        
        $result = $this->retrieve($sql, $params);
        
        $metrics = array();
        while (!$result->EOF) {
            $row = $result->getRowAssoc(false);
            $row['success_rate'] = $row['total_requests'] > 0 ? 
                ($row['successful_requests'] / $row['total_requests']) * 100 : 0;
            $metrics[] = $row;
            $result->MoveNext();
        }
        
        $result->Close();
        return $metrics;
    }
    
    /**
     * Clean up old agent communications (for maintenance)
     * @param int $daysToKeep Number of days to keep
     * @return int Number of records deleted
     */
    public function cleanupOldCommunications($daysToKeep = 30) {
        $cutoffDate = date('Y-m-d H:i:s', strtotime("-{$daysToKeep} days"));
        
        $result = $this->update(
            'DELETE FROM skz_agent_communications WHERE timestamp < ?',
            array($cutoffDate)
        );
        
        return $result;
    }
    
    /**
     * Get agent workflow summary
     * @param int $submissionId Submission ID
     * @return array Workflow summary
     */
    public function getWorkflowSummary($submissionId) {
        $sql = 'SELECT 
                    ac.agent_name,
                    ac.action,
                    ac.timestamp,
                    ac.success,
                    ags.state_data
                FROM skz_agent_communications ac
                LEFT JOIN skz_agent_states ags ON ac.agent_name = ags.agent_id
                WHERE ags.submission_id = ?
                ORDER BY ac.timestamp ASC';
        
        $result = $this->retrieve($sql, array($submissionId));
        
        $summary = array();
        while (!$result->EOF) {
            $row = $result->getRowAssoc(false);
            if ($row['state_data']) {
                $row['state_data'] = json_decode($row['state_data'], true);
            }
            $summary[] = $row;
            $result->MoveNext();
        }
        
        $result->Close();
        return $summary;
    }
    
    /**
     * Check if database tables exist
     * @return bool Tables exist status
     */
    public function tablesExist() {
        $result = $this->retrieve("SHOW TABLES LIKE 'skz_%'");
        $count = $result->RecordCount();
        $result->Close();
        
        return $count >= 2; // Expecting at least 2 tables
    }
    
    /**
     * Install database schema
     * @return bool Installation success
     */
    public function installSchema() {
        $sqls = array(
            "CREATE TABLE IF NOT EXISTS skz_agent_states (
                agent_id VARCHAR(50) PRIMARY KEY,
                state_data JSON,
                last_updated DATETIME,
                submission_id INT,
                INDEX idx_submission_id (submission_id),
                INDEX idx_last_updated (last_updated)
            )",
            
            "CREATE TABLE IF NOT EXISTS skz_agent_communications (
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
            )"
        );
        
        foreach ($sqls as $sql) {
            $result = $this->update($sql);
            if ($result === false) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Uninstall database schema
     * @return bool Uninstallation success
     */
    public function uninstallSchema() {
        $sqls = array(
            "DROP TABLE IF EXISTS skz_agent_communications",
            "DROP TABLE IF EXISTS skz_agent_states"
        );
        
        foreach ($sqls as $sql) {
            $result = $this->update($sql);
            if ($result === false) {
                return false;
            }
        }
        
        return true;
    }
}

?>