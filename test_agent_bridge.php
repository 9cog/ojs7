<?php

/**
 * Test script to verify SKZ Agent Bridge integration
 * Tests communication between OJS and the autonomous agents framework
 */

// Minimal setup for testing outside full OJS context
define('DIRECTORY_SEPARATOR', '/');
define('PKP_LIB_PATH', __DIR__ . DIRECTORY_SEPARATOR . 'lib' . DIRECTORY_SEPARATOR . 'pkp');

// Mock the required classes for testing
class Application {
    public static function getVersion() {
        return '3.4.0';
    }
    
    public static function getRequest() {
        return new MockRequest();
    }
}

class MockRequest {
    public function getContext() {
        return new MockContext();
    }
    
    public function getUser() {
        return new MockUser();
    }
}

class MockUser {
    public function getId() {
        return 1;
    }
    
    public function getUsername() {
        return 'test_user';
    }
}

class MockContext {
    public function getId() {
        return 1;
    }
    
    public function getLocalizedName() {
        return 'Test Journal';
    }
}

class Config {
    private static $config = array();
    
    public static function getVar($section, $key, $default = null) {
        return isset(self::$config[$section][$key]) ? self::$config[$section][$key] : $default;
    }
    
    public static function setVar($section, $key, $value) {
        self::$config[$section][$key] = $value;
    }
}

class PluginRegistry {
    public static function getPlugin($category, $name) {
        return null; // Will use defaults from Config
    }
}

class DAORegistry {
    public static function getDAO($name) {
        return new MockDAO();
    }
}

class MockDAO {
    public function insertLogEntry($data) {
        return true;
    }
    
    public function logAgentCommunication($data) {
        return true;
    }
}

// Set default config for testing  
// Note: getAgentStatus adds '/status' to base URL, callAgent adds '/agent_name/action'
Config::setVar('skz', 'agent_base_url', 'http://localhost:5000/api/v1/agents');
Config::setVar('skz', 'api_key', 'test-key');
Config::setVar('skz', 'timeout', 30);

// Include the agent bridge class
require_once(__DIR__ . '/plugins/generic/skzAgents/classes/SKZAgentBridge.inc.php');

echo "🤖 SKZ Agent Bridge Integration Test\n";
echo "====================================\n\n";

// Test 1: Initialize Bridge
echo "📡 Test 1: Initializing Agent Bridge...\n";
$bridge = new SKZAgentBridge();
echo "✅ Agent Bridge initialized successfully\n\n";

// Test 2: Test agent status call
echo "📊 Test 2: Getting Agent Status...\n";
try {
    $status = $bridge->getAgentStatus();
    if ($status && isset($status['agents'])) {
        echo "✅ Found " . count($status['agents']) . " active agents\n";
        foreach ($status['agents'] as $agent) {
            if (isset($agent['name']) && isset($agent['status'])) {
                echo "  - " . $agent['name'] . ": " . $agent['status'] . "\n";
            }
        }
    } else {
        echo "❌ No agent status data received\n";
    }
} catch (Exception $e) {
    echo "❌ Error getting agent status: " . $e->getMessage() . "\n";
}
echo "\n";

// Test 3: Test specific agent call
echo "🔬 Test 3: Testing Research Discovery Agent...\n";
try {
    $testData = array(
        'manuscript_id' => 'test-manuscript-001',
        'title' => 'Test Manuscript for Agent Integration',
        'content' => 'This is a test manuscript to verify agent integration.',
        'keywords' => array('autonomous agents', 'academic publishing', 'workflow automation')
    );
    
    $result = $bridge->callAgent('research_discovery', 'action', $testData);
    
    if ($result && isset($result['success']) && $result['success']) {
        echo "✅ Research Discovery Agent responded successfully\n";
        if (isset($result['result'])) {
            echo "  - Processing time: " . (isset($result['processing_time']) ? round($result['processing_time'], 2) . 's' : 'unknown') . "\n";
            if (isset($result['result']['papers_found'])) {
                echo "  - Papers found: " . $result['result']['papers_found'] . "\n";
            }
            if (isset($result['result']['recommendations'])) {
                echo "  - Recommendations: " . count($result['result']['recommendations']) . "\n";
            }
        }
    } else {
        echo "❌ Research Discovery Agent call failed\n";
        if (isset($result['error'])) {
            echo "  Error: " . $result['error'] . "\n";
        }
    }
} catch (Exception $e) {
    echo "❌ Error calling Research Discovery Agent: " . $e->getMessage() . "\n";
}
echo "\n";

// Test 4: Test submission assistant agent
echo "📝 Test 4: Testing Submission Assistant Agent...\n";
try {
    $submissionData = array(
        'manuscript_id' => 'test-submission-001',
        'title' => 'Test Manuscript Submission',
        'abstract' => 'This is a test abstract for manuscript submission validation.',
        'authors' => array('Test Author'),
        'keywords' => array('test', 'submission', 'validation')
    );
    
    $result = $bridge->callAgent('submission_assistant', 'action', $submissionData);
    
    if ($result && isset($result['success']) && $result['success']) {
        echo "✅ Submission Assistant Agent responded successfully\n";
        if (isset($result['processing_time'])) {
            echo "  - Processing time: " . round($result['processing_time'], 2) . "s\n";
        }
    } else {
        echo "❌ Submission Assistant Agent call failed\n";
    }
} catch (Exception $e) {
    echo "❌ Error calling Submission Assistant Agent: " . $e->getMessage() . "\n";
}
echo "\n";

// Test 5: Test all agents status
echo "🌐 Test 5: Testing All Agents Availability...\n";
$agentTypes = array(
    'research_discovery' => 'Research Discovery Agent',
    'submission_assistant' => 'Submission Assistant Agent', 
    'editorial_orchestration' => 'Editorial Orchestration Agent',
    'review_coordination' => 'Review Coordination Agent',
    'content_quality' => 'Content Quality Agent',
    'publishing_production' => 'Publishing Production Agent',
    'analytics_monitoring' => 'Analytics Monitoring Agent'
);

$activeAgents = 0;
foreach ($agentTypes as $type => $name) {
    try {
        $testData = array('test' => true, 'agent_type' => $type);
        $result = $bridge->callAgent($type, 'action', $testData);
        
        if ($result && isset($result['success']) && $result['success']) {
            echo "✅ $name: Active\n";
            $activeAgents++;
        } else {
            echo "❌ $name: Failed to respond\n";
        }
    } catch (Exception $e) {
        echo "❌ $name: Error - " . $e->getMessage() . "\n";
    }
}

echo "\n📊 Results Summary:\n";
echo "================\n";
echo "Total Agents: " . count($agentTypes) . "\n";
echo "Active Agents: $activeAgents\n";
echo "Success Rate: " . round(($activeAgents / count($agentTypes)) * 100, 1) . "%\n";

if ($activeAgents == count($agentTypes)) {
    echo "\n🎉 ALL TESTS PASSED! SKZ Agent Bridge integration is working correctly.\n";
    echo "The 7 autonomous agents are successfully integrated with OJS workflows.\n";
} else {
    echo "\n⚠️  Some agents are not responding. Please check agent framework status.\n";
}

echo "\n";
?>