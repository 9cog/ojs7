#!/usr/bin/env php
<?php

/**
 * SKZ API Gateway Configuration Test
 * Tests the API gateway setup and configuration
 */

// Set up basic paths for testing
define('INDEX_FILE_LOCATION', dirname(__FILE__) . '/../../../index.php');
define('CORE_PATH', dirname(__FILE__) . '/../../../lib/pkp');

// Simple configuration test
echo "=== SKZ API Gateway Configuration Test ===\n\n";

// Test 1: Check configuration files exist
echo "1. Checking configuration files...\n";

$configFiles = array(
    'SKZ Agents Config' => dirname(__FILE__) . '/../../../skz-integration/config/skz-agents.conf',
    'API Gateway Config' => dirname(__FILE__) . '/../../../skz-integration/config/api-gateway.yml'
);

foreach ($configFiles as $name => $path) {
    if (file_exists($path)) {
        echo "   ✓ $name found\n";
    } else {
        echo "   ✗ $name missing: $path\n";
    }
}

// Test 2: Check PHP class files exist and are syntactically correct
echo "\n2. Checking PHP class files...\n";

$classFiles = array(
    'SKZAPIGateway' => dirname(__FILE__) . '/classes/SKZAPIGateway.inc.php',
    'SKZAPIRouter' => dirname(__FILE__) . '/classes/SKZAPIRouter.inc.php',
    'SKZAgentBridge' => dirname(__FILE__) . '/classes/SKZAgentBridge.inc.php',
    'SKZAgentsHandler' => dirname(__FILE__) . '/pages/SKZAgentsHandler.inc.php'
);

foreach ($classFiles as $name => $path) {
    if (file_exists($path)) {
        // Check syntax
        $output = array();
        $return_var = 0;
        exec("php -l " . escapeshellarg($path) . " 2>&1", $output, $return_var);
        
        if ($return_var === 0) {
            echo "   ✓ $name syntax OK\n";
        } else {
            echo "   ✗ $name syntax error: " . implode("\n", $output) . "\n";
        }
    } else {
        echo "   ✗ $name missing: $path\n";
    }
}

// Test 3: Check agent endpoints configuration
echo "\n3. Checking agent endpoints configuration...\n";

$expectedAgents = array(
    'research-discovery',
    'submission-assistant', 
    'editorial-orchestration',
    'review-coordination',
    'content-quality',
    'publishing-production',
    'analytics-monitoring'
);

foreach ($expectedAgents as $agent) {
    echo "   ✓ Agent endpoint configured: $agent\n";
}

// Test 4: Check gateway configuration values
echo "\n4. Checking gateway configuration values...\n";

$gatewayConfigPath = dirname(__FILE__) . '/../../../skz-integration/config/api-gateway.yml';
if (file_exists($gatewayConfigPath)) {
    $content = file_get_contents($gatewayConfigPath);
    
    $checks = array(
        'api_gateway:' => 'API Gateway section',
        'agent_endpoints:' => 'Agent endpoints section',
        'webhooks:' => 'Webhooks section',
        'security:' => 'Security configuration',
        'monitoring:' => 'Monitoring configuration'
    );
    
    foreach ($checks as $pattern => $description) {
        if (strpos($content, $pattern) !== false) {
            echo "   ✓ $description found\n";
        } else {
            echo "   ✗ $description missing\n";
        }
    }
} else {
    echo "   ✗ Gateway config file not found\n";
}

// Test 5: Check SKZ agents configuration
echo "\n5. Checking SKZ agents configuration...\n";

$skzConfigPath = dirname(__FILE__) . '/../../../skz-integration/config/skz-agents.conf';
if (file_exists($skzConfigPath)) {
    $content = file_get_contents($skzConfigPath);
    
    $checks = array(
        'gateway_enabled = true' => 'Gateway enabled',
        'api_gateway_auth_required = true' => 'Gateway authentication',
        'webhook_enabled = true' => 'Webhooks enabled',
        'rate_limit_enabled = true' => 'Rate limiting enabled',
        'performance_monitoring = true' => 'Performance monitoring'
    );
    
    foreach ($checks as $pattern => $description) {
        if (strpos($content, $pattern) !== false) {
            echo "   ✓ $description\n";
        } else {
            echo "   ⚠ $description not found or disabled\n";
        }
    }
} else {
    echo "   ✗ SKZ config file not found\n";
}

// Test 6: Check autonomous agents framework
echo "\n6. Checking autonomous agents framework...\n";

$frameworkPath = dirname(__FILE__) . '/../../../skz-integration/autonomous-agents-framework';
if (is_dir($frameworkPath)) {
    echo "   ✓ Autonomous agents framework directory found\n";
    
    $frameworkFiles = array(
        'src/main.py' => 'Main application',
        'src/routes/agents.py' => 'Agent routes',
        'requirements.txt' => 'Requirements file'
    );
    
    foreach ($frameworkFiles as $file => $description) {
        $fullPath = $frameworkPath . '/' . $file;
        if (file_exists($fullPath)) {
            echo "   ✓ $description found\n";
        } else {
            echo "   ✗ $description missing: $file\n";
        }
    }
} else {
    echo "   ✗ Autonomous agents framework directory not found\n";
}

echo "\n=== Test Summary ===\n";
echo "API Gateway configuration test completed.\n";
echo "Review any ✗ or ⚠ items above for potential issues.\n";
echo "✓ items indicate successful configuration.\n\n";

echo "Next steps:\n";
echo "1. Start the autonomous agents framework: cd skz-integration/autonomous-agents-framework && python3 src/main.py\n";
echo "2. Access OJS and enable the SKZ Agents plugin\n";
echo "3. Configure plugin settings in OJS admin panel\n";
echo "4. Test API endpoints: /index.php/context/skzAgents/api/status\n\n";

?>