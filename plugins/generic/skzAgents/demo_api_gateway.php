#!/usr/bin/env php
<?php

/**
 * SKZ API Gateway Demo Script
 * Demonstrates the API gateway functionality with example requests
 */

echo "=== SKZ API Gateway Demo ===\n\n";

// Demo configuration
$baseUrl = 'http://localhost/ojs'; // Update this to your OJS installation URL
$apiKey = 'demo-api-key-12345';   // Update this to your configured API key

// Function to make API requests
function makeApiRequest($url, $data = null, $method = 'GET') {
    global $apiKey;
    
    $headers = array(
        'Content-Type: application/json',
        'X-API-Key: ' . $apiKey,
        'User-Agent: SKZ-Gateway-Demo/1.0'
    );
    
    $curl = curl_init();
    curl_setopt_array($curl, array(
        CURLOPT_URL => $url,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => 30,
        CURLOPT_HTTPHEADER => $headers,
        CURLOPT_SSL_VERIFYPEER => false,
        CURLOPT_CUSTOMREQUEST => $method
    ));
    
    if ($method === 'POST' && $data) {
        curl_setopt($curl, CURLOPT_POSTFIELDS, json_encode($data));
    }
    
    $response = curl_exec($curl);
    $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
    $error = curl_error($curl);
    curl_close($curl);
    
    return array(
        'response' => $response,
        'http_code' => $httpCode,
        'error' => $error
    );
}

// Demo 1: Check API Gateway Status
echo "1. Testing API Gateway Status\n";
echo "URL: {$baseUrl}/index.php/context/skzAgents/api/status\n";

$result = makeApiRequest("{$baseUrl}/index.php/context/skzAgents/api/status");

if ($result['error']) {
    echo "❌ Connection Error: " . $result['error'] . "\n";
} else {
    echo "📡 HTTP Code: " . $result['http_code'] . "\n";
    echo "📄 Response: " . $result['response'] . "\n";
}

echo "\n" . str_repeat("-", 60) . "\n\n";

// Demo 2: Research Discovery Agent - Literature Search
echo "2. Testing Research Discovery Agent - Literature Search\n";
echo "URL: {$baseUrl}/index.php/context/skzAgents/api/research-discovery/literature_search\n";

$searchData = array(
    'query' => 'cosmetic science safety assessment',
    'filters' => array(
        'year_range' => '2020-2024',
        'journal_type' => 'peer_reviewed',
        'language' => 'english'
    ),
    'limit' => 10
);

echo "📋 Request Data: " . json_encode($searchData, JSON_PRETTY_PRINT) . "\n";

$result = makeApiRequest(
    "{$baseUrl}/index.php/context/skzAgents/api/research-discovery/literature_search",
    $searchData,
    'POST'
);

if ($result['error']) {
    echo "❌ Connection Error: " . $result['error'] . "\n";
} else {
    echo "📡 HTTP Code: " . $result['http_code'] . "\n";
    echo "📄 Response: " . $result['response'] . "\n";
}

echo "\n" . str_repeat("-", 60) . "\n\n";

// Demo 3: Submission Assistant Agent - Quality Assessment
echo "3. Testing Submission Assistant Agent - Quality Assessment\n";
echo "URL: {$baseUrl}/index.php/context/skzAgents/api/submission-assistant/quality_assessment\n";

$qualityData = array(
    'manuscript_content' => 'This is a sample manuscript about cosmetic formulation safety...',
    'metadata' => array(
        'title' => 'Safety Assessment of Novel Cosmetic Formulations',
        'keywords' => array('cosmetics', 'safety', 'formulation', 'dermatology'),
        'research_type' => 'experimental'
    ),
    'assessment_criteria' => array(
        'scientific_rigor' => true,
        'methodology_soundness' => true,
        'data_quality' => true,
        'ethical_compliance' => true
    )
);

echo "📋 Request Data: " . json_encode($qualityData, JSON_PRETTY_PRINT) . "\n";

$result = makeApiRequest(
    "{$baseUrl}/index.php/context/skzAgents/api/submission-assistant/quality_assessment",
    $qualityData,
    'POST'
);

if ($result['error']) {
    echo "❌ Connection Error: " . $result['error'] . "\n";
} else {
    echo "📡 HTTP Code: " . $result['http_code'] . "\n";
    echo "📄 Response: " . $result['response'] . "\n";
}

echo "\n" . str_repeat("-", 60) . "\n\n";

// Demo 4: Webhook Registration
echo "4. Testing Webhook Registration\n";
echo "URL: {$baseUrl}/index.php/context/skzAgents/api/webhook/register\n";

$webhookData = array(
    'event' => 'agent.task.completed',
    'callback_url' => 'https://your-domain.com/webhook/handler',
    'description' => 'Notification when agent tasks are completed',
    'active' => true
);

echo "📋 Request Data: " . json_encode($webhookData, JSON_PRETTY_PRINT) . "\n";

$result = makeApiRequest(
    "{$baseUrl}/index.php/context/skzAgents/api/webhook/register",
    $webhookData,
    'POST'
);

if ($result['error']) {
    echo "❌ Connection Error: " . $result['error'] . "\n";
} else {
    echo "📡 HTTP Code: " . $result['http_code'] . "\n";
    echo "📄 Response: " . $result['response'] . "\n";
}

echo "\n" . str_repeat("-", 60) . "\n\n";

// Demo 5: Analytics Monitoring Agent - Performance Analytics
echo "5. Testing Analytics Monitoring Agent - Performance Analytics\n";
echo "URL: {$baseUrl}/index.php/context/skzAgents/api/analytics-monitoring/performance_analytics\n";

$analyticsData = array(
    'metrics_type' => 'system_performance',
    'time_range' => '7_days',
    'components' => array(
        'gateway_performance',
        'agent_response_times',
        'error_rates',
        'throughput_metrics'
    ),
    'aggregation' => 'hourly'
);

echo "📋 Request Data: " . json_encode($analyticsData, JSON_PRETTY_PRINT) . "\n";

$result = makeApiRequest(
    "{$baseUrl}/index.php/context/skzAgents/api/analytics-monitoring/performance_analytics",
    $analyticsData,
    'POST'
);

if ($result['error']) {
    echo "❌ Connection Error: " . $result['error'] . "\n";
} else {
    echo "📡 HTTP Code: " . $result['http_code'] . "\n";
    echo "📄 Response: " . $result['response'] . "\n";
}

echo "\n" . str_repeat("=", 60) . "\n\n";

// Summary
echo "📊 Demo Summary:\n";
echo "• API Gateway Status Check\n";
echo "• Research Discovery Agent Demo\n";
echo "• Submission Assistant Agent Demo\n";
echo "• Webhook Registration Demo\n";
echo "• Analytics Monitoring Agent Demo\n\n";

echo "🔧 Configuration Notes:\n";
echo "• Update \$baseUrl to your OJS installation URL\n";
echo "• Update \$apiKey to your configured API key\n";
echo "• Ensure SKZ agents framework is running\n";
echo "• Ensure OJS SKZ plugin is enabled\n\n";

echo "📚 For more information:\n";
echo "• See API_GATEWAY_DOCUMENTATION.md\n";
echo "• Run test_gateway_config.php for configuration validation\n";
echo "• Check OJS logs for detailed error information\n\n";

echo "=== Demo Complete ===\n";

?>