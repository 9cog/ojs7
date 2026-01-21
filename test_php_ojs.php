<?php
/**
 * OJS7 PHP Installation Verification Test
 * Tests all essential PHP components for OJS with SKZ Agents
 */

echo "=== OJS7 PHP Installation Verification ===\n\n";

// PHP Version Check
echo "PHP Version: " . PHP_VERSION . "\n";
echo "PHP SAPI: " . php_sapi_name() . "\n\n";

// Memory and Execution Limits
echo "=== Resource Limits ===\n";
echo "Memory Limit: " . ini_get('memory_limit') . "\n";
echo "Max Execution Time: " . ini_get('max_execution_time') . "s\n";
echo "Upload Max Filesize: " . ini_get('upload_max_filesize') . "\n";
echo "Post Max Size: " . ini_get('post_max_size') . "\n\n";

// Essential Extensions for OJS
echo "=== Essential OJS Extensions ===\n";
$required_extensions = [
    'mysqli' => 'MySQL Database Support',
    'pdo_mysql' => 'PDO MySQL Support', 
    'pgsql' => 'PostgreSQL Support',
    'pdo_pgsql' => 'PDO PostgreSQL Support',
    'curl' => 'HTTP Client Support',
    'openssl' => 'SSL/TLS Support',
    'mbstring' => 'Multibyte String Support',
    'intl' => 'Internationalization Support',
    'gd' => 'Image Processing Support',
    'zip' => 'Archive Support',
    'fileinfo' => 'File Type Detection',
    'xml' => 'XML Processing',
    'dom' => 'DOM Manipulation',
    'json' => 'JSON Support',
    'session' => 'Session Management',
    'filter' => 'Data Filtering',
    'hash' => 'Hashing Functions',
    'opcache' => 'Performance Optimization'
];

$missing_extensions = [];
foreach ($required_extensions as $ext => $description) {
    if (extension_loaded($ext)) {
        echo "✓ $ext - $description\n";
    } else {
        echo "✗ $ext - $description (MISSING)\n";
        $missing_extensions[] = $ext;
    }
}

echo "\n=== Database Connectivity Tests ===\n";

// Test MySQL/MariaDB connectivity capability
if (extension_loaded('mysqli')) {
    echo "✓ MySQLi extension available\n";
    echo "  MySQL Client Version: " . mysqli_get_client_info() . "\n";
} else {
    echo "✗ MySQLi extension not available\n";
}

// Test PostgreSQL connectivity capability
if (extension_loaded('pgsql')) {
    echo "✓ PostgreSQL extension available\n";
    if (function_exists('pg_version')) {
        echo "  PostgreSQL Client Version: Available\n";
    }
} else {
    echo "✗ PostgreSQL extension not available\n";
}

echo "\n=== Internationalization Support ===\n";
if (extension_loaded('intl')) {
    echo "✓ Intl extension loaded\n";
    echo "  ICU Version: " . INTL_ICU_VERSION . "\n";
    echo "  ICU Data Version: " . INTL_ICU_DATA_VERSION . "\n";
    
    // Test locale support
    $locales = ['en_US', 'fr_FR', 'de_DE', 'es_ES', 'pt_BR', 'zh_CN'];
    echo "  Supported Locales: ";
    foreach ($locales as $locale) {
        if (class_exists('Locale') && Locale::acceptFromHttp($locale)) {
            echo "$locale ";
        }
    }
    echo "\n";
} else {
    echo "✗ Intl extension not available\n";
}

echo "\n=== File Processing Capabilities ===\n";
if (extension_loaded('gd')) {
    echo "✓ GD extension loaded\n";
    $gd_info = gd_info();
    echo "  GD Version: " . $gd_info['GD Version'] . "\n";
    echo "  JPEG Support: " . ($gd_info['JPEG Support'] ? 'Yes' : 'No') . "\n";
    echo "  PNG Support: " . ($gd_info['PNG Support'] ? 'Yes' : 'No') . "\n";
    echo "  WebP Support: " . ($gd_info['WebP Support'] ? 'Yes' : 'No') . "\n";
} else {
    echo "✗ GD extension not available\n";
}

if (extension_loaded('zip')) {
    echo "✓ ZIP extension loaded\n";
    echo "  ZIP Version: " . phpversion('zip') . "\n";
} else {
    echo "✗ ZIP extension not available\n";
}

echo "\n=== Security & Encryption ===\n";
if (extension_loaded('openssl')) {
    echo "✓ OpenSSL extension loaded\n";
    echo "  OpenSSL Version: " . OPENSSL_VERSION_TEXT . "\n";
} else {
    echo "✗ OpenSSL extension not available\n";
}

if (extension_loaded('sodium')) {
    echo "✓ Sodium extension loaded (Modern Cryptography)\n";
    echo "  Sodium Version: " . phpversion('sodium') . "\n";
} else {
    echo "✗ Sodium extension not available\n";
}

echo "\n=== Performance Optimization ===\n";
if (extension_loaded('opcache')) {
    echo "✓ OPcache extension loaded\n";
    $opcache_status = opcache_get_status(false);
    if ($opcache_status !== false) {
        echo "  OPcache Enabled: " . ($opcache_status['opcache_enabled'] ? 'Yes' : 'No') . "\n";
        echo "  Memory Usage: " . round($opcache_status['memory_usage']['used_memory'] / 1024 / 1024, 2) . " MB\n";
    }
} else {
    echo "✗ OPcache extension not available\n";
}

echo "\n=== Directory Permissions ===\n";
$directories = [
    'D:\casto\oj7\logs' => 'Log Directory',
    'D:\casto\oj7\tmp' => 'Temporary Files',
    'D:\casto\oj7\tmp\sessions' => 'Session Storage'
];

foreach ($directories as $dir => $description) {
    if (is_dir($dir)) {
        $writable = is_writable($dir);
        echo ($writable ? "✓" : "✗") . " $description ($dir) - " . ($writable ? "Writable" : "Not Writable") . "\n";
    } else {
        echo "✗ $description ($dir) - Directory Missing\n";
    }
}

echo "\n=== Configuration Summary ===\n";
echo "Configuration File: " . php_ini_loaded_file() . "\n";
echo "Error Log: " . ini_get('error_log') . "\n";
echo "Session Save Path: " . session_save_path() . "\n";
echo "Upload Temp Dir: " . ini_get('upload_tmp_dir') . "\n";
echo "Default Timezone: " . date_default_timezone_get() . "\n";

echo "\n=== Test Results ===\n";
if (empty($missing_extensions)) {
    echo "🎉 SUCCESS: All essential extensions are loaded!\n";
    echo "✅ PHP installation is ready for OJS7 with SKZ Agents\n";
} else {
    echo "⚠️  WARNING: Missing extensions: " . implode(', ', $missing_extensions) . "\n";
    echo "❌ PHP installation needs attention before running OJS7\n";
}

echo "\n=== Next Steps ===\n";
echo "1. Verify database connectivity with actual credentials\n";
echo "2. Test OJS7 installation with this PHP setup\n";
echo "3. Configure web server (Apache/Nginx) to use this PHP\n";
echo "4. Test SKZ agents integration endpoints\n";

echo "\nTest completed at: " . date('Y-m-d H:i:s T') . "\n";
?>