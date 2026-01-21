# Quick PHP Installation for OJS Integration

## Manual Installation (No Admin Required)

### Step 1: Download PHP
1. Go to https://windows.php.net/download/
2. Download "PHP 8.1+ Thread Safe (x64)" ZIP file
3. Extract to `D:\casto\oj7\php\` (create this directory)

### Step 2: Configure PHP
```cmd
# Copy configuration file
copy D:\casto\oj7\php\php.ini-development D:\casto\oj7\php\php.ini
```

### Step 3: Add to PATH (Current Session)
```cmd
set PATH=%PATH%;D:\casto\oj7\php
```

### Step 4: Test Installation
```cmd
php --version
```

## Required Extensions for OJS

Edit `D:\casto\oj7\php\php.ini` and uncomment these lines:
```ini
extension=curl
extension=gd
extension=intl
extension=mbstring
extension=mysqli
extension=pdo_mysql
extension=zip
extension=json
extension=xml
extension=openssl
extension=fileinfo
```

## Quick Test Commands

```cmd
# Test PHP
php -r "echo 'PHP Working: ' . PHP_VERSION . PHP_EOL;"

# Test required extensions
php -r "
\$required = ['curl', 'gd', 'intl', 'mbstring', 'mysqli', 'zip'];
foreach(\$required as \$ext) {
    echo \$ext . ': ' . (extension_loaded(\$ext) ? 'OK' : 'MISSING') . PHP_EOL;
}
"
```

## Resume OJS Integration Workflow

Once PHP is working:
```cmd
# Test PHP bridge
php plugins/generic/skzAgents/tests/test_bridge.php

# Start Python API server  
cd skz-integration/autonomous-agents-framework/src && python simple_api_server.py 5000

# Run validation
bash validate_implementation.sh

# Deploy to OJS
php plugins/generic/skzAgents/deploy.php
```
