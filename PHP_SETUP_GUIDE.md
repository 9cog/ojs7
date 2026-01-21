# PHP Setup Guide for SKZ-OJS Integration

## Why PHP is Critical

- **OJS Core**: Open Journal Systems is built entirely in PHP
- **Plugin System**: All OJS plugins (including our SKZ agents plugin) are PHP-based
- **Database Integration**: OJS database operations require PHP
- **Future Updates**: Every OJS update will be in PHP - we must maintain compatibility
- **Bridge Integration**: PHP-Python bridges are essential for agent communication

## Recommended PHP Installation for Windows

### Option 1: PHP for Windows (Recommended)

1. **Download PHP 8.1+ for Windows**
   ```
   https://windows.php.net/download/
   ```
   - Choose "Thread Safe" version
   - Download ZIP file (e.g., php-8.1.x-Win32-vs16-x64.zip)

2. **Installation Steps**
   ```cmd
   # Create PHP directory
   mkdir C:\php
   
   # Extract PHP to C:\php
   # Add C:\php to system PATH
   ```

3. **Configure PHP**
   ```cmd
   # Copy php.ini-development to php.ini
   copy C:\php\php.ini-development C:\php\php.ini
   ```

### Option 2: XAMPP (Full Stack)

1. **Download XAMPP**
   ```
   https://www.apachefriends.org/download.html
   ```

2. **Installation**
   - Install to C:\xampp
   - Add C:\xampp\php to PATH

### Option 3: Chocolatey (Package Manager)

```cmd
# Install Chocolatey first (if not installed)
# Then install PHP
choco install php

# Verify installation
php --version
```

## Required PHP Extensions for OJS

Add these to php.ini:

```ini
# Core extensions for OJS
extension=curl
extension=gd
extension=intl
extension=mbstring
extension=mysqli
extension=pdo_mysql
extension=zip
extension=json
extension=xml
extension=simplexml
extension=dom
extension=xmlwriter
extension=xmlreader

# Additional extensions for SKZ integration
extension=openssl
extension=fileinfo
extension=filter
extension=hash
extension=session
extension=tokenizer
```

## Environment Configuration

### Add to System PATH
```
C:\php
# OR
C:\xampp\php
```

### Verify Installation
```cmd
php --version
php -m  # List installed modules
php -i  # Show PHP info
```

## OJS-Specific PHP Configuration

### Memory and Execution Limits
```ini
memory_limit = 512M
max_execution_time = 300
upload_max_filesize = 64M
post_max_size = 64M
```

### Database Configuration
```ini
# MySQL/MariaDB support
extension=mysqli
extension=pdo_mysql

# PostgreSQL support (if needed)
extension=pgsql
extension=pdo_pgsql
```

## Testing PHP Installation

### Basic Test
```cmd
php -r "echo 'PHP is working!' . PHP_EOL;"
```

### OJS Compatibility Test
```cmd
php -r "
echo 'PHP Version: ' . PHP_VERSION . PHP_EOL;
echo 'Extensions loaded:' . PHP_EOL;
foreach(['curl', 'gd', 'intl', 'mbstring', 'mysqli', 'zip'] as \$ext) {
    echo '  ' . \$ext . ': ' . (extension_loaded(\$ext) ? 'YES' : 'NO') . PHP_EOL;
}
"
```

## Next Steps After PHP Installation

1. **Verify PHP is in PATH**
   ```cmd
   php --version
   ```

2. **Test OJS Bridge Connection**
   ```cmd
   php plugins/generic/skzAgents/tests/test_bridge.php
   ```

3. **Continue OJS Integration Workflow**
   ```cmd
   # Resume /ojs-integration workflow
   php plugins/generic/skzAgents/tests/test_bridge.php
   python skz-integration/autonomous-agents-framework/src/simple_api_server.py 5000
   bash validate_implementation.sh
   php plugins/generic/skzAgents/deploy.php
   ```

## Troubleshooting

### Common Issues

1. **"php is not recognized"**
   - PHP not in system PATH
   - Restart command prompt after PATH changes

2. **Missing extensions**
   - Check php.ini configuration
   - Uncomment required extension lines

3. **Permission issues**
   - Run command prompt as Administrator
   - Check file permissions in PHP directory

### Verification Commands
```cmd
# Check PHP installation
php --version

# Check loaded extensions
php -m | findstr -i "curl gd mysqli"

# Test basic functionality
php -r "phpinfo();" | findstr "PHP Version"
```

## Integration with SKZ Agents

Once PHP is working, the integration flow will be:

```
OJS (PHP) ←→ SKZ Plugin (PHP) ←→ Python Agents (8001-8007)
```

This maintains the proper architecture where:
- OJS core remains in PHP
- SKZ plugin provides PHP-Python bridge
- Autonomous agents run as Python services
- All communication flows through proper PHP interfaces

## Security Considerations

- Keep PHP updated to latest stable version
- Configure php.ini with security best practices
- Ensure proper file permissions
- Use HTTPS for production deployments
