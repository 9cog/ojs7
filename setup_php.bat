@echo off
echo Setting up PHP for OJS Integration...

REM Set PHP path for current session
set PATH=%PATH%;D:\casto\oj7\php

REM Copy PHP configuration
if exist "D:\casto\oj7\php\php.ini-development" (
    copy "D:\casto\oj7\php\php.ini-development" "D:\casto\oj7\php\php.ini"
    echo PHP configuration copied
) else (
    echo Please extract PHP to D:\casto\oj7\php\ first
    pause
    exit /b 1
)

REM Test PHP installation
echo Testing PHP installation...
php --version

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ PHP is working!
    echo.
    echo Testing required extensions...
    php -r "echo 'Extensions check:' . PHP_EOL; $required = ['curl', 'gd', 'intl', 'mbstring', 'mysqli', 'zip', 'json', 'xml', 'openssl']; foreach($required as $ext) { echo '  ' . $ext . ': ' . (extension_loaded($ext) ? 'OK' : 'MISSING') . PHP_EOL; }"
    echo.
    echo Ready to continue with OJS integration workflow!
    echo Run: php plugins/generic/skzAgents/tests/test_bridge.php
) else (
    echo ✗ PHP not working. Please check installation.
)

pause
