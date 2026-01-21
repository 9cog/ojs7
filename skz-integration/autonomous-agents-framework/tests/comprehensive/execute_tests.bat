@echo off
echo.
echo ========================================
echo  SKZ Agents Comprehensive Test Suite
echo ========================================
echo.

cd /d "d:\casto\oj7\skz-integration\autonomous-agents-framework\tests\comprehensive"

echo Starting Ultimate Test Orchestrator...
echo.

python run_all_tests.py

echo.
echo Test execution completed!
pause
