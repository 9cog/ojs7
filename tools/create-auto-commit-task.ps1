#!/usr/bin/env pwsh
# Create-Auto-Commit-Task.ps1
# A masterful script to create a Windows scheduled task for automatic git commits
# This magnificent orchestration ensures flawless GitHub synchronization

# Requires elevation - must be run as administrator
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Output "This divine script requires elevation! Please run as Administrator."
    exit 1
}

# Configuration variables - the foundation of our masterpiece
$TASK_NAME = "OJSAutoCommit"
$SCRIPT_PATH = "D:\casto\ojs\ojs-3.3.0-7\tools\auto-commit.ps1"
$WORKING_DIR = "D:\casto\ojs\ojs-3.3.0-7"

# Create a dazzling description for this task
$TASK_DESCRIPTION = "Automatic GitHub synchronization for Open Journal Systems. This magnificent task runs the auto-commit script to ensure changes are committed and pushed to GitHub with impeccable reliability."

# Delete the task if it already exists (ensuring a clean slate for our creation)
try {
    Unregister-ScheduledTask -TaskName $TASK_NAME -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "Previous task removed, preparing to create a new masterpiece..."
} catch {
    Write-Output "No previous task found, creating fresh..."
}

# Create an action of unparalleled elegance
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$SCRIPT_PATH`"" `
    -WorkingDirectory $WORKING_DIR

# Define multiple majestic triggers for our symphony of automation
# 1. Run every 15 minutes - ensuring frequent synchronization
$trigger1 = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15)

# 2. Run at system startup - ensuring we never miss a beat
$trigger2 = New-ScheduledTaskTrigger -AtStartup

# Set breathtaking settings for reliability beyond compare
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

# Use the current user for authentication - a perfect balance of security and usability
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType S4U `
    -RunLevel Highest

# Register our magnificent task
Register-ScheduledTask -TaskName $TASK_NAME `
    -Action $action `
    -Trigger $trigger1,$trigger2 `
    -Settings $settings `
    -Principal $principal `
    -Description $TASK_DESCRIPTION `
    -Force

Write-Output "====================================================="
Write-Output "     🌟 Automatic Commit Task Created! 🌟"
Write-Output "====================================================="
Write-Output "Task Name: $TASK_NAME"
Write-Output "Script Path: $SCRIPT_PATH"
Write-Output "Trigger 1: Every 15 minutes"
Write-Output "Trigger 2: At system startup"
Write-Output ""
Write-Output "This masterpiece of automation will ensure your GitHub repository"
Write-Output "is synchronized with unfailing reliability and breathtaking precision!"
Write-Output "====================================================="