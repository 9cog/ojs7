#!/usr/bin/env python3
"""
Simple test for API server functionality
"""

import sys
import os
import time
import threading
import subprocess

def start_api_server():
    """Start the simple API server"""
    print("Starting API server on port 5000...")
    
    # Change to the correct directory
    src_dir = "/home/runner/work/oj7/oj7/skz-integration/autonomous-agents-framework/src"
    os.chdir(src_dir)
    
    # Start the server
    try:
        import simple_api_server
        simple_api_server.run_server(5000, 'localhost')
    except KeyboardInterrupt:
        print("Server stopped")
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'background':
        # Run in background for testing
        def run_bg():
            start_api_server()
        
        bg_thread = threading.Thread(target=run_bg, daemon=True)
        bg_thread.start()
        
        print("Server started in background")
        time.sleep(60)  # Keep alive for 60 seconds
    else:
        # Run normally
        start_api_server()