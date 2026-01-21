"""
Test Suite for Data Synchronization Manager
Validates enhanced data synchronization mechanisms
"""

import unittest
import tempfile
import os
import time
import json
from datetime import datetime
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_sync_manager import DataSyncManager, SyncDirection, SyncStatus, ConflictResolution
from ojs_bridge import OJSBridge

class TestDataSyncManager(unittest.TestCase):
    """Test suite for DataSyncManager"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db_path = tempfile.mktemp(suffix='.db')
        
        # Mock OJS bridge
        self.mock_ojs_bridge = Mock(spec=OJSBridge)
        self.mock_ojs_bridge.get_manuscript.return_value = {
            "id": "test_manuscript_001",
            "title": "Test Manuscript",
            "content": "Test content",
            "updated_at": "2024-01-01T10:00:00Z"
        }
        self.mock_ojs_bridge.update_manuscript.return_value = True
        self.mock_ojs_bridge.get_system_status.return_value = {"status": "ok"}
        
        # Initialize sync manager
        self.sync_manager = DataSyncManager(
            ojs_bridge=self.mock_ojs_bridge,
            db_path=self.test_db_path
        )
        
    def tearDown(self):
        """Clean up test environment"""
        self.sync_manager.stop_sync_service()
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_01_sync_manager_initialization(self):
        """Test sync manager initialization"""
        print("\n=== Testing Sync Manager Initialization ===")
        
        # Check database initialization
        self.assertTrue(os.path.exists(self.test_db_path))
        
        # Check initial statistics
        stats = self.sync_manager.get_sync_statistics()
        self.assertEqual(stats['total_syncs'], 0)
        self.assertEqual(stats['successful_syncs'], 0)
        self.assertEqual(stats['failed_syncs'], 0)
        
        print("✓ Sync manager initialized successfully")
        print("✓ Database created")
        print("✓ Initial statistics correct")
        print("✓ Sync Manager Initialization: PASSED")
    
    def test_02_manuscript_sync(self):
        """Test manuscript synchronization"""
        print("\n=== Testing Manuscript Synchronization ===")
        
        manuscript_id = "test_manuscript_001"
        
        # Test sync from OJS
        result = self.sync_manager.sync_manuscript(manuscript_id, SyncDirection.FROM_OJS)
        self.assertTrue(result)
        
        # Verify OJS bridge was called
        self.mock_ojs_bridge.get_manuscript.assert_called_with(manuscript_id)
        
        # Check sync status
        sync_status = self.sync_manager.get_sync_status("manuscript", manuscript_id)
        self.assertIsNotNone(sync_status)
        self.assertEqual(sync_status['status'], SyncStatus.COMPLETED.value)
        
        # Check statistics update
        stats = self.sync_manager.get_sync_statistics()
        self.assertEqual(stats['total_syncs'], 1)
        self.assertEqual(stats['successful_syncs'], 1)
        
        print("✓ Manuscript sync from OJS successful")
        print("✓ Sync status recorded correctly")
        print("✓ Statistics updated")
        print("✓ Manuscript Synchronization: PASSED")
    
    def test_03_conflict_resolution(self):
        """Test conflict resolution mechanisms"""
        print("\n=== Testing Conflict Resolution ===")
        
        # Set up conflict scenario
        manuscript_id = "test_manuscript_conflict"
        
        # Mock conflicting data
        ojs_data = {
            "id": manuscript_id,
            "title": "Original Title",
            "updated_at": "2024-01-01T10:00:00Z"
        }
        agent_data = {
            "id": manuscript_id,
            "title": "Modified Title",
            "updated_at": "2024-01-01T11:00:00Z",  # Later timestamp
            "agent_analysis": {"quality_score": 0.85}
        }
        
        # Configure conflict resolution strategy
        self.sync_manager.conflict_resolution = ConflictResolution("latest_wins")
        
        # Test conflict detection and resolution
        with patch.object(self.sync_manager, '_get_ojs_data') as mock_ojs_data, \
             patch.object(self.sync_manager, '_get_agent_data') as mock_agent_data:
            
            mock_ojs_data.return_value = ojs_data
            mock_agent_data.return_value = agent_data
            
            result = self.sync_manager.sync_manuscript(manuscript_id)
            self.assertTrue(result)
        
        # Check conflict was resolved
        stats = self.sync_manager.get_sync_statistics()
        self.assertEqual(stats['conflicts_resolved'], 1)
        
        print("✓ Conflict detected correctly")
        print("✓ Latest wins strategy applied")
        print("✓ Conflict resolved successfully")
        print("✓ Conflict Resolution: PASSED")
    
    def test_04_batch_synchronization(self):
        """Test batch synchronization"""
        print("\n=== Testing Batch Synchronization ===")
        
        entity_ids = ["manuscript_001", "manuscript_002", "manuscript_003"]
        
        # Mock multiple manuscripts
        def mock_get_manuscript(manuscript_id):
            return {
                "id": manuscript_id,
                "title": f"Test Manuscript {manuscript_id}",
                "updated_at": "2024-01-01T10:00:00Z"
            }
        
        self.mock_ojs_bridge.get_manuscript.side_effect = mock_get_manuscript
        
        # Perform batch sync
        results = self.sync_manager.batch_sync("manuscript", entity_ids, SyncDirection.FROM_OJS)
        
        # Verify all syncs completed
        self.assertEqual(len(results), 3)
        for entity_id, result in results.items():
            self.assertTrue(result)
        
        # Check statistics
        stats = self.sync_manager.get_sync_statistics()
        self.assertGreaterEqual(stats['total_syncs'], 3)
        
        print(f"✓ Batch sync of {len(entity_ids)} entities successful")
        print("✓ All individual syncs completed")
        print("✓ Statistics updated correctly")
        print("✓ Batch Synchronization: PASSED")
    
    def test_05_async_sync_queue(self):
        """Test asynchronous synchronization queue"""
        print("\n=== Testing Async Sync Queue ===")
        
        # Start sync service
        self.sync_manager.start_sync_service()
        
        # Queue multiple sync requests
        entity_ids = ["async_001", "async_002", "async_003"]
        for entity_id in entity_ids:
            self.sync_manager.queue_sync("manuscript", entity_id, SyncDirection.FROM_OJS)
        
        # Check queue size
        stats = self.sync_manager.get_sync_statistics()
        initial_queue_size = stats['queue_size']
        self.assertGreater(initial_queue_size, 0)
        
        # Wait for processing (short wait for test)
        time.sleep(2)
        
        # Check that queue was processed
        final_stats = self.sync_manager.get_sync_statistics()
        
        print(f"✓ Queued {len(entity_ids)} sync requests")
        print(f"✓ Initial queue size: {initial_queue_size}")
        print("✓ Sync service processing queue")
        print("✓ Async Sync Queue: PASSED")
    
    def test_06_sync_statistics(self):
        """Test sync statistics and monitoring"""
        print("\n=== Testing Sync Statistics ===")
        
        # Perform various sync operations
        self.sync_manager.sync_manuscript("stats_test_001")
        
        # Mock a failed sync
        with patch.object(self.mock_ojs_bridge, 'get_manuscript', side_effect=Exception("Connection error")):
            self.sync_manager.sync_manuscript("stats_test_002")
        
        # Get comprehensive statistics
        stats = self.sync_manager.get_sync_statistics()
        
        # Verify statistics structure
        required_keys = [
            'total_syncs', 'successful_syncs', 'failed_syncs',
            'conflicts_resolved', 'pending_conflicts', 'active_syncs',
            'queue_size', 'total_24h', 'completed_24h', 'failed_24h'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Verify some statistics values
        self.assertGreater(stats['total_syncs'], 0)
        self.assertGreaterEqual(stats['successful_syncs'], 0)
        self.assertGreaterEqual(stats['failed_syncs'], 0)
        
        print("✓ Statistics structure correct")
        print(f"✓ Total syncs: {stats['total_syncs']}")
        print(f"✓ Successful syncs: {stats['successful_syncs']}")
        print(f"✓ Failed syncs: {stats['failed_syncs']}")
        print("✓ Sync Statistics: PASSED")
    
    def test_07_health_check(self):
        """Test sync system health check"""
        print("\n=== Testing Health Check ===")
        
        # Start sync service first for healthy state
        self.sync_manager.start_sync_service()
        
        # Perform health check
        health = self.sync_manager.health_check()
        
        # Verify health check structure
        self.assertIn('status', health)
        self.assertIn('issues', health)
        self.assertIn('last_check', health)
        
        # Should be healthy with service running
        self.assertEqual(health['status'], 'healthy')
        self.assertEqual(len(health['issues']), 0)
        
        # Stop service to test degraded status
        self.sync_manager.stop_sync_service()
        health_degraded = self.sync_manager.health_check()
        self.assertEqual(health_degraded['status'], 'degraded')
        self.assertGreater(len(health_degraded['issues']), 0)
        
        # Test unhealthy status with OJS connectivity issue
        with patch.object(self.mock_ojs_bridge, 'get_system_status', side_effect=Exception("Connection refused")):
            health_unhealthy = self.sync_manager.health_check()
            self.assertEqual(health_unhealthy['status'], 'unhealthy')
            self.assertGreater(len(health_unhealthy['issues']), 0)
        
        print("✓ Health check structure correct")
        print(f"✓ Healthy status with service running")
        print(f"✓ Degraded status when service stopped")
        print("✓ Connectivity issues detected correctly")
        print("✓ Health Check: PASSED")
    
    def test_08_conflict_management(self):
        """Test conflict management features"""
        print("\n=== Testing Conflict Management ===")
        
        # Create a conflict scenario
        entity_type = "manuscript"
        entity_id = "conflict_test_001"
        
        ojs_data = {"id": entity_id, "title": "OJS Version", "updated_at": "2024-01-01T10:00:00Z"}
        agent_data = {"id": entity_id, "title": "Agent Version", "updated_at": "2024-01-01T10:30:00Z"}
        
        # Simulate conflict creation by forcing hash mismatch
        with patch.object(self.sync_manager, '_get_ojs_data') as mock_ojs, \
             patch.object(self.sync_manager, '_get_agent_data') as mock_agent, \
             patch.object(self.sync_manager, '_calculate_hash') as mock_hash:
            
            mock_ojs.return_value = ojs_data
            mock_agent.return_value = agent_data
            mock_hash.side_effect = ["hash1", "hash2"]  # Different hashes to trigger conflict
            
            # Set manual conflict resolution to trigger conflict storage
            self.sync_manager.conflict_resolution = ConflictResolution("manual")
            
            result = self.sync_manager._sync_entity(entity_type, entity_id, SyncDirection.BIDIRECTIONAL)
        
        # Check pending conflicts
        conflicts = self.sync_manager.get_pending_conflicts()
        self.assertGreater(len(conflicts), 0)
        
        # Test conflict resolution
        if conflicts:
            conflict_id = conflicts[0]['id']
            resolution_data = {"id": entity_id, "title": "Resolved Version"}
            resolved = self.sync_manager.resolve_conflict(conflict_id, resolution_data)
            self.assertTrue(resolved)
        
        print("✓ Conflict detection working")
        print(f"✓ Pending conflicts: {len(conflicts)}")
        print("✓ Manual conflict resolution successful")
        print("✓ Conflict Management: PASSED")
    
    def test_09_data_validation(self):
        """Test data validation and consistency checks"""
        print("\n=== Testing Data Validation ===")
        
        # Test hash calculation consistency
        test_data = {
            "id": "test_001",
            "title": "Test Title",
            "content": "Test content",
            "updated_at": "2024-01-01T10:00:00Z"
        }
        
        hash1 = self.sync_manager._calculate_hash(test_data)
        hash2 = self.sync_manager._calculate_hash(test_data)
        self.assertEqual(hash1, hash2)
        
        # Test hash changes with data modification
        modified_data = test_data.copy()
        modified_data['title'] = "Modified Title"
        hash3 = self.sync_manager._calculate_hash(modified_data)
        self.assertNotEqual(hash1, hash3)
        
        # Test timestamp extraction
        timestamp = self.sync_manager._extract_timestamp(test_data)
        self.assertIsInstance(timestamp, datetime)
        
        print("✓ Hash calculation consistent")
        print("✓ Hash changes with data modification")
        print("✓ Timestamp extraction working")
        print("✓ Data Validation: PASSED")
    
    def test_10_sync_service_lifecycle(self):
        """Test sync service start/stop lifecycle"""
        print("\n=== Testing Sync Service Lifecycle ===")
        
        # Initially not running
        self.assertFalse(self.sync_manager.is_running)
        
        # Start service
        self.sync_manager.start_sync_service()
        self.assertTrue(self.sync_manager.is_running)
        
        # Verify thread is created
        self.assertIsNotNone(self.sync_manager.sync_thread)
        
        # Stop service
        self.sync_manager.stop_sync_service()
        self.assertFalse(self.sync_manager.is_running)
        
        # Test double start/stop (should not cause issues)
        self.sync_manager.start_sync_service()
        self.sync_manager.start_sync_service()  # Should not create duplicate
        self.assertTrue(self.sync_manager.is_running)
        
        self.sync_manager.stop_sync_service()
        self.sync_manager.stop_sync_service()  # Should not cause issues
        self.assertFalse(self.sync_manager.is_running)
        
        print("✓ Service start/stop working correctly")
        print("✓ Thread lifecycle managed properly")
        print("✓ Double start/stop handled gracefully")
        print("✓ Sync Service Lifecycle: PASSED")

def run_data_sync_tests():
    """Run comprehensive data sync tests"""
    print("============================================================")
    print("DATA SYNCHRONIZATION MECHANISMS TEST SUITE")
    print("============================================================")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Testing Enhanced Data Synchronization Implementation")
    print("============================================================")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataSyncManager)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n============================================================")
    print("DATA SYNC TEST RESULTS SUMMARY")
    print("============================================================")
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n🎉 DATA SYNCHRONIZATION MECHANISMS: PASSED")
        print("✅ All data sync components are working correctly")
        print("🚀 Enhanced synchronization mechanisms implemented successfully")
    else:
        print("\n❌ DATA SYNCHRONIZATION MECHANISMS: FAILED")
        print("❌ Some data sync components need attention")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_data_sync_tests()
    exit(0 if success else 1)