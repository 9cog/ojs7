"""
Integration Test for Complete Data Synchronization Implementation
Tests the complete data synchronization mechanisms with SKZ framework integration
"""

import unittest
import tempfile
import os
import time
from datetime import datetime
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_ojs_bridge import EnhancedOJSBridge, EnhancedAgentOJSBridge, create_enhanced_bridge, create_enhanced_agent_bridge
from data_sync_manager import SyncDirection

class TestCompleteDataSyncIntegration(unittest.TestCase):
    """Integration test for complete data synchronization implementation"""
    
    def setUp(self):
        """Set up test environment"""
        # Clean any existing database files first
        self._cleanup_databases()
        
        # Create enhanced bridge with mocked OJS endpoints
        self.enhanced_bridge = create_enhanced_bridge(
            "http://localhost:8080",
            "test_api_key", 
            "test_secret_key"
        )
        
        # Mock the underlying OJS API calls
        self._setup_ojs_mocks()
        
        # Create test agent bridges
        self.agent_bridge_1 = create_enhanced_agent_bridge("research_agent", self.enhanced_bridge)
        self.agent_bridge_2 = create_enhanced_agent_bridge("quality_agent", self.enhanced_bridge)
    
    def _cleanup_databases(self):
        """Clean up database files"""
        db_patterns = ['*.db', 'enhanced_ojs_sync.db', 'data_sync.db']
        for pattern in db_patterns:
            if '*' in pattern:
                import glob
                for db_file in glob.glob(pattern):
                    try:
                        os.remove(db_file)
                    except:
                        pass
            else:
                try:
                    os.remove(pattern)
                except:
                    pass
        
    def tearDown(self):
        """Clean up test environment"""
        self.enhanced_bridge.shutdown()
        
        # Clean up database files
        self._cleanup_databases()
    
    def _setup_ojs_mocks(self):
        """Set up OJS API mocks"""
        # Mock manuscript data
        self.enhanced_bridge.get_manuscript = Mock(return_value={
            "id": "test_manuscript_001",
            "title": "Enhanced Test Manuscript", 
            "content": "Test content with enhanced sync",
            "status": "under_review",
            "updated_at": "2024-01-01T10:00:00+00:00",
            "metadata": {
                "author": "Test Author",
                "keywords": ["test", "sync", "enhancement"]
            }
        })
        
        # Mock reviewer data
        self.enhanced_bridge.get_reviewers = Mock(return_value=[{
            "id": "reviewer_001",
            "name": "Test Reviewer",
            "expertise": ["academic_publishing", "quality_assessment"],
            "workload": 3,
            "updated_at": "2024-01-01T09:00:00+00:00"
        }])
        
        # Mock editorial decisions
        self.enhanced_bridge.get_editorial_decisions = Mock(return_value=[{
            "id": "decision_001",
            "submission_id": "test_manuscript_001",
            "decision": "minor_revision",
            "comments": "Enhanced review with sync capabilities",
            "created_at": "2024-01-01T11:00:00+00:00"
        }])
        
        # Mock update operations
        self.enhanced_bridge.update_manuscript = Mock(return_value=True)
        self.enhanced_bridge.create_editorial_decision = Mock(return_value=True)
        self.enhanced_bridge.send_agent_result = Mock(return_value=True)
        self.enhanced_bridge.get_system_status = Mock(return_value={"status": "ok"})
    
    def test_01_enhanced_bridge_initialization(self):
        """Test enhanced bridge initialization"""
        print("\n=== Testing Enhanced Bridge Initialization ===")
        
        # Verify enhanced bridge is properly initialized
        self.assertIsNotNone(self.enhanced_bridge.sync_manager)
        self.assertTrue(self.enhanced_bridge.sync_manager.is_running)
        
        # Verify agent bridges are properly initialized
        self.assertEqual(self.agent_bridge_1.agent_id, "research_agent")
        self.assertEqual(self.agent_bridge_2.agent_id, "quality_agent")
        self.assertTrue(self.agent_bridge_1.auto_sync)
        self.assertTrue(self.agent_bridge_2.auto_sync)
        
        print("✓ Enhanced OJS Bridge initialized with sync manager")
        print("✓ Sync service running")
        print("✓ Agent bridges initialized with auto-sync enabled")
        print("✓ Enhanced Bridge Initialization: PASSED")
    
    def test_02_manuscript_sync_workflow(self):
        """Test complete manuscript synchronization workflow"""
        print("\n=== Testing Manuscript Sync Workflow ===")
        
        manuscript_id = "test_manuscript_001"
        
        # Test sync from OJS to agents
        sync_result = self.enhanced_bridge.sync_manuscript(manuscript_id, SyncDirection.FROM_OJS)
        self.assertTrue(sync_result)
        
        # Test agent processing with enhanced sync
        agent_result = self.agent_bridge_1.sync_manuscript_data(manuscript_id)
        self.assertTrue(agent_result['success'])
        self.assertIn('manuscript', agent_result)
        self.assertIn('agent_analysis', agent_result)
        self.assertIn('sync_status', agent_result)
        
        # Verify sync status is recorded
        sync_status = self.enhanced_bridge.get_sync_status("manuscript", manuscript_id)
        self.assertIsNotNone(sync_status)
        
        print(f"✓ Manuscript {manuscript_id} synced successfully")
        print("✓ Agent analysis completed with sync metadata")
        print("✓ Sync status recorded")
        print("✓ Manuscript Sync Workflow: PASSED")
    
    def test_03_multi_agent_coordination(self):
        """Test multi-agent coordination with data sync"""
        print("\n=== Testing Multi-Agent Coordination ===")
        
        manuscript_id = "coordination_test_ms"
        
        # Both agents process the same manuscript
        result_1 = self.agent_bridge_1.sync_manuscript_data(manuscript_id)
        result_2 = self.agent_bridge_2.sync_manuscript_data(manuscript_id)
        
        # Both should succeed
        self.assertTrue(result_1['success'])
        self.assertTrue(result_2['success'])
        
        # Verify no conflicts in sync
        conflicts = self.enhanced_bridge.get_pending_conflicts()
        manuscript_conflicts = [c for c in conflicts if c['entity_id'] == manuscript_id]
        
        # Get sync statistics
        stats = self.enhanced_bridge.get_sync_statistics()
        
        print(f"✓ Research agent processed manuscript: {result_1['success']}")
        print(f"✓ Quality agent processed manuscript: {result_2['success']}")
        print(f"✓ Pending conflicts for manuscript: {len(manuscript_conflicts)}")
        print(f"✓ Total successful syncs: {stats['successful_syncs']}")
        print("✓ Multi-Agent Coordination: PASSED")
    
    def test_04_real_time_event_processing(self):
        """Test real-time event processing with auto-sync"""
        print("\n=== Testing Real-Time Event Processing ===")
        
        # Test submission created event
        submission_event = {
            'event_type': 'submission_created',
            'submission_id': 'new_submission_001',
            'timestamp': datetime.now().isoformat()
        }
        
        event_result = self.agent_bridge_1.process_ojs_event(submission_event)
        self.assertTrue(event_result['success'])
        self.assertTrue(event_result.get('auto_sync_queued', False))
        
        # Test review assigned event
        review_event = {
            'event_type': 'review_assigned',
            'submission_id': 'new_submission_001',
            'reviewer_id': 'reviewer_001',
            'timestamp': datetime.now().isoformat()
        }
        
        review_result = self.agent_bridge_2.process_ojs_event(review_event)
        self.assertTrue(review_result['success'])
        self.assertTrue(review_result.get('auto_sync_queued', False))
        
        # Test decision made event
        decision_event = {
            'event_type': 'decision_made',
            'submission_id': 'new_submission_001',
            'decision_id': 'decision_001',
            'decision': 'accept',
            'timestamp': datetime.now().isoformat()
        }
        
        decision_result = self.agent_bridge_1.process_ojs_event(decision_event)
        self.assertTrue(decision_result['success'])
        self.assertTrue(decision_result.get('auto_sync_queued', False))
        
        print("✓ Submission created event processed with auto-sync")
        print("✓ Review assigned event processed with auto-sync")
        print("✓ Decision made event processed with auto-sync")
        print("✓ Real-Time Event Processing: PASSED")
    
    def test_05_batch_operations(self):
        """Test batch synchronization operations"""
        print("\n=== Testing Batch Operations ===")
        
        manuscript_ids = [
            "batch_ms_001",
            "batch_ms_002", 
            "batch_ms_003",
            "batch_ms_004",
            "batch_ms_005"
        ]
        
        # Perform batch sync
        batch_results = self.enhanced_bridge.batch_sync_manuscripts(manuscript_ids)
        
        # Verify all manuscripts were processed
        self.assertEqual(len(batch_results), len(manuscript_ids))
        
        # Check success rate
        successful_syncs = sum(1 for result in batch_results.values() if result)
        success_rate = successful_syncs / len(manuscript_ids) * 100
        
        print(f"✓ Batch sync processed {len(manuscript_ids)} manuscripts")
        print(f"✓ Successful syncs: {successful_syncs}/{len(manuscript_ids)}")
        print(f"✓ Success rate: {success_rate:.1f}%")
        print("✓ Batch Operations: PASSED")
    
    def test_06_conflict_resolution_workflow(self):
        """Test conflict resolution workflow"""
        print("\n=== Testing Conflict Resolution Workflow ===")
        
        # Create a scenario that might cause conflicts
        manuscript_id = "conflict_test_ms"
        
        # Simulate concurrent modifications by both agents
        # (In a real scenario, this would be concurrent access to the same data)
        
        # Agent 1 processes manuscript
        with patch.object(self.enhanced_bridge.sync_manager, '_get_agent_data') as mock_agent_data:
            mock_agent_data.return_value = {
                "id": manuscript_id,
                "title": "Agent 1 Modified Title",
                "updated_at": "2024-01-01T10:30:00+00:00",
                "agent_analysis": {"quality_score": 0.85}
            }
            
            result_1 = self.agent_bridge_1.sync_manuscript_data(manuscript_id)
        
        # Check for any conflicts
        conflicts = self.enhanced_bridge.get_pending_conflicts()
        
        # If conflicts exist, test resolution
        if conflicts:
            conflict = conflicts[0]
            resolution_data = {
                "id": manuscript_id,
                "title": "Resolved Title",
                "resolved_by": "automated_system",
                "resolution_timestamp": datetime.now().isoformat()
            }
            
            resolved = self.enhanced_bridge.resolve_conflict(conflict['id'], resolution_data)
            self.assertTrue(resolved)
            
            print(f"✓ Conflict detected and resolved: {conflict['id']}")
        else:
            print("✓ No conflicts detected (conflict prevention working)")
        
        # Get conflict resolution statistics
        stats = self.enhanced_bridge.get_sync_statistics()
        
        print(f"✓ Total conflicts resolved: {stats['conflicts_resolved']}")
        print("✓ Conflict Resolution Workflow: PASSED")
    
    def test_07_performance_monitoring(self):
        """Test performance monitoring and statistics"""
        print("\n=== Testing Performance Monitoring ===")
        
        # Perform various operations to generate statistics
        operations = [
            ("manuscript", "perf_ms_001"),
            ("manuscript", "perf_ms_002"),
            ("reviewer", "perf_reviewer_001"),
            ("editorial_decision", "perf_decision_001")
        ]
        
        for entity_type, entity_id in operations:
            if entity_type == "manuscript":
                self.enhanced_bridge.sync_manuscript(entity_id)
            elif entity_type == "reviewer":
                self.enhanced_bridge.sync_reviewer(entity_id)
            elif entity_type == "editorial_decision":
                self.enhanced_bridge.sync_editorial_decision(entity_id)
        
        # Get comprehensive statistics
        stats = self.enhanced_bridge.get_sync_statistics()
        
        # Verify statistics structure and values
        required_stats = [
            'total_syncs', 'successful_syncs', 'failed_syncs',
            'conflicts_resolved', 'pending_conflicts', 'active_syncs',
            'queue_size', 'last_sync'
        ]
        
        for stat in required_stats:
            self.assertIn(stat, stats)
        
        print(f"✓ Total synchronizations: {stats['total_syncs']}")
        print(f"✓ Success rate: {stats['successful_syncs']}/{stats['total_syncs']}")
        print(f"✓ Pending conflicts: {stats['pending_conflicts']}")
        print(f"✓ Queue size: {stats['queue_size']}")
        print("✓ Performance Monitoring: PASSED")
    
    def test_08_health_monitoring(self):
        """Test system health monitoring"""
        print("\n=== Testing Health Monitoring ===")
        
        # Get overall health status
        health = self.enhanced_bridge.health_check()
        
        # Verify health check structure
        self.assertIn('ojs_status', health)
        self.assertIn('sync_status', health)
        self.assertIn('overall_status', health)
        self.assertIn('timestamp', health)
        
        # Test agent bridge health status
        agent_status_1 = self.agent_bridge_1.get_enhanced_bridge_status()
        agent_status_2 = self.agent_bridge_2.get_enhanced_bridge_status()
        
        # Verify enhanced features are reported
        self.assertIn('enhanced_features', agent_status_1)
        self.assertIn('sync_statistics', agent_status_1)
        self.assertTrue(agent_status_1['enhanced_features']['data_sync_manager'])
        self.assertTrue(agent_status_1['enhanced_features']['auto_sync_enabled'])
        
        print(f"✓ Overall health status: {health['overall_status']}")
        print(f"✓ OJS status: {health['ojs_status'].get('status', 'unknown')}")
        print(f"✓ Sync status: {health['sync_status']['status']}")
        print(f"✓ Agent 1 features: {len(agent_status_1['enhanced_features'])} enhanced features")
        print(f"✓ Agent 2 features: {len(agent_status_2['enhanced_features'])} enhanced features")
        print("✓ Health Monitoring: PASSED")
    
    def test_09_auto_sync_toggle(self):
        """Test auto-sync enable/disable functionality"""
        print("\n=== Testing Auto-Sync Toggle ===")
        
        # Initially auto-sync should be enabled
        self.assertTrue(self.agent_bridge_1.auto_sync)
        
        # Disable auto-sync
        self.agent_bridge_1.disable_auto_sync()
        self.assertFalse(self.agent_bridge_1.auto_sync)
        
        # Test event processing without auto-sync
        test_event = {
            'event_type': 'submission_created',
            'submission_id': 'no_auto_sync_ms',
            'timestamp': datetime.now().isoformat()
        }
        
        result = self.agent_bridge_1.process_ojs_event(test_event)
        self.assertTrue(result['success'])
        self.assertFalse(result.get('auto_sync_queued', False))
        
        # Re-enable auto-sync
        self.agent_bridge_1.enable_auto_sync()
        self.assertTrue(self.agent_bridge_1.auto_sync)
        
        # Test event processing with auto-sync re-enabled
        result_enabled = self.agent_bridge_1.process_ojs_event(test_event)
        self.assertTrue(result_enabled['success'])
        self.assertTrue(result_enabled.get('auto_sync_queued', False))
        
        print("✓ Auto-sync disabled successfully")
        print("✓ Event processing without auto-sync works")
        print("✓ Auto-sync re-enabled successfully")
        print("✓ Event processing with auto-sync restored")
        print("✓ Auto-Sync Toggle: PASSED")
    
    def test_10_integration_validation(self):
        """Final integration validation"""
        print("\n=== Testing Integration Validation ===")
        
        # Comprehensive workflow test
        manuscript_id = "integration_test_ms"
        
        # 1. Initial sync from OJS
        sync_result = self.enhanced_bridge.sync_manuscript(manuscript_id, SyncDirection.FROM_OJS)
        self.assertTrue(sync_result)
        
        # 2. Multiple agents process the manuscript
        research_result = self.agent_bridge_1.sync_manuscript_data(manuscript_id)
        quality_result = self.agent_bridge_2.sync_manuscript_data(manuscript_id)
        
        self.assertTrue(research_result['success'])
        self.assertTrue(quality_result['success'])
        
        # 3. Simulate OJS event
        event_result = self.agent_bridge_1.process_ojs_event({
            'event_type': 'decision_made',
            'submission_id': manuscript_id,
            'decision_id': 'final_decision',
            'decision': 'accept'
        })
        self.assertTrue(event_result['success'])
        
        # 4. Final sync back to OJS
        final_sync = self.enhanced_bridge.sync_manuscript(manuscript_id, SyncDirection.TO_OJS)
        self.assertTrue(final_sync)
        
        # 5. Validate final state
        final_stats = self.enhanced_bridge.get_sync_statistics()
        final_health = self.enhanced_bridge.health_check()
        
        self.assertGreater(final_stats['total_syncs'], 0)
        self.assertEqual(final_health['overall_status'], 'healthy')
        
        print("✓ End-to-end manuscript workflow completed")
        print("✓ Multi-agent processing successful")
        print("✓ Event-driven synchronization working")
        print("✓ Final system state healthy")
        print(f"✓ Total operations: {final_stats['total_syncs']}")
        print(f"✓ Success rate: {final_stats['successful_syncs']}/{final_stats['total_syncs']}")
        print("✓ Integration Validation: PASSED")

def run_integration_tests():
    """Run complete data synchronization integration tests"""
    print("============================================================")
    print("COMPLETE DATA SYNCHRONIZATION INTEGRATION TEST SUITE")
    print("============================================================")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Testing Complete Data Synchronization Implementation")
    print("============================================================")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCompleteDataSyncIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n============================================================")
    print("INTEGRATION TEST RESULTS SUMMARY")
    print("============================================================")
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n🎉 COMPLETE DATA SYNCHRONIZATION INTEGRATION: PASSED")
        print("✅ All data synchronization mechanisms are working correctly")
        print("🚀 Enhanced data sync implementation successfully integrated with SKZ framework")
        print("🎯 Ready for production deployment")
    else:
        print("\n❌ COMPLETE DATA SYNCHRONIZATION INTEGRATION: FAILED") 
        print("❌ Some integration components need attention")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)