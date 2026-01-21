"""
Enhanced OJS Bridge Integration with Data Synchronization
Integrates DataSyncManager with existing OJS Bridge functionality
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ojs_bridge import OJSBridge, AgentOJSBridge
from data_sync_manager import DataSyncManager, SyncDirection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedOJSBridge(OJSBridge):
    """
    Enhanced OJS Bridge with integrated data synchronization
    Extends the base OJS Bridge with robust sync capabilities
    """
    
    def __init__(self, ojs_base_url: str, api_key: str, secret_key: str):
        super().__init__(ojs_base_url, api_key, secret_key)
        
        # Initialize data sync manager
        self.sync_manager = DataSyncManager(self, "enhanced_ojs_sync.db")
        self.sync_manager.start_sync_service()
        
        logger.info("Initialized Enhanced OJS Bridge with data synchronization")
    
    def sync_manuscript(self, submission_id: str, direction: SyncDirection = SyncDirection.BIDIRECTIONAL) -> bool:
        """Synchronize manuscript with enhanced sync capabilities"""
        return self.sync_manager.sync_manuscript(submission_id, direction)
    
    def sync_reviewer(self, reviewer_id: str, direction: SyncDirection = SyncDirection.BIDIRECTIONAL) -> bool:
        """Synchronize reviewer with enhanced sync capabilities"""
        return self.sync_manager.sync_reviewer(reviewer_id, direction)
    
    def sync_editorial_decision(self, decision_id: str, direction: SyncDirection = SyncDirection.BIDIRECTIONAL) -> bool:
        """Synchronize editorial decision with enhanced sync capabilities"""
        return self.sync_manager.sync_editorial_decision(decision_id, direction)
    
    def batch_sync_manuscripts(self, manuscript_ids: List[str]) -> Dict[str, bool]:
        """Batch synchronize multiple manuscripts"""
        return self.sync_manager.batch_sync("manuscript", manuscript_ids)
    
    def get_sync_status(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get synchronization status for an entity"""
        return self.sync_manager.get_sync_status(entity_type, entity_id)
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get comprehensive synchronization statistics"""
        return self.sync_manager.get_sync_statistics()
    
    def get_pending_conflicts(self) -> List[Dict[str, Any]]:
        """Get list of pending synchronization conflicts"""
        return self.sync_manager.get_pending_conflicts()
    
    def resolve_conflict(self, conflict_id: str, resolution_data: Dict[str, Any]) -> bool:
        """Resolve a synchronization conflict"""
        return self.sync_manager.resolve_conflict(conflict_id, resolution_data)
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check including sync status"""
        base_health = super().get_system_status()
        sync_health = self.sync_manager.health_check()
        
        return {
            'ojs_status': base_health,
            'sync_status': sync_health,
            'overall_status': 'healthy' if sync_health['status'] == 'healthy' else 'degraded',
            'timestamp': datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Properly shutdown the enhanced bridge"""
        self.sync_manager.stop_sync_service()
        logger.info("Enhanced OJS Bridge shutdown complete")

class EnhancedAgentOJSBridge(AgentOJSBridge):
    """
    Enhanced Agent OJS Bridge with automatic data synchronization
    Extends the agent bridge with real-time sync capabilities
    """
    
    def __init__(self, agent_id: str, enhanced_ojs_bridge: EnhancedOJSBridge):
        # Initialize with the base OJS bridge from enhanced bridge
        super().__init__(agent_id, enhanced_ojs_bridge)
        self.enhanced_bridge = enhanced_ojs_bridge
        self.auto_sync = True  # Enable automatic synchronization
        
        logger.info(f"Initialized Enhanced Agent OJS Bridge for {agent_id}")
    
    def sync_manuscript_data(self, submission_id: str) -> Dict[str, Any]:
        """Enhanced manuscript data synchronization"""
        try:
            # Use enhanced sync capabilities
            sync_success = self.enhanced_bridge.sync_manuscript(submission_id)
            
            if not sync_success:
                return {'success': False, 'error': 'Synchronization failed'}
            
            # Get synchronized manuscript data
            manuscript = self.ojs_bridge.get_manuscript(submission_id)
            if not manuscript:
                return {'success': False, 'error': 'Manuscript not found after sync'}
            
            # Get agent's analysis
            agent_analysis = self._analyze_manuscript(manuscript)
            
            # If auto-sync is enabled, automatically sync results back
            if self.auto_sync:
                result_success = self.ojs_bridge.send_agent_result(self.agent_id, {
                    'submission_id': submission_id,
                    'analysis': agent_analysis,
                    'timestamp': datetime.now().isoformat(),
                    'sync_metadata': {
                        'sync_timestamp': datetime.now().isoformat(),
                        'agent_id': self.agent_id,
                        'sync_version': '2.0'
                    }
                })
                
                if result_success:
                    # Queue a sync back to ensure consistency
                    self.enhanced_bridge.sync_manager.queue_sync(
                        "manuscript", submission_id, SyncDirection.TO_OJS
                    )
            
            return {
                'success': True,
                'manuscript': manuscript,
                'agent_analysis': agent_analysis,
                'sync_status': self.enhanced_bridge.get_sync_status("manuscript", submission_id)
            }
            
        except Exception as e:
            logger.error(f"Enhanced sync failed for manuscript {submission_id}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def process_ojs_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced OJS event processing with automatic synchronization"""
        event_type = event_data.get('event_type', 'unknown')
        
        try:
            # Process the event using base functionality
            base_result = super().process_ojs_event(event_data)
            
            # Add enhanced sync capabilities based on event type
            if event_type == 'submission_created':
                submission_id = event_data.get('submission_id')
                if submission_id and self.auto_sync:
                    # Queue automatic sync for new submission
                    self.enhanced_bridge.sync_manager.queue_sync(
                        "manuscript", submission_id, SyncDirection.FROM_OJS
                    )
                    base_result['auto_sync_queued'] = True
            
            elif event_type == 'review_assigned':
                submission_id = event_data.get('submission_id')
                reviewer_id = event_data.get('reviewer_id')
                
                if submission_id and reviewer_id and self.auto_sync:
                    # Sync both manuscript and reviewer data
                    self.enhanced_bridge.sync_manager.queue_sync(
                        "manuscript", submission_id, SyncDirection.BIDIRECTIONAL
                    )
                    self.enhanced_bridge.sync_manager.queue_sync(
                        "reviewer", reviewer_id, SyncDirection.BIDIRECTIONAL
                    )
                    base_result['auto_sync_queued'] = True
            
            elif event_type == 'decision_made':
                submission_id = event_data.get('submission_id')
                decision_id = event_data.get('decision_id')
                
                if submission_id and decision_id and self.auto_sync:
                    # Sync decision and manuscript data
                    self.enhanced_bridge.sync_manager.queue_sync(
                        "editorial_decision", decision_id, SyncDirection.BIDIRECTIONAL
                    )
                    self.enhanced_bridge.sync_manager.queue_sync(
                        "manuscript", submission_id, SyncDirection.BIDIRECTIONAL
                    )
                    base_result['auto_sync_queued'] = True
            
            # Add sync metadata to result
            base_result['sync_metadata'] = {
                'enhanced_sync_enabled': True,
                'auto_sync_enabled': self.auto_sync,
                'sync_timestamp': datetime.now().isoformat(),
                'agent_id': self.agent_id
            }
            
            return base_result
            
        except Exception as e:
            logger.error(f"Enhanced event processing failed: {str(e)}")
            return {
                'success': False,
                'error': f'Enhanced event processing failed: {str(e)}',
                'event_type': event_type
            }
    
    def get_enhanced_bridge_status(self) -> Dict[str, Any]:
        """Get comprehensive bridge status including sync information"""
        base_status = self.get_bridge_status()
        
        # Add enhanced sync information
        sync_stats = self.enhanced_bridge.get_sync_statistics()
        pending_conflicts = self.enhanced_bridge.get_pending_conflicts()
        health = self.enhanced_bridge.health_check()
        
        enhanced_status = {
            **base_status,
            'enhanced_features': {
                'auto_sync_enabled': self.auto_sync,
                'data_sync_manager': True,
                'conflict_resolution': True,
                'batch_operations': True,
                'real_time_sync': True
            },
            'sync_statistics': sync_stats,
            'pending_conflicts_count': len(pending_conflicts),
            'health_status': health['overall_status'],
            'enhanced_version': '2.0'
        }
        
        return enhanced_status
    
    def enable_auto_sync(self):
        """Enable automatic synchronization"""
        self.auto_sync = True
        logger.info(f"Auto-sync enabled for agent {self.agent_id}")
    
    def disable_auto_sync(self):
        """Disable automatic synchronization"""
        self.auto_sync = False
        logger.info(f"Auto-sync disabled for agent {self.agent_id}")

def create_enhanced_bridge(ojs_base_url: str, api_key: str, secret_key: str) -> EnhancedOJSBridge:
    """Factory function to create enhanced OJS bridge"""
    return EnhancedOJSBridge(ojs_base_url, api_key, secret_key)

def create_enhanced_agent_bridge(agent_id: str, enhanced_bridge: EnhancedOJSBridge) -> EnhancedAgentOJSBridge:
    """Factory function to create enhanced agent bridge"""
    return EnhancedAgentOJSBridge(agent_id, enhanced_bridge)

# Example usage and integration test
if __name__ == "__main__":
    # This demonstrates how to use the enhanced bridges
    
    # Create enhanced OJS bridge
    enhanced_bridge = create_enhanced_bridge(
        "http://localhost:8080",
        "test_api_key",
        "test_secret_key"
    )
    
    # Create enhanced agent bridge
    agent_bridge = create_enhanced_agent_bridge("test_agent", enhanced_bridge)
    
    # Example operations
    print("Enhanced OJS Bridge Integration Example")
    print("=" * 50)
    
    # Test sync operations
    print("Testing manuscript sync...")
    sync_result = enhanced_bridge.sync_manuscript("test_manuscript_001")
    print(f"Sync result: {sync_result}")
    
    # Test batch operations
    print("Testing batch sync...")
    batch_result = enhanced_bridge.batch_sync_manuscripts(["ms1", "ms2", "ms3"])
    print(f"Batch sync results: {batch_result}")
    
    # Test statistics
    print("Getting sync statistics...")
    stats = enhanced_bridge.get_sync_statistics()
    print(f"Sync statistics: {stats}")
    
    # Test health check
    print("Performing health check...")
    health = enhanced_bridge.health_check()
    print(f"Health status: {health['overall_status']}")
    
    # Test enhanced agent bridge
    print("Testing enhanced agent bridge...")
    agent_status = agent_bridge.get_enhanced_bridge_status()
    print(f"Agent bridge features: {agent_status['enhanced_features']}")
    
    # Cleanup
    enhanced_bridge.shutdown()
    print("Enhanced bridge shutdown complete")