"""
Comprehensive tests for Automated Review Coordination Engine
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json

# Import the modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.automated_coordination_engine import (
    AutomatedCoordinationEngine, CoordinationContext, ReviewStage,
    CoordinationStatus, InterventionType, AutomationRule
)
from models.reviewer_matcher import ManuscriptProfile, ReviewerProfile
from models.ojs_coordination_integrator import OJSCoordinationIntegrator

@pytest.fixture
def coordination_config():
    """Test configuration for coordination engine"""
    return {
        'reviewer_matching': {
            'ml_enabled': True,
            'optimization_level': 'test'
        },
        'communication': {
            'smtp': {'enabled': False},
            'auto_reminders': True,
            'escalation_enabled': True
        },
        'monitoring_interval': 60  # 1 minute for testing
    }

@pytest.fixture
def coordination_engine(coordination_config):
    """Create coordination engine for testing"""
    return AutomatedCoordinationEngine(coordination_config)

@pytest.fixture
def sample_manuscript():
    """Sample manuscript for testing"""
    return ManuscriptProfile(
        manuscript_id="test_ms_001",
        title="Advanced Machine Learning Techniques in Academic Publishing",
        abstract="This paper presents novel approaches to automated manuscript processing...",
        keywords=["machine learning", "automation", "academic publishing"],
        subject_areas=["computer science", "artificial intelligence"],
        authors=["Dr. Jane Smith", "Prof. Michael Chen"],
        author_institutions=["University A", "Institute B"],
        manuscript_type="research",
        urgency_level="medium",
        required_expertise=["machine learning", "AI"],
        submission_date=datetime.now().isoformat(),
        target_review_date=(datetime.now() + timedelta(days=30)).isoformat(),
        language="en",
        special_requirements=[]
    )

@pytest.fixture
def sample_reviewers():
    """Sample reviewers for testing"""
    return [
        ReviewerProfile(
            reviewer_id=1,
            name="Dr. Alice Johnson",
            email="alice.johnson@university.edu",
            expertise_areas=["machine learning", "data science"],
            keywords=["ML", "AI", "algorithms"],
            current_workload=2,
            max_workload=5,
            avg_review_time=14,
            quality_score=4.8,
            reliability_score=0.95,
            availability_status="available",
            last_assignment_date=None,
            preferred_manuscript_types=["research"],
            language_preferences=["en"],
            timezone="UTC",
            response_rate=0.92,
            past_collaborations=[],
            conflict_of_interest=[]
        ),
        ReviewerProfile(
            reviewer_id=2,
            name="Prof. Bob Wilson",
            email="bob.wilson@institute.org",
            expertise_areas=["artificial intelligence", "automation"],
            keywords=["AI", "automation", "systems"],
            current_workload=1,
            max_workload=4,
            avg_review_time=18,
            quality_score=4.6,
            reliability_score=0.88,
            availability_status="available",
            last_assignment_date=None,
            preferred_manuscript_types=["research", "review"],
            language_preferences=["en"],
            timezone="UTC",
            response_rate=0.85,
            past_collaborations=[],
            conflict_of_interest=[]
        )
    ]

class TestAutomatedCoordinationEngine:
    """Test suite for Automated Coordination Engine"""
    
    @pytest.mark.asyncio
    async def test_coordination_initiation(self, coordination_engine, sample_manuscript, sample_reviewers):
        """Test coordination initiation"""
        context = await coordination_engine.initiate_coordination(sample_manuscript, sample_reviewers)
        
        assert context is not None
        assert context.manuscript_id == sample_manuscript.manuscript_id
        assert context.current_stage == ReviewStage.INITIATED
        assert context.coordination_status == CoordinationStatus.ACTIVE
        assert len(context.assigned_reviewers) <= len(sample_reviewers)
    
    @pytest.mark.asyncio
    async def test_stage_progression(self, coordination_engine, sample_manuscript, sample_reviewers):
        """Test stage progression functionality"""
        context = await coordination_engine.initiate_coordination(sample_manuscript, sample_reviewers)
        
        initial_stage = context.current_stage
        await coordination_engine._progress_to_stage(context, ReviewStage.REVIEWER_ASSIGNMENT)
        
        assert context.current_stage == ReviewStage.REVIEWER_ASSIGNMENT
        assert context.current_stage != initial_stage
        assert f'stage_transition_{ReviewStage.REVIEWER_ASSIGNMENT.value}' in context.metadata
    
    @pytest.mark.asyncio
    async def test_reviewer_assignment(self, coordination_engine, sample_manuscript, sample_reviewers):
        """Test automated reviewer assignment"""
        context = await coordination_engine.initiate_coordination(sample_manuscript, sample_reviewers)
        
        # Check that reviewers were assigned
        assert len(context.review_assignments) > 0
        assert len(context.assigned_reviewers) > 0
        
        # Check that assignments have required fields
        for assignment in context.review_assignments:
            assert assignment.manuscript_id == sample_manuscript.manuscript_id
            assert assignment.reviewer_id in [r.reviewer_id for r in sample_reviewers]
            assert assignment.match_score > 0
            assert assignment.confidence > 0
    
    @pytest.mark.asyncio
    async def test_reviewer_response_accepted(self, coordination_engine, sample_manuscript, sample_reviewers):
        """Test processing accepted reviewer response"""
        context = await coordination_engine.initiate_coordination(sample_manuscript, sample_reviewers)
        
        # Get first assigned reviewer
        first_assignment = context.review_assignments[0]
        reviewer_id = first_assignment.reviewer_id
        
        # Process acceptance
        await coordination_engine.process_reviewer_response(
            context.manuscript_id, reviewer_id, 'accepted'
        )
        
        # Check that response was recorded
        acceptance_recorded = False
        for assignment in context.review_assignments:
            if assignment.reviewer_id == reviewer_id:
                if assignment.metadata and assignment.metadata.get('invitation_status') == 'accepted':
                    acceptance_recorded = True
                    break
        
        assert acceptance_recorded
        
        # Check communication history
        response_recorded = any(
            comm['type'] == 'reviewer_response' and comm['response'] == 'accepted'
            for comm in context.communication_history
        )
        assert response_recorded
    
    @pytest.mark.asyncio
    async def test_reviewer_response_declined(self, coordination_engine, sample_manuscript, sample_reviewers):
        """Test processing declined reviewer response"""
        context = await coordination_engine.initiate_coordination(sample_manuscript, sample_reviewers)
        
        # Get first assigned reviewer
        first_assignment = context.review_assignments[0]
        reviewer_id = first_assignment.reviewer_id
        
        # Process decline
        await coordination_engine.process_reviewer_response(
            context.manuscript_id, reviewer_id, 'declined'
        )
        
        # Check that decline was recorded
        decline_recorded = False
        for assignment in context.review_assignments:
            if assignment.reviewer_id == reviewer_id:
                if assignment.metadata and assignment.metadata.get('invitation_status') == 'declined':
                    decline_recorded = True
                    break
        
        assert decline_recorded
        
        # Check that replacement intervention was recorded
        replacement_intervention = any(
            intervention['type'] == InterventionType.REVIEWER_REASSIGNMENT.value
            for intervention in context.intervention_history
        )
        assert replacement_intervention
    
    @pytest.mark.asyncio
    async def test_review_submission(self, coordination_engine, sample_manuscript, sample_reviewers):
        """Test review submission processing"""
        context = await coordination_engine.initiate_coordination(sample_manuscript, sample_reviewers)
        
        # Accept invitation first
        first_assignment = context.review_assignments[0]
        reviewer_id = first_assignment.reviewer_id
        
        await coordination_engine.process_reviewer_response(
            context.manuscript_id, reviewer_id, 'accepted'
        )
        
        # Submit review
        review_data = {
            'recommendation': 'accept',
            'comments': 'Excellent paper with novel contributions',
            'technical_quality': 4.5,
            'originality': 4.2,
            'clarity': 4.0
        }
        
        await coordination_engine.process_review_submission(
            context.manuscript_id, reviewer_id, review_data
        )
        
        # Check that submission was recorded
        submission_recorded = False
        for assignment in context.review_assignments:
            if assignment.reviewer_id == reviewer_id:
                if assignment.metadata and assignment.metadata.get('review_submitted'):
                    submission_recorded = True
                    assert assignment.metadata.get('review_data') == review_data
                    break
        
        assert submission_recorded
    
    @pytest.mark.asyncio
    async def test_quality_assessment(self, coordination_engine, sample_manuscript, sample_reviewers):
        """Test automated quality assessment"""
        context = await coordination_engine.initiate_coordination(sample_manuscript, sample_reviewers)
        
        # Mock scenario where all reviews are submitted
        for assignment in context.review_assignments:
            if not assignment.metadata:
                assignment.metadata = {}
            assignment.metadata['invitation_status'] = 'accepted'
            assignment.metadata['review_submitted'] = True
            assignment.metadata['review_data'] = {
                'recommendation': 'accept',
                'comments': 'Good quality review'
            }
        
        # Trigger quality assessment
        await coordination_engine._perform_quality_assessment(context)
        
        # Check that quality metrics were calculated
        assert 'average_quality' in context.quality_metrics
        assert 'quality_variance' in context.quality_metrics
        assert 'assessment_date' in context.quality_metrics
        
        # Check that quality scores were assigned to assignments
        for assignment in context.review_assignments:
            if assignment.metadata and assignment.metadata.get('review_submitted'):
                assert 'quality_score' in assignment.metadata
    
    def test_automation_rule_evaluation(self, coordination_engine, sample_manuscript, sample_reviewers):
        """Test automation rule condition evaluation"""
        # Create test context
        context = CoordinationContext(
            manuscript_id=sample_manuscript.manuscript_id,
            manuscript_profile=sample_manuscript,
            current_stage=ReviewStage.REVIEW_IN_PROGRESS,
            coordination_status=CoordinationStatus.ACTIVE,
            assigned_reviewers=sample_reviewers,
            review_assignments=[],
            communication_history=[],
            quality_metrics={},
            timeline_metrics={'coordination_started': datetime.now().isoformat()},
            intervention_history=[],
            metadata={}
        )
        
        # Test rule that should trigger
        rule = AutomationRule(
            rule_id='test_rule',
            rule_name='Test Rule',
            trigger_conditions={
                'stage': ReviewStage.REVIEW_IN_PROGRESS,
                'days_since_assignment': 0  # Should trigger immediately
            },
            actions=[{'type': 'test_action'}],
            priority=5,
            enabled=True,
            execution_count=0,
            last_executed=None
        )
        
        # Run evaluation asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            coordination_engine._evaluate_rule_conditions(context, rule)
        )
        
        loop.close()
        
        assert result is True
        
        # Test rule that should not trigger
        rule.trigger_conditions['stage'] = ReviewStage.COMPLETED
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            coordination_engine._evaluate_rule_conditions(context, rule)
        )
        
        loop.close()
        
        assert result is False
    
    def test_coordination_metrics(self, coordination_engine):
        """Test coordination metrics tracking"""
        metrics = coordination_engine.get_coordination_metrics()
        
        assert hasattr(metrics, 'total_coordinated_reviews')
        assert hasattr(metrics, 'average_coordination_time')
        assert hasattr(metrics, 'successful_completions')
        assert hasattr(metrics, 'escalation_rate')
        assert hasattr(metrics, 'automation_success_rate')
        assert hasattr(metrics, 'timeline_adherence')
    
    def test_active_coordinations_tracking(self, coordination_engine, sample_manuscript, sample_reviewers):
        """Test active coordinations tracking"""
        # Initially no active coordinations
        assert len(coordination_engine.get_active_coordinations()) == 0
        
        # Start coordination
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        context = loop.run_until_complete(
            coordination_engine.initiate_coordination(sample_manuscript, sample_reviewers)
        )
        
        loop.close()
        
        # Should have one active coordination
        active_coordinations = coordination_engine.get_active_coordinations()
        assert len(active_coordinations) == 1
        assert sample_manuscript.manuscript_id in active_coordinations
        
        # Retrieved context should match
        retrieved_context = coordination_engine.get_coordination_status(sample_manuscript.manuscript_id)
        assert retrieved_context is not None
        assert retrieved_context.manuscript_id == context.manuscript_id

class TestOJSCoordinationIntegrator:
    """Test suite for OJS Coordination Integrator"""
    
    @pytest.fixture
    def ojs_config(self):
        """Test OJS configuration"""
        return {
            'base_url': 'http://localhost:8000',
            'api_key': 'test_key',
            'timeout': 30
        }
    
    @pytest.fixture
    def integrator(self, ojs_config, coordination_config):
        """Create OJS integrator for testing"""
        with patch('models.ojs_coordination_integrator.EnhancedOJSBridge'):
            return OJSCoordinationIntegrator(ojs_config, coordination_config)
    
    @pytest.mark.asyncio
    async def test_manuscript_profile_conversion(self, integrator):
        """Test conversion from OJS manuscript to coordination profile"""
        from models.ojs_coordination_integrator import OJSManuscriptData
        
        ojs_manuscript = OJSManuscriptData(
            submission_id=123,
            title="Test Manuscript",
            abstract="Test abstract",
            authors=[
                {'firstName': 'Jane', 'lastName': 'Smith', 'affiliation': 'University A'},
                {'firstName': 'John', 'lastName': 'Doe', 'affiliation': 'Institute B'}
            ],
            subject_classification=['computer science', 'AI'],
            keywords=['machine learning', 'automation'],
            submission_date=datetime.now().isoformat(),
            current_stage='external-review',
            current_status='in_review',
            editorial_notes='',
            file_urls=[],
            metadata={'priority': 'regular', 'articleType': 'research'}
        )
        
        manuscript_profile = await integrator._convert_to_manuscript_profile(ojs_manuscript)
        
        assert manuscript_profile.manuscript_id == '123'
        assert manuscript_profile.title == 'Test Manuscript'
        assert manuscript_profile.abstract == 'Test abstract'
        assert len(manuscript_profile.authors) == 2
        assert 'Jane Smith' in manuscript_profile.authors
        assert manuscript_profile.subject_areas == ['computer science', 'AI']
        assert manuscript_profile.manuscript_type == 'research'
        assert manuscript_profile.urgency_level == 'medium'
    
    @pytest.mark.asyncio
    async def test_reviewer_profile_conversion(self, integrator):
        """Test conversion from OJS reviewer to coordination profile"""
        from models.ojs_coordination_integrator import OJSReviewerData
        
        ojs_reviewers = [
            OJSReviewerData(
                user_id=1,
                username='reviewer1',
                first_name='Alice',
                last_name='Johnson',
                email='alice@university.edu',
                affiliation='University A',
                bio='Expert in machine learning',
                interests=['machine learning', 'AI'],
                review_history={
                    'averageReviewTime': 15,
                    'averageQuality': 4.5,
                    'reliabilityScore': 0.92,
                    'responseRate': 0.88
                },
                availability_status='available',
                workload_info={
                    'currentAssignments': 2,
                    'maxAssignments': 5
                }
            )
        ]
        
        reviewer_profiles = await integrator._convert_to_reviewer_profiles(ojs_reviewers)
        
        assert len(reviewer_profiles) == 1
        reviewer = reviewer_profiles[0]
        
        assert reviewer.reviewer_id == 1
        assert reviewer.name == 'Alice Johnson'
        assert reviewer.email == 'alice@university.edu'
        assert reviewer.expertise_areas == ['machine learning', 'AI']
        assert reviewer.avg_review_time == 15
        assert reviewer.quality_score == 4.5
        assert reviewer.current_workload == 2
        assert reviewer.max_workload == 5
    
    @pytest.mark.asyncio
    async def test_webhook_handling(self, integrator):
        """Test OJS webhook handling"""
        webhook_data = {
            'event_type': 'review_submitted',
            'submission_id': 123,
            'reviewer_id': 1,
            'review_data': {
                'recommendation': 'accept',
                'comments': 'Good paper'
            }
        }
        
        with patch.object(integrator, '_handle_review_submission', new_callable=AsyncMock) as mock_handler:
            response = await integrator.handle_ojs_webhook(webhook_data)
            
            assert response['status'] == 'processed'
            assert 'review_processed' in response['actions_taken']
            mock_handler.assert_called_once_with(123, webhook_data)
    
    def test_sync_mapping_creation(self, integrator):
        """Test sync mapping creation between OJS and coordination"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        loop.run_until_complete(
            integrator._create_sync_link(123, 'coord_ms_001')
        )
        
        loop.close()
        
        assert hasattr(integrator, 'sync_mappings')
        assert integrator.sync_mappings['123'] == 'coord_ms_001'
        assert integrator.sync_mappings['coord_ms_001'] == '123'
    
    @pytest.mark.asyncio
    async def test_integration_health_check(self, integrator):
        """Test integration health check"""
        with patch.object(integrator.ojs_bridge, 'health_check', new_callable=AsyncMock) as mock_ojs_health:
            mock_ojs_health.return_value = {'status': 'healthy'}
            
            health_status = await integrator.health_check()
            
            assert health_status['status'] in ['healthy', 'degraded']
            assert 'components' in health_status
            assert 'ojs_bridge' in health_status['components']
            assert 'coordination_engine' in health_status['components']
            assert 'integration_layer' in health_status['components']

class TestEnhancedReviewCoordinationAgent:
    """Test suite for Enhanced Review Coordination Agent"""
    
    @pytest.fixture
    def enhanced_agent(self):
        """Create enhanced agent for testing"""
        # Mock the heavy imports to speed up tests
        with patch('models.automated_coordination_engine.AutomatedCoordinationEngine'):
            with patch('models.ojs_coordination_integrator.OJSCoordinationIntegrator'):
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'microservices', 'review-coordination'))
                from app import EnhancedReviewCoordinationAgent
                return EnhancedReviewCoordinationAgent()
    
    def test_agent_initialization(self, enhanced_agent):
        """Test agent initialization with enhanced capabilities"""
        assert enhanced_agent.agent_type == 'review_coordination'
        assert 'automated_coordination' in enhanced_agent.capabilities
        assert 'intelligent_reviewer_matching' in enhanced_agent.capabilities
        assert 'ojs_integration' in enhanced_agent.capabilities
        assert enhanced_agent.performance['automation_success_rate'] > 0
    
    def test_agent_data_structure(self, enhanced_agent):
        """Test agent data structure"""
        agent_data = enhanced_agent.get_agent_data()
        
        assert agent_data['id'] == 'agent_enhanced_review_coordination'
        assert agent_data['version'] == '2.0-automated'
        assert 'coordination_stats' in agent_data
        assert 'automation_enabled' in agent_data['coordination_stats']
        assert agent_data['coordination_stats']['automation_enabled'] is True
    
    def test_coordination_action_processing(self, enhanced_agent):
        """Test coordination action processing"""
        test_data = {
            'action_type': 'coordinate',
            'manuscript': {
                'id': 'test_manuscript',
                'title': 'Test Paper',
                'subject_areas': ['AI', 'ML']
            }
        }
        
        result = enhanced_agent.process_action(test_data)
        
        assert result['coordination_initiated'] is True
        assert result['automation_level'] == 'full'
        assert 'estimated_completion' in result
        assert 'quality_prediction' in result
    
    def test_status_action_processing(self, enhanced_agent):
        """Test status action processing"""
        test_data = {
            'action_type': 'status',
            'manuscript_id': 'test_manuscript'
        }
        
        result = enhanced_agent.process_action(test_data)
        
        assert 'active_coordinations' in result
        assert 'automation_success_rate' in result
        assert 'system_health' in result
    
    def test_metrics_action_processing(self, enhanced_agent):
        """Test metrics action processing"""
        test_data = {
            'action_type': 'metrics'
        }
        
        result = enhanced_agent.process_action(test_data)
        
        assert 'total_coordinated' in result
        assert 'success_rate' in result
        assert 'escalation_rate' in result
        assert 'timeline_adherence' in result

@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration test scenarios for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_coordination_workflow(self, coordination_engine, sample_manuscript, sample_reviewers):
        """Test complete coordination workflow from start to finish"""
        # 1. Initiate coordination
        context = await coordination_engine.initiate_coordination(sample_manuscript, sample_reviewers)
        assert context.current_stage == ReviewStage.INITIATED
        
        # 2. Reviewers should be assigned
        assert len(context.review_assignments) > 0
        
        # 3. Process reviewer acceptances
        for assignment in context.review_assignments:
            await coordination_engine.process_reviewer_response(
                context.manuscript_id, assignment.reviewer_id, 'accepted'
            )
        
        # 4. Submit reviews
        for assignment in context.review_assignments:
            review_data = {
                'recommendation': 'accept',
                'comments': 'Excellent work',
                'quality_score': 4.5
            }
            await coordination_engine.process_review_submission(
                context.manuscript_id, assignment.reviewer_id, review_data
            )
        
        # 5. Verify final state
        final_context = coordination_engine.get_coordination_status(context.manuscript_id)
        assert len(final_context.communication_history) > 0
        assert 'quality_metrics' in final_context.__dict__
    
    def test_performance_under_load(self, coordination_engine):
        """Test system performance with multiple concurrent coordinations"""
        import time
        
        start_time = time.time()
        
        # Create multiple manuscripts
        manuscripts = []
        for i in range(10):
            manuscript = ManuscriptProfile(
                manuscript_id=f"load_test_ms_{i}",
                title=f"Load Test Manuscript {i}",
                abstract="Test abstract for load testing",
                keywords=["test"],
                subject_areas=["computer science"],
                authors=[f"Author {i}"],
                author_institutions=["Test University"],
                manuscript_type="research",
                urgency_level="medium",
                required_expertise=["testing"],
                submission_date=datetime.now().isoformat(),
                target_review_date=(datetime.now() + timedelta(days=30)).isoformat(),
                language="en",
                special_requirements=[]
            )
            manuscripts.append(manuscript)
        
        # Process all manuscripts
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def process_manuscripts():
            tasks = []
            for manuscript in manuscripts:
                task = coordination_engine.initiate_coordination(manuscript, [])
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        results = loop.run_until_complete(process_manuscripts())
        loop.close()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify performance
        assert len(results) == 10
        assert processing_time < 30  # Should complete within 30 seconds
        
        # Verify all coordinations were created
        active_coordinations = coordination_engine.get_active_coordinations()
        assert len(active_coordinations) >= 10

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])