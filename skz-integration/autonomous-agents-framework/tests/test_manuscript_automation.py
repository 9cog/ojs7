"""
Comprehensive Tests for Manuscript Processing Automation
Tests the complete automation workflow and API endpoints
"""
import pytest
import asyncio
import json
import uuid
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.manuscript_processing_automation import (
    ManuscriptProcessingAutomation,
    ManuscriptMetadata,
    ProcessingTask,
    AutomationWorkflow,
    ManuscriptStatus,
    ProcessingStage,
    AutomationPriority
)

class TestManuscriptProcessingAutomation:
    """Test suite for manuscript processing automation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = {
            'agent_endpoints': {
                'research_discovery': 'http://localhost:5001/api/agents',
                'submission_assistant': 'http://localhost:5002/api/agents', 
                'editorial_orchestration': 'http://localhost:5003/api/agents',
                'review_coordination': 'http://localhost:5004/api/agents',
                'content_quality': 'http://localhost:5005/api/agents',
                'publishing_production': 'http://localhost:5006/api/agents',
                'analytics_monitoring': 'http://localhost:5007/api/agents'
            },
            'timeout_seconds': 300,
            'max_retries': 3
        }
        
        self.automation = ManuscriptProcessingAutomation(self.config)
        
        # Sample manuscript data
        self.sample_manuscript = {
            'id': 'test_manuscript_001',
            'title': 'Novel Hyaluronic Acid Formulations for Anti-Aging Applications',
            'authors': [
                {'name': 'Dr. Sarah Johnson', 'email': 'sarah.johnson@cosmetics.com'},
                {'name': 'Prof. Michael Chen', 'email': 'michael.chen@research.edu'}
            ],
            'abstract': 'This study investigates the efficacy of novel hyaluronic acid formulations in cosmetic applications for anti-aging treatments. We developed three different molecular weight formulations and tested their penetration, hydration, and anti-aging effects.',
            'keywords': ['hyaluronic acid', 'anti-aging', 'cosmetic formulation', 'skin hydration', 'molecular weight'],
            'research_type': 'experimental',
            'field_of_study': 'cosmetic_science',
            'file_paths': ['/uploads/manuscript.pdf', '/uploads/supplementary.pdf'],
            'priority': 2,
            'special_requirements': ['inci_verification', 'safety_assessment']
        }
    
    @pytest.mark.asyncio
    async def test_manuscript_submission(self):
        """Test manuscript submission for automation"""
        # Submit manuscript
        workflow_id = await self.automation.submit_manuscript_for_automation(self.sample_manuscript)
        
        # Verify workflow was created
        assert workflow_id is not None
        assert workflow_id in self.automation.active_workflows
        
        workflow = self.automation.active_workflows[workflow_id]
        assert workflow.manuscript_id == self.sample_manuscript['id']
        assert workflow.status == ManuscriptStatus.RECEIVED
        assert workflow.current_stage == ProcessingStage.INTAKE
        assert len(workflow.processing_tasks) > 0
    
    def test_task_generation(self):
        """Test processing task generation based on manuscript type"""
        metadata = ManuscriptMetadata(
            manuscript_id='test_001',
            title='Test Manuscript',
            authors=[],
            abstract='Test abstract',
            keywords=['test'],
            research_type='experimental',
            field_of_study='cosmetic_science',
            submission_date=datetime.utcnow().isoformat(),
            file_paths=[],
            priority=AutomationPriority.NORMAL,
            special_requirements=[]
        )
        
        tasks = self.automation._generate_processing_tasks(metadata)
        
        # Verify task structure
        assert len(tasks) >= 3  # Should have at least initial validation, quality assessment, and editorial review
        
        # Check for required tasks
        task_types = [task.agent_type for task in tasks]
        assert 'submission_assistant' in task_types
        assert 'content_quality' in task_types
        assert 'editorial_orchestration' in task_types
        
        # Verify dependency structure
        initial_tasks = [task for task in tasks if not task.dependencies]
        assert len(initial_tasks) > 0  # Should have at least one task without dependencies
    
    @pytest.mark.asyncio
    async def test_task_processing(self):
        """Test individual task processing"""
        # Create a simple task
        task = ProcessingTask(
            task_id=str(uuid.uuid4()),
            task_name="Test Task",
            agent_type="submission_assistant",
            stage=ProcessingStage.INITIAL_VALIDATION,
            dependencies=[],
            input_data={'test': 'data'},
            expected_output={},
            timeout_minutes=10,
            retry_count=0,
            max_retries=3,
            created_at=datetime.utcnow().isoformat()
        )
        
        # Create workflow
        workflow = AutomationWorkflow(
            workflow_id=str(uuid.uuid4()),
            manuscript_id='test_001',
            manuscript_metadata=ManuscriptMetadata(
                manuscript_id='test_001',
                title='Test',
                authors=[],
                abstract='Test',
                keywords=[],
                research_type='test',
                field_of_study='test',
                submission_date=datetime.utcnow().isoformat(),
                file_paths=[],
                priority=AutomationPriority.NORMAL,
                special_requirements=[]
            ),
            processing_tasks=[task],
            current_stage=ProcessingStage.INTAKE,
            status=ManuscriptStatus.RECEIVED,
            progress_percentage=0.0,
            estimated_completion=datetime.utcnow().isoformat(),
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            notifications_sent=[]
        )
        
        # Process task
        await self.automation._process_task(task, workflow)
        
        # Verify task was processed (mock API should return success)
        assert task.status in ["completed", "pending"]  # pending for retry, completed for success
        if task.status == "completed":
            assert task.result is not None
            assert task.completed_at is not None
    
    def test_workflow_status_tracking(self):
        """Test workflow status tracking and progress calculation"""
        # Create workflow with multiple tasks
        tasks = []
        for i in range(3):
            task = ProcessingTask(
                task_id=str(uuid.uuid4()),
                task_name=f"Task {i+1}",
                agent_type="submission_assistant",
                stage=ProcessingStage.INITIAL_VALIDATION,
                dependencies=[],
                input_data={},
                expected_output={},
                timeout_minutes=10,
                retry_count=0,
                max_retries=3,
                created_at=datetime.utcnow().isoformat()
            )
            tasks.append(task)
        
        workflow_id = str(uuid.uuid4())
        workflow = AutomationWorkflow(
            workflow_id=workflow_id,
            manuscript_id='test_001',
            manuscript_metadata=ManuscriptMetadata(
                manuscript_id='test_001',
                title='Test',
                authors=[],
                abstract='Test',
                keywords=[],
                research_type='test',
                field_of_study='test',
                submission_date=datetime.utcnow().isoformat(),
                file_paths=[],
                priority=AutomationPriority.NORMAL,
                special_requirements=[]
            ),
            processing_tasks=tasks,
            current_stage=ProcessingStage.INTAKE,
            status=ManuscriptStatus.RECEIVED,
            progress_percentage=0.0,
            estimated_completion=datetime.utcnow().isoformat(),
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            notifications_sent=[]
        )
        
        self.automation.active_workflows[workflow_id] = workflow
        
        # Complete first task
        tasks[0].status = "completed"
        tasks[0].completed_at = datetime.utcnow().isoformat()
        
        # Update progress
        asyncio.run(self.automation._update_workflow_progress(workflow_id))
        
        # Check progress
        updated_workflow = self.automation.active_workflows[workflow_id]
        assert updated_workflow.progress_percentage == 33.33 or abs(updated_workflow.progress_percentage - 33.33) < 0.1
        
        # Complete all tasks
        for task in tasks:
            task.status = "completed"
            task.completed_at = datetime.utcnow().isoformat()
        
        asyncio.run(self.automation._update_workflow_progress(workflow_id))
        assert updated_workflow.progress_percentage == 100.0
    
    def test_automation_metrics(self):
        """Test automation performance metrics calculation"""
        # Create completed workflow
        workflow_id = str(uuid.uuid4())
        workflow = AutomationWorkflow(
            workflow_id=workflow_id,
            manuscript_id='test_001',
            manuscript_metadata=ManuscriptMetadata(
                manuscript_id='test_001',
                title='Test',
                authors=[],
                abstract='Test',
                keywords=[],
                research_type='test',
                field_of_study='test',
                submission_date=datetime.utcnow().isoformat(),
                file_paths=[],
                priority=AutomationPriority.NORMAL,
                special_requirements=[]
            ),
            processing_tasks=[],
            current_stage=ProcessingStage.COMPLETED,
            status=ManuscriptStatus.ACCEPTED,
            progress_percentage=100.0,
            estimated_completion=datetime.utcnow().isoformat(),
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            notifications_sent=[]
        )
        
        self.automation.completed_workflows[workflow_id] = workflow
        self.automation._update_performance_metrics()
        
        metrics = self.automation.get_automation_metrics()
        
        assert 'performance_metrics' in metrics
        assert metrics['active_workflows'] == 0
        assert metrics['completed_workflows'] == 1
        assert metrics['performance_metrics']['total_processed'] == 1
        assert metrics['performance_metrics']['success_rate'] == 1.0
    
    def test_routing_rules(self):
        """Test intelligent routing based on field of study"""
        # Test cosmetic science routing
        cosmetic_metadata = ManuscriptMetadata(
            manuscript_id='cosmetic_001',
            title='Test',
            authors=[],
            abstract='Test',
            keywords=[],
            research_type='experimental',
            field_of_study='cosmetic_science',
            submission_date=datetime.utcnow().isoformat(),
            file_paths=[],
            priority=AutomationPriority.NORMAL,
            special_requirements=[]
        )
        
        cosmetic_tasks = self.automation._generate_processing_tasks(cosmetic_metadata)
        cosmetic_agents = [task.agent_type for task in cosmetic_tasks]
        
        # Should include research discovery for cosmetic science
        assert 'research_discovery' in cosmetic_agents
        
        # Test clinical research routing
        clinical_metadata = ManuscriptMetadata(
            manuscript_id='clinical_001',
            title='Test',
            authors=[],
            abstract='Test',
            keywords=[],
            research_type='clinical',
            field_of_study='clinical_research',
            submission_date=datetime.utcnow().isoformat(),
            file_paths=[],
            priority=AutomationPriority.NORMAL,
            special_requirements=[]
        )
        
        clinical_tasks = self.automation._generate_processing_tasks(clinical_metadata)
        clinical_agents = [task.agent_type for task in clinical_tasks]
        
        # Clinical research should prioritize different agents
        assert 'content_quality' in clinical_agents
        assert 'editorial_orchestration' in clinical_agents
    
    def test_error_handling(self):
        """Test error handling and retry mechanisms"""
        # Create task that will fail
        task = ProcessingTask(
            task_id=str(uuid.uuid4()),
            task_name="Failing Task",
            agent_type="nonexistent_agent",  # Will cause error
            stage=ProcessingStage.INITIAL_VALIDATION,
            dependencies=[],
            input_data={},
            expected_output={},
            timeout_minutes=1,
            retry_count=0,
            max_retries=2,
            created_at=datetime.utcnow().isoformat()
        )
        
        workflow = AutomationWorkflow(
            workflow_id=str(uuid.uuid4()),
            manuscript_id='test_001',
            manuscript_metadata=ManuscriptMetadata(
                manuscript_id='test_001',
                title='Test',
                authors=[],
                abstract='Test',
                keywords=[],
                research_type='test',
                field_of_study='test',
                submission_date=datetime.utcnow().isoformat(),
                file_paths=[],
                priority=AutomationPriority.NORMAL,
                special_requirements=[]
            ),
            processing_tasks=[task],
            current_stage=ProcessingStage.INTAKE,
            status=ManuscriptStatus.RECEIVED,
            progress_percentage=0.0,
            estimated_completion=datetime.utcnow().isoformat(),
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            notifications_sent=[]
        )
        
        # Process task (should fail)
        asyncio.run(self.automation._process_task(task, workflow))
        
        # Should have error and be marked for retry or failed
        assert task.error_message is not None
        assert task.retry_count > 0 or task.status == "failed"


class TestManuscriptAutomationAPI:
    """Test suite for manuscript automation API endpoints"""
    
    def setup_method(self):
        """Setup API test environment"""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        
        from main import app
        from routes.manuscript_automation_api import init_automation
        
        # Initialize app for testing
        app.config['TESTING'] = True
        self.client = app.test_client()
        
        # Initialize automation
        config = {
            'agent_endpoints': {
                'submission_assistant': 'http://localhost:5002/api/agents'
            }
        }
        init_automation(config)
    
    def test_health_check(self):
        """Test automation health check endpoint"""
        response = self.client.get('/api/v1/automation/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'status' in data
        assert 'system_info' in data
    
    def test_submit_manuscript_api(self):
        """Test manuscript submission API endpoint"""
        manuscript_data = {
            'id': 'api_test_001',
            'title': 'API Test Manuscript',
            'authors': [{'name': 'Test Author', 'email': 'test@example.com'}],
            'abstract': 'This is a test manuscript for API testing.',
            'keywords': ['test', 'api'],
            'research_type': 'experimental',
            'field_of_study': 'cosmetic_science'
        }
        
        response = self.client.post(
            '/api/v1/automation/submit',
            data=json.dumps(manuscript_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'workflow_id' in data
        assert 'status' in data
    
    def test_get_metrics_api(self):
        """Test automation metrics API endpoint"""
        response = self.client.get('/api/v1/automation/metrics')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'performance_metrics' in data
        assert 'active_workflows' in data
    
    def test_invalid_submission(self):
        """Test API validation for invalid manuscript submission"""
        invalid_data = {
            'title': 'Missing required fields'
            # Missing authors and abstract
        }
        
        response = self.client.post(
            '/api/v1/automation/submit',
            data=json.dumps(invalid_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data

if __name__ == '__main__':
    pytest.main([__file__, '-v'])