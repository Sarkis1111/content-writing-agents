"""
Editor Agent Integration Tests

This module provides comprehensive integration tests for the Editor Agent system,
testing all components including LangGraph workflows, tool integration, quality assurance,
human escalation, and state management.

Test Categories:
- Core Editor Agent functionality
- LangGraph workflow execution  
- Tool integration and orchestration
- Quality assurance and gate systems
- Human escalation workflows
- State management and persistence
- Performance and reliability tests
"""

import asyncio
import os
import sys
import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, src_path)

# Test imports
try:
    from .editor_agent import EditorAgent, EditorAgentConfig, EditingResults
    from .tool_integration import ToolOrchestrator, ToolExecutionPlan, ExecutionMode, ToolType
    from .quality_assurance import (
        QualityAssessmentEngine, QualityCriteria, QualityDimension, 
        QualityMetrics, QualityLevel, AssessmentResult
    )
    from .human_escalation import (
        EscalationManager, EscalationContext, EscalationReason, 
        EscalationPriority, ReviewerExpertise
    )
    from .state_management import (
        StateManager, EditingSessionState, RevisionType, 
        WorkflowState, QualityGateDecision
    )
    
    MODULES_AVAILABLE = True
    
except ImportError as e:
    print(f"Warning: Some modules not available for testing: {e}")
    MODULES_AVAILABLE = False
    
    # Create mock classes for testing structure
    class EditorAgent:
        def __init__(self, config=None): pass
        async def edit_content(self, content, requirements=None): 
            return type('MockResult', (), {'final_quality_score': 85.0})()
    
    class ToolOrchestrator:
        def __init__(self): pass
        async def edit_content(self, content, requirements=None, mode=None):
            return type('MockResult', (), {'overall_score': 78.0})()


class TestEditorAgentCore:
    """Test core Editor Agent functionality"""
    
    @pytest.fixture
    def sample_content(self):
        """Sample content for testing"""
        return """
        This is a test content for editing validation. It has some grammer mistakes and could be improved.
        The SEO optimization is not great and readability might be better with shorter sentences.
        
        Overall this content needs significant editing to meet quality standards for professional publication.
        We want to make sure it's perfect before we publish it to our audience for maximum engagement.
        """
    
    @pytest.fixture
    def editing_requirements(self):
        """Sample editing requirements"""
        return {
            "target_keywords": ["content", "editing", "quality", "professional"],
            "target_audience": "business professionals",
            "content_type": "blog_post",
            "writing_style": "professional",
            "brand_voice": "authoritative",
            "enable_auto_fixes": True,
            "require_human_review": False,
            "seo_requirements": {
                "target_keywords": ["content editing", "quality assurance"],
                "meta_description_length": 160
            }
        }
    
    @pytest.fixture
    def editor_config(self):
        """Editor agent configuration"""
        return EditorAgentConfig(
            min_grammar_score=85.0,
            min_seo_score=75.0,
            min_readability_score=80.0,
            min_overall_quality=80.0,
            max_editing_iterations=3,
            enable_auto_fixes=True,
            require_human_review=False
        )
    
    @pytest.mark.asyncio
    async def test_editor_agent_initialization(self, editor_config):
        """Test editor agent initialization"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        agent = EditorAgent(editor_config)
        
        assert agent.config == editor_config
        assert agent.config.min_grammar_score == 85.0
        assert agent.config.enable_auto_fixes is True
        
        # Test performance stats initialization
        stats = agent.get_performance_stats()
        assert "total_edits" in stats
        assert stats["total_edits"] == 0
    
    @pytest.mark.asyncio
    async def test_content_editing_workflow(self, sample_content, editing_requirements, editor_config):
        """Test complete content editing workflow"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        agent = EditorAgent(editor_config)
        
        # Execute content editing
        results = await agent.edit_content(sample_content, editing_requirements)
        
        # Validate results structure
        assert isinstance(results, EditingResults)
        assert results.original_content == sample_content
        assert results.edited_content is not None
        assert len(results.edited_content) > 0
        
        # Validate quality scores
        assert "overall_score" in results.quality_scores
        assert results.final_quality_score >= 0
        assert results.final_quality_score <= 100
        
        # Validate processing metadata
        assert results.processing_time > 0
        assert results.total_iterations >= 1
        assert len(results.stages_completed) > 0
        
        print(f"✓ Content editing completed: {results.final_quality_score:.1f} score")
    
    @pytest.mark.asyncio
    async def test_multiple_editing_iterations(self, editor_config):
        """Test multiple editing iterations for challenging content"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        # Very poor quality content that should trigger multiple iterations
        poor_content = """
        this content has lots of grammer erors and bad seo and is hard to read with run on sentences that go on forever without proper punctuation or structure and needs alot of work
        """
        
        config = EditorAgentConfig(
            min_overall_quality=85.0,
            max_editing_iterations=3,
            enable_auto_fixes=True
        )
        
        agent = EditorAgent(config)
        results = await agent.edit_content(poor_content)
        
        # Should have multiple iterations for poor content
        assert results.total_iterations >= 2
        assert len(results.changes_applied) > 0
        
        print(f"✓ Multiple iterations completed: {results.total_iterations} iterations")
    
    @pytest.mark.asyncio
    async def test_human_escalation_trigger(self, sample_content, editor_config):
        """Test human escalation triggering"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        # Configure for human escalation
        config = EditorAgentConfig(
            min_overall_quality=95.0,  # Very high threshold
            escalation_quality_threshold=90.0,
            require_human_review=True
        )
        
        agent = EditorAgent(config)
        results = await agent.edit_content(sample_content)
        
        # Should trigger human review due to high thresholds
        assert results.human_review_required is True
        assert len(results.escalation_reasons) > 0
        
        print(f"✓ Human escalation triggered: {results.escalation_reasons}")


class TestToolIntegration:
    """Test tool integration and orchestration"""
    
    @pytest.fixture
    def orchestrator(self):
        """Tool orchestrator instance"""
        return ToolOrchestrator()
    
    @pytest.fixture
    def sample_content(self):
        """Sample content for tool testing"""
        return "This is sample content for testing tool integration with grammer errors and SEO optimization needs."
    
    @pytest.mark.asyncio
    async def test_tool_orchestrator_initialization(self, orchestrator):
        """Test tool orchestrator initialization"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        available_tools = orchestrator.get_available_tools()
        assert len(available_tools) > 0
        
        # Check expected tools are available
        expected_tools = ["grammar", "seo", "readability", "sentiment"]
        for tool in expected_tools:
            assert tool in available_tools
        
        print(f"✓ Tools available: {available_tools}")
    
    @pytest.mark.asyncio
    async def test_sequential_tool_execution(self, orchestrator, sample_content):
        """Test sequential tool execution"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        editing_requirements = {
            "target_keywords": ["testing", "integration"],
            "target_audience": "developers"
        }
        
        results = await orchestrator.edit_content(
            sample_content,
            editing_requirements,
            ExecutionMode.SEQUENTIAL
        )
        
        # Validate results
        assert results.overall_score > 0
        assert len(results.successful_tools) > 0
        assert results.total_execution_time > 0
        
        print(f"✓ Sequential execution: {len(results.successful_tools)} tools successful")
    
    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self, orchestrator, sample_content):
        """Test parallel tool execution"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        editing_requirements = {
            "target_keywords": ["testing", "parallel"],
            "enable_parallel": True
        }
        
        results = await orchestrator.edit_content(
            sample_content,
            editing_requirements,
            ExecutionMode.PARALLEL
        )
        
        # Parallel execution should be faster than sequential for multiple tools
        assert results.total_execution_time > 0
        assert len(results.individual_scores) > 0
        
        print(f"✓ Parallel execution completed in {results.total_execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_adaptive_tool_selection(self, orchestrator):
        """Test adaptive tool selection"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        # Marketing content should trigger SEO and sentiment tools
        marketing_content = "Buy our amazing product now! Limited time offer with incredible benefits!"
        marketing_requirements = {
            "content_type": "marketing_copy",
            "target_keywords": ["product", "offer"],
            "brand_voice": "exciting"
        }
        
        results = await orchestrator.edit_content(
            marketing_content,
            marketing_requirements,
            ExecutionMode.ADAPTIVE
        )
        
        # Should include SEO and sentiment analysis for marketing content
        tool_types = [ToolType.SEO.value, ToolType.SENTIMENT.value]
        successful_tool_values = [tool.value for tool in results.successful_tools]
        
        has_seo_or_sentiment = any(tool in successful_tool_values for tool in tool_types)
        assert has_seo_or_sentiment, f"Expected SEO or sentiment tools, got: {successful_tool_values}"
        
        print(f"✓ Adaptive selection: {successful_tool_values}")


class TestQualityAssurance:
    """Test quality assurance system"""
    
    @pytest.fixture
    def qa_engine(self):
        """Quality assessment engine"""
        from .quality_assurance import get_quality_assessment_engine
        return get_quality_assessment_engine()
    
    @pytest.fixture
    def sample_metrics(self):
        """Sample quality metrics"""
        if not MODULES_AVAILABLE:
            return {}
        
        return {
            QualityDimension.GRAMMAR: QualityMetrics(
                dimension=QualityDimension.GRAMMAR,
                score=88.0,
                level=QualityLevel.GOOD,
                recommendations=["Fix spelling errors"]
            ),
            QualityDimension.SEO: QualityMetrics(
                dimension=QualityDimension.SEO,
                score=75.0,
                level=QualityLevel.GOOD,
                recommendations=["Add target keywords"]
            ),
            QualityDimension.READABILITY: QualityMetrics(
                dimension=QualityDimension.READABILITY,
                score=82.0,
                level=QualityLevel.GOOD,
                recommendations=["Simplify sentences"]
            )
        }
    
    @pytest.mark.asyncio
    async def test_quality_assessment(self, qa_engine, sample_metrics):
        """Test quality assessment engine"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        content = "Test content for quality assessment."
        original_content = "Original test content."
        criteria = QualityCriteria()
        
        assessment = qa_engine.assess_quality(
            content, original_content, sample_metrics, criteria
        )
        
        # Validate assessment results
        assert assessment.overall_score > 0
        assert assessment.quality_level is not None
        assert assessment.assessment_result is not None
        assert len(assessment.dimension_metrics) > 0
        
        print(f"✓ Quality assessment: {assessment.overall_score:.1f} ({assessment.assessment_result.value})")
    
    @pytest.mark.asyncio
    async def test_quality_gate_decisions(self, sample_metrics):
        """Test quality gate decision logic"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        from .quality_assurance import get_quality_gate_manager
        
        gate_manager = get_quality_gate_manager()
        
        content = "Test content for quality gates."
        original_content = "Original test content."
        criteria = QualityCriteria()
        
        # Test quality gate evaluation
        gate_decision, assessment = gate_manager.evaluate_quality_gate(
            content, original_content, sample_metrics, criteria
        )
        
        assert gate_decision in [result.value for result in AssessmentResult]
        assert assessment is not None
        
        print(f"✓ Quality gate decision: {gate_decision}")
    
    @pytest.mark.asyncio
    async def test_escalation_conditions(self, qa_engine, sample_metrics):
        """Test escalation condition detection"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        # Create low-quality metrics that should trigger escalation
        poor_metrics = {
            QualityDimension.GRAMMAR: QualityMetrics(
                dimension=QualityDimension.GRAMMAR,
                score=45.0,  # Below threshold
                level=QualityLevel.POOR
            )
        }
        
        criteria = QualityCriteria(
            critical_fail_threshold=50.0
        )
        
        assessment = qa_engine.assess_quality(
            "Poor quality content", "Original content", poor_metrics, criteria
        )
        
        # Should trigger escalation due to poor quality
        assert assessment.assessment_result == AssessmentResult.CRITICAL_FAIL
        
        print(f"✓ Escalation triggered for poor quality: {assessment.overall_score:.1f}")


class TestHumanEscalation:
    """Test human escalation system"""
    
    @pytest.fixture
    def escalation_manager(self):
        """Escalation manager instance"""
        from .human_escalation import get_escalation_manager
        return get_escalation_manager()
    
    @pytest.fixture
    def escalation_context(self):
        """Sample escalation context"""
        return EscalationContext(
            content_type="marketing_copy",
            content_length=1200,
            domain="marketing",
            quality_score=55.0,
            critical_issues_count=2,
            editing_iterations=3,
            importance_level="high"
        )
    
    @pytest.mark.asyncio
    async def test_escalation_creation(self, escalation_manager, escalation_context):
        """Test escalation request creation"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        # Mock quality assessment for escalation
        quality_assessment = type('MockAssessment', (), {
            'overall_score': 55.0,
            'all_issues': [],
            'assessment_result': AssessmentResult.FAIL
        })()
        
        escalation = await escalation_manager.create_escalation(
            content="Content needing human review",
            original_content="Original content",
            quality_assessment=quality_assessment,
            context=escalation_context
        )
        
        assert escalation.escalation_id is not None
        assert len(escalation.reasons) > 0
        assert escalation.priority is not None
        
        print(f"✓ Escalation created: {escalation.escalation_id} ({escalation.priority.value})")
    
    @pytest.mark.asyncio
    async def test_reviewer_assignment(self, escalation_manager, escalation_context):
        """Test reviewer assignment process"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        # Create escalation
        quality_assessment = type('MockAssessment', (), {
            'overall_score': 55.0,
            'all_issues': [],
            'assessment_result': AssessmentResult.FAIL
        })()
        
        escalation = await escalation_manager.create_escalation(
            "Content for assignment test",
            "Original content",
            quality_assessment,
            escalation_context
        )
        
        # Test assignment
        assigned = await escalation_manager.assign_escalation(escalation.escalation_id)
        
        if assigned:
            status = escalation_manager.get_escalation_status(escalation.escalation_id)
            assert status is not None
            assert status["assigned_reviewer"] is not None
            
            print(f"✓ Escalation assigned: {status['assigned_reviewer']}")
        else:
            print("⚠ No available reviewers for assignment")
    
    @pytest.mark.asyncio
    async def test_escalation_workflow(self, escalation_manager, escalation_context):
        """Test complete escalation workflow"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        # Create and assign escalation
        quality_assessment = type('MockAssessment', (), {
            'overall_score': 55.0,
            'all_issues': [],
            'assessment_result': AssessmentResult.FAIL
        })()
        
        escalation = await escalation_manager.create_escalation(
            "Workflow test content",
            "Original content",
            quality_assessment,
            escalation_context
        )
        
        assigned = await escalation_manager.assign_escalation(escalation.escalation_id)
        
        if assigned:
            # Process feedback
            feedback_processed = await escalation_manager.process_human_feedback(
                escalation.escalation_id,
                "Content reviewed and improved",
                "Revised content with human improvements"
            )
            
            assert feedback_processed is True
            
            # Resolve escalation
            resolved = await escalation_manager.resolve_escalation(escalation.escalation_id)
            assert resolved is True
            
            print("✓ Complete escalation workflow successful")
        else:
            print("⚠ Escalation workflow test skipped - no reviewers available")


class TestStateManagement:
    """Test state management system"""
    
    @pytest.fixture
    def state_manager(self):
        """State manager instance"""
        from .state_management import get_state_manager
        return get_state_manager()
    
    @pytest.mark.asyncio
    async def test_session_creation(self, state_manager):
        """Test editing session creation"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        original_content = "Original content for state management testing."
        
        session = state_manager.create_session(
            original_content,
            editing_requirements={"test": True},
            quality_criteria={"min_score": 80.0}
        )
        
        assert session.session_id is not None
        assert session.original_content == original_content
        assert session.current_content == original_content
        assert len(session.checkpoint_history) == 1  # Initial checkpoint
        
        print(f"✓ Session created: {session.session_id}")
    
    @pytest.mark.asyncio
    async def test_revision_tracking(self, state_manager):
        """Test content revision tracking"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        # Create session
        session = state_manager.create_session("Original content for revision tracking.")
        
        # Update content
        revised_content = "Revised content for revision tracking."
        revision = state_manager.update_session_content(
            session.session_id,
            revised_content,
            RevisionType.GRAMMAR_FIX,
            stage="grammar_check",
            tool_used="grammar_checker",
            quality_before=65.0,
            quality_after=85.0
        )
        
        assert revision is not None
        assert revision.revision_type == RevisionType.GRAMMAR_FIX
        assert revision.quality_improvement == 20.0
        
        # Check session state
        updated_session = state_manager.get_session(session.session_id)
        assert updated_session.current_content == revised_content
        assert len(updated_session.revisions) == 1
        
        print(f"✓ Revision tracked: +{revision.quality_improvement} quality improvement")
    
    @pytest.mark.asyncio
    async def test_workflow_state_tracking(self, state_manager):
        """Test workflow state tracking"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        # Create session
        session = state_manager.create_session("Content for workflow state testing.")
        
        # Update workflow state
        checkpoint = state_manager.update_workflow_state(
            session.session_id,
            WorkflowState.SEO_OPTIMIZATION,
            completed_stages=["grammar_check", "style_review"],
            failed_stages=[]
        )
        
        assert checkpoint is not None
        assert checkpoint.current_state == WorkflowState.SEO_OPTIMIZATION
        assert "grammar_check" in checkpoint.completed_stages
        
        # Check session has multiple checkpoints
        updated_session = state_manager.get_session(session.session_id)
        assert len(updated_session.checkpoint_history) == 2  # Initial + new
        
        print(f"✓ Workflow state updated: {checkpoint.current_state.value}")
    
    @pytest.mark.asyncio
    async def test_session_analytics(self, state_manager):
        """Test session analytics generation"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        # Create session with activity
        session = state_manager.create_session("Analytics test content.")
        
        # Add some revisions and state changes
        state_manager.update_session_content(
            session.session_id,
            "First revision",
            RevisionType.GRAMMAR_FIX,
            quality_before=60.0,
            quality_after=75.0
        )
        
        state_manager.update_workflow_state(
            session.session_id,
            WorkflowState.COMPLETED
        )
        
        # Get analytics
        analytics = state_manager.get_session_analytics(session.session_id)
        
        assert analytics is not None
        assert "session_id" in analytics
        assert "content_metrics" in analytics
        assert "revision_analytics" in analytics
        assert "workflow_analytics" in analytics
        
        print(f"✓ Analytics generated: {analytics['revision_analytics']['total_revisions']} revisions")


class TestPerformanceAndReliability:
    """Test performance and reliability"""
    
    @pytest.mark.asyncio
    async def test_concurrent_editing_sessions(self):
        """Test multiple concurrent editing sessions"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        config = EditorAgentConfig(
            max_editing_iterations=2,
            min_overall_quality=70.0
        )
        
        # Create multiple agents for concurrent testing
        agents = [EditorAgent(config) for _ in range(3)]
        
        test_contents = [
            "First test content with grammer errors.",
            "Second test content needing SEO optimization.",
            "Third test content with readability issues and long sentences."
        ]
        
        # Run concurrent editing
        tasks = [
            agent.edit_content(content)
            for agent, content in zip(agents, test_contents)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Validate all succeeded
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == len(test_contents)
        
        for result in successful_results:
            assert result.final_quality_score > 0
        
        print(f"✓ Concurrent editing: {len(successful_results)} sessions completed")
    
    @pytest.mark.asyncio
    async def test_large_content_handling(self):
        """Test handling of large content"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        # Generate large content (5000+ characters)
        large_content = "This is a large content test. " * 200  # ~6000 characters
        large_content += "It has some grammer errors and needs editing for quality."
        
        config = EditorAgentConfig(
            max_editing_iterations=2,
            timeout_per_stage=90  # Longer timeout for large content
        )
        
        agent = EditorAgent(config)
        
        start_time = datetime.now()
        results = await agent.edit_content(large_content)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        assert results.final_quality_score > 0
        assert processing_time < 120  # Should complete within 2 minutes
        assert len(results.edited_content) > 0
        
        print(f"✓ Large content processed: {len(large_content)} chars in {processing_time:.1f}s")
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery mechanisms"""
        if not MODULES_AVAILABLE:
            pytest.skip("Modules not available")
        
        # Test with potentially problematic content
        problematic_content = "Content with unicode: éññörs and spëcîal characters that might cause issues."
        
        config = EditorAgentConfig(
            max_editing_iterations=1,
            enable_auto_fixes=True
        )
        
        agent = EditorAgent(config)
        
        try:
            results = await agent.edit_content(problematic_content)
            
            # Should handle errors gracefully and return results
            assert results is not None
            assert results.edited_content is not None
            
            print("✓ Error recovery successful")
            
        except Exception as e:
            # Should not raise unhandled exceptions
            pytest.fail(f"Unhandled exception in error recovery test: {e}")


async def run_integration_tests():
    """Run all integration tests"""
    
    print("Running Editor Agent Integration Tests")
    print("=" * 50)
    
    if not MODULES_AVAILABLE:
        print("⚠ Warning: Not all modules available - running limited tests")
    
    # Test categories
    test_classes = [
        TestEditorAgentCore,
        TestToolIntegration, 
        TestQualityAssurance,
        TestHumanEscalation,
        TestStateManagement,
        TestPerformanceAndReliability
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 30)
        
        # Get test methods
        test_methods = [
            method for method in dir(test_class) 
            if method.startswith('test_')
        ]
        
        # Run tests
        for method_name in test_methods:
            total_tests += 1
            
            try:
                # Create test instance with fixtures
                test_instance = test_class()
                
                # Setup fixtures manually (simplified)
                if hasattr(test_instance, 'sample_content'):
                    test_instance.sample_content = lambda: "Sample test content with grammer errors."
                
                # Run test method
                test_method = getattr(test_instance, method_name)
                
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()
                
                passed_tests += 1
                print(f"✓ {method_name}")
                
            except Exception as e:
                print(f"✗ {method_name}: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print(f"⚠ {total_tests - passed_tests} tests failed")
    
    return passed_tests, total_tests


if __name__ == "__main__":
    # Run integration tests
    asyncio.run(run_integration_tests())