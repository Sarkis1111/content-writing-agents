#!/usr/bin/env python3
"""
Simple Editor Agent Integration Test

This script provides direct integration testing for the Editor Agent system
without complex test framework dependencies.
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test that all Editor Agent modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        # Core Editor Agent
        from agents.editor.editor_agent import EditorAgent, EditorAgentConfig, EditingResults
        print("✅ editor_agent module imported")
        
        # Tool Integration
        from agents.editor.tool_integration import ToolOrchestrator, ExecutionMode, ToolType
        print("✅ tool_integration module imported")
        
        # Quality Assurance
        from agents.editor.quality_assurance import QualityAssessmentEngine, QualityCriteria
        print("✅ quality_assurance module imported")
        
        # Human Escalation
        from agents.editor.human_escalation import EscalationManager, EscalationContext
        print("✅ human_escalation module imported")
        
        # State Management
        from agents.editor.state_management import StateManager, EditingSessionState
        print("✅ state_management module imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of core components"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from agents.editor.editor_agent import EditorAgentConfig
        from agents.editor.tool_integration import ToolOrchestrator
        from agents.editor.state_management import get_state_manager
        
        # Test EditorAgentConfig
        config = EditorAgentConfig(
            min_grammar_score=85.0,
            min_seo_score=75.0,
            enable_auto_fixes=True
        )
        assert config.min_grammar_score == 85.0
        print("✅ EditorAgentConfig creation and configuration")
        
        # Test ToolOrchestrator
        orchestrator = ToolOrchestrator()
        available_tools = orchestrator.get_available_tools()
        assert len(available_tools) > 0
        expected_tools = ['grammar', 'seo', 'readability', 'sentiment']
        for tool in expected_tools:
            assert tool in available_tools, f"Missing tool: {tool}"
        print(f"✅ ToolOrchestrator with tools: {available_tools}")
        
        # Test StateManager
        state_manager = get_state_manager()
        assert state_manager is not None
        metrics = state_manager.get_global_metrics()
        assert 'total_sessions' in metrics
        print("✅ StateManager initialization and metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

async def test_editor_agent_workflow():
    """Test the main Editor Agent workflow"""
    print("\n🔄 Testing Editor Agent workflow...")
    
    try:
        from agents.editor.editor_agent import EditorAgent, EditorAgentConfig
        
        # Test content with intentional errors
        test_content = """
        This is a test content for editing validation. It has some grammer mistakes and could be improved.
        The SEO optimization is not great and readability might be better with shorter sentences.
        
        Overall this content needs significant editing to meet quality standards for professional publication.
        """
        
        # Create editor with reasonable settings
        config = EditorAgentConfig(
            min_grammar_score=80.0,
            min_seo_score=70.0,
            min_readability_score=75.0,
            min_overall_quality=75.0,
            max_editing_iterations=2,
            enable_auto_fixes=True,
            require_human_review=False
        )
        
        editor = EditorAgent(config)
        
        # Define editing requirements
        editing_requirements = {
            "target_keywords": ["content", "editing", "quality"],
            "target_audience": "business professionals",
            "content_type": "blog_post",
            "writing_style": "professional"
        }
        
        print("   📝 Starting content editing...")
        start_time = time.time()
        
        # Execute editing workflow
        results = await editor.edit_content(test_content, editing_requirements)
        
        processing_time = time.time() - start_time
        
        # Validate results
        assert results is not None, "No results returned"
        assert hasattr(results, 'final_quality_score'), "Missing final_quality_score"
        assert hasattr(results, 'edited_content'), "Missing edited_content"
        assert hasattr(results, 'processing_time'), "Missing processing_time"
        
        print(f"✅ Content editing completed:")
        print(f"   • Original length: {len(test_content)} chars")
        print(f"   • Final length: {len(results.edited_content)} chars")
        print(f"   • Quality score: {results.final_quality_score:.1f}")
        print(f"   • Processing time: {processing_time:.2f}s")
        print(f"   • Iterations: {results.total_iterations}")
        print(f"   • Stages completed: {len(results.stages_completed)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Editor Agent workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_tool_integration():
    """Test tool integration layer"""
    print("\n🛠️ Testing tool integration...")
    
    try:
        from agents.editor.tool_integration import ToolOrchestrator, ExecutionMode
        
        orchestrator = ToolOrchestrator()
        
        test_content = "This content has grammer errors and needs SEO optimization for better readability."
        
        editing_requirements = {
            "target_keywords": ["content", "optimization"],
            "target_audience": "general audience",
            "content_type": "article"
        }
        
        print("   🔄 Testing sequential execution...")
        results = await orchestrator.edit_content(
            test_content,
            editing_requirements,
            ExecutionMode.SEQUENTIAL
        )
        
        assert results.overall_score >= 0, "Invalid overall score"
        assert len(results.successful_tools) > 0, "No tools executed successfully"
        
        print(f"✅ Tool integration successful:")
        print(f"   • Overall score: {results.overall_score:.1f}")
        print(f"   • Successful tools: {len(results.successful_tools)}")
        print(f"   • Failed tools: {len(results.failed_tools)}")
        print(f"   • Execution time: {results.total_execution_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_state_management():
    """Test state management system"""
    print("\n💾 Testing state management...")
    
    try:
        from agents.editor.state_management import get_state_manager, RevisionType
        
        state_manager = get_state_manager()
        
        # Create editing session
        original_content = "Original content for state management testing."
        
        session = state_manager.create_session(
            original_content,
            editing_requirements={"test": True},
            quality_criteria={"min_score": 80.0}
        )
        
        assert session.session_id is not None, "Session ID not created"
        assert session.original_content == original_content, "Original content not stored"
        
        print(f"   📝 Session created: {session.session_id}")
        
        # Test content update with revision tracking
        revised_content = "Revised content for state management testing."
        revision = state_manager.update_session_content(
            session.session_id,
            revised_content,
            RevisionType.GRAMMAR_FIX,
            stage="grammar_check",
            quality_before=65.0,
            quality_after=85.0
        )
        
        assert revision is not None, "Revision not created"
        assert revision.quality_improvement == 20.0, "Quality improvement not calculated"
        
        print(f"   ✏️  Revision tracked: +{revision.quality_improvement} quality points")
        
        # Test session analytics
        analytics = state_manager.get_session_analytics(session.session_id)
        assert analytics is not None, "Analytics not generated"
        
        print(f"✅ State management successful:")
        print(f"   • Session ID: {session.session_id}")
        print(f"   • Revisions tracked: {len(session.revisions)}")
        print(f"   • Quality improvement: +{revision.quality_improvement}")
        
        return True
        
    except Exception as e:
        print(f"❌ State management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_assurance():
    """Test quality assurance system"""
    print("\n🎯 Testing quality assurance...")
    
    try:
        from agents.editor.quality_assurance import (
            get_quality_assessment_engine, QualityCriteria, 
            QualityDimension, QualityMetrics, QualityLevel
        )
        
        qa_engine = get_quality_assessment_engine()
        
        # Create sample quality metrics
        sample_metrics = {
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
        
        # Test quality assessment
        content = "Test content for quality assessment."
        original_content = "Original test content."
        criteria = QualityCriteria()
        
        assessment = qa_engine.assess_quality(
            content, original_content, sample_metrics, criteria
        )
        
        assert assessment.overall_score > 0, "Overall score not calculated"
        assert assessment.quality_level is not None, "Quality level not determined"
        assert len(assessment.dimension_metrics) > 0, "No dimension metrics"
        
        print(f"✅ Quality assurance successful:")
        print(f"   • Overall score: {assessment.overall_score:.1f}")
        print(f"   • Quality level: {assessment.quality_level.value}")
        print(f"   • Assessment result: {assessment.assessment_result.value}")
        print(f"   • Dimensions assessed: {len(assessment.dimension_metrics)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality assurance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_human_escalation():
    """Test human escalation system"""
    print("\n👥 Testing human escalation...")
    
    try:
        from agents.editor.human_escalation import get_escalation_manager, EscalationContext
        from agents.editor.quality_assurance import AssessmentResult
        
        escalation_manager = get_escalation_manager()
        
        # Create escalation context
        context = EscalationContext(
            content_type="marketing_copy",
            content_length=1200,
            domain="marketing",
            quality_score=55.0,
            critical_issues_count=2,
            editing_iterations=3,
            importance_level="high"
        )
        
        # Mock quality assessment with all required attributes
        quality_assessment = type('MockAssessment', (), {
            'overall_score': 55.0,
            'all_issues': [],
            'assessment_result': AssessmentResult.FAIL,
            'failed_dimensions': ['grammar', 'seo'],
            'criteria_violations': ['Grammar score below threshold', 'SEO score below threshold'],
            'quality_level': 'poor',
            'confidence': 0.85
        })()
        
        # Test escalation creation
        escalation = await escalation_manager.create_escalation(
            content="Content needing human review",
            original_content="Original content",
            quality_assessment=quality_assessment,
            context=context
        )
        
        assert escalation.escalation_id is not None, "Escalation ID not created"
        assert len(escalation.reasons) > 0, "No escalation reasons"
        
        print(f"✅ Human escalation successful:")
        print(f"   • Escalation ID: {escalation.escalation_id}")
        print(f"   • Priority: {escalation.priority.value}")
        print(f"   • Reasons: {[r.value for r in escalation.reasons]}")
        
        # Test reviewer assignment
        assigned = await escalation_manager.assign_escalation(escalation.escalation_id)
        
        if assigned:
            print(f"   • Reviewer assigned: ✅")
        else:
            print(f"   • No reviewer available: ⚠️")
        
        return True
        
    except Exception as e:
        print(f"❌ Human escalation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_comprehensive_tests():
    """Run all integration tests"""
    print("🧪 Editor Agent Integration Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: Module Imports
    test_results.append(("Module Imports", test_imports()))
    
    # Test 2: Basic Functionality
    test_results.append(("Basic Functionality", test_basic_functionality()))
    
    # Test 3: Editor Agent Workflow
    test_results.append(("Editor Agent Workflow", await test_editor_agent_workflow()))
    
    # Test 4: Tool Integration
    test_results.append(("Tool Integration", await test_tool_integration()))
    
    # Test 5: State Management
    test_results.append(("State Management", await test_state_management()))
    
    # Test 6: Quality Assurance
    test_results.append(("Quality Assurance", test_quality_assurance()))
    
    # Test 7: Human Escalation
    test_results.append(("Human Escalation", await test_human_escalation()))
    
    # Results Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Editor Agent is ready for production.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review and fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_tests())
    sys.exit(0 if success else 1)