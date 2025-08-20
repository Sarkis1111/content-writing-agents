#!/usr/bin/env python3
"""
Comprehensive Editor Agent Functionality Test

This script tests the complete Editor Agent system end-to-end to ensure
all components work together properly before marking Phase 3.4 as complete.
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

async def test_complete_editing_workflow():
    """Test the complete editing workflow from start to finish"""
    print("🔄 Testing Complete Editor Agent Workflow")
    print("=" * 60)
    
    try:
        from agents.editor.editor_agent import EditorAgent, EditorAgentConfig
        from agents.editor.tool_integration import ExecutionMode
        from agents.editor.quality_assurance import QualityCriteria
        
        # Test content with various issues
        test_content = """
        This is a comprehensive test of our editing system. The content has several grammer mistakes and could definitly be improved for better readability and SEO optimization.
        
        The sentences are sometimes to long and complex which makes them hard to read. We also need to ensure that the content is optimized for search engines with proper keywords and structure.
        
        Additionally, the tone might not be consistant throughout the document and some facts may need verification for accurracy.
        """
        
        # Create comprehensive editing requirements
        editing_requirements = {
            "target_keywords": ["editing system", "content optimization", "quality assurance"],
            "target_audience": "business professionals",
            "content_type": "blog_post",
            "writing_style": "professional",
            "brand_voice": "authoritative but approachable",
            "seo_requirements": {
                "target_keywords": ["content editing", "automated editing", "quality control"],
                "meta_description_length": 160,
                "readability_level": "professional"
            },
            "quality_requirements": {
                "min_grammar_score": 90.0,
                "min_readability_score": 85.0,
                "min_seo_score": 80.0,
                "min_overall_quality": 85.0
            }
        }
        
        # Configure editor for comprehensive testing
        config = EditorAgentConfig(
            min_grammar_score=90.0,
            min_seo_score=80.0,
            min_readability_score=85.0,
            min_overall_quality=85.0,
            max_editing_iterations=3,
            enable_auto_fixes=True,
            require_human_review=False,
            escalation_quality_threshold=75.0,
            timeout_per_stage=60.0
        )
        
        print(f"📝 Original content length: {len(test_content)} characters")
        print(f"⚙️  Editor configuration: min_quality={config.min_overall_quality}")
        print(f"🎯 Target audience: {editing_requirements['target_audience']}")
        print(f"🔍 Target keywords: {editing_requirements['target_keywords']}")
        print()
        
        # Create and run editor
        editor = EditorAgent(config)
        start_time = time.time()
        
        print("🚀 Starting comprehensive editing workflow...")
        results = await editor.edit_content(test_content, editing_requirements)
        
        processing_time = time.time() - start_time
        
        # Validate results thoroughly
        print("\n📊 EDITING RESULTS")
        print("-" * 40)
        print(f"✅ Editing completed successfully")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"🔄 Total iterations: {results.total_iterations}")
        print(f"📈 Final quality score: {results.final_quality_score:.1f}/100")
        print(f"📝 Content length change: {len(test_content)} → {len(results.edited_content)} chars")
        print(f"🎯 Stages completed: {len(results.stages_completed)}")
        
        # Check quality scores by dimension
        if results.quality_scores:
            print(f"\n🎯 QUALITY SCORES BY DIMENSION")
            print("-" * 40)
            for dimension, score in results.quality_scores.items():
                if dimension != "overall_score":
                    print(f"{dimension.capitalize():<15}: {score:.1f}/100")
        
        # Show changes applied
        if hasattr(results, 'changes_applied') and results.changes_applied:
            print(f"\n✏️  CHANGES APPLIED")
            print("-" * 40)
            for i, change in enumerate(results.changes_applied[:5], 1):  # Show first 5 changes
                print(f"{i}. {change}")
            if len(results.changes_applied) > 5:
                print(f"   ... and {len(results.changes_applied) - 5} more changes")
        
        # Human review status
        print(f"\n👥 HUMAN REVIEW STATUS")
        print("-" * 40)
        print(f"Human review required: {'Yes' if results.human_review_required else 'No'}")
        if hasattr(results, 'escalation_reasons') and results.escalation_reasons:
            print(f"Escalation reasons: {', '.join(results.escalation_reasons)}")
        
        # Show edited content sample
        print(f"\n📄 EDITED CONTENT PREVIEW")
        print("-" * 40)
        preview = results.edited_content[:200] + "..." if len(results.edited_content) > 200 else results.edited_content
        print(f"{preview}")
        
        # Success criteria
        success_criteria = [
            ("Processing completed", results is not None),
            ("Quality score acceptable", results.final_quality_score >= 70.0),
            ("Content was edited", results.edited_content != test_content),
            ("Processing time reasonable", processing_time < 120),
            ("Stages completed", len(results.stages_completed) > 0),
            ("Iterations performed", results.total_iterations >= 1)
        ]
        
        print(f"\n✅ SUCCESS CRITERIA")
        print("-" * 40)
        all_passed = True
        for criterion, passed in success_criteria:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{criterion:<25}: {status}")
            if not passed:
                all_passed = False
        
        return all_passed, results
        
    except Exception as e:
        print(f"❌ Complete workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

async def test_tool_integration_functionality():
    """Test tool integration layer functionality"""
    print("\n🛠️ Testing Tool Integration Layer")
    print("=" * 60)
    
    try:
        from agents.editor.tool_integration import ToolOrchestrator, ExecutionMode
        
        orchestrator = ToolOrchestrator()
        
        test_content = "This content has grammer errors and needs SEO optimization for better search rankings."
        
        editing_requirements = {
            "target_keywords": ["SEO optimization", "content quality"],
            "target_audience": "digital marketers",
            "content_type": "marketing_copy"
        }
        
        # Test all execution modes
        modes_to_test = [
            (ExecutionMode.SEQUENTIAL, "Sequential"),
            (ExecutionMode.PARALLEL, "Parallel"), 
            (ExecutionMode.ADAPTIVE, "Adaptive")
        ]
        
        mode_results = {}
        
        for mode, mode_name in modes_to_test:
            print(f"\n🔧 Testing {mode_name} execution mode...")
            
            start_time = time.time()
            result = await orchestrator.edit_content(
                test_content,
                editing_requirements, 
                mode
            )
            execution_time = time.time() - start_time
            
            mode_results[mode_name] = {
                'score': result.overall_score,
                'successful_tools': len(result.successful_tools),
                'failed_tools': len(result.failed_tools),
                'execution_time': execution_time
            }
            
            print(f"   Overall score: {result.overall_score:.1f}")
            print(f"   Successful tools: {len(result.successful_tools)}")
            print(f"   Execution time: {execution_time:.3f}s")
        
        # Compare results
        print(f"\n📊 EXECUTION MODE COMPARISON")
        print("-" * 40)
        for mode_name, metrics in mode_results.items():
            print(f"{mode_name:<12}: Score={metrics['score']:.1f}, Tools={metrics['successful_tools']}, Time={metrics['execution_time']:.3f}s")
        
        # Validate tool integration
        all_modes_successful = all(
            metrics['successful_tools'] > 0 for metrics in mode_results.values()
        )
        
        return all_modes_successful, mode_results
        
    except Exception as e:
        print(f"❌ Tool integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

async def test_quality_assurance_system():
    """Test the quality assurance system"""
    print("\n🎯 Testing Quality Assurance System")
    print("=" * 60)
    
    try:
        from agents.editor.quality_assurance import (
            get_quality_assessment_engine, 
            QualityCriteria,
            QualityDimension,
            QualityMetrics,
            QualityLevel,
            AssessmentResult
        )
        
        qa_engine = get_quality_assessment_engine()
        
        # Create test metrics for different quality scenarios
        test_scenarios = [
            ("High Quality Content", {
                QualityDimension.GRAMMAR: QualityMetrics(
                    dimension=QualityDimension.GRAMMAR,
                    score=92.0,
                    level=QualityLevel.EXCELLENT
                ),
                QualityDimension.SEO: QualityMetrics(
                    dimension=QualityDimension.SEO,
                    score=88.0,
                    level=QualityLevel.GOOD
                ),
                QualityDimension.READABILITY: QualityMetrics(
                    dimension=QualityDimension.READABILITY,
                    score=89.0,
                    level=QualityLevel.GOOD
                )
            }),
            ("Poor Quality Content", {
                QualityDimension.GRAMMAR: QualityMetrics(
                    dimension=QualityDimension.GRAMMAR,
                    score=55.0,
                    level=QualityLevel.POOR
                ),
                QualityDimension.SEO: QualityMetrics(
                    dimension=QualityDimension.SEO,
                    score=45.0,
                    level=QualityLevel.POOR
                ),
                QualityDimension.READABILITY: QualityMetrics(
                    dimension=QualityDimension.READABILITY,
                    score=50.0,
                    level=QualityLevel.POOR
                )
            })
        ]
        
        criteria = QualityCriteria()
        
        scenario_results = {}
        
        for scenario_name, dimension_metrics in test_scenarios:
            print(f"\n🧪 Testing: {scenario_name}")
            
            assessment = qa_engine.assess_quality(
                f"Test content for {scenario_name.lower()}",
                "Original test content",
                dimension_metrics,
                criteria
            )
            
            scenario_results[scenario_name] = assessment
            
            print(f"   Overall score: {assessment.overall_score:.1f}")
            print(f"   Quality level: {assessment.quality_level.value}")
            print(f"   Assessment result: {assessment.assessment_result.value}")
            print(f"   Failed dimensions: {len(assessment.failed_dimensions)}")
        
        # Validate QA system behavior
        high_quality_passed = scenario_results["High Quality Content"].assessment_result in [
            AssessmentResult.PASS, AssessmentResult.CONDITIONAL_PASS
        ]
        poor_quality_failed = scenario_results["Poor Quality Content"].assessment_result in [
            AssessmentResult.FAIL, AssessmentResult.CRITICAL_FAIL
        ]
        
        qa_system_working = high_quality_passed and poor_quality_failed
        
        print(f"\n✅ QA SYSTEM VALIDATION")
        print("-" * 40)
        print(f"High quality content passes: {'✅ YES' if high_quality_passed else '❌ NO'}")
        print(f"Poor quality content fails: {'✅ YES' if poor_quality_failed else '❌ NO'}")
        
        return qa_system_working, scenario_results
        
    except Exception as e:
        print(f"❌ Quality assurance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

async def test_human_escalation_system():
    """Test the human escalation system"""
    print("\n👥 Testing Human Escalation System")
    print("=" * 60)
    
    try:
        from agents.editor.human_escalation import (
            get_escalation_manager,
            EscalationContext,
            EscalationReason,
            EscalationPriority
        )
        from agents.editor.quality_assurance import AssessmentResult
        
        escalation_manager = get_escalation_manager()
        
        # Test escalation scenarios
        escalation_scenarios = [
            ("High Priority Marketing Content", {
                "content_type": "marketing_copy",
                "content_length": 2000,
                "domain": "marketing", 
                "quality_score": 65.0,
                "critical_issues_count": 2,
                "editing_iterations": 3,
                "importance_level": "high"
            }),
            ("Technical Documentation", {
                "content_type": "technical_documentation",
                "content_length": 5000,
                "domain": "technology",
                "quality_score": 70.0,
                "critical_issues_count": 1,
                "editing_iterations": 2,
                "importance_level": "medium"
            })
        ]
        
        escalation_results = {}
        
        for scenario_name, context_data in escalation_scenarios:
            print(f"\n🚨 Testing escalation: {scenario_name}")
            
            context = EscalationContext(**context_data)
            
            # Create mock quality assessment
            quality_assessment = type('MockAssessment', (), {
                'overall_score': context_data['quality_score'],
                'all_issues': [],
                'assessment_result': AssessmentResult.FAIL,
                'failed_dimensions': ['grammar', 'readability'],
                'criteria_violations': ['Grammar score below threshold'],
                'quality_level': 'poor',
                'confidence': 0.75
            })()
            
            # Test escalation creation
            escalation = await escalation_manager.create_escalation(
                content=f"Test content for {scenario_name.lower()}",
                original_content="Original test content",
                quality_assessment=quality_assessment,
                context=context
            )
            
            escalation_results[scenario_name] = escalation
            
            print(f"   Escalation ID: {escalation.escalation_id}")
            print(f"   Priority: {escalation.priority.value}")
            print(f"   Reasons: {[r.value for r in escalation.reasons]}")
            
            # Test reviewer assignment
            assigned = await escalation_manager.assign_escalation(escalation.escalation_id)
            assignment_status = "✅ ASSIGNED" if assigned else "⚠️  NO REVIEWER"
            print(f"   Assignment: {assignment_status}")
        
        # Validate escalation system
        escalations_created = all(
            result.escalation_id is not None 
            for result in escalation_results.values()
        )
        
        return escalations_created, escalation_results
        
    except Exception as e:
        print(f"❌ Human escalation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

async def test_state_management_system():
    """Test the state management system"""
    print("\n💾 Testing State Management System")
    print("=" * 60)
    
    try:
        from agents.editor.state_management import (
            get_state_manager,
            RevisionType,
            WorkflowState
        )
        
        state_manager = get_state_manager()
        
        # Test session management
        print("📝 Testing session creation and management...")
        
        original_content = "Original content for comprehensive state management testing."
        
        # Create session
        session = state_manager.create_session(
            original_content,
            editing_requirements={"test_mode": True, "quality_target": 85.0},
            quality_criteria={"min_score": 80.0, "max_iterations": 3}
        )
        
        print(f"   Session created: {session.session_id}")
        print(f"   Original content length: {len(session.original_content)}")
        
        # Test multiple content revisions
        revisions_data = [
            ("Grammar fixes applied", RevisionType.GRAMMAR_FIX, 65.0, 78.0),
            ("SEO optimization completed", RevisionType.SEO_OPTIMIZATION, 78.0, 84.0), 
            ("Readability improvements", RevisionType.READABILITY_IMPROVEMENT, 84.0, 89.0)
        ]
        
        print("\n✏️  Testing revision tracking...")
        for i, (content_desc, revision_type, quality_before, quality_after) in enumerate(revisions_data, 1):
            revised_content = f"{original_content} - Revision {i}: {content_desc}"
            
            revision = state_manager.update_session_content(
                session.session_id,
                revised_content,
                revision_type,
                stage=f"stage_{i}",
                tool_used=f"tool_{i}",
                quality_before=quality_before,
                quality_after=quality_after
            )
            
            print(f"   Revision {i}: {revision_type.value} (+{revision.quality_improvement:.1f} quality)")
        
        # Test workflow state updates  
        print("\n🔄 Testing workflow state tracking...")
        workflow_states = [
            WorkflowState.GRAMMAR_CHECK,
            WorkflowState.SEO_OPTIMIZATION, 
            WorkflowState.READABILITY_ASSESSMENT,
            WorkflowState.COMPLETED
        ]
        
        for i, state in enumerate(workflow_states, 1):
            checkpoint = state_manager.update_workflow_state(
                session.session_id,
                state,
                completed_stages=[f"stage_{j}" for j in range(1, i+1)],
                failed_stages=[]
            )
            print(f"   Workflow state: {state.value}")
        
        # Test analytics generation
        print("\n📊 Testing analytics generation...")
        analytics = state_manager.get_session_analytics(session.session_id)
        
        final_session = state_manager.get_session(session.session_id)
        
        print(f"   Total revisions: {len(final_session.revisions)}")
        print(f"   Workflow checkpoints: {len(final_session.checkpoint_history)}")
        print(f"   Quality improvement: {analytics['revision_analytics']['total_quality_improvement']:.1f}")
        print(f"   Processing stages: {len(analytics['workflow_analytics']['completed_stages'])}")
        
        # Validate state management
        state_management_working = (
            len(final_session.revisions) == 3 and
            len(final_session.checkpoint_history) >= 4 and  # Initial + 3 workflow states
            analytics['revision_analytics']['total_quality_improvement'] > 0
        )
        
        return state_management_working, {
            'session': final_session,
            'analytics': analytics
        }
        
    except Exception as e:
        print(f"❌ State management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

async def run_comprehensive_functionality_test():
    """Run all comprehensive functionality tests"""
    print("🧪 COMPREHENSIVE EDITOR AGENT FUNCTIONALITY TEST")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Track all test results
    test_results = {}
    overall_start_time = time.time()
    
    # Test 1: Complete Editing Workflow
    workflow_success, workflow_data = await test_complete_editing_workflow()
    test_results["Complete Editing Workflow"] = workflow_success
    
    # Test 2: Tool Integration
    tool_integration_success, tool_data = await test_tool_integration_functionality()
    test_results["Tool Integration"] = tool_integration_success
    
    # Test 3: Quality Assurance
    qa_success, qa_data = await test_quality_assurance_system()
    test_results["Quality Assurance"] = qa_success
    
    # Test 4: Human Escalation
    escalation_success, escalation_data = await test_human_escalation_system()
    test_results["Human Escalation"] = escalation_success
    
    # Test 5: State Management
    state_success, state_data = await test_state_management_system()
    test_results["State Management"] = state_success
    
    total_time = time.time() - overall_start_time
    
    # Final Results Summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(1 for success in test_results.values() if success)
    total_tests = len(test_results)
    
    for test_name, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<30}: {status}")
    
    print("-" * 80)
    print(f"TOTAL TESTS: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
    print(f"TOTAL TIME: {total_time:.2f} seconds")
    
    # System readiness assessment
    system_ready = passed_tests == total_tests
    
    if system_ready:
        print("\n🎉 SYSTEM STATUS: ✅ FULLY FUNCTIONAL AND READY FOR PRODUCTION")
        print("All core components are working correctly and integration is successful.")
    else:
        failed_tests = [name for name, success in test_results.items() if not success]
        print(f"\n⚠️  SYSTEM STATUS: ❌ ISSUES DETECTED")
        print(f"Failed components: {', '.join(failed_tests)}")
        print("System requires fixes before production deployment.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return system_ready, test_results

if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_functionality_test())
    sys.exit(0 if success[0] else 1)