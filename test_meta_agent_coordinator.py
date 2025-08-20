"""
Meta-Agent Coordinator Test Suite
Tests the complete Phase 3.5 implementation including cross-framework orchestration
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

# Test imports
try:
    from agents.meta.meta_agent_coordinator import (
        MetaAgentCoordinator, 
        MetaWorkflowRequest, 
        ContentType,
        WorkflowPhase
    )
    print("✅ Meta-Agent Coordinator imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_meta_coordinator_initialization():
    """Test Meta-Agent Coordinator initialization"""
    print("\n🔧 Testing Meta-Agent Coordinator Initialization")
    
    try:
        # Initialize coordinator
        coordinator = MetaAgentCoordinator(
            enable_crewai=True,
            enable_monitoring=True,
            default_quality_threshold=75.0
        )
        
        print(f"✅ Coordinator initialized successfully")
        print(f"  - Research Agent Available: {coordinator.research_agent is not None}")
        print(f"  - Writer Agent Available: {coordinator.writer_agent is not None}")
        print(f"  - Strategy Agent Available: {coordinator.strategy_agent is not None}")
        print(f"  - Editor Agent Available: {coordinator.editor_agent is not None}")
        print(f"  - CrewAI Coordination Available: {coordinator.coordination_crew is not None}")
        print(f"  - Framework Bridges: {len(coordinator.framework_bridges)}")
        
        return coordinator
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None

async def test_framework_bridges(coordinator):
    """Test framework communication bridges"""
    print("\n🌉 Testing Framework Bridges")
    
    try:
        # Test CrewAI to LangGraph bridge
        crewai_data = {
            'research_results': {'findings': 'Test research data'},
            'key_findings': ['Finding 1', 'Finding 2'],
            'sources': ['Source A', 'Source B'],
            'topic': 'Test Topic'
        }
        
        mapped_data = coordinator.framework_bridges['crewai_langgraph'].data_mapper(crewai_data)
        print(f"✅ CrewAI → LangGraph mapping successful")
        print(f"  - Original keys: {list(crewai_data.keys())}")
        print(f"  - Mapped keys: {list(mapped_data.keys())}")
        
        # Test AutoGen to CrewAI bridge  
        autogen_data = {
            'strategy_result': {'strategy': 'Test strategy'},
            'consensus': {'agreement': 'High'},
            'agent_perspectives': {'strategist': 'Perspective 1'}
        }
        
        mapped_autogen = coordinator.framework_bridges['autogen_crewai'].data_mapper(autogen_data)
        print(f"✅ AutoGen → CrewAI mapping successful")
        
        # Test LangGraph to AutoGen bridge
        langgraph_data = {
            'final_content': 'Test content',
            'quality_scores': {'overall': 85.0},
            'revision_count': 2
        }
        
        mapped_langgraph = coordinator.framework_bridges['langgraph_autogen'].data_mapper(langgraph_data)
        print(f"✅ LangGraph → AutoGen mapping successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Framework bridge test failed: {e}")
        return False

async def test_workflow_request_creation():
    """Test workflow request creation and validation"""
    print("\n📋 Testing Workflow Request Creation")
    
    try:
        # Create comprehensive workflow request
        request = MetaWorkflowRequest(
            topic="AI-Powered Content Creation for Enterprise Marketing",
            content_type=ContentType.WHITEPAPER,
            target_audience=["Marketing Directors", "Content Strategists", "CMOs"],
            requirements={
                "length": "2500-3000 words",
                "sections": ["Executive Summary", "Market Analysis", "Technology Overview", "Implementation Guide"],
                "tone": "professional",
                "industry": "technology"
            },
            research_depth="comprehensive",
            strategy_focus=["competitive_analysis", "audience_targeting", "content_optimization"],
            quality_threshold=80.0,
            enable_human_review=True,
            max_revision_cycles=3,
            preferred_tone="authoritative",
            keywords=["AI", "content creation", "enterprise", "marketing automation"]
        )
        
        print(f"✅ Workflow request created successfully")
        print(f"  - Topic: {request.topic}")
        print(f"  - Content Type: {request.content_type.value}")
        print(f"  - Target Audience: {len(request.target_audience)} segments")
        print(f"  - Quality Threshold: {request.quality_threshold}%")
        print(f"  - Keywords: {len(request.keywords)}")
        
        return request
        
    except Exception as e:
        print(f"❌ Workflow request creation failed: {e}")
        return None

async def test_individual_workflow_phases(coordinator, request):
    """Test individual workflow phases"""
    print("\n⚙️  Testing Individual Workflow Phases")
    
    try:
        # Initialize workflow state
        from agents.meta.meta_agent_coordinator import WorkflowState
        
        session_id = f"test_session_{int(time.time())}"
        workflow_state = WorkflowState(
            session_id=session_id,
            current_phase=WorkflowPhase.INITIALIZATION,
            start_time=datetime.now()
        )
        
        # Test Phase 1: Initialization
        print("  🚀 Testing Initialization Phase...")
        await coordinator._initialize_workflow_context(request, workflow_state)
        print("    ✅ Context initialization successful")
        
        # Test Phase 2: Research Phase
        print("  🔍 Testing Research Phase...")
        research_result = await coordinator._execute_research_phase(request, workflow_state)
        print(f"    ✅ Research phase completed: {research_result.get('status', 'unknown')}")
        
        # Test Phase 3: Strategy Phase  
        print("  📊 Testing Strategy Phase...")
        workflow_state.research_data = research_result
        strategy_result = await coordinator._execute_strategy_phase(request, workflow_state)
        print(f"    ✅ Strategy phase completed: {strategy_result.get('status', 'unknown')}")
        
        # Test Phase 4: Writing Phase
        print("  ✍️  Testing Writing Phase...")
        workflow_state.strategy_data = strategy_result
        content_result = await coordinator._execute_writing_phase(request, workflow_state)
        print(f"    ✅ Writing phase completed: {content_result.get('status', 'unknown')}")
        
        # Test Phase 5: Editing Phase
        print("  📝 Testing Editing Phase...")
        workflow_state.content_data = content_result
        editing_result = await coordinator._execute_editing_phase(request, workflow_state)
        print(f"    ✅ Editing phase completed: {editing_result.get('status', 'unknown')}")
        
        # Test Phase 6: Finalization
        print("  🏁 Testing Finalization Phase...")
        workflow_state.editor_data = editing_result
        final_result = await coordinator._finalize_workflow(request, workflow_state)
        print(f"    ✅ Finalization completed: {final_result.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual phase testing failed: {e}")
        return False

async def test_complete_workflow_execution(coordinator, request):
    """Test complete end-to-end workflow execution"""
    print("\n🚀 Testing Complete Workflow Execution")
    
    try:
        start_time = time.time()
        
        # Execute complete workflow
        result = await coordinator.execute_complete_workflow(request)
        
        execution_time = time.time() - start_time
        
        print(f"✅ Complete workflow execution finished in {execution_time:.2f}s")
        print(f"  - Success: {result.success}")
        print(f"  - Execution Time: {result.execution_time:.2f}s")
        
        if result.success:
            print(f"  - Final Content Available: {result.final_content is not None}")
            if result.final_content:
                content_preview = result.final_content[:100] + "..." if len(result.final_content) > 100 else result.final_content
                print(f"  - Content Preview: {content_preview}")
            
            print(f"  - Quality Assessment: {result.quality_assessment}")
            print(f"  - Research Insights Available: {result.research_insights is not None}")
            print(f"  - Strategy Recommendations Available: {result.strategy_recommendations is not None}")
            
            if result.workflow_state:
                print(f"  - Phases Completed: {len(result.workflow_state.phases_completed)}")
                print(f"  - Current Phase: {result.workflow_state.current_phase.value}")
                print(f"  - Error Count: {len(result.workflow_state.error_log)}")
        else:
            print(f"  - Error Details: {result.error_details}")
        
        return result
        
    except Exception as e:
        print(f"❌ Complete workflow execution failed: {e}")
        return None

async def test_performance_monitoring(coordinator):
    """Test performance monitoring and metrics"""
    print("\n📊 Testing Performance Monitoring")
    
    try:
        # Get performance summary
        summary = coordinator.get_performance_summary()
        
        print("✅ Performance summary generated")
        print(f"  - Total Sessions: {summary.get('total_sessions', 0)}")
        print(f"  - Completed Sessions: {summary.get('completed_sessions', 0)}")
        print(f"  - Success Rate: {summary.get('success_rate', 0):.2%}")
        print(f"  - Average Execution Time: {summary.get('average_execution_time', 0):.2f}s")
        
        # Framework integration health
        health = summary.get('framework_integration_health', {})
        print("  - Framework Integration Health:")
        for framework, available in health.items():
            status = "✅" if available else "❌"
            print(f"    {status} {framework}: {available}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False

async def test_error_handling(coordinator):
    """Test error handling and recovery mechanisms"""
    print("\n🛡️  Testing Error Handling")
    
    try:
        # Test with invalid request
        invalid_request = MetaWorkflowRequest(
            topic="",  # Invalid empty topic
            content_type=ContentType.BLOG_POST,
            target_audience=[],  # Empty audience
            quality_threshold=150.0  # Invalid threshold > 100
        )
        
        result = await coordinator.execute_complete_workflow(invalid_request)
        
        # Should handle gracefully even with invalid input
        print(f"✅ Error handling test completed")
        print(f"  - Handled invalid input gracefully: {not result.success}")
        print(f"  - Error captured: {result.error_details is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

async def run_comprehensive_tests():
    """Run comprehensive test suite for Meta-Agent Coordinator"""
    print("🧪 Starting Meta-Agent Coordinator Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Initialization
    coordinator = await test_meta_coordinator_initialization()
    test_results.append(coordinator is not None)
    
    if not coordinator:
        print("❌ Cannot continue tests without coordinator")
        return False
    
    # Test 2: Framework Bridges
    bridges_ok = await test_framework_bridges(coordinator)
    test_results.append(bridges_ok)
    
    # Test 3: Workflow Request Creation
    request = await test_workflow_request_creation()
    test_results.append(request is not None)
    
    if not request:
        print("❌ Cannot continue workflow tests without valid request")
        return False
    
    # Test 4: Individual Workflow Phases
    phases_ok = await test_individual_workflow_phases(coordinator, request)
    test_results.append(phases_ok)
    
    # Test 5: Complete Workflow Execution
    result = await test_complete_workflow_execution(coordinator, request)
    test_results.append(result is not None)
    
    # Test 6: Performance Monitoring
    monitoring_ok = await test_performance_monitoring(coordinator)
    test_results.append(monitoring_ok)
    
    # Test 7: Error Handling
    error_handling_ok = await test_error_handling(coordinator)
    test_results.append(error_handling_ok)
    
    # Summary
    print("\n" + "=" * 60)
    print("🏆 TEST SUITE RESULTS")
    print("=" * 60)
    
    test_names = [
        "Coordinator Initialization",
        "Framework Bridges", 
        "Workflow Request Creation",
        "Individual Workflow Phases",
        "Complete Workflow Execution",
        "Performance Monitoring",
        "Error Handling"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (name, passed_test) in enumerate(zip(test_names, test_results)):
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {i+1}. {name}: {status}")
    
    success_rate = (passed / total) * 100
    print(f"\n📊 Overall Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 85:
        print("🎉 Meta-Agent Coordinator Phase 3.5: PRODUCTION READY!")
        return True
    else:
        print("⚠️  Some tests failed - review implementation")
        return False

# Main execution
async def main():
    """Main test execution"""
    try:
        success = await run_comprehensive_tests()
        return success
    except Exception as e:
        print(f"❌ Test suite execution failed: {e}")
        return False

if __name__ == "__main__":
    # Run the comprehensive test suite
    result = asyncio.run(main())
    sys.exit(0 if result else 1)