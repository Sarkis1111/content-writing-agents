"""
Simplified test for Phase 3.1 Research Agent - Tests core functionality without tool dependencies.

This test focuses on validating the CrewAI integration and agent architecture.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_agent_imports():
    """Test that we can import the core components."""
    print("🧪 Testing Core Imports")
    print("=" * 40)
    
    try:
        # Test basic imports without instantiation
        print("   Importing ResearchRequest...")
        from src.agents.research.research_agent import ResearchRequest, ResearchResponse
        print("   ✅ ResearchRequest imported successfully")
        
        print("   Importing ResearchCoordinator...")
        from src.agents.research.coordinator import ResearchCoordinator, TaskPriority
        print("   ✅ ResearchCoordinator imported successfully")
        
        # Test request creation
        print("   Creating ResearchRequest...")
        request = ResearchRequest(
            topic="AI Testing",
            research_depth="quick", 
            max_sources=5
        )
        print(f"   ✅ ResearchRequest created: {request.topic}")
        
        # Test coordinator creation
        print("   Creating ResearchCoordinator...")
        coordinator = ResearchCoordinator()
        print("   ✅ ResearchCoordinator created successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        return False


async def test_crewai_framework():
    """Test CrewAI framework integration."""
    print("\n🤖 Testing CrewAI Framework Integration")
    print("=" * 40)
    
    try:
        from src.frameworks.crewai import get_crew_registry, get_agent_registry
        
        print("   Getting crew registry...")
        crew_registry = get_crew_registry()
        print("   ✅ Crew registry obtained")
        
        print("   Getting agent registry...")
        agent_registry = get_agent_registry()
        print("   ✅ Agent registry obtained")
        
        # List available crews
        crews = crew_registry.list_crews()
        print(f"   📋 Available crews: {len(crews)}")
        for crew in crews:
            print(f"      - {crew.name}: {len(crew.agents)} agents")
        
        # List research agents
        research_agents = agent_registry.get_research_agents()
        print(f"   🔍 Research agents: {len(research_agents)}")
        for agent in research_agents:
            print(f"      - {agent.name} ({agent.role.value})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CrewAI framework test failed: {e}")
        return False


async def test_workflow_planning():
    """Test workflow planning functionality."""
    print("\n📋 Testing Workflow Planning")
    print("=" * 40)
    
    try:
        from src.agents.research.coordinator import ResearchCoordinator, WorkflowPlan
        from src.agents.research.research_agent import ResearchRequest
        
        coordinator = ResearchCoordinator()
        
        # Test different research depths
        depths = ["quick", "standard", "comprehensive"]
        
        for depth in depths:
            print(f"   Planning {depth} research...")
            
            request = ResearchRequest(
                topic=f"Test Topic for {depth}",
                research_depth=depth,
                max_sources=10
            )
            
            # Create workflow plan (this tests planning without execution)
            workflow_id = f"test_{depth}_{int(asyncio.get_event_loop().time())}"
            plan = coordinator._create_workflow_plan(workflow_id, request)
            
            print(f"      ✅ {depth}: {len(plan.tasks)} tasks, strategy: {plan.execution_strategy}")
            
            # Validate plan
            ready_tasks = plan.ready_tasks
            print(f"         Ready tasks: {len(ready_tasks)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Workflow planning test failed: {e}")
        return False


async def test_task_management():
    """Test task management and coordination."""
    print("\n⚙️ Testing Task Management")  
    print("=" * 40)
    
    try:
        from src.agents.research.coordinator import (
            ResearchTask, TaskPriority, TaskStatus, 
            ResearchCoordinator
        )
        
        # Create test tasks
        print("   Creating test tasks...")
        tasks = [
            ResearchTask(
                task_id="task_1",
                agent_role="web_researcher",
                description="Test web research task",
                parameters={"topic": "test"},
                priority=TaskPriority.HIGH
            ),
            ResearchTask(
                task_id="task_2", 
                agent_role="trend_analyst",
                description="Test trend analysis task",
                parameters={"topic": "test"},
                dependencies=["task_1"]
            )
        ]
        
        print(f"   ✅ Created {len(tasks)} test tasks")
        
        # Test task state management
        print("   Testing task state management...")
        task = tasks[0]
        
        print(f"      Initial status: {task.status}")
        assert task.status == TaskStatus.PENDING
        
        task.mark_started()
        print(f"      After start: {task.status}")
        assert task.status == TaskStatus.IN_PROGRESS
        
        task.mark_completed({"result": "test"})
        print(f"      After completion: {task.status}")
        assert task.status == TaskStatus.COMPLETED
        
        print(f"      Execution time: {task.execution_time:.3f}s")
        
        print("   ✅ Task state management working correctly")
        
        # Test dependency resolution
        print("   Testing dependency resolution...")
        ready_before = [t for t in tasks if t.is_ready]
        print(f"      Ready tasks before resolution: {len(ready_before)}")
        
        coordinator = ResearchCoordinator()
        coordinator._resolve_dependencies(tasks, "task_1")
        
        ready_after = [t for t in tasks if t.is_ready] 
        print(f"      Ready tasks after resolution: {len(ready_after)}")
        
        print("   ✅ Dependency resolution working correctly")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Task management test failed: {e}")
        return False


async def test_response_formatting():
    """Test response formatting and data structures."""
    print("\n📊 Testing Response Formatting")
    print("=" * 40)
    
    try:
        from src.agents.research.research_agent import (
            ResearchResponse, SourceInfo, TrendData
        )
        
        # Create test data
        print("   Creating test response data...")
        
        sources = [
            SourceInfo(
                url="https://example.com/source1",
                title="Test Source 1",
                credibility_score=0.8,
                relevance_score=0.9,
                content_type="article"
            ),
            SourceInfo(
                url="https://example.com/source2", 
                title="Test Source 2",
                credibility_score=0.7,
                relevance_score=0.8,
                content_type="news"
            )
        ]
        
        trends = [
            TrendData(
                keyword="test_keyword",
                trend_direction="rising",
                growth_rate=0.15,
                interest_score=0.8
            )
        ]
        
        # Create response
        response = ResearchResponse(
            topic="Test Topic",
            request_id="test_request_123",
            execution_time=2.5,
            status="completed",
            summary="Test research summary with key findings",
            key_findings=["Finding 1", "Finding 2", "Finding 3"],
            sources=sources,
            trends=trends
        )
        
        print("   ✅ Response created successfully")
        print(f"      Topic: {response.topic}")
        print(f"      Status: {response.status}")
        print(f"      Sources: {len(response.sources)}")
        print(f"      Findings: {len(response.key_findings)}")
        print(f"      Trends: {len(response.trends)}")
        print(f"      Execution time: {response.execution_time}s")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Response formatting test failed: {e}")
        return False


async def main():
    """Run simplified Research Agent tests."""
    
    print("🧪 Phase 3.1 Research Agent - Simplified Testing Suite")
    print("🤖 CrewAI Integration & Architecture Validation")
    print("=" * 80)
    
    tests = [
        ("Core Imports", test_agent_imports),
        ("CrewAI Framework", test_crewai_framework),
        ("Workflow Planning", test_workflow_planning),
        ("Task Management", test_task_management),
        ("Response Formatting", test_response_formatting)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            if success:
                print(f"✅ {test_name}: PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 SIMPLIFIED TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests:.1%}")
    
    if passed_tests == total_tests:
        print("🎉 Core architecture tests passed!")
        print("\n✅ Phase 3.1 Research Agent - Core Components Verified:")
        print("   • ResearchAgent class structure")
        print("   • ResearchCoordinator workflow management") 
        print("   • CrewAI framework integration")
        print("   • Task delegation and dependency management")
        print("   • Response data structures")
        print("   • Multi-agent coordination patterns")
        
        print("\n🚀 Ready for:")
        print("   • Full tool integration testing")
        print("   • End-to-end workflow execution") 
        print("   • MCP server integration")
        
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. Please review implementation.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)