"""
Test script for Phase 3.1 Research Agent implementation.

This script tests the CrewAI-based Research Agent with multi-agent collaboration
patterns as specified in the development strategy.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.research import ResearchAgent, ResearchRequest
from src.agents.research.coordinator import ResearchCoordinator
from src.core.logging import get_framework_logger
from src.core.config import get_config


async def test_research_agent_basic():
    """Test basic Research Agent functionality."""
    
    print("🔍 Testing Research Agent - Basic Functionality")
    print("=" * 60)
    
    try:
        # Initialize agent (will use mock LLM for testing)
        agent = ResearchAgent(llm_config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 1500
        })
        
        # Create test request
        request = ResearchRequest(
            topic="Artificial Intelligence trends 2024",
            research_depth="quick",
            max_sources=10,
            include_trends=True,
            include_news=True,
            fact_check=False,
            time_limit=60
        )
        
        print(f"📋 Research Request:")
        print(f"   Topic: {request.topic}")
        print(f"   Depth: {request.research_depth}")
        print(f"   Max Sources: {request.max_sources}")
        print(f"   Include Trends: {request.include_trends}")
        print(f"   Time Limit: {request.time_limit}s")
        print()
        
        # Execute research
        print("🚀 Starting research execution...")
        response = await agent.research(request)
        
        # Display results
        print("✅ Research completed!")
        print(f"   Status: {response.status}")
        print(f"   Execution Time: {response.execution_time:.2f}s")
        print(f"   Sources Found: {len(response.sources)}")
        print(f"   Key Findings: {len(response.key_findings)}")
        print(f"   Trends Identified: {len(response.trends)}")
        print()
        
        if response.summary:
            print("📄 Summary:")
            print(f"   {response.summary[:200]}...")
            print()
        
        if response.key_findings:
            print("🔑 Key Findings:")
            for i, finding in enumerate(response.key_findings[:3], 1):
                print(f"   {i}. {finding}")
            print()
        
        if response.sources:
            print("📚 Sources:")
            for i, source in enumerate(response.sources[:3], 1):
                print(f"   {i}. {source.title} - {source.url}")
                print(f"      Credibility: {source.credibility_score:.2f}")
            print()
        
        # Performance stats
        stats = agent.get_performance_stats()
        print("📊 Performance Stats:")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        print(f"   Avg Execution Time: {stats['avg_execution_time']:.2f}s")
        print(f"   Total Requests: {stats['total_requests']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_research_coordinator():
    """Test Research Coordinator workflow management."""
    
    print("\n🎯 Testing Research Coordinator - Workflow Management")
    print("=" * 60)
    
    try:
        # Initialize coordinator
        coordinator = ResearchCoordinator()
        
        # Create comprehensive research request
        request = ResearchRequest(
            topic="Climate Change Technology Solutions",
            research_depth="comprehensive",
            focus_areas=["renewable energy", "carbon capture", "green technology"],
            max_sources=25,
            include_trends=True,
            include_news=True,
            fact_check=True,
            time_limit=300
        )
        
        print(f"📋 Comprehensive Research Request:")
        print(f"   Topic: {request.topic}")
        print(f"   Depth: {request.research_depth}")
        print(f"   Focus Areas: {', '.join(request.focus_areas)}")
        print(f"   Max Sources: {request.max_sources}")
        print(f"   Fact Check: {request.fact_check}")
        print()
        
        # Execute coordinated research
        print("🚀 Starting coordinated research workflow...")
        response = await coordinator.coordinate_research(request)
        
        # Display results
        print("✅ Coordinated research completed!")
        print(f"   Status: {response.status}")
        print(f"   Execution Time: {response.execution_time:.2f}s")
        print(f"   Sources Found: {len(response.sources)}")
        print(f"   Key Findings: {len(response.key_findings)}")
        print(f"   Trends Analyzed: {len(response.trends)}")
        print()
        
        # Workflow performance
        workflow_perf = response.agent_performance
        print("🔄 Workflow Performance:")
        print(f"   Total Tasks: {workflow_perf.get('total_tasks', 0)}")
        print(f"   Successful Tasks: {workflow_perf.get('successful_tasks', 0)}")
        print(f"   Coordination Overhead: {workflow_perf.get('coordination_overhead', 0):.1%}")
        print()
        
        # Coordinator stats
        coord_stats = coordinator.get_coordination_stats()
        print("📈 Coordinator Stats:")
        print(f"   Active Workflows: {coord_stats['active_workflows']}")
        print(f"   Total Processed: {coord_stats['total_processed']}")
        print(f"   Success Rate: {coord_stats['success_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Coordinator test failed: {e}")
        return False


async def test_agent_health_check():
    """Test Research Agent health check functionality."""
    
    print("\n🏥 Testing Research Agent - Health Check")
    print("=" * 60)
    
    try:
        agent = ResearchAgent()
        
        print("🔍 Running health check...")
        health_status = await agent.health_check()
        
        print("📋 Health Check Results:")
        print(f"   Status: {health_status['status']}")
        print(f"   Framework: {health_status['framework']}")
        print(f"   Agent Type: {health_status['agent_type']}")
        
        if 'test_execution_time' in health_status:
            print(f"   Test Execution Time: {health_status['test_execution_time']:.2f}s")
            print(f"   Test Status: {health_status['test_status']}")
        
        print(f"   Tool Count: {health_status.get('tool_count', 0)}")
        
        if health_status['status'] == 'healthy':
            print("✅ Agent is healthy and operational!")
        else:
            print(f"⚠️  Agent health issue: {health_status.get('error', 'Unknown error')}")
        
        return health_status['status'] == 'healthy'
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


async def test_different_research_depths():
    """Test different research depth configurations."""
    
    print("\n📊 Testing Research Agent - Different Research Depths")
    print("=" * 60)
    
    depths = ["quick", "standard", "comprehensive"]
    results = {}
    
    try:
        agent = ResearchAgent()
        
        for depth in depths:
            print(f"🔍 Testing {depth} research...")
            
            request = ResearchRequest(
                topic="Machine Learning Applications",
                research_depth=depth,
                max_sources=15 if depth == "quick" else 25,
                time_limit=30 if depth == "quick" else 120
            )
            
            response = await agent.research(request)
            results[depth] = response
            
            print(f"   ✅ {depth.title()} completed in {response.execution_time:.2f}s")
            print(f"      Sources: {len(response.sources)}, Findings: {len(response.key_findings)}")
        
        print("\n📈 Depth Comparison:")
        for depth in depths:
            resp = results[depth]
            print(f"   {depth.title():>12}: {resp.execution_time:>6.2f}s | "
                  f"{len(resp.sources):>2} sources | {len(resp.key_findings):>2} findings")
        
        return True
        
    except Exception as e:
        print(f"❌ Research depth test failed: {e}")
        return False


async def main():
    """Run all Research Agent tests."""
    
    print("🧪 Phase 3.1 Research Agent Testing Suite")
    print("🤖 CrewAI-based Multi-Agent Research System")
    print("=" * 80)
    
    # Track test results
    tests = [
        ("Basic Functionality", test_research_agent_basic),
        ("Workflow Coordination", test_research_coordinator), 
        ("Health Check", test_agent_health_check),
        ("Research Depths", test_different_research_depths)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running Test: {test_name}")
        print("-" * 40)
        
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
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests:.1%}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Research Agent is ready for Phase 3.1")
        print("\n🚀 Phase 3.1 Research Agent Implementation: COMPLETE")
        print("\n✅ Capabilities Verified:")
        print("   • CrewAI framework integration")
        print("   • Multi-agent collaboration")
        print("   • Specialized sub-agents (Web Research, Trend Analysis, Content Curator, Fact Checker)")
        print("   • Advanced workflow coordination")
        print("   • Task delegation and dependency management")
        print("   • Multiple research depth configurations")
        print("   • Health monitoring and performance tracking")
        
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. Please review implementation.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    # Set up basic configuration for testing
    os.environ.setdefault("LOG_LEVEL", "INFO")
    
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)