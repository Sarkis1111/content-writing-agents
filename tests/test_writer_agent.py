"""
Test Writer Agent Implementation - Phase 3.2 Validation

This module tests the Writer Agent LangGraph implementation with comprehensive
functionality testing including workflow execution, tool integration, and
quality assessment.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.writer.writer_agent import WriterAgent, WriterAgentConfig
from frameworks.langgraph.state import ContentType
from core.logging import get_framework_logger

logger = get_framework_logger("WriterAgentTest")


class WriterAgentTester:
    """Comprehensive tester for Writer Agent functionality"""
    
    def __init__(self):
        self.writer_agent = None
        self.test_results = {}
        
    async def setup(self):
        """Set up test environment"""
        try:
            logger.info("Setting up Writer Agent test environment...")
            
            # Initialize Writer Agent with test configuration
            config = WriterAgentConfig(
                max_revisions=2,
                quality_threshold=0.7,
                enable_human_review=False,  # Disable for automated testing
                enable_image_generation=False,  # Disable for faster testing
                default_model="gpt-3.5-turbo",
                default_temperature=0.7
            )
            
            self.writer_agent = WriterAgent(config)
            await self.writer_agent.initialize()
            
            logger.info("Writer Agent test setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Test setup failed: {e}")
            return False

    async def test_basic_initialization(self) -> Dict[str, Any]:
        """Test basic Writer Agent initialization"""
        test_name = "basic_initialization"
        logger.info(f"Testing: {test_name}")
        
        try:
            # Check initialization status
            assert self.writer_agent is not None, "Writer Agent not initialized"
            assert self.writer_agent.is_initialized, "Writer Agent initialization flag not set"
            
            # Check tool initialization
            assert self.writer_agent.content_writer is not None, "Content Writer not initialized"
            assert self.writer_agent.headline_generator is not None, "Headline Generator not initialized"
            assert self.writer_agent.sentiment_analyzer is not None, "Sentiment Analyzer not initialized"
            
            # Check framework components
            assert self.writer_agent.langgraph_framework is not None, "LangGraph framework not initialized"
            assert self.writer_agent.workflow_registry is not None, "Workflow registry not initialized"
            
            return {
                "test": test_name,
                "status": "passed",
                "message": "All initialization checks passed",
                "details": {
                    "agent_initialized": True,
                    "tools_loaded": True,
                    "framework_ready": True
                }
            }
            
        except Exception as e:
            return {
                "test": test_name,
                "status": "failed",
                "error": str(e),
                "message": f"Initialization test failed: {e}"
            }

    async def test_content_creation_workflow(self) -> Dict[str, Any]:
        """Test basic content creation workflow"""
        test_name = "content_creation_workflow"
        logger.info(f"Testing: {test_name}")
        
        try:
            # Test parameters
            topic = "The Future of Artificial Intelligence in Content Creation"
            content_type = ContentType.BLOG_POST
            requirements = {
                "target_audience": "Content creators and marketers",
                "tone": "professional",
                "style": "journalistic",
                "target_length": 800,
                "keywords": ["AI", "content creation", "automation", "marketing"]
            }
            
            # Mock research data
            research_data = {
                "sources": [
                    {
                        "title": "AI in Content Marketing Report",
                        "content": "AI is transforming content creation with automated writing, personalization, and optimization tools.",
                        "credibility_score": 0.85
                    }
                ],
                "key_findings": [
                    "AI tools are increasing content creation efficiency",
                    "Personalization is becoming more sophisticated",
                    "Human oversight remains crucial for quality"
                ],
                "trends": [
                    {"keyword": "AI content", "growth": 45.2}
                ]
            }
            
            logger.info(f"Starting content creation for: {topic}")
            start_time = datetime.now()
            
            # Execute content creation
            result = await self.writer_agent.create_content(
                topic=topic,
                content_type=content_type,
                requirements=requirements,
                research_data=research_data
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Validate results
            assert result is not None, "No result returned"
            assert "success" in result, "Success flag missing"
            assert "workflow_id" in result, "Workflow ID missing"
            
            if result["success"]:
                assert "content" in result, "Content missing from successful result"
                content_data = result["content"]
                
                # Validate content structure
                assert "content" in content_data, "Main content missing"
                assert "title" in content_data, "Title missing"
                assert "headlines" in content_data, "Headlines missing"
                assert "quality_score" in content_data, "Quality score missing"
                
                # Validate content quality
                content_text = content_data["content"]
                assert len(content_text) > 100, "Content too short"
                assert len(content_text.split()) > 50, "Content has too few words"
                
                return {
                    "test": test_name,
                    "status": "passed",
                    "message": "Content creation workflow completed successfully",
                    "details": {
                        "workflow_id": result["workflow_id"],
                        "content_length": len(content_text),
                        "word_count": len(content_text.split()),
                        "quality_score": content_data.get("quality_score", 0),
                        "execution_time": execution_time,
                        "revision_count": result.get("revision_count", 0),
                        "headlines_generated": len(content_data.get("headlines", [])),
                        "has_title": bool(content_data.get("title")),
                        "has_meta": bool(content_data.get("meta"))
                    }
                }
            else:
                return {
                    "test": test_name,
                    "status": "failed",
                    "message": f"Content creation failed: {result.get('error', 'Unknown error')}",
                    "details": {
                        "error": result.get("error"),
                        "failed_steps": result.get("failed_steps", []),
                        "execution_time": execution_time
                    }
                }
                
        except Exception as e:
            return {
                "test": test_name,
                "status": "failed",
                "error": str(e),
                "message": f"Content creation test failed: {e}"
            }

    async def test_different_content_types(self) -> Dict[str, Any]:
        """Test content creation with different content types"""
        test_name = "different_content_types"
        logger.info(f"Testing: {test_name}")
        
        content_types = [
            (ContentType.ARTICLE, "The Impact of Remote Work on Team Productivity"),
            (ContentType.SOCIAL_MEDIA, "Quick tips for better productivity"),
            (ContentType.EMAIL, "Weekly productivity newsletter")
        ]
        
        results = []
        
        try:
            for content_type, topic in content_types:
                logger.info(f"Testing content type: {content_type.value}")
                
                requirements = {
                    "target_audience": "Business professionals",
                    "tone": "professional",
                    "target_length": 300 if content_type == ContentType.SOCIAL_MEDIA else 600
                }
                
                result = await self.writer_agent.create_content(
                    topic=topic,
                    content_type=content_type,
                    requirements=requirements
                )
                
                test_result = {
                    "content_type": content_type.value,
                    "topic": topic,
                    "success": result.get("success", False),
                    "error": result.get("error")
                }
                
                if result.get("success") and result.get("content"):
                    content_data = result["content"]
                    test_result.update({
                        "content_length": len(content_data.get("content", "")),
                        "word_count": len(content_data.get("content", "").split()),
                        "quality_score": content_data.get("quality_score", 0)
                    })
                
                results.append(test_result)
            
            # Check if at least some content types worked
            successful_tests = sum(1 for r in results if r["success"])
            
            return {
                "test": test_name,
                "status": "passed" if successful_tests > 0 else "failed",
                "message": f"Content type testing completed: {successful_tests}/{len(content_types)} successful",
                "details": {
                    "total_types_tested": len(content_types),
                    "successful_types": successful_tests,
                    "results": results
                }
            }
            
        except Exception as e:
            return {
                "test": test_name,
                "status": "failed",
                "error": str(e),
                "message": f"Content type testing failed: {e}",
                "details": {"partial_results": results}
            }

    async def test_tool_integration(self) -> Dict[str, Any]:
        """Test individual tool integration"""
        test_name = "tool_integration"
        logger.info(f"Testing: {test_name}")
        
        try:
            integration_results = {}
            
            # Test Content Writer integration
            try:
                content_writer = self.writer_agent.content_writer
                assert content_writer is not None, "Content Writer not available"
                integration_results["content_writer"] = "available"
            except Exception as e:
                integration_results["content_writer"] = f"failed: {e}"
            
            # Test Headline Generator integration
            try:
                headline_generator = self.writer_agent.headline_generator
                assert headline_generator is not None, "Headline Generator not available"
                integration_results["headline_generator"] = "available"
            except Exception as e:
                integration_results["headline_generator"] = f"failed: {e}"
            
            # Test Sentiment Analyzer integration
            try:
                sentiment_analyzer = self.writer_agent.sentiment_analyzer
                assert sentiment_analyzer is not None, "Sentiment Analyzer not available"
                integration_results["sentiment_analyzer"] = "available"
            except Exception as e:
                integration_results["sentiment_analyzer"] = f"failed: {e}"
            
            # Count successful integrations
            successful_tools = sum(1 for status in integration_results.values() if status == "available")
            total_tools = len(integration_results)
            
            return {
                "test": test_name,
                "status": "passed" if successful_tools == total_tools else "partial",
                "message": f"Tool integration check: {successful_tools}/{total_tools} tools available",
                "details": {
                    "tool_availability": integration_results,
                    "successful_integrations": successful_tools,
                    "total_tools": total_tools
                }
            }
            
        except Exception as e:
            return {
                "test": test_name,
                "status": "failed",
                "error": str(e),
                "message": f"Tool integration test failed: {e}"
            }

    async def test_workflow_state_management(self) -> Dict[str, Any]:
        """Test workflow state management"""
        test_name = "workflow_state_management"
        logger.info(f"Testing: {test_name}")
        
        try:
            # Check state manager availability
            state_manager = self.writer_agent.state_manager
            assert state_manager is not None, "State manager not available"
            
            # Test state creation
            test_inputs = {
                "topic": "Test Topic",
                "content_type": ContentType.BLOG_POST,
                "requirements": {"test": True}
            }
            
            initial_state = state_manager.create_initial_state("content_creation", test_inputs)
            
            # Validate initial state structure
            assert "topic" in initial_state, "Topic not in initial state"
            assert "content_type" in initial_state, "Content type not in initial state"
            assert "workflow_status" in initial_state, "Workflow status not in initial state"
            assert "current_step" in initial_state, "Current step not in initial state"
            
            return {
                "test": test_name,
                "status": "passed",
                "message": "State management test passed",
                "details": {
                    "state_manager_available": True,
                    "state_creation_successful": True,
                    "state_keys": list(initial_state.keys()),
                    "initial_step": initial_state.get("current_step"),
                    "initial_status": initial_state.get("workflow_status")
                }
            }
            
        except Exception as e:
            return {
                "test": test_name,
                "status": "failed",
                "error": str(e),
                "message": f"State management test failed: {e}"
            }

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Writer Agent tests"""
        logger.info("Starting comprehensive Writer Agent testing...")
        
        # Setup
        setup_success = await self.setup()
        if not setup_success:
            return {
                "success": False,
                "error": "Test setup failed",
                "results": {}
            }
        
        # Run individual tests
        test_methods = [
            self.test_basic_initialization,
            self.test_tool_integration,
            self.test_workflow_state_management,
            self.test_content_creation_workflow,
            self.test_different_content_types
        ]
        
        test_results = {}
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                result = await test_method()
                test_results[result["test"]] = result
                
                if result["status"] == "passed":
                    passed_tests += 1
                    logger.info(f"✅ {result['test']}: PASSED")
                else:
                    logger.warning(f"❌ {result['test']}: {result['status'].upper()} - {result.get('message', 'No message')}")
                    
            except Exception as e:
                logger.error(f"❌ Test method failed: {e}")
                test_results[f"test_error_{len(test_results)}"] = {
                    "test": "unknown",
                    "status": "failed",
                    "error": str(e)
                }
        
        # Cleanup
        if self.writer_agent:
            await self.writer_agent.shutdown()
        
        # Summary
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        overall_success = passed_tests >= (total_tests * 0.7)  # 70% pass rate for success
        
        summary = {
            "success": overall_success,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": success_rate,
                "timestamp": datetime.now().isoformat()
            },
            "results": test_results
        }
        
        logger.info(f"Writer Agent testing completed: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        return summary


async def main():
    """Main test execution function"""
    print("=" * 80)
    print("Writer Agent - Phase 3.2 Testing")
    print("=" * 80)
    print()
    
    tester = WriterAgentTester()
    
    try:
        results = await tester.run_all_tests()
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        
        summary = results["summary"]
        print(f"Overall Success: {'✅ PASS' if results['success'] else '❌ FAIL'}")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Timestamp: {summary['timestamp']}")
        
        print("\n" + "-" * 80)
        print("INDIVIDUAL TEST RESULTS")
        print("-" * 80)
        
        for test_name, result in results["results"].items():
            status_icon = "✅" if result["status"] == "passed" else "⚠️" if result["status"] == "partial" else "❌"
            print(f"{status_icon} {test_name}: {result['status'].upper()}")
            print(f"   Message: {result.get('message', 'No message')}")
            if result.get('details'):
                print(f"   Details: {result['details']}")
            if result.get('error'):
                print(f"   Error: {result['error']}")
            print()
        
        print("=" * 80)
        
        # Write results to file
        results_file = "writer_agent_test_results.json"
        with open(results_file, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        
        print(f"Detailed results saved to: {results_file}")
        
        return results["success"]
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)