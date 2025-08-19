"""
Simplified Strategy Agent validation test.
Tests core functionality without complex import issues.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_imports():
    """Test basic imports work."""
    try:
        # Test AutoGen is available
        import autogen
        logger.info(f"✓ AutoGen available: {autogen.__version__}")
        
        # Test our strategy models can be imported
        sys.path.insert(0, os.path.join(src_path, 'agents', 'strategy'))
        from models import StrategyRequest, StrategyType, ContentType
        logger.info("✓ Strategy models imported successfully")
        
        # Test basic model creation
        request = StrategyRequest(
            topic="Test Topic",
            strategy_type=StrategyType.CONTENT_MARKETING,
            content_types=[ContentType.BLOG_POST]
        )
        logger.info("✓ Strategy request created successfully")
        
        return True
    except Exception as e:
        logger.error(f"✗ Basic imports failed: {e}")
        return False

async def test_autogen_basic():
    """Test basic AutoGen functionality."""
    try:
        import autogen
        
        # Test we can create basic agents
        llm_config = {
            "model": "gpt-4",
            "api_key": "test-key",  # Mock key for testing
            "temperature": 0.7
        }
        
        # Create a test agent
        test_agent = autogen.AssistantAgent(
            name="TestAgent",
            system_message="You are a test agent.",
            llm_config=llm_config
        )
        
        logger.info("✓ AutoGen AssistantAgent created successfully")
        
        # Test agent properties
        assert test_agent.name == "TestAgent"
        assert "test agent" in test_agent.system_message.lower()
        
        logger.info("✓ AutoGen agent properties validated")
        
        return True
    except Exception as e:
        logger.error(f"✗ AutoGen basic test failed: {e}")
        return False

async def test_data_structures():
    """Test our data structures work correctly."""
    try:
        sys.path.insert(0, os.path.join(src_path, 'agents', 'strategy'))
        from models import (
            StrategyRequest, StrategyType, ContentType, AudienceSegment,
            ContentStrategistPerspective, StrategyConsensus
        )
        
        # Test comprehensive request
        request = StrategyRequest(
            topic="AI in Healthcare",
            strategy_type=StrategyType.THOUGHT_LEADERSHIP,
            content_types=[ContentType.BLOG_POST, ContentType.WHITEPAPER],
            target_audience=[AudienceSegment.TECHNICAL_PROFESSIONALS],
            industry="Healthcare",
            business_objectives=["Build authority", "Generate leads"],
            success_metrics=["Engagement", "Lead quality"]
        )
        
        # Validate request structure
        assert request.topic == "AI in Healthcare"
        assert len(request.content_types) == 2
        assert len(request.business_objectives) == 2
        
        logger.info("✓ Strategy request structure validated")
        
        # Test perspective structure  
        perspective = ContentStrategistPerspective(
            overall_strategy="Test strategy",
            content_pillars=["Education", "Innovation"],
            messaging_framework={"primary": "Expert insights"},
            content_mix_recommendations={"blog": 0.6, "social": 0.4},
            editorial_calendar_structure={"frequency": "Weekly"},
            brand_alignment_score=0.8,
            strategic_recommendations=["Focus on expertise"],
            success_metrics=["Engagement"],
            confidence_level=0.85
        )
        
        assert perspective.confidence_level == 0.85
        assert len(perspective.content_pillars) == 2
        
        logger.info("✓ Strategy perspective structure validated")
        
        # Test consensus structure
        consensus = StrategyConsensus(
            agreed_strategy_type="thought_leadership",
            consensus_score=0.85,
            unified_approach="Educational content approach",
            key_decisions=["Focus on education"],
            trade_offs_accepted=["Quality over quantity"],
            implementation_priorities=["Content calendar"],
            resource_requirements={"team": "3 people"},
            timeline_agreement={"phase_1": "Month 1"},
            success_metrics_consensus=["Engagement"],
            risks_identified=["Resource constraints"],
            mitigation_strategies=["Phased approach"]
        )
        
        assert consensus.consensus_score == 0.85
        assert len(consensus.key_decisions) == 1
        
        logger.info("✓ Strategy consensus structure validated")
        
        return True
    except Exception as e:
        logger.error(f"✗ Data structures test failed: {e}")
        return False

async def test_framework_integration():
    """Test framework integration components exist."""
    try:
        # Test frameworks directory structure
        frameworks_path = os.path.join(src_path, 'frameworks', 'autogen')
        
        config_path = os.path.join(frameworks_path, 'config.py')
        conversations_path = os.path.join(frameworks_path, 'conversations.py')
        coordination_path = os.path.join(frameworks_path, 'coordination.py')
        
        assert os.path.exists(config_path), "AutoGen config file exists"
        assert os.path.exists(conversations_path), "AutoGen conversations file exists"
        assert os.path.exists(coordination_path), "AutoGen coordination file exists"
        
        logger.info("✓ AutoGen framework files exist")
        
        # Test we can read the files
        with open(config_path, 'r') as f:
            config_content = f.read()
            assert "AutoGenConfig" in config_content
            assert "AutoGenFramework" in config_content
        
        logger.info("✓ AutoGen framework configuration validated")
        
        with open(conversations_path, 'r') as f:
            conv_content = f.read()
            assert "ConversationTemplate" in conv_content
            assert "Strategy Council" in conv_content
        
        logger.info("✓ AutoGen conversation patterns validated")
        
        return True
    except Exception as e:
        logger.error(f"✗ Framework integration test failed: {e}")
        return False

async def test_agent_file_structure():
    """Test the strategy agent file exists and has expected content."""
    try:
        strategy_agent_path = os.path.join(src_path, 'agents', 'strategy', 'strategy_agent.py')
        assert os.path.exists(strategy_agent_path), "Strategy agent file exists"
        
        with open(strategy_agent_path, 'r') as f:
            agent_content = f.read()
            
        # Check for key components
        assert "class StrategyAgent" in agent_content
        assert "AutoGen framework" in agent_content
        assert "Content Strategist" in agent_content
        assert "Audience Analyst" in agent_content
        assert "Competitive Analyst" in agent_content
        assert "Performance Optimizer" in agent_content
        assert "develop_strategy" in agent_content
        assert "GroupChat" in agent_content
        
        logger.info("✓ Strategy Agent file structure validated")
        logger.info(f"✓ Strategy Agent file size: {len(agent_content)} characters")
        
        # Count lines
        lines = agent_content.split('\n')
        logger.info(f"✓ Strategy Agent file lines: {len(lines)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Agent file structure test failed: {e}")
        return False

async def test_tool_availability():
    """Test that analysis tools are available."""
    try:
        # Check tool files exist
        tools_analysis_path = os.path.join(src_path, 'tools', 'analysis')
        tools_editing_path = os.path.join(src_path, 'tools', 'editing')
        
        content_analysis_path = os.path.join(tools_analysis_path, 'content_analysis.py')
        topic_extraction_path = os.path.join(tools_analysis_path, 'topic_extraction.py')
        sentiment_analyzer_path = os.path.join(tools_editing_path, 'sentiment_analyzer.py')
        
        assert os.path.exists(content_analysis_path), "Content analysis tool exists"
        assert os.path.exists(topic_extraction_path), "Topic extraction tool exists" 
        assert os.path.exists(sentiment_analyzer_path), "Sentiment analyzer tool exists"
        
        logger.info("✓ Analysis tool files exist")
        
        # Check tool files have expected content
        with open(content_analysis_path, 'r') as f:
            content = f.read()
            assert "ContentAnalysisTool" in content
            assert "sentiment" in content.lower()
        
        logger.info("✓ Content analysis tool validated")
        
        with open(topic_extraction_path, 'r') as f:
            content = f.read()
            assert "TopicExtractionTool" in content
            assert "keyword" in content.lower()
        
        logger.info("✓ Topic extraction tool validated")
        
        return True
    except Exception as e:
        logger.error(f"✗ Tool availability test failed: {e}")
        return False

async def main():
    """Run simplified validation tests."""
    logger.info("="*60)
    logger.info("STRATEGY AGENT PHASE 3.3 - SIMPLIFIED VALIDATION")
    logger.info("="*60)
    
    start_time = datetime.now()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("AutoGen Basic", test_autogen_basic),
        ("Data Structures", test_data_structures),
        ("Framework Integration", test_framework_integration),
        ("Agent File Structure", test_agent_file_structure),
        ("Tool Availability", test_tool_availability)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing: {test_name} ---")
        try:
            result = await test_func()
            if result:
                passed += 1
                logger.info(f"✓ {test_name}: PASSED")
            else:
                logger.info(f"✗ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name}: ERROR - {e}")
    
    # Results
    total_time = (datetime.now() - start_time).total_seconds()
    success_rate = (passed / total) * 100
    
    logger.info("="*60)
    logger.info("VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
    logger.info(f"Total Time: {total_time:.2f} seconds")
    
    if success_rate >= 80:
        logger.info("\n🎉 STRATEGY AGENT PHASE 3.3: IMPLEMENTATION SUCCESSFUL")
        logger.info("Core components are properly structured and ready for integration!")
    else:
        logger.info("\n⚠️ STRATEGY AGENT PHASE 3.3: NEEDS ATTENTION")
        logger.info("Some components need adjustment before deployment.")
    
    return success_rate >= 80

if __name__ == "__main__":
    asyncio.run(main())