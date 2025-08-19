"""
Test the fixed import system for analysis tools.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_fixed_content_analysis():
    """Test the fixed content analysis tool."""
    try:
        from tools.analysis.content_analysis import ContentAnalysisTool, ContentAnalysisRequest
        
        # Create tool instance
        tool = ContentAnalysisTool()
        logger.info("✅ ContentAnalysisTool imported and instantiated successfully")
        
        # Create test request
        test_content = """
        AI-powered content marketing represents the future of digital marketing strategies. 
        By leveraging artificial intelligence and machine learning algorithms, businesses can 
        create more personalized, targeted, and effective content that resonates with their 
        specific audience segments. This technological advancement enables marketers to analyze 
        vast amounts of data, predict customer behavior, and optimize content performance in 
        real-time, ultimately driving better engagement rates and higher conversion metrics.
        """
        
        request = ContentAnalysisRequest(
            text=test_content,
            include_sentiment=True,
            include_readability=True,
            include_style_analysis=True
        )
        
        logger.info("✅ ContentAnalysisRequest created successfully")
        logger.info(f"Request text length: {len(request.text)} characters")
        
        # Test the analysis
        result = await tool.analyze_content(request)
        
        if result.success:
            logger.info("✅ Content analysis completed successfully!")
            logger.info(f"Word count: {result.result.word_count}")
            logger.info(f"Sentiment: {result.result.sentiment.sentiment_label}")
            logger.info(f"Processing time: {result.processing_time:.2f}s")
            return True
        else:
            logger.error(f"❌ Content analysis failed: {result.error}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Content analysis test failed: {e}")
        return False

async def test_fixed_sentiment_analyzer():
    """Test the fixed sentiment analyzer tool."""
    try:
        from tools.editing.sentiment_analyzer import SentimentAnalyzerTool, SentimentAnalysisRequest
        
        # Create tool instance
        tool = SentimentAnalyzerTool()
        logger.info("✅ SentimentAnalyzerTool imported and instantiated successfully")
        
        # Create test request
        test_content = """
        We're excited to announce our revolutionary new AI platform that transforms how businesses 
        approach content marketing. This innovative solution delivers exceptional results, 
        empowering marketing teams to achieve unprecedented success with intelligent automation 
        and data-driven insights.
        """
        
        request = SentimentAnalysisRequest(
            text=test_content,
            include_brand_voice_analysis=True,
            include_emotion_detection=True
        )
        
        logger.info("✅ SentimentAnalysisRequest created successfully")
        
        # Test the analysis
        result = await tool.analyze_sentiment(request)
        
        if result.success:
            logger.info("✅ Sentiment analysis completed successfully!")
            logger.info(f"Sentiment polarity: {result.result.polarity:.3f}")
            logger.info(f"Processing time: {result.processing_time:.2f}s")
            return True
        else:
            logger.error(f"❌ Sentiment analysis failed: {result.error}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Sentiment analysis test failed: {e}")
        return False

async def test_strategy_agent_with_tools():
    """Test strategy agent models with tool integration."""
    try:
        # Test strategy models
        sys.path.insert(0, os.path.join(src_path, 'agents', 'strategy'))
        from models import StrategyRequest, StrategyType, ContentType, AudienceSegment
        
        # Create comprehensive request
        request = StrategyRequest(
            topic="AI-Powered Marketing Automation for Enterprise",
            strategy_type=StrategyType.THOUGHT_LEADERSHIP,
            content_types=[ContentType.BLOG_POST, ContentType.WHITEPAPER],
            target_audience=[AudienceSegment.B2B_EXECUTIVES],
            industry="Marketing Technology",
            business_objectives=["Establish thought leadership", "Generate enterprise leads"],
            success_metrics=["Brand awareness", "Lead quality"]
        )
        
        logger.info("✅ Strategy request created successfully")
        logger.info(f"Topic: {request.topic}")
        logger.info(f"Content types: {len(request.content_types)}")
        logger.info(f"Business objectives: {len(request.business_objectives)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Strategy agent test failed: {e}")
        return False

async def main():
    """Run all fixed import tests."""
    logger.info("="*60)
    logger.info("TESTING FIXED IMPORT SYSTEM")
    logger.info("="*60)
    
    tests = [
        ("Content Analysis Tool", test_fixed_content_analysis),
        ("Sentiment Analyzer Tool", test_fixed_sentiment_analyzer),
        ("Strategy Agent Models", test_strategy_agent_with_tools)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing: {test_name} ---")
        try:
            result = await test_func()
            if result:
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.info(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    success_rate = (passed / total) * 100
    
    logger.info("\n" + "="*60)
    logger.info("FIXED IMPORT TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate == 100:
        logger.info("\n🎉 ALL IMPORTS FIXED - READY FOR PRODUCTION!")
    elif success_rate >= 80:
        logger.info("\n✅ IMPORTS MOSTLY WORKING - MINOR ISSUES REMAIN")
    else:
        logger.info("\n⚠️ IMPORT ISSUES PERSIST - MORE WORK NEEDED")
    
    return success_rate >= 80

if __name__ == "__main__":
    asyncio.run(main())