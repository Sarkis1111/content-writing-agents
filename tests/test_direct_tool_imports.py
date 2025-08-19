"""
Test direct tool imports bypassing the __init__.py files.
"""

import asyncio
import os
import sys

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

# Also add specific tool paths
sys.path.insert(0, os.path.join(src_path, 'tools', 'analysis'))
sys.path.insert(0, os.path.join(src_path, 'tools', 'editing'))

async def test_direct_tool_imports():
    """Test importing tools directly from their modules."""
    
    print("Testing direct tool imports...")
    
    # Test 1: Direct import of content analysis
    try:
        import content_analysis
        from content_analysis import ContentAnalysisTool, ContentAnalysisRequest
        
        print("✅ Content analysis imported directly")
        
        # Test creating tool instance
        tool = ContentAnalysisTool()
        print("✅ ContentAnalysisTool instance created")
        
        # Test creating request
        request = ContentAnalysisRequest(
            text="This is a test of the content analysis tool functionality.",
            include_sentiment=True,
            include_readability=True
        )
        print("✅ ContentAnalysisRequest created")
        
        # Test analysis
        result = await tool.analyze_content(request)
        
        if result.success:
            print("✅ Content analysis completed successfully!")
            print(f"   Word count: {result.result.word_count}")
            print(f"   Sentiment: {result.result.sentiment.sentiment_label}")
            print(f"   Processing time: {result.processing_time:.2f}s")
        else:
            print(f"❌ Content analysis failed: {result.error}")
            
    except Exception as e:
        print(f"❌ Direct content analysis import failed: {e}")
    
    print("\n" + "-"*50 + "\n")
    
    # Test 2: Direct import of sentiment analyzer
    try:
        import sentiment_analyzer
        from sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisRequest
        
        print("✅ Sentiment analyzer imported directly")
        
        # Test creating tool instance
        tool = SentimentAnalyzer()
        print("✅ SentimentAnalyzer instance created")
        
        # Test creating request
        request = SentimentAnalysisRequest(
            text="We are excited to announce our innovative new solution that delivers amazing results!",
            include_brand_voice_analysis=True,
            include_emotion_detection=True
        )
        print("✅ SentimentAnalysisRequest created")
        
        # Test analysis
        result = await tool.analyze_sentiment(request)
        
        if result.success:
            print("✅ Sentiment analysis completed successfully!")
            print(f"   Polarity: {result.result.polarity:.3f}")
            print(f"   Processing time: {result.processing_time:.2f}s")
        else:
            print(f"❌ Sentiment analysis failed: {result.error}")
            
    except Exception as e:
        print(f"❌ Direct sentiment analyzer import failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_direct_tool_imports())