#!/usr/bin/env python3
"""
API Integration Test for Phase 3.1 Research Agent
Tests the complete Research Agent with real API keys and external services.
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.research.research_agent import ResearchAgent, ResearchRequest
from src.core.logging.logger import setup_logging


async def test_research_with_apis():
    """Test Research Agent with real API integration."""
    
    print("🧪 Research Agent API Integration Test")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    # Check available API keys
    print("\n📋 API Configuration Check:")
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"), 
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
        "SERPAPI_KEY": os.getenv("SERPAPI_KEY"),
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY")
    }
    
    available_apis = []
    for name, key in api_keys.items():
        if key:
            print(f"✅ {name}: {'*' * 8}{key[-4:]}")
            available_apis.append(name)
        else:
            print(f"❌ {name}: Not configured")
    
    if not available_apis:
        print("\n⚠️  No API keys found. Please configure at least one API key in your .env file.")
        return False
    
    print(f"\n🚀 Initializing Research Agent...")
    try:
        agent = ResearchAgent()
        print("✅ Research Agent initialized successfully")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False
    
    # Create research request
    research_topic = "Latest AI writing tools and content generation trends 2024"
    print(f"\n📝 Research Topic: {research_topic}")
    
    request = ResearchRequest(
        topic=research_topic,
        research_depth="standard",
        focus_areas=["AI writing tools", "content generation", "GPT models", "automation trends"],
        include_trends=True,
        include_news=True,
        fact_check=True,
        max_sources=8
    )
    
    print(f"   Depth: {request.research_depth}")
    print(f"   Focus Areas: {', '.join(request.focus_areas)}")
    print(f"   Max Sources: {request.max_sources}")
    
    # Execute research
    print(f"\n🔍 Executing Research (this may take 2-3 minutes)...")
    start_time = datetime.now()
    
    try:
        response = await agent.research(request)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\n📊 Research Results:")
        print(f"   Success: {response.success}")
        print(f"   Processing Time: {processing_time:.2f}s")
        print(f"   Agent Processing Time: {response.execution_time:.2f}s")
        
        if response.success:
            print(f"\n📄 Research Summary:")
            summary_lines = response.summary.split('\n')[:3]  # First 3 lines
            for line in summary_lines:
                if line.strip():
                    print(f"   {line.strip()}")
            if len(response.summary.split('\n')) > 3:
                print("   ...")
            
            print(f"\n🔗 Sources Found: {len(response.sources)}")
            for i, source in enumerate(response.sources[:5], 1):  # Top 5 sources
                print(f"   {i}. {source.title[:60]}...")
                print(f"      URL: {source.url}")
                print(f"      Type: {source.source_type} | Relevance: {source.relevance_score:.2f}")
                print()
            
            if len(response.sources) > 5:
                print(f"   ... and {len(response.sources) - 5} more sources")
            
            if response.trends:
                print(f"\n📈 Key Trends ({len(response.trends)}):")
                for i, trend in enumerate(response.trends[:3], 1):
                    print(f"   {i}. {trend.name}")
                    print(f"      {trend.description[:80]}...")
                    print(f"      Confidence: {trend.confidence_score:.2f}")
                    print()
            
            if response.trending_topics:
                print(f"🔥 Trending Topics:")
                for topic in response.trending_topics[:5]:
                    print(f"   • {topic}")
            
            if response.recent_news:
                print(f"\n📰 Recent News ({len(response.recent_news)}):")
                for i, news in enumerate(response.recent_news[:3], 1):
                    title = news.get('title', 'No title')[:60]
                    print(f"   {i}. {title}...")
                    if 'published_date' in news:
                        print(f"      Published: {news['published_date']}")
            
            if response.key_themes:
                print(f"\n🎯 Key Themes:")
                for theme in response.key_themes[:5]:
                    print(f"   • {theme}")
            
            print(f"\n✅ Research Agent API Test: SUCCESS")
            
            # Save detailed results
            results_file = "research_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "success": True,
                    "topic": research_topic,
                    "processing_time": processing_time,
                    "summary": response.summary,
                    "sources_count": len(response.sources),
                    "trends_count": len(response.trends),
                    "news_count": len(response.recent_news),
                    "available_apis": available_apis,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"📁 Detailed results saved to: {results_file}")
            return True
            
        else:
            print(f"\n❌ Research Failed:")
            print(f"   Error: {response.error}")
            if response.sources:
                print(f"   Partial sources found: {len(response.sources)}")
            
            print(f"❌ Research Agent API Test: FAILED")
            return False
    
    except Exception as e:
        print(f"\n💥 Research Execution Failed:")
        print(f"   Error: {e}")
        print(f"   Error Type: {type(e).__name__}")
        
        # Print some debugging info
        import traceback
        print(f"\n🔧 Debug Information:")
        print(traceback.format_exc()[-500:])  # Last 500 chars of traceback
        
        return False


async def test_basic_tools():
    """Test individual research tools."""
    print("\n🔧 Testing Individual Tools:")
    
    try:
        # Test web search tool
        from src.tools.research.web_search import WebSearchTool
        web_tool = WebSearchTool()
        print("✅ WebSearchTool imported and initialized")
        
        # Test if we can create a basic search query
        from src.tools.research.web_search import SearchQuery
        search_query = SearchQuery(
            query="AI writing tools",
            max_results=3
        )
        print("✅ SearchQuery created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool testing failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🧪 Phase 3.1 Research Agent - Complete API Integration Test")
    print("=" * 70)
    
    # Test 1: Individual tools
    tools_success = await test_basic_tools()
    
    if not tools_success:
        print("\n❌ Basic tools test failed. Skipping full API test.")
        return False
    
    # Test 2: Full API integration
    print("\n" + "=" * 70)
    api_success = await test_research_with_apis()
    
    print("\n" + "=" * 70)
    print("📋 Final Test Results:")
    print(f"   Tools Test: {'✅ PASS' if tools_success else '❌ FAIL'}")
    print(f"   API Test: {'✅ PASS' if api_success else '❌ FAIL'}")
    
    overall_success = tools_success and api_success
    print(f"   Overall: {'✅ SUCCESS - Research Agent fully operational!' if overall_success else '❌ PARTIAL SUCCESS - Check configuration'}")
    
    if overall_success:
        print(f"\n🎉 Phase 3.1 Research Agent is ready for production use!")
    else:
        print(f"\n⚠️  Some issues detected. Check API keys and dependencies.")
    
    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)