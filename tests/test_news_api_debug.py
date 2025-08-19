#!/usr/bin/env python3
"""
Debug script for News API to understand why it's returning 0 articles.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_news_api_detailed():
    """Test News API with detailed debugging."""
    print("🔍 NEWS API DIAGNOSTIC TEST")
    print("="*40)
    
    try:
        from newsapi import NewsApiClient
        
        api_key = os.getenv('NEWS_API_KEY')
        print(f"API Key configured: {'Yes' if api_key else 'No'}")
        
        if not api_key:
            print("❌ No NEWS_API_KEY found in environment")
            return
        
        newsapi = NewsApiClient(api_key=api_key)
        
        # Test 1: Top headlines without query
        print("\n📰 Test 1: Top headlines (no query)")
        try:
            headlines = newsapi.get_top_headlines(
                language='en',
                country='us',
                page_size=5
            )
            
            print(f"   Status: {headlines.get('status', 'unknown')}")
            print(f"   Total results: {headlines.get('totalResults', 0)}")
            print(f"   Articles returned: {len(headlines.get('articles', []))}")
            
            articles = headlines.get('articles', [])
            for i, article in enumerate(articles[:3], 1):
                print(f"   - {i}: {article.get('title', 'No title')[:60]}...")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Test 2: Top headlines with AI query
        print("\n🤖 Test 2: AI-related headlines")
        try:
            ai_headlines = newsapi.get_top_headlines(
                q='artificial intelligence',
                language='en',
                page_size=5
            )
            
            print(f"   Status: {ai_headlines.get('status', 'unknown')}")
            print(f"   Total results: {ai_headlines.get('totalResults', 0)}")
            print(f"   Articles returned: {len(ai_headlines.get('articles', []))}")
            
            articles = ai_headlines.get('articles', [])
            for i, article in enumerate(articles[:3], 1):
                print(f"   - {i}: {article.get('title', 'No title')[:60]}...")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Test 3: Everything endpoint (broader search)
        print("\n🌐 Test 3: Everything search")
        try:
            everything = newsapi.get_everything(
                q='AI OR "artificial intelligence"',
                language='en',
                sort_by='publishedAt',
                page_size=5
            )
            
            print(f"   Status: {everything.get('status', 'unknown')}")
            print(f"   Total results: {everything.get('totalResults', 0)}")
            print(f"   Articles returned: {len(everything.get('articles', []))}")
            
            articles = everything.get('articles', [])
            for i, article in enumerate(articles[:3], 1):
                print(f"   - {i}: {article.get('title', 'No title')[:60]}...")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Test 4: General tech news
        print("\n💻 Test 4: General tech headlines")
        try:
            tech_headlines = newsapi.get_top_headlines(
                category='technology',
                language='en',
                country='us',
                page_size=5
            )
            
            print(f"   Status: {tech_headlines.get('status', 'unknown')}")
            print(f"   Total results: {tech_headlines.get('totalResults', 0)}")
            print(f"   Articles returned: {len(tech_headlines.get('articles', []))}")
            
            articles = tech_headlines.get('articles', [])
            for i, article in enumerate(articles[:3], 1):
                print(f"   - {i}: {article.get('title', 'No title')[:60]}...")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Test 5: Check API status and limits
        print("\n📊 Test 5: API Status Check")
        try:
            # Try a minimal request to check API status
            status_check = newsapi.get_top_headlines(page_size=1)
            print(f"   API Status: {status_check.get('status', 'unknown')}")
            print(f"   Error code: {status_check.get('code', 'none')}")
            print(f"   Message: {status_check.get('message', 'none')}")
            
        except Exception as e:
            print(f"   ❌ Status check failed: {e}")
    
    except ImportError:
        print("❌ newsapi-python not installed")
    except Exception as e:
        print(f"❌ General error: {e}")

if __name__ == "__main__":
    test_news_api_detailed()