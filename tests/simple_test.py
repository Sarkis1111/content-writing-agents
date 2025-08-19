#!/usr/bin/env python3
"""
Simple test to verify core tool functionality without complex imports.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports without the complex dependency chain."""
    print("🚀 TESTING TOOL IMPORTS (SIMPLIFIED)")
    print("="*50)
    
    # Test basic pydantic models first
    print("\n📋 Testing Basic Models:")
    try:
        from pydantic import BaseModel, Field
        
        class TestModel(BaseModel):
            name: str = Field(..., description="Test field")
            
        model = TestModel(name="test")
        print("✅ Pydantic models working")
    except Exception as e:
        print(f"❌ Pydantic failed: {e}")
        return False
    
    # Test individual tool files by examining their structure
    tool_files = {
        'research': ['web_search.py', 'content_retrieval.py', 'trend_finder.py', 'news_search.py'],
        'analysis': ['content_processing.py', 'topic_extraction.py', 'content_analysis.py', 'reddit_search.py'],
        'writing': ['content_writer.py', 'headline_generator.py', 'image_generator.py'],
        'editing': ['grammar_checker.py', 'seo_analyzer.py', 'readability_scorer.py', 'sentiment_analyzer.py']
    }
    
    total_files = 0
    found_files = 0
    
    for category, files in tool_files.items():
        print(f"\n📂 Testing {category.title()} category:")
        for filename in files:
            filepath = f"src/tools/{category}/{filename}"
            if os.path.exists(filepath):
                print(f"  ✅ {filename} - file exists")
                found_files += 1
                
                # Basic file content check
                with open(filepath, 'r') as f:
                    content = f.read()
                    if 'class' in content and 'Tool' in content:
                        print(f"     Contains tool class")
                    if 'mcp_' in content:
                        print(f"     Contains MCP functions")
            else:
                print(f"  ❌ {filename} - missing")
            
            total_files += 1
    
    print(f"\n📊 File Summary: {found_files}/{total_files} files found")
    return found_files == total_files

def test_dependencies():
    """Test if core dependencies are available."""
    print("\n🔧 TESTING CORE DEPENDENCIES")
    print("="*50)
    
    dependencies = [
        # Core Python libraries
        ('aiohttp', 'HTTP client for async requests'),
        ('pydantic', 'Data validation and settings'),
        ('requests', 'HTTP library'),
        
        # Research tools
        ('serpapi', 'SerpAPI for web search'),
        ('beautifulsoup4', 'HTML parsing'),
        ('pytrends', 'Google Trends API'),
        ('newsapi', 'News API client'),
        
        # Analysis tools  
        ('nltk', 'Natural language processing'),
        ('textblob', 'Text processing'),
        ('langdetect', 'Language detection'),
        ('praw', 'Reddit API wrapper'),
        
        # Writing tools
        ('openai', 'OpenAI API client'),
        ('pillow', 'Image processing'),
        
        # Editing tools
        ('textstat', 'Text statistics'),
        ('language_tool_python', 'Grammar checking')
    ]
    
    available = 0
    total = len(dependencies)
    
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"✅ {package:20} - {description}")
            available += 1
        except ImportError:
            print(f"❌ {package:20} - {description}")
    
    print(f"\n📊 Dependencies: {available}/{total} available")
    return available / total

def test_tool_structure():
    """Test if tool files have expected structure."""
    print("\n🏗️  TESTING TOOL STRUCTURE")
    print("="*50)
    
    # Look for key patterns in tool files
    patterns_to_check = [
        ('Tool class', r'class \w+Tool'),
        ('MCP function', r'def mcp_\w+'),
        ('Pydantic models', r'class \w+\(BaseModel\)'),
        ('Async methods', r'async def'),
    ]
    
    import re
    structure_results = {}
    
    for category in ['research', 'analysis', 'writing', 'editing']:
        print(f"\n📁 {category.title()} Tools:")
        category_path = f"src/tools/{category}"
        
        if os.path.exists(category_path):
            files = [f for f in os.listdir(category_path) if f.endswith('.py') and f != '__init__.py']
            
            for filename in files:
                filepath = os.path.join(category_path, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    tool_name = filename[:-3]  # Remove .py
                    print(f"  📄 {tool_name}:")
                    
                    patterns_found = []
                    for pattern_name, pattern in patterns_to_check:
                        if re.search(pattern, content):
                            patterns_found.append(pattern_name)
                            print(f"     ✅ {pattern_name}")
                        else:
                            print(f"     ❌ {pattern_name}")
                    
                    structure_results[f"{category}/{tool_name}"] = patterns_found
                    
                except Exception as e:
                    print(f"     ❌ Error reading file: {e}")
        else:
            print(f"  ❌ Category directory not found")
    
    return structure_results

def main():
    """Run all tests."""
    print("🧪 CONTENT WRITING AGENTS - SIMPLIFIED TESTING")
    print("Testing Phase 2 Implementation Without Complex Dependencies")
    print("="*60)
    
    # Test 1: Basic imports and file structure
    files_ok = test_basic_imports()
    
    # Test 2: Dependencies
    dep_ratio = test_dependencies()
    
    # Test 3: Tool structure
    structure_results = test_tool_structure()
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    if files_ok:
        print("✅ All tool files are present and properly located")
    else:
        print("❌ Some tool files are missing")
    
    if dep_ratio >= 0.8:
        print(f"✅ Dependencies: {dep_ratio:.1%} available (good)")
    elif dep_ratio >= 0.6:
        print(f"⚠️  Dependencies: {dep_ratio:.1%} available (partial)")
    else:
        print(f"❌ Dependencies: {dep_ratio:.1%} available (insufficient)")
    
    total_tools = len(structure_results)
    tools_with_good_structure = sum(1 for patterns in structure_results.values() if len(patterns) >= 3)
    
    print(f"🏗️  Tool Structure: {tools_with_good_structure}/{total_tools} tools have good structure")
    
    # Overall assessment
    if files_ok and dep_ratio >= 0.7 and tools_with_good_structure / total_tools >= 0.8:
        print("\n🎉 OVERALL: Phase 2 implementation looks good!")
        print("   ✅ Ready to proceed with fixing imports and testing")
    else:
        print("\n⚠️  OVERALL: Some issues detected")
        print("   🔧 May need fixes before full testing")
    
    return {
        'files_ok': files_ok,
        'dependency_ratio': dep_ratio,
        'structure_results': structure_results
    }

if __name__ == "__main__":
    results = main()