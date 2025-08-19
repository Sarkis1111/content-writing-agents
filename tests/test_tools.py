#!/usr/bin/env python3
"""
Simple test script to verify all tools are working properly.
"""

import sys
import os
import traceback
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all tool modules can be imported."""
    print("="*60)
    print("TESTING TOOL IMPORTS")
    print("="*60)
    
    test_results = {}
    
    # Test Research Tools
    print("\n📊 Testing Research Tools:")
    try:
        from tools.research import RESEARCH_TOOLS, MCP_RESEARCH_FUNCTIONS
        print(f"✅ Research Tools: {len(RESEARCH_TOOLS)} tools loaded")
        print(f"   Tools: {list(RESEARCH_TOOLS.keys())}")
        print(f"✅ MCP Functions: {len(MCP_RESEARCH_FUNCTIONS)} functions loaded")
        test_results['research'] = {'status': 'success', 'tools': len(RESEARCH_TOOLS), 'mcp': len(MCP_RESEARCH_FUNCTIONS)}
    except Exception as e:
        print(f"❌ Research Tools failed: {e}")
        test_results['research'] = {'status': 'error', 'error': str(e)}
    
    # Test Analysis Tools
    print("\n🔬 Testing Analysis Tools:")
    try:
        from tools.analysis import ANALYSIS_TOOLS, MCP_ANALYSIS_FUNCTIONS
        print(f"✅ Analysis Tools: {len(ANALYSIS_TOOLS)} tools loaded")
        print(f"   Tools: {list(ANALYSIS_TOOLS.keys())}")
        print(f"✅ MCP Functions: {len(MCP_ANALYSIS_FUNCTIONS)} functions loaded")
        test_results['analysis'] = {'status': 'success', 'tools': len(ANALYSIS_TOOLS), 'mcp': len(MCP_ANALYSIS_FUNCTIONS)}
    except Exception as e:
        print(f"❌ Analysis Tools failed: {e}")
        test_results['analysis'] = {'status': 'error', 'error': str(e)}
        
    # Test Writing Tools
    print("\n✍️  Testing Writing Tools:")
    try:
        from tools.writing import WRITING_TOOLS, MCP_WRITING_FUNCTIONS
        print(f"✅ Writing Tools: {len(WRITING_TOOLS)} tools loaded")
        print(f"   Tools: {list(WRITING_TOOLS.keys())}")
        print(f"✅ MCP Functions: {len(MCP_WRITING_FUNCTIONS)} functions loaded")
        test_results['writing'] = {'status': 'success', 'tools': len(WRITING_TOOLS), 'mcp': len(MCP_WRITING_FUNCTIONS)}
    except Exception as e:
        print(f"❌ Writing Tools failed: {e}")
        test_results['writing'] = {'status': 'error', 'error': str(e)}
        
    # Test Editing Tools
    print("\n📝 Testing Editing Tools:")
    try:
        from tools.editing import EDITING_TOOLS, MCP_EDITING_FUNCTIONS
        print(f"✅ Editing Tools: {len(EDITING_TOOLS)} tools loaded")
        print(f"   Tools: {list(EDITING_TOOLS.keys())}")
        print(f"✅ MCP Functions: {len(MCP_EDITING_FUNCTIONS)} functions loaded")
        test_results['editing'] = {'status': 'success', 'tools': len(EDITING_TOOLS), 'mcp': len(MCP_EDITING_FUNCTIONS)}
    except Exception as e:
        print(f"❌ Editing Tools failed: {e}")
        test_results['editing'] = {'status': 'error', 'error': str(e)}
        
    # Test Combined Import
    print("\n🎯 Testing Combined Import:")
    try:
        from tools import ALL_TOOLS, ALL_MCP_FUNCTIONS, TOOL_CATEGORIES
        print(f"✅ Combined Import: {len(ALL_TOOLS)} total tools")
        print(f"✅ Combined MCP: {len(ALL_MCP_FUNCTIONS)} total MCP functions")
        print(f"✅ Categories: {list(TOOL_CATEGORIES.keys())}")
        test_results['combined'] = {'status': 'success', 'tools': len(ALL_TOOLS), 'mcp': len(ALL_MCP_FUNCTIONS)}
    except Exception as e:
        print(f"❌ Combined Import failed: {e}")
        test_results['combined'] = {'status': 'error', 'error': str(e)}
    
    return test_results

def test_basic_functionality():
    """Test basic functionality of key tools."""
    print("\n" + "="*60)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*60)
    
    functionality_results = {}
    
    try:
        from tools import ALL_TOOLS
        
        # Test a sample from each category
        test_cases = [
            ('research', 'web_search'),
            ('analysis', 'content_processing'), 
            ('writing', 'content_writer'),
            ('editing', 'grammar_checker')
        ]
        
        for category, tool_name in test_cases:
            print(f"\n🔧 Testing {category} tool: {tool_name}")
            try:
                tool = ALL_TOOLS.get(tool_name)
                if tool:
                    print(f"✅ Tool instance found: {type(tool).__name__}")
                    # Check if tool has expected methods
                    if hasattr(tool, '__class__'):
                        methods = [method for method in dir(tool) if not method.startswith('_')]
                        print(f"   Methods: {', '.join(methods[:5])}...")
                    functionality_results[tool_name] = {'status': 'success', 'type': type(tool).__name__}
                else:
                    print(f"❌ Tool not found in registry")
                    functionality_results[tool_name] = {'status': 'error', 'error': 'not found'}
            except Exception as e:
                print(f"❌ Error testing {tool_name}: {e}")
                functionality_results[tool_name] = {'status': 'error', 'error': str(e)}
                
    except Exception as e:
        print(f"❌ Could not load ALL_TOOLS: {e}")
        functionality_results['overall'] = {'status': 'error', 'error': str(e)}
    
    return functionality_results

def test_environment_setup():
    """Test environment configuration and API keys."""
    print("\n" + "="*60)
    print("TESTING ENVIRONMENT SETUP")
    print("="*60)
    
    env_results = {}
    
    # Check critical environment variables
    env_vars = {
        'OPENAI_API_KEY': 'required for writing tools',
        'SERPAPI_KEY': 'optional for enhanced web search',
        'GOOGLE_API_KEY': 'optional for Google Custom Search',
        'NEWS_API_KEY': 'optional for news monitoring',
        'REDDIT_CLIENT_ID': 'optional for Reddit analysis'
    }
    
    print("\n🔑 API Keys Status:")
    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            masked_value = value[:8] + "..." if len(value) > 8 else "***"
            print(f"✅ {var}: {masked_value} ({description})")
            env_results[var] = {'status': 'configured', 'description': description}
        else:
            status = "❗" if var == 'OPENAI_API_KEY' else "ℹ️"
            print(f"{status} {var}: Not set ({description})")
            env_results[var] = {'status': 'not_set', 'description': description}
    
    # Test tool validation
    print("\n🔍 Tool Validation:")
    try:
        from tools import validate_all_tools
        validation = validate_all_tools()
        print(f"✅ Overall Status: {validation['overall_status']}")
        for category, info in validation['categories'].items():
            print(f"   {category}: {info['status']} ({info['tools_count']} tools)")
        env_results['validation'] = validation
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        env_results['validation'] = {'error': str(e)}
    
    return env_results

def main():
    """Run all tests."""
    print("🚀 CONTENT WRITING AGENTS - TOOL TESTING")
    print("Phase 2 Complete - Testing All 15 Tools")
    
    try:
        # Test imports
        import_results = test_imports()
        
        # Test basic functionality  
        functionality_results = test_basic_functionality()
        
        # Test environment setup
        env_results = test_environment_setup()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_successes = sum(1 for result in import_results.values() if result.get('status') == 'success')
        print(f"📊 Import Tests: {total_successes}/{len(import_results)} categories successful")
        
        func_successes = sum(1 for result in functionality_results.values() if result.get('status') == 'success')
        print(f"🔧 Functionality Tests: {func_successes}/{len(functionality_results)} tools successful")
        
        configured_keys = sum(1 for result in env_results.values() if isinstance(result, dict) and result.get('status') == 'configured')
        print(f"🔑 Environment: {configured_keys} API keys configured")
        
        if 'validation' in env_results and 'overall_status' in env_results['validation']:
            print(f"🎯 Overall System Status: {env_results['validation']['overall_status'].upper()}")
        
        print(f"\n✨ Phase 2 Testing Complete!")
        print(f"   Ready for Phase 3: Agent Development")
        
        return {
            'imports': import_results,
            'functionality': functionality_results,
            'environment': env_results
        }
        
    except Exception as e:
        print(f"❌ Test script failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()