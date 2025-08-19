"""
Debug import issues in the Strategy Agent implementation.
"""

import sys
import os

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

print(f"Current directory: {current_dir}")
print(f"Source path: {src_path}")
print(f"Python path: {sys.path[:3]}")

def test_import_paths():
    """Test different import approaches."""
    
    # Test 1: Direct imports from tools
    try:
        from tools.analysis.content_analysis import ContentAnalysisTool
        print("✅ Direct import from tools.analysis.content_analysis works")
    except Exception as e:
        print(f"❌ Direct import failed: {e}")
    
    # Test 2: Add tools to path
    try:
        tools_path = os.path.join(src_path, 'tools')
        sys.path.insert(0, tools_path)
        from analysis.content_analysis import ContentAnalysisTool
        print("✅ Import with tools in path works")
    except Exception as e:
        print(f"❌ Tools path import failed: {e}")
    
    # Test 3: Absolute imports
    try:
        sys.path.insert(0, os.path.join(src_path, 'tools', 'analysis'))
        import content_analysis
        print("✅ Direct module import works")
    except Exception as e:
        print(f"❌ Direct module import failed: {e}")
    
    # Test 4: Check file exists
    content_analysis_path = os.path.join(src_path, 'tools', 'analysis', 'content_analysis.py')
    print(f"Content analysis file exists: {os.path.exists(content_analysis_path)}")
    
    # Test 5: Strategy agent imports
    try:
        strategy_path = os.path.join(src_path, 'agents', 'strategy')
        sys.path.insert(0, strategy_path)
        from models import StrategyRequest
        print("✅ Strategy models import works")
    except Exception as e:
        print(f"❌ Strategy models import failed: {e}")

if __name__ == "__main__":
    test_import_paths()