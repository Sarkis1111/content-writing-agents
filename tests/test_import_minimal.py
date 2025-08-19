"""
Minimal test to identify specific import issues.
"""

import sys
import os

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

def test_specific_imports():
    """Test specific imports individually."""
    
    # Test 1: Try importing the content analysis module directly
    try:
        import tools.analysis.content_analysis as ca_module
        print("✅ Content analysis module imported directly")
        print(f"Module file: {ca_module.__file__}")
    except Exception as e:
        print(f"❌ Direct module import failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-"*50)
    
    # Test 2: Try importing specific classes
    try:
        from tools.analysis.content_analysis import ContentAnalysisRequest
        print("✅ ContentAnalysisRequest imported successfully")
        
        # Test creating an instance
        request = ContentAnalysisRequest(text="test")
        print("✅ ContentAnalysisRequest instance created")
        
    except Exception as e:
        print(f"❌ ContentAnalysisRequest import failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-"*50)
    
    # Test 3: Try importing the tool class
    try:
        from tools.analysis.content_analysis import ContentAnalysisTool
        print("✅ ContentAnalysisTool imported successfully")
        
        # Test creating an instance  
        tool = ContentAnalysisTool()
        print("✅ ContentAnalysisTool instance created")
        
    except Exception as e:
        print(f"❌ ContentAnalysisTool import failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_specific_imports()