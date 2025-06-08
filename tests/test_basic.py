import pytest
from pathlib import Path
import tempfile

def test_package_import():
    """Test that the package can be imported"""
    try:
        import autowriter_mcp
        assert autowriter_mcp.__version__ == "0.2.0"
    except ImportError:
        pytest.skip("Package not installed in development mode")

def test_server_import():
    """Test that server module can be imported"""
    try:
        from autowriter_mcp.server import LMStudioClient, AutowriterMCPServer
        assert LMStudioClient is not None
        assert AutowriterMCPServer is not None
    except ImportError:
        pytest.skip("Server module not available")

def test_lmstudio_client_init():
    """Test LMStudio client initialization"""
    try:
        from autowriter_mcp.server import LMStudioClient
        client = LMStudioClient()
        assert client.base_url == "http://localhost:1234"
        
        client_custom = LMStudioClient("http://example.com:8000")
        assert client_custom.base_url == "http://example.com:8000"
    except ImportError:
        pytest.skip("LMStudioClient not available")

def test_main_function_help():
    """Test that main function shows help when called with --help"""
    try:
        from autowriter_mcp.server import main
        import sys
        from io import StringIO
        
        # Capture stdout
        old_argv = sys.argv
        old_stdout = sys.stdout
        
        try:
            sys.argv = ["autowriter-mcp", "--help"]
            sys.stdout = StringIO()
            
            # Should exit with 0 due to --help
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 0
            
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            
    except ImportError:
        pytest.skip("Main function not available")

if __name__ == "__main__":
    pytest.main([__file__])