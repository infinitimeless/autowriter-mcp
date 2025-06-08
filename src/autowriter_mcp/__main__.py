"""
Main entry point for running autowriter-mcp as a module
python -m autowriter_mcp
"""

from .server import main

if __name__ == "__main__":
    exit(main())