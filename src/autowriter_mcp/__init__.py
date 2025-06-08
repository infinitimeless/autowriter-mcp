"""
Autowriter MCP Server Package

A token-saving Model Context Protocol server for coordinating automated 
writing workflows between Obsidian vaults and LMStudio.
"""

__version__ = "0.2.0"
__author__ = "autowriter-mcp contributors"
__license__ = "MIT"

from .server import AutowriterMCPServer, LMStudioClient

__all__ = ["AutowriterMCPServer", "LMStudioClient"]