#!/usr/bin/env python3
"""
Enhanced Autowriter MCP Server with Direct LMStudio Integration
Saves Claude tokens by generating and saving content locally
"""

import argparse
import asyncio
import json
import logging
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
    Resource,
    Prompt
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("autowriter-mcp")

class LMStudioClient:
    """Direct LMStudio API client for content generation"""
    
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 min timeout for long generations
        
    async def generate_content(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate content using LMStudio API"""
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "current",  # Use currently loaded model
                    "messages": [
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "stream": False
                }
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            logger.info(f"Generated content: {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"LMStudio generation failed: {str(e)}")
            raise
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


def main():
    """Main entry point for the Enhanced Autowriter MCP Server"""
    try:
        parser = argparse.ArgumentParser(
            description='Enhanced Autowriter MCP Server - Token-saving content generation with direct LMStudio integration'
        )
        parser.add_argument('vault_path', help='Path to the Obsidian vault directory')
        parser.add_argument('--index-file', default='book_index.md', help='Index file name (default: book_index.md)')
        parser.add_argument('--lmstudio-url', default='http://localhost:1234', help='LMStudio server URL (default: http://localhost:1234)')
        parser.add_argument('--version', action='version', version='autowriter-mcp 0.2.0 (Token-Saving Edition)')
        
        args = parser.parse_args()
        
        # Validate vault path
        vault_path = Path(args.vault_path)
        if not vault_path.exists():
            logger.error(f"Vault directory does not exist: {vault_path}")
            print(f"❌ Error: Vault directory does not exist: {vault_path}")
            return 1
        
        if not vault_path.is_dir():
            logger.error(f"Vault path is not a directory: {vault_path}")
            print(f"❌ Error: Vault path is not a directory: {vault_path}")
            return 1
        
        # Create configuration
        config = {
            "server_info": {
                "name": "autowriter-mcp",
                "version": "0.2.0",
                "description": "Token-saving MCP server with direct LMStudio integration"
            },
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": True
            },
            "vault_config": {
                "vault_path": str(vault_path.resolve()),
                "index_file": args.index_file
            },
            "lmstudio_config": {
                "base_url": args.lmstudio_url,
                "model": "current"
            }
        }
        
        # Initialize server - import here to avoid circular imports
        from .core import AutowriterMCPServer
        server = AutowriterMCPServer(config=config)
        
        # Run server
        logger.info("🚀 Starting Token-Saving Autowriter MCP Server...")
        server.mcp.run(transport='stdio')
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        return 0
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
