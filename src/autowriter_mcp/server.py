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

class AutowriterMCPServer:
    """
    Enhanced Autowriter MCP Server with direct LMStudio integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Autowriter MCP server with provided configuration"""
        self.config = config
        self.mcp = FastMCP(self.config["server_info"]["name"])
        
        # Initialize LMStudio client
        self.lmstudio = LMStudioClient(self.config["lmstudio_config"]["base_url"])
        
        # Track writing progress
        self.writing_queue: List[Dict[str, Any]] = []
        self.completed_sections: List[str] = []
        
        logger.info(f"Initialized {self.config['server_info']['name']} v{self.config['server_info']['version']}")
        logger.info(f"Vault path: {self.config['vault_config']['vault_path']}")
        logger.info(f"LMStudio URL: {self.config['lmstudio_config']['base_url']}")
        
        # Register tools
        self._register_tools()
        
    def _register_tools(self):
        """Register all MCP tools"""
        
        @self.mcp.tool()
        async def analyze_book_structure() -> str:
            """
            Analyze the book structure from the Obsidian vault and identify missing content.
            
            Returns:
                Analysis of current book structure and missing sections
            """
            try:
                vault_path = Path(self.config["vault_config"]["vault_path"])
                index_file = vault_path / self.config["vault_config"]["index_file"]
                
                if not vault_path.exists():
                    return f"❌ Vault directory not found: {vault_path}"
                    
                if not index_file.exists():
                    return f"❌ Index file not found: {index_file}\n💡 Create '{self.config['vault_config']['index_file']}' in your vault to get started!"
                
                # Read the index file
                with open(index_file, 'r', encoding='utf-8') as f:
                    index_content = f.read()
                
                # Analyze structure
                sections = self._parse_index_sections(index_content)
                missing_sections = self._identify_missing_sections(sections, vault_path)
                
                result = {
                    "total_sections": len(sections),
                    "completed_sections": len(sections) - len(missing_sections),
                    "missing_sections": missing_sections,
                    "next_to_write": missing_sections[:3] if missing_sections else []
                }
                
                logger.info(f"Book analysis complete: {result['completed_sections']}/{result['total_sections']} sections done")
                
                return self._format_analysis_result(result)
                
            except Exception as e:
                logger.error(f"Error analyzing book structure: {str(e)}")
                return f"❌ Error analyzing book structure: {str(e)}"
        
        @self.mcp.tool()
        async def generate_and_save_section(section_title: str, section_type: str = "chapter", word_count: int = 1000, writing_style: str = "informative") -> str:
            """
            🚀 TOKEN-SAVING: Generate content via LMStudio and save directly to vault.
            Claude never sees the generated content, saving 80-90% of tokens!
            
            Args:
                section_title: The title of the section to generate
                section_type: Type of content (chapter, subchapter, paragraph)
                word_count: Target word count for the section
                writing_style: Style of writing (informative, narrative, academic, etc.)
                
            Returns:
                Success message with metadata (NOT the content itself)
            """
            try:
                # Create generation prompt
                prompt = self._create_generation_prompt(section_title, section_type, word_count, writing_style)
                
                logger.info(f"Generating content for: {section_title}")
                
                # 🎯 KEY: Generate content locally via LMStudio
                generated_content = await self.lmstudio.generate_content(
                    prompt=prompt,
                    max_tokens=int(word_count * 1.5)  # Buffer for longer content
                )
                
                # 🎯 KEY: Save directly to vault WITHOUT sending to Claude
                file_path = await self._save_content_to_vault(section_title, generated_content)
                
                # Update tracking
                if section_title not in self.completed_sections:
                    self.completed_sections.append(section_title)
                
                # Remove from queue if it was queued
                self.writing_queue = [q for q in self.writing_queue if q.get('section_title') != section_title]
                
                # 🎯 KEY: Return only METADATA, not content (saves tokens!)
                return f"✅ **Successfully generated and saved '{section_title}'**\n\n" \
                       f"📝 Type: {section_type}\n" \
                       f"📊 Generated: {len(generated_content.split())} words\n" \
                       f"📁 Saved to: {file_path}\n" \
                       f"🎨 Style: {writing_style}\n\n" \
                       f"💡 **Token Saver**: Content generated locally and saved directly!\n" \
                       f"🔗 Use 'update_index_links' to add to your index."
                
            except Exception as e:
                logger.error(f"Error generating content: {str(e)}")
                return f"❌ Error generating content for '{section_title}': {str(e)}"
        
        @self.mcp.tool()
        async def get_writing_status() -> str:
            """Get current writing progress and queue status"""
            try:
                total_in_queue = len(self.writing_queue)
                completed_count = len(self.completed_sections)
                
                status_report = f"📊 **Writing Status Report**\n\n"
                status_report += f"✅ Completed sections: {completed_count}\n"
                status_report += f"⏳ Queued for writing: {total_in_queue}\n"
                status_report += f"📁 Vault: {self.config['vault_config']['vault_path']}\n"
                status_report += f"🔗 LMStudio: {self.config['lmstudio_config']['base_url']}\n\n"
                
                return status_report
                
            except Exception as e:
                logger.error(f"Error getting status: {str(e)}")
                return f"❌ Error getting status: {str(e)}"
    
    async def _save_content_to_vault(self, section_title: str, content: str) -> str:
        """Save content directly to vault and return file path"""
        vault_path = Path(self.config["vault_config"]["vault_path"])
        
        # Generate safe file name
        safe_title = "".join(c for c in section_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_').lower()
        file_path = f"{safe_title}.md"
        
        full_path = vault_path / file_path
        
        # Ensure directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create markdown content
        markdown_content = f"# {section_title}\n\n{content}\n"
        
        # Write to file
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Saved content to: {file_path}")
        return file_path
    
    def _create_generation_prompt(self, section_title: str, section_type: str, word_count: int, writing_style: str) -> str:
        """Create an effective prompt for content generation"""
        return f"""Write a {word_count}-word {section_type} titled "{section_title}".

Writing requirements:
- Style: {writing_style}
- Length: approximately {word_count} words
- Format: Clear, well-structured content suitable for a book
- Quality: Professional, engaging, and informative

Please write the {section_type} content directly without meta-commentary or introductory text."""
    
    def _parse_index_sections(self, content: str) -> List[str]:
        """Parse section titles from index content"""
        sections = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('#') or line.startswith('-') or line.startswith('*'):
                title = line.lstrip('#-* ').strip()
                if title and not title.startswith('[['):
                    sections.append(title)
        
        return sections
    
    def _identify_missing_sections(self, sections: List[str], vault_path: Path) -> List[str]:
        """Identify which sections don't have corresponding files"""
        missing = []
        
        for section in sections:
            safe_title = "".join(c for c in section if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_').lower()
            expected_file = vault_path / f"{safe_title}.md"
            
            if not expected_file.exists():
                missing.append(section)
        
        return missing
    
    def _format_analysis_result(self, result: Dict[str, Any]) -> str:
        """Format the analysis result for display"""
        output = f"📚 **Book Structure Analysis**\n\n"
        output += f"📊 Progress: {result['completed_sections']}/{result['total_sections']} sections completed\n"
        
        if result['total_sections'] > 0:
            output += f"📈 Completion: {(result['completed_sections']/result['total_sections']*100):.1f}%\n\n"
        else:
            output += f"📈 Completion: 0.0%\n\n"
        
        if result['missing_sections']:
            output += f"❌ Missing sections ({len(result['missing_sections'])}):\n"
            for section in result['missing_sections'][:10]:
                output += f"- {section}\n"
            if len(result['missing_sections']) > 10:
                output += f"... and {len(result['missing_sections']) - 10} more\n"
            output += "\n"
        
        if result['next_to_write']:
            output += "🎯 **Token-Saving Generation Options:**\n"
            for i, section in enumerate(result['next_to_write'], 1):
                output += f"{i}. Use 'generate_and_save_section' for: {section}\n"
            output += "\n💡 Generate content locally to save Claude tokens!"
        else:
            output += "🎉 **All sections are complete!**\n"
        
        return output

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
        
        # Initialize server
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
