# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-06-08

### 🚀 Major Changes
- **Breaking Change**: Removed hardcoded configuration file in favor of command-line arguments
- Added proper Python package structure with `pyproject.toml`
- Added PyPI and uvx installation support
- Open-sourced project with MIT license

### ✨ New Features
- Command-line argument configuration for vault path, index file, and LMStudio URL
- Multiple vault support through configurable paths
- Enhanced error handling and validation
- Comprehensive documentation and migration guide
- GitHub Actions ready for CI/CD
- Professional README with installation and usage instructions

### 🛠 Improvements
- Removed sensitive hardcoded paths from configuration
- Better logging and error messages
- Improved project structure following Python best practices
- Added development dependencies and tooling configuration
- Created comprehensive `.gitignore` for clean repository

### 🔧 Technical Changes
- Moved from hardcoded config file to argparse-based configuration
- Added proper package metadata and dependencies
- Implemented clean separation between development and production configurations
- Added support for multiple installation methods (uvx, pip, local development)

### 📚 Documentation
- Complete README with installation, configuration, and usage examples
- Migration guide for upgrading from v0.1.0
- License file (MIT)
- Comprehensive `.gitignore`
- Development setup instructions

### 🐛 Bug Fixes
- Fixed vault path validation
- Improved error handling for missing LMStudio connection
- Better handling of missing index files

### 🔐 Security
- Removed all hardcoded local paths and sensitive information
- Added `.gitignore` to prevent accidental commit of sensitive files
- Clean separation between example and actual configuration

## [0.1.0] - 2025-06-03

### Initial Release
- Basic MCP server functionality
- Obsidian vault integration
- LMStudio content generation
- Token-saving architecture
- Queue-based content generation
- Index link management

### Features
- `analyze_book_structure` tool
- `generate_and_save_section` tool with token-saving
- `batch_generate_sections` for multiple content generation
- `process_writing_queue` for automated processing
- `request_content_generation` for queue management
- `update_index_links` for Obsidian integration
- `get_writing_status` for progress tracking

### Architecture
- Direct LMStudio API integration
- Local content generation to save Claude tokens
- Obsidian markdown file management
- FastMCP framework usage