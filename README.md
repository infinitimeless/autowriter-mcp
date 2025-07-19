# Autowriter MCP Server

A token-saving Model Context Protocol (MCP) server that coordinates automated writing workflows between Obsidian vaults and LMStudio or Gemini free API. Automatic fallback to Gemini API when LMStudio is unavailable. This server enables AI-powered content generation while dramatically reducing Claude token usage by generating content locally via LMStudio or Gemini Free API and saving directly to Obsidian vaults.

## 🚀 Key Features

### Core Features
- **Token-Saving Architecture**: Save 80-90% of Claude tokens by generating content locally via LMStudio
- **Gemini Fallback**: Automatic fallback to Gemini API when LMStudio is unavailable
- **Robust Error Handling**: Retry logic and comprehensive error recovery
- **Direct Obsidian Integration**: Seamlessly integrates with Obsidian vaults and markdown files
- **Automated Book Writing**: Analyze book structure and generate missing sections automatically
- **Health Monitoring**: Built-in health checks for content generation services

### 🎓 Professional Book Features
- **Concept Definition Management**: Automatically tracks and validates concept consistency across your book
- **Framework Component Tracking**: Maintains coherent framework structures and relationships
- **Evidence Chain Management**: Validates claims with proper citations and evidence strength
- **Professional Content Generation**: Context-aware generation for technical, business, and academic books
- **Real-time Validation**: Ensures concept consistency, framework coherence, and evidence accuracy
- **Audience-Appropriate Content**: Adapts complexity level to target audience (professional, expert, general)

## 🛠 Prerequisites

- **Python 3.11+**: Required for running the MCP server
- **LMStudio**: Running locally on `http://localhost:1234` (or configurable URL) - **Primary**
- **Gemini API Key**: For fallback content generation (optional but recommended)
- **Claude Desktop**: With MCP support for connecting to the server
- **Obsidian**: For managing your writing vault (optional but recommended)

## 📦 Installation

### Method 1: Using uvx (Recommended)

```bash
# Install and run directly with uvx
uvx autowriter-mcp '/path/to/your/obsidian/vault'
```

### Method 2: Using pip

```bash
# Install from PyPI
pip install autowriter-mcp

# Run the server
autowriter-mcp '/path/to/your/obsidian/vault'
```

### Method 3: Local Development

```bash
# Clone the repository
git clone https://github.com/infinitimeless/autowriter-mcp.git
cd autowriter-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Run the server
autowriter-mcp '/path/to/your/obsidian/vault'
```

## ⚙️ Configuration

### Command Line Options

```bash
autowriter-mcp [vault_path] [options]

Arguments:
  vault_path              Path to your Obsidian vault directory (required)

Options:
  --index-file FILENAME   Index file name (default: book_index.md)
  --lmstudio-url URL      LMStudio server URL (default: http://localhost:1234)
  --gemini-api-key KEY    Gemini API key for fallback (or set GEMINI_API_KEY env var)
  --gemini-model NAME     Gemini model name (default: gemini-1.5-flash)
  --version               Show version information
  --help                  Show help message
```

### Claude Desktop Configuration

Add to your Claude Desktop MCP configuration:

#### For uvx installation:
```json
{
  "mcpServers": {
    "autowriter-mcp": {
      "command": "uvx",
      "args": ["autowriter-mcp", "/path/to/your/obsidian/vault"],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here",
        "GEMINI_MODEL_NAME": "gemini-1.5-flash"
      }
    }
  }
}
```

#### For local development:
```json
{
  "mcpServers": {
    "autowriter-mcp": {
      "command": "/path/to/autowriter-mcp/.venv/bin/python",
      "args": ["-m", "autowriter_mcp.server", "/path/to/your/obsidian/vault"],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here",
        "GEMINI_MODEL_NAME": "gemini-1.5-flash"
      }
    }
  }
}
```

## 🎯 Usage

### 1. Prepare Your Obsidian Vault

Create an index file (default: `book_index.md`) in your vault with your book structure:

```markdown
# My Book Title

## Chapter 1: Introduction
## Chapter 2: Getting Started
## Chapter 3: Advanced Topics
## Chapter 4: Conclusion
```

### 2. Available MCP Tools

#### Core Tools
- **`analyze_book_structure`**: Analyze your vault structure and identify missing content sections
- **`generate_and_save_section`**: 🚀 **Token-Saving** - Generate content locally via LMStudio with Gemini fallback
- **`get_writing_status`**: Get current progress and queue status
- **`check_generator_health`**: Check LMStudio and Gemini fallback health status

#### 🎓 Professional Book Tools
- **`generate_professional_content`**: 🚀 **Professional Generation** - Generate content with concept consistency, framework tracking, and evidence validation
- **`get_professional_status`**: Comprehensive professional book status with validation metrics

**Professional Book Types Supported:**
- **Technical Books**: Code examples, API documentation, technical specifications
- **Business Books**: Strategic frameworks, case studies, business methodologies  
- **Academic Books**: Theoretical frameworks, research methodology, peer-reviewed citations

#### Professional Content Generation Example
```python
generate_professional_content(
    section_title="Advanced Machine Learning Techniques",
    section_type="chapter", 
    word_count=1500,
    book_type="technical",
    target_audience="professional"
)
```

#### Professional Status Monitoring
```python
get_professional_status(book_type="technical")
```

## 🏗 Architecture

### Token-Saving Design

The server is specifically designed to minimize Claude token usage:

1. **Local Generation**: Content is generated by LMStudio, not Claude
2. **Direct File Writing**: Content is saved directly to vault files
3. **Metadata Only**: Claude only receives generation metadata, not content
4. **Batch Processing**: Multiple sections can be generated in one operation

### 🎓 Professional Book Architecture

The professional book system adds intelligent context management:

1. **Concept Registry**: Automatically tracks all concept definitions and their relationships
2. **Framework Mapping**: Maintains hierarchical framework structures and dependencies
3. **Evidence Validation**: Ensures claims are supported by appropriate citations and evidence
4. **Context Compilation**: Builds comprehensive professional context from existing vault content
5. **Real-time Validation**: Validates consistency as new content is generated
6. **Audience Adaptation**: Adjusts complexity level based on target audience

**Professional Context Elements:**
- **Concept Definitions**: Term definitions with complexity levels and relationships
- **Framework Components**: Structured frameworks with implementation steps and prerequisites
- **Evidence Chains**: Claims with supporting evidence strength and citation tracking
- **Methodology Steps**: Sequential processes with dependencies and success criteria
- **Citation Registry**: Consistent citation formats and source validation

### 🔄 Fallback Mechanism

The system provides robust content generation with automatic fallback:

1. **Primary Generation**: LMStudio for local, private content generation
2. **Automatic Fallback**: Gemini API when LMStudio is unavailable
3. **Retry Logic**: Automatic retries with exponential backoff
4. **Health Monitoring**: Real-time status checks for both services
5. **Transparent Operation**: Users are informed which service generated content

**Fallback Sequence:**
1. Try LMStudio (up to 3 attempts with retries)
2. If LMStudio fails, automatically switch to Gemini
3. Try Gemini (up to 3 attempts with retries)
4. Report detailed error information if both fail

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the [Model Context Protocol](https://modelcontextprotocol.io/)
- Integrates with [LMStudio](https://lmstudio.ai/) for local AI generation
- Designed for [Obsidian](https://obsidian.md/) vault management
- Compatible with [Claude Desktop](https://claude.ai/) MCP support

---

**🚀 Save Claude tokens while accelerating your writing workflow with autowriter-mcp!**
