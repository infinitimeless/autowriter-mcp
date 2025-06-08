# Migration Guide

## Upgrading from v0.1.0 to v0.2.0

### Breaking Changes

#### Configuration Method
**v0.1.0**: Used hardcoded configuration file `config/server_config.json`
**v0.2.0**: Uses command-line arguments for configuration

#### Migration Steps

1. **Update your Claude Desktop configuration**:

   **Old configuration**:
   ```json
   {
     "mcpServers": {
       "autowriter-mcp": {
         "command": "/path/to/autowriter-mcp/.venv/bin/python",
         "args": ["/path/to/autowriter-mcp/src/autowriter_mcp/server.py"]
       }
     }
   }
   ```

   **New configuration**:
   ```json
   {
     "mcpServers": {
       "autowriter-mcp": {
         "command": "/path/to/autowriter-mcp/.venv/bin/python",
         "args": ["-m", "autowriter_mcp.server", "/path/to/your/obsidian/vault"]
       }
     }
   }
   ```

2. **Remove old configuration file**:
   ```bash
   rm config/server_config.json
   ```

3. **Update command-line usage**:

   **Old usage**:
   ```bash
   python src/autowriter_mcp/server.py
   ```

   **New usage**:
   ```bash
   autowriter-mcp '/path/to/your/obsidian/vault'
   ```

### New Features in v0.2.0

- ✅ Command-line argument configuration
- ✅ PyPI package support
- ✅ uvx installation support
- ✅ Multiple vault support
- ✅ Improved error handling
- ✅ Better documentation

### Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `vault_path` | (required) | Path to your Obsidian vault |
| `--index-file` | `book_index.md` | Index file name |
| `--lmstudio-url` | `http://localhost:1234` | LMStudio server URL |

### Example Configurations

#### Basic Usage
```bash
autowriter-mcp '/Users/username/Documents/MyBook'
```

#### Custom Index File
```bash
autowriter-mcp '/Users/username/Documents/MyBook' --index-file 'table_of_contents.md'
```

#### Custom LMStudio URL
```bash
autowriter-mcp '/Users/username/Documents/MyBook' --lmstudio-url 'http://192.168.1.100:1234'
```

### Troubleshooting

#### "Vault directory does not exist" Error
- Ensure the path to your Obsidian vault is correct
- Use absolute paths for best compatibility
- Check directory permissions

#### "LMStudio connection failed" Error
- Verify LMStudio is running on the specified URL
- Check if the port is accessible
- Ensure a model is loaded in LMStudio

#### "Index file not found" Error
- Create the index file in your vault root directory
- Use the `--index-file` argument if using a different filename
- Ensure the file has `.md` extension

### Getting Help

- Check the [README](../README.md) for detailed installation instructions
- Review [configuration examples](../README.md#configuration)
- Open an [issue](https://github.com/infinitimeless/autowriter-mcp/issues) if you encounter problems