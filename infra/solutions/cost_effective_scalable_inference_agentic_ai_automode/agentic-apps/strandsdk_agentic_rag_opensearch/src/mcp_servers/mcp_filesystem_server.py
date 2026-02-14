#!/usr/bin/env python3
"""
Simple MCP Filesystem Server

This server provides basic filesystem operations through the Model Context Protocol.
It demonstrates how to create MCP tools that can be used by the Strands agents.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict
from mcp.server import FastMCP

# Create MCP server
mcp = FastMCP("Filesystem Server")

@mcp.tool(description="Read the contents of a file")
def read_file(file_path: str) -> str:
    """Read and return the contents of a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        The contents of the file as a string
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File {file_path} does not exist"
        
        if not path.is_file():
            return f"Error: {file_path} is not a file"
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

@mcp.tool(description="Write content to a file")
def write_file(file_path: str, content: str, append: bool = False) -> str:
    """Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        append: Whether to append to the file (default: False, overwrites)
        
    Returns:
        Success or error message
    """
    try:
        path = Path(file_path)
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        mode = 'a' if append else 'w'
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)
        
        action = "appended to" if append else "written to"
        return f"Successfully {action} {file_path}"
    except Exception as e:
        return f"Error writing to file {file_path}: {str(e)}"

@mcp.tool(description="List files and directories in a path")
def list_directory(dir_path: str = ".") -> str:
    """List the contents of a directory.
    
    Args:
        dir_path: Path to the directory to list (default: current directory)
        
    Returns:
        A formatted list of directory contents
    """
    try:
        path = Path(dir_path)
        if not path.exists():
            return f"Error: Directory {dir_path} does not exist"
        
        if not path.is_dir():
            return f"Error: {dir_path} is not a directory"
        
        items = []
        for item in sorted(path.iterdir()):
            if item.is_dir():
                items.append(f"📁 {item.name}/")
            else:
                size = item.stat().st_size
                items.append(f"📄 {item.name} ({size} bytes)")
        
        if not items:
            return f"Directory {dir_path} is empty"
        
        return f"Contents of {dir_path}:\n" + "\n".join(items)
    except Exception as e:
        return f"Error listing directory {dir_path}: {str(e)}"

@mcp.tool(description="Create a new directory")
def create_directory(dir_path: str) -> str:
    """Create a new directory.
    
    Args:
        dir_path: Path to the directory to create
        
    Returns:
        Success or error message
    """
    try:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return f"Successfully created directory {dir_path}"
    except Exception as e:
        return f"Error creating directory {dir_path}: {str(e)}"

@mcp.tool(description="Delete a file or directory")
def delete_path(path_to_delete: str) -> str:
    """Delete a file or directory.
    
    Args:
        path_to_delete: Path to the file or directory to delete
        
    Returns:
        Success or error message
    """
    try:
        path = Path(path_to_delete)
        if not path.exists():
            return f"Error: Path {path_to_delete} does not exist"
        
        if path.is_file():
            path.unlink()
            return f"Successfully deleted file {path_to_delete}"
        elif path.is_dir():
            # Only delete empty directories for safety
            try:
                path.rmdir()
                return f"Successfully deleted directory {path_to_delete}"
            except OSError:
                return f"Error: Directory {path_to_delete} is not empty. Only empty directories can be deleted."
        else:
            return f"Error: {path_to_delete} is neither a file nor a directory"
    except Exception as e:
        return f"Error deleting {path_to_delete}: {str(e)}"

@mcp.tool(description="Get file or directory information")
def get_path_info(path_to_check: str) -> str:
    """Get information about a file or directory.
    
    Args:
        path_to_check: Path to check
        
    Returns:
        Information about the path
    """
    try:
        path = Path(path_to_check)
        if not path.exists():
            return f"Path {path_to_check} does not exist"
        
        stat = path.stat()
        info = {
            "path": str(path.absolute()),
            "name": path.name,
            "type": "directory" if path.is_dir() else "file",
            "size": stat.st_size if path.is_file() else "N/A",
            "modified": stat.st_mtime,
            "permissions": oct(stat.st_mode)[-3:]
        }
        
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error getting info for {path_to_check}: {str(e)}"

if __name__ == "__main__":
    print("Starting MCP Filesystem Server on http://localhost:8001")
    print("Available tools:")
    print("- read_file: Read file contents")
    print("- write_file: Write content to file")
    print("- list_directory: List directory contents")
    print("- create_directory: Create new directory")
    print("- delete_path: Delete file or empty directory")
    print("- get_path_info: Get file/directory information")
    print("\nPress Ctrl+C to stop the server")
    
    mcp = FastMCP("my mcp", port=8001)
    # Run the server with Streamable HTTP transport
    mcp.run(transport="streamable-http")
