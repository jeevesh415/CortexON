# Standard library imports
from typing import List, Optional, Dict, Any
import json
import asyncio
import os
import logging
import signal
import sys

# Third-party imports
from anthropic import AsyncAnthropic
from mcp import ClientSession, StdioServerParameters

# Local application imports
from utils.models import MCPServerConfig, MCPToolAnalysis
from utils.ant_client import get_client

# Set up logging
logger = logging.getLogger(__name__)

async def analyze_mcp_server_tools(server_name: str, server_config: MCPServerConfig) -> dict:
    """
    Analyze MCP server tools using Claude LLM - PURE DYNAMIC DISCOVERY ONLY
    
    Args:
        server_name: Name of the MCP server
        server_config: Configuration of the MCP server
        
    Returns:
        Dictionary containing analyzed tool information or error if discovery fails
    """
    import asyncio
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    command = server_config.command
    args = server_config.args
    env = server_config.env

    try:
        logger.info(f"Starting dynamic tool discovery for {server_name}")
        
        # Connect to MCP server and get tools
        async with stdio_client(
            StdioServerParameters(command=command, args=args, env=env)
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Fetch available tools from the server
                tools_response = await session.list_tools()
                logger.info(f"Received tools response from {server_name}: {tools_response}")
        
        # Process the tools response
        if not tools_response or not hasattr(tools_response, 'tools'):
            logger.warning(f"No tools found in response from {server_name}")
            return {
                "error": f"No tools found for {server_name}",
                "server_name": server_name,
                "tools": [],
                "success": False
            }
        
        server_tools = tools_response.tools
        if not server_tools:
            logger.warning(f"Empty tools list from {server_name}")
            return {
                "error": f"Empty tools list for {server_name}",
                "server_name": server_name, 
                "tools": [],
                "success": False
            }
            
        logger.info(f"Processing {len(server_tools)} tools from {server_name}")
        
        # Convert tools to analyzable format
        tools_for_analysis = []
        for i, tool in enumerate(server_tools):
            try:
                tool_info = {
                    "name": f"{server_name}.{tool.name}",
                    "description": tool.description or f"Tool {tool.name} from {server_name}",
                    "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                }
                tools_for_analysis.append(tool_info)
                logger.debug(f"Added tool: {tool_info['name']}")
                
            except Exception as tool_error:
                logger.warning(f"Error processing tool {i} from {server_name}: {tool_error}")
                continue
        
        if not tools_for_analysis:
            return {
                "error": f"Failed to process any tools from {server_name}",
                "server_name": server_name,
                "tools": [],
                "success": False
            }
        
        logger.info(f"Successfully processed {len(tools_for_analysis)} tools for {server_name}")
        
        # Analyze tools with Claude
        return await debug_mcp_server_connection(server_name, server_config, tools_for_analysis)
        
    except Exception as e:
        logger.error(f"Error in dynamic tool discovery for {server_name}: {str(e)}")
        return {
            "error": f"Dynamic tool discovery failed: {str(e)}",
            "server_name": server_name,
            "tools": [],
            "success": False
        }

async def _analyze_tools(server_name: str, server_config: MCPServerConfig, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze discovered tools using Claude LLM
    """
    
    try:
        logger.info(f"Starting Claude analysis for {len(tools)} tools from {server_name}")
        
        # Prepare tools info for analysis
        tools_info = []
        for tool in tools:
            tool_data = {
                "name": tool.get("name", "unknown"),
                "description": tool.get("description", "No description available"),
                "input_schema": tool.get("input_schema", {})
            }
            tools_info.append(tool_data)
        
        analysis_prompt = f"""
        You are analyzing MCP (Model Context Protocol) server tools that were dynamically discovered.

        Server Name: {server_name}
        Server Description: {server_config.description}
        Number of Tools Found: {len(tools_info)}

        Discovered Tools:
        {json.dumps(tools_info, indent=2)}

        For each tool, analyze and provide:
        1. Required and optional parameters (based on input_schema)
        2. Usage format and examples
        3. Practical use cases and benefits

        Respond with valid JSON matching this exact schema:

        {{
            "server_analysis": {{
                "server_name": "{server_name}",
                "total_tools": {len(tools_info)},
                "analysis_summary": "Comprehensive summary of server capabilities"
            }},
            "tools": [
                {{
                    "name": "tool name",
                    "description": "tool description", 
                    "parameters": {{
                        "required": ["param1", "param2"],
                        "optional": ["param3"],
                        "parameter_details": {{
                            "param1": "description and type",
                            "param2": "description and type"
                        }}
                    }},
                    "usage": {{
                        "format": "tool_name(param1=value1, param2=value2)",
                        "example": "concrete usage example"
                    }},
                    "usefulness": {{
                        "primary_use": "main purpose",
                        "use_cases": ["case1", "case2", "case3"],
                        "benefits": "why this tool is valuable"
                    }}
                }}
            ],
            "success": true
        }}

        CRITICAL: Respond ONLY with valid JSON. No other text.
        """

        # Get Claude client and make the analysis request
        client = get_client()
        
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ]
        )
        
        # Parse response
        response_text = response.content[0].text.strip()
        logger.debug(f"Claude response length: {len(response_text)}")
        
        try:
            analysis_data = json.loads(response_text)
            
            # Validate structure
            if not isinstance(analysis_data, dict):
                raise ValueError("Response is not a dictionary")
            
            if "server_analysis" not in analysis_data or "tools" not in analysis_data:
                raise ValueError("Missing required fields in response")
            
            # Add success flag
            analysis_data["success"] = True
            analysis_data["discovery_method"] = "dynamic"
            
            logger.info(f"Claude analysis completed successfully for {server_name}")
            return analysis_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Claude returned invalid JSON for {server_name}: {e}")
            logger.error(f"Raw response: {response_text[:300]}...")
            raise ValueError(f"Claude analysis failed - invalid JSON: {e}")
            
    except Exception as e:
        logger.error(f"Claude analysis failed for {server_name}: {str(e)}")
        raise