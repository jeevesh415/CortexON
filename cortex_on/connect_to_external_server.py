import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import nest_asyncio
from colorama import Fore, Style, init
import logfire
from mcp import ClientSession, StdioServerParameters
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

init(autoreset=True)  # Initialize colorama with autoreset=True

from dotenv import load_dotenv
from pydantic_ai.mcp import MCPServerStdio

load_dotenv()  

class StdioServerProvider:
    """Class for creating and managing MCPServerStdio instances from a JSON configuration file"""
    
    def __init__(self, config_path: str = 'config/external_mcp_servers.json'):
        self.config_path = config_path
        self.servers: Dict[str, MCPServerStdio] = {}
        self.server_configs: Dict[str, Dict[str, Any]] = {}
        self.server_tools: Dict[str, List[Dict[str, Any]]] = {}
        self.server_status: Dict[str, Dict[str, Any]] = {}
        self.registered_tool_names: List[str] = []  # Track all registered tool names to ensure uniqueness
        self.active_servers: List[MCPServerStdio] = []  # Track currently active servers
        
    async def shutdown_servers(self):
        """Properly shut down all active servers"""
        for server in self.active_servers:
            try:
                if hasattr(server, 'close') and callable(server.close):
                    await server.close()
                elif hasattr(server, '__aexit__') and callable(server.__aexit__):
                    await server.__aexit__(None, None, None)
                logfire.info(f"Shut down MCP server: {server}")
            except Exception as e:
                print(f"Error shutting down server: {str(e)}")
        
        # Clear the active servers list
        self.active_servers = []
        self.servers = {}
        self.server_tools = {}
        print("All servers have been shut down")

    async def load_servers(self) -> Tuple[List[MCPServerStdio], str]:
        """Load server configurations from JSON and create MCPServerStdio instances"""
        # First shut down any existing servers
        await self.shutdown_servers()
        
        # Clear registered tool names before loading new servers
        self.registered_tool_names = []
        
        # Check if config file exists in the specified path or try to find it
        if not os.path.exists(self.config_path):
            # Try to find it relative to the current file
            alt_path = os.path.join(os.path.dirname(__file__), self.config_path)
            if os.path.exists(alt_path):
                self.config_path = alt_path
            else:
                # Try to find it in the parent directory
                parent_dir = os.path.dirname(os.path.dirname(__file__))
                alt_path = os.path.join(parent_dir, self.config_path)
                if os.path.exists(alt_path):
                    self.config_path = alt_path
                else:
                    raise FileNotFoundError(f"Could not find config file: {self.config_path}")
        
        print(f"Loading stdio server configuration from: {self.config_path}")
        
        # Load the configuration file
        with open(self.config_path, 'r') as f:
            self.server_configs = json.load(f)
        
        # Create MCPServerStdio instances for each server
        stdio_servers = []
        server_names = []
        
        for server_name, config in self.server_configs.items():
            # Skip servers without a command
            if 'command' not in config or not config['command']:
                print(f"Skipping {server_name} - no command specified")
                continue
            
            if config['status'] == "disabled":
                print(f"Skipping {server_name} - server is disabled")
                continue
            
            command = config['command']
            args = config.get('args', [])
            env = config.get('env', {})
            
            # Create the MCPServerStdio instance
            try:
                # Use namespaced tool names by setting the namespace parameter
                server = MCPServerStdio(
                    command,
                    args=args,
                    env=env
                )
                
                self.servers[server_name] = server
                stdio_servers.append(server)
                server_names.append(server_name)
                self.active_servers.append(server)  # Track in active servers list
                
                # Initialize server status
                self.server_status[server_name] = {
                    "status": "initializing",
                    "last_update": datetime.now().isoformat()
                }
                
                print(f"Created MCPServerStdio for {server_name} with command: {command} {' '.join(args)}")
            except Exception as e:
                print(f"Error creating MCPServerStdio for {server_name}: {str(e)}")
        
        # Wait for servers to initialize before attempting to discover tools
        await asyncio.sleep(2)
        
        # Try to discover tools from all servers
        for server_name in server_names:
            print(f"Attempting to discover tools for {server_name}...")
            await self.get_server_tools(server_name)
        
        # Generate a combined system prompt with server information
        system_prompt = self._generate_system_prompt(server_names)
        
        return stdio_servers, system_prompt
    
    def _generate_system_prompt(self, server_names: List[str]) -> str:
        """Generate a system prompt for the agent with information about available servers and their tools"""
        if not server_names:
            return "You are an AI assistant that can help with various tasks."
        
        servers_list = ", ".join([f"`{name}`" for name in server_names])
        
        prompt = f"""[EXTERNAL SERVER CAPABILITIES]
                """
        
        # Add details about each server and its tools if available
        for server_name in server_names:
            config = self.server_configs.get(server_name, {})
            description = config.get('description', f"MCP server for {server_name}")
            
            prompt += f"""- {server_name}:
                        Description: {description}
                        Usage scenarios:
                        """
            
            # Add general usage scenarios based on server description
            keywords = server_name.lower().split('-')
            if "github" in keywords:
                prompt += """    - Repository operations
                            - Code browsing and analysis
                            - Issue and PR management
                        """
            elif "google" in keywords and "maps" in keywords:
                prompt += """    - Geocoding and location services
                            - Directions and routing
                            - Place search and information
                        """
            else:
                # Generic description based on server name
                prompt += f"""    - {server_name.replace('-', ' ').title()} operations
                        """
            
            # Add tool information
            prompt += "  Available tools:\n"
            
            # Use any tools we've discovered
            tools = self.server_tools.get(server_name, [])
            if tools:
                for tool in tools:
                    tool_name = tool.get("name", f"{server_name}.unknown_tool")
                    tool_description = tool.get("description", "No description available")
                    prompt += f"    - {tool_name}: {tool_description}\n"
            else:
                # If no tools are discovered, provide generic information
                prompt += f"    - Various tools prefixed with '{server_name}.'\n"
            
        prompt += """
                [HOW TO USE EXTERNAL SERVERS]
                1. When a user's task requires capabilities from an external server:
                - Identify which server is most appropriate based on the task description
                - Use the server's tools directly with the server name prefix (e.g., {server}.tool_name)
                - Include all required parameters for the tool

                2. Server selection guidelines:
                """

        # Generate server selection guidelines
        server_selection = ""
        if server_names:
            server_selection = "When deciding which external server to use:\n"
            for i, server_name in enumerate(server_names, 1):
                config = self.server_configs.get(server_name, {})
                description = config.get('description', f"MCP server for {server_name}")
                server_title = server_name.replace('-', ' ').title()
                
                # Generate usage scenarios based on server type
                keywords = server_name.lower().split('-')
                # if "github" in keywords:
                #     usage = "repository operations, code analysis, issue management"
                # elif "google" in keywords and "maps" in keywords:
                #     usage = "location services, geocoding, directions, place search"
                # elif "weather" in keywords:
                #     usage = "weather information, forecasts, climate data"
                # elif "calendar" in keywords:
                #     usage = "scheduling, calendar management, event creation"
                # else:
                usage = f"{server_name.replace('-', ' ')} related operations"
                
                server_selection += f"{i + 2}. For {usage}: Use {server_name} server\n"
        else:
            server_selection = "No external servers currently configured. Use main server tools (coder_task, web_surfer_task) for all operations."

        prompt += server_selection
        
        prompt += """
                3. Important notes:
                - Always include the server name prefix with the tool name
                - Multiple servers can be used in the same task when needed
                - Provide detailed parameters based on the specific tool requirements
                        """
        
        return prompt

    def generate_dynamic_sections(self, server_names: List[str]) -> Dict[str, str]:
        """Generate content for each specific section that needs dynamic insertion using tool_analysis data"""
        
        # Always include main server tools
        main_server_tools = [
            {
                "name": "coder_task",
                "description": "Assigns coding tasks to the coder agent - handles code implementation and execution",
                "server": "main"
            },
            {
                "name": "web_surfer_task", 
                "description": "Assigns web surfing tasks to the web surfer agent - handles web browsing and authentication",
                "server": "main"
            }
        ]
        
        # Generate server selection guidelines using tool_analysis data
        server_selection_dynamic = ""
        if server_names:
            server_selection_dynamic = "When deciding which external server to use:\n"
            for i, server_name in enumerate(server_names, 1):
                config = self.server_configs.get(server_name, {})
                
                # Use tool_analysis data if available for more accurate descriptions
                if 'tool_analysis' in config and config['tool_analysis'].get('success'):
                    analysis = config['tool_analysis']
                    server_analysis = analysis.get('server_analysis', {})
                    analysis_summary = server_analysis.get('analysis_summary', f"MCP server for {server_name}")
                    
                    # Extract primary use cases from tool analysis
                    tools = analysis.get('tools', [])
                    if tools:
                        # Create usage description based on actual tool capabilities
                        primary_uses = set()
                        for tool in tools:
                            usefulness = tool.get('usefulness', {})
                            primary_use = usefulness.get('primary_use', '')
                            if primary_use:
                                primary_uses.add(primary_use.lower())
                        
                        if primary_uses:
                            usage_list = list(primary_uses)[:3]  # Take first 3 primary uses
                            usage = ", ".join(usage_list)
                        else:
                            usage = f"{server_name.replace('-', ' ')} operations"
                    else:
                        usage = f"{server_name.replace('-', ' ')} operations"
                        
                    server_selection_dynamic += f"{i + 2}. For {usage}: Use {server_name} server\n"
                    server_selection_dynamic += f"   - {analysis_summary}\n"
                else:
                    # Fallback to basic description if no tool_analysis
                    description = config.get('description', f"MCP server for {server_name}")
                    usage = f"{server_name.replace('-', ' ')} related operations"
                    server_selection_dynamic += f"{i + 2}. For {usage}: Use {server_name} server\n"
                    server_selection_dynamic += f"   - {description}\n"
        else:
            server_selection_dynamic = "No external servers currently configured. Use main server tools (coder_task, web_surfer_task) for all operations."
        
        # Generate available tools section
        available_tools = ""
        
        # First add main server tools
        for i, tool in enumerate(main_server_tools, 5):  # Start from 5 to continue after direct orchestrator tools
            available_tools += f"\n{i}. {tool['name']}:\n"
            available_tools += f"   - {tool['description']}\n"
            available_tools += f"   - Provided by {tool['server']} MCP server\n"
            available_tools += f"   - Use when plan requires {'coding' if 'coder' in tool['name'] else 'web browsing/authentication'} operations\n"
        
        # Then add external server tools using tool_analysis data
        tool_counter = 5 + len(main_server_tools)
        for server_name in server_names:
            config = self.server_configs.get(server_name, {})
            
            # Use tool_analysis data if available
            if 'tool_analysis' in config and config['tool_analysis'].get('success'):
                analysis = config['tool_analysis']
                analyzed_tools = analysis.get('tools', [])
                
                for tool in analyzed_tools:
                    tool_name = tool.get("name", f"{server_name}.unknown_tool")
                    tool_description = tool.get("description", "No description available")
                    usefulness = tool.get('usefulness', {})
                    primary_use = usefulness.get('primary_use', '')
                    use_cases = usefulness.get('use_cases', [])
                    parameters = tool.get('parameters', {})
                    usage = tool.get('usage', {})
                    
                    available_tools += f"\n{tool_counter}. {tool_name}:\n"
                    available_tools += f"   - Description: {tool_description}\n"
                    if primary_use:
                        available_tools += f"   - Primary use: {primary_use}\n"
                    if use_cases:
                        available_tools += f"   - Use cases: {', '.join(use_cases[:3])}\n"
                    
                    # Add parameter information
                    required_params = parameters.get('required', [])
                    optional_params = parameters.get('optional', [])
                    param_details = parameters.get('parameter_details', {})
                    
                    if required_params or optional_params:
                        available_tools += f"   - Parameters:\n"
                        if required_params:
                            available_tools += f"     * Required: {', '.join(required_params)}\n"
                        if optional_params:
                            available_tools += f"     * Optional: {', '.join(optional_params)}\n"
                        
                        # Add parameter details for key parameters
                        if param_details:
                            key_params = list(param_details.keys())[:2]  # Show first 2 parameter details
                            for param in key_params:
                                if param in param_details:
                                    available_tools += f"     * {param}: {param_details[param]}\n"
                    
                    # Add usage information
                    usage_format = usage.get('format', '')
                    usage_example = usage.get('example', '')
                    if usage_format:
                        available_tools += f"   - Usage format: {usage_format}\n"
                    if usage_example:
                        available_tools += f"   - Example: {usage_example}\n"
                    
                    available_tools += f"   - Provided by {server_name} server\n"
                    available_tools += f"   - Use when plan requires {server_name.replace('-', ' ')} operations\n"
                    tool_counter += 1
            else:
                # Fallback to basic discovery method
                tools = self.server_tools.get(server_name, [])
                if tools:
                    for tool in tools:
                        tool_name = tool.get("name", f"{server_name}.unknown_tool")
                        tool_description = tool.get("description", "No description available")
                        available_tools += f"\n{tool_counter}. {tool_name}:\n"
                        available_tools += f"   - {tool_description}\n"
                        available_tools += f"   - Provided by {server_name} server\n"
                        available_tools += f"   - Use when plan requires {server_name.replace('-', ' ')} operations\n"
                        tool_counter += 1
        
        # Generate servers with tools section
        servers_with_tools = ""
        
        # First add main server
        servers_with_tools += "\nI. MAIN MCP SERVER:\n"
        servers_with_tools += "   Description: Core MCP server providing coding and web browsing capabilities\n"
        for i, tool in enumerate(main_server_tools, 1):
            servers_with_tools += f"   {i}. {tool['name']}:\n"
            servers_with_tools += f"      - {tool['description']}\n"
            servers_with_tools += f"      - Updates UI with {tool['server']} server operation progress\n"
        servers_with_tools += "\n"
        
        # Then add external servers using tool_analysis data
        for i, server_name in enumerate(server_names, 1):
            config = self.server_configs.get(server_name, {})
            
            servers_with_tools += f"{i + 1}. {server_name.upper()} SERVER:\n"
            
            # Use tool_analysis data if available for better descriptions
            if 'tool_analysis' in config and config['tool_analysis'].get('success'):
                analysis = config['tool_analysis']
                server_analysis = analysis.get('server_analysis', {})
                analysis_summary = server_analysis.get('analysis_summary', f"MCP server for {server_name}")
                total_tools = server_analysis.get('total_tools', 0)
                
                servers_with_tools += f"   Description: {analysis_summary}\n"
                servers_with_tools += f"   Total Tools: {total_tools}\n"
                
                analyzed_tools = analysis.get('tools', [])
                if analyzed_tools:
                    for j, tool in enumerate(analyzed_tools, 1):
                        tool_name = tool.get("name", f"{server_name}.unknown_tool")
                        tool_description = tool.get("description", "No description available")
                        usefulness = tool.get('usefulness', {})
                        primary_use = usefulness.get('primary_use', '')
                        use_cases = usefulness.get('use_cases', [])
                        parameters = tool.get('parameters', {})
                        usage = tool.get('usage', {})
                        
                        servers_with_tools += f"   {j}. {tool_name}:\n"
                        servers_with_tools += f"      - Description: {tool_description}\n"
                        if primary_use:
                            servers_with_tools += f"      - Primary use: {primary_use}\n"
                        if use_cases:
                            servers_with_tools += f"      - Use cases: {', '.join(use_cases[:2])}\n"
                        
                        # Add parameter summary
                        required_params = parameters.get('required', [])
                        optional_params = parameters.get('optional', [])
                        if required_params:
                            servers_with_tools += f"      - Required params: {', '.join(required_params)}\n"
                        if optional_params:
                            servers_with_tools += f"      - Optional params: {', '.join(optional_params[:2])}\n"
                        
                        # Add usage example
                        usage_example = usage.get('example', '')
                        if usage_example:
                            servers_with_tools += f"      - Example: {usage_example}\n"
                        
                        servers_with_tools += f"      - Updates UI with {server_name} operation progress\n"
                else:
                    servers_with_tools += f"   - Various tools prefixed with '{server_name}.'\n"
            else:
                # Fallback to basic description
                description = config.get('description', f"MCP server for {server_name}")
                servers_with_tools += f"   Description: {description}\n"
                
                tools = self.server_tools.get(server_name, [])
                if tools:
                    for j, tool in enumerate(tools, 1):
                        tool_name = tool.get("name", f"{server_name}.unknown_tool")
                        tool_description = tool.get("description", "No description available")
                        servers_with_tools += f"   {j}. {tool_name}:\n"
                        servers_with_tools += f"      - {tool_description}\n"
                        servers_with_tools += f"      - Updates UI with {server_name} operation progress\n"
                else:
                    servers_with_tools += f"   - Various tools prefixed with '{server_name}.'\n"
            servers_with_tools += "\n"
        
        # Generate external MCP server tools guidelines
        external_tools_guidelines = ""
        
        if server_names:
            external_tools_guidelines = "Additional specialized tools available from external MCP servers:\n"
            
            external_tools_guidelines += "\n[CRITICAL: STATUS REPORTING REQUIREMENT]\n"
            external_tools_guidelines += "For all external server operations, you MUST use server_status_update:\n"
            external_tools_guidelines += "1. BEFORE calling tool: server_status_update(server_name, 'Connecting...', 10)\n"
            external_tools_guidelines += "2. BEFORE calling tool: server_status_update(server_name, 'Preparing request...', 30)\n"
            external_tools_guidelines += "3. AFTER calling tool: server_status_update(server_name, 'Processing response...', 80)\n"
            external_tools_guidelines += "4. AFTER calling tool: server_status_update(server_name, 'Operation completed', 100)\n\n"
            
            for server_name in server_names:
                config = self.server_configs.get(server_name, {})
                
                external_tools_guidelines += f"- {server_name} Server:\n"
                
                # Use tool_analysis data if available for comprehensive guidelines
                if 'tool_analysis' in config and config['tool_analysis'].get('success'):
                    analysis = config['tool_analysis']
                    server_analysis = analysis.get('server_analysis', {})
                    analysis_summary = server_analysis.get('analysis_summary', f"MCP server for {server_name}")
                    total_tools = server_analysis.get('total_tools', 0)
                    
                    external_tools_guidelines += f"  * Description: {analysis_summary}\n"
                    external_tools_guidelines += f"  * Total Available Tools: {total_tools}\n"
                    
                    # Extract and list detailed tool information from analysis
                    analyzed_tools = analysis.get('tools', [])
                    if analyzed_tools:
                        external_tools_guidelines += f"  * Available tools ({len(analyzed_tools)}):\n"
                        
                        for tool in analyzed_tools[:3]:  # Show details for first 3 tools to avoid overwhelming
                            tool_name = tool.get('name', '').replace(f'{server_name}.', '')
                            tool_description = tool.get('description', '')
                            usefulness = tool.get('usefulness', {})
                            primary_use = usefulness.get('primary_use', '')
                            parameters = tool.get('parameters', {})
                            usage = tool.get('usage', {})
                            
                            external_tools_guidelines += f"    - {tool_name}: {tool_description}\n"
                            if primary_use:
                                external_tools_guidelines += f"      • Primary use: {primary_use}\n"
                            
                            # Add parameter info
                            required_params = parameters.get('required', [])
                            if required_params:
                                external_tools_guidelines += f"      • Required: {', '.join(required_params)}\n"
                            
                            # Add usage example
                            usage_example = usage.get('example', '')
                            if usage_example:
                                external_tools_guidelines += f"      • Example: {usage_example}\n"
                        
                        # Show remaining tool names if there are more than 3
                        if len(analyzed_tools) > 3:
                            remaining_tools = [tool.get('name', '').replace(f'{server_name}.', '') for tool in analyzed_tools[3:]]
                            external_tools_guidelines += f"    - Additional tools: {', '.join(remaining_tools)}\n"
                        
                        # Add summary of primary capabilities
                        primary_uses = set()
                        for tool in analyzed_tools:
                            usefulness = tool.get('usefulness', {})
                            primary_use = usefulness.get('primary_use', '')
                            if primary_use:
                                primary_uses.add(primary_use)
                        
                        if primary_uses:
                            external_tools_guidelines += f"  * Primary capabilities: {', '.join(list(primary_uses)[:3])}\n"
                else:
                    # Fallback to basic description
                    description = config.get('description', f"MCP server for {server_name}")
                    external_tools_guidelines += f"  * Description: {description}\n"
                    
                    tools = self.server_tools.get(server_name, [])
                    if tools:
                        external_tools_guidelines += f"  * Available tools: {', '.join([tool.get('name', '') for tool in tools])}\n"
                
                external_tools_guidelines += f"  * Use when plan specifies {server_name.replace('-', ' ')} operations\n"
                external_tools_guidelines += f"  * REQUIRED: Use server_status_update before and after {server_name} tool calls\n"
                external_tools_guidelines += f"  * Tool format: {server_name}.tool_name\n"
                external_tools_guidelines += "\n"
            
            external_tools_guidelines += """Key guidelines for external server usage:
- CRITICAL: Always use server_status_update before and after external tool calls
- This creates step-by-step progress reporting similar to planner agent
- Users see real-time feedback showing connection, execution, and completion steps
- Each external server provides domain-specific tools
- Follow plan guidance for when to use external server tools
- All external server tools require proper server name prefixes
- Monitor server status and provide user feedback during operations"""
        else:
            external_tools_guidelines = """No external MCP servers currently configured.
Only the main MCP server tools (coder_task, web_surfer_task) are available.
External servers can be added by configuring them in the external_mcp_servers.json file."""
        
        return {
            "server_selection_guidelines": server_selection_dynamic.strip(),
            "available_tools": available_tools.strip(),
            "servers_available_to_you_with_list_of_their_tools": servers_with_tools.strip(),
            "external_mcp_server_tools": external_tools_guidelines.strip()
        }

    def generate_planner_sections(self, server_names: List[str]) -> Dict[str, str]:
        """Generate content for planner agent specific sections"""
        
        # Generate external MCP servers information for planner
        external_servers_info = ""
        external_task_formats = ""
        dynamic_server_selection_rules = ""
        
        if server_names:
            external_servers_info = "EXTERNAL MCP SERVERS (dynamically available):\n"
            external_task_formats = "\nFor external servers, use these formats:\n"
            dynamic_server_selection_rules = ""
            
            for server_name in server_names:
                config = self.server_configs.get(server_name, {})
                
                # Use tool_analysis data if available for comprehensive planner information
                if 'tool_analysis' in config and config['tool_analysis'].get('success'):
                    analysis = config['tool_analysis']
                    server_analysis = analysis.get('server_analysis', {})
                    analysis_summary = server_analysis.get('analysis_summary', f"MCP server for {server_name}")
                    total_tools = server_analysis.get('total_tools', 0)
                    
                    # Generate dynamic server selection rules based on actual capabilities
                    analyzed_tools = analysis.get('tools', [])
                    if analyzed_tools:
                        # Extract primary uses to create specific selection rules
                        primary_uses = []
                        use_cases = []
                        for tool in analyzed_tools:
                            usefulness = tool.get('usefulness', {})
                            primary_use = usefulness.get('primary_use', '')
                            tool_use_cases = usefulness.get('use_cases', [])
                            if primary_use:
                                primary_uses.append(primary_use.lower())
                            use_cases.extend(tool_use_cases)
                        
                        # Create specific selection rules based on actual capabilities
                        unique_uses = list(set(primary_uses + [case.lower() for case in use_cases]))
                        capability_keywords = ' '.join(unique_uses)
                        
                        if any(keyword in capability_keywords for keyword in ['geocoding', 'location', 'navigation', 'mapping', 'directions']):
                            dynamic_server_selection_rules += f"- Location/geocoding/mapping/navigation/directions tasks → USE {server_name} server (NOT web_surfer_agent)\n"
                        elif any(keyword in capability_keywords for keyword in ['repository', 'code', 'github', 'git']):
                            dynamic_server_selection_rules += f"- Repository/code analysis/GitHub tasks → USE {server_name} server (NOT web_surfer_agent)\n"
                        elif any(keyword in capability_keywords for keyword in ['vector', 'database', 'index', 'search']):
                            dynamic_server_selection_rules += f"- Vector database/index management/search tasks → USE {server_name} server (NOT web_surfer_agent)\n"
                        else:
                            # Use first few primary uses for the rule
                            if primary_uses:
                                rule_uses = '/'.join(primary_uses[:3])
                                dynamic_server_selection_rules += f"- {rule_uses.title()} tasks → USE {server_name} server (NOT web_surfer_agent)\n"
                    
                    # Add comprehensive server info using analysis
                    external_servers_info += f"\n{server_name} server:\n"
                    external_servers_info += f"- Description: {analysis_summary}\n"
                    external_servers_info += f"- Total Tools: {total_tools}\n"
                    
                    # Extract capabilities from tool analysis
                    if analyzed_tools:
                        capabilities = set()
                        use_for_items = set()
                        tool_names = []
                        
                        for tool in analyzed_tools:
                            tool_names.append(tool.get('name', '').replace(f'{server_name}.', ''))
                            usefulness = tool.get('usefulness', {})
                            primary_use = usefulness.get('primary_use', '')
                            if primary_use:
                                capabilities.add(primary_use)
                            
                            # Extract use cases
                            tool_use_cases = usefulness.get('use_cases', [])
                            for case in tool_use_cases[:2]:  # Take first 2 use cases per tool
                                use_for_items.add(case)
                        
                        if capabilities:
                            external_servers_info += f"- Capabilities: {', '.join(list(capabilities))}\n"
                        if use_for_items:
                            external_servers_info += f"- Use for: {', '.join(list(use_for_items)[:4])}\n"  # Limit to 4 items
                        if tool_names:
                            external_servers_info += f"- Available tools: {', '.join(tool_names)}\n"
                        
                        external_servers_info += "- Tool Details:\n"
                        for tool in analyzed_tools:
                            tool_name = tool.get('name', '').replace(f'{server_name}.', '')
                            tool_description = tool.get('description', '')
                            usefulness = tool.get('usefulness', {})
                            primary_use = usefulness.get('primary_use', '')
                            parameters = tool.get('parameters', {})
                            usage = tool.get('usage', {})
                            
                            external_servers_info += f"  • {tool_name}: {tool_description}\n"
                            if primary_use:
                                external_servers_info += f"    - Primary use: {primary_use}\n"
                            
                            required_params = parameters.get('required', [])
                            if required_params:
                                external_servers_info += f"    - Required: {', '.join(required_params)}\n"
                            
                            usage_format = usage.get('format', '')
                            if usage_format:
                                external_servers_info += f"    - Format: {usage_format}\n"
                    
                    # Create task format based on actual capabilities
                    if analyzed_tools:
                        primary_uses = [tool.get('usefulness', {}).get('primary_use', '') for tool in analyzed_tools]
                        main_capabilities = [use for use in primary_uses if use][:3]  # Take first 3
                        if main_capabilities:
                            capabilities_str = ', '.join(main_capabilities).lower()
                            external_task_formats += f"- [ ] Task description ({server_name} server) - for {capabilities_str}\n"
                        else:
                            external_task_formats += f"- [ ] Task description ({server_name} server) - for {server_name.replace('-', ' ')} operations\n"
                    else:
                        external_task_formats += f"- [ ] Task description ({server_name} server) - for {server_name.replace('-', ' ')} operations\n"
                else:
                    # Fallback to basic description
                    description = config.get('description', f"MCP server for {server_name}")
                    
                    # Generate basic selection rules
                    description_lower = description.lower()
                    if any(keyword in description_lower for keyword in ['github', 'repository', 'code analysis']):
                        dynamic_server_selection_rules += f"- GitHub/repository/code analysis tasks → USE {server_name} server (NOT web_surfer_agent)\n"
                    elif any(keyword in description_lower for keyword in ['google', 'maps', 'location', 'geocoding']):
                        dynamic_server_selection_rules += f"- Location/geocoding/mapping/navigation tasks → USE {server_name} server (NOT web_surfer_agent)\n"
                    elif any(keyword in description_lower for keyword in ['pinecone', 'vector', 'database', 'index']):
                        dynamic_server_selection_rules += f"- Vector database/index management/Pinecone tasks → USE {server_name} server (NOT web_surfer_agent)\n"
                    else:
                        server_title = server_name.replace('-', ' ').title()
                        dynamic_server_selection_rules += f"- {server_title} related tasks → USE {server_name} server (NOT web_surfer_agent)\n"
                    
                    # Add basic server info
                    external_servers_info += f"\n{server_name} server:\n"
                    external_servers_info += f"- Description: {description}\n"
                    
                    # Add basic capabilities
                    keywords = server_name.lower().split('-')
                    if "google" in keywords and "maps" in keywords:
                        external_servers_info += "- Capabilities: Location services, geocoding, directions, place search\n"
                        external_servers_info += "- Use for: Location-based tasks, mapping, navigation\n"
                        external_task_formats += f"- [ ] Task description ({server_name} server) - for location services, geocoding, navigation\n"
                    else:
                        external_servers_info += f"- Capabilities: {server_name.replace('-', ' ').title()} operations\n"
                        external_servers_info += f"- Use for: {server_name.replace('-', ' ')} related tasks\n"
                        external_task_formats += f"- [ ] Task description ({server_name} server) - for {server_name.replace('-', ' ')} operations\n"
                    
                    # Add basic tools if available
                    tools = self.server_tools.get(server_name, [])
                    if tools:
                        external_servers_info += f"- Available tools: {', '.join([tool.get('name', '') for tool in tools])}\n"
        else:
            external_servers_info = "EXTERNAL MCP SERVERS: None currently configured."
            external_task_formats = "\nNo external servers available. Use only Main MCP Server assignments."
            dynamic_server_selection_rules = "No external servers configured."
        
        return {
            "external_mcp_servers": external_servers_info.strip(),
            "external_server_task_formats": external_task_formats.strip(),
            "dynamic_server_selection_rules": dynamic_server_selection_rules.strip()
        }

    async def get_server_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """Fetch tool information from a running server by introspecting its capabilities"""
        if server_name not in self.servers:
            return []
            
        try:
            server = self.servers[server_name]
            tools = []
            
            # Access the private _mcp_api property to get tool information
            # Note: This is implementation-specific and might need adjustment
            # based on the actual MCPServerStdio implementation
            if hasattr(server, '_mcp_api') and server._mcp_api:
                api = server._mcp_api
                
                # Check if the API has a tool manager
                if hasattr(api, '_tool_manager'):
                    tool_manager = api._tool_manager
                    
                    # Get tools from the tool manager
                    if hasattr(tool_manager, '_tools') and tool_manager._tools:
                        for tool_name, tool_info in tool_manager._tools.items():
                            # Ensure tool name has server namespace prefix
                            if not tool_name.startswith(f"{server_name}."):
                                prefixed_name = f"{server_name}.{tool_name}"
                            else:
                                prefixed_name = tool_name
                            
                            # Check if this tool name is already registered
                            if prefixed_name in self.registered_tool_names:
                                # Make the name unique by adding a suffix
                                base_name = prefixed_name
                                suffix = 1
                                while f"{base_name}_{suffix}" in self.registered_tool_names:
                                    suffix += 1
                                prefixed_name = f"{base_name}_{suffix}"
                                
                            # Add to the registered names list
                            self.registered_tool_names.append(prefixed_name)
                                
                            # Extract description if available
                            description = "No description available"
                            if hasattr(tool_info, 'description') and tool_info.description:
                                description = tool_info.description
                            elif hasattr(tool_info, '__doc__') and tool_info.__doc__:
                                description = tool_info.__doc__.strip()
                                
                            tools.append({
                                "name": prefixed_name,
                                "description": description
                            })
            
            # If we couldn't get tools through introspection, try to get schema
            if not tools and hasattr(server, 'get_schema'):
                try:
                    # Some MCP servers might have a get_schema method
                    schema = await server.get_schema()
                    if schema and 'tools' in schema:
                        for tool in schema['tools']:
                            name = tool.get('name', '')
                            # Ensure tool name has server namespace prefix
                            if not name.startswith(f"{server_name}."):
                                prefixed_name = f"{server_name}.{name}"
                            else:
                                prefixed_name = name
                                
                            # Check if this tool name is already registered
                            if prefixed_name in self.registered_tool_names:
                                # Make the name unique by adding a suffix
                                base_name = prefixed_name
                                suffix = 1
                                while f"{base_name}_{suffix}" in self.registered_tool_names:
                                    suffix += 1
                                prefixed_name = f"{base_name}_{suffix}"
                                
                            # Add to the registered names list
                            self.registered_tool_names.append(prefixed_name)
                                
                            tools.append({
                                "name": prefixed_name,
                                "description": tool.get('description', f"Tool from {server_name}")
                            })
                except Exception as schema_err:
                    print(f"Error getting schema from {server_name}: {str(schema_err)}")
            
            # Fallback for when we can't directly access the tool information:
            # We'll use some typical tools based on server name to provide useful information
            if not tools:
                fallback_tools = []
                
                if server_name == "github":
                    fallback_tools = [
                        {"base_name": "search_repositories", "description": "Search for GitHub repositories"},
                        {"base_name": "get_repository", "description": "Get details about a specific repository"},
                        {"base_name": "list_issues", "description": "List issues for a repository"},
                        {"base_name": "create_issue", "description": "Create a new issue in a repository"},
                        {"base_name": "search_code", "description": "Search for code within repositories"}
                    ]
                elif server_name == "google-maps":
                    fallback_tools = [
                        {"base_name": "geocode", "description": "Convert addresses to geographic coordinates"},
                        {"base_name": "directions", "description": "Get directions between locations"},
                        {"base_name": "places", "description": "Search for places near a location"},
                        {"base_name": "distance_matrix", "description": "Calculate distance and travel time"}
                    ]
                else:
                    fallback_tools = [
                        {"base_name": "use", "description": f"Use the {server_name} service"}
                    ]
                    
                # Process the fallback tools with unique naming
                for tool_info in fallback_tools:
                    base_name = tool_info["base_name"]
                    prefixed_name = f"{server_name}.{base_name}"
                    
                    # Check if this tool name is already registered
                    if prefixed_name in self.registered_tool_names:
                        # Make the name unique by adding a suffix
                        suffix = 1
                        while f"{prefixed_name}_{suffix}" in self.registered_tool_names:
                            suffix += 1
                        prefixed_name = f"{prefixed_name}_{suffix}"
                    
                    # Add to the registered names list
                    self.registered_tool_names.append(prefixed_name)
                    
                    tools.append({
                        "name": prefixed_name,
                        "description": tool_info["description"]
                    })
                    
                print(f"Using fallback tool definitions for {server_name} - actual tools couldn't be discovered")
                
            # Store and return the tools
            self.server_tools[server_name] = tools
            print(f"Discovered {len(tools)} tools for {server_name}")
            return tools
        except Exception as e:
            print(f"Error discovering tools from {server_name}: {str(e)}")
            return []

    async def monitor_server_status(self, server_name: str, callback: callable) -> None:
        """
        Set up monitoring for a server's status and call the callback with updates
        
        Args:
            server_name: The name of the server to monitor
            callback: An async function to call with status updates (takes server_name and status dict)
        """
        if server_name not in self.servers:
            return
            
        try:
            # Set initial status
            status = {
                "status": "monitoring",
                "progress": 75,
                "last_update": datetime.now().isoformat()
            }
            
            # Call the callback with initial status
            try:
                await callback(server_name, status)
            except Exception as cb_err:
                logfire.error(f"Error calling status callback: {str(cb_err)}")
                
            # Check server health periodically
            check_count = 0
            while server_name in self.servers:
                check_count += 1
                
                # Get the server
                server = self.servers[server_name]
                is_healthy = False
                
                # Try to determine if the server is healthy
                try:
                    if hasattr(server, '_mcp_api') and server._mcp_api:
                        # We'll consider it healthy if it has an API
                        is_healthy = True
                        
                        # Get more detailed health info if available
                        if hasattr(server._mcp_api, 'health') and callable(server._mcp_api.health):
                            health_info = await server._mcp_api.health()
                            if isinstance(health_info, dict):
                                status.update(health_info)
                except Exception:
                    is_healthy = False
                
                # Update the status based on health check
                if is_healthy:
                    status = {
                        "status": "running",
                        "progress": 100,
                        "health": "ok",
                        "last_update": datetime.now().isoformat(),
                        "check_count": check_count
                    }
                else:
                    status = {
                        "status": "degraded",
                        "progress": 80,
                        "health": "degraded",
                        "last_update": datetime.now().isoformat(),
                        "check_count": check_count
                    }
                
                # Call the callback with the status
                try:
                    await callback(server_name, status)
                except Exception as cb_err:
                    logfire.error(f"Error calling status callback: {str(cb_err)}")
                
                # Wait before the next check
                await asyncio.sleep(5)  # Check every 5 seconds
                
        except Exception as e:
            logfire.error(f"Error monitoring server {server_name}: {str(e)}")
            
            # Try to send a final error status
            try:
                status = {
                    "status": "error",
                    "progress": 0,
                    "error": str(e),
                    "last_update": datetime.now().isoformat()
                }
                await callback(server_name, status)
            except Exception:
                pass

server_provider = StdioServerProvider()