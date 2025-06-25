from pydantic import BaseModel
from typing import Dict, Optional, List
from enum import Enum


class FactModel(BaseModel):
    facts: str


class PlanModel(BaseModel):
    plan: str


class LedgerAnswer(BaseModel):
    """Model for individual ledger answers"""

    answer: bool | str
    explanation: Optional[str] = None


class LedgerModel(BaseModel):
    """Main ledger state model"""

    is_request_satisfied: LedgerAnswer
    is_in_loop: LedgerAnswer
    is_progress_being_made: LedgerAnswer
    next_speaker: LedgerAnswer
    instruction_or_question: LedgerAnswer


class Action(str, Enum):
    enable = "enable"
    disable = "disable"


class MCPRequest(BaseModel):
    server_name: str
    action: Action
    server_secret: str


class MCPServerConfig(BaseModel):
    """Model for MCP server configuration"""

    command: str
    args: List[str]
    env: Dict[str, str]
    description: str
    status: str = "disabled"  # Default to disabled
    secret_key: str

    def has_valid_api_key(self) -> bool:
        """Check if the server has a valid API key in its environment variables"""
        return self.secret_key in self.env and bool(self.env[self.secret_key].strip())


class ToolParameters(BaseModel):
    """Model for tool parameters information"""
    required: List[str] = []
    optional: List[str] = []
    parameter_details: Dict[str, str] = {}


class ToolUsage(BaseModel):
    """Model for tool usage information"""
    format: str
    example: str


class ToolUsefulness(BaseModel):
    """Model for tool usefulness information"""
    primary_use: str
    use_cases: List[str] = []
    benefits: str


class AnalyzedTool(BaseModel):
    """Model for individual analyzed tool"""
    name: str
    description: str
    parameters: ToolParameters
    usage: ToolUsage
    usefulness: ToolUsefulness


class ServerAnalysis(BaseModel):
    """Model for server analysis summary"""
    server_name: str
    total_tools: int
    analysis_summary: str


class MCPToolAnalysis(BaseModel):
    """Model for complete MCP tool analysis"""
    server_analysis: ServerAnalysis
    tools: List[AnalyzedTool]
