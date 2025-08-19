"""
Strategy Agent Module - AutoGen Framework Implementation.

This module provides comprehensive content strategy development through collaborative
multi-agent discussions using the AutoGen framework.

Main Components:
- StrategyAgent: Main orchestrator using AutoGen GroupChat
- Strategy Models: Data structures and request/response schemas
- Tool Integration: Analysis tools for enhanced strategy insights

The Strategy Agent coordinates four specialized agents:
1. Content Strategist - Overall strategy and content planning
2. Audience Analyst - Audience research and targeting insights  
3. Competitive Analyst - Market positioning and differentiation
4. Performance Optimizer - Metrics, optimization, and ROI focus

Usage:
    from src.agents.strategy import get_strategy_agent, StrategyRequest
    
    agent = await get_strategy_agent()
    request = StrategyRequest(topic="AI in Healthcare", industry="Healthcare")
    response = await agent.develop_strategy(request)
"""

from .models import (
    # Enums
    StrategyType,
    ContentType, 
    AudienceSegment,
    CompetitivePosition,
    
    # Data Classes
    StrategyContext,
    ContentStrategistPerspective,
    AudienceAnalystPerspective,
    CompetitiveAnalystPerspective,
    PerformanceOptimizerPerspective,
    StrategyConsensus,
    StrategyResult,
    DiscussionMetrics,
    AgentContribution,
    StrategyWorkflowState,
    ToolAnalysisInput,
    ToolAnalysisResult,
    
    # Request/Response Models
    StrategyRequest,
    StrategyResponse,
    StrategySessionConfig
)

from .strategy_agent import (
    StrategyAgent,
    StrategyAgentConfig,
    get_strategy_agent,
    shutdown_strategy_agent
)

__all__ = [
    # Core Agent
    "StrategyAgent",
    "StrategyAgentConfig", 
    "get_strategy_agent",
    "shutdown_strategy_agent",
    
    # Data Models - Enums
    "StrategyType",
    "ContentType",
    "AudienceSegment", 
    "CompetitivePosition",
    
    # Data Models - Classes
    "StrategyContext",
    "ContentStrategistPerspective",
    "AudienceAnalystPerspective",
    "CompetitiveAnalystPerspective", 
    "PerformanceOptimizerPerspective",
    "StrategyConsensus",
    "StrategyResult",
    "DiscussionMetrics",
    "AgentContribution",
    "StrategyWorkflowState",
    "ToolAnalysisInput",
    "ToolAnalysisResult",
    
    # Request/Response
    "StrategyRequest",
    "StrategyResponse",
    "StrategySessionConfig"
]