"""CrewAI agent definitions and role management."""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from ...core.logging import get_framework_logger
from ...core.errors import CrewAIError, AgentError
from ...core.monitoring import get_metrics_collector


class AgentRole(Enum):
    """Predefined agent roles for CrewAI agents."""
    
    # Research Agents
    WEB_RESEARCHER = "web_researcher"
    TREND_ANALYST = "trend_analyst"
    CONTENT_CURATOR = "content_curator"
    FACT_CHECKER = "fact_checker"
    
    # Strategy Agents
    CONTENT_STRATEGIST = "content_strategist"
    AUDIENCE_ANALYST = "audience_analyst"
    COMPETITIVE_ANALYST = "competitive_analyst"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    
    # Coordination Agents
    WORKFLOW_COORDINATOR = "workflow_coordinator"
    QUALITY_CONTROLLER = "quality_controller"
    RESOURCE_MANAGER = "resource_manager"
    COMMUNICATION_HUB = "communication_hub"


@dataclass
class AgentDefinition:
    """Definition for a CrewAI agent."""
    
    name: str
    role: AgentRole
    goal: str
    backstory: str
    tools: List[str] = field(default_factory=list)
    max_iter: int = 5
    max_execution_time: Optional[int] = None
    verbose: bool = True
    allow_delegation: bool = False
    system_template: Optional[str] = None
    prompt_template: Optional[str] = None
    response_template: Optional[str] = None
    
    def to_crewai_config(self) -> Dict[str, Any]:
        """Convert to CrewAI agent configuration."""
        config = {
            "role": self.name,
            "goal": self.goal,
            "backstory": self.backstory,
            "max_iter": self.max_iter,
            "verbose": self.verbose,
            "allow_delegation": self.allow_delegation
        }
        
        if self.max_execution_time:
            config["max_execution_time"] = self.max_execution_time
        
        if self.system_template:
            config["system_template"] = self.system_template
            
        return config


class AgentRegistry:
    """Registry for managing CrewAI agent definitions."""
    
    def __init__(self):
        self.logger = get_framework_logger("CrewAI")
        self.metrics = get_metrics_collector()
        self.agents: Dict[str, AgentDefinition] = {}
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize default agent definitions."""
        
        # Research Agents
        self.register_agent(AgentDefinition(
            name="Web Research Specialist",
            role=AgentRole.WEB_RESEARCHER,
            goal="Gather comprehensive information from web sources using advanced search techniques",
            backstory="You are an expert web researcher with years of experience in finding, evaluating, and synthesizing information from online sources. You excel at identifying authoritative sources, cross-referencing information, and presenting findings in a structured manner.",
            tools=["web_search", "content_retrieval"],
            max_iter=5,
            system_template="You are a professional web researcher. Focus on finding accurate, up-to-date information from reliable sources. Always verify information across multiple sources when possible."
        ))
        
        self.register_agent(AgentDefinition(
            name="Trend Analysis Expert",
            role=AgentRole.TREND_ANALYST,
            goal="Identify and analyze market trends, emerging patterns, and future opportunities",
            backstory="You are a seasoned trend analyst with expertise in market research, data analysis, and pattern recognition. You have a keen eye for spotting emerging trends and understanding their implications for content strategy.",
            tools=["trend_analysis", "news_search"],
            max_iter=4,
            system_template="You are a trend analysis expert. Focus on identifying patterns, analyzing data trends, and providing actionable insights about market directions and opportunities."
        ))
        
        self.register_agent(AgentDefinition(
            name="Content Curator",
            role=AgentRole.CONTENT_CURATOR,
            goal="Curate, organize, and synthesize relevant content from various sources",
            backstory="You are an experienced content curator with excellent organizational skills and a talent for identifying valuable content. You excel at structuring information, removing redundancy, and presenting content in logical, accessible formats.",
            tools=["content_analyzer", "topic_extractor"],
            max_iter=3,
            system_template="You are a content curation specialist. Focus on organizing information logically, removing duplicates, and presenting content in clear, structured formats."
        ))
        
        self.register_agent(AgentDefinition(
            name="Fact Checker",
            role=AgentRole.FACT_CHECKER,
            goal="Verify information accuracy, check sources, and ensure content credibility",
            backstory="You are a meticulous fact-checker with extensive experience in source verification and information validation. You have a methodical approach to verifying claims and a keen eye for identifying potential misinformation.",
            tools=["web_search", "content_analyzer"],
            max_iter=3,
            system_template="You are a professional fact-checker. Focus on verifying claims, checking source credibility, and ensuring information accuracy. Be thorough and methodical in your verification process."
        ))
        
        # Strategy Agents
        self.register_agent(AgentDefinition(
            name="Content Strategist",
            role=AgentRole.CONTENT_STRATEGIST,
            goal="Develop comprehensive content strategies aligned with business objectives",
            backstory="You are a strategic content planning expert with deep understanding of audience engagement, brand positioning, and content marketing. You excel at creating comprehensive content strategies that drive business results.",
            tools=["content_analyzer", "trend_analysis"],
            max_iter=4,
            system_template="You are a content strategy expert. Focus on aligning content with business goals, understanding audience needs, and creating actionable strategic recommendations."
        ))
        
        self.register_agent(AgentDefinition(
            name="Audience Analyst",
            role=AgentRole.AUDIENCE_ANALYST,
            goal="Analyze target audiences and provide insights for content optimization",
            backstory="You are an audience research specialist with expertise in demographic analysis, behavioral patterns, and content preferences. You excel at understanding what resonates with different audience segments.",
            tools=["sentiment_analyzer", "content_analyzer"],
            max_iter=3,
            system_template="You are an audience analysis expert. Focus on understanding audience behaviors, preferences, and engagement patterns to inform content decisions."
        ))
        
        self.register_agent(AgentDefinition(
            name="Competitive Analyst",
            role=AgentRole.COMPETITIVE_ANALYST,
            goal="Analyze competitive landscape and identify content opportunities",
            backstory="You are a competitive intelligence expert with experience in market analysis and competitive positioning. You excel at identifying gaps in the market and opportunities for differentiation.",
            tools=["web_search", "content_analyzer"],
            max_iter=4,
            system_template="You are a competitive analysis expert. Focus on understanding the competitive landscape, identifying opportunities, and recommending positioning strategies."
        ))
        
        self.register_agent(AgentDefinition(
            name="Performance Optimizer",
            role=AgentRole.PERFORMANCE_OPTIMIZER,
            goal="Optimize content for maximum performance and measurable results",
            backstory="You are a performance optimization expert with deep knowledge of content metrics, SEO, and conversion optimization. You excel at making data-driven recommendations for content improvement.",
            tools=["seo_optimizer", "content_analyzer"],
            max_iter=3,
            system_template="You are a performance optimization expert. Focus on measurable improvements, data-driven recommendations, and optimization strategies that deliver results."
        ))
        
        # Coordination Agents
        self.register_agent(AgentDefinition(
            name="Workflow Coordinator",
            role=AgentRole.WORKFLOW_COORDINATOR,
            goal="Coordinate complex multi-agent workflows and ensure smooth execution",
            backstory="You are an expert workflow coordinator with experience in project management and process optimization. You excel at orchestrating complex operations and ensuring all team members work effectively together.",
            tools=["content_analyzer"],
            max_iter=2,
            allow_delegation=True,
            system_template="You are a workflow coordination expert. Focus on process management, task delegation, and ensuring efficient collaboration between team members."
        ))
        
        self.register_agent(AgentDefinition(
            name="Quality Controller",
            role=AgentRole.QUALITY_CONTROLLER,
            goal="Ensure consistent quality standards across all content and processes",
            backstory="You are a quality assurance expert with meticulous attention to detail and high standards for content excellence. You excel at identifying quality issues and implementing improvement processes.",
            tools=["content_analyzer", "sentiment_analyzer"],
            max_iter=3,
            system_template="You are a quality control expert. Focus on maintaining high standards, identifying quality issues, and recommending improvements across all deliverables."
        ))
    
    def register_agent(self, agent: AgentDefinition):
        """Register a new agent definition."""
        agent_id = f"{agent.role.value}_{agent.name.lower().replace(' ', '_')}"
        self.agents[agent_id] = agent
        self.logger.info(f"Registered CrewAI agent: {agent.name} ({agent.role.value})")
        self.metrics.record_counter("agent_registered", framework="crewai", role=agent.role.value)
    
    def get_agent(self, agent_id: str) -> Optional[AgentDefinition]:
        """Get an agent definition by ID."""
        return self.agents.get(agent_id)
    
    def get_agents_by_role(self, role: AgentRole) -> List[AgentDefinition]:
        """Get all agents with a specific role."""
        return [agent for agent in self.agents.values() if agent.role == role]
    
    def list_agents(self) -> List[AgentDefinition]:
        """List all registered agents."""
        return list(self.agents.values())
    
    def create_agent_instance(self, agent_id: str, llm, tools: Optional[Dict[str, Any]] = None):
        """Create a CrewAI agent instance from definition."""
        agent_def = self.get_agent(agent_id)
        if not agent_def:
            raise AgentError(f"Agent definition not found: {agent_id}")
        
        try:
            from crewai import Agent
            
            # Filter available tools based on agent definition
            agent_tools = []
            if tools and agent_def.tools:
                for tool_name in agent_def.tools:
                    if tool_name in tools:
                        agent_tools.append(tools[tool_name])
                    else:
                        self.logger.warning(f"Tool {tool_name} not available for agent {agent_def.name}")
            
            # Create CrewAI agent
            agent = Agent(
                role=agent_def.name,
                goal=agent_def.goal,
                backstory=agent_def.backstory,
                tools=agent_tools,
                llm=llm,
                max_iter=agent_def.max_iter,
                verbose=agent_def.verbose,
                allow_delegation=agent_def.allow_delegation,
                max_execution_time=agent_def.max_execution_time
            )
            
            self.logger.info(f"Created CrewAI agent instance: {agent_def.name}")
            self.metrics.record_counter("agent_created", framework="crewai", role=agent_def.role.value)
            
            return agent
            
        except ImportError as e:
            raise CrewAIError(f"CrewAI not installed: {e}")
        except Exception as e:
            raise AgentError(f"Failed to create agent {agent_def.name}: {e}")
    
    def get_research_agents(self) -> List[AgentDefinition]:
        """Get all research-related agents."""
        research_roles = [
            AgentRole.WEB_RESEARCHER,
            AgentRole.TREND_ANALYST,
            AgentRole.CONTENT_CURATOR,
            AgentRole.FACT_CHECKER
        ]
        return [agent for agent in self.agents.values() if agent.role in research_roles]
    
    def get_strategy_agents(self) -> List[AgentDefinition]:
        """Get all strategy-related agents."""
        strategy_roles = [
            AgentRole.CONTENT_STRATEGIST,
            AgentRole.AUDIENCE_ANALYST,
            AgentRole.COMPETITIVE_ANALYST,
            AgentRole.PERFORMANCE_OPTIMIZER
        ]
        return [agent for agent in self.agents.values() if agent.role in strategy_roles]
    
    def get_coordination_agents(self) -> List[AgentDefinition]:
        """Get all coordination-related agents."""
        coordination_roles = [
            AgentRole.WORKFLOW_COORDINATOR,
            AgentRole.QUALITY_CONTROLLER,
            AgentRole.RESOURCE_MANAGER,
            AgentRole.COMMUNICATION_HUB
        ]
        return [agent for agent in self.agents.values() if agent.role in coordination_roles]


# Global agent registry instance
_agent_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry