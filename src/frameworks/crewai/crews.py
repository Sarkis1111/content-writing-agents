"""CrewAI crew configuration templates and management."""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from ...core.logging import get_framework_logger
from ...core.errors import CrewAIError, WorkflowError
from ...core.monitoring import get_metrics_collector
from .agents import AgentRegistry, AgentDefinition, AgentRole, get_agent_registry


class ProcessType(Enum):
    """CrewAI process types."""
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"


@dataclass
class TaskDefinition:
    """Definition for a CrewAI task."""
    
    description: str
    agent_role: AgentRole
    expected_output: str
    context: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    async_execution: bool = False
    
    def to_crewai_config(self) -> Dict[str, Any]:
        """Convert to CrewAI task configuration."""
        config = {
            "description": self.description,
            "expected_output": self.expected_output,
            "async_execution": self.async_execution
        }
        
        if self.context:
            config["context"] = self.context
            
        return config


@dataclass
class CrewDefinition:
    """Definition for a CrewAI crew."""
    
    name: str
    description: str
    agents: List[AgentRole]
    tasks: List[TaskDefinition]
    process: ProcessType = ProcessType.SEQUENTIAL
    verbose: bool = True
    memory: bool = True
    max_execution_time: Optional[int] = None
    max_rpm: Optional[int] = None
    planning: bool = False
    
    def validate(self):
        """Validate crew definition."""
        if not self.agents:
            raise CrewAIError(f"Crew {self.name} must have at least one agent")
        
        if not self.tasks:
            raise CrewAIError(f"Crew {self.name} must have at least one task")
        
        # Validate that all task agents are in crew
        crew_roles = set(self.agents)
        task_roles = set(task.agent_role for task in self.tasks)
        missing_roles = task_roles - crew_roles
        
        if missing_roles:
            raise CrewAIError(
                f"Crew {self.name} missing agents for roles: {missing_roles}"
            )


class CrewTemplateRegistry:
    """Registry for managing CrewAI crew templates."""
    
    def __init__(self):
        self.logger = get_framework_logger("CrewAI")
        self.metrics = get_metrics_collector()
        self.agent_registry = get_agent_registry()
        self.crews: Dict[str, CrewDefinition] = {}
        self._initialize_default_crews()
    
    def _initialize_default_crews(self):
        """Initialize default crew templates."""
        
        # Research Crew
        research_crew = CrewDefinition(
            name="Research Crew",
            description="Comprehensive research and information gathering crew",
            agents=[
                AgentRole.WEB_RESEARCHER,
                AgentRole.TREND_ANALYST,
                AgentRole.CONTENT_CURATOR,
                AgentRole.FACT_CHECKER
            ],
            tasks=[
                TaskDefinition(
                    description="Research {topic} comprehensively using multiple web sources",
                    agent_role=AgentRole.WEB_RESEARCHER,
                    expected_output="Detailed research report with sources and key findings",
                    tools=["web_search", "content_retrieval"]
                ),
                TaskDefinition(
                    description="Analyze trends related to {topic}",
                    agent_role=AgentRole.TREND_ANALYST,
                    expected_output="Trend analysis with supporting data and insights",
                    tools=["trend_analysis", "news_search"],
                    context=["research_report"]
                ),
                TaskDefinition(
                    description="Curate and organize research findings",
                    agent_role=AgentRole.CONTENT_CURATOR,
                    expected_output="Organized content collection with categorized information",
                    tools=["content_analyzer", "topic_extractor"],
                    context=["research_report", "trend_analysis"]
                ),
                TaskDefinition(
                    description="Verify key facts and claims",
                    agent_role=AgentRole.FACT_CHECKER,
                    expected_output="Fact-checked research validation with credibility assessment",
                    tools=["web_search", "content_analyzer"],
                    context=["research_report", "organized_content"]
                )
            ],
            process=ProcessType.HIERARCHICAL,
            memory=True,
            max_execution_time=600
        )
        self.register_crew(research_crew)
        
        # Strategy Council Crew
        strategy_crew = CrewDefinition(
            name="Strategy Council",
            description="Strategic planning and content strategy development crew",
            agents=[
                AgentRole.CONTENT_STRATEGIST,
                AgentRole.AUDIENCE_ANALYST,
                AgentRole.COMPETITIVE_ANALYST,
                AgentRole.PERFORMANCE_OPTIMIZER
            ],
            tasks=[
                TaskDefinition(
                    description="Develop comprehensive content strategy for {topic} based on research",
                    agent_role=AgentRole.CONTENT_STRATEGIST,
                    expected_output="Content strategy document with objectives and approach",
                    tools=["content_analyzer", "trend_analysis"]
                ),
                TaskDefinition(
                    description="Analyze target audience for {topic} content",
                    agent_role=AgentRole.AUDIENCE_ANALYST,
                    expected_output="Audience analysis with demographics, preferences, and engagement insights",
                    tools=["sentiment_analyzer", "content_analyzer"],
                    context=["content_strategy"]
                ),
                TaskDefinition(
                    description="Analyze competitive landscape for {topic}",
                    agent_role=AgentRole.COMPETITIVE_ANALYST,
                    expected_output="Competitive analysis with opportunities and positioning recommendations",
                    tools=["web_search", "content_analyzer"],
                    context=["content_strategy", "audience_analysis"]
                ),
                TaskDefinition(
                    description="Optimize strategy for performance and measurable results",
                    agent_role=AgentRole.PERFORMANCE_OPTIMIZER,
                    expected_output="Performance optimization recommendations with KPIs and metrics",
                    tools=["seo_optimizer", "content_analyzer"],
                    context=["content_strategy", "audience_analysis", "competitive_analysis"]
                )
            ],
            process=ProcessType.SEQUENTIAL,
            memory=True,
            max_execution_time=400,
            planning=True
        )
        self.register_crew(strategy_crew)
        
        # Meta Coordination Crew
        meta_crew = CrewDefinition(
            name="Meta Coordination Crew",
            description="Cross-framework workflow coordination and quality management",
            agents=[
                AgentRole.WORKFLOW_COORDINATOR,
                AgentRole.QUALITY_CONTROLLER
            ],
            tasks=[
                TaskDefinition(
                    description="Coordinate cross-framework workflow for {workflow_type}",
                    agent_role=AgentRole.WORKFLOW_COORDINATOR,
                    expected_output="Workflow coordination plan with task assignments and timelines",
                    tools=["content_analyzer"]
                ),
                TaskDefinition(
                    description="Ensure quality standards across all workflow outputs",
                    agent_role=AgentRole.QUALITY_CONTROLLER,
                    expected_output="Quality assessment report with recommendations",
                    tools=["content_analyzer", "sentiment_analyzer"],
                    context=["workflow_plan"]
                )
            ],
            process=ProcessType.SEQUENTIAL,
            memory=False,
            max_execution_time=300
        )
        self.register_crew(meta_crew)
        
        # Quick Research Crew (for rapid workflows)
        quick_research_crew = CrewDefinition(
            name="Quick Research Crew",
            description="Rapid research crew for urgent content needs",
            agents=[
                AgentRole.WEB_RESEARCHER,
                AgentRole.CONTENT_CURATOR
            ],
            tasks=[
                TaskDefinition(
                    description="Quickly research essential information about {topic}",
                    agent_role=AgentRole.WEB_RESEARCHER,
                    expected_output="Essential research findings with key facts",
                    tools=["web_search", "content_retrieval"],
                    async_execution=True
                ),
                TaskDefinition(
                    description="Organize key research findings",
                    agent_role=AgentRole.CONTENT_CURATOR,
                    expected_output="Organized essential information",
                    tools=["content_analyzer"],
                    context=["research_findings"]
                )
            ],
            process=ProcessType.SEQUENTIAL,
            memory=False,
            max_execution_time=200,
            max_rpm=20
        )
        self.register_crew(quick_research_crew)
    
    def register_crew(self, crew: CrewDefinition):
        """Register a crew template."""
        try:
            crew.validate()
            crew_id = crew.name.lower().replace(" ", "_")
            self.crews[crew_id] = crew
            self.logger.info(f"Registered CrewAI crew template: {crew.name}")
            self.metrics.record_counter("crew_template_registered", framework="crewai")
        except Exception as e:
            raise CrewAIError(f"Failed to register crew {crew.name}: {e}")
    
    def get_crew(self, crew_id: str) -> Optional[CrewDefinition]:
        """Get a crew definition by ID."""
        return self.crews.get(crew_id)
    
    def list_crews(self) -> List[CrewDefinition]:
        """List all registered crew templates."""
        return list(self.crews.values())
    
    def create_crew_instance(
        self,
        crew_id: str,
        llm,
        tools: Optional[Dict[str, Any]] = None,
        variables: Optional[Dict[str, Any]] = None
    ):
        """Create a CrewAI crew instance from template."""
        crew_def = self.get_crew(crew_id)
        if not crew_def:
            raise CrewAIError(f"Crew template not found: {crew_id}")
        
        try:
            from crewai import Crew, Task, Process
            
            variables = variables or {}
            
            # Create agent instances
            agents = []
            for agent_role in crew_def.agents:
                # Find agent definition by role
                agent_defs = self.agent_registry.get_agents_by_role(agent_role)
                if not agent_defs:
                    raise CrewAIError(f"No agent definition found for role: {agent_role}")
                
                # Use first matching agent definition
                agent_def = agent_defs[0]
                agent_id = f"{agent_def.role.value}_{agent_def.name.lower().replace(' ', '_')}"
                
                agent = self.agent_registry.create_agent_instance(agent_id, llm, tools)
                agents.append(agent)
            
            # Create task instances
            tasks = []
            for task_def in crew_def.tasks:
                # Find the agent for this task
                task_agent = None
                for agent in agents:
                    # Match agent by role description (this is a simplified matching)
                    agent_role_name = task_def.agent_role.value
                    if agent_role_name in agent.role.lower().replace(" ", "_"):
                        task_agent = agent
                        break
                
                if not task_agent:
                    # Create a simple mapping by position
                    agent_index = list(crew_def.agents).index(task_def.agent_role)
                    task_agent = agents[agent_index] if agent_index < len(agents) else agents[0]
                
                # Replace variables in task description
                description = task_def.description
                for key, value in variables.items():
                    description = description.replace(f"{{{key}}}", str(value))
                
                task = Task(
                    description=description,
                    agent=task_agent,
                    expected_output=task_def.expected_output
                )
                tasks.append(task)
            
            # Create crew instance
            process = Process.hierarchical if crew_def.process == ProcessType.HIERARCHICAL else Process.sequential
            
            crew_config = {
                "agents": agents,
                "tasks": tasks,
                "process": process,
                "verbose": crew_def.verbose,
                "memory": crew_def.memory
            }
            
            if crew_def.max_execution_time:
                crew_config["max_execution_time"] = crew_def.max_execution_time
            
            if crew_def.max_rpm:
                crew_config["max_rpm"] = crew_def.max_rpm
            
            crew = Crew(**crew_config)
            
            self.logger.info(f"Created CrewAI crew instance: {crew_def.name}")
            self.metrics.record_counter(
                "crew_created", 
                framework="crewai",
                crew_name=crew_def.name
            )
            
            return crew
            
        except ImportError as e:
            raise CrewAIError(f"CrewAI not installed: {e}")
        except Exception as e:
            raise CrewAIError(f"Failed to create crew {crew_def.name}: {e}")
    
    def get_research_crews(self) -> List[CrewDefinition]:
        """Get all research-related crews."""
        return [crew for crew in self.crews.values() if "research" in crew.name.lower()]
    
    def get_strategy_crews(self) -> List[CrewDefinition]:
        """Get all strategy-related crews."""
        return [crew for crew in self.crews.values() if "strategy" in crew.name.lower()]
    
    def get_coordination_crews(self) -> List[CrewDefinition]:
        """Get all coordination-related crews."""
        return [crew for crew in self.crews.values() if "coordination" in crew.name.lower() or "meta" in crew.name.lower()]


# Global crew template registry instance
_crew_registry: Optional[CrewTemplateRegistry] = None


def get_crew_registry() -> CrewTemplateRegistry:
    """Get the global crew template registry instance."""
    global _crew_registry
    if _crew_registry is None:
        _crew_registry = CrewTemplateRegistry()
    return _crew_registry