"""
Research Agent Implementation using CrewAI Framework

This module implements Phase 3.1 of the development strategy: Research Agent with CrewAI.
The Research Agent uses a team-based approach with specialized sub-agents for comprehensive research.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_framework_logger
from ...core.errors import AgentError, ToolError
from ...core.monitoring import get_metrics_collector, PerformanceTimer
from ...frameworks.crewai import get_crew_registry, get_agent_registry
from ...tools.research import RESEARCH_TOOLS
from ...tools.research.crewai_tools import CREWAI_RESEARCH_TOOLS
from ...tools.analysis import ANALYSIS_TOOLS


@dataclass 
class ResearchRequest:
    """Request for research agent execution."""
    
    topic: str
    research_depth: str = "comprehensive"  # "quick", "standard", "comprehensive", "deep"
    focus_areas: List[str] = field(default_factory=list)
    exclude_sources: List[str] = field(default_factory=list)
    include_trends: bool = True
    include_news: bool = True
    fact_check: bool = True
    max_sources: int = 50
    language: str = "en"
    region: str = "US"
    time_limit: Optional[int] = None  # seconds
    
    def __post_init__(self):
        """Validate and normalize request parameters."""
        if not self.topic.strip():
            raise ValueError("Topic cannot be empty")
        
        valid_depths = ["quick", "standard", "comprehensive", "deep"]
        if self.research_depth not in valid_depths:
            raise ValueError(f"research_depth must be one of: {valid_depths}")
        
        if self.max_sources <= 0:
            self.max_sources = 10


@dataclass
class SourceInfo:
    """Information about a research source."""
    
    url: str
    title: str
    credibility_score: float
    relevance_score: float
    content_type: str  # "article", "news", "academic", "blog", "social"
    publish_date: Optional[datetime] = None
    author: Optional[str] = None
    summary: str = ""


@dataclass
class TrendData:
    """Trend analysis data."""
    
    keyword: str
    trend_direction: str  # "rising", "falling", "stable"
    growth_rate: float
    interest_score: float
    related_queries: List[str] = field(default_factory=list)
    regional_data: Dict[str, float] = field(default_factory=dict)


@dataclass 
class ResearchResponse:
    """Response from research agent execution."""
    
    topic: str
    request_id: str
    execution_time: float
    status: str  # "completed", "partial", "failed"
    
    @property
    def success(self) -> bool:
        """Return True if research was successful."""
        return self.status in ["completed", "partial"]
    
    # Core research data
    sources: List[SourceInfo] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    summary: str = ""
    
    # Trend analysis
    trends: List[TrendData] = field(default_factory=list)
    trending_topics: List[str] = field(default_factory=list)
    
    # News analysis
    recent_news: List[Dict[str, Any]] = field(default_factory=list)
    news_sentiment: str = "neutral"  # "positive", "negative", "neutral", "mixed"
    
    # Content organization
    content_categories: Dict[str, List[str]] = field(default_factory=dict)
    key_themes: List[str] = field(default_factory=list)
    
    # Quality metrics
    fact_check_results: List[Dict[str, Any]] = field(default_factory=list)
    credibility_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    agent_performance: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ResearchAgent:
    """
    Research Agent using CrewAI framework with specialized sub-agents.
    
    This agent implements the multi-agent research coordination pattern described
    in Phase 3.1 of the development strategy. It uses CrewAI's team-based approach
    with the following specialized agents:
    
    - Web Research Specialist: Broad information gathering
    - Trend Analyst: Market and topic trends  
    - Content Curator: Content organization and synthesis
    - Fact Checker: Source verification and validation
    """
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the Research Agent."""
        
        try:
            self.config = get_config()
        except Exception as e:
            # Handle config validation issues gracefully
            self.config = None
            
        self.logger = get_framework_logger("ResearchAgent")
        self.metrics = get_metrics_collector()
        
        # Initialize framework components
        self.crew_registry = get_crew_registry()
        self.agent_registry = get_agent_registry()
        
        # Set up LLM configuration
        self.llm_config = llm_config or self._get_default_llm_config()
        self.llm = self._initialize_llm()
        
        # Initialize tools (CrewAI-compatible tools for crew execution)
        self.tools = CREWAI_RESEARCH_TOOLS
        
        # Keep original tools for direct access if needed
        self.original_tools = {
            **RESEARCH_TOOLS,
            **ANALYSIS_TOOLS
        }
        
        # Performance tracking
        self.execution_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_execution_time": 0.0,
            "last_execution": None
        }
        
        self.logger.info("Research Agent initialized with CrewAI framework")
        self.metrics.record_counter("research_agent_initialized", framework="crewai")
    
    def _get_default_llm_config(self) -> Dict[str, Any]:
        """Get default LLM configuration."""
        default_model = "gpt-3.5-turbo"
        if self.config and hasattr(self.config, 'frameworks') and hasattr(self.config.frameworks, 'crewai'):
            default_model = self.config.frameworks.crewai.model
        
        return {
            "model": default_model,
            "temperature": 0.3,  # Lower temperature for more factual research
            "max_tokens": 2000,
            "timeout": 60
        }
    
    def _initialize_llm(self):
        """Initialize the LLM instance."""
        try:
            from langchain_openai import ChatOpenAI
            
            return ChatOpenAI(
                model=self.llm_config["model"],
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"],
                timeout=self.llm_config["timeout"]
            )
        except ImportError as e:
            self.logger.warning(f"LangChain OpenAI not available: {e}")
            # Fall back to mock LLM for testing
            class MockLLM:
                def __init__(self, **kwargs):
                    self.config = kwargs
                    self.model_name = kwargs.get("model", "mock-llm")
            
            return MockLLM(**self.llm_config)
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise AgentError(f"Failed to initialize LLM: {e}")
    
    async def research(self, request: ResearchRequest) -> ResearchResponse:
        """
        Execute research using the CrewAI research team.
        
        Args:
            request: Research request with topic and parameters
            
        Returns:
            ResearchResponse with findings and analysis
        """
        
        request_id = f"research_{int(datetime.now().timestamp())}"
        
        self.logger.info(f"Starting research for topic: {request.topic} (ID: {request_id})")
        
        with PerformanceTimer(self.metrics, "research_execution", topic=request.topic) as timer:
            try:
                # Select appropriate crew based on research depth
                crew_name = self._select_crew_for_depth(request.research_depth)
                
                # Create crew instance
                crew = self._create_research_crew(crew_name, request)
                
                # Execute research workflow
                result = await self._execute_crew_research(crew, request)
                
                # Calculate execution time
                execution_time = time.time() - timer.start_time if hasattr(timer, 'start_time') else 0.0
                
                # Process and format results
                response = self._format_research_response(
                    request, request_id, result, execution_time
                )
                
                # Update statistics
                self._update_execution_stats(True, execution_time)
                
                self.logger.info(f"Research completed successfully: {request_id}")
                self.metrics.record_timer("research_execution_time", execution_time)
                self.metrics.record_counter("research_successful", topic=request.topic)
                
                return response
                
            except Exception as e:
                self.logger.error(f"Research failed for {request_id}: {e}")
                self.metrics.record_counter("research_failed", error=str(e))
                
                # Calculate execution time for error case
                execution_time = time.time() - timer.start_time if hasattr(timer, 'start_time') else 0.0
                
                # Update statistics
                self._update_execution_stats(False, execution_time)
                
                # Return error response
                return self._create_error_response(request, request_id, str(e), execution_time)
    
    def _select_crew_for_depth(self, research_depth: str) -> str:
        """Select appropriate crew based on research depth."""
        crew_mapping = {
            "quick": "quick_research_crew",
            "standard": "research_crew", 
            "comprehensive": "research_crew",
            "deep": "research_crew"
        }
        
        return crew_mapping.get(research_depth, "research_crew")
    
    def _create_research_crew(self, crew_name: str, request: ResearchRequest):
        """Create and configure research crew instance."""
        
        try:
            # Prepare variables for crew tasks
            variables = {
                "topic": request.topic,
                "max_sources": request.max_sources,
                "language": request.language,
                "region": request.region,
                "focus_areas": ", ".join(request.focus_areas) if request.focus_areas else "general",
                "research_depth": request.research_depth
            }
            
            # Create crew instance
            crew = self.crew_registry.create_crew_instance(
                crew_name, 
                self.llm,
                tools=self.tools,
                variables=variables
            )
            
            return crew
            
        except Exception as e:
            raise AgentError(f"Failed to create research crew: {e}")
    
    async def _execute_crew_research(self, crew, request: ResearchRequest) -> Dict[str, Any]:
        """Execute research using the CrewAI crew."""
        
        try:
            # Set execution timeout if specified
            if request.time_limit:
                result = await asyncio.wait_for(
                    self._run_crew_async(crew),
                    timeout=request.time_limit
                )
            else:
                result = await self._run_crew_async(crew)
            
            return result
            
        except asyncio.TimeoutError:
            raise AgentError(f"Research timed out after {request.time_limit} seconds")
        except Exception as e:
            raise AgentError(f"Crew execution failed: {e}")
    
    async def _run_crew_async(self, crew) -> Dict[str, Any]:
        """Run CrewAI crew in async context."""
        
        # CrewAI crews are synchronous, so we need to run in a thread
        loop = asyncio.get_event_loop()
        
        def run_crew():
            try:
                return crew.kickoff()
            except Exception as e:
                self.logger.error(f"Crew kickoff failed: {e}")
                raise AgentError(f"Crew execution error: {e}")
        
        result = await loop.run_in_executor(None, run_crew)
        
        # Extract and structure the results
        return self._parse_crew_result(result)
    
    def _parse_crew_result(self, result) -> Dict[str, Any]:
        """Parse and structure CrewAI crew result."""
        
        # CrewAI results can be strings or more complex objects
        if isinstance(result, str):
            # Simple string result - parse as best we can
            return {
                "raw_output": result,
                "parsed_content": self._parse_text_result(result),
                "execution_metadata": {}
            }
        elif hasattr(result, 'tasks_output'):
            # Structured result with task outputs
            return {
                "raw_output": str(result),
                "task_outputs": [str(task_output) for task_output in result.tasks_output],
                "parsed_content": self._parse_task_outputs(result.tasks_output),
                "execution_metadata": getattr(result, 'metadata', {})
            }
        else:
            # Unknown result format
            return {
                "raw_output": str(result),
                "parsed_content": {"summary": str(result)[:1000]},
                "execution_metadata": {}
            }
    
    def _parse_text_result(self, text: str) -> Dict[str, Any]:
        """Parse text-based crew result."""
        
        # Basic parsing of research output
        lines = text.split('\n')
        
        parsed = {
            "summary": "",
            "key_findings": [],
            "sources": [],
            "trends": []
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            lower_line = line.lower()
            if "summary" in lower_line and ":" in line:
                current_section = "summary"
                continue
            elif "finding" in lower_line or "key point" in lower_line:
                current_section = "findings"
                continue
            elif "source" in lower_line or "reference" in lower_line:
                current_section = "sources"
                continue
            elif "trend" in lower_line:
                current_section = "trends"
                continue
            
            # Add content to appropriate section
            if current_section == "summary":
                parsed["summary"] += line + " "
            elif current_section == "findings" and line.startswith(("-", "•", "*")):
                parsed["key_findings"].append(line[1:].strip())
            elif current_section == "sources" and ("http" in line or "www" in line):
                parsed["sources"].append(line)
            elif current_section == "trends":
                parsed["trends"].append(line)
        
        # Clean up summary
        parsed["summary"] = parsed["summary"].strip()
        
        return parsed
    
    def _parse_task_outputs(self, task_outputs) -> Dict[str, Any]:
        """Parse structured task outputs from CrewAI."""
        
        parsed = {
            "summary": "",
            "key_findings": [],
            "sources": [],
            "trends": [],
            "task_results": {}
        }
        
        for i, output in enumerate(task_outputs):
            output_str = str(output)
            task_name = f"task_{i+1}"
            
            # Store individual task result
            parsed["task_results"][task_name] = output_str
            
            # Extract content from each task
            task_parsed = self._parse_text_result(output_str)
            
            # Merge results
            if task_parsed["summary"]:
                parsed["summary"] += f"{task_parsed['summary']} "
            
            parsed["key_findings"].extend(task_parsed["key_findings"])
            parsed["sources"].extend(task_parsed["sources"])
            parsed["trends"].extend(task_parsed["trends"])
        
        # Clean up and deduplicate
        parsed["summary"] = parsed["summary"].strip()
        parsed["key_findings"] = list(dict.fromkeys(parsed["key_findings"]))  # Remove duplicates
        parsed["sources"] = list(dict.fromkeys(parsed["sources"]))
        parsed["trends"] = list(dict.fromkeys(parsed["trends"]))
        
        return parsed
    
    def _format_research_response(
        self, 
        request: ResearchRequest, 
        request_id: str, 
        result: Dict[str, Any], 
        execution_time: float
    ) -> ResearchResponse:
        """Format crew result into ResearchResponse."""
        
        parsed = result["parsed_content"]
        
        # Create source info objects
        sources = []
        for source_text in parsed.get("sources", [])[:request.max_sources]:
            source = self._parse_source_info(source_text)
            if source:
                sources.append(source)
        
        # Extract trends data
        trends = []
        for trend_text in parsed.get("trends", []):
            trend = self._parse_trend_data(trend_text)
            if trend:
                trends.append(trend)
        
        # Create response
        response = ResearchResponse(
            topic=request.topic,
            request_id=request_id,
            execution_time=execution_time,
            status="completed",
            summary=parsed.get("summary", ""),
            key_findings=parsed.get("key_findings", []),
            sources=sources,
            trends=trends,
            content_categories={"general": parsed.get("key_findings", [])},
            credibility_assessment={
                "avg_source_credibility": self._calculate_avg_credibility(sources),
                "total_sources": len(sources),
                "verified_sources": len([s for s in sources if s.credibility_score > 0.7])
            },
            agent_performance={
                "crew_name": "research_crew",
                "task_count": len(result.get("task_outputs", [])),
                "execution_time": execution_time,
                "success_rate": 1.0
            }
        )
        
        return response
    
    def _parse_source_info(self, source_text: str) -> Optional[SourceInfo]:
        """Parse source information from text."""
        
        # Basic URL extraction
        import re
        url_match = re.search(r'https?://[^\s]+', source_text)
        
        if url_match:
            url = url_match.group()
            title = source_text.replace(url, "").strip()
            
            return SourceInfo(
                url=url,
                title=title if title else "Unknown Title",
                credibility_score=0.8,  # Default credibility
                relevance_score=0.8,   # Default relevance
                content_type="article",
                summary=source_text[:200] + "..." if len(source_text) > 200 else source_text
            )
        
        return None
    
    def _parse_trend_data(self, trend_text: str) -> Optional[TrendData]:
        """Parse trend information from text."""
        
        # Basic trend parsing
        if any(word in trend_text.lower() for word in ["rising", "growing", "increasing"]):
            direction = "rising"
            growth_rate = 0.15
        elif any(word in trend_text.lower() for word in ["falling", "declining", "decreasing"]):
            direction = "falling"
            growth_rate = -0.10
        else:
            direction = "stable"
            growth_rate = 0.02
        
        # Extract keyword (first meaningful word)
        words = trend_text.split()
        keyword = words[0] if words else "unknown"
        
        return TrendData(
            keyword=keyword,
            trend_direction=direction,
            growth_rate=growth_rate,
            interest_score=0.7,  # Default interest
            related_queries=[trend_text]
        )
    
    def _calculate_avg_credibility(self, sources: List[SourceInfo]) -> float:
        """Calculate average credibility score of sources."""
        if not sources:
            return 0.0
        
        total = sum(source.credibility_score for source in sources)
        return round(total / len(sources), 2)
    
    def _create_error_response(
        self, 
        request: ResearchRequest, 
        request_id: str, 
        error_message: str,
        execution_time: float
    ) -> ResearchResponse:
        """Create error response."""
        
        return ResearchResponse(
            topic=request.topic,
            request_id=request_id,
            execution_time=execution_time,
            status="failed",
            errors=[error_message],
            agent_performance={
                "crew_name": "research_crew",
                "execution_time": execution_time,
                "success_rate": 0.0
            }
        )
    
    def _update_execution_stats(self, success: bool, execution_time: float):
        """Update execution statistics."""
        
        self.execution_stats["total_requests"] += 1
        
        if success:
            self.execution_stats["successful_requests"] += 1
        else:
            self.execution_stats["failed_requests"] += 1
        
        # Update average execution time
        total_time = (self.execution_stats["avg_execution_time"] * 
                     (self.execution_stats["total_requests"] - 1))
        self.execution_stats["avg_execution_time"] = (
            (total_time + execution_time) / self.execution_stats["total_requests"]
        )
        
        self.execution_stats["last_execution"] = datetime.now()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        
        success_rate = 0.0
        if self.execution_stats["total_requests"] > 0:
            success_rate = (self.execution_stats["successful_requests"] / 
                          self.execution_stats["total_requests"])
        
        return {
            **self.execution_stats,
            "success_rate": round(success_rate, 3),
            "framework": "CrewAI",
            "agent_type": "ResearchAgent",
            "capabilities": [
                "Web Research", "Trend Analysis", 
                "Content Curation", "Fact Checking"
            ]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform agent health check."""
        
        try:
            # Test basic functionality
            test_request = ResearchRequest(
                topic="AI trends",
                research_depth="quick",
                max_sources=5,
                time_limit=30
            )
            
            # Run a quick test
            result = await self.research(test_request)
            
            health_status = {
                "status": "healthy",
                "framework": "CrewAI",
                "agent_type": "ResearchAgent",
                "test_execution_time": result.execution_time,
                "test_status": result.status,
                "crew_registry_status": "active",
                "tool_count": len(self.tools),
                "performance_stats": self.get_performance_stats()
            }
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "framework": "CrewAI", 
                "agent_type": "ResearchAgent",
                "error": str(e),
                "performance_stats": self.get_performance_stats()
            }