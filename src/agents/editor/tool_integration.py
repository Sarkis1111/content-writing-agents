"""
Editor Agent Tool Integration Layer

This module provides comprehensive integration and orchestration of all editing tools
for the Editor Agent. It offers a unified interface, tool coordination, parallel processing,
and intelligent tool selection based on content analysis and editing requirements.

Key Features:
- Unified tool interface for all editing operations
- Parallel and sequential tool execution
- Intelligent tool selection and optimization
- Tool result aggregation and normalization
- Error handling and fallback mechanisms
- Performance monitoring and caching
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, src_path)

# Core imports with fallbacks
try:
    from core.errors import ToolError, ValidationError, IntegrationError
    from core.logging.logger import get_logger
    from core.monitoring.metrics import get_metrics_collector
    from utils.simple_retry import with_retry
except ImportError:
    # Mock implementations
    import logging
    class ToolError(Exception): pass
    class ValidationError(Exception): pass
    class IntegrationError(Exception): pass
    
    def get_logger(name): return logging.getLogger(name)
    def get_metrics_collector(): return None
    def with_retry(*args, **kwargs):
        def decorator(func): return func
        return decorator

# Tool imports with fallbacks
try:
    sys.path.insert(0, os.path.join(src_path, 'tools', 'editing'))
    
    # Grammar checking
    from grammar_checker import GrammarChecker, GrammarCheckRequest, GrammarCheckResults
    
    # SEO analysis
    from seo_analyzer import SEOAnalyzer, SEOAnalysisRequest, SEOAnalysisResults
    
    # Readability scoring
    from readability_scorer import ReadabilityScorer, ReadabilityRequest, ReadabilityResults
    
    # Sentiment analysis
    from sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisRequest, SentimentAnalysisResults
    
    TOOLS_AVAILABLE = True
    
except ImportError as e:
    # Create mock tool classes for development
    TOOLS_AVAILABLE = False
    
    class MockGrammarChecker:
        async def check_grammar(self, request):
            return type('MockResult', (), {
                'overall_score': 85.0,
                'corrected_text': request.text if hasattr(request, 'text') else "corrected text",
                'errors': [],
                'suggestions': [],
                'processing_time': 1.2
            })()
    
    class MockSEOAnalyzer:
        async def analyze_seo(self, request):
            return type('MockResult', (), {
                'overall_score': 78.0,
                'keyword_density': {},
                'issues': [],
                'recommendations': [],
                'processing_time': 1.5
            })()
    
    class MockReadabilityScorer:
        async def score_readability(self, request):
            return type('MockResult', (), {
                'overall_score': 82.0,
                'primary_grade_level': 9.2,
                'improvement_suggestions': [],
                'priority_recommendations': [],
                'processing_time': 1.0
            })()
    
    class MockSentimentAnalyzer:
        async def analyze_sentiment(self, request):
            return type('MockResult', (), {
                'overall_sentiment': type('Sentiment', (), {'polarity': 0.3})(),
                'confidence_score': 78.0,
                'recommendations': [],
                'warnings': [],
                'processing_time': 0.8
            })()
    
    # Use mock classes
    GrammarChecker = MockGrammarChecker
    SEOAnalyzer = MockSEOAnalyzer
    ReadabilityScorer = MockReadabilityScorer
    SentimentAnalyzer = MockSentimentAnalyzer
    
    # Mock request classes
    class GrammarCheckRequest:
        def __init__(self, text, **kwargs):
            self.text = text
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class SEOAnalysisRequest:
        def __init__(self, content, **kwargs):
            self.content = content
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ReadabilityRequest:
        def __init__(self, text, **kwargs):
            self.text = text
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class SentimentAnalysisRequest:
        def __init__(self, text, **kwargs):
            self.text = text
            for k, v in kwargs.items():
                setattr(self, k, v)

logger = get_logger(__name__)


class ToolType(str, Enum):
    """Available editing tool types"""
    GRAMMAR = "grammar"
    SEO = "seo"
    READABILITY = "readability"
    SENTIMENT = "sentiment"


class ExecutionMode(str, Enum):
    """Tool execution modes"""
    SEQUENTIAL = "sequential"       # Run tools one after another
    PARALLEL = "parallel"          # Run tools simultaneously
    SELECTIVE = "selective"        # Run only selected tools
    ADAPTIVE = "adaptive"          # Intelligently select and order tools


class ToolPriority(str, Enum):
    """Tool execution priority levels"""
    CRITICAL = "critical"    # Must execute successfully
    HIGH = "high"           # Should execute, can fallback
    MEDIUM = "medium"       # Execute if time/resources allow
    LOW = "low"            # Optional execution


@dataclass
class ToolExecutionPlan:
    """Plan for executing editing tools"""
    
    tools_to_execute: List[ToolType] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    tool_priorities: Dict[ToolType, ToolPriority] = field(default_factory=dict)
    
    # Execution parameters
    max_parallel_tools: int = 3
    timeout_per_tool: int = 120
    retry_failed_tools: bool = True
    max_retries: int = 2
    
    # Tool-specific configurations
    tool_configs: Dict[ToolType, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tools_to_execute": [tool.value for tool in self.tools_to_execute],
            "execution_mode": self.execution_mode.value,
            "tool_priorities": {tool.value: priority.value for tool, priority in self.tool_priorities.items()},
            "max_parallel_tools": self.max_parallel_tools,
            "timeout_per_tool": self.timeout_per_tool,
            "retry_failed_tools": self.retry_failed_tools,
            "max_retries": self.max_retries,
            "tool_configs": {tool.value: config for tool, config in self.tool_configs.items()}
        }


@dataclass
class ToolResult:
    """Standardized result from a tool execution"""
    
    tool_type: ToolType
    success: bool
    
    # Core results
    score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional data
    raw_result: Any = None
    processed_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution info
    execution_time: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tool_type": self.tool_type.value,
            "success": self.success,
            "score": self.score,
            "recommendations": self.recommendations,
            "issues": self.issues,
            "processed_content": self.processed_content,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "retry_count": self.retry_count
        }


@dataclass
class IntegratedToolResults:
    """Aggregated results from all tool executions"""
    
    content: str
    original_content: str
    
    # Tool results
    tool_results: Dict[ToolType, ToolResult] = field(default_factory=dict)
    
    # Aggregated metrics
    overall_score: float = 0.0
    individual_scores: Dict[ToolType, float] = field(default_factory=dict)
    
    # Combined recommendations and issues
    all_recommendations: List[str] = field(default_factory=list)
    all_issues: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution summary
    execution_plan: Optional[ToolExecutionPlan] = None
    total_execution_time: float = 0.0
    successful_tools: List[ToolType] = field(default_factory=list)
    failed_tools: List[ToolType] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "original_content": self.original_content,
            "tool_results": {tool.value: result.to_dict() for tool, result in self.tool_results.items()},
            "overall_score": self.overall_score,
            "individual_scores": {tool.value: score for tool, score in self.individual_scores.items()},
            "all_recommendations": self.all_recommendations,
            "all_issues": self.all_issues,
            "execution_plan": self.execution_plan.to_dict() if self.execution_plan else None,
            "total_execution_time": self.total_execution_time,
            "successful_tools": [tool.value for tool in self.successful_tools],
            "failed_tools": [tool.value for tool in self.failed_tools]
        }


class ToolCache:
    """Caching system for tool results"""
    
    def __init__(self, max_size: int = 100, ttl_minutes: int = 60):
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self.cache: Dict[str, Tuple[ToolResult, datetime]] = {}
        self.logger = get_logger(__name__)
    
    def _generate_cache_key(self, tool_type: ToolType, content: str, config: Dict[str, Any]) -> str:
        """Generate cache key for tool execution"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        return f"{tool_type.value}:{content_hash}:{config_hash}"
    
    def get(self, tool_type: ToolType, content: str, config: Dict[str, Any]) -> Optional[ToolResult]:
        """Get cached tool result"""
        key = self._generate_cache_key(tool_type, content, config)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                self.logger.debug(f"Cache hit for {tool_type.value}")
                return result
            else:
                # Expired, remove from cache
                del self.cache[key]
        
        return None
    
    def put(self, tool_type: ToolType, content: str, config: Dict[str, Any], result: ToolResult):
        """Cache tool result"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entries
            oldest_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])[:10]
            for key in oldest_keys:
                del self.cache[key]
        
        key = self._generate_cache_key(tool_type, content, config)
        self.cache[key] = (result, datetime.now())
        self.logger.debug(f"Cached result for {tool_type.value}")
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.logger.info("Tool cache cleared")


class ToolOrchestrator:
    """Orchestrates execution of multiple editing tools"""
    
    def __init__(self, enable_caching: bool = True, cache_ttl: int = 60):
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # Initialize editing tools
        self.tools = self._initialize_tools()
        
        # Caching
        self.cache = ToolCache(ttl_minutes=cache_ttl) if enable_caching else None
        
        # Performance tracking
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "avg_execution_time": 0.0,
            "tool_success_rates": {tool.value: 0.0 for tool in ToolType}
        }
    
    def _initialize_tools(self) -> Dict[ToolType, Any]:
        """Initialize all editing tools"""
        tools = {}
        
        try:
            tools[ToolType.GRAMMAR] = GrammarChecker()
            tools[ToolType.SEO] = SEOAnalyzer()
            tools[ToolType.READABILITY] = ReadabilityScorer()
            tools[ToolType.SENTIMENT] = SentimentAnalyzer()
            
            self.logger.info(f"Initialized {len(tools)} editing tools (available: {TOOLS_AVAILABLE})")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize some editing tools: {e}")
        
        return tools
    
    def create_execution_plan(
        self,
        content: str,
        editing_requirements: Dict[str, Any],
        mode: ExecutionMode = ExecutionMode.ADAPTIVE
    ) -> ToolExecutionPlan:
        """Create intelligent execution plan based on content and requirements"""
        
        plan = ToolExecutionPlan()
        plan.execution_mode = mode
        
        # Analyze content to determine which tools are needed
        content_length = len(content)
        word_count = len(content.split())
        
        # Default tool selection
        if mode == ExecutionMode.ADAPTIVE:
            # Intelligent tool selection based on content and requirements
            
            # Grammar check - always needed
            plan.tools_to_execute.append(ToolType.GRAMMAR)
            plan.tool_priorities[ToolType.GRAMMAR] = ToolPriority.CRITICAL
            
            # SEO analysis - if SEO requirements specified
            if editing_requirements.get("target_keywords") or editing_requirements.get("seo_requirements"):
                plan.tools_to_execute.append(ToolType.SEO)
                plan.tool_priorities[ToolType.SEO] = ToolPriority.HIGH
            
            # Readability - for longer content or specific audience
            if word_count > 100 or editing_requirements.get("target_audience"):
                plan.tools_to_execute.append(ToolType.READABILITY)
                plan.tool_priorities[ToolType.READABILITY] = ToolPriority.HIGH
            
            # Sentiment analysis - for marketing or brand content
            content_type = editing_requirements.get("content_type", "")
            if "marketing" in content_type.lower() or "brand" in content_type.lower():
                plan.tools_to_execute.append(ToolType.SENTIMENT)
                plan.tool_priorities[ToolType.SENTIMENT] = ToolPriority.MEDIUM
        
        else:
            # Default: run all tools
            plan.tools_to_execute = list(ToolType)
            for tool in ToolType:
                plan.tool_priorities[tool] = ToolPriority.HIGH
        
        # Configure tool-specific parameters
        plan.tool_configs = self._create_tool_configs(editing_requirements)
        
        # Optimize execution based on content size and complexity
        if word_count > 1000:
            plan.execution_mode = ExecutionMode.PARALLEL if mode == ExecutionMode.ADAPTIVE else mode
            plan.max_parallel_tools = 2
        
        self.logger.info(f"Created execution plan: {len(plan.tools_to_execute)} tools, {plan.execution_mode.value} mode")
        return plan
    
    def _create_tool_configs(self, editing_requirements: Dict[str, Any]) -> Dict[ToolType, Dict[str, Any]]:
        """Create tool-specific configurations from editing requirements"""
        configs = {}
        
        # Grammar checker config
        configs[ToolType.GRAMMAR] = {
            "style": editing_requirements.get("writing_style", "business"),
            "check_spelling": True,
            "check_grammar": True,
            "check_style": True,
            "auto_correct": editing_requirements.get("enable_auto_fixes", True)
        }
        
        # SEO analyzer config
        configs[ToolType.SEO] = {
            "target_keywords": editing_requirements.get("target_keywords", []),
            "content_type": editing_requirements.get("content_type", "article"),
            "target_url": editing_requirements.get("target_url"),
            "competitor_urls": editing_requirements.get("competitor_urls", [])
        }
        
        # Readability scorer config
        configs[ToolType.READABILITY] = {
            "target_audience": editing_requirements.get("target_audience"),
            "content_purpose": editing_requirements.get("content_purpose", "informational"),
            "include_vocabulary_analysis": True,
            "include_sentence_analysis": True
        }
        
        # Sentiment analyzer config
        configs[ToolType.SENTIMENT] = {
            "target_brand_voice": editing_requirements.get("brand_voice"),
            "target_audience": editing_requirements.get("target_audience"),
            "content_context": editing_requirements.get("content_type"),
            "include_sentence_analysis": False  # Skip for performance
        }
        
        return configs
    
    async def execute_tools(
        self,
        content: str,
        execution_plan: ToolExecutionPlan
    ) -> IntegratedToolResults:
        """Execute tools according to the execution plan"""
        
        start_time = datetime.now()
        self.logger.info(f"Starting tool execution: {execution_plan.execution_mode.value} mode, {len(execution_plan.tools_to_execute)} tools")
        
        try:
            # Initialize results
            results = IntegratedToolResults(
                content=content,
                original_content=content,
                execution_plan=execution_plan
            )
            
            # Execute tools based on mode
            if execution_plan.execution_mode == ExecutionMode.PARALLEL:
                tool_results = await self._execute_parallel(content, execution_plan)
            else:
                tool_results = await self._execute_sequential(content, execution_plan)
            
            # Process and aggregate results
            results = self._aggregate_results(results, tool_results)
            
            # Calculate execution time
            results.total_execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance stats
            self._update_performance_stats(results)
            
            self.logger.info(f"Tool execution completed in {results.total_execution_time:.2f}s - Overall score: {results.overall_score:.1f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}")
            
            # Return error result
            error_results = IntegratedToolResults(
                content=content,
                original_content=content,
                execution_plan=execution_plan
            )
            error_results.total_execution_time = (datetime.now() - start_time).total_seconds()
            error_results.failed_tools = execution_plan.tools_to_execute
            
            return error_results
    
    async def _execute_parallel(
        self,
        content: str,
        execution_plan: ToolExecutionPlan
    ) -> Dict[ToolType, ToolResult]:
        """Execute tools in parallel"""
        
        # Create tasks for each tool
        tasks = []
        for tool_type in execution_plan.tools_to_execute:
            config = execution_plan.tool_configs.get(tool_type, {})
            task = asyncio.create_task(
                self._execute_single_tool(tool_type, content, config, execution_plan.timeout_per_tool)
            )
            tasks.append((tool_type, task))
        
        # Execute tools in batches to respect max_parallel_tools
        results = {}
        for i in range(0, len(tasks), execution_plan.max_parallel_tools):
            batch = tasks[i:i + execution_plan.max_parallel_tools]
            
            # Wait for batch completion
            batch_results = await asyncio.gather(
                *[task for _, task in batch],
                return_exceptions=True
            )
            
            # Process batch results
            for (tool_type, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Tool {tool_type.value} failed: {result}")
                    results[tool_type] = ToolResult(
                        tool_type=tool_type,
                        success=False,
                        error_message=str(result)
                    )
                else:
                    results[tool_type] = result
        
        return results
    
    async def _execute_sequential(
        self,
        content: str,
        execution_plan: ToolExecutionPlan
    ) -> Dict[ToolType, ToolResult]:
        """Execute tools sequentially"""
        
        results = {}
        current_content = content
        
        for tool_type in execution_plan.tools_to_execute:
            config = execution_plan.tool_configs.get(tool_type, {})
            
            try:
                result = await self._execute_single_tool(
                    tool_type,
                    current_content,
                    config,
                    execution_plan.timeout_per_tool
                )
                
                results[tool_type] = result
                
                # Use processed content for next tool if available and successful
                if result.success and result.processed_content:
                    current_content = result.processed_content
                
            except Exception as e:
                self.logger.error(f"Tool {tool_type.value} execution failed: {e}")
                results[tool_type] = ToolResult(
                    tool_type=tool_type,
                    success=False,
                    error_message=str(e)
                )
                
                # For critical tools, stop execution
                priority = execution_plan.tool_priorities.get(tool_type, ToolPriority.MEDIUM)
                if priority == ToolPriority.CRITICAL:
                    self.logger.warning(f"Critical tool {tool_type.value} failed, stopping execution")
                    break
        
        return results
    
    @with_retry(max_attempts=3, delay=1.0)
    async def _execute_single_tool(
        self,
        tool_type: ToolType,
        content: str,
        config: Dict[str, Any],
        timeout: int
    ) -> ToolResult:
        """Execute a single editing tool"""
        
        start_time = datetime.now()
        
        try:
            # Check cache first
            if self.cache:
                cached_result = self.cache.get(tool_type, content, config)
                if cached_result:
                    return cached_result
            
            # Get tool instance
            if tool_type not in self.tools:
                raise ToolError(f"Tool not available: {tool_type.value}")
            
            tool = self.tools[tool_type]
            
            # Execute tool with timeout
            result = await asyncio.wait_for(
                self._call_tool_method(tool_type, tool, content, config),
                timeout=timeout
            )
            
            # Process result into standardized format
            tool_result = self._process_tool_result(tool_type, result, content)
            tool_result.execution_time = (datetime.now() - start_time).total_seconds()
            
            # Cache successful result
            if self.cache and tool_result.success:
                self.cache.put(tool_type, content, config, tool_result)
            
            self.logger.debug(f"Tool {tool_type.value} completed: score={tool_result.score:.1f}, time={tool_result.execution_time:.2f}s")
            return tool_result
            
        except asyncio.TimeoutError:
            error_msg = f"Tool {tool_type.value} timed out after {timeout}s"
            self.logger.error(error_msg)
            return ToolResult(
                tool_type=tool_type,
                success=False,
                error_message=error_msg,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            error_msg = f"Tool {tool_type.value} failed: {str(e)}"
            self.logger.error(error_msg)
            return ToolResult(
                tool_type=tool_type,
                success=False,
                error_message=error_msg,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _call_tool_method(self, tool_type: ToolType, tool: Any, content: str, config: Dict[str, Any]) -> Any:
        """Call the appropriate method for each tool type"""
        
        if tool_type == ToolType.GRAMMAR:
            request = GrammarCheckRequest(text=content, **config)
            return await tool.check_grammar(request)
        
        elif tool_type == ToolType.SEO:
            request = SEOAnalysisRequest(content=content, **config)
            return await tool.analyze_seo(request)
        
        elif tool_type == ToolType.READABILITY:
            request = ReadabilityRequest(text=content, **config)
            return await tool.score_readability(request)
        
        elif tool_type == ToolType.SENTIMENT:
            request = SentimentAnalysisRequest(text=content, **config)
            return await tool.analyze_sentiment(request)
        
        else:
            raise ToolError(f"Unknown tool type: {tool_type}")
    
    def _process_tool_result(self, tool_type: ToolType, raw_result: Any, original_content: str) -> ToolResult:
        """Process raw tool result into standardized format"""
        
        try:
            if tool_type == ToolType.GRAMMAR:
                return self._process_grammar_result(raw_result, original_content)
            elif tool_type == ToolType.SEO:
                return self._process_seo_result(raw_result, original_content)
            elif tool_type == ToolType.READABILITY:
                return self._process_readability_result(raw_result, original_content)
            elif tool_type == ToolType.SENTIMENT:
                return self._process_sentiment_result(raw_result, original_content)
            else:
                return ToolResult(
                    tool_type=tool_type,
                    success=False,
                    error_message=f"Unknown tool type: {tool_type}"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to process {tool_type.value} result: {e}")
            return ToolResult(
                tool_type=tool_type,
                success=False,
                error_message=f"Result processing failed: {str(e)}"
            )
    
    def _process_grammar_result(self, result: Any, original_content: str) -> ToolResult:
        """Process grammar checker result"""
        score = getattr(result, 'overall_score', 0.0)
        corrected_text = getattr(result, 'corrected_text', None)
        errors = getattr(result, 'errors', [])
        
        # Convert errors to standardized format
        issues = []
        recommendations = []
        
        if hasattr(result, 'errors') and result.errors:
            for error in result.errors[:10]:  # Limit to top 10
                issues.append({
                    "type": getattr(error, 'error_type', {}).get('value', 'grammar') if hasattr(error, 'error_type') else 'grammar',
                    "message": getattr(error, 'message', ''),
                    "severity": getattr(error, 'severity', 'medium'),
                    "suggestion": getattr(error, 'suggestion', ''),
                    "position": getattr(error, 'start_pos', None)
                })
        
        if hasattr(result, 'suggestions') and result.suggestions:
            recommendations = [getattr(sugg, 'description', str(sugg)) for sugg in result.suggestions[:5]]
        
        return ToolResult(
            tool_type=ToolType.GRAMMAR,
            success=score > 0,
            score=score,
            recommendations=recommendations,
            issues=issues,
            raw_result=result,
            processed_content=corrected_text,
            metadata={
                "error_count": len(errors) if errors else 0,
                "auto_corrections": bool(corrected_text and corrected_text != original_content)
            }
        )
    
    def _process_seo_result(self, result: Any, original_content: str) -> ToolResult:
        """Process SEO analyzer result"""
        score = getattr(result, 'overall_score', 0.0)
        issues = getattr(result, 'issues', [])
        
        # Convert issues to standardized format
        standardized_issues = []
        recommendations = []
        
        if issues:
            for issue in issues[:10]:  # Limit to top 10
                standardized_issues.append({
                    "type": getattr(issue, 'issue_type', {}).get('value', 'seo') if hasattr(issue, 'issue_type') else 'seo',
                    "message": getattr(issue, 'description', ''),
                    "severity": "high" if getattr(issue, 'priority', '') == "critical" else "medium",
                    "suggestion": getattr(issue, 'recommendation', ''),
                    "impact": getattr(issue, 'impact_score', 0)
                })
                
                if hasattr(issue, 'recommendation') and issue.recommendation:
                    recommendations.append(issue.recommendation)
        
        keyword_density = getattr(result, 'keyword_density', {})
        
        return ToolResult(
            tool_type=ToolType.SEO,
            success=score > 0,
            score=score,
            recommendations=recommendations[:5],  # Limit recommendations
            issues=standardized_issues,
            raw_result=result,
            metadata={
                "keyword_density": keyword_density,
                "seo_issues_count": len(issues) if issues else 0
            }
        )
    
    def _process_readability_result(self, result: Any, original_content: str) -> ToolResult:
        """Process readability scorer result"""
        score = getattr(result, 'overall_score', 0.0)
        grade_level = getattr(result, 'primary_grade_level', 0.0)
        
        recommendations = []
        if hasattr(result, 'improvement_suggestions'):
            recommendations.extend(getattr(result, 'improvement_suggestions', [])[:5])
        if hasattr(result, 'priority_recommendations'):
            recommendations.extend(getattr(result, 'priority_recommendations', [])[:3])
        
        # Create issues from recommendations
        issues = []
        for i, rec in enumerate(recommendations[:5]):
            issues.append({
                "type": "readability",
                "message": rec,
                "severity": "medium",
                "suggestion": rec,
                "priority": i + 1
            })
        
        return ToolResult(
            tool_type=ToolType.READABILITY,
            success=score > 0,
            score=score,
            recommendations=recommendations[:5],
            issues=issues,
            raw_result=result,
            metadata={
                "grade_level": grade_level,
                "readability_analysis": True
            }
        )
    
    def _process_sentiment_result(self, result: Any, original_content: str) -> ToolResult:
        """Process sentiment analyzer result"""
        confidence_score = getattr(result, 'confidence_score', 0.0)
        
        # Get sentiment polarity
        sentiment_polarity = 0.0
        if hasattr(result, 'overall_sentiment'):
            overall_sentiment = result.overall_sentiment
            if hasattr(overall_sentiment, 'polarity'):
                sentiment_polarity = overall_sentiment.polarity
        
        # Convert polarity to score (0-100)
        score = max(0, (sentiment_polarity + 1) * 50)  # Convert -1,1 range to 0-100
        
        recommendations = getattr(result, 'recommendations', [])[:5]
        warnings = getattr(result, 'warnings', [])
        
        # Create issues from warnings
        issues = []
        for warning in warnings[:5]:
            issues.append({
                "type": "sentiment",
                "message": warning,
                "severity": "low",
                "suggestion": "Review content sentiment and tone"
            })
        
        return ToolResult(
            tool_type=ToolType.SENTIMENT,
            success=confidence_score > 0,
            score=confidence_score,
            recommendations=recommendations,
            issues=issues,
            raw_result=result,
            metadata={
                "sentiment_polarity": sentiment_polarity,
                "sentiment_confidence": confidence_score
            }
        )
    
    def _aggregate_results(self, base_results: IntegratedToolResults, tool_results: Dict[ToolType, ToolResult]) -> IntegratedToolResults:
        """Aggregate individual tool results into integrated results"""
        
        base_results.tool_results = tool_results
        
        # Categorize tools by success/failure
        for tool_type, result in tool_results.items():
            if result.success:
                base_results.successful_tools.append(tool_type)
                base_results.individual_scores[tool_type] = result.score
            else:
                base_results.failed_tools.append(tool_type)
        
        # Calculate overall score (weighted average of successful tools)
        score_weights = {
            ToolType.GRAMMAR: 0.3,
            ToolType.SEO: 0.25,
            ToolType.READABILITY: 0.25,
            ToolType.SENTIMENT: 0.2
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for tool_type, score in base_results.individual_scores.items():
            weight = score_weights.get(tool_type, 0.2)
            weighted_score += score * weight
            total_weight += weight
        
        base_results.overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Aggregate recommendations and issues
        for result in tool_results.values():
            if result.success:
                base_results.all_recommendations.extend(result.recommendations)
                base_results.all_issues.extend(result.issues)
        
        # Remove duplicate recommendations
        base_results.all_recommendations = list(dict.fromkeys(base_results.all_recommendations))
        
        # Use the best processed content (highest scoring tool that provided content)
        best_content = base_results.content
        best_score = 0.0
        
        for result in tool_results.values():
            if result.success and result.processed_content and result.score > best_score:
                best_content = result.processed_content
                best_score = result.score
        
        base_results.content = best_content
        
        return base_results
    
    def _update_performance_stats(self, results: IntegratedToolResults):
        """Update performance tracking statistics"""
        try:
            self.execution_stats["total_executions"] += 1
            
            # Update success rate
            if results.overall_score > 70:  # Consider 70+ as successful
                self.execution_stats["successful_executions"] += 1
            
            success_rate = self.execution_stats["successful_executions"] / self.execution_stats["total_executions"]
            self.execution_stats["success_rate"] = success_rate
            
            # Update average execution time
            total_time = self.execution_stats["avg_execution_time"] * (self.execution_stats["total_executions"] - 1)
            self.execution_stats["avg_execution_time"] = (total_time + results.total_execution_time) / self.execution_stats["total_executions"]
            
            # Update tool-specific success rates
            for tool_type in ToolType:
                if tool_type in results.successful_tools:
                    # Update success rate for this tool (simplified)
                    current_rate = self.execution_stats["tool_success_rates"][tool_type.value]
                    self.execution_stats["tool_success_rates"][tool_type.value] = min(100.0, current_rate + 1.0)
                elif tool_type in results.failed_tools:
                    current_rate = self.execution_stats["tool_success_rates"][tool_type.value]
                    self.execution_stats["tool_success_rates"][tool_type.value] = max(0.0, current_rate - 1.0)
            
            # Log metrics
            if self.metrics:
                self.metrics.record_counter("tool_orchestrator_executions")
                self.metrics.record_histogram("tool_orchestrator_execution_time", results.total_execution_time)
                self.metrics.record_gauge("tool_orchestrator_overall_score", results.overall_score)
                
                for tool_type, score in results.individual_scores.items():
                    self.metrics.record_gauge(f"tool_score_{tool_type.value}", score)
                    
        except Exception as e:
            self.logger.warning(f"Failed to update performance stats: {e}")
    
    # Public Interface
    
    async def edit_content(
        self,
        content: str,
        editing_requirements: Optional[Dict[str, Any]] = None,
        execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE
    ) -> IntegratedToolResults:
        """
        Main interface for editing content using all integrated tools
        
        Args:
            content: Content to edit
            editing_requirements: Editing requirements and preferences
            execution_mode: How to execute the tools
            
        Returns:
            Integrated results from all tools
        """
        
        editing_requirements = editing_requirements or {}
        
        try:
            self.logger.info(f"Starting integrated content editing: {len(content)} chars, {execution_mode.value} mode")
            
            # Create execution plan
            execution_plan = self.create_execution_plan(content, editing_requirements, execution_mode)
            
            # Execute tools
            results = await self.execute_tools(content, execution_plan)
            
            self.logger.info(f"Integrated editing completed: {len(results.successful_tools)}/{len(execution_plan.tools_to_execute)} tools successful")
            return results
            
        except Exception as e:
            self.logger.error(f"Integrated content editing failed: {e}")
            raise IntegrationError(f"Content editing integration failed: {str(e)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.execution_stats.copy()
    
    def clear_cache(self):
        """Clear tool result cache"""
        if self.cache:
            self.cache.clear()
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return [tool.value for tool in self.tools.keys()]


# Global orchestrator instance
_tool_orchestrator: Optional[ToolOrchestrator] = None


def get_tool_orchestrator() -> ToolOrchestrator:
    """Get the global tool orchestrator instance"""
    global _tool_orchestrator
    if _tool_orchestrator is None:
        _tool_orchestrator = ToolOrchestrator()
    return _tool_orchestrator


# Export main classes
__all__ = [
    'ToolOrchestrator',
    'ToolExecutionPlan',
    'ToolResult',
    'IntegratedToolResults',
    'ToolType',
    'ExecutionMode',
    'ToolPriority',
    'get_tool_orchestrator'
]


if __name__ == "__main__":
    # Example usage and testing
    async def test_tool_integration():
        """Test the tool integration functionality"""
        
        test_content = """
        This is a test content for editing integration. It has some grammer mistakes and could be improved.
        The SEO optimization is not great and readability might be better with shorter sentences.
        
        Overall this content needs significant editing to meet quality standards for professional publication.
        We want to make sure it's perfect before we publish it to our audience.
        """
        
        editing_requirements = {
            "target_keywords": ["content", "editing", "quality"],
            "target_audience": "professional writers",
            "content_type": "blog_post",
            "writing_style": "professional",
            "enable_auto_fixes": True
        }
        
        try:
            orchestrator = get_tool_orchestrator()
            
            results = await orchestrator.edit_content(
                test_content,
                editing_requirements,
                ExecutionMode.PARALLEL
            )
            
            print("Tool Integration Test Results:")
            print(f"Overall Score: {results.overall_score:.1f}")
            print(f"Total Execution Time: {results.total_execution_time:.2f}s")
            print(f"Successful Tools: {len(results.successful_tools)}")
            print(f"Failed Tools: {len(results.failed_tools)}")
            print(f"Total Recommendations: {len(results.all_recommendations)}")
            print(f"Total Issues: {len(results.all_issues)}")
            
            print("\nIndividual Tool Scores:")
            for tool, score in results.individual_scores.items():
                print(f"- {tool.value}: {score:.1f}")
            
            if results.all_recommendations:
                print("\nTop Recommendations:")
                for rec in results.all_recommendations[:3]:
                    print(f"- {rec}")
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Uncomment to run test
    # asyncio.run(test_tool_integration())