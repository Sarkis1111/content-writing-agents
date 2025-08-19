"""
Writer Agent - LangGraph Implementation for Iterative Content Creation

This module implements the Writer Agent using LangGraph framework for complex content workflows 
with multiple decision points, state management for iterative writing processes, and conditional 
logic for different content types.

Key Features:
- LangGraph StateGraph for workflow orchestration
- Iterative content creation with revision loops
- Self-review and improvement cycles
- Content type adaptation and quality gates
- Integration with writing tools (Content Writer, Headline Generator, Image Generator, Sentiment Analyzer)
- Human escalation for complex issues
- Multi-stage approval processes

Architecture:
- Research Analysis → Outline Creation → Content Writing → Self Review → Revision Loops → Finalization
- Quality gates and approval processes with conditional logic
- Dynamic workflow based on content requirements
- State persistence and recovery
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    from ...core.errors import AgentError, WorkflowError, ToolError
    from ...core.logging import get_framework_logger
    from ...core.monitoring import get_metrics_collector
    from ...frameworks.langgraph.config import get_langgraph_framework
    from ...frameworks.langgraph.state import (
        ContentCreationState, StateStatus, ContentType, QualityMetrics, get_state_manager
    )
    from ...frameworks.langgraph.workflows import get_workflow_registry, get_conditional_logic

    # Import writing tools
    from ...tools.writing.content_writer import (
        ContentWriter, ContentRequest, ContentType as WriterContentType, 
        Tone, Style, GPTModel
    )
    from ...tools.writing.headline_generator import (
        HeadlineGenerator, HeadlineRequest, HeadlineStyle, HeadlineTone, Platform
    )
    from ...tools.writing.image_generator import image_generator_tool
    from ...tools.editing.sentiment_analyzer import (
        SentimentAnalyzer, SentimentAnalysisRequest, BrandVoice
    )
    from ...tools.analysis.content_analysis import content_analysis_tool
except ImportError:
    # Handle relative import issues when running from different contexts
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from core.errors import AgentError, WorkflowError, ToolError
    from core.logging import get_framework_logger
    from core.monitoring import get_metrics_collector
    from frameworks.langgraph.config import get_langgraph_framework
    from frameworks.langgraph.state import (
        ContentCreationState, StateStatus, ContentType, QualityMetrics, get_state_manager
    )
    from frameworks.langgraph.workflows import get_workflow_registry, get_conditional_logic

    # Import writing tools
    from tools.writing.content_writer import (
        ContentWriter, ContentRequest, ContentType as WriterContentType, 
        Tone, Style, GPTModel
    )
    from tools.writing.headline_generator import (
        HeadlineGenerator, HeadlineRequest, HeadlineStyle, HeadlineTone, Platform
    )
    from tools.writing.image_generator import image_generator_tool
    from tools.editing.sentiment_analyzer import (
        SentimentAnalyzer, SentimentAnalysisRequest, BrandVoice
    )
    from tools.analysis.content_analysis import content_analysis_tool

logger = get_framework_logger("Writer")


class WriterAgentConfig:
    """Configuration for Writer Agent"""
    
    def __init__(
        self,
        max_revisions: int = 3,
        quality_threshold: float = 0.8,
        enable_human_review: bool = True,
        enable_image_generation: bool = True,
        default_model: str = "gpt-4o",
        default_temperature: float = 0.7
    ):
        self.max_revisions = max_revisions
        self.quality_threshold = quality_threshold
        self.enable_human_review = enable_human_review
        self.enable_image_generation = enable_image_generation
        self.default_model = default_model
        self.default_temperature = default_temperature


class WriterAgent:
    """
    Writer Agent using LangGraph for complex content creation workflows
    
    Features:
    - Iterative content creation with revision loops
    - Research analysis and insight extraction  
    - Outline generation with structure optimization
    - Content writing with multiple revision cycles
    - Self-review and improvement loops
    - Quality gates and approval processes
    - Content type adaptation
    - Human escalation for complex issues
    """
    
    def __init__(self, config: Optional[WriterAgentConfig] = None):
        """
        Initialize Writer Agent
        
        Args:
            config: Writer agent configuration
        """
        self.logger = get_framework_logger("WriterAgent")
        self.metrics = get_metrics_collector()
        self.config = config or WriterAgentConfig()
        
        # Initialize tools
        self.content_writer = ContentWriter()
        self.headline_generator = HeadlineGenerator()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Framework components
        self.langgraph_framework = None
        self.workflow_registry = get_workflow_registry()
        self.conditional_logic = get_conditional_logic()
        self.state_manager = get_state_manager()
        
        # Workflow graph
        self.content_creation_graph = None
        self.is_initialized = False
        
        self.logger.info("WriterAgent initialized")

    async def initialize(self):
        """Initialize the Writer Agent and LangGraph framework"""
        if self.is_initialized:
            self.logger.warning("WriterAgent already initialized")
            return
        
        try:
            self.logger.info("Initializing WriterAgent with LangGraph framework...")
            
            # Initialize LangGraph framework
            self.langgraph_framework = await get_langgraph_framework()
            
            # Create content creation workflow
            await self._create_content_creation_workflow()
            
            self.is_initialized = True
            self.logger.info("WriterAgent initialized successfully")
            self.metrics.record_counter("agent_initialized", agent="writer")
            
        except Exception as e:
            error_msg = f"Failed to initialize WriterAgent: {e}"
            self.logger.error(error_msg)
            raise AgentError(error_msg) from e

    async def _create_content_creation_workflow(self):
        """Create the content creation workflow using LangGraph templates"""
        
        try:
            # Define node functions for the workflow
            node_functions = {
                "analyze_research_data": self._analyze_research_data,
                "generate_content_outline": self._generate_content_outline,
                "generate_content": self._generate_content,
                "review_content_quality": self._review_content_quality,
                "escalate_to_human": self._escalate_to_human,
                "finalize_content_output": self._finalize_content_output
            }
            
            # Define condition functions for workflow logic
            condition_functions = {
                "should_revise": self.conditional_logic.should_revise,
                "needs_human_review": self.conditional_logic.needs_human_review,
                "quality_approved": self.conditional_logic.quality_approved
            }
            
            # Create graph from template
            workflow_graph = await self.workflow_registry.create_graph_from_template(
                template_id="content_creation_workflow",
                node_functions=node_functions,
                condition_functions=condition_functions
            )
            
            # Register and compile the graph
            self.langgraph_framework.register_graph("content_creation", workflow_graph)
            self.content_creation_graph = self.langgraph_framework.compile_graph("content_creation")
            
            self.logger.info("Content creation workflow created and compiled")
            
        except Exception as e:
            raise AgentError(f"Failed to create content creation workflow: {e}")

    # Node Functions for LangGraph Workflow

    async def _analyze_research_data(self, state: ContentCreationState) -> ContentCreationState:
        """
        Analyze research data and extract key insights for content creation
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with research insights
        """
        self.logger.info(f"Analyzing research data for topic: {state.get('topic')}")
        
        try:
            research_data = state.get("research_data")
            if not research_data:
                # Create minimal research data if none provided
                research_data = {
                    "sources": [],
                    "key_findings": [f"Primary topic: {state.get('topic')}"],
                    "trends": [],
                    "competitive_analysis": {},
                    "fact_checks": []
                }
            
            # Extract insights from research data
            insights = await self._extract_research_insights(research_data, state)
            
            # Update state
            updated_state = state.copy()
            updated_state.update({
                "research_insights": insights,
                "research_status": "completed",
                "current_step": "create_outline",
                "completed_steps": state.get("completed_steps", []) + ["analyze_research"]
            })
            
            self.logger.info("Research data analysis completed")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Research analysis failed: {e}")
            updated_state = state.copy()
            updated_state.update({
                "workflow_status": StateStatus.FAILED,
                "failed_steps": state.get("failed_steps", []) + ["analyze_research"],
                "error": str(e)
            })
            return updated_state

    async def _extract_research_insights(
        self, 
        research_data: Dict[str, Any], 
        state: ContentCreationState
    ) -> Dict[str, Any]:
        """Extract actionable insights from research data"""
        
        insights = {
            "key_themes": [],
            "supporting_data": [],
            "content_angles": [],
            "audience_interests": [],
            "competitive_gaps": []
        }
        
        try:
            # Extract key themes from findings
            key_findings = research_data.get("key_findings", [])
            if key_findings:
                insights["key_themes"] = key_findings[:5]  # Top 5 themes
            
            # Process sources for supporting data
            sources = research_data.get("sources", [])
            if sources:
                insights["supporting_data"] = [
                    {
                        "source": source.get("title", "Unknown"),
                        "data": source.get("content", "")[:200] + "..." if source.get("content", "") else "",
                        "credibility": source.get("credibility_score", 0.5)
                    }
                    for source in sources[:3]  # Top 3 sources
                ]
            
            # Generate content angles based on topic and content type
            content_type = state.get("content_type")
            topic = state.get("topic", "")
            
            if content_type == ContentType.BLOG_POST:
                insights["content_angles"] = [
                    f"Comprehensive guide to {topic}",
                    f"Expert insights on {topic}",
                    f"Latest trends in {topic}",
                    f"Common mistakes in {topic}"
                ]
            elif content_type == ContentType.ARTICLE:
                insights["content_angles"] = [
                    f"In-depth analysis of {topic}",
                    f"Research findings on {topic}",
                    f"Industry perspective on {topic}"
                ]
            else:
                insights["content_angles"] = [f"Key information about {topic}"]
            
            # Analyze competitive gaps
            competitive_analysis = research_data.get("competitive_analysis", {})
            if competitive_analysis:
                insights["competitive_gaps"] = [
                    "Opportunity for unique perspective",
                    "Missing detailed explanation",
                    "Lack of practical examples"
                ]
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"Insight extraction failed: {e}")
            return insights

    async def _generate_content_outline(self, state: ContentCreationState) -> ContentCreationState:
        """
        Generate structured content outline based on research insights
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with content outline
        """
        self.logger.info("Generating content outline")
        
        try:
            # Get research insights and requirements
            research_insights = state.get("research_insights", {})
            topic = state.get("topic", "")
            content_type = state.get("content_type", ContentType.BLOG_POST)
            requirements = state.get("requirements", {})
            
            # Generate outline using content writer
            outline_request = ContentRequest(
                topic=f"Create detailed outline for: {topic}",
                content_type=WriterContentType.BLOG_POST,  # Use blog post template for outlines
                tone=Tone.PROFESSIONAL,
                style=Style.JOURNALISTIC,
                target_length=500,
                key_points=research_insights.get("key_themes", [])[:5],
                model=GPTModel.GPT_4O,
                temperature=0.5,  # Lower temperature for structured outline
                custom_instructions="Create a detailed, structured outline with main headings and subpoints. Focus on logical flow and comprehensive coverage of the topic."
            )
            
            outline_result = await self.content_writer.generate_content(outline_request)
            outline_content = outline_result.content
            
            # Update state
            updated_state = state.copy()
            updated_state.update({
                "outline": outline_content,
                "outline_quality_score": outline_result.quality_score / 100,
                "current_step": "write_content", 
                "completed_steps": state.get("completed_steps", []) + ["create_outline"]
            })
            
            self.logger.info("Content outline generated successfully")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Outline generation failed: {e}")
            updated_state = state.copy()
            updated_state.update({
                "workflow_status": StateStatus.FAILED,
                "failed_steps": state.get("failed_steps", []) + ["create_outline"],
                "error": str(e)
            })
            return updated_state

    async def _generate_content(self, state: ContentCreationState) -> ContentCreationState:
        """
        Generate main content based on outline and research
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with generated content
        """
        self.logger.info("Generating main content")
        
        try:
            # Get content parameters
            topic = state.get("topic", "")
            outline = state.get("outline", "")
            research_insights = state.get("research_insights", {})
            content_type = state.get("content_type", ContentType.BLOG_POST)
            requirements = state.get("requirements", {})
            
            # Map ContentType to WriterContentType
            writer_content_type = self._map_content_type(content_type)
            
            # Determine target length based on content type
            target_length = self._determine_target_length(content_type, requirements)
            
            # Build context data from research
            context_data = {
                "outline": outline,
                "key_themes": research_insights.get("key_themes", []),
                "supporting_data": research_insights.get("supporting_data", []),
                "content_angles": research_insights.get("content_angles", [])
            }
            
            # Generate content
            content_request = ContentRequest(
                topic=topic,
                content_type=writer_content_type,
                tone=self._determine_tone(requirements),
                style=self._determine_style(content_type, requirements),
                target_length=target_length,
                target_audience=requirements.get("target_audience"),
                key_points=research_insights.get("key_themes", [])[:7],
                keywords=requirements.get("keywords", []),
                context_data=context_data,
                model=GPTModel(self.config.default_model),
                temperature=self.config.default_temperature,
                include_outline=False,  # We already have an outline
                include_meta=True
            )
            
            content_result = await self.content_writer.generate_content(content_request)
            
            # Generate headlines for the content
            headlines = await self._generate_headlines(topic, content_type, requirements)
            
            # Update state
            updated_state = state.copy()
            updated_state.update({
                "content": content_result.content,
                "content_title": content_result.title,
                "content_meta": {
                    "meta_description": content_result.meta_description,
                    "tags": content_result.tags,
                    "word_count": content_result.word_count,
                    "reading_time": content_result.estimated_reading_time
                },
                "headlines": headlines,
                "content_quality_initial": content_result.quality_score / 100,
                "current_step": "self_review",
                "completed_steps": state.get("completed_steps", []) + ["write_content"],
                "generation_stats": {
                    "generation_time": content_result.generation_time,
                    "model_used": content_result.model_used,
                    "tokens_used": content_result.prompt_tokens + content_result.completion_tokens,
                    "cost": content_result.total_cost
                }
            })
            
            self.logger.info(f"Content generated: {content_result.word_count} words")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Content generation failed: {e}")
            updated_state = state.copy()
            updated_state.update({
                "workflow_status": StateStatus.FAILED,
                "failed_steps": state.get("failed_steps", []) + ["write_content"],
                "error": str(e)
            })
            return updated_state

    async def _generate_headlines(
        self, 
        topic: str, 
        content_type: ContentType, 
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate headlines for the content"""
        
        try:
            # Map content type to headline style
            headline_style = HeadlineStyle.QUESTION
            if content_type == ContentType.BLOG_POST:
                headline_style = HeadlineStyle.HOW_TO
            elif content_type == ContentType.ARTICLE:
                headline_style = HeadlineStyle.NEWS
            elif content_type == ContentType.SOCIAL_MEDIA:
                headline_style = HeadlineStyle.CURIOSITY
            
            headline_request = HeadlineRequest(
                topic=topic,
                style=headline_style,
                tone=HeadlineTone.PROFESSIONAL,
                platform=Platform.BLOG,
                num_variants=3,
                target_audience=requirements.get("target_audience"),
                keywords=requirements.get("keywords", [])[:5],
                include_numbers=True,
                temperature=0.8
            )
            
            headline_results = await self.headline_generator.generate_headlines(headline_request)
            
            return [
                {
                    "headline": h.headline,
                    "score": h.analysis.overall_score,
                    "predicted_ctr": h.analysis.predicted_ctr
                }
                for h in headline_results.headlines
            ]
            
        except Exception as e:
            self.logger.warning(f"Headline generation failed: {e}")
            return [{"headline": topic, "score": 70.0, "predicted_ctr": 2.5}]

    async def _review_content_quality(self, state: ContentCreationState) -> ContentCreationState:
        """
        Review content quality and determine if revisions are needed
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with quality assessment
        """
        self.logger.info("Reviewing content quality")
        
        try:
            content = state.get("content", "")
            if not content:
                raise WorkflowError("No content to review")
            
            # Perform comprehensive quality analysis
            quality_metrics = await self._assess_content_quality(content, state)
            
            # Check if content meets quality threshold
            revision_count = state.get("revision_count", 0)
            max_revisions = self.config.max_revisions
            
            # Update state with quality metrics
            updated_state = state.copy()
            updated_state.update({
                "quality_metrics": quality_metrics,
                "current_quality_score": quality_metrics.overall_score,
                "revision_count": revision_count,
                "current_step": "quality_decision",
                "completed_steps": state.get("completed_steps", []) + ["self_review"]
            })
            
            # Add revision history
            revisions = state.get("revisions", [])
            revisions.append({
                "revision_number": revision_count + 1,
                "quality_score": quality_metrics.overall_score,
                "issues": quality_metrics.issues,
                "recommendations": quality_metrics.recommendations,
                "timestamp": datetime.now().isoformat()
            })
            updated_state["revisions"] = revisions
            
            self.logger.info(f"Quality review completed. Score: {quality_metrics.overall_score}/100")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Quality review failed: {e}")
            updated_state = state.copy()
            updated_state.update({
                "workflow_status": StateStatus.FAILED,
                "failed_steps": state.get("failed_steps", []) + ["self_review"],
                "error": str(e)
            })
            return updated_state

    async def _assess_content_quality(
        self, 
        content: str, 
        state: ContentCreationState
    ) -> QualityMetrics:
        """Assess content quality using multiple analysis tools"""
        
        try:
            # Sentiment analysis
            sentiment_request = SentimentAnalysisRequest(
                text=content,
                target_brand_voice=BrandVoice.PROFESSIONAL,
                target_audience=state.get("requirements", {}).get("target_audience"),
                include_sentence_analysis=False
            )
            
            sentiment_result = await self.sentiment_analyzer.analyze_sentiment(sentiment_request)
            
            # Content analysis (using existing tool)
            content_analysis_result = await content_analysis_tool.analyze_content({
                "content": content,
                "analysis_type": "comprehensive"
            })
            
            # Calculate individual scores
            grammar_score = 85.0  # Placeholder - would use grammar checker
            readability_score = content_analysis_result.get("readability_score", 75.0)
            seo_score = 80.0  # Placeholder - would use SEO analyzer
            sentiment_score = (sentiment_result.overall_sentiment.polarity + 1) * 50  # Convert -1,1 to 0,100
            brand_alignment_score = sentiment_result.brand_voice_analysis.voice_consistency
            factual_accuracy_score = 80.0  # Placeholder - would use fact checker
            
            # Calculate overall score
            scores = [
                grammar_score, readability_score, seo_score, 
                sentiment_score, brand_alignment_score, factual_accuracy_score
            ]
            overall_score = sum(scores) / len(scores)
            
            # Identify issues and recommendations
            issues = []
            recommendations = []
            
            if grammar_score < 80:
                issues.append("Grammar and style issues detected")
                recommendations.append("Review and fix grammar errors")
            
            if readability_score < 70:
                issues.append("Content may be difficult to read")
                recommendations.append("Simplify sentence structure and vocabulary")
            
            if sentiment_score < 40:
                issues.append("Negative sentiment may impact reader engagement")
                recommendations.append("Add more positive language and tone")
            
            if len(sentiment_result.warnings) > 0:
                issues.extend(sentiment_result.warnings)
                recommendations.extend(sentiment_result.recommendations)
            
            quality_metrics = QualityMetrics(
                grammar_score=grammar_score,
                readability_score=readability_score,
                seo_score=seo_score,
                sentiment_score=sentiment_score,
                brand_alignment_score=brand_alignment_score,
                factual_accuracy_score=factual_accuracy_score,
                overall_score=overall_score,
                issues=issues,
                recommendations=recommendations
            )
            
            quality_metrics.calculate_overall_score()
            return quality_metrics
            
        except Exception as e:
            self.logger.warning(f"Quality assessment partially failed: {e}")
            # Return basic quality metrics
            return QualityMetrics(
                grammar_score=75.0,
                readability_score=75.0,
                seo_score=75.0,
                sentiment_score=75.0,
                brand_alignment_score=75.0,
                factual_accuracy_score=75.0,
                overall_score=75.0,
                issues=["Quality assessment incomplete"],
                recommendations=["Manual review recommended"]
            )

    async def _escalate_to_human(self, state: ContentCreationState) -> ContentCreationState:
        """
        Escalate content to human for review
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with human review request
        """
        self.logger.info("Escalating content to human reviewer")
        
        try:
            # Create human review request
            review_request = {
                "content": state.get("content"),
                "quality_metrics": state.get("quality_metrics"),
                "revision_history": state.get("revisions", []),
                "escalation_reason": self._determine_escalation_reason(state),
                "requested_at": datetime.now().isoformat(),
                "workflow_id": state.get("workflow_id")
            }
            
            # Update state
            updated_state = state.copy()
            updated_state.update({
                "human_review_request": review_request,
                "workflow_status": StateStatus.WAITING,
                "current_step": "awaiting_human_review",
                "completed_steps": state.get("completed_steps", []) + ["escalate_to_human"],
                "review_required": True
            })
            
            self.logger.info("Content escalated to human reviewer")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Human escalation failed: {e}")
            # Continue to finalization if escalation fails
            updated_state = state.copy()
            updated_state.update({
                "current_step": "finalize_content",
                "escalation_failed": True,
                "escalation_error": str(e)
            })
            return updated_state

    def _determine_escalation_reason(self, state: ContentCreationState) -> str:
        """Determine reason for human escalation"""
        
        quality_metrics = state.get("quality_metrics")
        if not quality_metrics:
            return "Quality assessment incomplete"
        
        if quality_metrics.overall_score < 60:
            return "Overall quality score below acceptable threshold"
        elif len(quality_metrics.issues) > 3:
            return "Multiple quality issues identified"
        elif state.get("revision_count", 0) >= self.config.max_revisions:
            return "Maximum revision attempts reached"
        elif state.get("requirements", {}).get("review_required", False):
            return "Review explicitly requested"
        else:
            return "Quality concerns require human judgment"

    async def _finalize_content_output(self, state: ContentCreationState) -> ContentCreationState:
        """
        Finalize content for delivery
        
        Args:
            state: Current workflow state
            
        Returns:
            Final state with completed content
        """
        self.logger.info("Finalizing content output")
        
        try:
            # Prepare final content package
            final_content = {
                "content": state.get("content"),
                "title": state.get("content_title"),
                "headlines": state.get("headlines", []),
                "meta": state.get("content_meta", {}),
                "outline": state.get("outline"),
                "quality_score": state.get("current_quality_score"),
                "revision_count": state.get("revision_count", 0),
                "generation_stats": state.get("generation_stats", {}),
                "workflow_summary": {
                    "completed_steps": state.get("completed_steps", []),
                    "total_time": state.get("execution_time", 0),
                    "quality_improvements": len(state.get("revisions", [])),
                    "final_status": "completed"
                }
            }
            
            # Update state
            updated_state = state.copy()
            updated_state.update({
                "final_content": final_content,
                "workflow_status": StateStatus.COMPLETED,
                "current_step": "completed",
                "completed_steps": state.get("completed_steps", []) + ["finalize_content"],
                "completion_time": datetime.now().isoformat()
            })
            
            self.logger.info("Content finalization completed")
            self.metrics.record_counter("content_created", agent="writer", success="true")
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Content finalization failed: {e}")
            updated_state = state.copy()
            updated_state.update({
                "workflow_status": StateStatus.FAILED,
                "failed_steps": state.get("failed_steps", []) + ["finalize_content"],
                "error": str(e)
            })
            self.metrics.record_counter("content_created", agent="writer", success="false")
            return updated_state

    # Helper Methods

    def _map_content_type(self, content_type: ContentType) -> WriterContentType:
        """Map ContentType to WriterContentType"""
        
        mapping = {
            ContentType.BLOG_POST: WriterContentType.BLOG_POST,
            ContentType.ARTICLE: WriterContentType.ARTICLE,
            ContentType.SOCIAL_MEDIA: WriterContentType.SOCIAL_MEDIA,
            ContentType.EMAIL: WriterContentType.EMAIL,
            ContentType.PRESS_RELEASE: WriterContentType.PRESS_RELEASE,
            ContentType.DOCUMENTATION: WriterContentType.TECHNICAL_DOCS
        }
        
        return mapping.get(content_type, WriterContentType.ARTICLE)

    def _determine_target_length(
        self, 
        content_type: ContentType, 
        requirements: Dict[str, Any]
    ) -> int:
        """Determine target content length based on type and requirements"""
        
        if "target_length" in requirements:
            return requirements["target_length"]
        
        # Default lengths by content type
        defaults = {
            ContentType.BLOG_POST: 1200,
            ContentType.ARTICLE: 2000,
            ContentType.SOCIAL_MEDIA: 150,
            ContentType.EMAIL: 400,
            ContentType.PRESS_RELEASE: 600,
            ContentType.DOCUMENTATION: 1500
        }
        
        return defaults.get(content_type, 1000)

    def _determine_tone(self, requirements: Dict[str, Any]) -> Tone:
        """Determine content tone from requirements"""
        
        tone_str = requirements.get("tone", "professional").lower()
        
        tone_mapping = {
            "professional": Tone.PROFESSIONAL,
            "casual": Tone.CASUAL,
            "conversational": Tone.CONVERSATIONAL,
            "formal": Tone.FORMAL,
            "friendly": Tone.FRIENDLY,
            "authoritative": Tone.AUTHORITATIVE,
            "humorous": Tone.HUMOROUS,
            "persuasive": Tone.PERSUASIVE,
            "informative": Tone.INFORMATIVE,
            "inspirational": Tone.INSPIRATIONAL
        }
        
        return tone_mapping.get(tone_str, Tone.PROFESSIONAL)

    def _determine_style(
        self, 
        content_type: ContentType, 
        requirements: Dict[str, Any]
    ) -> Style:
        """Determine writing style from content type and requirements"""
        
        style_str = requirements.get("style")
        if style_str:
            style_mapping = {
                "academic": Style.ACADEMIC,
                "journalistic": Style.JOURNALISTIC,
                "creative": Style.CREATIVE,
                "technical": Style.TECHNICAL,
                "marketing": Style.MARKETING,
                "storytelling": Style.STORYTELLING,
                "listicle": Style.LISTICLE,
                "how_to": Style.HOW_TO,
                "interview": Style.INTERVIEW,
                "review": Style.REVIEW
            }
            if style_str.lower() in style_mapping:
                return style_mapping[style_str.lower()]
        
        # Default styles by content type
        defaults = {
            ContentType.BLOG_POST: Style.JOURNALISTIC,
            ContentType.ARTICLE: Style.JOURNALISTIC,
            ContentType.SOCIAL_MEDIA: Style.MARKETING,
            ContentType.EMAIL: Style.MARKETING,
            ContentType.PRESS_RELEASE: Style.JOURNALISTIC,
            ContentType.DOCUMENTATION: Style.TECHNICAL
        }
        
        return defaults.get(content_type, Style.JOURNALISTIC)

    # Public API Methods

    async def create_content(
        self,
        topic: str,
        content_type: ContentType = ContentType.BLOG_POST,
        requirements: Optional[Dict[str, Any]] = None,
        research_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create content using the Writer Agent workflow
        
        Args:
            topic: Content topic
            content_type: Type of content to create
            requirements: Content requirements and parameters
            research_data: Optional research data to use
            
        Returns:
            Complete content creation results
        """
        if not self.is_initialized:
            await self.initialize()
        
        workflow_id = f"writer_{id(topic)}_{int(datetime.now().timestamp())}"
        
        try:
            self.logger.info(f"Starting content creation for: {topic}")
            
            # Create initial state
            initial_inputs = {
                "topic": topic,
                "content_type": content_type,
                "requirements": requirements or {},
                "research_data": research_data,
                "workflow_id": workflow_id,
                "review_required": requirements.get("review_required", False) if requirements else False
            }
            
            initial_state = self.state_manager.create_initial_state("content_creation", initial_inputs)
            
            # Execute workflow
            config = {"configurable": {"thread_id": workflow_id}}
            
            result_state = await self.langgraph_framework.execute_graph(
                "content_creation",
                initial_state,
                config=config
            )
            
            # Extract results
            if result_state.get("workflow_status") == StateStatus.COMPLETED:
                self.logger.info(f"Content creation completed for: {topic}")
                return {
                    "success": True,
                    "workflow_id": workflow_id,
                    "content": result_state.get("final_content"),
                    "state": result_state,
                    "execution_time": result_state.get("execution_time", 0),
                    "quality_score": result_state.get("current_quality_score", 0),
                    "revision_count": result_state.get("revision_count", 0)
                }
            else:
                self.logger.error(f"Content creation failed for: {topic}")
                return {
                    "success": False,
                    "workflow_id": workflow_id,
                    "error": result_state.get("error", "Unknown error"),
                    "state": result_state,
                    "failed_steps": result_state.get("failed_steps", [])
                }
                
        except Exception as e:
            self.logger.error(f"Content creation workflow failed: {e}")
            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e),
                "error_type": type(e).__name__
            }

    async def continue_workflow(
        self, 
        workflow_id: str, 
        human_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Continue a paused workflow (e.g., after human review)
        
        Args:
            workflow_id: ID of the workflow to continue
            human_feedback: Optional feedback from human reviewer
            
        Returns:
            Updated workflow results
        """
        try:
            self.logger.info(f"Continuing workflow: {workflow_id}")
            
            # This would typically involve resuming from a checkpoint
            # For now, we'll return a placeholder response
            return {
                "success": True,
                "workflow_id": workflow_id,
                "status": "continued",
                "message": "Workflow continuation not yet implemented"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to continue workflow {workflow_id}: {e}")
            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e)
            }

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get current status of a workflow
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Current workflow status and progress
        """
        try:
            # This would typically query the workflow state
            # For now, we'll return a placeholder response
            return {
                "success": True,
                "workflow_id": workflow_id,
                "status": "active",
                "current_step": "unknown",
                "progress": 50.0,
                "message": "Status checking not yet implemented"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow status {workflow_id}: {e}")
            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e)
            }

    async def shutdown(self):
        """Shutdown the Writer Agent"""
        try:
            self.logger.info("Shutting down WriterAgent...")
            
            if self.langgraph_framework:
                await self.langgraph_framework.shutdown()
            
            self.is_initialized = False
            self.logger.info("WriterAgent shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during WriterAgent shutdown: {e}")


# Global Writer Agent instance
_writer_agent: Optional[WriterAgent] = None


async def get_writer_agent(config: Optional[WriterAgentConfig] = None) -> WriterAgent:
    """Get the global Writer Agent instance"""
    global _writer_agent
    
    if _writer_agent is None:
        _writer_agent = WriterAgent(config)
        await _writer_agent.initialize()
    
    return _writer_agent


async def shutdown_writer_agent():
    """Shutdown the global Writer Agent instance"""
    global _writer_agent
    
    if _writer_agent:
        await _writer_agent.shutdown()
        _writer_agent = None