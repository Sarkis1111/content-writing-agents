"""
Editor Agent Module

This module provides the Editor Agent implementation using LangGraph framework
for multi-stage content editing and quality assurance workflows.
"""

from .editor_agent import (
    EditorAgent,
    EditorAgentConfig,
    EditingResults,
    EditingIssue,
    EditingStage,
    EditingPriority,
    QualityGateResult
)

__all__ = [
    'EditorAgent',
    'EditorAgentConfig', 
    'EditingResults',
    'EditingIssue',
    'EditingStage',
    'EditingPriority',
    'QualityGateResult'
]

__version__ = "0.1.0"