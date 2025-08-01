"""
Chat module for LLM Orchestrator

This module handles chat functionality including:
- Chat endpoint logic
- Gemini function calling integration
- Legacy keyword-based tool detection
"""

from .handler import ChatHandler
from .function_calling import FunctionCallManager

__all__ = ["ChatHandler", "FunctionCallManager"]