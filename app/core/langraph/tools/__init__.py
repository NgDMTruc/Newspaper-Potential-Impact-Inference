"""LangGraph tools for enhanced language model capabilities.

This package contains custom tools that can be used with LangGraph to extend
the capabilities of language models. Currently includes tools for web search
and other external integrations.
"""

from langchain_core.tools.base import BaseTool

from .duckduckgo_search import duckduckgo_search_tool
from .web_scraper import NewsImpactTool

tools: list[BaseTool] = [duckduckgo_search_tool, NewsImpactTool()]