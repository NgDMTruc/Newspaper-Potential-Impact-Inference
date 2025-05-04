import requests
from bs4 import BeautifulSoup
import re
from typing import Optional, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field

class NewsImpactInput(BaseModel):
    """Input for news impact analysis."""
    url: str = Field(description="URL of the news article to analyze")
    field: str = Field(description="Field to analyze impact on (e.g., finance, technology)")

def extract_news_content(url: str, max_words: int = 500) -> Optional[str]:
    """
    Extract main content from a news article URL.
    
    Args:
        url (str): URL of the news article
        max_words (int, optional): Maximum number of words to extract. Defaults to 500.
    
    Returns:
        Optional[str]: Extracted article content or None if extraction fails
    """
    try:
        # Send a request with a user agent to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try multiple strategies to extract main content
        content_candidates = [
            soup.find('article'),
            soup.find('div', class_=re.compile(r'(article|content|main-content)', re.I)),
            soup.find('div', id=re.compile(r'(article|content|main-content)', re.I)),
            soup.find('body')
        ]
        
        # Find the first non-None content
        for candidate in content_candidates:
            if candidate:
                # Extract text, remove extra whitespace
                text = ' '.join(candidate.get_text(strip=True).split())
                
                # Limit to max_words
                words = text.split()[:max_words]
                return ' '.join(words)
        
        return None
    
    except (requests.RequestException, ValueError) as e:
        print(f"Error extracting content from {url}: {e}")
        return None

def analyze_news_impact(url: str, field: str, max_words: int = 500) -> Optional[str]:
    """
    Extract news content and prepare it for impact analysis.
    
    Args:
        url (str): URL of the news article
        field (str): Field to analyze impact on (e.g., finance, technology)
        max_words (int, optional): Maximum number of words to extract. Defaults to 500.
    
    Returns:
        Optional[str]: Prepared content for impact analysis or None if extraction fails
    """
    content = extract_news_content(url, max_words)
    if not content:
        return None
    
    # Prepare a prompt for impact analysis
    impact_prompt = f"""
    News Content: {content}
    
    Analyze the potential impact of this news on the {field} sector:
    - Identify key events or developments
    - Explain potential short-term and long-term consequences
    - Provide specific insights related to {field}
    """
    
    return impact_prompt

class NewsImpactTool(BaseTool):
    """Tool for analyzing news impact on a specific field."""
    
    name: str = "news_impact_analysis"
    description: str = "Analyze the potential impact of a news article on a specific field"
    args_schema: type[BaseModel] = NewsImpactInput

    def _run(
        self, 
        url: str, 
        field: str, 
        **kwargs: Any
    ) -> str:
        """
        Run the news impact analysis.
        
        Args:
            url (str): URL of the news article
            field (str): Field to analyze impact on
        
        Returns:
            str: Analysis of the news article's impact
        """
        # Call the analysis function
        result = analyze_news_impact(url, field)
        
        if result is None:
            return "Unable to extract or analyze the news article content."
        
        return result

    async def _arun(
        self, 
        url: str, 
        field: str, 
        **kwargs: Any
    ) -> str:
        """Async version of _run method."""
        return self._run(url, field, **kwargs)