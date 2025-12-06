"""Webpage fetch tool using Jina AI reader."""
from typing import Optional
from pydantic import BaseModel, Field


class WebpageArgs(BaseModel):
    url: str = Field(
        ..., 
        description="The URL of the webpage to fetch and convert to markdown/text format"
    )


def fetch_webpage(url: str) -> str:
    """
    Fetch a webpage and convert it to markdown/text format using Jina AI reader.
    
    Args:
        url: The URL of the webpage to fetch and convert
        
    Returns:
        String containing the webpage content in markdown/text format
    """
    import requests
    
    try:
        # Construct the Jina AI reader URL
        jina_url = f"https://r.jina.ai/{url}"
        
        # Make the request to Jina AI
        response = requests.get(jina_url, timeout=30)
        response.raise_for_status()
        
        return response.text
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching webpage: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")