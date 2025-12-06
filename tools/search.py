"""Search tool for Bluesky posts."""
from pydantic import BaseModel, Field
from typing import Optional


class SearchArgs(BaseModel):
    query: str = Field(..., description="Search query string")
    max_results: int = Field(default=25, description="Maximum number of results to return (max 100)")
    author: Optional[str] = Field(None, description="Filter by author handle (e.g., 'user.bsky.social')")
    sort: str = Field(default="latest", description="Sort order: 'latest' or 'top'")


def search_bluesky_posts(query: str, max_results: int = 25, author: str = None, sort: str = "latest") -> str:
    """
    Search for posts on Bluesky matching the given criteria.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (max 100)
        author: Filter by author handle (e.g., 'user.bsky.social')
        sort: Sort order: 'latest' or 'top'
        
    Returns:
        YAML-formatted search results with posts and metadata
    """
    import os
    import yaml
    import requests
    from datetime import datetime
    
    try:
        # Validate inputs
        max_results = min(max_results, 100)
        if sort not in ["latest", "top"]:
            sort = "latest"
        
        # Build search query
        search_query = query
        if author:
            search_query = f"from:{author} {query}"
        
        # Get credentials from environment
        username = os.getenv("BSKY_USERNAME")
        password = os.getenv("BSKY_PASSWORD")
        pds_host = os.getenv("PDS_URI", "https://bsky.social")
        
        if not username or not password:
            raise Exception("BSKY_USERNAME and BSKY_PASSWORD environment variables must be set")
        
        # Create session
        session_url = f"{pds_host}/xrpc/com.atproto.server.createSession"
        session_data = {
            "identifier": username,
            "password": password
        }
        
        try:
            session_response = requests.post(session_url, json=session_data, timeout=10)
            session_response.raise_for_status()
            session = session_response.json()
            access_token = session.get("accessJwt")
            
            if not access_token:
                raise Exception("Failed to get access token from session")
        except Exception as e:
            raise Exception(f"Authentication failed. ({str(e)})")
        
        # Search posts
        headers = {"Authorization": f"Bearer {access_token}"}
        search_url = f"{pds_host}/xrpc/app.bsky.feed.searchPosts"
        params = {
            "q": search_query,
            "limit": max_results,
            "sort": sort
        }
        
        try:
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            search_data = response.json()
        except Exception as e:
            raise Exception(f"Search failed. ({str(e)})")
        
        # Format results
        results = []
        for post in search_data.get("posts", []):
            author = post.get("author", {})
            record = post.get("record", {})
            
            post_data = {
                "author": {
                    "handle": author.get("handle", ""),
                    "display_name": author.get("displayName", ""),
                },
                "text": record.get("text", ""),
                "created_at": record.get("createdAt", ""),
                "uri": post.get("uri", ""),
                "cid": post.get("cid", ""),
                "like_count": post.get("likeCount", 0),
                "repost_count": post.get("repostCount", 0),
                "reply_count": post.get("replyCount", 0),
            }
            
            # Add reply info if present
            if "reply" in record and record["reply"]:
                post_data["reply_to"] = {
                    "uri": record["reply"].get("parent", {}).get("uri", ""),
                    "cid": record["reply"].get("parent", {}).get("cid", ""),
                }
            
            results.append(post_data)
        
        return yaml.dump({
            "search_results": {
                "query": query,
                "author_filter": author,
                "sort": sort,
                "result_count": len(results),
                "posts": results
            }
        }, default_flow_style=False, sort_keys=False)
        
    except Exception as e:
        raise Exception(f"Error searching Bluesky: {str(e)}")