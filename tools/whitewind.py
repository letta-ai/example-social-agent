"""Whitewind blog post creation tool."""
from typing import Optional
from pydantic import BaseModel, Field


class WhitewindPostArgs(BaseModel):
    title: str = Field(
        ..., 
        description="Title of the blog post"
    )
    content: str = Field(
        ..., 
        description="Main content of the blog post (Markdown supported)"
    )
    subtitle: Optional[str] = Field(
        default=None,
        description="Optional subtitle for the blog post"
    )


def create_whitewind_blog_post(title: str, content: str, subtitle: Optional[str] = None) -> str:
    """
    Create a new blog post on Whitewind.
    
    This tool creates blog posts using the com.whtwnd.blog.entry lexicon on the ATProto network.
    The posts are publicly visible and use the github-light theme.
    
    Args:
        title: Title of the blog post
        content: Main content of the blog post (Markdown supported)
        subtitle: Optional subtitle for the blog post
        
    Returns:
        Success message with the blog post URL
        
    Raises:
        Exception: If the post creation fails
    """
    import os
    import requests
    from datetime import datetime, timezone
    
    try:
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
        
        session_response = requests.post(session_url, json=session_data, timeout=10)
        session_response.raise_for_status()
        session = session_response.json()
        access_token = session.get("accessJwt")
        user_did = session.get("did")
        handle = session.get("handle", username)
        
        if not access_token or not user_did:
            raise Exception("Failed to get access token or DID from session")
        
        # Create blog post record
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        blog_record = {
            "$type": "com.whtwnd.blog.entry",
            "theme": "github-light",
            "title": title,
            "content": content,
            "createdAt": now,
            "visibility": "public"
        }
        
        # Add subtitle if provided
        if subtitle:
            blog_record["subtitle"] = subtitle
        
        # Create the record
        headers = {"Authorization": f"Bearer {access_token}"}
        create_record_url = f"{pds_host}/xrpc/com.atproto.repo.createRecord"
        
        create_data = {
            "repo": user_did,
            "collection": "com.whtwnd.blog.entry",
            "record": blog_record
        }
        
        post_response = requests.post(create_record_url, headers=headers, json=create_data, timeout=10)
        post_response.raise_for_status()
        result = post_response.json()
        
        # Extract the record key from the URI
        post_uri = result.get("uri")
        if post_uri:
            rkey = post_uri.split("/")[-1]
            # Construct the Whitewind blog URL
            blog_url = f"https://whtwnd.com/{handle}/{rkey}"
        else:
            blog_url = "URL generation failed"
        
        # Build success message
        success_parts = [
            f"Successfully created Whitewind blog post!",
            f"Title: {title}"
        ]
        if subtitle:
            success_parts.append(f"Subtitle: {subtitle}")
        success_parts.extend([
            f"URL: {blog_url}",
            f"Theme: github-light",
            f"Visibility: public"
        ])
        
        return "\n".join(success_parts)
        
    except Exception as e:
        raise Exception(f"Error creating Whitewind blog post: {str(e)}")
