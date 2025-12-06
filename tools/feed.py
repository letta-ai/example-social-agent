"""Feed tool for retrieving Bluesky feeds."""
from pydantic import BaseModel, Field
from typing import Optional


class FeedArgs(BaseModel):
    feed_name: Optional[str] = Field(None, description="Named feed preset. Available feeds: 'home' (timeline), 'discover' (what's hot), 'ai-for-grownups', 'atmosphere', 'void-cafe'. If not provided, returns home timeline")
    max_posts: int = Field(default=25, description="Maximum number of posts to retrieve (max 100)")


def get_bluesky_feed(feed_name: str = None, max_posts: int = 25) -> str:
    """
    Retrieve a Bluesky feed.
    
    Args:
        feed_name: Named feed preset - available options: 'home', 'discover', 'ai-for-grownups', 'atmosphere', 'void-cafe'
        max_posts: Maximum number of posts to retrieve (max 100)
        
    Returns:
        YAML-formatted feed data with posts and metadata
    """
    import os
    import yaml
    import requests
    
    try:
        # Predefined feed mappings (must be inside function for sandboxing)
        feed_presets = {
            "home": None,  # Home timeline (default)
            "discover": "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot",
            "ai-for-grownups": "at://did:plc:gfrmhdmjvxn2sjedzboeudef/app.bsky.feed.generator/ai-for-grownups", 
            "atmosphere": "at://did:plc:gfrmhdmjvxn2sjedzboeudef/app.bsky.feed.generator/the-atmosphere",
            "void-cafe": "at://did:plc:gfrmhdmjvxn2sjedzboeudef/app.bsky.feed.generator/void-cafe"
        }
        
        # Validate inputs
        max_posts = min(max_posts, 100)
        
        # Resolve feed URI from name
        if feed_name:
            # Handle case where agent passes 'FeedName.discover' instead of 'discover'
            if '.' in feed_name and feed_name.startswith('FeedName.'):
                feed_name = feed_name.split('.', 1)[1]
            
            # Look up named preset
            if feed_name not in feed_presets:
                available_feeds = list(feed_presets.keys())
                raise Exception(f"Invalid feed name '{feed_name}'. Available feeds: {available_feeds}")
            resolved_feed_uri = feed_presets[feed_name]
            feed_display_name = feed_name
        else:
            # Default to home timeline
            resolved_feed_uri = None
            feed_display_name = "home"
        
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
        
        # Get feed
        headers = {"Authorization": f"Bearer {access_token}"}
        
        if resolved_feed_uri:
            # Custom feed
            feed_url = f"{pds_host}/xrpc/app.bsky.feed.getFeed"
            params = {
                "feed": resolved_feed_uri,
                "limit": max_posts
            }
            feed_type = "custom"
        else:
            # Home timeline
            feed_url = f"{pds_host}/xrpc/app.bsky.feed.getTimeline"
            params = {
                "limit": max_posts
            }
            feed_type = "home"
        
        try:
            response = requests.get(feed_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            feed_data = response.json()
        except Exception as e:
            raise Exception(f"Failed to get feed. ({str(e)})")
        
        # Format posts
        posts = []
        for item in feed_data.get("feed", []):
            post = item.get("post", {})
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
            
            # Add repost info if present
            if "reason" in item and item["reason"]:
                reason = item["reason"]
                if reason.get("$type") == "app.bsky.feed.defs#reasonRepost":
                    by = reason.get("by", {})
                    post_data["reposted_by"] = {
                        "handle": by.get("handle", ""),
                        "display_name": by.get("displayName", ""),
                    }
            
            # Add reply info if present
            if "reply" in record and record["reply"]:
                parent = record["reply"].get("parent", {})
                post_data["reply_to"] = {
                    "uri": parent.get("uri", ""),
                    "cid": parent.get("cid", ""),
                }
            
            posts.append(post_data)
        
        # Format response
        feed_result = {
            "feed": {
                "type": feed_type,
                "name": feed_display_name,
                "post_count": len(posts),
                "posts": posts
            }
        }
        
        if resolved_feed_uri:
            feed_result["feed"]["uri"] = resolved_feed_uri
        
        return yaml.dump(feed_result, default_flow_style=False, sort_keys=False)
        
    except Exception as e:
        raise Exception(f"Error retrieving feed: {str(e)}")