"""Post tool for creating Bluesky posts."""
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class PostArgs(BaseModel):
    text: List[str] = Field(
        ..., 
        description="List of texts to create posts (each max 300 characters). Single item creates one post, multiple items create a thread."
    )
    lang: Optional[str] = Field(
        default="en-US",
        description="Language code for the posts (e.g., 'en-US', 'es', 'ja', 'th'). Defaults to 'en-US'"
    )
    
    @validator('text')
    def validate_text_list(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Text list cannot be empty")
        return v


def create_new_bluesky_post(text: List[str], lang: str = "en-US") -> str:
    """
    Create a NEW standalone post on Bluesky. This tool creates independent posts that
    start new conversations.

    IMPORTANT: This tool is ONLY for creating new posts. To reply to an existing post,
    use reply_to_bluesky_post instead.

    Args:
        text: List of post contents (each max 300 characters). Single item creates one post, multiple items create a thread.
        lang: Language code for the posts (e.g., 'en-US', 'es', 'ja', 'th'). Defaults to 'en-US'

    Returns:
        Success message with post URL(s)

    Raises:
        Exception: If the post fails or list is empty
    """
    import os
    import requests
    from datetime import datetime, timezone
    
    try:
        # Validate input
        if not text or len(text) == 0:
            raise Exception("Text list cannot be empty")
        
        # Validate character limits for all posts
        for i, post_text in enumerate(text):
            if len(post_text) > 300:
                raise Exception(f"Post {i+1} exceeds 300 character limit (current: {len(post_text)} characters)")
        
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
        
        if not access_token or not user_did:
            raise Exception("Failed to get access token or DID from session")
        
        # Create posts (single or thread)
        import re
        headers = {"Authorization": f"Bearer {access_token}"}
        create_record_url = f"{pds_host}/xrpc/com.atproto.repo.createRecord"
        
        post_urls = []
        previous_post = None
        root_post = None
        
        for i, post_text in enumerate(text):
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
            post_record = {
                "$type": "app.bsky.feed.post",
                "text": post_text,
                "createdAt": now,
                "langs": [lang]
            }
            
            # If this is part of a thread (not the first post), add reply references
            if previous_post:
                post_record["reply"] = {
                    "root": root_post,
                    "parent": previous_post
                }
            
            # Add facets for mentions and URLs
            facets = []
            
            # Parse mentions - fixed to handle @ at start of text
            mention_regex = rb"(?:^|[$|\W])(@([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)"
            text_bytes = post_text.encode("UTF-8")
            
            for m in re.finditer(mention_regex, text_bytes):
                handle = m.group(1)[1:].decode("UTF-8")  # Remove @ prefix
                # Adjust byte positions to account for the optional prefix
                mention_start = m.start(1)
                mention_end = m.end(1)
                try:
                    resolve_resp = requests.get(
                        f"{pds_host}/xrpc/com.atproto.identity.resolveHandle",
                        params={"handle": handle},
                        timeout=5
                    )
                    if resolve_resp.status_code == 200:
                        did = resolve_resp.json()["did"]
                        facets.append({
                            "index": {
                                "byteStart": mention_start,
                                "byteEnd": mention_end,
                            },
                            "features": [{"$type": "app.bsky.richtext.facet#mention", "did": did}],
                        })
                except:
                    continue
            
            # Parse URLs - fixed to handle URLs at start of text
            url_regex = rb"(?:^|[$|\W])(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*[-a-zA-Z0-9@%_\+~#//=])?)"

            for m in re.finditer(url_regex, text_bytes):
                url = m.group(1).decode("UTF-8")
                # Adjust byte positions to account for the optional prefix
                url_start = m.start(1)
                url_end = m.end(1)
                facets.append({
                    "index": {
                        "byteStart": url_start,
                        "byteEnd": url_end,
                    },
                    "features": [{"$type": "app.bsky.richtext.facet#link", "uri": url}],
                })

            # Parse hashtags
            hashtag_regex = rb"(?:^|[$|\s])#([a-zA-Z0-9_]+)"

            for m in re.finditer(hashtag_regex, text_bytes):
                tag = m.group(1).decode("UTF-8")  # Get tag without # prefix
                # Get byte positions for the entire hashtag including #
                tag_start = m.start(0)
                # Adjust start if there's a space/prefix
                if text_bytes[tag_start:tag_start+1] in (b' ', b'$'):
                    tag_start += 1
                tag_end = m.end(0)
                facets.append({
                    "index": {
                        "byteStart": tag_start,
                        "byteEnd": tag_end,
                    },
                    "features": [{"$type": "app.bsky.richtext.facet#tag", "tag": tag}],
                })

            if facets:
                post_record["facets"] = facets
            
            # Create the post
            create_data = {
                "repo": user_did,
                "collection": "app.bsky.feed.post",
                "record": post_record
            }
            
            post_response = requests.post(create_record_url, headers=headers, json=create_data, timeout=10)
            post_response.raise_for_status()
            result = post_response.json()
            
            post_uri = result.get("uri")
            post_cid = result.get("cid")
            handle = session.get("handle", username)
            rkey = post_uri.split("/")[-1] if post_uri else ""
            post_url = f"https://bsky.app/profile/{handle}/post/{rkey}"
            post_urls.append(post_url)
            
            # Set up references for thread continuation
            previous_post = {"uri": post_uri, "cid": post_cid}
            if i == 0:
                root_post = previous_post
        
        # Return appropriate message based on single post or thread
        if len(text) == 1:
            return f"Successfully posted to Bluesky!\nPost URL: {post_urls[0]}\nText: {text[0]}\nLanguage: {lang}"
        else:
            urls_text = "\n".join([f"Post {i+1}: {url}" for i, url in enumerate(post_urls)])
            return f"Successfully created thread with {len(text)} posts!\n{urls_text}\nLanguage: {lang}"
        
    except Exception as e:
        raise Exception(f"Error posting to Bluesky: {str(e)}")