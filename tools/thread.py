"""Thread tool for adding posts to Bluesky threads atomically."""
from typing import Optional
from pydantic import BaseModel, Field, validator


class ReplyThreadPostArgs(BaseModel):
    text: str = Field(
        ..., 
        description="Text content for the post (max 300 characters)"
    )
    lang: Optional[str] = Field(
        default="en-US",
        description="Language code for the post (e.g., 'en-US', 'es', 'ja', 'th'). Defaults to 'en-US'"
    )
    
    @validator('text')
    def validate_text_length(cls, v):
        if len(v) > 300:
            raise ValueError(f"Text exceeds 300 character limit (current: {len(v)} characters)")
        return v


def add_post_to_bluesky_reply_thread(text: str, lang: str = "en-US") -> str:
    """
    Add a single post to the current Bluesky reply thread. This tool indicates to the handler 
    that it should add this post to the ongoing reply thread context when responding to a notification.
    
    This is distinct from bluesky_reply which handles the complete reply process. Use this tool 
    when you want to build a reply thread incrementally, adding posts one at a time.
    
    This is an atomic operation - each call adds exactly one post. The handler (bsky.py)
    manages the thread state and ensures proper threading when multiple posts are queued.

    Args:
        text: Text content for the post (max 300 characters)
        lang: Language code for the post (e.g., 'en-US', 'es', 'ja', 'th'). Defaults to 'en-US'

    Returns:
        Confirmation message that the post has been queued for the reply thread

    Raises:
        Exception: If text exceeds character limit. On failure, the post will be omitted 
                  from the reply thread and the agent may try again with corrected text.
    """
    # Validate input
    if len(text) > 300:
        raise Exception(f"Text exceeds 300 character limit (current: {len(text)} characters). This post will be omitted from the thread. You may try again with shorter text.")
    
    # Return confirmation - the actual posting will be handled by bsky.py
    return f"Post queued for reply thread: {text[:50]}{'...' if len(text) > 50 else ''} (Language: {lang})"