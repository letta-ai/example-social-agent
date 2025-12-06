"""Reply tool for Bluesky - a simple tool for the Letta agent to indicate a reply."""
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class ReplyArgs(BaseModel):
    messages: List[str] = Field(
        ..., 
        description="List of reply messages (each max 300 characters, max 4 messages total). Single item creates one reply, multiple items create a threaded reply chain."
    )
    lang: Optional[str] = Field(
        default="en-US",
        description="Language code for the posts (e.g., 'en-US', 'es', 'ja', 'th'). Defaults to 'en-US'"
    )
    
    @validator('messages')
    def validate_messages(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Messages list cannot be empty")
        if len(v) > 4:
            raise ValueError(f"Cannot send more than 4 reply messages (current: {len(v)} messages)")
        for i, message in enumerate(v):
            if len(message) > 300:
                raise ValueError(f"Message {i+1} cannot be longer than 300 characters (current: {len(message)} characters)")
        return v


def bluesky_reply(messages: List[str], lang: str = "en-US") -> str:
    """
    This is a simple function that returns a string indicating reply thread will be sent.
    
    Args:
        messages: List of reply texts (each max 300 characters, max 4 messages total)
        lang: Language code for the posts (e.g., 'en-US', 'es', 'ja', 'th'). Defaults to 'en-US'
        
    Returns:
        Confirmation message with language info and message count
        
    Raises:
        Exception: If messages list is invalid or messages exceed limits
    """
    if not messages or len(messages) == 0:
        raise Exception("Messages list cannot be empty")
    if len(messages) > 4:
        raise Exception(f"Cannot send more than 4 reply messages (current: {len(messages)} messages)")
    
    for i, message in enumerate(messages):
        if len(message) > 300:
            raise Exception(f"Message {i+1} cannot be longer than 300 characters (current: {len(message)} characters)")
    
    if len(messages) == 1:
        return f'Reply sent (language: {lang})'
    else:
        return f'Reply thread with {len(messages)} messages sent (language: {lang})'