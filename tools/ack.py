"""Annotation tool for stream.thought.ack records."""
from typing import Optional
from pydantic import BaseModel, Field


class AnnotateAckArgs(BaseModel):
    note: str = Field(
        ..., 
        description="A note or annotation to attach to the acknowledgment record"
    )


def annotate_ack(note: str) -> str:
    """
    Add a note to the acknowledgment record for the current post interaction.
    
    This is a "dummy" tool that doesn't directly create records but signals to the system
    that a note should be included in the stream.thought.ack record when acknowledging
    the post you're replying to.
    
    Args:
        note: A note or annotation to attach to the acknowledgment
        
    Returns:
        Confirmation message
    """
    # This is a dummy tool - it just returns a confirmation
    # The actual note will be captured by the bot loop and passed to acknowledge_post
    return f"Your note will be added to the acknowledgment: \"{note}\""