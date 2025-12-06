"""Ignore notification tool for Bluesky."""
from pydantic import BaseModel, Field
from typing import Optional


class IgnoreNotificationArgs(BaseModel):
    reason: str = Field(..., description="Reason for ignoring this notification")
    category: Optional[str] = Field(
        default="bot",
        description="Category of ignored notification (e.g., 'bot', 'spam', 'not_relevant', 'handled_elsewhere')"
    )


def ignore_notification(reason: str, category: str = "bot") -> str:
    """
    Signal that the current notification should be ignored without a reply.
    
    This tool allows the agent to explicitly mark a notification as ignored
    rather than having it default to the no_reply folder. This is particularly
    useful for ignoring interactions from bots or spam accounts.
    
    Args:
        reason: Reason for ignoring this notification
        category: Category of ignored notification (default: 'bot')
        
    Returns:
        Confirmation message
    """
    return f"IGNORED_NOTIFICATION::{category}::{reason}"