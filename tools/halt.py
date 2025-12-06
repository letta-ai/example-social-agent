"""Emergency halt tool for TERMINAL shutdown of the bot process."""
from pydantic import BaseModel, Field


class HaltArgs(BaseModel):
    reason: str = Field(
        default="User requested halt",
        description="CRITICAL: Explain why you are performing an EMERGENCY SHUTDOWN of the entire bot"
    )


def halt_activity(reason: str = "User requested halt") -> str:
    """
    ⚠️ EMERGENCY SHUTDOWN TOOL - TERMINATES THE ENTIRE BOT PROCESS IMMEDIATELY ⚠️

    WARNING: This is a TERMINAL operation that will:
    - IMMEDIATELY STOP all bot activity
    - TERMINATE the bsky.py process completely
    - STOP processing all notifications
    - HALT all agent operations

    ⛔ DO NOT USE unless facing a SEVERE situation such as:
    - Critical system errors requiring immediate shutdown
    - Severe abuse or security concerns
    - Explicit user command to shut down the bot
    - Dangerous malfunction that could cause harm

    This is NOT for:
    - Regular conversation endings
    - Declining to respond to a message
    - Taking a break or pausing
    - Routine operations

    Use ignore_notification() instead for declining to respond to individual messages.

    Args:
        reason: REQUIRED explanation for why emergency shutdown is necessary

    Returns:
        Emergency halt signal that triggers immediate bot termination
    """
    return f"🛑 EMERGENCY HALT INITIATED: {reason}"