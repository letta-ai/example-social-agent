"""Flag archival memory for deletion tool."""
from pydantic import BaseModel, Field


class FlagArchivalMemoryForDeletionArgs(BaseModel):
    reason: str = Field(
        ...,
        description="The reason why this memory should be deleted"
    )
    memory_text: str = Field(
        ...,
        description="The exact text content of the archival memory to delete"
    )
    confirm: bool = Field(
        ...,
        description="Confirmation that you want to delete this memory (must be true to proceed)"
    )


def flag_archival_memory_for_deletion(reason: str, memory_text: str, confirm: bool) -> str:
    """
    Flag an archival memory for deletion based on its exact text content.

    This is a "dummy" tool that doesn't directly delete memories but signals to the system
    that the specified memory should be deleted at the end of the turn (if no halt_activity
    has been received).

    The system will search for all archival memories with this exact text and delete them.

    IMPORTANT: If multiple archival memories have identical text, ALL of them will be deleted.
    Make sure the memory_text is unique enough to avoid unintended deletions.

    Args:
        reason: The reason why this memory should be deleted
        memory_text: The exact text content of the archival memory to delete
        confirm: Confirmation that you want to delete this memory (must be true)

    Returns:
        Confirmation message
    """
    # This is a dummy tool - it just returns a confirmation
    # The actual deletion will be handled by the bot loop after the agent's turn completes
    if not confirm:
        return "Deletion cancelled - confirm must be set to true to delete the memory."

    return f"Memory flagged for deletion (reason: {reason}). Will be removed at the end of this turn if no halt is received."
