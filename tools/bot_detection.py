"""
Bot detection tools for checking known_bots memory block.
"""
import os
import random
import logging
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field
from letta_client import Letta

logger = logging.getLogger(__name__)


class CheckKnownBotsArgs(BaseModel):
    """Arguments for checking if users are in the known_bots list."""
    handles: List[str] = Field(..., description="List of user handles to check against known_bots")


def check_known_bots(handles: List[str], agent_state: "AgentState") -> str:
    """
    Check if any of the provided handles are in the known_bots memory block.
    
    Args:
        handles: List of user handles to check (e.g., ['horsedisc.bsky.social', 'user.bsky.social'])
        agent_state: The agent state object containing agent information
        
    Returns:
        JSON string with bot detection results
    """
    import json
    
    try:
        # Create Letta client inline (for cloud execution)
        client = Letta(api_key=os.environ["LETTA_API_KEY"])

        # Get all blocks attached to the agent to check if known_bots is mounted (SDK v1.0 returns page object)
        attached_blocks_page = client.agents.blocks.list(agent_id=str(agent_state.id))
        attached_blocks = attached_blocks_page.items if hasattr(attached_blocks_page, 'items') else attached_blocks_page
        attached_labels = {block.label for block in attached_blocks}
        
        if "known_bots" not in attached_labels:
            return json.dumps({
                "error": "known_bots memory block is not mounted to this agent",
                "bot_detected": False,
                "detected_bots": []
            })
        
        # Retrieve known_bots block content using agent-specific retrieval
        try:
            known_bots_block = client.agents.blocks.retrieve(
                agent_id=str(agent_state.id), 
                block_label="known_bots"
            )
        except Exception as e:
            return json.dumps({
                "error": f"Error retrieving known_bots block: {str(e)}",
                "bot_detected": False,
                "detected_bots": []
            })
        known_bots_content = known_bots_block.value
        
        # Parse known bots from content
        known_bot_handles = []
        for line in known_bots_content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract handle from lines like "- @handle.bsky.social" or "- @handle.bsky.social: description"
                if line.startswith('- @'):
                    handle = line[3:].split(':')[0].strip()
                    known_bot_handles.append(handle)
                elif line.startswith('-'):
                    handle = line[1:].split(':')[0].strip().lstrip('@')
                    known_bot_handles.append(handle)
        
        # Normalize handles for comparison
        normalized_input_handles = [h.lstrip('@').strip() for h in handles]
        normalized_bot_handles = [h.strip() for h in known_bot_handles]
        
        # Check for matches
        detected_bots = []
        for handle in normalized_input_handles:
            if handle in normalized_bot_handles:
                detected_bots.append(handle)
        
        bot_detected = len(detected_bots) > 0
        
        return json.dumps({
            "bot_detected": bot_detected,
            "detected_bots": detected_bots,
            "total_known_bots": len(normalized_bot_handles),
            "checked_handles": normalized_input_handles
        })
        
    except Exception as e:
        return json.dumps({
            "error": f"Error checking known_bots: {str(e)}",
            "bot_detected": False,
            "detected_bots": []
        })


def should_respond_to_bot_thread() -> bool:
    """
    Determine if we should respond to a bot thread (10% chance).
    
    Returns:
        True if we should respond, False if we should skip
    """
    return random.random() < 0.1


def extract_handles_from_thread(thread_data: dict) -> List[str]:
    """
    Extract all unique handles from a thread structure.
    
    Args:
        thread_data: Thread data dictionary from Bluesky API
        
    Returns:
        List of unique handles found in the thread
    """
    handles = set()
    
    def extract_from_post(post):
        """Recursively extract handles from a post and its replies."""
        if isinstance(post, dict):
            # Get author handle
            if 'post' in post and 'author' in post['post']:
                handle = post['post']['author'].get('handle')
                if handle:
                    handles.add(handle)
            elif 'author' in post:
                handle = post['author'].get('handle')
                if handle:
                    handles.add(handle)
            
            # Check replies
            if 'replies' in post:
                for reply in post['replies']:
                    extract_from_post(reply)
            
            # Check parent
            if 'parent' in post:
                extract_from_post(post['parent'])
    
    # Start extraction from thread root
    if 'thread' in thread_data:
        extract_from_post(thread_data['thread'])
    else:
        extract_from_post(thread_data)
    
    return list(handles)


# Tool configuration for registration
TOOL_CONFIG = {
    "type": "function",
    "function": {
        "name": "check_known_bots",
        "description": "Check if any of the provided handles are in the known_bots memory block",
        "parameters": CheckKnownBotsArgs.model_json_schema(),
    },
}