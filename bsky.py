# Rich imports removed - using simple text formatting
from time import sleep
from letta_client import Letta
from bsky_utils import thread_to_yaml_string, count_thread_posts
import os
import logging
import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time
import argparse

from utils import (
    upsert_block,
    upsert_agent
)
from config_loader import get_letta_config, get_config, get_queue_config

import bsky_utils
from datetime import date
from notification_db import NotificationDB

def extract_handles_from_data(data):
    """Recursively extract all unique handles from nested data structure."""
    handles = set()
    
    def _extract_recursive(obj):
        if isinstance(obj, dict):
            # Check if this dict has a 'handle' key
            if 'handle' in obj:
                handles.add(obj['handle'])
            # Recursively check all values
            for value in obj.values():
                _extract_recursive(value)
        elif isinstance(obj, list):
            # Recursively check all list items
            for item in obj:
                _extract_recursive(item)
    
    _extract_recursive(data)
    return list(handles)


def extract_dids_from_data(data):
    """Recursively extract all unique DIDs from nested data structure."""
    dids = set()
    
    def _extract_recursive(obj):
        if isinstance(obj, dict):
            # Check if this dict has a 'did' key
            if 'did' in obj and isinstance(obj['did'], str) and obj['did'].startswith('did:'):
                dids.add(obj['did'])
            # Recursively check all values
            for value in obj.values():
                _extract_recursive(value)
        elif isinstance(obj, list):
            # Recursively check all list items
            for item in obj:
                _extract_recursive(item)
    
    _extract_recursive(data)
    return list(dids)

# Logging will be configured after argument parsing
logger = None
prompt_logger = None
# Simple text formatting (Rich no longer used)
SHOW_REASONING = False
last_archival_query = "archival memory search"

def log_with_panel(message, title=None, border_color="white"):
    """Log a message with Unicode box-drawing characters"""
    if title:
        # Map old color names to appropriate symbols
        symbol_map = {
            "blue": "⚙",      # Tool calls
            "green": "✓",     # Success/completion
            "yellow": "◆",    # Reasoning
            "red": "✗",       # Errors
            "white": "▶",     # Default/mentions
            "cyan": "✎",      # Posts
        }
        symbol = symbol_map.get(border_color, "▶")
        
        print(f"\n{symbol} {title}")
        print(f"  {'─' * len(title)}")
        # Indent message lines
        for line in message.split('\n'):
            print(f"  {line}")
    else:
        print(message)


# Load Letta configuration from config.yaml (will be initialized later with custom path if provided)
letta_config = None
CLIENT = None

# Notification check delay
FETCH_NOTIFICATIONS_DELAY_SEC = 10  # Check every 10 seconds for faster response

# Check for new notifications every N queue items
CHECK_NEW_NOTIFICATIONS_EVERY_N_ITEMS = 2  # Check more frequently during processing

# Queue paths (will be initialized from config in main())
QUEUE_DIR = None
QUEUE_ERROR_DIR = None
QUEUE_NO_REPLY_DIR = None
PROCESSED_NOTIFICATIONS_FILE = None

# Maximum number of processed notifications to track
MAX_PROCESSED_NOTIFICATIONS = 10000

# Message tracking counters
message_counters = defaultdict(int)
start_time = time.time()

# Testing mode flag
TESTING_MODE = False

# Synthesis message tracking
last_synthesis_time = time.time()

# Database for notification tracking
NOTIFICATION_DB = None

# Bot name from config (used for file naming, logging, etc.)
BOT_NAME = None

def initialize_agent():
    logger.info("Starting agent initialization...")

    # Get the configured agent by ID
    logger.info("Loading agent from config...")
    agent_id = letta_config['agent_id']

    try:
        void_agent = CLIENT.agents.retrieve(agent_id=agent_id)
        logger.info(f"Successfully loaded agent: {void_agent.name} ({agent_id})")
    except Exception as e:
        logger.error(f"Failed to load agent {agent_id}: {e}")
        logger.error("Please ensure the agent_id in config.yaml is correct")
        raise e

    # Log agent details
    logger.info(f"Void agent details - ID: {void_agent.id}")
    logger.info(f"Agent name: {void_agent.name}")
    if hasattr(void_agent, 'llm_config'):
        logger.info(f"Agent model: {void_agent.llm_config.model}")
    if hasattr(void_agent, 'project_id') and void_agent.project_id:
        logger.info(f"Agent project_id: {void_agent.project_id}")
    if hasattr(void_agent, 'tools'):
        logger.info(f"Agent has {len(void_agent.tools)} tools")
        for tool in void_agent.tools[:3]:  # Show first 3 tools
            logger.info(f"  - Tool: {tool.name} (type: {tool.tool_type})")

    return void_agent


def process_mention(void_agent, atproto_client, notification_data, queue_filepath=None, testing_mode=False):
    """Process a mention and generate a reply using the Letta agent.
    
    Args:
        void_agent: The Letta agent instance
        atproto_client: The AT Protocol client
        notification_data: The notification data dictionary
        queue_filepath: Optional Path object to the queue file (for cleanup on halt)
    
    Returns:
        True: Successfully processed, remove from queue
        False: Failed but retryable, keep in queue
        None: Failed with non-retryable error, move to errors directory
        "no_reply": No reply was generated, move to no_reply directory
    """
    import uuid
    
    # Generate correlation ID for tracking this notification through the pipeline
    correlation_id = str(uuid.uuid4())[:8]
    
    # Track attached user blocks for cleanup (initialize early for finally block)
    attached_user_labels = []
    
    try:
        logger.info(f"[{correlation_id}] Starting process_mention", extra={
            'correlation_id': correlation_id,
            'notification_type': type(notification_data).__name__
        })
        
        # Handle both dict and object inputs for backwards compatibility
        if isinstance(notification_data, dict):
            uri = notification_data['uri']
            mention_text = notification_data.get('record', {}).get('text', '')
            author_handle = notification_data['author']['handle']
            author_name = notification_data['author'].get('display_name') or author_handle
        else:
            # Legacy object access
            uri = notification_data.uri
            mention_text = notification_data.record.text if hasattr(notification_data.record, 'text') else ""
            author_handle = notification_data.author.handle
            author_name = notification_data.author.display_name or author_handle
        
        logger.info(f"[{correlation_id}] Processing mention from @{author_handle}", extra={
            'correlation_id': correlation_id,
            'author_handle': author_handle,
            'author_name': author_name,
            'mention_uri': uri,
            'mention_text_length': len(mention_text),
            'mention_preview': mention_text[:100] if mention_text else ''
        })

        # Check if handle is in allowed list (if configured)
        allowed_handles = get_config().get('bot.allowed_handles', [])
        if allowed_handles and author_handle not in allowed_handles:
            logger.info(f"[{correlation_id}] Skipping mention from @{author_handle} (not in allowed_handles)")
            return True  # Remove from queue

        # Retrieve the entire thread associated with the mention
        try:
            thread = atproto_client.app.bsky.feed.get_post_thread({
                'uri': uri,
                'parent_height': 40,
                'depth': 10
            })
        except Exception as e:
            error_str = str(e)
            # Check if this is a NotFound error
            if 'NotFound' in error_str or 'Post not found' in error_str:
                logger.warning(f"Post not found for URI {uri}, removing from queue")
                return True  # Return True to remove from queue
            elif 'InternalServerError' in error_str:
                # Bluesky sometimes returns InternalServerError for deleted posts
                # Verify if post actually exists using getRecord
                try:
                    parts = uri.replace('at://', '').split('/')
                    repo, collection, rkey = parts[0], parts[1], parts[2]
                    atproto_client.com.atproto.repo.get_record({
                        'repo': repo, 'collection': collection, 'rkey': rkey
                    })
                    # Post exists, this is a real server error - re-raise
                    logger.error(f"Error fetching thread (post exists, server error): {e}")
                    raise
                except Exception as verify_e:
                    if 'RecordNotFound' in str(verify_e) or 'not found' in str(verify_e).lower():
                        logger.warning(f"Post deleted (verified via getRecord), removing from queue: {uri}")
                        return True  # Remove from queue
                    # Some other verification error, re-raise original
                    logger.error(f"Error fetching thread: {e}")
                    raise
            else:
                # Re-raise other errors
                logger.error(f"Error fetching thread: {e}")
                raise

        # Check thread length against configured maximum
        max_thread_posts = get_config().get('bot.max_thread_posts', 0)
        if max_thread_posts > 0:
            thread_post_count = count_thread_posts(thread)
            if thread_post_count > max_thread_posts:
                logger.info(f"Thread too long ({thread_post_count} posts > {max_thread_posts} max), skipping this mention")
                return True  # Return True to remove from queue

        # Get thread context as YAML string
        logger.debug("Converting thread to YAML string")
        try:
            thread_context = thread_to_yaml_string(thread)
            logger.debug(f"Thread context generated, length: {len(thread_context)} characters")
            
            # Check if #voidstop appears anywhere in the thread
            if "#voidstop" in thread_context.lower():
                logger.info("Found #voidstop in thread context, skipping this mention")
                return True  # Return True to remove from queue
            
            # Also check the mention text directly
            if "#voidstop" in mention_text.lower():
                logger.info("Found #voidstop in mention text, skipping this mention")
                return True  # Return True to remove from queue
            
            # Create a more informative preview by extracting meaningful content
            lines = thread_context.split('\n')
            meaningful_lines = []
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                    
                # Look for lines with actual content (not just structure)
                if any(keyword in line for keyword in ['text:', 'handle:', 'display_name:', 'created_at:', 'reply_count:', 'like_count:']):
                    meaningful_lines.append(line)
                    if len(meaningful_lines) >= 5:
                        break
            
            if meaningful_lines:
                preview = '\n'.join(meaningful_lines)
                logger.debug(f"Thread content preview:\n{preview}")
            else:
                # If no content fields found, just show it's a thread structure
                logger.debug(f"Thread structure generated ({len(thread_context)} chars)")
        except Exception as yaml_error:
            import traceback
            logger.error(f"Error converting thread to YAML: {yaml_error}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            logger.error(f"Thread type: {type(thread)}")
            if hasattr(thread, '__dict__'):
                logger.error(f"Thread attributes: {thread.__dict__}")
            # Try to continue with a simple context
            thread_context = f"Error processing thread context: {str(yaml_error)}"

        # Create a prompt for the Letta agent with thread context
        prompt = f"""You received a mention on Bluesky from @{author_handle} ({author_name or author_handle}).

MOST RECENT POST:
"{mention_text}"

THREAD CONTEXT:
```yaml
{thread_context}
```

If you choose to reply, use the add_post_to_bluesky_reply_thread tool. Each call creates one post (max 300 characters). You may use multiple calls to create a thread if needed."""

        # Extract all handles and DIDs from notification and thread data
        all_handles = set()
        all_handles.update(extract_handles_from_data(notification_data))
        all_handles.update(extract_handles_from_data(thread.model_dump()))
        unique_handles = list(all_handles)
        
        all_dids = set()
        all_dids.update(extract_dids_from_data(notification_data))
        all_dids.update(extract_dids_from_data(thread.model_dump()))
        unique_dids = list(all_dids)
        
        logger.debug(f"Found {len(unique_handles)} unique handles in thread: {unique_handles}")
        logger.debug(f"Found {len(unique_dids)} unique DIDs in thread: {unique_dids}")
        
        # Check if any handles are in known_bots list
        from tools.bot_detection import check_known_bots, should_respond_to_bot_thread, CheckKnownBotsArgs
        import json
        
        try:
            # Check for known bots in thread
            bot_check_result = check_known_bots(unique_handles, void_agent)
            bot_check_data = json.loads(bot_check_result)
            
            # TEMPORARILY DISABLED: Bot detection causing issues with normal users
            # TODO: Re-enable after debugging why normal users are being flagged as bots
            if False:  # bot_check_data.get("bot_detected", False):
                detected_bots = bot_check_data.get("detected_bots", [])
                logger.info(f"Bot detected in thread: {detected_bots}")
                
                # Decide whether to respond (10% chance)
                if not should_respond_to_bot_thread():
                    logger.info(f"Skipping bot thread (90% skip rate). Detected bots: {detected_bots}")
                    # Return False to keep in queue for potential later processing
                    return False
                else:
                    logger.info(f"Responding to bot thread (10% response rate). Detected bots: {detected_bots}")
            else:
                logger.debug("Bot detection disabled - processing all notifications")
                
        except Exception as bot_check_error:
            logger.warning(f"Error checking for bots: {bot_check_error}")
            # Continue processing if bot check fails

        # Get response from Letta agent
        # Format with Unicode characters
        title = f"MENTION FROM @{author_handle}"
        print(f"\n▶ {title}")
        print(f"  {'═' * len(title)}")
        # Indent the mention text
        for line in mention_text.split('\n'):
            print(f"  {line}")
        
        # Log prompt details to separate logger
        prompt_logger.debug(f"Full prompt being sent:\n{prompt}")
        
        # Log concise prompt info to main logger
        thread_handles_count = len(unique_handles)
        prompt_char_count = len(prompt)
        logger.debug(f"Sending to LLM: @{author_handle} mention | msg: \"{mention_text[:50]}...\" | context: {len(thread_context)} chars, {thread_handles_count} users | prompt: {prompt_char_count} chars")

        chunk_count = 0
        try:
            # Attach user blocks before sending to agent
            if unique_dids:
                logger.debug(f"Attaching user blocks for {len(unique_dids)} DIDs")
                success, attached_user_labels = attach_user_blocks(CLIENT, void_agent.id, unique_dids)
                if success and attached_user_labels:
                    logger.info(f"Attached {len(attached_user_labels)} user blocks: {attached_user_labels}")
                elif not success:
                    logger.warning("Failed to attach some user blocks, continuing anyway")
            # Use streaming to avoid 524 timeout errors (SDK v1.0: create() with streaming=True)
            request_params = {
                'agent_id': void_agent.id,
                'messages': [{"role": "user", "content": prompt[:100] + "..."}],  # Truncated for logging
                'streaming': True,
                'stream_tokens': False,
                'max_steps': 100
            }
            logger.debug(f"Sending streaming request: {request_params}")

            message_stream = CLIENT.agents.messages.create(
                agent_id=void_agent.id,
                messages=[{"role": "user", "content": prompt}],
                streaming=True,
                stream_tokens=False,  # Step streaming only (faster than token streaming)
                max_steps=100
            )

            # Collect the streaming response
            all_messages = []
            for chunk in message_stream:
                chunk_count += 1
                logger.debug(f"Received chunk {chunk_count}: type={getattr(chunk, 'message_type', 'unknown')}")
                # Log condensed chunk info
                if hasattr(chunk, 'message_type'):
                    if chunk.message_type == 'reasoning_message':
                        # Show full reasoning without truncation
                        if SHOW_REASONING:
                            # Format with Unicode characters
                            print("\n◆ Reasoning")
                            print("  ─────────")
                            # Indent reasoning lines
                            for line in chunk.reasoning.split('\n'):
                                print(f"  {line}")
                        else:
                            # Default log format (only when --reasoning is used due to log level)
                            # Format with Unicode characters
                            print("\n◆ Reasoning")
                            print("  ─────────")
                            # Indent reasoning lines
                            for line in chunk.reasoning.split('\n'):
                                print(f"  {line}")
                        
                        # Create ATProto record for reasoning (unless in testing mode)
                        if not testing_mode and hasattr(chunk, 'reasoning'):
                            try:
                                bsky_utils.create_reasoning_record(atproto_client, chunk.reasoning)
                            except Exception as e:
                                logger.debug(f"Failed to create reasoning record: {e}")
                    elif chunk.message_type == 'tool_call_message':
                        # Parse tool arguments for better display
                        tool_name = chunk.tool_call.name
                        
                        # Create ATProto record for tool call (unless in testing mode)
                        if not testing_mode:
                            try:
                                tool_call_id = chunk.tool_call.tool_call_id if hasattr(chunk.tool_call, 'tool_call_id') else None
                                bsky_utils.create_tool_call_record(
                                    atproto_client, 
                                    tool_name, 
                                    chunk.tool_call.arguments,
                                    tool_call_id
                                )
                            except Exception as e:
                                logger.debug(f"Failed to create tool call record: {e}")
                        
                        try:
                            args = json.loads(chunk.tool_call.arguments)
                            # Format based on tool type
                            if tool_name in ['add_post_to_bluesky_reply_thread', 'bluesky_reply']:
                                # Extract the text being posted
                                text = args.get('text', '')
                                if text:
                                    # Format with Unicode characters
                                    print("\n✎ Bluesky Post")
                                    print("  ────────────")
                                    # Indent post text
                                    for line in text.split('\n'):
                                        print(f"  {line}")
                                else:
                                    log_with_panel(chunk.tool_call.arguments[:150] + "...", f"Tool call: {tool_name}", "blue")
                            elif tool_name == 'archival_memory_search':
                                query = args.get('query', 'unknown')
                                global last_archival_query
                                last_archival_query = query
                                log_with_panel(f"query: \"{query}\"", f"Tool call: {tool_name}", "blue")
                            elif tool_name == 'archival_memory_insert':
                                content = args.get('content', '')
                                # Show the full content being inserted
                                log_with_panel(content, f"Tool call: {tool_name}", "blue")
                            elif tool_name == 'update_block':
                                label = args.get('label', 'unknown')
                                value_preview = str(args.get('value', ''))[:50] + "..." if len(str(args.get('value', ''))) > 50 else str(args.get('value', ''))
                                log_with_panel(f"{label}: \"{value_preview}\"", f"Tool call: {tool_name}", "blue")
                            else:
                                # Generic display for other tools
                                args_str = ', '.join(f"{k}={v}" for k, v in args.items() if k != 'request_heartbeat')
                                if len(args_str) > 150:
                                    args_str = args_str[:150] + "..."
                                log_with_panel(args_str, f"Tool call: {tool_name}", "blue")
                        except:
                            # Fallback to original format if parsing fails
                            log_with_panel(chunk.tool_call.arguments[:150] + "...", f"Tool call: {tool_name}", "blue")
                    elif chunk.message_type == 'tool_return_message':
                        # Enhanced tool result logging
                        tool_name = chunk.name
                        status = chunk.status
                        
                        if status == 'success':
                            # Try to show meaningful result info based on tool type
                            if hasattr(chunk, 'tool_return') and chunk.tool_return:
                                result_str = str(chunk.tool_return)
                                if tool_name == 'archival_memory_search':
                                    
                                    try:
                                        # Handle both string and list formats
                                        if isinstance(chunk.tool_return, str):
                                            # The string format is: "([{...}, {...}], count)"
                                            # We need to extract just the list part
                                            if chunk.tool_return.strip():
                                                # Find the list part between the first [ and last ]
                                                start_idx = chunk.tool_return.find('[')
                                                end_idx = chunk.tool_return.rfind(']')
                                                if start_idx != -1 and end_idx != -1:
                                                    list_str = chunk.tool_return[start_idx:end_idx+1]
                                                    # Use ast.literal_eval since this is Python literal syntax, not JSON
                                                    import ast
                                                    results = ast.literal_eval(list_str)
                                                else:
                                                    logger.warning("Could not find list in archival_memory_search result")
                                                    results = []
                                            else:
                                                logger.warning("Empty string returned from archival_memory_search")
                                                results = []
                                        else:
                                            # If it's already a list, use directly
                                            results = chunk.tool_return
                                        
                                        log_with_panel(f"Found {len(results)} memory entries", f"Tool result: {tool_name} ✓", "green")
                                        
                                        # Use the captured search query from the tool call
                                        search_query = last_archival_query
                                        
                                        # Combine all results into a single text block
                                        content_text = ""
                                        for i, entry in enumerate(results, 1):
                                            timestamp = entry.get('timestamp', 'N/A')
                                            content = entry.get('content', '')
                                            content_text += f"[{i}/{len(results)}] {timestamp}\n{content}\n\n"
                                        
                                        # Format with Unicode characters
                                        title = f"{search_query} ({len(results)} results)"
                                        print(f"\n⚙ {title}")
                                        print(f"  {'─' * len(title)}")
                                        # Indent content text
                                        for line in content_text.strip().split('\n'):
                                            print(f"  {line}")
                                        
                                    except Exception as e:
                                        logger.error(f"Error formatting archival memory results: {e}")
                                        log_with_panel(result_str[:100] + "...", f"Tool result: {tool_name} ✓", "green")
                                elif tool_name == 'add_post_to_bluesky_reply_thread':
                                    # Just show success for bluesky posts, the text was already shown in tool call
                                    log_with_panel("Post queued successfully", f"Bluesky Post ✓", "green")
                                elif tool_name == 'archival_memory_insert':
                                    # Skip archival memory insert results (always returns None)
                                    pass
                                elif tool_name == 'update_block':
                                    log_with_panel("Memory block updated", f"Tool result: {tool_name} ✓", "green")
                                else:
                                    # Generic success with preview
                                    preview = result_str[:100] + "..." if len(result_str) > 100 else result_str
                                    log_with_panel(preview, f"Tool result: {tool_name} ✓", "green")
                            else:
                                log_with_panel("Success", f"Tool result: {tool_name} ✓", "green")
                        elif status == 'error':
                            # Show error details
                            if tool_name == 'add_post_to_bluesky_reply_thread':
                                error_str = str(chunk.tool_return) if hasattr(chunk, 'tool_return') and chunk.tool_return else "Error occurred"
                                log_with_panel(error_str, f"Bluesky Post ✗", "red")
                            elif tool_name == 'archival_memory_insert':
                                # Skip archival memory insert errors too
                                pass
                            else:
                                error_preview = ""
                                if hasattr(chunk, 'tool_return') and chunk.tool_return:
                                    error_str = str(chunk.tool_return)
                                    error_preview = error_str[:100] + "..." if len(error_str) > 100 else error_str
                                    log_with_panel(f"Error: {error_preview}", f"Tool result: {tool_name} ✗", "red")
                                else:
                                    log_with_panel("Error occurred", f"Tool result: {tool_name} ✗", "red")
                        else:
                            logger.info(f"Tool result: {tool_name} - {status}")
                    elif chunk.message_type == 'assistant_message':
                        # Format with Unicode characters
                        print("\n▶ Assistant Response")
                        print("  ──────────────────")
                        # Indent response text
                        for line in chunk.content.split('\n'):
                            print(f"  {line}")
                    elif chunk.message_type == 'error_message':
                        # Dump full error object
                        logger.error(f"Agent error_message: {chunk}")
                        if hasattr(chunk, 'model_dump'):
                            logger.error(f"Agent error (dict): {chunk.model_dump()}")
                        elif hasattr(chunk, '__dict__'):
                            logger.error(f"Agent error (vars): {vars(chunk)}")
                    else:
                        # Filter out verbose message types
                        if chunk.message_type not in ['usage_statistics', 'stop_reason']:
                            logger.info(f"{chunk.message_type}: {str(chunk)[:150]}...")
                else:
                    logger.info(f"📦 Stream status: {chunk}")
                
                # Log full chunk for debugging
                logger.debug(f"Full streaming chunk: {chunk}")
                all_messages.append(chunk)
                if str(chunk) == 'done':
                    break
            
            # Convert streaming response to standard format for compatibility
            message_response = type('StreamingResponse', (), {
                'messages': [msg for msg in all_messages if hasattr(msg, 'message_type')]
            })()
        except Exception as api_error:
            import traceback
            error_str = str(api_error)
            logger.error(f"Letta API error: {api_error}")
            logger.error(f"Error type: {type(api_error).__name__}")
            logger.error(f"Chunks received before error: {chunk_count}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            logger.error(f"Mention text was: {mention_text}")
            logger.error(f"Author: @{author_handle}")
            logger.error(f"URI: {uri}")
            
            
            # Try to extract more info from different error types
            if hasattr(api_error, 'response'):
                logger.error(f"Error response object exists")
                if hasattr(api_error.response, 'text'):
                    logger.error(f"Response text: {api_error.response.text}")
                if hasattr(api_error.response, 'json') and callable(api_error.response.json):
                    try:
                        logger.error(f"Response JSON: {api_error.response.json()}")
                    except:
                        pass
            
            # Check for specific error types
            if hasattr(api_error, 'status_code'):
                logger.error(f"API Status code: {api_error.status_code}")
                if hasattr(api_error, 'body'):
                    logger.error(f"API Response body: {api_error.body}")
                if hasattr(api_error, 'headers'):
                    logger.error(f"API Response headers: {api_error.headers}")
                
                if api_error.status_code == 413:
                    logger.error("413 Payload Too Large - moving to errors directory")
                    return None  # Move to errors directory - payload is too large to ever succeed
                elif api_error.status_code == 524:
                    logger.error("524 error - timeout from Cloudflare, will retry later")
                    return False  # Keep in queue for retry
            
            # Check if error indicates we should remove from queue
            if 'status_code: 413' in error_str or 'Payload Too Large' in error_str:
                logger.warning("Payload too large error, moving to errors directory")
                return None  # Move to errors directory - cannot be fixed by retry
            elif 'status_code: 524' in error_str:
                logger.warning("524 timeout error, keeping in queue for retry")
                return False  # Keep in queue for retry
            
            raise

        # Log successful response
        logger.debug("Successfully received response from Letta API")
        logger.debug(f"Number of messages in response: {len(message_response.messages) if hasattr(message_response, 'messages') else 'N/A'}")

        # Extract successful add_post_to_bluesky_reply_thread tool calls from the agent's response
        reply_candidates = []
        tool_call_results = {}  # Map tool_call_id to status
        ack_note = None  # Track any note from annotate_ack tool
        flagged_memories = []  # Track memories flagged for deletion
        
        logger.debug(f"Processing {len(message_response.messages)} response messages...")
        
        # First pass: collect tool return statuses
        ignored_notification = False
        ignore_reason = ""
        ignore_category = ""

        logger.debug(f"Processing {len(message_response.messages)} messages from agent")

        for message in message_response.messages:
            # Log detailed message attributes for debugging
            msg_type = getattr(message, 'message_type', 'unknown')
            has_tool_call_id = hasattr(message, 'tool_call_id')
            has_status = hasattr(message, 'status')
            has_tool_return = hasattr(message, 'tool_return')

            logger.debug(f"Message type={msg_type}, has_tool_call_id={has_tool_call_id}, has_status={has_status}, has_tool_return={has_tool_return}")

            # Tool return messages are identified by having tool_return attribute, tool_call_id, and status
            if has_tool_call_id and has_status and has_tool_return:
                logger.debug(f"  -> tool_call_id={message.tool_call_id}, status={message.status}")

                # Store the result for ANY tool that has a return - we'll match by tool_call_id later
                tool_call_results[message.tool_call_id] = message.status
                logger.debug(f"Stored tool result: {message.tool_call_id} -> {message.status}")

                # Handle special processing for ignore_notification
                if message.status == 'success':
                    result_str = str(message.tool_return) if message.tool_return else ""
                    if 'IGNORED_NOTIFICATION::' in result_str:
                        parts = result_str.split('::')
                        if len(parts) >= 3:
                            ignore_category = parts[1]
                            ignore_reason = parts[2]
                            ignored_notification = True
                            logger.info(f"🚫 Notification ignored - Category: {ignore_category}, Reason: {ignore_reason}")

            # Check for deprecated tool in tool call messages
            elif hasattr(message, 'tool_call') and message.tool_call:
                if message.tool_call.name == 'bluesky_reply':
                    logger.error("DEPRECATED TOOL DETECTED: bluesky_reply is no longer supported!")
                    logger.error("Please use add_post_to_bluesky_reply_thread instead.")
                    logger.error("Update the agent's tools using register_tools.py")
                    logger.info("=== BOT TERMINATED DUE TO DEPRECATED TOOL USE ===")
                    exit(1)

        logger.debug(f"First pass complete. Collected {len(tool_call_results)} tool call results")
        logger.debug(f"tool_call_results: {tool_call_results}")

        # Second pass: process messages and check for successful tool calls
        for i, message in enumerate(message_response.messages, 1):
            # Log concise message info instead of full object
            msg_type = getattr(message, 'message_type', 'unknown')
            if hasattr(message, 'reasoning') and message.reasoning:
                logger.debug(f"  {i}. {msg_type}: {message.reasoning[:100]}...")
            elif hasattr(message, 'tool_call') and message.tool_call:
                tool_name = message.tool_call.name
                logger.debug(f"  {i}. {msg_type}: {tool_name}")
            elif hasattr(message, 'tool_return'):
                tool_name = getattr(message, 'name', 'unknown_tool')
                return_preview = str(message.tool_return)[:100] if message.tool_return else "None"
                status = getattr(message, 'status', 'unknown')
                logger.debug(f"  {i}. {msg_type}: {tool_name} -> {return_preview}... (status: {status})")
            elif hasattr(message, 'text'):
                logger.debug(f"  {i}. {msg_type}: {message.text[:100]}...")
            else:
                logger.debug(f"  {i}. {msg_type}: <no content>")

            # Check for halt_activity tool call
            if hasattr(message, 'tool_call') and message.tool_call:
                if message.tool_call.name == 'halt_activity':
                    logger.critical("=" * 60)
                    logger.critical("🛑 EMERGENCY HALT - BOT SHUTDOWN INITIATED 🛑")
                    logger.critical("=" * 60)
                    try:
                        args = json.loads(message.tool_call.arguments)
                        reason = args.get('reason', 'Agent requested halt')
                        logger.critical(f"SHUTDOWN REASON: {reason}")
                    except:
                        logger.critical("SHUTDOWN REASON: <unable to parse>")

                    # Delete the queue file before terminating
                    if queue_filepath and queue_filepath.exists():
                        queue_filepath.unlink()
                        logger.info(f"Deleted queue file: {queue_filepath.name}")

                        # Also mark as processed to avoid reprocessing
                        if NOTIFICATION_DB:
                            NOTIFICATION_DB.mark_processed(notification_data.get('uri', ''), status='processed')
                        else:
                            processed_uris = load_processed_notifications()
                            processed_uris.add(notification_data.get('uri', ''))
                            save_processed_notifications(processed_uris)

                    # Exit the program
                    logger.critical("=" * 60)
                    logger.critical("TERMINATING BOT PROCESS NOW")
                    logger.critical("=" * 60)
                    exit(0)
            
            # Check for deprecated bluesky_reply tool
            if hasattr(message, 'tool_call') and message.tool_call:
                if message.tool_call.name == 'bluesky_reply':
                    logger.error("DEPRECATED TOOL DETECTED: bluesky_reply is no longer supported!")
                    logger.error("Please use add_post_to_bluesky_reply_thread instead.")
                    logger.error("Update the agent's tools using register_tools.py")
                    logger.info("=== BOT TERMINATED DUE TO DEPRECATED TOOL USE ===")
                    exit(1)
                
                # Collect annotate_ack tool calls
                elif message.tool_call.name == 'annotate_ack':
                    try:
                        args = json.loads(message.tool_call.arguments)
                        note = args.get('note', '')
                        if note:
                            ack_note = note
                            logger.debug(f"Found annotate_ack with note: {note[:50]}...")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse annotate_ack arguments: {e}")

                # Collect flag_archival_memory_for_deletion tool calls
                elif message.tool_call.name == 'flag_archival_memory_for_deletion':
                    try:
                        args = json.loads(message.tool_call.arguments)
                        reason = args.get('reason', '')
                        memory_text = args.get('memory_text', '')
                        confirm = args.get('confirm', False)

                        # Only flag for deletion if confirmed and has all required fields
                        if confirm and memory_text and reason:
                            flagged_memories.append({
                                'reason': reason,
                                'memory_text': memory_text
                            })
                            logger.debug(f"Found memory flagged for deletion (reason: {reason}): {memory_text[:50]}...")
                        elif not confirm:
                            logger.debug(f"Memory deletion not confirmed, skipping: {memory_text[:50]}...")
                        elif not reason:
                            logger.warning(f"Memory deletion missing reason, skipping: {memory_text[:50]}...")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse flag_archival_memory_for_deletion arguments: {e}")

                # Collect archival_memory_insert tool calls for recording to AT Protocol
                elif message.tool_call.name == 'archival_memory_insert':
                    try:
                        args = json.loads(message.tool_call.arguments)
                        content = args.get('content', '')
                        tags_str = args.get('tags', None)

                        # Parse tags from string representation if present
                        tags = None
                        if tags_str:
                            try:
                                tags = json.loads(tags_str) if isinstance(tags_str, str) else tags_str
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse tags from archival_memory_insert: {tags_str[:50]}...")

                        if content:
                            # Create stream.thought.memory record
                            try:
                                memory_result = bsky_utils.create_memory_record(atproto_client, content, tags)
                                if memory_result:
                                    tags_info = f" ({len(tags)} tags)" if tags else ""
                                    logger.info(f"📝 Recorded archival memory to AT Protocol{tags_info}: {content[:100]}...")
                                else:
                                    logger.warning(f"Failed to record archival memory to AT Protocol")
                            except Exception as e:
                                logger.error(f"Error creating memory record: {e}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse archival_memory_insert arguments: {e}")

                # Collect add_post_to_bluesky_reply_thread tool calls - only if they were successful
                elif message.tool_call.name == 'add_post_to_bluesky_reply_thread':
                    tool_call_id = message.tool_call.tool_call_id
                    tool_status = tool_call_results.get(tool_call_id, 'unknown')

                    logger.debug(f"Found add_post_to_bluesky_reply_thread tool call: id={tool_call_id}, status={tool_status}")
                    logger.debug(f"Available tool_call_results: {tool_call_results}")

                    if tool_status == 'success':
                        try:
                            args = json.loads(message.tool_call.arguments)
                            reply_text = args.get('text', '')
                            reply_lang = args.get('lang', 'en-US')

                            if reply_text:  # Only add if there's actual content
                                reply_candidates.append((reply_text, reply_lang))
                                logger.debug(f"Found successful add_post_to_bluesky_reply_thread candidate: {reply_text[:50]}... (lang: {reply_lang})")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse tool call arguments: {e}")
                    elif tool_status == 'error':
                        logger.debug(f"Skipping failed add_post_to_bluesky_reply_thread tool call (status: error)")
                    else:
                        logger.warning(f"⚠️ Skipping add_post_to_bluesky_reply_thread tool call with unknown status: {tool_status}")
                        logger.warning(f"   Tool call ID '{tool_call_id}' not found in tool_call_results dict")

        # Handle archival memory deletion if any were flagged (only if no halt was received)
        if flagged_memories:
            logger.info(f"Processing {len(flagged_memories)} flagged memories for deletion")
            for flagged_memory in flagged_memories:
                reason = flagged_memory['reason']
                memory_text = flagged_memory['memory_text']

                try:
                    # Search for passages with this exact text
                    logger.debug(f"Searching for passages matching: {memory_text[:100]}...")
                    passages = CLIENT.agents.passages.list(
                        agent_id=void_agent.id,
                        query=memory_text
                    )

                    if not passages:
                        logger.warning(f"No passages found matching flagged memory: {memory_text[:50]}...")
                        continue

                    # Delete all matching passages
                    deleted_count = 0
                    for passage in passages:
                        # Check if the passage text exactly matches (to avoid partial matches)
                        if hasattr(passage, 'text') and passage.text == memory_text:
                            try:
                                CLIENT.agents.passages.delete(
                                    agent_id=void_agent.id,
                                    passage_id=str(passage.id)
                                )
                                deleted_count += 1
                                logger.debug(f"Deleted passage {passage.id}")
                            except Exception as delete_error:
                                logger.error(f"Failed to delete passage {passage.id}: {delete_error}")

                    if deleted_count > 0:
                        logger.info(f"🗑️ Deleted {deleted_count} archival memory passage(s) (reason: {reason}): {memory_text[:50]}...")
                    else:
                        logger.warning(f"No exact matches found for deletion: {memory_text[:50]}...")

                except Exception as e:
                    logger.error(f"Error processing memory deletion: {e}")

        # Check for conflicting tool calls
        if reply_candidates and ignored_notification:
            logger.error(f"⚠️ CONFLICT: Agent called both add_post_to_bluesky_reply_thread and ignore_notification!")
            logger.error(f"Reply candidates: {len(reply_candidates)}, Ignore reason: {ignore_reason}")
            logger.warning("Item will be left in queue for manual review")
            # Return False to keep in queue
            return False
        
        if reply_candidates:
            # Aggregate reply posts into a thread
            reply_messages = []
            reply_langs = []
            for text, lang in reply_candidates:
                reply_messages.append(text)
                reply_langs.append(lang)
            
            # Use the first language for the entire thread (could be enhanced later)
            reply_lang = reply_langs[0] if reply_langs else 'en-US'
            
            logger.debug(f"Found {len(reply_candidates)} add_post_to_bluesky_reply_thread calls, building thread")
            
            # Display the generated reply thread
            if len(reply_messages) == 1:
                content = reply_messages[0]
                title = f"Reply to @{author_handle}"
            else:
                content = "\n\n".join([f"{j}. {msg}" for j, msg in enumerate(reply_messages, 1)])
                title = f"Reply Thread to @{author_handle} ({len(reply_messages)} messages)"
            
            # Format with Unicode characters
            print(f"\n✎ {title}")
            print(f"  {'─' * len(title)}")
            # Indent content lines
            for line in content.split('\n'):
                print(f"  {line}")

            # Send the reply(s) with language (unless in testing mode)
            if testing_mode:
                logger.info("TESTING MODE: Skipping actual Bluesky post")
                response = True  # Simulate success
            else:
                if len(reply_messages) == 1:
                    # Single reply - use existing function
                    cleaned_text = bsky_utils.remove_outside_quotes(reply_messages[0])
                    logger.info(f"Sending single reply: {cleaned_text[:50]}... (lang: {reply_lang})")
                    response = bsky_utils.reply_to_notification(
                        client=atproto_client,
                        notification=notification_data,
                        reply_text=cleaned_text,
                        lang=reply_lang,
                        correlation_id=correlation_id
                    )
                else:
                    # Multiple replies - use new threaded function
                    cleaned_messages = [bsky_utils.remove_outside_quotes(msg) for msg in reply_messages]
                    logger.info(f"Sending threaded reply with {len(cleaned_messages)} messages (lang: {reply_lang})")
                    response = bsky_utils.reply_with_thread_to_notification(
                        client=atproto_client,
                        notification=notification_data,
                        reply_messages=cleaned_messages,
                        lang=reply_lang,
                        correlation_id=correlation_id
                    )

            if response:
                logger.info(f"[{correlation_id}] Successfully replied to @{author_handle}", extra={
                    'correlation_id': correlation_id,
                    'author_handle': author_handle,
                    'reply_count': len(reply_messages)
                })
                
                # Acknowledge the post we're replying to with stream.thought.ack
                try:
                    post_uri = notification_data.get('uri')
                    post_cid = notification_data.get('cid')
                    
                    if post_uri and post_cid:
                        ack_result = bsky_utils.acknowledge_post(
                            client=atproto_client,
                            post_uri=post_uri,
                            post_cid=post_cid,
                            note=ack_note
                        )
                        if ack_result:
                            if ack_note:
                                logger.info(f"Successfully acknowledged post from @{author_handle} with stream.thought.ack (note: \"{ack_note[:50]}...\")")
                            else:
                                logger.info(f"Successfully acknowledged post from @{author_handle} with stream.thought.ack")
                        else:
                            logger.warning(f"Failed to acknowledge post from @{author_handle}")
                    else:
                        logger.warning(f"Missing URI or CID for acknowledging post from @{author_handle}")
                except Exception as e:
                    logger.error(f"Error acknowledging post from @{author_handle}: {e}")
                    # Don't fail the entire operation if acknowledgment fails
                
                return True
            else:
                logger.error(f"Failed to send reply to @{author_handle}")
                return False
        else:
            # Check if notification was explicitly ignored
            if ignored_notification:
                logger.info(f"[{correlation_id}] Notification from @{author_handle} was explicitly ignored (category: {ignore_category})", extra={
                    'correlation_id': correlation_id,
                    'author_handle': author_handle,
                    'ignore_category': ignore_category
                })
                return "ignored"
            else:
                logger.warning(f"[{correlation_id}] No reply generated for mention from @{author_handle}, moving to no_reply folder", extra={
                    'correlation_id': correlation_id,
                    'author_handle': author_handle
                })
                return "no_reply"

    except Exception as e:
        logger.error(f"[{correlation_id}] Error processing mention: {e}", extra={
            'correlation_id': correlation_id,
            'error': str(e),
            'error_type': type(e).__name__,
            'author_handle': author_handle if 'author_handle' in locals() else 'unknown'
        })
        return False
    finally:
        # Always detach user blocks after processing
        if attached_user_labels:
            logger.debug(f"Detaching {len(attached_user_labels)} user blocks")
            detach_success = detach_user_blocks(CLIENT, void_agent.id, attached_user_labels)
            if not detach_success:
                logger.warning("Some user blocks may not have been detached properly")


def notification_to_dict(notification):
    """Convert a notification object to a dictionary for JSON serialization."""
    return {
        'uri': notification.uri,
        'cid': notification.cid,
        'reason': notification.reason,
        'is_read': notification.is_read,
        'indexed_at': notification.indexed_at,
        'author': {
            'handle': notification.author.handle,
            'display_name': notification.author.display_name,
            'did': notification.author.did
        },
        'record': {
            'text': getattr(notification.record, 'text', '') if hasattr(notification, 'record') else ''
        }
    }


def load_processed_notifications():
    """Load the set of processed notification URIs from database."""
    global NOTIFICATION_DB
    if NOTIFICATION_DB:
        return NOTIFICATION_DB.get_processed_uris(limit=MAX_PROCESSED_NOTIFICATIONS)
    return set()


def save_processed_notifications(processed_set):
    """Save the set of processed notification URIs to database."""
    # This is now handled by marking individual notifications in the DB
    # Keeping function for compatibility but it doesn't need to do anything
    pass


def save_notification_to_queue(notification, is_priority=None):
    """Save a notification to the queue directory with priority-based filename."""
    try:
        global NOTIFICATION_DB
        
        # Handle both notification objects and dicts
        if isinstance(notification, dict):
            notif_dict = notification
            notification_uri = notification.get('uri')
        else:
            notif_dict = notification_to_dict(notification)
            notification_uri = notification.uri
        
        # Check if already processed (using database if available)
        if NOTIFICATION_DB:
            if NOTIFICATION_DB.is_processed(notification_uri):
                logger.debug(f"Notification already processed (DB): {notification_uri}")
                return False
            # Add to database - if this fails, don't queue the notification
            if not NOTIFICATION_DB.add_notification(notif_dict):
                logger.warning(f"Failed to add notification to database, skipping: {notification_uri}")
                return False
        else:
            # Fall back to old JSON method
            processed_uris = load_processed_notifications()
            if notification_uri in processed_uris:
                logger.debug(f"Notification already processed: {notification_uri}")
                return False

        # Create JSON string
        notif_json = json.dumps(notif_dict, sort_keys=True)

        # Generate hash for filename (to avoid duplicates)
        notif_hash = hashlib.sha256(notif_json.encode()).hexdigest()[:16]

        # Determine priority based on author handle or explicit priority
        if is_priority is not None:
            priority_prefix = "0_" if is_priority else "1_"
        else:
            if isinstance(notification, dict):
                author_handle = notification.get('author', {}).get('handle', '')
            else:
                author_handle = getattr(notification.author, 'handle', '') if hasattr(notification, 'author') else ''
            # Prioritize cameron.pfiffer.org responses
            priority_prefix = "0_" if author_handle == "cameron.pfiffer.org" else "1_"

        # Create filename with priority, timestamp and hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reason = notif_dict.get('reason', 'unknown')
        filename = f"{priority_prefix}{timestamp}_{reason}_{notif_hash}.json"
        filepath = QUEUE_DIR / filename

        # Check if this notification URI is already in the queue
        for existing_file in QUEUE_DIR.glob("*.json"):
            if existing_file.name == "processed_notifications.json":
                continue
            try:
                with open(existing_file, 'r') as f:
                    existing_data = json.load(f)
                    if existing_data.get('uri') == notification_uri:
                        logger.debug(f"Notification already queued (URI: {notification_uri})")
                        return False
            except:
                continue

        # Write to file
        with open(filepath, 'w') as f:
            json.dump(notif_dict, f, indent=2)

        priority_label = "HIGH PRIORITY" if priority_prefix == "0_" else "normal"
        logger.info(f"Queued notification ({priority_label}): {filename}")
        return True

    except Exception as e:
        logger.error(f"Error saving notification to queue: {e}")
        return False


def load_and_process_queued_notifications(void_agent, atproto_client, testing_mode=False):
    """Load and process all notifications from the queue in priority order."""
    try:
        # Get all JSON files in queue directory (excluding processed_notifications.json)
        # Files are sorted by name, which puts priority files first (0_ prefix before 1_ prefix)
        all_queue_files = sorted([f for f in QUEUE_DIR.glob("*.json") if f.name != "processed_notifications.json"])

        # Filter out and delete like notifications immediately
        queue_files = []
        likes_deleted = 0
        
        for filepath in all_queue_files:
            try:
                with open(filepath, 'r') as f:
                    notif_data = json.load(f)
                
                # If it's a like, delete it immediately and don't process
                if notif_data.get('reason') == 'like':
                    filepath.unlink()
                    likes_deleted += 1
                    logger.debug(f"Deleted like notification: {filepath.name}")
                else:
                    queue_files.append(filepath)
            except Exception as e:
                logger.warning(f"Error checking notification file {filepath.name}: {e}")
                queue_files.append(filepath)  # Keep it in case it's valid
        
        if likes_deleted > 0:
            logger.info(f"Deleted {likes_deleted} like notifications from queue")

        if not queue_files:
            return

        logger.info(f"Processing {len(queue_files)} queued notifications")
        
        # Log current statistics
        elapsed_time = time.time() - start_time
        total_messages = sum(message_counters.values())
        messages_per_minute = (total_messages / elapsed_time * 60) if elapsed_time > 0 else 0
        
        logger.info(f"Session stats: {total_messages} total messages ({message_counters['mentions']} mentions, {message_counters['replies']} replies, {message_counters['follows']} follows) | {messages_per_minute:.1f} msg/min")

        for i, filepath in enumerate(queue_files, 1):
            # Determine if this is a priority notification
            is_priority = filepath.name.startswith("0_")
            
            # Check for new notifications periodically during queue processing
            # Also check immediately after processing each priority item
            should_check_notifications = (i % CHECK_NEW_NOTIFICATIONS_EVERY_N_ITEMS == 0 and i > 1)
            
            # If we just processed a priority item, immediately check for new priority notifications
            if is_priority and i > 1:
                should_check_notifications = True
            
            if should_check_notifications:
                logger.info(f"🔄 Checking for new notifications (processed {i-1}/{len(queue_files)} queue items)")
                try:
                    # Fetch and queue new notifications without processing them
                    new_count = fetch_and_queue_new_notifications(atproto_client)
                    
                    if new_count > 0:
                        logger.info(f"Added {new_count} new notifications to queue")
                        # Reload the queue files to include the new items
                        updated_queue_files = sorted([f for f in QUEUE_DIR.glob("*.json") if f.name != "processed_notifications.json"])
                        queue_files = updated_queue_files
                        logger.info(f"Queue updated: now {len(queue_files)} total items")
                except Exception as e:
                    logger.error(f"Error checking for new notifications: {e}")
            
            priority_label = " [PRIORITY]" if is_priority else ""
            logger.info(f"Processing queue file {i}/{len(queue_files)}{priority_label}: {filepath.name}")
            try:
                # Load notification data
                with open(filepath, 'r') as f:
                    notif_data = json.load(f)

                # Process based on type using dict data directly
                success = False
                if notif_data['reason'] == "mention":
                    success = process_mention(void_agent, atproto_client, notif_data, queue_filepath=filepath, testing_mode=testing_mode)
                    if success:
                        message_counters['mentions'] += 1
                elif notif_data['reason'] == "reply":
                    success = process_mention(void_agent, atproto_client, notif_data, queue_filepath=filepath, testing_mode=testing_mode)
                    if success:
                        message_counters['replies'] += 1
                elif notif_data['reason'] == "follow":
                    # author_handle = notif_data['author']['handle']
                    # author_display_name = notif_data['author'].get('display_name', 'no display name')
                    # follow_update = f"@{author_handle} ({author_display_name}) started following you."
                    # follow_message = f"Update: {follow_update}"
                    # logger.info(f"Notifying agent about new follower: @{author_handle} | prompt: {len(follow_message)} chars")
                    # CLIENT.agents.messages.create(
                    #     agent_id = void_agent.id,
                    #     messages = [{"role":"user", "content": follow_message}]
                    # )
                    success = True  # Follow updates are always successful
                    # if success:
                    #     message_counters['follows'] += 1
                elif notif_data['reason'] == "repost":
                    # Skip reposts silently
                    success = True  # Skip reposts but mark as successful to remove from queue
                    if success:
                        message_counters['reposts_skipped'] += 1
                elif notif_data['reason'] == "like":
                    # Skip likes silently
                    success = True  # Skip likes but mark as successful to remove from queue
                    if success:
                        message_counters.setdefault('likes_skipped', 0)
                        message_counters['likes_skipped'] += 1
                else:
                    logger.warning(f"Unknown notification type: {notif_data['reason']}")
                    success = True  # Remove unknown types from queue

                # Handle file based on processing result
                if success:
                    if testing_mode:
                        logger.info(f"TESTING MODE: Keeping queue file: {filepath.name}")
                    else:
                        filepath.unlink()
                        logger.info(f"Successfully processed and removed: {filepath.name}")
                        
                        # Mark as processed to avoid reprocessing
                        if NOTIFICATION_DB:
                            NOTIFICATION_DB.mark_processed(notif_data['uri'], status='processed')
                        else:
                            processed_uris = load_processed_notifications()
                            processed_uris.add(notif_data['uri'])
                            save_processed_notifications(processed_uris)
                    
                elif success is None:  # Special case for moving to error directory
                    error_path = QUEUE_ERROR_DIR / filepath.name
                    filepath.rename(error_path)
                    logger.warning(f"Moved {filepath.name} to errors directory")
                    
                    # Also mark as processed to avoid retrying
                    if NOTIFICATION_DB:
                        NOTIFICATION_DB.mark_processed(notif_data['uri'], status='error')
                    else:
                        processed_uris = load_processed_notifications()
                        processed_uris.add(notif_data['uri'])
                        save_processed_notifications(processed_uris)
                    
                elif success == "no_reply":  # Special case for moving to no_reply directory
                    no_reply_path = QUEUE_NO_REPLY_DIR / filepath.name
                    filepath.rename(no_reply_path)
                    logger.info(f"Moved {filepath.name} to no_reply directory")
                    
                    # Also mark as processed to avoid retrying
                    if NOTIFICATION_DB:
                        NOTIFICATION_DB.mark_processed(notif_data['uri'], status='error')
                    else:
                        processed_uris = load_processed_notifications()
                        processed_uris.add(notif_data['uri'])
                        save_processed_notifications(processed_uris)
                    
                elif success == "ignored":  # Special case for explicitly ignored notifications
                    # For ignored notifications, we just delete them (not move to no_reply)
                    filepath.unlink()
                    logger.info(f"🚫 Deleted ignored notification: {filepath.name}")
                    
                    # Also mark as processed to avoid retrying
                    if NOTIFICATION_DB:
                        NOTIFICATION_DB.mark_processed(notif_data['uri'], status='error')
                    else:
                        processed_uris = load_processed_notifications()
                        processed_uris.add(notif_data['uri'])
                        save_processed_notifications(processed_uris)
                    
                else:
                    logger.warning(f"⚠️  Failed to process {filepath.name}, keeping in queue for retry")

            except Exception as e:
                logger.error(f"💥 Error processing queued notification {filepath.name}: {e}")
                # Keep the file for retry later

    except Exception as e:
        logger.error(f"Error loading queued notifications: {e}")


def fetch_and_queue_new_notifications(atproto_client):
    """Fetch new notifications and queue them without processing."""
    try:
        global NOTIFICATION_DB
        
        # Get current time for marking notifications as seen
        logger.debug("Getting current time for notification marking...")
        last_seen_at = atproto_client.get_current_time_iso()
        
        # Get timestamp of last processed notification for filtering
        last_processed_time = None
        if NOTIFICATION_DB:
            last_processed_time = NOTIFICATION_DB.get_latest_processed_time()
            if last_processed_time:
                logger.debug(f"Last processed notification was at: {last_processed_time}")

        # Fetch ALL notifications using pagination
        all_notifications = []
        cursor = None
        page_count = 0
        max_pages = 20  # Safety limit to prevent infinite loops
        
        while page_count < max_pages:
            try:
                # Fetch notifications page
                if cursor:
                    notifications_response = atproto_client.app.bsky.notification.list_notifications(
                        params={'cursor': cursor, 'limit': 100}
                    )
                else:
                    notifications_response = atproto_client.app.bsky.notification.list_notifications(
                        params={'limit': 100}
                    )
                
                page_count += 1
                page_notifications = notifications_response.notifications
                
                if not page_notifications:
                    break
                
                all_notifications.extend(page_notifications)
                
                # Check if there are more pages
                cursor = getattr(notifications_response, 'cursor', None)
                if not cursor:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching notifications page {page_count}: {e}")
                break
        
        # Now process all fetched notifications
        new_count = 0
        if all_notifications:
            logger.info(f"📥 Fetched {len(all_notifications)} total notifications from API")
            
            # Mark as seen first
            try:
                atproto_client.app.bsky.notification.update_seen(
                    data={'seenAt': last_seen_at}
                )
                logger.debug(f"Marked {len(all_notifications)} notifications as seen at {last_seen_at}")
            except Exception as e:
                logger.error(f"Error marking notifications as seen: {e}")
            
            # Debug counters
            skipped_read = 0
            skipped_likes = 0
            skipped_processed = 0
            skipped_old_timestamp = 0
            processed_uris = load_processed_notifications()
            
            # Queue all new notifications (except likes)
            for notif in all_notifications:
                # Skip if older than last processed (when we have timestamp filtering)
                if last_processed_time and hasattr(notif, 'indexed_at'):
                    if notif.indexed_at <= last_processed_time:
                        skipped_old_timestamp += 1
                        logger.debug(f"Skipping old notification (indexed_at {notif.indexed_at} <= {last_processed_time})")
                        continue
                
                # Debug: Log is_read status but DON'T skip based on it
                if hasattr(notif, 'is_read') and notif.is_read:
                    skipped_read += 1
                    logger.debug(f"Notification has is_read=True (but processing anyway): {notif.uri if hasattr(notif, 'uri') else 'unknown'}")
                
                # Skip likes
                if hasattr(notif, 'reason') and notif.reason == 'like':
                    skipped_likes += 1
                    continue
                    
                notif_dict = notif.model_dump() if hasattr(notif, 'model_dump') else notif
                
                # Skip likes in dict form too
                if notif_dict.get('reason') == 'like':
                    continue
                
                # Check if already processed
                notif_uri = notif_dict.get('uri', '')
                if notif_uri in processed_uris:
                    skipped_processed += 1
                    logger.debug(f"Skipping already processed: {notif_uri}")
                    continue
                
                # Check if it's a priority notification
                is_priority = False
                
                # Priority for cameron.pfiffer.org notifications
                author_handle = notif_dict.get('author', {}).get('handle', '')
                if author_handle == "cameron.pfiffer.org":
                    is_priority = True
                
                # Also check for priority keywords in mentions
                if notif_dict.get('reason') == 'mention':
                    # Get the mention text to check for priority keywords
                    record = notif_dict.get('record', {})
                    text = record.get('text', '')
                    if any(keyword in text.lower() for keyword in ['urgent', 'priority', 'important', 'emergency']):
                        is_priority = True
                
                if save_notification_to_queue(notif_dict, is_priority=is_priority):
                    new_count += 1
                    logger.debug(f"Queued notification from @{author_handle}: {notif_dict.get('reason', 'unknown')}")
            
            # Log summary of filtering
            logger.info(f"📊 Notification processing summary:")
            logger.info(f"  • Total fetched: {len(all_notifications)}")
            logger.info(f"  • Had is_read=True: {skipped_read} (not skipped)")
            logger.info(f"  • Skipped (likes): {skipped_likes}")
            logger.info(f"  • Skipped (old timestamp): {skipped_old_timestamp}")
            logger.info(f"  • Skipped (already processed): {skipped_processed}")
            logger.info(f"  • Queued for processing: {new_count}")
        else:
            logger.debug("No new notifications to queue")
            
        return new_count
            
    except Exception as e:
        logger.error(f"Error fetching and queueing notifications: {e}")
        return 0


def process_notifications(void_agent, atproto_client, testing_mode=False):
    """Fetch new notifications, queue them, and process the queue."""
    try:
        # Fetch and queue new notifications
        new_count = fetch_and_queue_new_notifications(atproto_client)
        
        if new_count > 0:
            logger.info(f"Found {new_count} new notifications to process")

        # Now process the entire queue (old + new notifications)
        load_and_process_queued_notifications(void_agent, atproto_client, testing_mode)

    except Exception as e:
        logger.error(f"Error processing notifications: {e}")


def send_synthesis_message(client: Letta, agent_id: str, agent_name: str = None, atproto_client=None) -> None:
    """
    Send a synthesis message to the agent every 10 minutes.
    This prompts the agent to synthesize its recent experiences.

    Args:
        client: Letta client
        agent_id: Agent ID to send synthesis to
        agent_name: Agent name for temporal block labels
        atproto_client: Optional AT Protocol client for posting synthesis results
    """
    # Track attached temporal blocks for cleanup
    attached_temporal_labels = []

    try:
        logger.info("🧠 Preparing synthesis with temporal journal blocks")

        # Attach temporal blocks before synthesis
        success, attached_temporal_labels = attach_temporal_blocks(client, agent_id, agent_name)
        if not success:
            logger.warning("Failed to attach some temporal blocks, continuing with synthesis anyway")

        # Create synthesis prompt with agent-specific block names
        today = date.today()
        synthesis_prompt = f"""Time for synthesis.

This is your periodic opportunity to reflect on recent experiences and update your memory.

The following temporal journal blocks are temporarily available for this session:
- {agent_name}_day_{today.strftime('%Y_%m_%d')}: Today's journal ({today.strftime('%B %d, %Y')})
- {agent_name}_month_{today.strftime('%Y_%m')}: This month's journal ({today.strftime('%B %Y')})
- {agent_name}_year_{today.year}: This year's journal ({today.year})

You may use these blocks as you see fit. Synthesize your recent experiences into your memory as appropriate."""
        
        logger.info("🧠 Sending enhanced synthesis prompt to agent")

        # Send synthesis message with streaming to show tool use (SDK v1.0: create() with streaming=True)
        message_stream = client.agents.messages.create(
            agent_id=agent_id,
            messages=[{"role": "user", "content": synthesis_prompt}],
            streaming=True,
            stream_tokens=False,
            max_steps=100
        )
        
        # Track synthesis content for potential posting
        synthesis_posts = []
        ack_note = None
        
        # Process the streaming response
        for chunk in message_stream:
            if hasattr(chunk, 'message_type'):
                if chunk.message_type == 'reasoning_message':
                    if SHOW_REASONING:
                        print("\n◆ Reasoning")
                        print("  ─────────")
                        for line in chunk.reasoning.split('\n'):
                            print(f"  {line}")
                    
                    # Create ATProto record for reasoning (if we have atproto client)
                    if atproto_client and hasattr(chunk, 'reasoning'):
                        try:
                            bsky_utils.create_reasoning_record(atproto_client, chunk.reasoning)
                        except Exception as e:
                            logger.debug(f"Failed to create reasoning record during synthesis: {e}")
                elif chunk.message_type == 'tool_call_message':
                    tool_name = chunk.tool_call.name
                    
                    # Create ATProto record for tool call (if we have atproto client)
                    if atproto_client:
                        try:
                            tool_call_id = chunk.tool_call.tool_call_id if hasattr(chunk.tool_call, 'tool_call_id') else None
                            bsky_utils.create_tool_call_record(
                                atproto_client,
                                tool_name,
                                chunk.tool_call.arguments,
                                tool_call_id
                            )
                        except Exception as e:
                            logger.debug(f"Failed to create tool call record during synthesis: {e}")
                    try:
                        args = json.loads(chunk.tool_call.arguments)
                        if tool_name == 'archival_memory_search':
                            query = args.get('query', 'unknown')
                            log_with_panel(f"query: \"{query}\"", f"Tool call: {tool_name}", "blue")
                        elif tool_name == 'archival_memory_insert':
                            content = args.get('content', '')
                            log_with_panel(content[:200] + "..." if len(content) > 200 else content, f"Tool call: {tool_name}", "blue")

                            # Record archival memory insert to AT Protocol
                            if atproto_client:
                                try:
                                    tags_str = args.get('tags', None)
                                    tags = None
                                    if tags_str:
                                        try:
                                            tags = json.loads(tags_str) if isinstance(tags_str, str) else tags_str
                                        except json.JSONDecodeError:
                                            logger.warning(f"Failed to parse tags from archival_memory_insert: {tags_str[:50]}...")

                                    memory_result = bsky_utils.create_memory_record(atproto_client, content, tags)
                                    if memory_result:
                                        tags_info = f" ({len(tags)} tags)" if tags else ""
                                        logger.info(f"📝 Recorded archival memory to AT Protocol{tags_info}")
                                except Exception as e:
                                    logger.debug(f"Failed to create memory record during synthesis: {e}")
                        elif tool_name == 'update_block':
                            label = args.get('label', 'unknown')
                            value_preview = str(args.get('value', ''))[:100] + "..." if len(str(args.get('value', ''))) > 100 else str(args.get('value', ''))
                            log_with_panel(f"{label}: \"{value_preview}\"", f"Tool call: {tool_name}", "blue")
                        elif tool_name == 'annotate_ack':
                            note = args.get('note', '')
                            if note:
                                ack_note = note
                                log_with_panel(f"note: \"{note[:100]}...\"" if len(note) > 100 else f"note: \"{note}\"", f"Tool call: {tool_name}", "blue")
                        elif tool_name == 'add_post_to_bluesky_reply_thread':
                            text = args.get('text', '')
                            synthesis_posts.append(text)
                            log_with_panel(f"text: \"{text[:100]}...\"" if len(text) > 100 else f"text: \"{text}\"", f"Tool call: {tool_name}", "blue")
                        else:
                            args_str = ', '.join(f"{k}={v}" for k, v in args.items() if k != 'request_heartbeat')
                            if len(args_str) > 150:
                                args_str = args_str[:150] + "..."
                            log_with_panel(args_str, f"Tool call: {tool_name}", "blue")
                    except:
                        log_with_panel(chunk.tool_call.arguments[:150] + "...", f"Tool call: {tool_name}", "blue")
                elif chunk.message_type == 'tool_return_message':
                    if chunk.status == 'success':
                        log_with_panel("Success", f"Tool result: {chunk.name} ✓", "green")
                    else:
                        log_with_panel("Error", f"Tool result: {chunk.name} ✗", "red")
                elif chunk.message_type == 'assistant_message':
                    print("\n▶ Synthesis Response")
                    print("  ──────────────────")
                    for line in chunk.content.split('\n'):
                        print(f"  {line}")
                elif chunk.message_type == 'error_message':
                    # Dump full error object
                    logger.error(f"Synthesis error_message: {chunk}")
                    if hasattr(chunk, 'model_dump'):
                        logger.error(f"Synthesis error (dict): {chunk.model_dump()}")
                    elif hasattr(chunk, '__dict__'):
                        logger.error(f"Synthesis error (vars): {vars(chunk)}")

            if str(chunk) == 'done':
                break
        
        logger.info("🧠 Synthesis message processed successfully")
        
        # Handle synthesis acknowledgments if we have an atproto client
        if atproto_client and ack_note:
            try:
                result = bsky_utils.create_synthesis_ack(atproto_client, ack_note)
                if result:
                    logger.info(f"✓ Created synthesis acknowledgment: {ack_note[:50]}...")
                else:
                    logger.warning("Failed to create synthesis acknowledgment")
            except Exception as e:
                logger.error(f"Error creating synthesis acknowledgment: {e}")
        
        # Handle synthesis posts if any were generated
        if atproto_client and synthesis_posts:
            try:
                for post_text in synthesis_posts:
                    cleaned_text = bsky_utils.remove_outside_quotes(post_text)
                    response = bsky_utils.send_post(atproto_client, cleaned_text)
                    if response:
                        logger.info(f"✓ Posted synthesis content: {cleaned_text[:50]}...")
                    else:
                        logger.warning(f"Failed to post synthesis content: {cleaned_text[:50]}...")
            except Exception as e:
                logger.error(f"Error posting synthesis content: {e}")
        
    except Exception as e:
        logger.error(f"Error sending synthesis message: {e}")
    finally:
        # Always detach temporal blocks after synthesis
        if attached_temporal_labels:
            logger.info("🧠 Detaching temporal journal blocks after synthesis")
            detach_success = detach_temporal_blocks(client, agent_id, attached_temporal_labels, agent_name)
            if not detach_success:
                logger.warning("Some temporal blocks may not have been detached properly")


def attach_temporal_blocks(client: Letta, agent_id: str, agent_name: str = None) -> tuple:
    """
    Attach temporal journal blocks (day, month, year) to the agent for synthesis.
    Creates blocks if they don't exist.

    Args:
        client: Letta client
        agent_id: Agent ID
        agent_name: Agent name for prefixing block labels (prevents collision across agents)

    Returns:
        Tuple of (success: bool, attached_labels: list)
    """
    try:
        today = date.today()

        # Generate temporal block labels with agent-specific prefix
        day_label = f"{agent_name}_day_{today.strftime('%Y_%m_%d')}"
        month_label = f"{agent_name}_month_{today.strftime('%Y_%m')}"
        year_label = f"{agent_name}_year_{today.year}"
        
        temporal_labels = [day_label, month_label, year_label]
        attached_labels = []
        
        # Get current blocks attached to agent (SDK v1.0 returns page object)
        current_blocks_page = client.agents.blocks.list(agent_id=agent_id)
        current_blocks = current_blocks_page.items if hasattr(current_blocks_page, 'items') else current_blocks_page
        current_block_labels = {block.label for block in current_blocks}
        current_block_ids = {str(block.id) for block in current_blocks}

        for label in temporal_labels:
            try:
                # Skip if already attached
                if label in current_block_labels:
                    logger.debug(f"Temporal block already attached: {label}")
                    attached_labels.append(label)
                    continue

                # Check if block exists globally (SDK v1.0 returns page object)
                blocks_page = client.blocks.list(label=label)
                blocks = blocks_page.items if hasattr(blocks_page, 'items') else blocks_page

                if blocks and len(blocks) > 0:
                    block = blocks[0]
                    # Check if already attached by ID
                    if str(block.id) in current_block_ids:
                        logger.debug(f"Temporal block already attached by ID: {label}")
                        attached_labels.append(label)
                        continue
                else:
                    # Create new temporal block with appropriate header
                    if "day" in label:
                        header = f"# Daily Journal - {today.strftime('%B %d, %Y')}"
                        initial_content = f"{header}\n\nNo entries yet for today."
                    elif "month" in label:
                        header = f"# Monthly Journal - {today.strftime('%B %Y')}"
                        initial_content = f"{header}\n\nNo entries yet for this month."
                    else:  # year
                        header = f"# Yearly Journal - {today.year}"
                        initial_content = f"{header}\n\nNo entries yet for this year."
                    
                    block = client.blocks.create(
                        label=label,
                        value=initial_content,
                        limit=10000  # Larger limit for journal blocks
                    )
                    logger.info(f"Created new temporal block: {label}")
                
                # Attach the block
                client.agents.blocks.attach(
                    agent_id=agent_id,
                    block_id=str(block.id)
                )
                attached_labels.append(label)
                logger.info(f"Attached temporal block: {label}")
                
            except Exception as e:
                # Check for duplicate constraint errors
                error_str = str(e)
                if "duplicate key value violates unique constraint" in error_str:
                    logger.debug(f"Temporal block already attached (constraint): {label}")
                    attached_labels.append(label)
                else:
                    logger.warning(f"Failed to attach temporal block {label}: {e}")
        
        logger.info(f"Temporal blocks attached: {len(attached_labels)}/{len(temporal_labels)}")
        return True, attached_labels
        
    except Exception as e:
        logger.error(f"Error attaching temporal blocks: {e}")
        return False, []


def detach_temporal_blocks(client: Letta, agent_id: str, labels_to_detach: list = None, agent_name: str = None) -> bool:
    """
    Detach temporal journal blocks from the agent after synthesis.

    Args:
        client: Letta client
        agent_id: Agent ID
        labels_to_detach: Optional list of specific labels to detach.
                         If None, detaches all temporal blocks for this agent.
        agent_name: Agent name for prefixing block labels (prevents collision across agents)

    Returns:
        bool: Success status
    """
    try:
        # If no specific labels provided, generate today's labels
        if labels_to_detach is None:
            today = date.today()
            labels_to_detach = [
                f"{agent_name}_day_{today.strftime('%Y_%m_%d')}",
                f"{agent_name}_month_{today.strftime('%Y_%m')}",
                f"{agent_name}_year_{today.year}"
            ]
        
        # Get current blocks and build label to ID mapping (SDK v1.0 returns page object)
        current_blocks_page = client.agents.blocks.list(agent_id=agent_id)
        current_blocks = current_blocks_page.items if hasattr(current_blocks_page, 'items') else current_blocks_page
        block_label_to_id = {block.label: str(block.id) for block in current_blocks}
        
        detached_count = 0
        for label in labels_to_detach:
            if label in block_label_to_id:
                try:
                    client.agents.blocks.detach(
                        agent_id=agent_id,
                        block_id=block_label_to_id[label]
                    )
                    detached_count += 1
                    logger.debug(f"Detached temporal block: {label}")
                except Exception as e:
                    logger.warning(f"Failed to detach temporal block {label}: {e}")
            else:
                logger.debug(f"Temporal block not attached: {label}")
        
        logger.info(f"Detached {detached_count} temporal blocks")
        return True
        
    except Exception as e:
        logger.error(f"Error detaching temporal blocks: {e}")
        return False


def attach_user_blocks(client: Letta, agent_id: str, dids: list) -> tuple:
    """
    Attach user-specific memory blocks for the given DIDs.
    Creates blocks if they don't exist.

    Args:
        client: Letta client
        agent_id: Agent ID
        dids: List of DIDs to attach blocks for

    Returns:
        tuple: (success: bool, attached_labels: list)
    """
    if not dids:
        return True, []

    attached_labels = []

    try:
        # Get currently attached blocks
        current_blocks_page = client.agents.blocks.list(agent_id=agent_id)
        current_blocks = current_blocks_page.items if hasattr(current_blocks_page, 'items') else current_blocks_page
        current_block_labels = {block.label for block in current_blocks}

        for did in dids:
            # Create label from DID: user_did:plc:abc123 -> user_did_plc_abc123
            label = f"user_{did.replace(':', '_')}"
            
            # Skip if already attached
            if label in current_block_labels:
                logger.debug(f"User block already attached: {label}")
                attached_labels.append(label)
                continue
            
            try:
                # Check if block exists
                try:
                    existing_blocks = client.blocks.list(label=label, limit=1)
                    blocks_list = existing_blocks.items if hasattr(existing_blocks, 'items') else existing_blocks
                    block = blocks_list[0] if blocks_list else None
                except:
                    block = None
                
                # Create block if it doesn't exist
                if not block:
                    block = client.blocks.create(
                        label=label,
                        value=f"User {did}\n\nNo information about this user yet.",
                        description=f"Memory block for user {did}. Update this block with information learned about the user during conversations."
                    )
                    logger.debug(f"Created new user block: {label}")
                
                # Attach the block
                client.agents.blocks.attach(
                    agent_id=agent_id,
                    block_id=str(block.id)
                )
                attached_labels.append(label)
                logger.debug(f"Attached user block: {label}")
                
            except Exception as e:
                # Check for duplicate constraint errors (already attached)
                if 'duplicate' in str(e).lower() or 'unique constraint' in str(e).lower():
                    logger.debug(f"User block already attached (constraint): {label}")
                    attached_labels.append(label)
                else:
                    logger.warning(f"Failed to attach user block {label}: {e}")
        
        logger.debug(f"User blocks attached: {len(attached_labels)}/{len(dids)}")
        return True, attached_labels
        
    except Exception as e:
        logger.error(f"Error attaching user blocks: {e}")
        return False, []


def detach_user_blocks(client: Letta, agent_id: str, labels_to_detach: list) -> bool:
    """
    Detach user-specific memory blocks after processing.
    
    Args:
        client: Letta client
        agent_id: Agent ID
        labels_to_detach: List of block labels to detach
    
    Returns:
        bool: Success status
    """
    if not labels_to_detach:
        return True
    
    try:
        # Get currently attached blocks
        current_blocks_page = client.agents.blocks.list(agent_id=agent_id)
        current_blocks = current_blocks_page.items if hasattr(current_blocks_page, 'items') else current_blocks_page
        
        # Create mapping of label -> block_id
        block_label_to_id = {block.label: str(block.id) for block in current_blocks}
        
        detached_count = 0
        for label in labels_to_detach:
            if label in block_label_to_id:
                try:
                    client.agents.blocks.detach(
                        agent_id=agent_id,
                        block_id=block_label_to_id[label]
                    )
                    detached_count += 1
                    logger.debug(f"Detached user block: {label}")
                except Exception as e:
                    logger.warning(f"Failed to detach user block {label}: {e}")
            else:
                logger.debug(f"User block not attached: {label}")
        
        logger.debug(f"Detached {detached_count} user blocks")
        return True
        
    except Exception as e:
        logger.error(f"Error detaching user blocks: {e}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Void Bot - Bluesky autonomous agent')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file (default: configs/config.yaml)')
    parser.add_argument('--test', action='store_true', help='Run in testing mode (no messages sent, queue files preserved)')
    parser.add_argument('--simple-logs', action='store_true', help='Use simplified log format (void - LEVEL - message)')
    # --rich option removed as we now use simple text formatting
    parser.add_argument('--reasoning', action='store_true', help='Display reasoning in panels and set reasoning log level to INFO')
    parser.add_argument('--cleanup-interval', type=int, default=10, help='Run user block cleanup every N cycles (default: 10, 0 to disable)')
    parser.add_argument('--synthesis-interval', type=int, default=600, help='Send synthesis message every N seconds (default: 600 = 10 minutes, 0 to disable)')
    parser.add_argument('--synthesis-only', action='store_true', help='Run in synthesis-only mode (only send synthesis messages, no notification processing)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Initialize configuration with custom path
    global letta_config, CLIENT, QUEUE_DIR, QUEUE_ERROR_DIR, QUEUE_NO_REPLY_DIR, PROCESSED_NOTIFICATIONS_FILE, NOTIFICATION_DB, BOT_NAME
    get_config(args.config)  # Initialize the global config instance
    letta_config = get_letta_config()

    # Initialize queue paths from config
    queue_config = get_queue_config()
    QUEUE_DIR = Path(queue_config['base_dir'])
    QUEUE_ERROR_DIR = Path(queue_config['error_dir'])
    QUEUE_NO_REPLY_DIR = Path(queue_config['no_reply_dir'])
    PROCESSED_NOTIFICATIONS_FILE = Path(queue_config['processed_file'])

    # Get bot name for logging and file naming
    bot_name = queue_config['bot_name']
    BOT_NAME = bot_name

    # Create queue directories
    QUEUE_DIR.mkdir(exist_ok=True)
    QUEUE_ERROR_DIR.mkdir(exist_ok=True, parents=True)
    QUEUE_NO_REPLY_DIR.mkdir(exist_ok=True, parents=True)

    # Create Letta client with configuration
    CLIENT_PARAMS = {
        'api_key': letta_config['api_key'],
        'timeout': letta_config['timeout']
    }
    if letta_config.get('base_url'):
        CLIENT_PARAMS['base_url'] = letta_config['base_url']
    CLIENT = Letta(**CLIENT_PARAMS)

    # Configure logging based on command line arguments
    if args.simple_logs:
        log_format = f"{bot_name} - %(levelname)s - %(message)s"
    else:
        # Create custom formatter with symbols
        class SymbolFormatter(logging.Formatter):
            """Custom formatter that adds symbols for different log levels"""

            SYMBOLS = {
                logging.DEBUG: '◇',
                logging.INFO: '✓',
                logging.WARNING: '⚠',
                logging.ERROR: '✗',
                logging.CRITICAL: '‼'
            }

            def __init__(self, bot_name):
                super().__init__()
                self.bot_name = bot_name

            def format(self, record):
                # Get the symbol for this log level
                symbol = self.SYMBOLS.get(record.levelno, '•')

                # Format time as HH:MM:SS
                timestamp = self.formatTime(record, "%H:%M:%S")

                # Build the formatted message
                level_name = f"{record.levelname:<5}"  # Left-align, 5 chars

                # Use vertical bar as separator
                parts = [symbol, timestamp, '│', self.bot_name, '│', level_name, '│', record.getMessage()]

                return ' '.join(parts)

        # Reset logging configuration
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Create handler with custom formatter
        handler = logging.StreamHandler()
        if not args.simple_logs:
            handler.setFormatter(SymbolFormatter(bot_name))
        else:
            handler.setFormatter(logging.Formatter(log_format))

        # Configure root logger
        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(handler)
    
    global logger, prompt_logger, console
    logger = logging.getLogger("void_bot")
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    # Create a separate logger for prompts (set to WARNING to hide by default)
    prompt_logger = logging.getLogger("void_bot.prompts")
    if args.reasoning:
        prompt_logger.setLevel(logging.INFO)  # Show reasoning when --reasoning is used
    else:
        prompt_logger.setLevel(logging.WARNING)  # Hide by default
    
    # Disable httpx logging completely
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    
    # Create Rich console for pretty printing
    # Console no longer used - simple text formatting

    global TESTING_MODE, SHOW_REASONING
    TESTING_MODE = args.test

    # Store rich flag globally
    # Rich formatting no longer used
    
    # Store reasoning flag globally
    SHOW_REASONING = args.reasoning
    
    if TESTING_MODE:
        logger.info("=== RUNNING IN TESTING MODE ===")
        logger.info("   - No messages will be sent to Bluesky")
        logger.info("   - Queue files will not be deleted")
        logger.info("   - Notifications will not be marked as seen")
        print("\n")
    
    # Check for synthesis-only mode
    SYNTHESIS_ONLY = args.synthesis_only
    if SYNTHESIS_ONLY:
        logger.info("=== RUNNING IN SYNTHESIS-ONLY MODE ===")
        logger.info("   - Only synthesis messages will be sent")
        logger.info("   - No notification processing")
        logger.info("   - No Bluesky client needed")
        print("\n")
    """Main bot loop that continuously monitors for notifications."""
    global start_time
    start_time = time.time()
    logger.info("=== STARTING VOID BOT ===")
    void_agent = initialize_agent()
    logger.info(f"Void agent initialized: {void_agent.id}")
    
    # Initialize notification database with config-based path
    logger.info("Initializing notification database...")
    NOTIFICATION_DB = NotificationDB(db_path=queue_config['db_path'])
    
    # Migrate from old JSON format if it exists
    if PROCESSED_NOTIFICATIONS_FILE.exists():
        logger.info("Found old processed_notifications.json, migrating to database...")
        NOTIFICATION_DB.migrate_from_json(str(PROCESSED_NOTIFICATIONS_FILE))
    
    # Log database stats
    db_stats = NOTIFICATION_DB.get_stats()
    logger.info(f"Database initialized - Total notifications: {db_stats.get('total', 0)}, Recent (24h): {db_stats.get('recent_24h', 0)}")
    
    # Clean up old records
    NOTIFICATION_DB.cleanup_old_records(days=7)
    
    # Ensure correct tools are attached for Bluesky
    logger.info("Configuring tools for Bluesky platform...")
    try:
        from tool_manager import ensure_platform_tools
        ensure_platform_tools('bluesky', void_agent.id)
    except Exception as e:
        logger.error(f"Failed to configure platform tools: {e}")
        logger.warning("Continuing with existing tool configuration")
    
    # Check if agent has required tools
    if hasattr(void_agent, 'tools') and void_agent.tools:
        tool_names = [tool.name for tool in void_agent.tools]
        # Check for bluesky-related tools
        bluesky_tools = [name for name in tool_names if 'bluesky' in name.lower() or 'reply' in name.lower()]
        if not bluesky_tools:
            logger.error("⚠️  No Bluesky-related tools found! Agent will not be able to reply to notifications.")
            print("\n" + "="*60)
            print("⚠️  MISSING BLUESKY TOOLS")
            print("="*60)
            print("The agent does not have the required Bluesky tools attached.")
            print("Without these tools, the agent cannot reply to notifications.")
            print("\nWould you like to run the tool registration now?")
            response = input("Run register_tools.py? [Y/n]: ").strip().lower()
            if response in ['', 'y', 'yes']:
                print("\nRunning register_tools.py...")
                import subprocess
                import sys
                result = subprocess.run([sys.executable, 'register_tools.py'], cwd='.')
                if result.returncode == 0:
                    print("✓ Tools registered successfully! Please restart the agent.")
                    sys.exit(0)
                else:
                    print("✗ Tool registration failed. Please run 'python register_tools.py' manually.")
                    sys.exit(1)
            else:
                print("⚠️  Continuing without Bluesky tools. Agent will not be able to reply.")
                print("Run 'python register_tools.py' when ready to enable replies.\n")
    else:
        logger.warning("Agent has no tools registered!")

    # Initialize Bluesky client (needed for both notification processing and synthesis acks/posts)
    if not SYNTHESIS_ONLY:
        atproto_client = bsky_utils.default_login()
        logger.info("Connected to Bluesky")
    else:
        # In synthesis-only mode, still connect for acks and posts (unless in test mode)
        if not args.test:
            atproto_client = bsky_utils.default_login()
            logger.info("Connected to Bluesky (for synthesis acks/posts)")
        else:
            atproto_client = None
            logger.info("Skipping Bluesky connection (test mode)")

    # Configure intervals
    CLEANUP_INTERVAL = args.cleanup_interval
    SYNTHESIS_INTERVAL = args.synthesis_interval
    
    # Synthesis-only mode
    if SYNTHESIS_ONLY:
        if SYNTHESIS_INTERVAL <= 0:
            logger.error("Synthesis-only mode requires --synthesis-interval > 0")
            return
            
        logger.info(f"Starting synthesis-only mode, interval: {SYNTHESIS_INTERVAL} seconds ({SYNTHESIS_INTERVAL/60:.1f} minutes)")
        
        while True:
            try:
                # Send synthesis message immediately on first run
                logger.info("🧠 Sending synthesis message")
                send_synthesis_message(CLIENT, void_agent.id, void_agent.name, atproto_client)
                
                # Wait for next interval
                logger.info(f"Waiting {SYNTHESIS_INTERVAL} seconds until next synthesis...")
                sleep(SYNTHESIS_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("=== SYNTHESIS MODE STOPPED BY USER ===")
                break
            except Exception as e:
                logger.error(f"Error in synthesis loop: {e}")
                logger.info(f"Sleeping for {SYNTHESIS_INTERVAL} seconds due to error...")
                sleep(SYNTHESIS_INTERVAL)
    
    # Normal mode with notification processing
    logger.info(f"Starting notification monitoring, checking every {FETCH_NOTIFICATIONS_DELAY_SEC} seconds")

    cycle_count = 0
    
    if CLEANUP_INTERVAL > 0:
        logger.info(f"User block cleanup enabled every {CLEANUP_INTERVAL} cycles")
    else:
        logger.info("User block cleanup disabled")
    
    if SYNTHESIS_INTERVAL > 0:
        logger.info(f"Synthesis messages enabled every {SYNTHESIS_INTERVAL} seconds ({SYNTHESIS_INTERVAL/60:.1f} minutes)")
    else:
        logger.info("Synthesis messages disabled")
    
    while True:
        try:
            cycle_count += 1
            process_notifications(void_agent, atproto_client, TESTING_MODE)
            
            # Check if synthesis interval has passed
            if SYNTHESIS_INTERVAL > 0:
                current_time = time.time()
                global last_synthesis_time
                if current_time - last_synthesis_time >= SYNTHESIS_INTERVAL:
                    logger.info(f"⏰ {SYNTHESIS_INTERVAL/60:.1f} minutes have passed, triggering synthesis")
                    send_synthesis_message(CLIENT, void_agent.id, void_agent.name, atproto_client)
                    last_synthesis_time = current_time
            
            # Run periodic cleanup every N cycles
            if CLEANUP_INTERVAL > 0 and cycle_count % CLEANUP_INTERVAL == 0:
                # Check if autofollow is enabled and sync followers
                from config_loader import get_bluesky_config
                bluesky_config = get_bluesky_config()
                if bluesky_config.get('autofollow', False):
                    logger.info("🔄 Syncing followers (autofollow enabled)")
                    try:
                        sync_result = bsky_utils.sync_followers(atproto_client, dry_run=False)
                        if 'error' in sync_result:
                            logger.error(f"Autofollow failed: {sync_result['error']}")
                        else:
                            if sync_result['newly_followed']:
                                logger.info(f"✓ Followed {len(sync_result['newly_followed'])} new users: {', '.join(sync_result['newly_followed'])}")
                            else:
                                logger.debug(f"No new followers to follow back ({sync_result['followers_count']} followers, {sync_result['following_count']} following)")
                            if sync_result.get('errors'):
                                logger.warning(f"Some follows failed: {len(sync_result['errors'])} errors")
                    except Exception as e:
                        logger.error(f"Error during autofollow sync: {e}")

                # Also check database health when doing cleanup
                if NOTIFICATION_DB:
                    db_stats = NOTIFICATION_DB.get_stats()
                    pending = db_stats.get('status_pending', 0)
                    errors = db_stats.get('status_error', 0)

                    if pending > 50:
                        logger.warning(f"⚠️ Queue health check: {pending} pending notifications (may be stuck)")
                    if errors > 20:
                        logger.warning(f"⚠️ Queue health check: {errors} error notifications")

                    # Periodic cleanup of old records
                    if cycle_count % (CLEANUP_INTERVAL * 10) == 0:  # Every 100 cycles
                        logger.info("Running database cleanup of old records...")
                        NOTIFICATION_DB.cleanup_old_records(days=7)
            
            # Log cycle completion with stats
            elapsed_time = time.time() - start_time
            total_messages = sum(message_counters.values())
            messages_per_minute = (total_messages / elapsed_time * 60) if elapsed_time > 0 else 0
            
            if total_messages > 0:
                logger.info(f"Cycle {cycle_count} complete. Session totals: {total_messages} messages ({message_counters['mentions']} mentions, {message_counters['replies']} replies) | {messages_per_minute:.1f} msg/min")
            sleep(FETCH_NOTIFICATIONS_DELAY_SEC)

        except KeyboardInterrupt:
            # Final stats
            elapsed_time = time.time() - start_time
            total_messages = sum(message_counters.values())
            messages_per_minute = (total_messages / elapsed_time * 60) if elapsed_time > 0 else 0
            
            logger.info("=== BOT STOPPED BY USER ===")
            logger.info(f"Final session stats: {total_messages} total messages processed in {elapsed_time/60:.1f} minutes")
            logger.info(f"   - {message_counters['mentions']} mentions")
            logger.info(f"   - {message_counters['replies']} replies")
            logger.info(f"   - {message_counters['follows']} follows")
            logger.info(f"   - {message_counters['reposts_skipped']} reposts skipped")
            logger.info(f"   - Average rate: {messages_per_minute:.1f} messages/minute")
            
            # Close database connection
            if NOTIFICATION_DB:
                logger.info("Closing database connection...")
                NOTIFICATION_DB.close()
            
            break
        except Exception as e:
            logger.error(f"=== ERROR IN MAIN LOOP CYCLE {cycle_count} ===")
            logger.error(f"Error details: {e}")
            # Wait a bit longer on errors
            logger.info(f"Sleeping for {FETCH_NOTIFICATIONS_DELAY_SEC * 2} seconds due to error...")
            sleep(FETCH_NOTIFICATIONS_DELAY_SEC * 2)


if __name__ == "__main__":
    main()
