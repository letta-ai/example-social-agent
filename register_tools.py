#!/usr/bin/env python3
"""Register all Void tools with a Letta agent."""
import os
import sys
import logging
from typing import List
from letta_client import Letta
from rich.console import Console
from rich.table import Table
from config_loader import get_letta_config, get_bluesky_config, get_config

# Import standalone functions and their schemas
from tools.search import search_bluesky_posts, SearchArgs
from tools.post import create_new_bluesky_post, PostArgs
from tools.feed import get_bluesky_feed, FeedArgs
from tools.halt import halt_activity, HaltArgs
from tools.thread import add_post_to_bluesky_reply_thread, ReplyThreadPostArgs
from tools.ignore import ignore_notification, IgnoreNotificationArgs
from tools.whitewind import create_whitewind_blog_post, WhitewindPostArgs
from tools.ack import annotate_ack, AnnotateAckArgs
from tools.webpage import fetch_webpage, WebpageArgs
from tools.flag_memory_deletion import flag_archival_memory_for_deletion, FlagArchivalMemoryForDeletionArgs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


# Tool configurations: function paired with its args_schema and metadata
TOOL_CONFIGS = [
    {
        "func": search_bluesky_posts,
        "args_schema": SearchArgs,
        "description": "Search for posts on Bluesky matching the given criteria",
        "tags": ["bluesky", "search", "posts"]
    },
    {
        "func": create_new_bluesky_post,
        "args_schema": PostArgs,
        "description": "Create a new Bluesky post or thread",
        "tags": ["bluesky", "post", "create", "thread"]
    },
    {
        "func": get_bluesky_feed,
        "args_schema": FeedArgs,
        "description": "Retrieve a Bluesky feed (home timeline or custom feed)",
        "tags": ["bluesky", "feed", "timeline"]
    },
    {
        "func": halt_activity,
        "args_schema": HaltArgs,
        "description": "Signal to halt all bot activity and terminate bsky.py",
        "tags": ["control", "halt", "terminate"]
    },
    {
        "func": add_post_to_bluesky_reply_thread,
        "args_schema": ReplyThreadPostArgs,
        "description": "Add a single post to the current Bluesky reply thread atomically",
        "tags": ["bluesky", "reply", "thread", "atomic"]
    },
    {
        "func": ignore_notification,
        "args_schema": IgnoreNotificationArgs,
        "description": "Explicitly ignore a notification without replying (useful for ignoring bot interactions)",
        "tags": ["notification", "ignore", "control", "bot"]
    },
    {
        "func": create_whitewind_blog_post,
        "args_schema": WhitewindPostArgs,
        "description": "Create a blog post on Whitewind with markdown support",
        "tags": ["whitewind", "blog", "post", "markdown"]
    },
    {
        "func": annotate_ack,
        "args_schema": AnnotateAckArgs,
        "description": "Add a note to the acknowledgment record for the current post interaction",
        "tags": ["acknowledgment", "note", "annotation", "metadata"]
    },
    {
        "func": fetch_webpage,
        "args_schema": WebpageArgs,
        "description": "Fetch a webpage and convert it to markdown/text format using Jina AI reader",
        "tags": ["web", "fetch", "webpage", "markdown", "jina"]
    },
    {
        "func": flag_archival_memory_for_deletion,
        "args_schema": FlagArchivalMemoryForDeletionArgs,
        "description": "Flag an archival memory for deletion based on its exact text content",
        "tags": ["memory", "archival", "delete", "cleanup"]
    },
]


def register_tools(agent_id: str = None, tools: List[str] = None, set_env: bool = True):
    """Register tools with a Letta agent.

    Args:
        agent_id: ID of the agent to attach tools to. If None, uses config default.
        tools: List of tool names to register. If None, registers all tools.
        set_env: If True, set environment variables for tool execution. Defaults to True.
    """
    # Load config fresh (uses global config instance from get_config())
    letta_config = get_letta_config()

    # Use agent ID from config if not provided
    if agent_id is None:
        agent_id = letta_config['agent_id']

    try:
        # Initialize Letta client with API key and base_url from config
        client_params = {
            'api_key': letta_config['api_key'],
            'timeout': letta_config['timeout']
        }
        if letta_config.get('base_url'):
            client_params['base_url'] = letta_config['base_url']
        client = Letta(**client_params)

        # Get the agent by ID
        try:
            agent = client.agents.retrieve(agent_id=agent_id)
        except Exception as e:
            console.print(f"[red]Error: Agent '{agent_id}' not found[/red]")
            console.print(f"Error details: {e}")
            return

        # Set environment variables for tool execution if requested
        if set_env:
            try:
                bsky_config = get_bluesky_config()
                env_vars = {
                    'BSKY_USERNAME': bsky_config['username'],
                    'BSKY_PASSWORD': bsky_config['password'],
                    'PDS_URI': bsky_config['pds_uri']
                }

                console.print(f"\n[bold cyan]Setting tool execution environment variables:[/bold cyan]")
                console.print(f"  BSKY_USERNAME: {env_vars['BSKY_USERNAME']}")
                console.print(f"  PDS_URI: {env_vars['PDS_URI']}")
                console.print(f"  BSKY_PASSWORD: {'*' * len(env_vars['BSKY_PASSWORD'])}\n")

                # Update agent with environment variables (secrets in SDK v1.0)
                client.agents.update(
                    agent_id=agent_id,
                    secrets=env_vars
                )

                console.print("[green]✓ Environment variables set successfully[/green]\n")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to set environment variables: {e}[/yellow]\n")
                logger.warning(f"Failed to set environment variables: {e}")

        # Filter tools if specific ones requested
        tools_to_register = TOOL_CONFIGS
        if tools:
            tools_to_register = [t for t in TOOL_CONFIGS if t["func"].__name__ in tools]
            if len(tools_to_register) != len(tools):
                missing = set(tools) - {t["func"].__name__ for t in tools_to_register}
                console.print(f"[yellow]Warning: Unknown tools: {missing}[/yellow]")

        # Create results table
        table = Table(title=f"Tool Registration for Agent '{agent.name}' ({agent_id})")
        table.add_column("Tool", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Description")

        # Register each tool
        for tool_config in tools_to_register:
            func = tool_config["func"]
            tool_name = func.__name__

            try:
                # Create or update the tool using the standalone function
                created_tool = client.tools.upsert_from_function(
                    func=func,
                    args_schema=tool_config["args_schema"],
                    tags=tool_config["tags"]
                )

                # Get current agent tools (SDK v1.0 returns page object)
                current_tools_page = client.agents.tools.list(agent_id=str(agent.id))
                current_tools = current_tools_page.items if hasattr(current_tools_page, 'items') else current_tools_page
                tool_names = [t.name for t in current_tools]

                # Check if already attached
                if created_tool.name in tool_names:
                    table.add_row(tool_name, "Already Attached", tool_config["description"])
                else:
                    # Attach to agent
                    client.agents.tools.attach(
                        agent_id=str(agent.id),
                        tool_id=str(created_tool.id)
                    )
                    table.add_row(tool_name, "✓ Attached", tool_config["description"])

            except Exception as e:
                table.add_row(tool_name, f"✗ Error: {str(e)}", tool_config["description"])
                logger.error(f"Error registering tool {tool_name}: {e}")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"Fatal error: {e}")


def list_available_tools():
    """List all available tools."""
    table = Table(title="Available Void Tools")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Description")
    table.add_column("Tags", style="dim")

    for tool_config in TOOL_CONFIGS:
        table.add_row(
            tool_config["func"].__name__,
            tool_config["description"],
            ", ".join(tool_config["tags"])
        )

    console.print(table)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register Void tools with a Letta agent")
    parser.add_argument("--config", type=str, default='configs/config.yaml', help="Path to config file (default: configs/config.yaml)")
    parser.add_argument("--agent-id", help=f"Agent ID (default: from config)")
    parser.add_argument("--tools", nargs="+", help="Specific tools to register (default: all)")
    parser.add_argument("--list", action="store_true", help="List available tools")
    parser.add_argument("--no-env", action="store_true", help="Skip setting environment variables")

    args = parser.parse_args()

    # Initialize config with custom path (sets global config instance)
    get_config(args.config)

    if args.list:
        list_available_tools()
    else:
        # Load config and get agent ID
        letta_config = get_letta_config()
        agent_id = args.agent_id if args.agent_id else letta_config['agent_id']
        console.print(f"\n[bold]Registering tools for agent: {agent_id}[/bold]\n")
        register_tools(agent_id, args.tools, set_env=not args.no_env)
