#!/usr/bin/env python3
"""
Setup script for social-agents framework.
Imports the example agent and creates config.yaml if needed.
"""

import os
import sys
import json
from pathlib import Path
import yaml
from rich.console import Console
from rich.prompt import Prompt, Confirm
from letta_client import Letta

console = Console()


def check_config_exists():
    """Check if config.yaml exists."""
    return Path("configs/config.yaml").exists()


def import_example_agent(client: Letta) -> str:
    """Import the example agent and return its ID."""
    agent_file = Path("agents/example-social-agent.af")

    if not agent_file.exists():
        console.print(f"[red]Error: Example agent file not found at {agent_file}[/red]")
        sys.exit(1)

    console.print(f"\n[cyan]Importing example agent from {agent_file}...[/cyan]")

    try:
        # Import the agent - import_file expects an open file object
        # Explicitly pass the project_id from your account
        with open(agent_file, 'rb') as f:
            result = client.agents.import_file(file=f, project_id="1ebf49e9-9c69-4f4c-a032-e6ea9c3a96e2")
        
        # Get the first imported agent ID
        agent_id = result.agent_ids[0]
        
        # Fetch the agent details
        agent = client.agents.retrieve(agent_id)

        console.print(f"[green]✓ Successfully imported agent: {agent.name} (ID: {agent.id})[/green]")
        return agent.id

    except Exception as e:
        console.print(f"[red]Error importing agent: {e}[/red]")
        sys.exit(1)


def create_config(agent_id: str, letta_api_key: str = None):
    """Create a basic config.yaml file."""
    console.print("\n[bold cyan]Setting up configuration...[/bold cyan]")

    # Prompt for Bluesky credentials
    console.print("\n[yellow]Bluesky Configuration:[/yellow]")
    bsky_username = Prompt.ask("Enter your Bluesky username (e.g., yourname.bsky.social)")
    bsky_password = Prompt.ask("Enter your Bluesky app password", password=True)
    bsky_pds_uri = Prompt.ask("Enter your PDS URI", default="https://bsky.social")

    # Create config structure
    config = {
        'letta': {
            'api_key': letta_api_key if letta_api_key else 'your-letta-api-key',
            'agent_id': agent_id,
            'timeout': 600
        },
        'bluesky': {
            'username': bsky_username,
            'password': bsky_password,
            'pds_uri': bsky_pds_uri,
            'autofollow': False
        },
        'bot': {
            'max_thread_posts': 0
        }
    }

    # Write config file
    try:
        # Ensure configs directory exists
        Path('configs').mkdir(exist_ok=True)
        
        with open('configs/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        console.print(f"\n[green]✓ Created configs/config.yaml[/green]")
        console.print("\n[yellow]Important:[/yellow] Edit config.yaml and add your LETTA_API_KEY")
        console.print("You can get an API key from: https://app.letta.com")

    except Exception as e:
        console.print(f"[red]Error creating config.yaml: {e}[/red]")
        sys.exit(1)


def main():
    """Main setup flow."""
    console.print("\n[bold]Social Agents Setup[/bold]")
    console.print("=" * 50)

    # Check if config already exists
    if check_config_exists():
        console.print("\n[yellow]configs/config.yaml already exists![/yellow]")
        if not Confirm.ask("Do you want to overwrite it?"):
            console.print("[yellow]Setup cancelled.[/yellow]")
            sys.exit(0)

    # Check for LETTA_API_KEY
    letta_api_key = os.environ.get("LETTA_API_KEY")
    if not letta_api_key:
        console.print("\n[yellow]LETTA_API_KEY not found in environment.[/yellow]")
        console.print("You can:")
        console.print("  1. Set LETTA_API_KEY environment variable now")
        console.print("  2. Continue setup and add it to config.yaml later")

        if Confirm.ask("\nDo you have a Letta API key to use now?"):
            letta_api_key = Prompt.ask("Enter your Letta API key", password=True)
            os.environ["LETTA_API_KEY"] = letta_api_key
        else:
            console.print("\n[yellow]Skipping agent import. You'll need to:[/yellow]")
            console.print("  1. Get a Letta API key from https://app.letta.com")
            console.print("  2. Import agents/example-social-agent.af manually")
            console.print("  3. Update config.yaml with your agent ID")
            sys.exit(0)

    # Ask if they want to import the example agent
    console.print("\n[cyan]This will import the example agent to your Letta account.[/cyan]")
    if Confirm.ask("Import example agent?", default=True):
        # Create Letta client
        try:
            client = Letta(api_key=letta_api_key)
        except Exception as e:
            console.print(f"[red]Error connecting to Letta: {e}[/red]")
            sys.exit(1)

        # Import agent
        agent_id = import_example_agent(client)
    else:
        # Ask for existing agent ID
        console.print("\n[yellow]Please provide your existing Letta agent ID.[/yellow]")
        console.print("You can find agent IDs at: https://app.letta.com")
        agent_id = Prompt.ask("Enter your agent ID")

    # Create config
    create_config(agent_id, letta_api_key)

    # Next steps
    console.print("\n[bold green]Setup Complete![/bold green]")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("  1. Verify your settings in config.yaml")
    console.print("  2. Register tools: [bold]python register_tools.py[/bold]")
    console.print("  3. Run your agent: [bold]python bsky.py[/bold]")
    console.print("\nSee README.md for more information.")


if __name__ == "__main__":
    main()