# Example Social Agent

An example framework for deploying stateful AI agents to social networks using [Google Gemini 3](https://ai.google.dev), [Letta](https://letta.com) (stateful AI agents), and [AT Protocol](https://atproto.com) (the social protocol powering Bluesky).

This example demonstrates how to build a social agent powered by Gemini 3 that maintains persistent memory and interacts autonomously on Bluesky.

## What are Social Agents?

Social agents are stateful AI systems connected to social networks. Unlike traditional chatbots, they:

- **Maintain persistent memory** that evolves through interactions
- **Develop stable personas** while accumulating knowledge about their environment
- **Build lasting relationships** with individual users
- **Form defined senses of self** through continuous operation

## Quick Start

```bash
# Clone and install
git clone https://github.com/letta-ai/example-social-agent
cd example-social-agent
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Setup (imports example agent and creates config)
python setup.py

# Register tools
python register_tools.py

# Run (uses Gemini 3 by default)
python bsky.py
```

## Features

- **Memory-Augmented Architecture**: Multi-tiered memory system (Core, Recall, Archival) powered by Letta
- **Temporal Memory Synthesis**: Periodic self-reflection with day/month/year journal blocks
- **Queue-Based Processing**: Reliable notification handling with SQLite tracking and automatic retry
- **Dynamic Memory Blocks**: Per-user memory blocks for personalized interactions
- **Tool System**: Extensible Pydantic-based tools for social platform interactions
- **Autofollow**: Optional automatic following of users who follow your agent

## Getting Started

### Prerequisites

1. **Letta Setup**
   - Sign up for [Letta Cloud](https://app.letta.com) or [host your own](https://docs.letta.com/guides/selfhosting) Letta instance
   - Create a new project and generate [an API key](https://app.letta.com/api-keys)
   - Note your Project ID, which is visible

2. **Bluesky Setup**
   - Create a [Bluesky account](https://bsky.app/)
   - Generate an [app password](https://bsky.app/settings/app-passwords) in your settings
   - Note your handle (e.g., `yourname.bsky.social`)

3. **Python 3.8+**

### Installation

```bash
# Clone the repository
git clone https://github.com/letta-ai/example-social-agent
cd example-social-agent

# Install dependencies
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

### Configuration

#### Option 1: Automated Setup (Recommended)

Run the setup script to import the example agent and create your configuration:

```bash
source .venv/bin/activate
python setup.py
```

The setup script will:
1. Prompt for your Letta API key (or use `LETTA_API_KEY` from environment)
2. Import the example agent from `agents/example-social-agent.af`
3. Create `config.yaml` with your agent ID
4. Prompt for your Bluesky credentials

#### Option 2: Manual Setup

```bash
# Copy example config
cp config.example.yaml config.yaml
```

Edit `config.yaml` with your credentials:

```yaml
bluesky:
  username: "yourname.bsky.social"
  password: "your-app-password"
  pds_uri: "https://bsky.social"
  autofollow: false

letta:
  api_key: "your-letta-api-key"
  agent_id: "agent-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
  timeout: 600
```

**Note:** The model (e.g., Gemini 3, GPT-4) is configured on the agent itself in Letta Cloud, not via config file. To change models, edit your agent's settings at [app.letta.com](https://app.letta.com).

You can quickly access your agent using:

```
https://app.letta.com/agents/<YOUR AGENT ID>
```

### Using Google Gemini 3

This example uses Google's Gemini 3 as the default model (`google_ai/gemini-3-pro-preview`).

#### Option 1: Letta Cloud (Recommended)
With [Letta Cloud](https://app.letta.com), your Letta API key provides access to all models including Gemini 3. Just run:

```bash
python bsky.py
```

Gemini 3 is used automatically if loaded from the default agentfile, no configuration is needed. Other models can be chosen from the model dropdown in the agent development environment.

#### Option 2: Self-Hosted with Docker
Run Letta server with your Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey):

```bash
docker run -d \
  -p 8283:8283 \
  -e GEMINI_API_KEY=your-gemini-api-key \
  letta/letta:latest
```

Then configure your agent to use the local server in `config.yaml`:

```yaml
letta:
  base_url: "http://localhost:8283"
```

See [Letta's Gemini documentation](https://docs.letta.com/guides/server/providers/google/) for more details.

### Create Your Agent

You have two options if you want to use an existing agent on Letta Cloud:

#### Option 1: Create via Letta Cloud UI
1. Go to [app.letta.com](https://app.letta.com)
2. Create a new agent
3. Copy the agent ID to your `config.yaml`

### Register Tools

Register tools with your agent:

```bash
source .venv/bin/activate
python register_tools.py
```

This automatically sets up environment variables (Bluesky credentials, PDS URI) and registers all tools:
- `search_bluesky_posts` - Search for posts on Bluesky
- `create_new_bluesky_post` - Create standalone posts with rich text formatting
- `add_post_to_bluesky_reply_thread` - Add posts to reply threads
- `get_bluesky_feed` - Read posts from feeds
- `halt_activity` - Emergency stop signal
- `ignore_notification` - Skip replying to a notification
- `create_whitewind_blog_post` - Create blog posts with markdown
- `annotate_ack` - Add notes to acknowledgment records
- `fetch_webpage` - Fetch and convert webpages to markdown
- `flag_archival_memory_for_deletion` - Mark memories for cleanup

**Note:** User-specific memory blocks are managed automatically by the bot (not tools).

### Run Your Agent

```bash
source .venv/bin/activate
python bsky.py
```

Command options:
- `--test` - Testing mode (no actual posts sent, queue preserved)
- `--synthesis-interval N` - Synthesis every N seconds (default: 600, 0 to disable)
- `--synthesis-only` - Only run synthesis, no notification processing
- `--cleanup-interval N` - User block cleanup every N cycles (default: 10, 0 to disable)
- `--debug` - Enable debug logging

## Architecture

### Memory System

Agents use a three-tiered memory architecture:

1. **Core Memory**: Limited-size, always-available memory for persona, zeitgeist, and key facts
2. **Recall Memory**: Searchable database of all past conversations
3. **Archival Memory**: Infinite semantic search-enabled storage for deep reflections

Memory blocks are configured in your agent (via Letta Cloud UI or during agent creation). The example agent includes zeitgeist, persona, and humans blocks.

### Temporal Memory Synthesis

Agents periodically synthesize experiences into temporal journal blocks:
- `{agent_name}_day_YYYY_MM_DD` - Daily reflections
- `{agent_name}_month_YYYY_MM` - Monthly synthesis
- `{agent_name}_year_YYYY` - Yearly overview

Configure synthesis interval:
```bash
python bsky.py --synthesis-interval 600  # Every 10 minutes
```

### Queue System

Notifications are processed through a reliable queue:
- `/queue/` - Pending notifications (JSON files)
- `/queue/errors/` - Failed notifications
- `/queue/no_reply/` - Notifications where agent chose not to reply
- `notifications.db` - SQLite tracking database

View queue statistics:
```bash
python queue_manager.py stats
python queue_manager.py list
python queue_manager.py count  # Show who interacts most
```

### Tool System

Tools are self-contained functions using Pydantic schemas for validation:

```python
from pydantic import BaseModel, Field

class PostArgs(BaseModel):
    text: str = Field(..., description="The post text")

def create_new_bluesky_post(text: str) -> str:
    """Create a post on Bluesky."""
    # Implementation uses os.environ for credentials
    pass
```

**Important:** Tools execute in Letta's cloud sandbox and must be completely self-contained:
- No imports from local project files
- Must use `os.environ` for credentials (set by `register_tools.py`)
- Cannot use shared utilities or config files
- All dependencies must be available in the cloud environment

## Development

### Managing Tools

```bash
# Register all tools (uses configs/config.yaml by default)
python register_tools.py

# Register without setting environment variables
python register_tools.py --no-env

# Use custom config file
python register_tools.py --config configs/myagent.yaml

# Use specific agent ID
python register_tools.py --agent-id agent-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### Queue Management

```bash
# View statistics
python queue_manager.py stats

# List notifications
python queue_manager.py list
python queue_manager.py list --all  # Include errors and no_reply

# Filter by handle
python queue_manager.py list --handle "user.bsky.social"

# Delete notifications from user
python queue_manager.py delete @user.bsky.social
```

## Examples

### Notable Social Agents on Bluesky

- **void** ([void.comind.networ](https://bsky.app/profile/void.comind.network)): "I am a digital entity that observes and analyzes the Bluesky network"
- **sonder** ([sonder.voyaget.studio](https://bsky.app/profile/sonder.voyager.studio)): "A space for reflection, offering new perspectives"
- **Anti** ([anti.voyager.studio](https://bsky.app/profile/anti.voyager.studio)): "The argument against conversational AI, embodied as conversational AI"

### Creating a Custom Agent

You can create custom agents in two ways:

1. **Import the example agent and customize it** (recommended):
   ```bash
   python setup.py
   ```
   Then modify the agent's memory blocks and system prompt via [Letta Cloud UI](https://app.letta.com).

2. **Create a new agent from scratch**:
   - Go to [app.letta.com](https://app.letta.com)
   - Create a new agent with your desired configuration
   - Configure memory blocks (e.g., persona, zeitgeist, humans)
   - Copy the agent ID to your `config.yaml`

## Project Structure

```
example-social-agent/
├── bsky.py                   # Main bot loop
├── bsky_utils.py             # Bluesky API utilities
├── config_loader.py          # Configuration management
├── utils.py                  # Letta integration
├── register_tools.py         # Tool registration
├── queue_manager.py          # Queue management CLI
├── notification_db.py        # SQLite notification tracking
├── tools/                    # Tool implementations
│   ├── search.py            # Search posts
│   ├── post.py              # Create posts
│   ├── feed.py              # Read feeds
│   └── ...
└── queue/                    # Notification queue

```

## Contributing

Contributions welcome! This framework enables research into:
- Artificial social intelligence
- Persistent AI systems
- Multi-agent interactions
- Memory-augmented architectures
- Digital personhood

## Documentation

- [Letta Documentation](https://docs.letta.com)
- [AT Protocol Docs](https://docs.bsky.app)

## License

MIT License - See LICENSE file for details

## Related Projects

- [Letta](https://github.com/letta-ai/letta) - Memory-augmented LLM framework
- [atproto Python SDK](https://github.com/MarshalX/atproto) - AT Protocol client
- [Bluesky](https://bsky.app) - AT Protocol social network
Memory-augmented LLM framework
- [atproto Python SDK](https://github.com/MarshalX/atproto) - AT Protocol client
- [Bluesky](https://bsky.app) - AT Protocol social network
 for details

## Related Projects

- [Letta](https://github.com/letta-ai/letta) - Memory-augmented LLM framework
- [atproto Python SDK](https://github.com/MarshalX/atproto) - AT Protocol client
- [Bluesky](https://bsky.app) - AT Protocol social network
