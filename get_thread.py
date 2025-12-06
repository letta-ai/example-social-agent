#!/usr/bin/env python3
"""
Centralized script for retrieving Bluesky post threads from URIs.
Includes YAML-ified string conversion for easy LLM parsing.
"""

import argparse
import sys
import logging
from typing import Optional, Dict, Any
import yaml
from bsky_utils import default_login, thread_to_yaml_string

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("get_thread")


def get_thread_from_uri(uri: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a post thread from a Bluesky URI.
    
    Args:
        uri: The Bluesky post URI (e.g., at://did:plc:xyz/app.bsky.feed.post/abc123)
    
    Returns:
        Thread data or None if retrieval failed
    """
    try:
        client = default_login()
        logger.info(f"Fetching thread for URI: {uri}")
        
        thread = client.app.bsky.feed.get_post_thread({'uri': uri, 'parent_height': 80, 'depth': 10})
        return thread
        
    except Exception as e:
        logger.error(f"Error retrieving thread for URI {uri}: {e}")
        return None


# thread_to_yaml_string is now imported from bsky_utils


def main():
    """Main CLI interface for the thread retrieval script."""
    parser = argparse.ArgumentParser(
        description="Retrieve and display Bluesky post threads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_thread.py at://did:plc:xyz/app.bsky.feed.post/abc123
  python get_thread.py --raw at://did:plc:xyz/app.bsky.feed.post/abc123
  python get_thread.py --output thread.yaml at://did:plc:xyz/app.bsky.feed.post/abc123
        """
    )
    
    parser.add_argument(
        "uri", 
        help="Bluesky post URI to retrieve thread for"
    )
    
    parser.add_argument(
        "--raw", 
        action="store_true",
        help="Include all metadata fields (don't strip for LLM parsing)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file to write YAML to (default: stdout)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress info logging"
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Retrieve the thread
    thread = get_thread_from_uri(args.uri)
    
    if thread is None:
        logger.error("Failed to retrieve thread")
        sys.exit(1)
    
    # Convert to YAML
    yaml_output = thread_to_yaml_string(thread, strip_metadata=not args.raw)
    
    # Output the result
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(yaml_output)
            logger.info(f"Thread saved to {args.output}")
        except Exception as e:
            logger.error(f"Error writing to file {args.output}: {e}")
            sys.exit(1)
    else:
        print(yaml_output)


if __name__ == "__main__":
    main()