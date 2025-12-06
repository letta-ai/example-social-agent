#!/usr/bin/env python3
"""Queue management utilities for Void bot."""
import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm

console = Console()

# Queue directories
QUEUE_DIR = Path("queue")
QUEUE_ERROR_DIR = QUEUE_DIR / "errors"
QUEUE_NO_REPLY_DIR = QUEUE_DIR / "no_reply"


def load_notification(filepath: Path) -> dict:
    """Load a notification from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading {filepath}: {e}[/red]")
        return None


def list_notifications(handle_filter: str = None, show_all: bool = False):
    """List all notifications in the queue, optionally filtered by handle."""
    # Collect notifications from all directories if show_all is True
    if show_all:
        dirs_to_check = [QUEUE_DIR, QUEUE_ERROR_DIR, QUEUE_NO_REPLY_DIR]
    else:
        dirs_to_check = [QUEUE_DIR]
    
    all_notifications = []
    
    for directory in dirs_to_check:
        if not directory.exists():
            continue
            
        # Get source directory name for display
        if directory == QUEUE_DIR:
            source = "queue"
        elif directory == QUEUE_ERROR_DIR:
            source = "errors"
        elif directory == QUEUE_NO_REPLY_DIR:
            source = "no_reply"
        else:
            source = "unknown"
        
        for filepath in directory.glob("*.json"):
            # Skip subdirectories
            if filepath.is_dir():
                continue
                
            notif = load_notification(filepath)
            if notif and isinstance(notif, dict):
                notif['_filepath'] = filepath
                notif['_source'] = source
                
                # Apply handle filter if specified
                if handle_filter:
                    author_handle = notif.get('author', {}).get('handle', '')
                    if handle_filter.lower() not in author_handle.lower():
                        continue
                
                all_notifications.append(notif)
    
    # Sort by indexed_at
    all_notifications.sort(key=lambda x: x.get('indexed_at', ''), reverse=True)
    
    # Display results
    if not all_notifications:
        if handle_filter:
            console.print(f"[yellow]No notifications found for handle containing '{handle_filter}'[/yellow]")
        else:
            console.print("[yellow]No notifications found in queue[/yellow]")
        return
    
    table = Table(title=f"Queue Notifications ({len(all_notifications)} total)")
    table.add_column("File", style="cyan", width=20)
    table.add_column("Source", style="magenta", width=10)
    table.add_column("Handle", style="green", width=25)
    table.add_column("Display Name", width=25)
    table.add_column("Text", width=40)
    table.add_column("Time", style="dim", width=20)
    
    for notif in all_notifications:
        author = notif.get('author', {})
        handle = author.get('handle', 'unknown')
        display_name = author.get('display_name', '')
        text = notif.get('record', {}).get('text', '')[:40]
        if len(notif.get('record', {}).get('text', '')) > 40:
            text += "..."
        indexed_at = notif.get('indexed_at', '')[:19]  # Trim milliseconds
        filename = notif['_filepath'].name[:20]
        source = notif['_source']
        
        table.add_row(filename, source, f"@{handle}", display_name, text, indexed_at)
    
    console.print(table)
    return all_notifications


def delete_by_handle(handle: str, dry_run: bool = False, force: bool = False):
    """Delete all notifications from a specific handle."""
    # Remove @ if present
    handle = handle.lstrip('@')
    
    # Find all notifications from this handle
    console.print(f"\\n[bold]Searching for notifications from @{handle}...[/bold]\\n")
    
    to_delete = []
    dirs_to_check = [QUEUE_DIR, QUEUE_ERROR_DIR, QUEUE_NO_REPLY_DIR]
    
    for directory in dirs_to_check:
        if not directory.exists():
            continue
            
        for filepath in directory.glob("*.json"):
            if filepath.is_dir():
                continue
                
            notif = load_notification(filepath)
            if notif and isinstance(notif, dict):
                author_handle = notif.get('author', {}).get('handle', '')
                if author_handle.lower() == handle.lower():
                    to_delete.append({
                        'filepath': filepath,
                        'notif': notif,
                        'source': directory.name
                    })
    
    if not to_delete:
        console.print(f"[yellow]No notifications found from @{handle}[/yellow]")
        return
    
    # Display what will be deleted
    table = Table(title=f"Notifications to Delete from @{handle}")
    table.add_column("File", style="cyan")
    table.add_column("Location", style="magenta")
    table.add_column("Text", width=50)
    table.add_column("Time", style="dim")
    
    for item in to_delete:
        notif = item['notif']
        text = notif.get('record', {}).get('text', '')[:50]
        if len(notif.get('record', {}).get('text', '')) > 50:
            text += "..."
        indexed_at = notif.get('indexed_at', '')[:19]
        
        table.add_row(
            item['filepath'].name,
            item['source'],
            text,
            indexed_at
        )
    
    console.print(table)
    console.print(f"\\n[bold red]Found {len(to_delete)} notifications to delete[/bold red]")
    
    if dry_run:
        console.print("\\n[yellow]DRY RUN - No files were deleted[/yellow]")
        return
    
    # Confirm deletion
    if not force and not Confirm.ask("\\nDo you want to delete these notifications?"):
        console.print("[yellow]Deletion cancelled[/yellow]")
        return
    
    # Delete the files
    deleted_count = 0
    for item in to_delete:
        try:
            item['filepath'].unlink()
            deleted_count += 1
            console.print(f"[green]✓[/green] Deleted {item['filepath'].name}")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to delete {item['filepath'].name}: {e}")
    
    console.print(f"\\n[bold green]Successfully deleted {deleted_count} notifications[/bold green]")


def count_by_handle():
    """Show detailed count of notifications by handle."""
    handle_counts = {}
    
    # Collect counts from all directories
    for directory, location in [(QUEUE_DIR, 'queue'), (QUEUE_ERROR_DIR, 'errors'), (QUEUE_NO_REPLY_DIR, 'no_reply')]:
        if not directory.exists():
            continue
            
        for filepath in directory.glob("*.json"):
            if filepath.is_dir():
                continue
                
            notif = load_notification(filepath)
            if notif and isinstance(notif, dict):
                handle = notif.get('author', {}).get('handle', 'unknown')
                
                if handle not in handle_counts:
                    handle_counts[handle] = {'queue': 0, 'errors': 0, 'no_reply': 0, 'total': 0}
                
                handle_counts[handle][location] += 1
                handle_counts[handle]['total'] += 1
    
    if not handle_counts:
        console.print("[yellow]No notifications found in any queue[/yellow]")
        return
    
    # Sort by total count
    sorted_handles = sorted(handle_counts.items(), key=lambda x: x[1]['total'], reverse=True)
    
    # Display results
    table = Table(title=f"Notification Count by Handle ({len(handle_counts)} unique handles)")
    table.add_column("Handle", style="green", width=30)
    table.add_column("Queue", style="cyan", justify="right")
    table.add_column("Errors", style="red", justify="right")
    table.add_column("No Reply", style="yellow", justify="right")
    table.add_column("Total", style="bold magenta", justify="right")
    
    for handle, counts in sorted_handles:
        table.add_row(
            f"@{handle}",
            str(counts['queue']) if counts['queue'] > 0 else "-",
            str(counts['errors']) if counts['errors'] > 0 else "-",
            str(counts['no_reply']) if counts['no_reply'] > 0 else "-",
            str(counts['total'])
        )
    
    console.print(table)
    
    # Summary statistics
    total_notifications = sum(h['total'] for h in handle_counts.values())
    avg_per_handle = total_notifications / len(handle_counts)
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total notifications: {total_notifications}")
    console.print(f"  Unique handles: {len(handle_counts)}")
    console.print(f"  Average per handle: {avg_per_handle:.1f}")
    
    # Top user info
    if sorted_handles:
        top_handle, top_counts = sorted_handles[0]
        percentage = (top_counts['total'] / total_notifications) * 100
        console.print(f"  Most active: @{top_handle} ({top_counts['total']} notifications, {percentage:.1f}% of total)")


def stats():
    """Show queue statistics."""
    stats_data = {
        'queue': {'count': 0, 'handles': set()},
        'errors': {'count': 0, 'handles': set()},
        'no_reply': {'count': 0, 'handles': set()}
    }
    
    # Collect stats
    for directory, key in [(QUEUE_DIR, 'queue'), (QUEUE_ERROR_DIR, 'errors'), (QUEUE_NO_REPLY_DIR, 'no_reply')]:
        if not directory.exists():
            continue
            
        for filepath in directory.glob("*.json"):
            if filepath.is_dir():
                continue
                
            notif = load_notification(filepath)
            if notif and isinstance(notif, dict):
                stats_data[key]['count'] += 1
                handle = notif.get('author', {}).get('handle', 'unknown')
                stats_data[key]['handles'].add(handle)
    
    # Display stats
    table = Table(title="Queue Statistics")
    table.add_column("Location", style="cyan")
    table.add_column("Count", style="yellow")
    table.add_column("Unique Handles", style="green")
    
    for key, label in [('queue', 'Active Queue'), ('errors', 'Errors'), ('no_reply', 'No Reply')]:
        table.add_row(
            label,
            str(stats_data[key]['count']),
            str(len(stats_data[key]['handles']))
        )
    
    console.print(table)
    
    # Show top handles
    all_handles = {}
    for location_data in stats_data.values():
        for handle in location_data['handles']:
            all_handles[handle] = all_handles.get(handle, 0) + 1
    
    if all_handles:
        sorted_handles = sorted(all_handles.items(), key=lambda x: x[1], reverse=True)[:10]
        
        top_table = Table(title="Top 10 Handles by Notification Count")
        top_table.add_column("Handle", style="green")
        top_table.add_column("Count", style="yellow")
        
        for handle, count in sorted_handles:
            top_table.add_row(f"@{handle}", str(count))
        
        console.print("\\n")
        console.print(top_table)


def main():
    parser = argparse.ArgumentParser(description="Manage Void bot notification queue")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List notifications in queue')
    list_parser.add_argument('--handle', help='Filter by handle (partial match)')
    list_parser.add_argument('--all', action='store_true', help='Include errors and no_reply folders')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete notifications from a specific handle')
    delete_parser.add_argument('handle', help='Handle to delete notifications from')
    delete_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without deleting')
    delete_parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show queue statistics')
    
    # Count command
    count_parser = subparsers.add_parser('count', help='Show detailed count by handle')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_notifications(args.handle, args.all)
    elif args.command == 'delete':
        delete_by_handle(args.handle, args.dry_run, args.force)
    elif args.command == 'stats':
        stats()
    elif args.command == 'count':
        count_by_handle()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()