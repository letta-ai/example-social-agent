#!/usr/bin/env python3
"""Recovery tools for missed notifications."""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import bsky_utils
from notification_db import NotificationDB
from bsky import save_notification_to_queue, notification_to_dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def recover_notifications(hours=24, dry_run=True):
    """
    Recover notifications from the past N hours.
    
    Args:
        hours: Number of hours back to check for notifications
        dry_run: If True, only show what would be recovered without saving
    """
    logger.info(f"Recovering notifications from the past {hours} hours")
    logger.info(f"Dry run: {dry_run}")
    
    # Initialize Bluesky client
    client = bsky_utils.default_login()
    logger.info("Connected to Bluesky")
    
    # Initialize database
    db = NotificationDB()
    logger.info("Database initialized")
    
    # Fetch notifications
    all_notifications = []
    cursor = None
    page_count = 0
    max_pages = 50  # More pages for recovery
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    cutoff_iso = cutoff_time.isoformat()
    logger.info(f"Looking for notifications since: {cutoff_iso}")
    
    while page_count < max_pages:
        try:
            # Fetch notifications page
            if cursor:
                response = client.app.bsky.notification.list_notifications(
                    params={'cursor': cursor, 'limit': 100}
                )
            else:
                response = client.app.bsky.notification.list_notifications(
                    params={'limit': 100}
                )
            
            page_count += 1
            page_notifications = response.notifications
            
            if not page_notifications:
                break
            
            # Filter by time
            for notif in page_notifications:
                if hasattr(notif, 'indexed_at') and notif.indexed_at >= cutoff_iso:
                    all_notifications.append(notif)
                elif hasattr(notif, 'indexed_at') and notif.indexed_at < cutoff_iso:
                    # We've gone past our cutoff, stop fetching
                    logger.info(f"Reached notifications older than {hours} hours, stopping")
                    cursor = None
                    break
            
            # Check if there are more pages
            if cursor is None:
                break
            cursor = getattr(response, 'cursor', None)
            if not cursor:
                break
                
        except Exception as e:
            logger.error(f"Error fetching notifications page {page_count}: {e}")
            break
    
    logger.info(f"Found {len(all_notifications)} notifications in the time range")
    
    # Process notifications
    recovered = 0
    skipped_likes = 0
    already_processed = 0
    
    for notif in all_notifications:
        # Skip likes
        if hasattr(notif, 'reason') and notif.reason == 'like':
            skipped_likes += 1
            continue
        
        # Check if already processed
        notif_dict = notification_to_dict(notif)
        uri = notif_dict.get('uri', '')
        
        if db.is_processed(uri):
            already_processed += 1
            logger.debug(f"Already processed: {uri}")
            continue
        
        # Log what we would recover
        author = notif_dict.get('author', {})
        author_handle = author.get('handle', 'unknown')
        reason = notif_dict.get('reason', 'unknown')
        indexed_at = notif_dict.get('indexed_at', '')
        
        logger.info(f"Would recover: {reason} from @{author_handle} at {indexed_at}")
        
        if not dry_run:
            # Save to queue
            if save_notification_to_queue(notif_dict):
                recovered += 1
                logger.info(f"Recovered notification from @{author_handle}")
            else:
                logger.warning(f"Failed to queue notification from @{author_handle}")
        else:
            recovered += 1
    
    # Summary
    logger.info(f"Recovery summary:")
    logger.info(f"  • Total found: {len(all_notifications)}")
    logger.info(f"  • Skipped (likes): {skipped_likes}")
    logger.info(f"  • Already processed: {already_processed}")
    logger.info(f"  • {'Would recover' if dry_run else 'Recovered'}: {recovered}")
    
    if dry_run and recovered > 0:
        logger.info("Run with --execute to actually recover these notifications")
    
    return recovered


def check_database_health():
    """Check the health of the notification database."""
    db = NotificationDB()
    stats = db.get_stats()
    
    logger.info("Database Statistics:")
    logger.info(f"  • Total notifications: {stats.get('total', 0)}")
    logger.info(f"  • Pending: {stats.get('status_pending', 0)}")
    logger.info(f"  • Processed: {stats.get('status_processed', 0)}")
    logger.info(f"  • Ignored: {stats.get('status_ignored', 0)}")
    logger.info(f"  • No reply: {stats.get('status_no_reply', 0)}")
    logger.info(f"  • Errors: {stats.get('status_error', 0)}")
    logger.info(f"  • Recent (24h): {stats.get('recent_24h', 0)}")
    
    # Check for issues
    if stats.get('status_pending', 0) > 100:
        logger.warning(f"⚠️ High number of pending notifications: {stats.get('status_pending', 0)}")
    
    if stats.get('status_error', 0) > 50:
        logger.warning(f"⚠️ High number of error notifications: {stats.get('status_error', 0)}")
    
    return stats


def reset_notification_status(hours=1, dry_run=True):
    """
    Reset notifications from error/no_reply status back to pending.
    
    Args:
        hours: Reset notifications from the last N hours
        dry_run: If True, only show what would be reset
    """
    db = NotificationDB()
    cutoff_time = datetime.now() - timedelta(hours=hours)
    cutoff_iso = cutoff_time.isoformat()
    
    # Get notifications to reset
    cursor = db.conn.execute("""
        SELECT uri, status, indexed_at, author_handle 
        FROM notifications 
        WHERE status IN ('error', 'no_reply')
        AND indexed_at > ?
        ORDER BY indexed_at DESC
    """, (cutoff_iso,))
    
    notifications_to_reset = cursor.fetchall()
    
    if not notifications_to_reset:
        logger.info(f"No notifications to reset from the last {hours} hours")
        return 0
    
    logger.info(f"Found {len(notifications_to_reset)} notifications to reset")
    
    for notif in notifications_to_reset:
        logger.info(f"Would reset: {notif['status']} -> pending for @{notif['author_handle']} at {notif['indexed_at']}")
    
    if not dry_run:
        reset_count = db.conn.execute("""
            UPDATE notifications 
            SET status = 'pending', processed_at = NULL, error = NULL
            WHERE status IN ('error', 'no_reply')
            AND indexed_at > ?
        """, (cutoff_iso,)).rowcount
        
        db.conn.commit()
        logger.info(f"Reset {reset_count} notifications to pending status")
        return reset_count
    else:
        logger.info("Run with --execute to actually reset these notifications")
        return len(notifications_to_reset)


def main():
    parser = argparse.ArgumentParser(description="Notification recovery and management tools")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Recover command
    recover_parser = subparsers.add_parser('recover', help='Recover missed notifications')
    recover_parser.add_argument('--hours', type=int, default=24, 
                              help='Number of hours back to check (default: 24)')
    recover_parser.add_argument('--execute', action='store_true',
                              help='Actually recover notifications (default is dry run)')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Check database health')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset error notifications to pending')
    reset_parser.add_argument('--hours', type=int, default=1,
                            help='Reset notifications from last N hours (default: 1)')
    reset_parser.add_argument('--execute', action='store_true',
                            help='Actually reset notifications (default is dry run)')
    
    args = parser.parse_args()
    
    if args.command == 'recover':
        recover_notifications(hours=args.hours, dry_run=not args.execute)
    elif args.command == 'health':
        check_database_health()
    elif args.command == 'reset':
        reset_notification_status(hours=args.hours, dry_run=not args.execute)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()