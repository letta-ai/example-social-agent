#!/usr/bin/env python3
"""SQLite database for robust notification tracking."""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Set, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class NotificationDB:
    """Database for tracking notification processing state."""
    
    def __init__(self, db_path: str = "queue/notifications.db"):
        """Initialize the notification database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Create main notifications table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                uri TEXT PRIMARY KEY,
                indexed_at TEXT NOT NULL,
                processed_at TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                reason TEXT,
                author_handle TEXT,
                author_did TEXT,
                text TEXT,
                parent_uri TEXT,
                root_uri TEXT,
                error TEXT,
                metadata TEXT
            )
        """)
        
        # Create indexes for faster lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_indexed_at 
            ON notifications(indexed_at DESC)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status 
            ON notifications(status)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_author_handle 
            ON notifications(author_handle)
        """)
        
        # Create session tracking table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                last_seen_at TEXT,
                notifications_processed INTEGER DEFAULT 0,
                notifications_skipped INTEGER DEFAULT 0,
                notifications_error INTEGER DEFAULT 0
            )
        """)
        
        self.conn.commit()
    
    def add_notification(self, notif_dict: Dict) -> bool:
        """Add a notification to the database."""
        try:
            # Handle None input
            if not notif_dict:
                return False
                
            # Extract key fields
            uri = notif_dict.get('uri', '')
            if not uri:
                return False
            
            indexed_at = notif_dict.get('indexed_at', '')
            reason = notif_dict.get('reason', '')
            author = notif_dict.get('author', {}) if notif_dict.get('author') else {}
            author_handle = author.get('handle', '') if author else ''
            author_did = author.get('did', '') if author else ''
            
            # Extract text from record if available (handle None records)
            record = notif_dict.get('record') or {}
            text = record.get('text', '')[:500] if record else ''
            
            # Extract thread info
            parent_uri = None
            root_uri = None
            if record and 'reply' in record and record['reply']:
                reply_info = record['reply']
                if reply_info and isinstance(reply_info, dict):
                    parent_info = reply_info.get('parent', {})
                    root_info = reply_info.get('root', {})
                    if parent_info:
                        parent_uri = parent_info.get('uri')
                    if root_info:
                        root_uri = root_info.get('uri')
            
            # Store additional metadata as JSON
            metadata = {
                'cid': notif_dict.get('cid'),
                'labels': notif_dict.get('labels', []),
                'is_read': notif_dict.get('is_read', False)
            }
            
            self.conn.execute("""
                INSERT OR IGNORE INTO notifications 
                (uri, indexed_at, reason, author_handle, author_did, text, 
                 parent_uri, root_uri, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """, (uri, indexed_at, reason, author_handle, author_did, text,
                  parent_uri, root_uri, json.dumps(metadata)))
            
            self.conn.commit()
            return True
        
        except Exception as e:
            logger.error(f"Error adding notification to DB: {e}")
            return False
    
    def is_processed(self, uri: str) -> bool:
        """Check if a notification has been processed."""
        cursor = self.conn.execute("""
            SELECT status FROM notifications WHERE uri = ?
        """, (uri,))
        row = cursor.fetchone()
        
        if row:
            return row['status'] in ['processed', 'ignored', 'no_reply']
        return False
    
    def mark_processed(self, uri: str, status: str = 'processed', error: str = None):
        """Mark a notification as processed."""
        try:
            self.conn.execute("""
                UPDATE notifications 
                SET status = ?, processed_at = ?, error = ?
                WHERE uri = ?
            """, (status, datetime.now().isoformat(), error, uri))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error marking notification processed: {e}")
    
    def get_unprocessed(self, limit: int = 100) -> List[Dict]:
        """Get unprocessed notifications."""
        cursor = self.conn.execute("""
            SELECT * FROM notifications 
            WHERE status = 'pending'
            ORDER BY indexed_at ASC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor]
    
    def get_latest_processed_time(self) -> Optional[str]:
        """Get the timestamp of the most recently processed notification."""
        cursor = self.conn.execute("""
            SELECT MAX(indexed_at) as latest 
            FROM notifications 
            WHERE status IN ('processed', 'ignored', 'no_reply')
        """)
        row = cursor.fetchone()
        return row['latest'] if row and row['latest'] else None
    
    def cleanup_old_records(self, days: int = 7):
        """Remove records older than specified days."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        deleted = self.conn.execute("""
            DELETE FROM notifications 
            WHERE indexed_at < ? 
            AND status IN ('processed', 'ignored', 'no_reply', 'error')
        """, (cutoff_date,)).rowcount
        
        self.conn.commit()
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old notification records")
            # Vacuum to reclaim space
            self.conn.execute("VACUUM")
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        stats = {}
        
        # Count by status
        cursor = self.conn.execute("""
            SELECT status, COUNT(*) as count 
            FROM notifications 
            GROUP BY status
        """)
        
        for row in cursor:
            stats[f"status_{row['status']}"] = row['count']
        
        # Total count
        cursor = self.conn.execute("SELECT COUNT(*) as total FROM notifications")
        stats['total'] = cursor.fetchone()['total']
        
        # Recent activity (last 24h)
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        cursor = self.conn.execute("""
            SELECT COUNT(*) as recent 
            FROM notifications 
            WHERE indexed_at > ?
        """, (yesterday,))
        stats['recent_24h'] = cursor.fetchone()['recent']
        
        return stats
    
    def start_session(self) -> int:
        """Start a new processing session."""
        cursor = self.conn.execute("""
            INSERT INTO sessions (started_at, last_seen_at)
            VALUES (?, ?)
        """, (datetime.now().isoformat(), datetime.now().isoformat()))
        self.conn.commit()
        return cursor.lastrowid
    
    def update_session(self, session_id: int, processed: int = 0, skipped: int = 0, error: int = 0):
        """Update session statistics."""
        self.conn.execute("""
            UPDATE sessions 
            SET last_seen_at = ?,
                notifications_processed = notifications_processed + ?,
                notifications_skipped = notifications_skipped + ?,
                notifications_error = notifications_error + ?
            WHERE id = ?
        """, (datetime.now().isoformat(), processed, skipped, error, session_id))
        self.conn.commit()
    
    def end_session(self, session_id: int):
        """End a processing session."""
        self.conn.execute("""
            UPDATE sessions 
            SET ended_at = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), session_id))
        self.conn.commit()
    
    def get_processed_uris(self, limit: int = 10000) -> Set[str]:
        """Get set of processed URIs for compatibility with existing code."""
        cursor = self.conn.execute("""
            SELECT uri FROM notifications 
            WHERE status IN ('processed', 'ignored', 'no_reply')
            ORDER BY processed_at DESC
            LIMIT ?
        """, (limit,))
        
        return {row['uri'] for row in cursor}
    
    def migrate_from_json(self, json_path: str = "queue/processed_notifications.json"):
        """Migrate data from the old JSON format."""
        json_file = Path(json_path)
        if not json_file.exists():
            return
        
        try:
            with open(json_file, 'r') as f:
                uris = json.load(f)
            
            migrated = 0
            for uri in uris:
                # Add as processed with unknown timestamp
                self.conn.execute("""
                    INSERT OR IGNORE INTO notifications 
                    (uri, indexed_at, status, processed_at)
                    VALUES (?, ?, 'processed', ?)
                """, (uri, datetime.now().isoformat(), datetime.now().isoformat()))
                migrated += 1
            
            self.conn.commit()
            logger.info(f"Migrated {migrated} URIs from JSON to database")
            
            # Rename old file to backup
            backup_path = json_file.with_suffix('.json.backup')
            json_file.rename(backup_path)
            logger.info(f"Renamed old JSON file to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error migrating from JSON: {e}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()