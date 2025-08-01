"""
Session migration utilities for transitioning from Redis to PostgreSQL
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import redis.asyncio as redis
import os
import sys

# Add path to pyairtable-common
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../pyairtable-common'))

try:
    from pyairtable_common.database.session_repository import PostgreSQLSessionRepository
    from pyairtable_common.database import get_async_session
    POSTGRES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PostgreSQL dependencies not available: {e}")
    POSTGRES_AVAILABLE = False

logger = logging.getLogger(__name__)


class SessionMigrationManager:
    """
    Manages the migration of session data from Redis to PostgreSQL
    
    This allows for gradual migration and fallback during the transition period.
    """
    
    def __init__(self, redis_url: str, redis_password: Optional[str] = None):
        self.redis_url = redis_url
        self.redis_password = redis_password
        self.redis_client: Optional[redis.Redis] = None
        self.postgres_repo: Optional[PostgreSQLSessionRepository] = None
        
        if POSTGRES_AVAILABLE:
            self.postgres_repo = PostgreSQLSessionRepository()
    
    async def initialize(self):
        """Initialize Redis connection for migration"""
        try:
            # Build Redis URL with password if provided
            redis_url = self.redis_url
            if self.redis_password and "://" in redis_url:
                protocol, rest = redis_url.split("://", 1)
                redis_url = f"{protocol}://:{self.redis_password}@{rest}"
            
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            if self.postgres_repo:
                await self.postgres_repo.initialize()
            
            logger.info("Session migration manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize migration manager: {e}")
            raise
    
    async def close(self):
        """Close connections"""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.postgres_repo:
            await self.postgres_repo.close()
    
    async def get_redis_sessions(self) -> List[str]:
        """Get all session IDs from Redis"""
        if not self.redis_client:
            return []
        
        try:
            session_keys = await self.redis_client.keys("session:*")
            return [key.replace("session:", "") for key in session_keys]
        except Exception as e:
            logger.error(f"Error getting Redis sessions: {e}")
            return []
    
    async def migrate_session(self, session_id: str) -> bool:
        """
        Migrate a single session from Redis to PostgreSQL
        
        Args:
            session_id: Session ID to migrate
            
        Returns:
            bool: True if migration successful, False otherwise
        """
        if not self.redis_client or not self.postgres_repo:
            logger.error("Migration manager not properly initialized")
            return False
        
        try:
            # Get session data from Redis
            session_data_str = await self.redis_client.get(f"session:{session_id}")
            if not session_data_str:
                logger.warning(f"Session {session_id} not found in Redis")
                return False
            
            session_data = json.loads(session_data_str)
            
            # Check if session already exists in PostgreSQL
            existing_session = await self.postgres_repo.get_session(session_id)
            if existing_session.get("session_id") == session_id and existing_session.get("history"):
                logger.info(f"Session {session_id} already exists in PostgreSQL, skipping")
                return True
            
            # Migrate session metadata - create new session in PostgreSQL
            async with get_async_session() as db:
                # Create session record
                from pyairtable_common.database.models import ConversationSession
                from sqlalchemy import select
                from datetime import datetime, timezone
                
                # Check if session exists
                result = await db.execute(
                    select(ConversationSession).where(
                        ConversationSession.session_id == session_id
                    )
                )
                existing = result.scalar_one_or_none()
                
                if not existing:
                    # Create new session
                    session = ConversationSession(
                        session_id=session_id,
                        created_at=datetime.fromisoformat(session_data.get("created_at", datetime.now(timezone.utc).isoformat())),
                        last_activity=datetime.fromisoformat(session_data.get("last_activity", datetime.now(timezone.utc).isoformat())),
                        metadata=session_data.get("metadata", {})
                    )
                    db.add(session)
                
                # Migrate message history
                history = session_data.get("history", [])
                for message in history:
                    await self.postgres_repo.add_message(
                        session_id=session_id,
                        role=message.get("role", "user"),
                        message=message.get("message", ""),
                        tools_used=message.get("tools_used", []),
                        timestamp=message.get("timestamp")
                    )
                
                await db.commit()
            
            logger.info(f"Successfully migrated session {session_id} with {len(history)} messages")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating session {session_id}: {e}")
            return False
    
    async def migrate_all_sessions(self, batch_size: int = 10) -> Dict[str, Any]:
        """
        Migrate all sessions from Redis to PostgreSQL
        
        Args:
            batch_size: Number of sessions to migrate concurrently
            
        Returns:
            Dict with migration statistics
        """
        if not POSTGRES_AVAILABLE:
            logger.error("PostgreSQL not available for migration")
            return {"error": "PostgreSQL not available"}
        
        start_time = datetime.now()
        session_ids = await self.get_redis_sessions()
        
        if not session_ids:
            logger.info("No sessions found in Redis")
            return {
                "total_sessions": 0,
                "migrated": 0,
                "failed": 0,
                "duration_seconds": 0
            }
        
        logger.info(f"Starting migration of {len(session_ids)} sessions")
        
        migrated = 0
        failed = 0
        
        # Process sessions in batches
        for i in range(0, len(session_ids), batch_size):
            batch = session_ids[i:i + batch_size]
            
            # Migrate batch concurrently
            tasks = [self.migrate_session(session_id) for session_id in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                    logger.error(f"Migration failed: {result}")
                elif result:
                    migrated += 1
                else:
                    failed += 1
            
            logger.info(f"Migrated batch {i//batch_size + 1}: {len([r for r in results if r is True])} successful")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = {
            "total_sessions": len(session_ids),
            "migrated": migrated,
            "failed": failed,
            "duration_seconds": duration,
            "completed_at": datetime.now().isoformat()
        }
        
        logger.info(f"Migration completed: {result}")
        return result
    
    async def verify_migration(self, session_id: str) -> Dict[str, Any]:
        """
        Verify that a session was migrated correctly
        
        Args:
            session_id: Session ID to verify
            
        Returns:
            Dict with verification results
        """
        if not self.redis_client or not self.postgres_repo:
            return {"error": "Migration manager not initialized"}
        
        try:
            # Get from Redis
            redis_data_str = await self.redis_client.get(f"session:{session_id}")
            redis_data = json.loads(redis_data_str) if redis_data_str else None
            
            # Get from PostgreSQL
            postgres_data = await self.postgres_repo.get_session(session_id)
            
            if not redis_data and not postgres_data:
                return {"status": "session_not_found"}
            
            if not redis_data:
                return {"status": "only_in_postgres", "postgres_messages": len(postgres_data.get("history", []))}
            
            if not postgres_data or not postgres_data.get("history"):
                return {"status": "only_in_redis", "redis_messages": len(redis_data.get("history", []))}
            
            redis_messages = len(redis_data.get("history", []))
            postgres_messages = len(postgres_data.get("history", []))
            
            return {
                "status": "verified",
                "redis_messages": redis_messages,
                "postgres_messages": postgres_messages,
                "messages_match": redis_messages == postgres_messages,
                "session_id": session_id
            }
            
        except Exception as e:
            return {"error": str(e), "session_id": session_id}


async def main():
    """CLI tool for running session migration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate sessions from Redis to PostgreSQL")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--redis-password", help="Redis password")
    parser.add_argument("--session-id", help="Migrate specific session ID")
    parser.add_argument("--verify", help="Verify migration of specific session ID")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for migration")
    
    args = parser.parse_args()
    
    manager = SessionMigrationManager(args.redis_url, args.redis_password)
    
    try:
        await manager.initialize()
        
        if args.verify:
            result = await manager.verify_migration(args.verify)
            print(f"Verification result: {json.dumps(result, indent=2)}")
        elif args.session_id:
            success = await manager.migrate_session(args.session_id)
            print(f"Migration {'successful' if success else 'failed'} for session {args.session_id}")
        else:
            result = await manager.migrate_all_sessions(args.batch_size)
            print(f"Migration results: {json.dumps(result, indent=2)}")
            
    finally:
        await manager.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())