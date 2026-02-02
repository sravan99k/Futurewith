# System Design - Practice Exercises

## Table of Contents

1. [Scalable Web Application Design](#scalable-web-application-design)
2. [Distributed System Architecture](#distributed-system-architecture)
3. [Database Scaling and Sharding](#database-scaling-and-sharding)
4. [Caching Strategy Implementation](#caching-strategy-implementation)
5. [Load Balancing and Traffic Management](#load-balancing-and-traffic-management)
6. [Microservices Communication Patterns](#microservices-communication-patterns)
7. [Event-Driven Architecture Design](#event-driven-architecture-design)
8. [High Availability and Disaster Recovery](#high-availability-and-disaster-recovery)
9. [Performance Optimization and Monitoring](#performance-optimization-and-monitoring)
10. [Security and Compliance Architecture](#security-and-compliance-architecture)

## Practice Exercise 1: Scalable Web Application Design

### Objective

Design a complete scalable web application architecture that can handle millions of users with high availability and performance.

### Exercise Details

**Time Required**: 2-3 weeks with iterative design refinement
**Difficulty**: Advanced

### Week 1: Requirements Analysis and Initial Architecture

#### Project: Social Media Platform (Instagram-like)

**Business Requirements**:

- 50 million daily active users
- 500 million photos uploaded daily
- Real-time feed updates
- Global user base (multiple regions)
- 99.9% uptime requirement
- Mobile and web clients

#### System Requirements Analysis

```markdown
## Functional Requirements

### Core Features

1. **User Management**
   - User registration/authentication
   - Profile management
   - Follow/unfollow functionality
   - Privacy settings

2. **Content Management**
   - Photo/video upload and storage
   - Content processing (thumbnails, filters)
   - Content metadata (tags, location, etc.)
   - Content moderation

3. **Social Features**
   - News feed generation
   - Like/comment functionality
   - Direct messaging
   - Story features (temporary content)

4. **Discovery Features**
   - Search (users, hashtags, locations)
   - Trending content
   - Content recommendations

### Non-Functional Requirements

1. **Scale**
   - 50M DAU
   - 500M photos/day (~6K photos/second avg, 60K peak)
   - 5B feed requests/day (~58K requests/second avg, 580K peak)

2. **Performance**
   - Feed loading: < 200ms (p95)
   - Photo upload: < 2s for mobile networks
   - Search results: < 100ms
   - Real-time notifications: < 500ms

3. **Availability**
   - 99.9% uptime (8.76 hours downtime/year)
   - Graceful degradation during outages
   - Cross-region disaster recovery

4. **Storage**
   - Photos: ~2MB average size = 1PB/day
   - Metadata and user data: ~1TB/day
   - 7-year retention policy
```

#### High-Level Architecture Design

```markdown
## System Architecture Overview

### Tier Structure
```

┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Client Tier │ │ Service Tier │ │ Storage Tier │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ • Mobile Apps │◄──►│ • API Gateway │◄──►│ • User DB │
│ • Web App │ │ • Auth Service │ │ • Content DB │
│ • Third-party │ │ • User Service │ │ • Media Storage │
│ Integrations │ │ • Media Service │ │ • Cache Layer │
└─────────────────┘ │ • Feed Service │ │ • Search Index │
│ • Notification │ │ • Message Queue │
│ Service │ └─────────────────┘
└─────────────────┘

````

### Component Breakdown

#### API Gateway Layer
- **Load Balancer**: HAProxy/NGINX for traffic distribution
- **API Gateway**: Kong/Zuul for routing, rate limiting, authentication
- **CDN**: CloudFront/CloudFlare for static content delivery

#### Application Services
- **User Service**: Authentication, profile management, relationships
- **Media Service**: Upload processing, storage, metadata management
- **Feed Service**: Timeline generation, content ranking
- **Notification Service**: Real-time updates, push notifications
- **Search Service**: User/content discovery, trending analysis

#### Data Storage Layer
- **Primary Database**: PostgreSQL clusters for user/metadata
- **Media Storage**: Amazon S3/Google Cloud Storage for photos/videos
- **Cache**: Redis clusters for hot data and session storage
- **Search**: Elasticsearch for full-text search and analytics
- **Message Queue**: Apache Kafka for event streaming

### Detailed Component Design

#### 1. API Gateway Design
```yaml
# API Gateway Configuration
apiGateway:
  rateLimiting:
    authenticated: 1000/hour
    unauthenticated: 100/hour

  routing:
    - path: /api/v1/users/*
      service: user-service
      auth_required: true

    - path: /api/v1/media/*
      service: media-service
      auth_required: true
      upload_limit: 10MB

    - path: /api/v1/feed/*
      service: feed-service
      auth_required: true
      cache_ttl: 60s

  cors:
    allowed_origins: ["https://app.example.com"]
    allowed_methods: ["GET", "POST", "PUT", "DELETE"]
````

#### 2. User Service Architecture

```python
# User Service Design Pattern

class UserService:
    def __init__(self):
        self.user_db = PostgreSQLCluster(
            primary="user-db-primary",
            replicas=["user-db-replica-1", "user-db-replica-2"]
        )
        self.cache = RedisCluster(
            nodes=["cache-1", "cache-2", "cache-3"]
        )
        self.event_bus = KafkaProducer(topics=["user-events"])

    async def get_user_profile(self, user_id: str) -> UserProfile:
        # Try cache first
        cached_user = await self.cache.get(f"user:{user_id}")
        if cached_user:
            return UserProfile.from_cache(cached_user)

        # Fallback to database (read replica)
        user = await self.user_db.read_replica.fetch_user(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        # Cache for future requests
        await self.cache.setex(
            f"user:{user_id}",
            3600,  # 1 hour TTL
            user.to_cache_format()
        )

        return UserProfile.from_db(user)

    async def update_user_profile(self, user_id: str, updates: dict) -> UserProfile:
        # Validate updates
        validated_updates = UserProfileSchema.validate(updates)

        # Update primary database
        updated_user = await self.user_db.primary.update_user(
            user_id, validated_updates
        )

        # Invalidate cache
        await self.cache.delete(f"user:{user_id}")

        # Publish event for other services
        await self.event_bus.publish({
            "event_type": "user_profile_updated",
            "user_id": user_id,
            "changes": validated_updates,
            "timestamp": datetime.utcnow().isoformat()
        })

        return UserProfile.from_db(updated_user)

# Database Schema for User Service
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(100),
    bio TEXT,
    avatar_url VARCHAR(500),
    verified BOOLEAN DEFAULT FALSE,
    privacy_settings JSONB DEFAULT '{"profile_public": true}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    follower_id UUID NOT NULL REFERENCES users(id),
    following_id UUID NOT NULL REFERENCES users(id),
    status VARCHAR(20) DEFAULT 'active', -- active, blocked, pending
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(follower_id, following_id)
);

-- Indexes for performance
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_relationships_follower ON user_relationships(follower_id);
CREATE INDEX idx_relationships_following ON user_relationships(following_id);
```

#### 3. Media Service Architecture

```python
# Media Service with distributed storage

class MediaService:
    def __init__(self):
        self.storage = S3StorageAdapter(
            bucket="media-storage-primary",
            backup_bucket="media-storage-backup"
        )
        self.cdn = CloudFrontAdapter()
        self.image_processor = ImageProcessingService()
        self.metadata_db = PostgreSQLCluster()
        self.event_bus = KafkaProducer()

    async def upload_media(
        self,
        user_id: str,
        file_data: bytes,
        metadata: MediaMetadata
    ) -> MediaUploadResult:
        # Generate unique media ID
        media_id = str(uuid.uuid4())

        # Validate file
        validation_result = await self._validate_media_file(file_data, metadata)
        if not validation_result.is_valid:
            raise MediaValidationError(validation_result.errors)

        # Process image asynchronously
        processing_task = await self.image_processor.submit_job({
            "media_id": media_id,
            "user_id": user_id,
            "original_data": file_data,
            "generate_thumbnails": True,
            "apply_filters": metadata.get("filters", [])
        })

        # Store original file
        original_key = f"media/{user_id}/{media_id}/original.jpg"
        await self.storage.put_object(original_key, file_data)

        # Store metadata in database
        media_record = await self.metadata_db.primary.insert_media({
            "id": media_id,
            "user_id": user_id,
            "original_url": f"https://cdn.example.com/{original_key}",
            "status": "processing",
            "metadata": metadata,
            "created_at": datetime.utcnow()
        })

        # Publish event
        await self.event_bus.publish({
            "event_type": "media_uploaded",
            "media_id": media_id,
            "user_id": user_id,
            "processing_job_id": processing_task.job_id
        })

        return MediaUploadResult(
            media_id=media_id,
            status="processing",
            processing_job_id=processing_task.job_id
        )

    async def _validate_media_file(
        self,
        file_data: bytes,
        metadata: MediaMetadata
    ) -> ValidationResult:
        # File size validation
        if len(file_data) > 50 * 1024 * 1024:  # 50MB limit
            return ValidationResult(False, ["File too large"])

        # File type validation
        file_type = self._detect_file_type(file_data)
        if file_type not in ["image/jpeg", "image/png", "video/mp4"]:
            return ValidationResult(False, ["Unsupported file type"])

        # Content scanning (anti-malware, inappropriate content)
        scan_result = await self._scan_content(file_data)
        if not scan_result.is_safe:
            return ValidationResult(False, ["Content safety check failed"])

        return ValidationResult(True, [])

# Image Processing Service (separate microservice)
class ImageProcessingService:
    def __init__(self):
        self.processing_queue = CeleryApp(
            broker="redis://processing-queue-cluster"
        )
        self.storage = S3StorageAdapter()

    @celery.task(bind=True, max_retries=3)
    async def process_media(self, job_data: dict):
        media_id = job_data["media_id"]
        user_id = job_data["user_id"]

        try:
            # Download original file
            original_key = f"media/{user_id}/{media_id}/original.jpg"
            original_data = await self.storage.get_object(original_key)

            # Generate multiple sizes
            sizes = [
                {"name": "thumbnail", "width": 150, "height": 150},
                {"name": "small", "width": 320, "height": 320},
                {"name": "medium", "width": 640, "height": 640},
                {"name": "large", "width": 1080, "height": 1080}
            ]

            processed_urls = {}

            for size in sizes:
                # Resize image
                resized_data = await self._resize_image(
                    original_data,
                    size["width"],
                    size["height"]
                )

                # Upload resized version
                size_key = f"media/{user_id}/{media_id}/{size['name']}.jpg"
                await self.storage.put_object(size_key, resized_data)
                processed_urls[size["name"]] = f"https://cdn.example.com/{size_key}"

            # Update database with processed URLs
            await self.metadata_db.primary.update_media(media_id, {
                "status": "completed",
                "processed_urls": processed_urls,
                "processed_at": datetime.utcnow()
            })

            # Publish completion event
            await self.event_bus.publish({
                "event_type": "media_processing_completed",
                "media_id": media_id,
                "processed_urls": processed_urls
            })

        except Exception as e:
            # Handle processing failure
            await self._handle_processing_failure(media_id, str(e))
            raise self.retry(countdown=60)
```

### Week 2-3: Advanced System Design Patterns

#### Feed Generation System Design

```python
# High-Performance Feed Generation System

class FeedGenerationService:
    """
    Implements both push and pull models for feed generation
    to optimize for different user types and engagement patterns
    """

    def __init__(self):
        self.user_service = UserServiceClient()
        self.media_service = MediaServiceClient()
        self.cache = RedisCluster()
        self.feed_db = PostgreSQLCluster()
        self.recommendation_service = MLRecommendationService()

    async def generate_home_feed(
        self,
        user_id: str,
        page_size: int = 20,
        cursor: str = None
    ) -> FeedResponse:
        """
        Hybrid approach: pre-computed + real-time generation
        """
        # Try to get pre-computed feed from cache
        cached_feed = await self._get_cached_feed(user_id, cursor)
        if cached_feed and len(cached_feed) >= page_size:
            return self._format_feed_response(cached_feed[:page_size])

        # Generate feed in real-time for active users
        user_profile = await self.user_service.get_user_profile(user_id)

        if user_profile.activity_level == "high":
            # Use pull model for active users
            feed_posts = await self._generate_pull_feed(user_id, page_size, cursor)
        else:
            # Use push model for less active users
            feed_posts = await self._generate_push_feed(user_id, page_size, cursor)

        # Cache generated feed
        await self._cache_feed(user_id, feed_posts, ttl=600)  # 10 minutes

        return self._format_feed_response(feed_posts)

    async def _generate_pull_feed(
        self,
        user_id: str,
        page_size: int,
        cursor: str = None
    ) -> List[FeedPost]:
        """Pull model: Generate feed on-demand from followed users"""

        # Get users that this user follows
        following_users = await self.user_service.get_following_list(user_id)

        if not following_users:
            # New user - show trending/recommended content
            return await self._get_discovery_feed(user_id, page_size)

        # Get recent posts from followed users
        recent_posts_query = """
        WITH ranked_posts AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY p.user_id
                    ORDER BY p.created_at DESC
                ) as user_post_rank
            FROM posts p
            WHERE p.user_id = ANY($1)
            AND p.created_at >= NOW() - INTERVAL '7 days'
            AND p.is_active = true
        )
        SELECT * FROM ranked_posts
        WHERE user_post_rank <= 10  -- Max 10 recent posts per user
        ORDER BY created_at DESC
        LIMIT $2
        """

        recent_posts = await self.feed_db.replica.fetch(
            recent_posts_query,
            following_users,
            page_size * 2  # Get more for ranking
        )

        # Apply ML ranking for personalization
        ranked_posts = await self.recommendation_service.rank_posts(
            user_id,
            recent_posts
        )

        return ranked_posts[:page_size]

    async def _generate_push_feed(
        self,
        user_id: str,
        page_size: int,
        cursor: str = None
    ) -> List[FeedPost]:
        """Push model: Use pre-computed feed with fallback"""

        # Check pre-computed feed table
        precomputed_query = """
        SELECT pf.*, p.content, p.media_urls, u.username, u.avatar_url
        FROM precomputed_feeds pf
        JOIN posts p ON pf.post_id = p.id
        JOIN users u ON p.user_id = u.id
        WHERE pf.user_id = $1
        AND ($2 IS NULL OR pf.score < $2)  -- Cursor-based pagination
        ORDER BY pf.score DESC, pf.created_at DESC
        LIMIT $3
        """

        feed_posts = await self.feed_db.replica.fetch(
            precomputed_query,
            user_id,
            cursor,
            page_size
        )

        if len(feed_posts) < page_size:
            # Fallback to pull model if not enough pre-computed content
            additional_posts = await self._generate_pull_feed(
                user_id,
                page_size - len(feed_posts)
            )
            feed_posts.extend(additional_posts)

        return feed_posts

# Background Feed Pre-computation Service
class FeedPrecomputationService:
    """
    Background service that pre-computes feeds for less active users
    using a fanout-on-write approach
    """

    def __init__(self):
        self.event_consumer = KafkaConsumer(topics=["post-published"])
        self.user_service = UserServiceClient()
        self.feed_db = PostgreSQLCluster()
        self.worker_pool = ThreadPoolExecutor(max_workers=50)

    async def start_processing(self):
        """Start consuming post publication events"""
        async for event in self.event_consumer:
            if event.topic == "post-published":
                await self._handle_new_post(event.data)

    async def _handle_new_post(self, post_data: dict):
        """Fan out new post to followers' pre-computed feeds"""
        post_id = post_data["post_id"]
        author_id = post_data["user_id"]

        # Get followers of the post author
        followers = await self.user_service.get_followers_list(author_id)

        # Filter followers who use pre-computed feeds (less active users)
        precompute_followers = [
            f for f in followers
            if f.activity_level in ["low", "medium"]
        ]

        # Calculate post score for feed ranking
        post_score = await self._calculate_post_score(post_data)

        # Batch insert into precomputed feeds
        feed_entries = [
            {
                "user_id": follower.id,
                "post_id": post_id,
                "score": post_score,
                "created_at": datetime.utcnow()
            }
            for follower in precompute_followers
        ]

        if feed_entries:
            await self._batch_insert_feed_entries(feed_entries)

            # Clean up old entries to prevent unlimited growth
            await self._cleanup_old_feed_entries(
                [f.id for f in precompute_followers]
            )

    async def _calculate_post_score(self, post_data: dict) -> float:
        """Calculate engagement-based score for feed ranking"""
        author_follower_count = post_data.get("author_follower_count", 0)
        post_age_hours = (
            datetime.utcnow() -
            datetime.fromisoformat(post_data["created_at"])
        ).total_seconds() / 3600

        # Base score factors
        author_score = min(math.log10(author_follower_count + 1), 5.0)
        recency_score = max(0, 10.0 - (post_age_hours / 24))  # Decay over days

        # Content type boost
        content_boost = {
            "image": 1.2,
            "video": 1.5,
            "text": 1.0
        }.get(post_data.get("content_type", "text"), 1.0)

        return (author_score + recency_score) * content_boost

    async def _batch_insert_feed_entries(self, entries: List[dict]):
        """Efficiently insert multiple feed entries"""
        if not entries:
            return

        # Use PostgreSQL's COPY for fast bulk insert
        query = """
        INSERT INTO precomputed_feeds (user_id, post_id, score, created_at)
        VALUES %s
        ON CONFLICT (user_id, post_id) DO NOTHING
        """

        await self.feed_db.primary.execute_values(query, entries)

    async def _cleanup_old_feed_entries(self, user_ids: List[str]):
        """Remove old entries to keep feeds manageable"""
        cleanup_query = """
        DELETE FROM precomputed_feeds
        WHERE user_id = ANY($1)
        AND id NOT IN (
            SELECT id FROM (
                SELECT id FROM precomputed_feeds
                WHERE user_id = ANY($1)
                ORDER BY score DESC, created_at DESC
                LIMIT 1000  -- Keep top 1000 entries per user
            ) recent_entries
        )
        """

        await self.feed_db.primary.execute(cleanup_query, user_ids)

# Database Schema for Feed System
CREATE TABLE precomputed_feeds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    score DECIMAL(10, 4) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, post_id)
);

-- Indexes for efficient querying
CREATE INDEX idx_precomputed_feeds_user_score ON precomputed_feeds(user_id, score DESC, created_at DESC);
CREATE INDEX idx_precomputed_feeds_cleanup ON precomputed_feeds(user_id, id);

-- Partitioning by user_id hash for better performance
CREATE TABLE precomputed_feeds_partition_0 PARTITION OF precomputed_feeds
FOR VALUES WITH (MODULUS 10, REMAINDER 0);

-- Additional partitions...
```

---

## Practice Exercise 2: Distributed System Architecture

### Objective

Design resilient distributed systems that handle failures gracefully and maintain consistency across multiple services.

### Exercise Details

**Time Required**: 3-4 weeks with comprehensive implementation
**Difficulty**: Advanced

### Week 1: Distributed System Fundamentals

#### CAP Theorem Implementation Exercise

**Scenario**: Design a distributed key-value store that demonstrates CAP theorem trade-offs

```python
# Distributed Key-Value Store with CAP Theorem Considerations

from enum import Enum
from typing import Dict, List, Optional, Set
import asyncio
import hashlib
import time
from dataclasses import dataclass

class ConsistencyLevel(Enum):
    ONE = "one"           # Any single replica
    QUORUM = "quorum"     # Majority of replicas
    ALL = "all"           # All replicas

class ReplicationStrategy(Enum):
    EVENTUAL_CONSISTENCY = "eventual"
    STRONG_CONSISTENCY = "strong"

@dataclass
class NodeInfo:
    node_id: str
    host: str
    port: int
    is_alive: bool = True
    last_heartbeat: float = 0

@dataclass
class DataRecord:
    key: str
    value: str
    version: int
    timestamp: float
    replicas: Set[str]

class DistributedKeyValueStore:
    """
    Implementation showing different consistency models
    and partition tolerance strategies
    """

    def __init__(
        self,
        node_id: str,
        nodes: List[NodeInfo],
        replication_factor: int = 3
    ):
        self.node_id = node_id
        self.nodes = {node.node_id: node for node in nodes}
        self.replication_factor = min(replication_factor, len(nodes))
        self.data_store: Dict[str, DataRecord] = {}
        self.vector_clock = {}
        self.pending_writes = {}  # For eventual consistency

    def _hash_key(self, key: str) -> int:
        """Consistent hashing for key distribution"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _get_replica_nodes(self, key: str) -> List[str]:
        """Determine which nodes should store replicas of this key"""
        hash_value = self._hash_key(key)
        sorted_nodes = sorted(self.nodes.keys())

        # Find starting position in ring
        start_idx = hash_value % len(sorted_nodes)

        # Select nodes in clockwise order
        replica_nodes = []
        for i in range(self.replication_factor):
            node_idx = (start_idx + i) % len(sorted_nodes)
            replica_nodes.append(sorted_nodes[node_idx])

        return replica_nodes

    async def put(
        self,
        key: str,
        value: str,
        consistency_level: ConsistencyLevel = ConsistencyLevel.QUORUM
    ) -> bool:
        """
        Write operation with configurable consistency
        """
        replica_nodes = self._get_replica_nodes(key)

        # Create new data record
        record = DataRecord(
            key=key,
            value=value,
            version=self._get_next_version(key),
            timestamp=time.time(),
            replicas=set(replica_nodes)
        )

        if consistency_level == ConsistencyLevel.STRONG:
            return await self._strong_consistency_put(record, replica_nodes)
        else:
            return await self._eventual_consistency_put(record, replica_nodes)

    async def _strong_consistency_put(
        self,
        record: DataRecord,
        replica_nodes: List[str]
    ) -> bool:
        """Strong consistency using two-phase commit"""

        # Phase 1: Prepare
        prepare_tasks = [
            self._send_prepare_request(node_id, record)
            for node_id in replica_nodes
            if self.nodes[node_id].is_alive
        ]

        prepare_responses = await asyncio.gather(
            *prepare_tasks,
            return_exceptions=True
        )

        # Check if majority agreed to prepare
        successful_prepares = [
            r for r in prepare_responses
            if not isinstance(r, Exception) and r.success
        ]

        required_nodes = len(replica_nodes) // 2 + 1  # Majority

        if len(successful_prepares) < required_nodes:
            # Abort transaction
            await self._send_abort_requests(replica_nodes)
            return False

        # Phase 2: Commit
        commit_tasks = [
            self._send_commit_request(node_id, record)
            for node_id in replica_nodes
            if self.nodes[node_id].is_alive
        ]

        commit_responses = await asyncio.gather(
            *commit_tasks,
            return_exceptions=True
        )

        # Commit locally if we're a replica
        if self.node_id in replica_nodes:
            self.data_store[record.key] = record

        return True

    async def _eventual_consistency_put(
        self,
        record: DataRecord,
        replica_nodes: List[str]
    ) -> bool:
        """Eventual consistency with async replication"""

        # Write locally first if we're a replica
        if self.node_id in replica_nodes:
            self.data_store[record.key] = record

        # Asynchronously replicate to other nodes
        replication_tasks = [
            self._async_replicate(node_id, record)
            for node_id in replica_nodes
            if node_id != self.node_id and self.nodes[node_id].is_alive
        ]

        # Don't wait for replication to complete
        asyncio.create_task(
            self._handle_async_replication(replication_tasks, record)
        )

        return True

    async def get(
        self,
        key: str,
        consistency_level: ConsistencyLevel = ConsistencyLevel.QUORUM
    ) -> Optional[DataRecord]:
        """Read operation with configurable consistency"""

        replica_nodes = self._get_replica_nodes(key)

        if consistency_level == ConsistencyLevel.ONE:
            return await self._read_one(key, replica_nodes)
        elif consistency_level == ConsistencyLevel.QUORUM:
            return await self._read_quorum(key, replica_nodes)
        else:  # ALL
            return await self._read_all(key, replica_nodes)

    async def _read_quorum(
        self,
        key: str,
        replica_nodes: List[str]
    ) -> Optional[DataRecord]:
        """Read from majority of replicas and resolve conflicts"""

        read_tasks = [
            self._read_from_node(node_id, key)
            for node_id in replica_nodes
            if self.nodes[node_id].is_alive
        ]

        # Wait for majority of reads to complete
        required_reads = len(replica_nodes) // 2 + 1

        results = []
        for coro in asyncio.as_completed(read_tasks):
            try:
                result = await coro
                if result:
                    results.append(result)
                if len(results) >= required_reads:
                    break
            except Exception:
                continue

        if not results:
            return None

        # Resolve conflicts using vector clocks/timestamps
        return self._resolve_read_conflicts(results)

    def _resolve_read_conflicts(
        self,
        results: List[DataRecord]
    ) -> DataRecord:
        """Resolve conflicts between different versions"""

        # Simple last-write-wins based on timestamp
        # In production, would use vector clocks for better conflict resolution
        return max(results, key=lambda r: (r.version, r.timestamp))

# Partition Tolerance Implementation
class PartitionDetector:
    """Detect and handle network partitions"""

    def __init__(self, nodes: List[NodeInfo]):
        self.nodes = {node.node_id: node for node in nodes}
        self.partition_groups: List[Set[str]] = []
        self.is_partitioned = False

    async def monitor_partitions(self):
        """Continuously monitor for network partitions"""
        while True:
            await self._check_node_connectivity()
            await self._detect_partitions()
            await asyncio.sleep(5)  # Check every 5 seconds

    async def _check_node_connectivity(self):
        """Check connectivity between all nodes"""
        connectivity_matrix = {}

        for node_id in self.nodes:
            connectivity_matrix[node_id] = {}
            for other_node_id in self.nodes:
                if node_id != other_node_id:
                    is_connected = await self._ping_node(node_id, other_node_id)
                    connectivity_matrix[node_id][other_node_id] = is_connected

        self.connectivity_matrix = connectivity_matrix

    async def _detect_partitions(self):
        """Detect network partitions using connectivity matrix"""
        # Use graph algorithms to find connected components
        visited = set()
        partition_groups = []

        for node_id in self.nodes:
            if node_id not in visited:
                # Find all nodes reachable from this node
                reachable = self._find_reachable_nodes(node_id, visited)
                partition_groups.append(reachable)

        # Update partition state
        old_partition_count = len(self.partition_groups)
        self.partition_groups = partition_groups
        self.is_partitioned = len(partition_groups) > 1

        if len(partition_groups) != old_partition_count:
            await self._handle_partition_change()

    def _find_reachable_nodes(
        self,
        start_node: str,
        visited: Set[str]
    ) -> Set[str]:
        """DFS to find all reachable nodes"""
        stack = [start_node]
        reachable = set()

        while stack:
            current = stack.pop()
            if current in visited:
                continue

            visited.add(current)
            reachable.add(current)

            # Add connected neighbors
            for neighbor in self.nodes:
                if (neighbor not in visited and
                    self.connectivity_matrix.get(current, {}).get(neighbor, False)):
                    stack.append(neighbor)

        return reachable

    async def _handle_partition_change(self):
        """Handle partition topology changes"""
        if self.is_partitioned:
            # Find majority partition
            majority_partition = max(self.partition_groups, key=len)
            minority_partitions = [
                p for p in self.partition_groups if p != majority_partition
            ]

            # Implement split-brain prevention
            if len(majority_partition) > len(self.nodes) // 2:
                # Majority partition continues operating
                await self._enable_writes_for_partition(majority_partition)
                await self._disable_writes_for_partitions(minority_partitions)
            else:
                # No majority - make system read-only
                await self._enable_readonly_mode()
        else:
            # Partition healed - restore normal operations
            await self._restore_normal_operations()

# Consistency Model Implementations
class EventuallyConsistentStore(DistributedKeyValueStore):
    """Optimized for Availability and Partition Tolerance (AP)"""

    async def put(self, key: str, value: str) -> bool:
        # Always succeed writes locally
        # Async replication happens in background
        return await super().put(key, value, ConsistencyLevel.ONE)

    async def get(self, key: str) -> Optional[DataRecord]:
        # Read from any available replica
        # May return stale data
        return await super().get(key, ConsistencyLevel.ONE)

class StronglyConsistentStore(DistributedKeyValueStore):
    """Optimized for Consistency and Partition Tolerance (CP)"""

    async def put(self, key: str, value: str) -> bool:
        # Requires majority consensus
        # May fail during partitions
        return await super().put(key, value, ConsistencyLevel.QUORUM)

    async def get(self, key: str) -> Optional[DataRecord]:
        # Read from majority
        # Ensures strong consistency
        return await super().get(key, ConsistencyLevel.QUORUM)

class HighAvailabilityStore(DistributedKeyValueStore):
    """Optimized for Consistency and Availability (CA)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition_detector = PartitionDetector(list(self.nodes.values()))

    async def put(self, key: str, value: str) -> bool:
        if self.partition_detector.is_partitioned:
            # Reject writes during partitions to maintain consistency
            raise PartitionException("Cannot write during network partition")

        # Require all replicas to acknowledge
        return await super().put(key, value, ConsistencyLevel.ALL)

    async def get(self, key: str) -> Optional[DataRecord]:
        if self.partition_detector.is_partitioned:
            # Allow reads from available replicas
            return await super().get(key, ConsistencyLevel.ONE)

        # Normal operation - read from any replica
        return await super().get(key, ConsistencyLevel.ONE)
```

### Week 2: Microservices Communication Patterns

#### Service Mesh Implementation

```python
# Service Mesh Pattern Implementation

import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import jwt
import logging

class ServiceMeshComponent:
    """Base class for service mesh components"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(f"servicemesh.{service_name}")

@dataclass
class ServiceInstance:
    id: str
    name: str
    host: str
    port: int
    health_check_url: str
    metadata: Dict[str, str]
    last_heartbeat: float = 0
    is_healthy: bool = True

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASH = "consistent_hash"

class ServiceRegistry(ServiceMeshComponent):
    """Service discovery and registration"""

    def __init__(self):
        super().__init__("service-registry")
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.health_check_interval = 30  # seconds

    async def register_service(self, instance: ServiceInstance) -> bool:
        """Register a service instance"""
        if instance.name not in self.services:
            self.services[instance.name] = []

        # Remove existing instance with same ID
        self.services[instance.name] = [
            s for s in self.services[instance.name]
            if s.id != instance.id
        ]

        # Add new instance
        instance.last_heartbeat = time.time()
        self.services[instance.name].append(instance)

        self.logger.info(f"Registered service instance: {instance.name}/{instance.id}")
        return True

    async def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance"""
        if service_name in self.services:
            self.services[service_name] = [
                s for s in self.services[service_name]
                if s.id != instance_id
            ]
            self.logger.info(f"Deregistered service instance: {service_name}/{instance_id}")
            return True
        return False

    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover healthy instances of a service"""
        if service_name not in self.services:
            return []

        # Return only healthy instances
        return [
            instance for instance in self.services[service_name]
            if instance.is_healthy
        ]

    async def start_health_monitoring(self):
        """Start background health checking"""
        while True:
            await self._health_check_all_services()
            await asyncio.sleep(self.health_check_interval)

    async def _health_check_all_services(self):
        """Check health of all registered services"""
        for service_name, instances in self.services.items():
            for instance in instances:
                try:
                    is_healthy = await self._check_instance_health(instance)
                    instance.is_healthy = is_healthy
                    if is_healthy:
                        instance.last_heartbeat = time.time()
                except Exception as e:
                    self.logger.error(f"Health check failed for {instance.id}: {e}")
                    instance.is_healthy = False

class LoadBalancer(ServiceMeshComponent):
    """Intelligent load balancing between service instances"""

    def __init__(self, service_registry: ServiceRegistry):
        super().__init__("load-balancer")
        self.service_registry = service_registry
        self.connection_counts: Dict[str, int] = {}
        self.round_robin_counters: Dict[str, int] = {}

    async def route_request(
        self,
        service_name: str,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    ) -> Optional[ServiceInstance]:
        """Route request to best available service instance"""

        instances = await self.service_registry.discover_services(service_name)
        if not instances:
            return None

        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(service_name, instances)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(instances)
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(service_name, instances)
        elif strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._consistent_hash_select(instances)

        return instances[0]  # Fallback

    def _round_robin_select(
        self,
        service_name: str,
        instances: List[ServiceInstance]
    ) -> ServiceInstance:
        """Simple round-robin selection"""
        if service_name not in self.round_robin_counters:
            self.round_robin_counters[service_name] = 0

        index = self.round_robin_counters[service_name] % len(instances)
        self.round_robin_counters[service_name] += 1

        return instances[index]

    def _least_connections_select(
        self,
        instances: List[ServiceInstance]
    ) -> ServiceInstance:
        """Select instance with least active connections"""
        return min(
            instances,
            key=lambda i: self.connection_counts.get(i.id, 0)
        )

    def connection_start(self, instance_id: str):
        """Track connection start"""
        self.connection_counts[instance_id] = (
            self.connection_counts.get(instance_id, 0) + 1
        )

    def connection_end(self, instance_id: str):
        """Track connection end"""
        if instance_id in self.connection_counts:
            self.connection_counts[instance_id] = max(
                0,
                self.connection_counts[instance_id] - 1
            )

class CircuitBreaker(ServiceMeshComponent):
    """Circuit breaker pattern for fault tolerance"""

    def __init__(self, service_name: str):
        super().__init__(f"circuit-breaker-{service_name}")
        self.failure_threshold = 5
        self.recovery_timeout = 60  # seconds
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""

        if self.state == "OPEN":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenException("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)

            # Success - reset failure count
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                self.logger.info("Circuit breaker CLOSED - service recovered")

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.error(f"Circuit breaker OPEN - too many failures: {self.failure_count}")

            raise e

class ServiceMesh:
    """Complete service mesh implementation"""

    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.load_balancer = LoadBalancer(self.service_registry)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.request_timeout = 30  # seconds

    async def start(self):
        """Start service mesh components"""
        await asyncio.gather(
            self.service_registry.start_health_monitoring(),
            self._start_metrics_collection()
        )

    async def register_service(self, instance: ServiceInstance) -> bool:
        """Register a service with the mesh"""
        return await self.service_registry.register_service(instance)

    async def make_request(
        self,
        service_name: str,
        request_func,
        *args,
        **kwargs
    ):
        """Make a request through the service mesh"""

        # Get circuit breaker for service
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(service_name)

        circuit_breaker = self.circuit_breakers[service_name]

        # Route to appropriate service instance
        instance = await self.load_balancer.route_request(service_name)
        if not instance:
            raise ServiceUnavailableException(f"No healthy instances of {service_name}")

        # Track connection
        self.load_balancer.connection_start(instance.id)

        try:
            # Execute request with circuit breaker protection
            result = await circuit_breaker.call(
                self._execute_request,
                instance,
                request_func,
                *args,
                **kwargs
            )
            return result

        finally:
            # Always clean up connection tracking
            self.load_balancer.connection_end(instance.id)

    async def _execute_request(
        self,
        instance: ServiceInstance,
        request_func,
        *args,
        **kwargs
    ):
        """Execute the actual request with timeout"""

        # Add instance endpoint information
        kwargs['host'] = instance.host
        kwargs['port'] = instance.port

        # Execute with timeout
        return await asyncio.wait_for(
            request_func(*args, **kwargs),
            timeout=self.request_timeout
        )
```

### Week 3-4: Advanced Patterns and Optimizations

#### Event Sourcing and CQRS Implementation

```python
# Event Sourcing and CQRS Pattern Implementation

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import asyncio
from dataclasses import dataclass, asdict
import uuid

# Event Store Implementation
@dataclass
class DomainEvent:
    event_id: str
    event_type: str
    aggregate_id: str
    data: Dict[str, Any]
    timestamp: datetime
    version: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class EventStore:
    """High-performance event store with snapshotting"""

    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.event_handlers: Dict[str, List] = {}
        self.snapshots: Dict[str, Any] = {}
        self.snapshot_frequency = 100  # Create snapshot every N events

    async def append_events(
        self,
        aggregate_id: str,
        events: List[DomainEvent],
        expected_version: int = None
    ) -> bool:
        """Append events to stream with optimistic concurrency control"""

        # Check expected version for concurrency control
        if expected_version is not None:
            current_version = await self._get_current_version(aggregate_id)
            if current_version != expected_version:
                raise OptimisticConcurrencyException(
                    f"Expected version {expected_version}, got {current_version}"
                )

        # Store events atomically
        try:
            await self.storage.begin_transaction()

            for event in events:
                await self.storage.insert_event(event)

                # Publish event to handlers
                await self._publish_event(event)

            await self.storage.commit_transaction()

            # Check if snapshot is needed
            await self._maybe_create_snapshot(aggregate_id)

            return True

        except Exception as e:
            await self.storage.rollback_transaction()
            raise EventStoreException(f"Failed to append events: {e}")

    async def get_events(
        self,
        aggregate_id: str,
        from_version: int = 0,
        to_version: int = None
    ) -> List[DomainEvent]:
        """Retrieve events for an aggregate"""

        return await self.storage.get_events(
            aggregate_id,
            from_version,
            to_version
        )

    async def get_aggregate(self, aggregate_id: str, aggregate_type: type):
        """Rebuild aggregate from events with snapshot optimization"""

        # Try to get latest snapshot
        snapshot = await self._get_latest_snapshot(aggregate_id)

        if snapshot:
            # Start from snapshot
            aggregate = aggregate_type.from_snapshot(snapshot['data'])
            from_version = snapshot['version'] + 1
        else:
            # Start from beginning
            aggregate = aggregate_type()
            from_version = 0

        # Apply events since snapshot
        events = await self.get_events(aggregate_id, from_version)

        for event in events:
            aggregate.apply_event(event)

        return aggregate

    async def _maybe_create_snapshot(self, aggregate_id: str):
        """Create snapshot if enough events have accumulated"""

        event_count = await self._get_event_count(aggregate_id)
        last_snapshot_version = await self._get_last_snapshot_version(aggregate_id)

        if event_count - last_snapshot_version >= self.snapshot_frequency:
            # Rebuild aggregate and create snapshot
            aggregate = await self.get_aggregate(aggregate_id, type(aggregate))

            snapshot_data = {
                'aggregate_id': aggregate_id,
                'version': event_count,
                'data': aggregate.to_snapshot(),
                'timestamp': datetime.utcnow()
            }

            await self.storage.store_snapshot(snapshot_data)

# Aggregate Root Base Class
class AggregateRoot:
    """Base class for domain aggregates"""

    def __init__(self):
        self.id: str = str(uuid.uuid4())
        self.version: int = 0
        self.uncommitted_events: List[DomainEvent] = []

    def apply_event(self, event: DomainEvent):
        """Apply event to aggregate state"""
        # Call specific handler method
        handler_name = f"_handle_{event.event_type}"
        handler = getattr(self, handler_name, None)

        if handler:
            handler(event)

        self.version = event.version

    def raise_event(self, event_type: str, data: Dict[str, Any]):
        """Raise a domain event"""
        event = DomainEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            aggregate_id=self.id,
            data=data,
            timestamp=datetime.utcnow(),
            version=self.version + 1
        )

        # Apply event to self
        self.apply_event(event)

        # Track for persistence
        self.uncommitted_events.append(event)

    def mark_events_as_committed(self):
        """Clear uncommitted events after persistence"""
        self.uncommitted_events.clear()

    def to_snapshot(self) -> Dict[str, Any]:
        """Create snapshot data"""
        return asdict(self)

    @classmethod
    def from_snapshot(cls, snapshot_data: Dict[str, Any]):
        """Restore from snapshot"""
        instance = cls()
        for key, value in snapshot_data.items():
            setattr(instance, key, value)
        return instance

# Example: User Aggregate
class User(AggregateRoot):
    def __init__(self):
        super().__init__()
        self.username: str = ""
        self.email: str = ""
        self.followers: List[str] = []
        self.following: List[str] = []
        self.is_active: bool = True

    def create_user(self, username: str, email: str):
        """Create a new user"""
        if self.username:  # Already created
            raise DomainException("User already exists")

        self.raise_event("user_created", {
            "username": username,
            "email": email
        })

    def follow_user(self, target_user_id: str):
        """Follow another user"""
        if target_user_id in self.following:
            return  # Already following

        self.raise_event("user_followed", {
            "target_user_id": target_user_id
        })

    def unfollow_user(self, target_user_id: str):
        """Unfollow another user"""
        if target_user_id not in self.following:
            return  # Not following

        self.raise_event("user_unfollowed", {
            "target_user_id": target_user_id
        })

    # Event handlers
    def _handle_user_created(self, event: DomainEvent):
        self.username = event.data["username"]
        self.email = event.data["email"]

    def _handle_user_followed(self, event: DomainEvent):
        target_user_id = event.data["target_user_id"]
        if target_user_id not in self.following:
            self.following.append(target_user_id)

    def _handle_user_unfollowed(self, event: DomainEvent):
        target_user_id = event.data["target_user_id"]
        if target_user_id in self.following:
            self.following.remove(target_user_id)

# Command Handler (Write Side)
class UserCommandHandler:
    def __init__(self, event_store: EventStore):
        self.event_store = event_store

    async def handle_create_user(self, command):
        # Create new user aggregate
        user = User()
        user.create_user(command.username, command.email)

        # Persist events
        await self.event_store.append_events(
            user.id,
            user.uncommitted_events
        )

        user.mark_events_as_committed()
        return user.id

    async def handle_follow_user(self, command):
        # Load existing user
        user = await self.event_store.get_aggregate(command.user_id, User)

        # Execute command
        user.follow_user(command.target_user_id)

        # Persist events
        await self.event_store.append_events(
            user.id,
            user.uncommitted_events,
            expected_version=user.version - len(user.uncommitted_events)
        )

        user.mark_events_as_committed()

# Read Model (Query Side)
class UserReadModel:
    def __init__(self):
        self.users: Dict[str, Dict] = {}
        self.followers_index: Dict[str, List[str]] = {}
        self.following_index: Dict[str, List[str]] = {}

    async def handle_user_created(self, event: DomainEvent):
        """Update read model when user is created"""
        user_data = {
            'id': event.aggregate_id,
            'username': event.data['username'],
            'email': event.data['email'],
            'follower_count': 0,
            'following_count': 0,
            'created_at': event.timestamp
        }

        self.users[event.aggregate_id] = user_data

    async def handle_user_followed(self, event: DomainEvent):
        """Update indexes when user follows someone"""
        follower_id = event.aggregate_id
        target_id = event.data['target_user_id']

        # Update following index
        if follower_id not in self.following_index:
            self.following_index[follower_id] = []
        self.following_index[follower_id].append(target_id)

        # Update followers index
        if target_id not in self.followers_index:
            self.followers_index[target_id] = []
        self.followers_index[target_id].append(follower_id)

        # Update counts
        if follower_id in self.users:
            self.users[follower_id]['following_count'] += 1
        if target_id in self.users:
            self.users[target_id]['follower_count'] += 1

# Query Handler (Read Side)
class UserQueryHandler:
    def __init__(self, read_model: UserReadModel):
        self.read_model = read_model

    async def get_user_profile(self, user_id: str):
        """Get user profile from read model"""
        return self.read_model.users.get(user_id)

    async def get_user_followers(self, user_id: str):
        """Get list of user followers"""
        follower_ids = self.read_model.followers_index.get(user_id, [])
        return [
            self.read_model.users[fid]
            for fid in follower_ids
            if fid in self.read_model.users
        ]

    async def get_user_following(self, user_id: str):
        """Get list of users being followed"""
        following_ids = self.read_model.following_index.get(user_id, [])
        return [
            self.read_model.users[fid]
            for fid in following_ids
            if fid in self.read_model.users
        ]

# Event Handler Registration and Processing
class EventProcessor:
    """Process events and update read models"""

    def __init__(self):
        self.handlers: Dict[str, List] = {}

    def register_handler(self, event_type: str, handler):
        """Register event handler"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    async def process_event(self, event: DomainEvent):
        """Process event through all registered handlers"""
        handlers = self.handlers.get(event.event_type, [])

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                # Log error but don't stop other handlers
                print(f"Error processing event {event.event_id}: {e}")

# Usage Example
async def cqrs_example():
    # Setup
    event_store = EventStore(PostgreSQLEventStorage())
    user_read_model = UserReadModel()
    command_handler = UserCommandHandler(event_store)
    query_handler = UserQueryHandler(user_read_model)

    # Setup event processing
    event_processor = EventProcessor()
    event_processor.register_handler("user_created", user_read_model.handle_user_created)
    event_processor.register_handler("user_followed", user_read_model.handle_user_followed)

    # Execute commands
    user_id = await command_handler.handle_create_user(
        CreateUserCommand(username="john_doe", email="john@example.com")
    )

    # Query data
    user_profile = await query_handler.get_user_profile(user_id)
    print(f"Created user: {user_profile}")
```

---

## Additional Practice Exercises

### Exercise 3: Database Scaling and Sharding

**Focus**: Horizontal partitioning, database clustering, read/write splitting
**Duration**: 2-3 weeks
**Skills**: Sharding strategies, replication, consistency management

### Exercise 4: Caching Strategy Implementation

**Focus**: Multi-level caching, cache invalidation, distributed caching
**Duration**: 2-3 weeks  
**Skills**: Cache patterns, Redis clustering, CDN integration

### Exercise 5: Load Balancing and Traffic Management

**Focus**: Load balancing algorithms, traffic shaping, geographic distribution
**Duration**: 2-3 weeks
**Skills**: Load balancer configuration, traffic analysis, failover strategies

### Exercise 6: Microservices Communication Patterns

**Focus**: Service-to-service communication, API gateways, service discovery
**Duration**: 3-4 weeks
**Skills**: Inter-service protocols, circuit breakers, timeout handling

### Exercise 7: Event-Driven Architecture Design

**Focus**: Event sourcing, message queues, eventual consistency
**Duration**: 3-4 weeks
**Skills**: Event design, message brokers, saga patterns

### Exercise 8: High Availability and Disaster Recovery

**Focus**: Failover strategies, backup systems, geographic redundancy
**Duration**: 2-3 weeks
**Skills**: RTO/RPO planning, automated failover, data replication

### Exercise 9: Performance Optimization and Monitoring

**Focus**: Performance profiling, bottleneck identification, observability
**Duration**: 2-3 weeks
**Skills**: APM tools, performance testing, optimization techniques

### Exercise 10: Security and Compliance Architecture

**Focus**: Security patterns, compliance requirements, threat modeling
**Duration**: 2-3 weeks
**Skills**: Security design, audit trails, encryption strategies

---

## Monthly System Design Assessment

### System Design Skills Self-Evaluation

Rate your proficiency (1-10) in each area:

**Architecture Design**:

- [ ] Scalable system architecture planning
- [ ] Component interaction design
- [ ] Technology stack selection and justification
- [ ] Performance and scalability considerations

**Distributed Systems**:

- [ ] CAP theorem understanding and application
- [ ] Consensus algorithms and distributed coordination
- [ ] Microservices design and communication patterns
- [ ] Event-driven architecture implementation

**Data Management**:

- [ ] Database scaling and partitioning strategies
- [ ] Data consistency and replication patterns
- [ ] Caching strategies and implementation
- [ ] Data pipeline and ETL design

**Reliability and Performance**:

- [ ] High availability and disaster recovery planning
- [ ] Performance optimization and monitoring
- [ ] Circuit breaker and retry pattern implementation
- [ ] Load balancing and traffic management

### Growth Planning Framework

1. **Architecture Philosophy**: What principles guide your system design decisions?
2. **Scalability Understanding**: How well do you design for scale from the beginning?
3. **Trade-off Analysis**: Can you effectively evaluate and communicate design trade-offs?
4. **Real-world Application**: How do you apply theoretical concepts to practical problems?
5. **Technology Selection**: How do you choose appropriate technologies for different scenarios?
6. **Monitoring and Operations**: Do you design systems that are observable and maintainable?

### Continuous Learning Recommendations

- Study system designs of major tech companies (high-scale architectures)
- Practice system design interviews with increasingly complex scenarios
- Build and operate distributed systems in production environments
- Contribute to open-source distributed systems projects
- Stay current with cloud platform features and distributed system tools
- Learn from post-mortems and incident reports from major tech companies

Remember: System design is about making informed trade-offs based on requirements, constraints, and operational realities. Focus on understanding the principles behind different architectural patterns and when to apply them.
