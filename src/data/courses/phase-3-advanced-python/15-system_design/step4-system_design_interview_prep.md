# System Design - Interview Preparation Guide

## 🎯 System Design Interview Overview

### What is a System Design Interview?

```
Purpose:
- Evaluate architectural thinking and problem-solving
- Assess understanding of large-scale system concepts
- Test communication and collaboration skills
- Gauge experience with real-world engineering challenges

Format (45-60 minutes):
1. Problem Statement (5 minutes)
2. Requirements Gathering (10 minutes)
3. High-Level Design (15 minutes)
4. Deep Dive (15 minutes)
5. Scale and Optimize (10 minutes)
6. Wrap-up and Questions (5 minutes)

Evaluation Criteria:
✓ Problem understanding and clarification
✓ High-level architecture design
✓ Technical component knowledge
✓ Scalability considerations
✓ Trade-offs and alternatives discussion
✓ Communication and collaboration
```

### Preparation Strategy

```
Knowledge Areas to Master:
1. System Architecture Fundamentals
2. Database Design and Scaling
3. Caching Strategies
4. Load Balancing and CDNs
5. Microservices and Service Design
6. Message Queues and Event-Driven Architecture
7. Monitoring and Reliability
8. Security and Authentication

Practice Schedule (4-6 weeks):
Week 1-2: Fundamentals review and concept mapping
Week 3-4: Practice with common system design problems
Week 5-6: Mock interviews and advanced topics

Daily Practice:
- 1 system design problem (45-60 minutes)
- Review and improve previous solutions
- Read engineering blogs from major tech companies
- Watch system design videos and tutorials
```

## 📋 System Design Framework

### The REDSCO Method

```
Requirements (10 minutes):
R - Functional Requirements
E - Estimations and Scale
D - Design Goals
S - Success Criteria
C - Constraints
O - Out of Scope

Step-by-Step Process:

1. Requirements Gathering:
"Before we start designing, let me understand the requirements:

Functional Requirements:
- What are the core features users need?
- Who are the users and how will they interact with the system?
- What are the key user journeys?

Non-Functional Requirements:
- What scale are we expecting? (users, requests, data)
- What are the performance requirements? (latency, throughput)
- What are the reliability requirements? (availability, consistency)
- Are there any specific constraints? (budget, timeline, technology)

Example Questions:
- How many users do we expect?
- How many posts/messages/requests per day?
- What's the read-to-write ratio?
- Do we need real-time features?
- What's the expected data retention period?
- Are there geographic requirements?
- What devices will users use?"

2. Capacity Estimation:
"Let me estimate the scale we're dealing with:

User Scale:
- Daily Active Users (DAU): 10 million
- Monthly Active Users (MAU): 50 million
- Peak concurrent users: 1 million

Request Scale:
- Reads per second: 100,000 QPS
- Writes per second: 10,000 QPS
- Peak load: 5x average (500K reads, 50K writes)

Data Scale:
- Storage per user: 1 KB metadata + 10 MB content
- Total storage: 500 TB
- Daily data growth: 100 GB
- Cache requirements: 10% of total data (50 TB)

Network:
- Bandwidth requirements: 10 Gbps average, 50 Gbps peak
- CDN traffic: 80% of total requests
- Database connections: 10,000 concurrent"

3. High-Level Design:
"Based on the requirements, here's my high-level approach:

[Draw and explain architecture diagram]

Main Components:
- Load Balancer: Route traffic to application servers
- Web Servers: Handle HTTP requests, business logic
- Application Servers: Core application logic
- Databases: Data persistence and retrieval
- Cache Layer: Fast data access
- CDN: Static content delivery
- Message Queues: Asynchronous processing

Data Flow:
1. User makes request → Load Balancer
2. Load Balancer → Web Server
3. Web Server → Application Server
4. Application Server → Cache/Database
5. Response back through the chain"
```

### Common Architecture Patterns

```
Layered Architecture:
┌─────────────────────┐
│   Presentation      │ ← Web/Mobile clients
├─────────────────────┤
│   Load Balancer     │ ← Traffic distribution
├─────────────────────┤
│   Web Tier          │ ← Web servers (nginx/Apache)
├─────────────────────┤
│   Application Tier  │ ← Business logic (APIs)
├─────────────────────┤
│   Cache Tier        │ ← Redis/Memcached
├─────────────────────┤
│   Database Tier     │ ← Primary storage
└─────────────────────┘

Microservices Architecture:
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ User Service │    │ Post Service │    │ Media Service│
├──────────────┤    ├──────────────┤    ├──────────────┤
│   User DB    │    │   Post DB    │    │   Media DB   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                            │
                   ┌──────────────┐
                   │ API Gateway  │
                   └──────────────┘

Event-Driven Architecture:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Producer   │───▶│ Message Bus │───▶│  Consumer   │
│  Service    │    │ (Kafka/RMQ) │    │  Service    │
└─────────────┘    └─────────────┘    └─────────────┘

Lambda/Serverless Architecture:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ API Gateway │───▶│   Lambda    │───▶│  DynamoDB   │
└─────────────┘    │ Functions   │    │    RDS      │
                   └─────────────┘    └─────────────┘
```

## 🏗️ Core System Design Components

### Database Design and Scaling

````
SQL vs NoSQL Decision Matrix:

Use SQL (PostgreSQL, MySQL) when:
✅ ACID compliance required
✅ Complex relationships and joins
✅ Strong consistency needed
✅ Well-defined schema
✅ Transactional requirements

Use NoSQL when:
✅ Massive scale requirements (MongoDB, Cassandra)
✅ Flexible/evolving schema (Document stores)
✅ Key-value access patterns (DynamoDB, Redis)
✅ Geographic distribution (Cassandra)
✅ Real-time analytics (InfluxDB, TimeScale)

Database Scaling Strategies:

1. Vertical Scaling (Scale Up):
- Increase CPU, RAM, storage
- Simple but limited
- Single point of failure

2. Read Replicas:
Master (Write) ───┐
                  ├─▶ Slave 1 (Read)
                  ├─▶ Slave 2 (Read)
                  └─▶ Slave 3 (Read)

3. Horizontal Sharding:
Users 1-1000    ─▶ Shard 1
Users 1001-2000 ─▶ Shard 2
Users 2001-3000 ─▶ Shard 3

Sharding Strategies:
- Range-based: partition by ID ranges
- Hash-based: hash(user_id) % num_shards
- Directory-based: lookup table for shard mapping
- Geographic: partition by location

Example Database Schema:
```sql
-- Users table (sharded by user_id)
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Posts table (sharded by user_id for co-location)
CREATE TABLE posts (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    media_urls TEXT[],
    visibility VARCHAR(20) DEFAULT 'public',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Relationships table (denormalized for performance)
CREATE TABLE user_relationships (
    follower_id BIGINT,
    following_id BIGINT,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (follower_id, following_id)
);

-- Indexes for performance
CREATE INDEX idx_posts_user_id_created ON posts(user_id, created_at DESC);
CREATE INDEX idx_relationships_follower ON user_relationships(follower_id);
CREATE INDEX idx_relationships_following ON user_relationships(following_id);
````

NoSQL Example (Document Store):

```javascript
// User document (MongoDB)
{
  "_id": ObjectId("..."),
  "username": "john_doe",
  "email": "john@example.com",
  "profile": {
    "displayName": "John Doe",
    "bio": "Software engineer",
    "location": "San Francisco",
    "avatarUrl": "https://cdn.example.com/avatars/john.jpg"
  },
  "followers": [
    {"userId": ObjectId("..."), "since": ISODate("...")},
    // First 100 followers embedded for quick access
  ],
  "followersCount": 1500,
  "followingCount": 300,
  "postsCount": 50,
  "createdAt": ISODate("..."),
  "updatedAt": ISODate("...")
}

// Post document (partitioned by month)
{
  "_id": ObjectId("..."),
  "userId": ObjectId("..."),
  "content": "Hello world!",
  "mediaUrls": ["https://cdn.example.com/images/abc.jpg"],
  "tags": ["hello", "world"],
  "location": {
    "type": "Point",
    "coordinates": [-122.4194, 37.7749]
  },
  "engagement": {
    "likes": 25,
    "comments": 5,
    "shares": 2
  },
  "visibility": "public",
  "createdAt": ISODate("..."),
  "updatedAt": ISODate("...")
}
```

### Caching Strategies

````
Caching Layers:

1. Browser Cache (Client-side):
- Static assets (CSS, JS, images)
- API responses with proper headers
- Service worker caching

2. CDN (Edge Cache):
- Global content distribution
- Static content delivery
- Dynamic content caching with TTL

3. Application Cache:
- Redis/Memcached for session data
- Application-level object caching
- Query result caching

4. Database Cache:
- Query plan caching
- Buffer pool optimization
- Database-specific caching

Cache Patterns:

Cache-Aside (Lazy Loading):
```python
def get_user(user_id):
    # Check cache first
    user = redis.get(f"user:{user_id}")
    if user:
        return json.loads(user)

    # Cache miss - fetch from database
    user = db.get_user(user_id)
    if user:
        # Store in cache with TTL
        redis.setex(f"user:{user_id}", 3600, json.dumps(user))

    return user

def update_user(user_id, data):
    # Update database
    user = db.update_user(user_id, data)

    # Invalidate cache
    redis.delete(f"user:{user_id}")

    return user
````

Write-Through Cache:

```python
def update_user_write_through(user_id, data):
    # Update database
    user = db.update_user(user_id, data)

    # Update cache immediately
    redis.setex(f"user:{user_id}", 3600, json.dumps(user))

    return user
```

Write-Behind (Write-Back):

```python
def update_user_write_behind(user_id, data):
    # Update cache immediately
    redis.setex(f"user:{user_id}", 3600, json.dumps(data))

    # Queue database update for later
    task_queue.put({
        'type': 'update_user',
        'user_id': user_id,
        'data': data,
        'timestamp': time.time()
    })

    return data
```

Cache Invalidation Strategies:

- TTL (Time To Live): Automatic expiration
- Event-based: Invalidate on data changes
- Manual: Explicit cache clearing
- Version-based: Cache versioning with timestamps

Redis Configuration Example:

```redis
# Memory optimization
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence for cache durability
save 900 1    # Save if at least 1 key changed in 900 seconds
save 300 10   # Save if at least 10 keys changed in 300 seconds
save 60 10000 # Save if at least 10000 keys changed in 60 seconds

# Cluster configuration for high availability
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 15000
```

### Load Balancing and Traffic Management

````
Load Balancer Types:

Layer 4 (Transport Layer):
- Routes based on IP and port
- Faster, lower overhead
- Protocol agnostic
- No content inspection

Layer 7 (Application Layer):
- Routes based on HTTP content
- URL, headers, cookies-based routing
- SSL termination
- Content-based decisions

Load Balancing Algorithms:

1. Round Robin:
Server 1 ← Request 1
Server 2 ← Request 2
Server 3 ← Request 3
Server 1 ← Request 4 (cycle repeats)

2. Weighted Round Robin:
Server 1 (weight=3) ← Requests 1,2,3
Server 2 (weight=1) ← Request 4
Server 3 (weight=2) ← Requests 5,6
(Pattern repeats)

3. Least Connections:
Route to server with fewest active connections

4. IP Hash:
hash(client_ip) % server_count
Ensures session affinity

5. Geolocation:
Route based on client geographic location

NGINX Configuration Example:
```nginx
upstream backend {
    least_conn;  # Load balancing algorithm
    server web1.example.com:3000 weight=3;
    server web2.example.com:3000 weight=1;
    server web3.example.com:3000 backup;

    # Health checks
    health_check interval=30s;
}

server {
    listen 80;
    server_name example.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name example.com;

    # SSL configuration
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    # Static content
    location /static/ {
        alias /var/www/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # API endpoints
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Rate limiting
        limit_req zone=api burst=20 nodelay;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}
````

CDN Integration:

```javascript
// CDN configuration for static assets
const CDN_CONFIG = {
  images: "https://images.cdn.example.com",
  videos: "https://videos.cdn.example.com",
  static: "https://static.cdn.example.com",
};

// Dynamic content caching headers
app.get("/api/posts/trending", (req, res) => {
  // Cache for 5 minutes
  res.set({
    "Cache-Control": "public, max-age=300",
    ETag: generateETag(trendingPosts),
    Vary: "Accept-Encoding",
  });

  res.json(trendingPosts);
});

// User-specific content (no caching)
app.get("/api/user/feed", authenticateUser, (req, res) => {
  res.set({
    "Cache-Control": "private, no-cache, no-store, must-revalidate",
    Pragma: "no-cache",
    Expires: "0",
  });

  res.json(userFeed);
});
```

### Message Queues and Event-Driven Architecture

````
Message Queue Patterns:

1. Point-to-Point (Queue):
Producer → Queue → Consumer
- One message consumed by one consumer
- Message deleted after consumption

2. Publish-Subscribe (Topic):
Producer → Topic → Consumer 1
              ↓     Consumer 2
              ↓     Consumer 3
- One message consumed by multiple consumers
- Message persists until all subscribers process it

3. Request-Reply:
Client → Request Queue → Service
Client ← Reply Queue   ← Service

Popular Message Queue Systems:

Apache Kafka (High Throughput):
```javascript
// Kafka producer configuration
const kafka = require('kafkajs');

const client = kafka({
  clientId: 'user-service',
  brokers: ['kafka1:9092', 'kafka2:9092', 'kafka3:9092']
});

const producer = client.producer({
  maxInFlightRequests: 1,
  idempotent: true,
  transactionTimeout: 30000
});

// Produce message
await producer.send({
  topic: 'user-events',
  messages: [{
    partition: hashUserIdToPartition(userId),
    key: userId.toString(),
    value: JSON.stringify({
      eventType: 'user.created',
      userId: userId,
      timestamp: Date.now(),
      data: userData
    })
  }]
});

// Kafka consumer
const consumer = client.consumer({
  groupId: 'notification-service',
  sessionTimeout: 30000,
  heartbeatInterval: 3000
});

await consumer.subscribe({ topic: 'user-events' });

await consumer.run({
  eachMessage: async ({ topic, partition, message }) => {
    const event = JSON.parse(message.value.toString());

    try {
      await processUserEvent(event);
      // Message automatically committed
    } catch (error) {
      console.error('Processing failed:', error);
      // Implement dead letter queue logic
    }
  },
});
````

RabbitMQ (Flexible Routing):

```javascript
const amqp = require("amqplib");

// Publisher
class EventPublisher {
  async connect() {
    this.connection = await amqp.connect("amqp://localhost");
    this.channel = await this.connection.createChannel();

    // Declare exchange
    await this.channel.assertExchange("user_events", "topic", {
      durable: true,
    });
  }

  async publishEvent(eventType, data) {
    const message = Buffer.from(
      JSON.stringify({
        eventType,
        data,
        timestamp: new Date().toISOString(),
      }),
    );

    this.channel.publish("user_events", eventType, message, {
      persistent: true,
    });
  }
}

// Consumer
class EventConsumer {
  async connect(serviceName) {
    this.connection = await amqp.connect("amqp://localhost");
    this.channel = await this.connection.createChannel();

    // Declare queue
    const queue = await this.channel.assertQueue(`${serviceName}_queue`, {
      durable: true,
    });

    // Bind to events we care about
    await this.channel.bindQueue(queue.queue, "user_events", "user.created");
    await this.channel.bindQueue(queue.queue, "user_events", "user.updated");

    // Set prefetch to 1 for fair distribution
    this.channel.prefetch(1);

    // Start consuming
    this.channel.consume(queue.queue, async (message) => {
      if (message) {
        try {
          const event = JSON.parse(message.content.toString());
          await this.processEvent(event);
          this.channel.ack(message);
        } catch (error) {
          console.error("Processing failed:", error);
          this.channel.nack(message, false, false); // Send to DLQ
        }
      }
    });
  }

  async processEvent(event) {
    // Implement event processing logic
    switch (event.eventType) {
      case "user.created":
        await this.handleUserCreated(event.data);
        break;
      case "user.updated":
        await this.handleUserUpdated(event.data);
        break;
    }
  }
}
```

Event Sourcing Pattern:

```javascript
// Event store
class EventStore {
  async appendEvent(streamId, event) {
    await db.events.insert({
      streamId,
      eventType: event.type,
      eventData: event.data,
      eventVersion: await this.getNextVersion(streamId),
      timestamp: new Date(),
    });
  }

  async getEvents(streamId, fromVersion = 0) {
    return await db.events
      .find({
        streamId,
        eventVersion: { $gte: fromVersion },
      })
      .sort({ eventVersion: 1 });
  }
}

// Aggregate reconstruction
class UserAggregate {
  constructor(userId) {
    this.userId = userId;
    this.version = 0;
    this.state = {};
  }

  static async loadFromEvents(userId) {
    const aggregate = new UserAggregate(userId);
    const events = await eventStore.getEvents(userId);

    for (const event of events) {
      aggregate.apply(event);
    }

    return aggregate;
  }

  apply(event) {
    switch (event.eventType) {
      case "UserCreated":
        this.state = { ...event.eventData };
        break;
      case "UserEmailChanged":
        this.state.email = event.eventData.email;
        break;
      case "UserDeactivated":
        this.state.active = false;
        break;
    }
    this.version = event.eventVersion;
  }
}
```

## 🎯 Common System Design Problems

### Problem 1: Design a URL Shortener (like bit.ly)

````
Step 1: Requirements Gathering

Functional Requirements:
- Shorten long URLs to short URLs
- Redirect short URLs to original URLs
- Custom short URLs (optional)
- URL expiration (optional)
- Analytics on URL usage

Non-Functional Requirements:
- 100M URLs shortened per day
- 100:1 read-to-write ratio
- Low latency for redirects (<100ms)
- High availability (99.9%)
- URL should be as short as possible

Step 2: Capacity Estimation

Scale:
- Write: 100M / (24 * 3600) ≈ 1160 URLs/second
- Read: 1160 * 100 = 116,000 redirects/second
- Storage: 100M URLs * 365 days * 5 years = 182.5B URLs
- Storage per URL: 500 bytes average
- Total storage: 182.5B * 500B ≈ 91TB over 5 years

Step 3: High-Level Design

Components:
1. URL Shortening Service
2. URL Redirect Service
3. Database for URL mappings
4. Cache for popular URLs
5. Analytics Service
6. Rate Limiter

Step 4: Detailed Design

Database Schema:
```sql
CREATE TABLE urls (
    id BIGSERIAL PRIMARY KEY,
    short_url VARCHAR(7) UNIQUE NOT NULL,
    long_url TEXT NOT NULL,
    user_id BIGINT,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    click_count BIGINT DEFAULT 0
);

CREATE INDEX idx_short_url ON urls(short_url);
CREATE INDEX idx_user_id ON urls(user_id);
CREATE INDEX idx_expires_at ON urls(expires_at);
````

URL Encoding Algorithm:

```python
import hashlib
import base64

class URLShortener:
    def __init__(self):
        self.counter = 0
        self.base62_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def encode_base62(self, num):
        if num == 0:
            return self.base62_chars[0]

        result = ""
        while num > 0:
            result = self.base62_chars[num % 62] + result
            num //= 62
        return result

    def generate_short_url(self, long_url, user_id=None):
        # Method 1: Auto-incrementing ID (predictable but efficient)
        url_id = self.get_next_id()
        short_code = self.encode_base62(url_id)

        # Method 2: Hash-based (unpredictable)
        # hash_input = f"{long_url}{user_id}{timestamp}"
        # hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        # short_code = hash_value[:7]

        return f"https://short.ly/{short_code}"

    def get_next_id(self):
        # This would be a distributed counter in practice
        # Could use database sequence, Redis counter, or Snowflake-style ID
        self.counter += 1
        return self.counter
```

API Design:

```python
# Shorten URL endpoint
@app.post("/api/shorten")
async def shorten_url(request: ShortenRequest):
    # Rate limiting check
    if not rate_limiter.is_allowed(request.client_ip):
        raise HTTPException(429, "Rate limit exceeded")

    # Validate URL
    if not is_valid_url(request.long_url):
        raise HTTPException(400, "Invalid URL")

    # Check if URL already shortened
    existing = await db.find_url_by_long_url(request.long_url, request.user_id)
    if existing:
        return {"short_url": existing.short_url}

    # Generate short URL
    short_code = url_shortener.generate_short_url(request.long_url, request.user_id)

    # Store in database
    url_record = await db.create_url_mapping({
        "short_url": short_code,
        "long_url": request.long_url,
        "user_id": request.user_id,
        "expires_at": request.expires_at
    })

    return {"short_url": short_code}

# Redirect endpoint
@app.get("/{short_code}")
async def redirect_url(short_code: str):
    # Check cache first
    long_url = await cache.get(f"url:{short_code}")

    if not long_url:
        # Cache miss - query database
        url_record = await db.find_url_by_short_code(short_code)

        if not url_record:
            raise HTTPException(404, "URL not found")

        if url_record.expires_at and url_record.expires_at < datetime.now():
            raise HTTPException(410, "URL expired")

        long_url = url_record.long_url

        # Cache for future requests
        await cache.set(f"url:{short_code}", long_url, ttl=3600)

    # Async analytics update (don't block redirect)
    asyncio.create_task(analytics.record_click(short_code, request.client_ip))

    return RedirectResponse(url=long_url, status_code=301)
```

Scaling Considerations:

1. Database Sharding:

```python
def get_shard(short_code):
    # Hash-based sharding
    hash_value = hashlib.md5(short_code.encode()).hexdigest()
    shard_id = int(hash_value[:4], 16) % NUM_SHARDS
    return f"shard_{shard_id}"

def get_database_connection(short_code):
    shard = get_shard(short_code)
    return database_connections[shard]
```

2. Caching Strategy:

```python
# Multi-level caching
# L1: Application cache (local)
# L2: Redis cluster (distributed)
# L3: Database

async def get_long_url(short_code):
    # L1 Cache
    if short_code in local_cache:
        return local_cache[short_code]

    # L2 Cache
    long_url = await redis.get(f"url:{short_code}")
    if long_url:
        local_cache[short_code] = long_url
        return long_url

    # L3 Database
    url_record = await db.find_url_by_short_code(short_code)
    if url_record:
        long_url = url_record.long_url
        # Cache at both levels
        local_cache[short_code] = long_url
        await redis.set(f"url:{short_code}", long_url, ttl=3600)
        return long_url

    return None
```

3. Analytics with Message Queue:

```python
# Async analytics processing
class AnalyticsService:
    async def record_click(self, short_code, client_ip, user_agent):
        # Send to message queue for processing
        analytics_event = {
            "event_type": "url_click",
            "short_code": short_code,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "timestamp": datetime.utcnow().isoformat(),
            "geo_location": await geo_service.get_location(client_ip)
        }

        await message_queue.publish("analytics_events", analytics_event)

    async def process_analytics_events(self):
        # Background worker to process analytics
        async for event in message_queue.consume("analytics_events"):
            await self.store_analytics(event)
            await self.update_click_count(event["short_code"])
```

Step 5: Additional Considerations

Security:

- Rate limiting to prevent abuse
- Input validation and sanitization
- HTTPS enforcement
- Bot detection

Monitoring:

- Response time metrics
- Error rates and types
- Cache hit ratios
- Database performance
- Queue depth monitoring

Disaster Recovery:

- Database replication across regions
- Regular backups
- Failover procedures
- Data consistency checks

```

### Problem 2: Design a Chat System (like WhatsApp)
```

Step 1: Requirements Gathering

Functional Requirements:

- Send and receive messages in real-time
- One-on-one and group messaging
- Online presence indicators
- Message history and persistence
- File/image sharing
- Push notifications for offline users

Non-Functional Requirements:

- 500M daily active users
- Low latency messaging (<100ms)
- High availability (99.99%)
- End-to-end encryption
- Support for 100,000 concurrent connections per server

Step 2: Capacity Estimation

Scale:

- DAU: 500M users
- Concurrent users: 100M (20% of DAU)
- Messages per user per day: 50
- Total messages per day: 25B
- Messages per second: 25B / (24 \* 3600) ≈ 290,000 QPS

Storage:

- Message size: 100 bytes average
- Daily storage: 25B \* 100B = 2.5TB/day
- Total storage (5 years): 2.5TB _ 365 _ 5 ≈ 4.5PB

Step 3: High-Level Architecture

Components:

1. WebSocket Gateway (real-time connections)
2. Chat Service (message processing)
3. Notification Service (push notifications)
4. User Service (presence and profiles)
5. Message Storage (database)
6. Media Service (file uploads)
7. Message Queue (async processing)

Step 4: Detailed Design

Real-time Communication:

```javascript
// WebSocket Gateway Service
class WebSocketGateway {
  constructor() {
    this.connections = new Map(); // userId -> Set of WebSocket connections
    this.userSessions = new Map(); // userId -> session info
  }

  async handleConnection(ws, userId) {
    // Store connection
    if (!this.connections.has(userId)) {
      this.connections.set(userId, new Set());
    }
    this.connections.get(userId).add(ws);

    // Update user presence
    await this.updateUserPresence(userId, "online");

    // Send pending messages
    await this.sendPendingMessages(userId);

    ws.on("message", async (message) => {
      await this.handleMessage(userId, JSON.parse(message));
    });

    ws.on("close", async () => {
      await this.handleDisconnection(userId, ws);
    });
  }

  async sendMessageToUser(userId, message) {
    const userConnections = this.connections.get(userId);

    if (userConnections && userConnections.size > 0) {
      // User is online - send directly
      const messagePayload = JSON.stringify(message);
      userConnections.forEach((ws) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(messagePayload);
        }
      });
      return true;
    }

    // User offline - store for later delivery
    await this.storePendingMessage(userId, message);
    return false;
  }

  async handleMessage(senderId, messageData) {
    try {
      // Validate message
      if (!this.validateMessage(messageData)) {
        return;
      }

      // Process message through chat service
      const processedMessage = await chatService.processMessage(
        senderId,
        messageData,
      );

      // Send to recipients
      for (const recipientId of processedMessage.recipients) {
        await this.sendMessageToUser(recipientId, processedMessage);
      }

      // Send delivery confirmation to sender
      await this.sendDeliveryConfirmation(senderId, processedMessage.messageId);
    } catch (error) {
      console.error("Error handling message:", error);
      await this.sendError(senderId, "Message processing failed");
    }
  }
}
```

Message Processing Service:

```python
class ChatService:
    async def process_message(self, sender_id, message_data):
        # Generate unique message ID
        message_id = self.generate_message_id()

        # Determine recipients
        if message_data['type'] == 'direct':
            recipients = [message_data['recipient_id']]
        elif message_data['type'] == 'group':
            recipients = await self.get_group_members(message_data['group_id'])
            recipients.remove(sender_id)  # Don't send to self

        # Create message object
        message = {
            'message_id': message_id,
            'sender_id': sender_id,
            'recipients': recipients,
            'content': message_data['content'],
            'message_type': message_data.get('message_type', 'text'),
            'timestamp': datetime.utcnow().isoformat(),
            'conversation_id': message_data.get('conversation_id')
        }

        # Store message in database
        await self.store_message(message)

        # Add to message queue for async processing
        await self.enqueue_message_tasks(message)

        return message

    async def store_message(self, message):
        # Partition messages by conversation_id for better performance
        partition_key = self.get_partition_key(message['conversation_id'])

        await database.messages.insert({
            'partition_key': partition_key,
            'message_id': message['message_id'],
            'conversation_id': message['conversation_id'],
            'sender_id': message['sender_id'],
            'content': message['content'],
            'message_type': message['message_type'],
            'timestamp': message['timestamp'],
            'delivery_status': {}  # Track delivery per recipient
        })

    async def get_conversation_history(self, conversation_id, user_id, page_token=None):
        # Verify user has access to conversation
        if not await self.user_has_access(conversation_id, user_id):
            raise PermissionError("Access denied")

        # Get paginated message history
        messages = await database.messages.find({
            'conversation_id': conversation_id
        }).sort({'timestamp': -1}).limit(50)

        # Mark messages as read
        await self.mark_messages_as_read(conversation_id, user_id)

        return messages
```

Database Schema Design:

```sql
-- Users table
CREATE TABLE users (
    user_id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    phone_number VARCHAR(20) UNIQUE NOT NULL,
    profile_picture_url TEXT,
    last_seen TIMESTAMP,
    status VARCHAR(20) DEFAULT 'offline',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Conversations table (for both direct and group chats)
CREATE TABLE conversations (
    conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_type VARCHAR(10) NOT NULL, -- 'direct' or 'group'
    created_by BIGINT REFERENCES users(user_id),
    group_name VARCHAR(100),
    group_description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Conversation participants
CREATE TABLE conversation_participants (
    conversation_id UUID REFERENCES conversations(conversation_id),
    user_id BIGINT REFERENCES users(user_id),
    role VARCHAR(20) DEFAULT 'member', -- 'admin', 'member'
    joined_at TIMESTAMP DEFAULT NOW(),
    last_read_message_id UUID,
    PRIMARY KEY (conversation_id, user_id)
);

-- Messages table (partitioned by conversation_id hash)
CREATE TABLE messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    sender_id BIGINT REFERENCES users(user_id),
    content TEXT,
    message_type VARCHAR(20) DEFAULT 'text', -- 'text', 'image', 'file', 'audio'
    media_url TEXT,
    reply_to_message_id UUID,
    timestamp TIMESTAMP DEFAULT NOW(),
    edited_at TIMESTAMP
) PARTITION BY HASH (conversation_id);

-- Message delivery status
CREATE TABLE message_delivery (
    message_id UUID REFERENCES messages(message_id),
    recipient_id BIGINT REFERENCES users(user_id),
    status VARCHAR(20) DEFAULT 'sent', -- 'sent', 'delivered', 'read'
    timestamp TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (message_id, recipient_id)
);

-- Indexes for performance
CREATE INDEX idx_messages_conversation_timestamp
ON messages(conversation_id, timestamp DESC);

CREATE INDEX idx_message_delivery_recipient_status
ON message_delivery(recipient_id, status);

CREATE INDEX idx_conversations_participant
ON conversation_participants(user_id, conversation_id);
```

Presence Service:

```python
class PresenceService:
    def __init__(self):
        self.redis = redis.Redis()
        self.presence_ttl = 300  # 5 minutes

    async def update_user_presence(self, user_id, status, device_info=None):
        # Update in Redis with TTL
        presence_data = {
            'status': status,  # 'online', 'away', 'offline'
            'last_seen': datetime.utcnow().isoformat(),
            'device_info': device_info
        }

        await self.redis.setex(
            f"presence:{user_id}",
            self.presence_ttl,
            json.dumps(presence_data)
        )

        # Notify user's contacts about status change
        await self.notify_contacts_presence_change(user_id, status)

    async def get_user_presence(self, user_id):
        presence_data = await self.redis.get(f"presence:{user_id}")

        if presence_data:
            return json.loads(presence_data)

        # If not in Redis, user is offline
        # Get last seen from database
        user = await database.users.find_one({'user_id': user_id})
        return {
            'status': 'offline',
            'last_seen': user['last_seen'] if user else None
        }

    async def get_contacts_presence(self, user_id):
        # Get user's contacts
        contacts = await self.get_user_contacts(user_id)

        # Batch get presence for all contacts
        presence_pipeline = self.redis.pipeline()
        for contact_id in contacts:
            presence_pipeline.get(f"presence:{contact_id}")

        presence_results = await presence_pipeline.execute()

        contacts_presence = {}
        for i, contact_id in enumerate(contacts):
            if presence_results[i]:
                contacts_presence[contact_id] = json.loads(presence_results[i])
            else:
                contacts_presence[contact_id] = {'status': 'offline'}

        return contacts_presence
```

Push Notification Service:

```python
class NotificationService:
    async def send_push_notification(self, user_id, message_data):
        # Get user's device tokens
        device_tokens = await self.get_user_device_tokens(user_id)

        if not device_tokens:
            return

        # Prepare notification payload
        notification = {
            'title': f"New message from {message_data['sender_name']}",
            'body': self.truncate_message(message_data['content']),
            'data': {
                'message_id': message_data['message_id'],
                'conversation_id': message_data['conversation_id'],
                'sender_id': message_data['sender_id']
            },
            'badge_count': await self.get_unread_count(user_id)
        }

        # Send to all user devices
        tasks = []
        for device_token in device_tokens:
            if device_token['platform'] == 'ios':
                tasks.append(self.send_apns_notification(device_token['token'], notification))
            elif device_token['platform'] == 'android':
                tasks.append(self.send_fcm_notification(device_token['token'], notification))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def send_fcm_notification(self, device_token, notification):
        # Firebase Cloud Messaging implementation
        payload = {
            'to': device_token,
            'notification': {
                'title': notification['title'],
                'body': notification['body'],
                'sound': 'default'
            },
            'data': notification['data'],
            'priority': 'high'
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://fcm.googleapis.com/fcm/send',
                headers={
                    'Authorization': f'Bearer {FCM_SERVER_KEY}',
                    'Content-Type': 'application/json'
                },
                json=payload
            ) as response:
                if response.status != 200:
                    logger.error(f"FCM notification failed: {await response.text()}")
```

Scaling Considerations:

1. WebSocket Connection Distribution:

```python
# Consistent hashing for WebSocket connections
class ConnectionManager:
    def __init__(self, gateway_servers):
        self.gateway_servers = gateway_servers
        self.hash_ring = hashlib.sha256()

    def get_gateway_server(self, user_id):
        # Hash user_id to determine which gateway server
        hash_value = hashlib.sha256(str(user_id).encode()).hexdigest()
        server_index = int(hash_value[:8], 16) % len(self.gateway_servers)
        return self.gateway_servers[server_index]

    async def send_message_to_user(self, user_id, message):
        gateway_server = self.get_gateway_server(user_id)

        if gateway_server == current_server:
            # User connected to this server
            await self.send_local_message(user_id, message)
        else:
            # User connected to different server
            await self.send_remote_message(gateway_server, user_id, message)
```

2. Message Queue for Reliability:

```python
# Reliable message delivery with retry logic
class ReliableMessageQueue:
    async def publish_message(self, message):
        # Add message to queue with retry metadata
        queue_message = {
            'id': str(uuid.uuid4()),
            'payload': message,
            'retry_count': 0,
            'max_retries': 3,
            'next_retry': datetime.utcnow(),
            'created_at': datetime.utcnow()
        }

        await self.message_queue.publish('chat_messages', queue_message)

    async def process_messages(self):
        async for queue_message in self.message_queue.consume('chat_messages'):
            try:
                # Process the message
                await self.deliver_message(queue_message['payload'])
                await self.acknowledge_message(queue_message['id'])

            except Exception as error:
                # Handle retry logic
                if queue_message['retry_count'] < queue_message['max_retries']:
                    # Exponential backoff
                    delay = min(300, 2 ** queue_message['retry_count'] * 10)
                    queue_message['retry_count'] += 1
                    queue_message['next_retry'] = datetime.utcnow() + timedelta(seconds=delay)

                    await self.reschedule_message(queue_message)
                else:
                    # Max retries exceeded - send to dead letter queue
                    await self.send_to_dlq(queue_message)
```

Step 5: Additional Features

End-to-End Encryption:

```javascript
// Client-side encryption (simplified)
class E2EEManager {
  async generateKeyPair() {
    return await window.crypto.subtle.generateKey(
      {
        name: "RSA-OAEP",
        modulusLength: 2048,
        publicExponent: new Uint8Array([1, 0, 1]),
        hash: "SHA-256",
      },
      true,
      ["encrypt", "decrypt"],
    );
  }

  async encryptMessage(message, recipientPublicKey) {
    // Generate symmetric key for this message
    const symmetricKey = await this.generateSymmetricKey();

    // Encrypt message with symmetric key
    const encryptedMessage = await this.encryptWithSymmetricKey(
      message,
      symmetricKey,
    );

    // Encrypt symmetric key with recipient's public key
    const encryptedKey = await window.crypto.subtle.encrypt(
      { name: "RSA-OAEP" },
      recipientPublicKey,
      symmetricKey,
    );

    return {
      encryptedMessage: encryptedMessage,
      encryptedKey: encryptedKey,
    };
  }

  async decryptMessage(encryptedData, privateKey) {
    // Decrypt symmetric key
    const symmetricKey = await window.crypto.subtle.decrypt(
      { name: "RSA-OAEP" },
      privateKey,
      encryptedData.encryptedKey,
    );

    // Decrypt message
    return await this.decryptWithSymmetricKey(
      encryptedData.encryptedMessage,
      symmetricKey,
    );
  }
}
```

Media File Handling:

```python
class MediaService:
    async def upload_file(self, file_data, user_id, conversation_id):
        # Generate unique file ID
        file_id = str(uuid.uuid4())

        # Validate file type and size
        if not self.is_valid_file(file_data):
            raise ValueError("Invalid file type or size")

        # Upload to cloud storage (AWS S3, Google Cloud Storage)
        file_url = await self.cloud_storage.upload(
            file_data,
            f"chat_media/{conversation_id}/{file_id}",
            metadata={
                'uploaded_by': user_id,
                'conversation_id': conversation_id,
                'content_type': file_data.content_type
            }
        )

        # Generate thumbnails for images/videos
        if file_data.content_type.startswith('image/'):
            thumbnail_url = await self.generate_thumbnail(file_url)
        else:
            thumbnail_url = None

        # Store metadata in database
        await database.media_files.insert({
            'file_id': file_id,
            'conversation_id': conversation_id,
            'uploaded_by': user_id,
            'file_url': file_url,
            'thumbnail_url': thumbnail_url,
            'file_name': file_data.filename,
            'file_size': len(file_data.data),
            'content_type': file_data.content_type,
            'uploaded_at': datetime.utcnow()
        })

        return {
            'file_id': file_id,
            'file_url': file_url,
            'thumbnail_url': thumbnail_url
        }
```

```

## 🧠 Interview Strategy and Tips

### Systematic Approach
```

Time Management (45-60 minutes):
5 min - Problem clarification and requirements
10 min - High-level design and major components
15 min - Deep dive into 2-3 components
10 min - Scaling and optimization discussion
5 min - Additional considerations and questions

Communication Strategy:

1. Think out loud - explain your reasoning
2. Draw diagrams - visual communication is key
3. Ask questions - engage with the interviewer
4. Acknowledge trade-offs - show you understand complexity
5. Stay calm - don't panic if you don't know something

Common Mistakes to Avoid:
❌ Diving into details before high-level design
❌ Not asking clarifying questions
❌ Ignoring scale requirements
❌ Over-engineering the solution
❌ Not considering trade-offs
❌ Poor communication and explanation

Example Communication:
"I'm going to start with a high-level architecture and then dive into the specific components. Please let me know if you'd like me to focus on any particular area."

"For the database choice, I'm considering the trade-offs between SQL and NoSQL. Given our requirements for [specific need], I think [choice] would be better because [reasoning]. What are your thoughts on this approach?"

"I notice we haven't discussed monitoring and observability. In a production system, I would also consider [specific monitoring approaches]. Would you like me to elaborate on that?"

```

### Technical Deep Dive Preparation
```

Database Systems:

- ACID properties and when they matter
- CAP theorem practical implications
- Sharding strategies and challenges
- Replication types and trade-offs
- NoSQL database types and use cases

Distributed Systems:

- Consistency models (eventual, strong, weak)
- Consensus algorithms (Raft, PBFT)
- Distributed caching strategies
- Service discovery mechanisms
- Circuit breaker and timeout patterns

Performance and Optimization:

- Caching layers and invalidation strategies
- Content Delivery Networks (CDN)
- Database indexing and query optimization
- Load balancing algorithms
- Horizontal vs vertical scaling

Reliability and Monitoring:

- Health checks and service monitoring
- Alerting and incident response
- Backup and disaster recovery
- Error handling and graceful degradation
- Performance metrics and SLAs

Security Considerations:

- Authentication and authorization
- Data encryption (at rest and in transit)
- Rate limiting and DDoS protection
- Input validation and sanitization
- Security headers and HTTPS

```

---

*Comprehensive guide to master system design interviews with real-world examples and practical solutions*
```
