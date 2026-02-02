# System Design - Quick Reference Cheatsheet

## 🎯 Core Design Principles

### Scalability Patterns

```
Horizontal Scaling (Scale Out)
- Add more servers/instances
- Load balancing required
- Better for handling traffic spikes
- More complex deployment

Vertical Scaling (Scale Up)
- Increase server capacity (CPU, RAM)
- Simpler implementation
- Limited by hardware constraints
- Single point of failure
```

### CAP Theorem

```
Consistency + Availability + Partition Tolerance
Pick any TWO:

CA (Consistency + Availability)
- Traditional RDBMS (MySQL, PostgreSQL)
- Works when network is reliable

CP (Consistency + Partition Tolerance)
- MongoDB, HBase, Redis
- Prioritizes data consistency

AP (Availability + Partition Tolerance)
- Cassandra, DynamoDB, CouchDB
- Prioritizes system availability
```

## 🏗️ System Architecture Patterns

### Microservices Architecture

```
Advantages:
✅ Technology diversity
✅ Independent deployments
✅ Fault isolation
✅ Team autonomy

Challenges:
❌ Network complexity
❌ Data consistency
❌ Service discovery
❌ Distributed debugging
```

### Event-Driven Architecture

```
Event Flow:
Producer → Event Bus → Consumer(s)

Patterns:
- Event Sourcing: Store events, not state
- CQRS: Separate read/write models
- Saga Pattern: Distributed transactions
```

### Layered Architecture

```
Presentation Layer    (UI/API)
    ↓
Business Logic Layer  (Services)
    ↓
Data Access Layer     (Repositories)
    ↓
Database Layer        (Storage)
```

## ⚖️ Load Balancing

### Load Balancer Types

```
Layer 4 (Transport Layer)
- Routes based on IP/Port
- Faster, lower overhead
- Examples: AWS ALB, HAProxy

Layer 7 (Application Layer)
- Routes based on content (HTTP headers, URLs)
- More intelligent routing
- SSL termination capability
```

### Load Balancing Algorithms

```
Round Robin
- Equal distribution
- Simple implementation

Weighted Round Robin
- Servers have different capacities
- Assign weights based on capacity

Least Connections
- Routes to server with fewest active connections
- Good for long-lived connections

IP Hash
- Routes based on client IP hash
- Ensures session affinity
```

### Implementation Example

```nginx
# NGINX Load Balancer
upstream backend {
    least_conn;
    server backend1.example.com weight=3;
    server backend2.example.com weight=1;
    server backend3.example.com backup;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🗄️ Database Design Patterns

### RDBMS vs NoSQL Decision Matrix

```
Use RDBMS when:
✅ ACID compliance required
✅ Complex relationships
✅ Structured data
✅ Strong consistency needed

Use NoSQL when:
✅ Rapid scaling needed
✅ Flexible schema
✅ Big data processing
✅ Geographic distribution
```

### Database Sharding Strategies

```
Horizontal Partitioning (Sharding):

Range-based Sharding
- Partition by date ranges
- Example: Jan data → Shard1, Feb data → Shard2

Hash-based Sharding
- Hash user ID to determine shard
- Even distribution

Directory-based Sharding
- Lookup service maps keys to shards
- Most flexible but adds complexity
```

### Database Replication

```
Master-Slave Replication
Master (Write) → Slave1 (Read)
              → Slave2 (Read)

Master-Master Replication
Master1 ↔ Master2 (Both read/write)
Risk: Write conflicts

Read Replicas
- Asynchronous replication
- Eventually consistent
- Reduces read load on master
```

## 💾 Caching Strategies

### Caching Levels

```
Browser Cache
    ↓
CDN Cache (Geographic)
    ↓
Load Balancer Cache
    ↓
Application Cache (Redis/Memcached)
    ↓
Database Cache
```

### Caching Patterns

```
Cache-Aside (Lazy Loading)
1. Check cache
2. If miss, fetch from DB
3. Update cache

Write-Through
1. Write to cache
2. Cache writes to DB
3. Slower writes, fresh cache

Write-Behind (Write-Back)
1. Write to cache
2. Async write to DB later
3. Faster writes, risk of data loss

Cache Invalidation Strategies:
- TTL (Time To Live)
- Event-based invalidation
- Manual purge
```

### Redis Implementation

```javascript
// Cache-aside pattern
async function getUser(userId) {
  // Check cache first
  const cached = await redis.get(`user:${userId}`);
  if (cached) return JSON.parse(cached);

  // Cache miss - fetch from database
  const user = await db.user.findById(userId);

  // Update cache
  await redis.setex(`user:${userId}`, 3600, JSON.stringify(user));

  return user;
}

// Cache invalidation
async function updateUser(userId, data) {
  const user = await db.user.update(userId, data);
  await redis.del(`user:${userId}`); // Invalidate cache
  return user;
}
```

## 🔄 Message Queues

### Queue Patterns

```
Point-to-Point (Queue)
Producer → Queue → Consumer
- One message, one consumer

Publish-Subscribe (Topic)
Producer → Topic → Consumer1
              ↓     Consumer2
              ↓     Consumer3
- One message, multiple consumers
```

### Popular Queue Systems

```
RabbitMQ
- AMQP protocol
- Complex routing
- Strong consistency

Apache Kafka
- High throughput
- Distributed streaming
- Partition-based scaling

AWS SQS
- Managed service
- Simple setup
- At-least-once delivery

Redis Pub/Sub
- In-memory
- Real-time messaging
- No persistence by default
```

### Implementation Examples

```python
# RabbitMQ Producer
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='user_events', durable=True)

message = json.dumps({'user_id': 123, 'event': 'signup'})
channel.basic_publish(
    exchange='',
    routing_key='user_events',
    body=message,
    properties=pika.BasicProperties(delivery_mode=2)  # Persistent
)

# RabbitMQ Consumer
def process_user_event(ch, method, properties, body):
    event = json.loads(body)
    print(f"Processing event: {event}")

    # Acknowledge message after processing
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(
    queue='user_events',
    on_message_callback=process_user_event
)

channel.start_consuming()
```

## 🔍 Search & Indexing

### Search Solutions

```
Elasticsearch
- Full-text search
- Real-time indexing
- Distributed
- RESTful API

Apache Solr
- Similar to Elasticsearch
- XML configuration
- Strong admin UI

Database Full-Text Search
- PostgreSQL: GIN/GiST indexes
- MySQL: FULLTEXT indexes
- Limited compared to dedicated solutions
```

### Elasticsearch Basics

```json
// Create index with mapping
PUT /users
{
  "mappings": {
    "properties": {
      "name": { "type": "text", "analyzer": "standard" },
      "email": { "type": "keyword" },
      "created_at": { "type": "date" }
    }
  }
}

// Search query
GET /users/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "name": "John" } }
      ],
      "filter": [
        { "range": { "created_at": { "gte": "2023-01-01" } } }
      ]
    }
  }
}
```

## 📊 Monitoring & Observability

### The Three Pillars

```
Metrics (What happened?)
- CPU, Memory, Disk usage
- Request rate, error rate
- Business metrics

Logs (Detailed context)
- Application logs
- Access logs
- Error logs

Traces (Request flow)
- Distributed tracing
- Performance bottlenecks
- Service dependencies
```

### Monitoring Stack

```
Data Collection:
- Prometheus (metrics)
- Jaeger/Zipkin (tracing)
- ELK Stack (logs)

Visualization:
- Grafana dashboards
- Kibana (log analysis)

Alerting:
- AlertManager
- PagerDuty integration
```

### Application Metrics

```javascript
// Prometheus metrics in Node.js
const promClient = require("prom-client");

// Counter: Only increases
const httpRequests = new promClient.Counter({
  name: "http_requests_total",
  help: "Total HTTP requests",
  labelNames: ["method", "route", "status_code"],
});

// Histogram: Request duration
const httpDuration = new promClient.Histogram({
  name: "http_request_duration_seconds",
  help: "HTTP request duration",
  buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10],
});

// Middleware
app.use((req, res, next) => {
  const start = Date.now();

  res.on("finish", () => {
    const duration = (Date.now() - start) / 1000;

    httpRequests
      .labels(req.method, req.route?.path || "unknown", res.statusCode)
      .inc();

    httpDuration.observe(duration);
  });

  next();
});
```

## 🔒 Security Patterns

### Authentication vs Authorization

```
Authentication (Who are you?)
- Verifies identity
- Username/password, OAuth, JWT

Authorization (What can you do?)
- Verifies permissions
- RBAC, ABAC, ACL
```

### Security Headers

```http
# Security Headers
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'
```

### Rate Limiting Implementation

```javascript
// Token bucket algorithm
class TokenBucket {
  constructor(capacity, refillRate) {
    this.capacity = capacity;
    this.tokens = capacity;
    this.refillRate = refillRate; // tokens per second
    this.lastRefill = Date.now();
  }

  consume(tokens = 1) {
    this.refill();

    if (this.tokens >= tokens) {
      this.tokens -= tokens;
      return true;
    }

    return false; // Rate limited
  }

  refill() {
    const now = Date.now();
    const timePassed = (now - this.lastRefill) / 1000;
    const tokensToAdd = timePassed * this.refillRate;

    this.tokens = Math.min(this.capacity, this.tokens + tokensToAdd);
    this.lastRefill = now;
  }
}
```

## 📈 Performance Optimization

### Database Optimization

```sql
-- Indexing strategies
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_post_user_created ON posts(user_id, created_at);

-- Query optimization
-- Avoid N+1 queries
SELECT u.*, p.title
FROM users u
LEFT JOIN posts p ON u.id = p.user_id
WHERE u.active = true;

-- Use EXPLAIN to analyze queries
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'john@example.com';
```

### API Optimization

```javascript
// Pagination
app.get("/api/posts", async (req, res) => {
  const limit = Math.min(parseInt(req.query.limit) || 20, 100);
  const cursor = req.query.cursor;

  const posts = await Post.find({
    ...(cursor && { _id: { $gt: cursor } }),
  })
    .sort({ _id: 1 })
    .limit(limit + 1);

  const hasMore = posts.length > limit;
  const results = hasMore ? posts.slice(0, -1) : posts;

  res.json({
    data: results,
    hasMore,
    nextCursor: hasMore ? results[results.length - 1]._id : null,
  });
});

// Response compression
const compression = require("compression");
app.use(compression());

// HTTP/2 Server Push (Express 5+)
app.get("/", (req, res) => {
  res.push("/styles.css");
  res.push("/app.js");
  res.render("index");
});
```

## 🌐 CDN & Edge Computing

### CDN Strategies

```
Static Content Caching
- Images, CSS, JavaScript
- Long TTL (cache time)

Dynamic Content Caching
- API responses
- Shorter TTL
- Cache invalidation

Edge Side Includes (ESI)
- Partial page caching
- Mix cached and dynamic content
```

### CloudFront Configuration

```javascript
// Cache behaviors
const distribution = {
  Origins: [
    {
      Id: "origin1",
      DomainName: "api.example.com",
      CustomOriginConfig: {
        HTTPPort: 80,
        HTTPSPort: 443,
        OriginProtocolPolicy: "https-only",
      },
    },
  ],
  DefaultCacheBehavior: {
    TargetOriginId: "origin1",
    ViewerProtocolPolicy: "redirect-to-https",
    CachePolicyId: "caching-disabled", // For dynamic content
    AllowedMethods: [
      "GET",
      "HEAD",
      "OPTIONS",
      "PUT",
      "POST",
      "PATCH",
      "DELETE",
    ],
  },
  CacheBehaviors: [
    {
      PathPattern: "/static/*",
      TargetOriginId: "origin1",
      CachePolicyId: "caching-optimized", // For static content
      TTL: 86400, // 24 hours
    },
  ],
};
```

## 🔄 API Design Patterns

### GraphQL vs REST

```
REST
✅ Simple caching
✅ Wide tooling support
✅ Stateless
❌ Over/under-fetching
❌ Multiple requests

GraphQL
✅ Single endpoint
✅ Flexible queries
✅ Strong typing
❌ Complex caching
❌ Learning curve
```

### API Versioning Strategies

```
URL Versioning
GET /api/v1/users
GET /api/v2/users

Header Versioning
GET /api/users
Accept: application/vnd.api+json;version=1

Query Parameter
GET /api/users?version=1

Media Type
Accept: application/vnd.user.v1+json
```

### Pagination Patterns

```javascript
// Cursor-based (Recommended for large datasets)
GET /api/posts?cursor=eyJpZCI6MTIzfQ&limit=20
{
    "data": [...],
    "pagination": {
        "next_cursor": "eyJpZCI6MTQ0fQ",
        "has_more": true
    }
}

// Offset-based (Simple but less efficient)
GET /api/posts?page=2&limit=20
{
    "data": [...],
    "pagination": {
        "page": 2,
        "limit": 20,
        "total": 1000,
        "pages": 50
    }
}
```

## 🏗️ Design Process

### 1. Requirements Gathering

```
Functional Requirements:
- What features?
- User interactions?
- Business logic?

Non-Functional Requirements:
- Scale (users, requests/second)
- Performance (latency, throughput)
- Availability (uptime)
- Consistency requirements
```

### 2. Capacity Estimation

```
Read/Write Ratio: 100:1 (typical social media)

Daily Active Users (DAU): 10M
Reads per user per day: 50
Total reads: 10M × 50 = 500M reads/day
Read QPS: 500M / (24 × 3600) ≈ 5,800 reads/second

Writes per user per day: 0.5
Total writes: 10M × 0.5 = 5M writes/day
Write QPS: 5M / (24 × 3600) ≈ 58 writes/second

Peak QPS = Average QPS × 3
```

### 3. High-Level Design

```
Client → CDN → Load Balancer → API Gateway → Microservices
                                              ↓
                                         Database Cluster
                                         Cache Layer
                                         Message Queues
```

### 4. Detailed Component Design

```
For each component:
- API interfaces
- Data models
- Algorithms
- Storage schema
- Caching strategy
```

## 🛠️ Common System Design Questions

### URL Shortener (bit.ly)

```
Requirements:
- Shorten long URLs
- Redirect to original URL
- Analytics
- 100M URLs/day

Design:
1. Base62 encoding for short URLs
2. Database: URL mapping table
3. Cache: Popular URLs in Redis
4. Analytics: Message queue → Analytics service

URL Generation:
- Option 1: Auto-increment ID → Base62
- Option 2: Hash function
- Option 3: Random string with collision check
```

### Chat System (WhatsApp)

```
Requirements:
- 1-on-1 messaging
- Group messaging
- Online presence
- Message history

Design:
1. WebSocket connections for real-time
2. Message service with message queues
3. User service for authentication
4. Notification service for offline users
5. Database: User data, message history

Message Flow:
User A → Gateway → Message Service → Queue → User B
                      ↓
                 Message Storage
```

### News Feed (Facebook)

```
Requirements:
- Generate feed for users
- Follow other users
- Post content
- Real-time updates

Design:
1. User service (profiles, following)
2. Post service (create, store posts)
3. Feed generation service
4. Timeline service (user's feed cache)

Feed Generation:
- Pull model: Generate on request (lazy)
- Push model: Pre-compute feeds (eager)
- Hybrid: Push for active users, pull for others
```

## 📚 Quick Reference Commands

### Database Operations

```sql
-- PostgreSQL performance
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM users WHERE email = 'john@example.com';

-- Create index concurrently
CREATE INDEX CONCURRENTLY idx_user_email ON users(email);

-- MongoDB aggregation
db.posts.aggregate([
    { $match: { status: "published" } },
    { $group: { _id: "$author", count: { $sum: 1 } } },
    { $sort: { count: -1 } }
]);
```

### Docker Commands

```bash
# Multi-container setup
docker-compose up -d

# Scale services
docker-compose scale web=3

# View logs
docker-compose logs -f web

# Performance monitoring
docker stats

# Cleanup
docker system prune -a
```

### Kubernetes Commands

```bash
# Deploy application
kubectl apply -f deployment.yaml

# Scale deployment
kubectl scale deployment webapp --replicas=5

# Check pod status
kubectl get pods -o wide

# View logs
kubectl logs -f deployment/webapp

# Port forwarding
kubectl port-forward service/webapp 8080:80
```

## 🎯 System Design Interview Tips

### 1. Clarification Questions (5 minutes)

- Scale requirements (users, QPS)
- Features scope
- Performance requirements
- Budget constraints

### 2. High-Level Design (10 minutes)

- Draw main components
- Show data flow
- Identify bottlenecks

### 3. Deep Dive (15 minutes)

- Database schema
- API design
- Caching strategy
- Scaling approach

### 4. Scale the Design (10 minutes)

- Identify bottlenecks
- Propose solutions
- Trade-offs discussion

---

_Essential patterns and concepts for system design interviews and real-world applications_
