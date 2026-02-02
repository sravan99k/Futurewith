# DevOps & Containerization: Modern Development and Deployment (2025)

---

# Comprehensive Learning System

title: "DevOps & Containerization: Modern Development and Deployment"
level: "Intermediate to Advanced"
time_to_complete: "15-20 hours"
prerequisites: ["Basic Linux knowledge", "Programming fundamentals", "Understanding of web applications", "Basic cloud concepts"]
skills_gained: ["Containerization with Docker", "Kubernetes orchestration", "CI/CD pipeline design", "Infrastructure as Code", "Monitoring and observability", "Cloud platform expertise"]
success_criteria: ["Build and deploy containerized applications", "Implement CI/CD pipelines", "Design scalable infrastructure", "Configure monitoring and alerting", "Manage cloud resources effectively"]
tags: ["devops", "docker", "kubernetes", "ci/cd", "infrastructure", "cloud", "automation"]
description: "Master modern DevOps practices including containerization, orchestration, CI/CD pipelines, and cloud deployment strategies. Learn to build, deploy, and manage scalable applications using industry-standard tools and practices."

---

## Table of Contents

1. [DevOps Philosophy and Culture](#devops-philosophy-and-culture)
2. [Containerization with Docker](#containerization-with-docker)
3. [Container Orchestration with Kubernetes](#container-orchestration-with-kubernetes)
4. [CI/CD Pipeline Implementation](#cicd-pipeline-implementation)
5. [Infrastructure as Code](#infrastructure-as-code)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Security in DevOps (DevSecOps)](#security-in-devops-devsecops)
8. [Cloud Platforms and Services](#cloud-platforms-and-services)
9. [Performance Optimization and Scaling](#performance-optimization-and-scaling)
10. [DevOps Best Practices](#devops-best-practices)

---

## Learning Goals

By the end of this module, you will be able to:

1. **Build and Deploy Containerized Applications** - Create Docker containers and manage containerized workloads efficiently
2. **Implement Container Orchestration** - Deploy and manage applications at scale using Kubernetes
3. **Design CI/CD Pipelines** - Build automated testing, integration, and deployment workflows
4. **Apply Infrastructure as Code** - Manage infrastructure through declarative configuration and automation
5. **Implement Monitoring and Observability** - Set up comprehensive monitoring, logging, and alerting systems
6. **Integrate Security into DevOps** - Apply DevSecOps principles and security practices throughout the pipeline
7. **Optimize Cloud Resource Management** - Deploy and manage applications effectively on cloud platforms
8. **Drive DevOps Culture and Collaboration** - Foster team collaboration and continuous improvement practices

---

## TL;DR

DevOps is about breaking down silos between development and operations to enable faster, more reliable software delivery. Key practices: **containerization with Docker**, **orchestration with Kubernetes**, **automated CI/CD pipelines**, **Infrastructure as Code**, and **comprehensive monitoring**. Focus on automation, collaboration, and continuous improvement to deliver value faster and more reliably.

---

## DevOps Philosophy and Culture

### The DevOps Transformation

#### **Core DevOps Principles**

**1. Collaboration Over Silos**
Traditional development often creates barriers between teams:

- Development writes code
- Operations deploys and maintains
- Security reviews and approves
- QA tests and validates

DevOps breaks down these silos:

```
Traditional:  Dev → QA → Security → Ops (Sequential, Slow)
DevOps:      Dev ⟷ QA ⟷ Security ⟷ Ops (Collaborative, Fast)
```

**2. Automation Over Manual Processes**

- **Build automation:** Compile, test, package automatically
- **Deployment automation:** Consistent, repeatable deployments
- **Testing automation:** Continuous quality validation
- **Infrastructure automation:** Infrastructure provisioning and management

**3. Measurement Over Assumptions**

- **Metrics-driven decisions:** Use data to guide improvements
- **Continuous monitoring:** Real-time visibility into system health
- **Feedback loops:** Quick detection and resolution of issues
- **Performance tracking:** Measure and optimize continuously

#### **The Three Ways of DevOps**

**First Way: Flow**
Optimize the flow from development to operations:

```typescript
// Example: Automated deployment pipeline
class DeploymentPipeline {
  async deploy(application: Application): Promise<DeploymentResult> {
    try {
      // 1. Build and test
      const buildResult = await this.buildService.build(application);
      const testResult = await this.testService.runTests(buildResult);

      if (!testResult.passed) {
        throw new Error(`Tests failed: ${testResult.failures.join(", ")}`);
      }

      // 2. Security scan
      const securityResult = await this.securityService.scan(buildResult);
      if (securityResult.hasVulnerabilities) {
        throw new Error(
          `Security vulnerabilities found: ${securityResult.issues}`,
        );
      }

      // 3. Deploy to staging
      const stagingDeployment = await this.deployToEnvironment(
        "staging",
        buildResult,
      );

      // 4. Run integration tests
      const integrationTests =
        await this.testService.runIntegrationTests("staging");
      if (!integrationTests.passed) {
        await this.rollback("staging");
        throw new Error("Integration tests failed");
      }

      // 5. Deploy to production
      const prodDeployment = await this.deployToEnvironment(
        "production",
        buildResult,
      );

      return {
        success: true,
        version: buildResult.version,
        deploymentTime: new Date(),
        environments: ["staging", "production"],
      };
    } catch (error) {
      await this.handleDeploymentFailure(error);
      throw error;
    }
  }
}
```

**Second Way: Feedback**
Create fast feedback loops:

- **Monitoring and alerting:** Quick problem detection
- **Log aggregation:** Centralized troubleshooting
- **Customer feedback:** Direct user input on features
- **Performance metrics:** Real-time system health

**Third Way: Continuous Learning**
Foster a culture of experimentation and learning:

- **Blameless post-mortems:** Learn from failures without blame
- **Experimentation:** A/B testing and feature flags
- **Knowledge sharing:** Documentation and team learning
- **Risk tolerance:** Accept controlled failure as learning

### DevOps Metrics and KPIs

#### **DORA Metrics (DevOps Research and Assessment)**

**1. Deployment Frequency**
How often deployments occur:

```typescript
class DeploymentMetrics {
  async getDeploymentFrequency(
    timeRange: TimeRange,
  ): Promise<DeploymentFrequency> {
    const deployments = await this.getDeployments(timeRange);
    const days = this.getDaysBetween(timeRange.start, timeRange.end);

    return {
      deploymentsPerDay: deployments.length / days,
      frequency: this.categorizeFrequency(deployments.length / days),
      trend: this.calculateTrend(deployments),
    };
  }

  private categorizeFrequency(deploymentsPerDay: number): string {
    if (deploymentsPerDay >= 1) return "On-demand (Elite)";
    if (deploymentsPerDay >= 1 / 7) return "Weekly (High)";
    if (deploymentsPerDay >= 1 / 30) return "Monthly (Medium)";
    return "Less than monthly (Low)";
  }
}
```

**2. Lead Time for Changes**
Time from code commit to production deployment:

```typescript
interface ChangeLeadTime {
  commitTime: Date;
  deploymentTime: Date;
  leadTimeMinutes: number;
}

class LeadTimeTracker {
  async calculateLeadTime(commit: GitCommit): Promise<ChangeLeadTime> {
    const deployment = await this.findDeploymentForCommit(commit.sha);
    const leadTimeMs =
      deployment.timestamp.getTime() - commit.timestamp.getTime();

    return {
      commitTime: commit.timestamp,
      deploymentTime: deployment.timestamp,
      leadTimeMinutes: leadTimeMs / (1000 * 60),
    };
  }

  async getAverageLeadTime(timeRange: TimeRange): Promise<number> {
    const commits = await this.getCommitsInRange(timeRange);
    const leadTimes = await Promise.all(
      commits.map((commit) => this.calculateLeadTime(commit)),
    );

    return (
      leadTimes.reduce((sum, lt) => sum + lt.leadTimeMinutes, 0) /
      leadTimes.length
    );
  }
}
```

**3. Mean Time to Recovery (MTTR)**
Average time to recover from failures:

```typescript
class IncidentMetrics {
  async calculateMTTR(incidents: Incident[]): Promise<number> {
    const resolvedIncidents = incidents.filter((i) => i.resolvedAt);

    const recoveryTimes = resolvedIncidents.map((incident) => {
      const startTime = incident.detectedAt || incident.createdAt;
      const endTime = incident.resolvedAt!;
      return endTime.getTime() - startTime.getTime();
    });

    return (
      recoveryTimes.reduce((sum, time) => sum + time, 0) / recoveryTimes.length
    );
  }
}
```

**4. Change Failure Rate**
Percentage of deployments that result in failures:

```typescript
class FailureRateMetrics {
  async getChangeFailureRate(timeRange: TimeRange): Promise<number> {
    const deployments = await this.getDeployments(timeRange);
    const failedDeployments = deployments.filter(
      (d) => d.status === "failed" || d.rollbackRequired,
    );

    return (failedDeployments.length / deployments.length) * 100;
  }
}
```

---

## Containerization with Docker

### Docker Fundamentals

#### **Docker Architecture**

```
┌─────────────────────────────────────────────┐
│                Docker Client               │
│  (docker build, docker run, docker push)   │
└─────────────────┬───────────────────────────┘
                  │ REST API
┌─────────────────▼───────────────────────────┐
│              Docker Daemon                  │
│  ┌─────────────┐ ┌─────────────┐ ┌────────┐ │
│  │ Containers  │ │   Images    │ │Networks│ │
│  └─────────────┘ └─────────────┘ └────────┘ │
└─────────────────────────────────────────────┘
```

#### **Dockerfile Best Practices**

**Multi-stage Build for Node.js Application:**

```dockerfile
# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (including devDependencies)
RUN npm ci --only=production=false

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM node:18-alpine AS production

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install only production dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy built application from builder stage
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Start the application
CMD ["node", "dist/index.js"]
```

**Python Application with Poetry:**

```dockerfile
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Activate virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### **Docker Compose for Local Development**

**Complete Development Environment:**

```yaml
version: "3.8"

services:
  # Application
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgresql://user:password@postgres:5432/myapp
      - REDIS_URL=redis://redis:6379
    volumes:
      - .:/app
      - /app/node_modules # Anonymous volume for node_modules
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - app-network

  # Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d myapp"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - app-network

  # Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    networks:
      - app-network

  # Message Queue
  rabbitmq:
    image: rabbitmq:3-management-alpine
    environment:
      RABBITMQ_DEFAULT_USER: user
      RABBITMQ_DEFAULT_PASS: password
    ports:
      - "5672:5672"
      - "15672:15672" # Management UI
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "-q", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - app-network

  # Development tools
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.dev.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - app
    networks:
      - app-network

volumes:
  postgres_data:
  redis_data:
  rabbitmq_data:

networks:
  app-network:
    driver: bridge
```

### Container Security and Optimization

#### **Security Best Practices**

**1. Image Security Scanning:**

```bash
# Scan image for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image myapp:latest

# Scan during build
docker build -t myapp:latest .
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image --exit-code 1 myapp:latest
```

**2. Distroless Base Images:**

```dockerfile
# Using distroless for minimal attack surface
FROM gcr.io/distroless/nodejs18-debian11

WORKDIR /app

# Copy application and dependencies
COPY --from=builder /app/dist ./
COPY --from=builder /app/node_modules ./node_modules

# Expose port
EXPOSE 3000

# Run as non-root
USER 1001:1001

# Start application
CMD ["index.js"]
```

**3. Runtime Security:**

```yaml
# docker-compose.yml with security constraints
services:
  app:
    image: myapp:latest
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:rw,noexec,nosuid,size=100m
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETGID
      - SETUID
    ulimits:
      nproc: 65535
      nofile:
        soft: 65535
        hard: 65535
```

#### **Performance Optimization**

**1. Layer Caching Optimization:**

```dockerfile
# Bad: Changes to source code invalidate package installation
COPY . .
RUN npm install

# Good: Install packages first, then copy source
COPY package*.json ./
RUN npm ci --only=production
COPY . .
```

**2. Multi-architecture Builds:**

```dockerfile
# Dockerfile supporting multiple architectures
FROM --platform=$BUILDPLATFORM node:18-alpine AS base
ARG TARGETPLATFORM
ARG BUILDPLATFORM
RUN echo "Building on $BUILDPLATFORM for $TARGETPLATFORM"

FROM base AS builder
# Build logic here...

FROM base AS production
# Production logic here...
```

**3. Image Size Optimization:**

```dockerfile
# Use alpine variants
FROM node:18-alpine

# Remove package manager cache
RUN npm ci --only=production && npm cache clean --force

# Remove unnecessary files
RUN rm -rf /usr/share/man/* /usr/share/doc/* /tmp/* /var/tmp/*

# Use .dockerignore to exclude unnecessary files
# .dockerignore:
# node_modules
# npm-debug.log
# .git
# .gitignore
# README.md
# Dockerfile
# .dockerignore
```

---

## Container Orchestration with Kubernetes

### Kubernetes Architecture and Concepts

#### **Core Components**

```
Master Node (Control Plane):
├── API Server: REST API for cluster management
├── etcd: Distributed key-value store for cluster state
├── Scheduler: Assigns pods to nodes
├── Controller Manager: Manages cluster state
└── Cloud Controller Manager: Cloud provider integration

Worker Nodes:
├── kubelet: Node agent that manages pods
├── kube-proxy: Network proxy for services
├── Container Runtime: Docker, containerd, or CRI-O
└── Pods: Smallest deployable units
```

#### **Kubernetes Manifests**

**Deployment with ConfigMap and Secret:**

```yaml
# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: myapp-config
  namespace: production
data:
  database.properties: |
    database.host=postgres-service
    database.port=5432
    database.name=myapp
  app.properties: |
    app.env=production
    app.log.level=INFO
    app.feature.newui=true

---
# Secret
apiVersion: v1
kind: Secret
metadata:
  name: myapp-secrets
  namespace: production
type: Opaque
data:
  database.password: cGFzc3dvcmQxMjM= # base64 encoded
  api.key: YWJjZGVmZ2hpams= # base64 encoded

---
# Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-deployment
  namespace: production
  labels:
    app: myapp
    version: v1.2.3
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
        version: v1.2.3
    spec:
      serviceAccountName: myapp-service-account
      containers:
        - name: myapp
          image: myregistry/myapp:v1.2.3
          ports:
            - containerPort: 3000
              name: http
          env:
            - name: DATABASE_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: myapp-secrets
                  key: database.password
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: myapp-secrets
                  key: api.key
          volumeMounts:
            - name: config-volume
              mountPath: /app/config
            - name: tmp-volume
              mountPath: /tmp
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 2
          securityContext:
            runAsNonRoot: true
            runAsUser: 1001
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL
      volumes:
        - name: config-volume
          configMap:
            name: myapp-config
        - name: tmp-volume
          emptyDir: {}
      nodeSelector:
        nodeType: application
      tolerations:
        - key: "dedicated"
          operator: "Equal"
          value: "app"
          effect: "NoSchedule"

---
# Service
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
  namespace: production
  labels:
    app: myapp
spec:
  type: ClusterIP
  selector:
    app: myapp
  ports:
    - port: 80
      targetPort: http
      name: http

---
# Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: myapp-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - myapp.example.com
      secretName: myapp-tls
  rules:
    - host: myapp.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: myapp-service
                port:
                  number: 80
```

#### **Horizontal Pod Autoscaler (HPA)**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: myapp-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp-deployment
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 50
          periodSeconds: 60
        - type: Pods
          value: 4
          periodSeconds: 60
      selectPolicy: Max
```

### Advanced Kubernetes Patterns

#### **StatefulSets for Databases**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres-cluster
  namespace: database
spec:
  serviceName: postgres-cluster
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:15
          env:
            - name: POSTGRES_DB
              value: myapp
            - name: POSTGRES_USER
              value: postgres
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: password
            - name: POSTGRES_REPLICATION_MODE
              value: slave
            - name: POSTGRES_REPLICATION_USER
              value: replicator
            - name: POSTGRES_REPLICATION_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: replication-password
            - name: POSTGRES_MASTER_SERVICE
              value: postgres-cluster-0.postgres-cluster
          ports:
            - containerPort: 5432
              name: postgres
          volumeMounts:
            - name: postgres-storage
              mountPath: /var/lib/postgresql/data
            - name: postgres-config
              mountPath: /etc/postgresql
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "1"
      volumes:
        - name: postgres-config
          configMap:
            name: postgres-config
  volumeClaimTemplates:
    - metadata:
        name: postgres-storage
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 20Gi

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-cluster
  namespace: database
spec:
  clusterIP: None # Headless service for StatefulSet
  selector:
    app: postgres
  ports:
    - port: 5432
      name: postgres

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-read
  namespace: database
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      name: postgres
```

#### **Jobs and CronJobs**

```yaml
# One-time Job
apiVersion: batch/v1
kind: Job
metadata:
  name: database-migration
  namespace: production
spec:
  backoffLimit: 3
  activeDeadlineSeconds: 600 # 10 minutes timeout
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: migrator
          image: myregistry/db-migrator:latest
          command: ["node", "migrate.js"]
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: database-secret
                  key: url
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "200m"

---
# Scheduled CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: backup-database
  namespace: production
spec:
  schedule: "0 2 * * *" # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: backup
              image: myregistry/backup-tool:latest
              command: ["./backup.sh"]
              env:
                - name: DATABASE_URL
                  valueFrom:
                    secretKeyRef:
                      name: database-secret
                      key: url
                - name: S3_BUCKET
                  value: "myapp-backups"
                - name: AWS_ACCESS_KEY_ID
                  valueFrom:
                    secretKeyRef:
                      name: aws-credentials
                      key: access-key-id
                - name: AWS_SECRET_ACCESS_KEY
                  valueFrom:
                    secretKeyRef:
                      name: aws-credentials
                      key: secret-access-key
              volumeMounts:
                - name: backup-storage
                  mountPath: /backups
          volumes:
            - name: backup-storage
              emptyDir: {}
  concurrencyPolicy: Forbid # Don't run concurrent backup jobs
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
```

### Kubernetes Networking and Security

#### **NetworkPolicies for Security**

```yaml
# Default deny all traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress

---
# Allow app to communicate with database
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-app-to-db
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: myapp
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: database
          podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
    - to: [] # Allow DNS resolution
      ports:
        - protocol: UDP
          port: 53

---
# Allow ingress to app
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-ingress-to-app
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: myapp
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 3000
```

#### **RBAC (Role-Based Access Control)**

```yaml
# Service Account
apiVersion: v1
kind: ServiceAccount
metadata:
  name: myapp-service-account
  namespace: production

---
# Role for application
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: myapp-role
  namespace: production
rules:
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list"]

---
# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: myapp-role-binding
  namespace: production
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: myapp-role
subjects:
  - kind: ServiceAccount
    name: myapp-service-account
    namespace: production
```

---

## CI/CD Pipeline Implementation

### GitLab CI/CD Pipeline

#### **Complete Pipeline Configuration**

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - security
  - deploy-staging
  - integration-tests
  - deploy-production
  - monitoring

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  REGISTRY: $CI_REGISTRY_IMAGE

# Common configuration
.docker-auth: &docker-auth
  - echo "$CI_REGISTRY_PASSWORD" | docker login -u "$CI_REGISTRY_USER" --password-stdin $CI_REGISTRY

.kubectl-config: &kubectl-config
  - kubectl config use-context $KUBE_CONTEXT

# Test stage
unit-tests:
  stage: test
  image: node:18-alpine
  services:
    - postgres:13
    - redis:6-alpine
  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: test_user
    POSTGRES_PASSWORD: test_pass
  script:
    - npm ci
    - npm run test:unit
    - npm run test:integration
  coverage: '/All files[^|]*\|[^|]*\s+([\d\.]+)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml
      junit: junit.xml
    paths:
      - coverage/
    expire_in: 1 week
  only:
    - merge_requests
    - main

lint-and-format:
  stage: test
  image: node:18-alpine
  script:
    - npm ci
    - npm run lint
    - npm run format:check
    - npm run type-check
  only:
    - merge_requests
    - main

# Build stage
build-image:
  stage: build
  image: docker:20.10.16
  services:
    - docker:20.10.16-dind
  before_script:
    - *docker-auth
  script:
    - docker build --pull -t $REGISTRY:$CI_COMMIT_SHA -t $REGISTRY:latest .
    - docker push $REGISTRY:$CI_COMMIT_SHA
    - docker push $REGISTRY:latest
  only:
    - main

# Security stage
security-scan:
  stage: security
  image: docker:20.10.16
  services:
    - docker:20.10.16-dind
  before_script:
    - *docker-auth
    - apk add --no-cache curl
    - curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
  script:
    - trivy image --exit-code 1 --severity HIGH,CRITICAL $REGISTRY:$CI_COMMIT_SHA
  allow_failure: false
  only:
    - main

sast-scan:
  stage: security
  image: registry.gitlab.com/gitlab-org/security-products/analyzers/semgrep:latest
  script:
    - semgrep --config=auto --json --output=sast-report.json .
  artifacts:
    reports:
      sast: sast-report.json
  only:
    - merge_requests
    - main

# Deploy staging
deploy-staging:
  stage: deploy-staging
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: https://staging.myapp.com
  before_script:
    - *kubectl-config
  script:
    - sed -i "s|IMAGE_TAG|$CI_COMMIT_SHA|g" k8s/staging/deployment.yaml
    - kubectl apply -f k8s/staging/
    - kubectl rollout status deployment/myapp-deployment -n staging --timeout=300s
  only:
    - main

# Integration tests
integration-tests:
  stage: integration-tests
  image: cypress/included:10.0.0
  variables:
    CYPRESS_baseUrl: https://staging.myapp.com
  script:
    - cypress run --env environment=staging
  artifacts:
    when: always
    paths:
      - cypress/videos/**/*.mp4
      - cypress/screenshots/**/*.png
    expire_in: 1 week
    reports:
      junit:
        - cypress/results/*.xml
  only:
    - main

# Production deployment
deploy-production:
  stage: deploy-production
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: https://myapp.com
  before_script:
    - *kubectl-config
  script:
    - sed -i "s|IMAGE_TAG|$CI_COMMIT_SHA|g" k8s/production/deployment.yaml
    - kubectl apply -f k8s/production/
    - kubectl rollout status deployment/myapp-deployment -n production --timeout=600s
  when: manual
  only:
    - main

# Post-deployment monitoring
smoke-tests:
  stage: monitoring
  image: curlimages/curl:latest
  script:
    - sleep 60 # Wait for deployment to stabilize
    - curl -f https://myapp.com/health || exit 1
    - curl -f https://myapp.com/api/status || exit 1
  only:
    - main

performance-tests:
  stage: monitoring
  image: grafana/k6:latest
  script:
    - k6 run --out json=results.json performance-tests/load-test.js
  artifacts:
    reports:
      performance: results.json
  only:
    - main
  when: manual
```

### GitHub Actions Workflow

#### **Node.js Application CI/CD**

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Test job
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:6-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: "18"
          cache: "npm"

      - name: Install dependencies
        run: npm ci

      - name: Run linting
        run: npm run lint

      - name: Run type checking
        run: npm run type-check

      - name: Run unit tests
        run: npm run test:unit
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379

      - name: Run integration tests
        run: npm run test:integration
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379

      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage/lcov.info
          fail_ci_if_error: true

  # Security scanning
  security:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: "fs"
          scan-ref: "."
          format: "sarif"
          output: "trivy-results.sarif"

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: "trivy-results.sarif"

  # Build and push Docker image
  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix=sha-
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # Deploy to staging
  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment:
      name: staging
      url: https://staging.myapp.com

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2

      - name: Deploy to EKS
        run: |
          aws eks update-kubeconfig --name staging-cluster
          sed -i "s|IMAGE_TAG|sha-${{ github.sha }}|g" k8s/staging/deployment.yaml
          kubectl apply -f k8s/staging/
          kubectl rollout status deployment/myapp-deployment -n staging --timeout=300s

  # Integration tests
  integration-tests:
    needs: deploy-staging
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run Cypress tests
        uses: cypress-io/github-action@v6
        with:
          config: baseUrl=https://staging.myapp.com
          spec: cypress/e2e/**/*.cy.js
          wait-on: "https://staging.myapp.com/health"
          wait-on-timeout: 120

      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: cypress-results
          path: |
            cypress/videos
            cypress/screenshots

  # Deploy to production
  deploy-production:
    needs: integration-tests
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment:
      name: production
      url: https://myapp.com

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2

      - name: Deploy to EKS
        run: |
          aws eks update-kubeconfig --name production-cluster
          sed -i "s|IMAGE_TAG|sha-${{ github.sha }}|g" k8s/production/deployment.yaml
          kubectl apply -f k8s/production/
          kubectl rollout status deployment/myapp-deployment -n production --timeout=600s

      - name: Run smoke tests
        run: |
          sleep 60
          curl -f https://myapp.com/health
          curl -f https://myapp.com/api/status
```

---

## Infrastructure as Code

### Terraform for Cloud Infrastructure

#### **AWS EKS Cluster with Terraform**

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
  }

  backend "s3" {
    bucket = "myapp-terraform-state"
    key    = "eks-cluster/terraform.tfstate"
    region = "us-west-2"

    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }
}

# Variables
variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "myapp-cluster"
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.cluster_name}-vpc"
    Environment = var.environment
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.cluster_name}-igw"
    Environment = var.environment
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.cluster_name}-public-${count.index + 1}"
    Environment = var.environment
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb" = "1"
  }
}

# Private Subnets
resource "aws_subnet" "private" {
  count = 2

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "${var.cluster_name}-private-${count.index + 1}"
    Environment = var.environment
    "kubernetes.io/cluster/${var.cluster_name}" = "owned"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# NAT Gateways
resource "aws_eip" "nat" {
  count = 2
  domain = "vpc"

  tags = {
    Name = "${var.cluster_name}-eip-${count.index + 1}"
    Environment = var.environment
  }
}

resource "aws_nat_gateway" "main" {
  count = 2

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "${var.cluster_name}-nat-${count.index + 1}"
    Environment = var.environment
  }
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.cluster_name}-public"
    Environment = var.environment
  }
}

resource "aws_route_table" "private" {
  count = 2
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }

  tags = {
    Name = "${var.cluster_name}-private-${count.index + 1}"
    Environment = var.environment
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = 2

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = 2

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# EKS Cluster IAM Role
resource "aws_iam_role" "cluster" {
  name = "${var.cluster_name}-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.cluster.name
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  role_arn = aws_iam_role.cluster.arn
  version  = "1.27"

  vpc_config {
    subnet_ids              = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
  }

  encryption_config {
    provider {
      key_arn = aws_kms_key.eks.arn
    }
    resources = ["secrets"]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.cluster_policy,
  ]

  tags = {
    Environment = var.environment
  }
}

# KMS Key for EKS encryption
resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Name = "${var.cluster_name}-eks-encryption"
    Environment = var.environment
  }
}

# Node Group IAM Role
resource "aws_iam_role" "node_group" {
  name = "${var.cluster_name}-node-group-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "node_group_policy" {
  for_each = {
    worker = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
    cni    = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
    registry = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  }

  policy_arn = each.value
  role       = aws_iam_role.node_group.name
}

# EKS Node Group
resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.cluster_name}-workers"
  node_role_arn   = aws_iam_role.node_group.arn
  subnet_ids      = aws_subnet.private[*].id

  instance_types = ["t3.medium"]
  ami_type       = "AL2_x86_64"
  capacity_type  = "ON_DEMAND"

  scaling_config {
    desired_size = 2
    max_size     = 10
    min_size     = 1
  }

  update_config {
    max_unavailable = 1
  }

  depends_on = [
    aws_iam_role_policy_attachment.node_group_policy
  ]

  tags = {
    Environment = var.environment
  }
}

# Outputs
output "cluster_id" {
  description = "EKS cluster ID"
  value       = aws_eks_cluster.main.id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.main.arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}
```

### Helm Charts for Application Deployment

#### **Complete Helm Chart Structure**

```
myapp-chart/
├── Chart.yaml
├── values.yaml
├── values-staging.yaml
├── values-production.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   ├── serviceaccount.yaml
│   └── _helpers.tpl
└── charts/
    └── postgresql/
```

**Chart.yaml:**

```yaml
apiVersion: v2
name: myapp
description: A Helm chart for MyApp
type: application
version: 0.1.0
appVersion: "1.0.0"

dependencies:
  - name: postgresql
    version: "11.x.x"
    repository: "https://charts.bitnami.com/bitnami"
    condition: postgresql.enabled
  - name: redis
    version: "17.x.x"
    repository: "https://charts.bitnami.com/bitnami"
    condition: redis.enabled

maintainers:
  - name: DevOps Team
    email: devops@example.com
```

**values.yaml:**

```yaml
# Default values for myapp
replicaCount: 2

image:
  repository: myregistry/myapp
  pullPolicy: IfNotPresent
  tag: ""

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations: {}

podSecurityContext:
  fsGroup: 1001

securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1001

service:
  type: ClusterIP
  port: 80
  targetPort: 3000

ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
    - host: chart-example.local
      paths:
        - path: /
          pathType: ImplementationSpecific
  tls: []

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity: {}

podDisruptionBudget:
  enabled: true
  minAvailable: 1

config:
  app:
    env: production
    logLevel: INFO
  database:
    host: ""
    port: 5432
    name: myapp
  redis:
    host: ""
    port: 6379

secrets:
  database:
    password: ""
  api:
    key: ""

# Dependencies
postgresql:
  enabled: true
  auth:
    postgresPassword: "postgres123"
    database: myapp
  primary:
    persistence:
      enabled: true
      size: 10Gi

redis:
  enabled: true
  auth:
    enabled: false
  master:
    persistence:
      enabled: true
      size: 5Gi
```

**templates/deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "myapp.fullname" . }}
  labels:
    {{- include "myapp.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "myapp.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
        checksum/secret: {{ include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}
        {{- with .Values.podAnnotations }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
      labels:
        {{- include "myapp.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "myapp.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
        - name: {{ .Chart.Name }}
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.service.targetPort }}
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 2
          env:
            - name: DATABASE_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: {{ include "myapp.fullname" . }}-secrets
                  key: database-password
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ include "myapp.fullname" . }}-secrets
                  key: api-key
          envFrom:
            - configMapRef:
                name: {{ include "myapp.fullname" . }}-config
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: cache
              mountPath: /app/cache
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
      volumes:
        - name: tmp
          emptyDir: {}
        - name: cache
          emptyDir: {}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

**Deployment Commands:**

```bash
# Install chart
helm install myapp ./myapp-chart -f values-staging.yaml

# Upgrade chart
helm upgrade myapp ./myapp-chart -f values-staging.yaml

# Rollback to previous version
helm rollback myapp 1

# Template and validate
helm template myapp ./myapp-chart -f values-staging.yaml | kubectl apply --dry-run=client -f -

# Package chart
helm package myapp-chart

# Push to registry
helm push myapp-0.1.0.tgz oci://registry.example.com/helm-charts
```

---

## Common Confusions & Mistakes

### **1. "DevOps = Dev + Ops Team"**

**Confusion:** Thinking DevOps is just combining development and operations teams
**Reality:** DevOps is a cultural and organizational shift focusing on collaboration and shared responsibility
**Solution:** Focus on breaking down silos, shared goals, and collaborative processes rather than just team restructuring

### **2. "Container = Virtual Machine"**

**Confusion:** Treating containers like lightweight VMs with full operating systems
**Reality:** Containers share the host OS kernel and focus on application isolation, not hardware virtualization
**Solution:** Understand container isolation, use containers for application packaging, and VMs for full OS isolation

### **3. "Kubernetes is Always the Answer"**

**Confusion:** Using Kubernetes for every container deployment without considering complexity
**Reality:** Kubernetes adds operational overhead; simpler tools like Docker Compose may be sufficient for small deployments
**Solution:** Use appropriate tools for your scale: Docker Compose for development, ECS for simple production, Kubernetes for complex deployments

### **4. "CI/CD Pipeline = Code Deployment"**

**Confusion:** Thinking CI/CD is only about automated code deployment
**Reality:** CI/CD encompasses testing, security scanning, compliance checks, and automated feedback loops
**Solution:** Build comprehensive pipelines with testing, security, compliance, and rollback capabilities

### **5. "Infrastructure as Code = Configuration Management"**

**Confusion:** Using IaC tools (Terraform, CloudFormation) like traditional configuration management (Ansible, Chef)
**Reality:** IaC focuses on provisioning infrastructure while configuration management handles server configuration
**Solution:** Use both: Terraform for infrastructure provisioning, Ansible for server configuration and application deployment

### **6. "Monitoring = Alerting"**

**Confusion:** Setting up monitoring only for critical alerts
**Reality:** Effective monitoring includes metrics, logging, tracing, and proactive observability
**Solution:** Implement comprehensive observability with metrics, logs, traces, and business KPIs

### **7. "Security as an Afterthought"**

**Confusion:** Adding security checks at the end of the deployment pipeline
**Reality:** Security should be integrated throughout the development and deployment lifecycle
**Solution:** Implement DevSecOps: security scanning in CI/CD, infrastructure security scanning, and security policies as code

### **8. "Cloud Means No Infrastructure Management"**

**Confusion:** Believing cloud eliminates the need for infrastructure management
**Reality:** Cloud requires even more infrastructure management, just with different tools and responsibilities
**Solution:** Learn cloud-specific services, management tools, and cost optimization while maintaining infrastructure expertise

---

## Micro-Quiz (80% Required for Mastery)

**Question 1:** What's the main advantage of containers over virtual machines?
a) Better security isolation
b) Faster startup and lower resource overhead
c) Complete OS independence
d) Easier backup and restore

**Question 2:** In Kubernetes, what does a "Deployment" resource provide?
a) Network load balancing
b) Application deployment and scaling management
c) Persistent storage management
d) Service discovery

**Question 3:** What's the primary purpose of CI/CD pipelines?
a) To deploy code to production faster
b) To automate testing, integration, and deployment with feedback loops
c) To replace manual testing
d) To reduce development time

**Question 4:** Which tool is best for defining infrastructure as declarative configuration?
a) Ansible
b) Chef
c) Terraform
d) Docker

**Question 5:** What should you monitor first in a production system?
a) CPU and memory usage
b) Business metrics and user experience
c) Network bandwidth
d) Disk space

**Answer Key:** 1-b, 2-b, 3-b, 4-c, 5-b

---

## Reflection Prompts

**1. DevOps Culture Assessment:**
Think about a team you've worked with. Where do you see gaps in collaboration between development and operations? What specific practices could bridge these gaps? How would you measure success?

**2. Container Strategy Decision:**
You're deploying a web application that serves 1000 users now but expects growth to 100,000. What containerization strategy would you choose? How would your approach change for 1 million users?

**3. CI/CD Pipeline Design:**
You need to deploy a critical financial application with high security requirements. What security checks would you include in your CI/CD pipeline? How would you ensure compliance and audit trails?

**4. Infrastructure Scalability Planning:**
Your current application runs on 3 servers. Plan how you would scale to 1000 servers. What tools, processes, and strategies would you implement? Consider cost, management complexity, and reliability.

---

## Mini Sprint Project (20-40 minutes)

**Project:** Containerize and Deploy a Simple Web Application

**Scenario:** You have a simple Python Flask web application that needs to be containerized and deployed.

**Requirements:**

1. **Application Details:**
   - Simple Flask app with a few endpoints
   - Uses SQLite database
   - Serves HTML pages and APIs
   - Currently runs on localhost:5000

2. **Containerization Requirements:**
   - Create a Dockerfile for the application
   - Use multi-stage build for optimization
   - Include proper environment variables
   - Use non-root user for security

3. **Deployment Requirements:**
   - Deploy using Docker Compose
   - Include environment configuration
   - Set up basic health checks
   - Configure logging

**Deliverables:**

1. **Dockerfile** - Optimized container definition
2. **docker-compose.yml** - Complete deployment configuration
3. **Environment Configuration** - .env file with settings
4. **Health Check Strategy** - How to verify application is working
5. **Deployment Commands** - Step-by-step deployment instructions

**Success Criteria:**

- Application builds and runs successfully in containers
- Proper container security practices (non-root user, minimal base image)
- Easy deployment and management with Docker Compose
- Clear documentation for development and production
- Working health checks and logging

---

## Full Project Extension (6-10 hours)

**Project:** Build a Complete CI/CD Pipeline with Kubernetes Deployment

**Scenario:** Create a full DevOps pipeline for a microservices e-commerce application with automated testing, security scanning, and Kubernetes deployment.

**Extended Requirements:**

**1. Application Architecture (1-2 hours)**

- Multi-service application (user service, product service, order service)
- Each service with its own database
- API gateway for routing
- Frontend web application

**2. Containerization Strategy (1-2 hours)**

- Individual Dockerfiles for each service
- Optimized multi-stage builds
- Consistent base images and patterns
- Security best practices (non-root users, minimal images)

**3. CI/CD Pipeline (2-3 hours)**

- GitHub Actions or GitLab CI pipeline
- Automated testing (unit, integration, security)
- Code quality checks (linting, formatting)
- Security scanning (dependency vulnerabilities)
- Automated deployment to staging and production

**4. Kubernetes Deployment (2-3 hours)**

- Kubernetes manifests for all services
- ConfigMaps and Secrets for configuration
- Ingress for routing
- Horizontal Pod Autoscaling (HPA)
- Service mesh configuration (Istio optional)

**5. Monitoring and Observability (1-2 hours)**

- Prometheus for metrics collection
- Grafana for visualization
- ELK stack for logging
- Jaeger for distributed tracing
- Alerting rules and notifications

**6. Infrastructure as Code (1-2 hours)**

- Terraform for cloud infrastructure
- Environment-specific configurations
- State management and remote state
- Automated infrastructure provisioning

**Deliverables:**

1. **Complete source code repository** with all services
2. **Docker and Kubernetes configurations**
3. **CI/CD pipeline configuration** with security scanning
4. **Infrastructure as Code templates**
5. **Monitoring and observability setup**
6. **Deployment documentation** and runbooks
7. **Security configuration** and compliance checks
8. **Performance testing results** and scaling analysis

**Success Criteria:**

- Full automated pipeline from code commit to production
- Comprehensive security scanning and compliance
- Production-ready Kubernetes deployment
- Complete monitoring and observability stack
- Infrastructure automation with IaC
- Clear documentation for operations and maintenance
- Successful load testing and performance validation

**Bonus Challenges:**

- Multi-environment promotion (dev → staging → production)
- Blue-green or canary deployment strategies
- Multi-cluster Kubernetes deployment
- Disaster recovery and backup automation
- Cost optimization and resource management
- Compliance automation (SOC2, PCI DSS)

---

## Conclusion

DevOps and containerization represent a fundamental shift in how we develop, deploy, and operate software systems. Success requires mastering not just the tools, but the cultural and philosophical changes that enable teams to deliver software faster, more reliably, and with higher quality.

**Key DevOps Principles:**

- **Automation first:** Eliminate manual, error-prone processes
- **Measure everything:** Use data to drive improvement decisions
- **Fail fast, learn faster:** Embrace controlled failure as learning
- **Collaboration over handoffs:** Break down silos between teams
- **Security as code:** Build security into every step of the process

**Your DevOps Journey:**

**Foundation (Months 1-3):**

- Master Docker containerization and compose
- Learn basic CI/CD pipeline concepts and implementation
- Understand infrastructure as code principles
- Practice with cloud platforms and basic deployments

**Proficiency (Months 4-8):**

- Implement Kubernetes orchestration and advanced patterns
- Build complex CI/CD pipelines with security integration
- Master infrastructure as code with Terraform or CloudFormation
- Implement monitoring and observability solutions

**Expertise (Months 9-18):**

- Design and implement enterprise-scale DevOps platforms
- Lead DevOps transformation initiatives
- Architect secure, scalable, and resilient systems
- Mentor teams and drive DevOps culture change

**Mastery (18+ Months):**

- Shape DevOps practices and standards across organizations
- Drive innovation in development and deployment practices
- Lead complex multi-team, multi-environment transformations
- Contribute to open-source DevOps tools and communities

Remember: DevOps is not just about tools and automation—it's about creating a culture of collaboration, learning, and continuous improvement. The most successful DevOps implementations focus as much on people and processes as they do on technology.

---

_"DevOps is not a goal, but a never-ending process of continual improvement."_ - Jez Humble

## 🤔 Common Confusions

### DevOps Fundamentals

1. **DevOps vs Agile vs CI/CD confusion**: DevOps is a culture and practice, Agile is a methodology, CI/CD is specific technical implementation
2. **Container vs virtual machine differences**: Containers share host OS kernel, VMs include complete OS - different isolation and resource usage
3. **Infrastructure as Code benefits**: Version control, repeatability, automation vs manual configuration management
4. **Microservices vs monolith in DevOps context**: Independent deployment vs single deployment unit - different CI/CD strategies

### Continuous Integration/Deployment

5. **CI vs CD vs CD confusion**: CI (Integration), CD (Continuous Delivery), CD (Continuous Deployment) - three different concepts
6. **Pipeline stages misunderstanding**: Build, test, security scan, deploy phases - each serves different quality gates
7. **Environment promotion confusion**: Development → Staging → Production promotions with different validation requirements
8. **Rollback vs roll-forward strategies**: Different approaches to handling failed deployments and production issues

### Container Orchestration

9. **Kubernetes vs Docker comparison**: Docker is containerization, Kubernetes is orchestration - different layers of abstraction
10. **Service vs deployment in Kubernetes**: Services provide networking, deployments manage application lifecycle
11. **Horizontal vs vertical pod autoscaling**: Pod scaling vs resource scaling in Kubernetes clusters
12. **ConfigMap vs Secret confusion**: Configuration data vs sensitive data in Kubernetes

### Monitoring & Observability

13. **Metrics vs logs vs traces**: Different types of observability data with different use cases
14. **SLO vs SLA vs SLI differences**: Objectives, Agreements, Indicators - different commitment and measurement levels
15. **Alerting vs notification confusion**: Data collection and analysis vs communication to humans
16. **Log aggregation vs log analysis**: Collecting logs vs extracting insights from log data

---

## 📝 Micro-Quiz: DevOps & Containerization Fundamentals

**Instructions**: Answer these 6 questions. Need 5/6 (83%) to pass.

1. **Question**: What's the main difference between Continuous Integration and Continuous Deployment?
   - a) CI builds code, CD deploys applications
   - b) CI integrates code changes, CD automatically deploys to production
   - c) They are the same thing
   - d) CI is for testing, CD is for monitoring

2. **Question**: In Docker, what's the key difference between an image and a container?
   - a) Images are temporary, containers are permanent
   - b) Images are blueprints, containers are running instances
   - c) Images are for Windows, containers are for Linux
   - d) No difference, they are identical

3. **Question**: What's the primary purpose of Infrastructure as Code?
   - a) To write code instead of infrastructure
   - b) To manage infrastructure through code for consistency and automation
   - c) To replace all cloud services
   - d) To make infrastructure faster

4. **Question**: In Kubernetes, what's the difference between a Service and a Deployment?
   - a) Services are for networking, Deployments manage application lifecycle
   - b) Services are for databases, Deployments are for applications
   - c) Services are old, Deployments are new
   - d) No difference, they are interchangeable

5. **Question**: What's the main purpose of monitoring in DevOps?
   - a) To make systems look professional
   - b) To detect issues early and provide insights for improvement
   - c) To replace manual testing
   - d) To increase system performance

6. **Question**: What's the difference between horizontal and vertical scaling?
   - a) Horizontal scales servers, vertical scales applications
   - b) Horizontal adds more machines, vertical increases machine resources
   - c) Horizontal is for databases, vertical is for applications
   - d) There's no significant difference

**Answer Key**: 1-b, 2-b, 3-b, 4-a, 5-b, 6-b

---

## 🎯 Reflection Prompts

### 1. Development Process Evolution

Think about your current development workflow. How do you currently move code from your computer to production? What manual steps are involved? What could go wrong at each step? This reflection helps you understand why DevOps practices are essential and how they solve real problems in software delivery.

### 2. Team Collaboration Analysis

Consider a team project you've worked on. How did you coordinate between different team members? How did you handle conflicting code changes? How did you know if the system was working correctly? This thinking helps you understand how DevOps practices improve team collaboration and system reliability.

### 3. Career Path Planning

Consider the DevOps learning progression described in this chapter. Which skills interest you most? Do you prefer the coding aspect (infrastructure as code, automation) or the operational side (monitoring, incident response)? This reflection helps you understand the different career paths within DevOps and what to focus on learning.

---

## 🚀 Mini Sprint Project: CI/CD Pipeline Builder

**Time Estimate**: 3-4 hours  
**Difficulty**: Intermediate

### Project Overview

Create an interactive platform that helps users build and visualize CI/CD pipelines with drag-and-drop interface and real deployment simulation.

### Core Features

1. **Visual Pipeline Builder**
   - Drag-and-drop pipeline stage creation
   - Pre-built stages: build, test, security scan, deploy, rollback
   - Real-time pipeline validation and error checking
   - Pipeline template library for common use cases

2. **Deployment Simulation**
   - Simulated environment for testing pipelines
   - Mock deployment to different environments (dev, staging, prod)
   - Error injection and failure scenario testing
   - Performance metrics and timing analysis

3. **Integration Hub**
   - **Version Control**: GitHub, GitLab, Bitbucket integration
   - **Container Registry**: Docker Hub, AWS ECR, Google Container Registry
   - **Cloud Platforms**: AWS, Azure, Google Cloud deployment options
   - **Monitoring**: Integration with monitoring tools and alerting

4. **Security & Quality Gates**
   - Security scanning integration (SAST, DAST, dependency scanning)
   - Code quality checks and coverage requirements
   - Compliance validation and policy enforcement
   - Automated security vulnerability assessment

### Technical Requirements

- **Frontend**: React/Vue.js with interactive pipeline designer
- **Backend**: Node.js/Python for pipeline execution and simulation
- **Database**: PostgreSQL for pipeline definitions and execution history
- **Container**: Docker for isolated pipeline execution
- **APIs**: Integration with version control and cloud provider APIs

### Success Criteria

- [ ] Pipeline builder provides intuitive design experience
- [ ] Deployment simulation accurately represents real-world scenarios
- [ ] Integration hub connects to major development tools
- [ ] Security gates provide meaningful quality assurance
- [ ] Error handling provides educational feedback

### Extension Ideas

- Add machine learning pipeline templates
- Include multi-cloud deployment strategies
- Implement pipeline analytics and optimization
- Add collaborative pipeline design features

---

## 🌟 Full Project Extension: Enterprise DevOps Platform & SRE Toolchain

**Time Estimate**: 20-25 hours  
**Difficulty**: Advanced

### Project Overview

Build a comprehensive enterprise DevOps platform that provides end-to-end application lifecycle management, advanced observability, automated incident response, and SRE (Site Reliability Engineering) capabilities.

### Advanced Features

1. **Unified DevOps Platform**
   - **Application Lifecycle Management**: From code commit to production deployment
   - **Multi-Cloud Deployment**: Support for AWS, Azure, Google Cloud, on-premises
   - **Environment Orchestration**: Dev, staging, production with automated promotion
   - **Release Management**: Blue-green, canary, and rolling deployment strategies

2. **Advanced CI/CD Automation**
   - **Intelligent Pipeline Triggers**: Event-based, schedule-based, and manual triggers
   - **Dynamic Scaling**: Auto-scaling build agents based on demand
   - **Security Integration**: DevSecOps with automated security scanning and compliance
   - **Quality Gates**: Automated code quality, performance, and security validation

3. **Comprehensive Observability Stack**
   - **Metrics Collection**: Application, infrastructure, and business metrics
   - **Distributed Tracing**: Request flow analysis across microservices
   - **Log Aggregation**: Centralized logging with intelligent analysis
   - **APM Integration**: Application performance monitoring and profiling

4. **Site Reliability Engineering Suite**
   - **Automated Incident Response**: Self-healing systems and automated remediation
   - **SLO Management**: Service level objective tracking and alerting
   - **Capacity Planning**: Predictive scaling and resource optimization
   - **Chaos Engineering**: Automated testing of system resilience

5. **Governance & Compliance**
   - **Policy as Code**: Automated compliance checking and enforcement
   - **Audit Trail**: Comprehensive tracking of all platform activities
   - **Cost Optimization**: Automated resource optimization and cost tracking
   - **Risk Assessment**: Continuous security and compliance monitoring

### Technical Architecture

```
Enterprise DevOps Platform
├── Application Lifecycle/
│   ├── Multi-cloud deployment
│   ├── Environment orchestration
│   ├── Release management
│   └── Lifecycle automation
├── CI/CD Engine/
│   ├── Intelligent triggers
│   ├── Dynamic scaling
│   ├── Security integration
│   └── Quality gates
├── Observability Stack/
│   ├── Metrics collection
│   ├── Distributed tracing
│   ├── Log aggregation
│   └── APM integration
├── SRE Tools/
│   ├── Incident response
│   ├── SLO management
│   ├── Capacity planning
│   └── Chaos engineering
└── Governance/
    ├── Policy as code
    ├── Audit trail
    ├── Cost optimization
    └── Risk assessment
```

### Advanced Implementation Requirements

- **Enterprise Scale**: Support for thousands of applications and teams
- **High Availability**: 99.99% uptime with disaster recovery capabilities
- **Security First**: Zero-trust architecture with comprehensive security controls
- **Integration Ecosystem**: Deep integration with existing enterprise tools
- **AI-Powered Automation**: Machine learning for optimization and prediction

### Learning Outcomes

- Mastery of enterprise DevOps practices and platform engineering
- Advanced knowledge of site reliability engineering and SRE practices
- Expertise in multi-cloud deployment and infrastructure management
- Skills in building automated, self-healing systems
- Understanding of enterprise governance and compliance requirements

### Success Metrics

- [ ] Platform successfully manages application lifecycle across enterprise environments
- [ ] CI/CD automation reduces deployment time and increases reliability
- [ ] Observability stack provides comprehensive system visibility
- [ ] SRE features improve system reliability and reduce incident response time
- [ ] Governance features ensure compliance and cost optimization
- [ ] Platform performance and reliability meet enterprise requirements

This comprehensive platform will prepare you for senior DevOps engineer roles, SRE leadership positions, and platform engineering management, providing the skills and experience needed to design and operate enterprise-scale DevOps platforms.
