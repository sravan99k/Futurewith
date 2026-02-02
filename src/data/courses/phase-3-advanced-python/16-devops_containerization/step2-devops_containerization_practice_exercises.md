# DevOps & Containerization - Practice Exercises

## Table of Contents

1. [Container Fundamentals and Docker Mastery](#container-fundamentals-and-docker-mastery)
2. [Kubernetes Orchestration and Management](#kubernetes-orchestration-and-management)
3. [CI/CD Pipeline Implementation](#cicd-pipeline-implementation)
4. [Infrastructure as Code (IaC)](#infrastructure-as-code-iac)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Security and Compliance Automation](#security-and-compliance-automation)
7. [Multi-Cloud and Hybrid Deployments](#multi-cloud-and-hybrid-deployments)
8. [Automated Testing in DevOps](#automated-testing-in-devops)
9. [Configuration Management](#configuration-management)
10. [Disaster Recovery and Backup Automation](#disaster-recovery-and-backup-automation)

## Practice Exercise 1: Container Fundamentals and Docker Mastery

### Objective

Build comprehensive containerization skills from basic Docker concepts to advanced container orchestration patterns.

### Exercise Details

**Time Required**: 2-3 weeks with progressive complexity
**Difficulty**: Beginner to Advanced

### Week 1: Docker Fundamentals and Optimization

#### Project: Multi-Service Application Containerization

**Scenario**: Containerize a complete e-commerce application with multiple services

```dockerfile
# Advanced Dockerfile for Node.js API Service

# Multi-stage build for production optimization
FROM node:18-alpine AS builder

# Install build dependencies
RUN apk add --no-cache python3 make g++

WORKDIR /app

# Copy package files first for better cache utilization
COPY package*.json ./
COPY yarn.lock ./

# Install dependencies
RUN yarn install --frozen-lockfile --production=false

# Copy source code
COPY . .

# Build the application
RUN yarn build && \
    yarn install --frozen-lockfile --production=true --ignore-scripts --prefer-offline

# Production stage
FROM node:18-alpine AS production

# Create app user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Install runtime dependencies only
RUN apk add --no-cache \
    tini \
    curl \
    && rm -rf /var/cache/apk/*

WORKDIR /app

# Copy built application from builder stage
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./package.json

# Set up health check
COPY scripts/health-check.sh /usr/local/bin/health-check.sh
RUN chmod +x /usr/local/bin/health-check.sh

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Use tini as init system
ENTRYPOINT ["/sbin/tini", "--"]

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/health-check.sh"]

# Start application
CMD ["node", "dist/server.js"]
```

```dockerfile
# Optimized Dockerfile for Frontend (React)

FROM node:18-alpine AS dependencies

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY yarn.lock ./

# Install dependencies with cache mount
RUN --mount=type=cache,target=/root/.yarn \
    yarn install --frozen-lockfile

# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy dependencies from previous stage
COPY --from=dependencies /app/node_modules ./node_modules
COPY . .

# Set build environment variables
ARG REACT_APP_API_URL
ARG REACT_APP_VERSION
ENV REACT_APP_API_URL=$REACT_APP_API_URL
ENV REACT_APP_VERSION=$REACT_APP_VERSION

# Build application
RUN yarn build

# Production stage with NGINX
FROM nginx:alpine AS production

# Install security updates
RUN apk upgrade --no-cache

# Copy nginx configuration
COPY nginx/default.conf /etc/nginx/conf.d/default.conf
COPY nginx/nginx.conf /etc/nginx/nginx.conf

# Copy built application
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom error pages
COPY nginx/error-pages /usr/share/nginx/html/error-pages

# Set proper permissions
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html

# Create nginx user for better security
RUN adduser -D -s /bin/sh nginx || true

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Expose port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

#### Advanced Docker Compose Configuration

```yaml
# docker-compose.yml - Production-ready multi-service setup

version: "3.8"

services:
  # Frontend Service
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      args:
        - REACT_APP_API_URL=http://localhost:3001/api
        - REACT_APP_VERSION=${APP_VERSION:-latest}
    ports:
      - "3000:80"
    networks:
      - web-network
    depends_on:
      - api
    restart: unless-stopped
    environment:
      - NODE_ENV=production
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.frontend.rule=Host(`localhost`)"
    volumes:
      - ./nginx/logs:/var/log/nginx
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: "0.5"
        reservations:
          memory: 128M
          cpus: "0.25"

  # API Service
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    ports:
      - "3001:3000"
    networks:
      - web-network
      - database-network
    depends_on:
      database:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://app_user:${DB_PASSWORD}@database:5432/app_db
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=${JWT_SECRET}
    volumes:
      - ./api/uploads:/app/uploads
      - ./api/logs:/app/logs
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: "1.0"
        reservations:
          memory: 256M
          cpus: "0.5"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Database Service
  database:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=app_db
      - POSTGRES_USER=app_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d
      - ./database/backups:/backups
    networks:
      - database-network
    restart: unless-stopped
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app_user -d app_db"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: "1.0"
        reservations:
          memory: 512M
          cpus: "0.5"
    command: >
      postgres
      -c shared_preload_libraries=pg_stat_statements
      -c pg_stat_statements.max=10000
      -c pg_stat_statements.track=all
      -c max_connections=200
      -c shared_buffers=256MB
      -c effective_cache_size=1GB
      -c work_mem=4MB

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --maxmemory 256mb
      --maxmemory-policy allkeys-lru
      --save 900 1
      --save 300 10
      --save 60 10000
    volumes:
      - redis_data:/data
      - ./redis/redis.conf:/usr/local/etc/redis/redis.conf
    networks:
      - database-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: "0.5"
        reservations:
          memory: 128M
          cpus: "0.25"

  # Monitoring - Prometheus
  prometheus:
    image: prom/prometheus:latest
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--web.console.libraries=/etc/prometheus/console_libraries"
      - "--web.console.templates=/etc/prometheus/consoles"
      - "--storage.tsdb.retention.time=200h"
      - "--web.enable-lifecycle"
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    networks:
      - monitoring-network
    ports:
      - "9090:9090"
    restart: unless-stopped

  # Monitoring - Grafana
  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    networks:
      - monitoring-network
    ports:
      - "3002:3000"
    restart: unless-stopped
    depends_on:
      - prometheus

  # Log Aggregation
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - logging-network
    ports:
      - "9200:9200"
    restart: unless-stopped

  # Backup Service
  backup:
    build:
      context: ./backup
      dockerfile: Dockerfile
    environment:
      - DB_HOST=database
      - DB_NAME=app_db
      - DB_USER=app_user
      - DB_PASSWORD=${DB_PASSWORD}
      - BACKUP_RETENTION_DAYS=30
      - S3_BUCKET=${BACKUP_S3_BUCKET}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    volumes:
      - ./database/backups:/backups
      - backup_scripts:/scripts
    networks:
      - database-network
    depends_on:
      - database
    restart: unless-stopped
    profiles:
      - backup

networks:
  web-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
  database-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.21.0.0/16
  monitoring-network:
    driver: bridge
  logging-network:
    driver: bridge

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
  elasticsearch_data:
    driver: local
  backup_scripts:
    driver: local

# Environment configuration file
x-logging: &default-logging
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

#### Container Security Hardening

```bash
#!/bin/bash
# container-security-setup.sh

# Docker security best practices implementation

echo "Setting up Docker security configurations..."

# 1. Configure Docker daemon security
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "live-restore": true,
  "userland-proxy": false,
  "no-new-privileges": true,
  "seccomp-profile": "/etc/docker/seccomp/default.json",
  "apparmor-profile": "docker-default",
  "selinux-enabled": true,
  "userns-remap": "dockremap"
}
EOF

# 2. Create Docker user namespace mapping
sudo useradd -r -s /bin/false -M -d /var/lib/docker dockremap
echo 'dockremap:165536:65536' | sudo tee -a /etc/subuid
echo 'dockremap:165536:65536' | sudo tee -a /etc/subgid

# 3. Set up AppArmor profile for Docker
sudo tee /etc/apparmor.d/docker-containers > /dev/null <<EOF
#include <tunables/global>

profile docker-containers flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>

  # Deny dangerous capabilities
  deny capability sys_admin,
  deny capability sys_module,
  deny capability sys_time,

  # Allow necessary capabilities
  capability net_admin,
  capability net_raw,
  capability setuid,
  capability setgid,
  capability chown,
  capability dac_override,
  capability fowner,
  capability kill,

  # File system access
  /etc/hosts r,
  /etc/hostname r,
  /etc/resolv.conf r,

  # Deny sensitive mounts
  deny /proc/sys/kernel/** rwklx,
  deny /sys/** rwklx,
}
EOF

sudo apparmor_parser -r /etc/apparmor.d/docker-containers

# 4. Configure seccomp profile
sudo mkdir -p /etc/docker/seccomp
curl -o /etc/docker/seccomp/default.json \
  https://raw.githubusercontent.com/moby/moby/master/profiles/seccomp/default.json

# 5. Set up Docker content trust
export DOCKER_CONTENT_TRUST=1
export DOCKER_CONTENT_TRUST_SERVER=https://notary.docker.io

# 6. Configure log rotation
sudo tee /etc/logrotate.d/docker-containers > /dev/null <<EOF
/var/lib/docker/containers/*/*-json.log {
  daily
  rotate 7
  missingok
  notifempty
  compress
  copytruncate
}
EOF

# 7. Set up Docker Bench Security
git clone https://github.com/docker/docker-bench-security.git
cd docker-bench-security
sudo sh docker-bench-security.sh

echo "Docker security configuration completed!"
echo "Please restart Docker daemon: sudo systemctl restart docker"
```

### Week 2: Advanced Container Patterns

#### Multi-Architecture Container Builds

```dockerfile
# Dockerfile.multiarch - Multi-architecture build

FROM --platform=$BUILDPLATFORM golang:1.19-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

WORKDIR /src

# Cache dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy source
COPY . .

# Build for target platform
ARG TARGETOS TARGETARCH
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH \
    go build -ldflags="-w -s" -o /app/server ./cmd/server

# Final stage
FROM scratch

# Copy timezone data
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Copy CA certificates
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy binary
COPY --from=builder /app/server /server

# Create non-root user
USER 65534:65534

EXPOSE 8080

ENTRYPOINT ["/server"]
```

```yaml
# docker-buildx-setup.yml - Build automation for multiple architectures

version: "3.8"

services:
  multi-arch-builder:
    build:
      context: .
      dockerfile: Dockerfile.multiarch
      platforms:
        - linux/amd64
        - linux/arm64
        - linux/arm/v7
      tags:
        - myapp:latest
        - myapp:${VERSION}
      cache_from:
        - myapp:latest
      cache_to:
        - type=local,dest=/tmp/.buildx-cache
      target: production
```

```bash
#!/bin/bash
# multi-arch-build.sh

# Set up Docker Buildx for multi-architecture builds

echo "Setting up Docker Buildx for multi-architecture builds..."

# Create and use a new builder instance
docker buildx create --name multi-arch-builder --use --bootstrap

# Enable experimental features
export DOCKER_CLI_EXPERIMENTAL=enabled

# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64,linux/arm/v7 \
  --tag myregistry.com/myapp:latest \
  --tag myregistry.com/myapp:${VERSION} \
  --push \
  --cache-from type=registry,ref=myregistry.com/myapp:cache \
  --cache-to type=registry,ref=myregistry.com/myapp:cache,mode=max \
  .

# Inspect the manifest
docker buildx imagetools inspect myregistry.com/myapp:latest

echo "Multi-architecture build completed!"
```

#### Container Monitoring and Observability

```python
# Container metrics collector with Prometheus integration

import time
import psutil
import docker
import threading
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Dict, List
import logging

class ContainerMonitor:
    """Advanced container monitoring with custom metrics"""

    def __init__(self, docker_client=None):
        self.client = docker_client or docker.from_env()
        self.logger = logging.getLogger(__name__)

        # Prometheus metrics
        self.container_cpu_usage = Gauge(
            'container_cpu_usage_percent',
            'CPU usage percentage by container',
            ['container_name', 'container_id', 'image']
        )

        self.container_memory_usage = Gauge(
            'container_memory_usage_bytes',
            'Memory usage in bytes by container',
            ['container_name', 'container_id', 'image']
        )

        self.container_network_rx_bytes = Counter(
            'container_network_rx_bytes_total',
            'Total received bytes by container',
            ['container_name', 'container_id', 'interface']
        )

        self.container_network_tx_bytes = Counter(
            'container_network_tx_bytes_total',
            'Total transmitted bytes by container',
            ['container_name', 'container_id', 'interface']
        )

        self.container_disk_io_read = Counter(
            'container_disk_io_read_bytes_total',
            'Total disk read bytes by container',
            ['container_name', 'container_id']
        )

        self.container_disk_io_write = Counter(
            'container_disk_io_write_bytes_total',
            'Total disk write bytes by container',
            ['container_name', 'container_id']
        )

        self.container_uptime = Gauge(
            'container_uptime_seconds',
            'Container uptime in seconds',
            ['container_name', 'container_id', 'image']
        )

        self.container_restart_count = Counter(
            'container_restart_count_total',
            'Number of container restarts',
            ['container_name', 'container_id', 'image']
        )

        self.container_health_status = Gauge(
            'container_health_status',
            'Container health status (1=healthy, 0=unhealthy)',
            ['container_name', 'container_id', 'image']
        )

    def start_monitoring(self, interval: int = 30):
        """Start monitoring containers in background thread"""
        def monitor_loop():
            while True:
                try:
                    self._collect_metrics()
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Error collecting metrics: {e}")
                    time.sleep(5)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        self.logger.info(f"Started container monitoring (interval: {interval}s)")

    def _collect_metrics(self):
        """Collect metrics from all running containers"""
        try:
            containers = self.client.containers.list()

            for container in containers:
                self._collect_container_metrics(container)

        except Exception as e:
            self.logger.error(f"Error listing containers: {e}")

    def _collect_container_metrics(self, container):
        """Collect metrics for a specific container"""
        try:
            # Container metadata
            container_name = container.name
            container_id = container.short_id
            image_name = container.image.tags[0] if container.image.tags else "unknown"

            # Get container stats
            stats = container.stats(stream=False)

            # CPU metrics
            cpu_percent = self._calculate_cpu_percent(stats)
            self.container_cpu_usage.labels(
                container_name=container_name,
                container_id=container_id,
                image=image_name
            ).set(cpu_percent)

            # Memory metrics
            memory_usage = stats['memory_stats'].get('usage', 0)
            self.container_memory_usage.labels(
                container_name=container_name,
                container_id=container_id,
                image=image_name
            ).set(memory_usage)

            # Network metrics
            self._collect_network_metrics(container_name, container_id, stats)

            # Disk I/O metrics
            self._collect_disk_metrics(container_name, container_id, stats)

            # Container state metrics
            self._collect_state_metrics(container, container_name, container_id, image_name)

        except Exception as e:
            self.logger.error(f"Error collecting metrics for {container.name}: {e}")

    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU usage percentage"""
        try:
            cpu_stats = stats['cpu_stats']
            precpu_stats = stats['precpu_stats']

            cpu_delta = cpu_stats['cpu_usage']['total_usage'] - \
                       precpu_stats['cpu_usage']['total_usage']
            system_delta = cpu_stats['system_cpu_usage'] - \
                          precpu_stats['system_cpu_usage']

            if system_delta > 0:
                return (cpu_delta / system_delta) * \
                       len(cpu_stats['cpu_usage']['percpu_usage']) * 100.0

            return 0.0

        except (KeyError, ZeroDivisionError):
            return 0.0

    def _collect_network_metrics(self, container_name: str, container_id: str, stats: Dict):
        """Collect network metrics"""
        try:
            networks = stats.get('networks', {})

            for interface, net_stats in networks.items():
                # RX bytes
                rx_bytes = net_stats.get('rx_bytes', 0)
                self.container_network_rx_bytes.labels(
                    container_name=container_name,
                    container_id=container_id,
                    interface=interface
                )._value._value = rx_bytes

                # TX bytes
                tx_bytes = net_stats.get('tx_bytes', 0)
                self.container_network_tx_bytes.labels(
                    container_name=container_name,
                    container_id=container_id,
                    interface=interface
                )._value._value = tx_bytes

        except Exception as e:
            self.logger.debug(f"Error collecting network metrics: {e}")

    def _collect_disk_metrics(self, container_name: str, container_id: str, stats: Dict):
        """Collect disk I/O metrics"""
        try:
            blkio_stats = stats.get('blkio_stats', {})

            # Read bytes
            read_bytes = 0
            for item in blkio_stats.get('io_service_bytes_recursive', []):
                if item.get('op') == 'Read':
                    read_bytes += item.get('value', 0)

            self.container_disk_io_read.labels(
                container_name=container_name,
                container_id=container_id
            )._value._value = read_bytes

            # Write bytes
            write_bytes = 0
            for item in blkio_stats.get('io_service_bytes_recursive', []):
                if item.get('op') == 'Write':
                    write_bytes += item.get('value', 0)

            self.container_disk_io_write.labels(
                container_name=container_name,
                container_id=container_id
            )._value._value = write_bytes

        except Exception as e:
            self.logger.debug(f"Error collecting disk metrics: {e}")

    def _collect_state_metrics(self, container, container_name: str,
                              container_id: str, image_name: str):
        """Collect container state metrics"""
        try:
            # Uptime
            started_at = container.attrs['State']['StartedAt']
            import datetime
            start_time = datetime.datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            uptime = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()

            self.container_uptime.labels(
                container_name=container_name,
                container_id=container_id,
                image=image_name
            ).set(uptime)

            # Restart count
            restart_count = container.attrs['RestartCount']
            self.container_restart_count.labels(
                container_name=container_name,
                container_id=container_id,
                image=image_name
            )._value._value = restart_count

            # Health status
            health_status = 1 if container.status == 'running' else 0
            health_data = container.attrs.get('State', {}).get('Health')
            if health_data:
                health_status = 1 if health_data.get('Status') == 'healthy' else 0

            self.container_health_status.labels(
                container_name=container_name,
                container_id=container_id,
                image=image_name
            ).set(health_status)

        except Exception as e:
            self.logger.debug(f"Error collecting state metrics: {e}")

# Container log aggregation
class ContainerLogAggregator:
    """Aggregate and forward container logs"""

    def __init__(self, docker_client=None, elasticsearch_host=None):
        self.client = docker_client or docker.from_env()
        self.es_host = elasticsearch_host
        self.logger = logging.getLogger(__name__)

        if self.es_host:
            from elasticsearch import Elasticsearch
            self.es_client = Elasticsearch([self.es_host])

    def start_log_collection(self):
        """Start collecting logs from all containers"""
        containers = self.client.containers.list()

        for container in containers:
            threading.Thread(
                target=self._collect_container_logs,
                args=(container,),
                daemon=True
            ).start()

    def _collect_container_logs(self, container):
        """Collect logs from a specific container"""
        try:
            # Stream logs
            for log_line in container.logs(stream=True, follow=True):
                log_entry = {
                    'timestamp': time.time(),
                    'container_name': container.name,
                    'container_id': container.short_id,
                    'image': container.image.tags[0] if container.image.tags else "unknown",
                    'message': log_line.decode('utf-8').strip()
                }

                # Forward to Elasticsearch
                if self.es_client:
                    self._send_to_elasticsearch(log_entry)

                # Log to file
                self._log_to_file(log_entry)

        except Exception as e:
            self.logger.error(f"Error collecting logs for {container.name}: {e}")

    def _send_to_elasticsearch(self, log_entry):
        """Send log entry to Elasticsearch"""
        try:
            index_name = f"container-logs-{time.strftime('%Y.%m.%d')}"
            self.es_client.index(index=index_name, body=log_entry)
        except Exception as e:
            self.logger.error(f"Error sending to Elasticsearch: {e}")

    def _log_to_file(self, log_entry):
        """Log entry to file"""
        log_file = f"/var/log/containers/{log_entry['container_name']}.log"
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        with open(log_file, 'a') as f:
            f.write(f"{log_entry['timestamp']} {log_entry['message']}\n")

# Usage example
if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000)

    # Initialize monitoring
    monitor = ContainerMonitor()
    monitor.start_monitoring(interval=30)

    # Initialize log aggregation
    log_aggregator = ContainerLogAggregator(
        elasticsearch_host="http://localhost:9200"
    )
    log_aggregator.start_log_collection()

    # Keep running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down monitoring...")
```

---

## Practice Exercise 2: Kubernetes Orchestration and Management

### Objective

Master Kubernetes deployment, scaling, and management for production workloads.

### Exercise Details

**Time Required**: 3-4 weeks with comprehensive cluster management
**Difficulty**: Advanced

### Week 1: Kubernetes Cluster Setup and Basic Workloads

#### Production Kubernetes Cluster Setup

```bash
#!/bin/bash
# k8s-cluster-setup.sh - Production Kubernetes cluster setup

set -euo pipefail

# Variables
CLUSTER_NAME="production-cluster"
KUBERNETES_VERSION="1.28.0"
NODE_COUNT=3
NODE_SIZE="Standard_D4s_v3"
REGION="eastus"

echo "Setting up production Kubernetes cluster..."

# 1. Create resource group
az group create --name ${CLUSTER_NAME}-rg --location $REGION

# 2. Create AKS cluster with production configurations
az aks create \
  --resource-group ${CLUSTER_NAME}-rg \
  --name $CLUSTER_NAME \
  --kubernetes-version $KUBERNETES_VERSION \
  --node-count $NODE_COUNT \
  --node-vm-size $NODE_SIZE \
  --enable-addons monitoring,azure-policy,azure-keyvault-secrets-provider \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10 \
  --network-plugin azure \
  --network-policy azure \
  --service-cidr 10.0.0.0/16 \
  --dns-service-ip 10.0.0.10 \
  --docker-bridge-address 172.17.0.1/16 \
  --enable-managed-identity \
  --enable-aad \
  --enable-azure-rbac \
  --enable-private-cluster \
  --outbound-type loadBalancer \
  --load-balancer-sku standard \
  --vm-set-type VirtualMachineScaleSets \
  --zones 1 2 3

# 3. Get cluster credentials
az aks get-credentials --resource-group ${CLUSTER_NAME}-rg --name $CLUSTER_NAME

# 4. Install essential cluster components
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml

# 5. Install cert-manager for TLS certificates
kubectl apply -f https://github.com/jetstack/cert-manager/releases/latest/download/cert-manager.yaml

# 6. Install Prometheus operator for monitoring
kubectl create namespace monitoring
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set alertmanager.persistentVolume.size=10Gi \
  --set server.persistentVolume.size=50Gi

# 7. Install external-dns for automatic DNS management
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: external-dns
  namespace: kube-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: external-dns
rules:
- apiGroups: [""]
  resources: ["services","endpoints","pods"]
  verbs: ["get","watch","list"]
- apiGroups: ["extensions","networking.k8s.io"]
  resources: ["ingresses"]
  verbs: ["get","watch","list"]
- apiGroups: [""]
  resources: ["nodes"]
  verbs: ["list"]
EOF

echo "Kubernetes cluster setup completed!"
echo "Cluster endpoint: $(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')"
echo "Available nodes:"
kubectl get nodes
```

#### Advanced Deployment Manifests

```yaml
# namespace.yaml - Namespace with resource quotas and limits

apiVersion: v1
kind: Namespace
metadata:
  name: ecommerce-app
  labels:
    name: ecommerce-app
    environment: production
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ecommerce-quota
  namespace: ecommerce-app
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "10"
    services.loadbalancers: "2"
    count/deployments.apps: "20"
    count/services: "10"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: ecommerce-limits
  namespace: ecommerce-app
spec:
  limits:
    - default:
        cpu: 500m
        memory: 512Mi
      defaultRequest:
        cpu: 100m
        memory: 128Mi
      type: Container
    - max:
        cpu: "2"
        memory: 4Gi
      min:
        cpu: 50m
        memory: 64Mi
      type: Container
```

```yaml
# configmap.yaml - Application configuration

apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: ecommerce-app
data:
  app.properties: |
    server.port=8080
    server.servlet.context-path=/api

    # Database configuration
    spring.datasource.url=jdbc:postgresql://postgres-service:5432/ecommerce
    spring.datasource.username=app_user
    spring.jpa.hibernate.ddl-auto=validate
    spring.jpa.show-sql=false

    # Redis configuration
    spring.redis.host=redis-service
    spring.redis.port=6379
    spring.redis.timeout=2000ms

    # Logging configuration
    logging.level.com.company.ecommerce=INFO
    logging.level.org.springframework.security=DEBUG

    # Feature flags
    features.recommendation.enabled=true
    features.analytics.enabled=true
    features.cache.ttl=3600

  nginx.conf: |
    upstream backend {
        server api-service:8080 max_fails=3 fail_timeout=30s;
    }

    server {
        listen 80;
        server_name _;
        
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
        
        location /api {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 5s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
        
        location / {
            root /usr/share/nginx/html;
            try_files $uri $uri/ /index.html;
            expires 1h;
            add_header Cache-Control "public, immutable";
        }
    }
---
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: ecommerce-app
type: Opaque
data:
  database-password: cGFzc3dvcmQxMjM= # password123
  redis-password: cmVkaXNwYXNz # redispass
  jwt-secret: bXlqd3RzZWNyZXRrZXk= # myjwtsecretkey
  encryption-key: ZW5jcnlwdGlvbmtleTEyMzQ1Ng== # encryptionkey123456
```

```yaml
# deployment.yaml - Production-ready deployment

apiVersion: apps/v1
kind: Deployment
metadata:
  name: ecommerce-api
  namespace: ecommerce-app
  labels:
    app: ecommerce-api
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: ecommerce-api
  template:
    metadata:
      labels:
        app: ecommerce-api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/actuator/prometheus"
    spec:
      serviceAccountName: ecommerce-api
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
        - name: api
          image: myregistry.azurecr.io/ecommerce-api:v1.2.3
          imagePullPolicy: Always
          ports:
            - containerPort: 8080
              name: http
              protocol: TCP
          env:
            - name: SPRING_PROFILES_ACTIVE
              value: "production"
            - name: DATABASE_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: database-password
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: redis-password
            - name: JWT_SECRET
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: jwt-secret
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
            - name: NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
          volumeMounts:
            - name: config
              mountPath: /app/config
              readOnly: true
            - name: logs
              mountPath: /app/logs
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
              ephemeral-storage: "1Gi"
            limits:
              memory: "1Gi"
              cpu: "500m"
              ephemeral-storage: "2Gi"
          livenessProbe:
            httpGet:
              path: /actuator/health/liveness
              port: 8080
              scheme: HTTP
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3
            successThreshold: 1
          readinessProbe:
            httpGet:
              path: /actuator/health/readiness
              port: 8080
              scheme: HTTP
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
            successThreshold: 1
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            capabilities:
              drop:
                - ALL
        - name: log-forwarder
          image: fluent/fluent-bit:1.9
          volumeMounts:
            - name: logs
              mountPath: /app/logs
              readOnly: true
            - name: fluent-bit-config
              mountPath: /fluent-bit/etc
          resources:
            requests:
              memory: "64Mi"
              cpu: "50m"
            limits:
              memory: "128Mi"
              cpu: "100m"
      volumes:
        - name: config
          configMap:
            name: app-config
        - name: logs
          emptyDir: {}
        - name: fluent-bit-config
          configMap:
            name: fluent-bit-config
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values:
                        - ecommerce-api
                topologyKey: kubernetes.io/hostname
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: ecommerce-api
      terminationGracePeriodSeconds: 60
```

### Week 2: Advanced Kubernetes Features

#### Horizontal Pod Autoscaler with Custom Metrics

```yaml
# hpa-custom-metrics.yaml

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ecommerce-api-hpa
  namespace: ecommerce-app
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ecommerce-api
  minReplicas: 3
  maxReplicas: 50
  metrics:
    # CPU utilization
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    # Memory utilization
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    # Custom metric: requests per second
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
    # External metric: queue length
    - type: External
      external:
        metric:
          name: queue_length
          selector:
            matchLabels:
              queue: order-processing
        target:
          type: AverageValue
          averageValue: "10"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 20
          periodSeconds: 60
        - type: Pods
          value: 2
          periodSeconds: 60
      selectPolicy: Min
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
        - type: Pods
          value: 4
          periodSeconds: 15
      selectPolicy: Max
---
# Custom metrics server configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: adapter-config
  namespace: monitoring
data:
  config.yaml: |
    rules:
    - seriesQuery: 'http_requests_per_second{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)_per_second"
        as: "${1}_per_second"
      metricsQuery: 'avg(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
    - seriesQuery: 'rabbitmq_queue_messages{namespace!="",queue!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
      name:
        as: "queue_length"
      metricsQuery: 'sum(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
```

#### StatefulSet for Database Workloads

```yaml
# postgresql-statefulset.yaml

apiVersion: v1
kind: Service
metadata:
  name: postgres-headless
  namespace: ecommerce-app
  labels:
    app: postgres
spec:
  ports:
    - port: 5432
      name: postgres
  clusterIP: None
  selector:
    app: postgres
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: ecommerce-app
  labels:
    app: postgres
spec:
  ports:
    - port: 5432
      name: postgres
  selector:
    app: postgres
    role: primary
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: ecommerce-app
spec:
  serviceName: postgres-headless
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      securityContext:
        fsGroup: 999
      containers:
        - name: postgres
          image: postgres:15-alpine
          env:
            - name: POSTGRES_DB
              value: ecommerce
            - name: POSTGRES_USER
              value: app_user
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: database-password
            - name: POSTGRES_REPLICATION_MODE
              value: master
            - name: POSTGRES_REPLICATION_USER
              value: replicator
            - name: POSTGRES_REPLICATION_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: database-password
            - name: POD_IP
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
          ports:
            - containerPort: 5432
              name: postgres
          volumeMounts:
            - name: postgres-data
              mountPath: /var/lib/postgresql/data
            - name: postgres-config
              mountPath: /etc/postgresql/postgresql.conf
              subPath: postgresql.conf
            - name: postgres-config
              mountPath: /docker-entrypoint-initdb.d/init.sql
              subPath: init.sql
          livenessProbe:
            exec:
              command:
                - /bin/sh
                - -c
                - pg_isready -U app_user -d ecommerce
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 6
          readinessProbe:
            exec:
              command:
                - /bin/sh
                - -c
                - pg_isready -U app_user -d ecommerce
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
        - name: postgres-exporter
          image: prometheuscommunity/postgres-exporter:v0.11.1
          env:
            - name: DATA_SOURCE_NAME
              value: "postgresql://app_user:password123@localhost:5432/ecommerce?sslmode=disable"
          ports:
            - containerPort: 9187
              name: metrics
          resources:
            requests:
              memory: "64Mi"
              cpu: "50m"
            limits:
              memory: "128Mi"
              cpu: "100m"
      volumes:
        - name: postgres-config
          configMap:
            name: postgres-config
  volumeClaimTemplates:
    - metadata:
        name: postgres-data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 100Gi
```

#### Network Policies for Security

```yaml
# network-policies.yaml

# Deny all traffic by default
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: ecommerce-app
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
---
# Allow API to access database
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-to-database
  namespace: ecommerce-app
spec:
  podSelector:
    matchLabels:
      app: postgres
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: ecommerce-api
      ports:
        - protocol: TCP
          port: 5432
---
# Allow API to access Redis
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-to-redis
  namespace: ecommerce-app
spec:
  podSelector:
    matchLabels:
      app: redis
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: ecommerce-api
      ports:
        - protocol: TCP
          port: 6379
---
# Allow ingress traffic to API
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ingress-to-api
  namespace: ecommerce-app
spec:
  podSelector:
    matchLabels:
      app: ecommerce-api
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8080
---
# Allow egress to external services
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-external-egress
  namespace: ecommerce-app
spec:
  podSelector:
    matchLabels:
      app: ecommerce-api
  policyTypes:
    - Egress
  egress:
    # Allow DNS
    - to: []
      ports:
        - protocol: UDP
          port: 53
    # Allow HTTPS to external APIs
    - to: []
      ports:
        - protocol: TCP
          port: 443
    # Allow HTTP for health checks
    - to: []
      ports:
        - protocol: TCP
          port: 80
```

### Week 3-4: Advanced Kubernetes Operations

#### Custom Resource Definitions and Operators

```yaml
# custom-app-operator.yaml

apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: ecommerceapps.apps.company.com
spec:
  group: apps.company.com
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                replicas:
                  type: integer
                  minimum: 1
                  maximum: 100
                  default: 3
                version:
                  type: string
                  pattern: '^v\d+\.\d+\.\d+$'
                database:
                  type: object
                  properties:
                    enabled:
                      type: boolean
                      default: true
                    storage:
                      type: string
                      default: "10Gi"
                    replicas:
                      type: integer
                      default: 1
                cache:
                  type: object
                  properties:
                    enabled:
                      type: boolean
                      default: true
                    memory:
                      type: string
                      default: "256Mi"
                monitoring:
                  type: object
                  properties:
                    enabled:
                      type: boolean
                      default: true
                    retention:
                      type: string
                      default: "7d"
            status:
              type: object
              properties:
                phase:
                  type: string
                  enum: ["Pending", "Running", "Failed"]
                conditions:
                  type: array
                  items:
                    type: object
                    properties:
                      type:
                        type: string
                      status:
                        type: string
                      lastTransitionTime:
                        type: string
                      reason:
                        type: string
                      message:
                        type: string
                readyReplicas:
                  type: integer
                totalReplicas:
                  type: integer
      subresources:
        status: {}
        scale:
          specReplicasPath: .spec.replicas
          statusReplicasPath: .status.totalReplicas
          labelSelectorPath: .status.labelSelector
  scope: Namespaced
  names:
    plural: ecommerceapps
    singular: ecommerceapp
    kind: ECommerceApp
    shortNames:
      - eapp
```

```python
# ecommerce-operator.py - Custom Kubernetes Operator

import asyncio
import logging
from typing import Dict, Any
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
import yaml
import json

class ECommerceAppOperator:
    """Custom operator for ECommerceApp resources"""

    def __init__(self):
        # Load Kubernetes configuration
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()

        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.custom_api = client.CustomObjectsApi()

        self.logger = logging.getLogger(__name__)

        # Resource definitions
        self.group = "apps.company.com"
        self.version = "v1"
        self.plural = "ecommerceapps"

    async def start(self):
        """Start the operator event loop"""
        self.logger.info("Starting ECommerceApp operator...")

        # Watch for ECommerceApp resource changes
        w = watch.Watch()

        while True:
            try:
                stream = w.stream(
                    self.custom_api.list_cluster_custom_object,
                    group=self.group,
                    version=self.version,
                    plural=self.plural
                )

                for event in stream:
                    await self.handle_event(event)

            except Exception as e:
                self.logger.error(f"Error watching resources: {e}")
                await asyncio.sleep(5)

    async def handle_event(self, event: Dict[str, Any]):
        """Handle ECommerceApp resource events"""
        event_type = event['type']
        obj = event['object']

        name = obj['metadata']['name']
        namespace = obj['metadata']['namespace']
        spec = obj.get('spec', {})

        self.logger.info(f"Handling {event_type} event for {namespace}/{name}")

        try:
            if event_type in ['ADDED', 'MODIFIED']:
                await self.reconcile_ecommerce_app(namespace, name, spec)
            elif event_type == 'DELETED':
                await self.cleanup_ecommerce_app(namespace, name)

        except Exception as e:
            self.logger.error(f"Error handling event for {namespace}/{name}: {e}")
            await self.update_status(namespace, name, "Failed", str(e))

    async def reconcile_ecommerce_app(self, namespace: str, name: str, spec: Dict[str, Any]):
        """Reconcile ECommerceApp resource to desired state"""

        # Update status to Pending
        await self.update_status(namespace, name, "Pending", "Reconciling resources")

        try:
            # Create or update API deployment
            await self.reconcile_api_deployment(namespace, name, spec)

            # Create or update database if enabled
            if spec.get('database', {}).get('enabled', True):
                await self.reconcile_database(namespace, name, spec)

            # Create or update cache if enabled
            if spec.get('cache', {}).get('enabled', True):
                await self.reconcile_cache(namespace, name, spec)

            # Create or update monitoring if enabled
            if spec.get('monitoring', {}).get('enabled', True):
                await self.reconcile_monitoring(namespace, name, spec)

            # Create services and ingress
            await self.reconcile_services(namespace, name, spec)

            # Update status to Running
            await self.update_status(namespace, name, "Running", "All resources reconciled")

        except Exception as e:
            await self.update_status(namespace, name, "Failed", str(e))
            raise

    async def reconcile_api_deployment(self, namespace: str, name: str, spec: Dict[str, Any]):
        """Create or update API deployment"""

        replicas = spec.get('replicas', 3)
        version = spec.get('version', 'latest')

        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f"{name}-api",
                'namespace': namespace,
                'labels': {
                    'app': f"{name}-api",
                    'managed-by': 'ecommerce-operator'
                }
            },
            'spec': {
                'replicas': replicas,
                'selector': {
                    'matchLabels': {
                        'app': f"{name}-api"
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': f"{name}-api",
                            'version': version
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'api',
                            'image': f"myregistry.azurecr.io/{name}-api:{version}",
                            'ports': [{
                                'containerPort': 8080,
                                'name': 'http'
                            }],
                            'env': [
                                {'name': 'DATABASE_URL', 'value': f"postgresql://{name}-db:5432/{name}"},
                                {'name': 'REDIS_URL', 'value': f"redis://{name}-redis:6379"}
                            ],
                            'resources': {
                                'requests': {
                                    'memory': '512Mi',
                                    'cpu': '250m'
                                },
                                'limits': {
                                    'memory': '1Gi',
                                    'cpu': '500m'
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8080
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': 8080
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }

        await self.apply_manifest(deployment_manifest)

    async def reconcile_database(self, namespace: str, name: str, spec: Dict[str, Any]):
        """Create or update database StatefulSet"""

        db_spec = spec.get('database', {})
        storage = db_spec.get('storage', '10Gi')
        replicas = db_spec.get('replicas', 1)

        statefulset_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'StatefulSet',
            'metadata': {
                'name': f"{name}-db",
                'namespace': namespace,
                'labels': {
                    'app': f"{name}-db",
                    'managed-by': 'ecommerce-operator'
                }
            },
            'spec': {
                'serviceName': f"{name}-db-headless",
                'replicas': replicas,
                'selector': {
                    'matchLabels': {
                        'app': f"{name}-db"
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': f"{name}-db"
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'postgres',
                            'image': 'postgres:15-alpine',
                            'env': [
                                {'name': 'POSTGRES_DB', 'value': name},
                                {'name': 'POSTGRES_USER', 'value': 'app_user'},
                                {'name': 'POSTGRES_PASSWORD', 'value': 'password123'}
                            ],
                            'ports': [{'containerPort': 5432}],
                            'volumeMounts': [{
                                'name': 'data',
                                'mountPath': '/var/lib/postgresql/data'
                            }],
                            'resources': {
                                'requests': {
                                    'memory': '512Mi',
                                    'cpu': '250m'
                                },
                                'limits': {
                                    'memory': '1Gi',
                                    'cpu': '500m'
                                }
                            }
                        }]
                    }
                },
                'volumeClaimTemplates': [{
                    'metadata': {'name': 'data'},
                    'spec': {
                        'accessModes': ['ReadWriteOnce'],
                        'resources': {
                            'requests': {
                                'storage': storage
                            }
                        }
                    }
                }]
            }
        }

        await self.apply_manifest(statefulset_manifest)

    async def apply_manifest(self, manifest: Dict[str, Any]):
        """Apply Kubernetes manifest"""
        kind = manifest['kind']
        metadata = manifest['metadata']
        name = metadata['name']
        namespace = metadata.get('namespace', 'default')

        try:
            if kind == 'Deployment':
                try:
                    self.apps_v1.read_namespaced_deployment(name, namespace)
                    # Exists, update it
                    self.apps_v1.patch_namespaced_deployment(name, namespace, manifest)
                except ApiException as e:
                    if e.status == 404:
                        # Doesn't exist, create it
                        self.apps_v1.create_namespaced_deployment(namespace, manifest)
                    else:
                        raise

            elif kind == 'StatefulSet':
                try:
                    self.apps_v1.read_namespaced_stateful_set(name, namespace)
                    # Exists, update it
                    self.apps_v1.patch_namespaced_stateful_set(name, namespace, manifest)
                except ApiException as e:
                    if e.status == 404:
                        # Doesn't exist, create it
                        self.apps_v1.create_namespaced_stateful_set(namespace, manifest)
                    else:
                        raise

            # Add more resource types as needed

        except Exception as e:
            self.logger.error(f"Error applying {kind} {namespace}/{name}: {e}")
            raise

    async def update_status(self, namespace: str, name: str, phase: str, message: str):
        """Update ECommerceApp status"""
        try:
            # Get current resource
            resource = self.custom_api.get_namespaced_custom_object(
                group=self.group,
                version=self.version,
                namespace=namespace,
                plural=self.plural,
                name=name
            )

            # Update status
            if 'status' not in resource:
                resource['status'] = {}

            resource['status']['phase'] = phase
            resource['status']['conditions'] = [{
                'type': 'Ready',
                'status': 'True' if phase == 'Running' else 'False',
                'lastTransitionTime': '2023-01-01T00:00:00Z',  # Should be current time
                'reason': phase,
                'message': message
            }]

            # Patch the resource
            self.custom_api.patch_namespaced_custom_object_status(
                group=self.group,
                version=self.version,
                namespace=namespace,
                plural=self.plural,
                name=name,
                body=resource
            )

        except Exception as e:
            self.logger.error(f"Error updating status for {namespace}/{name}: {e}")

# Main operator entry point
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    operator = ECommerceAppOperator()
    asyncio.run(operator.start())
```

---

## Additional Practice Exercises

### Exercise 3: CI/CD Pipeline Implementation

**Focus**: Advanced pipeline design, GitOps, automated testing, security scanning
**Duration**: 3-4 weeks
**Skills**: Pipeline automation, security integration, deployment strategies

### Exercise 4: Infrastructure as Code (IaC)

**Focus**: Terraform, Ansible, cloud resource management, environment provisioning
**Duration**: 2-3 weeks
**Skills**: Infrastructure automation, version control, environment consistency

### Exercise 5: Monitoring and Observability

**Focus**: Prometheus, Grafana, distributed tracing, log aggregation, alerting
**Duration**: 2-3 weeks
**Skills**: Monitoring design, metrics collection, troubleshooting, SLI/SLO

### Exercise 6: Security and Compliance Automation

**Focus**: Security scanning, compliance automation, secret management, RBAC
**Duration**: 2-3 weeks
**Skills**: Security automation, policy enforcement, audit trails

### Exercise 7: Multi-Cloud and Hybrid Deployments

**Focus**: Multi-cloud strategies, hybrid architectures, cloud migration
**Duration**: 3-4 weeks
**Skills**: Cloud abstraction, disaster recovery, cost optimization

### Exercise 8: Automated Testing in DevOps

**Focus**: Test automation, quality gates, performance testing integration
**Duration**: 2-3 weeks
**Skills**: Testing frameworks, automation integration, quality assurance

### Exercise 9: Configuration Management

**Focus**: Ansible, Puppet, Chef, configuration drift detection, compliance
**Duration**: 2-3 weeks
**Skills**: Configuration automation, state management, compliance monitoring

### Exercise 10: Disaster Recovery and Backup Automation

**Focus**: Backup strategies, disaster recovery automation, business continuity
**Duration**: 2-3 weeks
**Skills**: Recovery planning, automation scripting, testing procedures

---

## Monthly DevOps Assessment

### DevOps Skills Self-Evaluation

Rate your proficiency (1-10) in each area:

**Containerization and Orchestration**:

- [ ] Docker containerization and optimization
- [ ] Kubernetes deployment and management
- [ ] Container security and compliance
- [ ] Service mesh implementation and management

**Automation and Pipelines**:

- [ ] CI/CD pipeline design and implementation
- [ ] Infrastructure as Code (IaC) development
- [ ] Configuration management automation
- [ ] Testing automation integration

**Monitoring and Observability**:

- [ ] Monitoring system design and implementation
- [ ] Log aggregation and analysis
- [ ] Alerting and incident response automation
- [ ] Performance optimization and troubleshooting

**Security and Compliance**:

- [ ] Security automation and scanning
- [ ] Compliance monitoring and reporting
- [ ] Secret management and RBAC implementation
- [ ] Vulnerability assessment and remediation

### Growth Planning Framework

1. **DevOps Philosophy**: What principles guide your approach to DevOps practices?
2. **Automation Mindset**: How effectively do you identify and implement automation opportunities?
3. **Tool Proficiency**: What tools and platforms do you need to master further?
4. **Security Integration**: How well do you integrate security into your DevOps practices?
5. **Collaboration Skills**: How effectively do you work across development and operations teams?
6. **Continuous Improvement**: How do you stay current with DevOps trends and best practices?

### Continuous Learning Recommendations

- Implement and operate production DevOps pipelines
- Contribute to open-source DevOps tools and practices
- Study and implement site reliability engineering (SRE) practices
- Learn cloud-native technologies and patterns
- Practice infrastructure management across multiple cloud providers
- Participate in DevOps community events and conferences

Remember: DevOps is about culture, collaboration, and continuous improvement. Focus on building systems that enable teams to deliver software safely, quickly, and reliably.
