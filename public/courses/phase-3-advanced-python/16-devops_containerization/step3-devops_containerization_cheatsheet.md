# DevOps & Containerization - Quick Reference Cheatsheet

## 🐳 Docker Essentials

### Core Docker Commands

```bash
# Image management
docker build -t myapp:latest .
docker pull ubuntu:20.04
docker images
docker rmi image_id
docker image prune

# Container lifecycle
docker run -d --name myapp -p 8080:80 nginx
docker ps                    # Running containers
docker ps -a                 # All containers
docker stop container_id
docker start container_id
docker restart container_id
docker rm container_id

# Container inspection
docker logs container_id
docker exec -it container_id /bin/bash
docker inspect container_id
docker stats container_id
```

### Dockerfile Best Practices

```dockerfile
# Multi-stage build
FROM node:16-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:16-alpine AS production
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

WORKDIR /app
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --chown=nextjs:nodejs . .

USER nextjs
EXPOSE 3000
ENV NODE_ENV=production
ENV PORT=3000

CMD ["npm", "start"]

# Optimization techniques
# - Use specific image versions
# - Minimize layers
# - Use .dockerignore
# - Don't run as root
# - Use multi-stage builds
# - Order instructions by frequency of change
```

### Docker Compose Essentials

```yaml
# docker-compose.yml
version: "3.8"

services:
  web:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8080:80"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://user:pass@db:5432/myapp
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:13-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    restart: unless-stopped

volumes:
  postgres_data:

networks:
  default:
    driver: bridge
```

### Docker Compose Commands

```bash
# Container management
docker-compose up -d          # Start in background
docker-compose down           # Stop and remove containers
docker-compose restart web    # Restart specific service
docker-compose scale web=3    # Scale service

# Monitoring
docker-compose logs -f web
docker-compose ps
docker-compose top

# Development
docker-compose exec web /bin/bash
docker-compose run --rm web npm test
```

## ☸️ Kubernetes (K8s) Fundamentals

### Core Concepts

```yaml
# Pod - smallest deployable unit
apiVersion: v1
kind: Pod
metadata:
  name: webapp-pod
  labels:
    app: webapp
spec:
  containers:
    - name: webapp
      image: myapp:latest
      ports:
        - containerPort: 8080
      env:
        - name: NODE_ENV
          value: "production"
```

### Deployment Configuration

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-deployment
  labels:
    app: webapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
  template:
    metadata:
      labels:
        app: webapp
    spec:
      containers:
        - name: webapp
          image: myapp:v1.2.3
          ports:
            - containerPort: 8080
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: url
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
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
```

### Service Configuration

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: webapp-service
spec:
  selector:
    app: webapp
  ports:
    - port: 80
      targetPort: 8080
      protocol: TCP
  type: ClusterIP

---
# LoadBalancer service for external access
apiVersion: v1
kind: Service
metadata:
  name: webapp-lb
spec:
  selector:
    app: webapp
  ports:
    - port: 80
      targetPort: 8080
  type: LoadBalancer
```

### ConfigMap & Secrets

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  NODE_ENV: "production"
  LOG_LEVEL: "info"
  API_URL: "https://api.example.com"

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  username: dXNlcg== # base64 encoded 'user'
  password: cGFzcw== # base64 encoded 'pass'
  url: cG9zdGdyZXNxbDovL3VzZXI6cGFzc0BkYjozMzA2L215YXBw
```

### Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: webapp-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
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
                name: webapp-service
                port:
                  number: 80
```

### Essential kubectl Commands

```bash
# Cluster info
kubectl cluster-info
kubectl get nodes
kubectl describe node node-name

# Workload management
kubectl apply -f deployment.yaml
kubectl get deployments
kubectl scale deployment webapp-deployment --replicas=5
kubectl rollout status deployment/webapp-deployment
kubectl rollout undo deployment/webapp-deployment

# Pod operations
kubectl get pods -o wide
kubectl describe pod pod-name
kubectl logs pod-name -f
kubectl exec -it pod-name -- /bin/bash
kubectl port-forward pod/pod-name 8080:8080

# Services and networking
kubectl get services
kubectl get ingress
kubectl describe service webapp-service

# Configuration
kubectl get configmaps
kubectl get secrets
kubectl create secret generic db-secret --from-literal=password=mypassword

# Debugging
kubectl top nodes
kubectl top pods
kubectl get events --sort-by=.metadata.creationTimestamp
```

## 🔄 CI/CD Pipelines

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: "18"
          cache: "npm"

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test

      - name: Run linting
        run: npm run lint

  build:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-

      - name: Build and push Docker image
        uses: docker/build-push-action@v3
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
      - uses: actions/checkout@v3

      - name: Configure kubectl
        uses: azure/k8s-set-context@v1
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBECONFIG }}

      - name: Deploy to Kubernetes
        run: |
          sed -i "s|IMAGE_TAG|${{ github.sha }}|g" k8s/deployment.yaml
          kubectl apply -f k8s/
          kubectl rollout status deployment/webapp-deployment
```

### GitLab CI/CD Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  KUBECONFIG: /etc/deploy/config

before_script:
  - docker info
  - echo $CI_REGISTRY_PASSWORD | docker login -u $CI_REGISTRY_USER --password-stdin $CI_REGISTRY

test:
  stage: test
  image: node:16-alpine
  cache:
    paths:
      - node_modules/
  script:
    - npm ci
    - npm test
    - npm run lint
  coverage: '/Lines\s*:\s*(\d+\.\d+)%/'

build:
  stage: build
  only:
    - main
  script:
    - docker build -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE

deploy_staging:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: https://staging.myapp.com
  only:
    - main
  script:
    - kubectl config use-context staging
    - sed -i "s|IMAGE_TAG|$CI_COMMIT_SHA|g" k8s/deployment.yaml
    - kubectl apply -f k8s/
    - kubectl rollout status deployment/webapp-deployment

deploy_production:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: https://myapp.com
  when: manual
  only:
    - main
  script:
    - kubectl config use-context production
    - sed -i "s|IMAGE_TAG|$CI_COMMIT_SHA|g" k8s/deployment.yaml
    - kubectl apply -f k8s/
    - kubectl rollout status deployment/webapp-deployment
```

### Jenkins Pipeline (Jenkinsfile)

```groovy
pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'myapp'
        DOCKER_TAG = "${BUILD_NUMBER}"
        KUBECONFIG = credentials('kubeconfig')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test') {
            steps {
                sh 'npm ci'
                sh 'npm test'
                sh 'npm run lint'
            }
            post {
                always {
                    publishTestResults testResultsPattern: 'test-results.xml'
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'coverage',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }
            }
        }

        stage('Build Docker Image') {
            when {
                branch 'main'
            }
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                    docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push()
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push('latest')
                    }
                }
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'main'
            }
            steps {
                sh """
                    kubectl config use-context staging
                    sed -i 's|IMAGE_TAG|${DOCKER_TAG}|g' k8s/deployment.yaml
                    kubectl apply -f k8s/
                    kubectl rollout status deployment/webapp-deployment
                """
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            input {
                message "Deploy to production?"
                ok "Deploy"
            }
            steps {
                sh """
                    kubectl config use-context production
                    sed -i 's|IMAGE_TAG|${DOCKER_TAG}|g' k8s/deployment.yaml
                    kubectl apply -f k8s/
                    kubectl rollout status deployment/webapp-deployment
                """
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        failure {
            emailext (
                subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build failed. Check console output at ${env.BUILD_URL}",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

## 🏗️ Infrastructure as Code (IaC)

### Terraform Basics

```hcl
# main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "terraform-state-bucket"
    key    = "myapp/terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.environment}-vpc"
    Environment = var.environment
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.environment}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Environment = var.environment
  }
}

# ECS Service
resource "aws_ecs_service" "webapp" {
  name            = "webapp"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.webapp.arn
  desired_count   = 2

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
  }

  network_configuration {
    subnets         = aws_subnet.private[*].id
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.webapp.arn
    container_name   = "webapp"
    container_port   = 8080
  }

  depends_on = [aws_lb_listener.webapp]
}

# Outputs
output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}
```

### Ansible Playbook

```yaml
# playbook.yml
---
- name: Deploy web application
  hosts: webservers
  become: yes
  vars:
    app_name: myapp
    app_version: "{{ version | default('latest') }}"

  tasks:
    - name: Update system packages
      apt:
        update_cache: yes
        upgrade: dist

    - name: Install Docker
      apt:
        name: docker.io
        state: present

    - name: Start Docker service
      systemd:
        name: docker
        state: started
        enabled: yes

    - name: Add user to docker group
      user:
        name: "{{ ansible_user }}"
        groups: docker
        append: yes

    - name: Pull application image
      docker_image:
        name: "{{ app_name }}:{{ app_version }}"
        source: pull

    - name: Stop existing container
      docker_container:
        name: "{{ app_name }}"
        state: absent

    - name: Run new container
      docker_container:
        name: "{{ app_name }}"
        image: "{{ app_name }}:{{ app_version }}"
        ports:
          - "80:8080"
        restart_policy: unless-stopped
        env:
          NODE_ENV: production
          DATABASE_URL: "{{ database_url }}"

    - name: Configure nginx
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/sites-available/{{ app_name }}
      notify: restart nginx

  handlers:
    - name: restart nginx
      systemd:
        name: nginx
        state: restarted
```

## 📊 Monitoring & Logging

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]

  - job_name: "webapp"
    kubernetes_sd_configs:
      - role: endpoints
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        action: keep
        regex: webapp-service
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Application Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### ELK Stack Configuration

```yaml
# docker-compose.elk.yml
version: "3.8"
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:7.15.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline:ro
    ports:
      - "5000:5000/tcp"
      - "5000:5000/udp"
    environment:
      LS_JAVA_OPTS: "-Xmx256m -Xms256m"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_URL: http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

```ruby
# logstash/pipeline/logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "webapp" {
    json {
      source => "message"
    }

    date {
      match => ["timestamp", "ISO8601"]
    }

    mutate {
      remove_field => ["@version", "beat", "offset"]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "logs-%{[fields][service]}-%{+YYYY.MM.dd}"
  }

  stdout {
    codec => rubydebug
  }
}
```

## 🔒 Security & Best Practices

### Docker Security

```dockerfile
# Security-hardened Dockerfile
FROM node:16-alpine

# Don't run as root
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Remove unnecessary packages
RUN apk del curl wget

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy application code
COPY --chown=nextjs:nodejs . .

# Switch to non-root user
USER nextjs

# Run application
EXPOSE 3000
CMD ["npm", "start"]
```

### Kubernetes Security Policies

```yaml
# pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - "configMap"
    - "emptyDir"
    - "projected"
    - "secret"
    - "downwardAPI"
    - "persistentVolumeClaim"
  runAsUser:
    rule: "MustRunAsNonRoot"
  seLinux:
    rule: "RunAsAny"
  fsGroup:
    rule: "RunAsAny"
```

### Network Security

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: webapp-network-policy
spec:
  podSelector:
    matchLabels:
      app: webapp
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: nginx-ingress
      ports:
        - protocol: TCP
          port: 8080
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: database
      ports:
        - protocol: TCP
          port: 5432
    - to: []
      ports:
        - protocol: TCP
          port: 443 # HTTPS
        - protocol: TCP
          port: 53 # DNS
```

## 🚀 Deployment Strategies

### Blue-Green Deployment

```yaml
# Blue-Green with Kubernetes
apiVersion: v1
kind: Service
metadata:
  name: webapp-active
spec:
  selector:
    app: webapp
    version: blue # Switch between 'blue' and 'green'
  ports:
    - port: 80
      targetPort: 8080

---
# Blue deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
      version: blue
  template:
    metadata:
      labels:
        app: webapp
        version: blue
    spec:
      containers:
        - name: webapp
          image: myapp:v1.0.0

---
# Green deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
      version: green
  template:
    metadata:
      labels:
        app: webapp
        version: green
    spec:
      containers:
        - name: webapp
          image: myapp:v2.0.0
```

### Canary Deployment

```yaml
# Canary deployment with Istio
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: webapp
spec:
  http:
    - match:
        - headers:
            canary:
              exact: "true"
      route:
        - destination:
            host: webapp
            subset: v2
          weight: 100
    - route:
        - destination:
            host: webapp
            subset: v1
          weight: 90
        - destination:
            host: webapp
            subset: v2
          weight: 10 # 10% traffic to new version

---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: webapp
spec:
  host: webapp
  subsets:
    - name: v1
      labels:
        version: v1
    - name: v2
      labels:
        version: v2
```

## 🛠️ Useful Commands & Scripts

### Docker Cleanup

```bash
#!/bin/bash
# docker-cleanup.sh

echo "Stopping all containers..."
docker stop $(docker ps -aq)

echo "Removing all containers..."
docker rm $(docker ps -aq)

echo "Removing all images..."
docker rmi $(docker images -q)

echo "Removing all volumes..."
docker volume prune -f

echo "Removing all networks..."
docker network prune -f

echo "Cleanup complete!"
```

### Kubernetes Debugging

```bash
#!/bin/bash
# k8s-debug.sh

POD_NAME=$1

if [ -z "$POD_NAME" ]; then
    echo "Usage: $0 <pod-name>"
    exit 1
fi

echo "Pod details:"
kubectl describe pod $POD_NAME

echo -e "\nPod logs:"
kubectl logs $POD_NAME --tail=50

echo -e "\nContainer processes:"
kubectl exec $POD_NAME -- ps aux

echo -e "\nEnvironment variables:"
kubectl exec $POD_NAME -- env

echo -e "\nNetwork connectivity:"
kubectl exec $POD_NAME -- nslookup kubernetes.default
```

### Health Check Script

```bash
#!/bin/bash
# health-check.sh

ENDPOINT=$1
TIMEOUT=${2:-10}

if [ -z "$ENDPOINT" ]; then
    echo "Usage: $0 <endpoint> [timeout]"
    exit 1
fi

check_health() {
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT $ENDPOINT/health)

    if [ "$response" = "200" ]; then
        echo "✅ Service is healthy"
        return 0
    else
        echo "❌ Service is unhealthy (HTTP $response)"
        return 1
    fi
}

# Check health with retries
for i in {1..5}; do
    if check_health; then
        exit 0
    else
        echo "Retry $i/5..."
        sleep 5
    fi
done

echo "Service failed health check after 5 retries"
exit 1
```

## 📚 Configuration Management

### Environment Configuration

```bash
# .env.example
NODE_ENV=production
PORT=3000
LOG_LEVEL=info

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/myapp
REDIS_URL=redis://localhost:6379

# Authentication
JWT_SECRET=your-secret-key
SESSION_SECRET=another-secret

# External APIs
API_KEY=your-api-key
WEBHOOK_SECRET=webhook-secret

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
```

### Helm Charts

```yaml
# Chart.yaml
apiVersion: v2
name: webapp
description: A Helm chart for webapp
type: application
version: 0.1.0
appVersion: "1.0"

---
# values.yaml
replicaCount: 2

image:
  repository: myapp
  tag: "latest"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: myapp.example.com
      paths:
        - path: /
          pathType: Prefix

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
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
```

---

_Essential commands and configurations for modern DevOps and container orchestration_
