# DevOps & Containerization - Interview Preparation Guide

## 🎯 DevOps Interview Overview

### What Interviewers Look For

```
Technical Competencies:
✓ Container orchestration understanding
✓ CI/CD pipeline design and implementation
✓ Infrastructure as Code (IaC) experience
✓ Monitoring and observability setup
✓ Security best practices
✓ Cloud platform expertise
✓ Troubleshooting and problem-solving skills

Soft Skills:
✓ Collaboration with development teams
✓ Process improvement mindset
✓ Automation-first thinking
✓ Incident response and communication
✓ Learning agility and adaptability

Interview Formats:
- Technical discussions (45-60 minutes)
- Live system troubleshooting
- Architecture design scenarios
- CI/CD pipeline design
- Infrastructure planning exercises
- On-call and incident response scenarios
```

### Common Question Categories

```
Container Technologies:
- Docker fundamentals and best practices
- Kubernetes architecture and concepts
- Container security and networking
- Orchestration patterns and strategies

Infrastructure Management:
- Infrastructure as Code tools and practices
- Configuration management
- Cloud services and architecture
- Network design and security

CI/CD and Automation:
- Pipeline design and optimization
- Testing strategies and automation
- Deployment patterns and strategies
- Build and release management

Monitoring and Reliability:
- Observability and monitoring setup
- Log aggregation and analysis
- Performance optimization
- Incident response and SRE practices
```

## 🐳 Docker and Containerization

### Docker Fundamentals

```
Core Concepts to Master:

1. Container vs VM Architecture:
Virtual Machines:
┌─────────────┐ ┌─────────────┐
│    App A    │ │    App B    │
├─────────────┤ ├─────────────┤
│  Guest OS   │ │  Guest OS   │
├─────────────┤ ├─────────────┤
│ Hypervisor  │ │ Hypervisor  │
├─────────────┴─┴─────────────┤
│        Host OS              │
├─────────────────────────────┤
│       Hardware              │
└─────────────────────────────┘

Containers:
┌─────────────┐ ┌─────────────┐
│    App A    │ │    App B    │
├─────────────┤ ├─────────────┤
│  Libraries  │ │  Libraries  │
├─────────────┴─┴─────────────┤
│    Container Runtime        │
├─────────────────────────────┤
│        Host OS              │
├─────────────────────────────┤
│       Hardware              │
└─────────────────────────────┘

Benefits of Containers:
✓ Resource efficiency
✓ Faster startup times
✓ Consistent environments
✓ Easy scalability
✓ Simplified deployment

2. Docker Architecture:
Client → Docker Daemon → Images/Containers
```

### Dockerfile Best Practices

```dockerfile
# Multi-stage build example
FROM node:16-alpine AS builder
WORKDIR /app

# Copy package files first for better layer caching
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY src/ ./src/
RUN npm run build

# Production stage
FROM node:16-alpine AS production

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

WORKDIR /app

# Copy built application from builder stage
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules

# Security: Don't run as root
USER nextjs

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Expose port and set command
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

### Docker Optimization Techniques

```dockerfile
# ❌ Bad practices
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y git
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt

# ✅ Optimized version
FROM python:3.9-slim

# Combine RUN commands to reduce layers
RUN apt-get update && apt-get install -y \
    git \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code last
COPY . .

# Use specific versions and slim images
# Remove unnecessary packages after installation
# Use .dockerignore to exclude unnecessary files
```

### Interview Questions and Answers

```
Q: "How do you optimize Docker image size?"
A: Detailed approach with examples:

1. Use multi-stage builds
2. Choose minimal base images (alpine, scratch)
3. Combine RUN commands
4. Remove package manager cache
5. Use .dockerignore
6. Order layers by frequency of change

Q: "What's the difference between CMD and ENTRYPOINT?"
A:
- CMD: Default command, can be overridden
- ENTRYPOINT: Always executed, CMD becomes arguments
- Best practice: Use both together

# Example:
ENTRYPOINT ["python", "app.py"]
CMD ["--help"]

# docker run myapp           -> python app.py --help
# docker run myapp --prod    -> python app.py --prod

Q: "How do you handle secrets in Docker containers?"
A: Security-focused approach:

1. Docker Secrets (Swarm mode)
2. External secret management (Vault, AWS Secrets Manager)
3. Init containers for secret retrieval
4. Environment variables (least secure)
5. Mounted files from secure volumes

# Example with Docker Secrets
docker service create \
  --name myapp \
  --secret source=db_password,target=/run/secrets/db_password \
  myapp:latest
```

## ☸️ Kubernetes Mastery

### Core Architecture Understanding

```
Kubernetes Cluster Components:

Master Node (Control Plane):
┌─────────────────────────────────────┐
│                API Server           │ ← Entry point
├─────────────────────────────────────┤
│                etcd                 │ ← Key-value store
├─────────────────────────────────────┤
│              Scheduler              │ ← Pod placement
├─────────────────────────────────────┤
│         Controller Manager          │ ← Desired state
└─────────────────────────────────────┘

Worker Nodes:
┌─────────────────────────────────────┐
│               kubelet               │ ← Node agent
├─────────────────────────────────────┤
│              kube-proxy             │ ← Network proxy
├─────────────────────────────────────┤
│         Container Runtime           │ ← Docker/containerd
└─────────────────────────────────────┘

Key Concepts:
- Pods: Smallest deployable units
- Services: Network abstraction for pods
- Deployments: Declarative pod management
- ConfigMaps/Secrets: Configuration management
- Namespaces: Resource isolation
- Ingress: External access management
```

### Production-Ready Manifests

```yaml
# Complete application deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-deployment
  labels:
    app: webapp
    version: v1.2.3
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
  selector:
    matchLabels:
      app: webapp
  template:
    metadata:
      labels:
        app: webapp
        version: v1.2.3
    spec:
      # Security context
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000

      # Init container for database migration
      initContainers:
        - name: db-migration
          image: webapp:v1.2.3
          command: ["npm", "run", "migrate"]
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: database-url

      containers:
        - name: webapp
          image: webapp:v1.2.3
          ports:
            - containerPort: 3000
              name: http

          # Environment variables
          env:
            - name: NODE_ENV
              value: "production"
            - name: PORT
              value: "3000"

          # Config from ConfigMap
          envFrom:
            - configMapRef:
                name: webapp-config

            # Secrets
            - secretRef:
                name: webapp-secrets

          # Resource limits and requests
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"

          # Health checks
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
            failureThreshold: 3

          # Startup probe for slow-starting containers
          startupProbe:
            httpGet:
              path: /health
              port: http
            failureThreshold: 30
            periodSeconds: 10

          # Security context
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL

          # Volume mounts
          volumeMounts:
            - name: temp-storage
              mountPath: /tmp
            - name: config-volume
              mountPath: /app/config
              readOnly: true

      volumes:
        - name: temp-storage
          emptyDir: {}
        - name: config-volume
          configMap:
            name: webapp-config

---
# Service for internal communication
apiVersion: v1
kind: Service
metadata:
  name: webapp-service
  labels:
    app: webapp
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: http
      protocol: TCP
      name: http
  selector:
    app: webapp

---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: webapp-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: webapp-deployment
  minReplicas: 3
  maxReplicas: 10
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

---
# Pod Disruption Budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: webapp-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: webapp

---
# Network Policy for security
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
              app: frontend
      ports:
        - protocol: TCP
          port: 3000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: database
      ports:
        - protocol: TCP
          port: 5432
    # Allow DNS
    - to: []
      ports:
        - protocol: UDP
          port: 53
```

### Kubernetes Troubleshooting Guide

```bash
# Pod troubleshooting commands
kubectl get pods -o wide
kubectl describe pod <pod-name>
kubectl logs <pod-name> -f
kubectl logs <pod-name> -c <container-name>
kubectl logs <pod-name> --previous  # Previous container logs

# Debug running containers
kubectl exec -it <pod-name> -- /bin/bash
kubectl exec -it <pod-name> -c <container-name> -- /bin/sh

# Resource inspection
kubectl top nodes
kubectl top pods
kubectl get events --sort-by=.metadata.creationTimestamp

# Network debugging
kubectl get services
kubectl get endpoints
kubectl describe service <service-name>

# Debug DNS issues
kubectl run debug-pod --image=busybox -it --rm -- nslookup <service-name>

# Check resource usage and limits
kubectl describe nodes
kubectl get limitranges
kubectl get resourcequotas

# Security and RBAC
kubectl auth can-i create pods
kubectl get rolebindings
kubectl describe rolebinding <binding-name>
```

### Advanced Kubernetes Concepts

```yaml
# Custom Resource Definition (CRD)
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: webapps.stable.example.com
spec:
  group: stable.example.com
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
                image:
                  type: string
  scope: Namespaced
  names:
    plural: webapps
    singular: webapp
    kind: WebApp

---
# Operator pattern example
apiVersion: stable.example.com/v1
kind: WebApp
metadata:
  name: my-webapp
spec:
  replicas: 3
  image: "webapp:v1.2.3"
  database:
    type: postgresql
    storage: 10Gi
```

### Interview Scenarios

```
Scenario 1: "A pod keeps restarting. How do you debug?"

Systematic Approach:
1. Check pod status and events
   kubectl get pods
   kubectl describe pod <name>

2. Examine logs
   kubectl logs <pod> --previous
   kubectl logs <pod> -f

3. Check resource constraints
   kubectl top pod <name>
   kubectl describe node <node-name>

4. Verify health checks
   - Review liveness/readiness probe configuration
   - Test health endpoints manually

5. Check dependencies
   - Database connectivity
   - External service availability
   - ConfigMap/Secret availability

6. Network issues
   - DNS resolution
   - Service discovery
   - Network policies

Scenario 2: "How do you perform zero-downtime deployment?"

Strategy:
1. Rolling updates with proper health checks
2. Pod Disruption Budgets
3. Blue-green deployment pattern
4. Canary deployments with traffic splitting

# Rolling update configuration
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 0  # Zero downtime

# Canary deployment with Istio
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: webapp-vs
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: webapp-service
        subset: canary
  - route:
    - destination:
        host: webapp-service
        subset: stable
      weight: 90
    - destination:
        host: webapp-service
        subset: canary
      weight: 10
```

## 🔄 CI/CD Pipeline Design

### Pipeline Architecture

```
Modern CI/CD Pipeline Flow:

Code Commit → Trigger Pipeline
     ↓
Build Stage (Compile, Test, Package)
     ↓
Security Scanning (SAST, Dependency Check)
     ↓
Artifact Storage (Container Registry)
     ↓
Deploy to Staging → Automated Testing
     ↓
Security Scanning (DAST)
     ↓
Manual Approval (Production)
     ↓
Deploy to Production → Health Checks
     ↓
Monitoring & Alerting
```

### GitHub Actions Pipeline

```yaml
# .github/workflows/ci-cd.yml
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
    strategy:
      matrix:
        node-version: [16, 18, 20]

    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: "npm"

      - name: Install dependencies
        run: npm ci

      - name: Run linter
        run: npm run lint

      - name: Run unit tests
        run: npm run test:unit
        env:
          NODE_ENV: test

      - name: Run integration tests
        run: npm run test:integration
        env:
          NODE_ENV: test
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/testdb

      - name: Generate test coverage
        run: npm run test:coverage

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          token: ${{ secrets.CODECOV_TOKEN }}

  security:
    runs-on: ubuntu-latest
    needs: test

    steps:
      - uses: actions/checkout@v4

      - name: Run security audit
        run: npm audit --audit-level moderate

      - name: SAST with CodeQL
        uses: github/codeql-action/analyze@v2
        with:
          languages: javascript

      - name: Container security scan
        uses: aquasec/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: "sarif"
          output: "trivy-results.sarif"

  build:
    runs-on: ubuntu-latest
    needs: [test, security]
    if: github.ref == 'refs/heads/main'

    permissions:
      contents: read
      packages: write

    outputs:
      image-digest: ${{ steps.build.outputs.digest }}

    steps:
      - uses: actions/checkout@v4

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
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker image
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            BUILDTIME=${{ fromJSON(steps.meta.outputs.json).labels['org.opencontainers.image.created'] }}
            VERSION=${{ fromJSON(steps.meta.outputs.json).labels['org.opencontainers.image.version'] }}

  deploy-staging:
    runs-on: ubuntu-latest
    needs: build
    environment:
      name: staging
      url: https://staging.myapp.com

    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/k8s-set-context@v1
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG_STAGING }}

      - name: Deploy to staging
        run: |
          # Update image tag in deployment
          sed -i "s|IMAGE_TAG|${{ github.sha }}|g" k8s/staging/

          # Apply configurations
          kubectl apply -f k8s/staging/

          # Wait for rollout to complete
          kubectl rollout status deployment/webapp-deployment -n staging --timeout=600s

      - name: Run smoke tests
        run: |
          # Wait for service to be ready
          sleep 30

          # Run basic health checks
          curl -f https://staging.myapp.com/health || exit 1
          curl -f https://staging.myapp.com/metrics || exit 1

      - name: Run E2E tests
        run: |
          npm install
          npm run test:e2e:staging

  deploy-production:
    runs-on: ubuntu-latest
    needs: [build, deploy-staging]
    environment:
      name: production
      url: https://myapp.com
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/k8s-set-context@v1
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG_PRODUCTION }}

      - name: Blue-Green Deployment
        run: |
          # Deploy to blue environment first
          sed -i "s|IMAGE_TAG|${{ github.sha }}|g" k8s/production/blue/
          kubectl apply -f k8s/production/blue/
          kubectl rollout status deployment/webapp-blue-deployment -n production

          # Run health checks on blue environment
          kubectl port-forward service/webapp-blue-service 8080:80 -n production &
          sleep 10
          curl -f http://localhost:8080/health || exit 1

          # Switch traffic to blue (this becomes the new green)
          kubectl patch service webapp-service -n production -p '{"spec":{"selector":{"version":"blue"}}}'

          # Clean up old green environment
          kubectl delete deployment webapp-green-deployment -n production --ignore-not-found=true

  notify:
    runs-on: ubuntu-latest
    needs: [deploy-production]
    if: always()

    steps:
      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          channel: "#deployments"
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
          fields: repo,message,commit,author,action,eventName,ref,workflow
```

### Advanced Pipeline Strategies

```yaml
# GitLab CI/CD with advanced features
stages:
  - validate
  - test
  - security
  - build
  - deploy-review
  - deploy-staging
  - deploy-production
  - cleanup

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

# Parallel testing strategy
.test_template: &test_template
  stage: test
  image: node:16-alpine
  services:
    - postgres:13-alpine
  variables:
    POSTGRES_DB: testdb
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: postgres
  before_script:
    - npm ci
    - npm run db:migrate
  cache:
    key: ${CI_COMMIT_REF_SLUG}
    paths:
      - node_modules/
      - .npm/

unit_tests:
  <<: *test_template
  script:
    - npm run test:unit
  coverage: '/Lines\s*:\s*(\d+\.\d+)%/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml
    expire_in: 1 week

integration_tests:
  <<: *test_template
  script:
    - npm run test:integration
  parallel: 3

performance_tests:
  stage: test
  image: grafana/k6
  script:
    - k6 run performance-tests/load-test.js
  artifacts:
    reports:
      performance: performance-report.json

# Container scanning
container_security:
  stage: security
  image: docker:stable
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker run --rm -v /var/run/docker.sock:/var/run/docker.sock
      -v $(pwd):/tmp/.cache/ aquasec/trivy image
      --format template --template "@contrib/sarif.tpl"
      -o /tmp/.cache/trivy-report.sarif $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  artifacts:
    reports:
      sast: trivy-report.sarif

# Dynamic environment creation
review_app:
  stage: deploy-review
  image: bitnami/kubectl
  script:
    - |
      # Create namespace for review app
      kubectl create namespace review-app-$CI_MERGE_REQUEST_IID || true

      # Deploy application with branch-specific configuration
      sed -i "s|{{IMAGE_TAG}}|$CI_COMMIT_SHA|g" k8s/review-app.yaml
      sed -i "s|{{NAMESPACE}}|review-app-$CI_MERGE_REQUEST_IID|g" k8s/review-app.yaml
      sed -i "s|{{BRANCH}}|$CI_COMMIT_REF_SLUG|g" k8s/review-app.yaml

      kubectl apply -f k8s/review-app.yaml -n review-app-$CI_MERGE_REQUEST_IID

      # Wait for deployment
      kubectl rollout status deployment/webapp -n review-app-$CI_MERGE_REQUEST_IID

      # Get service URL
      REVIEW_URL=$(kubectl get ingress webapp-ingress -n review-app-$CI_MERGE_REQUEST_IID -o jsonpath='{.spec.rules[0].host}')
      echo "Review app available at: https://$REVIEW_URL"
  environment:
    name: review-app-$CI_MERGE_REQUEST_IID
    url: https://review-app-$CI_MERGE_REQUEST_IID.example.com
    on_stop: cleanup_review_app
  only:
    - merge_requests

cleanup_review_app:
  stage: cleanup
  image: bitnami/kubectl
  script:
    - kubectl delete namespace review-app-$CI_MERGE_REQUEST_IID
  environment:
    name: review-app-$CI_MERGE_REQUEST_IID
    action: stop
  when: manual
  only:
    - merge_requests
```

## 🏗️ Infrastructure as Code

### Terraform Best Practices

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }

  backend "s3" {
    bucket         = "mycompany-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values for reusability
locals {
  name_prefix = "${var.environment}-${var.project_name}"

  common_tags = {
    Environment   = var.environment
    Project       = var.project_name
    ManagedBy    = "Terraform"
    Owner        = var.owner_email
    CostCenter   = var.cost_center
  }

  vpc_cidr = var.environment == "production" ? "10.0.0.0/16" : "10.1.0.0/16"
}

# VPC Module
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "${local.name_prefix}-vpc"
  cidr = local.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = [for i in range(3) : cidrsubnet(local.vpc_cidr, 8, i)]
  public_subnets  = [for i in range(3) : cidrsubnet(local.vpc_cidr, 8, i + 100)]

  enable_nat_gateway = true
  enable_vpn_gateway = var.enable_vpn
  single_nat_gateway = var.environment != "production"

  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = local.common_tags

  vpc_tags = {
    "kubernetes.io/cluster/${local.name_prefix}-eks" = "shared"
  }

  public_subnet_tags = {
    "kubernetes.io/cluster/${local.name_prefix}-eks" = "shared"
    "kubernetes.io/role/elb"                         = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${local.name_prefix}-eks" = "shared"
    "kubernetes.io/role/internal-elb"               = "1"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"

  cluster_name    = "${local.name_prefix}-eks"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Cluster access configuration
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  cluster_endpoint_public_access_cidrs = var.allowed_cidr_blocks

  # Encryption
  cluster_encryption_config = [
    {
      provider_key_arn = aws_kms_key.eks.arn
      resources        = ["secrets"]
    }
  ]

  # Node groups
  eks_managed_node_groups = {
    general = {
      min_size       = var.environment == "production" ? 3 : 1
      max_size       = var.environment == "production" ? 10 : 3
      desired_size   = var.environment == "production" ? 3 : 2
      instance_types = var.environment == "production" ? ["m5.large"] : ["t3.medium"]

      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "general"
      }

      tags = local.common_tags
    }

    spot = {
      min_size       = 0
      max_size       = 5
      desired_size   = 0
      instance_types = ["t3.medium", "t3a.medium", "t2.medium"]
      capacity_type  = "SPOT"

      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "spot"
      }

      taints = {
        spot = {
          key    = "spot-instance"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }

      tags = merge(local.common_tags, {
        NodeType = "spot"
      })
    }
  }

  # Cluster add-ons
  cluster_addons = {
    aws-ebs-csi-driver = {
      most_recent = true
    }
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
  }

  tags = local.common_tags
}

# KMS key for EKS encryption
resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = local.common_tags
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${local.name_prefix}-eks"
  target_key_id = aws_kms_key.eks.key_id
}

# RDS Database
resource "aws_db_subnet_group" "main" {
  name       = "${local.name_prefix}-db-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db-subnet-group"
  })
}

resource "aws_security_group" "rds" {
  name_prefix = "${local.name_prefix}-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-rds-sg"
  })
}

resource "aws_db_instance" "main" {
  identifier = "${local.name_prefix}-db"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.environment == "production" ? "db.r6g.large" : "db.t3.micro"

  allocated_storage     = var.environment == "production" ? 100 : 20
  max_allocated_storage = var.environment == "production" ? 1000 : 100
  storage_encrypted     = true

  db_name  = var.database_name
  username = var.database_username
  password = var.database_password

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  backup_retention_period = var.environment == "production" ? 30 : 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = var.environment != "production"
  deletion_protection = var.environment == "production"

  performance_insights_enabled = var.environment == "production"
  monitoring_interval         = var.environment == "production" ? 60 : 0

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-database"
  })
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.name_prefix}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name_prefix = "${local.name_prefix}-redis-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-redis-sg"
  })
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "${local.name_prefix}-redis"
  description                = "Redis cluster for ${local.name_prefix}"

  port               = 6379
  parameter_group_name = "default.redis7"
  node_type          = var.environment == "production" ? "cache.r6g.large" : "cache.t3.micro"

  num_cache_clusters = var.environment == "production" ? 3 : 1

  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  automatic_failover_enabled = var.environment == "production"
  multi_az_enabled          = var.environment == "production"

  tags = local.common_tags
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
  sensitive   = true
}
```

### Ansible Configuration Management

```yaml
# ansible/playbook.yml
---
- name: Configure Kubernetes cluster nodes
  hosts: k8s_nodes
  become: yes
  vars:
    kubernetes_version: "1.28.0"
    containerd_version: "1.7.0"

  tasks:
    - name: Update system packages
      apt:
        update_cache: yes
        upgrade: dist
        autoclean: yes
        autoremove: yes

    - name: Install required packages
      apt:
        name:
          - apt-transport-https
          - ca-certificates
          - curl
          - gnupg
          - lsb-release
          - software-properties-common
        state: present

    - name: Add Docker GPG key
      apt_key:
        url: https://download.docker.com/linux/ubuntu/gpg
        state: present

    - name: Add Docker repository
      apt_repository:
        repo: "deb [arch=amd64] https://download.docker.com/linux/ubuntu {{ ansible_distribution_release }} stable"
        state: present

    - name: Install containerd
      apt:
        name: containerd.io
        state: present

    - name: Configure containerd
      template:
        src: containerd-config.toml.j2
        dest: /etc/containerd/config.toml
      notify: restart containerd

    - name: Add Kubernetes GPG key
      apt_key:
        url: https://packages.cloud.google.com/apt/doc/apt-key.gpg
        state: present

    - name: Add Kubernetes repository
      apt_repository:
        repo: "deb https://apt.kubernetes.io/ kubernetes-xenial main"
        state: present

    - name: Install Kubernetes components
      apt:
        name:
          - kubelet={{ kubernetes_version }}-00
          - kubeadm={{ kubernetes_version }}-00
          - kubectl={{ kubernetes_version }}-00
        state: present

    - name: Hold Kubernetes packages
      dpkg_selections:
        name: "{{ item }}"
        selection: hold
      loop:
        - kubelet
        - kubeadm
        - kubectl

    - name: Configure kubelet
      template:
        src: kubelet-config.yaml.j2
        dest: /var/lib/kubelet/config.yaml
      notify: restart kubelet

    - name: Enable and start services
      systemd:
        name: "{{ item }}"
        enabled: yes
        state: started
      loop:
        - containerd
        - kubelet

    - name: Configure system settings
      sysctl:
        name: "{{ item.key }}"
        value: "{{ item.value }}"
        state: present
        reload: yes
      loop:
        - { key: "net.bridge.bridge-nf-call-ip6tables", value: "1" }
        - { key: "net.bridge.bridge-nf-call-iptables", value: "1" }
        - { key: "net.ipv4.ip_forward", value: "1" }

    - name: Disable swap
      command: swapoff -a
      when: ansible_swaptotal_mb > 0

    - name: Remove swap from fstab
      lineinfile:
        path: /etc/fstab
        regexp: '^.*\s+swap\s+'
        state: absent

  handlers:
    - name: restart containerd
      systemd:
        name: containerd
        state: restarted

    - name: restart kubelet
      systemd:
        name: kubelet
        state: restarted

# Security hardening playbook
- name: Harden Kubernetes nodes
  hosts: k8s_nodes
  become: yes
  tasks:
    - name: Configure firewall rules
      ufw:
        rule: "{{ item.rule }}"
        port: "{{ item.port }}"
        proto: "{{ item.proto }}"
        src: "{{ item.src | default(omit) }}"
      loop:
        - { rule: "allow", port: "22", proto: "tcp", src: "10.0.0.0/8" }
        - { rule: "allow", port: "6443", proto: "tcp" } # Kubernetes API
        - { rule: "allow", port: "2379:2380", proto: "tcp" } # etcd
        - { rule: "allow", port: "10250", proto: "tcp" } # Kubelet
        - { rule: "allow", port: "10251", proto: "tcp" } # kube-scheduler
        - { rule: "allow", port: "10252", proto: "tcp" } # kube-controller-manager

    - name: Enable firewall
      ufw:
        state: enabled
        policy: deny
        direction: incoming

    - name: Configure audit logging
      template:
        src: audit-policy.yaml.j2
        dest: /etc/kubernetes/audit-policy.yaml

    - name: Set file permissions
      file:
        path: "{{ item.path }}"
        mode: "{{ item.mode }}"
        owner: "{{ item.owner }}"
        group: "{{ item.group }}"
      loop:
        - {
            path: "/etc/kubernetes",
            mode: "0755",
            owner: "root",
            group: "root",
          }
        - {
            path: "/etc/kubernetes/manifests",
            mode: "0755",
            owner: "root",
            group: "root",
          }
        - {
            path: "/etc/kubernetes/pki",
            mode: "0700",
            owner: "root",
            group: "root",
          }
```

## 📊 Monitoring and Observability

### Prometheus and Grafana Setup

```yaml
# monitoring/prometheus-values.yaml
prometheus:
  prometheusSpec:
    retention: 30d
    retentionSize: 50GiB

    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 100Gi

    resources:
      limits:
        cpu: 2000m
        memory: 8Gi
      requests:
        cpu: 1000m
        memory: 4Gi

    ruleSelector:
      matchLabels:
        prometheus: kube-prometheus

    serviceMonitorSelector:
      matchLabels:
        team: platform

    additionalScrapeConfigs:
      - job_name: "kubernetes-pods"
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels:
              [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels:
              [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__

grafana:
  adminPassword: ${GRAFANA_ADMIN_PASSWORD}

  persistence:
    enabled: true
    storageClassName: gp3
    size: 10Gi

  resources:
    limits:
      cpu: 500m
      memory: 1Gi
    requests:
      cpu: 250m
      memory: 512Mi

  grafana.ini:
    server:
      root_url: https://grafana.example.com
    auth.github:
      enabled: true
      allow_sign_up: true
      client_id: ${GITHUB_CLIENT_ID}
      client_secret: ${GITHUB_CLIENT_SECRET}
      scopes: user:email,read:org
      auth_url: https://github.com/login/oauth/authorize
      token_url: https://github.com/login/oauth/access_token
      api_url: https://api.github.com/user
      allowed_organizations: mycompany

  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
        - name: "default"
          orgId: 1
          folder: ""
          type: file
          disableDeletion: false
          editable: true
          options:
            path: /var/lib/grafana/dashboards/default

  dashboards:
    default:
      kubernetes-cluster-monitoring:
        gnetId: 315
        revision: 3
        datasource: Prometheus
      node-exporter:
        gnetId: 1860
        revision: 27
        datasource: Prometheus
      nginx-ingress:
        gnetId: 9614
        revision: 1
        datasource: Prometheus

alertmanager:
  alertmanagerSpec:
    storage:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 10Gi

  config:
    global:
      slack_api_url: "${SLACK_API_URL}"

    route:
      group_by: ["alertname", "cluster", "service"]
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 12h
      receiver: "default"
      routes:
        - match:
            severity: critical
          receiver: "critical-alerts"
        - match:
            severity: warning
          receiver: "warning-alerts"

    receivers:
      - name: "default"
        slack_configs:
          - channel: "#alerts"
            title: "Kubernetes Alert"
            text: "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"

      - name: "critical-alerts"
        slack_configs:
          - channel: "#critical-alerts"
            title: "CRITICAL: {{ .GroupLabels.alertname }}"
            text: "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
        pagerduty_configs:
          - service_key: "${PAGERDUTY_SERVICE_KEY}"

      - name: "warning-alerts"
        slack_configs:
          - channel: "#warnings"
            title: "Warning: {{ .GroupLabels.alertname }}"
            text: "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
```

### Custom Monitoring Setup

```yaml
# Application monitoring
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: webapp-metrics
  labels:
    team: platform
spec:
  selector:
    matchLabels:
      app: webapp
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics

---
# Custom alerting rules
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: webapp-alerts
  labels:
    prometheus: kube-prometheus
spec:
  groups:
    - name: webapp.rules
      rules:
        - alert: WebAppDown
          expr: up{job="webapp"} == 0
          for: 1m
          labels:
            severity: critical
          annotations:
            summary: "WebApp is down"
            description: "WebApp has been down for more than 1 minute"

        - alert: WebAppHighLatency
          expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="webapp"}[5m])) > 0.5
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "WebApp high latency"
            description: "95th percentile latency is above 500ms for 5 minutes"

        - alert: WebAppHighErrorRate
          expr: rate(http_requests_total{job="webapp",status=~"5.."}[5m]) / rate(http_requests_total{job="webapp"}[5m]) > 0.05
          for: 2m
          labels:
            severity: critical
          annotations:
            summary: "WebApp high error rate"
            description: "Error rate is above 5% for 2 minutes"
```

## 🎯 Interview Questions and Scenarios

### Practical Scenarios

````
Scenario 1: "Our production Kubernetes cluster is experiencing intermittent pod failures. Walk me through your troubleshooting approach."

Systematic Troubleshooting:
1. Immediate Assessment:
   kubectl get pods --all-namespaces | grep -v Running
   kubectl get events --sort-by=.metadata.creationTimestamp | tail -20
   kubectl top nodes
   kubectl describe nodes

2. Pod-Level Analysis:
   kubectl describe pod <failing-pod>
   kubectl logs <failing-pod> --previous
   kubectl get pod <failing-pod> -o yaml

3. Resource Investigation:
   - Check resource limits and requests
   - Verify node capacity
   - Examine storage issues
   - Network connectivity tests

4. Cluster Health:
   - Control plane status
   - etcd health
   - CNI plugin status
   - DNS resolution

5. Application-Specific:
   - Health check configuration
   - External dependencies
   - Configuration and secrets

Scenario 2: "Design a CI/CD pipeline for a microservices application with 10 services."

Pipeline Design:
1. Source Control Strategy:
   - Mono-repo vs multi-repo decision
   - Branch protection rules
   - Code review requirements

2. Build Strategy:
   - Parallel builds for changed services
   - Shared library management
   - Dependency tracking

3. Testing Strategy:
   - Unit tests per service
   - Integration tests for service interactions
   - Contract testing for API compatibility
   - End-to-end tests for critical paths

4. Deployment Strategy:
   - Service dependency management
   - Blue-green or canary deployments
   - Database migration handling
   - Rollback procedures

Example Implementation:
```yaml
# Pipeline configuration
stages:
  - detect-changes
  - parallel-builds
  - integration-tests
  - deploy-staging
  - e2e-tests
  - deploy-production

# Change detection
detect-changes:
  script:
    - ./scripts/detect-changes.sh > changed-services.txt
  artifacts:
    paths:
      - changed-services.txt

# Dynamic parallel builds
.build-template: &build-template
  stage: parallel-builds
  script:
    - docker build -t $CI_REGISTRY/$SERVICE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY/$SERVICE:$CI_COMMIT_SHA
  only:
    changes:
      - services/$SERVICE/**/*

user-service-build:
  <<: *build-template
  variables:
    SERVICE: user-service

product-service-build:
  <<: *build-template
  variables:
    SERVICE: product-service
# ... repeat for each service

# Service deployment with dependencies
deploy-services:
  stage: deploy-staging
  script:
    - ./scripts/deploy-with-dependencies.sh
  environment:
    name: staging
````

Scenario 3: "How would you implement zero-downtime deployments for a stateful application?"

Strategy:

1. Blue-Green Deployment for Stateless Components:
   - Maintain two identical environments
   - Switch traffic after health validation
   - Database connections need careful handling

2. Rolling Updates for Stateful Components:
   - StatefulSet with ordered updates
   - Persistent volume management
   - Data consistency validation

3. Database Migration Strategy:
   - Backward-compatible schema changes
   - Feature flags for new functionality
   - Multi-phase rollout

Implementation:

```yaml
# StatefulSet with rolling update
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: database
spec:
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
  template:
    spec:
      initContainers:
        - name: schema-migration
          image: migrate/migrate
          command:
            - migrate
            - -path=/migrations
            - -database=postgres://...
            - up
```

### Technical Deep Dive Questions

````
Q: "Explain the difference between Kubernetes Deployment and StatefulSet"

A: Comprehensive comparison:

Deployment (Stateless Applications):
- Pods are interchangeable
- Random pod naming (deployment-abc123-xyz)
- No stable network identity
- Shared storage or no persistent storage
- Parallel scaling and updates
- Use cases: Web servers, API services

StatefulSet (Stateful Applications):
- Pods have stable, unique identity
- Ordered pod naming (web-0, web-1, web-2)
- Stable network hostname
- Dedicated persistent volumes per pod
- Ordered, sequential scaling and updates
- Use cases: Databases, message queues

Example:
```yaml
# Deployment
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: web
        image: nginx
        volumeMounts:
        - name: shared-data
          mountPath: /data
      volumes:
      - name: shared-data
        emptyDir: {}

# StatefulSet
apiVersion: apps/v1
kind: StatefulSet
spec:
  serviceName: "database"
  replicas: 3
  template:
    spec:
      containers:
      - name: postgres
        image: postgres
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
````

Q: "How do you handle secrets in Kubernetes securely?"

A: Multi-layered security approach:

1. External Secret Management:
   - AWS Secrets Manager
   - Azure Key Vault
   - HashiCorp Vault
   - External Secrets Operator

2. Kubernetes Native Security:
   - Encryption at rest (etcd)
   - RBAC for secret access
   - Network policies
   - Pod security standards

3. Application-Level Security:
   - Minimal privilege principle
   - Secret rotation
   - Audit logging

Implementation:

```yaml
# External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "example-role"

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: app-secrets
spec:
  refreshInterval: 15s
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: app-secrets
    creationPolicy: Owner
  data:
    - secretKey: database-password
      remoteRef:
        key: app/database
        property: password
```

Q: "How do you implement monitoring for microservices?"

A: Comprehensive observability strategy:

1. The Three Pillars:
   - Metrics: Quantitative measurements
   - Logs: Detailed event records
   - Traces: Request flow tracking

2. Implementation Stack:
   - Prometheus for metrics
   - ELK/EFK stack for logs
   - Jaeger for distributed tracing

3. Best Practices:
   - Structured logging
   - Correlation IDs
   - Service mesh observability
   - Business metrics tracking

Example:

```javascript
// Application instrumentation
const promClient = require("prom-client");
const express = require("express");

// Metrics
const httpRequestDuration = new promClient.Histogram({
  name: "http_request_duration_seconds",
  help: "Duration of HTTP requests in seconds",
  labelNames: ["route", "method", "status_code"],
});

const httpRequestTotal = new promClient.Counter({
  name: "http_requests_total",
  help: "Total number of HTTP requests",
  labelNames: ["route", "method", "status_code"],
});

// Middleware
app.use((req, res, next) => {
  const start = Date.now();

  res.on("finish", () => {
    const duration = (Date.now() - start) / 1000;
    const route = req.route?.path || "unknown";

    httpRequestDuration.observe(
      { route, method: req.method, status_code: res.statusCode },
      duration,
    );

    httpRequestTotal.inc({
      route,
      method: req.method,
      status_code: res.statusCode,
    });
  });

  next();
});

// Metrics endpoint
app.get("/metrics", async (req, res) => {
  res.set("Content-Type", promClient.register.contentType);
  res.end(await promClient.register.metrics());
});
```

```

---

*Complete preparation guide for DevOps and containerization interviews with hands-on examples*
```
