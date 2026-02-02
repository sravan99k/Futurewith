# Model Deployment & Production Systems - Interview Questions & Answers

## Table of Contents

1. [Technical Questions (50+ questions)](#technical-questions)
2. [Coding Challenges (30+ questions)](#coding-challenges)
3. [Behavioral Questions (20+ questions)](#behavioral-questions)
4. [System Design Questions (15+ questions)](#system-design-questions)
5. [Answers and Explanations](#answers-and-explanations)

---

## Technical Questions

### MLOps Fundamentals (Questions 1-15)

**Q1. What is MLOps and how does it differ from traditional DevOps?**
**Difficulty: Intermediate**
_Answer: MLOps (Machine Learning Operations) extends DevOps principles to include machine learning workflows. Key differences include: data dependency management, model versioning, experiment tracking, model monitoring for data drift, and additional deployment considerations for AI models._

**Q2. Explain the ML model lifecycle and where MLOps fits in.**
**Difficulty: Intermediate**
_Answer: The ML lifecycle includes: Data Collection → Data Preparation → Model Training → Model Evaluation → Model Deployment → Monitoring → Retraining. MLOps provides the infrastructure and processes to automate and manage transitions between these stages._

**Q3. What are the key components of an MLOps pipeline?**
**Difficulty: Intermediate**
_Answer: Data validation, feature engineering, model training, model validation, model packaging, deployment, monitoring, and retraining. Each component needs to be automated, versioned, and monitored._

**Q4. How do you handle data drift detection in production?**
**Difficulty: Intermediate**
_Answer: Implement statistical tests (Kolmogorov-Smirnov, Chi-square), monitor feature distributions, use concept drift detection algorithms, set up alerts for significant changes, and establish retraining triggers._

**Q5. What is model versioning and why is it important?**
**Difficulty: Beginner**
_Answer: Model versioning tracks changes to models, datasets, and code. It's crucial for: reproducing results, rollback capabilities, A/B testing, compliance, and understanding model performance over time._

**Q6. Explain the difference between online and offline model training.**
**Difficulty: Intermediate**
_Answer: Offline training uses historical batch data to train models periodically. Online training updates models incrementally as new data arrives, requiring real-time learning capabilities and robust infrastructure._

**Q7. What is feature store and why use it?**
**Difficulty: Intermediate**
_Answer: A feature store is a centralized repository for storing, managing, and serving machine learning features. Benefits include: consistency between training and serving, feature reuse, governance, and real-time serving capabilities._

**Q8. How do you ensure reproducibility in ML experiments?**
**Difficulty: Intermediate**
_Answer: Set random seeds, use version control for code/data/models, containerize environments, track all hyperparameters, log all outputs, and use experiment tracking tools (MLflow, Weights & Biases)._

**Q9. What is A/B testing for ML models?**
**Difficulty: Intermediate**
_Answer: A/B testing compares two or more model versions by serving them to different user segments. It helps determine which model performs better based on specific metrics like accuracy, revenue, or user engagement._

**Q10. Explain model serving patterns and their trade-offs.**
**Difficulty: Advanced**
_Answer: Patterns include: batch serving (scheduled predictions), real-time serving (API endpoints), stream processing (Kafka/Kinesis), and edge deployment. Trade-offs involve latency, cost, complexity, and update frequency._

**Q11. What is the difference between warm start and cold start in model deployment?**
**Difficulty: Intermediate**
_Answer: Cold start requires loading model from scratch (high latency, low memory usage). Warm start keeps model loaded in memory (low latency, higher memory usage). Choose based on request frequency and resource constraints._

**Q12. How do you handle model updates without downtime?**
**Difficulty: Advanced**
_Answer: Use blue-green deployments, canary releases, or rolling updates. Implement health checks, ensure model compatibility, use feature flags, and have rollback mechanisms ready._

**Q13. What is shadow deployment?**
**Difficulty: Intermediate**
_Answer: Shadow deployment runs the new model in parallel with the production model without affecting user traffic. The new model receives the same inputs but predictions aren't used, allowing performance comparison before full deployment._

**Q14. Explain model registry and its importance.**
**Difficulty: Intermediate**
_Answer: A model registry is a centralized repository for model metadata, versioning, and lifecycle management. It enables collaboration, governance, compliance, and automated deployment workflows._

**Q15. What are the key metrics to monitor in production ML systems?**
**Difficulty: Intermediate**
_Answer: Model performance metrics (accuracy, precision, recall), business metrics, data quality metrics, system performance (latency, throughput), resource utilization, and cost metrics._

### Deployment Strategies (Questions 16-25)

**Q16. Compare blue-green vs. canary deployment for ML models.**
**Difficulty: Advanced**
_Answer: Blue-green maintains two identical environments switching traffic between them. Canary deploys to small percentage of traffic first. Blue-green is simpler but uses more resources. Canary reduces risk but requires sophisticated routing._

**Q17. What is multi-armed bandit approach for model deployment?**
**Difficulty: Advanced**
_Answer: Multi-armed bandit algorithms automatically route traffic between model variants based on performance, optimizing for the best-performing model while learning. More sophisticated than static A/B testing._

**Q18. Explain shadow mode deployment and when to use it.**
**Difficulty: Intermediate**
_Answer: Shadow mode runs the new model alongside production without affecting user decisions. Use it when: testing high-risk models, validating performance with real data, or preparing for gradual rollout._

**Q19. What is the difference between rolling deployment and recreate deployment?**
**Difficulty: Intermediate**
_Answer: Rolling deployment gradually replaces old instances with new ones. Recreate deployment stops all old instances before starting new ones. Rolling provides zero downtime but takes longer; recreate is faster but has brief downtime._

**Q20. How do you handle model compatibility during updates?**
**Difficulty: Advanced**
_Answer: Maintain API versioning, ensure backward compatibility, implement contract testing, use feature flags, maintain both old and new model versions during transition, and have comprehensive testing._

**Q21. What is progressive delivery in ML context?**
**Difficulty: Advanced**
_Answer: Progressive delivery gradually releases new models to users through stages: development → testing → shadow mode → canary → full rollout. Each stage has success criteria before proceeding to the next._

**Q22. Explain feature flags in model deployment.**
**Difficulty: Intermediate**
_Answer: Feature flags allow toggling model features on/off without code deployment. They enable: A/B testing, gradual rollouts, quick rollbacks, and controlled feature exposure to user segments._

**Q23. What is dark launching for ML models?**
**Difficulty: Intermediate**
_Answer: Dark launching deploys new features to production but keeps them hidden from users. Predictions are computed and logged but not used for decisions, allowing real-world testing without user impact._

**Q24. How do you implement automatic rollback for failed deployments?**
**Difficulty: Advanced**
_Answer: Set up monitoring for key metrics, define rollback triggers, maintain previous model versions, implement health checks, use deployment orchestration tools, and automate rollback procedures._

**Q25. What is the difference between horizontal and vertical scaling for ML systems?**
**Difficulty: Intermediate**
_Answer: Horizontal scaling adds more instances (scaling out), vertical scaling increases resources per instance (scaling up). Horizontal provides better fault tolerance and load distribution; vertical is simpler but has limits._

### Production Systems (Questions 26-35)

**Q26. How do you handle high-throughput prediction requests?**
**Difficulty: Advanced**
_Answer: Implement request batching, use async processing, employ caching strategies, use load balancers, implement rate limiting, consider model quantization, and use distributed serving frameworks._

**Q27. What is model distillation and when is it useful?**
**Difficulty: Advanced**
_Answer: Model distillation trains a smaller "student" model to mimic a larger "teacher" model. Useful when: reducing inference latency, decreasing memory footprint, or deploying on resource-constrained devices._

**Q28. Explain model compression techniques for production deployment.**
**Difficulty: Advanced**
_Answer: Techniques include: pruning (removing unnecessary weights), quantization (reducing precision), distillation, low-rank factorization, and knowledge distillation. Trade-offs involve accuracy vs. efficiency._

**Q29. How do you implement model serving on edge devices?**
**Difficulty: Advanced**
_Answer: Use model quantization, apply pruning, employ specialized frameworks (TensorFlow Lite, ONNX Runtime), optimize for specific hardware, implement efficient batching, and consider model splitting._

**Q30. What is the role of load balancing in ML serving systems?**
**Difficulty: Intermediate**
_Answer: Load balancing distributes inference requests across multiple model instances, ensuring optimal resource utilization, improving response times, providing fault tolerance, and enabling horizontal scaling._

**Q31. How do you handle different prediction formats across model versions?**
**Difficulty: Advanced**
_Answer: Implement API versioning, use transformation layers, maintain compatibility bridges, version request/response schemas, and use serialization/deserialization strategies._

**Q32. What is the difference between synchronous and asynchronous prediction?**
**Difficulty: Intermediate**
_Answer: Synchronous returns predictions immediately (low latency requirement). Asynchronous queues requests and returns later (higher throughput, can handle batch processing, enables complex computations)._

**Q33. How do you implement model caching strategies?**
**Difficulty: Advanced**
_Answer: Cache based on: input features, model version, request patterns. Use techniques like: LRU eviction, TTL expiration, feature-based caching, and distributed caching systems like Redis._

**Q34. What is the role of message queues in ML deployment?**
**Difficulty: Intermediate**
_Answer: Message queues enable: async processing, load buffering, fault tolerance, decoupling of components, batch processing, and reliable message delivery in ML pipelines._

**Q35. How do you handle prediction failures gracefully?**
**Difficulty: Advanced**
_Answer: Implement circuit breakers, fallback strategies, retry logic with exponential backoff, error logging, alerting systems, graceful degradation, and user notification mechanisms._

### Monitoring & Observability (Questions 36-50)

**Q36. What are the key components of ML monitoring?**
**Difficulty: Intermediate**
_Answer: Model performance monitoring, data drift detection, system performance tracking, business metrics monitoring, feature store monitoring, and infrastructure health checks._

**Q37. How do you set up alerts for model performance degradation?**
**Difficulty: Advanced**
_Answer: Define performance thresholds, use statistical process control, implement gradual degradation alerts, set up multi-level alerts (warning/critical), create dashboard visualizations, and integrate with incident management systems._

**Q38. Explain the concept of service level objectives (SLOs) for ML systems.**
**Difficulty: Advanced**
_Answer: SLOs define target performance levels for ML systems: latency SLOs (p99 < 100ms), availability SLOs (99.9% uptime), accuracy SLOs (maintain >90% accuracy), and throughput SLOs (1000 requests/sec)._

**Q39. What is the difference between monitoring and observability in ML?**
**Difficulty: Intermediate**
_Answer: Monitoring tracks predefined metrics and thresholds. Observability enables understanding of system behavior through logs, traces, and metrics exploration, particularly for unexpected issues._

**Q40. How do you monitor data quality in production?**
**Difficulty: Advanced**
_Answer: Implement data validation, track data distribution changes, monitor missing values and outliers, set up data lineage tracking, implement freshness checks, and create data quality scorecards._

**Q41. What is canary analysis in ML monitoring?**
**Difficulty: Advanced**
_Answer: Canary analysis evaluates new model versions using production traffic by comparing metrics between canary and control groups, detecting performance issues before full deployment._

**Q42. How do you implement distributed tracing for ML pipelines?**
**Difficulty: Advanced**
_Answer: Use distributed tracing systems (Jaeger, Zipkin), implement correlation IDs, trace data flow through pipeline stages, monitor latency at each step, and identify bottlenecks._

**Q43. What is the role of dashboards in ML operations?**
**Difficulty: Intermediate**
_Answer: Dashboards provide: real-time system health visibility, model performance trends, data quality metrics, infrastructure utilization, and enable quick problem identification and decision making._

**Q44. How do you handle log aggregation in ML systems?**
**Difficulty: Advanced**
_Answer: Use centralized logging (ELK stack, Splunk), structure logs with consistent formats, implement log sampling for high-volume systems, create searchable indices, and set up log-based alerts._

**Q45. What is model explainability monitoring?**
**Difficulty: Advanced**
_Answer: Monitor model explanations (SHAP, LIME values) to ensure they're stable, consistent, and align with domain knowledge. Track explanation drift and fairness metrics._

**Q46. How do you implement cost monitoring for ML deployments?**
**Difficulty: Intermediate**
_Answer: Track infrastructure costs (compute, storage, network), cost per prediction, model training costs, and set up cost budgets and alerts. Consider total cost of ownership (TCO)._

**Q47. What is the difference between white-box and black-box monitoring?**
**Difficulty: Intermediate**
_Answer: White-box monitoring uses internal system metrics (application logs, internal metrics). Black-box monitoring tests external behavior (end-to-end testing, synthetic transactions)._

**Q48. How do you implement health checks for ML services?**
**Difficulty: Advanced**
_Answer: Implement: model availability checks, data quality validation, dependency health checks, resource utilization monitoring, and response time measurements. Design health endpoints for load balancers._

**Q49. What is the role of synthetic data in monitoring?**
**Difficulty: Intermediate**
_Answer: Synthetic data tests model performance without using real user data. Useful for: testing edge cases, validating model behavior, monitoring system health, and ensuring disaster recovery readiness._

**Q50. How do you set up incident management for ML systems?**
**Difficulty: Advanced**
_Answer: Create incident response runbooks, define severity levels, implement automated alerts, establish on-call rotations, document incident post-mortems, and implement continuous improvement processes._

---

## Coding Challenges

### Docker and Containerization (Questions 1-10)

**Challenge 1: Create a production-ready Docker image for an ML model**
**Difficulty: Intermediate**

Create a Dockerfile for serving a TensorFlow model with proper optimization:

```dockerfile
# Multi-stage build for TensorFlow model serving
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p /app/models && chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "app:app"]
```

**Challenge 2: Optimize Docker image size and build time**
**Difficulty: Advanced**

```dockerfile
# Use alpine base for smaller image
FROM python:3.9-alpine as builder

# Install build dependencies
RUN apk add --no-cache gcc musl-dev

# Install Python packages
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production image with specific version
FROM python:3.9-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY --from=builder /root/.local /home/app/.local
COPY app/ ./app/

# Add to PATH
ENV PATH=/home/app/.local/bin:$PATH

# Use non-root user
USER app

EXPOSE 8080
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
```

**Challenge 3: Implement health checks for ML service**
**Difficulty: Intermediate**

```python
# health_check.py
import requests
import logging
import time
from typing import Dict, Any

class MLModelHealthCheck:
    def __init__(self, model_endpoint: str, timeout: int = 5):
        self.model_endpoint = model_endpoint
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def check_model_health(self) -> Dict[str, Any]:
        """Check if model service is healthy"""
        start_time = time.time()

        try:
            # Check if service is responding
            response = requests.get(
                f"{self.model_endpoint}/health",
                timeout=self.timeout
            )
            response.raise_for_status()

            # Check if model is loaded
            model_status = requests.get(
                f"{self.model_endpoint}/model/status",
                timeout=self.timeout
            )
            model_status.raise_for_status()

            latency = time.time() - start_time

            return {
                "status": "healthy",
                "latency": latency,
                "service_responsive": True,
                "model_loaded": True,
                "timestamp": time.time()
            }

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "service_responsive": False,
                "model_loaded": False,
                "timestamp": time.time()
            }

    def check_data_freshness(self, max_age_minutes: int = 60) -> Dict[str, Any]:
        """Check if training data is fresh"""
        try:
            response = requests.get(
                f"{self.model_endpoint}/data/freshness",
                timeout=self.timeout
            )
            response.raise_for_status()

            data_info = response.json()
            last_updated = data_info.get("last_updated")

            if last_updated:
                age_minutes = (time.time() - last_updated) / 60
                return {
                    "status": "fresh" if age_minutes < max_age_minutes else "stale",
                    "age_minutes": age_minutes,
                    "max_age_minutes": max_age_minutes
                }

            return {"status": "unknown"}

        except Exception as e:
            return {"status": "check_failed", "error": str(e)}

# FastAPI health check endpoint
from fastapi import FastAPI

app = FastAPI()
health_checker = MLModelHealthCheck("http://localhost:8080")

@app.get("/health")
async def health_check():
    health_status = health_checker.check_model_health()
    status_code = 200 if health_status["status"] == "healthy" else 503
    return health_status
```

**Challenge 4: Create Kubernetes deployment with resource limits**
**Difficulty: Advanced**

```yaml
# ml-model-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
  labels:
    app: ml-model
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
        version: v1
    spec:
      containers:
        - name: ml-model
          image: ml-model:latest
          ports:
            - containerPort: 8080
              name: http
          env:
            - name: MODEL_PATH
              value: "/app/models/model.pkl"
            - name: LOG_LEVEL
              value: "INFO"
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          volumeMounts:
            - name: model-storage
              mountPath: /app/models
              readOnly: true
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: model-pvc
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
    - port: 80
      targetPort: 8080
      protocol: TCP
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
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
```

**Challenge 5: Implement blue-green deployment**
**Difficulty: Advanced**

```yaml
# blue-green-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: ml-model-rollout
spec:
  replicas: 10
  strategy:
    blueGreen:
      activeService: ml-model-active
      previewService: ml-model-preview
      autoPromotionEnabled: false
      scaleDownDelaySeconds: 30
      prePromotionAnalysis:
        templates:
          - templateName: success-rate
        args:
          - name: service-name
            value: ml-model-preview.default.svc.cluster.local
      postPromotionAnalysis:
        templates:
          - templateName: success-rate
        args:
          - name: service-name
            value: ml-model-active.default.svc.cluster.local
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
        - name: ml-model
          image: ml-model:latest
          ports:
            - containerPort: 8080
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  args:
    - name: service-name
  metrics:
    - name: success-rate
      interval: "1m"
      successCondition: result[0] >= 0.95
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: |
            sum(rate(http_requests_total{service="{{args.service-name}}",status!~"5.."}[2m])) /
            sum(rate(http_requests_total{service="{{args.service-name}}"}[2m]))
```

**Challenge 6: Create CI/CD pipeline for ML model**
**Difficulty: Advanced**

```yaml
# .github/workflows/ml-cicd.yml
name: ML Model CI/CD

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

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests
        run: |
          pytest tests/ --cov=app --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: "python:3.9-slim"
          format: "sarif"
          output: "trivy-results.sarif"

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: "trivy-results.sarif"

  build:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

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
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          echo "Deploying image ${{ needs.build.outputs.image-tag }} to staging"
          # kubectl set image deployment/ml-model ml-model=${{ needs.build.outputs.image-tag }}
          # kubectl rollout status deployment/ml-model

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying image ${{ needs.build.outputs.image-tag }} to production"
          # Implement blue-green or canary deployment
          # kubectl create deployment blue-green-deployment \
          #   --image=${{ needs.build.outputs.image-tag }} \
          #   --namespace=production
```

**Challenge 7: Implement model versioning system**
**Difficulty: Advanced**

```python
# model_versioning.py
import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import boto3
import mlflow
import mlflow.sklearn

@dataclass
class ModelVersion:
    version_id: str
    model_path: str
    metadata: Dict[str, Any]
    created_at: datetime
    created_by: str
    checksum: str
    parent_version: Optional[str] = None

class ModelRegistry:
    def __init__(self, registry_path: str, s3_bucket: Optional[str] = None):
        self.registry_path = registry_path
        self.s3_bucket = s3_bucket
        self.logger = logging.getLogger(__name__)

    def register_model(self,
                      model_path: str,
                      metadata: Dict[str, Any],
                      created_by: str,
                      parent_version: Optional[str] = None) -> ModelVersion:
        """Register a new model version"""

        # Calculate checksum
        checksum = self._calculate_checksum(model_path)

        # Generate version ID
        version_id = self._generate_version_id(checksum, metadata)

        # Create model version
        model_version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            metadata=metadata,
            created_at=datetime.utcnow(),
            created_by=created_by,
            checksum=checksum,
            parent_version=parent_version
        )

        # Upload to S3 if configured
        if self.s3_bucket:
            self._upload_to_s3(model_path, version_id)

        # Save metadata
        self._save_metadata(model_version)

        # Log to MLflow
        self._log_to_mlflow(model_version, model_path)

        self.logger.info(f"Model registered: {version_id}")
        return model_version

    def get_model_version(self, version_id: str) -> Optional[ModelVersion]:
        """Retrieve model version by ID"""
        metadata_path = os.path.join(self.registry_path, f"{version_id}.json")

        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, 'r') as f:
            data = json.load(f)
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            return ModelVersion(**data)

    def list_versions(self, limit: int = 100) -> List[ModelVersion]:
        """List all model versions"""
        versions = []

        for filename in os.listdir(self.registry_path):
            if filename.endswith('.json'):
                version = self.get_model_version(filename[:-5])
                if version:
                    versions.append(version)

        # Sort by creation date, newest first
        versions.sort(key=lambda x: x.created_at, reverse=True)
        return versions[:limit]

    def promote_model(self, version_id: str, environment: str) -> bool:
        """Promote model to specific environment"""
        model_version = self.get_model_version(version_id)
        if not model_version:
            return False

        # Update environment metadata
        model_version.metadata[f"promoted_to_{environment}"] = datetime.utcnow().isoformat()
        self._save_metadata(model_version)

        self.logger.info(f"Model {version_id} promoted to {environment}")
        return True

    def _calculate_checksum(self, model_path: str) -> str:
        """Calculate SHA256 checksum of model file"""
        hash_sha256 = hashlib.sha256()

        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    def _generate_version_id(self, checksum: str, metadata: Dict[str, Any]) -> str:
        """Generate unique version ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_name = metadata.get("model_name", "unknown")
        return f"{model_name}_{timestamp}_{checksum[:8]}"

    def _upload_to_s3(self, model_path: str, version_id: str):
        """Upload model to S3"""
        s3_client = boto3.client('s3')
        s3_key = f"models/{version_id}/model.pkl"

        s3_client.upload_file(model_path, self.s3_bucket, s3_key)
        self.logger.info(f"Model uploaded to s3://{self.s3_bucket}/{s3_key}")

    def _save_metadata(self, model_version: ModelVersion):
        """Save model version metadata"""
        os.makedirs(self.registry_path, exist_ok=True)
        metadata_path = os.path.join(self.registry_path, f"{model_version.version_id}.json")

        # Convert datetime to ISO format for JSON serialization
        data = asdict(model_version)
        data['created_at'] = model_version.created_at.isoformat()

        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _log_to_mlflow(self, model_version: ModelVersion, model_path: str):
        """Log model to MLflow"""
        try:
            with mlflow.start_run():
                # Log parameters
                for key, value in model_version.metadata.items():
                    if isinstance(value, (int, float, str)):
                        mlflow.log_param(key, value)

                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model_path,
                    artifact_path="model",
                    registered_model_name=model_version.metadata.get("model_name")
                )

                # Log version ID as tag
                mlflow.set_tag("version_id", model_version.version_id)

        except Exception as e:
            self.logger.warning(f"Failed to log to MLflow: {e}")

# Example usage
if __name__ == "__main__":
    registry = ModelRegistry("./model_registry", s3_bucket="my-ml-models-bucket")

    # Register new model version
    model_version = registry.register_model(
        model_path="./models/production_model.pkl",
        metadata={
            "model_name": "fraud_detection",
            "version": "1.2.0",
            "accuracy": 0.95,
            "training_data_date": "2023-10-01"
        },
        created_by="data_scientist_123"
    )

    # Promote to staging
    registry.promote_model(model_version.version_id, "staging")

    # List all versions
    for version in registry.list_versions(10):
        print(f"Version: {version.version_id}, Created: {version.created_at}")
```

**Challenge 8: Implement model monitoring service**
**Difficulty: Advanced**

```python
# model_monitoring.py
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, start_http_server

@dataclass
class MetricData:
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

class ModelMetrics:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Prometheus metrics
        self.request_count = Counter(
            f'{model_name}_requests_total',
            'Total number of requests',
            ['method', 'status']
        )

        self.request_duration = Histogram(
            f'{model_name}_request_duration_seconds',
            'Request duration in seconds',
            ['method']
        )

        self.model_accuracy = Gauge(
            f'{model_name}_accuracy',
            'Model accuracy score'
        )

        self.prediction_latency = Histogram(
            f'{model_name}_prediction_latency_seconds',
            'Time spent on predictions'
        )

        self.active_models = Gauge(
            f'{model_name}_active_models',
            'Number of active model instances'
        )

        # Internal storage for trend analysis
        self.metric_history: List[MetricData] = []
        self.performance_thresholds = {
            'accuracy_min': 0.85,
            'latency_max': 1.0,
            'error_rate_max': 0.05
        }

    def record_request(self, method: str, status: str, duration: float):
        """Record API request metrics"""
        self.request_count.labels(method=method, status=status).inc()
        self.request_duration.labels(method=method).observe(duration)

    def record_prediction(self, prediction_time: float, accuracy: Optional[float] = None):
        """Record prediction metrics"""
        self.prediction_latency.observe(prediction_time)

        if accuracy is not None:
            self.model_accuracy.set(accuracy)
            self.metric_history.append(MetricData(
                timestamp=time.time(),
                value=accuracy,
                labels={'metric_type': 'accuracy'}
            ))

    def record_error(self, error_type: str, context: Dict[str, Any] = None):
        """Record model errors"""
        self.logger.error(f"Model error: {error_type}", extra=context)

        # Could add error rate monitoring
        error_labels = {'error_type': error_type}
        # self.error_count.labels(**error_labels).inc()

    def check_performance_thresholds(self) -> List[Dict[str, Any]]:
        """Check if metrics exceed defined thresholds"""
        alerts = []
        current_time = time.time()

        # Check accuracy trend
        recent_accuracy = [
            m.value for m in self.metric_history
            if m.labels.get('metric_type') == 'accuracy'
            and current_time - m.timestamp < 3600  # Last hour
        ]

        if recent_accuracy:
            avg_accuracy = np.mean(recent_accuracy[-10:])  # Last 10 measurements
            if avg_accuracy < self.performance_thresholds['accuracy_min']:
                alerts.append({
                    'type': 'accuracy_degradation',
                    'current_value': avg_accuracy,
                    'threshold': self.performance_thresholds['accuracy_min'],
                    'severity': 'high'
                })

        # Add more threshold checks...

        return alerts

class ModelMonitoringService:
    def __init__(self, models: List[str], monitoring_interval: int = 60):
        self.models = {name: ModelMetrics(name) for name in models}
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)
        self.running = False

    async def start_monitoring(self, port: int = 8000):
        """Start the monitoring service"""
        self.running = True

        # Start Prometheus metrics server
        start_http_server(port)
        self.logger.info(f"Metrics server started on port {port}")

        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop the monitoring service"""
        self.running = False

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                for model_name, metrics in self.models.items():
                    await self._check_model_health(model_name, metrics)

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def _check_model_health(self, model_name: str, metrics: ModelMetrics):
        """Check individual model health"""
        try:
            # Simulate health check
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{model_name}:8080/health") as response:
                    if response.status == 200:
                        health_data = await response.json()

                        # Record system metrics
                        metrics.active_models.set(health_data.get('active_instances', 1))

                        # Check for alerts
                        alerts = metrics.check_performance_thresholds()
                        for alert in alerts:
                            await self._handle_alert(model_name, alert)

        except Exception as e:
            metrics.record_error("health_check_failed", {"error": str(e)})

    async def _handle_alert(self, model_name: str, alert: Dict[str, Any]):
        """Handle performance alerts"""
        alert_message = f"Alert for {model_name}: {alert}"
        self.logger.warning(alert_message)

        # Could send to external alerting system
        # await self._send_to_slack(alert_message)
        # await self._send_to_pagerduty(alert)

    def get_model_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get current metrics for a model"""
        if model_name not in self.models:
            return {}

        metrics = self.models[model_name]

        return {
            "model_name": model_name,
            "request_count": self._get_counter_value(metrics.request_count),
            "avg_latency": self._get_histogram_avg(metrics.request_duration),
            "accuracy": metrics.model_accuracy._value._value,
            "active_models": metrics.active_models._value._value,
            "recent_alerts": metrics.check_performance_thresholds()
        }

    def _get_counter_value(self, counter) -> int:
        """Get total counter value"""
        return sum(counter._value._values.values())

    def _get_histogram_avg(self, histogram) -> float:
        """Calculate average from histogram"""
        total_count = sum(histogram._sum._value._values.values())
        total_sum = sum(histogram._count._value._values.values())
        return total_sum / total_count if total_count > 0 else 0

# FastAPI integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
monitoring_service = ModelMonitoringService(["model1", "model2"])

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    confidence: float
    processing_time: float

@app.on_event("startup")
async def startup_event():
    await monitoring_service.start_monitoring()

@app.post("/predict/{model_name}")
async def predict(model_name: str, request: PredictionRequest):
    start_time = time.time()

    try:
        # Simulate model prediction
        prediction = np.random.random()
        processing_time = time.time() - start_time

        # Record metrics
        metrics = monitoring_service.models.get(model_name)
        if metrics:
            metrics.record_prediction(processing_time, accuracy=0.9)
            metrics.record_request("POST", "200", processing_time)

        return PredictionResponse(
            prediction=prediction,
            model_version="v1.2.0",
            confidence=0.95,
            processing_time=processing_time
        )

    except Exception as e:
        processing_time = time.time() - start_time
        metrics = monitoring_service.models.get(model_name)
        if metrics:
            metrics.record_request("POST", "500", processing_time)
            metrics.record_error("prediction_failed", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    all_metrics = {}
    for model_name in monitoring_service.models:
        all_metrics[model_name] = monitoring_service.get_model_metrics(model_name)
    return all_metrics
```

**Challenge 9: Implement model A/B testing framework**
**Difficulty: Expert**

```python
# ab_testing.py
import random
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

@dataclass
class ABTestConfig:
    test_name: str
    model_a: str  # Control model
    model_b: str  # Treatment model
    traffic_split: float = 0.5  # Percentage of traffic to model B
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    test_duration_days: int = 7
    success_metric: str = "accuracy"
    hypothesis: str = "Model B performs better than Model A"

@dataclass
class TestResult:
    test_id: str
    config: ABTestConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, stopped
    results: Dict[str, Any] = field(default_factory=dict)

class ABTestManager:
    def __init__(self, model_router: Callable):
        self.model_router = model_router
        self.active_tests: Dict[str, TestResult] = {}
        self.test_results: Dict[str, List[Dict]] = {}
        self.logger = logging.getLogger(__name__)

    def create_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test"""
        test_id = f"{config.test_name}_{int(time.time())}"

        test_result = TestResult(
            test_id=test_id,
            config=config,
            start_time=datetime.utcnow()
        )

        self.active_tests[test_id] = test_result
        self.test_results[test_id] = []

        self.logger.info(f"Created A/B test: {test_id}")
        return test_id

    def route_request(self, test_id: str, features: List[float]) -> Dict[str, Any]:
        """Route request to appropriate model based on A/B test"""
        if test_id not in self.active_tests:
            return self.model_router(features)

        test_config = self.active_tests[test_id].config

        # Determine which model to use
        if random.random() < test_config.traffic_split:
            selected_model = test_config.model_b
            variant = "B"
        else:
            selected_model = test_config.model_a
            variant = "A"

        # Get prediction from selected model
        prediction = self.model_router(features, model_name=selected_model)

        # Record test result
        self._record_result(test_id, variant, prediction, features)

        return {
            **prediction,
            "test_id": test_id,
            "variant": variant,
            "model_used": selected_model
        }

    def _record_result(self, test_id: str, variant: str, prediction: Dict, features: List[float]):
        """Record test result for analysis"""
        result = {
            "timestamp": datetime.utcnow(),
            "variant": variant,
            "prediction": prediction.get("prediction"),
            "confidence": prediction.get("confidence", 0.0),
            "features": features,
            "processing_time": prediction.get("processing_time", 0.0)
        }

        self.test_results[test_id].append(result)

    def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        if test_id not in self.test_results:
            raise ValueError(f"Test {test_id} not found")

        results = self.test_results[test_id]
        if len(results) < self.active_tests[test_id].config.min_sample_size:
            return {"status": "insufficient_data", "sample_size": len(results)}

        # Separate results by variant
        variant_a_results = [r for r in results if r["variant"] == "A"]
        variant_b_results = [r for r in results if r["variant"] == "B"]

        # Calculate metrics
        analysis = {
            "test_id": test_id,
            "analysis_time": datetime.utcnow(),
            "total_samples": len(results),
            "variant_a_samples": len(variant_a_results),
            "variant_b_samples": len(variant_b_results),
            "metrics": {}
        }

        # Analyze success metric (e.g., accuracy, confidence, processing time)
        metric = self.active_tests[test_id].config.success_metric

        a_values = [r[metric] for r in variant_a_results if metric in r]
        b_values = [r[metric] for r in variant_b_results if metric in r]

        if a_values and b_values:
            # Statistical test (t-test)
            from scipy import stats

            t_stat, p_value = stats.ttest_ind(a_values, b_values)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(a_values) - 1) * np.var(a_values) +
                                (len(b_values) - 1) * np.var(b_values)) /
                               (len(a_values) + len(b_values) - 2))
            cohens_d = (np.mean(b_values) - np.mean(a_values)) / pooled_std

            # Confidence interval
            confidence_level = self.active_tests[test_id].config.confidence_level
            alpha = 1 - confidence_level

            # Mean difference confidence interval
            mean_diff = np.mean(b_values) - np.mean(a_values)
            se_diff = pooled_std * np.sqrt(1/len(a_values) + 1/len(b_values))
            df = len(a_values) + len(b_values) - 2
            t_critical = stats.t.ppf(1 - alpha/2, df)
            ci_lower = mean_diff - t_critical * se_diff
            ci_upper = mean_diff + t_critical * se_diff

            analysis["metrics"] = {
                "metric_name": metric,
                "variant_a_mean": np.mean(a_values),
                "variant_b_mean": np.mean(b_values),
                "mean_difference": mean_diff,
                "confidence_interval": (ci_lower, ci_upper),
                "p_value": p_value,
                "effect_size": cohens_d,
                "statistically_significant": p_value < alpha,
                "confidence_level": confidence_level
            }

            # Determine winner
            if analysis["metrics"]["statistically_significant"]:
                if np.mean(b_values) > np.mean(a_values):
                    analysis["winner"] = "Model B"
                    analysis["recommendation"] = "Deploy Model B"
                else:
                    analysis["winner"] = "Model A"
                    analysis["recommendation"] = "Keep Model A"
            else:
                analysis["winner"] = "No significant difference"
                analysis["recommendation"] = "Continue testing or choose based on other factors"

        return analysis

    def stop_test(self, test_id: str) -> str:
        """Stop an A/B test and provide final analysis"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test_result = self.active_tests[test_id]
        test_result.end_time = datetime.utcnow()
        test_result.status = "completed"

        # Perform final analysis
        final_analysis = self.analyze_test(test_id)
        test_result.results = final_analysis

        # Move to completed tests
        self.logger.info(f"Test {test_id} completed. Recommendation: {final_analysis.get('recommendation')}")

        return final_analysis.get("recommendation", "Analysis complete")

    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get current test status"""
        if test_id not in self.active_tests:
            return {"error": "Test not found"}

        test_result = self.active_tests[test_id]
        current_time = datetime.utcnow()
        duration = (current_time - test_result.start_time).total_seconds() / 3600  # hours

        return {
            "test_id": test_id,
            "test_name": test_result.config.test_name,
            "status": test_result.status,
            "start_time": test_result.start_time,
            "duration_hours": duration,
            "sample_size": len(self.test_results.get(test_id, [])),
            "traffic_split": test_result.config.traffic_split,
            "success_metric": test_result.config.success_metric
        }

    def list_active_tests(self) -> List[Dict[str, Any]]:
        """List all active tests"""
        return [self.get_test_status(test_id) for test_id in self.active_tests]

# Example usage with FastAPI
from fastapi import FastAPI, HTTPException
import asyncio

app = FastAPI()

# Mock model router (replace with actual model serving)
def model_router(features: List[float], model_name: str = "default") -> Dict[str, Any]:
    # Simulate model prediction
    prediction = np.random.random()
    confidence = 0.9 if model_name == "model_a" else 0.95
    processing_time = 0.1 + np.random.random() * 0.05

    return {
        "prediction": prediction,
        "confidence": confidence,
        "processing_time": processing_time,
        "model_name": model_name
    }

# Initialize A/B test manager
ab_test_manager = ABTestManager(model_router)

@app.post("/ab-tests/create")
async def create_ab_test(config: ABTestConfig):
    test_id = ab_test_manager.create_test(config)
    return {"test_id": test_id, "status": "created"}

@app.post("/ab-tests/{test_id}/predict")
async def ab_test_predict(test_id: str, request: PredictionRequest):
    try:
        result = ab_test_manager.route_request(test_id, request.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ab-tests/{test_id}/analyze")
async def analyze_test(test_id: str):
    try:
        analysis = ab_test_manager.analyze_test(test_id)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ab-tests/{test_id}/stop")
async def stop_test(test_id: str):
    try:
        recommendation = ab_test_manager.stop_test(test_id)
        return {"test_id": test_id, "recommendation": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ab-tests/active")
async def list_active_tests():
    return {"active_tests": ab_test_manager.list_active_tests()}
```

**Challenge 10: Implement model serving with auto-scaling**
**Difficulty: Expert**

```python
# auto_scaling_serving.py
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil
import ray
from ray import serve

class ModelServer:
    def __init__(self, model_path: str, scaling_config: Dict[str, Any] = None):
        self.model_path = model_path
        self.model = None
        self.request_count = 0
        self.scaling_config = scaling_config or {
            "min_replicas": 1,
            "max_replicas": 5,
            "target_utilization": 0.7,
            "scale_up_threshold": 0.8,
            "scale_down_threshold": 0.3,
            "check_interval": 30
        }
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize the model and Ray serve"""
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()

        # Create Ray serve deployment
        @serve.deployment
        class ModelDeployment:
            def __init__(self, model_path: str):
                self.model_path = model_path
                self.model = self._load_model()
                self.request_count = 0

            def _load_model(self):
                """Load the ML model"""
                # Replace with actual model loading
                import joblib
                try:
                    return joblib.load(self.model_path)
                except:
                    # Fallback to mock model
                    return None

            async def predict(self, features: List[float]) -> Dict[str, Any]:
                start_time = time.time()
                self.request_count += 1

                try:
                    if self.model:
                        prediction = self.model.predict([features])[0]
                    else:
                        # Mock prediction
                        prediction = np.random.random()

                    processing_time = time.time() - start_time

                    return {
                        "prediction": float(prediction),
                        "confidence": 0.95,
                        "processing_time": processing_time,
                        "model_version": "1.0.0"
                    }
                except Exception as e:
                    self.logger.error(f"Prediction error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

        # Deploy to Ray serve
        self.deployment = ModelDeployment.bind(model_path=self.model_path)
        handle = serve.run(self.deployment, route_prefix="/model")

        return handle

class AutoScaler:
    def __init__(self, model_server: ModelServer):
        self.model_server = model_server
        self.running = False
        self.logger = logging.getLogger(__name__)

    async def start_auto_scaling(self):
        """Start the auto-scaling loop"""
        self.running = True
        self.logger.info("Auto-scaling started")

        while self.running:
            try:
                await self._check_and_scale()
                await asyncio.sleep(self.model_server.scaling_config["check_interval"])
            except Exception as e:
                self.logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(10)

    async def stop_auto_scaling(self):
        """Stop auto-scaling"""
        self.running = False
        self.logger.info("Auto-scaling stopped")

    async def _check_and_scale(self):
        """Check current metrics and scale if needed"""
        # Get current metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        # Get request metrics (simplified)
        current_replicas = await self._get_current_replicas()

        # Calculate utilization
        utilization = (cpu_usage + memory_usage) / 2
        current_load = self._calculate_load()

        self.logger.debug(f"Current utilization: {utilization:.2%}, "
                         f"current replicas: {current_replicas}, "
                         f"current load: {current_load:.2f}")

        # Check if scaling is needed
        scaling_needed = False
        new_replica_count = current_replicas

        if (utilization > self.model_server.scaling_config["scale_up_threshold"] and
            current_replicas < self.model_server.scaling_config["max_replicas"]):
            scaling_needed = True
            new_replica_count = min(
                current_replicas + 1,
                self.model_server.scaling_config["max_replicas"]
            )
            self.logger.info(f"Scale up: {current_replicas} -> {new_replica_count}")

        elif (utilization < self.model_server.scaling_config["scale_down_threshold"] and
              current_replicas > self.model_server.scaling_config["min_replicas"]):
            scaling_needed = True
            new_replica_count = max(
                current_replicas - 1,
                self.model_server.scaling_config["min_replicas"]
            )
            self.logger.info(f"Scale down: {current_replicas} -> {new_replica_count}")

        # Apply scaling
        if scaling_needed:
            await self._scale_to(new_replica_count)

    async def _get_current_replicas(self) -> int:
        """Get current number of replicas"""
        try:
            # This is a simplified version - in practice, use Ray's metrics
            return 1  # Placeholder
        except Exception as e:
            self.logger.error(f"Error getting replica count: {e}")
            return 1

    async def _scale_to(self, replica_count: int):
        """Scale to specified number of replicas"""
        try:
            # Use Ray serve to scale
            serve.get_deployment("ModelDeployment").options(
                num_replicas=replica_count
            )
            self.logger.info(f"Scaled to {replica_count} replicas")
        except Exception as e:
            self.logger.error(f"Error scaling: {e}")

    def _calculate_load(self) -> float:
        """Calculate current system load"""
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        return (cpu_usage + memory_usage) / 2

# FastAPI application
app = FastAPI(title="ML Model Auto-Scaling Server")

# Global instances
model_server = None
auto_scaler = None

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    processing_time: float
    model_version: str

@app.on_event("startup")
async def startup_event():
    global model_server, auto_scaler

    # Initialize model server
    model_server = ModelServer("./models/production_model.pkl")
    handle = await model_server.initialize()

    # Start auto-scaling
    auto_scaler = AutoScaler(model_server)
    asyncio.create_task(auto_scaler.start_auto_scaling())

    # Start Prometheus metrics server
    start_http_server(8000)

    print("Model server started with auto-scaling enabled")

@app.on_event("shutdown")
async def shutdown_event():
    global auto_scaler

    if auto_scaler:
        await auto_scaler.stop_auto_scaling()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not ready")

    try:
        start_time = time.time()

        # Use Ray serve for prediction
        handle = serve.get_deployment("ModelDeployment").get_handle()
        result = await handle.predict.remote(request.features)

        processing_time = time.time() - start_time

        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            processing_time=processing_time,
            model_version=result["model_version"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": model_server is not None,
        "auto_scaling": auto_scaler.running if auto_scaler else False
    }

@app.get("/scaling/status")
async def get_scaling_status():
    """Get current scaling status"""
    if not auto_scaler:
        raise HTTPException(status_code=503, detail="Auto-scaler not available")

    return {
        "auto_scaling_enabled": auto_scaler.running,
        "current_utilization": (psutil.cpu_percent() + psutil.virtual_memory().percent) / 2,
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "scaling_config": auto_scaler.model_server.scaling_config
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## Behavioral Questions

### Production Scenarios (Questions 1-10)

**Q1. Describe a time when a model you deployed to production started performing poorly. How did you handle it?**
**Difficulty: Intermediate**
_Answer Framework:_

- _Situation:_ Describe the model performance degradation
- _Task:_ Explain your responsibility in resolving it
- _Action:_ Detail the steps you took (investigation, rollback, analysis)
- _Result:_ Share the outcome and lessons learned

_Sample Answer:_
_"I deployed a fraud detection model that initially showed 95% accuracy in testing. Within a week, the false positive rate increased by 40%. I immediately investigated and found data drift caused by new fraud patterns. I rolled back to the previous model, set up enhanced monitoring, and worked with the data team to retrain with recent data. We implemented automated drift detection and created a rapid retraining pipeline."_

**Q2. How do you handle conflicting priorities between model accuracy and system latency?**
**Difficulty: Intermediate**
_Answer:_
_"This requires careful trade-off analysis. First, I define the acceptable latency thresholds based on business requirements. Then I explore optimization strategies like model compression, caching, or batch processing. If trade-offs are unacceptable, I propose alternative architectures like tiered serving (fast approximate model + slower accurate model) or asynchronous processing for non-critical predictions."_

**Q3. Describe a situation where you had to make a go/no-go decision on deploying a model. What factors did you consider?**
**Difficulty: Intermediate**
_Key Factors:_

- Model performance vs. current production model
- Business impact and risk tolerance
- Technical complexity and resource requirements
- Monitoring and rollback capabilities
- User experience implications

_Answer Framework:_
_"I evaluate models across multiple dimensions: statistical performance, business metrics, technical readiness, and operational complexity. For a customer churn model, I recommended no-go because while accuracy improved 2%, the 200ms latency increase would impact user experience without proportional business value."_

**Q4. How do you communicate model limitations and risks to non-technical stakeholders?**
**Difficulty: Intermediate**
_Answer:_
_"I use business-focused explanations and visualizations. For example, instead of saying 'F1-score dropped by 5%', I say 'We expect 50 additional customer churns per month.' I create impact assessments showing both technical metrics and business consequences, use simple analogies for model behavior, and provide clear recommendations with risk mitigation strategies."_

**Q5. Tell me about a time when you had to debug a production ML system under pressure.**
**Difficulty: Advanced**
_Answer Framework:_

- _Problem:_ Describe the critical system issue
- _Urgency:_ Explain the business impact
- _Investigation:_ Detail your debugging approach
- _Resolution:_ Share the solution and timeline
- _Prevention:_ Explain improvements made

_Sample:_
_"Our recommendation system was serving random predictions due to a model versioning bug. Within 15 minutes, I identified the issue in the model registry, implemented a hotfix, and rolled back to the last known good version. I then created additional safeguards to prevent similar issues."_

**Q6. How do you prioritize which models to retrain when resources are limited?**
**Difficulty: Intermediate**
_Answer:_
_"I use a prioritization framework considering:_

- _Business Impact:_ Revenue impact and user reach
- _Performance Degradation:_ Current accuracy vs. acceptable thresholds
- _Data Freshness:_ How outdated the training data is
- _Resource Requirements:_ Training time and computational cost
- _Regulatory Compliance:_ Models requiring recent data for compliance\*

_Example: 'I prioritized retraining our payment fraud model over product recommendation because fraud directly impacts revenue and shows faster concept drift.'"_

**Q7. Describe your experience working with data engineers vs. software engineers. How do you bridge the gap?**
**Difficulty: Intermediate**
_Answer:_
_"Data engineers focus on data pipelines and schema evolution, while software engineers focus on API contracts and scalability. I bridge the gap by:_

- _Creating clear interfaces between data and model components_
- _Establishing common monitoring and alerting systems_
- _Using shared tools and platforms (MLflow, feature stores)_
- _Regular cross-team syncs on requirements and constraints_
- _Documenting data contracts and model expectations"_

**Q8. How do you handle model bias and fairness concerns in production?**
**Difficulty: Advanced**
_Answer:_
_"I implement multi-layered fairness monitoring:_

- _Pre-deployment:_ Bias testing across demographic groups
- _Production monitoring:_ Fairness metrics tracked alongside performance
- _Alert systems:_ Automated alerts for fairness degradation
- _Regular audits:_ Scheduled reviews of model decisions
- _Feedback loops:_ Mechanisms for users to challenge predictions
- _Documentation:_ Clear explanations of model limitations and intended use"\*

**Q9. What's your approach to managing model lifecycle and technical debt?**
**Difficulty: Advanced**
_Answer:_
_"I treat models like code with lifecycle management:_

- _Versioning:_ All models versioned with dependencies
- _Documentation:_ Model cards with performance, limitations, training data
- _Retirement planning:_ Clear criteria for model decommissioning
- _Technical debt tracking:_ Legacy model dependencies and their risks
- _Automated cleanup:_ Scripts to remove old models and free resources
- _Regular reviews:_ Quarterly model portfolio reviews"\*

**Q10. How do you ensure your ML systems are robust to infrastructure failures?**
**Difficulty: Advanced**
_Answer:_
_"I design for resilience through:_

- _Redundancy:_ Multiple model instances across availability zones
- _Circuit breakers:_ Automatic fallback to simpler models or cached predictions
- _Graceful degradation:_ Reduced functionality rather than complete failure
- _Health monitoring:_ Comprehensive health checks and automatic restarts
- _Disaster recovery:_ Backup models and data with rapid recovery procedures
- _Testing:_ Chaos engineering to test failure scenarios"\*

### Team Collaboration (Questions 11-15)

**Q11. How do you collaborate with product managers on ML feature development?**
**Difficulty: Intermediate**
_Answer:_
_"I establish clear communication patterns:_

- _Requirements gathering:_ Understand business goals and success metrics
- _Feasibility assessment:_ Evaluate technical possibilities and constraints
- _Iterative development:_ Regular check-ins and demo sessions
- _Risk communication:_ Clear articulation of technical risks and mitigation
- _Success metrics:_ Define both technical and business KPIs
- _Post-deployment review:_ Measure actual vs. expected business impact"\*

**Q12. Describe a time when you had to explain a complex technical concept to a non-technical colleague.**
**Difficulty: Intermediate**
_Answer Framework:_

- _Audience awareness:_ Understand their background and needs
- _Simplification:_ Use analogies and avoid jargon
- _Visualization:_ Use diagrams or examples
- _Relevance:_ Connect to their domain knowledge
- _Verification:_ Confirm understanding

_Sample:_
_"When explaining model drift to a marketing manager, I used the analogy of customer behavior changing over time. I showed how a model trained on 2020 data would struggle with 2023 behavior patterns, using a simple chart to visualize the concept."_

**Q13. How do you handle disagreements with team members about model deployment decisions?**
**Difficulty: Intermediate**
_Answer:_
_"I approach disagreements constructively:_

- _Data-driven discussion:_ Focus on metrics and evidence
- _Risk assessment:_ Clearly articulate potential consequences
- _Compromise solutions:_ Look for middle ground (e.g., shadow deployment)
- _External input:_ Seek validation from domain experts
- _Small experiments:_ Test both approaches with limited scope
- _Documentation:_ Record the decision rationale for future reference"\*

**Q14. What's your experience with cross-functional ML projects?**
**Difficulty: Intermediate**
_Answer:_
_"In my last project, I worked with:_

- _Data Engineering:_ On real-time data pipelines
- _Product Team:_ On feature requirements and user impact
- _Legal/Compliance:_ On data privacy and model governance
- _Infrastructure:_ On deployment and scaling strategies
- _Customer Support:_ On handling model errors and user feedback\*

_Key success factors: regular sync meetings, shared documentation, clear interfaces, and mutual understanding of constraints and goals."_

**Q15. How do you mentor junior team members in MLOps?**
**Difficulty: Intermediate**
_Answer:_
_"My mentoring approach includes:_

- _Hands-on learning:_ Pair programming and code reviews
- _Documentation:_ Creating and maintaining runbooks and guides
- _Best practices:_ Sharing industry standards and lessons learned
- _Problem-solving:_ Guiding through debugging and troubleshooting
- _Career development:_ Identifying growth opportunities and next steps
- _Regular feedback:_ Weekly one-on-ones and project retrospectives"\*

### Deployment Challenges (Questions 16-20)

**Q16. How do you handle model deployment when you don't have production-like data for testing?**
**Difficulty: Advanced**
_Answer:_
_"I use several strategies:_

- _Synthetic data generation:_ Create realistic test data
- _Shadow deployment:_ Run model alongside production without impact
- _Canary testing:_ Gradually increase traffic with monitoring
- _Historical replay:_ Test with past production data
- _Stress testing:_ Evaluate performance under various conditions
- _Expert validation:_ Have domain experts review predictions"\*

**Q17. Describe a time when you had to optimize a model for edge deployment.**
**Difficulty: Advanced**
_Answer:_
_"For a mobile image classification model:_

- _Model compression:_ Applied quantization and pruning
- _Architecture optimization:_ Used MobileNet architecture
- _Feature selection:_ Removed redundant features
- _Caching strategies:_ Implemented intelligent caching
- _Testing:_ Extensive testing on various device types
- _Performance monitoring:_ Real-time performance tracking on devices"\*

**Q18. How do you ensure backward compatibility when updating ML models?**
**Difficulty: Advanced**
_Answer:_
_"Backward compatibility strategies:_

- _API versioning:_ Maintain multiple API versions
- _Schema evolution:_ Gradual transition of input/output formats
- _Compatibility layers:_ Translation between old and new schemas
- _Feature flags:_ Gradual rollout with user control
- _Contract testing:_ Automated compatibility verification
- _Rollback procedures:_ Quick reversion capability"\*

**Q19. What's your approach to handling real-time vs. batch prediction trade-offs?**
**Difficulty: Advanced**
_Answer:_
_"I evaluate based on:_

- _Latency requirements:_ Real-time (<100ms) vs. batch (hours/days)
- _Throughput needs:_ High-volume batch processing
- _Resource efficiency:_ Real-time resources vs. batch compute costs
- _Business context:_ Customer experience vs. analytical insights
- _Hybrid approaches:_ Combined real-time and batch systems
- _Example: 'Fraud detection uses real-time scoring, while customer segmentation runs nightly in batch mode'"_

**Q20. How do you manage model updates during high-traffic periods?**
**Difficulty: Advanced**
_Answer:_
_"High-traffic deployment strategies:_

- _Scheduled deployment:_ Avoid peak usage times
- _Blue-green deployment:_ Zero-downtime switching
- _Canary releases:_ Gradual traffic shifting
- _Load testing:_ Validate performance under expected load
- _Resource provisioning:_ Scale infrastructure before deployment
- _Monitoring:_ Enhanced monitoring during deployment
- _Rollback readiness:_ Pre-prepared rollback procedures"\*

---

## System Design Questions

### MLOps System Design (Questions 1-5)

**Q1. Design an end-to-end MLOps platform for a company deploying 50+ models.**
**Difficulty: Expert**

_Key Components:_

1. **Data Layer**
   - Data lake (S3, Azure Blob)
   - Feature store (Feast, Tecton)
   - Data validation (Great Expectations, TFDV)

2. **Model Development**
   - Experiment tracking (MLflow, Weights & Biases)
   - Model registry (MLflow Registry, SageMaker Model Registry)
   - Version control (Git for code, DVC for data/models)

3. **CI/CD Pipeline**
   - Automated testing and validation
   - Model packaging and containerization
   - Staging and production deployment

4. **Serving Infrastructure**
   - Real-time serving (SageMaker Endpoints, KServe)
   - Batch processing (Airflow, Kubeflow Pipelines)
   - API gateway and load balancing

5. **Monitoring & Observability**
   - Model performance monitoring
   - Data drift detection
   - System metrics (Prometheus, Grafana)
   - Logging and alerting (ELK stack)

6. **Governance & Security**
   - Model lineage tracking
   - Access control and permissions
   - Compliance and audit trails
   - Model documentation and approval workflows

_Architecture Diagram Concept:_

```
[Data Sources] → [Data Ingestion] → [Feature Store] → [Model Training]
                                                        ↓
[User Apps] ← [API Gateway] ← [Model Serving] ← [Model Registry]
                    ↓
            [Monitoring & Alerting]
                    ↓
            [Model Retraining Pipeline]
```

**Q2. Design a model monitoring system that can track 1000+ models in production.**
**Difficulty: Expert**

_System Architecture:_

1. **Collection Layer**
   - Distributed metrics collection (Prometheus)
   - Log aggregation (Fluentd, Logstash)
   - Custom metrics SDKs for different serving frameworks

2. **Storage Layer**
   - Time-series database (Prometheus, InfluxDB)
   - Log storage (Elasticsearch)
   - Data warehouse (BigQuery, Redshift) for historical analysis

3. **Processing Layer**
   - Real-time anomaly detection (Apache Kafka, Apache Flink)
   - Batch analysis (Apache Spark)
   - Statistical process control algorithms

4. **Alerting & Visualization**
   - Alert manager (Prometheus Alertmanager)
   - Custom dashboards (Grafana)
   - Incident management integration (PagerDuty, OpsGenie)

5. **Key Design Considerations**
   - Scalable metric storage and querying
   - Real-time processing for critical models
   - Automated root cause analysis
   - Multi-tenancy for different business units
   - Cost optimization for data retention

_Sample Monitoring Pipeline:_

```
Model Predictions → Metrics Collector → Message Queue → Stream Processor
                                                    ↓
Alert Manager ← Alert Rules Engine ← Anomaly Detector ← Data Storage
```

**Q3. Design a feature store architecture for a multi-team ML organization.**
**Difficulty: Expert**

_Architecture Components:_

1. **Storage Layer**
   - Online store (Redis, Cassandra) for low-latency serving
   - Offline store (Data warehouse, Data lake) for training
   - Unified feature registry (metadata and definitions)

2. **Feature Management**
   - Feature definition and registration
   - Feature computation pipelines
   - Feature versioning and lineage
   - Access control and permissions

3. **Serving APIs**
   - Low-latency online serving (<10ms)
   - High-throughput batch serving for training
   - Streaming updates for real-time features

4. **Data Quality & Governance**
   - Feature validation and quality checks
   - Data lineage tracking
   - Compliance and audit trails
   - Feature documentation and discovery

_Example: Feast Architecture_

```
[Data Sources] → [Feature Engineering] → [Feature Store] → [Model Training]
                                                        ↓
[Real-time Apps] ← [Online Store] ← [Feature API] ← [Feature Registry]
```

**Q4. Design a CI/CD pipeline for ML models with automated testing and deployment.**
**Difficulty: Advanced**

_Pipeline Stages:_

1. **Development Phase**
   - Code commit triggers pipeline
   - Unit tests and code quality checks
   - Model training with fixed data snapshot
   - Model evaluation and performance testing

2. **Validation Phase**
   - Integration tests with downstream systems
   - Performance and load testing
   - Security and compliance checks
   - Model bias and fairness testing

3. **Staging Deployment**
   - Containerized model deployment
   - Automated smoke tests
   - Shadow deployment for performance comparison
   - A/B testing setup

4. **Production Deployment**
   - Blue-green or canary deployment
   - Monitoring and alerting setup
   - Performance validation
   - Automated rollback triggers

_Example CI/CD Flow:_

```
[Git Commit] → [Build & Test] → [Train Model] → [Validate] → [Package]
      ↓              ↓             ↓           ↓          ↓
[GitHub Actions] → [pytest] → [MLflow] → [Great Expectations] → [Docker]
      ↓
[Deploy to Staging] → [Integration Tests] → [Deploy to Production]
```

**Q5. Design a model versioning and registry system.**
**Difficulty: Advanced**

_System Components:_

1. **Model Registry**
   - Centralized model metadata storage
   - Model lineage and versioning
   - Model approval workflows
   - Access control and permissions

2. **Storage Backend**
   - Artifact storage (S3, GCS, Azure Blob)
   - Model format standardization
   - Compression and deduplication
   - Multi-region replication

3. **API Layer**
   - RESTful APIs for model operations
   - SDK for different programming languages
   - Integration with ML frameworks
   - Bulk operations support

4. **Model Management**
   - Automatic model promotion
   - Model comparison and analysis
   - Dependency tracking
   - Lifecycle management

_Key Features:_

- Immutable model versions
- Semantic versioning
- Model performance tracking
- Automated model testing
- Rollback capabilities
- Audit trails

### Production Architecture (Questions 6-10)

**Q6. Design a scalable model serving architecture for real-time predictions.**
**Difficulty: Expert**

_Architecture Design:_

1. **Load Balancing Layer**
   - API Gateway (Kong, AWS API Gateway)
   - Load balancer (HAProxy, AWS ALB)
   - Rate limiting and throttling

2. **Model Serving Layer**
   - Containerized model services (Docker, Kubernetes)
   - Auto-scaling based on metrics
   - Model warm-up and caching
   - Circuit breaker patterns

3. **Data Access Layer**
   - Feature store integration
   - Caching (Redis, Memcached)
   - Database connections pooling
   - Data validation

4. **Monitoring & Observability**
   - Request tracing (Jaeger, Zipkin)
   - Performance metrics
   - Error tracking
   - Business metrics

_Example: Kubernetes-based Serving_

```
[Load Balancer] → [Model Services] → [Feature Store] → [Model Registry]
      ↓                ↓              ↓              ↓
[Ingress] ← [Pod Autoscaler] ← [Service Mesh] ← [Model Cache]
```

**Q7. Design a system for handling model drift detection and automatic retraining.**
**Difficulty: Expert**

_System Components:_

1. **Data Drift Detection**
   - Statistical tests (KS test, Chi-square)
   - Distribution monitoring
   - Feature importance changes
   - Performance degradation tracking

2. **Trigger Mechanisms**
   - Threshold-based alerts
   - Time-based schedules
   - Performance-based triggers
   - Business event triggers

3. **Automated Retraining Pipeline**
   - Data collection and preprocessing
   - Model training with new data
   - Model validation and comparison
   - Automated deployment

4. **Rollback System**
   - Performance comparison
   - A/B testing for new models
   - Quick rollback mechanisms
   - Model approval workflows

_Pipeline Flow:_

```
[Data Drift Detection] → [Alert Trigger] → [Data Collection] → [Model Training]
        ↓                      ↓                ↓              ↓
[Performance Check] ← [Validation] ← [Model Evaluation] ← [Hyperparameter Tuning]
        ↓
[Deploy New Model] → [Monitor Performance] → [Alert on Issues] → [Rollback if Needed]
```

**Q8. Design a multi-region ML deployment system for global availability.**
**Difficulty: Expert**

_Architecture Considerations:_

1. **Geographic Distribution**
   - Regional data centers
   - Edge computing for low latency
   - Data residency compliance
   - Network optimization

2. **Model Synchronization**
   - Model version consistency
   - Staged rollouts across regions
   - Cross-region model updates
   - Conflict resolution

3. **Data Management**
   - Regional data storage
   - Cross-region data replication
   - Feature store distribution
   - Data privacy compliance

4. **Failover and Disaster Recovery**
   - Automatic failover mechanisms
   - Regional backup strategies
   - Data synchronization
   - Service mesh routing

_Example: Multi-Region Setup_

```
[Global Load Balancer] → [Regional Load Balancers] → [Model Services]
         ↓                       ↓                       ↓
[User Location Detection] → [Regional API Gateways] → [Local Data Centers]
```

**Q9. Design a real-time feature computation system for online predictions.**
**Difficulty: Expert**

_System Architecture:_

1. **Stream Processing**
   - Apache Kafka for event streaming
   - Apache Flink for stream processing
   - Real-time aggregations
   - Window-based computations

2. **Feature Computation**
   - Stream processors for real-time features
   - State management for aggregations
   - Feature validation and quality checks
   - Feature caching and serving

3. **Low-Latency Serving**
   - In-memory feature store
   - Feature computation APIs
   - Caching strategies
   - Pre-computed feature lookups

4. **Data Quality**
   - Real-time data validation
   - Anomaly detection
   - Data lineage tracking
   - Feature documentation

_Example Pipeline:_

```
[User Events] → [Event Stream] → [Stream Processor] → [Feature Store] → [Model Serving]
      ↓              ↓               ↓                ↓              ↓
[Real-time API] ← [Feature Cache] ← [Aggregation] ← [Validation] ← [User Request]
```

**Q10. Design a model governance and compliance system for regulated industries.**
**Difficulty: Expert**

_Governance Framework:_

1. **Model Documentation**
   - Model cards and documentation
   - Training data lineage
   - Model version history
   - Performance metrics and limitations

2. **Approval Workflows**
   - Multi-stage review process
   - Risk assessment
   - Compliance verification
   - Stakeholder sign-offs

3. **Audit and Monitoring**
   - Complete audit trails
   - Decision logging
   - Performance monitoring
   - Bias and fairness tracking

4. **Risk Management**
   - Model risk assessment
   - Continuous monitoring
   - Incident response procedures
   - Regular model reviews

_Compliance Features:_

- Model interpretability
- Bias testing and mitigation
- Privacy protection (differential privacy)
- Explainable AI requirements
- Regulatory reporting
- Data retention policies

_Example Governance Flow:_

```
[Model Development] → [Documentation] → [Risk Assessment] → [Approval]
        ↓                ↓                 ↓              ↓
[Model Testing] ← [Compliance Check] ← [Stakeholder Review] ← [Production Deploy]
        ↓
[Continuous Monitoring] → [Audit Logging] → [Incident Management] → [Model Retirement]
```

### Advanced Monitoring Systems (Questions 11-15)

**Q11. Design a comprehensive ML observability platform.**
**Difficulty: Expert**

_Observability Pillars:_

1. **Metrics**
   - Model performance metrics
   - System performance metrics
   - Business impact metrics
   - Custom application metrics

2. **Logs**
   - Structured application logs
   - Model inference logs
   - Error and exception logs
   - Audit and security logs

3. **Traces**
   - End-to-end request tracing
   - Service dependency mapping
   - Performance bottleneck identification
   - Distributed transaction tracking

4. **Events**
   - Model deployment events
   - Performance threshold breaches
   - System health changes
   - Business events

_Platform Components:_

- **Collection:** OpenTelemetry, Prometheus
- **Storage:** Elasticsearch, ClickHouse
- **Visualization:** Grafana, Kibana
- **Alerting:** Prometheus Alertmanager
- **Analysis:** Custom ML for anomaly detection

**Q12. Design a system for detecting and preventing model poisoning attacks.**
**Difficulty: Expert**

_Security Framework:_

1. **Input Validation**
   - Data quality checks
   - Anomaly detection in inputs
   - Input sanitization
   - Schema validation

2. **Model Protection**
   - Model encryption
   - Access control and authentication
   - Model integrity verification
   - Watermarking techniques

3. **Adversarial Detection**
   - Adversarial example detection
   - Gradient-based detection
   - Statistical anomaly detection
   - Ensemble methods for robustness

4. **Response System**
   - Automatic model isolation
   - Incident response procedures
   - Forensic analysis tools
   - Recovery mechanisms

_Detection Techniques:_

- Input distribution analysis
- Model behavior monitoring
- Performance degradation tracking
- Feature importance changes

**Q13. Design a cost optimization system for ML infrastructure.**
**Difficulty: Advanced**

_Cost Management Framework:_

1. **Cost Monitoring**
   - Resource usage tracking
   - Cost allocation by team/project
   - Real-time cost alerts
   - Historical cost analysis

2. **Optimization Strategies**
   - Right-sizing compute resources
   - Spot instance usage
   - Reserved capacity planning
   - Model compression and optimization

3. **Automated Optimization**
   - Predictive scaling
   - Resource scheduling
   - Storage tier optimization
   - Network cost optimization

4. **Governance**
   - Budget enforcement
   - Cost center accountability
   - Resource quotas
   - Approval workflows

_Optimization Techniques:_

- Model quantization and pruning
- Dynamic model loading
- Caching and CDN usage
- Serverless computing for batch jobs

**Q14. Design a system for managing model explainability at scale.**
**Difficulty: Expert**

_Explainability Framework:_

1. **Explanation Methods**
   - Local explanations (SHAP, LIME)
   - Global explanations
   - Feature importance tracking
   - Counterfactual explanations

2. **Scalable Computation**
   - Explanation caching
   - Parallel computation
   - Approximation methods
   - Model-specific optimizations

3. **Storage and Retrieval**
   - Explanation storage system
   - Fast lookup mechanisms
   - Version control for explanations
   - Privacy-preserving storage

4. **User Interface**
   - Explanation APIs
   - Interactive visualization
   - Custom explanation requests
   - Batch explanation processing

_Implementation:_

- Distributed SHAP computation
- Explanation result caching
- Real-time explanation serving
- Explanation quality monitoring

**Q15. Design a chaos engineering system for ML applications.**
**Difficulty: Expert**

_Chaos Engineering Framework:_

1. **Fault Injection**
   - Network latency and failures
   - Model serving interruptions
   - Data corruption scenarios
   - Resource exhaustion

2. **Resilience Testing**
   - Model degradation testing
   - Failover mechanism testing
   - Recovery time validation
   - Data loss scenarios

3. **Automation**
   - Automated chaos experiments
   - Continuous resilience testing
   - Experiment scheduling
   - Result analysis

4. **Monitoring and Analysis**
   - Chaos experiment metrics
   - System behavior analysis
   - Improvement recommendations
   - Resilience score calculation

_Experiment Types:_

- Model serving failures
- Data pipeline disruptions
- Feature store unavailability
- Network partition scenarios
- Resource constraints

---

## Answers and Explanations

### Technical Questions - Detailed Solutions

**Q1: MLOps vs DevOps Differences**

```python
# Example showing MLOps-specific challenges
class MLDataPipeline:
    def __init__(self):
        self.data_version = None
        self.model_version = None
        self.feature_store = None

    def detect_data_drift(self, current_data, baseline_data):
        """MLOps-specific: Data drift detection"""
        from scipy import stats

        # Perform statistical tests
        p_values = []
        for column in current_data.columns:
            _, p_value = stats.ks_2samp(
                current_data[column],
                baseline_data[column]
            )
            p_values.append(p_value)

        # Alert if drift detected
        if min(p_values) < 0.05:
            self.trigger_model_retraining()

    def trigger_model_retraining(self):
        """MLOps-specific: Automated retraining pipeline"""
        # This is unique to MLOps - triggered by data changes
        print("Data drift detected, initiating model retraining...")
        # Trigger ML pipeline
```

**Q16: Blue-Green vs Canary Deployment Implementation**

```python
class DeploymentManager:
    def __init__(self):
        self.active_environment = "blue"
        self.environments = {
            "blue": {"model_version": "v1.0", "traffic": 100},
            "green": {"model_version": "v1.1", "traffic": 0}
        }

    def canary_deploy(self, new_version, traffic_percentage=10):
        """Canary deployment - gradual traffic shift"""
        self.environments["green"]["model_version"] = new_version
        self.environments["green"]["traffic"] = traffic_percentage
        self.environments["blue"]["traffic"] = 100 - traffic_percentage

        # Monitor for issues
        if self.monitor_performance():
            self.promote_canary()
        else:
            self.rollback_canary()

    def blue_green_deploy(self, new_version):
        """Blue-green deployment - instant switch"""
        # Deploy to green environment
        self.environments["green"]["model_version"] = new_version
        self.environments["green"]["traffic"] = 0

        # Test green environment
        if self.health_check("green"):
            # Switch all traffic to green
            self.active_environment = "green"
            self.environments["blue"]["traffic"] = 0
            self.environments["green"]["traffic"] = 100
```

**Q36: ML Monitoring Implementation**

```python
class MLMonitoringSystem:
    def __init__(self, model_endpoint):
        self.model_endpoint = model_endpoint
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()

    def monitor_model_performance(self, predictions, actuals):
        """Monitor key ML metrics"""
        metrics = {
            "accuracy": accuracy_score(actuals, predictions),
            "precision": precision_score(actuals, predictions),
            "recall": recall_score(actuals, predictions),
            "f1_score": f1_score(actuals, predictions)
        }

        # Check for performance degradation
        for metric_name, value in metrics.items():
            threshold = self.get_threshold(metric_name)
            if value < threshold:
                self.alert_manager.send_alert(
                    f"Model performance degraded: {metric_name} = {value}"
                )

        return metrics

    def monitor_data_drift(self, current_data, reference_data):
        """Monitor for data distribution changes"""
        drift_scores = {}

        for column in current_data.columns:
            # Use Jensen-Shannon divergence for drift detection
            drift_score = self.calculate_js_divergence(
                current_data[column],
                reference_data[column]
            )
            drift_scores[column] = drift_score

        # Alert if significant drift detected
        max_drift = max(drift_scores.values())
        if max_drift > 0.1:  # threshold
            self.alert_manager.send_alert(
                f"Data drift detected: max drift = {max_drift}"
            )

        return drift_scores
```

### System Design - Implementation Examples

**Q1: MLOps Platform Architecture**

```yaml
# Kubernetes-based MLOps platform
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlops-config
data:
  # MLflow configuration
  MLFLOW_TRACKING_URI: "http://mlflow:5000"
  MLFLOW_S3_ENDPOINT_URL: "http://minio:9000"

  # Feature store configuration
  FEAST_CORE_URL: "http://feast-core:8080"
  FEAST_ONLINE_STORE: "redis"

  # Monitoring configuration
  PROMETHEUS_URL: "http://prometheus:9090"
  GRAFANA_URL: "http://grafana:3000"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-tracking
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
        - name: mlflow
          image: python:3.9
          command:
            - /bin/bash
            - -c
            - |
              pip install mlflow boto3 psycopg2-binary
              mlflow server --host 0.0.0.0 --port 5000 \
                --backend-store-uri postgresql://user:pass@postgres:5432/mlflow \
                --default-artifact-root s3://mlflow-artifacts/
          ports:
            - containerPort: 5000
          envFrom:
            - configMapRef:
                name: mlops-config
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
spec:
  selector:
    app: mlflow
  ports:
    - port: 5000
      targetPort: 5000
```

**Q6: Scalable Model Serving Architecture**

```python
# Kubernetes-based scalable serving
from kubernetes import client, config
from kubernetes.client.rest import ApiException

class K8sModelServer:
    def __init__(self):
        config.load_incluster_config()  # For production cluster
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()

    def create_model_deployment(self, model_name: str, image: str, replicas: int = 3):
        """Create a Kubernetes deployment for model serving"""

        # Container specification
        container = client.V1Container(
            name=model_name,
            image=image,
            ports=[client.V1ContainerPort(container_port=8080)],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "500m", "memory": "1Gi"},
                limits={"cpu": "1000m", "memory": "2Gi"}
            ),
            liveness_probe=client.V1Probe(
                http_get=client.V1HTTPGetProbe(
                    path="/health", port=8080
                ),
                initial_delay_seconds=30,
                period_seconds=10
            ),
            readiness_probe=client.V1Probe(
                http_get=client.V1HTTPGetProbe(
                    path="/ready", port=8080
                ),
                initial_delay_seconds=5,
                period_seconds=5
            )
        )

        # Pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={"app": model_name, "version": "v1"}
            ),
            spec=client.V1PodSpec(
                containers=[container],
                node_selector={"ml-workload": "true"}
            )
        )

        # Deployment spec
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=f"{model_name}-deployment"),
            spec=client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(
                    match_labels={"app": model_name}
                ),
                template=pod_template
            )
        )

        # Create deployment
        try:
            api_response = self.apps_v1.create_namespaced_deployment(
                namespace="default", body=deployment
            )
            return api_response
        except ApiException as e:
            print(f"Exception when creating deployment: {e}")
            raise

    def create_hpa(self, model_name: str, min_replicas: int = 2, max_replicas: int = 10):
        """Create Horizontal Pod Autoscaler"""

        hpa = client.V2HorizontalPodAutoscaler(
            api_version="autoscaling/v2",
            kind="HorizontalPodAutoscaler",
            metadata=client.V1ObjectMeta(name=f"{model_name}-hpa"),
            spec=client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V2CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=f"{model_name}-deployment"
                ),
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                metrics=[
                    client.V2MetricSpec(
                        type="Resource",
                        resource=client.V2ResourceMetricSource(
                            name="cpu",
                            target=client.V2MetricTarget(
                                type="Utilization",
                                average_utilization=70
                            )
                        )
                    )
                ]
            )
        )

        try:
            api_response = self.apps_v1.create_namespaced_horizontal_pod_autoscaler(
                namespace="default", body=hpa
            )
            return api_response
        except ApiException as e:
            print(f"Exception when creating HPA: {e}")
            raise
```

### Advanced Implementation - Model Monitoring

**Q11: Comprehensive Observability Platform**

```python
import asyncio
import logging
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

@dataclass
class ObservabilityConfig:
    service_name: str
    jaeger_endpoint: str
    prometheus_port: int
    log_level: str
    trace_sampling_rate: float

class MLObservabilityPlatform:
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.setup_tracing()
        self.setup_metrics()
        self.setup_logging()
        self.trace = trace.get_tracer(__name__)

    def setup_tracing(self):
        """Configure distributed tracing"""
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer_provider()

        jaeger_exporter = JaegerExporter(
            agent_host_name=self.config.jaeger_endpoint.split(':')[0],
            agent_port=int(self.config.jaeger_endpoint.split(':')[1])
        )

        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer.add_span_processor(span_processor)

    def setup_metrics(self):
        """Setup Prometheus metrics"""
        # Model-specific metrics
        self.prediction_counter = Counter(
            'ml_predictions_total',
            'Total predictions made',
            ['model_name', 'prediction_type']
        )

        self.prediction_latency = Histogram(
            'ml_prediction_duration_seconds',
            'Time spent on predictions',
            ['model_name']
        )

        self.model_accuracy = Gauge(
            'ml_model_accuracy',
            'Current model accuracy',
            ['model_name']
        )

        self.data_drift_score = Gauge(
            'ml_data_drift_score',
            'Current data drift score',
            ['model_name', 'feature_name']
        )

        # System metrics
        self.cpu_usage = Gauge('ml_system_cpu_usage_percent', 'CPU usage')
        self.memory_usage = Gauge('ml_system_memory_usage_percent', 'Memory usage')

        # Start metrics server
        start_http_server(self.config.prometheus_port)

    def setup_logging(self):
        """Setup structured logging"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        self.logger = structlog.get_logger()

    async def track_prediction(self,
                             model_name: str,
                             prediction: Any,
                             features: List[float],
                             actual: Any = None):
        """Track model prediction with full observability"""

        with self.trace.start_as_current_span("ml_prediction") as span:
            # Add attributes
            span.set_attribute("model.name", model_name)
            span.set_attribute("prediction.value", str(prediction))
            span.set_attribute("features.count", len(features))

            try:
                # Record metrics
                self.prediction_counter.labels(
                    model_name=model_name,
                    prediction_type="inference"
                ).inc()

                # Simulate prediction timing
                prediction_time = np.random.uniform(0.01, 0.1)
                self.prediction_latency.labels(model_name=model_name).observe(prediction_time)

                # If actual value provided, calculate accuracy
                if actual is not None:
                    accuracy = self._calculate_accuracy(prediction, actual)
                    self.model_accuracy.labels(model_name=model_name).set(accuracy)
                    span.set_attribute("accuracy", accuracy)

                # Log structured event
                self.logger.info(
                    "ml_prediction_completed",
                    model_name=model_name,
                    prediction=prediction,
                    features_count=len(features),
                    prediction_time=prediction_time,
                    actual=actual
                )

                return prediction

            except Exception as e:
                # Log error with context
                self.logger.error(
                    "ml_prediction_failed",
                    model_name=model_name,
                    error=str(e),
                    features=features[:10]  # Truncate for logging
                )
                span.record_exception(e)
                raise

    def _calculate_accuracy(self, prediction: Any, actual: Any) -> float:
        """Calculate prediction accuracy (simplified)"""
        # Implement your accuracy calculation
        return 0.95 if prediction == actual else 0.85

    async def monitor_data_drift(self,
                                model_name: str,
                                current_data: np.ndarray,
                                reference_data: np.ndarray):
        """Monitor for data drift"""

        with self.trace.start_as_current_span("ml_data_drift_check") as span:
            drift_scores = {}

            for i, feature_name in enumerate([f"feature_{i}" for i in range(current_data.shape[1])]):
                # Calculate drift score (simplified)
                drift_score = abs(np.mean(current_data[:, i]) - np.mean(reference_data[:, i]))
                drift_scores[feature_name] = drift_score

                # Update metric
                self.data_drift_score.labels(
                    model_name=model_name,
                    feature_name=feature_name
                ).set(drift_score)

                # Set alert if drift is high
                if drift_score > 0.1:
                    self.logger.warning(
                        "ml_high_data_drift",
                        model_name=model_name,
                        feature_name=feature_name,
                        drift_score=drift_score
                    )
                    span.add_event("high_drift_detected", {
                        "feature_name": feature_name,
                        "drift_score": drift_score
                    })

            return drift_scores

    async def collect_system_metrics(self):
        """Collect system resource metrics"""
        import psutil

        # CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent

        self.cpu_usage.set(cpu_percent)
        self.memory_usage.set(memory_percent)

        # Log if thresholds exceeded
        if cpu_percent > 90:
            self.logger.warning("ml_high_cpu_usage", cpu_percent=cpu_percent)

        if memory_percent > 90:
            self.logger.warning("ml_high_memory_usage", memory_percent=memory_percent)

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }

        try:
            # Check system resources
            await self.collect_system_metrics()
            health_status["checks"]["system_metrics"] = "ok"
        except Exception as e:
            health_status["checks"]["system_metrics"] = f"error: {e}"
            health_status["status"] = "degraded"

        try:
            # Check trace exporter
            with self.trace.start_as_current_span("health_check") as span:
                health_status["checks"]["tracing"] = "ok"
        except Exception as e:
            health_status["checks"]["tracing"] = f"error: {e}"
            health_status["status"] = "degraded"

        return health_status

# Example usage
async def main():
    config = ObservabilityConfig(
        service_name="ml-model-server",
        jaeger_endpoint="jaeger:6831",
        prometheus_port=8000,
        log_level="INFO",
        trace_sampling_rate=0.1
    )

    observability = MLObservabilityPlatform(config)

    # Simulate ML workload
    model_name = "fraud-detection"

    # Generate sample data
    current_data = np.random.normal(0, 1, (100, 10))
    reference_data = np.random.normal(0.1, 1.1, (100, 10))

    # Monitor data drift
    drift_scores = await observability.monitor_data_drift(
        model_name, current_data, reference_data
    )

    # Track predictions
    for i in range(10):
        features = np.random.random(10).tolist()
        prediction = await observability.track_prediction(
            model_name=model_name,
            prediction=np.random.random(),
            features=features
        )

    # Health check
    health = await observability.health_check()
    print(f"Health status: {health}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Summary

This comprehensive interview question set covers all essential aspects of model deployment and production systems:

### Technical Skills Assessed

- **MLOps Fundamentals**: Pipeline design, model lifecycle management
- **Deployment Strategies**: Blue-green, canary, shadow deployments
- **Production Systems**: Scalability, fault tolerance, performance optimization
- **Monitoring & Observability**: Metrics collection, alerting, incident response

### Practical Coding Challenges

- **Containerization**: Production-ready Docker images
- **Orchestration**: Kubernetes deployments with auto-scaling
- **CI/CD**: Automated testing and deployment pipelines
- **Monitoring**: Real-time metrics and alerting systems

### Behavioral Scenarios

- **Crisis Management**: Handling production incidents
- **Cross-team Collaboration**: Working with diverse stakeholders
- **Decision Making**: Balancing competing priorities
- **Communication**: Technical concepts to business audiences

### System Design

- **Scalable Architecture**: Multi-region deployments
- **Data Management**: Feature stores, model registries
- **Security**: Model governance, compliance
- **Cost Optimization**: Resource management, automation

### Advanced Topics

- **A/B Testing**: Statistical validation frameworks
- **Auto-scaling**: Intelligent resource management
- **Chaos Engineering**: Resilience testing
- **Observability**: Comprehensive monitoring platforms

This question set ensures candidates demonstrate both theoretical knowledge and practical implementation skills required for senior MLOps roles in production environments.
