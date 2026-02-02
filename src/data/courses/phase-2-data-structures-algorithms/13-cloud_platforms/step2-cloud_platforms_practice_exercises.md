# Cloud Platforms Practice Exercises

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Cloud Concepts](#basic-cloud-concepts)
3. [Compute Services](#compute-services)
4. [Storage and Databases](#storage-and-databases)
5. [Networking](#networking)
6. [Security](#security)
7. [Scalability and High Availability](#scalability-and-high-availability)
8. [Cost Management](#cost-management)
9. [DevOps and Automation](#devops-and-automation)
10. [Project-Based Exercises](#project-based-exercises)
11. [Assessment Challenges](#assessment-challenges)

---

## Introduction

Welcome to the hands-on Cloud Platforms practice exercises! These exercises are designed to give you practical experience with cloud computing concepts through real-world scenarios and hands-on implementations.

**Sarah's Learning Philosophy**
"The best way to learn cloud computing is by doing," Sarah realized. "Reading about auto-scaling is helpful, but actually setting it up and watching it work during a load test - that's when it clicks."

**Exercise Structure:**

- **Beginner**: Basic cloud concepts and simple setups
- **Intermediate**: Building and deploying applications
- **Advanced**: Complex architectures and optimization
- **Challenge**: Real-world problem solving

**Prerequisites:**

- Basic knowledge of web development
- Understanding of databases and SQL
- Familiarity with command line interfaces
- A cloud provider account (AWS free tier recommended)

**Time Requirements:**

- Beginner exercises: 30-60 minutes each
- Intermediate exercises: 1-2 hours each
- Advanced exercises: 2-4 hours each
- Project-based: 4-8 hours each

---

## Basic Cloud Concepts

### Exercise 1.1: Setting Up Your First Cloud Environment

**Scenario**: Sarah is setting up her first cloud environment for a personal project. She needs to understand the basic building blocks.

**Task**: Set up a basic cloud environment and understand the fundamental components.

**Steps**:

1. **Create a Cloud Provider Account**
   - Sign up for AWS Free Tier (recommended for beginners)
   - Alternatively: Google Cloud Platform or Microsoft Azure
   - Complete account verification process

2. **Explore the Cloud Console**
   - Navigate through the dashboard
   - Identify key services: Compute, Storage, Database, Networking
   - Understand the service catalog and categories

3. **Set Up Your First Virtual Machine**

   ```bash
   # For AWS EC2
   # Go to EC2 Dashboard
   # Click "Launch Instance"
   # Choose Amazon Linux 2 AMI
   # Select t2.micro instance (free tier)
   # Configure security group (allow SSH, HTTP, HTTPS)
   # Create key pair and download
   # Launch instance
   ```

4. **Connect to Your Instance**

   ```bash
   # Make key file secure
   chmod 400 your-key.pem

   # Connect to instance
   ssh -i your-key.pem ec2-user@your-instance-ip

   # Update system
   sudo yum update -y  # Amazon Linux
   # or
   sudo apt update && sudo apt upgrade -y  # Ubuntu
   ```

5. **Install Basic Software**

   ```bash
   # Install web server
   sudo yum install httpd -y  # Amazon Linux
   # or
   sudo apt install nginx -y  # Ubuntu

   # Install development tools
   sudo yum groupinstall "Development Tools" -y

   # Install Python
   sudo yum install python3 -y
   ```

**Expected Outcome:**

- Successfully created and connected to a virtual machine
- Understanding of cloud console navigation
- Basic understanding of instance types and AMIs
- Familiarity with security groups and key pairs

**Key Learning Points:**

- Cloud computing service models (IaaS, PaaS, SaaS)
- Virtual machine concepts and instance types
- Security groups and network access control
- Key pair authentication

### Exercise 1.2: Understanding Service Models

**Scenario**: Sarah needs to understand the differences between IaaS, PaaS, and SaaS to choose the right services for her project.

**Task**: Compare different service models by implementing the same application using different approaches.

**Steps**:

1. **IaaS Implementation - Virtual Machine**

   ```bash
   # Create EC2 instance
   # Install Node.js
   curl -fsSL https://rpm.nodesource.com/setup_16.x | sudo bash -
   sudo yum install nodejs -y

   # Create simple Express app
   mkdir ~/my-app
   cd ~/my-app
   npm init -y
   npm install express

   # Create app.js
   cat > app.js << 'EOF'
   const express = require('express');
   const app = express();

   app.get('/', (req, res) => {
       res.send('Hello from IaaS (EC2)!');
   });

   app.listen(3000, () => {
       console.log('IaaS app listening on port 3000');
   });
   EOF

   # Start application
   node app.js
   ```

2. **PaaS Implementation - AWS Elastic Beanstalk**

   ```bash
   # Install EB CLI
   pip install awsebcli

   # Initialize Elastic Beanstalk application
   eb init my-paas-app

   # Create environment
   eb create production

   # Deploy application
   # Create a simple app and deploy
   ```

3. **SaaS vs Custom Implementation**

   ```javascript
   // SaaS approach - use third-party services
   const AWS = require("aws-sdk");
   const lambda = new AWS.Lambda();

   // Function as a Service (FaaS) - serverless
   exports.handler = async (event) => {
     return {
       statusCode: 200,
       body: JSON.stringify("Hello from FaaS (Lambda)!"),
     };
   };
   ```

**Comparison Table:**
| Aspect | IaaS (EC2) | PaaS (Elastic Beanstalk) | FaaS (Lambda) |
|--------|------------|--------------------------|---------------|
| Control | Full OS control | Limited OS control | No server control |
| Scaling | Manual | Automatic | Automatic |
| Management | Full responsibility | Shared responsibility | Provider managed |
| Cost | Fixed + variable | Variable | Pay-per-use |
| Time to Deploy | Minutes | Minutes | Seconds |

**Expected Outcome:**

- Understanding of service model differences
- Experience with different deployment approaches
- Awareness of management responsibilities
- Cost comparison insights

**Key Learning Points:**

- When to use each service model
- Trade-offs between control and convenience
- Cost implications of different models
- Management responsibilities

### Exercise 1.3: Cloud Regions and Availability Zones

**Scenario**: Sarah is deploying a global application and needs to understand how to choose regions and use availability zones for high availability.

**Task**: Set up resources in multiple regions and configure high availability.

**Steps**:

1. **Explore Available Regions**

   ```bash
   # AWS CLI to list regions
   aws ec2 describe-regions --output table

   # Understand region characteristics:
   # - Physical location
   # - Data residency requirements
   # - Service availability
   # - Latency to users
   ```

2. **Set Up Multi-AZ Deployment**

   ```yaml
   # Create VPC with multiple AZs
   Resources:
     VPC:
       Type: AWS::EC2::VPC
       Properties:
         CidrBlock: 10.0.0.0/16
         EnableDnsHostnames: true
         EnableDnsSupport: true

     PublicSubnet1:
       Type: AWS::EC2::Subnet
       Properties:
         VpcId: !Ref VPC
         CidrBlock: 10.0.1.0/24
         AvailabilityZone: !Select [0, !GetAZs ""]

     PublicSubnet2:
       Type: AWS::EC2::Subnet
       Properties:
         VpcId: !Ref VPC
         CidrBlock: 10.0.2.0/24
         AvailabilityZone: !Select [1, !GetAZs ""]
   ```

3. **Configure Load Balancer Across AZs**

   ```yaml
   ApplicationLoadBalancer:
     Type: AWS::ElasticLoadBalancingV2::LoadBalancer
     Properties:
       Subnets:
         - !Ref PublicSubnet1
         - !Ref PublicSubnet2
       SecurityGroups:
         - !Ref ALBSecurityGroup
   ```

4. **Test Cross-AZ Functionality**

   ```bash
   # Test load balancing
   curl -I http://your-load-balancer-dns

   # Check instance distribution
   aws ec2 describe-instances --filters Name=vpc-id,Values=vpc-id --output table

   # Test failover
   # Stop one instance and verify traffic routes to the other
   aws ec2 stop-instances --instance-ids i-1234567890abcdef0
   ```

**Expected Outcome:**

- Understanding of global cloud infrastructure
- Ability to deploy across multiple availability zones
- Experience with load balancing
- Awareness of high availability principles

**Key Learning Points:**

- Region selection criteria
- Availability zone concepts
- High availability design patterns
- Load balancing across zones

---

## Compute Services

### Exercise 2.1: Virtual Machine Deep Dive

**Scenario**: Sarah needs to set up a development environment that can scale based on demand and has specific performance requirements.

**Task**: Configure an auto-scaling group of virtual machines for a development environment.

**Steps**:

1. **Create Launch Template**

   ```json
   {
     "LaunchTemplate": {
       "LaunchTemplateName": "dev-environment-template",
       "LaunchTemplateData": {
         "ImageId": "ami-0c55b159cbfafe1d0",  # Amazon Linux 2
         "InstanceType": "t3.medium",
         "KeyName": "my-dev-key",
         "SecurityGroupIds": ["sg-12345678"],
         "UserData": {
           "Fn::Base64": {
             "Fn::Sub": |
               #!/bin/bash
               yum update -y
               yum install -y docker
               systemctl start docker
               usermod -a -G docker ec2-user
               curl -L "https://github.com/docker/compose/releases/download/v2.0.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
               chmod +x /usr/local/bin/docker-compose
         }
       }
     }
   }
   ```

2. **Set Up Auto Scaling Group**

   ```json
   {
     "AutoScalingGroup": {
       "AutoScalingGroupName": "dev-environment-asg",
       "LaunchTemplate": {
         "LaunchTemplateName": "dev-environment-template",
         "Version": "1"
       },
       "MinSize": 2,
       "MaxSize": 6,
       "DesiredCapacity": 3,
       "AvailabilityZones": ["us-east-1a", "us-east-1b", "us-east-1c"],
       "TargetGroupARNs": [
         "arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/dev-targets/1234567890123456"
       ]
     }
   }
   ```

3. **Configure Scaling Policies**

   ```json
   {
     "ScalingPolicy": {
       "PolicyName": "scale-up-cpu",
       "PolicyType": "TargetTrackingScaling",
       "AutoScalingGroupName": "dev-environment-asg",
       "TargetTrackingConfiguration": {
         "PredefinedMetricSpecification": {
           "PredefinedMetricType": "ASGAverageCPUUtilization"
         },
         "TargetValue": 70.0
       }
     }
   }
   ```

4. **Implement Health Checks**

   ```bash
   # Create health check script
   cat > /usr/local/bin/health-check.sh << 'EOF'
   #!/bin/bash
   # Health check for development environment

   # Check if Docker is running
   if ! systemctl is-active --quiet docker; then
       echo "Docker is not running"
       exit 1
   fi

   # Check if ports are listening
   if ! netstat -tuln | grep -q ":3000 "; then
       echo "Application port 3000 not listening"
       exit 1
   fi

   # Check disk space
   if df / | awk 'NR==2 {print $5}' | grep -q '[8-9][0-9]%'; then
       echo "Disk space critical"
       exit 1
   fi

   echo "Health check passed"
   exit 0
   EOF

   chmod +x /usr/local/bin/health-check.sh

   # Add to crontab for regular checks
   echo "*/5 * * * * /usr/local/bin/health-check.sh" | crontab -
   ```

5. **Test Auto Scaling**

   ```bash
   # Create CPU stress test
   sudo yum install stress -y

   # Simulate high load
   stress --cpu 4 --timeout 300s

   # Monitor auto scaling in real-time
   watch -n 10 "aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names dev-environment-asg --query 'AutoScalingGroups[0].Instances'"
   ```

**Expected Outcome:**

- Understanding of auto-scaling groups
- Experience with launch templates
- Ability to configure scaling policies
- Knowledge of health check implementation

**Key Learning Points:**

- Instance types and sizing
- Auto-scaling configuration
- Health check strategies
- Load testing and performance monitoring

### Exercise 2.2: Serverless Function Development

**Scenario**: Sarah wants to build a serverless API that can handle variable loads efficiently without managing servers.

**Task**: Create a serverless API with multiple functions and integrate with databases and storage.

**Steps**:

1. **Create Lambda Function for User Authentication**

   ```javascript
   // auth-handler.js
   const AWS = require("aws-sdk");
   const bcrypt = require("bcryptjs");
   const jwt = require("jsonwebtoken");
   const dynamodb = new AWS.DynamoDB.DocumentClient();

   const TABLE_NAME = "Users";
   const JWT_SECRET = process.env.JWT_SECRET;

   exports.handler = async (event) => {
     const { httpMethod, path, body } = event;
     const data = JSON.parse(body);

     try {
       switch (httpMethod) {
         case "POST":
           if (path === "/auth/register") {
             return await registerUser(data);
           } else if (path === "/auth/login") {
             return await loginUser(data);
           }
           break;
         default:
           return response(405, { error: "Method not allowed" });
       }
     } catch (error) {
       console.error("Error:", error);
       return response(500, { error: "Internal server error" });
     }
   };

   async function registerUser(data) {
     const { email, password, name } = data;

     // Check if user exists
     const existingUser = await dynamodb
       .get({
         TableName: TABLE_NAME,
         Key: { email },
       })
       .promise();

     if (existingUser.Item) {
       return response(409, { error: "User already exists" });
     }

     // Hash password
     const hashedPassword = await bcrypt.hash(password, 10);

     // Create user
     await dynamodb
       .put({
         TableName: TABLE_NAME,
         Item: {
           email,
           password: hashedPassword,
           name,
           createdAt: new Date().toISOString(),
           isActive: true,
         },
       })
       .promise();

     return response(201, { message: "User registered successfully" });
   }

   async function loginUser(data) {
     const { email, password } = data;

     // Get user
     const user = await dynamodb
       .get({
         TableName: TABLE_NAME,
         Key: { email },
       })
       .promise();

     if (!user.Item) {
       return response(401, { error: "Invalid credentials" });
     }

     // Verify password
     const validPassword = await bcrypt.compare(password, user.Item.password);
     if (!validPassword) {
       return response(401, { error: "Invalid credentials" });
     }

     // Generate token
     const token = jwt.sign(
       { email: user.Item.email, name: user.Item.name },
       JWT_SECRET,
       { expiresIn: "24h" },
     );

     return response(200, {
       token,
       user: { email: user.Item.email, name: user.Item.name },
     });
   }

   function response(statusCode, body) {
     return {
       statusCode,
       headers: {
         "Content-Type": "application/json",
         "Access-Control-Allow-Origin": "*",
         "Access-Control-Allow-Headers": "Content-Type,Authorization",
         "Access-Control-Allow-Methods": "POST,GET,OPTIONS",
       },
       body: JSON.stringify(body),
     };
   }
   ```

2. **Create Lambda Function for File Upload**

   ```javascript
   // file-upload-handler.js
   const AWS = require("aws-sdk");
   const S3 = new AWS.S3();
   const dynamodb = new AWS.DynamoDB.DocumentClient();

   const BUCKET_NAME = process.env.BUCKET_NAME;
   const TABLE_NAME = "Files";

   exports.handler = async (event) => {
     const { httpMethod, path, body } = event;

     try {
       if (httpMethod === "POST" && path === "/files/upload") {
         return await uploadFile(event);
       } else if (httpMethod === "GET" && path === "/files") {
         return await listFiles(event);
       }
       return response(405, { error: "Method not allowed" });
     } catch (error) {
       console.error("Error:", error);
       return response(500, { error: "Internal server error" });
     }
   };

   async function uploadFile(event) {
     // Parse the file from the event
     const body = JSON.parse(event.body);
     const { fileName, fileContent, fileType } = body;
     const userId = event.requestContext.authorizer.claims.sub;

     // Generate unique file key
     const fileKey = `${userId}/${Date.now()}-${fileName}`;

     // Upload to S3
     const s3Params = {
       Bucket: BUCKET_NAME,
       Key: fileKey,
       Body: Buffer.from(fileContent, "base64"),
       ContentType: fileType,
       ServerSideEncryption: "AES256",
     };

     await S3.putObject(s3Params).promise();

     // Save metadata to DynamoDB
     const fileRecord = {
       fileId: AWS.util.crypto.md5(fileKey),
       userId,
       fileName,
       fileType,
       fileKey,
       fileSize: Buffer.from(fileContent, "base64").length,
       createdAt: new Date().toISOString(),
     };

     await dynamodb
       .put({
         TableName: TABLE_NAME,
         Item: fileRecord,
       })
       .promise();

     return response(201, {
       message: "File uploaded successfully",
       fileId: fileRecord.fileId,
       fileKey,
     });
   }

   async function listFiles(event) {
     const userId = event.requestContext.authorizer.claims.sub;

     const result = await dynamodb
       .query({
         TableName: TABLE_NAME,
         KeyConditionExpression: "userId = :userId",
         ExpressionAttributeValues: {
           ":userId": userId,
         },
         ScanIndexForward: false,
         Limit: 10,
       })
       .promise();

     return response(200, { files: result.Items });
   }

   function response(statusCode, body) {
     return {
       statusCode,
       headers: {
         "Content-Type": "application/json",
         "Access-Control-Allow-Origin": "*",
         "Access-Control-Allow-Headers": "Content-Type,Authorization",
         "Access-Control-Allow-Methods": "POST,GET,OPTIONS",
       },
       body: JSON.stringify(body),
     };
   }
   ```

3. **Set Up API Gateway**

   ```yaml
   # serverless.yml
   service: serverless-api

   provider:
     name: aws
     runtime: nodejs14.x
     region: us-east-1
     environment:
       TABLE_NAME: ${self:service}-${opt:stage, 'dev'}-Users
       BUCKET_NAME: ${self:service}-${opt:stage, 'dev'}-files
       JWT_SECRET: ${env:JWT_SECRET}

   functions:
     auth:
       handler: auth-handler.handler
       events:
         - http:
             path: auth/{action}
             method: ANY
             cors: true

     files:
       handler: file-upload-handler.handler
       events:
         - http:
             path: files/{action}
             method: ANY
             cors: true
       environment:
         BUCKET_NAME: ${self:provider.environment.BUCKET_NAME}

   resources:
     Resources:
       UsersTable:
         Type: AWS::DynamoDB::Table
         Properties:
           TableName: ${self:provider.environment.TABLE_NAME}
           BillingMode: PAY_PER_REQUEST
           AttributeDefinitions:
             - AttributeName: email
               AttributeType: S
           KeySchema:
             - AttributeName: email
               KeyType: HASH

       FilesTable:
         Type: AWS::DynamoDB::Table
         Properties:
           TableName: ${self:provider.environment.TABLE_NAME}
           BillingMode: PAY_PER_REQUEST
           AttributeDefinitions:
             - AttributeName: fileId
               AttributeType: S
             - AttributeName: userId
               AttributeType: S
           KeySchema:
             - AttributeName: fileId
               KeyType: HASH
           GlobalSecondaryIndexes:
             - IndexName: UserIdIndex
               KeySchema:
                 - AttributeName: userId
                   KeyType: HASH
               Projection:
                 ProjectionType: ALL

       FilesBucket:
         Type: AWS::S3::Bucket
         Properties:
           BucketName: ${self:provider.environment.BUCKET_NAME}
           CorsConfiguration:
             CorsRules:
               - AllowedHeaders: ["*"]
                 AllowedMethods: ["GET", "PUT", "POST", "DELETE"]
                 AllowedOrigins: ["*"]
                 MaxAge: 3600
   ```

4. **Deploy and Test**

   ```bash
   # Install Serverless Framework
   npm install -g serverless

   # Deploy the application
   serverless deploy

   # Test the endpoints
   curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/dev/auth/register \
     -H "Content-Type: application/json" \
     -d '{"email": "test@example.com", "password": "testpass123", "name": "Test User"}'

   # Test file upload
   curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/dev/files/upload \
     -H "Content-Type: application/json" \
     -d '{"fileName": "test.txt", "fileContent": "SGVsbG8gV29ybGQ=", "fileType": "text/plain"}'
   ```

**Expected Outcome:**

- Understanding of serverless architecture
- Experience with Lambda functions
- Knowledge of API Gateway configuration
- Integration with DynamoDB and S3

**Key Learning Points:**

- Serverless vs traditional architecture
- Event-driven programming
- API Gateway configuration
- Authentication and authorization
- File handling in serverless environment

### Exercise 2.3: Container Orchestration

**Scenario**: Sarah needs to deploy a microservices application that requires multiple containers with different scaling requirements.

**Task**: Set up a Kubernetes cluster and deploy a multi-container application.

**Steps**:

1. **Set Up Kubernetes Cluster (EKS)**

   ```bash
   # Install kubectl
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

   # Install eksctl
   curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
   sudo mv /tmp/eksctl /usr/local/bin

   # Create cluster
   eksctl create cluster \
     --name my-cluster \
     --region us-east-1 \
     --nodes 3 \
     --node-type t3.medium \
     --managed
   ```

2. **Create Application Docker Images**

   ```dockerfile
   # Dockerfile for web frontend
   FROM node:14-alpine

   WORKDIR /app

   # Copy package files
   COPY package*.json ./
   RUN npm ci --only=production

   # Copy source code
   COPY . .

   # Build application
   RUN npm run build

   # Expose port
   EXPOSE 3000

   # Health check
   HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
     CMD curl -f http://localhost:3000/health || exit 1

   # Start application
   CMD ["npm", "start"]
   ```

3. **Create Kubernetes Manifests**

   ```yaml
   # web-deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: web-app
     labels:
       app: web-app
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: web-app
     template:
       metadata:
         labels:
           app: web-app
       spec:
         containers:
           - name: web-app
             image: my-registry/web-app:latest
             ports:
               - containerPort: 3000
             env:
               - name: DATABASE_URL
                 valueFrom:
                   secretKeyRef:
                     name: app-secrets
                     key: database-url
               - name: API_URL
                 value: "http://api-service:8080"
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
                 port: 3000
               initialDelaySeconds: 30
               periodSeconds: 10
             readinessProbe:
               httpGet:
                 path: /ready
                 port: 3000
               initialDelaySeconds: 5
               periodSeconds: 5
   ```

4. **Configure Service and Ingress**

   ```yaml
   # web-service.yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: web-app-service
   spec:
     selector:
       app: web-app
     ports:
       - port: 80
         targetPort: 3000
     type: ClusterIP

   ---
   # ingress.yaml
   apiVersion: networking.k8s.io/v1
   kind: Ingress
   metadata:
     name: web-app-ingress
     annotations:
       kubernetes.io/ingress.class: "alb"
       alb.ingress.kubernetes.io/scheme: internet-facing
       alb.ingress.kubernetes.io/target-type: ip
   spec:
     rules:
       - host: myapp.example.com
         http:
           paths:
             - path: /
               pathType: Prefix
               backend:
                 service:
                   name: web-app-service
                   port:
                     number: 80
   ```

5. **Set Up Horizontal Pod Autoscaler**

   ```yaml
   # hpa.yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: web-app-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: web-app
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

6. **Deploy and Monitor**

   ```bash
   # Apply all manifests
   kubectl apply -f web-deployment.yaml
   kubectl apply -f web-service.yaml
   kubectl apply -f ingress.yaml
   kubectl apply -f hpa.yaml

   # Check deployment status
   kubectl get deployments
   kubectl get pods
   kubectl get services

   # Check HPA status
   kubectl get hpa

   # Monitor scaling
   kubectl get pods -w

   # Test autoscale by generating load
   kubectl run -i --tty load-generator --rm --image=busybox --restart=Never -- /bin/sh
   # Inside the container:
   while true; do wget -q -O- http://web-app-service; done
   ```

**Expected Outcome:**

- Understanding of Kubernetes architecture
- Experience with container orchestration
- Knowledge of deployment strategies
- Ability to configure auto-scaling

**Key Learning Points:**

- Container vs VM differences
- Kubernetes components and concepts
- Service discovery and networking
- Auto-scaling mechanisms
- Monitoring and observability

---

## Storage and Databases

### Exercise 3.1: Object Storage Implementation

**Scenario**: Sarah needs to build a file sharing application that can handle large files and provide access control.

**Task**: Create a comprehensive file storage solution with S3, including lifecycle management and security.

**Steps**:

1. **Set Up S3 Bucket with Security**

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Sid": "PublicReadGetObject",
         "Effect": "Allow",
         "Principal": "*",
         "Action": "s3:GetObject",
         "Resource": "arn:aws:s3:::my-file-sharing-app/public/*"
       },
       {
         "Sid": "DenyPublicRead",
         "Effect": "Deny",
         "Principal": "*",
         "Action": "s3:PutObject",
         "Resource": "arn:aws:s3:::my-file-sharing-app/public/*",
         "Condition": {
           "Bool": {
             "aws:SecureTransport": "false"
           }
         }
       }
     ]
   }
   ```

2. **Configure Bucket Properties**

   ```yaml
   # CloudFormation template for S3 bucket
   Resources:
     FileSharingBucket:
       Type: AWS::S3::Bucket
       Properties:
         BucketName: my-file-sharing-app-${AWS::AccountId}-${AWS::Region}

         # Enable versioning
         VersioningConfiguration:
           Status: Enabled

         # Configure CORS
         CorsConfiguration:
           CorsRules:
             - AllowedHeaders: ["*"]
               AllowedMethods: ["GET", "POST", "PUT", "DELETE"]
               AllowedOrigins: ["https://myapp.example.com"]
               MaxAge: 3600

         # Enable static website hosting
         WebsiteConfiguration:
           IndexDocument:
             Suffix: index.html
           ErrorDocument:
             Key: error.html

         # Lifecycle rules
         LifecycleConfiguration:
           Rules:
             - Id: DeleteIncompleteMultipartUploads
               Status: Enabled
               AbortIncompleteMultipartUpload:
                 DaysAfterInitiation: 1
             - Id: MoveToGlacierAfter90Days
               Status: Enabled
               Transitions:
                 - TransitionInDays: 90
                   StorageClass: GLACIER
                 - TransitionInDays: 365
                   StorageClass: DEEP_ARCHIVE

         # Tags
         Tags:
           - Key: Environment
             Value: Production
           - Key: Project
             Value: FileSharingApp
           - Key: Owner
             Value: DevTeam
   ```

3. **Create File Upload Service**

   ```python
   # file_upload_service.py
   import boto3
   import json
   import uuid
   from datetime import datetime
   from botocore.exceptions import ClientError

   class FileUploadService:
       def __init__(self):
           self.s3 = boto3.client('s3')
           self.bucket_name = 'my-file-sharing-app'

       def initiate_multipart_upload(self, file_name, file_type, user_id):
           """Initiate multipart upload for large files"""
           try:
               # Generate unique key
               file_id = str(uuid.uuid4())
               key = f"user-files/{user_id}/{file_id}/{file_name}"

               # Create multipart upload
               response = self.s3.create_multipart_upload(
                   Bucket=self.bucket_name,
                   Key=key,
                   ContentType=file_type,
                   Metadata={
                       'file_id': file_id,
                       'user_id': user_id,
                       'file_name': file_name,
                       'upload_timestamp': datetime.now().isoformat()
                   }
               )

               return {
                   'upload_id': response['UploadId'],
                   'file_id': file_id,
                   'key': key,
                   'status': 'initiated'
               }

           except ClientError as e:
               raise Exception(f"Failed to initiate upload: {str(e)}")

       def upload_part(self, upload_id, key, part_number, data):
           """Upload a part of the multipart upload"""
           try:
               response = self.s3.upload_part(
                   Bucket=self.bucket_name,
                   Key=key,
                   PartNumber=part_number,
                   UploadId=upload_id,
                   Body=data
               )

               return {
                   'etag': response['ETag'],
                   'part_number': part_number
               }

           except ClientError as e:
               raise Exception(f"Failed to upload part: {str(e)}")

       def complete_multipart_upload(self, upload_id, key, parts):
           """Complete the multipart upload"""
           try:
               # Prepare parts list
               parts_list = [
                   {'ETag': part['etag'], 'PartNumber': part['part_number']}
                   for part in parts
               ]

               response = self.s3.complete_multipart_upload(
                   Bucket=self.bucket_name,
                   Key=key,
                   UploadId=upload_id,
                   MultipartUpload={'Parts': parts_list}
               )

               return {
                   'location': response['Location'],
                   'bucket': response['Bucket'],
                   'key': response['Key'],
                   'etag': response['ETag'],
                   'status': 'completed'
               }

           except ClientError as e:
               raise Exception(f"Failed to complete upload: {str(e)}")

       def generate_presigned_url(self, key, action='get', expires_in=3600):
           """Generate presigned URL for secure file access"""
           try:
               if action == 'get':
                   url = self.s3.generate_presigned_url(
                       'get_object',
                       Params={'Bucket': self.bucket_name, 'Key': key},
                       ExpiresIn=expires_in
                   )
               elif action == 'put':
                   url = self.s3.generate_presigned_url(
                       'put_object',
                       Params={'Bucket': self.bucket_name, 'Key': key},
                       ExpiresIn=expires_in
                   )

               return url

           except ClientError as e:
               raise Exception(f"Failed to generate presigned URL: {str(e)}")
   ```

4. **Implement File Metadata Management**

   ```python
   # file_metadata_service.py
   import boto3
   from boto3.dynamodb.conditions import Key, Attr
   import json

   class FileMetadataService:
       def __init__(self):
           self.dynamodb = boto3.resource('dynamodb')
           self.table = self.dynamodb.Table('FileMetadata')

       def create_file_record(self, file_id, user_id, file_name, file_type, file_size, s3_key):
           """Create a file metadata record"""
           try:
               file_record = {
                   'file_id': file_id,
                   'user_id': user_id,
                   'file_name': file_name,
                   'file_type': file_type,
                   'file_size': file_size,
                   's3_key': s3_key,
                   'upload_date': '2023-01-01',  # Current date
                   'download_count': 0,
                   'is_public': False,
                   'tags': [],
                   'metadata': {
                       'original_name': file_name,
                       'mime_type': file_type
                   }
               }

               self.table.put_item(Item=file_record)
               return file_record

           except Exception as e:
               raise Exception(f"Failed to create file record: {str(e)}")

       def get_user_files(self, user_id, limit=10):
           """Get all files for a user"""
           try:
               response = self.table.query(
                   KeyConditionExpression=Key('user_id').eq(user_id),
                   Limit=limit,
                   ScanIndexForward=False  # Most recent first
               )

               return response['Items']

           except Exception as e:
               raise Exception(f"Failed to get user files: {str(e)}")

       def update_file_tags(self, file_id, user_id, tags):
           """Update file tags"""
           try:
               self.table.update_item(
                   Key={'file_id': file_id, 'user_id': user_id},
                   UpdateExpression="SET tags = :tags",
                   ExpressionAttributeValues={
                       ':tags': tags
                   }
               )

               return True

           except Exception as e:
               raise Exception(f"Failed to update file tags: {str(e)}")

       def increment_download_count(self, file_id, user_id):
           """Increment file download count"""
           try:
               self.table.update_item(
                   Key={'file_id': file_id, 'user_id': user_id},
                   UpdateExpression="SET download_count = download_count + :val",
                   ExpressionAttributeValues={
                       ':val': 1
                   }
               )

               return True

           except Exception as e:
               raise Exception(f"Failed to increment download count: {str(e)}")
   ```

5. **Create Lambda Function for File Processing**

   ```python
   # file_processing_lambda.py
   import json
   import boto3
   from PIL import Image
   import os

   def lambda_handler(event, context):
       s3 = boto3.client('s3')

       # Process S3 event
       for record in event['Records']:
           bucket = record['s3']['bucket']['name']
           key = record['s3']['object']['key']

           # Skip if not an image
           if not key.lower().endswith(('.png', '.jpg', '.jpeg')):
               continue

           try:
               # Download file
               response = s3.get_object(Bucket=bucket, Key=key)
               image_data = response['Body'].read()

               # Process image
               process_image(image_data, bucket, key)

           except Exception as e:
               print(f"Error processing {key}: {str(e)}")

   def process_image(image_data, bucket, key):
       """Process uploaded image"""
       # Open image
       image = Image.open(io.BytesIO(image_data))

       # Create thumbnail
       thumbnail = image.copy()
       thumbnail.thumbnail((300, 300), Image.Resampling.LANCZOS)

       # Save thumbnail to S3
       thumbnail_key = f"thumbnails/{os.path.basename(key)}"

       buffer = io.BytesIO()
       thumbnail.save(buffer, format='JPEG')
       buffer.seek(0)

       s3.put_object(
           Bucket=bucket,
           Key=thumbnail_key,
           Body=buffer,
           ContentType='image/jpeg',
           Metadata={
               'original_key': key,
               'thumbnail': 'true'
           }
       )
   ```

6. **Test the File Storage System**

   ```python
   # test_file_storage.py
   import requests
   import base64

   def test_file_upload():
       # Test file upload service
       upload_service = FileUploadService()

       # Initiate upload
       result = upload_service.initiate_multipart_upload(
           file_name="test-image.jpg",
           file_type="image/jpeg",
           user_id="user123"
       )

       print(f"Upload initiated: {result}")

   def test_presigned_url():
       # Test presigned URL generation
       service = FileUploadService()

       # Generate presigned URL
       url = service.generate_presigned_url(
           key="user-files/user123/file-id/test-image.jpg",
           action="get",
           expires_in=3600
       )

       print(f"Presigned URL: {url}")

   def test_file_metadata():
       # Test metadata operations
       metadata_service = FileMetadataService()

       # Create file record
       file_record = metadata_service.create_file_record(
           file_id="file-123",
           user_id="user123",
           file_name="test-image.jpg",
           file_type="image/jpeg",
           file_size=1024000,
           s3_key="user-files/user123/file-123/test-image.jpg"
       )

       print(f"File record created: {file_record}")

   if __name__ == "__main__":
       test_file_upload()
       test_presigned_url()
       test_file_metadata()
   ```

**Expected Outcome:**

- Understanding of object storage concepts
- Experience with S3 security and permissions
- Knowledge of multipart uploads for large files
- Integration between S3 and DynamoDB

**Key Learning Points:**

- Object storage vs file system differences
- Presigned URLs for secure access
- Multipart upload for large files
- S3 event processing with Lambda
- Metadata management patterns

### Exercise 3.2: Database Design and Implementation

**Scenario**: Sarah needs to design a database schema for a social media application with user posts, comments, and relationships.

**Task**: Design and implement a NoSQL database schema using DynamoDB with proper partitioning and query patterns.

**Steps**:

1. **Design Database Schema**

   ```python
   # database_schema.py
   """
   Social Media App Database Schema Design

   Table: Users
   - Primary Key: user_id (Partition Key)
   - Attributes: email, name, created_date, profile_info

   Table: Posts
   - Primary Key: post_id (Partition Key)
   - GSI: user_id-index (user_id as partition key, created_date as sort key)
   - Attributes: user_id, content, image_url, created_date, likes_count, comments_count

   Table: Comments
   - Primary Key: comment_id (Partition Key)
   - GSI: post_id-index (post_id as partition key, created_date as sort key)
   - Attributes: post_id, user_id, content, created_date, likes_count

   Table: Likes
   - Primary Key: user_id#post_id (Partition Key)
   - Attributes: post_id, user_id, created_date, like_type
   """

   from enum import Enum
   from dataclasses import dataclass
   from typing import List, Optional, Dict, Any
   from datetime import datetime

   class LikeType(Enum):
       LIKE = "like"
       LOVE = "love"
       LAUGH = "laugh"
       ANGRY = "angry"

   @dataclass
   class User:
       user_id: str
       email: str
       name: str
       created_date: str
       profile_info: Dict[str, Any]
       followers_count: int = 0
       following_count: int = 0
       posts_count: int = 0

   @dataclass
   class Post:
       post_id: str
       user_id: str
       content: str
       image_url: Optional[str] = None
       created_date: str = ""
       updated_date: Optional[str] = None
       likes_count: int = 0
       comments_count: int = 0
       shares_count: int = 0
       tags: List[str] = None
       is_public: bool = True

   @dataclass
   class Comment:
       comment_id: str
       post_id: str
       user_id: str
       content: str
       created_date: str = ""
       likes_count: int = 0
       parent_comment_id: Optional[str] = None
   ```

2. **Create DynamoDB Tables**

   ```yaml
   # DynamoDB Tables Configuration
   Resources:
     UsersTable:
       Type: AWS::DynamoDB::Table
       Properties:
         TableName: Users
         BillingMode: PAY_PER_REQUEST
         AttributeDefinitions:
           - AttributeName: user_id
             AttributeType: S
         KeySchema:
           - AttributeName: user_id
             KeyType: HASH
         PointInTimeRecoverySpecification:
           PointInTimeRecoveryEnabled: true
         SSESpecification:
           SSEEnabled: true
         Tags:
           - Key: Environment
             Value: Production
           - Key: Project
             Value: SocialMediaApp

     PostsTable:
       Type: AWS::DynamoDB::Table
       Properties:
         TableName: Posts
         BillingMode: PAY_PER_REQUEST
         AttributeDefinitions:
           - AttributeName: post_id
             AttributeType: S
           - AttributeName: user_id
             AttributeType: S
           - AttributeName: created_date
             AttributeType: S
         KeySchema:
           - AttributeName: post_id
             KeyType: HASH
         GlobalSecondaryIndexes:
           - IndexName: UserIdIndex
             KeySchema:
               - AttributeName: user_id
                 KeyType: HASH
               - AttributeName: created_date
                 KeyType: RANGE
             Projection:
               ProjectionType: ALL
         PointInTimeRecoverySpecification:
           PointInTimeRecoveryEnabled: true
         SSESpecification:
           SSEEnabled: true

     CommentsTable:
       Type: AWS::DynamoDB::Table
       Properties:
         TableName: Comments
         BillingMode: PAY_PER_REQUEST
         AttributeDefinitions:
           - AttributeName: comment_id
             AttributeType: S
           - AttributeName: post_id
             AttributeType: S
           - AttributeName: created_date
             AttributeType: S
         KeySchema:
           - AttributeName: comment_id
             KeyType: HASH
         GlobalSecondaryIndexes:
           - IndexName: PostIdIndex
             KeySchema:
               - AttributeName: post_id
                 KeyType: HASH
               - AttributeName: created_date
                 KeyType: RANGE
             Projection:
               ProjectionType: ALL
         PointInTimeRecoverySpecification:
           PointInTimeRecoveryEnabled: true
         SSESpecification:
           SSEEnabled: true

     LikesTable:
       Type: AWS::DynamoDB::Table
       Properties:
         TableName: Likes
         BillingMode: PAY_PER_REQUEST
         AttributeDefinitions:
           - AttributeName: user_id
             AttributeType: S
           - AttributeName: post_id
             AttributeType: S
           - AttributeName: created_date
             AttributeType: S
         KeySchema:
           - AttributeName: user_id
             KeyType: HASH
           - AttributeName: post_id
             KeyType: RANGE
         GlobalSecondaryIndexes:
           - IndexName: PostIdIndex
             KeySchema:
               - AttributeName: post_id
                 KeyType: HASH
               - AttributeName: created_date
                 KeyType: RANGE
             Projection:
               ProjectionType: ALL
         PointInTimeRecoverySpecification:
           PointInTimeRecoveryEnabled: true
         SSESpecification:
           SSEEnabled: true
   ```

3. **Implement Data Access Layer**

   ```python
   # data_access_layer.py
   import boto3
   from boto3.dynamodb.conditions import Key, Attr
   import json
   from datetime import datetime
   import uuid

   class SocialMediaDataAccess:
       def __init__(self):
           self.dynamodb = boto3.resource('dynamodb')
           self.users_table = self.dynamodb.Table('Users')
           self.posts_table = self.dynamodb.Table('Posts')
           self.comments_table = self.dynamodb.Table('Comments')
           self.likes_table = self.dynamodb.Table('Likes')

       # User operations
       def create_user(self, email: str, name: str, profile_info: dict = None) -> dict:
           """Create a new user"""
           user_id = str(uuid.uuid4())
           current_time = datetime.now().isoformat()

           user = {
               'user_id': user_id,
               'email': email,
               'name': name,
               'created_date': current_time,
               'profile_info': profile_info or {},
               'followers_count': 0,
               'following_count': 0,
               'posts_count': 0
           }

           self.users_table.put_item(Item=user)
           return user

       def get_user(self, user_id: str) -> dict:
           """Get user by ID"""
           response = self.users_table.get_item(Key={'user_id': user_id})
           return response.get('Item', {})

       def get_user_by_email(self, email: str) -> dict:
           """Get user by email (using GSI)"""
           # In production, you'd create an email GSI
           response = self.users_table.scan(
               FilterExpression=Attr('email').eq(email)
           )
           return response.get('Items', [{}])[0] if response.get('Items') else {}

       # Post operations
       def create_post(self, user_id: str, content: str, image_url: str = None, tags: list = None) -> dict:
           """Create a new post"""
           post_id = str(uuid.uuid4())
           current_time = datetime.now().isoformat()

           post = {
               'post_id': post_id,
               'user_id': user_id,
               'content': content,
               'image_url': image_url,
               'created_date': current_time,
               'likes_count': 0,
               'comments_count': 0,
               'shares_count': 0,
               'tags': tags or []
           }

           # Put post in main table
           self.posts_table.put_item(Item=post)

           # Update user's posts count
           self.users_table.update_item(
               Key={'user_id': user_id},
               UpdateExpression="SET posts_count = posts_count + :val",
               ExpressionAttributeValues={':val': 1}
           )

           return post

       def get_user_posts(self, user_id: str, limit: int = 10) -> list:
           """Get posts for a user"""
           response = self.posts_table.query(
               KeyConditionExpression=Key('user_id').eq(user_id),
               IndexName='UserIdIndex',
               ScanIndexForward=False,  # Most recent first
               Limit=limit
           )
           return response.get('Items', [])

       def get_post(self, post_id: str) -> dict:
           """Get post by ID"""
           response = self.posts_table.get_item(Key={'post_id': post_id})
           return response.get('Item', {})

       def get_posts_by_tags(self, tags: list, limit: int = 20) -> list:
           """Get posts by tags (using GSI)"""
           # This would require a tags GSI for efficient querying
           response = self.posts_table.scan(
               FilterExpression=Attr('tags').contains(tags[0]) if tags else Attr('tags').size().gt(0)
           )
           return response.get('Items', [])[:limit]

       # Comment operations
       def create_comment(self, post_id: str, user_id: str, content: str, parent_comment_id: str = None) -> dict:
           """Create a new comment"""
           comment_id = str(uuid.uuid4())
           current_time = datetime.now().isoformat()

           comment = {
               'comment_id': comment_id,
               'post_id': post_id,
               'user_id': user_id,
               'content': content,
               'created_date': current_time,
               'likes_count': 0,
               'parent_comment_id': parent_comment_id
           }

           # Put comment in main table
           self.comments_table.put_item(Item=comment)

           # Update post's comments count
           self.posts_table.update_item(
               Key={'post_id': post_id},
               UpdateExpression="SET comments_count = comments_count + :val",
               ExpressionAttributeValues={':val': 1}
           )

           return comment

       def get_post_comments(self, post_id: str, limit: int = 20) -> list:
           """Get comments for a post"""
           response = self.comments_table.query(
               KeyConditionExpression=Key('post_id').eq(post_id),
               IndexName='PostIdIndex',
               ScanIndexForward=False,  # Oldest first
               Limit=limit
           )
           return response.get('Items', [])

       # Like operations
       def like_post(self, user_id: str, post_id: str, like_type: str = "like") -> bool:
           """Like a post"""
           current_time = datetime.now().isoformat()

           like_record = {
               'user_id': user_id,
               'post_id': post_id,
               'created_date': current_time,
               'like_type': like_type
           }

           # Add like
           self.likes_table.put_item(Item=like_record)

           # Update post's likes count
           self.posts_table.update_item(
               Key={'post_id': post_id},
               UpdateExpression="SET likes_count = likes_count + :val",
               ExpressionAttributeValues={':val': 1}
           )

           return True

       def unlike_post(self, user_id: str, post_id: str) -> bool:
           """Unlike a post"""
           # Remove like
           self.likes_table.delete_item(Key={'user_id': user_id, 'post_id': post_id})

           # Update post's likes count
           self.posts_table.update_item(
               Key={'post_id': post_id},
               UpdateExpression="SET likes_count = likes_count - :val",
               ExpressionAttributeValues={':val': 1}
           )

           return True

       def get_user_likes(self, user_id: str, limit: int = 50) -> list:
           """Get posts liked by a user"""
           response = self.likes_table.query(
               KeyConditionExpression=Key('user_id').eq(user_id),
               Limit=limit
           )
           return response.get('Items', [])

       # Advanced queries
       def get_user_feed(self, user_id: str, following_user_ids: list, limit: int = 20) -> list:
           """Get user's feed (posts from followed users)"""
           # This is a complex query that might require additional GSIs
           # For simplicity, we'll use a basic approach

           all_posts = []
           for following_id in following_user_ids:
               posts = self.get_user_posts(following_id, limit=10)
               all_posts.extend(posts)

           # Sort by created_date and limit results
           sorted_posts = sorted(all_posts, key=lambda x: x['created_date'], reverse=True)
           return sorted_posts[:limit]
   ```

4. **Implement Cache Layer**

   ```python
   # cache_layer.py
   import redis
   import json
   from datetime import timedelta
   import boto3
   from .data_access_layer import SocialMediaDataAccess

   class SocialMediaCache:
       def __init__(self):
           # Connect to ElastiCache (Redis)
           self.redis_client = redis.Redis(
               host='my-redis-cluster.cache.amazonaws.com',
               port=6379,
               decode_responses=True,
               socket_connect_timeout=5,
               socket_timeout=5,
               socket_keepalive=True,
               socket_keepalive_options={}
           )

           self.data_access = SocialMediaDataAccess()
           self.default_ttl = 3600  # 1 hour

       def get_user(self, user_id: str) -> dict:
           """Get user with caching"""
           cache_key = f"user:{user_id}"

           # Try to get from cache
           cached_user = self.redis_client.get(cache_key)
           if cached_user:
               return json.loads(cached_user)

           # Get from database
           user = self.data_access.get_user(user_id)
           if user:
               # Cache the result
               self.redis_client.setex(
                   cache_key,
                   self.default_ttl,
                   json.dumps(user, default=str)
               )

           return user

       def get_user_posts(self, user_id: str, limit: int = 10) -> list:
           """Get user posts with caching"""
           cache_key = f"user_posts:{user_id}:{limit}"

           # Try to get from cache
           cached_posts = self.redis_client.get(cache_key)
           if cached_posts:
               return json.loads(cached_posts)

           # Get from database
           posts = self.data_access.get_user_posts(user_id, limit)

           # Cache the result
           self.redis_client.setex(
               cache_key,
               self.default_ttl,
               json.dumps(posts, default=str)
           )

           return posts

       def invalidate_user_cache(self, user_id: str):
           """Invalidate user-related cache"""
           patterns_to_delete = [
               f"user:{user_id}",
               f"user_posts:{user_id}:*",
               f"user_feed:{user_id}:*"
           ]

           for pattern in patterns_to_delete:
               keys = self.redis_client.keys(pattern)
               if keys:
                   self.redis_client.delete(*keys)

       def get_post(self, post_id: str) -> dict:
           """Get post with caching"""
           cache_key = f"post:{post_id}"

           # Try to get from cache
           cached_post = self.redis_client.get(cache_key)
           if cached_post:
               return json.loads(cached_post)

           # Get from database
           post = self.data_access.get_post(post_id)
           if post:
               # Cache the result
               self.redis_client.setex(
                   cache_key,
                   self.default_ttl,
                   json.dumps(post, default=str)
               )

           return post

       def increment_post_views(self, post_id: str) -> int:
           """Increment post view count with caching"""
           cache_key = f"post_views:{post_id}"

           # Increment view count in cache
           views = self.redis_client.incr(cache_key)
           self.redis_client.expire(cache_key, 3600)  # 1 hour TTL

           # Optionally, update database asynchronously
           # (not implemented for simplicity)

           return views

       def get_trending_posts(self, timeframe_hours: int = 24) -> list:
           """Get trending posts (simplified implementation)"""
           cache_key = f"trending_posts:{timeframe_hours}h"

           # Try to get from cache
           cached_trending = self.redis_client.get(cache_key)
           if cached_trending:
               return json.loads(cached_trending)

           # In a real implementation, you'd query for posts with
           # high like/comment counts in the last timeframe
           # For now, return some sample data
           trending_posts = []  # Would be populated with actual data

           # Cache the result
           self.redis_client.setex(
               cache_key,
               1800,  # 30 minutes
               json.dumps(trending_posts, default=str)
           )

           return trending_posts
   ```

5. **Performance Testing and Optimization**

   ```python
   # performance_testing.py
   import time
   import concurrent.futures
   from locust import HttpUser, task, between

   class SocialMediaLoadTest(HttpUser):
       wait_time = between(1, 3)

       def on_start(self):
           """Login and get auth token"""
           response = self.client.post("/auth/login", json={
               "email": "test@example.com",
               "password": "testpass123"
           })
           if response.status_code == 200:
               self.token = response.json()["token"]
               self.headers = {"Authorization": f"Bearer {self.token}"}

       @task(3)
       def get_user_profile(self):
           """Test user profile endpoint"""
           self.client.get("/users/profile", headers=self.headers)

       @task(5)
       def get_user_feed(self):
           """Test user feed endpoint"""
           self.client.get("/feed", headers=self.headers)

       @task(2)
       def create_post(self):
           """Test post creation"""
           self.client.post("/posts", json={
               "content": "Test post content",
               "image_url": "https://example.com/image.jpg",
               "tags": ["test", "loadtest"]
           }, headers=self.headers)

       @task(4)
       def get_post_comments(self):
           """Test get comments"""
           self.client.get("/posts/12345/comments", headers=self.headers)
   ```

**Expected Outcome:**

- Understanding of NoSQL database design
- Experience with DynamoDB partitioning and GSIs
- Knowledge of query patterns and access patterns
- Experience with caching strategies
- Performance testing and optimization

**Key Learning Points:**

- Denormalization in NoSQL databases
- Hot partition prevention
- GSI design patterns
- Cache-aside pattern
- Load testing methodologies

This completes the first part of the Cloud Platforms practice exercises. The exercises cover fundamental cloud concepts, compute services, and storage/databases with hands-on implementations. Each exercise includes real-world scenarios, code examples, and practical applications to help understand cloud computing concepts through practice.

The remaining sections would cover networking, security, scalability, cost management, DevOps automation, project-based exercises, and assessment challenges to provide comprehensive hands-on experience with cloud platforms.
