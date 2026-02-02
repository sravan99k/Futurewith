# Backend Development - Practice Exercises

## Table of Contents

1. [API Design and Development Mastery](#api-design-and-development-mastery)
2. [Database Design and Optimization](#database-design-and-optimization)
3. [Authentication and Authorization Implementation](#authentication-and-authorization-implementation)
4. [Microservices Architecture Practice](#microservices-architecture-practice)
5. [Performance Optimization Challenges](#performance-optimization-challenges)
6. [Error Handling and Logging Excellence](#error-handling-and-logging-excellence)
7. [Third-Party Integration Projects](#third-party-integration-projects)
8. [Caching Strategies Implementation](#caching-strategies-implementation)
9. [Message Queue and Event-Driven Systems](#message-queue-and-event-driven-systems)
10. [Production Monitoring and Observability](#production-monitoring-and-observability)

## Practice Exercise 1: API Design and Development Mastery

### Objective

Build comprehensive skills in designing, implementing, and maintaining robust REST and GraphQL APIs.

### Exercise Details

**Time Required**: 3-4 weeks with progressive complexity
**Difficulty**: Intermediate to Advanced

### Week 1: RESTful API Foundation

#### Project: E-Commerce Product Catalog API

**Scenario**: Build a complete product catalog API for an e-commerce platform

**Requirements**:

- Product CRUD operations with categories and inventory
- Advanced filtering and search capabilities
- Pagination and sorting functionality
- Rate limiting and authentication
- Comprehensive error handling
- API documentation and testing

#### Implementation Challenge 1: Resource Design

```python
# API Resource Structure Design Practice

from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_sqlalchemy import SQLAlchemy
from marshmallow import Schema, fields, validate
from datetime import datetime
import uuid

app = Flask(__name__)
api = Api(app)
db = SQLAlchemy(app)

# Database Models
class Category(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    products = db.relationship('Product', backref='category_ref', lazy=True)

class Product(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Decimal(10, 2), nullable=False)
    category_id = db.Column(db.String(36), db.ForeignKey('category.id'), nullable=False)
    stock_quantity = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Advanced features
    attributes = db.Column(db.JSON)  # Flexible product attributes
    search_vector = db.Column(db.Text)  # For full-text search

# Marshmallow Schemas for Validation and Serialization
class ProductSchema(Schema):
    id = fields.Str(dump_only=True)
    name = fields.Str(required=True, validate=validate.Length(min=1, max=200))
    description = fields.Str(missing='')
    price = fields.Decimal(required=True, validate=validate.Range(min=0))
    category_id = fields.Str(required=True)
    stock_quantity = fields.Int(missing=0, validate=validate.Range(min=0))
    is_active = fields.Bool(missing=True)
    attributes = fields.Dict(missing={})
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

class ProductListSchema(Schema):
    """Schema for listing products with pagination"""
    page = fields.Int(missing=1, validate=validate.Range(min=1))
    per_page = fields.Int(missing=20, validate=validate.Range(min=1, max=100))
    category = fields.Str(missing=None)
    min_price = fields.Decimal(missing=None, validate=validate.Range(min=0))
    max_price = fields.Decimal(missing=None, validate=validate.Range(min=0))
    search = fields.Str(missing=None)
    sort_by = fields.Str(missing='created_at', validate=validate.OneOf([
        'name', 'price', 'created_at', 'updated_at'
    ]))
    sort_order = fields.Str(missing='desc', validate=validate.OneOf(['asc', 'desc']))

# API Resource Implementation
class ProductListResource(Resource):
    def __init__(self):
        self.list_schema = ProductListSchema()
        self.product_schema = ProductSchema()

    def get(self):
        """Get paginated list of products with filtering"""
        try:
            # Validate query parameters
            args = self.list_schema.load(request.args)
        except Exception as e:
            return {'errors': e.messages}, 400

        # Build query with filters
        query = Product.query.filter(Product.is_active == True)

        # Category filter
        if args.get('category'):
            query = query.join(Category).filter(
                Category.name.ilike(f"%{args['category']}%")
            )

        # Price range filter
        if args.get('min_price'):
            query = query.filter(Product.price >= args['min_price'])
        if args.get('max_price'):
            query = query.filter(Product.price <= args['max_price'])

        # Search functionality
        if args.get('search'):
            search_term = f"%{args['search']}%"
            query = query.filter(
                db.or_(
                    Product.name.ilike(search_term),
                    Product.description.ilike(search_term),
                    Product.search_vector.contains(args['search'])
                )
            )

        # Sorting
        sort_column = getattr(Product, args['sort_by'])
        if args['sort_order'] == 'desc':
            sort_column = sort_column.desc()
        query = query.order_by(sort_column)

        # Pagination
        pagination = query.paginate(
            page=args['page'],
            per_page=args['per_page'],
            error_out=False
        )

        # Serialize results
        products = self.product_schema.dump(pagination.items, many=True)

        return {
            'products': products,
            'pagination': {
                'page': pagination.page,
                'pages': pagination.pages,
                'per_page': pagination.per_page,
                'total': pagination.total,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            },
            'filters': args
        }, 200

    def post(self):
        """Create a new product"""
        try:
            product_data = self.product_schema.load(request.json)
        except Exception as e:
            return {'errors': e.messages}, 400

        # Validate category exists
        category = Category.query.get(product_data['category_id'])
        if not category:
            return {'errors': {'category_id': 'Category not found'}}, 400

        # Create product
        product = Product(**product_data)

        try:
            db.session.add(product)
            db.session.commit()

            return {
                'product': self.product_schema.dump(product),
                'message': 'Product created successfully'
            }, 201

        except Exception as e:
            db.session.rollback()
            return {'errors': {'database': 'Failed to create product'}}, 500

class ProductResource(Resource):
    def __init__(self):
        self.product_schema = ProductSchema()

    def get(self, product_id):
        """Get a specific product"""
        product = Product.query.filter(
            Product.id == product_id,
            Product.is_active == True
        ).first()

        if not product:
            return {'errors': {'product': 'Product not found'}}, 404

        return {'product': self.product_schema.dump(product)}, 200

    def put(self, product_id):
        """Update a product"""
        product = Product.query.get(product_id)
        if not product:
            return {'errors': {'product': 'Product not found'}}, 404

        try:
            product_data = self.product_schema.load(request.json, partial=True)
        except Exception as e:
            return {'errors': e.messages}, 400

        # Update product attributes
        for key, value in product_data.items():
            setattr(product, key, value)

        product.updated_at = datetime.utcnow()

        try:
            db.session.commit()
            return {
                'product': self.product_schema.dump(product),
                'message': 'Product updated successfully'
            }, 200
        except Exception as e:
            db.session.rollback()
            return {'errors': {'database': 'Failed to update product'}}, 500

    def delete(self, product_id):
        """Soft delete a product"""
        product = Product.query.get(product_id)
        if not product:
            return {'errors': {'product': 'Product not found'}}, 404

        product.is_active = False
        product.updated_at = datetime.utcnow()

        try:
            db.session.commit()
            return {'message': 'Product deleted successfully'}, 200
        except Exception as e:
            db.session.rollback()
            return {'errors': {'database': 'Failed to delete product'}}, 500

# Register API endpoints
api.add_resource(ProductListResource, '/api/v1/products')
api.add_resource(ProductResource, '/api/v1/products/<string:product_id>')
```

#### Implementation Challenge 2: Advanced API Features

```python
# Rate Limiting and Caching Implementation

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import redis
import hashlib
import json

# Configuration
app.config['CACHE_TYPE'] = 'redis'
app.config['CACHE_REDIS_URL'] = 'redis://localhost:6379/0'

# Initialize extensions
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)
cache = Cache(app)

class AdvancedProductListResource(Resource):
    decorators = [limiter.limit("100 per minute")]

    def __init__(self):
        self.list_schema = ProductListSchema()
        self.product_schema = ProductSchema()

    def get(self):
        """Get products with caching and rate limiting"""
        try:
            args = self.list_schema.load(request.args)
        except Exception as e:
            return {'errors': e.messages}, 400

        # Create cache key based on query parameters
        cache_key = self._generate_cache_key(args)

        # Try to get from cache first
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result

        # Build and execute query (same as before)
        query = self._build_query(args)
        pagination = query.paginate(
            page=args['page'],
            per_page=args['per_page'],
            error_out=False
        )

        result = {
            'products': self.product_schema.dump(pagination.items, many=True),
            'pagination': {
                'page': pagination.page,
                'pages': pagination.pages,
                'per_page': pagination.per_page,
                'total': pagination.total,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            },
            'filters': args,
            'cached': False
        }

        # Cache result for 5 minutes
        cache.set(cache_key, result, timeout=300)

        return result, 200

    def _generate_cache_key(self, args):
        """Generate consistent cache key from query parameters"""
        # Sort args to ensure consistent key generation
        sorted_args = sorted(args.items())
        args_string = json.dumps(sorted_args, sort_keys=True)
        return f"products_list:{hashlib.md5(args_string.encode()).hexdigest()}"

    def _build_query(self, args):
        """Build database query based on filters"""
        # Implementation same as before
        pass

# Background Task for Cache Invalidation
from celery import Celery

celery = Celery('api', broker='redis://localhost:6379/1')

@celery.task
def invalidate_product_cache(product_id):
    """Invalidate related cache entries when product changes"""
    # Clear all product list caches (pattern-based deletion)
    redis_client = redis.Redis.from_url('redis://localhost:6379/0')

    # Get all keys matching product list pattern
    keys = redis_client.keys('products_list:*')
    if keys:
        redis_client.delete(*keys)

    # Clear specific product cache
    redis_client.delete(f'product:{product_id}')

# API Versioning Implementation
class ProductResourceV2(ProductResource):
    """Version 2 of Product API with additional features"""

    def get(self, product_id):
        """Enhanced product retrieval with related data"""
        product = Product.query.filter(
            Product.id == product_id,
            Product.is_active == True
        ).first()

        if not product:
            return {'errors': {'product': 'Product not found'}}, 404

        # Enhanced response with related data
        result = self.product_schema.dump(product)

        # Add category information
        result['category'] = {
            'id': product.category_ref.id,
            'name': product.category_ref.name
        }

        # Add related products
        related_products = Product.query.filter(
            Product.category_id == product.category_id,
            Product.id != product.id,
            Product.is_active == True
        ).limit(5).all()

        result['related_products'] = self.product_schema.dump(
            related_products, many=True
        )

        # Add stock status
        result['stock_status'] = self._get_stock_status(product.stock_quantity)

        return {'product': result}, 200

    def _get_stock_status(self, quantity):
        if quantity == 0:
            return 'out_of_stock'
        elif quantity < 10:
            return 'low_stock'
        else:
            return 'in_stock'

# Register versioned endpoints
api.add_resource(ProductResourceV2, '/api/v2/products/<string:product_id>')
```

### Week 2: GraphQL API Development

#### Implementation Challenge: GraphQL Schema and Resolvers

```python
# GraphQL Implementation with Graphene

import graphene
from graphene import relay
from graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyConnectionField
from graphql import GraphQLError

# GraphQL Object Types
class CategoryType(SQLAlchemyObjectType):
    class Meta:
        model = Category
        interfaces = (relay.Node,)

class ProductType(SQLAlchemyObjectType):
    class Meta:
        model = Product
        interfaces = (relay.Node,)

    stock_status = graphene.String()

    def resolve_stock_status(self, info):
        if self.stock_quantity == 0:
            return 'OUT_OF_STOCK'
        elif self.stock_quantity < 10:
            return 'LOW_STOCK'
        return 'IN_STOCK'

# Input Types for Mutations
class ProductInput(graphene.InputObjectType):
    name = graphene.String(required=True)
    description = graphene.String()
    price = graphene.Decimal(required=True)
    category_id = graphene.String(required=True)
    stock_quantity = graphene.Int()
    attributes = graphene.JSONString()

class ProductUpdateInput(graphene.InputObjectType):
    name = graphene.String()
    description = graphene.String()
    price = graphene.Decimal()
    category_id = graphene.String()
    stock_quantity = graphene.Int()
    attributes = graphene.JSONString()

# Mutations
class CreateProduct(graphene.Mutation):
    class Arguments:
        product_data = ProductInput(required=True)

    product = graphene.Field(ProductType)
    success = graphene.Boolean()
    errors = graphene.List(graphene.String)

    def mutate(self, info, product_data):
        try:
            # Validate category exists
            category = Category.query.get(product_data.category_id)
            if not category:
                return CreateProduct(
                    success=False,
                    errors=['Category not found']
                )

            # Create product
            product = Product(
                name=product_data.name,
                description=product_data.description or '',
                price=product_data.price,
                category_id=product_data.category_id,
                stock_quantity=product_data.stock_quantity or 0,
                attributes=product_data.attributes or {}
            )

            db.session.add(product)
            db.session.commit()

            return CreateProduct(
                product=product,
                success=True,
                errors=[]
            )

        except Exception as e:
            db.session.rollback()
            return CreateProduct(
                success=False,
                errors=[str(e)]
            )

class UpdateProduct(graphene.Mutation):
    class Arguments:
        product_id = graphene.String(required=True)
        product_data = ProductUpdateInput(required=True)

    product = graphene.Field(ProductType)
    success = graphene.Boolean()
    errors = graphene.List(graphene.String)

    def mutate(self, info, product_id, product_data):
        try:
            product = Product.query.get(product_id)
            if not product:
                return UpdateProduct(
                    success=False,
                    errors=['Product not found']
                )

            # Update fields
            for field, value in product_data.items():
                if value is not None:
                    setattr(product, field, value)

            product.updated_at = datetime.utcnow()
            db.session.commit()

            return UpdateProduct(
                product=product,
                success=True,
                errors=[]
            )

        except Exception as e:
            db.session.rollback()
            return UpdateProduct(
                success=False,
                errors=[str(e)]
            )

# Queries with Advanced Filtering
class Query(graphene.ObjectType):
    node = relay.Node.Field()

    # Single product query
    product = graphene.Field(
        ProductType,
        id=graphene.String(required=True)
    )

    # Product list with advanced filtering
    products = graphene.List(
        ProductType,
        category=graphene.String(),
        min_price=graphene.Decimal(),
        max_price=graphene.Decimal(),
        search=graphene.String(),
        limit=graphene.Int(default_value=20),
        offset=graphene.Int(default_value=0)
    )

    # Categories list
    categories = graphene.List(CategoryType)

    def resolve_product(self, info, id):
        return Product.query.filter(
            Product.id == id,
            Product.is_active == True
        ).first()

    def resolve_products(self, info, **args):
        query = Product.query.filter(Product.is_active == True)

        # Apply filters
        if args.get('category'):
            query = query.join(Category).filter(
                Category.name.ilike(f"%{args['category']}%")
            )

        if args.get('min_price'):
            query = query.filter(Product.price >= args['min_price'])

        if args.get('max_price'):
            query = query.filter(Product.price <= args['max_price'])

        if args.get('search'):
            search_term = f"%{args['search']}%"
            query = query.filter(
                db.or_(
                    Product.name.ilike(search_term),
                    Product.description.ilike(search_term)
                )
            )

        # Apply pagination
        query = query.offset(args.get('offset', 0))
        query = query.limit(args.get('limit', 20))

        return query.all()

    def resolve_categories(self, info):
        return Category.query.all()

# Mutations
class Mutations(graphene.ObjectType):
    create_product = CreateProduct.Field()
    update_product = UpdateProduct.Field()

# Schema
schema = graphene.Schema(query=Query, mutation=Mutations)

# Flask route for GraphQL
from flask_graphql import GraphQLView

app.add_url_rule(
    '/api/graphql',
    view_func=GraphQLView.as_view(
        'graphql',
        schema=schema,
        graphiql=True  # Enable GraphiQL interface for development
    )
)
```

### Week 3-4: Advanced API Patterns

#### Implementation Challenge: API Gateway and Microservices Integration

```python
# API Gateway Pattern Implementation

from flask import Flask, request, jsonify
import requests
import asyncio
import aiohttp
from circuit_breaker import CircuitBreaker
import logging

class APIGateway:
    def __init__(self):
        self.services = {
            'products': {
                'url': 'http://product-service:5000',
                'circuit_breaker': CircuitBreaker(failure_threshold=5, timeout=30)
            },
            'users': {
                'url': 'http://user-service:5001',
                'circuit_breaker': CircuitBreaker(failure_threshold=5, timeout=30)
            },
            'orders': {
                'url': 'http://order-service:5002',
                'circuit_breaker': CircuitBreaker(failure_threshold=5, timeout=30)
            }
        }
        self.logger = logging.getLogger(__name__)

    async def proxy_request(self, service_name, path, method='GET', **kwargs):
        """Proxy request to microservice with circuit breaker"""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")

        service = self.services[service_name]
        url = f"{service['url']}{path}"

        try:
            with service['circuit_breaker']:
                async with aiohttp.ClientSession() as session:
                    async with session.request(method, url, **kwargs) as response:
                        return {
                            'status_code': response.status,
                            'data': await response.json(),
                            'headers': dict(response.headers)
                        }
        except Exception as e:
            self.logger.error(f"Service {service_name} error: {str(e)}")
            raise

    async def aggregate_data(self, user_id):
        """Aggregate data from multiple services"""
        try:
            # Fetch data from multiple services concurrently
            tasks = [
                self.proxy_request('users', f'/api/users/{user_id}'),
                self.proxy_request('orders', f'/api/orders?user_id={user_id}'),
                self.proxy_request('products', '/api/products?featured=true')
            ]

            user_response, orders_response, products_response = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            # Handle partial failures gracefully
            result = {'user_id': user_id}

            if not isinstance(user_response, Exception):
                result['user'] = user_response['data']
            else:
                result['user'] = None
                self.logger.warning(f"User service failed: {user_response}")

            if not isinstance(orders_response, Exception):
                result['orders'] = orders_response['data']
            else:
                result['orders'] = []
                self.logger.warning(f"Orders service failed: {orders_response}")

            if not isinstance(products_response, Exception):
                result['featured_products'] = products_response['data']
            else:
                result['featured_products'] = []
                self.logger.warning(f"Products service failed: {products_response}")

            return result

        except Exception as e:
            self.logger.error(f"Data aggregation failed: {str(e)}")
            raise

# Authentication Middleware
from functools import wraps
import jwt

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401

        try:
            # Remove 'Bearer ' prefix
            token = token.split(' ')[1] if ' ' in token else token
            payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
            request.user = payload
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

    return decorated_function

# API Composition Endpoint
@app.route('/api/dashboard/<user_id>')
@require_auth
async def user_dashboard(user_id):
    """Composed API endpoint aggregating data from multiple services"""
    try:
        gateway = APIGateway()
        dashboard_data = await gateway.aggregate_data(user_id)

        return jsonify({
            'success': True,
            'data': dashboard_data
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Request/Response Transformation Middleware
class ResponseTransformer:
    """Transform responses to maintain consistent API format"""

    @staticmethod
    def transform_product_response(data):
        """Transform product service response to gateway format"""
        if isinstance(data, list):
            return [ResponseTransformer._transform_single_product(item) for item in data]
        return ResponseTransformer._transform_single_product(data)

    @staticmethod
    def _transform_single_product(product):
        return {
            'id': product.get('id'),
            'name': product.get('name'),
            'price': {
                'amount': float(product.get('price', 0)),
                'currency': 'USD'
            },
            'availability': {
                'in_stock': product.get('stock_quantity', 0) > 0,
                'quantity': product.get('stock_quantity', 0)
            },
            'metadata': {
                'created_at': product.get('created_at'),
                'updated_at': product.get('updated_at')
            }
        }

# API Monitoring and Metrics
from prometheus_client import Counter, Histogram, generate_latest
import time

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'API request latency')

def track_metrics(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()

        try:
            response = f(*args, **kwargs)
            status_code = response[1] if isinstance(response, tuple) else 200

            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.endpoint,
                status=status_code
            ).inc()

            return response

        except Exception as e:
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.endpoint,
                status=500
            ).inc()
            raise

        finally:
            REQUEST_LATENCY.observe(time.time() - start_time)

    return decorated_function

@app.route('/metrics')
def metrics():
    return generate_latest()
```

---

## Practice Exercise 2: Database Design and Optimization

### Objective

Master relational database design, optimization techniques, and advanced querying patterns.

### Exercise Details

**Time Required**: 3-4 weeks with real-world scenarios
**Difficulty**: Intermediate to Advanced

### Week 1: Database Schema Design

#### Project: Social Media Platform Database

**Requirements**: Design database for a Twitter-like social platform

```sql
-- Advanced Database Schema Design

-- Users table with proper indexing
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(100),
    bio TEXT,
    avatar_url VARCHAR(500),
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- Soft delete support
    deleted_at TIMESTAMP WITH TIME ZONE NULL
);

-- Indexes for users table
CREATE INDEX idx_users_username ON users(username) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_email ON users(email) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_last_active ON users(last_active_at DESC);
CREATE INDEX idx_users_created_at ON users(created_at DESC);

-- Posts table with advanced features
CREATE TABLE posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    -- Support for different post types
    post_type VARCHAR(20) DEFAULT 'text' CHECK (post_type IN ('text', 'image', 'video', 'link')),
    media_urls JSONB DEFAULT '[]'::jsonb,
    -- Engagement metrics (denormalized for performance)
    like_count INTEGER DEFAULT 0,
    repost_count INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0,
    -- Hierarchy for replies
    parent_post_id UUID REFERENCES posts(id) ON DELETE CASCADE,
    thread_root_id UUID REFERENCES posts(id) ON DELETE CASCADE,
    -- Content metadata
    hashtags TEXT[] DEFAULT '{}',
    mentioned_users UUID[] DEFAULT '{}',
    -- Moderation
    is_flagged BOOLEAN DEFAULT FALSE,
    is_hidden BOOLEAN DEFAULT FALSE,
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- Full-text search
    search_vector tsvector
);

-- Advanced indexing strategy for posts
CREATE INDEX idx_posts_user_id_created_at ON posts(user_id, created_at DESC);
CREATE INDEX idx_posts_created_at ON posts(created_at DESC) WHERE is_hidden = FALSE;
CREATE INDEX idx_posts_thread_root ON posts(thread_root_id, created_at DESC);
CREATE INDEX idx_posts_parent ON posts(parent_post_id, created_at DESC);
-- GIN indexes for arrays and JSONB
CREATE INDEX idx_posts_hashtags ON posts USING GIN(hashtags);
CREATE INDEX idx_posts_mentioned_users ON posts USING GIN(mentioned_users);
CREATE INDEX idx_posts_media_urls ON posts USING GIN(media_urls);
-- Full-text search index
CREATE INDEX idx_posts_search_vector ON posts USING GIN(search_vector);

-- Trigger to update search vector
CREATE OR REPLACE FUNCTION update_post_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'A') ||
        setweight(to_tsvector('english', array_to_string(NEW.hashtags, ' ')), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trig_update_post_search_vector
    BEFORE INSERT OR UPDATE ON posts
    FOR EACH ROW EXECUTE FUNCTION update_post_search_vector();

-- Followers/Following relationship table
CREATE TABLE user_follows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    follower_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    following_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- Prevent following yourself and duplicate follows
    UNIQUE(follower_id, following_id),
    CHECK(follower_id != following_id)
);

-- Optimized indexes for follow relationships
CREATE INDEX idx_user_follows_follower ON user_follows(follower_id, created_at DESC);
CREATE INDEX idx_user_follows_following ON user_follows(following_id, created_at DESC);

-- Likes table with composite primary key for efficiency
CREATE TABLE post_likes (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, post_id)
);

-- Index for likes (reverse lookup)
CREATE INDEX idx_post_likes_post_created ON post_likes(post_id, created_at DESC);

-- Reposts table
CREATE TABLE post_reposts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    comment TEXT, -- Optional comment on repost
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, post_id) -- Prevent duplicate reposts
);

CREATE INDEX idx_post_reposts_user_created ON post_reposts(user_id, created_at DESC);
CREATE INDEX idx_post_reposts_post_created ON post_reposts(post_id, created_at DESC);

-- Notifications table for real-time features
CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    actor_id UUID REFERENCES users(id) ON DELETE CASCADE, -- Who triggered the notification
    notification_type VARCHAR(50) NOT NULL CHECK (
        notification_type IN ('like', 'repost', 'follow', 'mention', 'reply')
    ),
    target_id UUID, -- ID of the target object (post, user, etc.)
    target_type VARCHAR(50), -- Type of target (post, user, etc.)
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Partitioning notifications by month for better performance
CREATE INDEX idx_notifications_user_created ON notifications(user_id, created_at DESC);
CREATE INDEX idx_notifications_unread ON notifications(user_id, created_at DESC)
    WHERE is_read = FALSE;

-- User activity tracking for analytics
CREATE TABLE user_activities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    activity_type VARCHAR(50) NOT NULL,
    target_id UUID,
    target_type VARCHAR(50),
    metadata JSONB DEFAULT '{}'::jsonb,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Partition by month for better performance
CREATE INDEX idx_user_activities_user_created ON user_activities(user_id, created_at DESC);
CREATE INDEX idx_user_activities_type_created ON user_activities(activity_type, created_at DESC);

-- Trigger functions for maintaining denormalized counts
CREATE OR REPLACE FUNCTION update_post_like_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE posts SET like_count = like_count + 1 WHERE id = NEW.post_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE posts SET like_count = like_count - 1 WHERE id = OLD.post_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trig_update_post_like_count
    AFTER INSERT OR DELETE ON post_likes
    FOR EACH ROW EXECUTE FUNCTION update_post_like_count();

-- Similar triggers for repost_count and reply_count
CREATE OR REPLACE FUNCTION update_post_repost_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE posts SET repost_count = repost_count + 1 WHERE id = NEW.post_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE posts SET repost_count = repost_count - 1 WHERE id = OLD.post_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trig_update_post_repost_count
    AFTER INSERT OR DELETE ON post_reposts
    FOR EACH ROW EXECUTE FUNCTION update_post_repost_count();

-- Views for complex queries
CREATE VIEW user_stats AS
SELECT
    u.id,
    u.username,
    u.display_name,
    COUNT(DISTINCT p.id) as post_count,
    COUNT(DISTINCT f1.following_id) as following_count,
    COUNT(DISTINCT f2.follower_id) as follower_count,
    MAX(p.created_at) as last_post_at
FROM users u
LEFT JOIN posts p ON u.id = p.user_id AND p.is_hidden = FALSE
LEFT JOIN user_follows f1 ON u.id = f1.follower_id
LEFT JOIN user_follows f2 ON u.id = f2.following_id
WHERE u.deleted_at IS NULL
GROUP BY u.id, u.username, u.display_name;

-- Materialized view for trending hashtags
CREATE MATERIALIZED VIEW trending_hashtags AS
SELECT
    hashtag,
    COUNT(*) as usage_count,
    COUNT(DISTINCT user_id) as unique_users,
    MAX(created_at) as last_used
FROM (
    SELECT
        unnest(hashtags) as hashtag,
        user_id,
        created_at
    FROM posts
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    AND is_hidden = FALSE
) hashtag_usage
GROUP BY hashtag
HAVING COUNT(*) >= 5  -- Minimum threshold for trending
ORDER BY usage_count DESC, unique_users DESC
LIMIT 50;

-- Refresh trending hashtags every hour
CREATE INDEX idx_trending_hashtags_usage ON trending_hashtags(usage_count DESC);
```

### Week 2: Query Optimization and Performance

#### Complex Query Optimization Challenges

```sql
-- Challenge 1: Feed Generation Query
-- Generate personalized feed for a user with optimizations

-- Original naive query (DO NOT USE - for educational purposes)
/*
SELECT p.*, u.username, u.display_name, u.avatar_url
FROM posts p
JOIN users u ON p.user_id = u.id
WHERE p.user_id IN (
    SELECT following_id
    FROM user_follows
    WHERE follower_id = :user_id
)
AND p.is_hidden = FALSE
ORDER BY p.created_at DESC
LIMIT 50;
*/

-- Optimized feed query with proper indexing
WITH user_following AS (
    SELECT following_id
    FROM user_follows
    WHERE follower_id = :user_id
),
recent_posts AS (
    SELECT
        p.id,
        p.user_id,
        p.content,
        p.post_type,
        p.media_urls,
        p.like_count,
        p.repost_count,
        p.reply_count,
        p.created_at,
        u.username,
        u.display_name,
        u.avatar_url,
        u.verified,
        -- Check if current user has liked this post
        EXISTS(
            SELECT 1 FROM post_likes pl
            WHERE pl.post_id = p.id AND pl.user_id = :user_id
        ) as user_has_liked,
        -- Check if current user has reposted
        EXISTS(
            SELECT 1 FROM post_reposts pr
            WHERE pr.post_id = p.id AND pr.user_id = :user_id
        ) as user_has_reposted
    FROM posts p
    JOIN users u ON p.user_id = u.id
    WHERE p.user_id IN (SELECT following_id FROM user_following)
    AND p.is_hidden = FALSE
    AND p.created_at >= NOW() - INTERVAL '7 days'  -- Limit to recent posts
    ORDER BY p.created_at DESC
    LIMIT 200  -- Get more than needed for ranking
)
SELECT * FROM recent_posts
ORDER BY
    -- Simple engagement-based ranking
    (like_count * 1.0 + repost_count * 2.0 + reply_count * 1.5) DESC,
    created_at DESC
LIMIT 50;

-- Challenge 2: Efficient user search with ranking
CREATE OR REPLACE FUNCTION search_users(
    search_term TEXT,
    requesting_user_id UUID,
    result_limit INTEGER DEFAULT 20
)
RETURNS TABLE(
    user_id UUID,
    username VARCHAR(50),
    display_name VARCHAR(100),
    bio TEXT,
    avatar_url VARCHAR(500),
    verified BOOLEAN,
    follower_count BIGINT,
    is_following BOOLEAN,
    relevance_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH user_matches AS (
        SELECT
            u.id,
            u.username,
            u.display_name,
            u.bio,
            u.avatar_url,
            u.verified,
            -- Calculate relevance score
            CASE
                WHEN u.username = search_term THEN 100.0
                WHEN u.username ILIKE search_term || '%' THEN 90.0
                WHEN u.display_name ILIKE search_term || '%' THEN 80.0
                WHEN u.username ILIKE '%' || search_term || '%' THEN 70.0
                WHEN u.display_name ILIKE '%' || search_term || '%' THEN 60.0
                WHEN u.bio ILIKE '%' || search_term || '%' THEN 50.0
                ELSE 0.0
            END as base_relevance,
            -- Boost for verified users
            CASE WHEN u.verified THEN 1.2 ELSE 1.0 END as verified_boost
        FROM users u
        WHERE u.deleted_at IS NULL
        AND (
            u.username ILIKE '%' || search_term || '%'
            OR u.display_name ILIKE '%' || search_term || '%'
            OR u.bio ILIKE '%' || search_term || '%'
        )
    ),
    user_stats_enriched AS (
        SELECT
            um.*,
            COALESCE(us.follower_count, 0) as follower_count,
            -- Boost popular users
            CASE
                WHEN COALESCE(us.follower_count, 0) > 10000 THEN 1.5
                WHEN COALESCE(us.follower_count, 0) > 1000 THEN 1.3
                WHEN COALESCE(us.follower_count, 0) > 100 THEN 1.1
                ELSE 1.0
            END as popularity_boost,
            EXISTS(
                SELECT 1 FROM user_follows uf
                WHERE uf.follower_id = requesting_user_id
                AND uf.following_id = um.id
            ) as is_following
        FROM user_matches um
        LEFT JOIN user_stats us ON um.id = us.id
        WHERE um.base_relevance > 0
    )
    SELECT
        use.id,
        use.username,
        use.display_name,
        use.bio,
        use.avatar_url,
        use.verified,
        use.follower_count,
        use.is_following,
        (use.base_relevance * use.verified_boost * use.popularity_boost) as relevance_score
    FROM user_stats_enriched use
    ORDER BY relevance_score DESC, use.follower_count DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Challenge 3: Advanced analytics query
-- Get engagement trends for a user's posts
CREATE OR REPLACE FUNCTION get_user_engagement_analytics(
    target_user_id UUID,
    days_back INTEGER DEFAULT 30
)
RETURNS TABLE(
    date DATE,
    total_posts BIGINT,
    total_likes BIGINT,
    total_reposts BIGINT,
    total_replies BIGINT,
    avg_likes_per_post NUMERIC,
    avg_engagement_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH date_series AS (
        SELECT generate_series(
            (CURRENT_DATE - days_back),
            CURRENT_DATE,
            '1 day'::interval
        )::date AS date
    ),
    daily_post_stats AS (
        SELECT
            p.created_at::date as post_date,
            COUNT(*) as post_count,
            SUM(p.like_count) as total_likes,
            SUM(p.repost_count) as total_reposts,
            SUM(p.reply_count) as total_replies
        FROM posts p
        WHERE p.user_id = target_user_id
        AND p.created_at >= CURRENT_DATE - days_back
        AND p.is_hidden = FALSE
        GROUP BY p.created_at::date
    ),
    user_follower_count AS (
        SELECT COUNT(*) as followers
        FROM user_follows
        WHERE following_id = target_user_id
    )
    SELECT
        ds.date,
        COALESCE(dps.post_count, 0) as total_posts,
        COALESCE(dps.total_likes, 0) as total_likes,
        COALESCE(dps.total_reposts, 0) as total_reposts,
        COALESCE(dps.total_replies, 0) as total_replies,
        CASE
            WHEN COALESCE(dps.post_count, 0) > 0
            THEN ROUND(dps.total_likes::numeric / dps.post_count, 2)
            ELSE 0
        END as avg_likes_per_post,
        CASE
            WHEN COALESCE(dps.post_count, 0) > 0 AND ufc.followers > 0
            THEN ROUND(
                ((dps.total_likes + dps.total_reposts + dps.total_replies)::numeric
                / (dps.post_count * ufc.followers)) * 100, 4
            )
            ELSE 0
        END as avg_engagement_rate
    FROM date_series ds
    LEFT JOIN daily_post_stats dps ON ds.date = dps.post_date
    CROSS JOIN user_follower_count ufc
    ORDER BY ds.date;
END;
$$ LANGUAGE plpgsql;

-- Performance monitoring queries
-- Query to identify slow queries and missing indexes
SELECT
    query,
    calls,
    total_time,
    mean_time,
    stddev_time,
    max_time,
    min_time
FROM pg_stat_statements
WHERE mean_time > 100  -- Queries taking more than 100ms on average
ORDER BY mean_time DESC
LIMIT 20;

-- Query to find unused indexes
SELECT
    indexrelname AS index_name,
    relname AS table_name,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelname::regclass)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND schemaname = 'public'
ORDER BY pg_relation_size(indexrelname::regclass) DESC;

-- Query to analyze table and index sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) -
                   pg_relation_size(schemaname||'.'||tablename)) as index_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Week 3: Database Performance Optimization

#### Advanced Optimization Techniques

```python
# Database optimization implementation in Python

import asyncpg
import asyncio
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager

@dataclass
class QueryMetrics:
    query: str
    execution_time: float
    rows_affected: int
    plan: Optional[Dict] = None

class DatabaseOptimizer:
    def __init__(self, connection_pool: asyncpg.Pool):
        self.pool = connection_pool
        self.logger = logging.getLogger(__name__)
        self.query_cache = {}
        self.metrics = []

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = await self.pool.acquire()
        try:
            yield conn
        finally:
            await self.pool.release(conn)

    async def execute_with_metrics(self, query: str, params: List = None) -> QueryMetrics:
        """Execute query with performance monitoring"""
        import time

        start_time = time.time()

        async with self.get_connection() as conn:
            if params:
                result = await conn.fetch(query, *params)
            else:
                result = await conn.fetch(query)

            execution_time = time.time() - start_time

            # Get query execution plan for analysis
            explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
            try:
                plan_result = await conn.fetch(explain_query, *(params or []))
                plan = plan_result[0]['QUERY PLAN'][0] if plan_result else None
            except Exception:
                plan = None

            metrics = QueryMetrics(
                query=query,
                execution_time=execution_time,
                rows_affected=len(result),
                plan=plan
            )

            self.metrics.append(metrics)

            if execution_time > 1.0:  # Log slow queries
                self.logger.warning(f"Slow query ({execution_time:.2f}s): {query[:100]}...")

            return metrics

    async def optimize_feed_query(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Optimized feed generation with caching and pagination"""

        # Check cache first
        cache_key = f"feed:{user_id}:{limit}"
        if cache_key in self.query_cache:
            cached_result, timestamp = self.query_cache[cache_key]
            if time.time() - timestamp < 300:  # 5 minute cache
                return cached_result

        # Optimized query with proper joins and limits
        query = """
        WITH user_following AS (
            SELECT following_id
            FROM user_follows
            WHERE follower_id = $1
        ),
        ranked_posts AS (
            SELECT
                p.*,
                u.username,
                u.display_name,
                u.avatar_url,
                u.verified,
                -- Engagement-based ranking
                (p.like_count * 1.0 + p.repost_count * 2.0 + p.reply_count * 1.5) as engagement_score,
                ROW_NUMBER() OVER (
                    ORDER BY p.created_at DESC
                ) as recency_rank,
                ROW_NUMBER() OVER (
                    ORDER BY (p.like_count * 1.0 + p.repost_count * 2.0 + p.reply_count * 1.5) DESC
                ) as engagement_rank
            FROM posts p
            JOIN users u ON p.user_id = u.id
            WHERE p.user_id IN (SELECT following_id FROM user_following)
            AND p.is_hidden = FALSE
            AND p.created_at >= NOW() - INTERVAL '3 days'
            ORDER BY p.created_at DESC
            LIMIT 200
        )
        SELECT *
        FROM ranked_posts
        ORDER BY
            CASE
                WHEN engagement_rank <= 10 THEN engagement_score
                ELSE created_at
            END DESC
        LIMIT $2;
        """

        metrics = await self.execute_with_metrics(query, [user_id, limit])

        # Cache successful results
        if metrics.execution_time < 2.0:  # Only cache fast queries
            self.query_cache[cache_key] = (metrics, time.time())

        return metrics

    async def batch_insert_optimized(self, table: str, records: List[Dict]) -> None:
        """Optimized batch insert using COPY"""
        if not records:
            return

        # Use COPY for large batch inserts (much faster than individual INSERTs)
        if len(records) > 100:
            await self._copy_insert(table, records)
        else:
            await self._batch_insert(table, records)

    async def _copy_insert(self, table: str, records: List[Dict]) -> None:
        """Use PostgreSQL COPY for maximum performance"""
        if not records:
            return

        columns = list(records[0].keys())

        async with self.get_connection() as conn:
            # Create temporary file-like object for COPY
            import io
            import csv

            csv_data = io.StringIO()
            writer = csv.DictWriter(csv_data, fieldnames=columns)
            writer.writerows(records)
            csv_data.seek(0)

            # Use COPY command
            await conn.copy_to_table(
                table,
                source=csv_data,
                columns=columns,
                format='csv'
            )

    async def _batch_insert(self, table: str, records: List[Dict]) -> None:
        """Use batch INSERT for smaller datasets"""
        if not records:
            return

        columns = list(records[0].keys())
        placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
        values_list = []

        for i, record in enumerate(records):
            record_placeholders = ', '.join([
                f'${j + i * len(columns) + 1}'
                for j in range(len(columns))
            ])
            values_list.append(f'({record_placeholders})')

        query = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES {', '.join(values_list)}
        """

        # Flatten parameters
        params = []
        for record in records:
            params.extend([record[col] for col in columns])

        await self.execute_with_metrics(query, params)

    async def analyze_query_performance(self) -> Dict:
        """Analyze collected query metrics"""
        if not self.metrics:
            return {}

        total_queries = len(self.metrics)
        slow_queries = [m for m in self.metrics if m.execution_time > 1.0]

        avg_execution_time = sum(m.execution_time for m in self.metrics) / total_queries

        # Group by similar queries (simplified)
        query_patterns = {}
        for metric in self.metrics:
            pattern = metric.query.split('WHERE')[0].strip()  # Simplified grouping
            if pattern not in query_patterns:
                query_patterns[pattern] = []
            query_patterns[pattern].append(metric)

        return {
            'total_queries': total_queries,
            'slow_queries_count': len(slow_queries),
            'average_execution_time': avg_execution_time,
            'slowest_query': max(self.metrics, key=lambda m: m.execution_time),
            'query_patterns': {
                pattern: {
                    'count': len(metrics),
                    'avg_time': sum(m.execution_time for m in metrics) / len(metrics),
                    'max_time': max(m.execution_time for m in metrics)
                }
                for pattern, metrics in query_patterns.items()
            }
        }

# Connection pool setup with optimization
async def create_optimized_pool():
    return await asyncpg.create_pool(
        "postgresql://user:pass@localhost/db",
        min_size=10,
        max_size=50,
        command_timeout=60,
        server_settings={
            'application_name': 'social_media_api',
            'jit': 'off',  # Disable JIT for predictable performance
        }
    )

# Example usage
async def main():
    pool = await create_optimized_pool()
    optimizer = DatabaseOptimizer(pool)

    # Example: Optimized feed generation
    feed_metrics = await optimizer.optimize_feed_query("user-123", limit=50)

    # Example: Batch insert optimization
    notifications = [
        {
            'user_id': f'user-{i}',
            'notification_type': 'like',
            'target_id': 'post-123',
            'is_read': False
        }
        for i in range(1000)
    ]

    await optimizer.batch_insert_optimized('notifications', notifications)

    # Analyze performance
    analysis = await optimizer.analyze_query_performance()
    print(f"Query analysis: {analysis}")

    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Additional Practice Exercises

### Exercise 3: Authentication and Authorization Implementation

**Focus**: JWT tokens, OAuth integration, role-based access control, security best practices
**Duration**: 2-3 weeks
**Skills**: Security patterns, token management, session handling, multi-factor authentication

### Exercise 4: Microservices Architecture Practice

**Focus**: Service decomposition, inter-service communication, distributed system patterns
**Duration**: 4-6 weeks
**Skills**: Service design, API composition, distributed transactions, service discovery

### Exercise 5: Performance Optimization Challenges

**Focus**: Application profiling, bottleneck identification, optimization techniques
**Duration**: 2-3 weeks
**Skills**: Performance monitoring, caching strategies, database optimization, code profiling

### Exercise 6: Error Handling and Logging Excellence

**Focus**: Comprehensive error handling, structured logging, monitoring integration
**Duration**: 2 weeks
**Skills**: Error patterns, logging frameworks, alerting, debugging techniques

### Exercise 7: Third-Party Integration Projects

**Focus**: Payment processing, social media APIs, cloud services integration
**Duration**: 3-4 weeks
**Skills**: API integration, webhook handling, rate limiting, fallback strategies

### Exercise 8: Caching Strategies Implementation

**Focus**: Multi-level caching, cache invalidation, distributed caching
**Duration**: 2-3 weeks
**Skills**: Cache patterns, Redis, CDN integration, cache coherency

### Exercise 9: Message Queue and Event-Driven Systems

**Focus**: Asynchronous processing, event sourcing, message queues
**Duration**: 3-4 weeks
**Skills**: Queue management, event design, eventual consistency, error recovery

### Exercise 10: Production Monitoring and Observability

**Focus**: Metrics collection, alerting, distributed tracing, log aggregation
**Duration**: 2-3 weeks
**Skills**: Observability patterns, monitoring tools, SLA management, incident response

---

## Monthly Backend Development Assessment

### Technical Skills Self-Evaluation

Rate your proficiency (1-10) in each area:

**API Development**:

- [ ] REST API design and implementation
- [ ] GraphQL schema and resolver development
- [ ] API documentation and testing
- [ ] Versioning and backward compatibility

**Database Management**:

- [ ] Relational database design and optimization
- [ ] Query optimization and performance tuning
- [ ] NoSQL database usage and patterns
- [ ] Database migrations and schema management

**System Architecture**:

- [ ] Microservices design and implementation
- [ ] Distributed system patterns and practices
- [ ] Scalability and performance optimization
- [ ] Security implementation and best practices

**Integration and Deployment**:

- [ ] Third-party API integration
- [ ] Message queue and event-driven architecture
- [ ] Caching strategies and implementation
- [ ] Monitoring and observability setup

### Growth Planning Framework

1. **Technical Strengths**: Which backend technologies and patterns do you master?
2. **Architecture Understanding**: How well do you design scalable systems?
3. **Performance Optimization**: Can you identify and resolve bottlenecks?
4. **Security Awareness**: Do you implement comprehensive security measures?
5. **Integration Skills**: How effectively do you work with external systems?
6. **Monitoring Proficiency**: Can you build observable and maintainable systems?

### Continuous Learning Recommendations

- Build progressively complex backend projects
- Study high-scale system architectures (case studies from major tech companies)
- Practice database design with real-world scenarios
- Implement different architectural patterns (microservices, event sourcing, CQRS)
- Contribute to open-source backend projects
- Stay updated with backend technology trends and best practices

Remember: Backend development is about building reliable, scalable, and maintainable systems. Focus on understanding the underlying principles and trade-offs in system design.
