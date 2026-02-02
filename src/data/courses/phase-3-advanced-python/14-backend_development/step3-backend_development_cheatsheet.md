# Backend Development - Quick Reference Cheatsheet

## 🚀 Core Concepts

### RESTful API Principles

```
GET     /users          # Retrieve all users
GET     /users/{id}     # Retrieve specific user
POST    /users          # Create new user
PUT     /users/{id}     # Update entire user
PATCH   /users/{id}     # Partial update
DELETE  /users/{id}     # Delete user
```

### HTTP Status Codes

```
2xx Success
200 OK                  # Successful GET, PUT, PATCH
201 Created            # Successful POST
204 No Content         # Successful DELETE

4xx Client Error
400 Bad Request        # Invalid syntax
401 Unauthorized       # Authentication required
403 Forbidden         # Access denied
404 Not Found         # Resource not found
422 Unprocessable     # Validation errors

5xx Server Error
500 Internal Error    # Server error
502 Bad Gateway       # Invalid response
503 Service Unavailable
```

## 🔧 Node.js/Express Quick Setup

### Basic Express Server

```javascript
const express = require("express");
const app = express();

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.get("/api/users", (req, res) => {
  res.json({ users: [] });
});

app.post("/api/users", (req, res) => {
  const { name, email } = req.body;
  // Validation and creation logic
  res.status(201).json({ id: 1, name, email });
});

app.listen(3000, () => console.log("Server running on port 3000"));
```

### Middleware Patterns

```javascript
// Authentication Middleware
const authMiddleware = (req, res, next) => {
  const token = req.headers.authorization?.split(" ")[1];
  if (!token) return res.status(401).json({ error: "No token" });

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (error) {
    res.status(401).json({ error: "Invalid token" });
  }
};

// Error Handling Middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: "Something broke!" });
});
```

## 🐍 Python/FastAPI Quick Setup

### Basic FastAPI App

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    email: str

@app.get("/users")
async def get_users():
    return {"users": []}

@app.post("/users")
async def create_user(user: User):
    # Validation happens automatically
    return {"id": 1, **user.dict()}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if user_id < 1:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user_id, "name": "John"}

# Run with: uvicorn main:app --reload
```

### Dependency Injection

```python
from fastapi import Depends

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users")
async def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()
```

## 🗄️ Database Integration

### MongoDB with Mongoose

```javascript
const mongoose = require("mongoose");

// Schema Definition
const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  createdAt: { type: Date, default: Date.now },
});

const User = mongoose.model("User", userSchema);

// CRUD Operations
// Create
const user = new User({ name: "John", email: "john@example.com" });
await user.save();

// Read
const users = await User.find({ active: true });
const user = await User.findById(id);

// Update
await User.findByIdAndUpdate(id, { name: "John Updated" });

// Delete
await User.findByIdAndDelete(id);
```

### PostgreSQL with Sequelize

```javascript
const { Sequelize, DataTypes } = require("sequelize");

const sequelize = new Sequelize(DATABASE_URL);

const User = sequelize.define("User", {
  name: DataTypes.STRING,
  email: {
    type: DataTypes.STRING,
    allowNull: false,
    unique: true,
  },
});

// Associations
User.hasMany(Post);
Post.belongsTo(User);

// Queries
const users = await User.findAll({
  include: [Post],
  where: { active: true },
  order: [["createdAt", "DESC"]],
  limit: 10,
});
```

### SQLAlchemy (Python)

```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)

# Session management
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Usage
db = SessionLocal()
user = User(name="John", email="john@example.com")
db.add(user)
db.commit()
```

## 🔐 Authentication & Authorization

### JWT Implementation (Node.js)

```javascript
const jwt = require("jsonwebtoken");
const bcrypt = require("bcrypt");

// Generate Token
const generateToken = (payload) => {
  return jwt.sign(payload, process.env.JWT_SECRET, { expiresIn: "24h" });
};

// Hash Password
const hashPassword = async (password) => {
  return await bcrypt.hash(password, 12);
};

// Verify Password
const verifyPassword = async (password, hashedPassword) => {
  return await bcrypt.compare(password, hashedPassword);
};

// Login Route
app.post("/auth/login", async (req, res) => {
  const { email, password } = req.body;

  const user = await User.findOne({ email });
  if (!user) return res.status(401).json({ error: "Invalid credentials" });

  const isValid = await verifyPassword(password, user.password);
  if (!isValid) return res.status(401).json({ error: "Invalid credentials" });

  const token = generateToken({ id: user.id, email: user.email });
  res.json({ token, user: { id: user.id, name: user.name } });
});
```

### OAuth 2.0 Flow

```javascript
// Google OAuth with Passport.js
const GoogleStrategy = require("passport-google-oauth20").Strategy;

passport.use(
  new GoogleStrategy(
    {
      clientID: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      callbackURL: "/auth/google/callback",
    },
    async (accessToken, refreshToken, profile, done) => {
      // Find or create user
      const user = await User.findOrCreate({
        where: { googleId: profile.id },
        defaults: {
          name: profile.displayName,
          email: profile.emails[0].value,
        },
      });
      return done(null, user);
    },
  ),
);

app.get(
  "/auth/google",
  passport.authenticate("google", { scope: ["profile", "email"] }),
);
app.get(
  "/auth/google/callback",
  passport.authenticate("google"),
  (req, res) => {
    res.redirect("/dashboard");
  },
);
```

## 🔄 API Design Patterns

### Repository Pattern

```javascript
class UserRepository {
  async findById(id) {
    return await User.findById(id);
  }

  async findByEmail(email) {
    return await User.findOne({ email });
  }

  async create(userData) {
    const user = new User(userData);
    return await user.save();
  }

  async update(id, data) {
    return await User.findByIdAndUpdate(id, data, { new: true });
  }

  async delete(id) {
    return await User.findByIdAndDelete(id);
  }
}

class UserService {
  constructor(userRepository) {
    this.userRepository = userRepository;
  }

  async createUser(userData) {
    // Business logic
    const existingUser = await this.userRepository.findByEmail(userData.email);
    if (existingUser) throw new Error("User already exists");

    return await this.userRepository.create(userData);
  }
}
```

### Controller Pattern

```javascript
class UserController {
  constructor(userService) {
    this.userService = userService;
  }

  async getUsers(req, res, next) {
    try {
      const users = await this.userService.getAllUsers();
      res.json(users);
    } catch (error) {
      next(error);
    }
  }

  async createUser(req, res, next) {
    try {
      const user = await this.userService.createUser(req.body);
      res.status(201).json(user);
    } catch (error) {
      next(error);
    }
  }
}
```

## 📝 Input Validation

### Express Validator

```javascript
const { body, validationResult } = require("express-validator");

const userValidation = [
  body("name").notEmpty().trim().isLength({ min: 2, max: 50 }),
  body("email").isEmail().normalizeEmail(),
  body("password")
    .isLength({ min: 8 })
    .matches(/^(?=.*[A-Za-z])(?=.*\d)/),
];

app.post("/users", userValidation, (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(422).json({ errors: errors.array() });
  }

  // Proceed with user creation
});
```

### Joi Validation

```javascript
const Joi = require("joi");

const userSchema = Joi.object({
  name: Joi.string().min(2).max(50).required(),
  email: Joi.string().email().required(),
  password: Joi.string()
    .min(8)
    .pattern(/^(?=.*[A-Za-z])(?=.*\d)/)
    .required(),
  age: Joi.number().min(18).max(120).optional(),
});

const validateUser = (req, res, next) => {
  const { error } = userSchema.validate(req.body);
  if (error) {
    return res.status(400).json({ error: error.details[0].message });
  }
  next();
};
```

## 🚦 Error Handling

### Custom Error Classes

```javascript
class AppError extends Error {
  constructor(message, statusCode, isOperational = true) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = isOperational;
    Error.captureStackTrace(this, this.constructor);
  }
}

class ValidationError extends AppError {
  constructor(message) {
    super(message, 400);
  }
}

class NotFoundError extends AppError {
  constructor(resource) {
    super(`${resource} not found`, 404);
  }
}

// Global Error Handler
app.use((err, req, res, next) => {
  if (err instanceof AppError && err.isOperational) {
    return res.status(err.statusCode).json({
      error: err.message,
      status: "error",
    });
  }

  // Log unknown errors
  console.error(err);
  res.status(500).json({
    error: "Internal server error",
    status: "error",
  });
});
```

## 📊 API Documentation

### Swagger/OpenAPI Setup

```javascript
const swaggerJsdoc = require("swagger-jsdoc");
const swaggerUi = require("swagger-ui-express");

const options = {
  definition: {
    openapi: "3.0.0",
    info: {
      title: "My API",
      version: "1.0.0",
    },
  },
  apis: ["./routes/*.js"],
};

const specs = swaggerJsdoc(options);
app.use("/api-docs", swaggerUi.serve, swaggerUi.setup(specs));

/**
 * @swagger
 * /users:
 *   post:
 *     summary: Create a new user
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - name
 *               - email
 *             properties:
 *               name:
 *                 type: string
 *               email:
 *                 type: string
 */
```

## 🧪 Testing Patterns

### Unit Testing with Jest

```javascript
// userService.test.js
const UserService = require("./userService");

describe("UserService", () => {
  let userService;
  let mockRepository;

  beforeEach(() => {
    mockRepository = {
      findByEmail: jest.fn(),
      create: jest.fn(),
    };
    userService = new UserService(mockRepository);
  });

  test("should create user when email is unique", async () => {
    const userData = { name: "John", email: "john@example.com" };
    mockRepository.findByEmail.mockResolvedValue(null);
    mockRepository.create.mockResolvedValue({ id: 1, ...userData });

    const result = await userService.createUser(userData);

    expect(mockRepository.findByEmail).toHaveBeenCalledWith(userData.email);
    expect(mockRepository.create).toHaveBeenCalledWith(userData);
    expect(result).toEqual({ id: 1, ...userData });
  });
});
```

### Integration Testing with Supertest

```javascript
const request = require("supertest");
const app = require("../app");

describe("POST /api/users", () => {
  test("should create user with valid data", async () => {
    const userData = {
      name: "John Doe",
      email: "john@example.com",
      password: "SecurePass123",
    };

    const response = await request(app)
      .post("/api/users")
      .send(userData)
      .expect(201);

    expect(response.body).toHaveProperty("id");
    expect(response.body.name).toBe(userData.name);
  });
});
```

## 🔄 Caching Strategies

### Redis Implementation

```javascript
const redis = require("redis");
const client = redis.createClient();

// Cache middleware
const cache = (duration = 300) => {
  return async (req, res, next) => {
    const key = req.originalUrl;

    try {
      const cached = await client.get(key);
      if (cached) {
        return res.json(JSON.parse(cached));
      }
    } catch (error) {
      console.error("Cache error:", error);
    }

    // Intercept res.json to cache response
    const originalJson = res.json;
    res.json = function (data) {
      client.setex(key, duration, JSON.stringify(data));
      return originalJson.call(this, data);
    };

    next();
  };
};

// Usage
app.get("/api/users", cache(600), async (req, res) => {
  const users = await User.findAll();
  res.json(users);
});
```

## 📈 Performance Optimization

### Database Query Optimization

```javascript
// Inefficient
const posts = await Post.findAll();
const postsWithUsers = await Promise.all(
  posts.map(async (post) => {
    const user = await User.findById(post.userId);
    return { ...post, user };
  }),
);

// Efficient
const posts = await Post.findAll({
  include: [{ model: User }],
});
```

### Pagination

```javascript
app.get("/api/users", async (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 10;
  const offset = (page - 1) * limit;

  const { rows: users, count } = await User.findAndCountAll({
    offset,
    limit,
    order: [["createdAt", "DESC"]],
  });

  res.json({
    users,
    pagination: {
      page,
      limit,
      total: count,
      pages: Math.ceil(count / limit),
    },
  });
});
```

## 🔒 Security Best Practices

### Rate Limiting

```javascript
const rateLimit = require("express-rate-limit");

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: "Too many requests from this IP",
  standardHeaders: true,
  legacyHeaders: false,
});

app.use("/api/", limiter);

// Stricter rate limiting for auth endpoints
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
  skipSuccessfulRequests: true,
});

app.use("/api/auth", authLimiter);
```

### Security Headers

```javascript
const helmet = require("helmet");

app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        scriptSrc: ["'self'"],
        imgSrc: ["'self'", "data:", "https:"],
      },
    },
  }),
);
```

## 🚀 Deployment Checklist

### Environment Configuration

```bash
# .env
NODE_ENV=production
PORT=3000
DATABASE_URL=postgresql://user:pass@host:5432/dbname
JWT_SECRET=your-super-secret-key
REDIS_URL=redis://localhost:6379
```

### Production Optimizations

```javascript
// Compression
const compression = require("compression");
app.use(compression());

// CORS
const cors = require("cors");
app.use(
  cors({
    origin: process.env.ALLOWED_ORIGINS?.split(","),
    credentials: true,
  }),
);

// Health check
app.get("/health", (req, res) => {
  res.json({
    status: "OK",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  });
});

// Graceful shutdown
process.on("SIGTERM", () => {
  console.log("SIGTERM received, shutting down gracefully");
  server.close(() => {
    console.log("Process terminated");
  });
});
```

## 🛠️ Useful Commands

```bash
# Node.js package management
npm init -y
npm install express mongoose dotenv
npm install --save-dev nodemon jest supertest

# Start development server
npm run dev

# Run tests
npm test
npm run test:watch
npm run test:coverage

# Database migrations (Sequelize)
npx sequelize-cli db:migrate
npx sequelize-cli db:seed:all

# Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install Python dependencies
pip install fastapi uvicorn sqlalchemy psycopg2-binary

# Run FastAPI server
uvicorn main:app --reload
```

---

_Quick reference for backend development fundamentals and common patterns_
