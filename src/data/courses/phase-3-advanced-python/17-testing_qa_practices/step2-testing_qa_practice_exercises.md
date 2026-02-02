# Testing & QA Practices - Practice Exercises

## Overview

This file contains hands-on exercises, projects, and real-world scenarios to practice testing and QA skills. Complete these exercises to build practical expertise in testing methodologies, automation frameworks, and quality assurance practices.

## Exercise Categories

- 🧪 Unit Testing Labs
- 🔗 Integration Testing Projects
- 🌐 End-to-End Testing Scenarios
- 🤖 Test Automation Challenges
- 📊 Performance Testing Exercises
- 🔒 Security Testing Labs
- 📱 Mobile Testing Projects
- ♿ Accessibility Testing Challenges
- 📈 Test Strategy & Planning
- 🛠️ CI/CD Testing Integration

---

## 🧪 Unit Testing Labs

### Lab 1: JavaScript Unit Testing with Jest

**Objective**: Create comprehensive unit tests for a JavaScript utility library

**Setup**:

```bash
mkdir testing-lab-1
cd testing-lab-1
npm init -y
npm install --save-dev jest
```

**Task**: Create tests for the following utility functions:

```javascript
// utils.js
function validateEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

function calculateTax(price, rate) {
  if (typeof price !== "number" || typeof rate !== "number") {
    throw new Error("Price and rate must be numbers");
  }
  if (price < 0 || rate < 0) {
    throw new Error("Price and rate must be positive");
  }
  return price * (rate / 100);
}

function formatCurrency(amount, currency = "USD") {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: currency,
  }).format(amount);
}

function debounce(func, delay) {
  let timeoutId;
  return function (...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(this, args), delay);
  };
}
```

**Requirements**:

1. Write tests covering all edge cases
2. Achieve 100% code coverage
3. Use describe/it blocks for organization
4. Include setup and teardown where needed
5. Test error conditions
6. Use mocks for debounce function testing

**Expected Test File Structure**:

```javascript
// utils.test.js
describe("Email Validation", () => {
  describe("Valid emails", () => {
    // Test cases
  });

  describe("Invalid emails", () => {
    // Test cases
  });
});

describe("Tax Calculation", () => {
  // Test cases
});
// ... etc
```

### Lab 2: Python Unit Testing with Pytest

**Objective**: Build comprehensive tests for a Python class-based system

**Task**: Create tests for a Library Management System:

```python
# library.py
from datetime import datetime, timedelta
from typing import List, Optional

class Book:
    def __init__(self, isbn: str, title: str, author: str, copies: int = 1):
        self.isbn = isbn
        self.title = title
        self.author = author
        self.copies = copies
        self.available_copies = copies

class Member:
    def __init__(self, member_id: str, name: str, email: str):
        self.member_id = member_id
        self.name = name
        self.email = email
        self.borrowed_books: List[str] = []
        self.overdue_books: List[str] = []

class Library:
    def __init__(self):
        self.books = {}
        self.members = {}
        self.loans = {}

    def add_book(self, book: Book) -> None:
        if book.isbn in self.books:
            self.books[book.isbn].copies += book.copies
            self.books[book.isbn].available_copies += book.copies
        else:
            self.books[book.isbn] = book

    def register_member(self, member: Member) -> None:
        if member.member_id in self.members:
            raise ValueError("Member already exists")
        self.members[member.member_id] = member

    def borrow_book(self, member_id: str, isbn: str) -> bool:
        # Implementation details...
        pass

    def return_book(self, member_id: str, isbn: str) -> bool:
        # Implementation details...
        pass
```

**Requirements**:

1. Use fixtures for test data setup
2. Parametrize tests where appropriate
3. Test exception handling
4. Mock datetime for due date testing
5. Use pytest markers for test categories
6. Create custom test classes for organization

### Lab 3: Java Unit Testing with JUnit 5

**Objective**: Master modern JUnit testing patterns

**Task**: Test a Banking System with the following classes:

```java
// Account.java
public class Account {
    private String accountNumber;
    private double balance;
    private AccountType type;
    private List<Transaction> transactions;

    public Account(String accountNumber, AccountType type) {
        // Constructor implementation
    }

    public void deposit(double amount) throws InvalidAmountException {
        // Implementation
    }

    public void withdraw(double amount) throws InsufficientFundsException {
        // Implementation
    }

    public double calculateInterest() {
        // Implementation based on account type
    }
}
```

**Requirements**:

1. Use @Test, @BeforeEach, @AfterEach annotations
2. Implement parameterized tests with @ParameterizedTest
3. Use @TestMethodOrder for execution order
4. Create custom assertions
5. Use @Timeout for performance testing
6. Implement @Nested test classes

---

## 🔗 Integration Testing Projects

### Project 1: REST API Integration Testing

**Objective**: Test a complete REST API with database integration

**Scenario**: E-commerce API with User, Product, and Order endpoints

**Setup Requirements**:

- Node.js/Express API or Python/FastAPI
- PostgreSQL or MongoDB database
- Authentication middleware
- Rate limiting
- Input validation

**Test Scenarios**:

1. **User Management Flow**:

   ```javascript
   describe("User Management Integration", () => {
     test("Complete user lifecycle", async () => {
       // 1. Register new user
       const registerResponse = await request(app)
         .post("/api/auth/register")
         .send(validUserData);

       // 2. Verify user in database
       // 3. Login user
       // 4. Access protected route
       // 5. Update user profile
       // 6. Delete user
     });
   });
   ```

2. **Order Processing Flow**:
   - Create user account
   - Add products to cart
   - Apply discount codes
   - Process payment
   - Generate order confirmation
   - Update inventory
   - Send email notification

**Testing Tools**: Supertest, Jest/Mocha, Database seeding

### Project 2: Microservices Integration Testing

**Objective**: Test communication between multiple services

**Architecture**:

- User Service (Node.js)
- Product Service (Python)
- Order Service (Java)
- Notification Service (Go)
- API Gateway (NGINX)

**Test Scenarios**:

1. **Service-to-Service Communication**
2. **Circuit Breaker Testing**
3. **Database Transaction Testing**
4. **Message Queue Integration**
5. **Service Discovery Testing**

**Tools**: Docker Compose, Testcontainers, Wiremock

### Project 3: Database Integration Testing

**Objective**: Test complex database operations and transactions

**Scenarios**:

1. **Transaction Rollback Testing**
2. **Concurrent User Testing**
3. **Data Migration Testing**
4. **Backup and Recovery Testing**
5. **Performance Under Load**

---

## 🌐 End-to-End Testing Scenarios

### Scenario 1: E-commerce User Journey

**Objective**: Automate complete user workflows using Playwright

**Test Flow**:

```javascript
// e2e/ecommerce.spec.js
describe("E-commerce User Journey", () => {
  let page;

  beforeEach(async () => {
    page = await browser.newPage();
    await page.goto("/");
  });

  test("Complete purchase flow", async () => {
    // 1. User registration
    await page.click('[data-testid="register-button"]');
    await page.fill("#email", "test@example.com");
    await page.fill("#password", "SecurePass123");
    await page.click('[data-testid="submit-register"]');

    // 2. Product search and selection
    await page.fill('[data-testid="search-input"]', "laptop");
    await page.press('[data-testid="search-input"]', "Enter");
    await page.click(".product-item:first-child");

    // 3. Add to cart
    await page.selectOption("#size", "Medium");
    await page.selectOption("#color", "Blue");
    await page.click('[data-testid="add-to-cart"]');

    // 4. Checkout process
    await page.click('[data-testid="cart-icon"]');
    await page.click('[data-testid="checkout-button"]');

    // 5. Payment and confirmation
    await fillPaymentForm(page);
    await page.click('[data-testid="place-order"]');

    // 6. Verify order confirmation
    await expect(page.locator(".order-confirmation")).toBeVisible();
    const orderNumber = await page.textContent(".order-number");
    expect(orderNumber).toMatch(/^ORD-\d{6}$/);
  });
});
```

### Scenario 2: Social Media Platform Testing

**Objective**: Test complex user interactions and real-time features

**Features to Test**:

1. **User Authentication & Profile Management**
2. **Post Creation & Interaction** (like, comment, share)
3. **Real-time Messaging**
4. **File Upload & Media Processing**
5. **Search & Discovery**
6. **Privacy Settings**

**Test Implementation**:

```javascript
// Social media platform E2E tests
describe("Social Media Platform", () => {
  test("User interaction flow", async () => {
    // Create two user accounts
    const user1 = await createTestUser("alice@test.com");
    const user2 = await createTestUser("bob@test.com");

    // User 1 creates a post
    await user1.createPost({
      text: "Hello, testing world!",
      image: "test-image.jpg",
      privacy: "public",
    });

    // User 2 discovers and interacts with post
    await user2.searchPosts("testing world");
    await user2.likePost(postId);
    await user2.commentOnPost(postId, "Great post!");

    // Verify real-time updates
    await user1.expectNotification("New like on your post");
    await user1.expectNotification("New comment on your post");

    // Test messaging
    await user2.sendMessage(user1.id, "Hi Alice!");
    await user1.expectNewMessage("Hi Alice!");
  });
});
```

### Scenario 3: Banking Application Security Testing

**Objective**: Test security features and compliance requirements

**Security Test Cases**:

1. **Authentication Security**
2. **Session Management**
3. **Input Validation**
4. **SQL Injection Prevention**
5. **XSS Protection**
6. **CSRF Protection**

---

## 🤖 Test Automation Challenges

### Challenge 1: Dynamic Test Data Management

**Objective**: Build a robust test data management system

**Requirements**:

1. **Data Factory Pattern Implementation**:

   ```python
   # test_data_factory.py
   import factory
   from faker import Faker

   fake = Faker()

   class UserFactory(factory.Factory):
       class Meta:
           model = User

       email = factory.LazyAttribute(lambda obj: fake.email())
       first_name = factory.LazyAttribute(lambda obj: fake.first_name())
       last_name = factory.LazyAttribute(lambda obj: fake.last_name())
       is_active = True

   class ProductFactory(factory.Factory):
       class Meta:
           model = Product

       name = factory.LazyAttribute(lambda obj: fake.catch_phrase())
       price = factory.LazyAttribute(lambda obj: fake.pydecimal(2, 2, positive=True))
       category = factory.SubFactory(CategoryFactory)
   ```

2. **Environment-Specific Data Management**
3. **Test Data Cleanup Strategies**
4. **Data Privacy Compliance**

### Challenge 2: Cross-Browser Testing Automation

**Objective**: Implement comprehensive cross-browser test suite

**Implementation**:

```javascript
// cross-browser-config.js
const browsers = [
  { name: "chromium", headless: false },
  { name: "firefox", headless: false },
  { name: "webkit", headless: false },
];

const devices = ["Desktop Chrome", "iPad Pro", "iPhone 12", "Galaxy S21"];

// Test runner
browsers.forEach((browser) => {
  devices.forEach((device) => {
    describe(`${browser.name} - ${device}`, () => {
      // Test implementations
    });
  });
});
```

### Challenge 3: Visual Regression Testing

**Objective**: Implement automated visual testing pipeline

**Tools**: Percy, Applitools, or Playwright visual testing

**Implementation**:

```javascript
// visual-regression.spec.js
describe("Visual Regression Tests", () => {
  test("Homepage layout consistency", async () => {
    await page.goto("/");
    await page.screenshot({
      path: "screenshots/homepage.png",
      fullPage: true,
    });

    // Compare with baseline
    await expect(page).toHaveScreenshot("homepage-baseline.png");
  });

  test("Responsive design validation", async () => {
    const viewports = [
      { width: 1920, height: 1080 },
      { width: 1024, height: 768 },
      { width: 375, height: 667 },
    ];

    for (const viewport of viewports) {
      await page.setViewportSize(viewport);
      await expect(page).toHaveScreenshot(
        `homepage-${viewport.width}x${viewport.height}.png`,
      );
    }
  });
});
```

---

## 📊 Performance Testing Exercises

### Exercise 1: Load Testing with K6

**Objective**: Implement comprehensive load testing strategy

**Scenario**: API Load Testing

```javascript
// load-test.js
import http from "k6/http";
import { check, sleep } from "k6";
import { Rate } from "k6/metrics";

const errorRate = new Rate("errors");

export let options = {
  stages: [
    { duration: "2m", target: 100 }, // Ramp up
    { duration: "5m", target: 100 }, // Steady state
    { duration: "2m", target: 200 }, // Spike test
    { duration: "5m", target: 200 }, // Spike steady state
    { duration: "2m", target: 0 }, // Ramp down
  ],
  thresholds: {
    http_req_duration: ["p(99)<500"],
    errors: ["rate<0.1"],
  },
};

export default function () {
  // Authentication
  const loginResponse = http.post("https://api.example.com/auth/login", {
    email: "test@example.com",
    password: "password123",
  });

  check(loginResponse, {
    "login successful": (resp) => resp.status === 200,
    "token received": (resp) => resp.json("token") !== "",
  });

  const token = loginResponse.json("token");

  // API calls with authentication
  const headers = {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };

  // Get user profile
  const profileResponse = http.get("https://api.example.com/user/profile", {
    headers,
  });
  check(profileResponse, {
    "profile loaded": (resp) => resp.status === 200,
  }) || errorRate.add(1);

  // Create post
  const postData = {
    title: `Test Post ${Math.random()}`,
    content: "This is a test post for load testing",
  };

  const postResponse = http.post(
    "https://api.example.com/posts",
    JSON.stringify(postData),
    { headers },
  );
  check(postResponse, {
    "post created": (resp) => resp.status === 201,
  }) || errorRate.add(1);

  sleep(1);
}
```

### Exercise 2: Database Performance Testing

**Objective**: Test database performance under various loads

**Scenarios**:

1. **Concurrent Read/Write Operations**
2. **Large Dataset Queries**
3. **Index Performance Testing**
4. **Connection Pool Optimization**

**Implementation Example**:

```python
# db_performance_test.py
import asyncio
import aiopg
import time
from concurrent.futures import ThreadPoolExecutor

async def test_concurrent_operations():
    """Test database under concurrent load"""

    # Connection pool setup
    dsn = 'postgresql://user:pass@localhost/testdb'
    pool = await aiopg.create_pool(dsn, minsize=10, maxsize=100)

    async def read_operation():
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT * FROM users WHERE active = true")
                return await cur.fetchall()

    async def write_operation():
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO user_actions (user_id, action, timestamp) VALUES (%s, %s, %s)",
                    (random.randint(1, 1000), 'test_action', time.time())
                )

    # Concurrent test execution
    start_time = time.time()

    tasks = []
    for i in range(100):
        if i % 3 == 0:
            tasks.append(write_operation())
        else:
            tasks.append(read_operation())

    results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.time()

    # Performance analysis
    successful_operations = sum(1 for r in results if not isinstance(r, Exception))
    total_time = end_time - start_time

    print(f"Completed {successful_operations}/100 operations in {total_time:.2f}s")
    print(f"Operations per second: {successful_operations/total_time:.2f}")
```

---

## 🔒 Security Testing Labs

### Lab 1: OWASP Top 10 Testing

**Objective**: Test for common security vulnerabilities

**Vulnerability Tests**:

1. **SQL Injection Testing**:

   ```python
   # sql_injection_test.py
   def test_sql_injection_prevention():
       """Test SQL injection attack prevention"""

       # Malicious payloads
       payloads = [
           "'; DROP TABLE users; --",
           "' OR '1'='1' --",
           "' UNION SELECT username, password FROM admin_users --",
           "'; INSERT INTO users (username) VALUES ('hacker'); --"
       ]

       for payload in payloads:
           response = requests.post('/api/login', {
               'username': payload,
               'password': 'test'
           })

           # Should not return sensitive data or error messages
           assert response.status_code in [400, 401, 403]
           assert 'SQL' not in response.text
           assert 'syntax error' not in response.text.lower()
   ```

2. **XSS Testing**:

   ```javascript
   // xss_prevention_test.spec.js
   describe("XSS Prevention", () => {
     const xssPayloads = [
       '<script>alert("XSS")</script>',
       '<img src="x" onerror="alert(\'XSS\')">',
       "<svg onload=\"alert('XSS')\">",
       'javascript:alert("XSS")',
       "<iframe src=\"javascript:alert('XSS')\"></iframe>",
     ];

     xssPayloads.forEach((payload) => {
       test(`Should sanitize payload: ${payload}`, async () => {
         await page.fill("#comment-input", payload);
         await page.click("#submit-comment");

         // Check that script is not executed
         await page.waitForTimeout(1000);
         const alerts = await page.evaluate(() => window.alertCalled);
         expect(alerts).toBeFalsy();

         // Check that content is properly escaped
         const commentText = await page.textContent(".comment:last-child");
         expect(commentText).toBe(payload); // Should be displayed as text, not HTML
       });
     });
   });
   ```

3. **Authentication Security Testing**:

   ```python
   # auth_security_test.py
   def test_password_policy():
       """Test password strength requirements"""
       weak_passwords = [
           'password',
           '123456',
           'qwerty',
           'abc123',
           'password123'
       ]

       for weak_password in weak_passwords:
           response = client.post('/api/register', {
               'email': 'test@example.com',
               'password': weak_password
           })

           assert response.status_code == 400
           assert 'password' in response.json()['errors']

   def test_brute_force_protection():
       """Test account lockout mechanism"""

       # Attempt multiple failed logins
       for attempt in range(6):
           response = client.post('/api/login', {
               'email': 'test@example.com',
               'password': 'wrongpassword'
           })

       # Account should be locked after 5 attempts
       assert response.status_code == 429
       assert 'locked' in response.json()['message'].lower()
   ```

### Lab 2: API Security Testing

**Objective**: Comprehensive API security validation

**Test Areas**:

1. **Rate Limiting**
2. **Input Validation**
3. **Authorization Bypass**
4. **Data Exposure**
5. **CORS Configuration**

---

## 📱 Mobile Testing Projects

### Project 1: React Native App Testing

**Objective**: Test mobile app across platforms using Detox

**Setup**:

```bash
npm install --save-dev detox
detox init
```

**Test Implementation**:

```javascript
// e2e/mobile-app.e2e.js
describe("Mobile App E2E", () => {
  beforeEach(async () => {
    await device.reloadReactNative();
  });

  it("should complete user registration flow", async () => {
    // Welcome screen
    await expect(element(by.id("welcome-screen"))).toBeVisible();
    await element(by.id("get-started-button")).tap();

    // Registration form
    await element(by.id("email-input")).typeText("test@example.com");
    await element(by.id("password-input")).typeText("SecurePass123");
    await element(by.id("confirm-password-input")).typeText("SecurePass123");

    // Submit registration
    await element(by.id("register-button")).tap();

    // Verify success
    await expect(element(by.id("dashboard-screen"))).toBeVisible();
  });

  it("should handle offline scenarios", async () => {
    await device.setNetworkConnection(false);

    // App should show offline indicator
    await expect(element(by.id("offline-banner"))).toBeVisible();

    // Local data should still be accessible
    await element(by.id("cached-data-tab")).tap();
    await expect(element(by.id("data-list"))).toBeVisible();

    await device.setNetworkConnection(true);
    await expect(element(by.id("offline-banner"))).not.toBeVisible();
  });
});
```

### Project 2: iOS/Android Native Testing

**Objective**: Platform-specific testing with Appium

**Test Scenarios**:

1. **Platform-Specific UI Elements**
2. **Device Capabilities** (camera, GPS, notifications)
3. **Performance on Different Devices**
4. **Accessibility Features**

---

## ♿ Accessibility Testing Challenges

### Challenge 1: Automated A11y Testing

**Objective**: Implement comprehensive accessibility testing

**Tools**: axe-playwright, Pa11y, WAVE

**Implementation**:

```javascript
// accessibility.spec.js
const { injectAxe, checkA11y } = require("axe-playwright");

describe("Accessibility Tests", () => {
  beforeEach(async () => {
    await page.goto("/");
    await injectAxe(page);
  });

  test("Should pass WCAG AA standards", async () => {
    await checkA11y(page, null, {
      detailedReport: true,
      detailedReportOptions: { html: true },
    });
  });

  test("Keyboard navigation", async () => {
    // Test tab navigation
    await page.keyboard.press("Tab");
    let focusedElement = await page.evaluate(
      () => document.activeElement.tagName,
    );
    expect(focusedElement).toBe("BUTTON");

    // Test skip links
    await page.keyboard.press("Tab");
    await page.keyboard.press("Enter");

    focusedElement = await page.evaluate(() => document.activeElement.id);
    expect(focusedElement).toBe("main-content");
  });

  test("Screen reader compatibility", async () => {
    // Check for proper ARIA labels
    const buttons = await page.$$("button");
    for (const button of buttons) {
      const ariaLabel = await button.getAttribute("aria-label");
      const textContent = await button.textContent();

      expect(ariaLabel || textContent).toBeTruthy();
    }

    // Check heading hierarchy
    const headings = await page.$$eval("h1, h2, h3, h4, h5, h6", (elements) =>
      elements.map((el) => ({
        tag: el.tagName,
        level: parseInt(el.tagName[1]),
      })),
    );

    // Verify proper heading sequence
    for (let i = 1; i < headings.length; i++) {
      const prevLevel = headings[i - 1].level;
      const currentLevel = headings[i].level;
      expect(currentLevel - prevLevel).toBeLessThanOrEqual(1);
    }
  });
});
```

### Challenge 2: Manual Accessibility Testing

**Objective**: Create systematic manual testing procedures

**Testing Checklist**:

1. **Keyboard Navigation Testing**
2. **Screen Reader Testing**
3. **Color Contrast Validation**
4. **Focus Management**
5. **Alternative Text Verification**

---

## 📈 Test Strategy & Planning

### Exercise 1: Test Plan Creation

**Objective**: Develop comprehensive test strategy documents

**Components**:

1. **Test Strategy Document**
2. **Risk Assessment Matrix**
3. **Test Coverage Analysis**
4. **Resource Planning**
5. **Timeline & Milestones**

**Template**:

```markdown
# Test Plan Template

## 1. Test Scope

- Features to be tested
- Features not to be tested
- Test approach (manual/automated)

## 2. Test Objectives

- Quality gates
- Acceptance criteria
- Performance benchmarks

## 3. Test Environment

- Hardware requirements
- Software requirements
- Test data requirements

## 4. Test Schedule

- Test phases
- Milestones
- Dependencies

## 5. Risk Assessment

| Risk                 | Probability | Impact | Mitigation Strategy             |
| -------------------- | ----------- | ------ | ------------------------------- |
| API changes          | High        | High   | Mock services, contract testing |
| Environment issues   | Medium      | High   | Infrastructure as Code          |
| Test data corruption | Low         | High   | Automated data seeding          |
```

### Exercise 2: Defect Management Process

**Objective**: Establish efficient bug tracking and resolution workflow

**Workflow Implementation**:

1. **Bug Report Templates**
2. **Severity & Priority Classification**
3. **Triage Process**
4. **Resolution Tracking**
5. **Post-mortem Analysis**

---

## 🛠️ CI/CD Testing Integration

### Project 1: GitHub Actions Test Pipeline

**Objective**: Create comprehensive CI/CD testing workflow

```yaml
# .github/workflows/test-pipeline.yml
name: Test Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16, 18, 20]

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node-version }}
          cache: "npm"

      - name: Install dependencies
        run: npm ci

      - name: Run unit tests
        run: npm run test:unit -- --coverage

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests

    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: "18"
          cache: "npm"

      - name: Install dependencies
        run: npm ci

      - name: Run integration tests
        run: npm run test:integration
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/testdb

  e2e-tests:
    runs-on: ubuntu-latest
    needs: integration-tests

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: "18"
          cache: "npm"

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright browsers
        run: npx playwright install

      - name: Start application
        run: |
          npm run build
          npm run start &
          sleep 10

      - name: Run E2E tests
        run: npm run test:e2e

      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: failure()
        with:
          name: playwright-report
          path: playwright-report/

  performance-tests:
    runs-on: ubuntu-latest
    needs: e2e-tests

    steps:
      - uses: actions/checkout@v3

      - name: Run K6 performance tests
        uses: k6io/action@v0.1
        with:
          filename: performance-tests/load-test.js
          flags: --out json=results.json

      - name: Upload performance results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: results.json

  security-tests:
    runs-on: ubuntu-latest
    needs: unit-tests

    steps:
      - uses: actions/checkout@v3

      - name: Run OWASP ZAP security scan
        uses: zaproxy/action-baseline@v0.7.0
        with:
          target: "https://staging.yourapp.com"
          rules_file_name: ".zap/rules.tsv"
          cmd_options: "-a"
```

### Project 2: Test Reporting Dashboard

**Objective**: Create automated test result visualization

**Components**:

1. **Test Result Aggregation**
2. **Trend Analysis**
3. **Failure Analysis**
4. **Performance Metrics**
5. **Quality Gates Monitoring**

---

## 🎯 Advanced Testing Scenarios

### Scenario 1: Chaos Engineering Testing

**Objective**: Test system resilience under failure conditions

**Implementation**:

```python
# chaos_testing.py
import random
import time
import requests
from threading import Thread

class ChaosMonkey:
    def __init__(self, base_url):
        self.base_url = base_url
        self.running = True

    def kill_random_service(self):
        """Simulate random service failures"""
        services = ['user-service', 'product-service', 'order-service']
        service = random.choice(services)

        # Simulate service shutdown
        requests.post(f'{self.base_url}/admin/shutdown/{service}')
        print(f"Killed {service}")

        # Wait random time before recovery
        time.sleep(random.randint(30, 120))

        # Restart service
        requests.post(f'{self.base_url}/admin/start/{service}')
        print(f"Restarted {service}")

    def network_partition(self):
        """Simulate network partitions"""
        # Implementation for network chaos
        pass

    def resource_exhaustion(self):
        """Simulate resource constraints"""
        # Implementation for resource chaos
        pass

def test_system_resilience():
    """Test system behavior under chaos conditions"""
    chaos = ChaosMonkey('http://localhost:8080')

    # Start chaos monkey
    chaos_thread = Thread(target=chaos.kill_random_service)
    chaos_thread.daemon = True
    chaos_thread.start()

    # Run normal user flows during chaos
    for i in range(100):
        try:
            response = requests.post('/api/orders', {
                'user_id': f'user-{i}',
                'products': [{'id': 'product-1', 'quantity': 1}]
            })

            if response.status_code != 200:
                print(f"Order {i} failed: {response.status_code}")
            else:
                print(f"Order {i} succeeded despite chaos")

        except Exception as e:
            print(f"Order {i} exception: {e}")

        time.sleep(1)
```

### Scenario 2: Multi-Environment Testing

**Objective**: Validate application across different environments

**Environment Configurations**:

1. **Development Environment**
2. **Staging Environment**
3. **Production-like Environment**
4. **Performance Environment**

---

## 📝 Assessment Projects

### Final Project 1: Complete Testing Framework

**Objective**: Build a comprehensive testing framework for a full-stack application

**Requirements**:

1. **Multi-layer testing** (Unit, Integration, E2E)
2. **Cross-browser support**
3. **Mobile testing**
4. **Performance testing**
5. **Security testing**
6. **Accessibility testing**
7. **CI/CD integration**
8. **Test reporting**

**Deliverables**:

- Complete test suite (minimum 100 tests)
- Test strategy document
- CI/CD pipeline configuration
- Performance benchmarks
- Security test report
- Accessibility audit

### Final Project 2: Testing Automation Tool

**Objective**: Create a custom testing tool or framework

**Tool Options**:

1. **Test Data Generator**
2. **Visual Regression Testing Tool**
3. **API Testing Framework**
4. **Mobile App Testing Utility**
5. **Performance Monitoring Dashboard**

**Requirements**:

- Clean, maintainable code
- Comprehensive documentation
- Unit tests for the tool itself
- Example usage scenarios
- Performance considerations

---

## 🏆 Certification Preparation

### Practice Exam 1: ISTQB Foundation Level

**Topics Covered**:

- Testing fundamentals
- Test lifecycle
- Static testing
- Test design techniques
- Test management
- Tool support

### Practice Exam 2: Selenium WebDriver

**Practical Assessment**:

- Framework design
- Page Object Model implementation
- Data-driven testing
- Cross-browser testing
- Reporting integration

### Practice Exam 3: Performance Testing

**K6 Certification Preparation**:

- Load testing scenarios
- Performance scripting
- Results analysis
- Bottleneck identification

---

## 📚 Additional Resources

### Books & References

- "The Art of Software Testing" by Glenford Myers
- "Agile Testing" by Lisa Crispin
- "Google's Software Testing Framework"
- "Continuous Testing in DevOps"

### Online Communities

- Ministry of Testing
- Software Testing Help
- Test Automation University
- QA Stack Exchange

### Tools & Frameworks Reference

- **Unit Testing**: Jest, JUnit, PyTest, Mocha
- **Integration Testing**: Supertest, Spring Boot Test, Requests
- **E2E Testing**: Playwright, Cypress, Selenium
- **Mobile Testing**: Detox, Appium, Espresso
- **Performance**: K6, JMeter, Gatling
- **API Testing**: Postman, REST Assured, Insomnia

---

_Total Exercise Count: 45+ hands-on exercises and projects_
_Estimated Completion Time: 8-12 weeks (with dedicated practice)_
_Skill Level: Intermediate to Advanced_
