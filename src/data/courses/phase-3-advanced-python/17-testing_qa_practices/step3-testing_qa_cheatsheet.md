# Testing & QA Practices - Quick Reference Cheatsheet

## 🧪 Testing Fundamentals

### Test Pyramid

```
                 /\
                /  \
               /    \
              / UI   \
             /  E2E   \
            /__________\
           /            \
          /              \
         / Integration     \
        /    Tests         \
       /___________________\
      /                     \
     /                       \
    /        Unit Tests       \
   /_________________________\

Unit Tests: Fast, isolated, many
Integration Tests: Medium speed, component interaction
E2E Tests: Slow, full user journey, few
```

### Testing Types

```
Functional Testing:
✓ Unit Testing
✓ Integration Testing
✓ End-to-End Testing
✓ Acceptance Testing
✓ Smoke Testing
✓ Regression Testing

Non-Functional Testing:
✓ Performance Testing
✓ Security Testing
✓ Usability Testing
✓ Accessibility Testing
✓ Compatibility Testing
```

## 🔬 Unit Testing

### JavaScript/Jest

```javascript
// Basic test structure
describe("Calculator", () => {
  let calculator;

  beforeEach(() => {
    calculator = new Calculator();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe("add method", () => {
    it("should add two positive numbers", () => {
      const result = calculator.add(2, 3);
      expect(result).toBe(5);
    });

    it("should handle negative numbers", () => {
      const result = calculator.add(-1, 1);
      expect(result).toBe(0);
    });

    it("should throw error for non-numbers", () => {
      expect(() => calculator.add("a", 1)).toThrow("Invalid input");
    });
  });
});

// Mocking
const mockUserService = {
  getUser: jest.fn(),
  createUser: jest.fn(),
};

// Spy on methods
jest.spyOn(userService, "getUser").mockResolvedValue({ id: 1, name: "John" });

// Test async functions
test("should fetch user data", async () => {
  const mockUser = { id: 1, name: "John" };
  mockUserService.getUser.mockResolvedValue(mockUser);

  const result = await userController.getUser(1);

  expect(mockUserService.getUser).toHaveBeenCalledWith(1);
  expect(result).toEqual(mockUser);
});
```

### Python/Pytest

```python
# conftest.py
import pytest
from myapp import create_app, db
from myapp.models import User

@pytest.fixture
def app():
    app = create_app('testing')
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def sample_user():
    return User(name='John Doe', email='john@example.com')

# test_user.py
def test_create_user(client, app):
    """Test user creation endpoint"""
    user_data = {
        'name': 'John Doe',
        'email': 'john@example.com'
    }

    response = client.post('/api/users', json=user_data)

    assert response.status_code == 201
    assert response.json['name'] == user_data['name']

# Parametrized tests
@pytest.mark.parametrize("email,expected", [
    ("test@example.com", True),
    ("invalid-email", False),
    ("", False),
    ("test@", False)
])
def test_email_validation(email, expected):
    result = validate_email(email)
    assert result == expected

# Mocking with unittest.mock
from unittest.mock import patch, Mock

@patch('myapp.services.external_api')
def test_external_api_call(mock_api):
    mock_api.get_data.return_value = {'status': 'success'}

    result = my_service.process_data()

    mock_api.get_data.assert_called_once()
    assert result['status'] == 'success'
```

### Java/JUnit 5

```java
// Basic test structure
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class UserServiceTest {

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private UserService userService;

    @BeforeAll
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    @DisplayName("Should create user with valid data")
    void shouldCreateUserWithValidData() {
        // Given
        UserDto userDto = new UserDto("John", "john@example.com");
        User savedUser = new User(1L, "John", "john@example.com");

        when(userRepository.save(any(User.class))).thenReturn(savedUser);

        // When
        User result = userService.createUser(userDto);

        // Then
        assertThat(result.getId()).isEqualTo(1L);
        assertThat(result.getName()).isEqualTo("John");
        verify(userRepository).save(any(User.class));
    }

    @ParameterizedTest
    @ValueSource(strings = {"", " ", "ab", "a".repeat(51)})
    @DisplayName("Should throw exception for invalid names")
    void shouldThrowExceptionForInvalidNames(String invalidName) {
        UserDto userDto = new UserDto(invalidName, "john@example.com");

        assertThrows(ValidationException.class, () -> {
            userService.createUser(userDto);
        });
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.SECONDS)
    void shouldCompleteWithinTimeout() {
        // Test that completes within 5 seconds
    }
}
```

## 🔗 Integration Testing

### API Testing with Supertest (Node.js)

```javascript
const request = require("supertest");
const app = require("../app");

describe("User API", () => {
  beforeAll(async () => {
    await setupTestDatabase();
  });

  afterAll(async () => {
    await cleanupTestDatabase();
  });

  describe("POST /api/users", () => {
    it("should create a new user", async () => {
      const userData = {
        name: "John Doe",
        email: "john@example.com",
        password: "SecurePass123",
      };

      const response = await request(app)
        .post("/api/users")
        .send(userData)
        .expect(201)
        .expect("Content-Type", /json/);

      expect(response.body).toHaveProperty("id");
      expect(response.body.name).toBe(userData.name);
      expect(response.body).not.toHaveProperty("password");
    });

    it("should return 400 for invalid email", async () => {
      const userData = {
        name: "John Doe",
        email: "invalid-email",
        password: "SecurePass123",
      };

      await request(app).post("/api/users").send(userData).expect(400);
    });
  });

  describe("Authentication flow", () => {
    it("should complete registration and login flow", async () => {
      // Register user
      const userData = {
        name: "Jane Doe",
        email: "jane@example.com",
        password: "SecurePass123",
      };

      await request(app).post("/api/auth/register").send(userData).expect(201);

      // Login user
      const loginResponse = await request(app)
        .post("/api/auth/login")
        .send({
          email: userData.email,
          password: userData.password,
        })
        .expect(200);

      const token = loginResponse.body.token;
      expect(token).toBeDefined();

      // Access protected route
      await request(app)
        .get("/api/profile")
        .set("Authorization", `Bearer ${token}`)
        .expect(200);
    });
  });
});
```

### Database Integration Testing

```python
# test_database_integration.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from myapp.models import User, Post, Base

class TestDatabaseIntegration:

    @pytest.fixture(scope='class')
    def db_session(self):
        engine = create_engine('postgresql://test:test@localhost/test_db')
        Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        session = Session()

        yield session

        session.close()
        Base.metadata.drop_all(engine)

    def test_user_post_relationship(self, db_session):
        # Create user
        user = User(name='John', email='john@example.com')
        db_session.add(user)
        db_session.commit()

        # Create posts
        post1 = Post(title='Post 1', content='Content 1', user_id=user.id)
        post2 = Post(title='Post 2', content='Content 2', user_id=user.id)

        db_session.add_all([post1, post2])
        db_session.commit()

        # Test relationship
        assert len(user.posts) == 2
        assert post1.user.name == 'John'

    def test_cascade_delete(self, db_session):
        user = User(name='Jane', email='jane@example.com')
        db_session.add(user)
        db_session.commit()

        post = Post(title='Test Post', content='Content', user_id=user.id)
        db_session.add(post)
        db_session.commit()

        # Delete user should cascade to posts
        db_session.delete(user)
        db_session.commit()

        remaining_posts = db_session.query(Post).filter_by(user_id=user.id).all()
        assert len(remaining_posts) == 0
```

## 🌐 End-to-End Testing

### Playwright

```javascript
// playwright.config.js
module.exports = {
  testDir: "./tests",
  timeout: 30000,
  expect: {
    timeout: 5000,
  },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: "html",
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
    },
    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },
  ],
  webServer: {
    command: "npm run start",
    port: 3000,
  },
};

// tests/user-journey.spec.js
const { test, expect } = require("@playwright/test");

test.describe("User Registration and Login", () => {
  test("complete user journey", async ({ page }) => {
    // Navigate to registration
    await page.goto("/");
    await page.click('[data-testid="register-link"]');

    // Fill registration form
    await page.fill('[data-testid="name-input"]', "John Doe");
    await page.fill('[data-testid="email-input"]', "john@example.com");
    await page.fill('[data-testid="password-input"]', "SecurePass123");

    // Submit form
    await page.click('[data-testid="register-button"]');

    // Verify success message
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();

    // Login with new account
    await page.click('[data-testid="login-link"]');
    await page.fill('[data-testid="email-input"]', "john@example.com");
    await page.fill('[data-testid="password-input"]', "SecurePass123");
    await page.click('[data-testid="login-button"]');

    // Verify dashboard
    await expect(page).toHaveURL("/dashboard");
    await expect(page.locator("h1")).toContainText("Welcome, John");
  });

  test("form validation", async ({ page }) => {
    await page.goto("/register");

    // Try to submit empty form
    await page.click('[data-testid="register-button"]');

    // Check validation errors
    await expect(page.locator('[data-testid="name-error"]')).toContainText(
      "Name is required",
    );
    await expect(page.locator('[data-testid="email-error"]')).toContainText(
      "Email is required",
    );

    // Test invalid email
    await page.fill('[data-testid="email-input"]', "invalid-email");
    await page.click('[data-testid="register-button"]');
    await expect(page.locator('[data-testid="email-error"]')).toContainText(
      "Invalid email format",
    );
  });
});

// tests/shopping-cart.spec.js
test.describe("Shopping Cart", () => {
  test.beforeEach(async ({ page }) => {
    // Login as existing user
    await page.goto("/login");
    await page.fill('[data-testid="email-input"]', "test@example.com");
    await page.fill('[data-testid="password-input"]', "password123");
    await page.click('[data-testid="login-button"]');
  });

  test("add and remove items from cart", async ({ page }) => {
    // Browse products
    await page.goto("/products");

    // Add first product to cart
    await page.click('[data-testid="product-1"] [data-testid="add-to-cart"]');
    await expect(page.locator('[data-testid="cart-count"]')).toContainText("1");

    // Add second product
    await page.click('[data-testid="product-2"] [data-testid="add-to-cart"]');
    await expect(page.locator('[data-testid="cart-count"]')).toContainText("2");

    // View cart
    await page.click('[data-testid="cart-icon"]');

    // Verify items in cart
    await expect(page.locator('[data-testid="cart-item-1"]')).toBeVisible();
    await expect(page.locator('[data-testid="cart-item-2"]')).toBeVisible();

    // Remove item
    await page.click(
      '[data-testid="cart-item-1"] [data-testid="remove-button"]',
    );
    await expect(page.locator('[data-testid="cart-item-1"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="cart-count"]')).toContainText("1");
  });
});
```

### Cypress

```javascript
// cypress.config.js
const { defineConfig } = require("cypress");

module.exports = defineConfig({
  e2e: {
    baseUrl: "http://localhost:3000",
    supportFile: "cypress/support/e2e.js",
    specPattern: "cypress/e2e/**/*.cy.{js,jsx,ts,tsx}",
  },
  component: {
    devServer: {
      framework: "react",
      bundler: "webpack",
    },
  },
});

// cypress/e2e/auth.cy.js
describe("Authentication", () => {
  beforeEach(() => {
    cy.visit("/");
  });

  it("should register and login user", () => {
    // Registration
    cy.get('[data-cy="register-link"]').click();
    cy.get('[data-cy="name-input"]').type("John Doe");
    cy.get('[data-cy="email-input"]').type("john@example.com");
    cy.get('[data-cy="password-input"]').type("SecurePass123");
    cy.get('[data-cy="register-button"]').click();

    cy.get('[data-cy="success-message"]').should("be.visible");

    // Login
    cy.get('[data-cy="login-link"]').click();
    cy.get('[data-cy="email-input"]').type("john@example.com");
    cy.get('[data-cy="password-input"]').type("SecurePass123");
    cy.get('[data-cy="login-button"]').click();

    cy.url().should("include", "/dashboard");
    cy.get("h1").should("contain", "Welcome");
  });

  it("should handle login errors", () => {
    cy.get('[data-cy="login-link"]').click();
    cy.get('[data-cy="email-input"]').type("wrong@example.com");
    cy.get('[data-cy="password-input"]').type("wrongpassword");
    cy.get('[data-cy="login-button"]').click();

    cy.get('[data-cy="error-message"]').should(
      "contain",
      "Invalid credentials",
    );
  });
});

// Custom commands (cypress/support/commands.js)
Cypress.Commands.add("login", (email, password) => {
  cy.request({
    method: "POST",
    url: "/api/auth/login",
    body: {
      email: email,
      password: password,
    },
  }).then((response) => {
    window.localStorage.setItem("authToken", response.body.token);
  });
});

Cypress.Commands.add("createProduct", (productData) => {
  cy.request({
    method: "POST",
    url: "/api/products",
    headers: {
      Authorization: `Bearer ${window.localStorage.getItem("authToken")}`,
    },
    body: productData,
  });
});

// Usage in tests
it("should display user products", () => {
  cy.login("test@example.com", "password123");
  cy.createProduct({ name: "Test Product", price: 100 });

  cy.visit("/products");
  cy.get('[data-cy="product-name"]').should("contain", "Test Product");
});
```

## 📊 Performance Testing

### K6 Load Testing

```javascript
// load-test.js
import http from "k6/http";
import { check, sleep } from "k6";
import { Rate } from "k6/metrics";

const errorRate = new Rate("errors");

export const options = {
  stages: [
    { duration: "2m", target: 100 }, // Ramp up to 100 users
    { duration: "5m", target: 100 }, // Stay at 100 users
    { duration: "2m", target: 200 }, // Ramp up to 200 users
    { duration: "5m", target: 200 }, // Stay at 200 users
    { duration: "2m", target: 0 }, // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ["p(99)<500"], // 99% of requests must complete below 500ms
    errors: ["rate<0.1"], // Error rate must be below 10%
  },
};

export function setup() {
  // Setup data for the test
  const authResponse = http.post("https://api.example.com/auth/login", {
    email: "test@example.com",
    password: "password123",
  });

  return { token: authResponse.json("token") };
}

export default function (data) {
  const params = {
    headers: {
      Authorization: `Bearer ${data.token}`,
      "Content-Type": "application/json",
    },
  };

  // Test user profile endpoint
  const profileResponse = http.get(
    "https://api.example.com/user/profile",
    params,
  );
  check(profileResponse, {
    "profile status is 200": (r) => r.status === 200,
    "profile has user data": (r) => r.json("name") !== "",
  }) || errorRate.add(1);

  // Test creating a post
  const postData = JSON.stringify({
    title: `Test Post ${Math.random()}`,
    content: "This is a test post content",
  });

  const postResponse = http.post(
    "https://api.example.com/posts",
    postData,
    params,
  );
  check(postResponse, {
    "post created": (r) => r.status === 201,
    "post has id": (r) => r.json("id") !== null,
  }) || errorRate.add(1);

  sleep(1); // Wait 1 second between iterations
}

export function teardown(data) {
  // Cleanup after test
  console.log("Test completed");
}
```

### JMeter Test Plan (JMX)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<jmeterTestPlan version="1.2">
  <hashTree>
    <TestPlan>
      <elementProp name="TestPlan.arguments" elementType="Arguments" guiclass="ArgumentsPanel">
        <collectionProp name="Arguments.arguments">
          <elementProp name="baseUrl" elementType="Argument">
            <stringProp name="Argument.name">baseUrl</stringProp>
            <stringProp name="Argument.value">https://api.example.com</stringProp>
          </elementProp>
        </collectionProp>
      </elementProp>
    </TestPlan>

    <hashTree>
      <ThreadGroup>
        <stringProp name="ThreadGroup.num_threads">100</stringProp>
        <stringProp name="ThreadGroup.ramp_time">300</stringProp>
        <stringProp name="ThreadGroup.duration">600</stringProp>
        <boolProp name="ThreadGroup.scheduler">true</boolProp>

        <hashTree>
          <!-- HTTP Request Sampler -->
          <HTTPSamplerProxy>
            <stringProp name="HTTPSampler.domain">${__P(baseUrl)}</stringProp>
            <stringProp name="HTTPSampler.path">/api/users</stringProp>
            <stringProp name="HTTPSampler.method">GET</stringProp>
          </HTTPSamplerProxy>

          <!-- Response Assertion -->
          <ResponseAssertion>
            <stringProp name="Assertion.test_field">Assertion.response_code</stringProp>
            <stringProp name="Assertion.test_type">2</stringProp>
            <stringProp name="Assertion.test_string">200</stringProp>
          </ResponseAssertion>

          <!-- Throughput Timer -->
          <ConstantThroughputTimer>
            <stringProp name="throughput">60</stringProp>
          </ConstantThroughputTimer>
        </hashTree>
      </ThreadGroup>
    </hashTree>
  </hashTree>
</jmeterTestPlan>
```

## 🔒 Security Testing

### OWASP ZAP Automation

```python
# zap_security_test.py
import time
from zapv2 import ZAPv2

class SecurityTest:
    def __init__(self, proxy_url='http://localhost:8080'):
        self.zap = ZAPv2(proxies={'http': proxy_url, 'https': proxy_url})
        self.target = 'https://example.com'

    def spider_scan(self):
        """Spider scan to discover URLs"""
        print(f'Spidering target: {self.target}')
        scan_id = self.zap.spider.scan(self.target)

        while int(self.zap.spider.status(scan_id)) < 100:
            print(f'Spider progress: {self.zap.spider.status(scan_id)}%')
            time.sleep(2)

        print('Spider scan completed')

    def active_scan(self):
        """Active security scan"""
        print(f'Active scanning target: {self.target}')
        scan_id = self.zap.ascan.scan(self.target)

        while int(self.zap.ascan.status(scan_id)) < 100:
            print(f'Active scan progress: {self.zap.ascan.status(scan_id)}%')
            time.sleep(5)

        print('Active scan completed')

    def generate_report(self):
        """Generate security report"""
        alerts = self.zap.core.alerts()

        high_risk = [alert for alert in alerts if alert['risk'] == 'High']
        medium_risk = [alert for alert in alerts if alert['risk'] == 'Medium']

        print(f'High risk vulnerabilities: {len(high_risk)}')
        print(f'Medium risk vulnerabilities: {len(medium_risk)}')

        # Generate HTML report
        html_report = self.zap.core.htmlreport()
        with open('security_report.html', 'w') as f:
            f.write(html_report)

        return len(high_risk) == 0  # Pass if no high-risk vulnerabilities

if __name__ == '__main__':
    security_test = SecurityTest()
    security_test.spider_scan()
    security_test.active_scan()

    if security_test.generate_report():
        print('Security test PASSED')
        exit(0)
    else:
        print('Security test FAILED - High risk vulnerabilities found')
        exit(1)
```

### Manual Security Testing Checklist

```bash
# SQL Injection Testing
curl -X POST "https://api.example.com/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com'\'' OR 1=1 --", "password": "anything"}'

# XSS Testing
curl -X POST "https://api.example.com/comments" \
  -H "Content-Type: application/json" \
  -d '{"comment": "<script>alert(\"XSS\")</script>"}'

# CSRF Testing
curl -X POST "https://api.example.com/transfer" \
  -H "Content-Type: application/json" \
  -H "Origin: https://malicious-site.com" \
  -d '{"to": "attacker", "amount": 1000}'

# Authentication Bypass
curl -X GET "https://api.example.com/admin/users" \
  -H "Authorization: Bearer invalid_token"

# Rate Limiting Test
for i in {1..100}; do
  curl -X POST "https://api.example.com/login" \
    -H "Content-Type: application/json" \
    -d '{"email": "test@example.com", "password": "wrong"}'
done
```

## ♿ Accessibility Testing

### axe-core Integration

```javascript
// accessibility.test.js
const axe = require("axe-core");
const { chromium } = require("playwright");

describe("Accessibility Tests", () => {
  let browser, page;

  beforeAll(async () => {
    browser = await chromium.launch();
    page = await browser.newPage();
  });

  afterAll(async () => {
    await browser.close();
  });

  test("Homepage should pass WCAG AA standards", async () => {
    await page.goto("https://example.com");

    const results = await page.evaluate(() => {
      return axe.run(document, {
        rules: {
          "color-contrast": { enabled: true },
          "keyboard-navigation": { enabled: true },
          "focus-management": { enabled: true },
        },
      });
    });

    expect(results.violations).toHaveLength(0);

    // Log violations if any
    if (results.violations.length > 0) {
      console.log(
        "Accessibility violations:",
        JSON.stringify(results.violations, null, 2),
      );
    }
  });

  test("Form should have proper labels and ARIA attributes", async () => {
    await page.goto("https://example.com/contact");

    // Check for form labels
    const inputs = await page.$$("input");
    for (const input of inputs) {
      const id = await input.getAttribute("id");
      const ariaLabel = await input.getAttribute("aria-label");
      const label = await page.$(`label[for="${id}"]`);

      expect(label || ariaLabel).toBeTruthy();
    }

    // Check for proper heading hierarchy
    const headings = await page.$$eval("h1, h2, h3, h4, h5, h6", (elements) =>
      elements.map((el) => ({
        tag: el.tagName,
        level: parseInt(el.tagName[1]),
      })),
    );

    // Verify heading levels don't skip
    for (let i = 1; i < headings.length; i++) {
      const prevLevel = headings[i - 1].level;
      const currentLevel = headings[i].level;
      expect(currentLevel - prevLevel).toBeLessThanOrEqual(1);
    }
  });
});
```

### Keyboard Navigation Testing

```javascript
// keyboard-navigation.test.js
describe("Keyboard Navigation", () => {
  test("should navigate through form using Tab key", async () => {
    await page.goto("/contact-form");

    // Start from first focusable element
    await page.keyboard.press("Tab");

    let focusedElement = await page.evaluate(() => document.activeElement.id);
    expect(focusedElement).toBe("name-input");

    // Tab to next element
    await page.keyboard.press("Tab");
    focusedElement = await page.evaluate(() => document.activeElement.id);
    expect(focusedElement).toBe("email-input");

    // Tab to submit button
    await page.keyboard.press("Tab");
    await page.keyboard.press("Tab");
    focusedElement = await page.evaluate(() => document.activeElement.id);
    expect(focusedElement).toBe("submit-button");

    // Test skip links
    await page.keyboard.press("Tab");
    await page.keyboard.press("Enter");

    focusedElement = await page.evaluate(() => document.activeElement.id);
    expect(focusedElement).toBe("main-content");
  });

  test("should handle Escape key to close modals", async () => {
    await page.click('[data-testid="open-modal"]');
    await expect(page.locator('[data-testid="modal"]')).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(page.locator('[data-testid="modal"]')).not.toBeVisible();
  });
});
```

## 📱 Mobile Testing

### Responsive Testing with Playwright

```javascript
// mobile-responsive.test.js
const { devices } = require("@playwright/test");

const mobileDevices = [
  devices["iPhone 12"],
  devices["Galaxy S21"],
  devices["iPad Pro"],
];

for (const device of mobileDevices) {
  test.describe(`Mobile tests on ${device.name}`, () => {
    test.use({ ...device });

    test("should display mobile navigation", async ({ page }) => {
      await page.goto("/");

      // Mobile menu should be visible
      await expect(
        page.locator('[data-testid="mobile-menu-button"]'),
      ).toBeVisible();

      // Desktop menu should be hidden
      await expect(
        page.locator('[data-testid="desktop-menu"]'),
      ).not.toBeVisible();

      // Test mobile menu functionality
      await page.click('[data-testid="mobile-menu-button"]');
      await expect(page.locator('[data-testid="mobile-menu"]')).toBeVisible();
    });

    test("should handle touch interactions", async ({ page }) => {
      await page.goto("/gallery");

      // Test swipe gesture on image carousel
      const carousel = page.locator('[data-testid="image-carousel"]');
      const initialImage = await carousel.getAttribute("data-current-image");

      // Perform swipe left
      await carousel.dispatchEvent("touchstart", {
        touches: [{ clientX: 300, clientY: 200 }],
      });
      await carousel.dispatchEvent("touchmove", {
        touches: [{ clientX: 100, clientY: 200 }],
      });
      await carousel.dispatchEvent("touchend", {});

      await page.waitForTimeout(500); // Wait for animation

      const newImage = await carousel.getAttribute("data-current-image");
      expect(newImage).not.toBe(initialImage);
    });
  });
}
```

### App Testing with Detox (React Native)

```javascript
// detox.config.js
module.exports = {
  testRunner: "jest",
  runnerConfig: "e2e/config.json",
  apps: {
    "ios.debug": {
      type: "ios.app",
      binaryPath: "ios/build/MyApp.app",
      build:
        "xcodebuild -workspace ios/MyApp.xcworkspace -scheme MyApp -configuration Debug -sdk iphonesimulator -derivedDataPath ios/build",
    },
    "android.debug": {
      type: "android.apk",
      binaryPath: "android/app/build/outputs/apk/debug/app-debug.apk",
      build:
        "cd android && ./gradlew assembleDebug assembleAndroidTest -DtestBuildType=debug",
    },
  },
  devices: {
    simulator: {
      type: "ios.simulator",
      device: {
        type: "iPhone 12",
      },
    },
    emulator: {
      type: "android.emulator",
      device: {
        avdName: "Pixel_4_API_30",
      },
    },
  },
};

// e2e/app.test.js
describe("App E2E Tests", () => {
  beforeAll(async () => {
    await device.launchApp();
  });

  beforeEach(async () => {
    await device.reloadReactNative();
  });

  it("should complete onboarding flow", async () => {
    await expect(element(by.id("welcome-screen"))).toBeVisible();

    // Swipe through onboarding screens
    await element(by.id("onboarding-carousel")).swipe("left");
    await expect(element(by.id("screen-2"))).toBeVisible();

    await element(by.id("onboarding-carousel")).swipe("left");
    await expect(element(by.id("screen-3"))).toBeVisible();

    // Complete onboarding
    await element(by.id("get-started-button")).tap();
    await expect(element(by.id("login-screen"))).toBeVisible();
  });

  it("should handle offline state", async () => {
    await device.setNetworkConnection(false);

    await element(by.id("refresh-button")).tap();
    await expect(element(by.id("offline-message"))).toBeVisible();

    await device.setNetworkConnection(true);
    await element(by.id("refresh-button")).tap();
    await expect(element(by.id("offline-message"))).not.toBeVisible();
  });
});
```

## 🎯 Test Strategy & Planning

### Test Plan Template

```markdown
# Test Plan for [Project Name]

## 1. Test Scope

### Features to be Tested:

- User authentication
- Product catalog
- Shopping cart
- Payment processing
- Order management

### Features NOT to be Tested:

- Third-party payment gateway internals
- Email delivery service
- External API dependencies (will be mocked)

## 2. Test Approach

### Testing Types:

- **Unit Tests**: 80% coverage target
- **Integration Tests**: API endpoints and database operations
- **E2E Tests**: Critical user journeys
- **Performance Tests**: Load and stress testing
- **Security Tests**: OWASP Top 10 vulnerabilities

### Test Environments:

- **Development**: Local testing
- **Staging**: Full integration testing
- **Production**: Smoke tests only

## 3. Entry & Exit Criteria

### Entry Criteria:

- Code complete and deployed to test environment
- Test data available
- Test environment stable

### Exit Criteria:

- All critical and high priority defects resolved
- 90% test case execution
- Performance benchmarks met

## 4. Risk Assessment

| Risk                         | Probability | Impact | Mitigation                       |
| ---------------------------- | ----------- | ------ | -------------------------------- |
| API changes                  | High        | High   | Contract testing, API versioning |
| Test environment instability | Medium      | High   | Infrastructure monitoring        |
| Test data inconsistency      | Medium      | Medium | Automated data seeding           |

## 5. Test Schedule

- Week 1: Unit and integration tests
- Week 2: E2E test automation
- Week 3: Performance and security testing
- Week 4: User acceptance testing
```

### Defect Report Template

```markdown
# Bug Report #[ID]

**Summary**: Brief description of the issue

**Environment**:

- OS: Windows 10
- Browser: Chrome 91.0
- Version: v2.1.0

**Priority**: High/Medium/Low
**Severity**: Critical/Major/Minor

**Steps to Reproduce**:

1. Navigate to login page
2. Enter invalid email format
3. Click submit button

**Expected Result**:
Validation error message should be displayed

**Actual Result**:
Form submits successfully without validation

**Screenshots**:
[Attach screenshots/videos]

**Additional Information**:

- Console errors: [Any JS errors]
- Network requests: [Relevant API calls]
- User account: test@example.com

**Assigned To**: Developer Name
**Status**: Open/In Progress/Resolved
```

## 📚 Quick Commands & Scripts

### Jest Commands

```bash
# Run all tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run tests with coverage
npm test -- --coverage

# Run specific test file
npm test -- user.test.js

# Run tests matching pattern
npm test -- --testNamePattern="should create user"

# Update snapshots
npm test -- --updateSnapshot

# Run tests in parallel
npm test -- --maxWorkers=4

# Debug tests
npm test -- --detectOpenHandles
```

### Playwright Commands

```bash
# Install browsers
npx playwright install

# Run tests
npx playwright test

# Run specific test file
npx playwright test auth.spec.js

# Run tests in headed mode
npx playwright test --headed

# Run tests in debug mode
npx playwright test --debug

# Generate test report
npx playwright show-report

# Record new test
npx playwright codegen
```

### Cypress Commands

```bash
# Open Cypress Test Runner
npx cypress open

# Run tests in headless mode
npx cypress run

# Run specific spec file
npx cypress run --spec "cypress/e2e/auth.cy.js"

# Run tests in specific browser
npx cypress run --browser chrome

# Record test run
npx cypress run --record --key <record-key>

# Generate test report
npx cypress run --reporter mochawesome
```

---

_Essential testing patterns, tools, and best practices for quality assurance_
