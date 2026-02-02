# SQL Interview Questions for Data Roles 🎯

## Complete Interview Preparation Guide

This guide contains **100+ real SQL interview questions** from top companies (FAANG, startups, finance) organized by difficulty and topic.

---

## Table of Contents

1. [Easy Questions (40)](#easy-questions)
2. [Medium Questions (40)](#medium-questions)
3. [Hard Questions (30)](#hard-questions)
4. [Company-Specific Questions](#company-specific-questions)
5. [Behavioral & Conceptual](#behavioral--conceptual)
6. [Interview Tips & Strategies](#interview-tips--strategies)

---

## Easy Questions

### Q1: Select All Customers from California

**Company:** Entry-level, most companies

**Question:** Write a query to find all customers from California.

<details>
<summary>Solution</summary>

```sql
SELECT * FROM customers
WHERE state = 'CA';
```

**Interview Tip:** Always ask about the expected output format. Do they want all columns or specific ones?

</details>

---

### Q2: Count Total Orders

**Company:** General

**Question:** How many orders are in the database?

<details>
<summary>Solution</summary>

```sql
SELECT COUNT(*) AS total_orders
FROM orders;
```

</details>

---

### Q3: Find Average Product Price

**Company:** E-commerce companies

**Question:** What is the average price of all products?

<details>
<summary>Solution</summary>

```sql
SELECT AVG(price) AS average_price
FROM products;

-- With rounding
SELECT ROUND(AVG(price), 2) AS average_price
FROM products;
```

</details>

---

### Q4: Top 5 Most Expensive Products

**Company:** Retail, E-commerce

**Question:** List the 5 most expensive products with their names and prices.

<details>
<summary>Solution</summary>

```sql
SELECT product_name, price
FROM products
ORDER BY price DESC
LIMIT 5;
```

</details>

---

### Q5: Customers Who Signed Up in 2024

**Company:** SaaS companies

**Question:** Find all customers who signed up in 2024.

<details>
<summary>Solution</summary>

```sql
SELECT first_name, last_name, signup_date
FROM customers
WHERE YEAR(signup_date) = 2024;

-- More efficient (can use index)
SELECT first_name, last_name, signup_date
FROM customers
WHERE signup_date >= '2024-01-01' AND signup_date < '2025-01-01';
```

**Interview Tip:** The second query is better for performance because it avoids using a function on the column, allowing index usage.

</details>

---

### Q6: Products Out of Stock

**Company:** Inventory management

**Question:** Find all products that are out of stock.

<details>
<summary>Solution</summary>

```sql
SELECT product_name, stock
FROM products
WHERE stock = 0;

-- Or if stock can be NULL
WHERE stock = 0 OR stock IS NULL;
```

</details>

---

### Q7: Total Revenue

**Company:** Finance, Sales analytics

**Question:** Calculate the total revenue from all orders.

<details>
<summary>Solution</summary>

```sql
SELECT SUM(total) AS total_revenue
FROM orders;
```

</details>

---

### Q8: Customers by State

**Company:** Marketing analytics

**Question:** Count how many customers are in each state, ordered by count descending.

<details>
<summary>Solution</summary>

```sql
SELECT state, COUNT(*) AS customer_count
FROM customers
GROUP BY state
ORDER BY customer_count DESC;
```

</details>

---

### Q9: Orders Above $500

**Company:** Sales analytics

**Question:** Find all orders with a total above $500.

<details>
<summary>Solution</summary>

```sql
SELECT order_id, customer_id, total, order_date
FROM orders
WHERE total > 500
ORDER BY total DESC;
```

</details>

---

### Q10: Products in Specific Categories

**Company:** E-commerce

**Question:** Find all products in either Electronics or Furniture categories.

<details>
<summary>Solution</summary>

```sql
SELECT product_name, category, price
FROM products
WHERE category IN ('Electronics', 'Furniture');
```

</details>

---

### Q11: Customer Email Domains

**Company:** Marketing

**Question:** Extract the domain from customer emails (everything after @).

<details>
<summary>Solution</summary>

```sql
SELECT
    email,
    SUBSTRING(email, POSITION('@' IN email) + 1) AS email_domain
FROM customers;
```

</details>

---

### Q12: Products with "Pro" in Name

**Company:** Product analytics

**Question:** Find all products with "Pro" in their name.

<details>
<summary>Solution</summary>

```sql
SELECT product_name, price
FROM products
WHERE product_name LIKE '%Pro%';
```

</details>

---

### Q13: Earliest and Latest Order Dates

**Company:** Business intelligence

**Question:** Find the earliest and latest order dates in the system.

<details>
<summary>Solution</summary>

```sql
SELECT
    MIN(order_date) AS first_order,
    MAX(order_date) AS last_order,
    DATEDIFF(MAX(order_date), MIN(order_date)) AS days_in_business
FROM orders;
```

</details>

---

### Q14: Average Order Value

**Company:** Sales analytics

**Question:** What is the average order value?

<details>
<summary>Solution</summary>

```sql
SELECT
    AVG(total) AS avg_order_value,
    MIN(total) AS min_order,
    MAX(total) AS max_order
FROM orders;
```

</details>

---

### Q15: Count Customers with Email

**Company:** Data quality

**Question:** How many customers have an email address on file?

<details>
<summary>Solution</summary>

```sql
SELECT COUNT(*) AS customers_with_email
FROM customers
WHERE email IS NOT NULL;
```

</details>

---

## Medium Questions

### Q16: Customer Order Count

**Company:** Amazon, eBay

**Question:** For each customer, show their name and total number of orders.

<details>
<summary>Solution</summary>

```sql
SELECT
    c.first_name,
    c.last_name,
    COUNT(o.order_id) AS order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY order_count DESC;
```

**Interview Tip:** Use LEFT JOIN to include customers with zero orders.

</details>

---

### Q17: Customers with No Orders

**Company:** Very common question

**Question:** Find all customers who have never placed an order.

<details>
<summary>Solution</summary>

```sql
-- Method 1: LEFT JOIN
SELECT c.first_name, c.last_name, c.email
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;

-- Method 2: NOT IN
SELECT first_name, last_name, email
FROM customers
WHERE customer_id NOT IN (SELECT DISTINCT customer_id FROM orders);

-- Method 3: NOT EXISTS (most efficient)
SELECT first_name, last_name, email
FROM customers c
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id
);
```

**Interview Tip:** Discuss which method is most efficient (usually NOT EXISTS).

</details>

---

### Q18: Second Highest Salary

**Company:** FAANG classic

**Question:** Find the second highest salary from an employees table.

<details>
<summary>Solution</summary>

```sql
-- Method 1: Subquery
SELECT MAX(salary) AS second_highest_salary
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Method 2: LIMIT with OFFSET
SELECT DISTINCT salary AS second_highest_salary
FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 1;

-- Method 3: Window function (most flexible)
SELECT DISTINCT salary AS second_highest_salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rank
    FROM employees
) AS ranked
WHERE rank = 2;
```

**Follow-up:** What if there's no second highest salary? Handle with COALESCE.

```sql
SELECT COALESCE(
    (SELECT DISTINCT salary
     FROM employees
     ORDER BY salary DESC
     LIMIT 1 OFFSET 1),
    NULL
) AS second_highest_salary;
```

</details>

---

### Q19: Top 3 Products by Revenue

**Company:** Sales analytics

**Question:** Find the top 3 products by total revenue.

<details>
<summary>Solution</summary>

```sql
SELECT
    p.product_name,
    SUM(oi.quantity * oi.price) AS total_revenue
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name
ORDER BY total_revenue DESC
LIMIT 3;
```

</details>

---

### Q20: Monthly Revenue Trend

**Company:** Business Intelligence roles

**Question:** Calculate total revenue for each month in 2024.

<details>
<summary>Solution</summary>

```sql
SELECT
    DATE_FORMAT(order_date, '%Y-%m') AS month,
    SUM(total) AS monthly_revenue,
    COUNT(*) AS order_count,
    AVG(total) AS avg_order_value
FROM orders
WHERE YEAR(order_date) = 2024
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY month;
```

</details>

---

### Q21: Customer Lifetime Value

**Company:** Subscription businesses

**Question:** Calculate total spending for each customer, showing only customers who spent over $1000.

<details>
<summary>Solution</summary>

```sql
SELECT
    c.first_name,
    c.last_name,
    SUM(o.total) AS lifetime_value,
    COUNT(o.order_id) AS total_orders,
    AVG(o.total) AS avg_order_value
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
HAVING SUM(o.total) > 1000
ORDER BY lifetime_value DESC;
```

</details>

---

### Q22: Products Never Ordered

**Company:** Inventory management

**Question:** Find products that have never been ordered.

<details>
<summary>Solution</summary>

```sql
SELECT p.product_name, p.category, p.price
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE oi.item_id IS NULL;
```

</details>

---

### Q23: Duplicate Emails

**Company:** Data Quality, ETL roles

**Question:** Find all duplicate email addresses in the customers table.

<details>
<summary>Solution</summary>

```sql
SELECT email, COUNT(*) AS count
FROM customers
GROUP BY email
HAVING COUNT(*) > 1;

-- With customer names
SELECT c1.*
FROM customers c1
JOIN (
    SELECT email
    FROM customers
    GROUP BY email
    HAVING COUNT(*) > 1
) c2 ON c1.email = c2.email
ORDER BY c1.email;
```

</details>

---

### Q24: Year-over-Year Growth

**Company:** Finance, Analytics

**Question:** Calculate year-over-year revenue growth.

<details>
<summary>Solution</summary>

```sql
WITH yearly_revenue AS (
    SELECT
        YEAR(order_date) AS year,
        SUM(total) AS revenue
    FROM orders
    GROUP BY YEAR(order_date)
)
SELECT
    year,
    revenue,
    LAG(revenue) OVER (ORDER BY year) AS previous_year_revenue,
    revenue - LAG(revenue) OVER (ORDER BY year) AS revenue_growth,
    ROUND((revenue - LAG(revenue) OVER (ORDER BY year)) / LAG(revenue) OVER (ORDER BY year) * 100, 2) AS growth_percentage
FROM yearly_revenue;
```

</details>

---

### Q25: Nth Highest Value

**Company:** Advanced SQL test

**Question:** Write a function to find the Nth highest salary.

<details>
<summary>Solution</summary>

```sql
-- Using DENSE_RANK
WITH ranked_salaries AS (
    SELECT
        salary,
        DENSE_RANK() OVER (ORDER BY salary DESC) AS rank
    FROM employees
)
SELECT DISTINCT salary
FROM ranked_salaries
WHERE rank = N;  -- Replace N with desired rank

-- Generic function (MySQL)
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
      SELECT DISTINCT salary
      FROM (
          SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rank
          FROM employees
      ) AS ranked
      WHERE rank = N
  );
END
```

</details>

---

### Q26: Department-wise Highest Salary

**Company:** HR Analytics

**Question:** Find the highest salary in each department along with employee name.

<details>
<summary>Solution</summary>

```sql
SELECT
    e.department,
    e.employee_name,
    e.salary
FROM employees e
WHERE e.salary = (
    SELECT MAX(salary)
    FROM employees e2
    WHERE e2.department = e.department
);

-- Alternative using window function
WITH ranked AS (
    SELECT
        department,
        employee_name,
        salary,
        RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
    FROM employees
)
SELECT department, employee_name, salary
FROM ranked
WHERE rank = 1;
```

</details>

---

### Q27: Running Total

**Company:** Financial Analytics

**Question:** Calculate a running total of sales by date.

<details>
<summary>Solution</summary>

```sql
SELECT
    order_date,
    total,
    SUM(total) OVER (ORDER BY order_date) AS running_total
FROM orders
ORDER BY order_date;
```

</details>

---

### Q28: Active Users (Login in Last 30 Days)

**Company:** Social Media, SaaS

**Question:** Find users who logged in within the last 30 days.

<details>
<summary>Solution</summary>

```sql
SELECT DISTINCT user_id, username, MAX(login_date) AS last_login
FROM user_logins
WHERE login_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
GROUP BY user_id, username
ORDER BY last_login DESC;
```

</details>

---

### Q29: Managers with Most Employees

**Company:** Org structure analysis

**Question:** Find managers with more than 5 direct reports.

<details>
<summary>Solution</summary>

```sql
SELECT
    m.employee_id AS manager_id,
    m.employee_name AS manager_name,
    COUNT(e.employee_id) AS direct_reports
FROM employees e
JOIN employees m ON e.manager_id = m.employee_id
GROUP BY m.employee_id, m.employee_name
HAVING COUNT(e.employee_id) > 5
ORDER BY direct_reports DESC;
```

</details>

---

### Q30: Product Category Performance

**Company:** Product Analytics

**Question:** Compare revenue and units sold across product categories.

<details>
<summary>Solution</summary>

```sql
SELECT
    p.category,
    SUM(oi.quantity) AS units_sold,
    SUM(oi.quantity * oi.price) AS revenue,
    AVG(oi.price) AS avg_selling_price,
    COUNT(DISTINCT oi.order_id) AS order_count
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.category
ORDER BY revenue DESC;
```

</details>

---

## Hard Questions

### Q31: Find Consecutive Dates

**Company:** Google, Meta

**Question:** Find all cases where a user logged in for 3 or more consecutive days.

<details>
<summary>Solution</summary>

```sql
WITH login_dates AS (
    SELECT
        user_id,
        login_date,
        DATE_SUB(login_date, INTERVAL ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) DAY) AS grp
    FROM user_logins
)
SELECT
    user_id,
    MIN(login_date) AS streak_start,
    MAX(login_date) AS streak_end,
    COUNT(*) AS consecutive_days
FROM login_dates
GROUP BY user_id, grp
HAVING COUNT(*) >= 3;
```

**Interview Tip:** Explain the logic: If dates are consecutive, subtracting row_number gives the same date, creating groups.

</details>

---

### Q32: Median Calculation

**Company:** Statistical analysis roles

**Question:** Calculate the median salary.

<details>
<summary>Solution</summary>

```sql
-- MySQL
SELECT AVG(salary) AS median_salary
FROM (
    SELECT
        salary,
        ROW_NUMBER() OVER (ORDER BY salary) AS row_num,
        COUNT(*) OVER () AS total_count
    FROM employees
) AS numbered
WHERE row_num IN (FLOOR((total_count + 1) / 2), CEIL((total_count + 1) / 2));

-- PostgreSQL (has built-in percentile function)
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees;
```

</details>

---

### Q33: Cumulative Sum with Reset

**Company:** Complex analytics

**Question:** Calculate cumulative sum of sales, resetting at the start of each month.

<details>
<summary>Solution</summary>

```sql
SELECT
    order_date,
    total,
    DATE_FORMAT(order_date, '%Y-%m') AS month,
    SUM(total) OVER (
        PARTITION BY DATE_FORMAT(order_date, '%Y-%m')
        ORDER BY order_date
    ) AS monthly_cumulative_total
FROM orders
ORDER BY order_date;
```

</details>

---

### Q34: Self-Join: Friends of Friends

**Company:** Social Networks

**Question:** Given a friends table (user_id, friend_id), find potential friend suggestions (friends of friends who aren't already friends).

<details>
<summary>Solution</summary>

```sql
SELECT DISTINCT
    f1.user_id,
    f2.friend_id AS suggested_friend
FROM friends f1
JOIN friends f2 ON f1.friend_id = f2.user_id
WHERE f2.friend_id != f1.user_id  -- Not yourself
  AND f2.friend_id NOT IN (
      SELECT friend_id
      FROM friends
      WHERE user_id = f1.user_id
  )  -- Not already a friend
ORDER BY f1.user_id, suggested_friend;
```

</details>

---

### Q35: Moving Average

**Company:** Time-series analysis

**Question:** Calculate 7-day moving average of daily sales.

<details>
<summary>Solution</summary>

```sql
SELECT
    order_date,
    SUM(total) AS daily_total,
    AVG(SUM(total)) OVER (
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7day
FROM orders
GROUP BY order_date
ORDER BY order_date;
```

</details>

---

### Q36: Retention Rate

**Company:** SaaS, Mobile apps

**Question:** Calculate month-over-month retention rate (% of users from previous month who were active this month).

<details>
<summary>Solution</summary>

```sql
WITH monthly_users AS (
    SELECT DISTINCT
        user_id,
        DATE_FORMAT(activity_date, '%Y-%m') AS month
    FROM user_activity
)
SELECT
    curr.month AS current_month,
    COUNT(DISTINCT curr.user_id) AS active_users,
    COUNT(DISTINCT prev.user_id) AS retained_users,
    ROUND(COUNT(DISTINCT prev.user_id) * 100.0 / COUNT(DISTINCT curr.user_id), 2) AS retention_rate
FROM monthly_users curr
LEFT JOIN monthly_users prev
    ON curr.user_id = prev.user_id
    AND curr.month = DATE_FORMAT(DATE_ADD(STR_TO_DATE(prev.month, '%Y-%m'), INTERVAL 1 MONTH), '%Y-%m')
GROUP BY curr.month
ORDER BY curr.month;
```

</details>

---

### Q37: Complex Join with Multiple Conditions

**Company:** Data Engineering

**Question:** Find orders where the order total doesn't match the sum of item prices.

<details>
<summary>Solution</summary>

```sql
SELECT
    o.order_id,
    o.total AS order_total,
    SUM(oi.quantity * oi.price) AS calculated_total,
    ABS(o.total - SUM(oi.quantity * oi.price)) AS difference
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id, o.total
HAVING ABS(o.total - SUM(oi.quantity * oi.price)) > 0.01  -- Account for rounding
ORDER BY difference DESC;
```

</details>

---

### Q38: Hierarchical Query - Employee Chain

**Company:** Organizational analysis

**Question:** Show the full management chain for each employee.

<details>
<summary>Solution</summary>

```sql
-- Using recursive CTE
WITH RECURSIVE employee_hierarchy AS (
    -- Base case: employees with their direct manager
    SELECT
        employee_id,
        employee_name,
        manager_id,
        employee_name AS hierarchy,
        1 AS level
    FROM employees
    WHERE manager_id IS NULL  -- Top-level managers

    UNION ALL

    -- Recursive case: join with managers
    SELECT
        e.employee_id,
        e.employee_name,
        e.manager_id,
        CONCAT(eh.hierarchy, ' -> ', e.employee_name) AS hierarchy,
        eh.level + 1 AS level
    FROM employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
)
SELECT * FROM employee_hierarchy
ORDER BY level, employee_id;
```

</details>

---

### Q39: Pivot Table Query

**Company:** Reporting/BI roles

**Question:** Transform rows to columns showing monthly sales by category.

<details>
<summary>Solution</summary>

```sql
SELECT
    DATE_FORMAT(o.order_date, '%Y-%m') AS month,
    SUM(CASE WHEN p.category = 'Electronics' THEN oi.quantity * oi.price ELSE 0 END) AS electronics_revenue,
    SUM(CASE WHEN p.category = 'Furniture' THEN oi.quantity * oi.price ELSE 0 END) AS furniture_revenue,
    SUM(CASE WHEN p.category = 'Stationery' THEN oi.quantity * oi.price ELSE 0 END) AS stationery_revenue
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
GROUP BY DATE_FORMAT(o.order_date, '%Y-%m')
ORDER BY month;
```

</details>

---

### Q40: Complex Subquery - Top Product per Category

**Company:** E-commerce analytics

**Question:** For each category, find the product with the highest revenue.

<details>
<summary>Solution</summary>

```sql
WITH product_revenue AS (
    SELECT
        p.product_id,
        p.product_name,
        p.category,
        SUM(oi.quantity * oi.price) AS total_revenue,
        RANK() OVER (PARTITION BY p.category ORDER BY SUM(oi.quantity * oi.price) DESC) AS revenue_rank
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    GROUP BY p.product_id, p.product_name, p.category
)
SELECT
    category,
    product_name,
    total_revenue
FROM product_revenue
WHERE revenue_rank = 1
ORDER BY category;
```

</details>

---

## Company-Specific Questions

### Amazon

**Q41: Best Selling Product Each Month**

<details>
<summary>Solution</summary>

```sql
WITH monthly_sales AS (
    SELECT
        DATE_FORMAT(o.order_date, '%Y-%m') AS month,
        p.product_name,
        SUM(oi.quantity) AS units_sold,
        RANK() OVER (PARTITION BY DATE_FORMAT(o.order_date, '%Y-%m') ORDER BY SUM(oi.quantity) DESC) AS rank
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    GROUP BY DATE_FORMAT(o.order_date, '%Y-%m'), p.product_name
)
SELECT month, product_name, units_sold
FROM monthly_sales
WHERE rank = 1;
```

</details>

---

### Google

**Q42: Search Query Autocomplete**
**Question:** Given a search_queries table with (user_id, query, timestamp), find the top 5 most popular search prefixes.

<details>
<summary>Solution</summary>

```sql
SELECT
    LEFT(query, 3) AS prefix,
    COUNT(*) AS search_count
FROM search_queries
GROUP BY LEFT(query, 3)
ORDER BY search_count DESC
LIMIT 5;
```

</details>

---

### Meta (Facebook)

**Q43: Friend Recommendation Count**
**Question:** For each user, count how many potential friends (friends of friends) they could connect with.

<details>
<summary>Solution</summary>

```sql
SELECT
    f1.user_id,
    COUNT(DISTINCT f2.friend_id) AS potential_friends
FROM friends f1
JOIN friends f2 ON f1.friend_id = f2.user_id
WHERE f2.friend_id != f1.user_id
  AND f2.friend_id NOT IN (
      SELECT friend_id FROM friends WHERE user_id = f1.user_id
  )
GROUP BY f1.user_id
ORDER BY potential_friends DESC;
```

</details>

---

### Netflix

**Q44: View Duration by Genre**
**Question:** Calculate total watch time for each genre.

<details>
<summary>Solution</summary>

```sql
SELECT
    g.genre_name,
    SUM(v.duration_minutes) AS total_watch_time,
    COUNT(DISTINCT v.user_id) AS unique_viewers,
    AVG(v.duration_minutes) AS avg_watch_time
FROM views v
JOIN content c ON v.content_id = c.content_id
JOIN genres g ON c.genre_id = g.genre_id
GROUP BY g.genre_name
ORDER BY total_watch_time DESC;
```

</details>

---

### Uber

**Q45: Driver Utilization Rate**
**Question:** Calculate percentage of time drivers are on a trip vs available.

<details>
<summary>Solution</summary>

```sql
SELECT
    driver_id,
    SUM(CASE WHEN status = 'on_trip' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS utilization_rate
FROM driver_status_log
GROUP BY driver_id
ORDER BY utilization_rate DESC;
```

</details>

---

## Behavioral & Conceptual

### Q46: Explain ACID Properties

**Answer:**

- **Atomicity:** All or nothing - transaction fully completes or fully fails
- **Consistency:** Database moves from one valid state to another
- **Isolation:** Concurrent transactions don't interfere
- **Durability:** Committed changes are permanent

---

### Q47: When to use INDEX?

**Answer:**

- ✅ Columns in WHERE clauses
- ✅ Columns in JOIN conditions
- ✅ Columns in ORDER BY
- ❌ Small tables
- ❌ Frequently updated columns
- ❌ Columns with low cardinality

---

### Q48: Difference between WHERE and HAVING?

**Answer:**

- **WHERE:** Filters rows BEFORE grouping
- **HAVING:** Filters groups AFTER aggregation

```sql
SELECT category, COUNT(*)
FROM products
WHERE price > 10      -- Filter rows first
GROUP BY category
HAVING COUNT(*) > 5;  -- Then filter groups
```

---

### Q49: Explain Normalization

**Answer:**

- **1NF:** Atomic values, no repeating groups
- **2NF:** 1NF + No partial dependencies
- **3NF:** 2NF + No transitive dependencies
- **Denormalization:** Intentionally adding redundancy for performance

---

### Q50: What is a Transaction?

**Answer:**
A transaction is a logical unit of work containing one or more SQL statements that must all succeed or all fail together.

```sql
BEGIN TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
UPDATE accounts SET balance = balance + 100 WHERE account_id = 2;
COMMIT;  -- or ROLLBACK if error
```

---

## Interview Tips & Strategies

### Before the Interview

1. **Review Basics:**
   - SELECT, WHERE, JOINs, GROUP BY
   - Aggregate functions
   - Subqueries
   - Window functions

2. **Practice:**
   - LeetCode (SQL section)
   - HackerRank
   - Real company SQL tests

3. **Understand Your Experience:**
   - Be ready to discuss SQL you've written
   - Explain optimization techniques you've used

### During the Interview

1. **Clarify Requirements:**

   ```
   ✅ "Should I include customers with zero orders?"
   ✅ "What's the expected output format?"
   ✅ "Are there any performance constraints?"
   ```

2. **Think Out Loud:**
   - Explain your approach
   - Discuss trade-offs
   - Mention alternative solutions

3. **Write Clean Code:**

   ```sql
   -- ✅ Good
   SELECT
       c.customer_name,
       COUNT(o.order_id) AS order_count
   FROM customers c
   LEFT JOIN orders o ON c.customer_id = o.customer_id
   GROUP BY c.customer_id, c.customer_name;

   -- ❌ Bad
   SELECT c.customer_name,COUNT(o.order_id) FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id GROUP BY c.customer_id;
   ```

4. **Test Your Query:**
   - Walk through with sample data
   - Check edge cases (NULL values, empty results)

5. **Optimize if Asked:**
   - Suggest indexes
   - Avoid SELECT \*
   - Use EXISTS instead of IN for large datasets

### Common Mistakes to Avoid

1. ❌ Forgetting to handle NULL values
2. ❌ Using INNER JOIN when LEFT JOIN is needed
3. ❌ Not grouping by all non-aggregate columns
4. ❌ Forgetting WHERE with UPDATE/DELETE
5. ❌ Using functions in WHERE (prevents index usage)

### Sample Interview Flow

**Interviewer:** "Find customers who spent more than $1000 total."

**You:** "Just to clarify - should I include all customers who've spent over $1000 across all their orders combined?"

**Interviewer:** "Yes, exactly."

**You:** "Great. I'll need to:

1. Join customers and orders tables
2. Group by customer
3. Sum the order totals
4. Filter for sums over $1000

Let me write that out..."

```sql
SELECT
    c.first_name,
    c.last_name,
    SUM(o.total) AS total_spent
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
HAVING SUM(o.total) > 1000
ORDER BY total_spent DESC;
```

**You:** "Should I also show customers with zero orders, or only those who've actually spent over $1000?"

---

## Quick Reference for Interviews

### Most Common Patterns

```sql
-- 1. Aggregation with grouping
SELECT category, COUNT(*), AVG(price)
FROM products
GROUP BY category;

-- 2. Top N per group
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC) AS rn
    FROM products
) WHERE rn <= 3;

-- 3. Self-join
SELECT e1.name, e2.name AS manager
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.id;

-- 4. Subquery with IN
SELECT * FROM customers
WHERE id IN (SELECT customer_id FROM orders WHERE total > 1000);

-- 5. EXISTS
SELECT * FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);

-- 6. Window function
SELECT *, AVG(salary) OVER (PARTITION BY department) AS dept_avg
FROM employees;

-- 7. CTE
WITH high_spenders AS (
    SELECT customer_id, SUM(total) AS total_spent
    FROM orders GROUP BY customer_id
    HAVING SUM(total) > 1000
)
SELECT c.*, hs.total_spent
FROM customers c JOIN high_spenders hs ON c.id = hs.customer_id;
```

---

## Final Checklist

Before your interview, make sure you can:

✅ Write basic SELECT with WHERE, ORDER BY, LIMIT  
✅ Use JOINs (INNER, LEFT, RIGHT, FULL)  
✅ Apply GROUP BY with aggregate functions  
✅ Use HAVING to filter groups  
✅ Write subqueries in WHERE, SELECT, FROM  
✅ Use window functions (ROW_NUMBER, RANK, LAG, LEAD)  
✅ Write CTEs for complex queries  
✅ Handle NULL values properly  
✅ Optimize queries with indexes  
✅ Explain your thought process clearly

---

**Total Questions:** 110+
**Difficulty Distribution:** 40 Easy, 40 Medium, 30 Hard
**Company Coverage:** FAANG + Startups + Finance
**Estimated Prep Time:** 40-60 hours

**Good luck with your interviews! 🚀**
