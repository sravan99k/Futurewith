# SQL Ultimate Cheatsheet 📚

## Quick Reference Guide for All SQL Commands

---

## Table of Contents

1. [Database & Table Operations](#database--table-operations)
2. [Data Types](#data-types)
3. [Constraints](#constraints)
4. [CRUD Operations](#crud-operations)
5. [SELECT Queries](#select-queries)
6. [WHERE Clause](#where-clause)
7. [Operators](#operators)
8. [Aggregate Functions](#aggregate-functions)
9. [GROUP BY & HAVING](#group-by--having)
10. [JOINs](#joins)
11. [Subqueries](#subqueries)
12. [Window Functions](#window-functions)
13. [String Functions](#string-functions)
14. [Date & Time Functions](#date--time-functions)
15. [Mathematical Functions](#mathematical-functions)
16. [Conditional Logic](#conditional-logic)
17. [Set Operations](#set-operations)
18. [Indexes](#indexes)
19. [Transactions](#transactions)
20. [Common Patterns](#common-patterns)

---

## Database & Table Operations

### Create Database

```sql
CREATE DATABASE database_name;
USE database_name;
DROP DATABASE database_name;  -- Delete database
```

### Create Table

```sql
CREATE TABLE table_name (
    column1 datatype constraints,
    column2 datatype constraints,
    ...
);
```

**Example:**

```sql
CREATE TABLE employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    hire_date DATE DEFAULT CURRENT_DATE,
    salary DECIMAL(10, 2),
    department_id INT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);
```

### Modify Table Structure

```sql
-- Add column
ALTER TABLE table_name ADD column_name datatype;

-- Drop column
ALTER TABLE table_name DROP COLUMN column_name;

-- Modify column
ALTER TABLE table_name MODIFY COLUMN column_name new_datatype;

-- Rename column
ALTER TABLE table_name RENAME COLUMN old_name TO new_name;

-- Rename table
ALTER TABLE old_table_name RENAME TO new_table_name;
```

### Delete Table

```sql
DROP TABLE table_name;           -- Delete table permanently
TRUNCATE TABLE table_name;       -- Delete all data, keep structure
```

### View Table Structure

```sql
DESCRIBE table_name;              -- MySQL
\d table_name;                    -- PostgreSQL
EXEC sp_help 'table_name';        -- SQL Server
```

---

## Data Types

### Numeric Types

```sql
INT                    -- Integer: -2,147,483,648 to 2,147,483,647
SMALLINT               -- Small integer: -32,768 to 32,767
BIGINT                 -- Large integer: -9,223,372,036,854,775,808 to ...
DECIMAL(p, s)          -- Fixed precision: DECIMAL(10,2) = 99999999.99
NUMERIC(p, s)          -- Same as DECIMAL
FLOAT(p)               -- Floating point number
REAL                   -- Floating point (4 bytes)
DOUBLE                 -- Double precision float (8 bytes)
```

### String Types

```sql
CHAR(n)                -- Fixed-length string (n characters)
VARCHAR(n)             -- Variable-length string (up to n characters)
TEXT                   -- Large text (up to 65,535 characters)
MEDIUMTEXT             -- Medium text (up to 16,777,215 characters)
LONGTEXT               -- Long text (up to 4GB)
```

### Date & Time Types

```sql
DATE                   -- Date only: YYYY-MM-DD
TIME                   -- Time only: HH:MM:SS
DATETIME               -- Date and time: YYYY-MM-DD HH:MM:SS
TIMESTAMP              -- Timestamp (auto-updates)
YEAR                   -- Year in 4-digit format
```

### Boolean Type

```sql
BOOLEAN                -- TRUE (1) or FALSE (0)
BOOL                   -- Alias for BOOLEAN
```

### Binary Types

```sql
BLOB                   -- Binary large object
BINARY(n)              -- Fixed-length binary data
VARBINARY(n)           -- Variable-length binary data
```

---

## Constraints

```sql
PRIMARY KEY            -- Unique identifier, NOT NULL
FOREIGN KEY            -- Link to another table
UNIQUE                 -- No duplicate values allowed
NOT NULL               -- Must have a value
DEFAULT                -- Default value if none provided
CHECK                  -- Custom validation rule
AUTO_INCREMENT         -- Auto-generate sequential numbers (MySQL)
IDENTITY               -- Auto-generate sequential numbers (SQL Server)
```

### Constraint Examples

```sql
CREATE TABLE example (
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(100) UNIQUE NOT NULL,
    age INT CHECK (age >= 18),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add constraint after creation
ALTER TABLE table_name ADD CONSTRAINT constraint_name constraint_type (column);

-- Drop constraint
ALTER TABLE table_name DROP CONSTRAINT constraint_name;
```

---

## CRUD Operations

### INSERT (Create)

```sql
-- Insert single row
INSERT INTO table_name (column1, column2, column3)
VALUES (value1, value2, value3);

-- Insert multiple rows
INSERT INTO table_name (column1, column2, column3)
VALUES
    (value1a, value2a, value3a),
    (value1b, value2b, value3b),
    (value1c, value2c, value3c);

-- Insert from another table
INSERT INTO table_name (column1, column2)
SELECT column1, column2 FROM other_table WHERE condition;
```

### SELECT (Read)

```sql
-- Basic select
SELECT column1, column2 FROM table_name;

-- Select all columns
SELECT * FROM table_name;

-- Select with condition
SELECT * FROM table_name WHERE condition;
```

### UPDATE (Update)

```sql
-- Update specific rows
UPDATE table_name
SET column1 = value1, column2 = value2
WHERE condition;

-- Update all rows (DANGEROUS!)
UPDATE table_name
SET column1 = value1;

-- Update with calculation
UPDATE products
SET price = price * 1.10
WHERE category = 'Electronics';
```

### DELETE (Delete)

```sql
-- Delete specific rows
DELETE FROM table_name WHERE condition;

-- Delete all rows (DANGEROUS!)
DELETE FROM table_name;

-- Better: use TRUNCATE for deleting all
TRUNCATE TABLE table_name;
```

---

## SELECT Queries

### Basic SELECT

```sql
SELECT column1, column2 FROM table_name;
```

### SELECT with Aliases

```sql
SELECT
    column1 AS alias1,
    column2 AS 'Alias 2',
    column3 alias3
FROM table_name AS t;
```

### SELECT DISTINCT

```sql
SELECT DISTINCT column_name FROM table_name;

-- Count unique values
SELECT COUNT(DISTINCT column_name) FROM table_name;
```

### LIMIT / OFFSET

```sql
-- MySQL / PostgreSQL
SELECT * FROM table_name LIMIT 10;                    -- First 10 rows
SELECT * FROM table_name LIMIT 10 OFFSET 20;          -- Rows 21-30

-- SQL Server
SELECT TOP 10 * FROM table_name;                      -- First 10 rows
SELECT * FROM table_name ORDER BY id OFFSET 20 ROWS FETCH NEXT 10 ROWS ONLY;
```

### ORDER BY

```sql
-- Ascending (default)
SELECT * FROM table_name ORDER BY column1;
SELECT * FROM table_name ORDER BY column1 ASC;

-- Descending
SELECT * FROM table_name ORDER BY column1 DESC;

-- Multiple columns
SELECT * FROM table_name ORDER BY column1 DESC, column2 ASC;
```

---

## WHERE Clause

### Basic WHERE

```sql
SELECT * FROM table_name WHERE condition;
```

### Comparison Operators

```sql
WHERE age = 25                     -- Equal to
WHERE age != 25                    -- Not equal to
WHERE age <> 25                    -- Not equal to (alternative)
WHERE age > 25                     -- Greater than
WHERE age >= 25                    -- Greater than or equal to
WHERE age < 25                     -- Less than
WHERE age <= 25                    -- Less than or equal to
```

### Logical Operators

```sql
WHERE age > 25 AND city = 'NYC'    -- Both conditions true
WHERE age > 25 OR city = 'NYC'     -- At least one condition true
WHERE NOT age = 25                 -- Opposite of condition
```

### BETWEEN

```sql
WHERE age BETWEEN 25 AND 40        -- age >= 25 AND age <= 40
WHERE price BETWEEN 10.00 AND 50.00
```

### IN

```sql
WHERE city IN ('NYC', 'LA', 'Chicago')
WHERE age IN (25, 30, 35, 40)
WHERE id NOT IN (1, 2, 3)
```

### LIKE (Pattern Matching)

```sql
WHERE name LIKE 'J%'               -- Starts with J
WHERE name LIKE '%son'             -- Ends with son
WHERE name LIKE '%art%'            -- Contains art
WHERE name LIKE 'J___'             -- J followed by exactly 3 characters
WHERE name LIKE '[JMS]%'           -- Starts with J, M, or S (SQL Server)
WHERE name LIKE '[A-C]%'           -- Starts with A, B, or C (SQL Server)
```

**Wildcards:**

- `%` = Any number of characters
- `_` = Exactly one character
- `[]` = Any character in brackets (SQL Server only)

### IS NULL / IS NOT NULL

```sql
WHERE email IS NULL                -- Has no value
WHERE email IS NOT NULL            -- Has a value
```

---

## Operators

### Arithmetic Operators

```sql
SELECT price + tax AS total                    -- Addition
SELECT quantity - returned AS net_quantity     -- Subtraction
SELECT price * quantity AS total               -- Multiplication
SELECT total / quantity AS unit_price          -- Division
SELECT amount % 10 AS remainder                -- Modulo (remainder)
```

### Comparison Operators

```sql
=    -- Equal to
!=   -- Not equal to
<>   -- Not equal to (standard SQL)
>    -- Greater than
>=   -- Greater than or equal to
<    -- Less than
<=   -- Less than or equal to
```

### Logical Operators

```sql
AND  -- Both conditions must be true
OR   -- At least one condition must be true
NOT  -- Negates a condition
```

### Special Operators

```sql
BETWEEN ... AND ...      -- Range check (inclusive)
IN (...)                 -- Match any value in list
LIKE                     -- Pattern matching
IS NULL                  -- Check for NULL
IS NOT NULL              -- Check for non-NULL
EXISTS                   -- Check if subquery returns rows
ANY / SOME               -- Compare to any value in subquery
ALL                      -- Compare to all values in subquery
```

---

## Aggregate Functions

```sql
COUNT(*)                   -- Count all rows
COUNT(column)              -- Count non-NULL values
COUNT(DISTINCT column)     -- Count unique values

SUM(column)                -- Sum of all values
AVG(column)                -- Average of all values
MIN(column)                -- Minimum value
MAX(column)                -- Maximum value

-- Examples
SELECT COUNT(*) FROM customers;
SELECT AVG(price) FROM products;
SELECT MIN(order_date), MAX(order_date) FROM orders;
SELECT SUM(quantity * price) AS total_revenue FROM order_items;
```

### Statistical Functions

```sql
STDDEV(column)             -- Standard deviation
VARIANCE(column)           -- Variance
```

---

## GROUP BY & HAVING

### GROUP BY

```sql
-- Group by single column
SELECT category, COUNT(*) AS count
FROM products
GROUP BY category;

-- Group by multiple columns
SELECT category, brand, COUNT(*) AS count
FROM products
GROUP BY category, brand;

-- Group by with aggregate functions
SELECT
    category,
    COUNT(*) AS product_count,
    AVG(price) AS avg_price,
    MIN(price) AS min_price,
    MAX(price) AS max_price
FROM products
GROUP BY category;
```

### HAVING (Filter Groups)

```sql
-- HAVING filters groups (after GROUP BY)
SELECT category, COUNT(*) AS count
FROM products
GROUP BY category
HAVING COUNT(*) > 5;

-- Multiple HAVING conditions
SELECT category, AVG(price) AS avg_price
FROM products
GROUP BY category
HAVING AVG(price) > 100 AND COUNT(*) > 3;

-- WHERE vs HAVING
SELECT category, AVG(price) AS avg_price
FROM products
WHERE stock > 0              -- Filter rows BEFORE grouping
GROUP BY category
HAVING AVG(price) > 100;     -- Filter groups AFTER grouping
```

---

## JOINs

### INNER JOIN

```sql
-- Show only matching rows from both tables
SELECT a.column1, b.column2
FROM table_a a
INNER JOIN table_b b ON a.id = b.a_id;
```

### LEFT JOIN (LEFT OUTER JOIN)

```sql
-- Show ALL rows from left table, matching from right
SELECT a.column1, b.column2
FROM table_a a
LEFT JOIN table_b b ON a.id = b.a_id;
```

### RIGHT JOIN (RIGHT OUTER JOIN)

```sql
-- Show ALL rows from right table, matching from left
SELECT a.column1, b.column2
FROM table_a a
RIGHT JOIN table_b b ON a.id = b.a_id;
```

### FULL OUTER JOIN

```sql
-- Show ALL rows from both tables
SELECT a.column1, b.column2
FROM table_a a
FULL OUTER JOIN table_b b ON a.id = b.a_id;

-- MySQL doesn't support FULL OUTER JOIN directly
-- Workaround: UNION of LEFT and RIGHT JOIN
SELECT a.column1, b.column2 FROM table_a a LEFT JOIN table_b b ON a.id = b.a_id
UNION
SELECT a.column1, b.column2 FROM table_a a RIGHT JOIN table_b b ON a.id = b.a_id;
```

### CROSS JOIN (Cartesian Product)

```sql
-- Every row from table_a matched with every row from table_b
SELECT a.column1, b.column2
FROM table_a a
CROSS JOIN table_b b;
```

### SELF JOIN

```sql
-- Join table to itself
SELECT e1.name AS employee, e2.name AS manager
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.employee_id;
```

### Multiple JOINs

```sql
SELECT
    c.customer_name,
    o.order_date,
    p.product_name,
    oi.quantity
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
INNER JOIN products p ON oi.product_id = p.product_id;
```

---

## Subqueries

### Subquery in WHERE

```sql
-- Find products more expensive than average
SELECT product_name, price
FROM products
WHERE price > (SELECT AVG(price) FROM products);
```

### Subquery with IN

```sql
SELECT customer_name
FROM customers
WHERE customer_id IN (
    SELECT DISTINCT customer_id FROM orders WHERE total > 1000
);
```

### Subquery in SELECT

```sql
SELECT
    customer_name,
    (SELECT COUNT(*) FROM orders WHERE customer_id = c.customer_id) AS order_count
FROM customers c;
```

### Subquery in FROM (Derived Table)

```sql
SELECT category, avg_price
FROM (
    SELECT category, AVG(price) AS avg_price
    FROM products
    GROUP BY category
) AS category_averages
WHERE avg_price > 100;
```

### Correlated Subquery

```sql
-- Subquery references outer query
SELECT product_name, price
FROM products p1
WHERE price > (
    SELECT AVG(price)
    FROM products p2
    WHERE p2.category = p1.category
);
```

### EXISTS / NOT EXISTS

```sql
-- Check if subquery returns any rows
SELECT customer_name
FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id
);

SELECT customer_name
FROM customers c
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id
);
```

### ANY / ALL

```sql
-- ANY: True if comparison is true for ANY value
SELECT product_name, price
FROM products
WHERE price > ANY (SELECT price FROM products WHERE category = 'Electronics');

-- ALL: True if comparison is true for ALL values
SELECT product_name, price
FROM products
WHERE price > ALL (SELECT price FROM products WHERE category = 'Budget');
```

---

## Window Functions

### ROW_NUMBER

```sql
-- Assign sequential number to each row
SELECT
    product_name,
    price,
    ROW_NUMBER() OVER (ORDER BY price DESC) AS row_num
FROM products;
```

### RANK

```sql
-- Rank with gaps (1, 2, 2, 4, 5)
SELECT
    product_name,
    sales,
    RANK() OVER (ORDER BY sales DESC) AS sales_rank
FROM products;
```

### DENSE_RANK

```sql
-- Rank without gaps (1, 2, 2, 3, 4)
SELECT
    product_name,
    sales,
    DENSE_RANK() OVER (ORDER BY sales DESC) AS dense_rank
FROM products;
```

### PARTITION BY

```sql
-- Rank within each category
SELECT
    category,
    product_name,
    price,
    RANK() OVER (PARTITION BY category ORDER BY price DESC) AS category_rank
FROM products;
```

### LAG / LEAD

```sql
-- Access previous/next row values
SELECT
    order_date,
    total,
    LAG(total, 1) OVER (ORDER BY order_date) AS previous_total,
    LEAD(total, 1) OVER (ORDER BY order_date) AS next_total
FROM orders;
```

### SUM() OVER (Running Total)

```sql
SELECT
    order_date,
    total,
    SUM(total) OVER (ORDER BY order_date) AS running_total
FROM orders;
```

### AVG() OVER (Moving Average)

```sql
-- 3-day moving average
SELECT
    order_date,
    total,
    AVG(total) OVER (ORDER BY order_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg
FROM orders;
```

### NTILE

```sql
-- Divide rows into N groups
SELECT
    customer_name,
    total_spent,
    NTILE(4) OVER (ORDER BY total_spent DESC) AS quartile
FROM customer_totals;
```

---

## String Functions

```sql
-- Concatenation
CONCAT(str1, str2, ...)              -- 'Hello' + 'World' = 'HelloWorld'
CONCAT_WS(separator, str1, str2)     -- 'Hello', ' ', 'World' = 'Hello World'

-- Case conversion
UPPER(string)                        -- 'hello' → 'HELLO'
LOWER(string)                        -- 'HELLO' → 'hello'

-- Substring
SUBSTRING(string, start, length)     -- SUBSTRING('Hello', 1, 3) = 'Hel'
LEFT(string, length)                 -- LEFT('Hello', 3) = 'Hel'
RIGHT(string, length)                -- RIGHT('Hello', 3) = 'llo'

-- Trimming
TRIM(string)                         -- Remove leading/trailing spaces
LTRIM(string)                        -- Remove leading spaces
RTRIM(string)                        -- Remove trailing spaces

-- Length
LENGTH(string)                       -- Number of characters
CHAR_LENGTH(string)                  -- Number of characters (Unicode safe)

-- Replace
REPLACE(string, old, new)            -- REPLACE('Hello', 'l', 'L') = 'HeLLo'

-- Position
POSITION(substring IN string)        -- Find position of substring
INSTR(string, substring)             -- Find position (MySQL)

-- Padding
LPAD(string, length, pad_string)     -- LPAD('5', 3, '0') = '005'
RPAD(string, length, pad_string)     -- RPAD('5', 3, '0') = '500'

-- Reverse
REVERSE(string)                      -- REVERSE('Hello') = 'olleH'

-- Repeat
REPEAT(string, count)                -- REPEAT('Ha', 3) = 'HaHaHa'
```

### Examples

```sql
-- Full name
SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM customers;

-- Email username
SELECT LEFT(email, POSITION('@' IN email) - 1) AS username FROM customers;

-- Format phone number
SELECT CONCAT('(', SUBSTRING(phone, 1, 3), ') ', SUBSTRING(phone, 4, 3), '-', SUBSTRING(phone, 7, 4))
FROM customers;
```

---

## Date & Time Functions

### Current Date/Time

```sql
CURRENT_DATE                         -- 2025-11-09
CURRENT_TIME                         -- 13:20:53
CURRENT_TIMESTAMP                    -- 2025-11-09 13:20:53
NOW()                                -- 2025-11-09 13:20:53 (MySQL)
```

### Extract Parts

```sql
YEAR(date)                           -- Extract year (2025)
MONTH(date)                          -- Extract month (11)
DAY(date)                            -- Extract day (9)
HOUR(time)                           -- Extract hour
MINUTE(time)                         -- Extract minute
SECOND(time)                         -- Extract second
DAYNAME(date)                        -- Day name (Saturday)
MONTHNAME(date)                      -- Month name (November)
DAYOFWEEK(date)                      -- Day of week (1=Sunday, 7=Saturday)
DAYOFYEAR(date)                      -- Day of year (1-366)
WEEK(date)                           -- Week number (0-53)
QUARTER(date)                        -- Quarter (1-4)
```

### Date Arithmetic

```sql
-- Add/Subtract dates (MySQL)
DATE_ADD(date, INTERVAL value unit)
DATE_SUB(date, INTERVAL value unit)

-- Examples
DATE_ADD(CURRENT_DATE, INTERVAL 7 DAY)           -- 7 days from now
DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH)         -- 1 month ago
DATE_ADD(NOW(), INTERVAL 2 HOUR)                 -- 2 hours from now

-- PostgreSQL
date + INTERVAL '7 days'
date - INTERVAL '1 month'

-- SQL Server
DATEADD(DAY, 7, GETDATE())
DATEADD(MONTH, -1, GETDATE())
```

### Date Difference

```sql
-- MySQL
DATEDIFF(date1, date2)               -- Difference in days
TIMESTAMPDIFF(unit, date1, date2)    -- Difference in specified unit

-- PostgreSQL
AGE(date1, date2)                    -- Interval between dates

-- SQL Server
DATEDIFF(unit, date1, date2)
```

### Date Formatting

```sql
-- MySQL
DATE_FORMAT(date, format)
-- DATE_FORMAT('2025-11-09', '%M %d, %Y') → 'November 09, 2025'

-- PostgreSQL
TO_CHAR(date, format)

-- SQL Server
FORMAT(date, format)
```

**Common Format Codes (MySQL):**

```
%Y  Year (4 digits)          → 2025
%y  Year (2 digits)          → 25
%M  Month name               → November
%m  Month (numeric)          → 11
%d  Day                      → 09
%H  Hour (24h)               → 13
%h  Hour (12h)               → 01
%i  Minutes                  → 20
%s  Seconds                  → 53
%p  AM/PM                    → PM
%W  Weekday name             → Saturday
```

---

## Mathematical Functions

```sql
ABS(number)                          -- Absolute value
CEIL(number) / CEILING(number)       -- Round up
FLOOR(number)                        -- Round down
ROUND(number, decimals)              -- Round to decimals
TRUNCATE(number, decimals)           -- Truncate to decimals

MOD(number, divisor)                 -- Modulo (remainder)
POWER(base, exponent)                -- base^exponent
SQRT(number)                         -- Square root
EXP(number)                          -- e^number
LN(number)                           -- Natural logarithm
LOG(number)                          -- Logarithm base 10

RAND()                               -- Random number 0-1
SIGN(number)                         -- -1, 0, or 1
GREATEST(val1, val2, ...)            -- Largest value
LEAST(val1, val2, ...)               -- Smallest value

-- Examples
SELECT ROUND(price, 2) FROM products;
SELECT CEIL(AVG(rating)) FROM reviews;
SELECT FLOOR(RAND() * 100) AS random_number;
```

---

## Conditional Logic

### CASE Statement

```sql
-- Simple CASE
SELECT
    product_name,
    price,
    CASE
        WHEN price < 50 THEN 'Cheap'
        WHEN price BETWEEN 50 AND 200 THEN 'Moderate'
        WHEN price > 200 THEN 'Expensive'
        ELSE 'Unknown'
    END AS price_category
FROM products;

-- CASE in aggregation
SELECT
    SUM(CASE WHEN status = 'completed' THEN total ELSE 0 END) AS completed_revenue,
    SUM(CASE WHEN status = 'pending' THEN total ELSE 0 END) AS pending_revenue
FROM orders;
```

### COALESCE (Return first non-NULL)

```sql
SELECT COALESCE(phone, email, 'No contact') AS contact FROM customers;
```

### NULLIF (Return NULL if equal)

```sql
SELECT NULLIF(column1, column2) FROM table_name;
-- If column1 = column2, return NULL, else return column1
```

### IFNULL / ISNULL / NVL

```sql
-- MySQL
IFNULL(column, 'default_value')

-- SQL Server
ISNULL(column, 'default_value')

-- Oracle
NVL(column, 'default_value')
```

---

## Set Operations

### UNION (Combine results, remove duplicates)

```sql
SELECT column FROM table1
UNION
SELECT column FROM table2;
```

### UNION ALL (Combine results, keep duplicates)

```sql
SELECT column FROM table1
UNION ALL
SELECT column FROM table2;
```

### INTERSECT (Only rows in both)

```sql
SELECT column FROM table1
INTERSECT
SELECT column FROM table2;
```

### EXCEPT / MINUS (Rows in first but not second)

```sql
-- PostgreSQL / SQL Server
SELECT column FROM table1
EXCEPT
SELECT column FROM table2;

-- Oracle
SELECT column FROM table1
MINUS
SELECT column FROM table2;
```

---

## Indexes

### Create Index

```sql
-- Single column index
CREATE INDEX idx_name ON table_name(column_name);

-- Composite index (multiple columns)
CREATE INDEX idx_name ON table_name(column1, column2);

-- Unique index
CREATE UNIQUE INDEX idx_name ON table_name(column_name);
```

### Drop Index

```sql
-- MySQL
DROP INDEX idx_name ON table_name;

-- PostgreSQL / SQL Server
DROP INDEX idx_name;
```

### View Indexes

```sql
-- MySQL
SHOW INDEX FROM table_name;

-- PostgreSQL
\d table_name

-- SQL Server
sp_helpindex 'table_name';
```

---

## Transactions

### Basic Transaction

```sql
BEGIN TRANSACTION;  -- Start transaction (or START TRANSACTION;)

-- Your SQL commands here
INSERT INTO accounts (balance) VALUES (1000);
UPDATE accounts SET balance = balance - 100 WHERE id = 1;

COMMIT;             -- Save changes permanently
-- OR
ROLLBACK;           -- Undo all changes
```

### Savepoints

```sql
BEGIN TRANSACTION;

INSERT INTO customers (name) VALUES ('John');
SAVEPOINT sp1;

UPDATE customers SET city = 'NYC' WHERE name = 'John';
SAVEPOINT sp2;

-- Undo to sp2
ROLLBACK TO sp2;

COMMIT;
```

### Transaction Isolation Levels

```sql
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

---

## Common Patterns

### Pagination

```sql
-- MySQL / PostgreSQL
SELECT * FROM products
ORDER BY product_id
LIMIT 20 OFFSET 40;  -- Page 3 (rows 41-60)

-- SQL Server
SELECT * FROM products
ORDER BY product_id
OFFSET 40 ROWS FETCH NEXT 20 ROWS ONLY;
```

### Find Duplicates

```sql
SELECT column, COUNT(*) AS count
FROM table_name
GROUP BY column
HAVING COUNT(*) > 1;
```

### Delete Duplicates (Keep one)

```sql
-- Using CTE and ROW_NUMBER
WITH duplicates AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS row_num
    FROM customers
)
DELETE FROM duplicates WHERE row_num > 1;
```

### Top N per Group

```sql
-- Using window functions
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC) AS rank
    FROM products
)
SELECT * FROM ranked WHERE rank <= 3;
```

### Running Total

```sql
SELECT
    order_date,
    total,
    SUM(total) OVER (ORDER BY order_date) AS running_total
FROM orders;
```

### Pivot Table (Dynamic Columns)

```sql
SELECT
    customer_id,
    SUM(CASE WHEN product = 'Laptop' THEN quantity ELSE 0 END) AS laptop_qty,
    SUM(CASE WHEN product = 'Mouse' THEN quantity ELSE 0 END) AS mouse_qty,
    SUM(CASE WHEN product = 'Keyboard' THEN quantity ELSE 0 END) AS keyboard_qty
FROM orders
GROUP BY customer_id;
```

### Ranking with Ties

```sql
SELECT
    product_name,
    sales,
    RANK() OVER (ORDER BY sales DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY sales DESC) AS dense_rank
FROM products;
```

### Conditional Counting

```sql
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_count,
    SUM(CASE WHEN status = 'inactive' THEN 1 ELSE 0 END) AS inactive_count
FROM users;
```

### Generate Date Series

```sql
-- PostgreSQL
SELECT generate_series(
    '2025-01-01'::date,
    '2025-12-31'::date,
    '1 day'::interval
) AS date;

-- MySQL (recursive CTE)
WITH RECURSIVE dates AS (
    SELECT '2025-01-01' AS date
    UNION ALL
    SELECT DATE_ADD(date, INTERVAL 1 DAY)
    FROM dates
    WHERE date < '2025-12-31'
)
SELECT * FROM dates;
```

---

## Performance Tips

### Use Indexes Wisely

```sql
-- ✅ Good: Index on WHERE clause columns
CREATE INDEX idx_customer_state ON customers(state);
SELECT * FROM customers WHERE state = 'CA';

-- ❌ Bad: Function in WHERE prevents index use
SELECT * FROM customers WHERE UPPER(state) = 'CA';  -- Can't use index!

-- ✅ Good: Store uppercase in column or use functional index
CREATE INDEX idx_customer_state_upper ON customers(UPPER(state));
```

### Select Only Needed Columns

```sql
-- ❌ Bad
SELECT * FROM customers;

-- ✅ Good
SELECT first_name, last_name, email FROM customers;
```

### Use EXISTS Instead of IN for Large Sets

```sql
-- ❌ Slower
SELECT * FROM customers WHERE id IN (SELECT customer_id FROM orders);

-- ✅ Faster
SELECT * FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);
```

### Avoid SELECT DISTINCT When Possible

```sql
-- ❌ Slower
SELECT DISTINCT category FROM products;

-- ✅ Faster (if appropriate)
SELECT category FROM products GROUP BY category;
```

### Use LIMIT

```sql
-- Always limit results when testing
SELECT * FROM large_table LIMIT 100;
```

---

## Quick Reference Card

```sql
-- DATABASE
CREATE DATABASE db_name;
USE db_name;
DROP DATABASE db_name;

-- TABLE
CREATE TABLE t (col1 INT, col2 VARCHAR(50));
ALTER TABLE t ADD col3 DATE;
DROP TABLE t;

-- CRUD
INSERT INTO t VALUES (1, 'data');
SELECT * FROM t WHERE condition;
UPDATE t SET col1 = value WHERE condition;
DELETE FROM t WHERE condition;

-- JOIN
SELECT * FROM t1 JOIN t2 ON t1.id = t2.id;

-- AGGREGATE
SELECT COUNT(*), SUM(col), AVG(col), MIN(col), MAX(col) FROM t;

-- GROUP BY
SELECT col, COUNT(*) FROM t GROUP BY col HAVING COUNT(*) > 1;

-- SUBQUERY
SELECT * FROM t WHERE col IN (SELECT col FROM t2);

-- TRANSACTION
BEGIN; ... COMMIT; / ROLLBACK;
```

---

**Total Lines:** 500+
**Suitable for:** Quick reference during coding & interviews
**Print this:** Keep it handy! 📌
