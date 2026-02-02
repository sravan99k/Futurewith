# 🧠 Machine Learning Practice Questions & Exercises

## Simple Level Questions for Everyone!

_Based on the Machine Learning Complete Simple Guide_

---

## 🎯 **SECTION A: SUPERVISED LEARNING BASICS** (Beginner Level)

### **Question 1: Regression vs Classification**

Circle the correct answer for each scenario:

1. Predicting house prices ($200,000, $350,000)
   a) Regression b) Classification

2. Sorting emails into "spam" or "not spam"
   a) Regression b) Classification

3. Predicting tomorrow's temperature (72°F)
   a) Regression b) Classification

4. Classifying photos as "cat" or "dog"
   a) Regression b) Classification

5. Predicting your exam score (85 points)
   a) Regression b) Classification

**Answers:** 1-Regression, 2-Classification, 3-Regression, 4-Classification, 5-Regression

---

### **Question 2: Algorithm Matching**

Draw lines to match each problem with the best algorithm:

| **Problem**                                 | **Best Algorithm**        |
| ------------------------------------------- | ------------------------- |
| Predict house prices                        | K-Nearest Neighbors (KNN) |
| Email spam detection                        | Logistic Regression       |
| Group customers by shopping behavior        | K-Means Clustering        |
| Find the 5 most similar movies to "Titanic" | Decision Tree             |

**Correct Matching:**

- Predict house prices → Linear Regression or Decision Tree
- Email spam detection → Logistic Regression or Naive Bayes
- Group customers by shopping behavior → K-Means Clustering
- Find the 5 most similar movies to "Titanic" → K-Nearest Neighbors (KNN)

---

### **Question 3: Fill in the Blanks**

Complete these sentences using the words: **training, testing, supervised, features, target**

1. ******\_****** learning is like studying with a teacher who has all the answers.
2. We use ******\_****** data to teach the computer patterns.
3. We use ******\_****** data to see how well the computer learned.
4. ******\_****** are the inputs we give to the computer (like house size, number of rooms).
5. The ******\_****** is what we want the computer to predict (like house price).

**Answers:** 1-Supervised, 2-training, 3-testing, 4-Features, 5-target

---

### **Question 4: Simple Predictions**

For each scenario, decide if you would use Linear Regression, Decision Tree, or Random Forest:

1. **Student wants to predict exam score based on study hours**
   Answer: ******\_******

2. **Real estate agent wants to predict house prices with high accuracy**
   Answer: ******\_******

3. **Doctor wants to explain medical decisions clearly to patients**
   Answer: ******\_******

**Answers:** 1-Linear Regression, 2-Random Forest, 3-Decision Tree

---

## 🎯 **SECTION B: REGRESSION ALGORITHMS** (Medium Level)

### **Question 5: Linear Regression Understanding**

A linear regression model found this relationship:
**House Price = House Size × 500 + Base Price**

If Base Price = $50,000, predict prices for these houses:

| House Size (sq ft) | Predicted Price   |
| ------------------ | ----------------- |
| 1000               | $******\_\_****** |
| 1500               | $******\_\_****** |
| 2000               | $******\_\_****** |

**Answers:**

- 1000 sq ft: $550,000
- 1500 sq ft: $800,000
- 2000 sq ft: $1,050,000

---

### **Question 6: Decision Tree Logic**

A decision tree for house prices works like this:

```
Is house size > 1500 sq ft?
├─ Yes → Is location = "Good"?
│   ├─ Yes → Price: $300,000+
│   └─ No → Price: $200,000-300,000
└─ No → Price: $100,000-200,000
```

Predict prices for these houses:

1. 1800 sq ft, Good location → $******\_\_******
2. 1200 sq ft, Bad location → $******\_\_******
3. 2000 sq ft, Bad location → $******\_\_******
4. 1400 sq ft, Good location → $******\_\_******

**Answers:**

1. $300,000+
2. $100,000-200,000
3. $200,000-300,000
4. $100,000-200,000

---

### **Question 7: Random Forest vs Single Decision Tree**

Why would you choose Random Forest over a single Decision Tree?

a) Random Forest is always faster
b) Random Forest combines many trees for better accuracy
c) Random Forest is easier to understand
d) Random Forest uses less memory

**Answer: b) Random Forest combines many trees for better accuracy**

---

## 🎯 **SECTION C: CLASSIFICATION ALGORITHMS** (Medium Level)

### **Question 8: Classification Predictions**

A logistic regression model gives these spam probabilities:

| Email Content                      | Spam Probability | Prediction   |
| ---------------------------------- | ---------------- | ------------ |
| "You won $1,000,000! Click here!"  | 0.95             | ****\_\_**** |
| "Meeting at 3pm tomorrow"          | 0.15             | ****\_\_**** |
| "Free iPhone! Limited time offer!" | 0.88             | ****\_\_**** |
| "Thanks for your purchase"         | 0.08             | ****\_\_**** |

**Rule:** If probability > 0.5, predict "Spam"; otherwise "Not Spam"

**Answers:**

- "You won $1,000,000, Click here!" → Spam (0.95 > 0.5)
- "Meeting at 3pm tomorrow" → Not Spam (0.15 < 0.5)
- "Free iPhone! Limited time offer!" → Spam (0.88 > 0.5)
- "Thanks for your purchase" → Not Spam (0.08 < 0.5)

---

### **Question 9: KNN Neighbor Voting**

For K-Nearest Neighbors with k=5, find the most similar neighbors:

| Neighbor | Similarity Score | Class |
| -------- | ---------------- | ----- |
| A        | 0.95             | Cat   |
| B        | 0.88             | Dog   |
| C        | 0.92             | Cat   |
| D        | 0.85             | Cat   |
| E        | 0.78             | Dog   |

**Question:** What would you predict for the new item?
a) Cat b) Dog c) Uncertain

**Answer:** a) Cat (3 Cat votes vs 2 Dog votes)

---

### **Question 10: Algorithm Comparison**

For each scenario, choose the best classification algorithm:

1. **Text spam detection** (many words to analyze)
   Answer: ******\_******

2. **Medical diagnosis** (doctor needs to explain reasoning)
   Answer: ******\_******

3. **Movie recommendation** (find similar movies)
   Answer: ******\_******

4. **High accuracy needed** (important business decision)
   Answer: ******\_******

**Answers:**

1. Naive Bayes (good with text)
2. Decision Tree (easy to explain)
3. K-Nearest Neighbors (finds similar items)
4. Random Forest (high accuracy)

---

## 🎯 **SECTION D: UNSUPERVISED LEARNING** (Advanced Level)

### **Question 11: Clustering Types**

Match each scenario with the right clustering algorithm:

| **Scenario**                                    | **Algorithm** |
| ----------------------------------------------- | ------------- |
| Group customers without knowing how many groups | DBSCAN        |
| Find exactly 4 customer types                   | K-Means       |
| Build a hierarchy of customer types             | Hierarchical  |
| Find unusual customers (outliers)               | DBSCAN        |

---

### **Question 12: K-Means vs DBSCAN**

Explain the difference:

**K-Means:**

- Expects ******\_****** number of groups
- Works best with ******\_****** shaped clusters
- Doesn't identify ******\_******

**DBSCAN:**

- Finds ******\_****** number of groups automatically
- Works with ******\_****** shaped clusters
- Can identify ******\_****** (people who don't fit any group)

**Answers:**

- K-Means: specific, round, outliers
- DBSCAN: variable, any, outliers/noise

---

### **Question 13: Dimensionality Reduction**

You have student data with 50 features (test scores, attendance, participation, etc.). You want to:

1. **Reduce to 2 main patterns for visualization**
   Algorithm: ******\_******

2. **Remove redundant features before using other ML**
   Algorithm: ******\_******

3. **Create a beautiful map showing student relationships**
   Algorithm: ******\_******

**Answers:** 1-PCA, 2-PCA, 3-t-SNE

---

### **Question 14: PCA Component Interpretation**

A PCA analysis of student performance shows:

- **Component 1 (60% of variance):** Math, Science, English scores
- **Component 2 (25% of variance):** Attendance, Participation, Homework completion

What do these components represent?
Component 1: ******\_******
Component 2: ******\_******

**Answers:**

- Component 1: Academic Knowledge/Intelligence
- Component 2: Student Engagement/Work Ethic

---

## 🎯 **SECTION E: REAL-WORLD SCENARIOS** (Advanced Level)

### **Question 15: Algorithm Selection**

For each business problem, recommend the best ML approach:

**Scenario 1: E-commerce Company**

- **Goal:** Predict which customers will buy again
- **Data:** Purchase history, website visits, demographics
- **Answer:** ******\_******

**Scenario 2: Hospital**

- **Goal:** Group patients by similar symptoms for treatment research
- **Data:** Symptoms, test results, treatment outcomes
- **Answer:** ******\_******

**Scenario 3: News Website**

- **Goal:** Automatically categorize articles as politics, sports, technology, entertainment
- **Data:** Article text, keywords, author information
- **Answer:** ******\_******

**Sample Answers:**

1. **E-commerce:** Supervised Classification (predict buy/not buy)
2. **Hospital:** Unsupervised Clustering (find patient groups)
3. **News:** Supervised Classification (categorize articles)

---

### **Question 16: Model Evaluation**

You trained three models with these accuracies:

| Model | Training Accuracy | Testing Accuracy |
| ----- | ----------------- | ---------------- |
| A     | 95%               | 70%              |
| B     | 80%               | 78%              |
| C     | 85%               | 85%              |

**Questions:**

1. **Which model is overfitting?** ******\_******
2. **Which model generalizes best?** ******\_******
3. **Which model would you choose and why?** ******\_******

**Answers:**

1. Model A (huge gap between training and testing)
2. Model C (consistent performance)
3. Model C (best generalization, no overfitting)

---

### **Question 17: Data Quality Issues**

You notice these problems in your dataset:

| Problem                                                   | How to Fix?    |
| --------------------------------------------------------- | -------------- |
| 50% of age values are missing                             | ******\_****** |
| Some prices are negative                                  | ******\_****** |
| House sizes range from 100 to 100,000 sq ft (unrealistic) | ******\_****** |
| City names have typos ("New York" vs "Newyork")           | ******\_****** |

**Sample Answers:**

- 50% missing ages → Remove rows or fill with median/mean
- Negative prices → Check data source, likely errors
- Extreme house sizes → Remove outliers or check data collection
- City name typos → Clean and standardize text data

---

### **Question 18: Performance Optimization**

Your model takes 2 hours to train on 1 million records. You need to reduce training time.

**Which strategies would help? (Check all that apply)**

□ Use a simpler algorithm
□ Use more powerful hardware
□ Reduce number of features
□ Use data sampling (train on subset)
□ Use cloud computing
□ Use a more efficient library

**All answers are valid strategies for reducing training time**

---

## 🎯 **SECTION F: CODING CONCEPTS** (Intermediate Level)

### **Question 19: Code Understanding**

Look at this code and answer questions:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Prepare data
X = data[['age', 'income', 'education']]
y = data['job_change']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
```

**Questions:**

1. **What does `test_size=0.2` mean?**
   Answer: ******\_******

2. **What does `n_estimators=100` mean?**
   Answer: ******\_******

3. **What does the `.score()` method return?**
   Answer: ******\_******

4. **What type of problem is this?**
   Answer: ******\_******

**Answers:**

1. 20% of data used for testing, 80% for training
2. Uses 100 decision trees in the random forest
3. Accuracy percentage on test data
4. Classification problem

---

### **Question 20: Data Preprocessing**

You have this data that needs cleaning before ML:

| Feature   | Current Values                                | Problem        | Solution       |
| --------- | --------------------------------------------- | -------------- | -------------- |
| Age       | [25, 30, , 45, 50]                            | ******\_****** | ******\_****** |
| Income    | [50000, 60000, 70000, 80000, 90000]           | ******\_****** | ******\_****** |
| Education | ["High School", "Bachelor", "PhD", "Masters"] | ******\_****** | ******\_****** |

**Answers:**

- Age: Missing values → Fill with median or remove row
- Income: No problem (clean numerical data)
- Education: Categorical text → Convert to numbers (0,1,2,3) or one-hot encode

---

## 🎯 **SECTION G: ADVANCED CONCEPTS** (Expert Level)

### **Question 21: Bias vs Variance**

Explain in simple terms:

**High Bias (Underfitting):**

- Problem: ******\_******
- Example: ******\_******
- Fix: ******\_******

**High Variance (Overfitting):**

- Problem: ******\_******
- Example: ******\_******
- Fix: ******\_******

**Sample Answers:**

- High Bias: Model too simple, misses patterns → Use more complex model
- High Variance: Model too complex, memorizes noise → Use simpler model or more data

---

### **Question 22: Feature Engineering**

You're predicting house prices. For each feature, decide if you should keep, modify, or create new features:

| Feature             | Action         | Reason         |
| ------------------- | -------------- | -------------- |
| House size in sq ft | ******\_****** | ******\_****** |
| Year built          | ******\_****** | ******\_****** |
| Address             | ******\_****** | ******\_****** |
| Number of bedrooms  | ******\_****** | ******\_****** |

**Sample Answers:**

- House size: Keep (directly affects price)
- Year built: Modify to age (newer = more valuable)
- Address: Create new features (distance to city, school district quality)
- Number of bedrooms: Keep (affects price)

---

### **Question 23: Ensemble Methods**

Why do ensemble methods often work better than single models?

1. ***
2. ***
3. ***

**Sample Answers:**

1. Combine multiple perspectives (like asking many experts)
2. Reduce overfitting by averaging predictions
3. Handle different types of patterns in data

---

### **Question 24: Model Deployment**

You've built a great model that predicts customer churn. What do you need to consider for deployment?

**Technical Considerations:**

- ***
- ***
- ***

**Business Considerations:**

- ***
- ***
- ***

**Sample Answers:**
Technical: API development, model monitoring, data pipeline
Business: User training, ROI measurement, gradual rollout

---

## 🎯 **SECTION H: FUN CHALLENGES** (For All Levels)

### **Challenge 1: Build Your Own Algorithm**

Design a simple algorithm to sort students by academic performance:

**Your Rules:**

1. If test score > 90 → ******\_******
2. If test score 70-90 AND attendance > 95% → ******\_******
3. If test score 70-90 OR attendance > 95% → ******\_******
4. Otherwise → ******\_******

**Sample Answer:**

1. "Excellent"
2. "Good"
3. "Average"
4. "Needs Improvement"

---

### **Challenge 2: Debug the Model**

This model has problems. Can you spot them?

```python
# Problematic code
model = LinearRegression()
model.fit(X_train, y_train)
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_train, y_train)  # BUG HERE!
print(f"Model is good if accuracy > 80%")
```

**Problems you can find:**

1. ***
2. ***
3. ***

**Answers:**

1. Used training data for testing (line 4)
2. No data splitting mentioned
3. Accuracy threshold is vague

---

### **Challenge 3: Algorithm Storytelling**

Complete this story about choosing ML algorithms:

"Once upon a time, there was a data scientist named Alex who had to solve three problems:

Problem 1: Predict how much money customers will spend
Alex chose ******\_****** because ******\_******

Problem 2: Group customers into shopping types
Alex chose ******\_****** because ******\_******

Problem 3: Sort product reviews into positive/negative
Alex chose ******\_****** because ******\_******

Alex's secret was always asking: 'What am I trying to accomplish?' before choosing an algorithm."

**Sample Answers:**

- Problem 1: Linear Regression (predicting numbers)
- Problem 2: K-Means (finding groups without labels)
- Problem 3: Logistic Regression or Naive Bayes (classification with text)

---

## 🎯 **SECTION I: ANSWER KEY & LEARNING TIPS**

### **Understanding Your Mistakes:**

**If you got Algorithm Selection wrong:**

- Remember: Numbers = Regression, Categories = Classification
- Data with labels = Supervised, No labels = Unsupervised
- Want accuracy? Try Random Forest
- Want to explain decisions? Try Decision Tree

**If you got Clustering wrong:**

- K-Means: You decide number of groups
- DBSCAN: Finds outliers and works with any shape
- Hierarchical: Shows relationships between groups

**If you got PCA/t-SNE wrong:**

- PCA: Simplifies data for algorithms
- t-SNE: Makes beautiful visualizations

### **🎊 CONGRATULATIONS!**

**You've completed all practice questions for Machine Learning!**

### **What Your Scores Mean:**

- **20-24 correct:** 🌟 ML Expert - You're ready for Deep Learning!
- **16-19 correct:** 🎯 ML Practitioner - Review tricky concepts
- **12-15 correct:** 📚 ML Learner - Keep practicing
- **Under 12:** 💪 ML Beginner - Study the guide again

### **Learning Path Recommendations:**

**If you scored 20-24:**
✅ You understand ML concepts well  
✅ Ready for Step 3: Deep Learning Neural Networks  
✅ Can start building real projects

**If you scored 16-19:**
✅ Good understanding, some gaps  
✅ Review sections you got wrong  
✅ Practice with simple projects

**If you scored 12-15:**
✅ Basic concepts understood  
✅ Need more practice with algorithms  
✅ Review the main guide carefully

**If you scored under 12:**
✅ Start with fundamentals  
✅ Focus on supervised vs unsupervised  
✅ Practice with simple examples

### **Next Steps:**

1. **Build a simple project** using what you learned
2. **Try different algorithms** on the same dataset
3. **Explain ML concepts** to someone else
4. **Get ready for Step 3: Deep Learning!** 🚀

### **🏆 Final Challenge: Teach Someone Else**

The best way to learn is to teach! Explain these concepts to a friend or family member:

- "Supervised learning is like studying with a teacher..."
- "Random Forest combines many decision trees..."
- "Clustering finds groups without knowing the groups first..."

**Remember: Every expert was once a beginner!** 🎓
