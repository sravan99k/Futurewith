# AI Cheat Sheets & Quick Reference Guide - Enhanced Visual Edition

## Your Complete Visual AI Assistant

_📊 Visual guides, flowcharts, and quick reference cards for instant AI problem-solving_

---

## 🎯 Quick Navigation

| Feature                      | What You'll Find                                   |
| ---------------------------- | -------------------------------------------------- |
| 🔍 **Algorithm Selection**   | Interactive flowcharts to choose the right AI tool |
| 📊 **Visual Comparisons**    | Performance tables and capability matrices         |
| 🚀 **Quick Reference Cards** | Condensed cheat cards for each algorithm           |
| 📈 **Process Diagrams**      | Flowcharts showing how algorithms work             |
| ⚡ **Instant Lookup**        | One-line definitions for instant recall            |
| 🖨️ **Print-Ready**           | Optimized for printing reference cards             |

---

## 🎯 How to Use This Guide

### 📚 **For Absolute Beginners (Start Here!)**

- Begin with **Algorithm Quick Reference** - explains what each AI tool does
- Use **Simple Examples** sections to understand concepts
- Don't worry about technical details yet!

### ⚡ **For Quick Answers**

- Jump to any section using the table below
- Each section has a **Simple Summary** first
- Technical details come after for those who want them

### 🔧 **For Hands-On Practice**

- Use **Code Templates Library** with step-by-step instructions
- Try **Practice Exercises** that build understanding
- Start simple, then add complexity

### 📖 **Table of Contents**

1. [Algorithm Quick Reference - What Does Each AI Tool Do?](#algorithm-quick-reference)
2. [Model Comparison - Which Tool is Best?](#model-comparison-matrix)
3. [Code Templates - Ready-to-Use Examples](#code-templates-library)
4. [Library Quick References - Popular AI Tools](#library-quick-references)
5. [Data Prep Cheatsheet - Getting Data Ready](#data-preprocessing-cheatsheet)
6. [Model Evaluation - How Good is Your AI?](#model-evaluation--metrics)
7. [Tuning Guide - Making Your AI Better](#hyperparameter-tuning-guide)
8. [Deep Learning - Advanced AI Brains](#deep-learning-architecture-reference)
9. [Computer Vision - Teaching AI to See](#computer-vision-quick-reference)
10. [Text Processing - Teaching AI Language](#nlp-processing-pipeline)
11. [Deployment Checklists - Going Live](#deployment-checklists)
12. [Problem Solving - Fix Common Issues](#debugging--troubleshooting-guide)
13. [Hardware Guide - What Equipment You Need](#hardware-requirements-summary)
14. [Speed Tips - Making AI Faster](#performance-optimization-tips)
15. [Error Solutions - Common Problems Fixed](#common-error-solutions)
16. [Best Practices - Do's and Don'ts](#best-practices-summary)

---

## 🎯 ALGORITHM SELECTION FLOWCHART

### Interactive Decision Tree - Choose Your AI Tool

```mermaid
flowchart TD
    A[What's Your Problem?] --> B{Predict Numbers?}
    B -->|Yes| C[Continuous Values]
    B -->|No| D{Categorical/Groups?}

    C --> E{How much data?}
    E -->|< 1K| F[Linear Regression]
    E -->|1K-100K| G[Random Forest Regressor]
    E -->|100K+| H[Gradient Boosting/XGBoost]

    D --> I{Binary Decision?}
    I -->|Yes| J[Yes/No Classification]
    I -->|No| K{Multiple Categories?}

    J --> L{Data Size?}
    L -->|< 1K| M[Logistic Regression]
    L -->|1K-100K| N[Random Forest]
    L -->|100K+| O[XGBoost]

    K --> P{Multiple Categories}
    P -->|Any Size| Q[Random Forest]

    C --> R[For Images: CNN]
    C --> S[For Text: BERT/Transformers]
    C --> T[For Time Series: LSTM]

    D --> U[For Clustering: K-Means]
    D --> V[For Anomaly Detection: Isolation Forest]

    style A fill:#e1f5fe
    style F fill:#c8e6c9
    style G fill:#c8e6c9
    style H fill:#c8e6c9
    style M fill:#fff3e0
    style N fill:#fff3e0
    style O fill:#fff3e0
    style Q fill:#f3e5f5
```

---

## 🃏 VISUAL QUICK REFERENCE CARDS

### 📈 Supervised Learning Cards

### Simple Summary

Think of AI algorithms like **different tools in a toolbox**:

- **Hammer (Linear Regression)**: Great for simple, straight-line problems
- **Multi-tool (Decision Tree)**: Versatile, can handle many different problems
- **Precision instrument (SVM)**: Excellent for complex, high-tech problems

```
╔══════════════════════════════════════════════════════════════╗
║                    LINEAR REGRESSION 📈                      ║
╠══════════════════════════════════════════════════════════════╣
║  One-line: Finds the best straight line through data         ║
║  Best for: Predicting continuous numbers                     ║
║  Difficulty: ⭐⭐⭐⭐⭐ (Very Easy)                            ║
║  Example: House prices, temperature, exam scores             ║
║  Code: from sklearn.linear_model import LinearRegression     ║
╚══════════════════════════════════════════════════════════════╝

```

╔══════════════════════════════════════════════════════════════╗
║ LOGISTIC REGRESSION 🎯 ║
╠══════════════════════════════════════════════════════════════╣
║ One-line: Predicts probabilities for yes/no decisions ║
║ Best for: Binary classification, spam detection ║
║ Difficulty: ⭐⭐⭐⭐ (Easy) ║
║ Example: Will it rain? Is email spam? ║
║ Code: from sklearn.linear_model import LogisticRegression ║
╚══════════════════════════════════════════════════════════════╝

```
╔══════════════════════════════════════════════════════════════╗
║                      DECISION TREE 🌳                       ║
╠══════════════════════════════════════════════════════════════╣
║  One-line: "If this, then that" decision flowchart          ║
║  Best for: Interpretable rules and decisions                 ║
║  Difficulty: ⭐⭐⭐⭐⭐ (Very Easy)                            ║
║  Example: Loan approval, medical diagnosis                   ║
║  Code: from sklearn.tree import DecisionTreeClassifier       ║
╚══════════════════════════════════════════════════════════════╝

```

╔══════════════════════════════════════════════════════════════╗
║ RANDOM FOREST 🌲🌲 ║
╠══════════════════════════════════════════════════════════════╣
║ One-line: 100 decision trees vote for the best answer ║
║ Best for: High accuracy, robust predictions ║
║ Difficulty: ⭐⭐⭐ (Medium) ║
║ Example: Customer behavior prediction ║
║ Code: from sklearn.ensemble import RandomForestClassifier ║
╚══════════════════════════════════════════════════════════════╝

```
╔══════════════════════════════════════════════════════════════╗
║              SUPPORT VECTOR MACHINE (SVM) 🎯                ║
╠══════════════════════════════════════════════════════════════╣
║  One-line: Finds the widest boundary between groups          ║
║  Best for: Complex classification, high-dimensional data     ║
║  Difficulty: ⭐⭐ (Harder)                                   ║
║  Example: Image recognition, handwritten digits              ║
║  Code: from sklearn.svm import SVC                           ║
╚══════════════════════════════════════════════════════════════╝

```

╔══════════════════════════════════════════════════════════════╗
║ K-NEAREST NEIGHBORS (KNN) 👥 ║
╠══════════════════════════════════════════════════════════════╣
║ One-line: Looks at closest neighbors for decisions ║
║ Best for: Simple pattern recognition, recommendations ║
║ Difficulty: ⭐⭐⭐⭐ (Easy) ║
║ Example: Movie recommendations, similar customer lookup ║
║ Code: from sklearn.neighbors import KNeighborsClassifier ║
╚══════════════════════════════════════════════════════════════╝

```
╔══════════════════════════════════════════════════════════════╗
║                        NAIVE BAYES 📊                       ║
╠══════════════════════════════════════════════════════════════╣
║  One-line: Uses probability and word patterns                ║
║  Best for: Text classification, spam detection               ║
║  Difficulty: ⭐⭐⭐ (Medium)                                  ║
║  Example: Email sorting, sentiment analysis                  ║
║  Code: from sklearn.naive_bayes import GaussianNB           ║
╚══════════════════════════════════════════════════════════════╝

```

╔══════════════════════════════════════════════════════════════╗
║ GRADIENT BOOSTING 🚀 ║
╠══════════════════════════════════════════════════════════════╣
║ One-line: Learns from mistakes and improves each time ║
║ Best for: Maximum accuracy, competitions ║
║ Difficulty: ⭐⭐ (Harder) ║
║ Example: Kaggle competitions, high-stakes predictions ║
║ Code: import xgboost as xgb ║
╚══════════════════════════════════════════════════════════════╝

````

---

## 📊 ALGORITHM PROCESS DIAGRAMS

### How Linear Regression Works

```mermaid
graph TD
    A[Input Data Points] --> B[Plot X vs Y]
    B --> C[Calculate Best Fit Line]
    C --> D[Minimize Sum of Squared Errors]
    D --> E[Equation: Y = mx + b]
    E --> F[Use Line to Predict New Values]

    style A fill:#e3f2fd
    style F fill:#c8e6c9
    style C fill:#fff3e0
````

### How Random Forest Works

```mermaid
graph TD
    A[Training Data] --> B[Create Random Subsets]
    B --> C[Train Multiple Decision Trees]
    C --> D[Each Tree on Different Subset]
    D --> E[Each Tree Makes Prediction]
    E --> F[Voting Process]
    F --> G[Final Answer = Majority Vote]

    style A fill:#e3f2fd
    style G fill:#c8e6c9
    style C fill:#f3e5f5
```

### How Neural Networks Learn

```mermaid
graph LR
    A[Input Layer] --> B[Hidden Layer 1]
    B --> C[Hidden Layer 2]
    C --> D[Output Layer]

    B -.-> E[Weights Adjusted]
    C -.-> F[Weights Adjusted]
    E --> G[Backpropagation]
    F --> G
    G --> B

    style A fill:#e3f2fd
    style D fill:#c8e6c9
    style B fill:#fff3e0
    style C fill:#fff3e0
    style G fill:#ffebee
```

### How Deep Learning Training Works

```mermaid
flowchart TD
    A[Start with Random Weights] --> B[Forward Pass: Calculate Prediction]
    B --> C[Compare with Actual Answer]
    C --> D[Calculate Loss/Error]
    D --> E[Backward Pass: Calculate Gradients]
    E --> F[Update Weights]
    F --> G{Performance Good Enough?}
    G -->|No| B
    G -->|Yes| H[Training Complete]

    style A fill:#e3f2fd
    style H fill:#c8e6c9
    style D fill:#ffebee
```

---

## 📈 PERFORMANCE COMPARISON TABLE

### Algorithm Performance Matrix

| Algorithm               | Speed      | Accuracy   | Memory       | Interpretability | Data Size    | Best Use Case        |
| ----------------------- | ---------- | ---------- | ------------ | ---------------- | ------------ | -------------------- |
| **Linear Regression**   | ⚡⚡⚡⚡⚡ | ⭐⭐⭐     | 🟢 Low       | 🟢 High          | Small-Medium | Simple predictions   |
| **Logistic Regression** | ⚡⚡⚡⚡⚡ | ⭐⭐⭐     | 🟢 Low       | 🟢 High          | Small-Medium | Binary decisions     |
| **Decision Tree**       | ⚡⚡⚡⚡   | ⭐⭐⭐⭐   | 🟡 Medium    | 🟢 High          | Small-Medium | Rule-based decisions |
| **Random Forest**       | ⚡⚡⚡     | ⭐⭐⭐⭐⭐ | 🟡 Medium    | 🟡 Medium        | Medium-Large | General-purpose      |
| **SVM**                 | ⚡⚡       | ⭐⭐⭐⭐   | 🔴 High      | 🔴 Low           | Small-Medium | Complex boundaries   |
| **KNN**                 | ⚡⚡       | ⭐⭐⭐⭐   | 🔴 High      | 🟡 Medium        | Small-Medium | Similarity matching  |
| **Naive Bayes**         | ⚡⚡⚡⚡⚡ | ⭐⭐⭐     | 🟢 Low       | 🟡 Medium        | Medium-Large | Text classification  |
| **XGBoost**             | ⚡⚡       | ⭐⭐⭐⭐⭐ | 🔴 High      | 🔴 Low           | Large        | Competition winning  |
| **Neural Networks**     | ⚡         | ⭐⭐⭐⭐⭐ | 🔴 Very High | 🔴 Very Low      | Large        | Complex patterns     |

### Speed Legend

- ⚡⚡⚡⚡⚡ = Very Fast (< 1 second)
- ⚡⚡⚡⚡ = Fast (1-10 seconds)
- ⚡⚡⚡ = Medium (10 seconds - 1 minute)
- ⚡⚡ = Slow (1-10 minutes)
- ⚡ = Very Slow (10+ minutes)

### Accuracy Legend

- ⭐⭐⭐⭐⭐ = Excellent (>95%)
- ⭐⭐⭐⭐ = Very Good (90-95%)
- ⭐⭐⭐ = Good (80-90%)
- ⭐⭐ = Fair (70-80%)
- ⭐ = Poor (<70%)

---

### 🗺️ Unsupervised Learning Cards

```
╔══════════════════════════════════════════════════════════════╗
║                    K-MEANS CLUSTERING 🎯                    ║
╠══════════════════════════════════════════════════════════════╣
║  One-line: Groups similar data like organizing by color      ║
║  Best for: Customer segmentation, data grouping              ║
║  Difficulty: ⭐⭐⭐ (Medium)                                  ║
║  Example: Netflix movie genres, customer types               ║
║  Code: from sklearn.cluster import KMeans                    ║
╚══════════════════════════════════════════════════════════════╝

```

╔══════════════════════════════════════════════════════════════╗
║ DBSCAN CLUSTERING 🌌 ║
╠══════════════════════════════════════════════════════════════╣
║ One-line: Finds clusters + odd ones out automatically ║
║ Best for: Outlier detection, irregular shapes ║
║ Difficulty: ⭐⭐⭐⭐ (Medium-Hard) ║
║ Example: Fraud detection, noise removal ║
║ Code: from sklearn.cluster import DBSCAN ║
╚══════════════════════════════════════════════════════════════╝

```
╔══════════════════════════════════════════════════════════════╗
║                 HIERARCHICAL CLUSTERING 🏗️                 ║
╠══════════════════════════════════════════════════════════════╣
║  One-line: Builds family tree of relationships              ║
║  Best for: Taxonomies, organizational structures            ║
║  Difficulty: ⭐⭐⭐⭐ (Medium-Hard)                           ║
║  Example: Species classification, product categories         ║
║  Code: from sklearn.cluster import AgglomerativeClustering   ║
╚══════════════════════════════════════════════════════════════╝

```

╔══════════════════════════════════════════════════════════════╗
║ PCA (PRINCIPAL COMPONENT) 📉 ║
╠══════════════════════════════════════════════════════════════╣
║ One-line: Keep only the most important information ║
║ Best for: Data compression, visualization ║
║ Difficulty: ⭐⭐⭐ (Medium) ║
║ Example: Image compression, feature reduction ║
║ Code: from sklearn.decomposition import PCA ║
╚══════════════════════════════════════════════════════════════╝

```
╔══════════════════════════════════════════════════════════════╗
║                         t-SNE 🎨                           ║
╠══════════════════════════════════════════════════════════════╣
║  One-line: Creates beautiful 2D maps of complex data         ║
║  Best for: Visualizing high-dimensional data                 ║
║  Difficulty: ⭐⭐⭐⭐ (Medium-Hard)                           ║
║  Example: Gene data visualization, embedding analysis        ║
║  Code: from sklearn.manifold import TSNE                    ║
╚══════════════════════════════════════════════════════════════╝

```

╔══════════════════════════════════════════════════════════════╗
║ UMAP 🗺️ ║
╠══════════════════════════════════════════════════════════════╣
║ One-line: Like t-SNE but faster for big datasets ║
║ Best for: Large dataset visualization ║
║ Difficulty: ⭐⭐⭐⭐ (Medium-Hard) ║
║ Example: Real-time analysis, big data patterns ║
║ Code: import umap-learn ║
╚══════════════════════════════════════════════════════════════╝

````

### How K-Means Clustering Works

```mermaid
graph TD
    A[Choose K Clusters] --> B[Randomly Place Centroids]
    B --> C[Assign Points to Nearest Centroid]
    C --> D[Recalculate Centroid Positions]
    D --> E{Changed Significantly?}
    E -->|Yes| C
    E -->|No| F[Clustering Complete]

    style A fill:#e3f2fd
    style F fill:#c8e6c9
    style B fill:#fff3e0
    style D fill:#f3e5f5
````

### How PCA Reduces Dimensionality

```mermaid
graph LR
    A[High-Dimensional Data] --> B[Find Principal Components]
    B --> C[Components Capture Max Variance]
    C --> D[Project to Lower Dimensions]
    D --> E[Keep Most Important Features]

    style A fill:#e3f2fd
    style E fill:#c8e6c9
    style B fill:#fff3e0
```

### 🧠 Deep Learning Architecture Cards

```
╔══════════════════════════════════════════════════════════════╗
║              FEEDFORWARD NETWORKS (MLP) 🧠                  ║
╠══════════════════════════════════════════════════════════════╣
║  One-line: Simple layered brain processing forward           ║
║  Best for: Basic patterns, tabular data                      ║
║  Difficulty: ⭐⭐⭐ (Medium)                                  ║
║  Example: Credit scoring, price prediction                   ║
║  Code: torch.nn.Sequential(nn.Linear(), nn.ReLU())          ║
╚══════════════════════════════════════════════════════════════╝

```

╔══════════════════════════════════════════════════════════════╗
║ CONVOLUTIONAL NETWORKS (CNN) 👁️ ║
╠══════════════════════════════════════════════════════════════╣
║ One-line: AI with super-powered eyes for image patterns ║
║ Best for: Image recognition, computer vision ║
║ Difficulty: ⭐⭐⭐⭐ (Medium-Hard) ║
║ Example: Face recognition, medical imaging ║
║ Code: torch.nn.Conv2d(in_channels=3, out_channels=64) ║
╚══════════════════════════════════════════════════════════════╝

```
╔══════════════════════════════════════════════════════════════╗
║              RECURRENT NETWORKS (RNN) 🔄                    ║
╠══════════════════════════════════════════════════════════════╣
║  One-line: Memory-enhanced networks for sequences            ║
║  Best for: Time series, text, speech                         ║
║  Difficulty: ⭐⭐⭐⭐ (Medium-Hard)                           ║
║  Example: Language translation, sentiment analysis           ║
║  Code: torch.nn.LSTM(input_size, hidden_size, num_layers)   ║
╚══════════════════════════════════════════════════════════════╝

```

╔══════════════════════════════════════════════════════════════╗
║ LSTM NETWORKS 💾 ║
╠══════════════════════════════════════════════════════════════╣
║ One-line: Smart memory that knows what to remember ║
║ Best for: Long sequences, complex temporal patterns ║
║ Difficulty: ⭐⭐⭐⭐⭐ (Hard) ║
║ Example: Stock prediction, video analysis ║
║ Code: torch.nn.LSTM(input_size, hidden_size, num_layers) ║
╚══════════════════════════════════════════════════════════════╝

```
╔══════════════════════════════════════════════════════════════╗
║                     TRANSFORMERS ⚡                         ║
╠══════════════════════════════════════════════════════════════╣
║  One-line: Attention mechanism for focus on important parts  ║
║  Best for: Language understanding, complex sequences         ║
║  Difficulty: ⭐⭐⭐⭐⭐ (Very Hard)                            ║
║  Example: ChatGPT, BERT, language models                     ║
║  Code: from transformers import AutoModel                   ║
╚══════════════════════════════════════════════════════════════╝
```

### How CNN Processes Images

```mermaid
graph TD
    A[Input Image] --> B[Convolutional Layer]
    B --> C[Activation Function]
    C --> D[Pooling Layer]
    D --> E[Multiple Conv+Pool Layers]
    E --> F[Flatten]
    F --> G[Fully Connected Layers]
    G --> H[Output Prediction]

    style A fill:#e3f2fd
    style H fill:#c8e6c9
    style B fill:#fff3e0
    style D fill:#f3e5f5
```

### How LSTM Memory Works

```mermaid
graph LR
    A[Current Input] --> B[Forget Gate]
    A --> C[Input Gate]
    A --> D[Output Gate]

    B --> E[Cell State Update]
    C --> E
    E --> F[Hidden State]
    F --> G[Next Time Step]

    style E fill:#fff3e0
    style F fill:#c8e6c9
```

### 📊 DETAILED PERFORMANCE COMPARISON TABLE

| Algorithm                 | Training Speed        | Inference Speed       | Memory Usage | Interpretability | Best For            | GPU Required |
| ------------------------- | --------------------- | --------------------- | ------------ | ---------------- | ------------------- | ------------ |
| **Linear Regression**     | ⚡⚡⚡⚡⚡ Ultra Fast | ⚡⚡⚡⚡⚡ Ultra Fast | 🟢 Very Low  | 🟢 High          | Simple predictions  | ❌ No        |
| **Logistic Regression**   | ⚡⚡⚡⚡⚡ Ultra Fast | ⚡⚡⚡⚡⚡ Ultra Fast | 🟢 Very Low  | 🟢 High          | Binary decisions    | ❌ No        |
| **Decision Trees**        | ⚡⚡⚡⚡ Fast         | ⚡⚡⚡⚡ Fast         | 🟡 Medium    | 🟢 High          | Rule-based logic    | ❌ No        |
| **Random Forest**         | ⚡⚡⚡ Medium         | ⚡⚡⚡ Medium         | 🟡 Medium    | 🟡 Medium        | General purpose     | ❌ No        |
| **SVM**                   | ⚡⚡ Slow             | ⚡⚡⚡ Medium         | 🔴 High      | 🔴 Low           | Complex boundaries  | ❌ No        |
| **KNN**                   | ⚡⚡⚡ Medium         | ⚡⚡ Slow             | 🔴 High      | 🟡 Medium        | Similarity matching | ❌ No        |
| **Naive Bayes**           | ⚡⚡⚡⚡ Fast         | ⚡⚡⚡⚡ Fast         | 🟢 Low       | 🟡 Medium        | Text classification | ❌ No        |
| **XGBoost**               | ⚡⚡ Slow             | ⚡⚡⚡ Medium         | 🔴 High      | 🔴 Low           | Competition winning | ❌ No        |
| **Neural Networks (MLP)** | ⚡ Slow               | ⚡⚡ Slow             | 🟡 Medium    | 🔴 Low           | Complex patterns    | ✅ Optional  |
| **CNNs**                  | ⚡ Very Slow          | ⚡⚡ Slow             | 🔴 High      | 🔴 Low           | Image recognition   | ✅ Yes       |
| **RNNs/LSTMs**            | ⚡ Very Slow          | ⚡⚡ Slow             | 🔴 High      | 🔴 Low           | Sequential data     | ✅ Yes       |
| **Transformers**          | ⚡ Very Slow          | ⚡ Slow               | 🔴 Very High | 🔴 Very Low      | Language models     | ✅ Yes       |

### 🎯 ALGORITHM SUCCESS RATES BY PROBLEM TYPE

| Problem Type               | Best Algorithm      | Success Rate | When to Use             |
| -------------------------- | ------------------- | ------------ | ----------------------- |
| **House Price Prediction** | Random Forest       | 92%          | General real estate     |
| **Spam Email Detection**   | Naive Bayes         | 95%          | Text-based filtering    |
| **Image Classification**   | ResNet/EfficientNet | 97%          | Standard vision tasks   |
| **Customer Segmentation**  | K-Means             | 85%          | Marketing groups        |
| **Stock Price Prediction** | LSTM                | 78%          | Time series analysis    |
| **Sentiment Analysis**     | BERT                | 94%          | Social media, reviews   |
| **Medical Diagnosis**      | Random Forest       | 89%          | Structured medical data |
| **Fraud Detection**        | Isolation Forest    | 88%          | Anomaly detection       |
| **Recommendation Systems** | Neural Networks     | 91%          | Personalized content    |
| **Language Translation**   | Transformers        | 96%          | Multilingual support    |

### 🔧 TROUBLESHOOTING FLOWCHART

```
MODEL NOT WORKING?
│
├─ Check Data Quality
│  ├─ Missing values? → Impute data
│  ├─ Wrong data types? → Convert types
│  └─ Outliers? → Handle outliers
│
├─ Check Model Selection
│  ├─ Simple problem, complex model? → Use simpler model
│  ├─ Complex problem, simple model? → Use complex model
│  └─ Wrong algorithm for data type? → Choose correctly
│
├─ Check Model Parameters
│  ├─ Overfitting? → Add regularization
│  ├─ Underfitting? → Increase model capacity
│  └─ Poor convergence? → Tune learning rate
│
└─ Check Evaluation
   ├─ Wrong metrics? → Use appropriate metrics
   ├─ Data leakage? → Fix train/test split
   └─ Sample size? → Get more data
```

---

## 🏆 ENHANCED ALGORITHM SELECTION MATRIX

### 📊 Visual Algorithm Selection Guide

```
╔══════════════════════════════════════════════════════════════╗
║                    PROBLEM TYPE SELECTOR                     ║
╠══════════════════════════════════════════════════════════════╣
║  🔢 NUMBERS (Regression)         📊 CATEGORIES (Classification)   ║
║  ├─ Linear/Polynomial Curves    ├─ Yes/No Decisions             ║
║  ├─ Continuous Values           ├─ Multiple Groups              ║
║  └─ Trend Prediction            └─ Pattern Recognition          ║
║                                                              ║
║  🔍 HIDDEN PATTERNS (Unsupervised)  🧠 COMPLEX PATTERNS (Deep)  ║
║  ├─ Customer Segmentation       ├─ Images & Videos              ║
║  ├─ Anomaly Detection           ├─ Natural Language             ║
║  └─ Data Compression            └─ Sequential Data              ║
╚══════════════════════════════════════════════════════════════╝
```

### 🎯 Data Size vs Algorithm Selection

| Your Data Size       | Classification                 | Regression               | Clustering         | Deep Learning             |
| -------------------- | ------------------------------ | ------------------------ | ------------------ | ------------------------- |
| **< 1K samples**     | 🟢 **Logistic Regression**     | 🟢 **Linear Regression** | 🟢 **K-Means**     | 🔴 Not Recommended        |
| **1K-10K samples**   | 🟢 **Random Forest**           | 🟢 **Random Forest**     | 🟢 **K-Means**     | 🟡 **Small CNNs**         |
| **10K-100K samples** | 🟡 **XGBoost**                 | 🟡 **XGBoost**           | 🟡 **DBSCAN**      | 🟢 **CNNs/LSTMs**         |
| **100K+ samples**    | 🟢 **XGBoost/Neural Networks** | 🟢 **Neural Networks**   | 🟢 **All Methods** | 🟢 **Large Transformers** |

### ⚡ Speed vs Accuracy Trade-off

```
╔══════════════════════════════════════════════════════════════╗
║                     SPEED vs ACCURACY MAP                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ⚡⚡⚡⚡⚡  Ultra Fast          ⚡⚡⚡⚡  Fast            ⚡⚡⚡  Medium        ║
║  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐ ║
║  │ Linear Reg  │           │ Decision    │           │ Random      │ ║
║  │ Logistic Reg│           │ Tree        │           │ Forest      │ ║
║  │ Naive Bayes │           │ KNN         │           │ SVM         │ ║
║  └─────────────┘           └─────────────┘           └─────────────┘ ║
║  Accuracy: ⭐⭐⭐            Accuracy: ⭐⭐⭐⭐           Accuracy: ⭐⭐⭐⭐   ║
║                                                              ║
║  ⚡⚡  Slow               ⚡  Very Slow                       ║
║  ┌─────────────┐           ┌─────────────┐                 ║
║  │ XGBoost     │           │ Neural      │                 ║
║  │ Deep Nets   │           │ Networks    │                 ║
║  └─────────────┘           └─────────────┘                 ║
║  Accuracy: ⭐⭐⭐⭐⭐         Accuracy: ⭐⭐⭐⭐⭐               ║
╚══════════════════════════════════════════════════════════════╝
```

### 🚀 One-Line Algorithm Explanations

- **Linear Regression**: Draws the best straight line through your data points
- **Logistic Regression**: Predicts probabilities for yes/no decisions
- **Decision Tree**: Asks "if this, then that" questions to make decisions
- **Random Forest**: 100 decision trees vote together for the best answer
- **Support Vector Machine**: Finds the widest possible boundary between groups
- **K-Means**: Groups data into K clusters based on similarity
- **Principal Component Analysis**: Reduces data complexity by keeping only important parts
- **Neural Network**: Simulates brain neurons to learn complex patterns
- **Convolutional Neural Network**: Specialized for understanding images like human vision
- **Recurrent Neural Network**: Remembers previous information to understand sequences
- **Transformer**: Uses attention to focus on what's important in data
- **Gradient Boosting**: Learns from mistakes and improves each time

---

## 🏆 Model Comparison - Which Tool is Best for Your Problem?

### ⚡ Quick Start Commands for Common Tasks

#### **Setup Commands**

```bash
# Install core ML libraries
pip install scikit-learn pandas numpy matplotlib seaborn

# Install deep learning libraries
pip install torch torchvision tensorflow

# Install advanced libraries
pip install xgboost lightgbm transformers

# For computer vision
pip install opencv-python pillow

# For data processing
pip install matplotlib seaborn plotly
```

#### **Training Commands**

```python
# Traditional ML training template
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Deep learning training template
import torch
model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Training loop here...
```

#### **Inference Commands**

```python
# Traditional ML prediction
predictions = model.predict(X_new)

# Deep learning inference
model.eval()
with torch.no_grad():
    outputs = model(input_tensor)
    predictions = torch.argmax(outputs, dim=1)

# Batch prediction
predictions = model.predict_proba(X_batch)
```

### Traditional ML vs Deep Learning - The Big Picture

#### **Traditional Machine Learning** 🎯

- **Like**: A skilled craftsperson with hand tools
- **Best for**: Small to medium datasets (100-10,000 examples)
- **Pros**:
  - ✅ Fast to train (minutes to hours)
  - ✅ Works well with small data
  - ✅ Easy to understand and explain
  - ✅ No special hardware needed
- **Cons**:
  - ❌ Requires manual feature engineering
  - ❌ May plateau on very large datasets

#### **Deep Learning** 🧠

- **Like**: A massive automated factory
- **Best for**: Large datasets (10,000+ examples)
- **Pros**:
  - ✅ Automatically finds patterns
  - ✅ Excellent on very large datasets
  - ✅ State-of-the-art results
- **Cons**:
  - ❌ Needs lots of data and computing power
  - ❌ Hard to understand why it makes decisions
  - ❌ Takes a long time to train

### 🏗️ HARDWARE REQUIREMENTS GUIDE

```
╔══════════════════════════════════════════════════════════════╗
║                   HARDWARE REQUIREMENTS                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🟢 LAPTOP/GPU NOT NEEDED                                    ║
║  ├─ Linear/Logistic Regression                               ║
║  ├─ Decision Trees                                           ║
║  ├─ K-Means Clustering                                       ║
║  └─ Naive Bayes                                              ║
║  Requirements: 4GB RAM, CPU sufficient                      ║
║                                                              ║
║  🟡 MODERATE GPU HELPFUL                                    ║
║  ├─ Random Forest                                            ║
║  ├─ SVM                                                      ║
║  ├─ XGBoost                                                  ║
║  └─ Small Neural Networks                                    ║
║  Requirements: 8GB RAM, GPU optional                        ║
║                                                              ║
║  🔴 GPU STRONGLY RECOMMENDED                                 ║
║  ├─ Convolutional Neural Networks                            ║
║  ├─ Recurrent Neural Networks                                ║
║  ├─ Transformers                                             ║
║  └─ Large Language Models                                    ║
║  Requirements: 16GB+ RAM, NVIDIA GPU (6GB+ VRAM)            ║
╚══════════════════════════════════════════════════════════════╝
```

### 🚀 DECISION FLOWCHART - START HERE

```
START → What type of problem?
│
├─ Numbers/Prices → Regression
│  ├─ Simple relationship → Linear Regression
│  └─ Complex patterns → Random Forest → XGBoost
│
├─ Yes/No/Groups → Classification
│  ├─ Binary → Logistic Regression → Random Forest
│  └─ Multiple → Random Forest → XGBoost
│
├─ Images/Videos → Deep Learning
│  ├─ Basic → Small CNN
│  └─ Advanced → ResNet/EfficientNet
│
├─ Text/Language → Deep Learning
│  ├─ Simple → Naive Bayes/TF-IDF
│  └─ Advanced → BERT/Transformers
│
└─ Find Patterns → Unsupervised
   ├─ Group similar → K-Means
   ├─ Find outliers → Isolation Forest
   └─ Reduce dimensions → PCA
```

### 🖨️ PRINTABLE REFERENCE CARDS

#### Quick Algorithm Cards (Print & Cut)

```
╔═══════════════════════════════════════════╗
║           LINEAR REGRESSION               ║
╠═══════════════════════════════════════════╣
║ Problem: Predict continuous numbers       ║
║ Speed: ⚡⚡⚡⚡⚡ (Very Fast)                ║
║ Accuracy: ⭐⭐⭐ (Good)                     ║
║ Code: from sklearn.linear_model           ║
║ import LinearRegression                   ║
╚═══════════════════════════════════════════╝

╔═══════════════════════════════════════════╗
║          RANDOM FOREST                    ║
╠═══════════════════════════════════════════╣
║ Problem: General classification/regression║
║ Speed: ⚡⚡⚡ (Medium)                      ║
║ Accuracy: ⭐⭐⭐⭐⭐ (Excellent)             ║
║ Code: from sklearn.ensemble import        ║
║ RandomForestClassifier                   ║
╚═══════════════════════════════════════════╝

╔═══════════════════════════════════════════╗
║             XGBOOST                      ║
╠═══════════════════════════════════════════╣
║ Problem: High-accuracy competitions       ║
║ Speed: ⚡⚡ (Slow)                         ║
║ Accuracy: ⭐⭐⭐⭐⭐ (Excellent)             ║
║ Code: import xgboost as xgb              ║
║ model = xgb.XGBClassifier()              ║
╚═══════════════════════════════════════════╝

╔═══════════════════════════════════════════╗
║           CNN (Deep Learning)             ║
╠═══════════════════════════════════════════╣
║ Problem: Image/video recognition          ║
║ Speed: ⚡ (Requires GPU)                  ║
║ Accuracy: ⭐⭐⭐⭐⭐ (State-of-art)          ║
║ Code: import torchvision.models          ║
║ model = models.resnet18(pretrained=True) ║
╚═══════════════════════════════════════════╝

╔═══════════════════════════════════════════╗
║         TRANSFORMERS (BERT)               ║
╠═══════════════════════════════════════════╣
║ Problem: Advanced text understanding      ║
║ Speed: ⚡ (Requires GPU)                  ║
║ Accuracy: ⭐⭐⭐⭐⭐ (State-of-art)          ║
║ Code: from transformers import            ║
║ AutoModel, AutoTokenizer                 ║
╚═══════════════════════════════════════════╝
```

---

## 🖨️ PRINT-READY QUICK REFERENCE SUMMARY

### 📋 Complete Algorithm Cheat Sheet

```
┌─────────────────────────────────────────────────────────────┐
│                    AI ALGORITHM QUICK SELECTOR              │
├─────────────────────────────────────────────────────────────┤
│  STEP 1: What are you trying to predict?                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ NUMBERS     │ │ YES/NO      │ │ FIND GROUPS │           │
│  │ (Regression)│ │(Classification│ │(Clustering) │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│                                                             │
│  STEP 2: How much data do you have?                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ < 1K        │ │ 1K-100K     │ │ 100K+       │           │
│  │ Simple ML   │ │ Ensemble    │ │ Deep Learning│          │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│                                                             │
│  STEP 3: Choose your algorithm:                             │
│  • Linear Reg → Logistic Reg → Random Forest → XGBoost     │
│  • K-Means → DBSCAN → Neural Networks                       │
└─────────────────────────────────────────────────────────────┘
```

### 🎯 INSTANT REFERENCE DECK (Cut and Keep)

```
┌──────────────────┬──────────────────┬──────────────────┐
│ LINEAR REGRESSION │LOGISTIC REGRESSION│  DECISION TREE   │
│ 📊 Predict Numbers│  🎯 Yes/No       │   🌳 If/Then     │
│ ⚡⚡⚡⚡⚡ Speed      │ ⚡⚡⚡⚡⚡ Speed   │ ⚡⚡⚡⚡ Speed     │
│ ⭐⭐⭐ Accuracy     │ ⭐⭐⭐ Accuracy   │ ⭐⭐⭐⭐ Accuracy   │
│ from sklearn     │ from sklearn     │ from sklearn     │
└──────────────────┴──────────────────┴──────────────────┘

┌──────────────────┬──────────────────┬──────────────────┐
│  RANDOM FOREST   │     XGBOOST      │      CNN         │
│ 🌲🌲 Tree Ensemble│ 🚀 Gradient Boost│ 👁️ Image Expert  │
│ ⚡⚡⚡ Speed       │ ⚡⚡ Speed        │ ⚡ Speed (GPU)    │
│ ⭐⭐⭐⭐⭐ Accuracy │ ⭐⭐⭐⭐⭐ Accuracy │⭐⭐⭐⭐⭐ Accuracy  │
│ from sklearn     │ import xgboost   │ torchvision      │
└──────────────────┴──────────────────┴──────────────────┘

┌──────────────────┬──────────────────┬──────────────────┐
│       LSTM       │   TRANSFORMERS   │    K-MEANS       │
│ 💾 Memory Expert │ ⚡ Attention     │ 🎯 Group Similar │
│ ⚡ Speed (GPU)    │ ⚡ Speed (GPU)    │ ⚡⚡⚡ Speed       │
│ ⭐⭐⭐⭐ Accuracy  │ ⭐⭐⭐⭐⭐ Accuracy │ ⭐⭐⭐ Accuracy    │
│ torch.nn.LSTM    │ transformers     │ from sklearn     │
└──────────────────┴──────────────────┴──────────────────┘
```

### 🏆 PERFORMANCE BENCHMARKS

| Use Case                  | Recommended Algorithm | Typical Accuracy | Training Time | Memory  |
| ------------------------- | --------------------- | ---------------- | ------------- | ------- |
| **Beginner Project**      | Logistic Regression   | 85%              | < 1 min       | < 100MB |
| **Quick Prototyping**     | Random Forest         | 92%              | 1-10 min      | < 500MB |
| **Competition Ready**     | XGBoost               | 95%              | 10-60 min     | < 2GB   |
| **Image Recognition**     | ResNet50              | 97%              | 1-4 hours     | < 1GB   |
| **Text Analysis**         | BERT                  | 94%              | 2-8 hours     | < 3GB   |
| **Time Series**           | LSTM                  | 88%              | 30min-2hrs    | < 1GB   |
| **Customer Segmentation** | K-Means               | 85%              | < 5 min       | < 200MB |
| **Anomaly Detection**     | Isolation Forest      | 88%              | < 1 min       | < 100MB |

### ⚡ SPEED OPTIMIZATION CHECKLIST

```
┌─────────────────────────────────────────────────────────────┐
│                   SPEED UP YOUR MODELS                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔧 DATA OPTIMIZATION                                        │
│  ├─ Use efficient data types (int8 vs float64)             │
│  ├─ Remove unnecessary features                            │
│  ├─ Use sparse matrices for high sparsity                  │
│  └─ Parallelize data preprocessing                         │
│                                                             │
│  🚀 MODEL OPTIMIZATION                                      │
│  ├─ Start with simple models (benchmark)                   │
│  ├─ Use appropriate sampling for large datasets            │
│  ├─ Enable multi-threading (n_jobs=-1)                     │
│  └─ Use GPU acceleration when available                    │
│                                                             │
│  💾 MEMORY OPTIMIZATION                                     │
│  ├─ Process data in batches                                │
│  ├─ Use generators for large files                         │
│  ├─ Clear unnecessary variables                            │
│  └─ Monitor memory usage during training                   │
│                                                             │
│  ⚙️  SYSTEM OPTIMIZATION                                     │
│  ├─ Use SSD storage for faster I/O                         │
│  ├─ Increase system RAM if possible                        │
│  ├─ Close unnecessary applications                         │
│  └─ Use cloud computing for large projects                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 VISUAL CODE TEMPLATES - QUICK START

### 🚀 Step-by-Step: Your First AI Project

#### **Template 1: Basic AI Pipeline (Like Following a Recipe)**

**What this does**: Takes your data and trains an AI to make predictions
**When to use**: For any prediction problem (prices, categories, etc.)

```python
# Step 1: Get your tools ready (like gathering ingredients)
import pandas as pd              # For working with data
import numpy as np               # For math calculations
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For cleaning data
from sklearn.ensemble import RandomForestClassifier  # Your AI model

# Step 2: Load your data (like getting your ingredients)
df = pd.read_csv('your_data.csv')  # Load your data file
print("Data shape:", df.shape)     # Show how much data you have
print(df.head())                   # Look at first few rows

# Step 3: Separate what you want to predict from everything else
X = df.drop('target_column', axis=1)  # Everything except the answer
y = df['target_column']               # What you want to predict

# Step 4: Clean up missing information (like removing bad ingredients)
# For numbers: fill with average
X = X.fillna(X.mean())
# For text: fill with most common
X = X.fillna(X.mode().iloc[0])

# Step 5: Convert text to numbers (AI likes numbers)
le = LabelEncoder()
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Step 6: Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Train your AI (like cooking your meal)
model = RandomForestClassifier()  # Create your AI
model.fit(X_train, y_train)       # Train it with your data

# Step 8: Test how good your AI is
accuracy = model.score(X_test, y_test)
print(f"Your AI is {accuracy:.1%} accurate!")

# Step 9: Make predictions on new data
predictions = model.predict(new_data)
print("Predictions:", predictions)
```

**💡 What each step means:**

- **Step 1**: Get the necessary tools (libraries)
- **Step 2**: Load your data from a file
- **Step 3**: Separate the question from the answers
- **Step 4**: Handle missing or incomplete data
- **Step 5**: Convert text to numbers (AI speaks numbers)
- **Step 6**: Split data for training and testing
- **Step 7**: Train the AI model
- **Step 8**: Check how accurate your AI is
- **Step 9**: Use your AI to make new predictions

#### **Template 2: Quick Classification (For Yes/No or Categories)**

**What this does**: Sorts things into groups (like spam/not spam)
**When to use**: Email filtering, image classification, customer types

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Create and train the classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# See detailed results
print("Classification Report:")
print(classification_report(y_test, y_pred))

# See confusion matrix (what did it get right/wrong?)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

#### **Template 3: Regression (For Predicting Numbers)**

**What this does**: Predicts continuous values (like prices, temperatures)
**When to use**: House prices, stock prices, exam scores

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train the regressor
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Calculate error
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f} (closer to 1.0 is better)")
```

# Scale features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

````

#### Image Data Pipeline
```python
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder('data/', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# For training with custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data.iloc[idx]['image_path'])
        label = self.data.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label
````

#### Text Data Pipeline

```python
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    tokens = [stemmer.stem(token) for token in tokens
              if token not in stop_words and token.isalpha()]

    return ' '.join(tokens)

# Using pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts, max_length=512):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

# Example usage
texts = ["This is a sample text.", "Another example."]
tokens = tokenize_texts(texts)
```

### Model Training Templates

#### Traditional ML Training

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Train model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))
```

#### Deep Learning Training (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

# Define model
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 56 * 56, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Initialize model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average loss: {epoch_loss:.4f}')

# Train model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
```

#### Transfer Learning (PyTorch)

```python
import torchvision.models as models

# Use pre-trained model
model = models.resnet18(pretrained=True)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Train only the final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Fine-tuning approach (unfreeze some layers)
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {'params': model.fc.parameters()},
    {'params': model.layer4.parameters()}
], lr=0.0001)
```

### Model Evaluation Templates

#### Classification Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def evaluate_classification(y_true, y_pred, y_prob=None):
    """Comprehensive classification evaluation"""

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # ROC-AUC if probabilities available
    if y_prob is not None:
        if len(np.unique(y_true)) == 2:  # Binary classification
            auc_score = roc_auc_score(y_true, y_prob)
            print(f"ROC-AUC: {auc_score:.4f}")

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

#### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_regression(y_true, y_pred):
    """Comprehensive regression evaluation"""

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Calculate percentage errors
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
```

---

## Library Quick References

### Scikit-learn Quick Reference

#### Most Used Classes and Functions

```python
# Import essentials
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN

# Model Selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

# Grid Search
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

#### Common Preprocessing Steps

```python
# Standardization (for normal distribution)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalization (for 0-1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encoding
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X_categorical).toarray()

# Label encoding
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

### TensorFlow/Keras Quick Reference

#### Model Building

```python
import tensorflow as tf
from tensorflow import keras

# Sequential model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Functional API
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(128, activation='relu')(inputs)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

#### Training and Callbacks

```python
# Training
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5', save_best_only=True
)
```

### PyTorch Quick Reference

#### Model Definition

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(input_size=784, hidden_size=128, num_classes=10).to(device)
```

#### Training Loop

```python
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
```

### Hugging Face Transformers Quick Reference

#### Model Loading and Usage

```python
from transformers import AutoTokenizer, AutoModel, pipeline

# Load pre-trained model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Text classification pipeline
classifier = pipeline("text-classification",
                     model="distilbert-base-uncased-finetuned-sst-2-english")

# Sentiment analysis
result = classifier("I love this movie!")

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)

# Named Entity Recognition
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
result = ner("John Doe works at Google in New York")

# Fill mask
fill_mask = pipeline("fill-mask", model="bert-base-uncased")
result = fill_mask("The [MASK] is very happy today.")
```

#### Tokenization and Encoding

```python
# Tokenize text
texts = ["Hello world!", "How are you?"]
encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Access tokens
tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
print(tokens)

# Decode back to text
decoded = tokenizer.decode(encoded["input_ids"][0])
print(decoded)
```

---

## Data Preprocessing Cheatsheet

### Numerical Data Preprocessing

#### Missing Value Handling

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# Check missing values
missing_info = df.isnull().sum()
print("Missing values per column:")
print(missing_info[missing_info > 0])

# Remove rows with missing values
df_clean = df.dropna()

# Fill with mean/median/mode
df['column'] = df['column'].fillna(df['column'].median())

# Advanced imputation
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

#### Outlier Detection and Treatment

```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Visual inspection
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, y='column')
plt.title('Box Plot for Outlier Detection')

plt.subplot(1, 2, 2)
plt.hist(df['column'], bins=30)
plt.title('Histogram for Distribution')
plt.show()

# Z-score method
z_scores = np.abs(stats.zscore(df['column']))
outliers = df[z_scores > 3]
print(f"Number of outliers: {len(outliers)}")

# IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['column'] < lower_bound) | (df['column'] > upper_bound)]

# Remove outliers
df_clean = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]

# Cap outliers
df['column'] = df['column'].clip(lower=lower_bound, upper=upper_bound)
```

#### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard Scaler (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Min-Max Scaler (0-1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Robust Scaler (median and IQR, good for outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Power transformation (for skewed distributions)
from sklearn.preprocessing import PowerTransformer
transformer = PowerTransformer(method='yeo-johnson')
X_transformed = transformer.fit_transform(X)
```

### Categorical Data Preprocessing

#### Encoding Methods

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoding (ordinal categories)
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# One-Hot Encoding (nominal categories)
pd_dummies = pd.get_dummies(df['category'], prefix='cat')

# Or using sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['category']])
feature_names = encoder.get_feature_names_out(['category'])
df_encoded = pd.DataFrame(encoded, columns=feature_names)

# Target Encoding (for high cardinality)
target_mean = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(target_mean)
```

#### Text Data Preprocessing

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """Basic text cleaning"""
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

def advanced_text_preprocessing(text):
    """Advanced text preprocessing"""
    # Clean text
    text = clean_text(text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)

# Apply preprocessing
df['text_clean'] = df['text_column'].apply(clean_text)
df['text_advanced'] = df['text_column'].apply(advanced_text_preprocessing)
```

### Image Data Preprocessing

#### Basic Image Operations

```python
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load image
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize
resized = cv2.resize(image, (224, 224))

# Normalize (for neural networks)
normalized = image.astype(np.float32) / 255.0

# Data augmentation for training
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# For inference (no augmentation)
transform_inference = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

## Model Evaluation & Metrics

### Classification Metrics

#### Quick Metric Calculation

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)

# Calculate all metrics at once
def get_all_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

    if y_prob is not None:
        if len(np.unique(y_true)) == 2:  # Binary classification
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)

    return metrics

# Usage
metrics = get_all_metrics(y_test, y_pred, y_prob)
print(metrics)

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
```

#### Class Imbalance Handling

```python
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Use weights in training
model = RandomForestClassifier(class_weight=class_weight_dict)
model.fit(X_train, y_train)

# Oversampling with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train, y_train)
```

### Regression Metrics

#### Comprehensive Evaluation

```python
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                           r2_score, explained_variance_score)

def evaluate_regression_model(y_true, y_pred):
    """Complete regression model evaluation"""

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    explained_var = explained_variance_score(y_true, y_pred)

    # Calculate percentage errors
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    # Create results dictionary
    results = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Explained Variance': explained_var,
        'MAPE (%)': mape,
        'SMAPE (%)': smape
    }

    # Print results
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return results

# Visual evaluation
import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))

    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')

    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    plt.tight_layout()
    plt.show()
```

### Cross-Validation Strategies

#### K-Fold Cross Validation

```python
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

# Basic k-fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Stratified k-fold (for classification)
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring='f1_weighted')

# Custom scoring
from sklearn.metrics import make_scorer

def custom_scorer(y_true, y_pred):
    """Custom scoring function"""
    return np.mean((y_true - y_pred) ** 2)  # MSE

custom_score = make_scorer(custom_scorer, greater_is_better=False)
scores = cross_val_score(model, X, y, cv=kfold, scoring=custom_score)
```

---

## Hyperparameter Tuning Guide

### Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all processors
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'learning_rate': uniform(0.01, 0.3)
}

# Random search
random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
```

### Bayesian Optimization (Optuna)

```python
import optuna
from sklearn.ensemble import RandomForestClassifier

def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Create and train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    # Cross-validation score
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return score.mean()

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Best parameters
print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value}")
```

---

## Deep Learning Architecture Reference

### CNN Architectures

#### ResNet Building Blocks

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Create ResNet-18
def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])
```

### Transformer Architecture

#### Basic Transformer Block

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformations and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.W_o(context)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x
```

### RNN/LSTM Architectures

#### LSTM Implementation

```python
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.W_i = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden

        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)

        # Input gate
        i = torch.sigmoid(self.W_i(combined))
        # Forget gate
        f = torch.sigmoid(self.W_f(combined))
        # Output gate
        o = torch.sigmoid(self.W_o(combined))
        # Cell candidate
        c_tilde = torch.tanh(self.W_c(combined))

        # Update cell state
        c = f * c_prev + i * c_tilde
        # Update hidden state
        h = o * torch.tanh(c)

        return h, c

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList([LSTMCell(input_size, hidden_size, bias)])
        for _ in range(num_layers - 1):
            self.cells.append(LSTMCell(hidden_size, hidden_size, bias))

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            hidden = (h0, c0)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = hidden[0][layer_idx], hidden[1][layer_idx]
                h_t, c_t = cell(x_t, (h_prev, c_prev))
                x_t = h_t
                hidden[0][layer_idx] = h_t
                hidden[1][layer_idx] = c_t

            outputs.append(x_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), hidden
```

---

## This comprehensive cheat sheet provides quick reference materials for all aspects of AI development, from basic algorithms to advanced deep learning architectures. Each section contains practical code templates and essential information for rapid development and deployment.

## Computer Vision Quick Reference

### Pre-trained Models Overview

| Model                  | Input Size | Parameters | FLOPs | Top-1 Accuracy | Use Case                 |
| ---------------------- | ---------- | ---------- | ----- | -------------- | ------------------------ |
| **MobileNet V2**       | 224x224    | 3.5M       | 300M  | 72.0%          | Mobile/Edge deployment   |
| **EfficientNet B0**    | 224x224    | 5.3M       | 390M  | 77.1%          | Good accuracy/efficiency |
| **ResNet-50**          | 224x224    | 25.6M      | 4.1B  | 76.2%          | Standard baseline        |
| **ResNet-101**         | 224x224    | 44.5M      | 7.8B  | 77.4%          | Higher accuracy          |
| **EfficientNet B7**    | 600x600    | 66.3M      | 37B   | 84.4%          | Best accuracy            |
| **Vision Transformer** | 384x384    | 86M        | 17B   | 88.5%          | State-of-the-art         |
| **ConvNeXt Large**     | 224x224    | 198M       | 34B   | 87.8%          | Modern CNN alternative   |

### Object Detection Models

| Model             | Input Size | Speed (FPS) | mAP@0.5 | Model Size | Real-time |
| ----------------- | ---------- | ----------- | ------- | ---------- | --------- |
| **YOLOv5s**       | 640x640    | 45          | 37.4    | 14MB       | ✅        |
| **YOLOv5m**       | 640x640    | 40          | 45.4    | 42MB       | ✅        |
| **YOLOv5l**       | 640x640    | 25          | 49.0    | 93MB       | ✅        |
| **YOLOv5x**       | 640x640    | 20          | 50.7    | 172MB      | ❌        |
| **Faster R-CNN**  | 800x600    | 7           | 42.1    | 160MB      | ❌        |
| **SSD MobileNet** | 300x300    | 27          | 23.2    | 27MB       | ✅        |
| **DETR**          | 800x600    | 6           | 42.0    | 41MB       | ❌        |

### Computer Vision Pipeline Templates

#### Image Classification Pipeline

```python
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np

# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Classification function
def classify_image(image_path, model, transform, class_names=None):
    """Classify a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Get top predictions
    top_prob, top_class = torch.topk(probabilities, k=5)

    results = []
    for i in range(top_prob.size(1)):
        prob = top_prob[0][i].item()
        class_idx = top_class[0][i].item()

        if class_names:
            class_name = class_names[class_idx]
        else:
            class_name = f"Class {class_idx}"

        results.append({
            'class': class_name,
            'confidence': prob
        })

    return results

# Batch processing
def batch_classify(image_paths, model, transform, batch_size=32):
    """Process multiple images"""
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []

        for path in batch_paths:
            image = Image.open(path).convert('RGB')
            image_tensor = transform(image)
            batch_images.append(image_tensor)

        # Stack images
        batch_tensor = torch.stack(batch_images)

        # Batch inference
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Process results
        for j, prob_dist in enumerate(probabilities):
            top_prob, top_class = torch.topk(prob_dist, k=3)
            results.append({
                'image_path': batch_paths[j],
                'predictions': [
                    {'class_idx': top_class[k].item(), 'confidence': top_prob[k].item()}
                    for k in range(3)
                ]
            })

    return results
```

#### Object Detection Pipeline

```python
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import yolo_model
import cv2
import numpy as np

# Initialize YOLO model
def load_yolo_model(model_size='yolov5s', device='cuda'):
    """Load YOLO model for object detection"""
    if model_size == 'yolov5s':
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    elif model_size == 'yolov5m':
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
    elif model_size == 'yolov5l':
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

    model.to(device)
    model.eval()
    return model

# Object detection function
def detect_objects(image_path, model, conf_threshold=0.4, iou_threshold=0.5):
    """Detect objects in image"""
    results = model(image_path)

    # Extract detections
    detections = []
    for detection in results.pred[0]:
        x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()

        if confidence >= conf_threshold:
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': int(class_id),
                'class_name': results.names[int(class_id)]
            })

    return detections

# Real-time detection
def real_time_detection(camera_id=0, model=None, conf_threshold=0.4):
    """Real-time object detection from camera"""
    cap = cv2.VideoCapture(camera_id)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        results = model(frame_rgb)

        # Draw detections on frame
        for detection in results.pred[0]:
            x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()

            if confidence >= conf_threshold:
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Draw label
                label = f"{results.names[int(class_id)]} {confidence:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('Object Detection', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Batch detection for video
def process_video(input_path, output_path, model, conf_threshold=0.4):
    """Process video and save detections"""
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        results = model(frame_rgb)

        # Draw detections on frame
        for detection in results.pred[0]:
            x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()

            if confidence >= conf_threshold:
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Draw label
                label = f"{results.names[int(class_id)]} {confidence:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame
        out.write(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
```

### Image Processing Operations

#### Common Image Operations

```python
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

def resize_image(image, width, height, interpolation=cv2.INTER_LINEAR):
    """Resize image with specified interpolation"""
    return cv2.resize(image, (width, height), interpolation=interpolation)

def rotate_image(image, angle, center=None, scale=1.0):
    """Rotate image by angle degrees"""
    if center is None:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Apply rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated

def enhance_image(image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0):
    """Enhance image properties"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Brightness and contrast
    pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)

    # Saturation
    pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

    # Sharpness
    pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)

    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def apply_filter(image, filter_type='gaussian', kernel_size=5):
    """Apply various filters to image"""
    if filter_type == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif filter_type == 'sharpen':
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == 'edge_detect':
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    else:
        return image

def edge_detection(image, method='canny', low_threshold=50, high_threshold=150):
    """Detect edges in image"""
    if method == 'canny':
        edges = cv2.Canny(image, low_threshold, high_threshold)
        return edges
    elif method == 'sobel':
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        return np.uint8(sobel / sobel.max() * 255)
    elif method == 'laplacian':
        return cv2.Laplacian(image, cv2.CV_64F)

def feature_detection(image, method='sift', max_features=500):
    """Detect keypoints and compute descriptors"""
    if method == 'sift':
        detector = cv2.SIFT_create(max_features)
    elif method == 'surf':
        detector = cv2.xfeatures2d.SURF_create(max_features)
    elif method == 'orb':
        detector = cv2.ORB_create(max_features)
    elif method == 'akaze':
        detector = cv2.AKAZE_create()

    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors
```

---

## NLP Processing Pipeline

### Text Preprocessing Pipeline

#### Complete Text Preprocessing

```python
import re
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Download spaCy model: python -m spacy download en_core_web_sm")

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, lemmatize=True, remove_punctuation=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_punctuation = remove_punctuation

        # Initialize components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def tokenize_text(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)

    def remove_punctuation_tokens(self, tokens):
        """Remove punctuation from tokens"""
        return [token for token in tokens if token.isalnum()]

    def remove_stopwords_tokens(self, tokens):
        """Remove stopwords from tokens"""
        return [token for token in tokens if token not in self.stop_words]

    def stem_tokens(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_tokens(self, tokens):
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess_text(self, text, return_tokens=False):
        """Complete preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize_text(text)

        # Remove punctuation if specified
        if self.remove_punctuation:
            tokens = self.remove_punctuation_tokens(tokens)

        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = self.remove_stopwords_tokens(tokens)

        # Lemmatize if specified
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)

        # Return tokens or joined text
        if return_tokens:
            return tokens
        else:
            return ' '.join(tokens)

# Advanced preprocessing with spaCy
class AdvancedTextPreprocessor:
    def __init__(self):
        self.nlp = nlp

    def extract_entities(self, text):
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def extract_pos_tags(self, text):
        """Extract part-of-speech tags"""
        doc = self.nlp(text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        return pos_tags

    def extract_dependencies(self, text):
        """Extract dependency parsing"""
        doc = self.nlp(text)
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        return dependencies

    def process_document(self, text):
        """Complete spaCy processing"""
        doc = self.nlp(text)

        processed = {
            'tokens': [token.text for token in doc],
            'lemmas': [token.lemma_ for token in doc],
            'pos_tags': [(token.text, token.pos_) for token in doc],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'dependencies': [(token.text, token.dep_, token.head.text) for token in doc],
            'sentences': [sent.text for sent in doc.sents]
        }

        return processed

# Feature extraction
class TextFeatureExtractor:
    def __init__(self, method='tfidf', max_features=10000, ngram_range=(1, 1)):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range

        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                lowercase=True,
                stop_words='english'
            )
        elif method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                lowercase=True,
                stop_words='english'
            )

    def fit_transform(self, texts):
        """Fit vectorizer and transform texts"""
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """Transform new texts using fitted vectorizer"""
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()

# Usage example
def preprocess_text_dataset(texts, labels=None, test_size=0.2):
    """Complete text preprocessing pipeline"""
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lemmatize=True,
        remove_punctuation=True
    )

    # Process texts
    processed_texts = [preprocessor.preprocess_text(text) for text in texts]

    # Extract features
    feature_extractor = TextFeatureExtractor(method='tfidf', max_features=5000)
    X = feature_extractor.fit_transform(processed_texts)

    # Split data if labels provided
    if labels is not None:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42, stratify=labels
        )
        return X_train, X_test, y_train, y_test, feature_extractor

    return X, feature_extractor
```

### Transformer-based Text Processing

#### BERT and RoBERTa Implementation

```python
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np

class TransformerTextProcessor:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def encode_text(self, texts, max_length=512, padding=True, truncation=True):
        """Encode texts using tokenizer"""
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors='pt',
            return_attention_mask=True
        )
        return encoded

    def get_embeddings(self, texts, max_length=512):
        """Get text embeddings"""
        encoded = self.encode_text(texts, max_length)

        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    def get_sentence_embeddings(self, texts, max_length=512):
        """Get sentence-level embeddings using mean pooling"""
        encoded = self.encode_text(texts, max_length)

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            embeddings = embeddings.cpu().numpy()

        return embeddings

# Pre-trained pipelines
def create_classification_pipeline(model_name, task='sentiment-analysis'):
    """Create ready-to-use classification pipeline"""
    if task == 'sentiment-analysis':
        classifier = pipeline(task, model=model_name, return_all_scores=True)

        def classify_text(text):
            results = classifier(text)
            scores = [result['score'] for result in results[0]]
            labels = [result['label'] for result in results[0]]
            return labels[np.argmax(scores)], max(scores)

        return classify_text

    elif task == 'text-generation':
        generator = pipeline(task, model=model_name, max_length=100)
        return generator

    elif task == 'question-answering':
        qa_pipeline = pipeline(task, model=model_name)
        return qa_pipeline

    elif task == 'named-entity-recognition':
        ner_pipeline = pipeline(task, model=model_name)
        return ner_pipeline

# Fine-tuning template
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def fine_tune_model(model_name, train_dataset, eval_dataset, num_labels, output_dir):
    """Fine-tune transformer model for text classification"""

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy'
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: {
            'accuracy': np.mean(eval_pred.predictions.argmax(axis=1) == eval_pred.label_ids)
        }
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model()

    return trainer
```

### Text Generation Templates

#### GPT-style Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class TextGenerator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_text(self, prompt, max_length=100, num_return_sequences=1,
                     temperature=1.0, top_k=50, top_p=0.95):
        """Generate text from prompt"""

        # Encode prompt
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length + len(inputs[0]),
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                no_repeat_ngram_size=2
            )

        # Decode generated text
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt from the generated text
            generated_text = text[len(prompt):].strip()
            generated_texts.append(generated_text)

        return generated_texts

    def generate_multiple_sequences(self, prompt, num_sequences=5, max_length=100):
        """Generate multiple diverse sequences"""
        sequences = []
        for i in range(num_sequences):
            # Vary generation parameters for diversity
            temp = 0.7 + (i * 0.1)  # Vary temperature
            generated = self.generate_text(
                prompt,
                max_length=max_length,
                temperature=temp,
                num_return_sequences=1
            )
            sequences.extend(generated)

        return sequences

    def complete_text(self, text, max_additional_length=100):
        """Complete partially written text"""
        # Find a good stopping point (end of sentence)
        if '.' in text:
            last_period = text.rfind('.')
            if last_period < len(text) - 10:  # Ensure enough context before the period
                prompt = text[:last_period + 1]
            else:
                prompt = text
        else:
            prompt = text

        generated = self.generate_text(prompt, max_length=len(prompt) + max_additional_length)
        return generated[0] if generated else ""

# Advanced text generation with fine-tuning
def train_text_generator(train_texts, model_name='gpt2', output_dir='./fine_tuned_model'):
    """Fine-tune GPT model on custom text data"""

    from transformers import TextDataset, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Add pad token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    # Create dataset
    def load_dataset(file_path, tokenizer, block_size=128):
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        tokenized = tokenizer.encode(text)
        examples = []

        for i in range(0, len(tokenized) - block_size + 1, block_size):
            examples.append(tokenized[i:i + block_size])

        return examples

    train_dataset = load_dataset(train_texts, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        warmup_steps=500,
        fp16=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    # Train
    trainer.train()
    trainer.save_model()

    return trainer, tokenizer
```

---

## Deployment Checklists

### Model Deployment Checklist

#### Pre-Deployment Checklist

```markdown
# Model Deployment Checklist

## Model Validation

- [ ] Model achieves required performance metrics
- [ ] Model passes all test cases
- [ ] Model handles edge cases properly
- [ ] No data leakage in training pipeline
- [ ] Model interpretability requirements met
- [ ] Bias and fairness testing completed
- [ ] Model robustness testing passed

## Code Quality

- [ ] Code follows style guidelines (PEP 8)
- [ ] All functions have proper documentation
- [ ] Unit tests written and passing
- [ ] Integration tests completed
- [ ] Error handling implemented
- [ ] Logging properly configured
- [ ] Security review completed

## Performance Requirements

- [ ] Inference time meets SLA requirements
- [ ] Memory usage within limits
- [ ] CPU/GPU utilization optimized
- [ ] Model size acceptable for deployment
- [ ] Batch processing capabilities tested
- [ ] Concurrent request handling validated

## Data Pipeline

- [ ] Input data validation implemented
- [ ] Feature engineering pipeline tested
- [ ] Data preprocessing production-ready
- [ ] Error handling for invalid inputs
- [ ] Data quality monitoring in place

## Monitoring Setup

- [ ] Performance metrics tracking configured
- [ ] Model drift detection implemented
- [ ] Data drift monitoring setup
- [ ] Alerting system configured
- [ ] Dashboard for model monitoring
- [ ] Error logging and tracking

## Compliance and Security

- [ ] Data privacy regulations compliance
- [ ] Model versioning implemented
- [ ] Access control configured
- [ ] Audit logging enabled
- [ ] API rate limiting implemented
- [ ] Input sanitization verified

## Deployment Infrastructure

- [ ] Production environment ready
- [ ] Load balancing configured
- [ ] Auto-scaling policies defined
- [ ] Backup and recovery procedures
- [ ] Disaster recovery plan
- [ ] Rollback procedures documented

## Documentation

- [ ] API documentation complete
- [ ] Model documentation updated
- [ ] Deployment runbook created
- [ ] Monitoring guide provided
- [ ] Troubleshooting guide available
- [ ] Business stakeholders informed
```

#### Production Deployment Template

```python
import logging
import time
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import joblib
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions')
ERROR_COUNT = Counter('errors_total', 'Total errors', ['error_type'])
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')

# Pydantic models for API
class PredictionRequest(BaseModel):
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    prediction: Any
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: float
    timestamp: str

class ModelMetadata(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    features: List[str]
    last_trained: str
    performance_metrics: Dict[str, float]

class ProductionModel:
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.model_metadata = None
        self.feature_names = None
        self.load_model()

    def load_model(self):
        """Load model and metadata"""
        try:
            # Load model
            if self.model_path.endswith('.joblib'):
                self.model = joblib.load(self.model_path)
            elif self.model_path.endswith('.pkl'):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            elif self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
                self.model = torch.load(self.model_path, map_location='cpu')
                self.model.eval()

            # Load metadata
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.model_metadata = ModelMetadata(**config)
                self.feature_names = config.get('features', [])

            logger.info(f"Model {self.model_metadata.model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and preprocess input data"""
        try:
            # Check required features
            missing_features = set(self.feature_names) - set(data.keys())
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            # Convert to DataFrame for consistent preprocessing
            df = pd.DataFrame([data])

            # Add any data preprocessing here
            # Example: handle missing values, encode categoricals, etc.

            return df

        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise ValueError(f"Invalid input: {str(e)}")

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction on input data"""
        try:
            start_time = time.time()

            # Validate and preprocess
            processed_data = self.validate_input(data)

            # Make prediction
            if isinstance(self.model, torch.nn.Module):
                with torch.no_grad():
                    # Convert to tensor if using PyTorch
                    input_tensor = torch.FloatTensor(processed_data.values).unsqueeze(0)
                    prediction = self.model(input_tensor)

                    # Convert back to numpy
                    prediction = prediction.numpy().flatten()
                    confidence = float(np.max(prediction)) if prediction.shape[0] > 1 else 1.0

            else:
                # For scikit-learn models
                prediction = self.model.predict(processed_data)
                confidence = getattr(self.model, 'predict_proba', lambda x: np.array([[1.0]]))(processed_data)

                if hasattr(confidence, 'max'):
                    confidence = float(confidence.max())
                else:
                    confidence = 1.0

            # Convert numpy types to Python types
            if hasattr(prediction, 'item'):
                prediction = prediction.item()

            processing_time = time.time() - start_time

            return {
                'prediction': prediction,
                'confidence': confidence,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            logger.error(traceback.format_exc())
            ERROR_COUNT.labels(error_type='prediction_error').inc()
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata"""
        return self.model_metadata

# Initialize FastAPI app
app = FastAPI(
    title="AI Model API",
    description="Production ML Model API",
    version="1.0.0"
)

# Global model instance
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    try:
        model = ProductionModel(
            model_path="./models/production_model.joblib",
            config_path="./models/model_metadata.json"
        )
        logger.info("Model loaded successfully on startup")

        # Start Prometheus metrics server
        start_http_server(8000)
        logger.info("Prometheus metrics server started on port 8000")

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

@app.get("/metadata", response_model=ModelMetadata)
async def get_model_metadata():
    """Get model metadata"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return model.get_metadata()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction endpoint"""
    start_time = time.time()
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()

    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Make prediction
        result = model.predict(request.data)

        # Create response
        response = PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            metadata=request.metadata,
            processing_time=result['processing_time'],
            timestamp=result['timestamp']
        )

        # Record metrics
        PREDICTION_COUNT.inc()
        REQUEST_LATENCY.observe(time.time() - start_time)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        ERROR_COUNT.labels(error_type='endpoint_error').inc()
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict_batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    start_time = time.time()
    REQUEST_COUNT.labels(method="POST", endpoint="/predict_batch").inc()

    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        results = []
        for request in requests:
            result = model.predict(request.data)
            results.append({
                'request_id': id(request),
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'processing_time': result['processing_time']
            })

        # Record metrics
        REQUEST_LATENCY.observe(time.time() - start_time)

        return {
            'results': results,
            'total_processing_time': time.time() - start_time,
            'average_time_per_request': (time.time() - start_time) / len(requests)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        ERROR_COUNT.labels(error_type='batch_error').inc()
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(
        "model_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=4,
        access_log=True
    )
```

#### Docker Deployment

```dockerfile
# Dockerfile for model deployment
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "model_server.py"]
```

```yaml
# docker-compose.yml for complete deployment
version: "3.8"

services:
  model-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/production_model.joblib
      - CONFIG_PATH=/app/models/model_metadata.json
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - model-api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--web.console.libraries=/etc/prometheus/console_libraries"
      - "--web.console.templates=/etc/prometheus/consoles"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
```

---

## Debugging & Troubleshooting Guide

### Common Error Solutions

#### Memory Errors

```python
import gc
import psutil
import torch

def diagnose_memory_usage():
    """Diagnose memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()

    print(f"RSS Memory: {memory_info.rss / 1024**3:.2f} GB")
    print(f"VMS Memory: {memory_info.vms / 1024**3:.2f} GB")

    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

def fix_memory_errors():
    """Fix common memory errors"""
    # Clear Python garbage collection
    gc.collect()

    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reduce batch size if needed
    # Enable gradient checkpointing
    # Use mixed precision training
    # Use data loader workers efficiently

# CUDA Out of Memory Error
def handle_cuda_oom():
    """Handle CUDA Out of Memory error"""
    solutions = [
        "Reduce batch size",
        "Enable gradient checkpointing",
        "Use mixed precision training",
        "Clear CUDA cache: torch.cuda.empty_cache()",
        "Use smaller model or architecture",
        "Process data in smaller chunks",
        "Use CPU for preprocessing"
    ]

    for solution in solutions:
        print(f"- {solution}")

# DataLoader Memory Leak
def fix_dataloader_memory_leak():
    """Fix DataLoader memory leak"""
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,  # Set to 0 for debugging
        pin_memory=False,  # Set to False if causing issues
        persistent_workers=False  # Disable persistent workers
    )

    # Add garbage collection in training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            # Training code here

            # Clear memory periodically
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
```

#### Training Instability

```python
import torch
import numpy as np

def diagnose_training_problems():
    """Diagnose common training problems"""

    # NaN Loss
    if torch.isnan(loss).any():
        print("NaN loss detected!")
        solutions = [
            "Check learning rate (reduce if too high)",
            "Check input data for NaN values",
            "Add gradient clipping",
            "Check model initialization",
            "Use smaller batch size",
            "Add batch normalization"
        ]
        for solution in solutions:
            print(f"- {solution}")

    # Exploding Gradients
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    if total_norm > 10:
        print(f"Large gradient norm detected: {total_norm:.4f}")
        print("- Apply gradient clipping")
        print("- Reduce learning rate")
        print("- Check model architecture")

    # Vanishing Gradients
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    if total_norm < 1e-6:
        print(f"Small gradient norm detected: {total_norm:.6f}")
        print("- Check activation functions")
        print("- Add skip connections")
        print("- Use residual blocks")

def fix_training_stability():
    """Fix training stability issues"""

    # Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Proper Initialization
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)

    model.apply(init_weights)

    # Learning Rate Scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### Data Pipeline Issues

```python
def debug_data_pipeline():
    """Debug data pipeline issues"""

    # Check data loading
    print(f"Dataset length: {len(dataset)}")
    print(f"First sample: {dataset[0]}")

    # Check data shapes
    sample = dataset[0]
    if isinstance(sample, tuple):
        data, target = sample
        print(f"Data shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Target type: {type(target)}")

    # Check for data corruption
    if torch.isnan(data).any():
        print("NaN values detected in data!")

    if torch.isinf(data).any():
        print("Inf values detected in data!")

    # Check class distribution
    if hasattr(dataset, 'targets'):
        unique, counts = torch.unique(dataset.targets, return_counts=True)
        print("Class distribution:")
        for class_idx, count in zip(unique, counts):
            print(f"  Class {class_idx}: {count} samples")

def fix_data_loading():
    """Fix common data loading issues"""

    # Memory-efficient data loading
    class EfficientDataset(Dataset):
        def __init__(self, data_path):
            self.data = pd.read_csv(data_path)
            # Don't load all data into memory
            self.file_paths = self.data['file_path'].tolist()
            self.labels = self.data['label'].tolist()

        def __getitem__(self, idx):
            # Load data on demand
            image = Image.open(self.file_paths[idx]).convert('RGB')
            label = self.labels[idx]

            # Apply transformations
            if self.transform:
                image = self.transform(image)

            return image, label

    # Pin memory for GPU training
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
```

### Performance Troubleshooting

#### Slow Training

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(description):
    start = time.time()
    yield
    end = time.time()
    print(f'{description}: {(end - start):.2f} seconds')

def diagnose_slow_training():
    """Diagnose slow training issues"""

    # Time different components
    with timer('Data loading'):
        # Check data loading speed
        for batch in dataloader:
            break

    with timer('Model forward pass'):
        # Check model inference speed
        with torch.no_grad():
            _ = model(data)

    with timer('Model backward pass'):
        # Check backpropagation speed
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

    # Check hardware utilization
    if torch.cuda.is_available():
        gpu_util = torch.cuda.utilization()
        print(f"GPU Utilization: {gpu_util}%")

def optimize_training_speed():
    """Optimize training speed"""

    # Data loading optimization
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=8,  # Increase workers
        prefetch_factor=2,  # Prefetch batches
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive
    )

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Gradient accumulation
    accumulation_steps = 4

    # Efficient model architecture
    model = torch.compile(model)  # PyTorch 2.0 compile

    # Use appropriate precision
    model.half()  # Use FP16
```

#### Model Performance Issues

```python
def diagnose_model_performance():
    """Diagnose model performance issues"""

    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Check model architecture
    print("\nModel architecture:")
    print(model)

    # Profile model
    with torch.profiler.profile() as prof:
        with torch.profiler.record_function("model_inference"):
            _ = model(data)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

def fix_underfitting():
    """Fix underfitting issues"""

    # Increase model capacity
    # Add more layers
    # Add more parameters
    # Reduce regularization

    # Remove dropout
    model = nn.Sequential(
        nn.Linear(input_size, 512),
        nn.ReLU(),
        # Remove dropout if over-regularizing
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, output_size)
    )

    # Reduce weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Increase learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Add more training time
    num_epochs += 50

def fix_overfitting():
    """Fix overfitting issues"""

    # Regularization techniques
    # Add dropout
    model = nn.Sequential(
        nn.Linear(input_size, 512),
        nn.ReLU(),
        nn.Dropout(0.5),  # Add dropout
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),  # Add dropout
        nn.Linear(256, output_size)
    )

    # Weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Early stopping
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Increase training data
    # Use ensemble methods
```

---

## 📱 MOBILE-FRIENDLY QUICK REFERENCE

### Text-Based Algorithm Selector (For Phone)

```
🔢 NUMBERS? → Regression
├─ Simple line → Linear
└─ Complex curve → Random Forest → XGBoost

🎯 YES/NO? → Classification
├─ Simple decision → Logistic
└─ Complex patterns → Random Forest → XGBoost

🖼️ IMAGES? → Deep Learning
├─ Basic recognition → CNN
└─ Advanced → ResNet/EfficientNet

📝 TEXT? → Deep Learning
├─ Simple spam → Naive Bayes
└─ Understanding → BERT

👥 GROUPS? → Clustering
├─ Round groups → K-Means
└─ Irregular → DBSCAN
```

### 🎓 LEARNING PATH RECOMMENDATIONS

```
BEGINNER PATH (0-6 months)
├─ Week 1-2: Linear/Logistic Regression
├─ Week 3-4: Decision Trees & Random Forest
├─ Week 5-6: Data preprocessing & evaluation
└─ Week 7-8: First complete project

INTERMEDIATE PATH (6-18 months)
├─ Month 4-6: Deep Learning basics
├─ Month 7-9: CNN for images
├─ Month 10-12: NLP fundamentals
└─ Month 13-18: Specialized domains

ADVANCED PATH (18+ months)
├─ Transformers & BERT
├─ Advanced architectures
├─ MLOps & deployment
└─ Research & innovation
```

### 📞 EMERGENCY TROUBLESHOOTING

| Problem              | Quick Fix              | Advanced Fix           |
| -------------------- | ---------------------- | ---------------------- |
| **Model too slow**   | Reduce data size       | Use GPU, optimize code |
| **Low accuracy**     | Check data quality     | Feature engineering    |
| **Overfitting**      | Add regularization     | More data, dropout     |
| **Out of memory**    | Reduce batch size      | Use smaller model      |
| **NaN errors**       | Check input data       | Gradient clipping      |
| **Slow convergence** | Increase learning rate | Better initialization  |

### 🎯 SUCCESS METRICS BY PROBLEM TYPE

```
CLASSIFICATION SUCCESS BENCHMARKS
├─ Spam Detection: >95% accuracy
├─ Image Recognition: >90% accuracy
├─ Sentiment Analysis: >85% accuracy
└─ Medical Diagnosis: >95% accuracy

REGRESSION SUCCESS BENCHMARKS
├─ House Prices: R² >0.85
├─ Stock Prices: R² >0.70
├─ Temperature: MAE <2°C
└─ Sales Forecast: MAPE <15%

CLUSTERING SUCCESS BENCHMARKS
├─ Customer Groups: Silhouette >0.5
├─ Anomaly Detection: Precision >0.80
└─ Image Segmentation: IoU >0.70
```

---

## 🎨 VISUAL DESIGN ELEMENTS USED

### ✅ Enhanced Features Added:

1. **Algorithm Selection Flowcharts** - Interactive decision trees
2. **Visual Quick Reference Cards** - Box-formatted algorithm cards
3. **Performance Comparison Tables** - Speed/Accuracy matrices
4. **Process Flow Diagrams** - How algorithms work internally
5. **Hardware Requirements Guide** - GPU/CPU recommendations
6. **Printable Reference Cards** - Cut-and-keep format
7. **Troubleshooting Flowcharts** - Problem-solving guides
8. **Mobile-Friendly Selectors** - Text-based quick access

### 📊 Data Visualization Elements:

- ASCII art boxes and tables
- Mermaid flowcharts and diagrams
- Speed/Accuracy rating systems
- Color-coded difficulty levels
- Hardware requirement indicators
- Success rate benchmarks

### 🖨️ Print Optimization:

- Compact reference cards
- One-page summary tables
- Cut-and-keep format
- Mobile-friendly text versions
- Emergency quick fixes

---

**🎉 COMPLETE ENHANCEMENT SUMMARY**

Your AI Cheat Sheets reference guide is now enhanced with:

- ✅ **12 Algorithm Selection Flowcharts**
- ✅ **20+ Visual Quick Reference Cards**
- ✅ **Performance Comparison Tables**
- ✅ **Process Flow Diagrams**
- ✅ **Hardware Requirements Guide**
- ✅ **Printable Reference Format**
- ✅ **Troubleshooting Flowcharts**
- ✅ **Mobile-Friendly Selectors**

**Ready for immediate use - both digital and print! 🚀**
