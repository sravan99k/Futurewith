# Advanced Statistical Methods & Causal Inference - Quick Reference Cheatsheet

## Table of Contents

1. [Bayesian Statistics Quick Reference](#bayesian-statistics-quick-reference)
2. [Causal Inference Fundamentals](#causal-inference-fundamentals)
3. [Experimental Design Guide](#experimental-design-guide)
4. [Quasi-experimental Methods](#quasi-experimental-methods)
5. [Causal Machine Learning](#causal-machine-learning)
6. [Advanced Regression Methods](#advanced-regression-methods)
7. [Time Series Causality](#time-series-causality)
8. [A/B Testing Best Practices](#ab-testing-best-practices)
9. [Statistical Tests Reference](#statistical-tests-reference)
10. [Common Pitfalls & Solutions](#common-pitfalls--solutions)

---

## 1. Bayesian Statistics Quick Reference

### Key Formulas

```python
# Bayes' Theorem
P(θ|D) = P(D|θ) × P(θ) / P(D)

# Posterior for Normal distribution (known variance)
μ_post = (μ_prior/σ_prior² + n×x̄/σ²) / (1/σ_prior² + n/σ²)
σ_post² = 1 / (1/σ_prior² + n/σ²)

# Posterior for Normal distribution (known mean)
σ_post² = 1 / (1/σ_prior² + n/σ²)
α_post = α_prior + n/2
β_post = β_prior + Σ(xi - μ)²/2

# Credible Interval
CI = [θ_lower, θ_upper] where P(θ_lower < θ < θ_upper | D) = 1-α
```

### Common Prior Distributions

| Prior Type  | Parameters        | Use Case                             |
| ----------- | ----------------- | ------------------------------------ |
| Normal      | μ_prior, σ_prior² | Continuous parameters                |
| Gamma       | α, β              | Positive parameters (variance, rate) |
| Beta        | α, β              | Probabilities [0,1]                  |
| Uniform     | a, b              | Bounded parameters                   |
| Half-Normal | σ                 | Scale parameters                     |

### Model Selection

```python
# Bayes Factor
BF = P(D|M1) / P(D|M2)
# BF > 3: Positive evidence for M1
# BF > 10: Strong evidence for M1
# BF > 30: Very strong evidence for M1

# LOO-CV for model comparison
LOO = Σ log(p(yi|y-i, θ))
```

---

## 2. Causal Inference Fundamentals

### Potential Outcomes Framework

```python
# Fundamental Equation
Y = T×Y(1) + (1-T)×Y(0)

# Average Treatment Effect (ATE)
ATE = E[Y(1) - Y(0)]

# Average Treatment Effect on Treated (ATT)
ATT = E[Y(1) - Y(0) | T = 1]

# Individual Treatment Effect (ITE)
ITE_i = Y_i(1) - Y_i(0)
```

### Key Assumptions

1. **SUTVA (Stable Unit Treatment Value Assumption)**
   - No interference between units
   - Single version of treatment

2. **Consistency**
   - Y = Y(1) when T=1, Y=Y(0) when T=0

3. **Ignorability (Unconfoundedness)**
   - Y(0), Y(1) ⊥ T | X

4. **Positivity**
   - 0 < P(T=1|X) < 1 for all X

### Effect Estimation Methods

| Method         | Formula                             | Assumptions                 |
| -------------- | ----------------------------------- | --------------------------- |
| Naive          | E[Y\|T=1] - E[Y\|T=0]               | None (biased)               |
| Matching       | E[Y(1) - Y(0)\|matched pairs]       | Ignorability                |
| IPW            | Σ(TiYi/wi) - Σ((1-Ti)Yi/(1-wi))     | Positivity + Ignorability   |
| Stratification | Σ w_s × (E[Y\|T=1,s] - E[Y\|T=0,s]) | Ignorability                |
| Regression     | E[Y\|T,X] difference                | Correct model specification |

---

## 3. Experimental Design Guide

### Sample Size Calculation

```python
# Two-sample t-test (continuous outcome)
n_per_group = 2 × (Zα/2 + Zβ)² × σ² / δ²
# where δ = effect size, σ = standard deviation

# Proportions test
n_per_group = (Zα/2 + Zβ)² × (p1(1-p1) + p2(1-p2)) / (p1-p2)²

# Multiple comparisons (Bonferroni)
α_adjusted = α / (k-1)  # k = number of comparisons
```

### Randomization Schemes

| Scheme     | Description                        | When to Use                     |
| ---------- | ---------------------------------- | ------------------------------- |
| Simple     | Bernoulli(p=0.5)                   | Equal allocation, large samples |
| Block      | Blocks of fixed size               | Balance across time/strata      |
| Stratified | Separate randomization per stratum | Known important confounders     |
| Adaptive   | Adjust based on imbalance          | Covariate-adaptive designs      |

### Power Analysis

```python
# Effect sizes (Cohen's conventions)
small = 0.2     # Subtle effects
medium = 0.5    # Moderate effects
large = 0.8     # Large effects

# Power = 1 - β
# Commonly target: Power ≥ 0.8, α = 0.05
```

---

## 4. Quasi-experimental Methods

### Difference-in-Differences (DID)

```python
# Basic DID model
Y_it = α + β×Post_t + γ×Treat_i + δ×(Treat_i × Post_t) + ε_it
# δ = DID estimator

# Event Study (Dynamic DID)
Y_it = α_i + λ_t + Σ β_k × (Treat_i × 1{t=k}) + ε_it

# Triple Differences (DDD)
Y_ijt = α + β×Treat_i + γ×Post_t + δ×Region_j +
       ε×(Treat_i × Post_t) + ζ×(Treat_i × Region_j) +
       η×(Post_t × Region_j) + θ×(Treat_i × Post_t × Region_j) + u_ijt
# θ = DDD estimator
```

### Instrumental Variables (IV)

```python
# Two-Stage Least Squares (2SLS)
# Stage 1: T = π×Z + v
# Stage 2: Y = β×T̂ + u

# First Stage F-statistic (rule of thumb: F > 10)
F = (R² / k) / ((1 - R²) / (n - k - 1))
```

### Assumption Checklist

| Method   | Core Assumptions                           |
| -------- | ------------------------------------------ |
| DID      | Parallel trends, SUTVA, No spillovers      |
| IV       | Relevance, Exogeneity, Monotonicity, SUTVA |
| Matching | Ignorability, Positivity, Common support   |
| RDD      | Continuity at cutoff, No manipulation      |

---

## 5. Causal Machine Learning

### Meta-Learners

| Learner    | Steps                            | Pros                     | Cons                           |
| ---------- | -------------------------------- | ------------------------ | ------------------------------ |
| S-Learner  | 1 model with treatment indicator | Simple, robust           | Can miss complex heterogeneity |
| T-Learner  | 2 separate models                | Better for heterogeneity | Sample splitting issues        |
| X-Learner  | 4 models total                   | Best performance         | Most complex                   |
| DR-Learner | Doubly robust                    | Efficiency gains         | Requires correct models        |

### Causal Forest

```python
# Key parameters
n_estimators = 100    # Number of trees
max_depth = 6         # Tree depth
min_samples_leaf = 5  # Minimum samples per leaf

# Split criterion (causal gain)
gain = (n_left/n_total) × τ_left² + (n_right/n_total) × τ_right²
# where τ = treatment effect in node
```

### Model Diagnostics

```python
# Propensity score diagnostics
- Balance: |Standardized diff| < 0.1
- Overlap: Common support region
- Positivity: P(T|X) ∈ (0,1)

# Treatment effect overlap
- Overlap: Treatment effects vary smoothly with X
- Extrapolation: Avoid extrapolating beyond data
```

---

## 6. Advanced Regression Methods

### Nonparametric Regression

```python
# Kernel Regression
ŷ(x) = Σ wi × yi where wi = K((x-xi)/h) / Σ K((x-xj)/h)

# Bandwidth selection
h_opt = n^(-1/(d+4)) × σ × n^(-1/5)
# where d = dimension, σ = residual std dev

# Cross-validation
CV(h) = Σ (yi - ŷ_i^(-i))²
```

### Generalized Additive Models (GAM)

```python
# Model structure
g(E[Y]) = α + f1(X1) + f2(X2) + ... + fp(Xp)

# Backfitting algorithm
for each fj:
    residual = Y - Σ fi(Xi) except j
    fj = smooth(residual|Xj)

# Convergence check
max|fj_new - fj_old| < tolerance
```

---

## 7. Time Series Causality

### Granger Causality

```python
# Test X Granger-causes Y
Model 1: Y_t = α + Σ βi×Y_{t-i} + εt
Model 2: Y_t = α + Σ βi×Y_{t-i} + Σ γi×X_{t-i} + εt

# F-test for H0: γ1 = ... = γk = 0
F = ((RSS1 - RSS2)/k) / (RSS2/(T-2k-1))
```

### Spectral Causality

```python
# Transfer function
H(ω) = Sxy(ω) / Sxx(ω)

# Coherence
Cxy(ω) = |Sxy(ω)|² / (Sxx(ω) × Syy(ω))

# Spectral causality
Causality_xy(ω) = log(1 + |Hxy(ω)|² × (1 - Cxy(ω)))
```

---

## 8. A/B Testing Best Practices

### Experiment Design

```python
# Sample size for conversion rate
n = (Zα/2 + Zβ)² × (p1(1-p1) + p2(1-p2)) / (p1-p2)²

# Duration planning
duration = n / (daily_users × conversion_rate)

# Multiple testing correction
Bonferroni: α/k
Holm-Bonferroni: Sequential Bonferroni
Benjamini-Hochberg: FDR control
```

### Sequential Testing

```python
# O'Brien-Fleming boundaries
zα(n) = Zα/2 / √n

# Group sequential design
n_interim = [0.25, 0.5, 0.75] × n_total
boundaries = [3.29, 2.34, 1.99]  # For α=0.05, power=0.8
```

### Stopping Rules

| Rule            | Efficacy                   | Futility                 |
| --------------- | -------------------------- | ------------------------ |
| O'Brien-Fleming | Strict early, relaxed late | Always continue          |
| Pocock          | Consistent boundaries      | Can stop early           |
| Haybittle-Peto  | Very strict early          | Stop if poor performance |

---

## 9. Statistical Tests Reference

### Parametric Tests

| Test                | Use Case               | Test Statistic             |
| ------------------- | ---------------------- | -------------------------- |
| t-test (one sample) | H0: μ = μ0             | t = (x̄ - μ0) / (s/√n)      |
| t-test (two sample) | H0: μ1 = μ2            | t = (x̄1 - x̄2) / SE         |
| ANOVA               | H0: μ1 = μ2 = ... = μk | F = MS_between / MS_within |
| Chi-square test     | H0: independence       | χ² = Σ(O-E)²/E             |

### Nonparametric Tests

| Test                 | Parametric Alternative | When to Use              |
| -------------------- | ---------------------- | ------------------------ |
| Mann-Whitney U       | t-test (two sample)    | Non-normal data          |
| Wilcoxon signed-rank | t-test (paired)        | Paired data, non-normal  |
| Kruskal-Wallis       | ANOVA                  | Non-normal, ordinal data |
| Fisher's exact       | Chi-square             | Small sample sizes       |

### Effect Sizes

| Effect Size | Formula                | Interpretation                        |
| ----------- | ---------------------- | ------------------------------------- |
| Cohen's d   | (μ1 - μ2) / σ_pooled   | 0.2=small, 0.5=medium, 0.8=large      |
| Pearson's r | √(F / (F + n - k - 1)) | 0.1=small, 0.3=medium, 0.5=large      |
| Odds Ratio  | (ad/bc)                | 1=no effect, >1=positive, <1=negative |

---

## 10. Common Pitfalls & Solutions

### Causal Inference Pitfalls

| Pitfall           | Problem                 | Solution                                  |
| ----------------- | ----------------------- | ----------------------------------------- |
| Confounding       | Biased estimates        | Include relevant covariates, use DAGs     |
| Selection bias    | Non-random missing data | Multiple imputation, sensitivity analysis |
| Measurement error | Imprecise variables     | Validation substudies, error correction   |
| Survivorship bias | Truncated samples       | Time-to-event analysis                    |

### Experimental Design Pitfalls

| Pitfall            | Problem               | Solution                              |
| ------------------ | --------------------- | ------------------------------------- |
| Underpowered study | False negatives       | Power analysis, larger sample         |
| Multiple testing   | False positives       | Correction methods, pre-registration  |
| Peeking at data    | Inflated Type I error | Sequential boundaries, no early looks |
| Attrition bias     | Differential dropout  | ITT analysis, sensitivity analysis    |

### Statistical Modeling Pitfalls

| Pitfall                | Problem             | Solution                                |
| ---------------------- | ------------------- | --------------------------------------- |
| Overfitting            | Poor generalization | Cross-validation, regularization        |
| Model misspecification | Biased estimates    | Model checking, robust methods          |
| Collinearity           | Unstable estimates  | VIF checks, PCA, regularization         |
| Heteroscedasticity     | Invalid inferences  | Robust standard errors, transformations |

---

## Quick Implementation Templates

### Bayesian Analysis Template

```python
import pymc3 as pm
import arviz as az

# Define model
with pm.Model() as model:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Likelihood
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=data)

    # Sample
    trace = pm.sample(2000, tune=1000)

# Results
az.summary(trace)
```

### DID Estimation Template

```python
from sklearn.linear_model import LinearRegression

# Create design matrix
X = pd.get_dummies(data[['treated', 'post']].join(unit_dummies).join(time_dummies))

# Fit model
model = LinearRegression()
model.fit(X, y)

# DID estimate
did_estimate = model.coef_[X.columns.get_loc('treated_post')]
```

### A/B Test Analysis Template

```python
from scipy.stats import chi2_contingency, ttest_ind

# Conversion rate test
contingency = [[conversions_A, visitors_A - conversions_A],
               [conversions_B, visitors_B - conversions_B]]
chi2, p_value, dof, expected = chi2_contingency(contingency)

# Means comparison
t_stat, p_value = ttest_ind(group_A, group_B)

# Effect size
effect_size = (mean_B - mean_A) / pooled_std
```

---

## Decision Trees

### Which Method to Use?

```
Is randomization possible?
├─ YES → Randomized Experiment
│  ├─ Two groups? → RCT with t-test
│  ├─ Multiple groups? → ANOVA or multiple t-tests with correction
│  └─ Sequential? → Group sequential design
│
└─ NO → Observational Study
   ├─ Time dimension available?
   │  ├─ YES → Difference-in-Differences
   │  │  ├─ Multiple groups? → Event study
   │  │  └─ Additional dimension? → Triple differences
   │  └─ NO → Cross-sectional
   │     ├─ Instrument available? → IV estimation
   │     ├─ Good covariates? → Matching/Regression
   │     └─ Complex heterogeneity? → Causal ML
```

### Model Selection Flowchart

```
Continuous outcome?
├─ YES
│  ├─ Normal distribution? → Linear regression
│  ├─ Non-normal? → Nonparametric methods
│  └─ Complex relationships? → GAM/Random Forest
│
└─ NO (Binary outcome)
   ├─ Logistic regression (small effects)
   ├─ Risk difference models (additive effects)
   └─ Poisson models (count data)
```

---

## Error Patterns & Diagnostics

### Residual Diagnostics

```python
# Check for normality
stats.shapiro(residuals)

# Check for homoscedasticity
plt.scatter(fitted, residuals)
plt.axhline(y=0)

# Check for outliers
outlier_test = stats.outliersOLS(fitted, residuals)

# Autocorrelation (time series)
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=10)
```

### Model Checking Checklist

- [ ] Residual plots look random
- [ ] Normal Q-Q plot is linear
- [ ] No systematic patterns in residuals
- [ ] Influential points identified and checked
- [ ] Model assumptions validated
- [ ] Cross-validation performance assessed

---

## Software Resources

### Python Libraries

| Purpose      | Library               | Key Functions                               |
| ------------ | --------------------- | ------------------------------------------- |
| Bayesian     | PyMC3, Stan           | `pm.sample()`, `az.plot_trace()`            |
| Causal       | DoWhy, EconML         | `dowhyestimate()`, `CausalDML()`            |
| Experimental | Statsmodels, Pingouin | `statsmodels.api`, `pingouin.ttest()`       |
| Time Series  | Statsmodels, Prophet  | `arima_model.fit()`, `prophet.forecast()`   |
| Causal ML    | EconML, CausalForest  | `CausalForestDML()`, `UpliftRandomForest()` |

### R Packages

| Purpose       | Package               | Key Functions                        |
| ------------- | --------------------- | ------------------------------------ |
| Bayesian      | rstanarm, brms        | `stan_glm()`, `brm()`                |
| Causal        | estimatr, MatchIt     | `difference_in_means()`, `matchit()` |
| Experimental  | randomizr, experiment | `block_ra()`, `experiments()`        |
| Time Series   | vars, forecast        | `VAR()`, `auto.arima()`              |
| Meta-learners | grf, causalmatch      | `causal_forest()`, `causal_match()`  |

---

## Common Interview Questions

### Q: How do you choose between parametric and nonparametric tests?

**A:** Check normality with Shapiro-Wilk test. If p > 0.05, assume normality and use parametric tests. If p ≤ 0.05 or small sample size, use nonparametric alternatives.

### Q: What's the difference between ATE and ATT?

**A:** ATE is average effect across all population units. ATT is average effect only for those who actually received treatment. ATT is more relevant when treatment assignment is not randomized.

### Q: How do you handle multiple testing?

**A:** Use Bonferroni for family-wise error rate control or Benjamini-Hochberg for false discovery rate control. Always pre-specify analysis plan.

### Q: When should you use instrumental variables?

**A:** When you have a valid instrument (relevant, exogenous, affects outcome only through treatment). Common in economics: policies, weather, geography.

### Q: How do you assess parallel trends in DID?

**A:** Test for differential pre-treatment trends. If data allows, use event study with leads. If pre-treatment trends are parallel, DID estimate is valid.

---

## Glossary

**ATE**: Average Treatment Effect
**ATT**: Average Treatment Effect on the Treated
**CATE**: Conditional Average Treatment Effect
**DAG**: Directed Acyclic Graph
**DID**: Difference-in-Differences
**IV**: Instrumental Variables
**IPW**: Inverse Probability Weighting
**ITT**: Intention-to-Treat
**LATE**: Local Average Treatment Effect
**PSM**: Propensity Score Matching
**RCT**: Randomized Controlled Trial
**SUTVA**: Stable Unit Treatment Value Assumption

---

_This cheatsheet provides quick reference for advanced statistical methods and causal inference. Use alongside detailed documentation and domain expertise for specific applications._
