# Advanced Statistical Methods & Causal Inference - Theory

## Course Overview

This module covers advanced statistical methods essential for AI/ML applications, with a special focus on causal inference - one of the most critical but under-represented areas in modern AI education.

## Table of Contents

1. [Advanced Probability Theory](#advanced-probability-theory)
2. [Bayesian Statistics](#bayesian-statistics)
3. [Causal Inference Fundamentals](#causal-inference-fundamentals)
4. [Potential Outcomes Framework](#potential-outcomes-framework)
5. [Directed Acyclic Graphs (DAGs)](#directed-acyclic-graphs-dags)
6. [Experimental Design](#experimental-design)
7. [Quasi-Experimental Methods](#quasi-experimental-methods)
8. [Causal Machine Learning](#causal-machine-learning)
9. [Advanced Regression Methods](#advanced-regression-methods)
10. [Time Series Causality](#time-series-causality)
11. [A/B Testing & Experimentation](#ab-testing--experimentation)
12. [Causal AI Applications](#causal-ai-applications)

---

## 1. Advanced Probability Theory

### Probability Spaces and σ-Algebras

- **Sample Space (Ω)**: All possible outcomes
- **σ-Algebra (F)**: Collection of measurable subsets
- **Probability Measure (P)**: Function mapping σ-algebra to [0,1]

```python
import numpy as np
import scipy.stats as stats
from scipy.special import gamma, beta
import matplotlib.pyplot as plt

class ProbabilitySpace:
    def __init__(self, sample_space, sigma_algebra, probability_measure):
        self.omega = sample_space
        self.f = sigma_algebra
        self.p = probability_measure

    def conditional_probability(self, event_a, event_b):
        """Calculate P(A|B) = P(A∩B) / P(B)"""
        intersection = event_a & event_b
        if self.p(event_b) == 0:
            raise ValueError("Probability of B cannot be zero")
        return self.p(intersection) / self.p(event_b)

    def independence_test(self, events):
        """Test for statistical independence"""
        n = len(events)
        for i in range(n):
            for j in range(i+1, n):
                prob_intersection = self.p(events[i] & events[j])
                prob_product = self.p(events[i]) * self.p(events[j])
                if abs(prob_intersection - prob_product) > 1e-6:
                    return False, (events[i], events[j], prob_intersection, prob_product)
        return True, None
```

### Advanced Distributions

#### Stable Distributions

```python
class StableDistribution:
    """Stable distributions for heavy-tailed data"""
    def __init__(self, alpha, beta, scale, location):
        self.alpha = alpha  # Stability parameter (0 < α ≤ 2)
        self.beta = beta    # Skewness parameter (-1 ≤ β ≤ 1)
        self.scale = scale  # Scale parameter
        self.location = location  # Location parameter

    def pdf(self, x):
        """Probability density function of stable distribution"""
        # Stable distributions don't have closed-form PDF
        # This is a numerical approximation
        from scipy import integrate

        def integrand(t):
            if t == 0:
                return 0
            return np.exp(-self.scale * t**self.alpha * np.cos(self.beta * t))

        # Numerical integration required for stable distributions
        result, _ = integrate.quad(lambda t: integrand(t), -np.inf, np.inf)
        return result

    def sample(self, n):
        """Generate samples from stable distribution"""
        # Use Chambers-Mallows-Stuck method for sampling
        U = np.random.uniform(-np.pi/2, np.pi/2, n)
        V = np.random.exponential(1, n)

        if self.alpha == 1:
            # Cauchy distribution
            return self.location + self.scale * np.tan(U)
        else:
            # General stable distribution
            c = np.cos(U)
            w = np.sin(self.alpha * U)
            z = c / ((np.pi * V * c * w)**(1/self.alpha))
            return self.location + self.scale * z
```

#### Skewed Distributions

```python
class SkewedNormal:
    """Skewed normal distribution"""
    def __init__(self, location=0, scale=1, shape=0):
        self.location = location
        self.scale = scale
        self.shape = shape  # Shape parameter controlling skewness

    def pdf(self, x):
        """PDF of skewed normal distribution"""
        z = (x - self.location) / self.scale
        phi = stats.norm.pdf(z)
        phi_sym = stats.norm.cdf(self.shape * z)
        return 2 * phi * phi_sym / self.scale

    def cdf(self, x):
        """CDF of skewed normal distribution"""
        z = (x - self.location) / self.scale
        cdf_norm = stats.norm.cdf(z)
        cdf_skewed = stats.norm.cdf(self.shape * z) * cdf_norm
        return 2 * cdf_norm - cdf_skewed

    def rvs(self, size=1):
        """Generate random variates"""
        # Use transformation method
        U = np.random.uniform(0, 1, size)
        V = np.random.normal(0, 1, size)

        if self.shape == 0:
            # Standard normal
            return self.location + self.scale * V
        else:
            # Skewed transformation
            skew_factor = np.sign(self.shape) * U
            return self.location + self.scale * V * np.exp(0.5 * self.shape * skew_factor**2)

# Example usage
dist = SkewedNormal(location=0, scale=1, shape=2)
x = np.linspace(-5, 5, 100)
pdf_values = [dist.pdf(val) for val in x]

plt.figure(figsize=(10, 6))
plt.plot(x, pdf_values, label='Skewed Normal (α=2)')
plt.plot(x, stats.norm.pdf(x), label='Standard Normal')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Skewed Normal vs Standard Normal Distribution')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 2. Bayesian Statistics

### Bayesian Inference Framework

#### Posterior Distribution

Bayes' Theorem: P(θ|D) = P(D|θ) × P(θ) / P(D)

Where:

- P(θ|D) = Posterior probability of parameters given data
- P(D|θ) = Likelihood of data given parameters
- P(θ) = Prior probability of parameters
- P(D) = Evidence/marginal likelihood

```python
import pymc3 as pm
import arviz as az

class BayesianInference:
    """Bayesian inference framework"""

    def __init__(self, prior_params, likelihood_func):
        self.prior_params = prior_params
        self.likelihood_func = likelihood_func
        self.posterior = None
        self.trace = None

    def specify_prior(self, param_name, distribution, **params):
        """Specify prior distribution for parameter"""
        if not hasattr(self, 'priors'):
            self.priors = {}

        self.priors[param_name] = {
            'distribution': distribution,
            'params': params
        }

    def compute_posterior(self, data):
        """Compute posterior distribution using MCMC"""
        with pm.Model() as model:
            # Define priors
            priors = {}
            for param, spec in self.priors.items():
                if spec['distribution'] == 'normal':
                    priors[param] = pm.Normal(
                        param,
                        mu=spec['params']['mu'],
                        sigma=spec['params']['sigma']
                    )
                elif spec['distribution'] == 'gamma':
                    priors[param] = pm.Gamma(
                        param,
                        alpha=spec['params']['alpha'],
                        beta=spec['params']['beta']
                    )
                elif spec['distribution'] == 'uniform':
                    priors[param] = pm.Uniform(
                        param,
                        lower=spec['params']['lower'],
                        upper=spec['params']['upper']
                    )

            # Define likelihood
            likelihood_params = {k: v for k, v in priors.items()}
            likelihood_params['observed'] = data

            # Add additional likelihood parameters if needed
            if 'sigma' not in priors:
                priors['sigma'] = pm.HalfNormal('sigma', sigma=10)

            likelihood = pm.Normal('likelihood', **likelihood_params)

            # Sample from posterior
            trace = pm.sample(2000, tune=1000, return_inferencedata=True)

        self.trace = trace
        return trace

    def posterior_summary(self, param_name):
        """Get summary statistics of posterior"""
        if self.trace is None:
            raise ValueError("Must compute posterior first")

        posterior_samples = self.trace.posterior[param_name].values.flatten()

        return {
            'mean': np.mean(posterior_samples),
            'std': np.std(posterior_samples),
            'q025': np.percentile(posterior_samples, 2.5),
            'q975': np.percentile(posterior_samples, 97.5),
            'hpd': self.compute_hpd(posterior_samples)
        }

    def compute_hpd(self, samples, alpha=0.05):
        """Compute highest posterior density interval"""
        n_samples = len(samples)
        sorted_samples = np.sort(samples)
        interval_width = int(np.floor((1 - alpha) * n_samples))
        min_width = np.inf
        hpd_interval = None

        for i in range(0, n_samples - interval_width):
            interval = [sorted_samples[i], sorted_samples[i + interval_width]]
            width = interval[1] - interval[0]

            if width < min_width:
                min_width = width
                hpd_interval = interval

        return hpd_interval

# Example: Bayesian Linear Regression
def bayesian_linear_regression_example():
    """Demonstrate Bayesian linear regression"""

    # Generate synthetic data
    np.random.seed(42)
    n = 100
    x = np.random.uniform(-2, 2, n)
    true_slope = 2.5
    true_intercept = 0.5
    true_sigma = 1.0

    y = true_intercept + true_slope * x + np.random.normal(0, true_sigma, n)

    # Setup Bayesian inference
    bi = BayesianInference(None, None)

    # Specify priors
    bi.specify_prior('intercept', 'normal', mu=0, sigma=10)
    bi.specify_prior('slope', 'normal', mu=0, sigma=10)

    # Compute posterior
    trace = bi.compute_posterior(y)

    # Print posterior summary
    print("Posterior Summary:")
    print("Intercept:", bi.posterior_summary('intercept'))
    print("Slope:", bi.posterior_summary('slope'))
    print("Sigma:", bi.posterior_summary('sigma'))

    return bi, trace

# Bayesian Model Selection
class BayesianModelSelection:
    """Bayesian model selection using Bayes factors"""

    def __init__(self, models):
        self.models = models
        self.model_posteriors = None
        self.bayes_factors = None

    def compute_model_evidence(self, data, model_spec):
        """Compute marginal likelihood (model evidence)"""
        # Use PyMC3 to compute model evidence
        with pm.Model() as model:
            # Define model structure
            for param, spec in model_spec['priors'].items():
                if spec['type'] == 'normal':
                    param_var = pm.Normal(param, mu=spec['mu'], sigma=spec['sigma'])
                elif spec['type'] == 'gamma':
                    param_var = pm.Gamma(param, alpha=spec['alpha'], beta=spec['beta'])

            # Define likelihood
            if 'likelihood' in model_spec:
                likelihood = model_spec['likelihood'](data, **model_spec['priors'])

            # Compute model evidence using built-in functions
            model_evidence = pm.loo(model)
            return model_evidence

    def compare_models(self, data, model_specs):
        """Compare multiple models using Bayes factors"""
        evidences = {}

        for model_name, model_spec in model_specs.items():
            evidence = self.compute_model_evidence(data, model_spec)
            evidences[model_name] = evidence

        # Calculate Bayes factors
        model_names = list(evidences.keys())
        model_values = list(evidences.values())

        # Relative Bayes factors
        self.bayes_factors = {}
        for i, model1 in enumerate(model_names):
            self.bayes_factors[model1] = {}
            for j, model2 in enumerate(model_names):
                if i != j:
                    bf = np.exp(model_values[j] - model_values[i])
                    self.bayes_factors[model1][model2] = bf

        return self.bayes_factors, evidences
```

---

## 3. Causal Inference Fundamentals

### The Fundamental Problem of Causal Inference

- We can only observe one potential outcome per unit
- We cannot directly observe the counterfactual
- Need to infer causal relationships from observed data

### Key Concepts

#### Causal Effect

```python
class CausalEffect:
    """Class for computing and analyzing causal effects"""

    def __init__(self, outcome, treatment, confounders=None):
        self.outcome = outcome
        self.treatment = treatment
        self.confounders = confounders or []

    def average_treatment_effect(self, data):
        """Compute average treatment effect (ATE)"""
        # E[Y(1)] - E[Y(0)]
        treated_outcomes = data[data[self.treatment] == 1][self.outcome]
        control_outcomes = data[data[self.treatment] == 0][self.outcome]

        ate = np.mean(treated_outcomes) - np.mean(control_outcomes)

        # Standard error
        var_treated = np.var(treated_outcomes, ddof=1) / len(treated_outcomes)
        var_control = np.var(control_outcomes, ddof=1) / len(control_outcomes)
        se = np.sqrt(var_treated + var_control)

        return {
            'ate': ate,
            'standard_error': se,
            'confidence_interval': (ate - 1.96*se, ate + 1.96*se),
            'p_value': 2 * (1 - stats.norm.cdf(abs(ate/se)))
        }

    def conditional_average_treatment_effect(self, data, effect_modifiers):
        """Compute conditional average treatment effect (CATE)"""
        from sklearn.linear_model import LinearRegression

        # Model treatment effect as function of effect modifiers
        if len(effect_modifiers) == 0:
            return self.average_treatment_effect(data)

        # Create interaction terms
        X = data[effect_modifiers].values
        treatment = data[self.treatment].values
        interaction_features = []

        for i, modifier in enumerate(effect_modifiers):
            interaction = treatment * X[:, i]
            interaction_features.append(interaction)

        X_interactions = np.column_stack(interaction_features)

        # Fit model for effect modifiers
        model = LinearRegression()
        model.fit(X_interactions, data[self.outcome].values)

        # CATE is the coefficient of the interaction terms
        cate_estimates = model.coef_

        return {
            'cate_estimates': cate_estimates,
            'effect_modifiers': effect_modifiers,
            'model_score': model.score(X_interactions, data[self.outcome].values)
        }
```

#### Individual Treatment Effect

```python
class IndividualTreatmentEffect:
    """Estimate individual treatment effects"""

    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model_0 = None  # Model for control group
        self.model_1 = None  # Model for treatment group
        self.is_fitted = False

    def fit(self, X, treatment, outcome):
        """Fit models for treated and control groups"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        # Split data by treatment
        treated_mask = treatment == 1
        control_mask = treatment == 0

        X_treated = X[treated_mask]
        y_treated = outcome[treated_mask]
        X_control = X[control_mask]
        y_control = outcome[control_mask]

        # Fit models
        if self.model_type == 'random_forest':
            self.model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'linear':
            self.model_1 = LinearRegression()
            self.model_0 = LinearRegression()

        self.model_1.fit(X_treated, y_treated)
        self.model_0.fit(X_control, y_control)

        self.is_fitted = True
        return self

    def predict_ite(self, X):
        """Predict individual treatment effects"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Predict outcomes under both treatment conditions
        y1_pred = self.model_1.predict(X)  # Potential outcome under treatment
        y0_pred = self.model_0.predict(X)  # Potential outcome under control

        # Individual treatment effect
        ite = y1_pred - y0_pred

        return ite

    def confidence_intervals(self, X, confidence_level=0.95):
        """Compute confidence intervals for ITE estimates"""
        from sklearn.ensemble import RandomForestRegressor
        from scipy import stats

        ite_predictions = []
        n_estimators = 100

        # Bootstrap estimates
        for i in range(100):
            # Sample with replacement
            bootstrap_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[bootstrap_idx]

            # Fit models on bootstrap sample
            ite_estimator = IndividualTreatmentEffect(model_type=self.model_type)
            ite_estimator.fit(X[bootstrap_idx], treatment[bootstrap_idx], outcome[bootstrap_idx])

            # Predict ITE
            ite_pred = ite_estimator.predict_ite(X)
            ite_predictions.append(ite_pred)

        ite_predictions = np.array(ite_predictions)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_bound = np.percentile(ite_predictions, 100 * alpha/2, axis=0)
        upper_bound = np.percentile(ite_predictions, 100 * (1 - alpha/2), axis=0)

        return lower_bound, upper_bound
```

---

## 4. Potential Outcomes Framework

### Rubin Causal Model

#### Key Assumptions

1. **Stable Unit Treatment Value Assumption (SUTVA)**
2. **Consistency**: Y = Y(1) × T + Y(0) × (1-T)
3. **Ignorability**: Y(0), Y(1) ⊥ T | X (Unconfoundedness)
4. **Positivity**: 0 < P(T=1|X) < 1

```python
class RubinCausalModel:
    """Implementation of Rubin's Causal Model"""

    def __init__(self, outcome_col, treatment_col, covariate_cols):
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.covariate_cols = covariate_cols

    def assess_unconfoundedness(self, data):
        """Assess balance in observed covariates"""
        from sklearn.metrics import roc_auc_score
        from sklearn.linear_model import LogisticRegression

        X = data[self.covariate_cols].values
        treatment = data[self.treatment_col].values

        # Train classifier to predict treatment assignment
        lr = LogisticRegression()
        lr.fit(X, treatment)

        # Predict treatment probabilities
        treatment_pred = lr.predict_proba(X)[:, 1]

        # Calculate propensity scores
        auc = roc_auc_score(treatment, treatment_pred)

        # Check balance for each covariate
        balance_results = {}
        for i, covar in enumerate(self.covariate_cols):
            treated_mean = data[data[self.treatment_col] == 1][covar].mean()
            control_mean = data[data[self.treatment_col] == 0][covar].mean()

            # Standardized difference
            pooled_std = np.sqrt(
                (data[data[self.treatment_col] == 1][covar].var() +
                 data[data[self.treatment_col] == 0][covar].var()) / 2
            )

            std_diff = abs(treated_mean - control_mean) / pooled_std

            balance_results[covar] = {
                'treated_mean': treated_mean,
                'control_mean': control_mean,
                'standardized_difference': std_diff,
                'balanced': std_diff < 0.1  # Common threshold
            }

        return {
            'propensity_score_auc': auc,
            'balance_results': balance_results,
            'overall_balanced': all(r['balanced'] for r in balance_results.values())
        }

    def estimate_ate(self, data, method='ipw'):
        """Estimate Average Treatment Effect using various methods"""
        if method == 'ipw':
            return self._inverse_probability_weighting(data)
        elif method == 'matching':
            return self._propensity_score_matching(data)
        elif method == 'doubly_robust':
            return self._doubly_robust_estimation(data)
        elif method == 'tmle':
            return self._targeted_maximum_likelihood_estimation(data)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _inverse_probability_weighting(self, data):
        """Estimate ATE using Inverse Probability Weighting"""
        from sklearn.linear_model import LogisticRegression

        # Estimate propensity scores
        X = data[self.covariate_cols].values
        treatment = data[self.treatment_col].values
        outcome = data[self.outcome_col].values

        ps_model = LogisticRegression()
        ps_model.fit(X, treatment)
        propensity_scores = ps_model.predict_proba(X)[:, 1]

        # Calculate IPW estimates
        treated_weights = treatment / propensity_scores
        control_weights = (1 - treatment) / (1 - propensity_scores)

        # Weighted means
        treated_mean = np.sum(outcome * treated_weights) / np.sum(treated_weights)
        control_mean = np.sum(outcome * control_weights) / np.sum(control_weights)

        ate = treated_mean - control_mean

        return {
            'ate': ate,
            'treated_mean': treated_mean,
            'control_mean': control_mean,
            'method': 'Inverse Probability Weighting',
            'propensity_score_model_score': ps_model.score(X, treatment)
        }

    def _propensity_score_matching(self, data, k=1):
        """Estimate ATE using Propensity Score Matching"""
        from sklearn.neighbors import NearestNeighbors

        # Estimate propensity scores
        X = data[self.covariate_cols].values
        treatment = data[self.treatment_col].values

        ps_model = LogisticRegression()
        ps_model.fit(X, treatment)
        propensity_scores = ps_model.predict_proba(X)[:, 1]

        # Separate treated and control units
        treated_indices = np.where(treatment == 1)[0]
        control_indices = np.where(treatment == 0)[0]

        if len(treated_indices) == 0 or len(control_indices) == 0:
            raise ValueError("Need both treated and control units")

        # Find matches for each treated unit
        matches = []
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nbrs.fit(propensity_scores[control_indices].reshape(-1, 1))

        for treated_idx in treated_indices:
            treated_ps = propensity_scores[treated_idx]
            distances, indices = nbrs.kneighbors([treated_ps])

            # Remove self-match if treated unit is also in control set
            valid_matches = indices[0][indices[0] < len(control_indices)]

            if len(valid_matches) > 0:
                closest_control_idx = control_indices[valid_matches[0]]
                matches.append((treated_idx, closest_control_idx))

        # Calculate treatment effect
        treatment_effects = []
        for treated_idx, control_idx in matches:
            treated_outcome = data.iloc[treated_idx][self.outcome_col]
            control_outcome = data.iloc[control_idx][self.outcome_col]
            treatment_effects.append(treated_outcome - control_outcome)

        ate = np.mean(treatment_effects)

        return {
            'ate': ate,
            'treatment_effects': treatment_effects,
            'method': f'Propensity Score Matching (k={k})',
            'num_matches': len(matches),
            'propensity_score_model_score': ps_model.score(X, treatment)
        }
```

---

## 5. Directed Acyclic Graphs (DAGs)

### Causal DAGs and Graph Theory

```python
class CausalDAG:
    """Class for working with causal Directed Acyclic Graphs"""

    def __init__(self):
        self.nodes = set()
        self.edges = {}  # Dict: parent -> set of children
        self.adjacency_matrix = None

    def add_node(self, node):
        """Add a node to the DAG"""
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = set()

    def add_edge(self, parent, child):
        """Add a directed edge from parent to child"""
        self.add_node(parent)
        self.add_node(child)
        self.edges[parent].add(child)

    def remove_edge(self, parent, child):
        """Remove a directed edge"""
        if parent in self.edges and child in self.edges[parent]:
            self.edges[parent].remove(child)

    def has_path(self, start, end, visited=None):
        """Check if there's a path from start to end"""
        if visited is None:
            visited = set()

        if start == end:
            return True

        if start in visited:
            return False

        visited.add(start)

        if start not in self.edges:
            return False

        for child in self.edges[start]:
            if self.has_path(child, end, visited.copy()):
                return True

        return False

    def get_ancestors(self, node):
        """Get all ancestors of a node"""
        ancestors = set()
        queue = [node]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)

            # Find all parents
            for parent, children in self.edges.items():
                if current in children and parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)

        return ancestors

    def get_descendants(self, node):
        """Get all descendants of a node"""
        descendants = set()
        queue = [node]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)

            if current in self.edges:
                for child in self.edges[current]:
                    if child not in descendants:
                        descendants.add(child)
                        queue.append(child)

        return descendants

    def is_dag(self):
        """Check if the graph is acyclic (DAG)"""
        # Use DFS to detect cycles
        def dfs(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            for child in self.edges.get(node, []):
                if child not in visited:
                    if dfs(child, visited, rec_stack):
                        return True
                elif child in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited = set()
        for node in self.nodes:
            if node not in visited:
                if dfs(node, visited, set()):
                    return False
        return True

    def topological_sort(self):
        """Return a topological ordering of the DAG"""
        if not self.is_dag():
            raise ValueError("Graph contains cycles")

        # Calculate in-degree for each node
        in_degree = {node: 0 for node in self.nodes}
        for parent, children in self.edges.items():
            for child in children:
                in_degree[child] += 1

        # Find nodes with in-degree 0
        queue = [node for node in self.nodes if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Decrease in-degree of children
            for child in self.edges.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

# DAG-based causal discovery
class CausalDiscovery:
    """Methods for discovering causal structure from data"""

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def pc_algorithm(self, data):
        """PC algorithm for causal discovery"""
        n_vars = data.shape[1]
        var_names = list(data.columns)

        # Step 1: Find skeleton (undirected graph)
        skeleton = self._find_skeleton(data, var_names)

        # Step 2: Orient edges
        dag = self._orient_edges(data, skeleton, var_names)

        return dag

    def _find_skeleton(self, data, var_names, max_conditioning_set_size=None):
        """Find the skeleton (undirected graph)"""
        n_vars = len(var_names)
        # Initialize complete graph
        skeleton = [[True] * n_vars for _ in range(n_vars)]

        # Remove self-loops
        for i in range(n_vars):
            skeleton[i][i] = False

        # Remove edges based on conditional independence tests
        max_size = max_conditioning_set_size or n_vars

        for level in range(max_size):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if not skeleton[i][j]:
                        continue

                    # Find conditioning sets of size 'level'
                    remaining_vars = [k for k in range(n_vars) if k != i and k != j and skeleton[i][k]]

                    if level >= len(remaining_vars):
                        continue

                    # Test independence conditioning on each subset
                    for conditioning_set in itertools.combinations(remaining_vars, level):
                        if self._test_conditional_independence(data, var_names, i, j, conditioning_set):
                            skeleton[i][j] = skeleton[j][i] = False
                            break

        return skeleton

    def _test_conditional_independence(self, data, var_names, i, j, conditioning_set):
        """Test conditional independence between variables i and j"""
        from scipy.stats import chi2_contingency

        # Get variable names
        var_i = var_names[i]
        var_j = var_names[j]
        conditioning_vars = [var_names[k] for k in conditioning_set]

        # Create contingency table
        if len(conditioning_vars) == 0:
            # Simple independence test
            crosstab = pd.crosstab(data[var_i], data[var_j])
        else:
            # Conditional independence test
            groups = data.groupby(conditioning_vars)
            test_results = []

            for name, group in groups:
                if len(group) < 5:  # Skip small groups
                    continue

                try:
                    crosstab = pd.crosstab(group[var_i], group[var_j])
                    chi2, p, dof, expected = chi2_contingency(crosstab)
                    test_results.append(p)
                except:
                    continue

            if len(test_results) == 0:
                return False

            # Combine p-values (Fisher's method)
            combined_statistic = -2 * sum(np.log(max(p, 1e-10)) for p in test_results)
            combined_p = 1 - stats.chi2.cdf(combined_statistic, 2 * len(test_results))
            return combined_p > self.alpha

        try:
            chi2, p, dof, expected = chi2_contingency(crosstab)
            return p > self.alpha  # Independent if p > alpha
        except:
            return False

    def _orient_edges(self, data, skeleton, var_names):
        """Orient edges to form a DAG"""
        n_vars = len(var_names)
        dag = CausalDAG()

        # Initialize DAG with nodes
        for var in var_names:
            dag.add_node(var)

        # Add edges based on skeleton
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if skeleton[i][j]:
                    dag.add_edge(var_names[i], var_names[j])

        # Apply orientation rules
        # Rule 1: Orient colliders
        oriented = True
        while oriented:
            oriented = False

            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if not skeleton[i][j]:
                        continue

                    # Find common neighbors
                    common_neighbors = []
                    for k in range(n_vars):
                        if k != i and k != j and skeleton[i][k] and skeleton[j][k]:
                            common_neighbors.append(k)

                    # Check if i and j are non-adjacent when conditioning on common neighbors
                    # If they become independent, orient towards forming a collider
                    for conditioning_set in itertools.combinations(common_neighbors, 2):
                        if self._test_conditional_independence(data, var_names, i, j, conditioning_set):
                            # Orient edge from i to j or j to i
                            # (Implementation would require more sophisticated rules)
                            pass

        return dag
```

---

## 6. Experimental Design

### Randomized Controlled Trials (RCTs)

```python
class RandomizedControlledTrial:
    """Design and analyze randomized controlled trials"""

    def __init__(self, outcomes, treatments, covariates=None):
        self.outcomes = outcomes
        self.treatments = treatments
        self.covariates = covariates or []
        self.analysis_plan = None

    def power_analysis(self, effect_size, alpha=0.05, power=0.8):
        """Calculate required sample size for desired power"""
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - alpha/2)  # Critical value for alpha
        z_beta = norm.ppf(power)  # Critical value for power

        # Sample size calculation for two-sample t-test
        n = 2 * (z_alpha + z_beta)**2 / effect_size**2

        return {
            'required_sample_size_per_group': int(np.ceil(n)),
            'total_required_sample_size': int(np.ceil(2 * n)),
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power
        }

    def randomization_scheme(self, n_units, n_treatments=2, allocation_ratio=1):
        """Create randomization scheme"""
        if n_treatments == 2:
            # Simple randomization
            treatment_assignments = np.random.binomial(1, 0.5, n_units)
        elif n_treatments > 2:
            # Block randomization for multiple treatments
            block_size = n_treatments * allocation_ratio
            n_blocks = n_units // block_size

            treatment_assignments = []
            for _ in range(n_blocks):
                block = list(range(n_treatments)) * allocation_ratio
                np.random.shuffle(block)
                treatment_assignments.extend(block)

            # Add remaining assignments
            remaining = n_units - len(treatment_assignments)
            additional = list(range(n_treatments)) * (remaining // n_treatments + 1)
            treatment_assignments.extend(additional[:remaining])

        return np.array(treatment_assignments)

    def analyze_rct(self, data):
        """Analyze randomized controlled trial"""
        from sklearn.linear_model import LinearRegression
        from scipy import stats

        # Basic analysis: difference in means
        treated_mean = data[data[self.treatments] == 1][self.outcomes].mean()
        control_mean = data[data[self.treatments] == 0][self.outcomes].mean()

        naive_ate = treated_mean - control_mean

        # T-test
        treated_values = data[data[self.treatments] == 1][self.outcomes]
        control_values = data[data[self.treatments] == 0][self.outcomes]

        t_stat, p_value = stats.ttest_ind(treated_values, control_values)

        # Regression adjustment
        X = data[self.treatments].values.reshape(-1, 1)
        if len(self.covariates) > 0:
            X = np.column_stack([X] + [data[covar].values.reshape(-1, 1) for covar in self.covariates])

        y = data[self.outcomes].values

        reg_model = LinearRegression()
        reg_model.fit(X, y)

        # Treatment effect is coefficient of treatment variable
        treatment_effect_regression = reg_model.coef_[0]

        # Covariate balance check
        balance_results = self._check_covariate_balance(data)

        return {
            'naive_ate': naive_ate,
            'regression_ate': treatment_effect_regression,
            't_statistic': t_stat,
            'p_value': p_value,
            'treated_mean': treated_mean,
            'control_mean': control_mean,
            'regression_model_score': reg_model.score(X, y),
            'covariate_balance': balance_results
        }

    def _check_covariate_balance(self, data):
        """Check balance of covariates between treatment groups"""
        balance_results = {}

        for covar in self.covariates:
            treated_mean = data[data[self.treatments] == 1][covar].mean()
            control_mean = data[data[self.treatments] == 0][covar].mean()

            # Standardized difference
            pooled_std = np.sqrt(
                (data[data[self.treatments] == 1][covar].var() +
                 data[data[self.treatments] == 0][covar].var()) / 2
            )

            std_diff = abs(treated_mean - control_mean) / pooled_std

            # T-test for difference in means
            treated_values = data[data[self.treatments] == 1][covar]
            control_values = data[data[self.treatments] == 0][covar]
            _, p_value = stats.ttest_ind(treated_values, control_values)

            balance_results[covar] = {
                'treated_mean': treated_mean,
                'control_mean': control_mean,
                'standardized_difference': std_diff,
                'p_value': p_value,
                'balanced': std_diff < 0.1 and p_value > 0.05
            }

        return balance_results

# Adaptive Trials
class AdaptiveTrial:
    """Adaptive randomized controlled trials"""

    def __init__(self, outcome_type='continuous'):
        self.outcome_type = outcome_type
        self.current_assignments = []
        self.current_outcomes = []
        self.adaptation_rule = None

    def set_adaptation_rule(self, adaptation_func):
        """Set rule for treatment assignment adaptation"""
        self.adaptation_rule = adaptation_func

    def assign_treatment(self, covariates, current_data):
        """Assign treatment based on current data and adaptation rule"""
        if self.adaptation_rule is None:
            # Default: equal allocation
            return np.random.binomial(1, 0.5)

        # Use adaptation rule to determine allocation probability
        allocation_prob = self.adaptation_rule(covariates, current_data)
        return np.random.binomial(1, allocation_prob)

    def interim_analysis(self, alpha=0.05):
        """Perform interim analysis for early stopping"""
        from scipy.stats import ttest_1samp

        if len(self.current_outcomes) < 10:
            return {'continue': True, 'reason': 'Insufficient data'}

        # Calculate current effect size
        treated_outcomes = [o for i, o in enumerate(self.current_outcomes)
                          if self.current_assignments[i] == 1]
        control_outcomes = [o for i, o in enumerate(self.current_outcomes)
                           if self.current_assignments[i] == 0]

        if len(treated_outcomes) == 0 or len(control_outcomes) == 0:
            return {'continue': True, 'reason': 'No outcomes in one group'}

        current_ate = np.mean(treated_outcomes) - np.mean(control_outcomes)

        # Test for early stopping
        t_stat, p_value = ttest_1samp([current_ate], 0)

        # O'Brien-Fleming boundaries (simplified)
        if p_value < alpha / (1 + len(self.current_assignments) / 10):
            return {
                'continue': False,
                'reason': 'Significant effect found',
                'current_effect': current_ate,
                'p_value': p_value
            }

        # Check for futility (no effect)
        sample_size = len(self.current_assignments)
        expected_effect = 0.5  # Assume minimal clinically important difference

        if abs(current_ate) < expected_effect * 0.2:  # Very small effect
            return {
                'continue': False,
                'reason': 'Futility - no meaningful effect detected',
                'current_effect': current_ate,
                'sample_size': sample_size
            }

        return {
            'continue': True,
            'reason': 'Continue trial',
            'current_effect': current_ate,
            'sample_size': sample_size
        }
```

---

## 7. Quasi-Experimental Methods

### Difference-in-Differences

```python
class DifferenceInDifferences:
    """Difference-in-differences estimation"""

    def __init__(self, outcome_col, treatment_col, time_col, unit_col):
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.time_col = time_col
        self.unit_col = unit_col

    def estimate_did(self, data):
        """Estimate difference-in-differences effect"""
        # Create treatment and post indicators
        data = data.copy()
        data['treated'] = data[self.treatment_col]
        data['post'] = (data[self.time_col] > data[self.time_col].min())
        data['treated_post'] = data['treated'] * data['post']

        # Estimate DID using OLS
        from sklearn.linear_model import LinearRegression

        X = data[['treated', 'post', 'treated_post']]
        y = data[self.outcome_col]

        did_model = LinearRegression()
        did_model.fit(X, y)

        did_effect = did_model.coef_[-1]  # Coefficient of treated_post

        # Calculate group-time specific effects
        effects = self._calculate_group_time_effects(data)

        # Test parallel trends assumption
        parallel_trends_test = self._test_parallel_trends(data)

        return {
            'did_effect': did_effect,
            'did_model_coefficients': did_model.coef_,
            'did_model_intercept': did_model.intercept_,
            'model_r_squared': did_model.score(X, y),
            'group_time_effects': effects,
            'parallel_trends_test': parallel_trends_test
        }

    def _calculate_group_time_effects(self, data):
        """Calculate effects for each group-time combination"""
        effects = {}

        for unit in data[self.unit_col].unique():
            for time in data[self.time_col].unique():
                unit_time_data = data[(data[self.unit_col] == unit) &
                                    (data[self.time_col] == time)]

                if len(unit_time_data) > 0:
                    treated = unit_time_data[self.treatment_col].iloc[0]
                    outcome = unit_time_data[self.outcome_col].iloc[0]

                    effects[(unit, time)] = {
                        'treated': treated,
                        'outcome': outcome
                    }

        return effects

    def _test_parallel_trends(self, data):
        """Test the parallel trends assumption"""
        from sklearn.linear_model import LinearRegression

        # Focus on pre-treatment periods
        pre_treatment_data = data[data['post'] == 0].copy()

        if len(pre_treatment_data) == 0:
            return {'test_possible': False, 'reason': 'No pre-treatment data'}

        # Test if treated and control groups have parallel trends in pre-period
        X = pre_treatment_data[['treated', 'time_trend']].values
        y = pre_treatment_data[self.outcome_col].values

        # Add time trend
        time_values = pre_treatment_data[self.time_col].values
        min_time = time_values.min()
        time_trend = time_values - min_time
        X[:, 1] = time_trend

        model = LinearRegression()
        model.fit(X, y)

        # Test if there's a significant difference in trends
        # The interaction term between treated and time trend
        time_trend_coef = model.coef_[1]

        return {
            'time_trend_coefficient': time_trend_coef,
            'test_possible': True,
            'interpretation': 'Small coefficients suggest parallel trends'
        }

# Instrumental Variables
class InstrumentalVariables:
    """Instrumental variables estimation"""

    def __init__(self, outcome_col, treatment_col, instrument_col, covariate_cols=None):
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.instrument_col = instrument_col
        self.covariate_cols = covariate_cols or []

    def iv_2sls(self, data):
        """Two-stage least squares estimation"""
        from sklearn.linear_model import LinearRegression

        y = data[self.outcome_col].values
        D = data[self.treatment_col].values
        Z = data[self.instrument_col].values

        X_covariates = None
        if len(self.covariate_cols) > 0:
            X_covariates = data[self.covariate_cols].values

        # Stage 1: Regress treatment on instrument
        X1 = Z.reshape(-1, 1)
        if X_covariates is not None:
            X1 = np.column_stack([Z, X_covariates])

        stage1_model = LinearRegression()
        stage1_model.fit(X1, D)

        # Predicted treatment values
        D_hat = stage1_model.predict(X1)

        # Stage 2: Regress outcome on predicted treatment
        X2 = D_hat.reshape(-1, 1)
        if X_covariates is not None:
            X2 = np.column_stack([D_hat, X_covariates])

        stage2_model = LinearRegression()
        stage2_model.fit(X2, y)

        # IV estimate is coefficient of predicted treatment
        iv_estimate = stage2_model.coef_[0]

        # Calculate fit statistics
        stage1_r2 = stage1_model.score(X1, D)
        stage2_r2 = stage2_model.score(X2, y)

        # Test instrument relevance (F-statistic)
        f_stat = self._calculate_f_statistic(D, Z, X_covariates)

        # Test overidentifying restrictions (if multiple instruments)
        sargan_test = self._sargan_test(y, D, Z, X_covariates)

        return {
            'iv_estimate': iv_estimate,
            'stage1_coefficients': stage1_model.coef_,
            'stage1_intercept': stage1_model.intercept_,
            'stage2_coefficients': stage2_model.coef_,
            'stage2_intercept': stage2_model.intercept_,
            'stage1_r2': stage1_r2,
            'stage2_r2': stage2_r2,
            'f_statistic': f_stat,
            'sargan_test': sargan_test
        }

    def _calculate_f_statistic(self, treatment, instrument, covariates=None):
        """Calculate F-statistic for instrument relevance"""
        from sklearn.linear_model import LinearRegression
        from scipy.stats import f

        X = instrument.reshape(-1, 1)
        if covariates is not None:
            X = np.column_stack([instrument, covariates])

        model = LinearRegression()
        model.fit(X, treatment)

        # Calculate F-statistic
        y_pred = model.predict(X)
        ss_res = np.sum((treatment - y_pred)**2)
        ss_tot = np.sum((treatment - np.mean(treatment))**2)

        r2 = 1 - (ss_res / ss_tot)
        n = len(treatment)
        k = X.shape[1]

        f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
        p_value = 1 - f.cdf(f_stat, k, n - k - 1)

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'relevant': f_stat > 10  # Rule of thumb: F > 10
        }

    def _sargan_test(self, outcome, treatment, instrument, covariates=None):
        """Sargan test for overidentifying restrictions"""
        # This is a simplified version - full implementation would be more complex
        return {
            'test_possible': False,
            'reason': 'Requires multiple instruments for overidentification'
        }
```

---

## 8. Causal Machine Learning

### Meta-Learners

```python
class SLearner:
    """S-learner (Single-learner) for heterogeneous treatment effects"""

    def __init__(self, base_learner):
        self.base_learner = base_learner
        self.fitted_model = None

    def fit(self, X, treatment, outcome):
        """Fit S-learner model"""
        # Create feature matrix with treatment indicator
        X_with_treatment = np.column_stack([X, treatment.reshape(-1, 1)])

        # Fit model on full dataset
        self.fitted_model = self.base_learner()
        self.fitted_model.fit(X_with_treatment, outcome)

        return self

    def predict_ite(self, X):
        """Predict individual treatment effects"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        # Predict under treatment and control
        X_treated = np.column_stack([X, np.ones((len(X), 1))])
        X_control = np.column_stack([X, np.zeros((len(X), 1))])

        y1_pred = self.fitted_model.predict(X_treated)
        y0_pred = self.fitted_model.predict(X_control)

        # Individual treatment effect
        ite = y1_pred - y0_pred

        return ite

class TLearner:
    """T-learner (Two-learner) for heterogeneous treatment effects"""

    def __init__(self, base_learner):
        self.base_learner = base_learner
        self.model_1 = None  # Model for treatment group
        self.model_0 = None  # Model for control group

    def fit(self, X, treatment, outcome):
        """Fit T-learner models"""
        # Split data by treatment
        treated_mask = treatment == 1
        control_mask = treatment == 0

        X_treated = X[treated_mask]
        y_treated = outcome[treated_mask]

        X_control = X[control_mask]
        y_control = outcome[control_mask]

        # Fit separate models for each group
        self.model_1 = self.base_learner()
        self.model_0 = self.base_learner()

        self.model_1.fit(X_treated, y_treated)
        self.model_0.fit(X_control, y_control)

        return self

    def predict_ite(self, X):
        """Predict individual treatment effects"""
        if self.model_1 is None or self.model_0 is None:
            raise ValueError("Models must be fitted first")

        # Predict outcomes for each group
        y1_pred = self.model_1.predict(X)
        y0_pred = self.model_0.predict(X)

        # Individual treatment effect
        ite = y1_pred - y0_pred

        return ite

class XLearner:
    """X-learner for heterogeneous treatment effects"""

    def __init__(self, base_learner):
        self.base_learner = base_learner
        self.models_1 = None  # Models for treatment group
        self.models_0 = None  # Models for control group
        self.treatment_effect_model = None

    def fit(self, X, treatment, outcome):
        """Fit X-learner"""
        # Step 1: Fit models for each group
        treated_mask = treatment == 1
        control_mask = treatment == 0

        X_treated = X[treated_mask]
        y_treated = outcome[treated_mask]

        X_control = X[control_mask]
        y_control = outcome[control_mask]

        self.models_1 = self.base_learner()
        self.models_0 = self.base_learner()

        self.models_1.fit(X_treated, y_treated)
        self.models_0.fit(X_control, y_control)

        # Step 2: Estimate treatment effects
        # For treated units: estimate effect using control model
        y0_hat_treated = self.models_0.predict(X_treated)
        treatment_effects_treated = y_treated - y0_hat_treated

        # For control units: estimate effect using treatment model
        y1_hat_control = self.models_1.predict(X_control)
        treatment_effects_control = y1_hat_control - y_control

        # Step 3: Fit treatment effect models
        self.treatment_effect_model_1 = self.base_learner()
        self.treatment_effect_model_0 = self.base_learner()

        self.treatment_effect_model_1.fit(X_treated, treatment_effects_treated)
        self.treatment_effect_model_0.fit(X_control, treatment_effects_control)

        return self

    def predict_ite(self, X):
        """Predict individual treatment effects"""
        # Predict treatment effects using both models
        effect_1 = self.treatment_effect_model_1.predict(X)
        effect_0 = self.treatment_effect_model_0.predict(X)

        # Combine estimates (could use propensity score weighting)
        ite = (effect_1 + effect_0) / 2

        return ite

# Causal Forest
class CausalForest:
    """Causal Random Forest for heterogeneous treatment effects"""

    def __init__(self, n_estimators=100, min_samples_leaf=10, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []

    def fit(self, X, treatment, outcome):
        """Fit causal forest"""
        self.n_features = X.shape[1]

        for i in range(self.n_estimators):
            # Bootstrap sample
            n_samples = len(X)
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)

            X_bootstrap = X[bootstrap_idx]
            treatment_bootstrap = treatment[bootstrap_idx]
            outcome_bootstrap = outcome[bootstrap_idx]

            # Grow tree
            tree = self._grow_tree(X_bootstrap, treatment_bootstrap, outcome_bootstrap)
            self.trees.append(tree)

        return self

    def _grow_tree(self, X, treatment, outcome, depth=0, max_depth=10):
        """Grow a single tree"""
        n_samples = len(X)

        # Stopping criteria
        if (n_samples < self.min_samples_leaf or
            depth >= max_depth or
            len(np.unique(treatment)) == 1):
            return self._create_leaf(outcome)

        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, treatment, outcome)

        if best_gain is None:
            return self._create_leaf(outcome)

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return self._create_leaf(outcome)

        # Recursively grow children
        left_child = self._grow_tree(X[left_mask], treatment[left_mask], outcome[left_mask], depth + 1, max_depth)
        right_child = self._grow_tree(X[right_mask], treatment[right_mask], outcome[right_mask], depth + 1, max_depth)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_child,
            'right': right_child,
            'depth': depth
        }

    def _find_best_split(self, X, treatment, outcome):
        """Find best split for current node"""
        best_gain = 0
        best_feature = None
        best_threshold = None

        n_features = int(np.sqrt(X.shape[1])) if self.max_features == 'sqrt' else X.shape[1]
        feature_indices = np.random.choice(X.shape[1], n_features, replace=False)

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                gain = self._calculate_gain(outcome[left_mask], outcome[right_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain if best_gain > 0 else None

    def _calculate_gain(self, left_outcome, right_outcome):
        """Calculate information gain from split"""
        # Calculate variance reduction
        total_variance = np.var(np.concatenate([left_outcome, right_outcome]))

        left_variance = np.var(left_outcome) if len(left_outcome) > 1 else 0
        right_variance = np.var(right_outcome) if len(right_outcome) > 1 else 0

        weighted_variance = (len(left_outcome) * left_variance + len(right_outcome) * right_variance) / (len(left_outcome) + len(right_outcome))

        gain = total_variance - weighted_variance
        return gain

    def _create_leaf(self, outcome):
        """Create leaf node"""
        return {
            'is_leaf': True,
            'mean_outcome': np.mean(outcome),
            'variance': np.var(outcome),
            'n_samples': len(outcome)
        }

    def predict_ite(self, X):
        """Predict individual treatment effects"""
        ite_predictions = []

        for tree in self.trees:
            ite_tree = self._predict_tree_ite(tree, X)
            ite_predictions.append(ite_tree)

        # Average across trees
        ite_predictions = np.array(ite_predictions)
        ite = np.mean(ite_predictions, axis=0)

        return ite

    def _predict_tree_ite(self, tree, X):
        """Predict ITE for a single tree"""
        # For causal forest, we need to modify the prediction logic
        # This is a simplified version
        predictions = np.zeros(len(X))

        for i, x in enumerate(X):
            predictions[i] = self._predict_tree_ite_single(tree, x)

        return predictions

    def _predict_tree_ite_single(self, tree, x):
        """Predict ITE for a single sample"""
        if tree['is_leaf']:
            return tree['mean_outcome']

        if x[tree['feature']] <= tree['threshold']:
            return self._predict_tree_ite_single(tree['left'], x)
        else:
            return self._predict_tree_ite_single(tree['right'], x)
```

---

## 9. Advanced Regression Methods

### Nonparametric Regression

```python
import numpy as np
from sklearn.neighbors import KernelDensity

class KernelRegression:
    """Nonparametric regression using kernel methods"""

    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.fitted = False

    def _gaussian_kernel(self, u):
        """Gaussian kernel function"""
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

    def _epanechnikov_kernel(self, u):
        """Epanechnikov kernel function"""
        return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

    def _uniform_kernel(self, u):
        """Uniform kernel function"""
        return np.where(np.abs(u) <= 1, 0.5, 0)

    def fit(self, X, y):
        """Fit kernel regression model"""
        self.X_train = X
        self.y_train = y
        self.fitted = True
        return self

    def predict(self, X):
        """Predict using kernel regression"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        predictions = []

        for x in X:
            # Calculate weights using kernel
            distances = np.linalg.norm(self.X_train - x, axis=1)

            if self.kernel == 'gaussian':
                weights = self._gaussian_kernel(distances / self.bandwidth)
            elif self.kernel == 'epanechnikov':
                weights = self._epanechnikov_kernel(distances / self.bandwidth)
            elif self.kernel == 'uniform':
                weights = self._uniform_kernel(distances / self.bandwidth)
            else:
                raise ValueError(f"Unknown kernel: {self.kernel}")

            # Normalize weights
            weights = weights / np.sum(weights)

            # Weighted average
            prediction = np.sum(weights * self.y_train)
            predictions.append(prediction)

        return np.array(predictions)

    def cross_validate_bandwidth(self, X, y, bandwidths):
        """Cross-validate to find optimal bandwidth"""
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=5)
        cv_scores = []

        for bandwidth in bandwidths:
            scores = []

            for train_idx, val_idx in kf.split(X):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]

                model = KernelRegression(bandwidth=bandwidth, kernel=self.kernel)
                model.fit(X_train_fold, y_train_fold)

                y_pred = model.predict(X_val_fold)
                mse = np.mean((y_val_fold - y_pred)**2)
                scores.append(mse)

            cv_scores.append(np.mean(scores))

        best_bandwidth = bandwidths[np.argmin(cv_scores)]
        best_score = np.min(cv_scores)

        return best_bandwidth, best_score, dict(zip(bandwidths, cv_scores))

# Generalized Additive Models
class GeneralizedAdditiveModel:
    """Generalized Additive Model (GAM)"""

    def __init__(self, link='identity', max_iter=100, learning_rate=0.1):
        self.link = link
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fitted = False

    def _link_function(self, eta):
        """Link function"""
        if self.link == 'identity':
            return eta
        elif self.link == 'log':
            return np.exp(eta)
        elif self.link == 'logit':
            return 1 / (1 + np.exp(-eta))
        else:
            raise ValueError(f"Unknown link function: {self.link}")

    def _inverse_link(self, mu):
        """Inverse link function"""
        if self.link == 'identity':
            return mu
        elif self.link == 'log':
            return np.log(mu)
        elif self.link == 'logit':
            return np.log(mu / (1 - mu))
        else:
            raise ValueError(f"Unknown link function: {self.link}")

    def fit(self, X, y, feature_functions=None):
        """Fit GAM using backfitting algorithm"""
        n_samples, n_features = X.shape

        # Initialize smoothing functions
        if feature_functions is None:
            # Default: use smoothing splines for each feature
            feature_functions = []
            for i in range(n_features):
                feature_functions.append(self._default_smoothing_function(X[:, i]))

        self.feature_functions = feature_functions

        # Initialize coefficients
        eta = np.mean(y) * np.ones(n_samples)

        for iteration in range(self.max_iter):
            old_eta = eta.copy()

            # Update each feature function
            for i in range(n_features):
                # Partial residuals
                partial_residuals = y - (eta - feature_functions[i](X[:, i]))

                # Update smoothing function
                feature_functions[i] = self._update_smoothing_function(
                    X[:, i], partial_residuals, feature_functions[i]
                )

                # Update eta
                eta = eta - feature_functions[i](X[:, i]) + \
                      np.mean(y - np.mean(y)) + \
                      np.sum([fj(X[:, i]) for fj in feature_functions], axis=0)

            # Check convergence
            if np.max(np.abs(eta - old_eta)) < 1e-6:
                break

        self.fitted = True
        self.eta_final = eta
        return self

    def _default_smoothing_function(self, x):
        """Default smoothing function using polynomial regression"""
        def smooth_func(x_new):
            # Fit polynomial of degree 3
            poly_features = np.column_stack([
                np.ones(len(x)),
                x,
                x**2,
                x**3
            ])

            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(poly_features, np.zeros(len(x)))

            poly_features_new = np.column_stack([
                np.ones(len(x_new)),
                x_new,
                x_new**2,
                x_new**3
            ])

            return model.predict(poly_features_new)

        return smooth_func

    def _update_smoothing_function(self, x, residuals, current_func):
        """Update smoothing function using backfitting"""
        # This is a simplified version
        # In practice, you'd use more sophisticated smoothing methods
        return current_func

    def predict(self, X):
        """Make predictions using fitted GAM"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        # Calculate linear predictor
        eta = 0
        for i in range(len(self.feature_functions)):
            eta += self.feature_functions[i](X[:, i])

        # Apply link function
        predictions = self._link_function(eta)

        return predictions
```

---

## 10. Time Series Causality

### Granger Causality

```python
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd

class GrangerCausalityTest:
    """Test for Granger causality in time series"""

    def __init__(self, max_lags=10, test_type='ssr_ftest'):
        self.max_lags = max_lags
        self.test_type = test_type

    def test_granger_causality(self, y, x, include_constant=True):
        """
        Test if x Granger-causes y

        Null hypothesis: x does not Granger-cause y
        Alternative hypothesis: x Granger-causes y
        """
        # Combine series
        data = pd.DataFrame({'y': y, 'x': x})

        # Test different lag lengths
        results = {}

        for lag in range(1, self.max_lags + 1):
            try:
                # Perform Granger causality test
                gc_result = grangercausalitytests(
                    data[['y', 'x']], maxlag=lag, verbose=False
                )

                # Extract test statistic and p-value
                if self.test_type in gc_result[lag][0]:
                    test_stat = gc_result[lag][0][self.test_type][0]
                    p_value = gc_result[lag][0][self.test_type][1]
                else:
                    # Use first available test if specified test not found
                    test_name = list(gc_result[lag][0].keys())[0]
                    test_stat = gc_result[lag][0][test_name][0]
                    p_value = gc_result[lag][0][test_name][1]

                results[lag] = {
                    'test_statistic': test_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

            except Exception as e:
                results[lag] = {
                    'error': str(e)
                }

        return results

    def select_optimal_lag(self, granger_results, criterion='aic'):
        """Select optimal lag length using information criteria"""
        valid_lags = [lag for lag, result in granger_results.items()
                     if 'error' not in result]

        if len(valid_lags) == 0:
            return None

        # Simple approach: use lag with minimum p-value
        min_p_lag = min(valid_lags, key=lambda lag: granger_results[lag]['p_value'])

        return min_p_lag

    def bidirectional_granger_test(self, y, x):
        """Test for bidirectional Granger causality"""
        # Test x -> y
        xy_results = self.test_granger_causality(y, x)

        # Test y -> x
        yx_results = self.test_granger_causality(x, y)

        # Select optimal lags
        xy_optimal_lag = self.select_optimal_lag(xy_results)
        yx_optimal_lag = self.select_optimal_lag(yx_results)

        return {
            'x_to_y': {
                'results': xy_results,
                'optimal_lag': xy_optimal_lag,
                'significant': xy_optimal_lag and xy_results[xy_optimal_lag]['significant']
            },
            'y_to_x': {
                'results': yx_results,
                'optimal_lag': yx_optimal_lag,
                'significant': yx_optimal_lag and yx_results[yx_optimal_lag]['significant']
            }
        }

# Spectral Causality
class SpectralCausality:
    """Frequency domain approach to causality testing"""

    def __init__(self):
        pass

    def compute_transfer_function(self, x, y, frequencies=None):
        """Compute frequency domain transfer function"""
        from scipy.fft import fft, ifft

        if frequencies is None:
            frequencies = np.fft.fftfreq(len(x))

        # Compute FFTs
        X_fft = fft(x)
        Y_fft = fft(y)

        # Cross-spectral density
        cross_spectrum = Y_fft * np.conj(X_fft)

        # Power spectral density
        x_psd = np.abs(X_fft)**2
        y_psd = np.abs(Y_fft)**2

        # Transfer function
        transfer_function = cross_spectrum / x_psd

        return frequencies, transfer_function

    def coherence_function(self, x, y, frequencies=None):
        """Compute coherence function"""
        from scipy.fft import fft

        if frequencies is None:
            frequencies = np.fft.fftfreq(len(x))

        # Compute FFTs
        X_fft = fft(x)
        Y_fft = fft(y)

        # Coherence calculation
        cross_spectrum = Y_fft * np.conj(X_fft)
        x_psd = np.abs(X_fft)**2
        y_psd = np.abs(Y_fft)**2

        coherence = np.abs(cross_spectrum)**2 / (x_psd * y_psd)

        return frequencies, coherence

    def granger_causality_spectral(self, x, y, frequency_range=(0, np.pi)):
        """Compute spectral Granger causality"""
        from scipy import signal

        # Compute transfer function
        frequencies, transfer_func = self.compute_transfer_function(x, y)

        # Magnitude squared coherence
        frequencies, coherence = self.coherence_function(x, y)

        # Find frequency in specified range
        freq_mask = (frequencies >= frequency_range[0]) & (frequencies <= frequency_range[1])

        # Calculate causality measure
        causality_measure = np.abs(transfer_func[freq_mask])**2 * (1 - coherence[freq_mask])

        return {
            'frequencies': frequencies[freq_mask],
            'transfer_function': transfer_func[freq_mask],
            'coherence': coherence[freq_mask],
            'causality_measure': causality_measure
        }
```

---

## 11. A/B Testing & Experimentation

### Advanced A/B Testing Methods

```python
class AdvancedABTest:
    """Advanced A/B testing with multiple variants and covariates"""

    def __init__(self, outcome_col, variant_col, covariate_cols=None):
        self.outcome_col = outcome_col
        self.variant_col = variant_col
        self.covariate_cols = covariate_cols or []
        self.variants = None
        self.results = {}

    def calculate_sample_size(self, effect_size, alpha=0.05, power=0.8, variants=2):
        """Calculate required sample size for multi-variant test"""
        from scipy.stats import norm

        # For multiple variants, use Bonferroni correction
        alpha_corrected = alpha / (variants - 1)

        z_alpha = norm.ppf(1 - alpha_corrected / 2)
        z_beta = norm.ppf(power)

        # Sample size per group
        n_per_group = 2 * (z_alpha + z_beta)**2 / effect_size**2

        # Total sample size
        total_n = variants * n_per_group

        return {
            'n_per_group': int(np.ceil(n_per_group)),
            'total_n': int(np.ceil(total_n)),
            'alpha_corrected': alpha_corrected
        }

    def multi_variant_analysis(self, data):
        """Analyze multi-variant experiment"""
        from scipy.stats import f_oneway
        from sklearn.linear_model import LinearRegression

        variants = data[self.variant_col].unique()
        variant_means = {}

        # Calculate means for each variant
        for variant in variants:
            variant_data = data[data[self.variant_col] == variant]
            variant_means[variant] = variant_data[self.outcome_col].mean()

        # ANOVA test
        variant_groups = [data[data[self.variant_col] == variant][self.outcome_col].values
                         for variant in variants]

        f_stat, p_value = f_oneway(*variant_groups)

        # Post-hoc analysis: pairwise comparisons
        pairwise_results = {}
        for i, variant1 in enumerate(variants):
            for j, variant2 in enumerate(variants):
                if i < j:
                    comparison_name = f"{variant1}_vs_{variant2}"

                    group1 = data[data[self.variant_col] == variant1][self.outcome_col]
                    group2 = data[data[self.variant_col] == variant2][self.outcome_col]

                    # T-test for pairwise comparison
                    from scipy.stats import ttest_ind
                    t_stat, t_p_value = ttest_ind(group1, group2)

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(group1) - 1) * group1.var() +
                                        (len(group2) - 1) * group2.var()) /
                                       (len(group1) + len(group2) - 2))
                    cohens_d = (group1.mean() - group2.mean()) / pooled_std

                    pairwise_results[comparison_name] = {
                        'mean_diff': group1.mean() - group2.mean(),
                        't_statistic': t_stat,
                        'p_value': t_p_value,
                        'cohens_d': cohens_d,
                        'significant': t_p_value < (0.05 / len(pairwise_results))  # Bonferroni correction
                    }

        # Covariate adjustment
        adjusted_results = self._covariate_adjusted_analysis(data)

        return {
            'variant_means': variant_means,
            'anova_results': {
                'f_statistic': f_stat,
                'p_value': p_value,
                'overall_significant': p_value < 0.05
            },
            'pairwise_comparisons': pairwise_results,
            'covariate_adjusted': adjusted_results
        }

    def _covariate_adjusted_analysis(self, data):
        """Perform covariate-adjusted analysis"""
        from sklearn.linear_model import LinearRegression

        if len(self.covariate_cols) == 0:
            return None

        # Create design matrix
        X = pd.get_dummies(data[self.variant_col], prefix='variant')

        # Add covariates
        for covar in self.covariate_cols:
            if data[covar].dtype in ['object', 'category']:
                # One-hot encode categorical covariates
                covar_dummies = pd.get_dummies(data[covar], prefix=covar)
                X = pd.concat([X, covar_dummies], axis=1)
            else:
                # Numeric covariate
                X[covar] = data[covar]

        y = data[self.outcome_col]

        # Fit regression model
        model = LinearRegression()
        model.fit(X, y)

        # Results for each variant (compare to reference group)
        reference_group = f"variant_{data[self.variant_col].iloc[0]}"
        variant_coeffs = {}

        for col in X.columns:
            if col.startswith('variant_') and col != reference_group:
                variant_name = col.replace('variant_', '')
                variant_coeffs[variant_name] = {
                    'coefficient': model.coef_[X.columns.get_loc(col)],
                    'p_value': None  # Would need standard errors for p-values
                }

        return {
            'model_r2': model.score(X, y),
            'variant_effects': variant_coeffs,
            'reference_group': reference_group
        }

    def sequential_testing(self, data, interim_analysis_points):
        """Perform sequential testing with multiple analyses"""
        results = []

        for n_obs in interim_analysis_points:
            if n_obs > len(data):
                continue

            # Use first n_obs observations
            interim_data = data.iloc[:n_obs]

            # Perform analysis
            analysis_result = self.multi_variant_analysis(interim_data)

            # Check stopping criteria
            stopping_decision = self._check_sequential_stopping(analysis_result)

            results.append({
                'n_observations': n_obs,
                'analysis_result': analysis_result,
                'stopping_decision': stopping_decision
            })

            if stopping_decision['should_stop']:
                break

        return results

    def _check_sequential_stopping(self, analysis_result):
        """Check if trial should be stopped at interim analysis"""
        # Simplified stopping rule
        overall_significant = analysis_result['anova_results']['overall_significant']

        if overall_significant:
            return {
                'should_stop': True,
                'reason': 'Significant effect detected',
                'recommendation': 'Stop for efficacy'
            }
        else:
            # Check for futility (very small effect sizes)
            max_effect = max([
                abs(comp['mean_diff'])
                for comp in analysis_result['pairwise_comparisons'].values()
            ])

            if max_effect < 0.1:  # Assume 0.1 is minimal important difference
                return {
                    'should_stop': True,
                    'reason': 'No meaningful effect detected',
                    'recommendation': 'Stop for futility'
                }

        return {
            'should_stop': False,
            'reason': 'Continue trial',
            'recommendation': 'Collect more data'
        }

    def heterogeneous_effects_analysis(self, data, effect_modifiers):
        """Analyze heterogeneous treatment effects across subgroups"""
        from sklearn.linear_model import LinearRegression

        results = {}

        for modifier in effect_modifiers:
            if modifier not in data.columns:
                continue

            modifier_values = data[modifier].unique()
            subgroup_results = {}

            for value in modifier_values:
                subgroup_data = data[data[modifier] == value]

                if len(subgroup_data) < 20:  # Skip small subgroups
                    continue

                subgroup_analysis = self.multi_variant_analysis(subgroup_data)
                subgroup_results[value] = subgroup_analysis

            results[modifier] = subgroup_results

        return results
```

---

## 12. Causal AI Applications

### Real-World Applications

#### Healthcare: Treatment Effect Heterogeneity

```python
class HealthcareCausalAnalysis:
    """Causal analysis for healthcare applications"""

    def __init__(self):
        pass

    def analyze_treatment_heterogeneity(self, patient_data, treatment_col, outcome_col,
                                      subgroup_vars):
        """Analyze how treatment effects vary across patient subgroups"""

        results = {}

        for subgroup_var in subgroup_vars:
            if subgroup_var not in patient_data.columns:
                continue

            subgroup_values = patient_data[subgroup_var].unique()
            heterogeneity_results = {}

            for value in subgroup_values:
                subgroup_data = patient_data[patient_data[subgroup_var] == value]

                if len(subgroup_data) < 50:  # Ensure sufficient sample size
                    continue

                # Calculate treatment effect for this subgroup
                treated = subgroup_data[subgroup_data[treatment_col] == 1]
                control = subgroup_data[subgroup_data[treatment_col] == 0]

                if len(treated) == 0 or len(control) == 0:
                    continue

                treated_mean = treated[outcome_col].mean()
                control_mean = control[outcome_col].mean()

                # Statistical test
                from scipy.stats import ttest_ind
                t_stat, p_value = ttest_ind(
                    treated[outcome_col],
                    control[outcome_col]
                )

                # Effect size (Cohen's d)
                pooled_std = np.sqrt((treated[outcome_col].var() + control[outcome_col].var()) / 2)
                cohens_d = (treated_mean - control_mean) / pooled_std

                heterogeneity_results[value] = {
                    'treatment_effect': treated_mean - control_mean,
                    'treated_mean': treated_mean,
                    'control_mean': control_mean,
                    'sample_size_treated': len(treated),
                    'sample_size_control': len(control),
                    'p_value': p_value,
                    'effect_size': cohens_d,
                    'clinically_significant': abs(cohens_d) > 0.5  # Cohen's d > 0.5
                }

            results[subgroup_var] = heterogeneity_results

        return results

    def estimate_dose_response(self, dose_data, dose_col, outcome_col, confounders=None):
        """Estimate dose-response relationship"""
        from sklearn.ensemble import RandomForestRegressor
        from scipy.interpolate import UnivariateSpline

        X = dose_data[[dose_col]].values
        y = dose_data[outcome_col].values

        if confounders:
            confounder_data = dose_data[confounders].values
            X = np.column_stack([X, confounder_data])

        # Fit dose-response model
        if confounders:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
        else:
            # Use spline for simple dose-response
            sorted_indices = np.argsort(dose_data[dose_col])
            doses_sorted = dose_data[dose_col].iloc[sorted_indices]
            outcomes_sorted = y[sorted_indices]

            # Fit smoothing spline
            spline = UnivariateSpline(doses_sorted, outcomes_sorted, s=len(doses_sorted))

        # Generate dose-response curve
        dose_range = np.linspace(dose_data[dose_col].min(), dose_data[dose_col].max(), 100)

        if confounders:
            # Average over confounders (simplified)
            # In practice, you'd average over observed confounders
            response_curve = np.zeros(len(dose_range))
            for i, dose in enumerate(dose_range):
                X_test = np.column_stack([np.ones(len(confounder_data)) * dose, confounder_data])
                response_curve[i] = model.predict(X_test).mean()
        else:
            response_curve = spline(dose_range)

        return {
            'dose_range': dose_range,
            'response_curve': response_curve,
            'optimal_dose': self._find_optimal_dose(dose_range, response_curve)
        }

    def _find_optimal_dose(self, doses, responses):
        """Find optimal dose (maximize response while minimizing side effects)"""
        # Simplified approach: find dose with maximum response
        max_response_idx = np.argmax(responses)
        optimal_dose = doses[max_response_idx]

        return {
            'optimal_dose': optimal_dose,
            'max_response': responses[max_response_idx],
            'dose_response_curve': list(zip(doses, responses))
        }
```

#### Finance: Causal Analysis for Trading

```python
class FinanceCausalAnalysis:
    """Causal analysis for financial applications"""

    def __init__(self):
        pass

    def analyze_market_events(self, price_data, event_data, event_type_col='event_type'):
        """Analyze causal effects of market events on prices"""

        results = {}

        for event_type in event_data[event_type_col].unique():
            event_subset = event_data[event_data[event_type_col] == event_type]
            event_results = {}

            for _, event in event_subset.iterrows():
                event_date = event['event_date']
                event_window = (event_date - pd.Timedelta(days=5),
                              event_date + pd.Timedelta(days=5))

                # Extract price window around event
                window_prices = price_data[
                    (price_data.index >= event_window[0]) &
                    (price_data.index <= event_window[1])
                ]

                if len(window_prices) < 10:  # Insufficient data
                    continue

                # Calculate abnormal returns
                pre_event_prices = window_prices[
                    window_prices.index < event_date
                ]['price']

                post_event_prices = window_prices[
                    window_prices.index >= event_date
                ]['price']

                if len(pre_event_prices) == 0 or len(post_event_prices) == 0:
                    continue

                # Calculate returns
                pre_returns = pre_event_prices.pct_change().dropna()
                post_returns = post_event_prices.pct_change().dropna()

                # Market model adjustment (simplified)
                # In practice, you'd use market index for adjustment
                abnormal_returns = post_returns - pre_returns.mean()

                # Cumulative abnormal returns
                cumulative_abnormal_returns = abnormal_returns.cumsum()

                # Statistical test
                from scipy.stats import ttest_1samp
                t_stat, p_value = ttest_1samp(abnormal_returns, 0)

                event_results[event_date] = {
                    'cumulative_abnormal_return': cumulative_abnormal_returns.iloc[-1],
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'abnormal_returns': abnormal_returns.tolist(),
                    'cumulative_abnormal_returns': cumulative_abnormal_returns.tolist()
                }

            results[event_type] = event_results

        return results

    def identify_causal_factors(self, return_data, factor_data):
        """Identify causal factors affecting returns"""
        from sklearn.linear_model import Lasso
        from scipy.stats import pearsonr

        results = {}

        # Correlation analysis
        correlations = {}
        for factor in factor_data.columns:
            if factor in return_data.columns:
                corr, p_value = pearsonr(return_data['return'], factor_data[factor])
                correlations[factor] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

        # Lasso regression for factor selection
        X = factor_data.values
        y = return_data['return'].values

        # Remove NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 20:
            return {'correlations': correlations, 'factor_selection': None}

        # Lasso regression
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_clean, y_clean)

        # Selected factors (non-zero coefficients)
        selected_factors = []
        factor_names = list(factor_data.columns)

        for i, coef in enumerate(lasso.coef_):
            if abs(coef) > 1e-6:  # Non-zero coefficient
                selected_factors.append({
                    'factor': factor_names[i],
                    'coefficient': coef,
                    'selected': True
                })

        results['correlations'] = correlations
        results['factor_selection'] = {
            'selected_factors': selected_factors,
            'model_score': lasso.score(X_clean, y_clean),
            'alpha_used': 0.01
        }

        return results
```

#### Education: Learning Effectiveness

```python
class EducationCausalAnalysis:
    """Causal analysis for educational interventions"""

    def __init__(self):
        pass

    def analyze_teaching_method_effectiveness(self, student_data, teaching_method_col,
                                           outcome_col, covariate_cols=None):
        """Analyze effectiveness of different teaching methods"""

        if covariate_cols is None:
            covariate_cols = ['prior_score', 'age', 'socioeconomic_status']

        results = {}

        methods = student_data[teaching_method_col].unique()

        # Compare all pairs of teaching methods
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i >= j:
                    continue

                method1_data = student_data[student_data[teaching_method_col] == method1]
                method2_data = student_data[teaching_method_col] == method2

                # Propensity score matching
                matched_pairs = self._propensity_score_matching(
                    method1_data, method2_data, covariate_cols
                )

                if len(matched_pairs) == 0:
                    continue

                # Calculate treatment effect
                treatment_effects = []
                for pair in matched_pairs:
                    outcome1 = pair[0][outcome_col]
                    outcome2 = pair[1][outcome_col]
                    treatment_effects.append(outcome1 - outcome2)

                # Statistical test
                from scipy.stats import ttest_1samp
                t_stat, p_value = ttest_1samp(treatment_effects, 0)

                comparison_name = f"{method1}_vs_{method2}"
                results[comparison_name] = {
                    'treatment_effect': np.mean(treatment_effects),
                    'standard_error': np.std(treatment_effects) / np.sqrt(len(treatment_effects)),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'confidence_interval': self._calculate_confidence_interval(treatment_effects),
                    'num_matched_pairs': len(matched_pairs),
                    'effect_size': np.mean(treatment_effects) / np.std(treatment_effects)
                }

        return results

    def _propensity_score_matching(self, group1_data, group2_data, covariate_cols):
        """Perform propensity score matching between two groups"""
        from sklearn.linear_model import LogisticRegression

        # Combine data and create treatment indicator
        group1_data = group1_data.copy()
        group2_data = group2_data.copy()

        group1_data['treatment'] = 1
        group2_data['treatment'] = 0

        combined_data = pd.concat([group1_data, group2_data], ignore_index=True)

        # Estimate propensity scores
        X = combined_data[covariate_cols].values
        treatment = combined_data['treatment'].values

        ps_model = LogisticRegression()
        ps_model.fit(X, treatment)
        propensity_scores = ps_model.predict_proba(X)[:, 1]

        # Separate groups with propensity scores
        group1_ps = propensity_scores[treatment == 1]
        group2_ps = propensity_scores[treatment == 0]

        # Greedy matching
        matched_pairs = []
        used_indices = set()

        for i, ps1 in enumerate(group1_ps):
            if i in used_indices:
                continue

            # Find closest match in group 2
            ps_differences = np.abs(group2_ps - ps1)
            closest_idx = np.argmin(ps_differences)

            if closest_idx not in used_indices and ps_differences[closest_idx] < 0.1:  # Caliper
                matched_pairs.append((group1_data.iloc[i], group2_data.iloc[closest_idx]))
                used_indices.add(i)
                used_indices.add(closest_idx)

        return matched_pairs

    def _calculate_confidence_interval(self, values, confidence_level=0.95):
        """Calculate confidence interval for a list of values"""
        from scipy.stats import t

        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)

        # t-critical value
        alpha = 1 - confidence_level
        t_critical = t.ppf(1 - alpha/2, n - 1)

        margin_error = t_critical * (std / np.sqrt(n))

        return (mean - margin_error, mean + margin_error)
```

---

## Summary

This comprehensive guide to Advanced Statistical Methods & Causal Inference covers:

### Key Topics Covered:

1. **Advanced Probability Theory** - Stable distributions, skewed distributions
2. **Bayesian Statistics** - Posterior inference, model selection
3. **Causal Inference Fundamentals** - Potential outcomes, treatment effects
4. **Potential Outcomes Framework** - Rubin causal model, assumptions
5. **Directed Acyclic Graphs** - Causal discovery, graphical models
6. **Experimental Design** - RCTs, adaptive trials, power analysis
7. **Quasi-Experimental Methods** - Difference-in-differences, IV estimation
8. **Causal Machine Learning** - Meta-learners, causal forests
9. **Advanced Regression** - Nonparametric regression, GAMs
10. **Time Series Causality** - Granger causality, spectral methods
11. **A/B Testing** - Multi-variant testing, sequential analysis
12. **Real-World Applications** - Healthcare, finance, education

### Practical Applications:

- Treatment effect heterogeneity analysis
- Policy evaluation and impact assessment
- A/B testing and experimentation
- Financial market analysis
- Educational intervention evaluation
- Healthcare treatment optimization

### Methodologies:

- Potential outcomes framework
- Directed acyclic graphs
- Randomized experiments
- Quasi-experimental designs
- Machine learning approaches
- Bayesian inference

This module provides the statistical and causal inference foundation needed for rigorous data analysis and evidence-based decision making across AI/ML applications.
