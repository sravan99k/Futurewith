# Advanced Statistical Methods & Causal Inference - Practice Exercises

## Table of Contents

1. [Bayesian Statistics Practice](#bayesian-statistics-practice)
2. [Causal Inference Hands-on](#causal-inference-hands-on)
3. [Experimental Design Practice](#experimental-design-practice)
4. [Quasi-experimental Methods](#quasi-experimental-methods)
5. [Causal Machine Learning](#causal-machine-learning)
6. [Advanced Regression Methods](#advanced-regression-methods)
7. [Time Series Causality](#time-series-causality)
8. [A/B Testing Implementation](#ab-testing-implementation)
9. [Real-world Applications](#real-world-applications)
10. [Integration Projects](#integration-projects)

---

## 1. Bayesian Statistics Practice

### Exercise 1.1: Bayesian Parameter Estimation

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pymc3 as pm
import arviz as az

# Generate synthetic data
np.random.seed(42)
n_samples = 100
true_theta = 2.5
sigma = 1.0

# Generate data from normal distribution
data = np.random.normal(true_theta, sigma, n_samples)

print(f"Generated {n_samples} samples from Normal({true_theta}, {sigma}^2)")
print(f"Sample mean: {np.mean(data):.3f}")
print(f"Sample std: {np.std(data, ddof=1):.3f}")

# Exercise: Implement Bayesian estimation
class BayesianParameterEstimator:
    def __init__(self):
        self.trace = None
        self.posterior_samples = None

    def bayesian_normal_mean(self, data, prior_mu=0, prior_sigma=10, known_sigma=1):
        """Bayesian estimation of normal mean with known variance"""

        # Posterior parameters
        n = len(data)
        x_bar = np.mean(data)

        # Posterior precision
        posterior_precision = 1/prior_sigma**2 + n/known_sigma**2

        # Posterior mean
        posterior_mean = (prior_mu/prior_sigma**2 + n*x_bar/known_sigma**2) / posterior_precision

        # Posterior variance
        posterior_var = 1/posterior_precision

        # Generate posterior samples
        posterior_samples = np.random.normal(posterior_mean, np.sqrt(posterior_var), 10000)

        return {
            'posterior_mean': posterior_mean,
            'posterior_var': posterior_var,
            'posterior_std': np.sqrt(posterior_var),
            'posterior_samples': posterior_samples,
            'credible_interval': np.percentile(posterior_samples, [2.5, 97.5])
        }

    def bayesian_normal_variance(self, data, prior_alpha=0.1, prior_beta=0.1):
        """Bayesian estimation of normal variance"""

        n = len(data)
        sample_var = np.var(data, ddof=1)

        # Posterior parameters for inverse gamma
        posterior_alpha = prior_alpha + n/2
        posterior_beta = prior_beta + (n-1)*sample_var/2

        # Generate posterior samples
        posterior_samples = 1/np.random.gamma(posterior_alpha, 1/posterior_beta, 10000)

        return {
            'posterior_alpha': posterior_alpha,
            'posterior_beta': posterior_beta,
            'posterior_mean': posterior_beta/(posterior_alpha-1) if posterior_alpha > 1 else np.nan,
            'posterior_samples': posterior_samples,
            'credible_interval': np.percentile(posterior_samples, [2.5, 97.5])
        }

# Test the implementation
estimator = BayesianParameterEstimator()

# Estimate mean
mean_result = estimator.bayesian_normal_mean(data, prior_mu=0, prior_sigma=10, known_sigma=sigma)
print(f"\nBayesian Mean Estimation:")
print(f"True value: {true_theta}")
print(f"Posterior mean: {mean_result['posterior_mean']:.3f}")
print(f"Posterior std: {mean_result['posterior_std']:.3f}")
print(f"95% Credible Interval: [{mean_result['credible_interval'][0]:.3f}, {mean_result['credible_interval'][1]:.3f}]")

# Estimate variance
var_result = estimator.bayesian_normal_variance(data)
print(f"\nBayesian Variance Estimation:")
print(f"True value: {sigma**2}")
print(f"Posterior mean: {var_result['posterior_mean']:.3f}")
print(f"95% Credible Interval: [{var_result['credible_interval'][0]:.3f}, {var_result['credible_interval'][1]:.3f}]")

# Plot posterior distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Posterior for mean
ax1.hist(mean_result['posterior_samples'], bins=50, density=True, alpha=0.7, color='skyblue')
ax1.axvline(true_theta, color='red', linestyle='--', label='True value')
ax1.axvline(mean_result['posterior_mean'], color='orange', linestyle='-', label='Posterior mean')
ax1.set_xlabel('Parameter Value')
ax1.set_ylabel('Density')
ax1.set_title('Posterior Distribution of Mean')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Posterior for variance
ax2.hist(var_result['posterior_samples'], bins=50, density=True, alpha=0.7, color='lightgreen')
ax2.axvline(sigma**2, color='red', linestyle='--', label='True value')
ax2.axvline(var_result['posterior_mean'], color='orange', linestyle='-', label='Posterior mean')
ax2.set_xlabel('Parameter Value')
ax2.set_ylabel('Density')
ax2.set_title('Posterior Distribution of Variance')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Exercise 1.2: Bayesian Model Selection with PyMC3

```python
# Exercise: Implement Bayesian model selection
class BayesianModelSelector:
    def __init__(self):
        self.models = {}
        self.evidences = {}

    def fit_models(self, data, models_spec):
        """Fit multiple models and compute model evidence"""

        for model_name, spec in models_spec.items():
            print(f"Fitting model: {model_name}")

            with pm.Model() as model:
                # Define priors based on specification
                priors = {}
                for param, prior_spec in spec['priors'].items():
                    if prior_spec['type'] == 'normal':
                        priors[param] = pm.Normal(
                            param,
                            mu=prior_spec['mu'],
                            sigma=prior_spec['sigma']
                        )
                    elif prior_spec['type'] == 'uniform':
                        priors[param] = pm.Uniform(param, lower=prior_spec['lower'], upper=prior_spec['upper'])
                    elif prior_spec['type'] == 'half_normal':
                        priors[param] = pm.HalfNormal(param, sigma=prior_spec['sigma'])

                # Define likelihood
                if spec['likelihood'] == 'normal':
                    if 'sigma' in priors:
                        likelihood = pm.Normal('likelihood', mu=priors['mu'], sigma=priors['sigma'], observed=data)
                    else:
                        # Estimate sigma
                        sigma_est = pm.HalfNormal('sigma', sigma=1)
                        likelihood = pm.Normal('likelihood', mu=priors['mu'], sigma=sigma_est, observed=data)
                elif spec['likelihood'] == 'student_t':
                    if 'sigma' in priors:
                        likelihood = pm.StudentT('likelihood', nu=spec.get('nu', 2), mu=priors['mu'], sigma=priors['sigma'], observed=data)
                    else:
                        sigma_est = pm.HalfNormal('sigma', sigma=1)
                        likelihood = pm.StudentT('likelihood', nu=spec.get('nu', 2), mu=priors['mu'], sigma=sigma_est, observed=data)

                # Sample from posterior
                trace = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=42)

                # Compute model evidence using LOO
                model_evidence = pm.loo(trace, model)

                self.models[model_name] = {
                    'trace': trace,
                    'model': model,
                    'evidence': model_evidence
                }

        # Calculate Bayes factors
        self.calculate_bayes_factors()

        return self.models

    def calculate_bayes_factors(self):
        """Calculate Bayes factors between models"""
        model_names = list(self.models.keys())
        loo_values = [self.models[name]['evidence'].loo for name in model_names]

        # Calculate relative Bayes factors
        max_loo = max(loo_values)
        self.bayes_factors = {}

        for i, model1 in enumerate(model_names):
            self.bayes_factors[model1] = {}
            for j, model2 in enumerate(model_names):
                if i != j:
                    bf = np.exp(loo_values[j] - max_loo) / np.exp(loo_values[i] - max_loo)
                    self.bayes_factors[model1][model2] = bf

        # Store model ranking
        sorted_models = sorted(zip(model_names, loo_values), key=lambda x: x[1], reverse=True)
        self.model_ranking = [{'model': name, 'loo': loo} for name, loo in sorted_models]

    def get_best_model(self):
        """Get the best model based on evidence"""
        return self.model_ranking[0]['model']

    def model_comparison_summary(self):
        """Get summary of model comparison"""
        print("Model Comparison Summary:")
        print("=" * 50)

        for rank, model_info in enumerate(self.model_ranking, 1):
            model_name = model_info['model']
            loo_value = model_info['loo']
            p_loo = self.models[model_name]['evidence'].p_loo

            # Calculate weight
            max_loo = self.model_ranking[0]['loo']
            weight = np.exp(loo_value - max_loo)

            print(f"Rank {rank}: {model_name}")
            print(f"  LOO: {loo_value:.2f}")
            print(f"  p_loo: {p_loo:.2f}")
            print(f"  Weight: {weight:.3f}")
            print()

# Generate data for model selection
np.random.seed(42)
n = 200

# True model: y ~ Normal(2 + 3x, 1)
x = np.random.uniform(0, 10, n)
true_mu = 2 + 3*x
y_normal = true_mu + np.random.normal(0, 1, n)

# Outlier-contaminated data
y_outliers = true_mu + np.random.normal(0, 1, n)
outlier_indices = np.random.choice(n, size=int(0.05*n), replace=False)
y_outliers[outlier_indices] += np.random.normal(0, 5, len(outlier_indices))

# Test model selection
selector = BayesianModelSelector()

# Define competing models
models_spec = {
    'normal_linear': {
        'priors': {
            'alpha': {'type': 'normal', 'mu': 0, 'sigma': 10},
            'beta': {'type': 'normal', 'mu': 0, 'sigma': 10}
        },
        'likelihood': 'normal'
    },
    'student_t_linear': {
        'priors': {
            'alpha': {'type': 'normal', 'mu': 0, 'sigma': 10},
            'beta': {'type': 'normal', 'mu': 0, 'sigma': 10}
        },
        'likelihood': 'student_t',
        'nu': 2
    }
}

# Test with normal data
print("Testing with normal data:")
models_normal = selector.fit_models(y_normal, models_spec)
best_model_normal = selector.get_best_model()
print(f"Best model for normal data: {best_model_normal}")

# Test with outlier data
print("\nTesting with outlier data:")
selector_outliers = BayesianModelSelector()
models_outliers = selector_outliers.fit_models(y_outliers, models_spec)
best_model_outliers = selector_outliers.get_best_model()
print(f"Best model for outlier data: {best_model_outliers}")

# Visualize model comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Model comparison for normal data
model_names = [m['model'] for m in selector.model_ranking]
loo_values = [m['loo'] for m in selector.model_ranking]

ax1.bar(model_names, loo_values, color=['skyblue', 'lightcoral'])
ax1.set_ylabel('LOO Value')
ax1.set_title('Model Comparison - Normal Data')
ax1.tick_params(axis='x', rotation=45)

# Model comparison for outlier data
model_names_out = [m['model'] for m in selector_outliers.model_ranking]
loo_values_out = [m['loo'] for m in selector_outliers.model_ranking]

ax2.bar(model_names_out, loo_values_out, color=['skyblue', 'lightcoral'])
ax2.set_ylabel('LOO Value')
ax2.set_title('Model Comparison - Outlier Data')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

---

## 2. Causal Inference Hands-on

### Exercise 2.1: Potential Outcomes Framework

```python
# Exercise: Implement potential outcomes framework
class PotentialOutcomesFramework:
    def __init__(self):
        self.data = None
        self.assumptions_validated = False

    def generate_synthetic_data(self, n_units=1000, treatment_prob=0.5,
                               treatment_effect=2.0, outcome_noise=1.0):
        """Generate synthetic data for causal inference"""

        np.random.seed(42)

        # Generate covariates
        age = np.random.normal(40, 15, n_units)
        education = np.random.randint(12, 20, n_units)
        income = np.random.normal(50000, 20000, n_units)

        # Confounders (affect both treatment assignment and outcome)
        confounders = {
            'age': age,
            'education': education,
            'baseline_income': income
        }

        # Propensity score model (probability of treatment)
        logit_ps = (
            -2 + 0.05 * age + 0.1 * education + 0.00001 * income
        )
        propensity_scores = 1 / (1 + np.exp(-logit_ps))

        # Treatment assignment
        treatment = np.random.binomial(1, propensity_scores)

        # Potential outcomes
        # Y(0) - outcome under control
        y0 = 10 + 0.5 * age + 0.2 * education + 0.0001 * income + np.random.normal(0, outcome_noise, n_units)

        # Y(1) - outcome under treatment
        y1 = y0 + treatment_effect + np.random.normal(0, outcome_noise * 0.1, n_units)

        # Observed outcome (only one potential outcome is observed)
        observed_outcome = treatment * y1 + (1 - treatment) * y0

        # Create DataFrame
        data = pd.DataFrame({
            'unit_id': range(n_units),
            'age': age,
            'education': education,
            'baseline_income': income,
            'treatment': treatment,
            'outcome': observed_outcome,
            'propensity_score': propensity_scores,
            'y0': y0,  # Potential outcome under control (not observed)
            'y1': y1,  # Potential outcome under treatment (not observed)
        })

        # Calculate true treatment effects
        data['true_ite'] = y1 - y0  # Individual Treatment Effect
        data['observed_ite'] = np.nan  # Cannot be observed

        return data

    def calculate_true_effects(self, data):
        """Calculate true causal effects (only possible with synthetic data)"""

        results = {}

        # Average Treatment Effect (ATE)
        results['ate'] = (data['y1'] - data['y0']).mean()

        # Average Treatment Effect on the Treated (ATT)
        treated_data = data[data['treatment'] == 1]
        results['att'] = (treated_data['y1'] - treated_data['y0']).mean()

        # Average Treatment Effect on the Controls (ATC)
        control_data = data[data['treatment'] == 0]
        results['atc'] = (control_data['y1'] - control_data['y0']).mean()

        # Conditional Average Treatment Effect (CATE) for age
        # Divide into age quartiles
        age_quartiles = pd.qcut(data['age'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        data['age_quartile'] = age_quartiles

        results['cate_by_age'] = {}
        for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
            quartile_data = data[data['age_quartile'] == quartile]
            results['cate_by_age'][quartile] = (quartile_data['y1'] - quartile_data['y0']).mean()

        return results

    def validate_sutva(self, data):
        """Validate Stable Unit Treatment Value Assumption"""

        results = {}

        # Check for interference (units affecting each other)
        # For simplicity, we'll check if treatment assignment is independent across units
        # In real data, this is more complex and requires domain knowledge

        # Check for multiple versions of treatment
        treatment_versions = data['treatment'].nunique()
        results['single_treatment_version'] = treatment_versions == 2  # Binary treatment

        return results

    def validate_ignorability(self, data, covariates):
        """Validate ignorability assumption using balance tests"""

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        results = {}

        X = data[covariates].values
        treatment = data['treatment'].values

        # Check if covariates predict treatment (they shouldn't if ignorable)
        ps_model = LogisticRegression()
        ps_model.fit(X, treatment)
        ps_predictions = ps_model.predict_proba(X)[:, 1]

        auc = roc_auc_score(treatment, ps_predictions)
        results['propensity_score_auc'] = auc
        results['ignorable'] = auc < 0.7  # Heuristic threshold

        # Check covariate balance
        balance_results = {}
        for covariate in covariates:
            treated_mean = data[data['treatment'] == 1][covariate].mean()
            control_mean = data[data['treatment'] == 0][covariate].mean()

            # Standardized difference
            pooled_std = np.sqrt(
                (data[data['treatment'] == 1][covariate].var() +
                 data[data['treatment'] == 0][covariate].var()) / 2
            )

            std_diff = abs(treated_mean - control_mean) / pooled_std

            balance_results[covariate] = {
                'treated_mean': treated_mean,
                'control_mean': control_mean,
                'standardized_difference': std_diff,
                'balanced': std_diff < 0.1  # Rule of thumb
            }

        results['balance'] = balance_results
        results['overall_balanced'] = all(r['balanced'] for r in balance_results.values())

        return results

    def estimate_ate_methods(self, data):
        """Estimate ATE using various methods"""

        results = {}

        # Naive estimator (biased)
        treated_mean = data[data['treatment'] == 1]['outcome'].mean()
        control_mean = data[data['treatment'] == 0]['outcome'].mean()
        results['naive_ate'] = treated_mean - control_mean

        # Stratification on propensity score
        # Create propensity score strata
        data['ps_stratum'] = pd.cut(data['propensity_score'], bins=5, labels=False)

        stratum_ates = []
        for stratum in data['ps_stratum'].unique():
            stratum_data = data[data['ps_stratum'] == stratum]
            if len(stratum_data[stratum_data['treatment'] == 1]) > 0 and \
               len(stratum_data[stratum_data['treatment'] == 0]) > 0:

                stratum_treated = stratum_data[stratum_data['treatment'] == 1]['outcome'].mean()
                stratum_control = stratum_data[stratum_data['treatment'] == 0]['outcome'].mean()
                stratum_ate = stratum_treated - stratum_control
                stratum_weight = len(stratum_data) / len(data)
                stratum_ates.append(stratum_ate * stratum_weight)

        results['stratification_ate'] = sum(stratum_ates)

        # IPW estimator
        treated_weights = data['treatment'] / data['propensity_score']
        control_weights = (1 - data['treatment']) / (1 - data['propensity_score'])

        treated_mean_weighted = np.sum(data['outcome'] * treated_weights) / np.sum(treated_weights)
        control_mean_weighted = np.sum(data['outcome'] * control_weights) / np.sum(control_weights)

        results['ipw_ate'] = treated_mean_weighted - control_mean_weighted

        # Regression adjustment
        from sklearn.linear_model import LinearRegression

        # Include treatment and covariates
        X = data[['treatment'] + ['age', 'education', 'baseline_income']].values
        y = data['outcome'].values

        reg_model = LinearRegression()
        reg_model.fit(X, y)

        # Predict under treatment and control
        X_treated = X.copy()
        X_treated[:, 0] = 1  # Set treatment = 1

        X_control = X.copy()
        X_control[:, 0] = 0  # Set treatment = 0

        y1_pred = reg_model.predict(X_treated)
        y0_pred = reg_model.predict(X_control)

        results['regression_ate'] = y1_pred.mean() - y0_pred.mean()

        return results

# Run the exercises
framework = PotentialOutcomesFramework()

# Generate synthetic data
print("Generating synthetic data...")
data = framework.generate_synthetic_data(n_units=2000, treatment_effect=3.0)
print(f"Generated data with {len(data)} units")
print(f"Treatment rate: {data['treatment'].mean():.3f}")

# Calculate true effects (only possible with synthetic data)
print("\nCalculating true causal effects...")
true_effects = framework.calculate_true_effects(data)
print(f"True ATE: {true_effects['ate']:.3f}")
print(f"True ATT: {true_effects['att']:.3f}")
print(f"True ATC: {true_effects['atc']:.3f}")

print("\nCATE by Age Quartile:")
for quartile, effect in true_effects['cate_by_age'].items():
    print(f"  {quartile}: {effect:.3f}")

# Validate assumptions
print("\nValidating assumptions...")
sutva_results = framework.validate_sutva(data)
ignorability_results = framework.validate_ignorability(data, ['age', 'education', 'baseline_income'])

print(f"SUTVA valid: {sutva_results['single_treatment_version']}")
print(f"Ignorability plausible (AUC < 0.7): {ignorability_results['ignorable']}")
print(f"Overall covariate balance: {ignorability_results['overall_balanced']}")

# Estimate effects using different methods
print("\nEstimating ATE using different methods...")
estimated_effects = framework.estimate_ate_methods(data)

print("Method Comparison:")
print(f"Naive estimator: {estimated_effects['naive_ate']:.3f}")
print(f"Stratification: {estimated_effects['stratification_ate']:.3f}")
print(f"IPW estimator: {estimated_effects['ipw_ate']:.3f}")
print(f"Regression adjustment: {estimated_effects['regression_ate']:.3f}")

# Calculate bias for each method
print("\nBias compared to true ATE:")
methods = ['naive_ate', 'stratification_ate', 'ipw_ate', 'regression_ate']
for method in methods:
    bias = estimated_effects[method] - true_effects['ate']
    print(f"  {method}: {bias:.3f}")

# Visualize results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. Covariate balance
covariates = ['age', 'education', 'baseline_income']
balance_values = [ignorability_results['balance'][cov]['standardized_difference'] for cov in covariates]

ax1.bar(covariates, balance_values, color=['red' if x > 0.1 else 'green' for x in balance_values])
ax1.axhline(y=0.1, color='black', linestyle='--', label='Balance threshold')
ax1.set_ylabel('Standardized Difference')
ax1.set_title('Covariate Balance Check')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# 2. Propensity score distribution
treated_ps = data[data['treatment'] == 1]['propensity_score']
control_ps = data[data['treatment'] == 0]['propensity_score']

ax2.hist(treated_ps, bins=30, alpha=0.7, label='Treated', color='blue')
ax2.hist(control_ps, bins=30, alpha=0.7, label='Control', color='red')
ax2.set_xlabel('Propensity Score')
ax2.set_ylabel('Frequency')
ax2.set_title('Propensity Score Distribution')
ax2.legend()

# 3. True vs estimated effects
method_names = ['Naive', 'Stratification', 'IPW', 'Regression']
estimated_values = [estimated_effects[m] for m in methods]

x_pos = range(len(method_names))
bars = ax3.bar(x_pos, estimated_values, color='skyblue', alpha=0.7, label='Estimated')
ax3.axhline(y=true_effects['ate'], color='red', linestyle='--', linewidth=2, label='True ATE')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, estimated_values)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{value:.2f}', ha='center', va='bottom')

ax3.set_xlabel('Estimation Method')
ax3.set_ylabel('Treatment Effect')
ax3.set_title('ATE Estimation Methods Comparison')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(method_names, rotation=45)
ax3.legend()

# 4. Individual treatment effects by age
age_bins = pd.qcut(data['age'], 10, retbins=True, labels=False)
data['age_bin'] = age_bins[0]

ite_by_age = data.groupby('age_bin')['true_ite'].mean()
age_centers = data.groupby('age_bin')['age'].mean()

ax4.scatter(age_centers, ite_by_age, color='purple', alpha=0.6)
ax4.plot(age_centers, ite_by_age, color='purple', alpha=0.3)
ax4.set_xlabel('Age')
ax4.set_ylabel('Individual Treatment Effect')
ax4.set_title('Heterogeneous Treatment Effects by Age')

plt.tight_layout()
plt.show()
```

### Exercise 2.2: Individual Treatment Effect Estimation

```python
# Exercise: Implement Individual Treatment Effect (ITE) estimation methods
class IndividualTreatmentEffectEstimator:
    def __init__(self):
        self.models = {}

    def s_learner(self, base_learner, X, treatment, outcome):
        """S-learner for ITE estimation"""

        # Create feature matrix with treatment indicator
        X_with_treatment = np.column_stack([X, treatment.reshape(-1, 1)])

        # Fit model on full dataset
        from sklearn.ensemble import RandomForestRegressor
        if base_learner == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif base_learner == 'linear':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X_with_treatment, outcome)

        # Predict under both treatment conditions
        X_treated = np.column_stack([X, np.ones((len(X), 1))])
        X_control = np.column_stack([X, np.zeros((len(X), 1))])

        y1_pred = model.predict(X_treated)
        y0_pred = model.predict(X_control)

        ite = y1_pred - y0_pred

        return ite, model

    def t_learner(self, base_learner, X, treatment, outcome):
        """T-learner for ITE estimation"""

        # Separate data by treatment
        treated_mask = treatment == 1
        control_mask = treatment == 0

        X_treated = X[treated_mask]
        y_treated = outcome[treated_mask]

        X_control = X[control_mask]
        y_control = outcome[control_mask]

        # Fit separate models
        from sklearn.ensemble import RandomForestRegressor
        if base_learner == 'random_forest':
            model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
            model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
        elif base_learner == 'linear':
            from sklearn.linear_model import LinearRegression
            model_1 = LinearRegression()
            model_0 = LinearRegression()
        else:
            model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
            model_0 = RandomForestRegressor(n_estimators=100, random_state=42)

        model_1.fit(X_treated, y_treated)
        model_0.fit(X_control, y_control)

        # Predict outcomes
        y1_pred = model_1.predict(X)
        y0_pred = model_0.predict(X)

        ite = y1_pred - y0_pred

        return ite, model_1, model_0

    def x_learner(self, base_learner, X, treatment, outcome):
        """X-learner for ITE estimation"""

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        # Step 1: Fit models for each group
        treated_mask = treatment == 1
        control_mask = treatment == 0

        X_treated = X[treated_mask]
        y_treated = outcome[treated_mask]

        X_control = X[control_mask]
        y_control = outcome[control_mask]

        if base_learner == 'random_forest':
            model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
            model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
        elif base_learner == 'linear':
            model_1 = LinearRegression()
            model_0 = LinearRegression()
        else:
            model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
            model_0 = RandomForestRegressor(n_estimators=100, random_state=42)

        model_1.fit(X_treated, y_treated)
        model_0.fit(X_control, y_control)

        # Step 2: Estimate treatment effects
        y0_hat_treated = model_0.predict(X_treated)  # Potential outcome under control for treated
        y1_hat_control = model_1.predict(X_control)  # Potential outcome under treatment for control

        treatment_effects_1 = y_treated - y0_hat_treated  # For treated units
        treatment_effects_0 = y1_hat_control - y_control  # For control units

        # Step 3: Fit treatment effect models
        if base_learner == 'random_forest':
            effect_model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
            effect_model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
        elif base_learner == 'linear':
            effect_model_1 = LinearRegression()
            effect_model_0 = LinearRegression()
        else:
            effect_model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
            effect_model_0 = RandomForestRegressor(n_estimators=100, random_state=42)

        effect_model_1.fit(X_treated, treatment_effects_1)
        effect_model_0.fit(X_control, treatment_effects_0)

        # Step 4: Predict ITE
        effect_1 = effect_model_1.predict(X)
        effect_0 = effect_model_0.predict(X)

        ite = (effect_1 + effect_0) / 2

        return ite, effect_model_1, effect_model_0

    def causal_forest(self, X, treatment, outcome, n_trees=100):
        """Simplified causal forest implementation"""

        def fit_tree(X, treatment, outcome, depth=0, max_depth=6):
            n_samples = len(X)

            # Stopping criteria
            if (n_samples < 5 or depth >= max_depth or
                len(np.unique(treatment)) == 1):
                return {
                    'is_leaf': True,
                    'mean_outcome': outcome.mean(),
                    'treatment_effect': self._estimate_leaf_treatment_effect(treatment, outcome)
                }

            # Find best split
            best_feature, best_threshold = self._find_best_split(X, treatment, outcome)

            if best_feature is None:
                return {
                    'is_leaf': True,
                    'mean_outcome': outcome.mean(),
                    'treatment_effect': self._estimate_leaf_treatment_effect(treatment, outcome)
                }

            # Split data
            left_mask = X[:, best_feature] <= best_threshold
            right_mask = ~left_mask

            if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                return {
                    'is_leaf': True,
                    'mean_outcome': outcome.mean(),
                    'treatment_effect': self._estimate_leaf_treatment_effect(treatment, outcome)
                }

            # Recursively grow children
            left_child = fit_tree(X[left_mask], treatment[left_mask], outcome[left_mask], depth + 1, max_depth)
            right_child = fit_tree(X[right_mask], treatment[right_mask], outcome[right_mask], depth + 1, max_depth)

            return {
                'feature': best_feature,
                'threshold': best_threshold,
                'left': left_child,
                'right': right_child,
                'depth': depth
            }

        def predict_tree(tree, x):
            if tree['is_leaf']:
                return tree['treatment_effect']

            if x[tree['feature']] <= tree['threshold']:
                return predict_tree(tree['left'], x)
            else:
                return predict_tree(tree['right'], x)

        # Build forest
        trees = []
        for i in range(n_trees):
            # Bootstrap sample
            n_samples = len(X)
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)

            X_bootstrap = X[bootstrap_idx]
            treatment_bootstrap = treatment[bootstrap_idx]
            outcome_bootstrap = outcome[bootstrap_idx]

            # Grow tree
            tree = fit_tree(X_bootstrap, treatment_bootstrap, outcome_bootstrap)
            trees.append(tree)

        # Predict ITE for all samples
        ite_predictions = []
        for x in X:
            tree_predictions = [predict_tree(tree, x) for tree in trees]
            ite_predictions.append(np.mean(tree_predictions))

        return np.array(ite_predictions), trees

    def _estimate_leaf_treatment_effect(self, treatment, outcome):
        """Estimate treatment effect in a leaf node"""
        treated_outcomes = outcome[treatment == 1]
        control_outcomes = outcome[treatment == 0]

        if len(treated_outcomes) == 0 or len(control_outcomes) == 0:
            return 0  # No clear treatment effect

        return treated_outcomes.mean() - control_outcomes.mean()

    def _find_best_split(self, X, treatment, outcome):
        """Find best split for causal tree"""
        best_gain = 0
        best_feature = None
        best_threshold = None

        n_features = int(np.sqrt(X.shape[1]))
        feature_indices = np.random.choice(X.shape[1], n_features, replace=False)

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue

                gain = self._calculate_causal_gain(
                    outcome[left_mask], treatment[left_mask],
                    outcome[right_mask], treatment[right_mask]
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_causal_gain(self, y_left, t_left, y_right, t_right):
        """Calculate causal gain from split"""
        # Simple variance reduction with treatment consideration

        # Calculate effect in each child
        effect_left = self._estimate_leaf_treatment_effect(t_left, y_left)
        effect_right = self._estimate_leaf_treatment_effect(t_right, y_right)

        # Gain is weighted sum of squared effects
        total_samples = len(y_left) + len(y_right)
        weighted_gain = (len(y_left) * effect_left**2 + len(y_right) * effect_right**2) / total_samples

        return weighted_gain

    def evaluate_ite(self, ite_estimated, ite_true):
        """Evaluate ITE estimation performance"""

        mse = np.mean((ite_estimated - ite_true)**2)
        mae = np.mean(np.abs(ite_estimated - ite_true))

        # Correlation
        correlation = np.corrcoef(ite_estimated, ite_true)[0, 1]

        # Classification accuracy (if ITE has consistent sign)
        signs_match = np.sign(ite_estimated) == np.sign(ite_true)
        sign_accuracy = np.mean(signs_match)

        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'sign_accuracy': sign_accuracy
        }

# Test ITE estimation methods
estimator = IndividualTreatmentEffectEstimator()

# Use synthetic data from previous exercise
X = data[['age', 'education', 'baseline_income']].values
treatment = data['treatment'].values
outcome = data['outcome'].values
ite_true = data['true_ite'].values

print("Estimating Individual Treatment Effects...")
print(f"Sample size: {len(X)}")
print(f"True ITE range: [{ite_true.min():.3f}, {ite_true.max():.3f}]")
print(f"True ITE mean: {ite_true.mean():.3f}")

# Test S-learner
print("\nTesting S-learner...")
ite_s, model_s = estimator.s_learner('random_forest', X, treatment, outcome)
eval_s = estimator.evaluate_ite(ite_s, ite_true)
print(f"S-learner - MSE: {eval_s['mse']:.3f}, Correlation: {eval_s['correlation']:.3f}")

# Test T-learner
print("Testing T-learner...")
ite_t, model_t1, model_t0 = estimator.t_learner('random_forest', X, treatment, outcome)
eval_t = estimator.evaluate_ite(ite_t, ite_true)
print(f"T-learner - MSE: {eval_t['mse']:.3f}, Correlation: {eval_t['correlation']:.3f}")

# Test X-learner
print("Testing X-learner...")
ite_x, model_x1, model_x0 = estimator.x_learner('random_forest', X, treatment, outcome)
eval_x = estimator.evaluate_ite(ite_x, ite_true)
print(f"X-learner - MSE: {eval_x['mse']:.3f}, Correlation: {eval_x['correlation']:.3f}")

# Test Causal Forest
print("Testing Causal Forest...")
ite_cf, trees = estimator.causal_forest(X, treatment, outcome, n_trees=50)
eval_cf = estimator.evaluate_ite(ite_cf, ite_true)
print(f"Causal Forest - MSE: {eval_cf['mse']:.3f}, Correlation: {eval_cf['correlation']:.3f}")

# Visualize results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

methods = ['S-learner', 'T-learner', 'X-learner', 'Causal Forest']
mse_values = [eval_s['mse'], eval_t['mse'], eval_x['mse'], eval_cf['mse']]
correlation_values = [eval_s['correlation'], eval_t['correlation'], eval_x['correlation'], eval_cf['correlation']]

# MSE comparison
bars1 = ax1.bar(methods, mse_values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('ITE Estimation: MSE Comparison')
ax1.tick_params(axis='x', rotation=45)

# Add value labels
for bar, value in zip(bars1, mse_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

# Correlation comparison
bars2 = ax2.bar(methods, correlation_values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
ax2.set_ylabel('Correlation with True ITE')
ax2.set_title('ITE Estimation: Correlation Comparison')
ax2.set_ylim(0, 1)
ax2.tick_params(axis='x', rotation=45)

# Add value labels
for bar, value in zip(bars2, correlation_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{value:.3f}', ha='center', va='bottom')

# Scatter plots: estimated vs true ITE
ax3.scatter(ite_true, ite_s, alpha=0.6, color='skyblue', label='S-learner')
ax3.plot([ite_true.min(), ite_true.max()], [ite_true.min(), ite_true.max()], 'r--', label='Perfect')
ax3.set_xlabel('True ITE')
ax3.set_ylabel('Estimated ITE')
ax3.set_title('S-learner: Estimated vs True ITE')
ax3.legend()

ax4.scatter(ite_true, ite_cf, alpha=0.6, color='pink', label='Causal Forest')
ax4.plot([ite_true.min(), ite_true.max()], [ite_true.min(), ite_true.max()], 'r--', label='Perfect')
ax4.set_xlabel('True ITE')
ax4.set_ylabel('Estimated ITE')
ax4.set_title('Causal Forest: Estimated vs True ITE')
ax4.legend()

plt.tight_layout()
plt.show()

# Analyze heterogeneity by covariates
print("\nAnalyzing treatment effect heterogeneity...")

# Age-based heterogeneity
age_quartiles = pd.qcut(data['age'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
heterogeneity_analysis = {}

for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
    mask = age_quartiles == quartile
    quartile_ite_true = ite_true[mask]
    quartile_ite_s = ite_s[mask]

    heterogeneity_analysis[quartile] = {
        'true_mean': quartile_ite_true.mean(),
        'estimated_mean': quartile_ite_s.mean(),
        'mse': np.mean((quartile_ite_s - quartile_ite_true)**2)
    }

print("Heterogeneity by Age Quartile:")
for quartile, results in heterogeneity_analysis.items():
    print(f"  {quartile}: True={results['true_mean']:.3f}, "
          f"Estimated={results['estimated_mean']:.3f}, "
          f"MSE={results['mse']:.3f}")
```

---

## 3. Experimental Design Practice

### Exercise 3.1: Randomized Controlled Trial Design

```python
# Exercise: Design and analyze randomized controlled trials
class ExperimentalDesign:
    def __init__(self):
        self.trial_data = None
        self.analysis_plan = None

    def power_analysis(self, effect_size, alpha=0.05, power=0.8, groups=2):
        """Calculate required sample size for desired power"""

        from scipy.stats import norm

        z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed test
        z_beta = norm.ppf(power)

        if groups == 2:
            # Two-sample t-test
            n_per_group = 2 * (z_alpha + z_beta)**2 / effect_size**2
        else:
            # ANOVA (simplified - using Bonferroni correction)
            alpha_corrected = alpha / (groups - 1)
            z_alpha_corrected = norm.ppf(1 - alpha_corrected/2)
            n_per_group = 2 * (z_alpha_corrected + z_beta)**2 / effect_size**2

        return {
            'n_per_group': int(np.ceil(n_per_group)),
            'total_n': int(np.ceil(groups * n_per_group)),
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'groups': groups
        }

    def randomization_schemes(self, n_units, n_treatments=2, allocation_ratio=1,
                            block_size=None):
        """Generate different randomization schemes"""

        schemes = {}

        # Simple randomization
        if n_treatments == 2:
            treatment_assignments = np.random.binomial(1, 0.5, n_units)
        else:
            treatment_assignments = np.random.choice(n_treatments, n_units,
                                                   p=[1/n_treatments]*n_treatments)

        schemes['simple'] = treatment_assignments

        # Block randomization
        if block_size is None:
            block_size = n_treatments * allocation_ratio

        n_blocks = n_units // block_size
        treatment_assignments = []

        for _ in range(n_blocks):
            block = list(range(n_treatments)) * allocation_ratio
            np.random.shuffle(block)
            treatment_assignments.extend(block)

        # Add remaining assignments
        remaining = n_units - len(treatment_assignments)
        if remaining > 0:
            additional = list(range(n_treatments)) * (remaining // n_treatments + 1)
            treatment_assignments.extend(additional[:remaining])

        schemes['block'] = np.array(treatment_assignments)

        # Stratified randomization
        # For simplicity, assume one stratification variable
        # In practice, you'd have multiple stratification factors
        stratified_assignments = self._stratified_randomization(n_units, n_treatments)
        schemes['stratified'] = stratified_assignments

        return schemes

    def _stratified_randomization(self, n_units, n_treatments):
        """Implement stratified randomization"""
        # Create stratification variable (simulated)
        strata = np.random.choice(3, n_units)  # 3 strata

        stratified_assignments = np.zeros(n_units, dtype=int)

        for stratum in range(3):
            stratum_mask = strata == stratum
            stratum_size = np.sum(stratum_mask)

            if stratum_size > 0:
                stratum_assignments = np.random.choice(n_treatments, stratum_size,
                                                     p=[1/n_treatments]*n_treatments)
                stratified_assignments[stratum_mask] = stratum_assignments

        return stratified_assignments

    def covariate_adaptive_randomization(self, n_units, n_treatments=2,
                                       imbalance_parameter=1):
        """Implement covariate-adaptive randomization (Pocock-Simon)"""

        # Simulate covariates
        covariates = np.random.randn(n_units, 3)  # 3 covariates

        assignments = np.zeros(n_units, dtype=int)

        for i in range(n_units):
            # Calculate imbalance for each treatment option
            imbalances = []

            for treatment in range(n_treatments):
                # Calculate imbalance if this treatment is assigned
                temp_assignments = assignments.copy()
                temp_assignments[i] = treatment

                # Calculate imbalance score
                imbalance = self._calculate_imbalance(temp_assignments, covariates[:i+1])
                imbalances.append(imbalance)

            # Choose treatment with minimum imbalance
            optimal_treatment = np.argmin(imbalances)
            assignments[i] = optimal_treatment

        return assignments

    def _calculate_imbalance(self, assignments, covariates):
        """Calculate imbalance for covariate-adaptive randomization"""
        imbalance = 0

        for j in range(covariates.shape[1]):
            covariate = covariates[:, j]

            # Divide into high/low based on median
            median_val = np.median(covariate)
            high_mask = covariate > median_val
            low_mask = covariate <= median_val

            # Calculate imbalance within each group
            for group_mask in [high_mask, low_mask]:
                if np.sum(group_mask) > 0:
                    group_assignments = assignments[group_mask]
                    for treatment in range(max(assignments) + 1):
                        treatment_count = np.sum(group_assignments == treatment)
                        imbalance += treatment_count**2

        return imbalance

    def adaptive_trial_design(self, n_max=1000, interim_analysis_points=[200, 400, 600, 800]):
        """Design adaptive trial with sample size re-estimation"""

        design = {
            'n_max': n_max,
            'interim_analysis_points': interim_analysis_points,
            'early_stopping_rules': {},
            'sample_size_adjustment': {}
        }

        # Define stopping boundaries (O'Brien-Fleming style)
        for i, n_current in enumerate(interim_analysis_points):
            # Simplified O'Brien-Fleming boundaries
            if i == 0:  # First interim analysis
                design['early_stopping_rules'][n_current] = {'alpha': 0.001, 'futility': 0.05}
            elif i == 1:  # Second interim analysis
                design['early_stopping_rules'][n_current] = {'alpha': 0.004, 'futility': 0.1}
            elif i == 2:  # Third interim analysis
                design['early_stopping_rules'][n_current] = {'alpha': 0.019, 'futility': 0.2}
            else:  # Final analysis
                design['early_stopping_rules'][n_current] = {'alpha': 0.05, 'futility': 1.0}

        return design

    def simulate_rct(self, n_per_group, effect_size=0.5, variance=1,
                    dropout_rate=0.1, non_compliance_rate=0.05):
        """Simulate a randomized controlled trial"""

        np.random.seed(42)

        total_n = 2 * n_per_group

        # Generate baseline covariates
        baseline_covariates = {
            'age': np.random.normal(50, 15, total_n),
            'gender': np.random.binomial(1, 0.5, total_n),  # 1 = female
            'baseline_score': np.random.normal(100, 20, total_n)
        }

        # Randomization
        treatment = np.random.binomial(1, 0.5, total_n)

        # Generate outcomes
        # Y = β₀ + β₁*T + β₂*Age + β₃*Gender + β₄*Baseline + ε
        intercept = 100
        treatment_effect = effect_size
        age_coef = 0.1
        gender_coef = 5.0
        baseline_coef = 0.8

        outcome = (intercept +
                  treatment_effect * treatment +
                  age_coef * baseline_covariates['age'] +
                  gender_coef * baseline_covariates['gender'] +
                  baseline_coef * baseline_covariates['baseline_score'] +
                  np.random.normal(0, variance, total_n))

        # Apply dropout
        dropout_mask = np.random.binomial(1, dropout_rate, total_n)
        outcome[dropout_mask] = np.nan

        # Apply non-compliance
        compliance_mask = np.random.binomial(1, 1 - non_compliance_rate, total_n)
        # For simplicity, non-compliant patients get no treatment effect
        outcome = outcome + (treatment_effect * (1 - compliance_mask))

        return pd.DataFrame({
            'patient_id': range(total_n),
            'treatment': treatment,
            'outcome': outcome,
            **baseline_covariates,
            'dropout': dropout_mask,
            'compliant': compliance_mask
        })

    def analyze_rct(self, data, analysis_type='intention_to_treat'):
        """Analyze randomized controlled trial"""

        results = {}

        # Remove missing outcomes
        complete_data = data.dropna(subset=['outcome'])

        # Intention-to-treat analysis
        if analysis_type in ['intention_to_treat', 'both']:

            treated_outcomes = complete_data[complete_data['treatment'] == 1]['outcome']
            control_outcomes = complete_data[complete_data['treatment'] == 0]['outcome']

            # Calculate treatment effect
            itt_effect = treated_outcomes.mean() - control_outcomes.mean()

            # Statistical test
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(treated_outcomes, control_outcomes)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                (treated_outcomes.var() + control_outcomes.var()) / 2
            )
            cohens_d = itt_effect / pooled_std

            results['intention_to_treat'] = {
                'treatment_effect': itt_effect,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'n_treated': len(treated_outcomes),
                'n_control': len(control_outcomes),
                'mean_treated': treated_outcomes.mean(),
                'mean_control': control_outcomes.mean()
            }

        # Per-protocol analysis
        if analysis_type in ['per_protocol', 'both']:

            # Include only compliant patients
            protocol_data = complete_data[complete_data['compliant'] == 1]

            if len(protocol_data[protocol_data['treatment'] == 1]) > 0 and \
               len(protocol_data[protocol_data['treatment'] == 0]) > 0:

                treated_outcomes = protocol_data[protocol_data['treatment'] == 1]['outcome']
                control_outcomes = protocol_data[protocol_data['treatment'] == 0]['outcome']

                pp_effect = treated_outcomes.mean() - control_outcomes.mean()

                from scipy.stats import ttest_ind
                t_stat_pp, p_value_pp = ttest_ind(treated_outcomes, control_outcomes)

                pooled_std_pp = np.sqrt(
                    (treated_outcomes.var() + control_outcomes.var()) / 2
                )
                cohens_d_pp = pp_effect / pooled_std_pp

                results['per_protocol'] = {
                    'treatment_effect': pp_effect,
                    'p_value': p_value_pp,
                    'cohens_d': cohens_d_pp,
                    'n_treated': len(treated_outcomes),
                    'n_control': len(control_outcomes),
                    'mean_treated': treated_outcomes.mean(),
                    'mean_control': control_outcomes.mean()
                }

        return results

    def covariate_balance_check(self, data):
        """Check covariate balance between treatment groups"""

        covariates = ['age', 'gender', 'baseline_score']
        balance_results = {}

        for covar in covariates:
            treated_values = data[data['treatment'] == 1][covar]
            control_values = data[data['treatment'] == 0][covar]

            # Calculate means and standardized difference
            treated_mean = treated_values.mean()
            control_mean = control_values.mean()

            # Standardized difference
            pooled_std = np.sqrt((treated_values.var() + control_values.var()) / 2)
            std_diff = abs(treated_mean - control_mean) / pooled_std

            # Statistical test
            from scipy.stats import ttest_ind, chi2_contingency

            if covar == 'gender':  # Categorical variable
                # Chi-square test for independence
                contingency_table = pd.crosstab(data['treatment'], data[covar])
                chi2, p_value = chi2_contingency(contingency_table)[:2]
                test_stat = chi2
                test_name = 'chi_square'
            else:  # Continuous variable
                t_stat, p_value = ttest_ind(treated_values, control_values)
                test_stat = t_stat
                test_name = 't_test'

            balance_results[covar] = {
                'treated_mean': treated_mean,
                'control_mean': control_mean,
                'standardized_difference': std_diff,
                'test_statistic': test_stat,
                'p_value': p_value,
                'test_name': test_name,
                'balanced': std_diff < 0.1 and p_value > 0.05
            }

        return balance_results

# Run experimental design exercises
design = ExperimentalDesign()

print("=== Experimental Design Exercises ===\n")

# Power analysis
print("1. Power Analysis")
effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large effects
print("Required sample sizes for different effect sizes:")
for effect_size in effect_sizes:
    power_result = design.power_analysis(effect_size=effect_size, power=0.8)
    print(f"  Effect size {effect_size}: {power_result['n_per_group']} per group "
          f"(Total: {power_result['total_n']})")

# Randomization schemes comparison
print("\n2. Randomization Schemes Comparison")
n_units = 200
schemes = design.randomization_schemes(n_units, n_treatments=2)

print(f"Comparison of randomization schemes (n={n_units}):")
for scheme_name, assignments in schemes.items():
    treatment_rate = np.mean(assignments)
    print(f"  {scheme_name.capitalize()}: Treatment rate = {treatment_rate:.3f}")

# Simulate and analyze RCT
print("\n3. RCT Simulation and Analysis")
sim_data = design.simulate_rct(n_per_group=100, effect_size=0.5)

print(f"Simulated trial with {len(sim_data)} patients")
print(f"Treatment rate: {sim_data['treatment'].mean():.3f}")
print(f"Completion rate: {(1 - sim_data['dropout'].mean()):.3f}")
print(f"Compliance rate: {sim_data['compliant'].mean():.3f}")

# Analyze RCT
analysis_results = design.analyze_rct(sim_data, analysis_type='both')

print("\nRCT Analysis Results:")
for analysis_type, results in analysis_results.items():
    print(f"\n{analysis_type.replace('_', ' ').title()}:")
    print(f"  Treatment effect: {results['treatment_effect']:.3f}")
    print(f"  P-value: {results['p_value']:.4f}")
    print(f"  Effect size (Cohen's d): {results['cohens_d']:.3f}")
    print(f"  Sample sizes: Treated={results['n_treated']}, Control={results['n_control']}")

# Check covariate balance
print("\n4. Covariate Balance Check")
balance_results = design.covariate_balance_check(sim_data)

print("Covariate balance assessment:")
for covar, results in balance_results.items():
    status = "✓ Balanced" if results['balanced'] else "✗ Not balanced"
    print(f"  {covar}: {status}")
    print(f"    Standardized difference: {results['standardized_difference']:.3f}")
    print(f"    P-value: {results['p_value']:.4f}")

# Visualize results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. Randomization schemes comparison
scheme_names = list(schemes.keys())
treatment_rates = [np.mean(assignments) for assignments in schemes.values()]

bars = ax1.bar(scheme_names, treatment_rates, color=['skyblue', 'lightgreen', 'orange'])
ax1.axhline(y=0.5, color='red', linestyle='--', label='Target (0.5)')
ax1.set_ylabel('Treatment Assignment Rate')
ax1.set_title('Randomization Schemes Comparison')
ax1.legend()

# Add value labels
for bar, rate in zip(bars, treatment_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{rate:.3f}', ha='center', va='bottom')

# 2. Sample size vs effect size
effect_sizes_plot = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
sample_sizes = [design.power_analysis(es, power=0.8)['n_per_group'] for es in effect_sizes_plot]

ax2.plot(effect_sizes_plot, sample_sizes, 'bo-', linewidth=2, markersize=6)
ax2.set_xlabel('Effect Size')
ax2.set_ylabel('Required Sample Size per Group')
ax2.set_title('Sample Size vs Effect Size (Power = 0.8)')
ax2.grid(True, alpha=0.3)

# 3. Treatment effect comparison
analysis_types = list(analysis_results.keys())
effects = [analysis_results[at]['treatment_effect'] for at in analysis_types]
colors = ['skyblue', 'lightcoral']

bars = ax3.bar(analysis_types, effects, color=colors)
ax3.set_ylabel('Treatment Effect')
ax3.set_title('Treatment Effect: ITT vs Per-Protocol')
ax3.tick_params(axis='x', rotation=45)

# Add value labels
for bar, effect in zip(bars, effects):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{effect:.3f}', ha='center', va='bottom')

# 4. Covariate balance plot
covariates = list(balance_results.keys())
std_diffs = [balance_results[covar]['standardized_difference'] for covar in covariates]

bars = ax4.bar(covariates, std_diffs, color=['red' if x > 0.1 else 'green' for x in std_diffs])
ax4.axhline(y=0.1, color='black', linestyle='--', label='Balance threshold (0.1)')
ax4.set_ylabel('Standardized Difference')
ax4.set_title('Covariate Balance Assessment')
ax4.legend()

# Add value labels
for bar, diff in zip(bars, std_diffs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{diff:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

---

## 4. Quasi-experimental Methods

### Exercise 4.1: Difference-in-Differences Implementation

```python
# Exercise: Implement Difference-in-Differences analysis
class DifferenceInDifferencesAnalyzer:
    def __init__(self):
        pass

    def create_panel_data(self, n_units=100, n_periods=10, treatment_start_period=6):
        """Create synthetic panel data for DID analysis"""

        np.random.seed(42)

        # Unit and time identifiers
        units = np.repeat(range(n_units), n_periods)
        time_periods = np.tile(range(n_periods), n_units)

        # Treatment assignment (group and time)
        treated_units = np.random.choice(n_units, size=n_units//2, replace=False)
        treated_group = np.isin(units, treated_units).astype(int)

        # Treatment indicator
        post_treatment = (time_periods >= treatment_start_period).astype(int)
        treatment = treated_group * post_treatment

        # Time trends
        time_trend = time_periods

        # Generate outcomes with common trends
        # Y_it = α + β₁*Treat_i + β₂*Post_t + β₃*(Treat_i × Post_t) + γ*X_it + ε_it

        alpha = 10  # Constant
        beta1 = 1.0  # Group fixed effect
        beta2 = 0.5  # Time trend
        beta3 = 2.0  # Treatment effect (DID estimate)
        gamma = 0.3  # Covariate effect

        # Individual and time fixed effects
        unit_fe = np.random.normal(0, 2, n_units)
        time_fe = np.random.normal(0, 1, n_periods)
        unit_fe_expanded = unit_fe[units]
        time_fe_expanded = time_fe[time_periods]

        # Time-varying covariates
        covariates = np.random.randn(len(units), 2)  # Two covariates

        # Outcome
        outcome = (alpha +
                  beta1 * treated_group +
                  beta2 * post_treatment +
                  beta3 * treatment +
                  gamma * covariates.sum(axis=1) +
                  unit_fe_expanded +
                  time_fe_expanded +
                  np.random.normal(0, 1, len(units)))

        # Create DataFrame
        data = pd.DataFrame({
            'unit': units,
            'time': time_periods,
            'treated': treated_group,
            'post': post_treatment,
            'treatment': treatment,
            'outcome': outcome,
            'covariate1': covariates[:, 0],
            'covariate2': covariates[:, 1],
            'time_trend': time_trend
        })

        return data

    def estimate_did(self, data, unit_col='unit', time_col='time',
                    outcome_col='outcome', treatment_col='treatment'):
        """Estimate DID using OLS"""

        from sklearn.linear_model import LinearRegression

        # Create design matrix
        X = data[[treatment_col]].values

        # Add unit and time fixed effects
        unit_dummies = pd.get_dummies(data[unit_col], prefix='unit')
        time_dummies = pd.get_dummies(data[time_col], prefix='time')

        # Combine all variables
        X_full = pd.concat([data[[treatment_col]], unit_dummies, time_dummies], axis=1)

        y = data[outcome_col].values

        # Fit DID model
        model = LinearRegression()
        model.fit(X_full, y)

        # DID estimate is coefficient of treatment variable
        did_estimate = model.coef_[0]  # Treatment coefficient

        # Calculate standard errors (simplified)
        predictions = model.predict(X_full)
        residuals = y - predictions
        mse = np.mean(residuals**2)

        # Simplified standard error calculation
        n = len(y)
        p = X_full.shape[1]
        se_did = np.sqrt(mse / n)  # Very simplified

        # Calculate group-time specific effects
        group_time_effects = self._calculate_group_time_effects(data)

        # Test parallel trends (pre-treatment only)
        parallel_trends_test = self._test_parallel_trends(data)

        return {
            'did_estimate': did_estimate,
            'standard_error': se_did,
            't_statistic': did_estimate / se_did,
            'p_value': 2 * (1 - stats.norm.cdf(abs(did_estimate / se_did))),
            'model_r2': model.score(X_full, y),
            'group_time_effects': group_time_effects,
            'parallel_trends_test': parallel_trends_test,
            'model': model
        }

    def _calculate_group_time_effects(self, data):
        """Calculate effects for each group-time combination"""

        effects = {}

        for group in [0, 1]:
            for time in sorted(data['time'].unique()):
                group_time_data = data[(data['treated'] == group) & (data['time'] == time)]

                if len(group_time_data) > 0:
                    effects[(group, time)] = {
                        'mean_outcome': group_time_data['outcome'].mean(),
                        'n_obs': len(group_time_data)
                    }

        return effects

    def _test_parallel_trends(self, data):
        """Test parallel trends assumption using pre-treatment data"""

        # Focus on pre-treatment periods
        pre_treatment_data = data[data['post'] == 0].copy()

        if len(pre_treatment_data) == 0:
            return {'test_possible': False, 'reason': 'No pre-treatment data'}

        # Create interaction between treated and time trend
        pre_treatment_data['treated_time'] = pre_treatment_data['treated'] * pre_treatment_data['time']

        from sklearn.linear_model import LinearRegression

        X = pre_treatment_data[['treated', 'time', 'treated_time']].values
        y = pre_treatment_data['outcome'].values

        model = LinearRegression()
        model.fit(X, y)

        # Test if there's differential time trend
        time_trend_coef = model.coef_[2]  # Coefficient of treated_time interaction

        # Statistical test (simplified)
        predictions = model.predict(X)
        residuals = y - predictions
        mse = np.mean(residuals**2)
        se_coef = np.sqrt(mse / len(y))  # Very simplified

        t_stat = time_trend_coef / se_coef
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return {
            'time_trend_coefficient': time_trend_coef,
            'standard_error': se_coef,
            't_statistic': t_stat,
            'p_value': p_value,
            'parallel_trends_satisfied': abs(t_stat) < 1.96,  # 5% significance
            'interpretation': 'Small coefficient suggests parallel trends'
        }

    def event_study(self, data, treatment_start_period, outcome_col='outcome'):
        """Perform event study analysis"""

        # Create relative time indicators
        data['relative_time'] = data['time'] - treatment_start_period

        # Create event study dummies (excluding pre-treatment period)
        event_dummies = []
        relative_times = sorted(data['relative_time'].unique())

        for rt in relative_times:
            if rt < -1:  # Exclude -1 as reference period
                dummy_name = f'lead_{abs(rt)}' if rt < 0 else f'lag_{rt}'
                data[dummy_name] = (data['relative_time'] == rt).astype(int)
                event_dummies.append(dummy_name)

        # Estimate event study model
        from sklearn.linear_model import LinearRegression

        # Base variables
        base_vars = ['treated', 'post']

        # Unit and time fixed effects
        unit_dummies = pd.get_dummies(data['unit'], prefix='unit')
        time_dummies = pd.get_dummies(data['time'], prefix='time')

        # Full design matrix
        X_full = pd.concat([
            data[base_vars],
            data[event_dummies],
            unit_dummies,
            time_dummies
        ], axis=1)

        y = data[outcome_col].values

        model = LinearRegression()
        model.fit(X_full, y)

        # Extract coefficients for event study variables
        event_coefficients = {}
        event_standard_errors = {}

        for dummy in event_dummies:
            if dummy in X_full.columns:
                coef_idx = X_full.columns.get_loc(dummy)
                event_coefficients[dummy] = model.coef_[coef_idx]
                # Simplified standard error
                predictions = model.predict(X_full)
                residuals = y - predictions
                mse = np.mean(residuals**2)
                event_standard_errors[dummy] = np.sqrt(mse / len(y))

        return {
            'event_coefficients': event_coefficients,
            'standard_errors': event_standard_errors,
            'model_r2': model.score(X_full, y),
            'relative_times': relative_times
        }

    def triple_differences(self, data, group_col='treated', time_col='time',
                         outcome_col='outcome', third_dimension_col='region'):
        """Estimate Triple Differences (DDD)"""

        # Create triple interaction
        data['triple_diff'] = (data[group_col] * data['post'] * data[third_dimension_col])

        from sklearn.linear_model import LinearRegression

        # Design matrix for DDD
        base_vars = [group_col, 'post', third_dimension_col]
        interactions = [f'{group_col}:post', f'{group_col}:{third_dimension_col}',
                       f'post:{third_dimension_col}', 'triple_diff']

        # Add all variables to design matrix
        X_vars = base_vars + interactions

        # Add fixed effects
        unit_dummies = pd.get_dummies(data['unit'], prefix='unit')
        time_dummies = pd.get_dummies(data['time'], prefix='time')
        third_dummies = pd.get_dummies(data[third_dimension_col], prefix=third_dimension_col)

        X_full = pd.concat([data[X_vars], unit_dummies, time_dummies, third_dummies], axis=1)
        y = data[outcome_col].values

        model = LinearRegression()
        model.fit(X_full, y)

        # DDD estimate is coefficient of triple_diff
        if 'triple_diff' in X_full.columns:
            ddd_estimate = model.coef_[X_full.columns.get_loc('triple_diff')]
        else:
            ddd_estimate = np.nan

        return {
            'ddd_estimate': ddd_estimate,
            'model_r2': model.score(X_full, y),
            'model': model
        }

# Run DID analysis
didd_analyzer = DifferenceInDifferencesAnalyzer()

print("=== Difference-in-Differences Analysis ===\n")

# Create synthetic panel data
print("1. Creating synthetic panel data...")
panel_data = didd_analyzer.create_panel_data(n_units=100, n_periods=10, treatment_start_period=6)

print(f"Panel data created: {len(panel_data)} observations")
print(f"Units: {panel_data['unit'].nunique()}")
print(f"Time periods: {panel_data['time'].nunique()}")
print(f"Treatment group size: {panel_data['treated'].sum() / len(panel_data):.3f}")

# Estimate DID
print("\n2. Estimating Difference-in-Differences...")
did_results = didd_analyzer.estimate_did(panel_data)

print("DID Results:")
print(f"  DID Estimate: {did_results['did_estimate']:.3f}")
print(f"  Standard Error: {did_results['standard_error']:.3f}")
print(f"  T-statistic: {did_results['t_statistic']:.3f}")
print(f"  P-value: {did_results['p_value']:.4f}")
print(f"  R-squared: {did_results['model_r2']:.3f}")

# Test parallel trends
print("\n3. Testing Parallel Trends Assumption...")
parallel_results = didd_analyzer._test_parallel_trends(panel_data)

if parallel_results['test_possible']:
    print("Parallel Trends Test:")
    print(f"  Time trend coefficient: {parallel_results['time_trend_coefficient']:.3f}")
    print(f"  P-value: {parallel_results['p_value']:.4f}")
    print(f"  Parallel trends satisfied: {parallel_results['parallel_trends_satisfied']}")
else:
    print(f"  Test not possible: {parallel_results['reason']}")

# Event study analysis
print("\n4. Event Study Analysis...")
event_study_results = didd_analyzer.event_study(panel_data, treatment_start_period=5)

print("Event Study Results:")
print(f"  Model R-squared: {event_study_results['model_r2']:.3f}")
print("  Event study coefficients:")
for var, coef in event_study_results['event_coefficients'].items():
    se = event_study_results['standard_errors'][var]
    t_stat = coef / se
    print(f"    {var}: {coef:.3f} (t={t_stat:.2f})")

# Triple differences (if applicable)
print("\n5. Triple Differences Analysis...")
# Add third dimension
panel_data['region'] = np.random.choice([0, 1], len(panel_data))
ddd_results = didd_analyzer.triple_differences(panel_data)

print(f"DDD Estimate: {ddd_results['ddd_estimate']:.3f}")
print(f"Model R-squared: {ddd_results['model_r2']:.3f}")

# Visualize results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. Group-time trends
group_0_data = panel_data[panel_data['treated'] == 0].groupby('time')['outcome'].mean()
group_1_data = panel_data[panel_data['treated'] == 1].groupby('time')['outcome'].mean()

ax1.plot(group_0_data.index, group_0_data.values, 'b-', label='Control Group', linewidth=2)
ax1.plot(group_1_data.index, group_1_data.values, 'r-', label='Treatment Group', linewidth=2)
ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.7, label='Treatment Start')
ax1.set_xlabel('Time Period')
ax1.set_ylabel('Mean Outcome')
ax1.set_title('Outcome Trends by Group')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. DID estimation visualization
categories = ['Pre-Treatment', 'Post-Treatment']
control_means = [
    panel_data[(panel_data['treated'] == 0) & (panel_data['post'] == 0)]['outcome'].mean(),
    panel_data[(panel_data['treated'] == 0) & (panel_data['post'] == 1)]['outcome'].mean()
]
treatment_means = [
    panel_data[(panel_data['treated'] == 1) & (panel_data['post'] == 0)]['outcome'].mean(),
    panel_data[(panel_data['treated'] == 1) & (panel_data['post'] == 1)]['outcome'].mean()
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x - width/2, control_means, width, label='Control', color='skyblue')
bars2 = ax2.bar(x + width/2, treatment_means, width, label='Treatment', color='lightcoral')

ax2.set_xlabel('Time Period')
ax2.set_ylabel('Mean Outcome')
ax2.set_title('DID Visualization')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()

# Add DID estimate arrow
ax2.annotate('', xy=(1.2, treatment_means[1]), xytext=(1.2, control_means[1]),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax2.text(1.25, (treatment_means[1] + control_means[1])/2, f'DID={did_results["did_estimate"]:.2f}',
         rotation=90, va='center', color='red', fontweight='bold')

# 3. Event study plot
relative_times = sorted([float(var.split('_')[1]) for var in event_study_results['event_coefficients'].keys()])
coefficients = [event_study_results['event_coefficients'][f'{"lead" if rt < 0 else "lag"}_{abs(int(rt))}'] for rt in relative_times]
standard_errors = [event_study_results['standard_errors'][f'{"lead" if rt < 0 else "lag"}_{abs(int(rt))}'] for rt in relative_times]

ax3.errorbar(relative_times, coefficients, yerr=standard_errors, fmt='bo-', capsize=5)
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Treatment Start')
ax3.set_xlabel('Relative Time')
ax3.set_ylabel('Event Study Coefficient')
ax3.set_title('Event Study Results')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Parallel trends test
if parallel_results['test_possible']:
    pre_periods = panel_data[panel_data['post'] == 0]['time'].unique()
    treated_pre = panel_data[(panel_data['treated'] == 1) & (panel_data['post'] == 0)]
    control_pre = panel_data[(panel_data['treated'] == 0) & (panel_data['post'] == 0)]

    treated_trend = treated_pre.groupby('time')['outcome'].mean()
    control_trend = control_pre.groupby('time')['outcome'].mean()

    ax4.plot(treated_trend.index, treated_trend.values, 'r-', label='Treatment (Pre)', linewidth=2)
    ax4.plot(control_trend.index, control_trend.values, 'b-', label='Control (Pre)', linewidth=2)

    # Fit and plot trend lines
    treated_slope, treated_intercept = np.polyfit(treated_trend.index, treated_trend.values, 1)
    control_slope, control_intercept = np.polyfit(control_trend.index, control_trend.values, 1)

    ax4.plot(treated_trend.index, treated_slope * treated_trend.index + treated_intercept,
             'r--', alpha=0.7, label='Treatment Trend')
    ax4.plot(control_trend.index, control_slope * control_trend.index + control_intercept,
             'b--', alpha=0.7, label='Control Trend')

    ax4.set_xlabel('Time (Pre-Treatment)')
    ax4.set_ylabel('Mean Outcome')
    ax4.set_title('Parallel Trends Test')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add text box with test results
    textstr = f"Trend Diff: {parallel_results['time_trend_coefficient']:.3f}\nP-value: {parallel_results['p_value']:.4f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Summary
print("\n=== Analysis Summary ===")
print(f"✅ DID Estimate: {did_results['did_estimate']:.3f} (SE: {did_results['standard_error']:.3f})")
if 'parallel_trends_satisfied' in parallel_results and parallel_results['test_possible']:
    print(f"✅ Parallel Trends: {'Satisfied' if parallel_results['parallel_trends_satisfied'] else 'Violated'}")
else:
print(f"⚠️ Parallel Trends: Test not possible")
print(f"✅ Model Fit: R² = {did_results['model_r2']:.3f}")
```

---

## 5. Real-world Applications

### Exercise 5.1: Healthcare Causal Analysis

```python
# Exercise: Healthcare treatment effect analysis
class HealthcareCausalAnalysis:
    def __init__(self):
        pass

    def generate_clinical_trial_data(self, n_patients=1000):
        """Generate realistic clinical trial data"""

        np.random.seed(42)

        # Patient characteristics
        age = np.random.normal(65, 12, n_patients)
        age = np.clip(age, 18, 90)  # Realistic age bounds

        gender = np.random.binomial(1, 0.55, n_patients)  # Slightly more females

        # Medical history (confounders)
        diabetes = np.random.binomial(1, 0.25, n_patients)
        hypertension = np.random.binomial(1, 0.45, n_patients)
        bmi = np.random.normal(28, 5, n_patients)
        bmi = np.clip(bmi, 15, 50)  # Realistic BMI bounds

        # Baseline severity score (0-100)
        baseline_severity = np.random.normal(60, 15, n_patients)
        baseline_severity = np.clip(baseline_severity, 0, 100)

        # Treatment assignment (stratified by severity)
        # Higher severity patients more likely to receive treatment
        severity_prob = 0.3 + 0.4 * (baseline_severity / 100)
        treatment = np.random.binomial(1, severity_prob)

        # Compliance (affected by age and side effects)
        # Older patients less compliant
        age_effect = 0.9 - 0.3 * ((age - 18) / 72)  # Age 18 -> 0.9, age 90 -> 0.6
        compliance = np.random.binomial(1, age_effect)

        # Remove compliance for non-treated patients
        compliance = compliance * treatment

        # Side effects (affect adherence)
        side_effects = np.random.binomial(1, 0.15)  # 15% experience side effects

        # Potential outcomes (symptom improvement)
        # Base improvement (control group)
        control_improvement = (
            20 +  # Base improvement
            0.3 * age +  # Younger patients improve more
            5 * (gender == 0) +  # Male advantage
            0.2 * (100 - baseline_severity) +  # Less severe cases improve more
            10 * diabetes -  # Diabetes reduces improvement
            5 * hypertension -  # Hypertension reduces improvement
            0.1 * bmi +  # Higher BMI slightly reduces improvement
            np.random.normal(0, 10, n_patients)  # Random variation
        )

        # Treatment effect (additional improvement from treatment)
        treatment_effect = (
            15 +  # Base treatment effect
            5 * compliance +  # Compliance increases effectiveness
            3 * (100 - baseline_severity) +  # More effective for less severe cases
            np.random.normal(0, 3, n_patients)  # Random variation
        )

        # Potential outcomes
        y0 = control_improvement  # Outcome under control
        y1 = y0 + treatment_effect  # Outcome under treatment

        # Observed outcome
        observed_improvement = treatment * y1 + (1 - treatment) * y0

        # Adverse events (for safety analysis)
        adverse_events = (
            0.05 +  # Base rate
            0.02 * treatment +  # Treatment increases risk slightly
            0.01 * age +  # Older patients higher risk
            0.03 * (diabetes == 1) +  # Diabetes increases risk
            np.random.normal(0, 0.02, n_patients)  # Random variation
        )
        adverse_events = np.clip(adverse_events, 0, 0.5)  # Bounded

        # Create DataFrame
        data = pd.DataFrame({
            'patient_id': range(n_patients),
            'age': age,
            'gender': gender,  # 0=male, 1=female
            'diabetes': diabetes,
            'hypertension': hypertension,
            'bmi': bmi,
            'baseline_severity': baseline_severity,
            'treatment': treatment,
            'compliance': compliance,
            'side_effects': side_effects,
            'y0': y0,
            'y1': y1,
            'improvement': observed_improvement,
            'adverse_events': adverse_events
        })

        # Calculate individual treatment effects
        data['ite'] = data['y1'] - data['y0']

        return data

    def analyze_treatment_effectiveness(self, data):
        """Analyze treatment effectiveness with multiple methods"""

        results = {}

        # 1. Intention-to-Treat Analysis
        treated = data[data['treatment'] == 1]
        control = data[data['treatment'] == 0]

        itt_effect = treated['improvement'].mean() - control['improvement'].mean()

        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(treated['improvement'], control['improvement'])

        results['intention_to_treat'] = {
            'effect': itt_effect,
            'p_value': p_value,
            'n_treated': len(treated),
            'n_control': len(control),
            'mean_treated': treated['improvement'].mean(),
            'mean_control': control['improvement'].mean()
        }

        # 2. Per-Protocol Analysis
        protocol_patients = data[data['compliance'] == 1]
        protocol_treated = protocol_patients[protocol_patients['treatment'] == 1]
        protocol_control = protocol_patients[protocol_patients['treatment'] == 0]

        if len(protocol_treated) > 0 and len(protocol_control) > 0:
            pp_effect = protocol_treated['improvement'].mean() - protocol_control['improvement'].mean()
            t_stat_pp, p_value_pp = ttest_ind(protocol_treated['improvement'], protocol_control['improvement'])

            results['per_protocol'] = {
                'effect': pp_effect,
                'p_value': p_value_pp,
                'n_treated': len(protocol_treated),
                'n_control': len(protocol_control),
                'mean_treated': protocol_treated['improvement'].mean(),
                'mean_control': protocol_control['improvement'].mean()
            }

        # 3. Subgroup Analysis
        subgroups = ['gender', 'diabetes', 'hypertension']
        subgroup_results = {}

        for subgroup in subgroups:
            subgroup_results[subgroup] = {}

            for subgroup_value in data[subgroup].unique():
                subgroup_data = data[data[subgroup] == subgroup_value]
                subgroup_treated = subgroup_data[subgroup_data['treatment'] == 1]
                subgroup_control = subgroup_data[subgroup_data['treatment'] == 0]

                if len(subgroup_treated) > 5 and len(subgroup_control) > 5:
                    effect = subgroup_treated['improvement'].mean() - subgroup_control['improvement'].mean()
                    subgroup_results[subgroup][subgroup_value] = {
                        'effect': effect,
                        'n_treated': len(subgroup_treated),
                        'n_control': len(subgroup_control)
                    }

        results['subgroup_analysis'] = subgroup_results

        # 4. Dose-Response Analysis (by compliance level)
        compliance_levels = [0, 1]
        dose_response = {}

        for compliance_level in compliance_levels:
            if compliance_level == 1:
                level_data = data[data['compliance'] == 1]
                effect = level_data[level_data['treatment'] == 1]['improvement'].mean() - \
                        level_data[level_data['treatment'] == 0]['improvement'].mean()
            else:
                level_data = data[data['compliance'] == 0]
                effect = level_data[level_data['treatment'] == 1]['improvement'].mean() - \
                        level_data[level_data['treatment'] == 0]['improvement'].mean()

            dose_response[f'compliance_{compliance_level}'] = {
                'effect': effect,
                'n': len(level_data)
            }

        results['dose_response'] = dose_response

        return results

    def heterogeneous_effects_analysis(self, data, effect_modifiers):
        """Analyze heterogeneous treatment effects"""

        from sklearn.ensemble import RandomForestRegressor

        results = {}

        # Filter to treated patients only
        treated_data = data[data['treatment'] == 1].copy()

        for modifier in effect_modifiers:
            if modifier not in treated_data.columns:
                continue

            modifier_values = treated_data[modifier].unique()
            modifier_effects = {}

            for value in modifier_values:
                value_data = treated_data[treated_data[modifier] == value]

                if len(value_data) < 20:  # Skip small subgroups
                    continue

                # Calculate effect for this subgroup
                # Compare to overall control group (as reference)
                control_reference = data[data['treatment'] == 0]

                if len(value_data) > 0 and len(control_reference) > 0:
                    effect = value_data['improvement'].mean() - control_reference['improvement'].mean()

                    modifier_effects[value] = {
                        'effect': effect,
                        'n': len(value_data),
                        'mean_improvement': value_data['improvement'].mean()
                    }

            results[modifier] = modifier_effects

        return results

    def propensity_score_analysis(self, data, covariates):
        """Propensity score matching analysis"""

        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors

        # Estimate propensity scores
        X = data[covariates].values
        treatment = data['treatment'].values

        ps_model = LogisticRegression()
        ps_model.fit(X, treatment)
        propensity_scores = ps_model.predict_proba(X)[:, 1]

        # Propensity score matching
        treated_mask = treatment == 1
        control_mask = treatment == 0

        matched_pairs = []
        used_controls = set()

        # Find matches for each treated unit
        treated_ps = propensity_scores[treated_mask]
        control_ps = propensity_scores[control_mask]

        treated_indices = np.where(treated_mask)[0]
        control_indices = np.where(control_mask)[0]

        for i, treated_idx in enumerate(treated_indices):
            treated_ps_val = propensity_scores[treated_idx]

            # Find closest control
            ps_differences = np.abs(control_ps - treated_ps_val)
            closest_control_idx = np.argmin(ps_differences)
            closest_control_original_idx = control_indices[closest_control_idx]

            if closest_control_original_idx not in used_controls:
                matched_pairs.append((treated_idx, closest_control_original_idx))
                used_controls.add(closest_control_original_idx)

        # Calculate treatment effect from matched pairs
        matched_effects = []
        for treated_idx, control_idx in matched_pairs:
            treated_outcome = data.iloc[treated_idx]['improvement']
            control_outcome = data.iloc[control_idx]['improvement']
            matched_effects.append(treated_outcome - control_outcome)

        ps_ate = np.mean(matched_effects) if matched_effects else np.nan

        # Propensity score stratification
        data['ps_stratum'] = pd.cut(propensity_scores, bins=5, labels=False)

        stratified_effects = []
        weights = []

        for stratum in data['ps_stratum'].unique():
            stratum_data = data[data['ps_stratum'] == stratum]

            stratum_treated = stratum_data[stratum_data['treatment'] == 1]
            stratum_control = stratum_data[stratum_data['treatment'] == 0]

            if len(stratum_treated) > 0 and len(stratum_control) > 0:
                stratum_effect = (stratum_treated['improvement'].mean() -
                                stratum_control['improvement'].mean())
                stratum_weight = len(stratum_data) / len(data)

                stratified_effects.append(stratum_effect * stratum_weight)
                weights.append(stratum_weight)

        stratified_ate = sum(stratified_effects)

        return {
            'propensity_score_auc': ps_model.score(X, treatment),
            'matching_ate': ps_ate,
            'stratification_ate': stratified_ate,
            'num_matched_pairs': len(matched_pairs),
            'propensity_scores': propensity_scores
        }

    def safety_analysis(self, data):
        """Analyze treatment safety and adverse events"""

        # Adverse event rates
        treated_ae_rate = data[data['treatment'] == 1]['adverse_events'].mean()
        control_ae_rate = data[data['treatment'] == 0]['adverse_events'].mean()

        # Statistical test for difference in AE rates
        from scipy.stats import chi2_contingency

        # Convert to binary outcome
        ae_binary_treated = (data[data['treatment'] == 1]['adverse_events'] > 0.1).astype(int)
        ae_binary_control = (data[data['treatment'] == 0]['adverse_events'] > 0.1).astype(int)

        contingency_table = np.array([
            [ae_binary_treated.sum(), len(ae_binary_treated) - ae_binary_treated.sum()],
            [ae_binary_control.sum(), len(ae_binary_control) - ae_binary_control.sum()]
        ])

        chi2, p_value_ae, dof, expected = chi2_contingency(contingency_table)

        # Risk ratio and confidence interval
        risk_ratio = treated_ae_rate / control_ae_rate if control_ae_rate > 0 else np.nan

        return {
            'treated_ae_rate': treated_ae_rate,
            'control_ae_rate': control_ae_rate,
            'risk_difference': treated_ae_rate - control_ae_rate,
            'risk_ratio': risk_ratio,
            'chi2_statistic': chi2,
            'p_value': p_value_ae,
            'ae_binary_treated': ae_binary_treated.mean(),
            'ae_binary_control': ae_binary_control.mean()
        }

# Run healthcare analysis
healthcare_analyzer = HealthcareCausalAnalysis()

print("=== Healthcare Causal Analysis ===\n")

# Generate clinical trial data
print("1. Generating clinical trial data...")
clinical_data = healthcare_analyzer.generate_clinical_trial_data(n_patients=800)

print(f"Clinical trial data: {len(clinical_data)} patients")
print(f"Treatment rate: {clinical_data['treatment'].mean():.3f}")
print(f"Compliance rate: {clinical_data['compliance'].mean():.3f}")
print(f"Mean age: {clinical_data['age'].mean():.1f}")
print(f"Female proportion: {clinical_data['gender'].mean():.3f}")

# Calculate true treatment effects
true_ate = clinical_data['ite'].mean()
true_att = clinical_data[clinical_data['treatment'] == 1]['ite'].mean()

print(f"\nTrue Average Treatment Effect: {true_ate:.3f}")
print(f"True Average Treatment Effect on Treated: {true_att:.3f}")

# Analyze treatment effectiveness
print("\n2. Treatment Effectiveness Analysis...")
effectiveness_results = healthcare_analyzer.analyze_treatment_effectiveness(clinical_data)

print("Treatment Effectiveness Results:")
print(f"  Intention-to-Treat: {effectiveness_results['intention_to_treat']['effect']:.3f} (p={effectiveness_results['intention_to_treat']['p_value']:.4f})")

if 'per_protocol' in effectiveness_results:
    print(f"  Per-Protocol: {effectiveness_results['per_protocol']['effect']:.3f} (p={effectiveness_results['per_protocol']['p_value']:.4f})")

print("\n  Subgroup Analysis:")
for subgroup, results in effectiveness_results['subgroup_analysis'].items():
    print(f"    {subgroup}:")
    for value, result in results.items():
        print(f"      {value}: {result['effect']:.3f} (n={result['n_treated']})")

# Safety analysis
print("\n3. Safety Analysis...")
safety_results = healthcare_analyzer.safety_analysis(clinical_data)

print("Safety Results:")
print(f"  Adverse Event Rate (Treated): {safety_results['treated_ae_rate']:.3f}")
print(f"  Adverse Event Rate (Control): {safety_results['control_ae_rate']:.3f}")
print(f"  Risk Difference: {safety_results['risk_difference']:.3f}")
print(f"  Risk Ratio: {safety_results['risk_ratio']:.3f}")
print(f"  P-value: {safety_results['p_value']:.4f}")

# Heterogeneous effects analysis
print("\n4. Heterogeneous Effects Analysis...")
heterogeneity_modifiers = ['age', 'gender', 'baseline_severity']
heterogeneity_results = healthcare_analyzer.heterogeneous_effects_analysis(
    clinical_data, heterogeneity_modifiers
)

for modifier, results in heterogeneity_results.items():
    print(f"  {modifier}:")
    for value, result in results.items():
        print(f"    {value}: {result['effect']:.3f} (n={result['n']})")

# Propensity score analysis
print("\n5. Propensity Score Analysis...")
covariates = ['age', 'gender', 'diabetes', 'hypertension', 'bmi', 'baseline_severity']
ps_results = healthcare_analyzer.propensity_score_analysis(clinical_data, covariates)

print("Propensity Score Results:")
print(f"  Propensity Score AUC: {ps_results['propensity_score_auc']:.3f}")
print(f"  Matching ATE: {ps_results['matching_ate']:.3f}")
print(f"  Stratification ATE: {ps_results['stratification_ate']:.3f}")
print(f"  Number of Matched Pairs: {ps_results['num_matched_pairs']}")

# Compare all methods
print("\n6. Method Comparison:")
print("Method Comparison (Treatment Effect Estimates):")
print(f"  True ATE: {true_ate:.3f}")
print(f"  True ATT: {true_att:.3f}")
print(f"  ITT Estimate: {effectiveness_results['intention_to_treat']['effect']:.3f}")
if 'per_protocol' in effectiveness_results:
    print(f"  Per-Protocol: {effectiveness_results['per_protocol']['effect']:.3f}")
print(f"  Matching: {ps_results['matching_ate']:.3f}")
print(f"  Stratification: {ps_results['stratification_ate']:.3f}")

# Visualize results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. Treatment effect by compliance level
compliance_data = clinical_data.groupby(['treatment', 'compliance'])['improvement'].mean().unstack()
compliance_data.plot(kind='bar', ax=ax1, color=['lightcoral', 'skyblue'])
ax1.set_title('Mean Improvement by Treatment and Compliance')
ax1.set_ylabel('Mean Improvement')
ax1.set_xlabel('Treatment Group')
ax1.legend(['Non-compliant', 'Compliant'])
ax1.tick_params(axis='x', rotation=0)

# 2. Heterogeneous effects by age
age_bins = pd.cut(clinical_data['age'], bins=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
age_effects = clinical_data.groupby([age_bins, 'treatment'])['improvement'].mean().unstack()
age_effects['difference'] = age_effects[1] - age_effects[0]

age_effects['difference'].plot(kind='bar', ax=ax2, color='green', alpha=0.7)
ax2.set_title('Treatment Effect by Age Quintile')
ax2.set_ylabel('Treatment Effect')
ax2.set_xlabel('Age Quintile')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
ax2.tick_params(axis='x', rotation=45)

# 3. Propensity score distribution
treated_ps = clinical_data[clinical_data['treatment'] == 1]['propensity_scores']
control_ps = clinical_data[clinical_data['treatment'] == 0]['propensity_scores']

ax3.hist(treated_ps, bins=20, alpha=0.7, label='Treated', color='blue')
ax3.hist(control_ps, bins=20, alpha=0.7, label='Control', color='red')
ax3.set_xlabel('Propensity Score')
ax3.set_ylabel('Frequency')
ax3.set_title('Propensity Score Distribution')
ax3.legend()

# 4. Method comparison
methods = ['True ATE', 'ITT', 'Per-Protocol', 'Matching', 'Stratification']
values = [
    true_ate,
    effectiveness_results['intention_to_treat']['effect'],
    effectiveness_results.get('per_protocol', {}).get('effect', np.nan),
    ps_results['matching_ate'],
    ps_results['stratification_ate']
]

# Remove NaN values
valid_methods = [(m, v) for m, v in zip(methods, values) if not np.isnan(v)]
methods_clean, values_clean = zip(*valid_methods)

bars = ax4.bar(range(len(methods_clean)), values_clean,
               color=['red', 'skyblue', 'lightgreen', 'orange', 'purple'], alpha=0.7)
ax4.set_ylabel('Treatment Effect')
ax4.set_title('Method Comparison')
ax4.set_xticks(range(len(methods_clean)))
ax4.set_xticklabels(methods_clean, rotation=45)
ax4.axhline(y=true_ate, color='red', linestyle='--', alpha=0.7, label='True ATE')
ax4.legend()

# Add value labels
for bar, value in zip(bars, values_clean):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n=== Healthcare Analysis Complete ===")
```

This practice guide provides comprehensive hands-on exercises for implementing and testing advanced statistical methods and causal inference techniques across multiple domains and applications.
