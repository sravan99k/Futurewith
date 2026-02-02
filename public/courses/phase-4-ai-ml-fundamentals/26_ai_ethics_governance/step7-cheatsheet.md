# AI Ethics & Governance - Cheatsheet

## Quick Reference Guide

## Bias Detection and Fairness Metrics

### Common Fairness Metrics

```python
# Demographic Parity
def demographic_parity(y_pred, protected_attr):
    groups = np.unique(protected_attr)
    rates = {group: np.mean(y_pred[protected_attr == group]) for group in groups}
    return max(rates.values()) - min(rates.values())

# Equalized Odds (TPR parity)
def equalized_odds(y_true, y_pred, protected_attr):
    groups = np.unique(protected_attr)
    tpr = {}
    for group in groups:
        mask = protected_attr == group
        tpr[group] = np.sum((y_true[mask] == 1) & (y_pred[mask] == 1)) / np.sum(y_true[mask] == 1) if np.sum(y_true[mask] == 1) > 0 else 0
    return max(tpr.values()) - min(tpr.values())

# Calibration
def calibration(y_true, y_pred_prob, protected_attr):
    groups = np.unique(protected_attr)
    calibration_error = {}
    for group in groups:
        mask = protected_attr == group
        pred_rate = np.mean(y_pred_prob[mask])
        actual_rate = np.mean(y_true[mask])
        calibration_error[group] = abs(pred_rate - actual_rate)
    return max(calibration_error.values()) - min(calibration_error.values())
```

### Bias Detection Statistical Tests

```python
from scipy import stats

def statistical_bias_test(predictions, protected_attr, test_type='chi_square'):
    groups = np.unique(protected_attr)
    results = {}

    for i, group1 in enumerate(groups):
        for group2 in groups[i+1:]:
            group1_mask = protected_attr == group1
            group2_mask = protected_attr == group2

            group1_preds = predictions[group1_mask]
            group2_preds = predictions[group2_mask]

            if test_type == 'chi_square':
                contingency = np.array([
                    [np.sum(group1_preds), len(group1_preds) - np.sum(group1_preds)],
                    [np.sum(group2_preds), len(group2_preds) - np.sum(group2_preds)]
                ])
                stat, p_value = stats.chi2_contingency(contingency)[:2]
            elif test_type == 'ks_test':
                stat, p_value = stats.ks_2samp(group1_preds, group2_preds)

            results[f"{group1}_vs_{group2}"] = {
                'statistic': stat,
                'p_value': p_value,
                'biased': p_value < 0.05
            }

    return results
```

## Privacy-Preserving Techniques

### Differential Privacy

```python
import numpy as np

class DifferentialPrivacy:
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def laplace_noise(self, sensitivity):
        """Add Laplace noise for ε-differential privacy"""
        scale = sensitivity / self.epsilon
        return np.random.laplace(0, scale)

    def gaussian_noise(self, sensitivity, delta=1e-5):
        """Add Gaussian noise for (ε, δ)-differential privacy"""
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / self.epsilon
        return np.random.normal(0, sigma)

    def private_count(self, true_count, sensitivity=1):
        return true_count + self.laplace_noise(sensitivity)

    def private_mean(self, values, sensitivity=None):
        if sensitivity is None:
            sensitivity = (np.max(values) - np.min(values)) / len(values)
        return np.mean(values) + self.laplace_noise(sensitivity)
```

### Federated Learning

```python
class FederatedLearning:
    def __init__(self, global_model, privacy_budget=1.0):
        self.global_model = global_model
        self.privacy_budget = privacy_budget

    def secure_aggregation(self, client_updates):
        """Secure aggregation with privacy preservation"""
        # Add local differential privacy to each update
        private_updates = []
        for update in client_updates:
            # Clip to bound sensitivity
            clipped_update = np.clip(update, -1, 1)
            # Add noise
            noise_scale = self.privacy_budget / len(client_updates)
            noisy_update = clipped_update + np.random.laplace(0, noise_scale, size=update.shape)
            private_updates.append(noisy_update)

        # Aggregate
        return np.mean(private_updates, axis=0)
```

## Explainable AI (XAI)

### LIME Implementation

```python
import lime
import lime.lime_tabular

def create_lime_explainer(X_train, feature_names, class_names):
    """Create LIME explainer for tabular data"""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    return explainer

def explain_prediction(explainer, instance, model, num_features=10):
    """Generate explanation for a prediction"""
    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=num_features
    )
    return explanation
```

### SHAP Implementation

```python
import shap

def create_shap_explainer(model, X_train):
    """Create SHAP explainer"""
    if hasattr(model, 'tree_'):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, X_train[:100])
    return explainer

def explain_with_shap(explainer, X_test, instance_idx=0):
    """Generate SHAP explanation"""
    shap_values = explainer.shap_values(X_test[instance_idx:instance_idx+1])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    return shap_values[0]
```

## Ethical AI Development Checklist

### Pre-Development Checklist

- [ ] Ethical impact assessment completed
- [ ] Stakeholder consultation conducted
- [ ] Data collection ethics reviewed
- [ ] Bias sources identified
- [ ] Privacy implications assessed
- [ ] Regulatory compliance review
- [ ] Governance structure established

### Development Checklist

- [ ] Fairness metrics defined
- [ ] Bias testing framework implemented
- [ ] Privacy-preserving techniques applied
- [ ] Explainability features added
- [ ] Human oversight mechanisms designed
- [ ] Documentation created
- [ ] Testing with diverse datasets

### Deployment Checklist

- [ ] Bias validation on deployment data
- [ ] Privacy impact assessment updated
- [ ] Transparency measures activated
- [ ] Monitoring systems deployed
- [ ] Incident response procedures ready
- [ ] Stakeholder communication plan active

### Post-Deployment Checklist

- [ ] Regular bias audits scheduled
- [ ] Performance monitoring active
- [ ] Stakeholder feedback collection
- [ ] Model drift detection implemented
- [ ] Regular ethical reviews
- [ ] Continuous improvement process

## Governance Frameworks

### AI Ethics Board Structure

```python
class AIEthicsBoard:
    def __init__(self):
        self.members = {
            'ethics_officer': {'role': 'Chair', 'responsibilities': ['Overall oversight']},
            'technical_lead': {'role': 'Member', 'responsibilities': ['Technical assessment']},
            'legal_counsel': {'role': 'Member', 'responsibilities': ['Legal compliance']},
            'privacy_officer': {'role': 'Member', 'responsibilities': ['Privacy oversight']},
            'stakeholder_rep': {'role': 'Member', 'responsibilities': ['User representation']}
        }

    def review_ai_system(self, system_info):
        """Review AI system for ethical compliance"""
        review_criteria = {
            'bias_assessment': self._assess_bias(system_info),
            'privacy_compliance': self._assess_privacy(system_info),
            'transparency_check': self._assess_transparency(system_info),
            'risk_evaluation': self._assess_risks(system_info)
        }
        return self._make_decision(review_criteria)
```

### Risk Assessment Matrix

| Risk Category        | Likelihood | Impact | Risk Level | Mitigation Priority |
| -------------------- | ---------- | ------ | ---------- | ------------------- |
| Model Bias           | High       | High   | Critical   | Immediate           |
| Privacy Breach       | Medium     | High   | High       | High                |
| Regulatory Violation | Medium     | High   | High       | High                |
| Reputation Damage    | High       | Medium | High       | Medium              |
| System Failure       | Low        | High   | Medium     | Medium              |

## Compliance Frameworks

### GDPR Compliance Checklist

- [ ] Lawful basis for processing identified
- [ ] Data minimization principle applied
- [ ] Purpose limitation implemented
- [ ] Data retention periods defined
- [ ] Security measures implemented
- [ ] Data subject rights supported
- [ ] Privacy by design implemented
- [ ] DPO appointed (if required)
- [ ] DPIA conducted (if high risk)
- [ ] International transfers assessed

### EU AI Act Compliance

- [ ] AI system risk category identified
- [ ] Conformity assessment completed (high-risk)
- [ ] CE marking obtained (high-risk)
- [ ] Quality management system implemented
- [ ] Risk management system established
- [ ] Data governance measures implemented
- [ ] Technical documentation created
- [ ] Human oversight implemented
- [ ] Transparency measures activated
- [ ] Conformity marking affixed

## Monitoring and Alerting

### Bias Monitoring

```python
class BiasMonitor:
    def __init__(self, thresholds):
        self.thresholds = thresholds  # e.g., {'demographic_parity': 0.1}
        self.alerts = []

    def check_bias(self, predictions, protected_attributes):
        """Check for bias violations"""
        bias_violations = []

        # Check each protected attribute
        for attr in np.unique(protected_attributes):
            # Calculate fairness metrics
            dp_violation = self._check_demographic_parity(predictions, protected_attributes, attr)

            if dp_violation > self.thresholds.get('demographic_parity', 0.1):
                bias_violations.append({
                    'attribute': attr,
                    'violation_type': 'demographic_parity',
                    'severity': 'high' if dp_violation > 0.2 else 'medium',
                    'value': dp_violation
                })

        return bias_violations

    def _check_demographic_parity(self, predictions, protected_attributes, attr):
        """Calculate demographic parity violation"""
        groups = np.unique(protected_attributes)
        rates = {}

        for group in groups:
            mask = protected_attributes == group
            rates[group] = np.mean(predictions[mask])

        return max(rates.values()) - min(rates.values())
```

### Privacy Breach Detection

```python
class PrivacyMonitor:
    def __init__(self):
        self.access_logs = []
        self.anomaly_threshold = 0.05

    def detect_privacy_breach(self, access_pattern):
        """Detect potential privacy breaches"""
        anomalies = []

        # Check for unusual access patterns
        if access_pattern['failed_attempts'] > 5:
            anomalies.append({
                'type': 'excessive_failed_attempts',
                'severity': 'high',
                'details': access_pattern
            })

        # Check for data exfiltration
        if access_pattern['data_volume'] > self._get_baseline_volume() * 10:
            anomalies.append({
                'type': 'unusual_data_volume',
                'severity': 'critical',
                'details': access_pattern
            })

        return anomalies
```

## Incident Response

### AI Incident Response Plan

```python
class AIIncidentResponse:
    def __init__(self):
        self.incident_types = {
            'bias_incident': {
                'severity': 'high',
                'response_team': ['ethics_officer', 'ml_engineering', 'legal'],
                'timeline': 'immediate',
                'escalation': ['ethics_board', 'executive']
            },
            'privacy_incident': {
                'severity': 'critical',
                'response_team': ['dpo', 'legal', 'security'],
                'timeline': 'immediate',
                'escalation': ['regulatory_bodies', 'executive']
            },
            'bias_drift': {
                'severity': 'medium',
                'response_team': ['ml_engineering', 'ethics_officer'],
                'timeline': '24_hours',
                'escalation': ['ethics_board']
            }
        }

    def respond_to_incident(self, incident_type, details):
        """Respond to AI system incident"""
        if incident_type not in self.incident_types:
            return {'error': 'Unknown incident type'}

        response_plan = self.incident_types[incident_type]

        # Immediate response
        self._notify_response_team(response_plan['response_team'])
        self._create_incident_ticket(details)

        # Document incident
        incident_record = {
            'id': self._generate_incident_id(),
            'type': incident_type,
            'severity': response_plan['severity'],
            'timestamp': datetime.now().isoformat(),
            'details': details,
            'status': 'open'
        }

        # Execute response actions
        if incident_type == 'bias_incident':
            self._execute_bias_response(details)
        elif incident_type == 'privacy_incident':
            self._execute_privacy_response(details)

        return incident_record
```

## Documentation Templates

### Model Card Template

```markdown
# Model Card: [Model Name]

## Model Details

- **Name**: [Model name]
- **Version**: [Version number]
- **Date**: [Training date]
- **Developer**: [Organization]
- **Model Type**: [Architecture type]
- **License**: [Usage license]

## Intended Use

- **Primary Uses**: [Intended applications]
- **Primary Users**: [Target user groups]
- **Out of Scope**: [Prohibited uses]

## Training Data

- **Description**: [Data description]
- **Size**: [Dataset size]
- **Demographics**: [Demographic information]
- **Known Biases**: [Identified biases]

## Performance

- **Evaluation Metrics**: [Metrics used]
- **Results**: [Performance results]
- **Limitations**: [Known limitations]

## Ethical Considerations

- **Bias Analysis**: [Bias assessment results]
- **Fairness**: [Fairness evaluation]
- **Privacy**: [Privacy implications]

## Limitations

- **Technical**: [Technical limitations]
- **Ethical**: [Ethical limitations]
- **Operational**: [Operational limitations]
```

### Bias Audit Report Template

```markdown
# Bias Audit Report

## Executive Summary

- **Audit Date**: [Date]
- **System**: [AI system name]
- **Auditor**: [Auditor name]
- **Overall Rating**: [Rating]

## Methodology

- **Data Sources**: [Data used]
- **Methods**: [Audit methods]
- **Protected Attributes**: [Attributes tested]

## Findings

- **Statistical Tests**: [Test results]
- **Fairness Metrics**: [Metric results]
- **Performance Analysis**: [Performance across groups]

## Recommendations

- [List of recommendations]

## Compliance Status

- [Regulatory compliance status]
```

## Stakeholder Engagement

### Stakeholder Mapping

```python
class StakeholderMapping:
    def __init__(self, ai_system):
        self.stakeholders = {
            'primary': {
                'end_users': {'influence': 'high', 'interest': 'high'},
                'affected_individuals': {'influence': 'medium', 'interest': 'high'},
                'developers': {'influence': 'high', 'interest': 'high'}
            },
            'secondary': {
                'regulators': {'influence': 'high', 'interest': 'medium'},
                'advocacy_groups': {'influence': 'medium', 'interest': 'high'},
                'researchers': {'influence': 'medium', 'interest': 'medium'}
            }
        }

    def get_engagement_priority(self):
        """Get stakeholder engagement priorities"""
        priorities = {}

        for category, groups in self.stakeholders.items():
            for group, characteristics in groups.items():
                priority_score = characteristics['influence'] * characteristics['interest']
                priorities[group] = {
                    'category': category,
                    'priority_score': priority_score,
                    'engagement_strategy': self._get_engagement_strategy(characteristics)
                }

        return priorities
```

### Feedback Collection

```python
def collect_stakeholder_feedback(ai_system, stakeholder_groups):
    """Collect feedback from stakeholders"""
    feedback_methods = {
        'surveys': 'Structured questionnaires',
        'interviews': 'In-depth conversations',
        'focus_groups': 'Group discussions',
        'public_forums': 'Community meetings',
        'advisory_boards': 'Ongoing consultation'
    }

    feedback_results = {}

    for group in stakeholder_groups:
        group_feedback = {}

        for method, description in feedback_methods.items():
            # Collect feedback using appropriate method
            feedback = self._collect_method_feedback(group, method)
            group_feedback[method] = feedback

        feedback_results[group] = group_feedback

    return feedback_results
```

## Ethical AI Metrics

### Key Performance Indicators

```python
ethical_kpis = {
    'fairness_metrics': {
        'demographic_parity_violations': 0,
        'equalized_odds_gaps': 0,
        'calibration_errors': 0
    },
    'privacy_metrics': {
        'privacy_breaches': 0,
        'data_retention_compliance': 100,
        'consent_rates': 95
    },
    'transparency_metrics': {
        'explanation_availability': 100,
        'documentation_completeness': 90,
        'audit_trail_coverage': 100
    },
    'governance_metrics': {
        'compliance_score': 85,
        'incident_response_time': 24,  # hours
        'stakeholder_satisfaction': 4.2  # out of 5
    }
}
```

### Ethical Score Calculation

```python
def calculate_ethical_score(ai_system):
    """Calculate overall ethical score for AI system"""

    components = {
        'fairness': calculate_fairness_score(ai_system),
        'privacy': calculate_privacy_score(ai_system),
        'transparency': calculate_transparency_score(ai_system),
        'accountability': calculate_accountability_score(ai_system),
        'human_agency': calculate_human_agency_score(ai_system)
    }

    # Weighted average (weights can be customized)
    weights = {'fairness': 0.3, 'privacy': 0.25, 'transparency': 0.2,
              'accountability': 0.15, 'human_agency': 0.1}

    total_score = sum(components[key] * weights[key] for key in weights)

    return {
        'total_score': total_score,
        'component_scores': components,
        'grade': get_ethical_grade(total_score)
    }

def get_ethical_grade(score):
    """Convert score to letter grade"""
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    elif score >= 60: return 'D'
    else: return 'F'
```

## Quick Commands Reference

### Bias Testing Commands

```bash
# Run bias detection
python -m bias_detection --model model.pkl --data data.csv --protected-attributes gender,race

# Generate fairness report
python -m fairness_metrics --model model.pkl --test-data test_data.csv

# Audit model for bias
python -m audit_framework --system-name "Credit Approval AI" --compliance-level strict
```

### Privacy Analysis Commands

```bash
# Privacy impact assessment
python -m privacy_assessment --data user_data.csv --epsilon 1.0

# Differential privacy analysis
python -m privacy_tools --method laplace --sensitivity 1.0 --epsilon 1.0

# Federated learning privacy check
python -m federated_privacy --clients 10 --privacy-budget 1.0
```

### Transparency Tools

```bash
# Generate model card
python -m model_card_generator --model model.pkl --output model_card.md

# Create explanation dashboard
python -m explainability_dashboard --model model.pkl --port 8080

# Audit trail analysis
python -m audit_analysis --log-file audit.log --output report.html
```

### Compliance Checking

```python
# GDPR compliance check
def check_gdpr_compliance(ai_system):
    checklist = {
        'lawful_basis': ai_system.has_lawful_basis(),
        'data_minimization': ai_system.implements_data_minimization(),
        'purpose_limitation': ai_system.enforces_purpose_limitation(),
        'retention_limits': ai_system.has_retention_limits(),
        'security_measures': ai_system.implements_security(),
        'data_subject_rights': ai_system.supports_data_subject_rights()
    }

    compliance_score = sum(checklist.values()) / len(checklist) * 100
    return compliance_score, checklist
```

This cheatsheet provides quick reference for the most commonly used ethical AI practices, tools, and frameworks. Keep it handy for daily ethical AI development and assessment work!
