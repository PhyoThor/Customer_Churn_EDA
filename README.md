# Customer Churn Prediction — Exploratory Data Analysis (EDA)

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/phyothaw/customer-churn-prediction-dataset-eda)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Summary](#dataset-summary)
3. [Feature Descriptions](#feature-descriptions)
4. [Project Structure](#project-structure)
5. [EDA Methodology](#eda-methodology)
6. [Key Findings](#key-findings)
7. [Important Limitations](#important-limitations)
8. [Conclusions](#conclusions)
9. [Technologies Used](#technologies-used)

---

## Project Overview

This project performs an **Exploratory Data Analysis (EDA)** on a large-scale customer churn prediction dataset sourced from Kaggle. The goal is to understand the characteristics of churned vs. non-churned customers, identify the most influential features, and surface any data quality concerns before predictive modelling.

**Churn** refers to the event of a customer discontinuing their subscription or service. Predicting churn is critical for businesses to proactively retain customers and reduce revenue loss.

---

## Dataset Summary

| Property | Value |
|---|---|
| **Source** | [Kaggle – Customer Churn Prediction Dataset 1M](https://www.kaggle.com/datasets/isandeep06/customer-churn-prediction-dataset-1m) |
| **Rows** | 1,000,000 |
| **Columns** | 33 |
| **Target variable** | `churn` (binary: 0 = not churned, 1 = churned) |
| **Class distribution** | Not churned: 900,773 (90.08%) · Churned: 99,227 (9.92%) |

The dataset is **highly imbalanced** — the majority class (non-churned) accounts for ~90% of all records. Any predictive model built on this data must explicitly handle class imbalance (e.g., oversampling, undersampling, or class-weight adjustment).

### Missing Values

Several columns contain missing values and require imputation or removal before modelling:

| Column | Missing (approx.) |
|---|---|
| `avg_monthly_gb` | highest missing rate |
| `credit_score` | moderate |
| `annual_income` | moderate |
| `num_complaints` | moderate |
| `customer_satisfaction` | moderate |

---

## Feature Descriptions

### Customer Profile Features

| Feature | Description |
|---|---|
| `age` | Customer age in years |
| `gender` | Customer gender |
| `annual_income` | Annual income (USD) |
| `education` | Education level |
| `marital_status` | Marital status |
| `dependents` | Number of dependents |
| `senior_citizen` | Whether the customer is a senior citizen (0/1) |

### Subscription & Service Features

| Feature | Description |
|---|---|
| `num_services` | Total number of services subscribed |
| `has_phone_service` | Phone service subscription (0/1) |
| `has_internet_service` | Internet service subscription (0/1) |
| `has_online_security` | Online security add-on (0/1) |
| `has_online_backup` | Online backup add-on (0/1) |
| `has_device_protection` | Device protection add-on (0/1) |
| `has_tech_support` | Tech support add-on (0/1) |
| `has_streaming_tv` | Streaming TV add-on (0/1) |
| `has_streaming_movies` | Streaming movies add-on (0/1) |

### Billing & Usage Features

| Feature | Description |
|---|---|
| `monthlycharges` | Monthly bill amount (USD) |
| `totalcharges` | Cumulative charges to date (USD) |
| `late_payments` | Number of late payments |
| `avg_monthly_gb` | Average monthly data usage (GB) |
| `payment_method` | Payment method type |
| `paperless_billing` | Whether customer uses paperless billing (0/1) |
| `contract` | Contract type (month-to-month, one year, two year) |

### Customer Experience Features

| Feature | Description |
|---|---|
| `customer_satisfaction` | Satisfaction score |
| `num_complaints` | Number of complaints filed |
| `num_service_calls` | Number of service/support calls |
| `days_since_last_interaction` | Days since the customer last interacted with the service |
| `credit_score` | Credit score |
| `tenure` | Number of months the customer has been with the company |

---

## Project Structure

```
Customer_Churn_EDA/
├── README.md                                    # Project documentation (this file)
└── Customer_Churn_Prediction_Dataset_EDA.ipynb  # Main EDA notebook
```

---

## EDA Methodology

The notebook follows a structured EDA workflow:

1. **Data Loading** — Load the 1M-row CSV directly from the Kaggle dataset path.
2. **Initial Inspection** — `head()`, `shape`, `info()` to understand structure and types.

3. **Missing Value Analysis**
   - Percentage of nulls per column
   - Missing value heatmap (seaborn)
   - Null counts for churned vs. non-churned customers separately
4. **Duplicate Check** — Percentage of duplicate rows.
5. **Target Distribution** — Count plot of `churn` (churned vs. not churned).
6. **Outlier Detection** — Boxplots for all numerical columns within the churned subset.
7. **Demographic Analysis** — Age-group distribution comparison (churned vs. not churned) using 10-year interval bins.
8. **Service Usage Analysis**
   - Number of services subscribed by churned customers (bar chart)
   - Total service subscriptions (donut chart)
   - Side-by-side comparison of churned vs. non-churned service totals with percentages
   - Churn rates broken down by number of services
9. **Complaint Analysis** — Total complaints per service among churned customers.
10. **Satisfaction vs. Services** — Mean customer satisfaction by number of services for churned vs. non-churned groups.
11. **Inactivity Analysis** — Distribution of churned customers by `days_since_last_interaction` (days 0–30).
12. **Tenure Analysis** — Churn count by tenure month with line chart.
13. **Correlation Heatmap** — Pearson correlation matrix of all numerical features.

---

## Key Findings

### 1. Severe Class Imbalance
The dataset is highly imbalanced (~90% not churned vs. ~10% churned). Techniques like SMOTE, class-weight adjustment, or under-sampling are required before training any classifier.

### 2. Services Subscribed and Churn
Churned customers are disproportionately concentrated in the **3 to 6 services** range:
- Customers with **4 or 5 active services** represent the single largest churned segment (~78.79% of churned customers fall in the 3–6 range).
- This counter-intuitive finding (more services → more churn) may indicate that higher engagement leads to more exposure to service issues, or that bundled packages contain less value for certain customer profiles.

### 3. Customer Satisfaction
Churned customers consistently show **lower customer satisfaction scores** than non-churned customers across all service levels, making `customer_satisfaction` one of the most useful predictors.

### 4. Customer Experience Features Outperform Demographics
Variables like `num_complaints`, `num_service_calls`, and `customer_satisfaction` appear more informative than demographic features (`age`, `gender`, `annual_income`).

### 5. Tenure Pattern (U-shaped)
- Churn is **highest at month 1** of tenure — suggesting onboarding issues or dataset-specific rules.
- Churn rates remain lower through the middle tenure range.
- A secondary spike appears at **tenure = 72 months** (the maximum), possibly reflecting contract-end behaviour or a data-generation artifact.

### 6. Weak Individual Linear Correlations with Churn
The correlation heatmap shows that **no single variable has a strong linear relationship with `churn`**. Churn is likely driven by a combination of features, which favours ensemble methods (e.g., Random Forest, XGBoost) over simple linear models.

---

## Important Limitations

### `days_since_last_interaction` — Questionable Churn Definition
One of the most critical findings in this EDA is that even **1 day of inactivity** is associated with churn in this dataset. This likely reflects an overly broad or incorrect churn definition in the data-generation process. In real-world settings, a single inactive day would almost never constitute customer churn. This variable should be either:
- **Excluded** from predictive features, or
- **Treated with extreme caution** and validated against business definitions before use.

### Missing Values
Five columns have non-trivial missing rates. Imputation strategy will affect downstream model quality.

### Class Imbalance
Standard accuracy metrics are misleading on this dataset. Evaluation should use **AUC-ROC**, **F1-score**, **precision-recall curve**, or other imbalance-aware metrics.

---

## Conclusions

This EDA demonstrates that the customer churn dataset is **large, feature-rich, and analytically valuable**, but several issues must be resolved before building a reliable predictive model:

| Issue | Recommended Action |
|---|---|
| Class imbalance (90/10 split) | Apply SMOTE, class weighting, or undersampling |
| Missing values in 5+ columns | Impute or remove depending on missingness pattern |
| Questionable churn definition (`days_since_last_interaction`) | Exclude or validate against business rules |
| Potential outliers in numerical features | Detect and treat before modelling |
| Weak single-feature correlations | Use ensemble or non-linear models |

The strongest practical signals for churn come from **customer experience and service usage behaviour** (satisfaction scores, complaint count, service calls). Demographic variables alone are unlikely to produce a strong churn model.

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3 | Core programming language |
| pandas | Data loading, cleaning, and aggregation |
| seaborn | Statistical visualizations (heatmaps, count plots) |
| matplotlib | Custom charts and subplots |
| IPython / Jupyter | Interactive notebook environment |
| Kaggle Notebooks | Cloud compute and dataset hosting |
