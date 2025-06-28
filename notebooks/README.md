# 📊 Exploratory Data Analysis (EDA) Insights

The initial **Exploratory Data Analysis (EDA)** on the `data.csv` dataset uncovered crucial patterns and risk areas that will inform further **data preprocessing**, **feature engineering**, and **model development**.

---

## 🔍 Key Findings

### 1. 🎯 Extremely Low Fraud Rate – Severe Class Imbalance

- **Insight:**
  Only **0.20%** of transactions are fraudulent, while **99.80%** are legitimate.
- **Implication:**
  This extreme imbalance presents a serious challenge for traditional machine learning models. Specialized techniques like:

  - **Oversampling** (e.g., SMOTE)
  - **Undersampling**
  - **Class-weighted models** (e.g., tree-based methods)
  - **Anomaly detection** (e.g., Isolation Forest)
    are essential to improve fraud detection.

---

### 2. 🛒 Fraud Clusters in Specific Product Categories

- **Insight:**
  Fraud is disproportionately concentrated in:

  - `transport` (**8.00%** fraud rate)
  - `utility_bill` (**0.62%**)
  - `financial_services` (**0.35%**)
    Other categories like `movies`, `tv`, `data_bundles`, `ticket`, and `other` showed **no fraud**.

- **Implication:**
  These high-risk categories suggest targeted fraud strategies. Features related to product type are valuable for modeling and **risk segmentation**.

---

### 3. 🌐 Channel-Specific Fraud Vulnerability

- **Insight:**
  Fraud rates vary across channels:

  - `ChannelId_1`: **0.74%**
  - `ChannelId_3`: **0.32%**
  - `ChannelId_2`: **0.01%**
  - `ChannelId_5`: **0.00%**

- **Implication:**
  Channels differ in fraud exposure, possibly due to:

  - Security policies
  - Transaction types
  - User demographics
    Prioritizing **ChannelId_1** and **ChannelId_3** for deeper analysis is recommended.

---

### 4. 💱 Currency Code as a Contextual Indicator

- **Insight:**
  `CurrencyCode` **UGX (Ugandan Shilling)** aligns with the overall fraud rate (**0.20%**).
- **Implication:**
  While not significant in this subset, `CurrencyCode` may reveal **regional fraud trends** if additional currencies are included.

---

## 🧭 Next Steps

- Address the class imbalance using sampling techniques
- Engineer features from high-risk product and channel attributes

---
