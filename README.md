# Credit Risk Probability Model

This project aims to develop a credit risk model to assess the likelihood of default for individuals or entities.

---

## Credit Scoring Business Understanding

This section addresses fundamental business considerations in developing a credit risk model, particularly within a regulated financial context.

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord places significant emphasis on banks accurately measuring and managing their credit risk exposure to ensure capital adequacy. This directly influences the need for an interpretable and well-documented model in several ways:

1.  **Regulatory Compliance:** Basel II mandates that banks use robust and reliable risk measurement systems. Regulators require full transparency into how credit risk is assessed, meaning models cannot be black boxes. An interpretable model allows regulators to understand the underlying logic, variables, and assumptions, ensuring compliance with capital requirements and risk management guidelines.
2.  **Model Validation:** For internal ratings-based (IRB) approaches, banks must extensively validate their models. Interpretability facilitates this validation process, allowing model validators to scrutinize the model's logic, identify potential biases, and verify that it aligns with financial theory and business intuition. Without interpretability, validation becomes significantly more challenging, if not impossible.
3.  **Risk Management and Decision-Making:** Beyond compliance, an interpretable model helps credit officers and management understand _why_ a particular credit decision is made. This understanding is crucial for setting risk appetite, designing effective risk mitigation strategies, explaining decisions to customers, and continuously improving credit policies. Well-documented models ensure that the knowledge is retained and transferable within the organization, fostering consistency in application.
4.  **Auditability and Accountability:** In a highly regulated environment, models are subject to internal and external audits. An interpretable and well-documented model provides the necessary audit trail, demonstrating accountability and adherence to established risk management frameworks.

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In many real-world credit risk scenarios, a direct, universally agreed-upon "default" label might not be readily available, or the true default event might occur long after the initial assessment. Creating a proxy variable for default becomes necessary for the following reasons:

1.  **Data Availability:** Direct default flags can be rare, delayed, or inconsistently defined across different data sources or time periods. A proxy allows us to leverage available data to infer a "bad" outcome.
2.  **Timeliness of Prediction:** We often need to predict risk before a formal, legal default occurs. A proxy (e.g., severe delinquency like 90 days past due, bankruptcy filing, loan write-off) can serve as an early warning indicator.
3.  **Model Training Requirement:** Machine learning models require a target variable (label) for supervised learning. If a direct default label isn't present or suitable, a proxy provides the necessary target for the model to learn from.

However, relying on a proxy variable introduces significant business risks:

1.  **Proxy Mismatch Risk:** The most critical risk is that the proxy definition might not perfectly align with the true definition of "default" that the business or regulators care about. For example, 90 days past due might not always lead to a full write-off, or a loan that _does_ default might never reach 90 days past due if it's restructured early. This mismatch can lead to inaccurate risk assessments.
2.  **Underestimation/Overestimation of Risk:** An imperfect proxy could cause the model to systematically underestimate (e.g., missing true defaults that don't trigger the proxy) or overestimate (e.g., flagging accounts as "bad" that eventually recover) actual credit risk. This has direct implications for capital allocation, provisioning, and profitability.
3.  **Suboptimal Business Decisions:** If the model predicts based on a flawed proxy, credit decisions (e.g., loan approvals, interest rates, credit limits) may be suboptimal. This could lead to approving too many high-risk loans, rejecting too many good customers, or setting incorrect pricing.
4.  **Reputational Damage:** Making decisions based on an inaccurate proxy can harm customer relationships if loans are unfairly denied or if the bank is perceived as mismanaging risk.
5.  **Regulatory Scrutiny:** Regulators will meticulously examine the definition and justification of any proxy variable used for risk measurement, especially for capital requirement calculations. An ill-defined proxy can lead to regulatory penalties.

Therefore, meticulous justification, back-testing, and continuous monitoring of the chosen proxy's correlation with true default events are crucial.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

In a regulated financial context, the choice between simple, interpretable models (e.g., Logistic Regression with Weight of Evidence - WoE) and complex, high-performance models (e.g., Gradient Boosting) involves significant trade-offs:

**Simple, Interpretable Models (e.g., Logistic Regression with WoE):**

- **Pros:**
  - **Interpretability:** Easy to understand how each variable contributes to the prediction. WoE transformation makes relationships monotonic and more intuitive for business users. This is paramount for regulatory approval (Basel II/III), internal validation, and communicating risk decisions.
  - **Transparency and Explainability:** Allows clear justification for individual credit decisions and provides insights into underlying risk drivers.
  - **Regulatory Acceptance:** Generally preferred and more readily accepted by regulators due to their transparency and ease of audit.
  - **Stability:** Often more stable over time if the underlying relationships between variables and default remain consistent.
  - **Ease of Implementation & Monitoring:** Simpler to build, deploy, and monitor for model drift.
- **Cons:**
  - **Lower Predictive Performance:** May not capture complex, non-linear relationships in the data as effectively as more sophisticated models, potentially leading to lower accuracy and discrimination power (e.g., lower AUC).
  - **Feature Engineering Overhead:** Often requires significant manual feature engineering (like WoE transformation, binning) to improve performance and ensure linearity, which can be time-consuming.
  - **Assumptions:** Relies on assumptions about linearity and independence of variables.

**Complex, High-Performance Models (e.g., Gradient Boosting - XGBoost, LightGBM):**

- **Pros:**
  - **Superior Predictive Performance:** Often achieve higher accuracy, AUC, and better discrimination, especially with large and complex datasets, by capturing intricate non-linear relationships and interactions between features.
  - **Automated Feature Interaction:** Can automatically discover and leverage complex interactions between features, reducing the need for extensive manual feature engineering.
  - **Robustness to Missing Data/Outliers:** Some algorithms are inherently more robust to outliers and missing values.
- **Cons:**
  - **Lack of Interpretability (Black Box):** The primary drawback. It's difficult to understand _why_ a specific prediction was made or how individual features contribute to the final outcome. This is a major hurdle for regulatory approval and risk communication.
  - **Regulatory Scrutiny and Model Risk:** Regulators are highly cautious of "black-box" models due to the difficulty in validating their logic, assessing biases, and ensuring compliance. This can lead to higher capital requirements or outright rejection if interpretability cannot be sufficiently addressed (e.g., through SHAP, LIME).
  - **Higher Model Risk:** More prone to overfitting if not carefully tuned, and harder to diagnose the causes of poor performance or unexpected behavior.
  - **Complex Validation and Monitoring:** More challenging to validate and continuously monitor for drift or unexpected behavior due to their complexity. Requires advanced explainability techniques (SHAP, LIME, Partial Dependence Plots) which themselves need validation.
  - **Operational Overhead:** Can be computationally more intensive to train and deploy, though inference can be fast.

**Overall Trade-off:**

In a highly regulated environment like credit risk, there's a strong **trade-off between model performance and interpretability/explainability**. While complex models might offer superior predictive power, the imperative for regulatory compliance, transparency, and effective risk management often steers practitioners towards simpler, more interpretable models.

Hybrid approaches are sometimes employed, where complex models are used for internal analysis or early warning, but a simpler, more transparent model is the "model of record" for regulatory capital calculations and formal decision-making. Alternatively, significant effort is invested in developing and validating explainability techniques (like SHAP values) for complex models to make them acceptable to regulators.
