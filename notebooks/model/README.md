# 📊 Model Training Results

This report summarizes the performance of two classification models — **Logistic Regression** and **Random Forest** — trained for **credit risk assessment** using customer transaction data. The models were evaluated on standard classification metrics and logged using **MLflow** for reproducibility and experiment tracking.

---

## 🔍 Performance Summary

| Model               | Accuracy | F1 Score | ROC-AUC |
| ------------------- | -------- | -------- | ------- |
| Logistic Regression | 0.9983   | 0.2727   | 0.5833  |
| Random Forest       | 0.9998   | 0.9565   | 0.9583  |

> ⚠️ **Note:** The dataset is highly imbalanced (few fraudulent cases), so **F1 Score** and **ROC-AUC** are more reliable than accuracy for model comparison.

---

## 📈 Interpretation of Results

### 🧠 Logistic Regression

- **Accuracy (0.9983):** Appears high, but this is misleading due to class imbalance.
- **F1 Score (0.2727):** Very low, indicating poor balance between precision and recall on the minority (fraud) class.
- **ROC-AUC (0.5833):** Barely above random guessing (0.5), suggesting weak discrimination between fraud and non-fraud.

> ⚠️ Despite its simplicity, Logistic Regression fails to capture meaningful patterns in the data related to fraud detection.

---

### 🌲 Random Forest

- **Accuracy (0.9998):** Extremely high, though not the best metric in this context.
- **F1 Score (0.9565):** Excellent performance in detecting fraud while minimizing false alarms.
- **ROC-AUC (0.9583):** Outstanding ability to distinguish between fraudulent and legitimate transactions.

> ✅ Random Forest handles non-linear relationships and variable interactions well, making it a strong fit for this task.

---

## 🧾 Conclusion

Given the nature of the problem and class imbalance:

> 🎯 **Random Forest significantly outperforms Logistic Regression** in detecting credit risk and fraudulent transactions.

Its superior F1 and ROC-AUC scores make it the best candidate for production deployment or further hyperparameter optimization.

---

## 🧪 MLflow Tracking

All experiments were tracked and logged using **MLflow**, including:

- Model parameters and versions
- Evaluation metrics
- Input schema and example data
- Registered models for comparison and reproducibility

To inspect experiments:

```bash
mlflow ui
```
