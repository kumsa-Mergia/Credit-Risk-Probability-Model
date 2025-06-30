import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import schema definitions
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, DataType

def main():
    data = pd.read_csv("../../data/raw/data.csv")
    
    columns_to_drop = [
        'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
        'CustomerId', 'ProviderId', 'ProductId', 'ChannelId', 
        'TransactionStartTime', 
        'FraudResult' 
    ]
    
    X = data.drop(columns=columns_to_drop, errors='ignore') 
    y = data['FraudResult']
    
    # Handle categorical features in X
    for col in X.select_dtypes(include=['object']).columns:
        X = pd.concat([X, pd.get_dummies(X[col], prefix=col, drop_first=True)], axis=1)
        X = X.drop(columns=[col])

    # Convert remaining numerical columns to numeric, coercing errors to NaN
    # and ensure they are float64.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').astype(np.float64) # Force float64

    X = X.fillna(0) # Impute NaNs

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier()
    }
    
    mlflow.set_experiment("Credit Risk Modeling")
    
    # Define the input schema explicitly
    # All features are assumed to be double (float64) to avoid the integer warning
    input_schema_cols = [ColSpec(DataType.double, col_name) for col_name in X_train.columns]
    input_schema = Schema(input_schema_cols)
    
    # The output is a single prediction (0 or 1), which can be double or long
    # We'll use double for consistency, as models might output probabilities.
    output_schema = Schema([ColSpec(DataType.double, "prediction")])
    
    # Combine into a model signature
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            roc_auc = roc_auc_score(y_test, preds)
            
            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            
            # Log model with explicit signature
            mlflow.sklearn.log_model(
                model, 
                name="model", 
                input_example=X_train.head(5), 
                signature=signature # Pass the explicit signature here
            ) 
            
            print(f"{name} - Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()