import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# --- 1. Custom Transformers ---


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if self.datetime_col not in X_transformed.columns:
            raise ValueError(f"Datetime column '{self.datetime_col}' not found.")

        X_transformed[self.datetime_col] = pd.to_datetime(
            X_transformed[self.datetime_col], errors="coerce"
        )
        original_rows = X_transformed.shape[0]
        X_transformed.dropna(subset=[self.datetime_col], inplace=True)

        if X_transformed.shape[0] < original_rows:
            print(
                f"Warning: Dropped {original_rows - X_transformed.shape[0]} "
                f"rows due to invalid datetime."
            )

        X_transformed["transaction_hour"] = X_transformed[self.datetime_col].dt.hour
        X_transformed["transaction_day_of_week"] = X_transformed[
            self.datetime_col
        ].dt.dayofweek
        X_transformed["transaction_day_of_month"] = X_transformed[
            self.datetime_col
        ].dt.day
        X_transformed["transaction_month"] = X_transformed[self.datetime_col].dt.month
        X_transformed["transaction_year"] = X_transformed[self.datetime_col].dt.year

        return X_transformed.drop(columns=[self.datetime_col])


class CustomWOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, overall_woe_value=0.0):
        self.variables = variables
        self.overall_woe_value = overall_woe_value
        self.woe_maps = {}
        self.overall_target_log_odds = None

    def fit(self, X, y):
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)

        if not y.isin([0, 1]).all():
            raise ValueError("Target must be binary (0/1).")

        epsilon = 1e-6
        good = (y == 0).sum()
        bad = (y == 1).sum()
        self.overall_target_log_odds = np.log((good + epsilon) / (bad + epsilon))

        if self.variables is None:
            self.variables = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        else:
            self.variables = [col for col in self.variables if col in X.columns]

        for col in self.variables:
            temp_df = pd.DataFrame({"feature": X[col], "target": y})
            woe_table = (
                temp_df.groupby("feature")["target"]
                .agg(
                    total_count="count",
                    good_count=lambda x: (x == 0).sum(),
                    bad_count=lambda x: (x == 1).sum(),
                )
                .reset_index()
            )

            woe_table["good_rate"] = (woe_table["good_count"] + epsilon) / (
                good + epsilon
            )
            woe_table["bad_rate"] = (woe_table["bad_count"] + epsilon) / (bad + epsilon)
            woe_table["woe"] = np.log(woe_table["good_rate"] / woe_table["bad_rate"])
            self.woe_maps[col] = woe_table.set_index("feature")["woe"].to_dict()

        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            if self.variables is None:
                raise ValueError(
                    "Cannot convert ndarray to DataFrame without column names."
                )
            X = pd.DataFrame(X, columns=self.variables)
        X_transformed = X.copy()

        for col in self.variables:
            if col not in X_transformed.columns:
                continue

            na_placeholder = "__WOE_MISSING__"
            X_transformed[col] = X_transformed[col].fillna(na_placeholder)
            X_transformed[col] = X_transformed[col].apply(
                lambda x: self.woe_maps[col].get(x, self.overall_target_log_odds)
            )

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.variables
        return np.array(input_features)


# --- 2. Aggregation Function ---


def create_aggregated_features(
    df, id_col="AccountId", aggregate_cols=["Amount", "Value"]
):
    if id_col not in df.columns:
        raise ValueError(f"'{id_col}' not in DataFrame.")

    valid_cols = [
        col
        for col in aggregate_cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not valid_cols:
        return df

    agg_dict = {
        "Amount": ["sum", "mean", "std", "count"],
        "Value": ["sum", "mean", "std"],
    }
    agg_dict = {k: v for k, v in agg_dict.items() if k in valid_cols}

    agg_df = df.groupby(id_col).agg(agg_dict)
    agg_df.columns = [f"{col}_{func}_{id_col.lower()}" for col, func in agg_df.columns]
    agg_df = agg_df.reset_index()

    if "Amount" in valid_cols:
        agg_df.rename(
            columns={
                f"Amount_count_{id_col.lower()}": f"transaction_count_{id_col.lower()}"
            },
            inplace=True,
        )

    df = pd.merge(df, agg_df, on=id_col, how="left")
    for col in [f"amount_std_{id_col.lower()}", f"value_std_{id_col.lower()}"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# --- 3. Preprocessing Pipeline ---


def get_preprocessing_pipeline(
    numerical_features, categorical_ohe_features, categorical_woe_features
):
    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    ohe_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    woe_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "to_df",
                FunctionTransformer(
                    lambda X: pd.DataFrame(X, columns=categorical_woe_features),
                    validate=False,
                ),
            ),
            ("woe_encoder", CustomWOEEncoder(variables=categorical_woe_features)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numerical_features),
            ("cat_ohe", ohe_pipe, categorical_ohe_features),
            ("cat_woe", woe_pipe, categorical_woe_features),
        ],
        remainder="passthrough",
    )

    return preprocessor


def run_preprocessing(df_raw, target_column="FraudResult"):
    print("Starting preprocessing...")
    y = df_raw[target_column]
    X = df_raw.drop(columns=[target_column])

    datetime_extractor = DateTimeFeatureExtractor()
    X = datetime_extractor.fit_transform(X)
    y = y.loc[X.index]

    X = create_aggregated_features(
        X, id_col="AccountId", aggregate_cols=["Amount", "Value"]
    )
    y = y.loc[X.index]

    categorical_ohe_features = ["CurrencyCode"]
    categorical_woe_features = ["ProductCategory", "ChannelId"]
    numerical_features = [
        "Amount",
        "Value",
        "CountryCode",
        "PricingStrategy",
        "transaction_hour",
        "transaction_day_of_week",
        "transaction_day_of_month",
        "transaction_month",
        "transaction_year",
        "amount_sum_accountid",
        "amount_mean_accountid",
        "amount_std_accountid",
        "value_sum_accountid",
        "value_mean_accountid",
        "value_std_accountid",
        "transaction_count_accountid",
    ]

    all_columns = X.columns.tolist()
    numerical_features = [col for col in numerical_features if col in all_columns]
    categorical_ohe_features = [
        col for col in categorical_ohe_features if col in all_columns
    ]
    categorical_woe_features = [
        col for col in categorical_woe_features if col in all_columns
    ]

    drop_cols = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CustomerId",
        "ProviderId",
        "ProductId",
    ]
    drop_cols = [col for col in drop_cols if col in X.columns]
    X = X.drop(columns=drop_cols)

    pipeline = get_preprocessing_pipeline(
        numerical_features, categorical_ohe_features, categorical_woe_features
    )
    X_array = pipeline.fit_transform(X, y)

    try:
        feature_names = pipeline.get_feature_names_out()
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

    X_final = pd.DataFrame(X_array, columns=feature_names, index=y.index)

    print("Preprocessing complete.")
    return X_final, y


# --- 4. Example Usage ---

if __name__ == "__main__":
    print("Testing preprocessing pipeline...")

    n_rows = 1000
    df_raw_test = pd.DataFrame(
        {
            "TransactionId": range(n_rows),
            "BatchId": np.random.randint(1, 50, n_rows),
            "AccountId": np.random.randint(1000, 5000, n_rows),
            "SubscriptionId": np.random.randint(100, 300, n_rows),
            "CustomerId": np.random.randint(1000, 5000, n_rows),
            "CurrencyCode": np.random.choice(["KES", "USD", "EUR"], n_rows),
            "CountryCode": np.random.randint(254, 300, n_rows),
            "ProviderId": np.random.randint(1, 10, n_rows),
            "ProductId": np.random.randint(10000, 10010, n_rows),
            "ProductCategory": np.random.choice(
                ["Bills", "Airtime", "Data", "Other"], n_rows
            ),
            "ChannelId": np.random.choice(
                ["Web", "Android", "IOS", "Pay Later"], n_rows
            ),
            "Amount": np.random.uniform(-10000, 50000, n_rows),
            "Value": np.random.uniform(0, 50000, n_rows),
            "TransactionStartTime": pd.to_datetime("2024-01-01")
            + pd.to_timedelta(
                np.random.randint(0, 365 * 24 * 60 * 60, n_rows), unit="s"
            ),
            "PricingStrategy": np.random.choice([1, 2, 3, 4, 5], n_rows),
            "FraudResult": np.random.choice([0, 1], n_rows, p=[0.95, 0.05]),
        }
    )

    X_processed, y_processed = run_preprocessing(df_raw_test)

    print("\nProcessed Features:")
    print(X_processed.head())

    print("\nTarget:")
    print(y_processed.head())
