import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class RiskLabelEngineer:
    def __init__(self, transactions_df: pd.DataFrame, main_df: pd.DataFrame):
        self.transactions_df = transactions_df.copy()
        self.main_df = main_df.copy()
        self.rfm_df = None
        self.high_risk_cluster = None

    def compute_rfm(self):
        # Ensure datetime format
        self.transactions_df["TransactionStartTime"] = pd.to_datetime(
            self.transactions_df["TransactionStartTime"]
        )

        # Define snapshot date (1 day after last transaction)
        snapshot_date = self.transactions_df[
            "TransactionStartTime"
        ].max() + pd.Timedelta(days=1)

        rfm = (
            self.transactions_df.groupby("CustomerId")
            .agg(
                {
                    "TransactionStartTime": lambda x: (
                        snapshot_date - x.max()
                    ).days,  # Recency
                    "TransactionId": "count",  # Frequency
                    "Amount": "sum",  # Monetary
                }
            )
            .rename(
                columns={
                    "TransactionStartTime": "Recency",
                    "TransactionId": "Frequency",
                    "Amount": "Monetary",
                }
            )
            .reset_index()
        )

        self.rfm_df = rfm

    def cluster_customers(self, n_clusters=3, random_state=42):
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(
            self.rfm_df[["Recency", "Frequency", "Monetary"]]
        )

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.rfm_df["Cluster"] = kmeans.fit_predict(rfm_scaled)

        # Identify the high-risk cluster: highest Recency, lowest Frequency & Monetary
        cluster_summary = self.rfm_df.groupby("Cluster")[
            ["Recency", "Frequency", "Monetary"]
        ].mean()
        self.high_risk_cluster = cluster_summary.sort_values(
            by=["Frequency", "Monetary", "Recency"], ascending=[True, True, False]
        ).index[0]

        self.rfm_df["is_high_risk"] = (
            self.rfm_df["Cluster"] == self.high_risk_cluster
        ).astype(int)

    def merge_target(self):
        # Merge is_high_risk into main_df
        self.main_df = self.main_df.merge(
            self.rfm_df[["CustomerId", "is_high_risk"]], on="CustomerId", how="left"
        ).fillna(
            {"is_high_risk": 0}
        )  # Default to 0 if missing

    def run(self):
        self.compute_rfm()
        self.cluster_customers()
        self.merge_target()
        return self.main_df
