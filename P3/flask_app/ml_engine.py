"""
ml_engine.py — Machine Learning Engine for Customer Segments GUI
Extracts and wraps all ML logic from the customer_segments.ipynb notebook.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_PATH = os.path.join(DATA_DIR, "customers.csv")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load the wholesale customers dataset and return (data, full_data).
    *data* has Region/Channel dropped; *full_data* keeps them.
    """
    full_data = pd.read_csv(CSV_PATH)
    data = full_data.drop(["Region", "Channel"], axis=1)
    return data, full_data


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def get_statistics(data: pd.DataFrame) -> dict:
    """Return descriptive statistics as a JSON-friendly dict."""
    desc = data.describe().round(2)
    return {
        "columns": list(desc.columns),
        "stats": {idx: row.to_dict() for idx, row in desc.iterrows()},
    }


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------
def get_samples(data: pd.DataFrame, indices: list) -> dict:
    """Return the rows at *indices* as a dict."""
    samples = data.loc[indices].reset_index(drop=True)
    return {
        "columns": list(samples.columns),
        "rows": samples.values.tolist(),
        "indices": indices,
    }


# ---------------------------------------------------------------------------
# Feature relevance
# ---------------------------------------------------------------------------
def feature_relevance(data: pd.DataFrame, feature: str, random_state: int = 42) -> dict:
    """Train a DecisionTreeRegressor to predict *feature* from the rest.
    Returns the R² score and a textual interpretation.
    """
    new_data = data.drop(columns=[feature])
    target = data[feature]
    X_train, X_test, y_train, y_test = train_test_split(
        new_data, target, test_size=0.25, random_state=random_state
    )
    reg = DecisionTreeRegressor(random_state=random_state)
    reg.fit(X_train, y_train)
    score = round(r2_score(y_test, reg.predict(X_test)), 4)

    if score >= 0.7:
        interpretation = (
            f"R² = {score}. High score — '{feature}' can be well predicted from the other "
            "features. This suggests it is somewhat redundant and might not be necessary."
        )
    elif score >= 0.4:
        interpretation = (
            f"R² = {score}. Moderate score — '{feature}' is partially predictable, "
            "meaning it shares some information with other features."
        )
    else:
        interpretation = (
            f"R² = {score}. Low score — '{feature}' is hard to predict from others, "
            "indicating it carries unique and important information."
        )

    return {"feature": feature, "r2_score": score, "interpretation": interpretation}


# ---------------------------------------------------------------------------
# Log transform
# ---------------------------------------------------------------------------
def log_transform(data: pd.DataFrame) -> pd.DataFrame:
    """Apply a natural-log transform to all features."""
    return np.log(data + 1)


# ---------------------------------------------------------------------------
# Outlier detection (IQR)
# ---------------------------------------------------------------------------
def detect_outliers(log_data: pd.DataFrame) -> dict:
    """Find outliers via the 1.5×IQR rule for every feature.
    Returns per-feature outlier indices and the set of indices that appear
    in more than one feature ("multi-outliers").
    """
    per_feature: dict = {}
    all_outlier_indices: list = []

    for feature in log_data.columns:
        q1 = np.percentile(log_data[feature], 25)
        q3 = np.percentile(log_data[feature], 75)
        step = 1.5 * (q3 - q1)
        mask = ~((log_data[feature] >= q1 - step) & (log_data[feature] <= q3 + step))
        outlier_idx = list(log_data.index[mask])
        per_feature[feature] = outlier_idx
        all_outlier_indices.extend(outlier_idx)

    # Indices appearing in 2+ features
    from collections import Counter
    counts = Counter(all_outlier_indices)
    multi = sorted([idx for idx, cnt in counts.items() if cnt >= 2])

    return {
        "per_feature": per_feature,
        "multi_outliers": multi,
        "total_removed": len(multi),
    }


def remove_outliers(log_data: pd.DataFrame, outlier_indices: list) -> pd.DataFrame:
    """Drop outlier rows and reset the index."""
    return log_data.drop(log_data.index[outlier_indices]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def apply_pca(good_data: pd.DataFrame, n_components: int = 6):
    """Fit PCA and return (pca_object, reduced_data_df, results_dict)."""
    pca = PCA(n_components=n_components)
    pca.fit(good_data)

    dims = [f"Dimension {i+1}" for i in range(n_components)]
    components_df = pd.DataFrame(
        np.round(pca.components_, 4), columns=list(good_data.columns)
    )
    components_df.index = dims
    ratios = np.round(pca.explained_variance_ratio_, 4)

    results = {
        "dimensions": dims,
        "explained_variance": ratios.tolist(),
        "components": {
            dim: components_df.loc[dim].to_dict() for dim in dims
        },
        "features": list(good_data.columns),
    }

    # Also produce 2-component reduced data for visualisation
    pca2 = PCA(n_components=2)
    reduced = pca2.fit_transform(good_data)
    reduced_df = pd.DataFrame(reduced, columns=["Dimension 1", "Dimension 2"])

    return pca, pca2, reduced_df, results


# ---------------------------------------------------------------------------
# Clustering (Gaussian Mixture)
# ---------------------------------------------------------------------------
def cluster_data(reduced_df: pd.DataFrame, n_clusters: int = 2):
    """Fit a GaussianMixture model and return predictions + centres."""
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(reduced_df)
    preds = gmm.predict(reduced_df)
    centres = gmm.means_

    return {
        "predictions": preds.tolist(),
        "centres": centres.tolist(),
        "n_clusters": n_clusters,
        "dim1": reduced_df["Dimension 1"].tolist(),
        "dim2": reduced_df["Dimension 2"].tolist(),
    }


# ---------------------------------------------------------------------------
# Channel comparison
# ---------------------------------------------------------------------------
def channel_comparison(full_data: pd.DataFrame, outlier_indices: list,
                       reduced_df: pd.DataFrame) -> dict:
    """Compare clusters with the actual Channel labels."""
    channel = full_data["Channel"].copy()
    channel = channel.drop(channel.index[outlier_indices]).reset_index(drop=True)
    return {
        "channels": channel.tolist(),
        "dim1": reduced_df["Dimension 1"].tolist(),
        "dim2": reduced_df["Dimension 2"].tolist(),
    }


# ---------------------------------------------------------------------------
# Full pipeline (convenience)
# ---------------------------------------------------------------------------
def run_full_pipeline(n_clusters: int = 2, sample_indices=None):
    """Execute the entire analysis pipeline and return all results."""
    data, full_data = load_data()
    stats = get_statistics(data)

    if sample_indices is None:
        sample_indices = [0, 100, 300]
    samples = get_samples(data, sample_indices)

    log_data = log_transform(data)
    outlier_info = detect_outliers(log_data)
    good_data = remove_outliers(log_data, outlier_info["multi_outliers"])

    pca, pca2, reduced_df, pca_results = apply_pca(good_data)
    cluster_res = cluster_data(reduced_df, n_clusters)
    channel_res = channel_comparison(full_data, outlier_info["multi_outliers"], reduced_df)

    return {
        "stats": stats,
        "samples": samples,
        "outliers": outlier_info,
        "pca": pca_results,
        "clusters": cluster_res,
        "channels": channel_res,
        "n_samples": len(data),
        "n_features": len(data.columns),
        "features": list(data.columns),
    }


# ---------------------------------------------------------------------------
# Predict segment for new customer data
# ---------------------------------------------------------------------------
def predict_segment(values: dict, n_clusters: int = 2):
    """Given a dict of {feature: spending_value}, predict the cluster."""
    data, full_data = load_data()
    log_data = log_transform(data)
    outlier_info = detect_outliers(log_data)
    good_data = remove_outliers(log_data, outlier_info["multi_outliers"])

    pca2 = PCA(n_components=2)
    reduced = pca2.fit_transform(good_data)
    reduced_df = pd.DataFrame(reduced, columns=["Dimension 1", "Dimension 2"])

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(reduced_df)

    # Transform the new point
    new_point = pd.DataFrame([values])
    new_log = np.log(new_point + 1)
    new_reduced = pca2.transform(new_log)
    prediction = int(gmm.predict(new_reduced)[0])

    return {
        "cluster": prediction,
        "reduced_point": new_reduced[0].tolist(),
    }
