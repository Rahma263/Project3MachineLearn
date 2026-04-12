"""
app.py — Flask application for Customer Segments Dashboard
"""

import os
import sys
from flask import Flask, render_template, request, jsonify

# Ensure the flask_app package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_engine as ml

app = Flask(__name__)


# -----------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------------------------------------------------
# API — Dataset Statistics
# -----------------------------------------------------------------------
@app.route("/api/statistics", methods=["POST"])
def api_statistics():
    data, _ = ml.load_data()
    return jsonify(ml.get_statistics(data))


# -----------------------------------------------------------------------
# API — Sample Customers
# -----------------------------------------------------------------------
@app.route("/api/samples", methods=["POST"])
def api_samples():
    body = request.get_json(force=True)
    indices = body.get("indices", [0, 100, 300])
    data, _ = ml.load_data()
    return jsonify(ml.get_samples(data, indices))


# -----------------------------------------------------------------------
# API — Feature Relevance
# -----------------------------------------------------------------------
@app.route("/api/feature_relevance", methods=["POST"])
def api_feature_relevance():
    body = request.get_json(force=True)
    feature = body.get("feature", "Grocery")
    data, _ = ml.load_data()
    return jsonify(ml.feature_relevance(data, feature))


# -----------------------------------------------------------------------
# API — Outlier Detection
# -----------------------------------------------------------------------
@app.route("/api/outliers", methods=["POST"])
def api_outliers():
    data, _ = ml.load_data()
    log_data = ml.log_transform(data)
    return jsonify(ml.detect_outliers(log_data))


# -----------------------------------------------------------------------
# API — PCA Analysis
# -----------------------------------------------------------------------
@app.route("/api/pca", methods=["POST"])
def api_pca():
    data, _ = ml.load_data()
    log_data = ml.log_transform(data)
    outlier_info = ml.detect_outliers(log_data)
    good_data = ml.remove_outliers(log_data, outlier_info["multi_outliers"])
    _, _, _, results = ml.apply_pca(good_data)
    return jsonify(results)


# -----------------------------------------------------------------------
# API — Clustering
# -----------------------------------------------------------------------
@app.route("/api/cluster", methods=["POST"])
def api_cluster():
    body = request.get_json(force=True)
    n_clusters = body.get("n_clusters", 2)

    data, full_data = ml.load_data()
    log_data = ml.log_transform(data)
    outlier_info = ml.detect_outliers(log_data)
    good_data = ml.remove_outliers(log_data, outlier_info["multi_outliers"])

    _, _, reduced_df, _ = ml.apply_pca(good_data)
    cluster_res = ml.cluster_data(reduced_df, n_clusters)
    channel_res = ml.channel_comparison(full_data, outlier_info["multi_outliers"], reduced_df)

    return jsonify({"clusters": cluster_res, "channels": channel_res})


# -----------------------------------------------------------------------
# API — Predict Segment
# -----------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True)
    values = body.get("values", {})
    n_clusters = body.get("n_clusters", 2)
    return jsonify(ml.predict_segment(values, n_clusters))


# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
