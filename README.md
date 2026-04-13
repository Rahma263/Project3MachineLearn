# 🎯 Customer Segments - ML Web Application

## Unsupervised Learning | Machine Learning Engineer Nanodegree

![Project Banner](flask_app/static/assets/banner.jpg)
_Interactive customer segmentation with Flask web interface_

---

## 📽️ Project Demo

<p align="center">
  ▶️ <a href="P3\video\APPFlask.mp4">Watch the GUI demo</a>
</p>

&lt;/div&gt;

---

## 📸 Web Application Interface

&lt;div align="center"&gt;

|     📊 Dashboard Overview      |       🔍 Sample Analysis       |         ⚡ ML Pipeline         |
| :----------------------------: | :----------------------------: | :----------------------------: |
| ![Page 1](p3/images/page1.PNG) | ![Page 2](p3/images/page2.PNG) | ![Page 3](p3/images/page3.PNG) |

**Dark-themed interactive dashboard for customer segmentation analysis**

&lt;/div&gt;

### Features

- ✅ **Real-time Dataset Statistics** - 440 customers across 6 product categories
- ✅ **Sample Customer Tracking** - Select and analyze specific customer indices
- ✅ **Feature Relevance Testing** - Decision Tree regressor to identify redundant features
- ✅ **Outlier Detection** - 1.5×IQR rule on log-transformed data
- ✅ **PCA Analysis** - Dimensionality reduction with explained variance visualization
- ✅ **Gaussian Mixture Clustering** - Soft clustering with segment prediction
- ✅ **Live Segment Prediction** - Input spending data to classify new customers

---

## 🔬 Machine Learning Analysis

### Data Exploration & Correlation

&lt;div align="center"&gt;

|                     Feature Correlation Matrix                      |
| :-----------------------------------------------------------------: |
|              ![Scatter Matrix](p3/images/output1.png)               |
| _Log-transformed data scatter matrix showing feature relationships_ |

&lt;/div&gt;

**Key Findings:**

- Strong correlation between `Grocery`, `Milk`, and `Detergents_Paper` (retail pattern)
- `Fresh` and `Frozen` show different purchasing patterns (food service pattern)
- Log transformation normalized the heavily right-skewed distributions

### Principal Component Analysis

&lt;div align="center"&gt;

|                     PCA Component Weights                     |            PCA Biplot             |
| :-----------------------------------------------------------: | :-------------------------------: |
|             ![PCA Weights](p3/images/output.png)              | ![Biplot](p3/images/output3.png)  |
| _Explained variance: Dim 1 (44.24%) + Dim 2 (27.66%) = 71.9%_ | _Feature projections on PC plane_ |

&lt;/div&gt;

**Component Interpretation:**

- **Dimension 1**: Retail vs. Food Service (Detergents_Paper vs. Fresh/Frozen)
- **Dimension 2**: Fresh food emphasis (Fresh positive, Frozen negative)
- First 2 components capture **71.9%** of total variance

### Clustering Results

&lt;div align="center"&gt;

|                     K-Means Clustering                     |            Ground Truth Comparison            |
| :--------------------------------------------------------: | :-------------------------------------------: |
|             ![Clusters](p3/images/output4.png)             |   ![Channel Labels](p3/images/output5.png)    |
| _2 clusters identified (k=2 optimal via silhouette score)_ | _Red: Hotel/Restaurant/Cafe, Green: Retailer_ |

&lt;/div&gt;

**Clustering Performance:**

- **Algorithm**: K-Means (k=2, silhouette score: 0.42)
- **Segment 0** (Red): High Grocery/Milk/Detergents_Paper → **Retailers**
- **Segment 1** (Magenta): High Fresh/Frozen/Delicatessen → **Restaurants/Hotels/Cafes**
- Alignment with actual Channel labels confirms validity

---

## 🚀 Quick Start

### Prerequisites

| Requirement  | Version | Link                                       |
| ------------ | ------- | ------------------------------------------ |
| Python       | 2.7     | [Download](http://continuum.io/downloads)  |
| NumPy        | Latest  | [Docs](http://www.numpy.org/)              |
| Pandas       | Latest  | [Docs](http://pandas.pydata.org)           |
| Matplotlib   | Latest  | [Docs](http://matplotlib.org/)             |
| scikit-learn | Latest  | [Docs](http://scikit-learn.org/stable/)    |
| Flask        | Latest  | [Docs](https://flask.palletsprojects.com/) |
| Jupyter      | Latest  | [Docs](http://jupyter.org/index.html)      |

&gt; 💡 **Pro Tip**: Install [Anaconda](http://continuum.io/downloads) (Python 2.7) for easy setup!

### Installation

```bash
# Clone the repository
git clone https://github.com/Rahma263/Project3MachineLearn.git
cd Project3MachineLearn/P3/

# Install dependencies
pip install numpy pandas matplotlib scikit-learn flask

# Or if you have a requirements.txt
pip install -r requirements.txt

Running the Application
Option 1: Flask Web App
# Navigate to the flask_app directory
cd flask_app/

# Run the Flask application
python app.py

# Open browser at http://localhost:5000
Option 2: Jupyter Notebook
# From the P3 directory
jupyter notebook customer_segments.ipynb
# or
ipython notebook customer_segments.ipynb

📁 Project Structure

Project3MachineLearn/
└── P3/                              # Main project directory
    ├── 📄 README.md                 # This file
    ├── 📓 customer_segments.ipynb     # Jupyter notebook analysis
    ├── 📊 customers.csv              # Dataset (440 samples)
    ├── 🐍 visuals.py                # Visualization utilities
    ├── 🖼️ output.png                # PCA component weights
    ├── 🖼️ output1.png               # Scatter matrix
    ├── 🖼️ output3.png               # PCA biplot
    ├── 🖼️ output4.png               # Clustering results
    ├── 🖼️ output5.png               # Channel comparison
    ├── 🖼️ page1.PNG                 # Web app dashboard
    ├── 🖼️ page2.PNG                 # Web app samples
    ├── 🖼️ page3.PNG                 # Web app ML pipeline
    └── 🌐 flask_app/                # Flask web application
        ├── 🐍 app.py                # Flask main application
        ├── 🐍 ml_engine.py          # ML prediction engine
        ├── 🎨 static/               # Static assets
        │   ├── 📝 style.css         # Application styling
        │   └── 🖼️ assets/           # Additional media
        └── 📝 templates/            # HTML templates
            └── index.html           # Main web interface

📊 Dataset Overview
Source
    * UCI Machine Learning Repository - Wholesale Customers Dataset

    * Location: Lisbon, Portugal
    * Data Points: 440 wholesale clients
    * Unit: Monetary Units (m.u.)

Features

| Feature            | Mean   | Description                              |
| ------------------ | ------ | ---------------------------------------- |
| `Fresh`            | 12,000 | Annual spending on fresh products        |
| `Milk`             | 5,796  | Annual spending on milk products         |
| `Grocery`          | 7,951  | Annual spending on grocery products      |
| `Frozen`           | 3,072  | Annual spending on frozen products       |
| `Detergents_Paper` | 2,881  | Annual spending on detergents and paper  |
| `Delicatessen`     | 1,525  | Annual spending on delicatessen products |
| `Channel`          | -      | Hotel/Restaurant/Cafe (1) or Retail (2)  |
| `Region`           | -      | Lisbon (1), Oporto (2), or Other (3)     |

⚠️ Note: Channel and Region excluded from clustering to focus on purchasing behavior.

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Input    │────▶│   Flask App     │────▶│   ML Engine     │
│  (Web Form)     │     │   (app.py)      │     │  (ml_engine.py) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                           │
┌─────────────────┐       ┌─────────────────┐              │
│  Display Results  │◀────│  Pre-trained    │◀─────────────┘
│  (Segment +       │     │  Model (K-Means)│
│   Visualizations) │     │                 │
└─────────────────┘       └─────────────────┘

Key Components

| Component             | File                         | Purpose                          |
| --------------------- | ---------------------------- | -------------------------------- |
| **Web Interface**     | `flask_app/app.py`           | Routes, form handling, rendering |
| **ML Engine**         | `flask_app/ml_engine.py`     | Model loading, PCA, prediction   |
| **Analysis Notebook** | `customer_segments.ipynb`    | Complete data science workflow   |
| **Visualizations**    | `visuals.py`                 | Custom plotting functions        |
| **Styling**           | `flask_app/static/style.css` | Dark theme UI design             |

🎓 Methodology
1. Data Preprocessing

# Log transformation for skewed data
log_data = np.log(data)

# Outlier removal using Tukey's method (1.5×IQR)
# Removed: indices 65, 66, 75, 128, 154 (outliers in ≥2 features)
2. Feature Engineering

    * Correlation Analysis: Identified Grocery as partially redundant (R² = 0.68)
    * PCA: Reduced 6 features → 2 components (71.9% variance retained)

3. Clustering

# K-Means with optimal k=2 (silhouette score: 0.42)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(reduced_data)

4. Web Deployment

    * Flask backend with real-time prediction API
    * Interactive frontend for data exploration

🎯 Customer Segments

| Segment | Characteristics                        | Business Type                    | Delivery Strategy        |
| ------- | -------------------------------------- | -------------------------------- | ------------------------ |
| **0**   | High: Milk, Grocery, Detergents\_Paper | 🏪 **Retailers/Supermarkets**    | 3 days/week (acceptable) |
| **1**   | High: Fresh, Frozen, Delicatessen      | 🍽️ **Restaurants/Hotels/Cafes** | 5 days/week (critical)   |

Business Impact

    * A/B Testing: Different delivery schedules per segment
    * Resource Allocation: Optimize logistics based on segment needs
    * Customer Acquisition: Predict segment for new customers

🛠️ Technologies Stack
<p align="left">
  <img src="https://img.shields.io/badge/Python-2.7-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-Web%20Framework-lightgrey?logo=flask" alt="Flask"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-red?logo=scikit-learn" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-Computing-orange?logo=numpy" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Matplotlib-Viz-blue?logo=matplotlib" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/HTML5-Frontend-orange?logo=html5" alt="HTML5"/>
  <img src="https://img.shields.io/badge/CSS3-Styling-blue?logo=css3" alt="CSS3"/>
</p>

🏆 Results Summary

| Metric               | Value                              |
| -------------------- | ---------------------------------- |
| **Dataset Size**     | 440 customers                      |
| **Features**         | 6 product categories               |
| **Optimal Clusters** | 2 (silhouette: 0.42)               |
| **PCA Variance**     | 71.9% (2 components)               |
| **Outliers Removed** | 5 data points                      |
| **Segment Accuracy** | High alignment with Channel labels |


Instructions to Use:

   1. Save this README as README.md in your P3/ folder (same level as customer_segments.ipynb)
   2. Your images are already referenced correctly:
        page1.PNG, page2.PNG, page3.PNG - Web app screenshots
        output.png - PCA weights
        output1.png - Scatter matrix
        output3.png - Biplot
        output4.png - Clustering results
        output5.png - Channel comparison
    3. For videos: Add them to flask_app/static/assets/ and update the video links in the README
    4. Optional banner: Add flask_app/static/assets/banner.jpg for the header image
```
