# 🌸 Iris Flower Classification — Machine Learning Project

> Classify Iris flowers into three species — **Setosa, Versicolor, and Virginica** — using petal and sepal measurements, by training and comparing five machine learning classification models.

---

## 📌 Table of Contents

- [`Project Overview`](#project-overview)
- [`Dataset Description`](#dataset-description)
- [`Project Workflow`](#project-workflow)
- [`Exploratory Data Analysis`](#exploratory-data-analysis)
- [`Data Preprocessing`](#data-preprocessing)
- [`Model Building`](#model-building)
- [`Model Evaluation Metrics`](#model-evaluation-metrics)
- [`Results & Comparison`](#results--comparison)
- [`Final Conclusion`](#final-conclusion)
- [`Technologies Used`](#technologies-used)
- [`How to Run`](#how-to-run)

---

## 📖 Project Overview

This project demonstrates a complete **multi-class classification pipeline** on the classic Iris dataset. Five machine learning models are trained, evaluated with confusion matrices and classification reports, and compared against each other.

| Model | Type |
|---|---|
| Logistic Regression | Linear, probabilistic classifier |
| Decision Tree Classifier | Rule-based, tree structure |
| Random Forest Classifier | Ensemble of decision trees |
| Support Vector Classifier (SVC) | Margin-based, kernel methods |
| K-Nearest Neighbors (KNN) | Distance-based, instance learning |

---

## 📂 Dataset Description

- **Source:** Kaggle — [`saurabh00007/iriscsv`](https://www.kaggle.com/datasets/saurabh00007/iriscsv) (downloaded via `kagglehub`)
- **File:** `Iris.csv`
- **Shape:** 150 rows × 6 columns
- **Target Variable:** `Species` — 3 classes (balanced)
- **Missing Values:**  None
- **Duplicates:**  None

### Columns

| Column | Description | Type |
|---|---|---|
| `Id` | Row identifier (dropped before modeling) | Integer |
| `SepalLengthCm` | Sepal length in centimeters | Float |
| `SepalWidthCm` | Sepal width in centimeters | Float |
| `PetalLengthCm` | Petal length in centimeters | Float |
| `PetalWidthCm` | Petal width in centimeters | Float |
| `Species` | Flower species — target variable | Categorical → Encoded |

### Target Class Encoding

| Original Label | Encoded Value |
|---|---|
| Iris-setosa | 0 |
| Iris-versicolor | 1 |
| Iris-virginica | 2 |

### Class Balance

| Class | Count |
|---|---|
| Setosa (0) | 50 |
| Versicolor (1) | 50 |
| Virginica (2) | 50 |

> ✅ **Perfectly balanced dataset** — no oversampling or undersampling required.

---

## 🔄 Project Workflow

```
Download Dataset (Kaggle via kagglehub)
     ↓
Load Dataset (pd.read_csv)
     ↓
Exploratory Data Analysis (EDA)
  → Shape, info, describe
  → Null & duplicate check
  → Correlation heatmap
     ↓
Data Preprocessing
  → Label Encoding (Species: text → 0, 1, 2)
  → Drop Id column (not useful)
  → Check class balance
     ↓
Define Features
  → X = [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]
  → y = Species
     ↓
Train / Test Split (80% / 20%, random_state=42)
     ↓
Train 5 Classification Models
     ↓
Evaluate Each Model
  → Accuracy Score
  → Confusion Matrix
  → Classification Report (Precision, Recall, F1-Score)
     ↓
Compare All Models & Select Best
```

---

## 📊 Exploratory Data Analysis

### Statistical Summary

| Feature | Min | Mean | Max | Std |
|---|---|---|---|---|
| SepalLengthCm | 4.30 | 5.84 | 7.90 | 0.83 |
| SepalWidthCm | 2.00 | 3.05 | 4.40 | 0.43 |
| PetalLengthCm | 1.00 | 3.76 | 6.90 | 1.76 |
| PetalWidthCm | 0.10 | 1.20 | 2.50 | 0.76 |

### Correlation Heatmap Insights

A correlation heatmap was plotted for all numeric features:

- **PetalLengthCm ↔ PetalWidthCm:** Very strong positive correlation **(0.96)** — larger petals are both longer and wider.
- **SepalLengthCm ↔ PetalLengthCm / PetalWidthCm:** Strong positive correlation — sepal length grows with petal size.
- **SepalWidthCm:** Weak or negative correlation with all other features.

> 💡 **Key Insight:** Petal features are far more discriminative than sepal features for classifying Iris species.

---

## ⚙️ Data Preprocessing

### 1. Label Encoding
```python
le = LabelEncoder()
for cols in iris_data.columns:
    if iris_data[cols].dtypes == 'object':
        iris_data[cols] = le.fit_transform(iris_data[cols])
```
Converts `Species` string labels to integers: `0, 1, 2`.

### 2. Feature / Target Split
```python
x = iris_data.drop(['Id', 'Species'], axis=1)
y = iris_data['Species']
```
- `Id` is dropped — it's just a row index with no predictive value.
- `Species` is the target.

### 3. Train/Test Split
```python
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=42)
```

| Split | Shape |
|---|---|
| Training set | (120, 4) |
| Test set | (30, 4) |

---

## 🧠 Model Building

### 1. Logistic Regression
```python
LogisticRegression()
```
A probabilistic linear classifier that models the probability of each class. Serves as the interpretable **baseline model**.

- Train Accuracy: **97.5%**
- Test Accuracy: **100.0%**
- Train ≈ Test → **No overfitting**

**Logistic Regression Prediction Curve** was plotted to visualize class separation using `SepalLengthCm`.

---

### 2. Decision Tree Classifier
```python
DecisionTreeClassifier()
```
Recursively splits data using feature thresholds. Highly interpretable but prone to overfitting on larger datasets. On this small, clean dataset, it achieves perfect accuracy.

---

### 3. Random Forest Classifier
```python
RandomForestClassifier(n_estimators=100, random_state=42)
```
An ensemble of 100 decision trees, each trained on a random data and feature subset. Reduces variance and improves generalization over a single decision tree.

---

### 4. Support Vector Classifier (SVC)
Tested with three different kernels:

| Kernel | Accuracy |
|---|---|
| `rbf` (Radial Basis Function) | 100.0% |
| `linear` | 100.0% |
| `poly` (Polynomial) | 100.0% |

All kernels performed identically on this dataset. Final evaluation used the `poly` kernel result.

---

### 5. K-Nearest Neighbors (KNN)
```python
KNeighborsClassifier()
```
Classifies each sample based on the majority vote of its K nearest neighbors (default K=5). Works well on small, well-separated datasets like Iris.

---

## 📐 Model Evaluation Metrics

### Metrics Explained

| Metric | Formula | What It Means |
|---|---|---|
| **Accuracy** | Correct Predictions / Total Predictions | Overall % of correct classifications |
| **Precision** | TP / (TP + FP) | Of all predicted positives, how many are actually positive |
| **Recall** | TP / (TP + FN) | Of all actual positives, how many were correctly predicted |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Grid of TP, TN, FP, FN | Shows exactly which classes were confused |

### Support Values on Test Set (30 samples)

| Class | Support (Actual Samples) |
|---|---|
| Setosa (0) | 10 |
| Versicolor (1) | 9 |
| Virginica (2) | 11 |

---

## 📈 Results & Comparison

### Accuracy Scores

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| **Logistic Regression** | 97.50% | **100.0%** |
| **Decision Tree Classifier** | 100.0% | **100.0%** |
| **Random Forest Classifier** | 100.0% | **100.0%** |
| **SVC (rbf / linear / poly)** | 100.0% | **100.0%** |
| **KNN** | 100.0% | **100.0%** |

### Classification Report (All Models — Identical)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Setosa (0) | 1.00 | 1.00 | 1.00 | 10 |
| Versicolor (1) | 1.00 | 1.00 | 1.00 | 9 |
| Virginica (2) | 1.00 | 1.00 | 1.00 | 11 |
| **Macro Avg** | **1.00** | **1.00** | **1.00** | **30** |
| **Weighted Avg** | **1.00** | **1.00** | **1.00** | **30** |

### Confusion Matrix (All Models — Identical)

```
              Predicted
              0    1    2
Actual  0  [ 10    0    0 ]   ← All 10 Setosa correct
        1  [  0    9    0 ]   ← All 9 Versicolor correct
        2  [  0    0   11 ]   ← All 11 Virginica correct
```

> ✅ **Zero misclassifications across all models.** All off-diagonal values = 0.

### Interpretation

- **Logistic Regression:** Train (97.5%) ≈ Test (100%) → confirms the model is **not overfitting**. The slight train-test difference is normal for a clean, small dataset.
- **Decision Tree:** Perfect on both sets — the dataset's clear feature boundaries make it tree-friendly, though this model would be prone to overfitting on noisier data.
- **Random Forest:** Perfect accuracy with the added benefit of ensemble robustness.
- **SVC:** All three kernels (RBF, Linear, Poly) achieve perfect classification — demonstrates that Iris classes are **linearly separable** in feature space.
- **KNN:** Perfect classification with default K=5, confirming that same-species flowers are tightly clustered in feature space.

---

## ✅ Final Conclusion

All five models achieved **100% accuracy** on the 30-sample test set, with perfect precision, recall, and F1-scores across all three classes.

> 🏆 **All models perform equally well** — the best model for production depends on the deployment context.

### Why Perfect Accuracy is Valid Here:
- The Iris dataset is **small (150 samples)** and **perfectly balanced (50 per class)**
- **Zero missing values or noise** in the data
- Petal features provide **very strong linear separation** between species
- Logistic Regression's Train (97.5%) vs Test (100%) confirms this is genuine performance, not overfitting

### Model Recommendation by Use Case

| Use Case | Recommended Model |
|---|---|
| Interpretability / explainability | Logistic Regression |
| Fast training + good accuracy | Decision Tree |
| Best generalization on larger data | Random Forest |
| High-dimensional or complex boundaries | SVC (RBF kernel) |
| Simple, no-training-needed baseline | KNN |

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, analysis |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Heatmaps, regression plots |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `kagglehub` | Dataset download from Kaggle |

---

## ▶️ How to Run

```bash
# 1. Clone or download the project
git clone https://github.com/your-username/iris-flower-classification.git
cd iris-flower-classification

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub

# 3. Configure Kaggle API (for dataset download)
# Place your kaggle.json API token in ~/.kaggle/kaggle.json
# OR set environment variables:
# export KAGGLE_USERNAME=your_username
# export KAGGLE_KEY=your_api_key

# 4. Run the notebook
jupyter notebook Task_1_Iris_Flower_Classification_-_ML.ipynb
```

> **Note:** The dataset is auto-downloaded via `kagglehub.dataset_download("saurabh00007/iriscsv")` on first run. No manual CSV download needed.

---

## 📁 Project Structure

```
iris-flower-classification/
│
├── Task_1_Iris_Flower_Classification_-_ML.ipynb  # Main notebook
└── README.md                                      # This file
```

---

*Built with Python & Scikit-learn | Iris Flower Multi-Class Classification*
