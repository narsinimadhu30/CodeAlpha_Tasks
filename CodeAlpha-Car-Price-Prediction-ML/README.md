# 🚗 Car Price Prediction — Machine Learning Project

> Predict the **selling price of a used car** using regression models trained on real-world features like showroom price, mileage, fuel type, and car age.

---

## 📌 Table of Contents

- [`Project Overview`](#project-overview)
- [`Dataset Description`](#dataset-description)
- [`Project Workflow`](#project-workflow)
- [`Exploratory Data Analysis`](#exploratory-data-analysis)
- [`Feature Engineering & Preprocessing`](#feature-engineering--preprocessing)
- [`Model Building`](#model-building)
- [`Model Evaluation Metrics`](#model-evaluation-metrics)
- [`Results & Comparison`](#results--comparison)
- [`Final Conclusion`](#final-conclusion)
- [`Technologies Used`](#technologies-used)
- [`How to Run`](#how-to-run)

---

## 📖 Project Overview

This project builds a **machine learning pipeline** to predict the resale price of used cars. Three regression models are trained and evaluated:

| Model | Type |
|---|---|
| `Linear Regression` | `Parametric, baseline model` |
| `Decision Tree Regressor` | `Non-parametric, rule-based` |
| `Random Forest Regressor` | `Ensemble of decision trees` |

The best model is selected based on **R² Score**, **RMSE**, and **MAE**.

---

## 📂 Dataset Description

- **File:** `car data.csv`
- **Shape:** 301 rows × 9 columns (after loading)
- **Target Variable:** `Selling_Price` (in Lakhs ₹)

### Original Columns

| Column | Description | Type |
|---|---|---|
| `Car_Name` | Name/model of the car | Categorical |
| `Year` | Manufacturing year | Numeric |
| `Selling_Price` | Resale price (target) | Float |
| `Present_Price` | Showroom/ex-showroom price | Float |
| `Driven_kms` | Total kilometers driven | Integer |
| `Fuel_Type` | Petrol / Diesel / CNG | Categorical |
| `Selling_type` | Dealer or Individual | Categorical |
| `Transmission` | Manual or Automatic | Categorical |
| `Owner` | Number of previous owners | Integer |

---

## 🔄 Project Workflow

```
Load Dataset
     ↓
Exploratory Data Analysis (EDA)
     ↓
Feature Engineering
  → Rename Present_Price → Showroom_Price
  → Create Car_Age = 2026 - Year, drop Year
  → Drop Car_Name (not useful)
     ↓
Data Preprocessing
  → Check & drop nulls (0 found)
  → Remove duplicates (2 removed)
  → Outlier analysis (retained, all cols are informative)
  → One-Hot Encoding (pd.get_dummies)
  → Convert bool columns to int
     ↓
Train / Test Split (80% / 20%, random_state=47)
     ↓
Train 3 Models
     ↓
Evaluate: R², RMSE, MAE, MSE
     ↓
Compare & Select Best Model
```

---

## 📊 Exploratory Data Analysis

### Dataset Info
- **Total records:** 301
- **Missing values:**  None (`0`)
- **Duplicate rows:** `2` — removed during cleaning
- **Data types:** Float (2), Integer (3), Object/Categorical (4)

### Key Observations

- **Fuel Type Distribution:**
  - Petrol: 239 cars
  - Diesel: 58 cars
  - CNG: 2 cars

- **Owner Values:** `0` = First Owner, `1` = Second Owner, `3` = Third Owner

- **Outliers:** Present in `Selling_Price`, `Showroom_Price`, `Driven_kms`, `Car_Age`, `Owner` — but **retained** because all are genuinely informative for price prediction:
  - Lower car age → higher price
  - Lower driven kms → higher price
  - Premium brands drive legitimate high-price outliers

---

## ⚙️ Feature Engineering & Preprocessing

### 1. Column Rename
```python
car_data.rename(columns={'Present_Price': 'Showroom_Price'}, inplace=True)
```

### 2. Car Age Feature
```python
car_data["Car_Age"] = 2026 - car_data["Year"]
car_data.drop("Year", axis=1, inplace=True)
```
Year alone is not meaningful for prediction; **Car Age** directly captures depreciation.

### 3. Drop Car Name
```python
car_data.drop('Car_Name', axis=1, inplace=True)
```
Too many unique values with no ordinal meaning — excluded to prevent noise.

### 4. One-Hot Encoding
```python
car_data = pd.get_dummies(car_data, drop_first=True)
```

Resulting encoded columns:
| Original Feature | Encoded As |
|---|---|
| Fuel_Type | `Fuel_Type_Diesel`, `Fuel_Type_Petrol` |
| Selling_type | `Selling_type_Individual` |
| Transmission | `Transmission_Manual` |

### 5. Final Feature Set (X)

| Feature | Description |
|---|---|
| `Showroom_Price` | Ex-showroom price in Lakhs |
| `Driven_kms` | Total kilometers driven |
| `Owner` | Number of previous owners |
| `Car_Age` | Age of car in years |
| `Fuel_Type_Diesel` | 1 if Diesel, else 0 |
| `Fuel_Type_Petrol` | 1 if Petrol, else 0 |
| `Selling_type_Individual` | 1 if sold by individual, else 0 |
| `Transmission_Manual` | 1 if Manual, else 0 |

---

## 🧠 Model Building

### Train/Test Split
```python
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=47)
```

| Split | Shape |
|---|---|
| Training set | (239, 8) |
| Test set | (60, 8) |

---

### Models Used

#### 1. Linear Regression
```python
LinearRegression()
```
A parametric model that fits a straight-line (hyperplane) relationship between features and price. Serves as the **baseline** model.

#### 2. Decision Tree Regressor
```python
DecisionTreeRegressor()
```
Splits data recursively using feature thresholds to form decision rules. Captures **non-linear** relationships but can overfit.

#### 3. Random Forest Regressor
```python
RandomForestRegressor()
```
An **ensemble** of multiple decision trees — each trained on a random subset of data and features. Reduces overfitting and improves generalization.

---

### Evaluation Plots (per model)

For each model, three diagnostic plots were generated:

| Plot | Purpose |
|---|---|
| **Actual vs Predicted** | How closely predictions track real values |
| **Residual Plot** | Checks for patterns in errors (should be random) |
| **Error Distribution** | Checks normality of prediction errors |

---

## 📐 Model Evaluation Metrics

### Metrics Explained

| Metric | Formula | What It Means |
|---|---|---|
| **R² Score** | 1 - (SS_res / SS_tot) | % of variance explained. Higher = better. |
| **MAE** | mean(|y - ŷ|) | Average absolute error in Lakhs ₹ |
| **MSE** | mean((y - ŷ)²) | Penalizes large errors more heavily |
| **RMSE** | √MSE | Same unit as target (Lakhs ₹). Lower = better. |

---

## 📈 Results & Comparison

### Model Performance on Test Set

| Model | R² Score | MSE | RMSE | MAE |
|---|---|---|---|---|
| **`Linear Regression`** | 80.61% | 4.509 | 2.123 Lakhs | 1.301 Lakhs |
| **`Decision Tree`** | 90.36% | 2.242 | 1.497 Lakhs | 0.645 Lakhs |
| **`Random Forest`** | **`90.40%`** | **`2.233`** | **`1.494 Lakhs`** | **`0.578 Lakhs`** |

### Interpretation

#### 🔵 Linear Regression — R²: 80.61%
- Achieved a decent baseline of **80.6% accuracy**
- Residual plot shows a **systematic pattern** → model is **slightly underfitting**
- Error distribution is **right-skewed**, indicating the model struggles with high-priced cars
- RMSE: ~2.12 Lakhs | MAE: ~1.30 Lakhs

#### 🟡 Decision Tree — R²: 90.36%
- Significant improvement over Linear Regression (+~10%)
- Residual plot shows **sharp/jagged patterns** → model is **overfitting**
- Error distribution is right-skewed but narrower
- RMSE: ~1.50 Lakhs | MAE: ~0.64 Lakhs

#### 🟢 Random Forest — R²: 90.40% ✅ Best
- Marginally better R² than Decision Tree, but **much better generalization**
- Residual plot shows more **randomly scattered** residuals → healthier fit
- Lowest MAE: **0.578 Lakhs (~₹57,800 average error)**
- RMSE: ~1.494 Lakhs

---

## ✅ Final Conclusion

After comparing all three models across R² Score, RMSE, and MAE:

> 🏆 **Random Forest Regressor** is selected as the final model.

### Why Random Forest Wins:
- Highest R² score: **90.40%**
- Lowest MAE: **~₹57,800** average prediction error
- Best residual behavior — reduced overfitting compared to Decision Tree
- Ensemble approach naturally handles outliers and non-linear relationships

### Final Model Metrics Summary

| Metric | Value | Interpretation |
|---|---|---|
| R² Score | **90.40%** | Model explains 90.4% of price variance |
| RMSE | **1.494 Lakhs** | ~₹1,49,400 typical prediction deviation |
| MAE | **0.578 Lakhs** | ~₹57,800 average absolute error |
| MSE | **2.233** | Squared error (for optimization reference) |

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, feature engineering |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical plots (boxplot, regplot) |
| `scikit-learn` | ML models, train-test split, metrics |

---

## ▶️ How to Run

```bash
# 1. Clone or download the project
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Place the dataset
# Add 'car data.csv' to your working directory or update the path in the notebook

# 4. Run the notebook
jupyter notebook Task_3_Car_Price_Prediction_-_ML.ipynb
```

---

## 📁 Project Structure

```
car-price-prediction/
│
├── car data.csv                           # Raw dataset
├── Task_3_Car_Price_Prediction_-_ML.ipynb # Main notebook
└── README.md                              # This file
```

---

*Built with Python & Scikit-learn | Car Price Prediction using Regression Models*

