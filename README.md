
# 🪐 Astrophysical Dataset Analysis: Orbital Risk Prediction Using Machine Learning

This project explores an **astrophysical dataset** to identify potentially hazardous asteroids based on orbital parameters and physical features. It combines advanced **data preprocessing**, **exploratory analysis**, **feature engineering**, and **machine learning techniques** to accurately assess asteroid threat levels.

---

## 📌 Objective

To predict whether an asteroid poses a threat to Earth using structured orbital data by applying **classification models** and extracting meaningful patterns through scientific computation and feature analysis.

---

## 📊 Dataset Overview

- **Data Type**: Structured data with both numerical and categorical features.
- **Total Samples**: ~4,000 asteroid entries.
- **Key Features**:
  - `Name`: Asteroid ID
  - `Semi-Major Axis`, `Aphelion Distance`, `Miss Distance`
  - `Relative Velocity`, `Jupiter Tisserand Invariant`
  - `Approach Date`, `Mean Anomaly`, `Orbital Period`
- **Target Variable**: `Hazardous` (binary classification: 0 or 1)

---

## 🧹 Data Preprocessing

### ✔️ Handling Missing Values
- Applied `bfill`, `ffill`, and interpolation for numeric features.
- Used `mode()` to fill missing categorical values.
- Context-specific transformations for `approach_date` and `distance` fields.

### 🔁 Unit & Format Conversion
- All distances converted to **Astronomical Units (AU)**.
- Dates converted to proper `datetime` formats for easier time-based operations.

### ⚙️ Data Type Adjustments
- Numerical values converted to `int` or `float`.
- Categorical values encoded to optimize memory and performance.

### 📅 Time-Based Feature Creation
- Created `Time_Until_Approach` feature using:
  ---

## 📈 Exploratory Data Analysis (EDA)

### 📉 Descriptive Statistics
- Relative Velocity: Avg ≈ 50,516 km/h; Range: 1,207–160,681 km/h.
- Miss Distance (AU): Avg ≈ 0.258 AU
- Tisserand Invariant: Mean ≈ 5.126, Range: 2.36 – 9.02

### 📊 Visualizations
- **Histograms** reveal clustering and skewness in velocity, miss distance, and year.
- **Boxplots** show:
- High variability in orbital features.
- Significant outliers in Perihelion Time and Epoch Osculation.
- **Heatmap** reveals:
- Strong correlation: `Semi-Major Axis` ↔ `Aphelion Distance`
- Negative correlation: `Semi-Major Axis` ↔ `Mean Motion` (supports Kepler's law)

---

## 🏗 Feature Engineering

### 🆕 New Features
- `True Anomaly`, `Eccentric Anomaly` (from Kepler's equation)
- `Heliocentric Distance`, `Orbital Period`
- `Velocity at Aphelion`, `Escape Velocity`
- `Miss Distance / Semi-Major Axis Ratio`

### 📏 Normalization
- Applied **Min-Max Scaling** to bring all numeric features to [0, 1] range.
- Essential for neural networks and distance-based models.

### 🚨 Outlier Detection
- Used **Z-Score** method (threshold = 3), flagged 341 outliers.
- Mainly in:
- `Miss Distance (km)`
- `Relative Velocity`
- `Jupiter Tisserand Invariant`

### ↔️ Skewness Analysis
- Many features were right-skewed (e.g., `Heliocentric Distance = 1.43`)
- Addressed using feature transformation to improve model behavior.

---

## 🤖 Modeling: Neural Network

### 🧠 Model Architecture
- Input Layer: Size matched to features
- Dense Layer 1: 32 units, `ReLU`
- Dense Layer 2: 16 units, `Tanh`
- Output: 1 unit, `Sigmoid` for binary classification

### ⚙️ Model Compilation
- **Loss**: `BinaryCrossentropy`
- **Optimizer**: `Adam`

### 🔄 K-Fold Cross Validation
- **k = 10 folds**
- Trained and validated across 10 partitions of the dataset
- Plotted **Loss vs Epochs** and **Accuracy vs Epochs**

### 🔍 GridSearch Hyperparameter Tuning
- Tuned:
- `units`: [16, 32, 64]
- `learning_rate`: [0.001, 0.01, 0.1]
- Used **GridSearchCV** with 3-fold cross-validation.

---

## 📉 Evaluation

### ✅ Metrics
- **Accuracy**: ~83%
- **AUC-ROC**: 0.73
- **Confusion Matrix**:
- TP = 24, FP = 40
- TN = 728, FN = 115

### 📊 Visualizations
- **ROC Curve** to analyze threshold tradeoffs
- **Residual & Confusion Matrix Plots**

---

## 💡 Feature Importance

### 🔍 SHAP (SHapley Additive Explanations)
- Feature 0 (possibly Miss Distance) was most influential.
- Red/Blue color scale indicates impact of high/low feature values.

### 🔀 Permutation Importance
- Top Features:
- `Name`
- `Miss Distance (AU)`
- `Relative Velocity`
- Least Important:
- `Heliocentric Distance`, `Mean Anomaly`, `Orbital Energy Ratio`

---

## 🚨 Anomaly Detection

### ✅ Methods Used
1. **Isolation Forest**: ML-based method with contamination = 0.05
2. **Z-Score**: Statistical method (threshold = 3.05)

### 📊 Comparison
- Used a **confusion matrix** to compare anomaly results.
- Heatmap plotted overlap between both techniques.

---

## 🧪 Libraries Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `tensorflow.keras`, `scikeras`
- `sklearn.model_selection`, `sklearn.metrics`
- `shap`, `scipy`, `stats`


## ✅ Conclusion

This project demonstrates a full pipeline of **asteroid risk classification**, from preprocessing and visualization to advanced modeling and evaluation. By applying **deep feature engineering**, **cross-validation**, and **explainability techniques**, it delivers actionable insights into celestial object behavior and risk prediction.
---

## 👨‍🚀 Author

**[Mayank Nagar]**  
Astrophysics Enthusiast | Machine Learning Explorer  
