import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Define file and variables
file_name = 'medium_charge_charges.csv'
X_VAR = 'age'
Y_VAR = 'charges'

# --- Outlier Removal Function (Standard IQR) ---
def remove_outliers_iqr(df, column, factor=1.5):
    """Removes outliers based on a customizable IQR factor."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (factor * IQR)
    upper_bound = Q3 + (factor * IQR)
    non_outlier_mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    return df[non_outlier_mask]

# --- 1. Load and Clean Data (Standard IQR Cleaning) ---
df = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/data/Kaggle_medical_insurance_dataset/medium_charge_charges.csv')
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['charges'] = pd.to_numeric(df['charges'], errors='coerce')
df.dropna(inplace=True)
initial_rows = len(df)

# Apply standard IQR cleaning on both Age and Charges
df_clean = remove_outliers_iqr(df, X_VAR)
df_clean = remove_outliers_iqr(df_clean, Y_VAR)

removed_rows = initial_rows - len(df_clean)

X = df_clean[[X_VAR]].values
y = df_clean[Y_VAR].values

comparison_results = []
segment_name = file_name.replace('_charges.csv', '').replace('_', ' ').title()

# --- 2. Model 1: Quadratic Polynomial (Degree 2) ---
poly2_features = PolynomialFeatures(degree=2)
X_poly2 = poly2_features.fit_transform(X)
model2 = LinearRegression()
model2.fit(X_poly2, y)
r_squared2 = model2.score(X_poly2, y)

comparison_results.append({
    'Model': 'Quadratic Polynomial', 
    'R^2': r_squared2,
    'Equation': f'y = {model2.intercept_:.2f} + {model2.coef_[1]:.2f}*Age + {model2.coef_[2]:.4f}*Age^2'
})

# --- 3. Model 2: Cubic Polynomial (Degree 3) ---
poly3_features = PolynomialFeatures(degree=3)
X_poly3 = poly3_features.fit_transform(X)
model3 = LinearRegression()
model3.fit(X_poly3, y)
r_squared3 = model3.score(X_poly3, y)

comparison_results.append({
    'Model': 'Cubic Polynomial', 
    'R^2': r_squared3,
    'Equation': f'y = {model3.intercept_:.2f} + {model3.coef_[1]:.2f}*Age + {model3.coef_[2]:.4f}*Age^2 + {model3.coef_[3]:.6f}*Age^3'
})

# --- 4. Model 3: Power Law (y = c * X^k) ---
# Transform to log-linear form: log(y) = log(c) + k * log(X)
X_log = np.log(X)
y_log = np.log(y)

model_power = LinearRegression()
model_power.fit(X_log, y_log)

# R^2 is calculated on the log-transformed data
r_squared_power_log = model_power.score(X_log, y_log) 

# Coefficients: log(c) = intercept, k = slope
log_c = model_power.intercept_
k = model_power.coef_.flatten()[0] 
c = np.exp(log_c)

comparison_results.append({
    'Model': 'Power Law (Log-Transformed)', 
    'R^2': r_squared_power_log,
    'Equation': f'y = {c:.4f} * Age^{k:.4f}'
})

# --- 5. Create Comparison Table (DataFrame) ---
results_df = pd.DataFrame(comparison_results)
results_df['R^2'] = results_df['R^2'].round(4)
results_df = results_df.sort_values(by='R^2', ascending=False)
# Table printed in code output

# --- 6. Visualization (Plot all 3 fits) ---
plt.figure(figsize=(12, 7))
sns.scatterplot(x=df_clean[X_VAR], y=df_clean[Y_VAR], alpha=0.5, label='Cleaned Data Points')

# Create smooth X range for plotting curves
X_fit_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# Plot 1: Quadratic
X_fit_poly2 = poly2_features.transform(X_fit_range)
y_fit_pred2 = model2.predict(X_fit_poly2)
plt.plot(X_fit_range, y_fit_pred2, color='blue', linewidth=2, label=f'Quadratic (R²: {r_squared2:.4f})')

# Plot 2: Cubic
X_fit_poly3 = poly3_features.transform(X_fit_range)
y_fit_pred3 = model3.predict(X_fit_poly3)
plt.plot(X_fit_range, y_fit_pred3, color='orange', linewidth=2, label=f'Cubic (R²: {r_squared3:.4f})')

# Plot 3: Power Law
y_fit_pred_power = c * (X_fit_range ** k)
plt.plot(X_fit_range, y_fit_pred_power, color='green', linewidth=2, linestyle='--', label=f'Power Law (R² Log: {r_squared_power_log:.4f})')

plt.title(f'Comparison of Non-Linear Fits for {segment_name} Segment (IQR Cleaned)')
plt.xlabel('Age')
plt.ylabel('Charges (USD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

output_filename = f'non_linear_comparison_{segment_name.lower().replace(" ", "_")}.png'
plt.savefig(output_filename)
plt.show()
