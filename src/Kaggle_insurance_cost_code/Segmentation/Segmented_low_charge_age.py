import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Define file and variables
file_name = 'low_charge_charges.csv'
X_VAR = 'age'
Y_VAR = 'charges'
degree = 2 # Quadratic

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

# --- 1. Load and Initial Clean ---
df = pd.read_csv(file_name)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['charges'] = pd.to_numeric(df['charges'], errors='coerce')
df.dropna(inplace=True)
initial_rows = len(df)

# --- 2. Aggressive Outlier Deletion (Targeted Cleaning) ---

# A. Apply standard IQR (catches general outliers)
df_clean = remove_outliers_iqr(df, X_VAR)
df_clean = remove_outliers_iqr(df_clean, Y_VAR)

# B. Apply custom filter to remove the "left upper corner" points:
#    (Age < 30 AND Charges > 12000)
targeted_outlier_mask = ~((df_clean[X_VAR] < 30) & (df_clean[Y_VAR] > 12000))
df_clean = df_clean[targeted_outlier_mask].copy()

removed_rows = initial_rows - len(df_clean)

X = df_clean[[X_VAR]]
y = df_clean[Y_VAR]

# --- 3. Fit Polynomial Regression (Degree 2) ---
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# Get Polynomial Function and Metrics
intercept = model.intercept_
coefficients = model.coef_[1:] 
r_squared = model.score(X_poly, y) # R^2
pearson_r, _ = pearsonr(X[X_VAR], y)
y_pred = model.predict(X_poly)


# --- 4. Report and Visualization ---
segment_name = file_name.replace('_charges.csv', '').replace('_', ' ').title()

# Print polynomial function and metrics
print(f"--- Polynomial Regression Results: {segment_name} Segment (Targeted Cleaning) ---")
print(f"Total Outliers Deleted: {removed_rows} rows.")
print(f"R-squared (R²): {r_squared:.4f}")
print(f"Pearson Correlation (r): {pearson_r:.4f}")
print("\n--- Polynomial Function ---")
print(f"y = {intercept:,.2f} + {coefficients[0]:,.2f} * Age + {coefficients[1]:,.4f} * Age^2")


# Plotting the result
X_fit = np.linspace(X[X_VAR].min(), X[X_VAR].max(), 100).reshape(-1, 1)
X_fit_poly = poly_features.transform(X_fit)
y_fit_pred = model.predict(X_fit_poly)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_VAR, y=Y_VAR, data=df_clean, alpha=0.6, label='Cleaned Data Points')
plt.plot(X_fit, y_fit_pred, color='red', linewidth=3, label=f'Polynomial Regression (Order {degree})')

# Annotation
plt.text(0.05, 0.95, 
         f'R²: {r_squared:.4f}\nPearson r: {pearson_r:.4f}\nOutliers Deleted: {removed_rows}', 
         transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', 
         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))

plt.title(f'Polynomial Regression: Charges vs. Age ({segment_name}, Targeted Cleaning)')
plt.xlabel('Age')
plt.ylabel('Charges (USD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
