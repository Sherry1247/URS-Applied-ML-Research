import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Load the data
df = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/data/Kaggle_medical_insurance_dataset/insurance.csv')

# 2. Convert Categorical Variables to Dummy Variables (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# 3. Rename columns for cleaner visualization
df_encoded.rename(columns={
    'sex_male': 'Sex_Male',
    'smoker_yes': 'Smoker_Yes',
    'region_northwest': 'Region_Northwest',
    'region_southeast': 'Region_Southeast',
    'region_southwest': 'Region_Southwest'
}, inplace=True)

# 4. Define X (features) and y (target) - Including ALL non-interactive features
# These are: age, bmi, children, Sex_Male, Smoker_Yes, Region dummies
X = df_encoded[['age', 'bmi', 'children', 'Sex_Male', 'Smoker_Yes', 'Region_Northwest', 'Region_Southeast', 'Region_Southwest']]
y = df_encoded['charges']

# 5. Split data (for robustness)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Initialize and train the MLR model
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

# 7. Generate and Print the Equation

# Start the equation string with the intercept, rounding to two decimal places
equation = f"Charges = {mlr_model.intercept_:.2f}"

# Iterate through coefficients and add them to the equation string
for feature, coef in zip(X.columns, mlr_model.coef_):
    # Determine the sign for the string
    sign = "+" if coef >= 0 else "-"
    
    # Append the term in the format: + 123.45 * feature
    equation += f" {sign} {abs(coef):.2f} * {feature}"

# Print the final equation
print("\n--- Final Multiple Linear Regression Equation (Standard Features) ---")
print(equation)

r_squared = mlr_model.score(X_test, y_test)
print(f"\nR-squared (R²) on Test Set: {r_squared:.4f}")
