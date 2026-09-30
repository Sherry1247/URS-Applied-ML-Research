import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Load the data (Adjust the path if your file is in a different location)
df = pd.read_csv('data/Kaggle_medical_insurance_dataset/insurance.csv')

# --- Define the Segments ---
# Charges: 0-17000 (Low), 15000-32000 (Medium), 31000-60000 (High)
segments = {
    'Low Charges':    (0, 17000, 'blue'),
    'Medium Charges': (15000, 32000, 'orange'),
    'High Charges':   (31000, 60000, 'red')
}

# --- Plotting Setup ---
X_VAR = 'age'
Y_VAR = 'charges'

plt.figure(figsize=(12, 7))
sns.scatterplot(x=X_VAR, y=Y_VAR, data=df, alpha=0.3, color='gray', label='All Data Points')

# --- Fit and Plot Regression for Each Segment ---
for name, (min_c, max_c, color) in segments.items():
    # 1. Filter the data into the charge segment
    segment_df = df[(df[Y_VAR] >= min_c) & (df[Y_VAR] <= max_c)].copy()

    if not segment_df.empty:
        # 2. Prepare data for sklearn
        X = segment_df[[X_VAR]]
        y = segment_df[Y_VAR]

        # 3. Fit the Linear Regression Model
        model = LinearRegression()
        model.fit(X, y)

        # 4. Calculate R^2 for display in the legend
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        # 5. Generate prediction line over the segment's age range
        x_min = X[X_VAR].min()
        x_max = X[X_VAR].max()
        x_range = np.array([[x_min], [x_max]])
        y_plot = model.predict(x_range)

        # 6. Plot the Regression Line
        plt.plot(x_range, y_plot, color=color, linewidth=3,
                 label=f'{name} (R²: {r_squared:.2f})')
        
        # Print the equation
        print(f"{name} Equation: Charges = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Age")

# --- Final Plot Customization ---
plt.title(f'Segmented Linear Regression of Charges vs. {X_VAR.capitalize()} by Charge Range')
plt.xlabel(X_VAR.capitalize())
plt.ylabel('Charges (USD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
