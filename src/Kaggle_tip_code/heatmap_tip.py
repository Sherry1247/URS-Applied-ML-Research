"""
CORRELATION HEATMAP FOR TIP DATASET
This visualization shows correlations between all variables for ANN model preparation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/data/Kaggle_Tip_dataset/tip.csv')

# Calculate tip percentage
df['tip_pct'] = (df['tip'] / df['total_bill']) * 100

# Encode categorical variables numerically for correlation analysis
df_encoded = df.copy()

# Binary encoding for categorical variables
df_encoded['sex_encoded'] = df['sex'].map({'Female': 0, 'Male': 1})
df_encoded['smoker_encoded'] = df['smoker'].map({'No': 0, 'Yes': 1})
df_encoded['time_encoded'] = df['time'].map({'Lunch': 0, 'Dinner': 1})

# Ordinal encoding for day (assuming temporal order)
day_order = {'Thur': 0, 'Fri': 1, 'Sat': 2, 'Sun': 3}
df_encoded['day_encoded'] = df['day'].map(day_order)

# Select features for correlation matrix
features_for_correlation = [
    'total_bill', 'tip', 'tip_pct', 'size',
    'sex_encoded', 'smoker_encoded', 'day_encoded', 'time_encoded'
]

# Rename for better display
feature_names = {
    'total_bill': 'Total Bill',
    'tip': 'Tip ($)',
    'tip_pct': 'Tip %',
    'size': 'Party Size',
    'sex_encoded': 'Sex (M=1)',
    'smoker_encoded': 'Smoker (Y=1)',
    'day_encoded': 'Day (0-3)',
    'time_encoded': 'Time (D=1)'
}

# Calculate correlation matrix
corr_matrix = df_encoded[features_for_correlation].corr()
corr_matrix.rename(index=feature_names, columns=feature_names, inplace=True)

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap with enhanced styling
sns.heatmap(corr_matrix, 
            annot=True,           # Show correlation values
            fmt='.3f',            # 3 decimal places
            cmap='coolwarm',      # Color scheme (blue=negative, red=positive)
            center=0,             # Center colormap at 0
            square=True,          # Square cells
            linewidths=1.5,       # Grid lines
            linecolor='white',    # Grid line color
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
            vmin=-1, vmax=1,      # Scale from -1 to 1
            ax=ax)

# Styling
ax.set_title('Correlation Heatmap: Tips Dataset Features', 
             fontsize=16, fontweight='bold', pad=20)

# Rotate labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)

# Adjust layout
plt.tight_layout()

# Save and display
plt.savefig('correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()

