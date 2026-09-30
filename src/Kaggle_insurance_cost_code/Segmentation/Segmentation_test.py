import pandas as pd
import numpy as np

# --- 1. Load and Initial Clean ---
FILE_PATH = 'insurance.csv'
try:
    df = pd.read_csv('data/Kaggle_medical_insurance_dataset/insurance.csv')
except FileNotFoundError:
    print("Error: 'insurance.csv' not found. Please check the file path.")
    exit()

# Ensure numeric columns are clean (handle initial non-numeric strings/NaNs)
for col in ['age', 'bmi', 'children', 'charges']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True) 

# --- 2. Define Segments and Columns ---
segments = {
    'low_charge': (0, 17000),
    'medium_charge': (15000, 32000),
    'high_charge': (31000, 60000)
}
numeric_cols = ['age', 'bmi', 'children', 'charges']

# --- 3. Outlier Detection Function (IQR Method) ---
def detect_outliers_iqr(df_segment, column):
    """Identifies rows where the value in the specified column is an IQR outlier."""
    Q1 = df_segment[column].quantile(0.25)
    Q3 = df_segment[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    
    # Return the DataFrame containing only the outliers
    outliers = df_segment[(df_segment[column] < lower_bound) | (df_segment[column] > upper_bound)]
    return outliers

# --- 4. Segment, Save, and Detect Outliers for Each Part ---
print("--- Segmented Data Analysis and Outlier Detection ---")
print("-" * 50)

all_outlier_rows = pd.DataFrame()

for name, (min_c, max_c) in segments.items():
    # Filter the data
    df_segment = df[(df['charges'] >= min_c) & (df['charges'] <= max_c)].copy()
    
    # --- Task 1: Segment and Save ---
    output_filename = f'{name}_charges.csv'
    df_segment.to_csv(output_filename, index=False)
    
    print(f"\nSEGMENT: {name.upper()}")
    print(f"File saved: {output_filename} ({len(df_segment)} rows)")
    
    # --- Task 3: Outlier Detection ---
    outlier_rows = set()
    
    for col in numeric_cols:
        outliers = detect_outliers_iqr(df_segment, col)
        
        if not outliers.empty:
            print(f"  {col.upper()}: Found {len(outliers)} IQR Outliers.")
            # Record the index of the outlier rows
            outlier_rows.update(outliers.index.tolist())
        else:
            print(f"  {col.upper()}: No IQR Outliers found.")

    print(f"Total Unique Outlier Rows (across all numeric columns): {len(outlier_rows)}")

print("-" * 50)
