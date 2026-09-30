import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load and Clean the Data (MANDATORY step to define 'df') ---
# IMPORTANT: Adjust the file path if 'insurance.csv' is not at this location
FILE_PATH = 'data/Kaggle_medical_insurance_dataset/insurance.csv'
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    # Fallback to the file uploaded to this session
    df = pd.read_csv('insurance.csv')

# Clean data: convert to numeric and drop NaNs to prevent plotting errors
for col in ['age', 'bmi', 'children', 'charges']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True) 

plt.figure(figsize=(10, 6))
sns.violinplot(x='region', y='charges', data=df, palette='Set2')
plt.title('Distribution of Charges by Geographic Region')
plt.xlabel('Region')
plt.ylabel('Charges (USD)')
plt.show() # <-- Displays the Region plot
