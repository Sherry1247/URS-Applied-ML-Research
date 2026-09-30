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

# Clean data: convert to numeric and drop NaNs (Good practice)
for col in ['age', 'bmi', 'children', 'charges']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True) 

# --- 2. Violin Plot: Sex vs. Charges ---
plt.figure(figsize=(8, 6))
sns.violinplot(x='sex', y='charges', data=df, palette={'male': 'skyblue', 'female': 'lightcoral'})
plt.title('Distribution of Charges by Sex')
plt.xlabel('Sex')
plt.ylabel('Charges (USD)')
plt.show() 
