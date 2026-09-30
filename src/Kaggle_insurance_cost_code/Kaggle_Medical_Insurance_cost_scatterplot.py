# source: https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset
# Author: Siqi Dai
# Description: Data visualization of Medical Insurance Cost Dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/Kaggle_medical_insurance_dataset/insurance.csv')

target_variable = 'charges'
numerical_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

# =================================================================
# 1. Scatter Plots for Numerical Variables vs. Charges
# =================================================================
print("Generating Scatter Plots for Numerical Features...")
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    
    # Use standard scatter plot for numerical features
    sns.scatterplot(x=feature, y=target_variable, data=df, alpha=0.6)
    
    plt.title(f'Insurance Charges vs. {feature.capitalize()}')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Charges (USD)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'scatter_charges_vs_{feature}.png')
    plt.close()

# =================================================================
# 2. Swarm Plots for Categorical Variables vs. Charges
# (Swarm plot is the best "scatter-like" plot for categorical data)
# =================================================================
print("Generating Swarm Plots for Categorical Features...")
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    
    # Use swarm plot to show individual points for categorical features
    # 'size' and 'palette' are optional, but improve visual clarity
    sns.swarmplot(x=feature, y=target_variable, data=df, size=3, palette='viridis')
    
    plt.title(f'Insurance Charges by {feature.capitalize()} (Swarm Plot)')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Charges (USD)')
    plt.savefig(f'swarm_charges_vs_{feature}.png')
    plt.close()

print("All six visualization files have been generated and saved.")
