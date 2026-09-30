import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
# Assuming the full dataset is named 'insurance.csv'
df = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/data/Kaggle_medical_insurance_dataset/insurance.csv')

# 2. Convert Categorical Variables to Dummy Variables (One-Hot Encoding)
# Drop the original categorical columns after creating dummies
# 'drop_first=True' is used to avoid perfect multicollinearity,
# making one category (the first one alphabetically) the reference category.
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# 3. Rename columns for cleaner visualization
df_encoded.rename(columns={
    'sex_male': 'Sex_Male',
    'smoker_yes': 'Smoker_Yes',
    'region_northwest': 'Region_Northwest',
    'region_southeast': 'Region_southeast',
    'region_southwest': 'Region_Southwest'
}, inplace=True)

# 4. Calculate the Correlation Matrix
corr_matrix = df_encoded.corr()

# 5. Generate the Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    linewidths=.5, 
    cbar=True
)
plt.title('Full Correlation Heatmap: All Variables vs. Charges')
plt.show() 
