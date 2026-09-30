import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/data/Kaggle_Tip_dataset/tip.csv'
df = pd.read_csv(file_path)

# List of categorical/discrete variables for violin plots
categorical_vars = ['sex', 'smoker', 'day', 'time', 'size']

# --- Individual Violin Plot Function ---
# This function generates one plot per variable.
def plot_individual_violin(data, var, index):
    plt.figure(figsize=(6, 6))
    
    # Order for 'day' to be sequential
    order = ['Thur', 'Fri', 'Sat', 'Sun'] if var == 'day' else None
    
    # Uses the original seaborn violinplot style
    sns.violinplot(x=var, y='tip', data=data, palette='Set3', order=order)
    
    plt.title(f'{index}. Tip Distribution by {var.title()}')
    plt.xlabel(var.title())
    plt.ylabel('Tip ($)')
    plt.show()

# Execute and show the 5 individual violin plots:
print("Generating 5 individual violin plots...")

plot_individual_violin(df, 'sex', 1)
plot_individual_violin(df, 'smoker', 2)
plot_individual_violin(df, 'day', 3)
plot_individual_violin(df, 'time', 4)
plot_individual_violin(df, 'size', 5)
