import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/data/Kaggle_Tip_dataset/tip.csv')

# Create figure for Total Bill vs Tip analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Tip vs Total Bill with regression line and R²
axes[0].scatter(df['total_bill'], df['tip'], alpha=0.6, color='steelblue', s=60, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('Total Bill ($)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Tip ($)', fontsize=13, fontweight='bold')
axes[0].set_title('Tip vs Total Bill', fontsize=15, fontweight='bold')
axes[0].grid(True, alpha=0.3, linestyle='--')

# Add regression line
z = np.polyfit(df['total_bill'], df['tip'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['total_bill'].min(), df['total_bill'].max(), 100)
axes[0].plot(x_line, p(x_line), "r--", linewidth=2.5, label=f'y = {z[0]:.3f}x + {z[1]:.3f}')

# Calculate R²
from sklearn.metrics import r2_score
y_pred = p(df['total_bill'])
r2 = r2_score(df['tip'], y_pred)

# Add R² to the plot
axes[0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0].transAxes, 
            fontsize=14, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
axes[0].legend(fontsize=11, loc='lower right')

# Plot 2: Tip vs Total Bill colored by Party Size
scatter = axes[1].scatter(df['total_bill'], df['tip'], c=df['size'], cmap='viridis', 
                         alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
axes[1].set_xlabel('Total Bill ($)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Tip ($)', fontsize=13, fontweight='bold')
axes[1].set_title('Tip vs Total Bill (Colored by Party Size)', fontsize=15, fontweight='bold')
axes[1].grid(True, alpha=0.3, linestyle='--')

# Add colorbar
cbar = plt.colorbar(scatter, ax=axes[1])
cbar.set_label('Party Size', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('tip_vs_total_bill.png', dpi=200, bbox_inches='tight')
plt.show()

print("✓ Total Bill analysis plots saved successfully!")
print(f"\nRegression Equation: Tip = {z[0]:.4f} × Total_Bill + {z[1]:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Correlation Coefficient: {df['total_bill'].corr(df['tip']):.4f}")
