"""
TIP AMOUNT AND TIP PERCENTAGE BY PARTY SIZE ANALYSIS
1. Tip Amount by Party Size (absolute dollar analysis)
2. Tip Percentage by Party Size (relative generosity analysis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/data/Kaggle_Tip_dataset/tip.csv')

# Calculate tip percentage
df['tip_pct'] = (df['tip'] / df['total_bill']) * 100

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ========================================
# PLOT 1: Tip Amount by Party Size
# ========================================
ax1 = axes[0]

# Create box plot
size_groups = [df[df['size'] == s]['tip'].values for s in sorted(df['size'].unique())]
bp1 = ax1.boxplot(size_groups, labels=sorted(df['size'].unique()), 
                  patch_artist=True, widths=0.6,
                  boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                  medianprops=dict(color='red', linewidth=2),
                  whiskerprops=dict(color='black', linewidth=1.5),
                  capprops=dict(color='black', linewidth=1.5))

# Overlay mean values
means = [df[df['size'] == s]['tip'].mean() for s in sorted(df['size'].unique())]
positions = range(1, len(means) + 1)
ax1.plot(positions, means, 'D-', color='darkgreen', markersize=8, linewidth=2, 
         label='Mean Tip', markeredgecolor='black', markeredgewidth=1)

# Add regression line
X_size = np.array(sorted(df['size'].unique())).reshape(-1, 1)
y_mean = np.array(means)
lr = LinearRegression()
lr.fit(X_size, y_mean)
y_pred_line = lr.predict(X_size)
ax1.plot(positions, y_pred_line, 'r--', linewidth=2, 
         label=f'Trend: +${lr.coef_[0]:.2f}/person')

ax1.set_xlabel('Party Size', fontsize=13, fontweight='bold')
ax1.set_ylabel('Tip Amount ($)', fontsize=13, fontweight='bold')
ax1.set_title('Tip Amount by Party Size: Absolute Dollar Analysis', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add sample sizes
for i, size in enumerate(sorted(df['size'].unique()), 1):
    count = len(df[df['size'] == size])
    ax1.text(i, -0.5, f'n={count}', ha='center', fontsize=9, style='italic')

# Add interpretation box
interp_text = (
    "✓ Clear upward trend\n"
    "✓ Each additional guest\n"
    f"   adds ~${lr.coef_[0]:.2f} to tip\n"
    "⚠ High variability in\n"
    "   sizes 3-5 (wider boxes)"
)
ax1.text(0.98, 0.02, interp_text, transform=ax1.transAxes, 
         fontsize=10, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, 
                  edgecolor='black', linewidth=1.5))

# ========================================
# PLOT 2: Tip PERCENTAGE by Party Size
# ========================================
ax2 = axes[1]

# Create box plot for tip percentage
size_groups_pct = [df[df['size'] == s]['tip_pct'].values for s in sorted(df['size'].unique())]
bp2 = ax2.boxplot(size_groups_pct, labels=sorted(df['size'].unique()), 
                  patch_artist=True, widths=0.6,
                  boxprops=dict(facecolor='lightcoral', edgecolor='black', linewidth=1.5),
                  medianprops=dict(color='darkblue', linewidth=2),
                  whiskerprops=dict(color='black', linewidth=1.5),
                  capprops=dict(color='black', linewidth=1.5))

# Overlay mean values
means_pct = [df[df['size'] == s]['tip_pct'].mean() for s in sorted(df['size'].unique())]
ax2.plot(positions, means_pct, 's-', color='purple', markersize=8, linewidth=2, 
         label='Mean Tip %', markeredgecolor='black', markeredgewidth=1)

# Add reference lines
ax2.axhline(y=15, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, 
            label='15% (Standard)')
ax2.axhline(y=20, color='green', linestyle='--', linewidth=1.5, alpha=0.7, 
            label='20% (Generous)')

ax2.set_xlabel('Party Size', fontsize=13, fontweight='bold')
ax2.set_ylabel('Tip Percentage (%)', fontsize=13, fontweight='bold')
ax2.set_title('Tip Percentage by Party Size: Relative Generosity Analysis', 
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add sample sizes
for i, size in enumerate(sorted(df['size'].unique()), 1):
    count = len(df[df['size'] == size])
    ax2.text(i, 3, f'n={count}', ha='center', fontsize=9, style='italic')

# Add KEY INSIGHT box
insight_text = (
    "⚠ IMPORTANT FINDING:\n"
    "Tip % DECREASES with\n"
    "larger party sizes!\n\n"
    "• Size 1-2: ~17-22% tip\n"
    "• Size 3-6: ~14-16% tip\n\n"
    "Larger groups tip less\n"
    "as a percentage of bill"
)
ax2.text(0.98, 0.98, insight_text, transform=ax2.transAxes, 
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, 
                  edgecolor='red', linewidth=2))

plt.tight_layout()
plt.savefig('tip_by_party_size_analysis.png', dpi=200, bbox_inches='tight')
plt.show()

print("✓ Visualization saved: tip_by_party_size_analysis.png")
print("\nStatistics Summary:")
print("="*60)
print("\nTip Amount by Party Size:")
for size in sorted(df['size'].unique()):
    mean_tip = df[df['size'] == size]['tip'].mean()
    print(f"  Size {size}: ${mean_tip:.2f} average tip")

print(f"\n  → Trend: +${lr.coef_[0]:.2f} per additional guest")

print("\nTip Percentage by Party Size:")
for size in sorted(df['size'].unique()):
    mean_pct = df[df['size'] == size]['tip_pct'].mean()
    print(f"  Size {size}: {mean_pct:.2f}% average")

correlation = df['size'].corr(df['tip_pct'])
print(f"\n  → Correlation (Size ↔ Tip %): {correlation:.3f} (NEGATIVE!)")
print("="*60)
