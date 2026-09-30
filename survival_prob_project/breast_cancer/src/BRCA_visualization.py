# BRCA Breast Cancer Data Visualization - ENHANCED WITH YEAR ANALYSIS
# Part 1: Exploratory Data Analysis including Temporal Trends

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("="*80)
print("BRCA BREAST CANCER - ENHANCED DATA VISUALIZATION")
print("Part 1: Exploratory Data Analysis + Temporal Trends")
print("="*80)

# =============================================================================
# LOAD DATA
# =============================================================================

# Load original BRCA dataset
brca_path = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/breast_cancer/data/BRCA.csv'
brca = pd.read_csv(brca_path)

print(f"\nOriginal dataset: {brca.shape[0]} patients, {brca.shape[1]} features")

# Convert dates and extract years
brca['Date_of_Surgery'] = pd.to_datetime(brca['Date_of_Surgery'], format='%d-%b-%y', errors='coerce')
brca['Date_of_Last_Visit'] = pd.to_datetime(brca['Date_of_Last_Visit'], format='%d-%b-%y', errors='coerce')
brca['Surgery_Year'] = brca['Date_of_Surgery'].dt.year
brca['Last_Visit_Year'] = brca['Date_of_Last_Visit'].dt.year

# Clean dataset
brca_clean = brca.dropna(subset=['Patient_Status']).copy()
brca_clean = brca_clean.dropna(subset=['Age', 'Protein1', 'Protein2', 'Protein3', 
                                        'Protein4', 'Tumour_Stage', 'ER status', 
                                        'PR status', 'HER2 status', 'Surgery_Year'])

print(f"After cleaning: {brca_clean.shape[0]} patients")
print(f"Survival rate: {(brca_clean['Patient_Status']=='Alive').mean():.2%}")
print(f"Year range: {brca_clean['Surgery_Year'].min():.0f} - {brca_clean['Surgery_Year'].max():.0f}")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

print("\nFeature engineering...")

df = brca_clean.copy()

# Encode categorical variables
stage_mapping = {'I': 1, 'II': 2, 'III': 3}
df['Tumour_Stage_Encoded'] = df['Tumour_Stage'].map(stage_mapping)
df['HER2_Binary'] = (df['HER2 status'] == 'Positive').astype(int)
df['ER_Binary'] = (df['ER status'] == 'Positive').astype(int)
df['PR_Binary'] = (df['PR status'] == 'Positive').astype(int)

# One-hot encoding
histology_dummies = pd.get_dummies(df['Histology'], prefix='Histology', drop_first=True)
surgery_dummies = pd.get_dummies(df['Surgery_type'], prefix='Surgery', drop_first=True)
df = pd.concat([df, histology_dummies, surgery_dummies], axis=1)

# Target variable
df['Survival_Binary'] = (df['Patient_Status'] == 'Alive').astype(int)

# Define feature lists
numerical_features = ['Age', 'Protein1', 'Protein2', 'Protein3', 'Protein4', 'Surgery_Year']
encoded_features = ['Tumour_Stage_Encoded', 'HER2_Binary', 'ER_Binary', 'PR_Binary']
histology_features = [col for col in df.columns if col.startswith('Histology_')]
surgery_features = [col for col in df.columns if col.startswith('Surgery_')]

all_features = numerical_features + encoded_features + histology_features + surgery_features

print(f"Total features for modeling: {len(all_features)}")

# Save encoded dataset
encoded_save_path = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/breast_cancer/brca_encoded.csv'
df.to_csv(encoded_save_path, index=False)
print(f"✅ Encoded dataset saved to: {encoded_save_path}")

# =============================================================================
# VISUALIZATION 1: AGE DISTRIBUTION BY SURVIVAL
# =============================================================================

print("\nCreating Visualization 1: Age distribution by survival...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

survived = df[df['Survival_Binary'] == 1]
died = df[df['Survival_Binary'] == 0]

ax.hist([survived['Age'], died['Age']], 
        label=['Survived (n={})'.format(len(survived)), 
               'Died (n={})'.format(len(died))],
        bins=20, alpha=0.7, color=['green', 'red'])
ax.set_xlabel('Age (years)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Age Distribution by Survival Status', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

median_survived = survived['Age'].median()
median_died = died['Age'].median()
ax.axvline(median_survived, color='green', linestyle='--', linewidth=2, 
           label=f'Median Survived: {median_survived:.1f}')
ax.axvline(median_died, color='red', linestyle='--', linewidth=2, 
           label=f'Median Died: {median_died:.1f}')
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('viz_01_age_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: viz_01_age_distribution.png")
plt.close()

# =============================================================================
# VISUALIZATION 2: SURVIVAL BY TUMOR STAGE
# =============================================================================

print("Creating Visualization 2: Survival by tumor stage...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

stage_survival = df.groupby('Tumour_Stage')['Survival_Binary'].agg(['mean', 'count'])
stage_survival = stage_survival.reindex(['I', 'II', 'III'])

x = np.arange(len(stage_survival))
bars = ax.bar(x, stage_survival['mean'] * 100, color=['green', 'orange', 'red'], 
              alpha=0.7, edgecolor='black', linewidth=1.5)

for i, (bar, val, count) in enumerate(zip(bars, stage_survival['mean'], stage_survival['count'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val*100:.1f}%\n(n={int(count)})',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Tumour Stage', fontsize=12)
ax.set_ylabel('Survival Rate (%)', fontsize=12)
ax.set_title('Survival Rate by Tumour Stage', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Stage I\n(Early)', 'Stage II\n(Intermediate)', 'Stage III\n(Advanced)'])
ax.set_ylim([0, 100])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('viz_02_survival_by_stage.png', dpi=300, bbox_inches='tight')
print("✅ Saved: viz_02_survival_by_stage.png")
plt.close()

# =============================================================================
# VISUALIZATION 3: PROTEIN EXPRESSION BY SURVIVAL
# =============================================================================

print("Creating Visualization 3: Protein expression patterns...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
proteins = ['Protein1', 'Protein2', 'Protein3', 'Protein4']

for idx, (ax, protein) in enumerate(zip(axes.flatten(), proteins)):
    survived_data = df[df['Survival_Binary'] == 1][protein]
    died_data = df[df['Survival_Binary'] == 0][protein]
    
    ax.hist([survived_data, died_data], 
            label=['Survived', 'Died'],
            bins=25, alpha=0.6, color=['green', 'red'])
    
    ax.set_xlabel(f'{protein} Expression Level', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{protein} Distribution by Survival', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    med_surv = survived_data.median()
    med_died = died_data.median()
    ax.axvline(med_surv, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(med_died, color='red', linestyle='--', linewidth=2, alpha=0.7)

plt.tight_layout()
plt.savefig('viz_03_protein_expression.png', dpi=300, bbox_inches='tight')
print("✅ Saved: viz_03_protein_expression.png")
plt.close()

# =============================================================================
# NEW VISUALIZATION 4: SURVIVAL TRENDS BY YEAR
# =============================================================================

print("Creating Visualization 4: Survival trends by surgery year...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Survival rate by year
survival_by_year = df.groupby('Surgery_Year').agg({
    'Survival_Binary': ['mean', 'count']
}).round(4)
survival_by_year.columns = ['survival_rate', 'patient_count']
survival_by_year['survival_pct'] = survival_by_year['survival_rate'] * 100

years = survival_by_year.index
bars = ax1.bar(years, survival_by_year['survival_pct'], 
              color=['steelblue', 'coral', 'mediumseagreen'], 
              alpha=0.7, edgecolor='black', linewidth=1.5, width=0.6)

for bar, year in zip(bars, years):
    height = bar.get_height()
    count = int(survival_by_year.loc[year, 'patient_count'])
    rate = survival_by_year.loc[year, 'survival_pct']
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{rate:.1f}%\n(n={count})',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add overall mean line
overall_mean = df['Survival_Binary'].mean() * 100
ax1.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2,
           label=f'Overall Mean: {overall_mean:.1f}%', zorder=3)

ax1.set_xlabel('Surgery Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Survival Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Survival Rate by Surgery Year', fontsize=13, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticks(years)

# Panel 2: Death count by year
death_by_year = df.groupby('Surgery_Year')['Survival_Binary'].apply(
    lambda x: (x == 0).sum()
)

bars2 = ax2.bar(years, death_by_year.values, 
               color=['darkred', 'firebrick', 'crimson'], 
               alpha=0.7, edgecolor='black', linewidth=1.5, width=0.6)

for bar, year in zip(bars2, years):
    height = bar.get_height()
    total = int(survival_by_year.loc[year, 'patient_count'])
    deaths = int(death_by_year[year])
    death_pct = (deaths / total * 100) if total > 0 else 0
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{deaths}\n({death_pct:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_xlabel('Surgery Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Deaths', fontsize=12, fontweight='bold')
ax2.set_title('Death Count by Surgery Year', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(years)

plt.tight_layout()
plt.savefig('viz_04_survival_trends_by_year.png', dpi=300, bbox_inches='tight')
print("✅ Saved: viz_04_survival_trends_by_year.png")
plt.close()

# =============================================================================
# NEW VISUALIZATION 5: HER2 STATUS BY YEAR AND SURVIVAL
# =============================================================================

print("Creating Visualization 5: HER2 status analysis by year...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: HER2 distribution by year
ax1 = axes[0, 0]
her2_by_year = pd.crosstab(df['Surgery_Year'], df['HER2 status'], normalize='index') * 100

x = np.arange(len(her2_by_year.index))
width = 0.35

bars1 = ax1.bar(x - width/2, her2_by_year['Negative'], width, 
               label='HER2 Negative', color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax1.bar(x + width/2, her2_by_year['Positive'], width,
               label='HER2 Positive', color='coral', alpha=0.7, edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Surgery Year', fontsize=11, fontweight='bold')
ax1.set_ylabel('Percentage of Patients (%)', fontsize=11, fontweight='bold')
ax1.set_title('HER2 Status Distribution by Year', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(her2_by_year.index.astype(int))
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0, 100])

# Panel 2: Survival by HER2 status and year
ax2 = axes[0, 1]

her2_survival = df.groupby(['Surgery_Year', 'HER2 status'])['Survival_Binary'].mean() * 100
her2_survival_df = her2_survival.unstack()

x = np.arange(len(her2_survival_df.index))
bars1 = ax2.bar(x - width/2, her2_survival_df['Negative'], width,
               label='HER2 Negative', color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax2.bar(x + width/2, her2_survival_df['Positive'], width,
               label='HER2 Positive', color='coral', alpha=0.7, edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Surgery Year', fontsize=11, fontweight='bold')
ax2.set_ylabel('Survival Rate (%)', fontsize=11, fontweight='bold')
ax2.set_title('Survival Rate by HER2 Status and Year', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(her2_survival_df.index.astype(int))
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 100])

# Panel 3: Patient count by HER2 and year
ax3 = axes[1, 0]

her2_count = pd.crosstab(df['Surgery_Year'], df['HER2 status'])

x = np.arange(len(her2_count.index))
bars1 = ax3.bar(x - width/2, her2_count['Negative'], width,
               label='HER2 Negative', color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax3.bar(x + width/2, her2_count['Positive'], width,
               label='HER2 Positive', color='coral', alpha=0.7, edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax3.set_xlabel('Surgery Year', fontsize=11, fontweight='bold')
ax3.set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
ax3.set_title('Patient Count by HER2 Status and Year', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(her2_count.index.astype(int))
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: HER2+ patients survival comparison across years
ax4 = axes[1, 1]

# Overall HER2 survival (combining all years)
overall_her2_survival = df.groupby('HER2 status')['Survival_Binary'].agg(['mean', 'count'])
overall_her2_survival['survival_pct'] = overall_her2_survival['mean'] * 100

x_pos = [0, 1]
bars = ax4.bar(x_pos, overall_her2_survival['survival_pct'].values,
              color=['steelblue', 'coral'], alpha=0.7, edgecolor='black', linewidth=1.5)

for bar, her2_status in zip(bars, overall_her2_survival.index):
    height = bar.get_height()
    count = int(overall_her2_survival.loc[her2_status, 'count'])
    rate = overall_her2_survival.loc[her2_status, 'survival_pct']
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{rate:.1f}%\n(n={count})',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4.set_xlabel('HER2 Status', fontsize=11, fontweight='bold')
ax4.set_ylabel('Survival Rate (%)', fontsize=11, fontweight='bold')
ax4.set_title('Overall Survival Rate by HER2 Status\n(2017-2019 Combined)', 
             fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(overall_her2_survival.index)
ax4.set_ylim([0, 100])
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('viz_05_her2_analysis_by_year.png', dpi=300, bbox_inches='tight')
print("✅ Saved: viz_05_her2_analysis_by_year.png")
plt.close()

# =============================================================================
# VISUALIZATION 6: CORRELATION HEATMAP (UPDATED WITH YEAR)
# =============================================================================

print("Creating Visualization 6: Enhanced correlation heatmap...")

# Select numeric features including year
correlation_features = all_features + ['Survival_Binary']
correlation_data = df[correlation_features].copy()

# Verify all columns are numeric
non_numeric_cols = correlation_data.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"⚠️  Warning: Removing non-numeric columns: {non_numeric_cols}")
    correlation_data = correlation_data.select_dtypes(include=[np.number])

# Calculate correlation matrix
correlation_matrix = correlation_data.corr()

# Extract correlations with survival
survival_correlations = correlation_matrix['Survival_Binary'].drop('Survival_Binary').sort_values(ascending=False)

# Create figure with two panels
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Panel 1: Full correlation heatmap
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            ax=axes[0], annot_kws={'fontsize': 7})
axes[0].set_title('Feature Correlation Matrix\n(All Features + Surgery Year)', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Features', fontsize=12)
axes[0].set_ylabel('Features', fontsize=12)

# Panel 2: Survival correlation bar chart
survival_corr_df = pd.DataFrame({
    'Feature': survival_correlations.index,
    'Correlation': survival_correlations.values
})

colors = ['green' if x > 0 else 'red' for x in survival_corr_df['Correlation']]
axes[1].barh(survival_corr_df['Feature'], survival_corr_df['Correlation'], 
             color=colors, alpha=0.7)
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[1].set_xlabel('Correlation with Survival Probability', fontsize=12)
axes[1].set_title('Feature Correlation with Survival\n(Positive = Higher Survival)', 
                  fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].invert_yaxis()

# Highlight Surgery_Year if present
if 'Surgery_Year' in survival_corr_df['Feature'].values:
    year_idx = survival_corr_df[survival_corr_df['Feature'] == 'Surgery_Year'].index[0]
    axes[1].get_children()[year_idx].set_edgecolor('blue')
    axes[1].get_children()[year_idx].set_linewidth(2)

plt.tight_layout()
plt.savefig('viz_06_correlation_heatmap_with_year.png', dpi=300, bbox_inches='tight')
print("✅ Saved: viz_06_correlation_heatmap_with_year.png")
plt.close()

# =============================================================================
# PRINT STATISTICS
# =============================================================================

print("\n" + "="*80)
print("CORRELATION ANALYSIS RESULTS")
print("="*80)

print("\nTop 5 features POSITIVELY correlated with survival:")
for i, (feat, corr) in enumerate(survival_correlations.head().items(), 1):
    print(f"{i}. {feat:40s}: {corr:+.4f}")

print("\nTop 5 features NEGATIVELY correlated with survival:")
for i, (feat, corr) in enumerate(survival_correlations.tail().items(), 1):
    print(f"{i}. {feat:40s}: {corr:+.4f}")

# Year-specific analysis
print("\n" + "="*80)
print("TEMPORAL ANALYSIS (BY SURGERY YEAR)")
print("="*80)

print("\nSurvival Rate by Year:")
for year in sorted(df['Surgery_Year'].unique()):
    year_data = df[df['Surgery_Year'] == year]
    survival_rate = year_data['Survival_Binary'].mean() * 100
    n_patients = len(year_data)
    n_deaths = (year_data['Survival_Binary'] == 0).sum()
    print(f"  {int(year)}: {survival_rate:.2f}% survival ({n_patients} patients, {n_deaths} deaths)")

print("\nHER2+ Survival by Year:")
for year in sorted(df['Surgery_Year'].unique()):
    year_her2 = df[(df['Surgery_Year'] == year) & (df['HER2 status'] == 'Positive')]
    if len(year_her2) > 0:
        survival_rate = year_her2['Survival_Binary'].mean() * 100
        print(f"  {int(year)}: {survival_rate:.2f}% (n={len(year_her2)})")

print("\nHER2- Survival by Year:")
for year in sorted(df['Surgery_Year'].unique()):
    year_her2 = df[(df['Surgery_Year'] == year) & (df['HER2 status'] == 'Negative')]
    if len(year_her2) > 0:
        survival_rate = year_her2['Survival_Binary'].mean() * 100
        print(f"  {int(year)}: {survival_rate:.2f}% (n={len(year_her2)})")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  1. viz_01_age_distribution.png")
print("  2. viz_02_survival_by_stage.png")
print("  3. viz_03_protein_expression.png")
print("  4. viz_04_survival_trends_by_year.png         [NEW!]")
print("  5. viz_05_her2_analysis_by_year.png           [NEW!]")
print("  6. viz_06_correlation_heatmap_with_year.png   [UPDATED!]")
print(f"  7. brca_encoded.csv (saved for modeling)")
print("\n✅ All visualizations saved successfully!")
print("\n📊 New temporal analysis complete - includes year trends and HER2 by year!")
