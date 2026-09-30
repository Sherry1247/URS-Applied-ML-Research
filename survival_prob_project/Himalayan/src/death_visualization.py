"""
Data Visualization for Mountaineering Deaths Dataset
Analyzing how technology, time period, and socioeconomic factors affect survival
Each visualization saved as a separate file + combined analysis
CORRECTED - Handles abbreviated season format (Spr, Aut, Win, Sum)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Define paths
DATA_PATH = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/Himalayan/data/deaths.csv'
OUTPUT_DIR = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/Himalayan/visualization'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading deaths data...")
deaths = pd.read_csv(DATA_PATH)
print(f"Loaded {len(deaths)} death records")

# Extract year
deaths['year'] = deaths['yr_season'].astype(str).str.extract(r'(\d{4})').astype(int)

# Extract ABBREVIATED season (Spr, Aut, Win, Sum) - no case sensitivity needed
deaths['season_abbr'] = deaths['yr_season'].astype(str).str.extract(r'(Spr|Aut|Win|Sum)')[0]

# Map to full season names
season_map = {
    'Spr': 'Spring',
    'Sum': 'Summer',
    'Aut': 'Autumn',
    'Win': 'Winter'
}
deaths['season'] = deaths['season_abbr'].map(season_map)

print(f"\nSeason extraction successful!")
print(f"Season distribution:")
print(deaths['season'].value_counts())

# Create time period categories
deaths['time_period'] = pd.cut(deaths['year'], 
                                bins=[1900, 1960, 1980, 2000, 2020],
                                labels=['Early Era (1905-1960)', 'Modern Era (1961-1980)', 
                                       'Contemporary (1981-2000)', 'Recent (2001-2019)'])

# Clean oxygen usage data
deaths['oxygen_used'] = deaths['is_o2_used'].map({'Y': 'Yes', 'N': 'No', 'No': 'No'})
deaths['oxygen_used'] = deaths['oxygen_used'].fillna('Unknown')

print("\nGenerating visualizations...")

# ============================================================================
# SEPARATE VISUALIZATIONS
# ============================================================================

# 1. Deaths by Time Period
print("1. Creating time period visualization...")
fig, ax = plt.subplots(figsize=(12, 8))
deaths_by_period = deaths.groupby('time_period', observed=True).size()
colors_period = ['#8B0000', '#DC143C', '#FF6347', '#FFA07A']
bars = ax.bar(range(len(deaths_by_period)), deaths_by_period.values, color=colors_period, 
              edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(deaths_by_period)))
ax.set_xticklabels(deaths_by_period.index, fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Deaths', fontsize=12, fontweight='bold')
ax.set_title('Deaths by Time Period: Technological Evolution in Mountaineering', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'deaths_by_time_period.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: deaths_by_time_period.png")

# 2. Oxygen Usage Over Time
print("2. Creating oxygen usage trends visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

o2_by_period = deaths[deaths['oxygen_used'] != 'Unknown'].groupby(['time_period', 'oxygen_used'], observed=True).size().unstack(fill_value=0)
x = np.arange(len(o2_by_period))
width = 0.35
bars1 = ax1.bar(x - width/2, o2_by_period['No'], width, label='No Oxygen',
                color='#FF4444', edgecolor='black', linewidth=1.2)
bars2 = ax1.bar(x + width/2, o2_by_period['Yes'], width, label='Oxygen Used',
                color='#44AA44', edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Time Period', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Deaths', fontsize=11, fontweight='bold')
ax1.set_title('Deaths by Oxygen Usage Across Time Periods', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(o2_by_period.index, fontsize=9, rotation=45, ha='right')
ax1.legend(framealpha=0.9)
ax1.grid(axis='y', alpha=0.3)

o2_pct = o2_by_period.div(o2_by_period.sum(axis=1), axis=0) * 100
o2_pct.plot(kind='bar', stacked=True, ax=ax2, color=['#FF4444', '#44AA44'], 
            edgecolor='black', linewidth=1.2)
ax2.set_title('Oxygen Usage % Distribution by Period', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time Period', fontsize=11, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax2.legend(title='Oxygen Used', loc='upper left', framealpha=0.9)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Technology Adoption: Oxygen Usage Trends', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'oxygen_usage_trends.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: oxygen_usage_trends.png")

# 3. Deaths by Gender
print("3. Creating gender distribution visualization...")
fig, ax = plt.subplots(figsize=(10, 8))
gender_counts = deaths['gender'].value_counts()
colors_gender = ['#4169E1', '#FF69B4']
explode = (0.05, 0.05) if len(gender_counts) >= 2 else (0.05,)
wedges, texts, autotexts = ax.pie(gender_counts.values, labels=gender_counts.index, 
                                    autopct='%1.1f%%', colors=colors_gender[:len(gender_counts)], explode=explode,
                                    startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'},
                                    wedgeprops={'edgecolor': 'black', 'linewidth': 2})
ax.set_title(f'Deaths by Gender\nTotal Deaths: {len(deaths)}', 
             fontsize=13, fontweight='bold', pad=15)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'deaths_by_gender.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: deaths_by_gender.png")

# 4. Age Distribution at Death by Gender
print("4. Creating age distribution visualization...")
fig, ax = plt.subplots(figsize=(12, 8))
male_ages = deaths[deaths['gender'] == 'M']['age'].dropna()
female_ages = deaths[deaths['gender'] == 'F']['age'].dropna()
ax.hist([male_ages, female_ages], bins=15, label=['Male', 'Female'], 
        color=['#4169E1', '#FF69B4'], edgecolor='black', linewidth=1.2, alpha=0.7)
if len(male_ages) > 0:
    ax.axvline(male_ages.mean(), color='#4169E1', linestyle='--', linewidth=2.5, 
               label=f'Male Mean: {male_ages.mean():.1f}')
if len(female_ages) > 0:
    ax.axvline(female_ages.mean(), color='#FF69B4', linestyle='--', linewidth=2.5, 
               label=f'Female Mean: {female_ages.mean():.1f}')
ax.set_xlabel('Age at Death', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Deaths', fontsize=11, fontweight='bold')
ax.set_title('Age Distribution at Death by Gender', fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'age_distribution_by_gender.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: age_distribution_by_gender.png")

# 5. Deaths by Season - FIXED!
print("5. Creating seasonal distribution visualization...")
fig, ax = plt.subplots(figsize=(12, 8))
season_counts = deaths['season'].value_counts()
# Order by typical climbing season
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
season_counts = season_counts.reindex(season_order, fill_value=0)

colors_season = ['#90EE90', '#FFD700', '#FF8C00', '#87CEEB']
bars = ax.bar(range(len(season_counts)), season_counts.values, 
              color=colors_season, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(season_counts)))
ax.set_xticklabels(season_counts.index, fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Deaths', fontsize=11, fontweight='bold')
ax.set_title('Deaths by Season', fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'deaths_by_season.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: deaths_by_season.png")

# 6. Top 15 Nationalities
print("6. Creating nationality distribution visualization...")
fig, ax = plt.subplots(figsize=(12, 10))
top_nations = deaths['citizenship'].value_counts().head(15)
colors_nations = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_nations)))
bars = ax.barh(range(len(top_nations)), top_nations.values, color=colors_nations, 
               edgecolor='black', linewidth=1.2)
ax.set_yticks(range(len(top_nations)))
ax.set_yticklabels(top_nations.index, fontsize=10, fontweight='bold')
ax.set_xlabel('Number of Deaths', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Countries by Mountaineering Deaths\n(Socioeconomic Access)', 
             fontsize=13, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
for i, (bar, value) in enumerate(zip(bars, top_nations.values)):
    ax.text(value + 1, i, str(int(value)), va='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top_nationalities.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: top_nationalities.png")

# 7. Cause of Death Analysis
print("7. Creating cause of death visualization...")
fig, ax = plt.subplots(figsize=(14, 8))
deaths['cause_category'] = deaths['cause_of_death'].astype(str).str.extract(
    r'(Fall|Avalanche|AMS|Exposure|Crevasse|Illness|Disappearance)')[0]
deaths['cause_category'] = deaths['cause_category'].fillna('Other')
cause_counts = deaths['cause_category'].value_counts().head(10)
colors_cause = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(cause_counts)))
bars = ax.bar(range(len(cause_counts)), cause_counts.values, 
              color=colors_cause, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(cause_counts)))
ax.set_xticklabels(cause_counts.index, fontsize=11, fontweight='bold', rotation=45, ha='right')
ax.set_ylabel('Number of Deaths', fontsize=11, fontweight='bold')
ax.set_title('Primary Causes of Death in Mountaineering', fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'causes_of_death.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: causes_of_death.png")

# ============================================================================
# COMBINED COMPREHENSIVE VISUALIZATION
# ============================================================================

print("8. Creating combined comprehensive visualization...")
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. Deaths by Time Period
ax1 = fig.add_subplot(gs[0, 0])
deaths_by_period = deaths.groupby('time_period', observed=True).size()
colors_period = ['#8B0000', '#DC143C', '#FF6347', '#FFA07A']
bars = ax1.bar(range(len(deaths_by_period)), deaths_by_period.values, color=colors_period, 
              edgecolor='black', linewidth=1)
ax1.set_xticks(range(len(deaths_by_period)))
ax1.set_xticklabels(deaths_by_period.index, fontsize=8, rotation=45, ha='right')
ax1.set_ylabel('Deaths', fontsize=9, fontweight='bold')
ax1.set_title('Deaths by Time Period', fontsize=10, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 2. Oxygen Usage
ax2 = fig.add_subplot(gs[0, 1])
o2_counts = deaths[deaths['oxygen_used'] != 'Unknown']['oxygen_used'].value_counts()
colors_o2 = ['#FF4444', '#44AA44']
wedges, texts, autotexts = ax2.pie(o2_counts.values, labels=o2_counts.index, 
                                     autopct='%1.1f%%', colors=colors_o2,
                                     startangle=90, textprops={'fontsize': 9, 'fontweight': 'bold'},
                                     wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
ax2.set_title('Oxygen Usage Distribution', fontsize=10, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')

# 3. Gender Distribution
ax3 = fig.add_subplot(gs[0, 2])
gender_counts = deaths['gender'].value_counts()
colors_gender = ['#4169E1', '#FF69B4']
wedges, texts, autotexts = ax3.pie(gender_counts.values, labels=gender_counts.index, 
                                     autopct='%1.1f%%', colors=colors_gender[:len(gender_counts)],
                                     startangle=90, textprops={'fontsize': 9, 'fontweight': 'bold'},
                                     wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
ax3.set_title('Gender Distribution', fontsize=10, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')

# 4. Age Distribution
ax4 = fig.add_subplot(gs[1, 0])
male_ages = deaths[deaths['gender'] == 'M']['age'].dropna()
female_ages = deaths[deaths['gender'] == 'F']['age'].dropna()
ax4.hist([male_ages, female_ages], bins=12, label=['Male', 'Female'], 
        color=['#4169E1', '#FF69B4'], edgecolor='black', linewidth=0.8, alpha=0.7)
if len(male_ages) > 0:
    ax4.axvline(male_ages.mean(), color='#4169E1', linestyle='--', linewidth=2)
if len(female_ages) > 0:
    ax4.axvline(female_ages.mean(), color='#FF69B4', linestyle='--', linewidth=2)
ax4.set_xlabel('Age', fontsize=9, fontweight='bold')
ax4.set_ylabel('Count', fontsize=9, fontweight='bold')
ax4.set_title('Age Distribution by Gender', fontsize=10, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(axis='y', alpha=0.3)

# 5. Season Distribution
ax5 = fig.add_subplot(gs[1, 1])
season_counts = deaths['season'].value_counts()
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
season_counts = season_counts.reindex(season_order, fill_value=0)
colors_season = ['#90EE90', '#FFD700', '#FF8C00', '#87CEEB']
bars = ax5.bar(range(len(season_counts)), season_counts.values, 
              color=colors_season, edgecolor='black', linewidth=1)
ax5.set_xticks(range(len(season_counts)))
ax5.set_xticklabels(season_counts.index, fontsize=8)
ax5.set_ylabel('Deaths', fontsize=9, fontweight='bold')
ax5.set_title('Deaths by Season', fontsize=10, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 6. Top 10 Nationalities
ax6 = fig.add_subplot(gs[1, 2])
top_nations = deaths['citizenship'].value_counts().head(10)
colors_nations = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_nations)))
bars = ax6.barh(range(len(top_nations)), top_nations.values, color=colors_nations, 
               edgecolor='black', linewidth=0.8)
ax6.set_yticks(range(len(top_nations)))
ax6.set_yticklabels(top_nations.index, fontsize=8)
ax6.set_xlabel('Deaths', fontsize=9, fontweight='bold')
ax6.set_title('Top 10 Countries', fontsize=10, fontweight='bold')
ax6.invert_yaxis()
ax6.grid(axis='x', alpha=0.3)
for i, (bar, value) in enumerate(zip(bars, top_nations.values)):
    ax6.text(value + 0.5, i, str(int(value)), va='center', fontsize=8, fontweight='bold')

# 7. Causes of Death
ax7 = fig.add_subplot(gs[2, :])
cause_counts = deaths['cause_category'].value_counts().head(10)
colors_cause = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(cause_counts)))
bars = ax7.bar(range(len(cause_counts)), cause_counts.values, 
              color=colors_cause, edgecolor='black', linewidth=1.5)
ax7.set_xticks(range(len(cause_counts)))
ax7.set_xticklabels(cause_counts.index, fontsize=9, fontweight='bold', rotation=45, ha='right')
ax7.set_ylabel('Deaths', fontsize=9, fontweight='bold')
ax7.set_title('Primary Causes of Death', fontsize=10, fontweight='bold')
ax7.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Comprehensive Mountaineering Deaths Analysis (1905-2019)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig(os.path.join(OUTPUT_DIR, 'deaths_comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: deaths_comprehensive_analysis.png")

# Print key statistics
print("\n" + "="*60)
print("KEY STATISTICS FROM DEATHS DATASET")
print("="*60)
print(f"Total Deaths: {len(deaths)}")
avg_age = deaths['age'].mean()
print(f"Average Age at Death: {avg_age:.1f} years")
print(f"Age Range: {deaths['age'].min():.0f} - {deaths['age'].max():.0f} years")
print(f"\nGender Distribution:")
print(deaths['gender'].value_counts())
print(f"\nOxygen Usage:")
print(deaths['oxygen_used'].value_counts())
o2_known = deaths[deaths['oxygen_used'] != 'Unknown']
if len(o2_known) > 0:
    o2_yes_pct = (o2_known['oxygen_used'] == 'Yes').sum() / len(o2_known) * 100
    print(f"Oxygen usage rate: {o2_yes_pct:.1f}%")
print(f"\nSeason Distribution:")
print(deaths['season'].value_counts())
print(f"\nDeaths by Time Period:")
print(deaths['time_period'].value_counts().sort_index())
print("\n" + "="*60)
print("All visualizations saved to:")
print(OUTPUT_DIR)
print("="*60)
