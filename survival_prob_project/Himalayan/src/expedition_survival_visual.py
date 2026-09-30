"""
Data Visualization for Mountaineering Expedition Survival Analysis
Analyzing survival rates, technology impact, and socioeconomic factors
Each visualization saved as a separate file + combined analysis
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
DATA_PATH = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/Himalayan/data/exped.csv'
OUTPUT_DIR = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/Himalayan/visualization'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading expedition data...")
exped = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(exped)} expeditions")

# Handle missing values in key columns
exped['year'] = pd.to_numeric(exped['year'], errors='coerce')

# Handle season - may be full names or abbreviations
exped['season'] = exped['season'].fillna('Unknown')
# Map abbreviated seasons to full names if needed
season_map = {'Spr': 'Spring', 'Sum': 'Summer', 'Aut': 'Autumn', 'Win': 'Winter'}
exped['season'] = exped['season'].replace(season_map)

exped['mdeaths'] = pd.to_numeric(exped['mdeaths'], errors='coerce').fillna(0)
exped['hdeaths'] = pd.to_numeric(exped['hdeaths'], errors='coerce').fillna(0)
exped['totmembers'] = pd.to_numeric(exped['totmembers'], errors='coerce').fillna(0)
exped['tothired'] = pd.to_numeric(exped['tothired'], errors='coerce').fillna(0)

# Create time period categories
exped['time_period'] = pd.cut(exped['year'], 
                               bins=[1900, 1960, 1980, 2000, 2025],
                               labels=['Early Era (1905-1960)', 'Modern Era (1961-1980)', 
                                      'Contemporary (1981-2000)', 'Recent (2001-2024)'])

# Calculate survival metrics
exped['total_deaths'] = exped['mdeaths'] + exped['hdeaths']
exped['total_participants'] = exped['totmembers'] + exped['tothired']
exped['had_deaths'] = (exped['total_deaths'] > 0).astype(int)

# Clean oxygen usage data - o2used column
exped['oxygen_used'] = (exped['o2used'] == 'Y').astype(int)

# Expedition size categories
exped['size_category'] = pd.cut(exped['total_participants'], 
                                bins=[0, 5, 10, 20, 50, 1000],
                                labels=['Small (1-5)', 'Medium (6-10)', 'Large (11-20)', 
                                       'Very Large (21-50)', 'Massive (50+)'])

print("Generating visualizations...")

# ============================================================================
# SEPARATE VISUALIZATIONS
# ============================================================================

# 1. Survival Rate Over Time
print("1. Creating survival rate trends visualization...")
fig, ax = plt.subplots(figsize=(14, 8))
survival_by_decade = exped.groupby(exped['year'] // 10 * 10).agg({
    'total_deaths': 'sum',
    'total_participants': 'sum'
})
survival_by_decade = survival_by_decade[survival_by_decade['total_participants'] > 0]
survival_by_decade['survival_rate'] = (1 - survival_by_decade['total_deaths'] / survival_by_decade['total_participants']) * 100
ax.plot(survival_by_decade.index, survival_by_decade['survival_rate'], 
        marker='o', linewidth=3, markersize=10, color='#2E8B57')
ax.set_xlabel('Decade', fontsize=12, fontweight='bold')
ax.set_ylabel('Survival Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Survival Rate Trends Over Time (1900s-2020s)', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.set_ylim([97, 100])
for x, y in zip(survival_by_decade.index, survival_by_decade['survival_rate']):
    ax.text(x, y - 0.15, f'{y:.2f}%', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'survival_rate_trends.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: survival_rate_trends.png")

# 2. Oxygen Usage Impact on Survival
print("2. Creating oxygen usage impact visualization...")
fig, ax = plt.subplots(figsize=(12, 8))
oxygen_survival = exped.groupby('oxygen_used').agg({
    'total_deaths': 'sum',
    'total_participants': 'sum'
})
oxygen_survival = oxygen_survival[oxygen_survival['total_participants'] > 0]
oxygen_survival['survival_rate'] = (1 - oxygen_survival['total_deaths'] / oxygen_survival['total_participants']) * 100
oxygen_survival['death_rate'] = (oxygen_survival['total_deaths'] / oxygen_survival['total_participants']) * 100
colors_o2 = ['#FF6B6B', '#4ECDC4']
bars = ax.bar(range(len(oxygen_survival)), oxygen_survival['survival_rate'], 
              color=colors_o2, edgecolor='black', linewidth=2)
ax.set_xticks(range(len(oxygen_survival)))
ax.set_xticklabels(['No Oxygen', 'Oxygen Used'], fontsize=12, fontweight='bold')
ax.set_ylabel('Survival Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Survival Rate: Oxygen vs No Oxygen\n(Technology Impact)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim([98, 100])
ax.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, oxygen_survival['survival_rate'])):
    ax.text(bar.get_x() + bar.get_width()/2., val + 0.05,
            f'{val:.2f}%', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'oxygen_impact_survival.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: oxygen_impact_survival.png")

# 3. Member vs Hired Staff Death Rates (Socioeconomic Structure)
print("3. Creating member vs hired staff comparison...")
fig, ax = plt.subplots(figsize=(12, 8))
total_members = exped['totmembers'].sum()
total_hired = exped['tothired'].sum()
member_deaths = exped['mdeaths'].sum()
hired_deaths = exped['hdeaths'].sum()
member_death_rate = (member_deaths / total_members) * 100 if total_members > 0 else 0
hired_death_rate = (hired_deaths / total_hired) * 100 if total_hired > 0 else 0
categories = ['Expedition Members', 'Hired Staff']
death_rates = [member_death_rate, hired_death_rate]
colors_staff = ['#FF8C42', '#8B4C9B']
bars = ax.bar(range(len(categories)), death_rates, color=colors_staff, 
              edgecolor='black', linewidth=2)
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylabel('Death Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Death Rates: Expedition Members vs Hired Staff\n(Socioeconomic Factors)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
for bar, val, cat in zip(bars, death_rates, categories):
    deaths = member_deaths if cat == 'Expedition Members' else hired_deaths
    total = total_members if cat == 'Expedition Members' else total_hired
    ax.text(bar.get_x() + bar.get_width()/2., val + 0.05,
            f'{val:.2f}%\n({int(deaths)}/{int(total)})',
            ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'member_vs_hired_deaths.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: member_vs_hired_deaths.png")

# 4. Season Impact on Survival
print("4. Creating season impact visualization...")
fig, ax = plt.subplots(figsize=(12, 8))
season_survival = exped[exped['season'] != 'Unknown'].groupby('season').agg({
    'total_deaths': 'sum',
    'total_participants': 'sum'
})
season_survival = season_survival[season_survival['total_participants'] > 0]
season_survival['survival_rate'] = (1 - season_survival['total_deaths'] / season_survival['total_participants']) * 100
season_survival = season_survival.sort_values('survival_rate', ascending=False)
colors_season = ['#90EE90', '#FFD700', '#FF8C00', '#87CEEB']
bars = ax.bar(range(len(season_survival)), season_survival['survival_rate'], 
              color=colors_season[:len(season_survival)], edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(season_survival)))
ax.set_xticklabels(season_survival.index, fontsize=11, fontweight='bold')
ax.set_ylabel('Survival Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Survival Rate by Season', fontsize=13, fontweight='bold', pad=15)
ax.set_ylim([98, 100])
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, season_survival['survival_rate']):
    ax.text(bar.get_x() + bar.get_width()/2., val + 0.05,
            f'{val:.2f}%', ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'season_impact_survival.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: season_impact_survival.png")

# 5. Success Rate vs Survival Rate
print("5. Creating success vs survival visualization...")
fig, ax = plt.subplots(figsize=(12, 8))
period_stats = exped.groupby('time_period', observed=True).agg({
    'termreason': lambda x: (x == 'Success (main peak)').sum(),
    'total_deaths': 'sum',
    'total_participants': 'sum',
    'expid': 'count'
})
period_stats['success_rate'] = (period_stats['termreason'] / period_stats['expid']) * 100
mask = period_stats['total_participants'] > 0
period_stats.loc[mask, 'survival_rate'] = (1 - period_stats.loc[mask, 'total_deaths'] / period_stats.loc[mask, 'total_participants']) * 100
period_stats = period_stats[mask]

x = np.arange(len(period_stats))
width = 0.35
bars1 = ax.bar(x - width/2, period_stats['success_rate'], width, label='Success Rate',
               color='#4ECDC4', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, period_stats['survival_rate'], width, label='Survival Rate',
               color='#FF6B6B', edgecolor='black', linewidth=1.2)
ax.set_xlabel('Time Period', fontsize=11, fontweight='bold')
ax.set_ylabel('Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Success Rate vs Survival Rate Across Time Periods', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(period_stats.index, fontsize=9, rotation=45, ha='right')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'success_vs_survival.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: success_vs_survival.png")

# 6. Expedition Size Impact
print("6. Creating expedition size impact visualization...")
fig, ax = plt.subplots(figsize=(14, 8))
size_survival = exped.groupby('size_category', observed=True).agg({
    'total_deaths': 'sum',
    'total_participants': 'sum',
    'expid': 'count'
})
size_survival = size_survival[size_survival['total_participants'] > 0]
size_survival['survival_rate'] = (1 - size_survival['total_deaths'] / size_survival['total_participants']) * 100
size_survival['death_rate'] = (size_survival['total_deaths'] / size_survival['total_participants']) * 100
colors_size = plt.cm.viridis(np.linspace(0.2, 0.9, len(size_survival)))
bars = ax.bar(range(len(size_survival)), size_survival['survival_rate'], 
              color=colors_size, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(size_survival)))
ax.set_xticklabels(size_survival.index, fontsize=10, fontweight='bold', rotation=30, ha='right')
ax.set_ylabel('Survival Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Survival Rate by Expedition Size', fontsize=13, fontweight='bold', pad=15)
ax.set_ylim([98, 100])
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, size_survival['survival_rate']):
    ax.text(bar.get_x() + bar.get_width()/2., val + 0.05,
            f'{val:.2f}%', ha='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'expedition_size_impact.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: expedition_size_impact.png")

# 7. Deaths Timeline
print("7. Creating deaths timeline visualization...")
fig, ax = plt.subplots(figsize=(14, 8))
deaths_by_year = exped.groupby('year')['total_deaths'].sum()
deaths_by_year = deaths_by_year[deaths_by_year.index.notna()]
ax.fill_between(deaths_by_year.index, deaths_by_year.values, alpha=0.3, color='#FF6B6B')
ax.plot(deaths_by_year.index, deaths_by_year.values, linewidth=2, color='#C44536', marker='o', markersize=3)
ax.set_xlabel('Year', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Deaths', fontsize=11, fontweight='bold')
ax.set_title('Mountaineering Deaths Over Time (1905-2024)', fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)
ax.axvspan(1905, 1960, alpha=0.1, color='#8B0000', label='Early Era')
ax.axvspan(1961, 1980, alpha=0.1, color='#DC143C', label='Modern Era')
ax.axvspan(1981, 2000, alpha=0.1, color='#FF6347', label='Contemporary')
ax.axvspan(2001, 2024, alpha=0.1, color='#FFA07A', label='Recent')
ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'deaths_timeline.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: deaths_timeline.png")

# ============================================================================
# COMBINED COMPREHENSIVE VISUALIZATION (continued in next section due to length)
# ============================================================================

print("8. Creating combined comprehensive visualization...")
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. Survival Rate Over Time
ax1 = fig.add_subplot(gs[0, :])
survival_by_decade = exped.groupby(exped['year'] // 10 * 10).agg({
    'total_deaths': 'sum',
    'total_participants': 'sum'
})
survival_by_decade = survival_by_decade[survival_by_decade['total_participants'] > 0]
survival_by_decade['survival_rate'] = (1 - survival_by_decade['total_deaths'] / survival_by_decade['total_participants']) * 100
ax1.plot(survival_by_decade.index, survival_by_decade['survival_rate'], 
        marker='o', linewidth=2.5, markersize=8, color='#2E8B57')
ax1.set_xlabel('Decade', fontsize=10, fontweight='bold')
ax1.set_ylabel('Survival Rate (%)', fontsize=10, fontweight='bold')
ax1.set_title('Survival Rate Trends Over Time', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([97, 100])
for x, y in zip(survival_by_decade.index, survival_by_decade['survival_rate']):
    ax1.text(x, y - 0.15, f'{y:.2f}%', ha='center', fontsize=8, fontweight='bold')

# 2. Oxygen Impact
ax2 = fig.add_subplot(gs[1, 0])
oxygen_survival = exped.groupby('oxygen_used').agg({
    'total_deaths': 'sum',
    'total_participants': 'sum'
})
oxygen_survival = oxygen_survival[oxygen_survival['total_participants'] > 0]
oxygen_survival['survival_rate'] = (1 - oxygen_survival['total_deaths'] / oxygen_survival['total_participants']) * 100
colors_o2 = ['#FF6B6B', '#4ECDC4']
bars = ax2.bar(range(len(oxygen_survival)), oxygen_survival['survival_rate'], 
              color=colors_o2, edgecolor='black', linewidth=1.5)
ax2.set_xticks(range(len(oxygen_survival)))
ax2.set_xticklabels(['No O2', 'O2 Used'], fontsize=9, fontweight='bold')
ax2.set_ylabel('Survival %', fontsize=9, fontweight='bold')
ax2.set_title('Oxygen Impact', fontsize=10, fontweight='bold')
ax2.set_ylim([98, 100])
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, oxygen_survival['survival_rate']):
    ax2.text(bar.get_x() + bar.get_width()/2., val + 0.05,
            f'{val:.2f}%', ha='center', fontweight='bold', fontsize=8)

# 3. Member vs Hired
ax3 = fig.add_subplot(gs[1, 1])
categories = ['Members', 'Hired']
death_rates = [member_death_rate, hired_death_rate]
colors_staff = ['#FF8C42', '#8B4C9B']
bars = ax3.bar(range(len(categories)), death_rates, color=colors_staff, 
              edgecolor='black', linewidth=1.5)
ax3.set_xticks(range(len(categories)))
ax3.set_xticklabels(categories, fontsize=9, fontweight='bold')
ax3.set_ylabel('Death Rate %', fontsize=9, fontweight='bold')
ax3.set_title('Member vs Hired Deaths', fontsize=10, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, death_rates):
    ax3.text(bar.get_x() + bar.get_width()/2., val + 0.05,
            f'{val:.2f}%', ha='center', fontweight='bold', fontsize=8)

# 4. Season Impact
ax4 = fig.add_subplot(gs[1, 2])
season_survival = exped[exped['season'] != 'Unknown'].groupby('season').agg({
    'total_deaths': 'sum',
    'total_participants': 'sum'
})
season_survival = season_survival[season_survival['total_participants'] > 0]
season_survival['survival_rate'] = (1 - season_survival['total_deaths'] / season_survival['total_participants']) * 100
season_survival = season_survival.sort_values('survival_rate', ascending=False)
colors_season = ['#90EE90', '#FFD700', '#FF8C00', '#87CEEB']
bars = ax4.bar(range(len(season_survival)), season_survival['survival_rate'], 
              color=colors_season[:len(season_survival)], edgecolor='black', linewidth=1)
ax4.set_xticks(range(len(season_survival)))
ax4.set_xticklabels(season_survival.index, fontsize=8)
ax4.set_ylabel('Survival %', fontsize=9, fontweight='bold')
ax4.set_title('Season Impact', fontsize=10, fontweight='bold')
ax4.set_ylim([98, 100])
ax4.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, season_survival['survival_rate']):
    ax4.text(bar.get_x() + bar.get_width()/2., val + 0.05,
            f'{val:.1f}%', ha='center', fontweight='bold', fontsize=7)

# 5. Success vs Survival by Period
ax5 = fig.add_subplot(gs[2, 0])
period_stats = exped.groupby('time_period', observed=True).agg({
    'termreason': lambda x: (x == 'Success (main peak)').sum(),
    'total_deaths': 'sum',
    'total_participants': 'sum',
    'expid': 'count'
})
period_stats['success_rate'] = (period_stats['termreason'] / period_stats['expid']) * 100
mask = period_stats['total_participants'] > 0
period_stats.loc[mask, 'survival_rate'] = (1 - period_stats.loc[mask, 'total_deaths'] / period_stats.loc[mask, 'total_participants']) * 100
period_stats = period_stats[mask]
x = np.arange(len(period_stats))
width = 0.35
bars1 = ax5.bar(x - width/2, period_stats['success_rate'], width, label='Success',
               color='#4ECDC4', edgecolor='black', linewidth=0.8)
bars2 = ax5.bar(x + width/2, period_stats['survival_rate'], width, label='Survival',
               color='#FF6B6B', edgecolor='black', linewidth=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels([str(i).split('(')[0].strip() for i in period_stats.index], fontsize=7, rotation=45, ha='right')
ax5.set_ylabel('Rate %', fontsize=9, fontweight='bold')
ax5.set_title('Success vs Survival by Period', fontsize=10, fontweight='bold')
ax5.legend(fontsize=7)
ax5.grid(axis='y', alpha=0.3)

# 6. Expedition Size Impact
ax6 = fig.add_subplot(gs[2, 1])
size_survival = exped.groupby('size_category', observed=True).agg({
    'total_deaths': 'sum',
    'total_participants': 'sum'
})
size_survival = size_survival[size_survival['total_participants'] > 0]
size_survival['survival_rate'] = (1 - size_survival['total_deaths'] / size_survival['total_participants']) * 100
colors_size = plt.cm.viridis(np.linspace(0.2, 0.9, len(size_survival)))
bars = ax6.bar(range(len(size_survival)), size_survival['survival_rate'], 
              color=colors_size, edgecolor='black', linewidth=1)
ax6.set_xticks(range(len(size_survival)))
ax6.set_xticklabels([str(i).split('(')[0] for i in size_survival.index], fontsize=7, rotation=30, ha='right')
ax6.set_ylabel('Survival %', fontsize=9, fontweight='bold')
ax6.set_title('Size Impact', fontsize=10, fontweight='bold')
ax6.set_ylim([98, 100])
ax6.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, size_survival['survival_rate']):
    ax6.text(bar.get_x() + bar.get_width()/2., val + 0.05,
            f'{val:.1f}%', ha='center', fontweight='bold', fontsize=7)

# 7. Deaths Timeline with periods
ax7 = fig.add_subplot(gs[2, 2])
deaths_by_year_small = exped.groupby('year')['total_deaths'].sum()
deaths_by_year_small = deaths_by_year_small[deaths_by_year_small.index.notna()]
deaths_sampled = deaths_by_year_small[::5]
ax7.plot(deaths_sampled.index, deaths_sampled.values, linewidth=1.5, color='#C44536', marker='o', markersize=3)
ax7.fill_between(deaths_sampled.index, deaths_sampled.values, alpha=0.2, color='#FF6B6B')
ax7.set_xlabel('Year', fontsize=9, fontweight='bold')
ax7.set_ylabel('Deaths', fontsize=9, fontweight='bold')
ax7.set_title('Deaths Timeline', fontsize=10, fontweight='bold')
ax7.grid(True, alpha=0.3)

plt.suptitle('Comprehensive Expedition Survival Analysis (1905-2024)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig(os.path.join(OUTPUT_DIR, 'expedition_comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: expedition_comprehensive_analysis.png")

# Print key statistics
print("\n" + "="*60)
print("KEY STATISTICS FROM EXPEDITION DATASET")
print("="*60)
print(f"Total Expeditions: {len(exped):,}")
print(f"Expeditions with Deaths: {exped['had_deaths'].sum():,} ({(exped['had_deaths'].sum()/len(exped)*100):.2f}%)")
print(f"Total Deaths: {exped['total_deaths'].sum():,}")
print(f"  - Member Deaths: {exped['mdeaths'].sum():,}")
print(f"  - Hired Staff Deaths: {exped['hdeaths'].sum():,}")
print(f"\nTotal Participants: {exped['total_participants'].sum():,.0f}")
print(f"  - Members: {exped['totmembers'].sum():,.0f}")
print(f"  - Hired Staff: {exped['tothired'].sum():,.0f}")
overall_survival = (1 - exped['total_deaths'].sum()/exped['total_participants'].sum())*100
print(f"\nOverall Survival Rate: {overall_survival:.4f}%")
print(f"\nExpeditions Using Oxygen: {exped['oxygen_used'].sum():,} ({(exped['oxygen_used'].sum()/len(exped)*100):.2f}%)")
print(f"\nTime Period Distribution:")
print(exped['time_period'].value_counts().sort_index())
print("\n" + "="*60)
print("All visualizations saved to:")
print(OUTPUT_DIR)
print("="*60)
