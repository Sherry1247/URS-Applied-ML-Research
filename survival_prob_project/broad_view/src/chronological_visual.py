import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
FILE_PATH = "/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/broad_view/Annual cause death numbers new.csv"
OUTPUT_PATH = "/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/broad_view/visualization/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
print("="*80)
print("COMPREHENSIVE GLOBAL MORTALITY ANALYSIS")
print("SEPARATED GRAPHS VERSION - Each chart is standalone for clarity")
print("="*80)

print(f"\n📂 Loading data from: {FILE_PATH}")

try:
    df = pd.read_csv(FILE_PATH)
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: File not found at: {FILE_PATH}")
    exit()

# Clean column names
df.columns = [col.strip().replace(' fatalities', '').replace('\n', '').replace('\r', '') for col in df.columns]

EXCLUDE_COLS = ['Entity', 'Code', 'Year']
fatality_cols = [col for col in df.columns if col not in EXCLUDE_COLS]

for col in fatality_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=fatality_cols, inplace=True)

# =============================================================================
# 2. REGION MAPPING
# =============================================================================
print("\n🌍 Applying region mapping...")

aggregate_entities = [
    'African Region who', 'East Asia & Pacific wb', 'Eastern Mediterranean Region who',
    'Europe & Central Asia wb', 'European Region who', 'G20', 'Latin America & Caribbean wb',
    'Middle East & North Africa wb', 'North America wb', 'OECD Countries', 
    'Region of the Americas who', 'South Asia wb', 'SouthEast Asia Region who',
    'SubSaharan Africa wb', 'Western Pacific Region who', 'World',
    'World Bank High Income', 'World Bank Low Income', 'World Bank Lower Middle Income',
    'World Bank Upper Middle Income'
]

def get_region(country):
    sub_saharan_africa = [
        'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 
        'Cape Verde', 'Central African Republic', 'Chad', 'Comoros', 'Congo',
        'Cote dIvoire', 'Democratic Republic of Congo', 'Djibouti', 'Equatorial Guinea',
        'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea',
        'GuineaBissau', 'Kenya', 'Lesotho', 'Liberia', 'Madagascar', 'Malawi',
        'Mali', 'Mauritania', 'Mauritius', 'Mozambique', 'Namibia', 'Niger',
        'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles',
        'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania',
        'Togo', 'Uganda', 'Zambia', 'Zimbabwe'
    ]
    north_africa_middle_east = [
        'Algeria', 'Bahrain', 'Egypt', 'Iran', 'Iraq', 'Israel', 'Jordan',
        'Kuwait', 'Lebanon', 'Libya', 'Morocco', 'Oman', 'Palestine', 'Qatar',
        'Saudi Arabia', 'Syria', 'Tunisia', 'Turkey', 'United Arab Emirates', 'Yemen'
    ]
    south_asia = ['Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Maldives', 'Nepal', 'Pakistan', 'Sri Lanka']
    east_southeast_asia = [
        'Brunei', 'Cambodia', 'China', 'East Timor', 'Indonesia', 'Japan',
        'Laos', 'Malaysia', 'Mongolia', 'Myanmar', 'North Korea', 'Philippines',
        'Singapore', 'South Korea', 'Taiwan', 'Thailand', 'Vietnam'
    ]
    central_asia = ['Kazakhstan', 'Kyrgyzstan', 'Tajikistan', 'Turkmenistan', 'Uzbekistan']
    europe = [
        'Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus',
        'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus',
        'Czechia', 'Denmark', 'England', 'Estonia', 'Finland', 'France',
        'Georgia', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy',
        'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco',
        'Montenegro', 'Netherlands', 'North Macedonia', 'Northern Ireland',
        'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino',
        'Scotland', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden',
        'Switzerland', 'Ukraine', 'United Kingdom', 'Wales'
    ]
    north_america = ['Bermuda', 'Canada', 'Greenland', 'Mexico', 'United States', 'United States Virgin Islands', 'Puerto Rico']
    central_america_caribbean = [
        'Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Costa Rica',
        'Cuba', 'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada',
        'Guatemala', 'Guam', 'Haiti', 'Honduras', 'Jamaica', 'Nicaragua',
        'Northern Mariana Islands', 'Panama', 'Saint Kitts and Nevis',
        'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago'
    ]
    south_america = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela']
    oceania = [
        'American Samoa', 'Australia', 'Cook Islands', 'Fiji', 'Kiribati',
        'Marshall Islands', 'Micronesiacountry', 'Nauru', 'New Zealand', 'Niue',
        'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tokelau',
        'Tonga', 'Tuvalu', 'Vanuatu'
    ]
    
    if country in sub_saharan_africa: return 'Sub-Saharan Africa'
    elif country in north_africa_middle_east: return 'North Africa & Middle East'
    elif country in south_asia: return 'South Asia'
    elif country in east_southeast_asia: return 'East & Southeast Asia'
    elif country in central_asia: return 'Central Asia'
    elif country in europe: return 'Europe'
    elif country in north_america: return 'North America'
    elif country in central_america_caribbean: return 'Central America & Caribbean'
    elif country in south_america: return 'South America'
    elif country in oceania: return 'Oceania'
    else: return 'Other'

df_countries = df[~df['Entity'].isin(aggregate_entities)].copy()
df_countries['Region'] = df_countries['Entity'].apply(get_region)
df_countries = df_countries[df_countries['Region'] != 'Other']

print(f"✅ {df_countries['Entity'].nunique()} countries mapped to {df_countries['Region'].nunique()} regions")

# =============================================================================
# 3. DISEASE CLASSIFICATIONS
# =============================================================================
print("\n📊 Creating disease classifications...")

infectious_cols = ['Meningitis', 'Malaria', 'HIV/AIDS', 'Tuberculosis', 'Diarrheal disease', 'Acute hepatitis', 'Measles', 'Lower respiratory']
ncd_cardiovascular = ['Cardiovascular']
ncd_metabolic = ['Diabetes', 'Chronic kidney']
ncd_cancer = ['Neoplasm']
ncd_neurological = ['Dementia', 'Parkinson s']
ncd_respiratory = ['Chronic respiratory']
ncd_digestive = ['Digestive disease', 'Chronic liver']
injuries_violence = ['Interpersonal violence', 'Self harm', 'Conflict']
injuries_accidents = ['Drowning', 'Road injury', 'Fire', 'Poisoning', 'Forces of nature']
substance_disorders = ['Drug disorder', 'Alcohol disorder']
maternal_neonatal_nutritional = ['Maternal disorder', 'Neonatal disorder', 'Nutritional deficiency', 'Protein energy malnutrition']
environmental = ['Environmental exposure']

df_countries['Infectious'] = df_countries[infectious_cols].sum(axis=1)
df_countries['NCD_Cardiovascular'] = df_countries[ncd_cardiovascular].sum(axis=1)
df_countries['NCD_Metabolic'] = df_countries[ncd_metabolic].sum(axis=1)
df_countries['NCD_Cancer'] = df_countries[ncd_cancer].sum(axis=1)
df_countries['NCD_Neurological'] = df_countries[ncd_neurological].sum(axis=1)
df_countries['NCD_Respiratory'] = df_countries[ncd_respiratory].sum(axis=1)
df_countries['NCD_Digestive'] = df_countries[ncd_digestive].sum(axis=1)
df_countries['NCDs_Total'] = df_countries[['NCD_Cardiovascular', 'NCD_Metabolic', 'NCD_Cancer', 'NCD_Neurological', 'NCD_Respiratory', 'NCD_Digestive']].sum(axis=1)
df_countries['Injuries_Violence'] = df_countries[injuries_violence].sum(axis=1)
df_countries['Injuries_Accidents'] = df_countries[injuries_accidents].sum(axis=1)
df_countries['Injuries_Total'] = df_countries[['Injuries_Violence', 'Injuries_Accidents']].sum(axis=1)
df_countries['Substance_Disorders'] = df_countries[substance_disorders].sum(axis=1)
df_countries['Maternal_Neonatal_Nutritional'] = df_countries[maternal_neonatal_nutritional].sum(axis=1)
df_countries['Environmental'] = df_countries[environmental].sum(axis=1)

print("✅ Disease classifications created")

latest_year = df_countries['Year'].max()
df_latest = df_countries[df_countries['Year'] == latest_year]
print(f"\n📅 Latest year in data: {latest_year}")

# =============================================================================
# GRAPH 1: GLOBAL DISTRIBUTION PIE CHART (Main Categories)
# =============================================================================
print("\n🎨 Graph 1: Global Distribution of Fatalities...")

fig, ax = plt.subplots(figsize=(12, 10))

main_categories = {
    'Infectious Diseases': df_latest['Infectious'].sum(),
    'NCDs (Non-Communicable)': df_latest['NCDs_Total'].sum(),
    'Injuries': df_latest['Injuries_Total'].sum(),
    'Maternal/Neonatal/Nutritional': df_latest['Maternal_Neonatal_Nutritional'].sum(),
    'Substance Disorders': df_latest['Substance_Disorders'].sum(),
    'Environmental': df_latest['Environmental'].sum()
}

main_colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#95a5a6']
main_explode = [0, 0.08, 0, 0, 0, 0]

wedges, texts, autotexts = ax.pie(
    main_categories.values(),
    labels=main_categories.keys(),
    autopct='%1.1f%%',
    startangle=90,
    colors=main_colors,
    explode=main_explode,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 14}
)

for autotext in autotexts:
    autotext.set_fontsize(13)
    autotext.set_fontweight('bold')

ax.set_title(f'Global Distribution of Fatalities by Category ({latest_year})\nTotal: {sum(main_categories.values())/1e6:.1f} Million Deaths', 
             fontsize=18, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'graph_01_global_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: graph_01_global_distribution.png")
plt.show()

# =============================================================================
# GRAPH 2: NCD BREAKDOWN PIE CHART
# =============================================================================
print("\n🎨 Graph 2: NCD Breakdown...")

fig, ax = plt.subplots(figsize=(12, 10))

ncd_breakdown = {
    'Cardiovascular': df_latest['NCD_Cardiovascular'].sum(),
    'Cancer (Neoplasms)': df_latest['NCD_Cancer'].sum(),
    'Metabolic (Diabetes, Kidney)': df_latest['NCD_Metabolic'].sum(),
    'Neurological (Dementia, Parkinson)': df_latest['NCD_Neurological'].sum(),
    'Chronic Respiratory': df_latest['NCD_Respiratory'].sum(),
    'Digestive/Liver': df_latest['NCD_Digestive'].sum()
}

ncd_colors = ['#c0392b', '#e74c3c', '#f39c12', '#d35400', '#e67e22', '#f1c40f']

wedges, texts, autotexts = ax.pie(
    ncd_breakdown.values(),
    labels=ncd_breakdown.keys(),
    autopct='%1.1f%%',
    startangle=90,
    colors=ncd_colors,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 13}
)

for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

total_ncds = sum(ncd_breakdown.values())
total_all = sum(main_categories.values())
ax.set_title(f'Non-Communicable Disease (NCD) Breakdown ({latest_year})\nTotal NCDs: {total_ncds/1e6:.1f}M ({total_ncds/total_all*100:.1f}% of all deaths)', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'graph_02_ncd_breakdown.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: graph_02_ncd_breakdown.png")
plt.show()

# =============================================================================
# GRAPH 3: INJURIES BREAKDOWN PIE CHART
# =============================================================================
print("\n🎨 Graph 3: Injuries Breakdown...")

fig, ax = plt.subplots(figsize=(12, 10))

injuries_breakdown = {
    'Violence/Conflict\n(Homicide, War, Self-harm)': df_latest['Injuries_Violence'].sum(),
    'Accidents\n(Road, Drowning, Fire, etc.)': df_latest['Injuries_Accidents'].sum()
}

injuries_colors = ['#8e44ad', '#f39c12']

wedges, texts, autotexts = ax.pie(
    injuries_breakdown.values(),
    labels=injuries_breakdown.keys(),
    autopct='%1.1f%%',
    startangle=90,
    colors=injuries_colors,
    explode=[0.05, 0.05],
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 14}
)

for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')

ax.set_title(f'Injuries Breakdown ({latest_year})\nTotal: {sum(injuries_breakdown.values())/1e6:.1f}M Deaths', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'graph_03_injuries_breakdown.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: graph_03_injuries_breakdown.png")
plt.show()

# =============================================================================
# GRAPH 4: VIOLENCE DETAIL PIE CHART
# =============================================================================
print("\n🎨 Graph 4: Violence-Related Deaths Detail...")

fig, ax = plt.subplots(figsize=(12, 10))

violence_detail = {
    'Interpersonal Violence (Homicide)': df_latest['Interpersonal violence'].sum(),
    'Self-harm (Suicide)': df_latest['Self harm'].sum(),
    'Conflict (War/Terrorism)': df_latest['Conflict'].sum()
}

violence_colors = ['#c0392b', '#8e44ad', '#2c3e50']

wedges, texts, autotexts = ax.pie(
    violence_detail.values(),
    labels=violence_detail.keys(),
    autopct=lambda pct: f'{pct:.1f}%\n({pct/100*sum(violence_detail.values())/1e3:.0f}K)',
    startangle=90,
    colors=violence_colors,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 13}
)

for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

ax.set_title(f'Violence-Related Deaths Detail ({latest_year})\nTotal: {sum(violence_detail.values())/1e6:.2f}M Deaths', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'graph_04_violence_detail.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: graph_04_violence_detail.png")
plt.show()

# =============================================================================
# GRAPH 5: REGIONAL STACKED BAR CHART
# =============================================================================
print("\n🎨 Graph 5: Fatalities by Region...")

fig, ax = plt.subplots(figsize=(14, 10))

regional_data = df_latest.groupby('Region')[['Infectious', 'NCDs_Total', 'Injuries_Total', 'Maternal_Neonatal_Nutritional']].sum()
regional_data = regional_data.sort_values('NCDs_Total', ascending=True)

x = np.arange(len(regional_data))
width = 0.7

bars1 = ax.barh(x, regional_data['Infectious']/1e6, width, label='Infectious', color='#3498db', alpha=0.8)
bars2 = ax.barh(x, regional_data['NCDs_Total']/1e6, width, left=regional_data['Infectious']/1e6, label='NCDs', color='#e74c3c', alpha=0.8)
bars3 = ax.barh(x, regional_data['Injuries_Total']/1e6, width, left=(regional_data['Infectious']+regional_data['NCDs_Total'])/1e6, label='Injuries', color='#f39c12', alpha=0.8)
bars4 = ax.barh(x, regional_data['Maternal_Neonatal_Nutritional']/1e6, width, left=(regional_data['Infectious']+regional_data['NCDs_Total']+regional_data['Injuries_Total'])/1e6, label='Maternal/Neonatal/Nutritional', color='#9b59b6', alpha=0.8)

ax.set_yticks(x)
ax.set_yticklabels(regional_data.index, fontsize=12)
ax.set_xlabel('Fatalities (Millions)', fontsize=14, fontweight='bold')
ax.set_title(f'Fatalities by Region and Cause Category ({latest_year})', fontsize=16, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'graph_05_regional_stacked_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: graph_05_regional_stacked_bar.png")
plt.show()

# =============================================================================
# GRAPH 6: NCD TYPE HEATMAP BY REGION
# =============================================================================
print("\n🎨 Graph 6: NCD Type Distribution Heatmap...")

fig, ax = plt.subplots(figsize=(14, 10))

ncd_regional = df_latest.groupby('Region')[['NCD_Cardiovascular', 'NCD_Cancer', 'NCD_Metabolic', 'NCD_Neurological', 'NCD_Respiratory', 'NCD_Digestive']].sum()
ncd_regional.columns = ['Cardiovascular', 'Cancer', 'Metabolic', 'Neurological', 'Respiratory', 'Digestive']
ncd_regional = ncd_regional.div(ncd_regional.sum(axis=1), axis=0) * 100
ncd_regional = ncd_regional.sort_values('Cardiovascular', ascending=True)

sns.heatmap(ncd_regional, annot=True, fmt='.1f', cmap='Reds', ax=ax,
            cbar_kws={'label': '% of Regional NCDs'}, linewidths=0.5,
            annot_kws={'fontsize': 11})
ax.set_title(f'NCD Type Distribution by Region ({latest_year})\n(% of Total Regional NCD Deaths)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('NCD Type', fontsize=13, fontweight='bold')
ax.set_ylabel('Region', fontsize=13, fontweight='bold')
ax.tick_params(axis='both', labelsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'graph_06_ncd_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: graph_06_ncd_heatmap.png")
plt.show()

# =============================================================================
# GRAPH 7: NCD/INFECTIOUS RATIO BY REGION
# =============================================================================
print("\n🎨 Graph 7: NCD/Infectious Disease Ratio by Region...")

fig, ax = plt.subplots(figsize=(14, 10))

ratio_data = df_latest.groupby('Region')[['Infectious', 'NCDs_Total']].sum()
ratio_data['NCD_to_Infectious_Ratio'] = ratio_data['NCDs_Total'] / ratio_data['Infectious']
ratio_data = ratio_data.sort_values('NCD_to_Infectious_Ratio')

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(ratio_data)))
bars = ax.barh(ratio_data.index, ratio_data['NCD_to_Infectious_Ratio'], color=colors, edgecolor='black', height=0.6)

ax.set_xlabel('NCD to Infectious Disease Ratio', fontsize=14, fontweight='bold')
ax.set_title(f'NCD/Infectious Disease Ratio by Region ({latest_year})\n(Higher = More Developed Health Profile)', fontsize=16, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='x')
ax.tick_params(axis='both', labelsize=12)

for bar, val in zip(bars, ratio_data['NCD_to_Infectious_Ratio']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}x', va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'graph_07_ncd_infectious_ratio.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: graph_07_ncd_infectious_ratio.png")
plt.show()

# =============================================================================
# GRAPH 8: INJURIES BY REGION (Violence vs Accidents)
# =============================================================================
print("\n🎨 Graph 8: Injury Deaths by Region...")

fig, ax = plt.subplots(figsize=(14, 10))

injuries_regional = df_latest.groupby('Region')[['Injuries_Violence', 'Injuries_Accidents']].sum()
injuries_regional['Total'] = injuries_regional['Injuries_Violence'] + injuries_regional['Injuries_Accidents']
injuries_regional = injuries_regional.sort_values('Total', ascending=True)
injuries_regional = injuries_regional.drop('Total', axis=1)

x = np.arange(len(injuries_regional))
width = 0.35

bars1 = ax.barh(x - width/2, injuries_regional['Injuries_Violence']/1e3, width, label='Violence (Conflict, Homicide, Self-harm)', color='#c0392b', alpha=0.8)
bars2 = ax.barh(x + width/2, injuries_regional['Injuries_Accidents']/1e3, width, label='Accidents (Road, Drowning, Fire, etc.)', color='#f39c12', alpha=0.8)

ax.set_yticks(x)
ax.set_yticklabels(injuries_regional.index, fontsize=12)
ax.set_xlabel('Fatalities (Thousands)', fontsize=14, fontweight='bold')
ax.set_title(f'Injury Deaths by Region ({latest_year})\nViolence vs Accidents', fontsize=16, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'graph_08_injuries_by_region.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: graph_08_injuries_by_region.png")
plt.show()

# =============================================================================
# GRAPH 9: TOP 15 CAUSES OF DEATH GLOBALLY
# =============================================================================
print("\n🎨 Graph 9: Top 15 Causes of Death Globally...")

fig, ax = plt.subplots(figsize=(14, 10))

global_causes = df_latest[fatality_cols].sum().sort_values(ascending=True).tail(15)
colors_causes = plt.cm.viridis(np.linspace(0.1, 0.9, len(global_causes)))
bars = ax.barh(global_causes.index, global_causes.values/1e6, color=colors_causes, edgecolor='black', linewidth=0.5, height=0.7)

ax.set_xlabel('Fatalities (Millions)', fontsize=14, fontweight='bold')
ax.set_title(f'Top 15 Causes of Death Globally ({latest_year})', fontsize=16, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='x')
ax.tick_params(axis='both', labelsize=11)

for bar, val in zip(bars, global_causes.values):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f'{val/1e6:.2f}M', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'graph_09_top15_causes.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: graph_09_top15_causes.png")
plt.show()

# =============================================================================
# GRAPH 10: INFECTIOUS VS NCDs OVER TIME
# =============================================================================
print("\n🎨 Graph 10: Infectious vs NCDs Over Time...")

fig, ax = plt.subplots(figsize=(14, 8))

df_yearly = df_countries.groupby('Year')[['Infectious', 'NCDs_Total', 'Injuries_Total', 'NCD_Cardiovascular', 'NCD_Cancer', 'NCD_Metabolic', 'NCD_Neurological', 'Injuries_Violence', 'Injuries_Accidents']].sum()

ax.plot(df_yearly.index, df_yearly['Infectious']/1e6, 'b-o', linewidth=3, markersize=6, label='Infectious Diseases')
ax.plot(df_yearly.index, df_yearly['NCDs_Total']/1e6, 'r-s', linewidth=3, markersize=6, label='NCDs')

ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Fatalities (Millions)', fontsize=14, fontweight='bold')
ax.set_title('Infectious vs Non-Communicable Disease Trends\n(Reflects Technology & Development Impact)', fontsize=16, fontweight='bold', pad=15)
ax.legend(loc='center right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'graph_10_infectious_vs_ncd_trend.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: graph_10_infectious_vs_ncd_trend.png")
plt.show()

# =============================================================================
# GRAPH 11: NCD BREAKDOWN OVER TIME (Stacked Area)
# =============================================================================
print("\n🎨 Graph 11: NCD Breakdown Over Time...")

fig, ax = plt.subplots(figsize=(14, 8))

ax.stackplot(df_yearly.index, 
             df_yearly['NCD_Cardiovascular']/1e6,
             df_yearly['NCD_Cancer']/1e6,
             df_yearly['NCD_Metabolic']/1e6,
             df_yearly['NCD_Neurological']/1e6,
             labels=['Cardiovascular', 'Cancer', 'Metabolic', 'Neurological'],
             colors=['#c0392b', '#e74c3c', '#f39c12', '#d35400'], alpha=0.8)

ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Fatalities (Millions)', fontsize=14, fontweight='bold')
ax.set_title('NCD Breakdown Trends Over Time', fontsize=16, fontweight='bold', pad=15)
ax.legend(loc='upper left', fontsize=12)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'graph_11_ncd_breakdown_trend.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: graph_11_ncd_breakdown_trend.png")
plt.show()

# =============================================================================
# GRAPH 12: VIOLENCE VS ACCIDENTS OVER TIME
# =============================================================================
print("\n🎨 Graph 12: Violence vs Accidents Over Time...")

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(df_yearly.index, df_yearly['Injuries_Violence']/1e6, 'purple', linewidth=3, marker='o', markersize=6, label='Violence (Conflict, Homicide, Self-harm)')
ax.plot(df_yearly.index, df_yearly['Injuries_Accidents']/1e6, 'orange', linewidth=3, marker='s', markersize=6, label='Accidents (Road, Drowning, Fire, etc.)')

ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Fatalities (Millions)', fontsize=14, fontweight='bold')
ax.set_title('Injury Deaths: Violence vs Accidents Over Time', fontsize=16, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'graph_12_violence_vs_accidents_trend.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: graph_12_violence_vs_accidents_trend.png")
plt.show()

# =============================================================================
# GRAPH 13: NCD/INFECTIOUS RATIO OVER TIME (Development Indicator)
# =============================================================================
print("\n🎨 Graph 13: NCD/Infectious Ratio Over Time...")

fig, ax = plt.subplots(figsize=(14, 8))

ratio_trend = df_yearly['NCDs_Total'] / df_yearly['Infectious']
ax.plot(df_yearly.index, ratio_trend, 'g-o', linewidth=3, markersize=6, color='#27ae60')
ax.fill_between(df_yearly.index, ratio_trend, alpha=0.3, color='#27ae60')

ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('NCD to Infectious Ratio', fontsize=14, fontweight='bold')
ax.set_title('Global NCD/Infectious Disease Ratio Trend\n(Higher = Better Development & Healthcare)', fontsize=16, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=12)

ax.annotate(f'{ratio_trend.iloc[0]:.1f}x\n({df_yearly.index[0]})', xy=(df_yearly.index[0], ratio_trend.iloc[0]), fontsize=12, ha='center', fontweight='bold')
ax.annotate(f'{ratio_trend.iloc[-1]:.1f}x\n({df_yearly.index[-1]})', xy=(df_yearly.index[-1], ratio_trend.iloc[-1]), fontsize=12, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'graph_13_ncd_infectious_ratio_trend.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: graph_13_ncd_infectious_ratio_trend.png")
plt.show()

# =============================================================================
# GRAPHS 14-23: INDIVIDUAL REGION PIE CHARTS
# =============================================================================
print("\n🎨 Graphs 14-23: Individual Region Pie Charts...")

regions = df_latest.groupby('Region')['Infectious'].sum().sort_values(ascending=False).index.tolist()

for i, region in enumerate(regions):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    region_data = df_latest[df_latest['Region'] == region]
    
    totals = {
        'Infectious': region_data['Infectious'].sum(),
        'Cardiovascular': region_data['NCD_Cardiovascular'].sum(),
        'Cancer': region_data['NCD_Cancer'].sum(),
        'Other NCDs': region_data[['NCD_Metabolic', 'NCD_Neurological', 'NCD_Respiratory', 'NCD_Digestive']].sum().sum(),
        'Violence': region_data['Injuries_Violence'].sum(),
        'Accidents': region_data['Injuries_Accidents'].sum(),
        'Maternal/Neonatal': region_data['Maternal_Neonatal_Nutritional'].sum()
    }
    
    colors = ['#3498db', '#c0392b', '#e74c3c', '#f39c12', '#8e44ad', '#f1c40f', '#9b59b6']
    totals = {k: v for k, v in totals.items() if v > 0}
    
    wedges, texts, autotexts = ax.pie(
        totals.values(),
        labels=totals.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=colors[:len(totals)],
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 12}
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    total_deaths = sum(totals.values())
    ax.set_title(f'{region}\nCause of Death Distribution ({latest_year})\nTotal: {total_deaths/1e6:.2f}M Deaths', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    region_filename = region.replace(' ', '_').replace('&', 'and').replace('/', '_')
    plt.savefig(OUTPUT_PATH + f'graph_{14+i:02d}_region_{region_filename}.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: graph_{14+i:02d}_region_{region_filename}.png")
    plt.show()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE!")
print("="*80)

print(f"\n📊 Generated {13 + len(regions)} separate graphs:")
print("  Graph 01: Global Distribution of Fatalities")
print("  Graph 02: NCD Breakdown")
print("  Graph 03: Injuries Breakdown")
print("  Graph 04: Violence-Related Deaths Detail")
print("  Graph 05: Regional Stacked Bar Chart")
print("  Graph 06: NCD Type Heatmap by Region")
print("  Graph 07: NCD/Infectious Ratio by Region")
print("  Graph 08: Injuries by Region (Violence vs Accidents)")
print("  Graph 09: Top 15 Causes of Death Globally")
print("  Graph 10: Infectious vs NCDs Over Time")
print("  Graph 11: NCD Breakdown Over Time")
print("  Graph 12: Violence vs Accidents Over Time")
print("  Graph 13: NCD/Infectious Ratio Over Time")
print("  Graphs 14-23: Individual Region Pie Charts")

print(f"\n📁 All saved to: {OUTPUT_PATH}")
print("\n✅ Each graph is now SEPARATE and FULL-SIZE for clear readability!")
