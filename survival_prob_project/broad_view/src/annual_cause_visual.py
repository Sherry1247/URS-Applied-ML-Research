import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the uploaded file
FILE_PATH = "/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/broad_view/Annual cause death numbers new.csv"

# --- 1. Setup and Data Loading ---
print(f"Attempting to load data from: {FILE_PATH}")

try:
    df = pd.read_csv(FILE_PATH)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file was not found at the specified path: {FILE_PATH}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during file loading: {e}")
    exit()

# Clean column names for easier access: remove extra spaces and ' fatalities'
df.columns = [col.strip().replace(' fatalities', '').replace('\n', '') for col in df.columns]

# Identify fatality columns (assuming all non-Entity, Code, Year are death causes)
EXCLUDE_COLS = ['Entity', 'Code', 'Year']
fatality_cols = [col for col in df.columns if col not in EXCLUDE_COLS]

# Convert fatality columns to numeric, coercing errors to NaN
for col in fatality_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN in critical columns (though typically we fill them or proceed)
df.dropna(subset=fatality_cols, inplace=True)


# --- 2. Country-to-Region Mapping ---

# A robust, simplified mapping for common countries in mortality datasets
# This is crucial for the regional analysis requested by the user.
def get_region(country):
    if country in ['China', 'India', 'Indonesia', 'Pakistan', 'Bangladesh', 'Japan', 'Philippines', 'Vietnam', 'Turkey', 'Iran', 'Thailand', 'South Korea', 'Iraq', 'Afghanistan', 'Saudi Arabia', 'Malaysia', 'Uzbekistan', 'Nepal', 'Yemen', 'North Korea', 'Sri Lanka', 'Kazakhstan', 'Syria', 'Cambodia', 'Jordan', 'Laos', 'Myanmar', 'Mongolia', 'Taiwan']:
        return 'Asia'
    elif country in ['United States', 'Brazil', 'Mexico', 'Colombia', 'Argentina', 'Canada', 'Peru', 'Venezuela', 'Chile', 'Ecuador', 'Guatemala', 'Cuba', 'Dominican Republic', 'Haiti', 'Bolivia']:
        return 'Americas'
    elif country in ['Nigeria', 'Egypt', 'South Africa', 'DR Congo', 'Tanzania', 'Kenya', 'Algeria', 'Sudan', 'Uganda', 'Morocco', 'Angola', 'Mozambique', 'Ghana', 'Madagascar', 'Cameroon', 'Burkina Faso', 'Mali', 'Niger', 'Zambia', 'Zimbabwe', 'Rwanda', 'Senegal', 'Chad', 'Somalia', 'Guinea']:
        return 'Africa'
    elif country in ['Russia', 'Germany', 'United Kingdom', 'France', 'Italy', 'Spain', 'Ukraine', 'Poland', 'Romania', 'Netherlands', 'Belgium', 'Greece', 'Portugal', 'Sweden', 'Hungary', 'Czechia', 'Austria', 'Switzerland', 'Serbia', 'Ireland', 'Norway', 'Finland', 'Denmark']:
        return 'Europe'
    elif country in ['Australia', 'New Zealand']:
        return 'Oceania'
    else:
        return 'Other'

df['Region'] = df['Entity'].apply(get_region)


# --- 3. Visualization 1: Global Cause of Death (Pie Chart) ---

# Group major causes into broader categories for clarity
df['Infectious'] = df[['Meningitis', 'Malaria', 'HIV/AIDS', 'Tuberculosis', 'Diarrheal disease', 'Acute hepatitis', 'Measles']].sum(axis=1)
df['NCDs (Non-Communicable)'] = df[['Dementia', 'Parkinson s', 'Cardiovascular', 'Neoplasm', 'Diabetes', 'Chronic kidney', 'Chronic respiratory', 'Chronic liver', 'Digestive disease']].sum(axis=1)
df['Injuries'] = df[['Drowning', 'Interpersonal violence', 'Self harm', 'Road injury', 'Fire', 'Poisoning']].sum(axis=1)
df['Other/Environmental/Nutritional'] = df[['Nutritional deficiency', 'Maternal disorder', 'Drug disorder', 'Alcohol disorder', 'Forces of nature', 'Environmental exposure', 'Conflict', 'Protein energy malnutrition']].sum(axis=1)

# Analyze the latest year available
latest_year = df['Year'].max()
df_latest = df[df['Year'] == latest_year]

# Calculate global totals for new categories
global_totals = df_latest[['Infectious', 'NCDs (Non-Communicable)', 'Injuries', 'Other/Environmental/Nutritional']].sum()
global_totals = global_totals[global_totals > 0] # Filter out zero totals

plt.figure(figsize=(10, 10))
plt.pie(
    global_totals.values,
    labels=global_totals.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette('pastel'),
    wedgeprops={'edgecolor': 'black', 'linewidth': 0.8}
)
plt.title(f'Global Distribution of Fatalities by Cause Group ({latest_year})', fontsize=16, pad=20)
plt.show()


# --- 4. Visualization 2: Regional Mortality Trends (Line Chart) ---

df_regional = df.groupby(['Year', 'Region'])[['Infectious', 'NCDs (Non-Communicable)', 'Injuries', 'Other/Environmental/Nutritional']].sum().reset_index()
df_regional['Total Fatalities'] = df_regional[['Infectious', 'NCDs (Non-Communicable)', 'Injuries', 'Other/Environmental/Nutritional']].sum(axis=1)

# Plotting the total fatalities trend by region
plt.figure(figsize=(14, 7))
sns.lineplot(
    data=df_regional,
    x='Year',
    y='Total Fatalities',
    hue='Region',
    marker='o',
    dashes=False,
    linewidth=2.5
)
plt.title('Total Annual Fatalities Trend by Geographical Region', fontsize=16, pad=15)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Total Fatalities (Millions)', fontsize=14)
plt.ticklabel_format(style='plain', axis='y') # Prevent scientific notation
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{x/1e6:,.0f}M'))
plt.legend(title='Region', loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# --- 5. Visualization 3: Socioeconomic/Technology Influence on Survival Probability ---
# Proxy analysis: The shift from infectious diseases (often preventable/treatable with basic tech/sanitation)
# to NCDs (often linked to lifestyle, longevity, and complex healthcare) indicates socioeconomic and technological advancement.
# A higher NCDs/Infectious ratio generally indicates better societal development and survival past infectious risks.

df_global_causes = df_regional.groupby('Year')[['Infectious', 'NCDs (Non-Communicable)', 'Injuries']].sum().reset_index()

# Reshape data for line plot
df_melted_causes = df_global_causes.melt(
    id_vars='Year',
    value_vars=['Infectious', 'NCDs (Non-Communicable)'],
    var_name='Cause Type',
    value_name='Fatalities'
)

plt.figure(figsize=(14, 7))
sns.lineplot(
    data=df_melted_causes,
    x='Year',
    y='Fatalities',
    hue='Cause Type',
    marker='o',
    dashes=False,
    linewidth=2.5,
    palette=['#3b82f6', '#10b981'] # Blue for Infectious, Green for NCDs
)
plt.title('Global Trends in Infectious vs. Non-Communicable Disease Fatalities', fontsize=16, pad=15)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Annual Fatalities (Millions)', fontsize=14)
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{x/1e6:,.0f}M'))
plt.legend(title='Cause Type', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\n--- Socio-economic/Technology Influence Analysis (Proxy) ---")
print("The third plot compares Infectious Diseases (high-impact in low-resource settings) vs.")
print("Non-Communicable Diseases (high-impact in aging/developed populations).")
print("A growing ratio of NCDs to Infectious fatalities suggests global improvements")
print("in sanitation, vaccination, and basic healthcare (a key aspect of increased survival probability).")
