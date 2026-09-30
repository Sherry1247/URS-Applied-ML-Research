import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import matplotlib.ticker as ticker

# Configuration for Data Files
# Ensure these files are in the same folder as your script
HABERMAN_FILE = "/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/breast_cancer/data/haberman.csv"
BRCA_FILE = "/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/breast_cancer/data/BRCA.csv"

# --- 1. Data Loading and Setup ---

# Load Haberman Survival Data (Breast Cancer, Operations 1958-1969)
try:
    # Haberman data columns: Age, Op_Year, Axil_Nodes, Survival_Status (1=lived 5+ years, 2=died < 5 years)
    df_haberman = pd.read_csv(HABERMAN_FILE, header=0, names=['Age', 'Op_Year', 'Axil_Nodes', 'Survival_Status'])
    
    # Haberman years are already in 'relative' format (e.g., 58, 60), which corresponds to (1958-1900)
    # We ensure they are treated as the Timeline_Year requested (Year - 1900)
    df_haberman['Timeline_Year'] = df_haberman['Op_Year']
    
    # Event: 1=Death (Status 2), 0=Alive (Status 1)
    df_haberman['Event'] = df_haberman['Survival_Status'].apply(lambda x: 1 if x == 2 else 0) 
    df_haberman['Survival_Time'] = 5 # Historical fixed follow-up
    df_haberman['Source'] = 'Haberman (1958-1969)'
    
except Exception as e:
    print(f"Error loading Haberman data: {e}")
    df_haberman = pd.DataFrame()

# Load BRCA Survival Data (Breast Cancer, Operations ~2017-2021)
try:
    df_brca = pd.read_csv(BRCA_FILE)
    
    # Convert dates
    df_brca['Date_of_Surgery'] = pd.to_datetime(df_brca['Date_of_Surgery'], errors='coerce')
    df_brca['Date_of_Last_Visit'] = pd.to_datetime(df_brca['Date_of_Last_Visit'], errors='coerce')
    
    # Calculate survival time in years
    df_brca['Survival_Time'] = (df_brca['Date_of_Last_Visit'] - df_brca['Date_of_Surgery']).dt.days / 365.25
    
    # Event Status: 1=Dead, 0=Alive
    df_brca['Event'] = df_brca['Patient_Status'].apply(lambda x: 1 if x == 'Dead' else 0) 
    df_brca['Op_Year'] = df_brca['Date_of_Surgery'].dt.year
    df_brca['Source'] = 'BRCA (2017-2021)'
    
    # Clean up
    df_brca.dropna(subset=['Op_Year', 'Survival_Time', 'Event'], inplace=True) 
    
    # Calculate the requested Timeline_Year (Year - 1900)
    # e.g., 2017 becomes 117
    df_brca['Timeline_Year'] = df_brca['Op_Year'] - 1900

except Exception as e:
    print(f"Error loading BRCA data: {e}")
    df_brca = pd.DataFrame()


# --- 2. Trend Calculation ---

# Combine the datasets
df_survival = pd.concat([
    df_haberman[['Timeline_Year', 'Survival_Time', 'Event', 'Source', 'Survival_Status']], 
    df_brca[['Timeline_Year', 'Survival_Time', 'Event', 'Source']]
], ignore_index=True)

# Function to calculate survival rate per year
def calculate_survival_rate(group):
    
    # Haberman Logic: Simple Proportion
    if group['Source'].iloc[0] == 'Haberman (1958-1969)':
        survived_count = len(group[group['Survival_Status'] == 1])
        total_count = len(group)
        survival_rate = survived_count / total_count if total_count > 0 else np.nan
        return pd.Series({'Survival_Rate': survival_rate})

    # BRCA Logic: Kaplan-Meier Estimation
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(group['Survival_Time'], event_observed=group['Event'])
        # Use the probability at the end of the observed window
        survival_prob = kmf.survival_function_.iloc[-1, 0] 
        return pd.Series({'Survival_Rate': survival_prob})
    except:
        return pd.Series({'Survival_Rate': np.nan})

# Group by the Timeline Year (Year - 1900) and Source to keep them separate
df_trend = df_survival.groupby(['Timeline_Year', 'Source']).apply(calculate_survival_rate).reset_index()
df_trend.dropna(subset=['Survival_Rate'], inplace=True)


# --- 3. Visualization ---

if df_trend.empty:
    print("Error: No data to plot.")
    exit()

sns.set_style("whitegrid")
plt.figure(figsize=(14, 8))

# Main Line Chart
sns.lineplot(
    data=df_trend,
    x='Timeline_Year',
    y='Survival_Rate',
    hue='Source', 
    marker='o',
    dashes=False,
    linewidth=3,
    palette=['#2b6cb0', '#2b6cb0'], 
    legend=False
)

# --- DYNAMIC X-AXIS SETUP ---
# Determine min and max years from the actual data
data_min = int(df_trend['Timeline_Year'].min())
data_max = int(df_trend['Timeline_Year'].max())

# Add padding to the visual range
x_padding = 5
plot_min = data_min - x_padding
plot_max = data_max + x_padding

# Generate ticks every 10 years covering the full range
# Round start down to nearest decade
start_tick = data_min - (data_min % 10)
ticks = list(range(start_tick, data_max + 15, 10))
labels = [f"{1900 + t} ({t})" for t in ticks]

plt.xticks(ticks=ticks, labels=labels, rotation=45)
plt.xlim(plot_min, plot_max)

# Visual Context (Shaded Areas based on actual data range)
# We find the min/max for each source dynamically
haberman_years = df_trend[df_trend['Source'].str.contains('Haberman')]['Timeline_Year']
brca_years = df_trend[df_trend['Source'].str.contains('BRCA')]['Timeline_Year']

if not haberman_years.empty:
    plt.axvspan(haberman_years.min(), haberman_years.max(), color='gray', alpha=0.15, label='Historical Era (Haberman)')

if not brca_years.empty:
    plt.axvspan(brca_years.min(), brca_years.max(), color='green', alpha=0.15, label='Modern Era (BRCA)')

# Labels and Titles
plt.xlabel("Year of Operation\n[Format: Actual Year (Years since 1900)]", fontsize=14, fontweight='bold')
plt.ylabel("Estimated 5-Year Survival Rate", fontsize=14, fontweight='bold')
plt.title("Impact of Technology on Breast Cancer Survival (Chronological Timeline)", fontsize=16, pad=20)

# Format Y-Axis as Percentage
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
plt.ylim(0.4, 1.05) 

# Dynamic Annotation Placement
mid_x = (data_min + data_max) / 2
plt.text(
    mid_x, 0.85, 
    "Gap indicates years with\nno dataset coverage", 
    fontsize=11, color='gray', ha='center',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)

plt.legend(loc='lower right', frameon=True, fontsize=12)
plt.tight_layout()

# Show Plot
plt.show()
