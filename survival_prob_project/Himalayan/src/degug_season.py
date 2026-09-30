import pandas as pd

deaths = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/survival_prob_project/Himalayan/data/deaths.csv')

print("First 20 values of yr_season column:")
print(deaths['yr_season'].head(20))
print("\n\nUnique season values after current extraction:")
deaths['season'] = deaths['yr_season'].str.extract(r'(Spring|Summer|Autumn|Winter)')
print(deaths['season'].value_counts(dropna=False))
print(f"\nNull count: {deaths['season'].isna().sum()} out of {len(deaths)}")

print("\n\nLet's check the actual format:")
print("Sample yr_season values:")
for i, val in enumerate(deaths['yr_season'].head(30)):
    print(f"{i}: '{val}'")
