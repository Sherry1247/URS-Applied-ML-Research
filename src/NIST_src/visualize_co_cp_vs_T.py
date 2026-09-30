import re
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

file_path = "data/C-093.txt"

with open(file_path, "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

start_idx = next(i for i, l in enumerate(lines) if re.match(r"^\d", l))
data_text = "\n".join(lines[start_idx:])

cols = ["T_K", "Cp", "S", "GminusH_over_T", "HminusH_Tr",
        "deltaf_H", "deltaf_G", "logKf"]

df = pd.read_csv(StringIO(data_text), sep=r"\s+", header=None,
                 names=cols, na_values=["INFINITE", "INFINITY"])

print("First 5 rows of CO data:")
print(df.head())

# Plot Cp vs T
plt.figure(figsize=(7,5))
plt.plot(df["T_K"], df["Cp"], marker=".", color="red")
plt.xlabel("Temperature (K)")
plt.ylabel("Heat Capacity, Cp (J·K⁻¹·mol⁻¹)")
plt.title("CO — Heat Capacity vs Temperature (JANAF)")
plt.grid(True)
plt.show()
