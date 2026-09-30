import pandas as pd  # use pandas to laoad and manipulate data
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

df_income = pd.read_csv('http://bit.ly/income-file')

df_income.info()

print(df_income.head())
print(df_income.describe())
sns.displot(df_income, x="Income");
print(df_income.corr())

# output:
#    ID  Income  Age  Education  Gender
# 0   1     113   69         12       1
# 1   2      91   52         18       0
# 2   3     121   65         14       0
# 3   4      81   58         12       0
# 4   5      68   31         16       1
#                 ID       Income          Age    Education       Gender
# count  1500.000000  1500.000000  1500.000000  1500.000000  1500.000000
# mean    750.500000    75.986000    43.582000    14.681333     0.490000
# std     433.157015    20.005215    15.169466     2.693812     0.500067
# min       1.000000    14.000000    18.000000    10.000000     0.000000
# 25%     375.750000    62.000000    30.000000    12.000000     0.000000
# 50%     750.500000    76.000000    44.000000    15.000000     0.000000
# 75%    1125.250000    91.000000    57.000000    16.000000     1.000000
# max    1500.000000   134.000000    70.000000    20.000000     1.000000
#                  ID    Income       Age  Education    Gender
# ID         1.000000 -0.038846 -0.037770  -0.074147  0.005246
# Income    -0.038846  1.000000  0.761486   0.256634 -0.045060
# Age       -0.037770  0.761486  1.000000   0.026254 -0.027242
# Education -0.074147  0.256634  0.026254   1.000000 -0.004843
# Gender     0.005246 -0.045060 -0.027242  -0.004843  1.000000

# heatmap of correlation matrix
corr = df_income.iloc[:,1:5].corr()
sns.heatmap(corr, annot = True)

# pairplot
sns.pairplot(df_income.iloc[:, 1:4], kind="kde")


plt.show()
