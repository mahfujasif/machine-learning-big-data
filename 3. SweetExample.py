import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from subprocess import check_output


df_sweets = pd.read_excel('datasets/Celebrations_Data.xlsx', sheet_name='RawDataTub')
with pd.option_context('display.max_columns', None):
    print(df_sweets)
# print(df_sweets.describe())

# print(df_sweets.dtypes)

m = df_sweets.loc[:, 'Person Supplying Data'].value_counts()
# print(m)


# print(pd.isnull(df_sweets).sum())


chocolates = df_sweets.select_dtypes(include='int64').iloc[:, :-1]
chocolates.mean().plot(kind='barh')
plt.title('average distribution of choc')
# plt.show()

# print(df_sweets.shape)


plt.clf()
sns.set(style="whitegrid")
ax = sns.violinplot(data=chocolates)
# plt.show()

plt.clf()
for i,j in enumerate(chocolates.columns):
    sns.distplot(chocolates.iloc[:,i], hist=False, label=j)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=3, fancybox=True, shadow=True)
plt.title('histogram of all choc dist', y=1.3)
plt.show()


plt.clf()
corr = chocolates.corr()
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns, annot=True)
plt.show()