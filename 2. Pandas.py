import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({'c1': [1,3,6,2,4,3], 'c2': [4,5,8,3,4,8]})
# print(df)

# print(df.info())
#
# print(df.mean())
#
# print(df.describe())

# df['c1'].plot.hist(bins=10)
# plt.show()

# df.hist()
# plt.show()

df.plot.scatter(x='c1',y='c2', linestyle='-', marker='o', color='red')
plt.show()