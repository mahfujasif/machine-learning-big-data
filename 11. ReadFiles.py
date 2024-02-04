import numpy as np
import pandas as pd
import sklearn
import matplotlib.pylab as plt
import seaborn as sns


# df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', sep='\s+')
# print(df.head())

dct = {'a':[1,2,3], 'b':[3,2,3], 'c':[2,5, None]}
df = pd.DataFrame(dct)
print(df.head())
print("-".ljust(30, "-"))
df.insert(df.shape[1], 'd', [6, 5, 3])
df.insert(df.shape[1], 'e', [6, 5, 3])
print(df.head())
print(df.shape[0])
print(df.shape[1])