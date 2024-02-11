import numpy as np
import pandas as pd
import sklearn
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek



df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', sep='\s+')
df.columns=['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']
# print(df.head())

# print(df.describe())


# print(df.shape)
# # 0 for rows and 1 for columns
# print('data has a cardinality size {}'.format(df.shape[0]) +
#       ' and dimensionality size {}'.format(df.shape[1]))


# print(df.isna().sum())
# print(df.isna())

# print(df.info())

# print(df.select_dtypes('int'))
# print(df.select_dtypes(include='object'))


# print(df.horsepower.unique())

# print(df['horsepower']== '?')
# print(df[df['horsepower']== '?'])
# df[df['car name']== 'renault 18i']


df.horsepower = pd.to_numeric(df.horsepower.replace('?', np.nan))
# print(df.horsepower.unique())
# print(df.isnull().sum())


df_zero = df.copy()
df_zero = df_zero.fillna(0)
# print(df_zero.isnull().sum())



df_mean = df.copy()
## df_mean['horsepower'].fillna(df_mean['horsepower'].mean(), inplace=True)
df_mean.fillna({'horsepower': df_mean['horsepower'].mean()}, inplace=True)

# print(df_mean.horsepower.unique())
# print(df_mean.isnull().sum())

# print(df_mean[df_mean['car name'] == 'renault 18i'].to_string())


df_median = df.copy()
df_mode = df.copy()

## df_median['horsepower'].fillna(df_median['horsepower'].median(), inplace=True)
## df_mode['horsepower'].fillna(df_mode['horsepower'].mode(), inplace=True)
df_median.fillna({'horsepower': df_median['horsepower'].median()}, inplace=True)
df_mode.fillna({'horsepower': df_mode['horsepower'].mode()}, inplace=True)

# print(df_median[df_median['car name'] == 'renault 18i'].to_string())
# print(df_median[df_mode['car name'] == 'renault 18i'].to_string())



df_linear = df.copy()
df_linear = df_linear.interpolate(method="linear")
# print(df_linear.isnull().sum())

df_linear.horsepower.interpolate()
df_linear.horsepower.isnull().sum()

# print(df_linear[df_mean['car name'] == 'renault 18i'].to_string())


forward = df.copy()
backward = df.copy()

backward = backward.interpolate(method='linear', limit_direction='backward')
backward = forward.interpolate(method='linear', limit_direction='forward')


poly = df.copy()
poly = poly.interpolate(method="polynomial", order=2)
# print(poly[poly['car name'] == 'renault 18i'].to_string())


# df_rm = df.copy()
# df_rm = df_rm.fillna(df_rm.rolling(6, min_periods=1).mean())
# print(df_rm[df_rm['car name'] == 'renault 18i'].to_string())



df_knn = df.copy(deep=True)
knn_imputer = KNNImputer(n_neighbors=2, weights="uniform")
df_knn['horsepower'] = knn_imputer.fit_transform(df_knn[['horsepower']])
# print(df_knn.isnull().sum())
# print(df_knn[df_knn['car name'] == 'renault 18i'].to_string())



# try to fill nan values with a custom function
def f(x):
  return np.std(x)/np.mean(x)

col = 'horsepower'
df_mpg = df.copy()
df_mpg[col].fillna(f(df_mpg[col]), inplace=True)
# print(df_mpg[df_mpg['car name'] == 'renault 18i'].to_string())

# print(df.duplicated().sum())



df['Class'] = pd.cut(df['mpg'], bins=[df['mpg'].min(), 28, df['mpg'].max()] , labels=[0,1])
# df["Class"].hist()
# plt.show()
# plt.clf()
# print(df["Class"].value_counts())


class_count = df.Class.value_counts()
# print('Class 0:', class_count[0])
# print('Class 1:', class_count[1])
# print('Proportion:', round(class_count[0] / class_count[1], 2), ': 1')
# class_count.plot(kind='bar', title='Count (target)')
# plt.show()
# plt.clf()




count_class_0, count_class_1 = df.Class.value_counts()

# Divide by class
df_class_0 = df[df['Class'] == 0]
df_class_1 = df[df['Class'] == 1]


df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.Class.value_counts())

# df_test_under.Class.value_counts().plot(kind='bar', title='Count (class)')
# plt.show()
# plt.clf()



df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.Class.value_counts())

# df_test_over.Class.value_counts().plot(kind='bar', title='Count (Class)');
# plt.show()
# plt.clf()









X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=100, random_state=10
)

data = pd.DataFrame(X)
data['target'] = y
# data.target.value_counts().plot(kind='bar', title='Count (target)')
# plt.show()
# plt.clf()









def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    plt.clf()


pca = PCA(n_components=2)
X = pca.fit_transform(X)

# plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')







rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X, y)
# print('Removed indexes:', id_rus)
# plot_2d_space(X_rus, y_rus, 'Random under-sampling')



ros = RandomOverSampler()
# X_ros, y_ros = ros.fit_resample(X, y)
# plot_2d_space(X_ros, y_ros, 'Random over-sampling')
# print(X_ros.shape[0] - X.shape[0], 'new random picked points')




smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)
# plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')





smt = SMOTETomek()
X_smt, y_smt = smt.fit_resample(X, y)
# plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')



fig, ax = plt.subplots()
df['mpg'].hist(color='#abd3a9', edgecolor='black',
                          grid=False)
ax.set_title('mpg', fontsize=12)
ax.set_xlabel('mpg', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
# plt.show()
# plt.clf()





# df_cut = pd.cut(df['mpg'], bins=4)
# # df_cut = pd.cut(df_binary['mpg'], bins=4, labels=False)
# print(df_cut, '\n')
# print(df_cut.value_counts(), '\n')
# # Manually define cut bins with names
# cut_labels = ['Low', 'Moderate', 'High']
# cut_bins = [0, 20, 35, 47]
# print(pd.cut(df['mpg'], bins=cut_bins, labels=cut_labels))




quantile_list = [0, .25, .5, .75, 1.]
quantiles = df['acceleration'].quantile(quantile_list)
# print(quantiles)



'The red lines shows our bins.'
fig, ax = plt.subplots()
df['acceleration'].hist(bins=10, color='#abd3a9',
                             edgecolor='black', grid=False)
for quantile in quantiles:
    qvl = plt.axvline(quantile, color='r')
ax.legend([qvl], ['Quantiles'], fontsize=10)
ax.set_title('acceleration with Quantiles',
             fontsize=12)
ax.set_xlabel('acceleration', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
# plt.show()
# plt.clf()





df["d_consumption"] =  pd.qcut(df['mpg'], q=3)
df["d_consumption"].value_counts()
df['d_consumption'] = pd.qcut(df['mpg'] ,q = 3 , labels=["1","2","3"])
df["d_consumption"].hist()
# print(df["d_consumption"].value_counts())
# plt.show()
# plt.clf()