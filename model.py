import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('creditcard (1).csv')
data.head()

data.duplicated().sum()

data.drop_duplicates(inplace=True)
def countplot(col):
    plt.figure(figsize=(6, 5))
    sns.set_style("darkgrid")
    sns.countplot(data=data,
                x=col,
                palette='dark',
                width=0.5)
    plt.title(f"{col}'s Countplot",
              fontsize=14,
              weight="bold")
    plt.xlabel(col, fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.show()

countplot("Class")

data.isna().sum()


data.describe().T

df=data
zeros=df[df["Class"]==0].sample(492)
ones=df[df["Class"]==1]
df=pd.concat([zeros,ones],axis=0)
df.Class.value_counts()

def countplot(col):
    plt.figure(figsize=(6, 5))
    sns.set_style("darkgrid")
    sns.countplot(data=df,
                x=col,
                palette='dark',
                width=0.5)
    plt.title(f"{col}'s Countplot",
              fontsize=14,
              weight="bold")
    plt.xlabel(col, fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.show()

countplot("Class")

occ = data['Class'].value_counts(normalize=True)
occ

corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

def plot_data(X, y):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    return plt.show()

def prep_data(data):
    X=data.iloc[:, :-1]
    y=data.iloc[:, -1]
    return X,y


X ,y = prep_data(data)
plot_data(X.values, y.values)


fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = data['Amount'].values
time_val = data['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()


print()




