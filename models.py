import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns



df=pd.read_csv("creditcard (1).csv")
df.shape
df.Class.value_counts()


df.isnull().sum()

zeros=df[df["Class"]==0].sample(492)
ones=df[df["Class"]==1]
df=pd.concat([zeros,ones],axis=0)
df.Class.value_counts()


from sklearn.decomposition import PCA

pca=PCA(n_components=1)

new_columns=pca.fit_transform(df.iloc[:,1:-2])
df["V"]=new_columns


df=df.drop(df.columns[1:-3],axis=1)
df=df.reset_index()
df=df.drop("index",axis=1)
print(df.shape)


label=df["Class"]
new_df=df.drop("Class",axis=1)

new_data=pd.concat([new_df,label],axis=1)
new_data

def remove(data,i):

    q25=data[i].quantile(0.25)
    q75=data[i].quantile(0.75)
    iqr=q75-q25
    lower=q25-1.5*iqr
    upper=q75+1.5*iqr
    data=data[data[i]>=lower]
    data=data[data[i]<=upper]
    return data

for i in new_data.columns:
    new_data=remove(new_data,i)


X=new_data.iloc[:,:-1]
y=new_data.iloc[:,-1]


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)




def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f'{model_name} Accuracy: {accuracy:.2f}')
    print(f'{model_name} Classification Report:\n{class_report}')

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Cancer', 'Has Cancer'])
    disp.plot(cmap='Purples')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

    return accuracy











dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_accuracy =evaluate_model(dt, x_test, y_test, 'DecisionTreeClassifier')
print(dt_accuracy)
import joblib
joblib.dump(dt,"DT")


rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)

rfc_accuracy =evaluate_model(rfc, x_test, y_test, 'RandomForestClassifier')
print(rfc_accuracy)
joblib.dump(dt,"RFC")


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
lr_accuracy =evaluate_model(lr, x_test, y_test, 'LogisticRegression')
print(lr_accuracy)

joblib.dump(lr,"LR")



from sklearn.neighbors import KNeighborsClassifier


knn=KNeighborsClassifier()
knn.fit(x_train,y_train)

knn_accuracy =evaluate_model(knn, x_test, y_test, 'KNeighborsClassifier')
print(knn_accuracy)

joblib.dump(knn,"KNN")


from sklearn.svm import SVC

svc=SVC()
svc.fit(x_train,y_train)

joblib.dump(svc,"SVM")

svc_accuracy =evaluate_model(svc, x_test, y_test, 'Support Vector Classifier')
print(svc_accuracy)


from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)

joblib.dump(gbc,"GBC")


gbc_accuracy =evaluate_model(svc, x_test, y_test, 'GradientBoostingClassifier')
print(gbc_accuracy)



print(dt.predict([list(new_data.iloc[500,:-1])]),rfc.predict([list(new_data.iloc[500,:-1])]),lr.predict([list(new_data.iloc[500,:-1])]))
print(new_data.iloc[500,:-1])
