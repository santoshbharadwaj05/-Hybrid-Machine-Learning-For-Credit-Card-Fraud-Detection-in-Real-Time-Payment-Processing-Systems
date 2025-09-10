from flask import Flask, request, render_template,session,redirect,url_for
import sqlite3
import os
import numpy as np
import pandas as pd




df=pd.read_csv("creditcard (1).csv")

zeros=df[df["Class"]==0].sample(492)
ones=df[df["Class"]==1]
df=pd.concat([zeros,ones],axis=0)
df.Class.value_counts()
df1=df


from sklearn.decomposition import PCA

pca=PCA(n_components=1)

new_columns=pca.fit_transform(df.iloc[:,1:-2])
df["V"]=new_columns


df=df.drop(df.columns[1:-3],axis=1)
df=df.reset_index()
df=df.drop("index",axis=1)
df


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

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)





import joblib


KNN=joblib.load("KNN")
DT=joblib.load("DT")
LR=joblib.load("LR")
RFC=joblib.load("RFC")
GBC=joblib.load("GBC")
SVM=joblib.load("SVM")




from sklearn.preprocessing import StandardScaler

sc=StandardScaler()



app=Flask(__name__)
app.secret_key = os.urandom(16)
app.config['UPLOAD_FOLDER'] = 'uploads/'




l=["Fraud Credit card","Valid Credit card"]


@app.route('/')
def start():
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['password']
        psw1 = request.form['confirm_password']
        if(username !="" and password!="" and password==psw1):
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('Select * from user where email=(?) and password=(?)',(username,password))
            data=cursor.fetchone()
            if(data):
                msg="user already exists please login"
                return render_template('reg.html',msg=msg)
            else:
                if(not data):
                    cursor.execute("INSERT INTO user (email, password) VALUES (?, ?)", (username, password))
                    conn.commit()
                    conn.close()
                msg="successfully login"
                return render_template('login.html',msg=msg)
        else:
            msg="invalid values"
            return render_template("reg.html",msg=msg)

    else:
        msg = "request to register page"
        return render_template('reg.html',msg=msg)






@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('Select * from user where email=(?) and password=(?)', (username, password))
        d = cursor.fetchone()
        if(d):
            return render_template('home.html')
        else:
            msg="invalid password or invalid username"
            return render_template('login.html',msg=msg)
    else:
        msg = "login page request"
        return render_template("login.html",msg=msg)


@app.route('/main_page',methods=['GET', 'POST'])
def main_page():
    if request.method == "POST":

        time=float(request.form["time"])
        amount=float(request.form['amount'])
        v=float(request.form['v'])


        values=[[time,amount,v]]
        values=sc.fit_transform(values)

        model=request.form['model']

        global accuracy
        global prediction_values
        if model == "rfc":
            prediction = RFC.predict(values)
            accuracy = RFC.score(X_test, y_test)
            prediction_values=RFC.predict(X_test)
        elif model == "gbc":
            prediction = GBC.predict(values)
            accuracy = GBC.score(X_test, y_test)
            prediction_values = GBC.predict(X_test)
        elif model == "dt":
            prediction = DT.predict(values)
            accuracy = DT.score(X_test, y_test)
            prediction_values = DT.predict(X_test)
        elif model == "knn":
            prediction = KNN.predict(values)
            accuracy = KNN.score(X_test, y_test)
            prediction_values = KNN.predict(X_test)
        elif model == "svm":
            prediction = SVM.predict(values)
            accuracy = SVM.score(X_test, y_test)
            prediction_values = SVM.predict(X_test)
        elif model == "lr":
            prediction = LR.predict(values)
            accuracy = LR.score(X_test, y_test)
            prediction_values = LR.predict(X_test)



        return render_template("main_page.html",prediction=l[prediction[0]])

    return render_template("main_page.html")


@app.route('/data')
def data():
    return render_template("data.html",data_set=df1)


@app.route("/clean_data")
def clean_data():
    return render_template("clean data.html",data_set=new_data)
@app.route('/predict')
def prediction():
    try:
        if(len(prediction_values)):
            return render_template("prediction.html",prediction_values=prediction_values)
    except Exception:
        return render_template("prediction.html",prediction_value="we can not find")

@app.route("/accuracy")
def accuracy():
    try:
        return render_template("accuracy.html",r2_score=round(accuracy,2))
    except Exception:
        return render_template("accuracy.html",r2_score="we can not find accuracy")

@app.route("/data_analytics")
def Data_Analytics():
    return render_template("da.html")




if __name__ == '__main__':
    app.run(debug=True)

