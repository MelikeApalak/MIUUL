# Sales prediction with Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' %x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score

# Simple Linear Regression with OLS Using Scikit-Learn

df = pd.read_csv("machine_learning/datasets/advertising.csv")
df.shape

X = df[["TV"]]
y = df[["sales"]]

# Model

reg_model = LinearRegression().fit(X,y)

#y_hat = b + w*x

# sabit (bias-b)
reg_model.intercept_[0]

# katsayı (ağırlık, w1) (tv'nin katsayısı)
reg_model.coef_[0][0]

#predict

#150 birimlik tv harcamasında ne kadar satış olması beklenir?
reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

#500 birimlik tv harcaması olsa ne kadar satış olur?
reg_model.intercept_[0] + reg_model.coef_[0][0] * 500

df.describe().T

# visualization of the model
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b','s':9},
                ci=False, color="r")
g.set_title(f"Model denklemi: Sales= {round(reg_model.intercept_[0],2)} + TV*{round(reg_model.coef_[0][0],2)}")
g.set_ylabel("Satış sayısı")
g.set_xlabel("TV harcamaları")
plt.xlim(-10,310)
plt.ylim(bottom=0)
plt.show()

# prediction success

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y,y_pred) #10.5
y.mean()
y.std()

# RMSE
np.sqrt(mean_squared_error(y,y_pred)) #3.24

# MAE
mean_absolute_error(y,y_pred) #2.54

# R-square -> veri setindeki bağımsız değişkenin bağımlı değişkeni açıklama yüzdesidir.
#değişken sayısı arttıkça r^2 artmaya meyillidir.
reg_model.score(X,y)


# multiple linear regression

df = pd.read_csv("machine_learning/datasets/advertising.csv")

X= df.drop('sales',axis=1)
y = df[["sales"]]

# model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=1)

X_test.shape
y_test.shape

reg_model = LinearRegression().fit(X_train,y_train)

#sabit (b-bias)
reg_model.intercept_

#coefficients (w-weights)
reg_model.coef_

# predict
# tv:30, radio:10, newspaper:40, y:2.90, w:0.04,0.17,0.002
# y + w * x
#sales = 2.90 + tv*0.04 + radio*0.17 + newspaper*0.002

yeni_veri = [[30],[10],[40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)

# tahmin başarısı değerlendirme

#train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred)) #1.73

#train RKARE
reg_model.score(X_train,y_train)

#test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred)) #1.41

#test RKARE
reg_model.score(X_test,y_test)

# 10 katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                X,
                                y,
                                cv=10,
                                scoring="neg_mean_squared_error")))

#Simple Linear Regression with Gradient Descent from Scratch

#Cost function
def cost_function(Y,b,w,X):
    m = len(Y) #gözlem sayısı
    sse = 0
    for i in range(0,m):
        y_hat= b+ w *X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2
    mse = sse / m #toplam hata değerini gözlem sayısına bölüp MSE hesaplandı
    return mse

def update_weigths(Y,b,w,X,learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0,m):
        y_hat= b+w*X[i]
        y = Y[i]
        b_deriv_sum += (y_hat-y)
        w_deriv_sum += (y_hat-y) * X[i]
    new_b = b - (learning_rate*1 /m * b_deriv_sum)
    new_w = w - (learning_rate*1 /m * w_deriv_sum)
    return new_b,new_w


def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w={1}, mse={2}",format(initial_b,initial_w,
                                                                        cost_function(Y,initial_b,initial_w,X)))
    b = initial_b
    w= initial_w
    cost_history = []

    for i in range(num_iters):
        b,w= update_weigths(Y,b,w,X,learning_rate)
        mse= cost_function(Y,b,w,X)
        cost_history.append(mse)

        if i%100 ==0 :
            print("iter={:d}   b={:.2f}    w ={:.4f}   mse={:.4}".format(i,b,w,mse))
    print("After {0} iterations b={1}, w={2}, mse={3}".format(num_iters,b,w,cost_function(Y,b,w,X)))
    return cost_history,b,w

df= pd.read_csv('machine_learning/datasets/advertising.csv')
X = df["radio"]
Y = df["sales"]

#hyperparameters
learning_rate = 0.001
inital_b = 0.001
initial_w = 0.001
num_iters = 10000
train(Y,inital_b,initial_w,X,learning_rate,num_iters)

