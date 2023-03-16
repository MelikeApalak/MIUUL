# Diabetes Prediction with Logistic Regression

#iş problemi
#özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
#edebilecek bir makine öğrenmesi modeli geliştirebilir misiniz?

#veri seti ABD'deki ulusal diyabet sindirim böbtrek hastalıkları estitülerinde tutulan
#büyük veri setinin parçasıdır.
#768 gözlem ve 8 sayısal bağımsız değişkenden oluşmaktadır. hedef değişken "outcome" olarak belirtilmiş olup;
#1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

#değişkenler
#pregnancies: hamilelik sayısı
#Glucose : glikoz
#BloodPressure: kan basıncı
#SkinThickness : cilt kalınlığı
#Insulin : insülin
#BMI : Beden kitle indeksi
#DiabetesPedigreeFunction : soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan fonk.
#age : yaş
#outcome : kişinin diyabet olup olmadığı bilgisi (hasta 1, hasta değil 0)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report,plot_roc_curve
from sklearn.model_selection import train_test_split,cross_validate

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr',False)

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
   quartile1 = dataframe[col_name].quantile(q1)
   quartile3 = dataframe[col_name].quantile(q3)
   interquantile_range = quartile3- quartile1
   up_limit = quartile3 + 1.5 * interquantile_range
   low_limit = quartile1 - 1.5 * interquantile_range
   return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name]<low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df = pd.read_csv("machine_learning/datasets/diabetes.csv")
df.head()
df.shape

#target analizi
df["Outcome"].value_counts()
sns.countplot(x="Outcome",data=df)
plt.show()

100 * df["Outcome"].value_counts() / len(df) #oran bilgisi (veri setinin %35inde 1 sınıfı var)

#feature analizi
df.describe().T

#sayısal değişken görselleştirmesi -> boxplot, histogram
df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()

def plot_numerical_col(dataframe,numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show()

for col in df.columns:
    plot_numerical_col(df,col)

#outcome dışarıda bırakmak için
cols = [col for col in df.columns if "Outcome" not in col]

#bağımlı değişkeni barındırmaz(outcome)
for col in cols:
    plot_numerical_col(df,col)

#target vs features
df.groupby("Outcome").agg({"Pregnancies": "mean"})

#target ve featureların birlikte analizi
def target_summary_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n\n")
for col in cols:
    target_summary_with_num(df,"Outcome",col)

#data preprocessing
df.shape
df.head()

df.isnull().sum()

#numeric bağımsız değişkenler
df.describe().T

for col in cols:
    print(col,check_outlier(df,col))

replace_with_thresholds(df,"Insulin")

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])
#robust aykırı değerlere daha dayanıklıdır.

df.head()

#model & prediction
y = df["Outcome"]
X = df.drop(["Outcome"],axis=1)
log_model = LogisticRegression().fit(X,y)

log_model.intercept_ #sabit
log_model.coef_ #ağırlık

y_pred = log_model.predict(X)
y_pred[0:10]
y[0:10]

#model evaluation
def plot_confusion_matrix(y,y_pred):
    acc = round(accuracy_score(y,y_pred),2)
    cm=confusion_matrix(y,y_pred)
    sns.heatmap(cm,annot=True,fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc),size=10)
    plt.show()

plot_confusion_matrix(y,y_pred)

print(classification_report(y,y_pred))

#accuracy : 0.78
#precision : 0.74
#recall : 0.58
#F!-score : 0.65

#ROC AUC
y_prob = log_model.predict_proba(X)[:,1]
roc_auc_score(y,y_prob)
# 0.8393

#model validation : holdout

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=17)
log_model = LogisticRegression().fit(X_train,y_train)
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:,1]
print(classification_report(y_test,y_pred))

plot_roc_curve(log_model,X_test,y_test)
plt.title('ROC Curve')
plt.plot([0,1], [0,1], 'r--')
plt.show()

#AUC
roc_auc_score(y_test,y_prob)

#Model Validation : 10-Fold Cross  Validation
y = df["Outcome"]
X= df.drop(["Outcome"],axis=1)

log_model = LogisticRegression().fit(X,y)

cv_results = cross_validate(log_model,
                            X,y,
                            cv=5,
                            scoring=["accuracy","precision","recall","f1","roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#prediction for a new obversation

X.columns
random_user = X.sample(1,random_state=45)
log_model.predict(random_user)
