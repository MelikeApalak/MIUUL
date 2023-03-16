#Decision Tree Classification : CART

#1: exploratory data analysis
#2: data preprocessing & feature engineering
#3: modeling using cart
#4: Hyperparameter optimization with GridSearchCV
#5: final model
#6: feature importance
#7: analyzing model complexity with learning curves
#8: visualizing the decision tree
#9 : extracting decision rules
#10: extracting python / sql / excel codes of decision rules
#11: prediction using python codes
#12: saving and loading model

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz,export_text
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate,validation_curve
from skompiler import skompile

pd.set_option('display.max_columns',None)
warnings.simplefilter(action='ignore',category=Warning)

#1. Exploratory Data Analysis

#2. Data Preprocessing & Feature Engineering

#3. Modeling using CART
df = pd.read_csv("machine_learning/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"],axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X,y)

#confusion matrix için y_pred
y_pred = cart_model.predict(X) #tüm gözlemler için tahmin edilen değer

#AUC için y_prob
y_prob = cart_model.predict_proba(X)[:,1]

#Confusion matrix
print(classification_report(y,y_pred)) #doğruluk değeri 1 çıktı.

# Holdout yöntemi ile başarı değerlendirme

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=45)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train,y_train)

#Train hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:,1]
print(classification_report(y_train,y_pred))
roc_auc_score(y_train,y_prob)

# Test hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:,1]
print(classification_report(y_test,y_pred))
roc_auc_score(y_test,y_prob)
#overfit oldu. test verisinde başarı ciddi oranda düştü.

#CV(cross validation) ile başarı değerlendirme
cart_model = DecisionTreeClassifier(random_state=17).fit(X,y)

cv_results = cross_validate(cart_model,
                            X,y,
                            cv=5,
                           scoring=["accuracy","f1","roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
#amaç başarının artması değil,  daha doğru sonuca (hatanın doğruluğu) gidiyor olmak.

#model başarısını yeni gözlemler, yeni değişkenler ekleyerek veri ön işleme yaparak
#hiperparametre optimizasyonu yaparak gerçekleştirebiliriz.

# Hyperparameter Optimization with GridSearchCV
cart_model.get_params() #mevcut modelin hiperparametreleri

cart_params = {'max_depth': range(1,11),
               "min_samples_split":range(2,20)}

cart_best_grid = GridSearchCV(cart_model,cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X,y)

#tüm veriyi kullanarak gidebiliriz.

cart_best_grid.best_params_

cart_best_grid.best_score_

random = X.sample(1,random_state = 45)
cart_best_grid.predict(random)

#final model
cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_,random_state=17).fit(X,y)
cart_final.get_params()
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X,y)

cv_restuls = cross_validate(cart_final,
                            X,y,
                            cv=5,scoring=["accuracy","f1","roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Feature Importance (değişken önem düzeyi)
#bir değişkenin değerlerini belirli bir noktadan bölüp hangi noktadan bölüneceğine karar vermek.
cart_final.feature_importances_ #değişken önem düzeyleri. formatı değiştirmemiz lazım.

#amaç düşük hatalı tahmin yapmak sse küçültmek, saflık ölçümlerini küçültmek.gini,entropi
#modeldeki değişkenlerin önem düzeyleri
def plot_importance(model,features,num=len(X),save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x='Value',y='Feature',data=feature_imp.sort_values(by="Value",
                                                                   ascending=False)[0:num])

    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(cart_final,X,num=5)

#7. Analyzing Model Complexity with Learning Curves
#öğrenme eğrileri kullanarak model karmaşıklığını analiz etme

train_score, test_score = validation_curve(cart_final,X,y,
                                           param_name="max_depth",
                                           param_range=range(1,11), #derinlik sayıları
                                           scoring="roc_auc",
                                           cv=10) #10 katlı çapraz doğrulama
#9 parça ile model kurar 1i ile test eder.
mean_train_score = np.mean(train_score,axis=1)
mean_test_score = np.mean(test_score,axis=1)

plt.plot(range(1,11),mean_train_score,
         label="Training Score", color='b')

plt.plot(range(1,11),mean_test_score,
         label="Validation Score", color='g')

plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show()
#training score validation scoredan ayrıldığı nokta modelin ezberlendiği noktadır.

#modelin ismini dinamik şekilde almak.
def val_curve_params(model,X,y,param_name,param_range,scoring="roc_auc",cv=10):
    train_score,test_score = validation_curve(
        model, X=X, y=y,param_name=param_name, param_range=param_range,scoring=scoring,cv=cv)

    mean_train_score = np.mean(train_score,axis=1)
    mean_test_score = np.mean(test_score,axis=1)

    plt.plot(param_range,mean_train_score,
             label = "Training Score",color='b')

    plt.plot(param_range,mean_test_score,
             label="Validation Score",color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

val_curve_params(cart_final,X,y,"max_depth",range(1,11),scoring="f1")

#birden çok hiperparametre olması durumu ve değer aralıkları
cart_val_params = [["max_depth",range(1,11)], ["min_samples_split",range(2,20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_model,X,y,cart_val_params[i][0],cart_val_params[i][1])

#8. Visualizing the Decision Tree
#karar ağaçlarını görselleştirme
import graphviz
def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model,feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

tree_graph(model=cart_final,col_names=X.columns, file_name="cart_final.png")
#kurulan modelin yer aldığı png dosyası

#9. Extracting Decision Rules
#karar kuralları çıkartmak
tree_rules = export_text(cart_final,feature_names=list(X.columns))
print(tree_rules)

#10. Extracting Python Codes of Decision Rules
#pip install scikit-learn==0.23.1

print(skompile(cart_final.predict).to('python/code'))
#görsel teknikle elde edilen karar ağacının fonksiyonlaştırabileceği karar kuralları

#karar kurallarını sql kodu olarak yazmak
print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))

#excel kodu
print(skompile(cart_final.predict).to('excel'))

#11. Prediction using Python Codes
#karar kuralları fonksiyonu
def predict_with_rules(x):
    return 1


X.columns

x=[12,13,20,23,4,55,12,7]
predict_with_rules(x)

x=[6,148,70,35,0,30,0.62,50]
predict_with_rules(x)

#12. saving and loading model

joblib.dump(cart_final,"cart_final.pkl")
cart_model_from_disc = joblib.load("cart_final.pkl")
x=[12,13,20,23,4,55,12,7]
cart_model_from_disc.predict(pd.DataFrame(x).T)

