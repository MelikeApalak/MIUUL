#feature engineering & data pre-processing

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#!pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f' % x)
pd.set_option('display.width',500)

def load_application_train():
    data = pd.read_csv(r"feature_engineering/datasets/application_train.csv")
    return data

dff = load_application_train()
dff.head()

def load():
    data = pd.read_csv(r"feature_engineering/datasets/titanic.csv")
    return data

df = load()
df.head()

#1. outliers
#aykırı değerleri yakalama

#grafik teknikle aykırı değer(boxplot, histogram)
sns.boxplot(x=df["Age"])
plt.show()

#aykırı değer nasıl yakalanır ? ->
# eşik değerlere erişmek.
# çeyrek değerlere erişelim ki IQR hesabı yapalım.
q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5* iqr
low = q1 - 1.5* iqr

df[(df["Age"]<low) | (df["Age"]>up) ] # aykırı değerleri listeler
df[(df["Age"]<low) | (df["Age"]>up) ].index # aykırı değerlerin indexleri


df[(df["Age"]<low) | (df["Age"]>up)].any(axis=None) # aykırı değer var mı? yoksa false, varsa true.
df[(df["Age"]<low)].any(axis=None)

#eşik değer belirler.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
   quartile1 = dataframe[col_name].quantile(q1)
   quartile3 = dataframe[col_name].quantile(q3)
   interquantile_range = quartile3- quartile1
   up_limit = quartile3 + 1.5 * interquantile_range
   low_limit = quartile1 - 1.5 * interquantile_range
   return low_limit, up_limit

low, up = outlier_thresholds(df, "Age")
low, up = outlier_thresholds(df, "Fare")

#aykırı değer varlığını kontrol eder.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name]<low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df,"Age")

#verisetindeki sayısal,kategorik değişkenleri otomatik olarak getirir.
#numerik, kategorik, kategorik görünen ama kategorik olmayan (kardinal), numerik görünen ama kategorik olan.
def grab_col_names(dataframe,cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    #cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    #numerik ama kategorik değişkenler

    #değişkenin içindeki sayısal değişkenin sınıf sayısı < 10 ise sayısal görünümlü kategoriktir.
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"] #-> değişken tipi object değilse

    #eğer bir kategorik değişkenin 20 den fazla sınıfı varsa ve tipi de kategorikse kardinaldir.
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    #kateogorik değişkenleri güncelleme
    cat_cols = cat_cols + num_but_cat

    #kategorik ama kardinal değişkenlerde olmayanlar
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col,check_outlier(df,col))


cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]
for col in num_cols:
    print(col,check_outlier(dff,col))

#aykırı değerlerin kendilerine erişmek
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:# aykırı değer sayısı 10 dan çoksa headini al
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index #outlierlerın indekslerine erişme

grab_outliers(df, "Age")
age_index = grab_outliers(df, "Age",index=True)

# aykırı değer problemi çözme

#silme

low, up = outlier_thresholds(df, "Fare") #alt, üst limit
df.shape #-> veri setindeki gözlem sayısı
df[~((df["Fare"]<low) | (df["Fare"]>up))].shape

#birden fazla değişken için aykırılıkları silme
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]
df.shape

for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0] # -> kaç tane değişiklik oldu? 116 gözlem silinmiş

#baskılama yöntemi
#(re-assignment with thresholds)

low, up = outlier_thresholds(df, "Fare")

df.loc[((df["Fare"]<low) | (df["Fare"]>up)),"Fare"] #-> belirli değerlerden büyük ve küçük olan fare lar gelir.

df.loc[(df["Fare"]> up), "Fare"] = up #-> büyük olan değerleri up a eşitledi.
#df.loc[(df["Fare"]< low), "Fare"] = low

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col,check_outlier(df,col))

for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:
    print(col, check_outlier(df, col))

#Recap

df =load()
outlier_thresholds(df,"Age") #aykırı değerleri thresholdlara göre saptama
check_outlier(df,"Age")
grab_outliers(df,"Age",index=True) #outlier

remove_outlier(df,"Age").shape #aykırı değer silme
replace_with_thresholds(df,"Age") #eşikdeğerlerle değiştir


#COK DEGİSKENLİ AYKIRI DEGER ANALİZİ: LOCAL OUTLIER FACTOR
# 17 yasında 3 kez evlenmek anormaldir. tek basına aykırı olmayan degerler birikte ele alınınca aykırılık yaratabilir.

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()
df.shape

for col in df.columns:
    print(col, check_outlier(df, col)) #aykırı değişkenler

low, up = outlier_thresholds(df, "carat")
df[((df["carat"] < low) | (df["carat"] > up))].shape  #karat değişkeninde kaç tane aykırı değer var

low, up = outlier_thresholds(df, "depth")
df[((df["depth"] < low) | (df["depth"] > up))].shape

clf = LocalOutlierFactor(n_neighbors=20)
#komşuluk sayısı 20 de çok değişkenli outlier inceleme
clf.fit_predict(df)  #skorları getirdi lof skorları

df_scores = clf.negative_outlier_factor_ #skorları takip etme
df_scores[0:5] #eksi değerlerle beraber gözlemlendi

#df.scores = -df_scores
#elbow yöntemine göre bakacağımızdan biz bu değerleri pozitif olarak gözlemlemek istiyoruz.

np.sort(df_scores)[0:5]  #en kötü beş gözlem skoru

#dirsek(elbow) yöntemi ile
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()
th = np.sort(df_scores)[3]
th
df[df_scores<th]
df[df_scores < th].shape
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index
df[df_scores< th].drop(axis=0, labels=df[df_scores < th].index)

#baskılamak için gözlemler kullanılır.

