#encoding -> değişkenlerin temsil şekillerinin değiştirilmesi

#label encoding -> labelları yeniden kodlamak demektir.kullanacak
#olduğumuz yöntemlerin anlayacağı dile çevirme işlemi.
#bir kategorik değşikenin sınıfları labellardır.
#string ifadeleri 0,1 olarak ifade etmektir.

#ENCODING (LABEL ENCODING, ONE-HOT ENCODING, RARE ENCODING)
#label encoding & binary encoding
#(kategorik değişken , 2den fazla sınıfa sahip)
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
import seaborn as sns
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f' % x)
pd.set_option('display.width',500)

def load():
    data = pd.read_csv(r"C:\Users\MelikeApalak\Desktop\feature_engineering\datasets\titanic.csv")
    return data

def load_application_train():
    data = pd.read_csv(r"C:\Users\MelikeApalak\Desktop\feature_engineering\datasets\application_train.csv")
    return data

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

df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5] #-> alfabetik sıraya göre
le.inverse_transform([0,1])

#fonksiyonlaştırma
def label_encoder(dataframe,binary_col):
    label_encoder=LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

#yaygın problem ölcekleniyor yapıyor olmak. (elimizde yüzlerce değişken olduğunda ne olacak)
#iki sınıflı kategorik değişkenleri secmenin yolunu bulmalıyız.

binary_cols = [col for col in df.columns if df[col].dtype not in [int,float]
               and df[col].nunique()==2]

#unique methodu değşikenin içindeki eksik değerleri de sınıf olarak görür.
#ilgili değişkenin unique ine bakılır.

for col in binary_cols:
    label_encoder(df,col)

df.head()

df = load_application_train()
df.shape

#değişkenlerin hangisi 2 sınıflı kategorik değişken sorgusu
binary_cols = [col for col in df.columns if df[col].dtype not in [int,float]
               and df[col].nunique()==2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df,col)
    #eksik değerlere sahip kısma 2 değerini verdi.(NaN)

df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique() #eksik değeri göz önünde bulundurmaz.
len(df["Embarked"].unique()) #NaN değerini de sayar.

#One Hot Encoding
#sınıflar arası fark olmadığında label encoding uygulamak yanlış olacaktır.
#sınıflar değişkenlere dönüşür. (her biri sütunlara dönüşür)
        #gs         #fb       #bjk
#gs      1          0          0
#fb      0          1          0
#bjk     0          0          1

#ilk sınıfı drop etmek gerekir. (dropfirst) (birbiri üzerinden oluşturulma durumu kaldırılmaya çalışılır)
#dummy değişken tuzağı

df= load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df,columns=["Embarked"]).head()
pd.get_dummies(df,columns=["Embarked"],drop_first=True).head() #bir sınıfı siler
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head() #ilgili değişkendeki eksik değerler sınıf olarak oluşturulur, gelir.

pd.get_dummies(df,columns=["Sex","Embarked"],drop_first=True).head()
#değişkenin sınıf sayısı 2 ise drop_first=True yapınca değişken binary encode edilirç

def one_hot_encoder(dataframe,categorical_cols,drop_first=False):
    dataframe =pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return dataframe


df = load()
cat_cols,num_cols, cat_but_car = grab_col_names(df)

#kategorik değişkenlerin hepsini one hot encoderdan geçirebiliriz.

#eşsiz değer sayısı ikiden büyük olsun.
ohe_cols = [col for col in df.columns if 10>= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()
df.head()

#rare encoding
#eşik değer belirlenir, frekansı o eşik değerden düşük olan sınıflar bir araya getirilir.

#rare encoding uygulama
#1. kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
#2. rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
#3. rare encoder yazacağız.

#1. kategorik değişkenlerin azlık çokluk durumunun analizi
df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("################")

    if plot:
        sns.countplot(x=dataframe[col_name], data = dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df,col)

#rare kategoriler ile bağımlı değişkenler arasındaki ilişkinin analiz edilmesi
df["NAME_INCOME_TYPE"].value_counts()
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

def rare_analyser(dataframe,target,cat_cols):
    for col in cat_cols:
        print(col,":",len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(), #kat. değişkenin kaç sınıfı var
                            "RATIO": dataframe[col].value_counts() / len(dataframe), #oranları
                           "TARGET_MEAN" : dataframe.groupby(col)[target].mean()}),end="\n\n\n")

rare_analyser(df,"TARGET",cat_cols)

#rare encoder yazılması
def rare_encoder(dataframe,rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == '0'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

new_df = rare_encoder(df,0.01)
rare_analyser(new_df,"TARGET",cat_cols)
df["occupatıon_type"].value_counts()

#özellik ölçeklendirme (feature scaling)
#StandardScaler : klasik standartlaştırma.
#tüm gözlem birimlerinden ortalamayı çıkar, standart sapmaya böl. z = (x-u) / s
#z standartlaştırılması

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

#RobustScaler : medyanı çıkar iqr'a böl.
#aykırı değerlere karşı daha dayanıklı.(etkilenmiyor)

rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

#MinMaxScaler : verilen iki değer arasında değişken dönüşümü
#x_std = (x- x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
#x_scaled = x_std * (max-min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T
df.head()

age_cols = [col for col in df.columns if "Age" in col]

#sayısal değişkenlerin çeyrek değerlerini gösterip grafiğini oluşturur.
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df,col,plot=True)

#numeric to categorical: sayısal değişkenleri kategorik değişkenlere çevirme
#binning
df["Age_qcut"] = pd.qcut(df['Age'], 5)


