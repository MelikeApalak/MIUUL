#missing values(eksik değerler)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import missingno as msno


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

def load():
    data = pd.read_csv(r"C:\Users\MelikeApalak\Desktop\feature_engineering\datasets\titanic.csv")
    return data

df = load()
df.head()

#eksik gözlem var mı sorgusu
df.isnull().values.any() #output -> true, false

#degiskenlerdeki eksik deger sayısı
df.isnull().sum()

#degiskenlerdeki tam deger sayısı
df.notnull().sum()

#veri setindeki toplam eksik deger sayısı
df.isnull().sum().sum() # -> bir satırda en az 1 tane eksik olan satır sayısı

#en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

#tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

#eksik degiskenleri azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

#yüzdelik olarak eksiklik oranları
(df.isnull().sum() / df.shape[0]* 100).sort_values(ascending=False)

#sadece eksik değere sahip degiskenlerin isimleri
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

#fonksiyonlaştırma
def missing_values_table(dataframe,na_name= False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe.isnull().sum() / dataframe.shape[0]* 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss,np.round(ratio,2)], axis=1, keys=['n_miss','ratio'])
    print(missing_df,end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
missing_values_table(df,True)

#eksik değer problemi çözme
#çözüm 1 : hızlıca silmek
#bir satırda en az 1 tane bile eksik değer varsa dropna onları uçurur.
#gözlem sayısı azalır.

df.dropna().shape

#çözüm 2 : basit atama yöntemleriyle doldurmak (ortalama,median,sabit değer)
df["Age"].fillna(df["Age"].mean())
df["Age"].fillna(df["Age"].mean()).isnull().sum() #-> kontrol

df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()


#df.apply(lambda x: x.fillna(x.mean()), axis=0) #sayısal ve kategorik değişkenler barındırdığı için hata verir.


df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "0" else x, axis=0).head()
# değişken object değil ise bu değişkeni ortalamayla doldurur,
# object ise olduğu gibi bırakır

#sayısal değişkenleri otomatik olarak doldurur.
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "0" else x, axis=0)
dff.isnull().sum().sort_values(ascending=False)

#kategorik değişkenlerin doldurma yöntemi modunu almaktır.
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
df["Embarked"].fillna("missing")

#kategorik değişkenleri otomatik olarak doldurur.
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <=10) else x, axis=0).isnull().sum()

#kategorik değişken kırılımında değer atama
df.groupby("Sex")["Age"].mean()
df["Age"].mean()

#eksik değerlere cinsiyete göre ortalama ataması yapmak mantıklıdır.
#cinsiyet kırılımı

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"] == "female")] #-> yaş değişkeninde eksiklik olup kadın olanlar
df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"),"Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"),"Age"] = df.groupby("Sex")["Age"].mean()["male"]
df.isnull().sum()

#tahmine dayalı atama ile doldurma

#eksikliğe sahip değişken -> bağımlı değişken
#eksikliğe sahip olmayan değişken -> bağımsız değişken

df = load()
cat_cols, num_cols, cat_but_car= grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()

#değişkenlerin standartlaştırılması (0-1 arasında)
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff),columns=dff.columns)
dff.head()

#knn uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff),columns=dff.columns)

df["age_imputed_knn"] = dff[["Age"]]
df.loc[df["Age"].isnull(), ["Age","age_imputed_knn"]]
df.loc[df["Age"].isnull()] #tüm değişkenler


#gelişmiş analizler

msno.bar(df)
plt.show() #veri setindeki tam olan gözlemlerin sayısı

msno.matrix(df)
plt.show() #değişkenlerdeki eksikliklerin bir arada meydana gelip gelmediğini gösterir.

msno.heatmap(df)
plt.show() #eksiklikler üzerine kurulu ısı haritasıdır.
#eksik değerlerin belirli bir korelasyon ile ortaya çıkıp çıkmadığı

#eksik değerlerin bağımlı değşikenler ile analizi

missing_values_table(df,True)
na_cols = missing_values_table(df,True)

def missing_vs_target(dataframe,target,na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col+ '_NA_FLAG'] = np.where(temp_df[col].isnull(),1,0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}),end="\n\n\n")

missing_vs_target(df,"Survived",na_cols)

