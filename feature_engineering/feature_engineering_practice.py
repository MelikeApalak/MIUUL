from datetime import date
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f' % x)
pd.set_option('display.width',500)

def load():
    data = pd.read_csv(r"feature_engineering/datasets/titanic.csv")
    return data
df = load()
df.head()

df.columns = [col.upper() for col in df.columns]

# feature engineering

#cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
#name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
#name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
#name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x:len([x for x in x.split() if x.startswith("Dr")]))
#name title
df["NEW_TITLE"] = df.NAME.str.extract(' ([A-Za-z]+)\.',expand=False)
#family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] +1
#age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
#is alone
df.loc[((df["SIBSP"] + df["PARCH"]) >0), "NEW_IS_ALONE"] ="NO"
df.loc[((df["SIBSP"] + df["PARCH"]) == 0), "NEW_IS_ALONE"] ="YES"
#age level
df.loc[(df["AGE"] <18), "NEW_AGE_CAT"] ="young"
df.loc[(df["AGE"] >=18), "NEW_AGE_CAT"] ="mature"
df.loc[(df["AGE"] >=56), "NEW_AGE_CAT"] ="senior"

#sex x age
df.loc[(df["SEX"]=='male') & (df["AGE"]<=21),"NEW_SEX_CAT"] ='youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE']) <=50, 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 51), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE']) <=50, 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 51), 'NEW_SEX_CAT'] = 'seniorfemale'
df.head()
df.shape



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

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#outliers
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
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

for col in num_cols:
    print(col,check_outlier(df,col))

for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:
    print(col,check_outlier(df,col))

#missing values

def missing_values_table(dataframe,na_name= False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe.isnull().sum() / dataframe.shape[0]* 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss,np.round(ratio,2)], axis=1, keys=['n_miss','ratio'])
    print(missing_df,end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

df.drop("CABIN", inplace=True,axis=1)

remove_cols = ["TICKET","NAME"]
df.drop(remove_cols,inplace=True,axis=1)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

df["NEW_AGE_PCLASS"] = df["AGE"]* df["PCLASS"]


df.loc[(df["AGE"] <18), "NEW_AGE_CAT"] ="young"
df.loc[(df["AGE"] >=18), "NEW_AGE_CAT"] ="mature"
df.loc[(df["AGE"] >=56), "NEW_AGE_CAT"] ="senior"

df.loc[(df["SEX"]=='male') & (df["AGE"]<=21),"NEW_SEX_CAT"] ='youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE']) <=50, 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 51), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE']) <=50, 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 51), 'NEW_SEX_CAT'] = 'seniorfemale'

df = df.apply(lambda x:x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <=10) else x,axis=0)

#label encoding
binary_cols = [col for col in df.columns if df[col].dtype not in [int,float]
               and df[col].nunique()==2]

def label_encoder(dataframe,binary_col):
    label_encoder=LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoder(df,col)

#rare encoding
def rare_analyser(dataframe,target,cat_cols):
    for col in cat_cols:
        print(col,":",len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(), #kat. değişkenin kaç sınıfı var
                            "RATIO": dataframe[col].value_counts() / len(dataframe), #oranları
                           "TARGET_MEAN" : dataframe.groupby(col)[target].mean()}),end="\n\n\n")

def rare_encoder(dataframe,rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == '0'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

rare_analyser(df,"SURVIVED",cat_cols)
df = rare_encoder(df,0.01)

df["NEW_TITLE"].value_counts()

#One-Hot Encoding

def one_hot_encoder(dataframe,categorical_cols,drop_first=False):
    dataframe =pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10>= df[col].nunique() > 2]
df=one_hot_encoder(df,ohe_cols)

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]
rare_analyser(df,"SURVIVED",cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df)< 0.01).any(axis=None)]
#df.drop(useless_cols, axis=1, inplace=True)

#standart scaler
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()
df.head()
df.shape

#model
y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred,y_test)


#hiçbir işlem yapılmadan elde edilecek accuracy_score ?
dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff,columns=["Sex","Embarked"], drop_first=True)
y = dff["Survived"]
X= dff.drop(["PassengerId", "Survived", "Name", "Ticket","Cabin"],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train,y_train)

