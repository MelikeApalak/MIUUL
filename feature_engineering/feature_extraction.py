#özellik çıkarımı (feature extraction)
#ham veriden değişken üretmek
from datetime import date
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
import seaborn as sns
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f' % x)
pd.set_option('display.width',500)

def load():
    data = pd.read_csv(r"feature_engineering/datasets/titanic.csv")
    return data
df = load()
df.head()
#dolulara 1, boşlara 0 yazıldı.
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

test_stat, pvalue = proportions_ztest(count= [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                              df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs= [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


df.loc[((df['SibSp'] + df['Parch']) > 0) , "NEW_IS_ALONE"] ="NO"
df.loc[((df['SibSp'] + df['Parch']) == 0) , "NEW_IS_ALONE"] ="YES"


df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

#textler üzerinden özellik türetmek

df.head()

#harfleri saydırma
#letter count

df["NEW_NAME_COUNT"] = df["Name"].str.len()

#word count
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

#özel yapıları yakalamak

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

#regex ile değişken üretmek(regular expression)

df.head()
df['NEW_TITLE'] = df.Name.str.extract( ' ([A-Za-z]+)\.', expand=False)
df[["NEW_TITLE","Survived","Age"]].groupby(["NEW_TITLE"]).agg({"Survived" : "mean", "Age": ["count","mean"]})

#date değişkenleri üretmek
dff = pd.read_csv(r"feature_engineering/datasets/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")
#year
dff['year'] = dff['Timestamp'].dt.year

#month
dff['month'] = dff['Timestamp'].dt.month

#year diff
dff['year_diff']= date.today().year - dff['Timestamp'].dt.year

#iki tarih arasındaki ay farkı
#öncelikle yıl farkı hesaplanıp 12 ile çarpılarak, ay cinsine cevrilmeli
dff['month_diff']= (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month

#day name
dff['day_name'] = dff['Timestamp'].dt.day_name()
dff.head()

#feature interaction (özellik etkileşimi)
#değişkenlerin birbiriyle etkileşime girmesi
df = load()
df.head()
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21),'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age']) <=50, 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 51), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age']) <=50, 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 51), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.groupby("NEW_SEX_CAT")["Survived"].mean()

