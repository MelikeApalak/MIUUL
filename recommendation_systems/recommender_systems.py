import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr',False)

df_= pd.read_excel("recommendation_systems/datasets/online_retail_II.xlsx",
                   sheet_name="Year 2010-2011",engine="openpyxl")

df = df_.copy()

df.head()

#sadece sayısal değişkenler
df.describe().T

df.isnull().sum()
df.shape

def retail_data_prep(dataframe):
    #eksik değer, iadeler, sıfırdan küçük olma durumları.
    dataframe.dropna(inplace=True)
    dataframe=dataframe[~dataframe["Invoice"].str.contains("C",na=False)]
    dataframe = dataframe[dataframe["Quantity"]>0]
    dataframe = dataframe[dataframe["Price"]>0]
    return dataframe

df = retail_data_prep(df)
df.describe().T
df.isnull().sum()

#değişkenler için eşik değer belirleme
def outlier_thresholds(dataframe,variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1 # iqr değeri
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit,up_limit

#baskılama
def replace_with_thresholds(dataframe,variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable]<low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable]>up_limit), variable] = up_limit

#update
def retail_data_prep(dataframe):
    #eksik değer, iadeler, sıfırdan küçük olma durumları.
    dataframe.dropna(inplace=True)
    dataframe=dataframe[~dataframe["Invoice"].str.contains("C",na=False)]
    dataframe = dataframe[dataframe["Quantity"]>0]
    dataframe = dataframe[dataframe["Price"]>0]
    replace_with_thresholds(dataframe,"Quantity")
    replace_with_thresholds(dataframe,"Price")
    return dataframe

df = retail_data_prep(df)
df.isnull().sum()
df.describe().T

#ARL veri yapısı hazırlama (Invoice-Product Matrix)
df.head()

#satırlarda invoice sütunlarda ürünler
#1-0 larla ifade eden matrix yapısı

df_fr = df[df['Country'] == "France"]
df_fr.groupby(['Invoice','Description']).agg({"Quantity":"sum"}).head(20)
df_fr.groupby(['Invoice','Description']).agg({"Quantity":"sum"}).unstack().iloc[0:5,0:5]

df_fr.groupby(['Invoice','Description']). \
    agg({"Quantity":"sum"}). \
    unstack(). \
    fillna(0).iloc[0:5]
#fillna : boşluklar 0 la dolar

df_fr.groupby(['Invoice','Description']). \
    agg({"Quantity":"sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x : 1 if x>0 else 0).iloc[0:5,0:5]

#applymap ile dolu değerleri 1 e sabitledik.
#apply satır ya da sütun bilgisine göre gezer. applymap tüm gözlemleri gezer.


df_fr.groupby(['Invoice','StockCode']). \
    agg({"Quantity":"sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x : 1 if x>0 else 0).iloc[0:5,0:5]

def create_invoice_product_df(dataframe,id=False):
    if id:
        return dataframe.groupby(['Invoice','StockCode']). \
    agg({"Quantity":"sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x : 1 if x>0 else 0).iloc[0:5,0:5]

    else:
        return dataframe.groupby(['Invoice','Description']). \
    agg({"Quantity":"sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x : 1 if x>0 else 0).iloc[0:5,0:5]


fr_inv_pro_df = create_invoice_product_df(df_fr)
fr_inv_pro_df = create_invoice_product_df(df_fr,id = True)


def check_id(dataframe,stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_fr, 10002)

#birliktelik kurallarının çıkarılması

#apriori ile olası tüm ürün birlikteliklerin olasılıklarını bulmak
frequent_itemsets = apriori(fr_inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support",ascending=False)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) % (rules["lift"]>5) ]

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) % (rules["lift"]>5) ].sort_values("confidence",ascending=False)

#script
def outlier_thresholds(dataframe,variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1 # iqr değeri
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit,up_limit

#baskılama
def replace_with_thresholds(dataframe,variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable]<low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable]>up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    #eksik değer, iadeler, sıfırdan küçük olma durumları.
    dataframe.dropna(inplace=True)
    dataframe=dataframe[~dataframe["Invoice"].str.contains("C",na=False)]
    dataframe = dataframe[dataframe["Quantity"]>0]
    dataframe = dataframe[dataframe["Price"]>0]
    replace_with_thresholds(dataframe,"Quantity")
    replace_with_thresholds(dataframe,"Price")
    return dataframe

def create_invoice_product_df(dataframe,id=False):
    if id:
        return dataframe.groupby(['Invoice','StockCode']). \
    agg({"Quantity":"sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x : 1 if x>0 else 0).iloc[0:5,0:5]

    else:
        return dataframe.groupby(['Invoice','Description']). \
    agg({"Quantity":"sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x : 1 if x>0 else 0).iloc[0:5,0:5]

def check_id(dataframe,stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

def create_rules(dataframe,id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe,id)
    frequent_itemsets = apriori(dataframe,min_support=0.01,use_colnames=True)
    rules = association_rules(frequent_itemsets,metric="support",min_threshold=0.01)
    return rules

df = df_.copy()
df = retail_data_prep(df)
rules = create_rules(df)

#sepet aşamasındaki kullanıcılara ürün önerisi
#örnek: kullanıcı örnek ürün id : 22492

product_id = 22492
check_id(df,product_id)

sorted_rules = rules.sort_values("lift",ascending=False)

recommendation_list = []

for i,product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j== product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

recommendation_list[0:3]
check_id(df,22556)

def arl_recommender(rules_df,product_id,rec_count=1):
    sorted_rules=rules_df.sort_values("lift",ascending=False)
    recommendation_list = []

    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules,22492,1)