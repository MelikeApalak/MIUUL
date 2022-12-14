import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import scipy.stats as st
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.float_format',lambda x: '%.5f' % x)

df = pd.read_csv("measurement_problems/datasets/product_sorting.csv")
print(df.shape)
df.head(10)

##sorting by rating
df.sort_values("rating",ascending=False).head(20)

##sorting by comment count or purchase count
df.sort_values("purchase_count",ascending=False).head(20)
df.sort_values("commment_count",ascending=False).head(20)

##sorting by rating, comment and purchase
df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1,5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df.describe().T

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1,5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

#satın alma sayısı ile yorum sayısını ağırlıklandırabiliriz.
(df["comment_count_scaled"]*32/100+
 df["purchase_count_scaled"]*26/100+
 df["rating"]*42/100) # output =>birçok faktörün ağırlığı ile oluşturulmuş skorlar.

def weighted_sorting_score(dataframe,w1=32,w2=26,w3=42):
    return (dataframe["comment_count_scaled"]*w1/100+
    dataframe["purchase_count_scaled"]*w2/100+
    dataframe["rating"]*w3/100)

df["weighted_sorting_score"] = weighted_sorting_score(df)
df.sort_values("weighted_sorting_score",ascending=False).head(20)
#df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weigthed_sorting_score",ascending=False).head(20)

##BAYES ORTALAMA DERECELENDİRME (BAYESIAN AVERAGE RATING SCORE)
#sorting products with 5 star rated
def bayesian_average_rating(n,confidence=0.95):
    #örneğin n = 5 ise 1,2,3,4,5 yıldızlardan kaç değer girildiğini tutar.
    #puan dağılımlarının üzerinden ağırlıklı bir şekilde
    #olasılıksal ortalama hesabı yapar.
    if sum(n)==0:
        return 0
    K = len(n)
    z = st.norm.ppf(1-(1-confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k,n_k in enumerate(n):
        first_part += (k+1) * (n[k]+1) / (N+K)
        second_part += (k+1) * (k+1) * (n[k]+1) / (N+K)
    score = first_part - z * math.sqrt((second_part- first_part*first_part) / (N+K+1))
    return score

df.head()

df["bar_score"] = df.apply(lambda x : bayesian_average_rating(x[["1_point",
                                                                 "2_point",
                                                                 "3_point",
                                                                 "4_point",
                                                                 "5_point"]]),axis =1)
df.sort_values("weighted_sorting_score",ascending=False).head(20)
df.sort_values("bar_score",ascending=False).head(20)

#belirleyici ana etken sadece puanlar olduğu için sıralama tamamen doğru gelmez.
df[df["course_name"].index.isin([5,1])].sort_values("bar_score",ascending=False)

#hybrid sorting : BAR score + diğer faktörler

def hybrid_sorting_score(dataframe,bar_w=60,wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                 "2_point",
                                                                 "3_point",
                                                                 "4_point",
                                                                 "5_point"]]),axis=1)
    wss_score = weighted_sorting_score(dataframe)
    return bar_score*bar_w/100 + wss_score*wss_w/100

df["hybrid_sorting_score"] = hybrid_sorting_score(df)
df.sort_values("hybrid_sorting_score",ascending=False).head(20)











