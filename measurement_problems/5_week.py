#kullanıcı ve zaman ağırlıklı kurs puanı hesaplama

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.float_format',lambda x: '%.5f' % x)

df= pd.read_csv("measurement_problems/datasets/course_reviews.csv")
df.head()
df.shape
df["Rating"].value_counts()
df["Questions Asked"].value_counts()
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating":"mean"})

#ortalama puan
df["Rating"].mean()

#puan zamanlarına göre ağırlıklı ortalama hesabı(time-based weigthed average)
df.head()
df.info() #timestamp değişkeni-> object. bunu zamana çevirmeliyiz.
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

#yorumları gün cinsinden ifade etmemiz gerekir. (3 gün önce gibi)
#bu günün tarihinden yorumların yapıldığı günün tarihini çıkarırız.

current_date = pd.to_datetime('2021-02-10 0:0:0')
df["days"] = (current_date - df["Timestamp"]).dt.days

#son 30 günde yapılan yorumların puan ortalaması
df.loc[df["days"] <=30, "Rating"].mean()
df.loc[(df["days"]>30) & (df["days"]<=90),"Rating"].mean()
df.loc[(df["days"]>90) & (df["days"]<=180),"Rating"].mean()
df.loc[(df["days"]>180) ,"Rating"].mean()

#zamana göre ağırlıklı ortalama (güncel olan yorumlara en fazla ağırlığı verdik)
df.loc[df["days"] <=30, "Rating"].mean() * 28/100 + \
df.loc[(df["days"]>30) & (df["days"]<=90),"Rating"].mean() * 26/100 + \
df.loc[(df["days"]>90) & (df["days"]<=180),"Rating"].mean() *24/100 + \
df.loc[(df["days"]>180) ,"Rating"].mean() * 22/100

#ilgili verinin zamana göre ağırlıklı ortalamasını getiren fonksiyon
def time_based_weigthed_average(dataframe,w1=28,w2=26,w3=24,w4=22):
    return dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 / 100 + \
        dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
        dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
        dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

time_based_weigthed_average(df)
time_based_weigthed_average(df,30,26,22,22)

#tüm kullanıcılar aynı ağırlığa mı sahip olmalı ?

df.groupby("Progress").agg({"Rating": "mean"})
df.loc[df["Progress"] <=10, "Rating"].mean() * 22/100 + \
    df.loc[(df["Progress"]>10) & (df["Progress"]<=45),"Rating"].mean() * 24/100 + \
    df.loc[(df["Progress"]>45) & (df["Progress"]<=75),"Rating"].mean() *26/100 + \
    df.loc[(df["Progress"]>75) ,"Rating"].mean() * 28/100

#fonksiyonlaştırma
def user_based_weigthed_average(dataframe,w1=22,w2=24,w3=26,w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
        dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
        dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
        dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100


user_based_weigthed_average(df,20,24,26,30)

#Ağırlıklı Ortalama
#time-based & user-based bir araya getirilecek.
def course_weigthed_rating(dataframe,time_w = 50,user_w=50):
    return time_based_weigthed_average(dataframe) * time_w /100 +\
    user_based_weigthed_average(dataframe) * user_w/100

course_weigthed_rating(df)
course_weigthed_rating(df,time_w=40,user_w=60)

#ÜRÜN SIRALAMA
#UYGULAMA : (KURS SIRALAMA)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.float_format',lambda x: '%.5f' % x)

df = pd.read_csv("measurement_problems/datasets/product_sorting.csv")
print(df.shape)
df.head(10)
df.sort_values("rating",ascending=False).head(20)

#yorum sayısı ve satın alma sayısına göre sıralama
df.sort_values("purchase_count",ascending=False).head(20)
#satın almaya göre sıraladığımızda yorumu ve puanı az olan ücretsiz
#kurslar da gelebilir. bu yüzden yeterli değildir.
df.sort_values("commment_count",ascending=False).head(20)

#bu önemli 3 faktörü tek bir çizgiye çekip standarlaştırmalıyız.
#(puan, yorum, satın alma)

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1,5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df.describe().T
df["comment_count_scaled"] = MinMaxScaler(feature_range=(1,5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

(df["comment_count_scaled"]*32/100+
 df["purchase_count_scaled"]*26/100+
 df["rating"]*42/100) # output =>birçok faktörün ağırlığı ile oluşturulmuş skorlar.

def weigthed_sorting_score(dataframe,w1=32,w2=26,w3=42):
    return (dataframe["comment_count_scaled"]*w1/100+
    dataframe["purchase_count_scaled"]*w2/100+
    dataframe["rating"]*w3/100)

df["weigthed_sorting_score"] = weigthed_sorting_score(df)
df.sort_values("weigthed_sorting_score",ascending=False).head(20)

#df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weigthed_sorting_score",ascending=False).head(20)

##BAYES ORTALAMA DERECELENDİRME (BAYESIAN AVERAGE RATING SCORE)
#sorting products with 5 star rated
def bayesian_average_rating(n,confidence=0.95):
    #örneğin n = 5 ise 1,2,3,4,5 yıldızlardan kaç değer girildiğini tutar.
    #puan dağılımlarının üzerinden ağırlıklı bir şekilde
    #olasılıksal ortalama hesabı yapar.
    import math
    import scipy.stats as st
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
df.sort_values("weigthed_sorting_score",ascending=False).head(20)
df.sort_values("bar_score",ascending=False).head(20)

df[df["course_name"].index.isin([5,1])].sort_values("bar_score",ascending=False)

##Karma Sıralama (BAR score + diğer değerler)

def hybrid_sorting_score(dataframe,bar_w=60,wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                 "2_point",
                                                                 "3_point",
                                                                 "4_point",
                                                                 "5_point"]]),axis=1)
    wss_score = weigthed_sorting_score(dataframe)
    return bar_score*bar_w/100 + wss_score*wss_w/100

df["hybrid_sorting_score"] = hybrid_sorting_score(df)
df.sort_values("hybrid_sorting_score",ascending=False).head(20)

##IMDB Movie Scoring & Sorting

import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns',None)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.float_format',lambda x: '%.5f' % x)

df = pd.read_csv("measurement_problems/datasets/movies_metadata.csv",
                 low_memory=False) #DtypeWarning kapamak için

df= df[["title","vote_average","vote_count"]]
df.head()
df.shape

#Vote Average'a Göre Sıralama
df.sort_values("vote_average",ascending=False).head(20)
df["vote_count"].describe([0.10,0.25,0.50,0.70,0.80,0.90,0.95,0.99]).T

#oy sayısı 400den büyük olanları sıralama
df[df["vote_count"]>400].sort_values("vote_average",ascending=False).head(20)

from sklearn.preprocessing import MinMaxScaler

df["vote_count_score"] = MinMaxScaler(feature_range=(1,10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values("average_count_score",ascending= False).head(20)

#IMDB Weigthed Rating

#weigthed_rating = (v/(v+M)* r) + (M/(v+M) * C)
#sol tarafın görevi filmlerin aldığı oy sayısı açısından bakarak
#gereken min oy sayısı ile işlem yapmak.
#sağ taraf bütün kitlenin ortalaması
# r= vote average
# v=vote count
# M = minimum votes required to be listed in the top 250
# C = the mean vote acrros the whole report (currently 7.0)

M = 2500
C = df['vote_average'].mean()

def weighted_rating(r,v,M,C):
    return (v/(v+M)* r) + (M/(v+M) * C)

df.sort_values("average_count_score",ascending=False).head(10)

weighted_rating(7.4000,11444.00000,M,C)
weighted_rating(8.10000,14075.00000,M,C)
weighted_rating(8.50000,8358.00000,M,C)
df["weighted_rating"] = weighted_rating(df["vote_average"],
                                         df["vote_count"],M,C)
df.sort_values("weighted_rating",ascending=False).head(10)

#bayesian average rating score

def bayesian_average_rating(n,confidence=0.95):
    #örneğin n = 5 ise 1,2,3,4,5 yıldızlardan kaç değer girildiğini tutar.
    #puan dağılımlarının üzerinden ağırlıklı bir şekilde
    #olasılıksal ortalama hesabı yapar.
    import math
    import scipy.stats as st
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

bayesian_average_rating([34733,4355,4704,6561,13515,26183,87368,273082,600260,1295351])
bayesian_average_rating([37128,5879,6268,8419,16603,30016,78538,199430,402518,837905])

df= pd.read_csv("measurement_problems/datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]
df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one","two","three","four","five",
                                                                "six","seven","eight","nine","ten"]]),axis=1)

df.sort_values("bar_score",ascending=False).head(20)

#YORUM SIRALAMA(SORTING REVİEWS)

#üst-alt farkı skoru (up-down diff score)

#review 1 : 600 up and 400 down total 1000
#review 2 : 5500 up and 4500 down total 10000

def score_up_down_diff(up,down):
    return up-down

score_up_down_diff(600,400)
score_up_down_diff(5500,4500)

#average rating (ortalama puan)
#score = average rating = up_rating / all_rating
def score_average_rating(up,down):
    if up+down == 0:
        return 0
    return up/ (up+down)

score_average_rating(600,400)
score_average_rating(5500,4500)

#review 1: 2 up 0 down total 2
#review 2 : 100 up 1 down total 101

score_average_rating(2,0)
score_average_rating(100,1)


#wilson lower bound score
#600-400 0.6

def wilson_lower_bound(up,down,confidence=0.95):
    """
    wilson lower bound score
    - bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB soku olarak kabul edilir.
    - hesaplanacak skor ürün sıralaması için kullanılır.
    - not :
    eğer skorlar 1-5 arasındaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulliye uygun hale getirilir.
    bu beraberinde bazı problemleri de getirir. bu sebeple bayesian average rating yapmak gerekir.

    :param up: int
        up count
    :param down: int
        down count
    :param confidence: float
    :return: wilson score: float
    """

    n = up + down
    if n==0:
        return 0
    z=st.norm.ppf(1-(1-confidence)/2)
    phat = 1.0 * up/n
    return (phat + z *z / (2*n)- z*math.sqrt((phat*(1-phat) + z*z / (4*n)) /n)) / (1+z*z/n)

wilson_lower_bound(600,400)
wilson_lower_bound(5500,4500)
wilson_lower_bound(2,0)
wilson_lower_bound(100,1)

#case study
import pandas as pd
up= [15,70,14,4,2,5,8,37,21,52,28,147,61,30,23,40,37,61,54,18,12,68]
down = [0,2,2,2,15,2,6,5,23,8,12,2,1,1,5,1,2,6,2,0,2,2]
comments = pd.DataFrame({"up":up,"down":down})

comments["score_pos_neg_diff"] = comments.apply(lambda x:score_up_down_diff(x["up"],
                                                                            x["down"]),axis=1)

comments["score_average_rating"] = comments.apply(lambda x:score_average_rating(x["up"],
                                                                            x["down"]),axis=1)
comments["wilson_lower_bound"] = comments.apply(lambda x:wilson_lower_bound(x["up"],
                                                                            x["down"]),axis=1)

comments.sort_values("wilson_lower_bound",ascending=False)