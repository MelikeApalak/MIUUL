##IMDB Movie Scoring & Sorting

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
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

df["vote_count_score"] = MinMaxScaler(feature_range=(1,10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values("average_count_score",ascending= False).head(20)

#IMDB Weighted Rating

#weighted_rating = (v/(v+M)* r) + (M/(v+M) * C)
#sol tarafın görevi filmlerin aldığı oy sayısı açısından bakarak
#gereken min oy sayısı ile işlem yapmak.
#sağ taraf bütün kitlenin ortalaması
# r= vote average
# v=vote count
# M = minimum votes required to be listed in the top 250
# C = the mean vote acrros the whole report (currently 7.0)

#film 1: r=8 , M=500, v=1000
#(1000 / (1000+500))*8 = 5.33
# 500 / (1000+500) * 7 = 2.33
#toplam : 5.33 + 2.33 = 7.66

#film2 : r=8, M=500, v=3000
#(3000 / (3000+500))*8= 6.85
#500 / (3000+5000) *7 = 1
#toplam : 6.85 + 1 = 7.85

M = 2500
C = df['vote_average'].mean()
def weighted_rating(r,v,M,C):
    return (v/(v+M)* r) + (M/(v+M) * C)

df.sort_values("average_count_score",ascending=False).head(20)

weighted_rating(7.4000,11444.00000,M,C)
weighted_rating(8.10000,14075.00000,M,C)
weighted_rating(8.50000,8358.00000,M,C)
df["weighted_rating"] = weighted_rating(df["vote_average"],
                                         df["vote_count"],M,C)
df.sort_values("weighted_rating",ascending=False).head(10)

##Bayesian Average Rating Score

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

# 1 yıldızdan 10 yıldıza kadar kaç tane almış sorusunun bilgisi(puan dağılımları)
bayesian_average_rating([34733,4355,4704,6561,13515,26183,87368,273082,600260,1295351])
bayesian_average_rating([37128,5879,6268,8419,16603,30016,78538,199430,402518,837905])

df= pd.read_csv("measurement_problems/datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]
df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one","two","three","four","five",
                                                                "six","seven","eight","nine","ten"]]),axis=1)

df.sort_values("bar_score",ascending=False).head(20)

