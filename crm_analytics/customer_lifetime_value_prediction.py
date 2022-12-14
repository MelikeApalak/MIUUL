#BG-NBD & GAMMA-GAMMA ile CLTV Prediction

#1. VERİNİN HAZIRLANMASI
#2. BG-NBD MODELİ İLE EXPECTED NUMBER OF TRANSACTION
#3. GAMMA-GAMMA MODELİ İLE EXPECTED AVERAGE PROFIT
#4. BG-NBD VE GAMMA-GAMMA MODELİ İLE CLTV HESAPLANMASI
#5. CLTV'YE GÖRE SEGMENTLERİN OLUŞTURULMASI
#6. ÇALIŞMANIN FONKSİYONLAŞTIRILMASI

#Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
#pazarlama stratejileri belirlemek istiyor.

#veri setindeki değişkenler:

#InvoiceNo : Fatura numarası. Her işleme yani faturaya ait eşssiz numara. C ile başlıyorsa iptal edilen işlem.
#StockCode : Ürün kodu. Her bir ürün için eşsiz numara.
#Description : Ürün ismi
#Quantity : Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını gösterir.
#InvoiceDate : Fatura tarihi ve zamanı
#UnitPrice : Ürün fiyatı (sterlin)
#CustomerID : Eşsiz müşteri numarası
#Country : müşterinin yaşadığı ülke ismi.

#gerekli kütüphaneler
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
pd.set_option('display.float_format',lambda x : '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


#aykırı değerleri tespit edip baskılama yöntemiyle belirlenen aykırı değerleri
#belirli bir eşik ile değiştireceğiz.
#değişkenin genel davranışının dışında olan değerleri baskılamak gerekiyor.
def outlier_thresholds(dataframe,variable):
    #kendine girilen değişken için eşik değer belirler.
    # %25 lik ve %75 lik çeyrek değer hesaplanır.
    #önce çeyrek değerleri hesaplarız.
    #3. çeyrek değerin 1.5 iqr üstü
    #1. çeyrek değerin 1.5 iqr altındaki değerler üst ve alt eşik değerler ile belirlenir.
    quartile1 = dataframe[variable].quantile(0.01) #%25
    quartile3 = dataframe[variable].quantile(0.99) #%75
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit,up_limit

def replace_with_thresholds(dataframe,variable):
    #aykırı değer baskılama
    #aykırı değerleri üst ve alt limitlere atar.
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    #dataframe.loc[(dataframe[variable]< low_limit),variable] = low_limit
    dataframe.loc[(dataframe[variable]> up_limit), variable] = up_limit

#1. verinin okunması

df_ = pd.read_excel('crm_analytics/datasets/online_retail_II.xlsx',sheet_name="Year 2010-2011")

df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

# Veri Ön İşleme

df.dropna(inplace= True)
df = df[~df["Invoice"].str.contains('C',na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"]> 0]

replace_with_thresholds(df,"Quantity")
replace_with_thresholds(df,"Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]
today_date = dt.datetime(2011,12,11)
df.describe().T

#Lifetime veri yapısının hazırlanması

#recency : son satın alma üzerinden geçen zaman. haftalık.
# T : müşteri yaşı. haftalık.
# frequency : tekrar eden toplam satın alma başına ortalama kazanç
# monetart_value : satın alma başına ortalama kazanç

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max()- InvoiceDate.min()).days,
                                                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
                                                        'Invoice': lambda Invoice: Invoice.nunique(),
                                                        'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns =['recency','T','frequency','monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df.describe().T

cltv_df = cltv_df[(cltv_df['frequency']> 1)]

#haftalık cinsten ifade etmek
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7


# 2. BG-NBD modelinin kurulması
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

### 1 hafta içinde en çok satın alma bekledğiniz 10 müşteri ?
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

#BG/NBD için geçerli, gamma-gamma modeli için geçerli değil.
cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                    cltv_df['frequency'],
                                    cltv_df['recency'],
                                    cltv_df['T'])

### 1 ay içinde en çok satın alma bekledğiniz 10 müşteri ?
bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                cltv_df['frequency'],
                                                cltv_df['recency'],
                                                cltv_df['T'])

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

# 3 ayda tüm şirketin beklenen satış sayısı nedir ?
bgf.predict(4*3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()
cltv_df["expected_purc_3_month"] = bgf.predict(4*3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

#Tahmin sonuçlarının değerlendirilmesi
plot_period_transactions(bgf)
plt.show()

# 3. GAMMA- GAMMA MODELİNİN KURULMASI (Average Profit modellemesi)
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary'])


cltv_df.sort_values("expected_average_profit",ascending=False).head(10)

## 4. BG-NBD VE GG MODELİ İLE CLTV HESAPLANMASI


cltv = ggf.customer_lifetime_value(bgf,
                                cltv_df['frequency'],
                                cltv_df['recency'],
                                cltv_df['T'],
                                cltv_df['monetary'],
                                time=3, #3 aylık
                                freq = "W", #T'nin frekans bilgisi
                                discount_rate=0.01)

cltv.head()
cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv,on="Customer ID",how="left")
cltv_final.sort_values(by="clv",ascending=False).head(10)
#düzenli olan bir müşteri recency değeri arttıkça müşterinin satın alma olasılığı yaklaşıyordur.


## 5. CLTV'ye göre segmentlerin oluşturulması

cltv_final["segment"] = pd.qcut(cltv_final["clv"],4,labels=["D","C","B","A"])

cltv_final.sort_values(by="clv",ascending=False).head(50)

cltv_final.groupby("segment").agg({"count","mean","sum"})


##ÇALIŞMANIN FONKSİYONLAŞTIRILMASI

    #1. VERİ ÖN İŞLEME
def create_cltv_p(dataframe,month=3):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains('C', na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    #2. BG-NBD modelinin kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)

    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])
    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])
    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    # 3.GAMMA GAMMA MODELİNİN KURULMASI
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    #4. BG-NBD VE GG MODELİ İLE CLTV HESAPLANMASI

    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,
                                       freq="W",
                                       discount_rate=0.01)
    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final


df_ = pd.read_excel('crm_analytics/datasets/online_retail_II.xlsx',sheet_name="Year 2010-2011")
df = df_.copy()

cltv_final_2 = create_cltv_p(df)

cltv_final_2.to_csv("cltv_prediction.csv")











