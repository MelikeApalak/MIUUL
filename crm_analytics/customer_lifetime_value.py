# CUSTOMER LIFETIME VALUE

#veri setindeki değişkenler:

#InvoiceNo : Fatura numarası. Her işleme yani faturaya ait eşssiz numara. C ile başlıyorsa iptal edilen işlem.
#StockCode : Ürün kodu. Her bir ürün için eşsiz numara.
#Description : Ürün ismi
#Quantity : Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını gösterir.
#InvoiceDate : Fatura tarihi ve zamanı
#UnitPrice : Ürün fiyatı (sterlin)
#CustomerID : Eşsiz müşteri numarası
#Country : müşterinin yaşadığı ülke ismi.

# 1.VERİ HAZIRLAMA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns',None)
pd.set_option('display.float_format', lambda x : '%.5f' % x)

df_ = pd.read_excel('crm_analytics/datasets/online_retail_II.xlsx',sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.shape
df.isnull().sum()

#gözlem birimindeki invoice değişkeninde c varsa iptal edilen ürün demektir.
# başında c olan ürünleri silmemiz gerekir.

df = df[~df["Invoice"].str.contains("C",na=False)]
df.describe().T

df = df[(df["Quantity"] > 0)]
#customerID deki eksik değerleri yok ediyoruz.
df.dropna(inplace=True)

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                        'Quantity':lambda x : x.sum(),
                                        'TotalPrice': lambda x : x.sum()})

cltv_c.columns = ['total_transaction','total_unit','total_price']
cltv_c.head()

# 2. Averege Order Value
cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

# 3. Purchase Frequency
cltv_c.head()
cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0] #(total_number_of_customers)

cltv_c.shape #dataframe satırları eşsiz müşterileri temsil ediyor. customer ıd.

# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri / tüm müşteriler)

repeat_rate = cltv_c[cltv_c["total_transaction"]>1].shape[0] / cltv_c.shape[0]
churn_rate = 1- repeat_rate
#sadece shape gözlem ve değişken
#sayısını birlikte verir. 0. index eşsiz müşteri sayısını verir.

# 5. Profit Margin (profit_margin = total_price * 0.10)

cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10

# 6. Customer Value (customer_value = average_order_value * purchase_frequency)

cltv_c['customer_value']= cltv_c['average_order_value'] * cltv_c['purchase_frequency']

# 7. Customer Lifetime value ( CLTV = (customer_value / churn_rate) x profit_margin)

cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

cltv_c.sort_values(by="cltv",ascending = False).head()

# 8. Segmentlerin oluşturulması
cltv_c.sort_values(by="cltv",ascending = False).tail() # cltv açısından önem düzeyleri

cltv_c["segment"] = pd.qcut(cltv_c["cltv"],4,labels = ["D","C","B","A"])
cltv_c.sort_values(by="cltv",ascending = False).head()

cltv_c.groupby("segment").agg({"count","mean","sum"})

cltv_c.to_csv("cltv_c.csv")

# 9. tüm işlemlerin fonksiyonlaştırılması

def create_cltv_c(dataframe,profit=0.10):

    #veri hazırlama
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe["Quantity"] > 0)]
    dataframe.dropna(inplace=True)
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

    cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                            'Quantity': lambda x: x.sum(),
                                            'TotalPrice': lambda x: x.sum()})
    cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']
    #avg_order_value
    cltv_c["avg_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]
    #purchase frequency
    cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]  # (total_number_of_customers)
    #repeat rate & churn rate
    repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate
    #profit margin
    cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10
    #customer value
    cltv_c['customer_value']= cltv_c['avg_order_value'] * cltv_c['purchase_frequency']
    #customer life time value
    cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]
    #segment
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c


df= df_.copy()
clv = create_cltv_c(df)