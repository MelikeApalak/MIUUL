# content based recommendation
#film overview'larına göre tavsiye geliştirme
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#TF-IDF matrisi
#Cosine Similarity matrisi
#Benzerliklere göre öneriler

pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr',False) #çıktının tek satırda olması

df = pd.read_csv('recommendation_systems/datasets/the_movies_dataset/movies_metadata.csv', low_memory=False)
df.head()
df.shape
df["overview"].head()

def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words="english") #ölçüm değeri taşımayan kelimeleri siler.
    dataframe['overview'] = dataframe['overview'].fillna('')  # NaN değerleri siler.
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    #tfidf.get_feature_names()
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = calculate_cosine_sim(df)

def content_based_recommender(title,cosine_sim, dataframe):
    #indexleri oluşturma
    indices = pd.Series(df.index, index=df['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    #title'ın indexini yakalama
    movie_index=indices[title]
    #title'a göre benzerlik skorlarını hesaplama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index],columns=['score'])
    #kendisi hariç ilk 10 filmi getirir
    movie_indices = similarity_scores.sort_values("score",ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)
content_based_recommender("The Matrix", cosine_sim, df)

