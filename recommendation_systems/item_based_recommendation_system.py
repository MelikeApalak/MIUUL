#Item-Based Collaborative Filtering

#1. veri setinin hazırlanması
#2. user movie df'nin oluşturulması
#3. Item-Based film önerilerinin yapılması
#4. çalışma scriptinin hazırlanması

#veri seti hazırlama
import pandas as pd
pd.set_option('display.max_columns',500)

movie = pd.read_csv('recommendation_systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('recommendation_systems/datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating,how="left",on="movieId") #dataframe birleştirme
df.head()

#user movie df oluşturulması

df.head()
df.shape

df["title"].nunique() #eşsiz film sayısı

df["title"].value_counts().head()

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape
common_movies["title"].nunique()
df["title"].nunique()

#satırlara kullanıcılar, sütunlara film isimleri getirilecek
user_movie_df = common_movies.pivot_table(index=["userId"],columns=["title"],values="rating")

#item-based film önerilerinin yapılması

movie_name = "Matrix, The (1999)"

movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

#rastgele film seçimi ve sonrasında önerimi
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

def check_film(keyword,user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Insomnia", user_movie_df)

#4. script hazırlama
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('recommendation_systems/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('recommendation_systems/datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()
def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)",user_movie_df)

