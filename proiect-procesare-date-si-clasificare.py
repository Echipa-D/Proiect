import pandas as pd 
import numpy as np

pd.set_option('display.max_colwidth', 4000) 
pd.set_option("display.max_rows", -1) 
 
tweet_df = pd.read_excel('news.xls', sep=';', encoding='latin-1', header=None, names = ["categorie","comentariu"])
print(tweet_df)

from nltk import word_tokenize
tweet_df["token"] = tweet_df["comentariu"].apply(lambda text: word_tokenize(text))
print(tweet_df["token"])

##################               PUNCTUATION                   ################# 

import string
import re
tweet_df.replace("-", " ")
punctuation=[",",".","!","?",":",";","/","//",'"',"'","%","(",")","„","”"]

def remove_punctuation(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text  = "".join([char for char in text if char not in punctuation])
    text = re.sub('[0-9]+', '', text)
    text = text.lower()
    return text

tweet_df['punctuation'] = tweet_df['comentariu'].apply(lambda x: remove_punctuation(x))
tweet_df["punctuation"]
tweet_df.head(10)



import spacy
nlp = spacy.load("ro_core_news_lg")

stopwords = spacy.lang.ro.stop_words.STOP_WORDS
print(stopwords)

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords])

tweet_df["text_stopwords"] = tweet_df["punctuation"].apply(lambda text: remove_stopwords(text))
tweet_df.head()


tweet_df['parsed_news'] = tweet_df["text_stopwords"].apply(lambda x: [(y.lemma_, y.pos_) for y in  nlp(x)])
print(tweet_df["parsed_news"])
    
################              FREQUENT WORDS       ###################

from collections import Counter
cnt = Counter()
for text in tweet_df["parsed_news"].values:
    for word in text:
        cnt[word] += 1
        
cnt.most_common(200)


customised_stopwords=["șia","sieși"]

def remove_customised_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in customised_stopwords])

tweet_df["news"] = tweet_df["parsed_news"].apply(lambda text: remove_customised_stopwords(text))
tweet_df.head()


################           CLASSIFICATION           ##################

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(tweet_df.parsed_news, tweet_df.categorie,test_size = 0.3 , random_state = 0)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


from sklearn.feature_extraction.text import CountVectorizer
# creating a variable for count vectorizer which gives us features using the whole text of data.
count_vec = CountVectorizer(lowercase=True, max_features=4000, ngram_range=(1,2), max_df=0.9, min_df=0)

x_train_features = count_vec.fit_transform(x_train).todense()
x_test_features = count_vec.transform(x_test).todense()
x_train_features.shape, x_test_features.shape


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

lr = LogisticRegression()
lr.fit(x_train_features, y_train)
y_pred = lr.predict(x_test_features)
print(accuracy_score(y_test,y_pred)*100)



























