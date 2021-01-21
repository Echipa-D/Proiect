import pandas as pd 
pd.set_option('display.max_colwidth', 4000) 
pd.set_option("display.max_rows", -1) 

tweet_df = pd.read_excel('financial_news.xls', header=None, names = ["categorie","comentariu"])
print(tweet_df)


#tweet_df['comentariu'] = tweet_df['comentariu'].str.count(' ') + 1
#tweet_df['comentariu'].sum()
#############               TEXT VISUALIZATION         #####################

import seaborn as sns
sns.countplot(x="categorie",data=tweet_df)

from nltk import word_tokenize
tweet_df["token"] = tweet_df["comentariu"].apply(lambda text: word_tokenize(text))
print(tweet_df["token"])


##################               PUNCTUATION               ################# 

import re
import string

def remove_punctuation(text):
    text = re.sub('-',' ', text) 
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    text = text.lower()
    return text

tweet_df['punctuation'] = tweet_df['comentariu'].apply(lambda x: remove_punctuation(x))
tweet_df['punctuation']
tweet_df.head(10)



###################        STOPWORDS ELIMINATION     ######################


import spacy
nlp = spacy.load("ro_core_news_lg")

stopwords = spacy.lang.ro.stop_words.STOP_WORDS
print(stopwords)

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords])

tweet_df["text_stopwords"] = tweet_df["punctuation"].apply(lambda text: remove_stopwords(text))
tweet_df.head()


######################       lEMMATIZATION      ########################3

tweet_df['parsed_news'] = tweet_df["text_stopwords"].apply(lambda x: [(y.lemma_, y.pos_) for y in  nlp(x)])
print(tweet_df["parsed_news"])
    
################              FREQUENT WORDS       ###################

from collections import Counter
cnt = Counter()
for text in tweet_df["parsed_news"].values:
    for word in text:
        cnt[word] += 1
        
cnt.most_common(400)


customised_stopwords=["sieși","câtva"]

def remove_customised_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in customised_stopwords])

tweet_df["news"] = tweet_df["parsed_news"].apply(lambda text: remove_customised_stopwords(text))
tweet_df.head()


#####    label encoding

from sklearn.preprocessing import LabelEncoder
tweet_df['categorie_codificata'] = LabelEncoder().fit_transform(tweet_df['categorie'])
tweet_df[["categorie", "categorie_codificata"]] 

#number = tweet_df["news"].str.count(' ') + 1
#number.sum()

################           CLASSIFICATION           ##################

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix,classification_report

x_train,x_test,y_train,y_test = train_test_split(tweet_df.news,tweet_df.categorie_codificata,test_size = 0.3 , random_state = 0)

x_train.shape,x_test.shape,y_train.shape,y_test.shape




####            Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', MultinomialNB())])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("MULTINOMIAL NAIVE BAYES")
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

####           K- NEAREST NEIGHBOUR CLASSIFIER MODEL

from sklearn.neighbors import KNeighborsClassifier
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', KNeighborsClassifier(n_neighbors = 10,weights = 'distance',algorithm = 'brute'))])
model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("K NEAREST NEIGHBOR")
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

####           XGBOOST Classification Model
from xgboost import XGBClassifier
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', XGBClassifier(loss = 'deviance',
                                                   learning_rate = 0.01,
                                                   n_estimators = 10,
                                                   max_depth = 5,
                                                   random_state=2020))])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("XGBOOST")
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))



####      DECISION TREE CLASSIFICATION MODEL

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 10, 
                                           splitter='best', 
                                           random_state=2020))])
    
model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("DECISION TREE")
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
cm=confusion_matrix(y_test, prediction)
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

####             LOGISTIC REGRESSION 

pipe = Pipeline([('vect', CountVectorizer(lowercase=False)),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression(solver='lbfgs', multi_class='auto'))])

model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("MODEL - LOGISTIC REGRESSION")
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
classes = np.unique(y_test)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_yticklabels(labels=classes, rotation=0)
plt.show()

test = ['Banca Mondială avertizează că perspectivele pe termen scurt sunt foarte incerte şi sunt posibile diferite scenarii.',
               'Potrivit studiilor realizate, economia Chinei este deja mult mai mare decat inainte de pandemie. ',
               'Totuşi, pe tot anul, Guvernul pare optimist faţă de economişti şi estimează o creştere economică de 5,5% comparativ cu 2018']
for test in test:
    resultx = model.predict([test])
    dict = {0: 'Negativ', 1: 'Neutru', 2: 'Pozitiv'}
    print(test + '-> ' + dict[resultx[0]])

