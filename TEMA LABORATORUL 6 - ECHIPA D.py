


# ------------------------ ACTIVARE PACHETELOR UTILIZATE--------------------------------

import string
import nltk
nltk.download('stopwords')
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

#------------------------IMPORTARE BAZA DE DATE -------------------------------

tweet_df = pd.read_excel('data.xls', sep=';', encoding='latin-1', header=None, names = ["categorie","comentariu"])


    
#------------------------TASK 1 - TOKENIZARE-----------------------------------


import pandas as pd
from nltk import word_tokenize
tweet_df = pd.read_excel('data.xls', sep=';', encoding='latin-1', header=None, names = ["categorie","comentariu"])
tweet_df["token"] = tweet_df["comentariu"].apply(word_tokenize)
print(tweet_df["token"])        
        
#------------------TASK 2 - IMPLEMENT LEMNATIZATION ON YOUR CORPUS ------------
   
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

tweet_df["text_lemmatized"] = tweet_df["comentariu"].apply(lambda text: lemmatize_words(text))
tweet_df.head()
     
#----------------------------TASK 3 - PART OF SPEECH FUNCTION -----------------

from nltk.corpus import wordnet


def get_simple_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
#-------------------------------DATASET CLEANING ------------------------------

def clean_text(text):
    #lower text
    text = text.lower()
    # tokenize text and remove punctuation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('romanian')
    text = [x for x in text if not x in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_simple_pos(t[1])) for t in pos_tags]
    #remove special characters
    text = [re.sub('\W+',' ', word ) for word in text]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# clean text data
tweet_df["comentariu"] = tweet_df["comentariu"].apply(lambda x: clean_text(x))
print(tweet_df["comentariu"])

    
# ---------------------------POST TAG TEXT------------------------------------

tagged_tokens=[]
for token in tweet_df["token"]:
    tagged_tokens.append(nltk.pos_tag(token))

print(tagged_tokens)
--------------------------------

    from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='romanian',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(tweet_df['comentariu'])

#----- TASK 4 - EXTRACT THE MOST FREQUENT WORDS OUT OF YOUR CORPUS------------
top_ten=[]
from nltk.probability import FreqDist
fdist = FreqDist(tweet_df["token"])
top_ten = fdist.most_common(10)
print(top_ten)

# -----------------TASK 5 - MOST FREQUENT PART OF SPEECH TAGS----------------- 
from nltk.probability import FreqDist
most_frequent = nltk.FreqDist(tag for (word, tag) in tagged_tokens)
print(most_frequent)


#get most frequent word
def getMostFrecvWords(list_pos_tags, top):
    words = [word for (word, tag) in list_pos_tags]
    freq_dist = FreqDist(words)
    return freq_dist.most_common(top)







    