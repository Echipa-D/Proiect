

""""Activarea pachetelor"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import string
import re

"""Importarea bazei de date"""

def load_data():
    data = pd.read_csv('all-data-RO.csv', sep=',', encoding='latin-1',names = ["categorie","comentariu"])
    return data
print(data)

"""Prezentarea variabilelor"""
tweet_df = load_data()
df=load_data()
tweet_df.head()

print(tweet_df.shape)
print("COLUMN NAMES" , tweet_df.columns)

print(tweet_df.info())




