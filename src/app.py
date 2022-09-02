#imports
import pandas as pd 
import regex as reg
import re
import matplotlib.pyplot as plt
import unicodedata
import nltk
import string
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection, svm
from sklearn.metrics import classification_report, accuracy_score

from nltk.corpus import stopwords

#load
df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv')

#clean
df['is_spam'] = df['is_spam'].apply(lambda x: 1 if x == True else 0)

df = df.drop_duplicates()
df = df.reset_index(inplace = False)[['url','is_spam']]

df['url'] = df['url'].str.lower()

data = df.copy()

cleaner = []

for p in range(len(data.url)):
    desc = data['url'][p]
    
    desc = re.sub('[^a-zA-Z]', ' ', desc)
    
    desc=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)
    
    desc=re.sub("(\\d|\\W)+"," ",desc)
    
    cleaner.append(desc)

data['url'] = cleaner
        
data.head()
data['url'].str.split(expand=True).stack().value_counts()[:60]

stop_words = ['http','www','com','you','your','for','not','have','is','in','im','from','to','https','e','c','v','b','f','p']

for i in data['url'].str.split(expand=True).stack().value_counts().index:
    if len(i)<3 :
        stop_words.append(i)

stop_words=list(set(stop_words))

def remove_stopwords(message):
  if message is not None:
    words = message.strip().split()
    words_filtered = []
    for word in words:
      if word not in stop_words:
        words_filtered.append(word) 
    result = " ".join(words_filtered)         
  else:
    result = None

  return result 

data['url']=data['url'].apply(remove_stopwords)
data['url'].str.split(expand=True).stack().value_counts()[:60]

X = data['url']
y = data['is_spam']
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=123)


#Vector
vec = CountVectorizer(stop_words='english')
X_train = vec.fit_transform(X_train).toarray()
X_test = vec.transform(X_test).toarray()


#Model

nb = MultinomialNB()

nb.fit(X_train, y_train)

predictions = nb.predict(X_train)
print(classification_report(y_train, predictions))

predictionstest = nb.predict(X_test)
print(classification_report(y_test, predictionstest))

message_vectorizer = CountVectorizer().fit_transform(df['url'])

X_train, X_test, y_train, y_test = train_test_split(message_vectorizer, df['is_spam'], test_size = 0.2, random_state = 121, shuffle = True)

cl = svm.SVC(C=1.0, kernel='linear', degree=4, gamma='auto')
cl.fit(X_train, y_train)

pred = cl.predict(X_train)
print(classification_report(y_train, pred))

pred = cl.predict(X_test)
print(classification_report(y_test, pred))

pickle.dump(cl, open('../models/nlp_model.pkl', 'wb'))