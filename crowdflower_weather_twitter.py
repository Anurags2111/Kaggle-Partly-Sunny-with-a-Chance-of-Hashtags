#Anurag Sharma
# ML/DL 1st hackathon code
#Task: https://www.kaggle.com/c/crowdflower-weather-twitter

#Step 1: importing datasets
import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Step 2: Cleaning train
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(len(train)):
    tweet = re.sub('[^a-zA-Z]', ' ', train['tweet'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)
    
#Step 3: Creating the Bag of Words model (train)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)
X_train = cv.fit_transform(corpus).toarray()

y_train = train.iloc[:, 4:].values

from sklearn.preprocessing import StandardScaler
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

#Step 4: Cleaning test
corpus = []
for i in range(len(test)):
    tweet = re.sub('[^a-zA-Z]', ' ', test['tweet'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)
    
#Step 5: Creating the Bag of Words model (test)
cv = CountVectorizer(max_features=1000)
X_test = cv.fit_transform(corpus).toarray()

#Step 6: Naive Bayes Classification
R = []
for i in range(24):
    xx = y_train[:, i]
    xx =xx.astype(np.int64)
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, xx)
    y_pred = classifier.predict(X_test)
    R.append(y_pred)

Result = np.transpose(R)

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
Res_Scaled = min_max_scaler.fit_transform(Result)

tmp = []

for j in range(len(Res_Scaled)):
    tmp = Res_Scaled[j, 0:5]
    pos = np.argmax(tmp)
    
    #selecting 'sentiment' label with max score and writing rest sentiments=0
    for k in range(5):
        if k != pos:
            Res_Scaled[j][k]=0
    
    #selecting 'when' label with max score and writing rest 'when' labels=0
    tmp = Res_Scaled[j, 5:9]
    pos = np.argmax(tmp)
    for k in range(5,9):
        if k != pos:
            Res_Scaled[j][k]=0

Preditions = Res_Scaled #Predicted Result





    