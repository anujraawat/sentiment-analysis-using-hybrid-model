import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import partial
import sklearn

import re
import sys
import csv
from sklearn import tree
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation,GlobalMaxPooling1D
from keras.layers.embeddings import Embedding


import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier



def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    word = re.sub(r'(-|\')', '', word)
    #word = word.replace("'t", " not")
    #word = word.replace("'ll", " will")
    #word = word.replace("'m", " am")

    return word




def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))','Smiley',tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)','Laughing',tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)','Love',tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)','Wink',tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)','Sad',tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()','Cry',tweet)
    return tweet


def preprocess_tweet(tweet):
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'s", " is", tweet)
    tweet = re.sub(r"\'d", " would", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r"\'m", " am", tweet)
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', '', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = re.sub('[^A-Za-z0-9]+', ' ', tweet)
    tweet= re.sub(" \d+", " ", tweet)

    return tweet

""" Creates a dictionary with slangs and their equivalents and replaces them """
with open('slang.txt') as file:
    slang_map = dict(map(str.strip, line.partition('\t')[::2])
                     for line in file if line.strip())

slang_words = sorted(slang_map, key=len, reverse=True)  # longest first for regex
regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1)])

output=[]
f=open("testdata.csv","r+")
f2=open("ststd2.csv","w+")


str1=['','']
res=[]
data=[]
corpus=[]
data2=[]
wr=csv.writer(f2)
mycsv = csv.reader(f)


for row in mycsv:

    o1=row[0]
    o2=row[5]
    output.append((row[0],row[5]))
    str1[0]=o1
    if (o1 == '4'):
        {
            data.append(1)

        }
    if (o1 == '0'):
        {
            data.append(0)

        }
    if (o1 == '2'):
        {

        }

    if (o1 == '4'):
        {
            data2.append(1)

        }
    if (o1 == '0'):
        {
            data2.append(-1)

        }
    if (o1 == '2'):
        {
            data2.append(0)
        }



    o2 = handle_emojis(o2)
    o2=preprocess_word(o2)



    o2=preprocess_tweet(o2)
    o2 = replaceSlang(o2)
    str1[1]=o2
    tokens = word_tokenize(str1[1])
    #print(tokens)
    stop_words = stopwords.words('english')
    #print([i for i in tokens if i not in stop_words])

    str1[1]=([i for i in tokens if i not in stop_words])
    tweet = ' '.join(str1[1])
    if (o1 == '2'):
        {
        corpus.append(tweet)
        }
    else:
        res.append(tweet)
        corpus.append(tweet)
    wr.writerow(str1)

f.close()
f2.close()


#print(res)
#print(data)
#print(output)

'''
# feautre extraction
# bow
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(res)
print(bow)

# tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(res)
print(tfidf)

'''


# -----------------------------------LOGISTIC RERGRESSION-----------------------------------
bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("----------------------------------------------------------------------------------")
print("________LOGISTIC RERGRESSION________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)
hybrid = hstack([tfidf, bow])
hy = hybrid.tocsr()
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(hy, data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")

# ---------------------------------PASSIVE AGGRESSIVE CLASSIFIER--------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=0)
pac = pac.fit(X=X_train, y=y_train)
y_pred = pac.predict(X_test)
print("----------------------------------------------------------------------------------")
print("_____PASSIVE AGGRESSIVE CLASSIFIER_____")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=0)
pac = pac.fit(X=X_train, y=y_train)
y_pred = pac.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(res)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(res)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(res)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(hy, data, train_size=0.80, test_size=0.20, random_state=1234)
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=0)
pac = pac.fit(X=X_train, y=y_train)
y_pred = pac.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -------------------------------------MULTINOMIAL NB-------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
nb = MultinomialNB()
nb = nb.fit(X=X_train, y=y_train)
y_pred = nb.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________MULTINOMIAL NB__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
nb = MultinomialNB()
nb = nb.fit(X=X_train, y=y_train)
y_pred = nb.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(res)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(hy, data, train_size=0.80, test_size=0.20, random_state=1234)
nb = MultinomialNB()
nb = nb.fit(X=X_train, y=y_train)
y_pred = nb.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -------------------------------------PERCEPTRON-------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
per = Perceptron(tol=1e-3, random_state=0)
per = per.fit(X=X_train, y=y_train)
y_pred = per.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________PERCEPTRON__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
per = Perceptron(tol=1e-3, random_state=0)
per = per.fit(X=X_train, y=y_train)
y_pred = per.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(res)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(hy, data, train_size=0.80, test_size=0.20, random_state=1234)
per = Perceptron(tol=1e-3, random_state=0)
per = per.fit(X=X_train, y=y_train)
y_pred = per.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -----------------------------------RIDGE CLASSIFIER------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
rc = RidgeClassifier()
rc = rc.fit(X=X_train, y=y_train)
y_pred = rc.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________RIDGE CLASSIFIER__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
rc = RidgeClassifier()
rc = rc.fit(X=X_train, y=y_train)
y_pred = rc.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(res)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(hy, data, train_size=0.80, test_size=0.20, random_state=1234)
rc = RidgeClassifier()
rc = rc.fit(X=X_train, y=y_train)
y_pred = rc.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -------------------------------------LINEAR SVC-------------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
LSVC = LinearSVC()
LSVC = LSVC.fit(X_train, y_train)
y_pred = LSVC.predict(X_test)
print("----------------------------------------------------------------------------------")
print("__________LINEAR SVC__________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
LSVC = LinearSVC()
LSVC = LSVC.fit(X_train, y_train)
y_pred = LSVC.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(res)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(hy, data, train_size=0.80, test_size=0.20, random_state=1234)
LSVC = LinearSVC()
LSVC = LSVC.fit(X_train, y_train)
y_pred = LSVC.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


# -----------------------------------DECISION TREE CLASSIFIER-----------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("----------------------------------------------------------------------------------")
print("_____DECISION TREE CLASSIFIER_____")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)
train_tfidf = tfidf[8678:, :]
test_tfidf = tfidf[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("TF-IDF --------->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


bow_vectorizer = CountVectorizer()
# bow = bow_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
# tfidf = tfidf_vectorizer.fit_transform(corpus)
hy_vectorizer = FeatureUnion([('bow', bow_vectorizer), ('tfidf', tfidf_vectorizer)])
hy = hy_vectorizer.fit_transform(res)
# hybrid = hstack([tfidf, bow])
# hy = hybrid.tocsr()
train_hy = hy[8678:, :]
test_hy = hy[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(hy, data, train_size=0.80, test_size=0.20, random_state=1234)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("Bow + Tf-idf --->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)

X_train, X_test, y_train, y_test = train_test_split(bow,data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = RandomForestClassifier()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("----------------------------------------------------------------------------------")
print("________Random Forest________")
print("BagOfWords ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)
X_train, X_test, y_train, y_test = train_test_split(tfidf,data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = RandomForestClassifier()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("----------------------------------------------------------------------------------")
print("________Random Forest________")
print("TF-IDF ----->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")


tfidf_vectorizer2 = TfidfVectorizer()
tfidf2 = tfidf_vectorizer2.fit_transform(corpus)



#------------------------------------GLOVE classifier---------------------------------------


#embeddings
embeddings_index = dict()
f = open("glove.6B.300d.txt",encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
#print('Found %s word vectors.' % len(embeddings_index))
t=Tokenizer()
t.fit_on_texts(corpus)
vocabulary_size=len(t.word_index)+1
encoded_docs = t.texts_to_sequences(corpus)#encode words to integers
#print(encoded_docs)
# pad documents to a max length of 4 words



max_length = 6
padded_docs = pad_sequences(encoded_docs, maxlen=max_length)#pad these words to same length

t2=Tokenizer()
t2.fit_on_texts(res)
vocabulary_size2=len(t.word_index)+1
encoded_docs2 = t.texts_to_sequences(res)#encode words to integers

# pad documents to a max length of 4 words
max_length = 6
padded_docs2 = pad_sequences(encoded_docs2, maxlen=max_length)#pad these words to same length


#print(padded_docs)
embedding_matrix = np.zeros((vocabulary_size,300))
for word, index in t.word_index.items():
    if index >vocabulary_size-1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

#print(embedding_vector)
#print('hola\n\n')
#print(embedding_matrix)
            
model_glove  = Sequential()
e = Embedding(vocabulary_size, 300, weights=[embedding_matrix], input_length=max_length,trainable=False)
model_glove .add(e)

model_glove .add(Conv1D(256, 3, activation='relu'))

model_glove .add(Flatten())

model_glove .add(Dense(1, activation='sigmoid'))

#-------------------------------CNN-------------------------------------------------------
'''
model_glove = Sequential()
e = Embedding(vocabulary_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False)
model_glove.add(e)

model_glove.add(Flatten())

model_glove.add(Dense(units=32, activation='relu'))
model_glove.add(Dense(units=1, activation='sigmoid'))
'''
# compile the model
model_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model

print(model_glove.summary())
# fit the model

model_glove.fit(padded_docs2[:250],np.array(data[:250]),batch_size=64,epochs=10)
loss, accuracy = model_glove.evaluate(padded_docs2,np.array(data), verbose=0)
print('Accuracy of gloVe with cnn is: %f' % (accuracy))




#--------------------------------bow+tfidf+glove-------------------------------------------------
hybrid = hstack([bow,tfidf,padded_docs2])
hy = hybrid.tocsr()
X_train, X_test, y_train, y_test = train_test_split(hy,data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = RandomForestClassifier()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("BoW+TFIDF+GloVe with Random forest--->  " + str(accuracy_score(y_test, y_pred) * 100) + "%")

model_glove = Sequential()
e = Embedding(vocabulary_size, 300, weights=[embedding_matrix], input_length=2890, trainable=False)
model_glove.add(e)
model_glove.add(Flatten())
model_glove.add(Dense(units=1, activation='sigmoid'))
# compile the model
model_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model_glove.summary())
# fit the model

model_glove.fit(hy[:300],np.array(data[:300]),batch_size=64,epochs=10)
loss, accuracy = model_glove.evaluate(hy,np.array(data), verbose=0)
print('Accuracy of hybrid with cnn is: %f' % (accuracy))


#---------------------------------lstm----------------------------------------------------------------
model = Sequential()
model.add(Embedding(vocabulary_size, 300, input_length=6))
model.add(Dropout(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_docs2[:300],np.array(data[:300]),batch_size=32,epochs=15)
loss, accuracy = model.evaluate(padded_docs2,np.array(data), verbose=0)
print('Accuracy of gloVe with lstm is: %f' % (accuracy))

#-----------------------------------cnn+lstm----------------------------------------------------------

model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, 300, input_length=6, weights=[embedding_matrix], trainable=False))
#model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(32, 6, activation='relu'))

model_glove.add(MaxPooling1D(pool_size=1))
model_glove.add(Conv1D(64, 1, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=1))
model_glove.add(Conv1D(128, 1, activation='relu'))

model_glove.add(MaxPooling1D(pool_size=1))
model_glove.add(Conv1D(16, 1, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=1))

model_glove.add(LSTM(100))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## Fit train data
print(model_glove.summary())
model_glove.fit(padded_docs2, np.array(data), validation_split=0.2, epochs = 20)
loss, accuracy = model_glove.evaluate(padded_docs2,np.array(data), verbose=0)
print('Accuracy of gloVe with cnn and lstm is: %f' % (accuracy))




