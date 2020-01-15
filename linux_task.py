from nltk.corpus import conll2002
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize


def word2features(sent, i, model):
    try:
        word_as_vector = model.wv[sent[i][0]]
    except Exception as e:
        word_as_vector = np.array([0 for element in model.wv['de']])
    postag = sent[i][1]
    if i:
        old_postag = sent[i-1][1]
    else:
        old_postag = 'BOS'
    if i < len(sent) - 1:
        next_postag = sent[i+1][1]
    else:
        next_postag = 'EOS'
    features = [
        word_as_vector.reshape(-1, 1)[0],
        float(''.join(format(ord(x), 'b') for x in old_postag)),
        float(''.join(format(ord(x), 'b') for x in next_postag)),
        float(''.join(format(ord(x), 'b') for x in postag))
    ]
    return features

# todo delete this:
# Eitan9840


def sent2features(sent, model):
    return [word2features(sent, i, model) for i in range(len(sent))]


def sent2features2(sent):
    return [sent[i][0] for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


etr = conll2002.iob_sents('esp.train') # In Spanish
eta = conll2002.iob_sents('esp.testa') # In Spanish
etb = conll2002.iob_sents('esp.testb') # In Spanish

dtr = conll2002.iob_sents('ned.train') # In Dutch
dta = conll2002.iob_sents('ned.testa') # In Dutch
dtb = conll2002.iob_sents('ned.testb') # In Dutch

train_sents = etr
test_sents = etb

data = [sent2features2(s) for s in train_sents]
data = [item for sublist in data for item in sublist]

with open('file_of_text.txt', 'w') as f:
    for item in data:
        f.write("%s " % item)

new_data = []

sample = open("file_of_text.txt", "r")
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")

# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []

    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())

    new_data.append(temp)

old_data = data
data = new_data

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count = 1,
                              size = 100, window = 5)


model1.train(list(old_data), total_examples=1, epochs=1)


X_train = [sent2features(s, model1) for s in train_sents]
X_train = [item for sublist in X_train for item in sublist]

# normalizing the values of x:
for index in range(len(X_train[0])):
    mean = np.mean(np.array([row[index] for row in X_train]))
    sd = np.std(np.array([row[index] for row in X_train]))
    for row in X_train:
        row[index] = (row[index] - mean) / sd
y_train = [sent2labels(s) for s in train_sents]
y_train = [item for sublist in y_train for item in sublist]

X_test = [sent2features(s, model1) for s in test_sents]
X_test = [item for sublist in X_test for item in sublist]
# normalizing the values of x:
for index in range(len(X_test[0])):
    mean = np.mean(np.array([row[index] for row in X_train]))
    sd = np.std(np.array([row[index] for row in X_train]))
    for row in X_test:
        row[index] = (row[index] - mean) / sd
y_test = [sent2labels(s) for s in test_sents]
y_test = [item for sublist in y_test for item in sublist]



lr = LogisticRegression()

# pre processing the data to fit to the logistic Regression form

for index in range(len(X_train)):
    X_train[index] = np.array(X_train[index])


lr.fit(np.array(X_train), np.array(y_train))

Y_pred = lr.predict(np.array(X_test))

recall = recall_score(y_test, Y_pred, average='weighted', zero_division=1)
precision = precision_score(y_test, Y_pred, average='weighted', zero_division=1)
fscore = (2 * precision * recall) / (precision + recall)
performance = [precision, recall, fscore]
print(performance)
