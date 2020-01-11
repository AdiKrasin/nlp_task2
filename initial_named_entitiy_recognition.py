from nltk.corpus import conll2002
import numpy as np

def word2features(sent, i):
    word = sent[i][0][0]
    postag = sent[i][0][1]
    features = [
        'bias',
        word.lower(),
        word[-3:],
        word[-2:],
        word[-1:],
        word[:1],
        word[:2],
        word[:3],
        word.isupper(),
        word.isdigit(),
        postag,
    ]
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent)) if type(sent[i]) != tuple]


def sent2labels(sent):
    labels = []
    for i in range(len(sent)):
        if type(sent[i]) != tuple:
            label = sent[i]._label
            labels.append(label)
    return labels


def sent2tokens(sent):
    return [token for token, postag, label in sent]


etr = conll2002.chunked_sents('esp.train') # In Spanish
eta = conll2002.chunked_sents('esp.testa') # In Spanish
etb = conll2002.chunked_sents('esp.testb') # In Spanish

dtr = conll2002.chunked_sents('ned.train') # In Dutch
dta = conll2002.chunked_sents('ned.testa') # In Dutch
dtb = conll2002.chunked_sents('ned.testb') # In Dutch

train_sents = etr
test_sents = etb

X_train = [sent2features(s) for s in train_sents]
X_train = [item for sublist in X_train for item in sublist]
# normalizing the values of x:
for index in range(len(X_train[0])):
    mean = np.mean(np.array([row[index] for row in X_train]))
    sd = np.std(np.array([row[index] for row in X_train]))
    for row in X_train:
        row[index] = (row[index] - mean) / sd
y_train = [sent2labels(s) for s in train_sents]
y_train = [item for sublist in y_train for item in sublist]

X_test = [sent2features(s) for s in test_sents]
X_test = [item for sublist in X_test for item in sublist]
# normalizing the values of x:
for index in range(len(X_test[0])):
    mean = np.mean(np.array([row[index] for row in X_train]))
    sd = np.std(np.array([row[index] for row in X_train]))
    for row in X_test:
        row[index] = (row[index] - mean) / sd
y_test = [sent2labels(s) for s in test_sents]
y_test = [item for sublist in y_test for item in sublist]
