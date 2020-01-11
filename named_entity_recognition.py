from nltk.corpus import conll2002
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def word2features(sent, i):
    postag = sent[i][0][1]
    not_found = True
    j = i-1
    while not_found:
        try:
            if type(sent[j]) != tuple:
                old_postag = sent[i-1][0][1]
                not_found = False
        except Exception as e:
            if j < 0:
                old_postag = 0
                not_found = False
        j -= 1
        continue
    not_found = True
    j = i+1
    while not_found:
        try:
            if type(sent[j]) != tuple:
                next_postag = sent[i-1][0][1]
                not_found = False
        except Exception as e:
            if j >= len(sent):
                next_postag = 0
                not_found = False
        j += 1
        continue
    features = [
        0 if not old_postag else float(''.join(format(ord(x), 'b') for x in old_postag)),
        0 if not next_postag else float(''.join(format(ord(x), 'b') for x in next_postag)),
        float(''.join(format(ord(x), 'b') for x in postag))
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


# from here on just for 3.1.2

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
