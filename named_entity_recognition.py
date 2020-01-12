from nltk.corpus import conll2002
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def word2features(sent, i):
    postag = sent[i][0][1]
    try:
        old_postag = sent[i-1][0][1]
    except Exception as e:
        old_postag = 'O'
    try:
        next_postag = sent[i+1][0][1]
    except Exception as e:
        next_postag = 'O'
    features = [
        float(''.join(format(ord(x), 'b') for x in old_postag)),
        float(''.join(format(ord(x), 'b') for x in next_postag)),
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

'''
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
'''

# this is just for 3.1.3

new_train_sents = list(conll2002.iob_sents('esp.train'))

illegal_combinations = {'o-ix': 0, 'ix-iy': 0, 'bx-iy': 0}

for sent in new_train_sents:
    index = 0
    for element in sent:
        if index and index < len(sent) - 1:
            if element[2] == 'O' and sent[index+1][2][0] == 'I':
                illegal_combinations['o-ix'] += 1
            elif element[2][0] == 'I' and sent[index+1][2][0] == 'I' and element[2][1] != sent[index+1][2][1]:
                illegal_combinations['ix-iy'] += 1
            elif element[2][0] == 'B' and sent[index+1][2][0] == "I" and element[2][1] != sent[index+1][2][1]:
                illegal_combinations['bx-iy'] += 1
        index += 1

print(illegal_combinations)
