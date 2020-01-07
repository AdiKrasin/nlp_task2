#############################################################
## ASSIGNMENT 2_1 CODE SKELETON
#############################################################

from collections import defaultdict
import gzip


#### Q1.1 Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    '''
    Precision = TruePositives / (TruePositives + FalsePositives)
    TruePositives is the amount of Positive class members classified as Positive class members.
    FalsePositives is the amount of Negative class members classified as Positive class members.
    For this assignment, complex words are considered positive examples, and simple words are considered negative
     examples.
    '''
    true_positives = 0
    false_positive = 0
    index = 0

    for y in y_true:
        if y == 1:
            if y_pred[index] == 1:
                true_positives += 1
        else:
            if y_pred[index] == 1:
                false_positive += 1
        index += 1

    # escaping division by 0
    if true_positives == false_positive == 0:
        precision = 'inf'
    else:
        precision = true_positives / (true_positives + false_positive)

    return precision


## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    '''
    Recall = TruePositives / (TruePositives + FalseNegatives)
    TruePositives is the amount of Positive class members classified as Positive class members.
    FalseNegatives is the amount of Positive class members classified as Negative class members.
    For this assignment, complex words are considered positive examples, and simple words are considered negative
     examples.
    '''

    true_positives = 0
    false_negatives = 0
    index = 0

    for y in y_true:
        if y == 1:
            if y_pred[index] == 1:
                true_positives += 1
            else:
                false_negatives += 1
        index += 1

    # escaping division by 0
    if true_positives == false_negatives == 0:
        recall = 'inf'
    else:
        recall = true_positives / (true_positives + false_negatives)

    return recall


## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    '''
    F-Measure = (2 * Precision * Recall) / (Precision + Recall)
    '''
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    if precision != 'inf' and recall != 'inf':
        fscore = (2 * precision * recall) / (precision + recall)
    else:
        fscore = 'inf'

    return fscore

def test_predictions(y_pred, y_true):
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    print('this is the precision: {}\n this is the recall: {}\n and this is the fscore: {}'.format(precision, recall,
                                                                                                   fscore))


#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


### 1.2.1: A very simple baseline

## Labels every word complex
def all_complex(data_file):
    lables = data_file[1]
    all_positive = [1 for lable in lables]
    precision = get_precision(all_positive, lables)
    recall = get_recall(all_positive, lables)
    fscore = get_fscore(all_positive, lables)
    performance = [precision, recall, fscore]
    return performance


### 1.2.2: Word length thresholding

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    ## YOUR CODE HERE
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance


### 1.2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file):
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt', errors='ignore') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts


# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set
def word_frequency_threshold(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance


### 1.3.1: Naive Bayes

## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance


if __name__ == "__main__":
    training_file = "../data/complex_words_training.txt"
    development_file = "../data/complex_words_development.txt"
    test_file = "../data/complex_words_test_unlabeled.txt"
    train_data = load_file(training_file)
    development_data = load_file(development_file)
    print(all_complex(train_data))
    print(all_complex(development_data))
