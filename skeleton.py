#############################################################
## ASSIGNMENT 2_1 CODE SKELETON
#############################################################

from collections import defaultdict
import gzip
import matplotlib.pyplot as plt
import numpy as np


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
    train_data = load_file(training_file)
    development_data = load_file(development_file)
    train_lables = train_data[1]
    train_words = train_data[0]
    develop_lables = development_data[1]
    develop_words = development_data[0]

    # looking only in the relevant range - meaning len varying from 0 to longest word len + 1
    longest_word_len = 0

    for word in train_words:
        if len(word) > longest_word_len:
            longest_word_len = len(word)

    training_range = range(longest_word_len + 2)
    maximizer = 0
    max_fscore = -1
    recall_for_plot = []
    precision_for_plot = []

    # maximizing the fscore
    for num in training_range:
        y_pred = []
        for word in train_words:
            if len(word) >= num:
                y_pred.append(1)
            else:
                y_pred.append(0)
        fscore = get_fscore(train_lables, y_pred)
        recall_for_plot.append(get_recall(train_lables, y_pred))
        precision_for_plot.append(get_precision(train_lables, y_pred))
        if fscore != 'inf' and fscore > max_fscore:
            max_fscore = fscore
            maximizer = num

    while 'inf' in precision_for_plot:
        index = precision_for_plot.index('inf')
        precision_for_plot = precision_for_plot[:index] + precision_for_plot[index+1:]
        recall_for_plot = recall_for_plot[:index] + recall_for_plot[index+1:]
    while 'inf' in recall_for_plot:
        index = recall_for_plot.index('inf')
        precision_for_plot = precision_for_plot[:index] + precision_for_plot[index+1:]
        recall_for_plot = recall_for_plot[:index] + recall_for_plot[index+1:]

    plt.plot(recall_for_plot, precision_for_plot, color='green', linestyle='dashed', linewidth=3, marker='o',
             markerfacecolor='blue', markersize=12)
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.ylim = max(precision_for_plot)
    plt.xlim = max(recall_for_plot)
    plt.show()

    y_pred = []
    for word in train_words:
        if len(word) >= maximizer:
            y_pred.append(1)
        else:
            y_pred.append(0)

    tprecision = get_precision(y_pred, train_lables)
    trecall = get_recall(y_pred, train_lables)
    tfscore = get_fscore(y_pred, train_lables)

    y_pred = []
    for word in develop_words:
        if len(word) >= maximizer:
            y_pred.append(1)
        else:
            y_pred.append(0)

    dprecision = get_precision(y_pred, develop_lables)
    drecall = get_recall(y_pred, develop_lables)
    dfscore = get_fscore(y_pred, develop_lables)

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
    train_data = load_file(training_file)
    development_data = load_file(development_file)
    train_lables = train_data[1]
    train_words = train_data[0]
    develop_lables = development_data[1]
    develop_words = development_data[0]

    # looking only in the relevant range - meaning frequency can be the average of the all frequencies,
    # or the average of the 9/10 most un common, or the average of the 8/10 most un common and so on
    all_frequencies = []
    training_range = []

    for word in train_words:
        all_frequencies.append(counts[word])

    for n in range(1, 11):
        training_range.append(np.sum(all_frequencies[:int(n*len(all_frequencies)/10)]))

    maximizer = 0
    max_fscore = -1
    recall_for_plot = []
    precision_for_plot = []

    # maximizing the fscore
    for num in training_range:
        y_pred = []
        for word in train_words:
            if counts[word] < num:
                y_pred.append(1)
            else:
                y_pred.append(0)
        fscore = get_fscore(train_lables, y_pred)
        recall_for_plot.append(get_recall(train_lables, y_pred))
        precision_for_plot.append(get_precision(train_lables, y_pred))
        if fscore != 'inf' and fscore > max_fscore:
            max_fscore = fscore
            maximizer = num

    while 'inf' in precision_for_plot:
        index = precision_for_plot.index('inf')
        precision_for_plot = precision_for_plot[:index] + precision_for_plot[index+1:]
        recall_for_plot = recall_for_plot[:index] + recall_for_plot[index+1:]
    while 'inf' in recall_for_plot:
        index = recall_for_plot.index('inf')
        precision_for_plot = precision_for_plot[:index] + precision_for_plot[index+1:]
        recall_for_plot = recall_for_plot[:index] + recall_for_plot[index+1:]

    plt.plot(recall_for_plot, precision_for_plot, color='green', linestyle='dashed', linewidth=3, marker='o',
             markerfacecolor='blue', markersize=12)
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.ylim = max(precision_for_plot)
    plt.xlim = max(recall_for_plot)
    plt.show()

    y_pred = []
    for word in train_words:
        if counts[word] < maximizer:
            y_pred.append(1)
        else:
            y_pred.append(0)

    tprecision = get_precision(y_pred, train_lables)
    trecall = get_recall(y_pred, train_lables)
    tfscore = get_fscore(y_pred, train_lables)

    y_pred = []
    for word in develop_words:
        if counts[word] < maximizer:
            y_pred.append(1)
        else:
            y_pred.append(0)

    dprecision = get_precision(y_pred, develop_lables)
    drecall = get_recall(y_pred, develop_lables)
    dfscore = get_fscore(y_pred, develop_lables)

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
    '''
    # this is just for 1.2.1
    development_data = load_file(development_file)
    print(all_complex(train_data))
    print(all_complex(development_data))
    '''
    '''
    # this is just for 1.2.2
    print(word_length_threshold(training_file, development_file))
    '''
    '''
    # this is just for 1.2.3
    dic = load_ngram_counts('../ngram_counts.txt.gz')
    print(word_frequency_threshold(training_file, development_file, dic))
    '''
