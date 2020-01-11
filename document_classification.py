from nltk.corpus import reuters
import numpy as np


def collection_stats():
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents")
    '''
    train_docs = list(filter(lambda doc: doc.startswith("train"),
                             documents))
    print(str(len(train_docs)) + " total train documents")

    test_docs = list(filter(lambda doc: doc.startswith("test"),
                            documents))
    print(str(len(test_docs)) + " total test documents")
    '''

    # List of categories
    categories = reuters.categories()
    print(str(len(categories)) + " categories")

    docs_per_category = list()
    # Documents in a category
    for category in categories:
        #category_docs = reuters.fileids("acq")
        category_docs = reuters.fileids(category)
        print('for the following category: {} this is the amount of docs: {}'.format(category, len(category_docs)))
        docs_per_category.append(len(category_docs))

    print('mean of docs per category is: {}'.format(np.mean(np.array(docs_per_category))))
    print('standard deviation of docs per category is: {}'.format(np.std(np.array(docs_per_category))))
    print('this is the min docs per category: {}'.format(min(docs_per_category)))
    print('this is the max docs per category: {}'.format(max(docs_per_category)))

'''
    # Words for a document
    document_id = category_docs[0]
    document_words = reuters.words(category_docs[0])
    print(document_words)

    # Raw document
    print(reuters.raw(document_id))

'''
collection_stats()
