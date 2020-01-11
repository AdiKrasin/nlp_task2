from nltk.corpus import reuters
import numpy as np


def collection_stats():
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents")

    # List of categories
    categories = reuters.categories()
    print(str(len(categories)) + " categories")

    docs_per_category = list()
    # Documents in a category
    for category in categories:
        category_docs = reuters.fileids(category)
        print('for the following category: {} this is the amount of docs: {}'.format(category, len(category_docs)))
        docs_per_category.append(len(category_docs))

    print('mean of docs per category is: {}'.format(np.mean(np.array(docs_per_category))))
    print('standard deviation of docs per category is: {}'.format(np.std(np.array(docs_per_category))))
    print('this is the min docs per category: {}'.format(min(docs_per_category)))
    print('this is the max docs per category: {}'.format(max(docs_per_category)))


def additional_stats():
    documents = reuters.fileids()
    categories = reuters.categories()
    total_amount_of_documents_words = 0
    total_amount_of_document_characters = 0
    for category in categories:
        category_docs = reuters.fileids(category)
        # Words for a document
        document_id = category_docs[0]
        document_words = reuters.words(category_docs[0])
        total_amount_of_documents_words += len(document_words)
        for word in document_words:
            total_amount_of_document_characters += len(word)
    print('total amount of documents words: {}'.format(total_amount_of_documents_words))
    print('total amount of documents characters: {}'.format(total_amount_of_document_characters))

'''
# this is just for 2.1.1
collection_stats()
'''

# this is just for 2.1.2
additional_stats()
