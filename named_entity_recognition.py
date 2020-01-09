from nltk.corpus import conll2002

etr = conll2002.chunked_sents('esp.train') # In Spanish
eta = conll2002.chunked_sents('esp.testa') # In Spanish
etb = conll2002.chunked_sents('esp.testb') # In Spanish

dtr = conll2002.chunked_sents('ned.train') # In Dutch
dta = conll2002.chunked_sents('ned.testa') # In Dutch
dtb = conll2002.chunked_sents('ned.testb') # In Dutch