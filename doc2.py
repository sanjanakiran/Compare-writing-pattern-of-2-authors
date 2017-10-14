
# coding: utf-8

# In[19]:

import nltk
import string
from urllib import request

#text from online gutenberg
url = "http://www.gutenberg.org/files/16/16-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')

#text of the book is separated into tokens with word tokenizer
#and converted all the characters to lowercase
#ABtokens = nltk.word_tokenize(raw)
ABtokens = nltk.word_tokenize(raw)
ABwords = [w.lower() for w in ABtokens]

#print first 200 words which is tokenized and are in lowercase
print("Tokenized and lowercase")
print(ABwords[:200])
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

#to print the stopwords list
stopwords =nltk.corpus.stopwords.words('english')  + list(string.punctuation)

print("List of stopwords")
print(stopwords) 

#to remove the stopwords
stoppedABwords = [w for w in ABwords if not w in stopwords]
filtered_words = []
for w in ABwords:
    if w not in stopwords:
        filtered_words.append(w)
#tokenized, lowercase list without stopwords
print("tokenized, lowercase list without stopwords")
print(filtered_words[:200]) 



from nltk import FreqDist
from nltk.collocations import *
#list the top 50 words by frequency (normalized by the length of the document)
ABdist = FreqDist(filtered_words)
ABitems = ABdist.most_common(50)
print("top 50 words by frequency")
for item in ABitems:
    print(item[0], '\t', item[1])

ABbigrams = list(nltk.bigrams(filtered_words))
print("Sample Bigrams")
print(ABbigrams[:50])

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(ABwords)
scored = finder.score_ngrams(bigram_measures.raw_freq)
#print(type(scored))
#first = scored[0]
#print(type(first), first)
print ("without any filter")
for bscore in scored[:50]:
    print (bscore)
    
finder2 = BigramCollocationFinder.from_words(ABwords)
finder2.apply_freq_filter(2)
scored = finder2.score_ngrams(bigram_measures.raw_freq)
print ("removed low frequency words")
for bscore in scored[:30]:
    print (bscore)

finder.apply_word_filter(lambda w: w in stopwords)
scored = finder.score_ngrams(bigram_measures.raw_freq)
print ("Bigrams after removing stopwords")
for bscore in scored[:50]:
    print (bscore)


finder3 = BigramCollocationFinder.from_words(ABwords)
scored = finder3.score_ngrams(bigram_measures.pmi)
print ("pmi on raw")
for bscore in scored[:50]:
    print (bscore)

finder3.apply_freq_filter(5)
scored = finder3.score_ngrams(bigram_measures.pmi)
print ("pmi on filtered data")
for bscore in scored[:50]:
    print (bscore)
    
from nltk.collocations import *
trigram_measures = nltk.collocations.TrigramAssocMeasures()
print ("trigrams raw")
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder7 = TrigramCollocationFinder.from_words(ABwords)
scored = finder7.score_ngrams(trigram_measures.pmi)
for bscore in scored[:50]:
    print (bscore)

print ("trigrams pmi")
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder.apply_freq_filter(2)
finder7 = TrigramCollocationFinder.from_words(ABwords)
scored = finder7.score_ngrams(trigram_measures.pmi)
for bscore in scored[:50]:
    print (bscore)
    




# In[ ]:




# In[ ]:



