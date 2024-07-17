#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


from nltk.corpus import names
import random


# In[3]:


def gender_features(word):
    return{'last_letter' : word[-1]}


# In[4]:


gender_features('Shrek')


# In[5]:


names = nltk.corpus.names


# In[6]:


names.fileids()


# In[7]:


male_names = names.words('male.txt')


# In[8]:


female_name = names.words('female.txt')


# In[9]:


names = ([(name, 'male') for name in male_names] +
[(name, 'female') for name in female_name])


# In[10]:


names


# In[15]:


random.shuffle(names)


# In[16]:


names


# In[17]:


featuresets1 = [(gender_features(n), g) for (n,g) in names]


# In[18]:


featuresets1


# In[19]:


train_set , test_set = featuresets1[500:], featuresets1[:500]


# In[21]:


classifier1 = nltk.NaiveBayesClassifier.train(train_set)


# In[22]:


classifier1.classify(gender_features('Neo'))


# In[23]:


classifier1.classify(gender_features('Adela'))


# In[24]:


classifier1.classify(gender_features('Trinity'))


# In[25]:


print(nltk.classify.accuracy(classifier1, test_set))


# In[26]:


classifier1.show_most_informative_features(10)


# In[32]:


def gender_features2(name):
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz' :
        features["count(%s)" % letter] = name.lower().count(letter)
        features["has(%s)" %letter] = (letter in name.lower())
    return features


# In[33]:


gender_features2('John')


# In[34]:


featuresets2 = [(gender_features2(n),g) for (n,g) in names]


# In[35]:


train_set, test_set = featuresets2[500:], featuresets2[:500]


# In[37]:


classifier2 = nltk.NaiveBayesClassifier.train(train_set)


# In[38]:


print(nltk.classify.accuracy(classifier2, test_set))


# In[39]:


# chooseing right features


# In[40]:


train_names = names[1500:]
devtest_names = names[500:1500]
test_set = names[:500]


# In[43]:


train_set = [(gender_features(n), g) for (n,g) in train_names]


# In[44]:


devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]


# In[45]:


test_set = [(gender_features(n), g) for (n,g) in test_set]


# In[47]:


classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[48]:


print(nltk.classify.accuracy(classifier, test_set))


# In[49]:


errors = []


# In[52]:


for (names, tag) in devtest_names:
    guess = classifier1.classify(gender_features(names))
    if guess !=tag:
        errors.append((tag,guess,names))


# In[58]:


for(tag, geuess, name) in sorted(errors):
    print('correct=%-8s geuss=%-8s name=%-30s' % (tag, geuess, name))


# In[63]:


def gender_features(word):
    return { 'suffix1' : word[-1:],
           'suffix2' : word[-2:]}


# In[65]:


train_set = [(gender_features(n), g) for (n,g) in train_names]
devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
classifier = nltk.NaiveBayesClassifier.train(test_set)


# In[67]:


print(nltk.classify.accuracy(classifier, devtest_set))


# In[75]:


from nltk.corpus import movie_reviews


# In[76]:


nltk.download("movie_reviews")


# In[78]:


documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]


# In[81]:


random.shuffle(documents)


# In[82]:


movie_reviews.categories()


# In[83]:


movie_reviews.fileids()


# In[84]:


movie_reviews.fileids()[1000:]


# In[85]:


all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())


# In[86]:


all_words


# In[87]:


word_feature = all_words.keys()


# In[88]:


word_feature 


# In[90]:


def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_feature:
        features['contains(%s)' %word] = (word in document_words)
    return features


# In[92]:


print(document_features(movie_reviews.words('pos/cv957_8737.txt')))


# In[95]:


featuresets = [
    (document_features(d), c ) for (d,c) in documents  
]
train_set, test_set = featuresets[100:], featuresets[:100]


# In[96]:


classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[97]:


print(nltk.classify.accuracy(classifier, test_set))


# In[98]:


classifier.show_most_informative_featureshow_most_informative_features(10)


# In[ ]:




