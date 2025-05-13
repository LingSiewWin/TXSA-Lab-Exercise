#!/usr/bin/env python
# coding: utf-8

# ### VADER (Valence Aware Dictionary and sEntiment Reasoner)

# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. It is fully open-sourced under the [MIT License] (we sincerely appreciate all attributions and readily accept most contributions, but please don't hold us liable).

# In[1]:


# pip install vaderSentiment


# In[6]:


import vaderSentiment


# In[15]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentences = ["VADER is smart, handsome, and funny"]

analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print(vs)
    print("{:-<50} {}".format(sentence, str(vs)))


# For more knowledge, pls refer the following link
# https://github.com/cjhutto/vaderSentiment

# In[ ]:




