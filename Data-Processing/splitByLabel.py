
# coding: utf-8

# In[1]:

import pandas as pd
pd.set_option('display.max_rows', None)


# In[106]:

df = pd.read_csv('tweet.out', sep='\t', names=['twitterID', 'userID', 'label', 'twitter'])


# In[3]:

df.head()


# In[ ]:




# Remove Chinese Double Quotes

# In[8]:

import re


# In[ ]:




# Tweeking: read file as plain text, replace chars, and write back

# In[107]:

with open("tweet.out", "r", encoding="utf8") as infile, open("tweet_processed.out", "w", encoding="utf8") as outfile:
    for line in infile:
        outfile.write(re.sub('”|"|“', '', line))


# In[103]:

df = pd.read_csv('tweet_processed.out', sep='\t', names=['twitterID', 'userID', 'label', 'twitter'])


# In[ ]:




# In[87]:

df['twitter'] = df['twitter'].map(lambda x: re.sub('”', '', x))


# In[ ]:




# Split based on label

# In[104]:

label_0_twitters = df[df['label'] == 0]['twitter']


# In[105]:

label_0_twitters.to_csv('tweet_label_0.out', index=False)


# In[4]:

label_1_twitters = df[df['label'] == 1]['twitter']


# In[5]:

label_1_twitters.to_csv('tweet_label_1.out', index=False)

