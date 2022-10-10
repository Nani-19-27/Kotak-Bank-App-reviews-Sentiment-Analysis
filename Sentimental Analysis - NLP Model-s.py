#!/usr/bin/env python
# coding: utf-8

# # Kotak Playstore Application Reviews Scarping and NLP

# In[1]:


import google_play_scraper

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator

from transformers import pipeline
sentiment_analysis = pipeline('sentiment-analysis')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score


# In[2]:


from google_play_scraper import reviews,Sort


# In[3]:


kotak_app = reviews('com.msf.kbank.mobile',count=1000,sort=Sort.NEWEST)


# In[4]:


df_kotak = pd.DataFrame(kotak_app[0])


# In[5]:


df_kotak_ = df_kotak[['userName','content','score','reviewCreatedVersion','at']]


# In[6]:


df_kotak_


# ### Data Cleaning

# In[7]:


filtered_comments=[]

for i in range(0,len(df_kotak_.content)):
    text = df_kotak_.content[i].lower()
    text = re.sub('[^a-zA-Z]',' ',text.strip())
    text = re.sub('\n','',text.strip())
    filtered_comments.append(text.strip())


# In[8]:


df_kotak_['filtered_comments'] = filtered_comments


# In[9]:


df_kotak_[df_kotak_['filtered_comments']==''].head()


# In[10]:


df_kotak_.drop(df_kotak_[df_kotak_['filtered_comments']==''].index.tolist(),axis=0,inplace=True)

df_kotak_.reset_index(inplace=True)

df_kotak_.drop(columns=['index'],inplace=True)


# In[11]:


count=[]

for i in range(0,len(df_kotak_.filtered_comments)):
    text = len(df_kotak_.filtered_comments[i])
    if text <= 3:
        a=''
    else:
        a=text
    count.append(a)


# In[12]:


df_kotak_['num'] = count

df_kotak_[df_kotak_['num']==''].head()


# In[13]:


df_kotak_.drop(df_kotak_[df_kotak_['num']==''].index.tolist(),axis=0,inplace=True)

df_kotak_.reset_index(inplace=True)

df_kotak_.drop(columns=['index','num'],inplace=True)


# In[14]:


response =[]

for i in range(0,len(df_kotak_.filtered_comments)):
    text = sentiment_analysis(df_kotak_.filtered_comments[i])[0]['label']
    response.append(text)


# In[15]:


df_kotak_['response'] = response


# In[16]:


#df_kotak_.to_excel("C:\\Users\\Manikanta\\Downloads\\kotak.xlsx")


# In[17]:


df_kotak_.info()


# In[18]:


df_kotak_.reviewCreatedVersion.replace('None',np.nan,inplace=True)


# ## Word Cloud

# In[19]:


wc = df_kotak_['filtered_comments']


# In[20]:


toz_ = [word_tokenize(word) for word in wc]


# In[21]:


stop_words =[]

for i in (toz_):
    a=[]
    for j in i:
        if len(j) > 2:
            if not j in stopwords.words('english'):
                a.append(j)
    stop_words.append(a)           


# In[22]:


lemm = WordNetLemmatizer()

lemmatize=[]

for i in stop_words:
    for j in i:
        text = ''.join(j)
        lemmatize.append(text)
        text =' '.join(lemmatize)


# In[23]:


plt.figure(figsize=(10,10))

text = text
stopwords = set(STOPWORDS)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white",scale=3,random_state=0).generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off");


# ## Observations By Insights

# In[24]:


plt.figure(figsize=(7,5))
sns.set_style('darkgrid')

plt.title('Sentimental Analysis',fontdict={'family':'times new roman','size':25})

sns.countplot(x=df_kotak_.response,order=df_kotak_.response.value_counts().index,hatch='///')

sns.despine();


# In[25]:


plt.rcParams.update({'figure.figsize':(25,6)})

df_kotak_.groupby(['response','reviewCreatedVersion'])[['reviewCreatedVersion']].count().plot.bar(hatch='||').set_title('Version Wise Sentimental Analysis',fontsize=20)

plt.xticks(fontsize=15);


# In[26]:


plt.figure(figsize=(15,7))

sample = pd.DataFrame(df_kotak_.groupby('reviewCreatedVersion')['score'].mean().round()).reset_index()
plt.title('Version Vs Avg-Rating',fontsize=20)
sns.barplot(x=sample.reviewCreatedVersion,y=sample.score,hatch='...');


# ## NLP Model

# In[27]:


df_kotak_.head()


# In[28]:


x_var = df_kotak_.filtered_comments 
y_var = df_kotak_.response

x_train,x_test,y_train,y_test = train_test_split(x_var,y_var,test_size=0.20,random_state=0)


# In[29]:


vec= TfidfVectorizer()
log = LogisticRegression(solver='lbfgs')

model = Pipeline([('vectorizer',vec),('classifier',log)])

model.fit(x_train,y_train)

pred_y = model.predict(x_test)


# In[30]:


cf = confusion_matrix(pred_y,y_test)

print(cf)


# In[31]:


plt.figure(figsize=(10,5))

sns.heatmap(cf,annot=True);


# In[32]:


print('\nAccuarcy Score {:.3f}\n'.format(accuracy_score(y_test,pred_y)))

print('Precision Score {:.3f}\n'.format(precision_score(y_test,pred_y,average='weighted')))

print('Recall Score {:.3f}'.format(recall_score(y_test,pred_y,average='weighted')))


# In[33]:


comment = ['so many bugs are there some times it is not working quickly']

print(model.predict(comment))


# In[35]:


comment =['it is very easy to operate and i have good experience with the app so far']

print(model.predict(comment))

