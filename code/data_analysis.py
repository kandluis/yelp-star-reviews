
# coding: utf-8

# In[5]:


import os

import numpy as np
import pandas as pd


# In[6]:


# Read datasets.
DATA_DIR = "../yelp_data/dataset"
PHOTO_DIR= "../yelp_data/yelp_photos"


# In[7]:


def loadCSV(data):
    df = pd.read_csv(os.path.join(DATA_DIR, "%s.csv" % data))
    return df


# In[8]:


businesses = loadCSV("business")


# In[52]:


# How many businesses
len(businesses)


# In[53]:


len(businesses.columns)


# In[39]:


photos = loadCSV("photos")


# In[40]:


# How many photos
len(photos)


# In[41]:


photos


# In[23]:


len(photos.columns)


# In[42]:


# How many tips
tip = loadCSV("tip")


# In[43]:


len(tip)


# In[44]:


tip


# In[45]:


# How many checkins
checkin = loadCSV("checkin")


# In[46]:


len(checkin)


# In[57]:


len(set(businesses['business_id']))


# In[58]:


# Load user data (user network)
users = loadCSV("user")


# In[84]:


len(users)


# In[66]:


sum([len(neighbors) for neighbors in users.friends]) / 2
    


# In[68]:


reviews = loadCSV("review")


# In[85]:


len(reviews)


# In[86]:


reviews


# In[87]:


import snap


# In[91]:


N = len(set(users))
E = sum([len(neighbors) for neighbors in users.friends]) / 2
userNetwork = snap.TUNGraph.New(N, E)


# In[96]:


import matplotlib.pyplot as plt


# In[97]:


users.review_count.hist()
plt.show()


# In[98]:


users[users.review_count > 0].review_count.hist()


# In[100]:


from collections import Counter


# In[105]:


users['num_friends'] = users.friends.map(lambda x: len(x))


# In[114]:


def histogramForField(field):
    hist = Counter(sorted(users[field]))
    return zip(*sorted(zip(hist.keys(), hist.values())))


# In[146]:


fields = ['review_count', 'num_friends', "useful", "funny",
          "cool", "fans"]
for field in fields:
    X,Y = histogramForField(field)
    plt.loglog(X,Y)


# In[147]:


plt.title("Distribution of User Characteristics (Log-Log)")
plt.ylabel("Frequency")
plt.xlabel("By user")
plt.legend(fields)
plt.savefig("../shared/figures/distribution_user_characteristics", dpi=400)
plt.show()


# In[151]:


fields = ['average_stars']
for field in fields:
    X,Y = histogramForField(field)
    plt.plot(X,np.log(Y))


# In[152]:


plt.title("Average Rating Distribution")
plt.ylabel("Frequency (log)")
plt.xlabel("Average Rating for User")
plt.savefig("../shared/figures/average_user_rating_distribution", dpi=400)
plt.show()


# In[154]:


fields = ["compliment_hot", "compliment_more", "compliment_profile",
          "compliment_cute", "compliment_list", "compliment_note",
          "compliment_plain", "compliment_cool", "compliment_funny",
          "compliment_writer", "compliment_photos"]
for field in fields:
    X,Y = histogramForField(field)
    plt.loglog(X,Y)


# In[155]:


plt.title("Compliment Type Distribution (log-log)")
plt.ylabel("Frequency")
plt.xlabel("Count for User")
plt.legend(fields)
plt.savefig("../shared/figures/compliement_type_distribution", dpi=400)
plt.show()


# In[158]:


def histogramForField(df, field):
    hist = Counter(sorted(df[field]))
    return zip(*sorted(zip(hist.keys(), hist.values())))


# In[168]:


fields = ['review_count']
for field in fields:
    X,Y = histogramForField(businesses, field)
    plt.loglog(X,Y)


# In[169]:


plt.title("Review Count Distribution for Businesses")
plt.ylabel("Frequency")
plt.xlabel("Number of Reviews Received")
plt.legend(fields)
plt.savefig("../shared/figures/business_review_count_distribution", dpi=400)
plt.show()


# In[172]:


fields = ['stars']
for field in fields:
    X,Y = histogramForField(businesses, field)
    plt.plot(X,np.log(Y))


# In[173]:


plt.title("Review Count Distribution for Businesses")
plt.ylabel("Frequency")
plt.xlabel("Average Star Rating")
plt.legend(fields)
plt.savefig("../shared/figures/business_star_distribution", dpi=400)
plt.show()

