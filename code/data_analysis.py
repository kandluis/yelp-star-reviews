# coding: utf-8
import os

import numpy as np
import pandas as pd

# Read JSON files
DATA_DIR = "../yelp_data/dataset"
PHOTO_DIR = "../yelp_data/yelp_photos"


def loadCSV(data):
  df = pd.read_csv(os.path.join(DATA_DIR, "%s.csv" % data))
  return df
businesses = loadCSV("business")
# How many businesses
len(businesses)
len(businesses.columns)
photos = loadCSV("photos")
# How many photos
len(photos)
photos
len(photos.columns)
# How many tips
tip = loadCSV("tip")
len(tip)
tip
# How many checkins
checkin = loadCSV("checkin")
len(checkin)
len(set(businesses['business_id']))
# Load user data (user network)
users = loadCSV("user")
len(users)
sum([len(neighbors) for neighbors in users.friends]) / 2

reviews = loadCSV("review")
len(reviews)
reviews
import snap
N = len(set(users))
E = sum([len(neighbors) for neighbors in users.friends]) / 2
userNetwork = snap.TUNGraph.New(N, E)
import matplotlib.pyplot as plt
users.review_count.hist()
plt.show()
users[users.review_count > 0].review_count.hist()
from collections import Counter
users['num_friends'] = users.friends.map(lambda x: len(x))


def histogramForField(field):
  hist = Counter(sorted(users[field]))
  return zip(*sorted(zip(hist.keys(), hist.values())))
fields = ['review_count', 'num_friends', "useful", "funny",
          "cool", "fans"]
for field in fields:
  X, Y = histogramForField(field)
  plt.loglog(X, Y)
plt.title("Distribution of User Characteristics (Log-Log)")
plt.ylabel("Frequency")
plt.xlabel("By user")
plt.legend(fields)
plt.savefig("../shared/figures/distribution_user_characteristics", dpi=400)
plt.show()
fields = ['average_stars']
for field in fields:
  X, Y = histogramForField(field)
  plt.plot(X, np.log(Y))
plt.title("Average Rating Distribution")
plt.ylabel("Frequency (log)")
plt.xlabel("Average Rating for User")
plt.savefig("../shared/figures/average_user_rating_distribution", dpi=400)
plt.show()
fields = ["compliment_hot", "compliment_more", "compliment_profile",
          "compliment_cute", "compliment_list", "compliment_note",
          "compliment_plain", "compliment_cool", "compliment_funny",
          "compliment_writer", "compliment_photos"]
for field in fields:
  X, Y = histogramForField(field)
  plt.loglog(X, Y)
plt.title("Compliment Type Distribution (log-log)")
plt.ylabel("Frequency")
plt.xlabel("Count for User")
plt.legend(fields)
plt.savefig("../shared/figures/compliement_type_distribution", dpi=400)
plt.show()


def histogramForField(df, field):
  hist = Counter(sorted(df[field]))
  return zip(*sorted(zip(hist.keys(), hist.values())))
fields = ['review_count']
for field in fields:
  X, Y = histogramForField(businesses, field)
  plt.loglog(X, Y)
plt.title("Review Count Distribution for Businesses")
plt.ylabel("Frequency")
plt.xlabel("Number of Reviews Received")
plt.legend(fields)
plt.savefig("../shared/figures/business_review_count_distribution", dpi=400)
plt.show()
fields = ['stars']
for field in fields:
  X, Y = histogramForField(businesses, field)
  plt.plot(X, np.log(Y))
plt.title("Review Count Distribution for Businesses")
plt.ylabel("Frequency")
plt.xlabel("Average Star Rating")
plt.legend(fields)
plt.savefig("../shared/figures/business_star_distribution", dpi=400)
plt.show()
