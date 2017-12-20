
# coding: utf-8

# link: https://www.kaggle.com/kswamy15/mercari-using-pytorch

# In[*]

import torch
from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F


# In[*]

import numpy as np
import pandas as pd
import time


# In[*]

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import math


# In[*]

train_df = pd.read_csv('data/mercari/train.tsv', sep='\t')
test_df = pd.read_csv('data/mercari/test.tsv', sep='\t')


# In[*]

train_df.describe()


# In[*]

train_df.shape


# In[*]

train_df.head()


# In[*]

print('Train columns with null values:\n', train_df.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', test_df.isnull().sum())
print("-"*10)

train_df.describe(include = 'all')


# Calculate how much of the brand names are not there.

# In[*]

print(632682/train_df.shape[0])


# Around 50% is not there hence we should probably not consider this.

# In[*]

train_df = train_df.drop(['brand_name'], axis=1)
test_df = test_df.drop(['brand_name'], axis=1)


# # categories

# We need to check how many categories there are

# In[*]

# Merge the two dataframes
frames = [train_df, test_df]
combined_df = pd.concat(frames)


# In[*]

combined_cat_df = combined_df['category_name']
def split_cat(text):
    try: return text.split("/")
    except: pass

combined_cat_df = combined_cat_df.apply(lambda x: split_cat(x))


# In[*]

def no_of_cats(cat_list):
    try: return len(cat_list)
    except: return 0
    
no_of_cats = pd.DataFrame(combined_cat_df.apply(lambda x: no_of_cats(x)))


# In[*]

# no_of_cats['category_name'].max(axis=1)
index_whr_max_categories = no_of_cats['category_name'].argmax()
print(index_whr_max_categories)
max_num_of_categories = len(split_cat(combined_df.iloc[[index_whr_max_categories]]['category_name'].tolist()[0]))
print('there are a maximum of {} categories and this is happened in row:'.format(max_num_of_categories))
combined_df.iloc[[index_whr_max_categories]]


# In[*]

def split_cat(text, max_num_of_categories):
    return_val = ["None"] * max_num_of_categories
    try:
        text_list = text.split("/") + return_val
        return text_list[:max_num_of_categories]
    except:
        return return_val


# Change the category name for train and test and total dataframes

# In[*]

train_df['category_name'] = train_df['category_name'].apply(lambda x: split_cat(x, max_num_of_categories))
test_df['category_name'] = test_df['category_name'].apply(lambda x: split_cat(x, max_num_of_categories))
combined_df['category_name'] = combined_df['category_name'].apply(lambda x: split_cat(x, max_num_of_categories))


# In[*]

train_df.head()


# now we know that there are 5 categories so we will try to find the unknown ones category per category. so we will make predictions based on the 5 categories

# ### Running category encoding on the first category

# In[*]

combined_cat1_list = [x[0] for x in combined_df['category_name'].tolist()]
combined_cat1_list[:5]


# In[*]

cat1_le = LabelEncoder()
cat1_le.fit(combined_cat1_list)


# In[*]

cat1_le.transform(['Men', 'Electronics', 'Women', 'Home', 'Women'])


# In[*]

cat1_le.inverse_transform([ 5,  1, 10,  3, 10])


# Thus we are able to build a label encoder state space for the first category

# In[*]




# In[*]




# In[*]




# In[*]




# In[*]

train_df[train_df.isnull().any(axis=1)]


# In[*]

train_df[train_df.isnull().any(axis=1)]


# In[*]

value_list = ['iPhone']
train_df[train_df.name.isin(value_list)]


# For the missing category names we should try to find some unsupervised learning so that some amount filling of the data should be present.

# # Running NLP on the categories

# We will first try to classify the documents and see if we can get some meaningful classification based on that.

# Idea is to use only the name to predict the category name

# So we will drop all the remaining columns

# In[*]

print(train_df.columns.tolist())


# 

# In[*]




# In[*]




# In[*]




# In[*]




# In[*]




# In[*]




# In[*]




# In[*]




# In[*]




# In[*]

from copy import deepcopy
category_df = deepcopy(train_df)


# In[*]

category_df = category_df.drop(['train_id', 'item_condition_id', 'price', 'shipping'], axis=1)


# In[*]

category_df.sample(2)


# In[*]

predict_category_df = category_df[pd.isnull(category_df['category_name'])]
train_test_categry_df = category_df[pd.notnull(category_df['category_name'])]
train_categry_df, test_categry_df = train_test_split(train_test_categry_df, test_size=0.2, random_state=42)
print('separated into predict, train and test')
print(category_df.shape, predict_category_df.shape, train_categry_df.shape, test_categry_df.shape)
print(predict_category_df.shape[0] + train_categry_df.shape[0] + test_categry_df.shape[0])


# In[*]

X_train_category_df = train_categry_df[['name', 'item_description']]
y_train_category_df = train_categry_df[['category_name']]
X_test_category_df = test_categry_df[['name', 'item_description']]
y_test_category_df = test_categry_df[['category_name']]
print('separate to x and y')
print(X_train_category_df.shape, y_train_category_df.shape, X_test_category_df.shape, y_test_category_df.shape)


# category names are based on parent -> sub category -> subcategory etc. Need to find how many categories are there.

# In[*]




# In[*]

y_train_category_df.head()


# In[*]




# In[*]

X_category_df = category_df[['name', 'item_description']]
y_category_df = category_df[['category_name']]

