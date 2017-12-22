
# coding: utf-8

# References:
# 
# https://www.kaggle.com/kswamy15/mercari-using-pytorch
# 
# http://scikit-learn.org/stable/modules/ensemble.html

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
from time import time


# In[*]

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from sklearn import metrics
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
combined_cat1_list = [x for x in combined_cat1_list if not x == 'None']


# In[*]

cat1_le = LabelEncoder()
cat1_le.fit(combined_cat1_list)


# In[*]

cat1_le.transform(['Men', 'Electronics', 'Women', 'Home', 'Women'])


# In[*]

cat1_le.inverse_transform([5, 1, 9, 3, 9])


# Thus we are able to build a label encoder state space for the first category

# In[*]

def convert_catname_cat1(le, catlist):
    try:
        return le.transform(catlist[:1])[0]
    except:
        return np.nan


# In[*]

train_df['category_name'] = train_df['category_name'].apply(lambda x: convert_catname_cat1(cat1_le, x))


# In[*]

train_df.head()


# There are some null values in item description so will need to make fill them

# In[*]

train_df[train_df['item_description'].isnull()]


# In[*]

train_df['item_description'] = train_df['item_description'].fillna("")


# In[*]

train_df[train_df['item_description'].isnull()]


# No null values in item description

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

predict_category_df.head()


# In[*]

X_train_category_df = train_categry_df[['name', 'item_description']]
y_train_category_df = train_categry_df[['category_name']]
X_test_category_df = test_categry_df[['name', 'item_description']]
y_test_category_df = test_categry_df[['category_name']]
print('separate to x and y')
print(X_train_category_df.shape, y_train_category_df.shape, X_test_category_df.shape, y_test_category_df.shape)


# In[*]

y_train_category_df.head()


# In[*]

y_test_category_df.head()


# Combine the name and item_description

# In[*]

X_train_category_df.head()


# In[*]

X_train_category_df['total_text'] = X_train_category_df['name'] + " " +  X_train_category_df['item_description']
X_train_category_df.head()


# In[*]

X_test_category_df['total_text'] = X_test_category_df['name'] + " " +  X_test_category_df['item_description']
X_test_category_df.head()


# In[*]

print('Extracting features from the training data using a sparse vectorizer')
t0 = time()
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
x_train = vectorizer.fit_transform(X_train_category_df['total_text'])
duration = time() - t0
print("done in %fs" % (duration))    
print("n_samples: %d, n_features: %d" % x_train.shape)
print()


# In[*]

print('Extracting features from the testing data using a sparse vectorizer')
t0 = time()
x_test = vectorizer.transform(X_test_category_df['total_text'])
duration = time() - t0
print("done in %fs" % (duration))    
print("n_samples: %d, n_features: %d" % x_train.shape)
print()


# In[*]

feature_names = vectorizer.get_feature_names()


# In[*]

# #############################################################################  
# Benchmark classifiers                                                          
def fit_and_benchmark(clf, X_train, y_train, X_test, y_test, target_names):                                                              
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
                                                                                 
    if hasattr(clf, 'coef_'):                                                    
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
                                                                                 
        if feature_names is not None:                       
            print("top 10 keywords per class:")                                  
            for i, label in enumerate(target_names):                             
                top10 = np.argsort(clf.coef_[i])[-10:]
        print()
                                                                                 
    print("classification report:")
    print(metrics.classification_report(y_test, pred, target_names=target_names))          
                                                                                 
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))                            
                                                                                 
    print()                                                                      
    clf_descr = str(clf).split('(')[0]                                           
    return clf_descr, score, train_time, test_time 


# In[*]

# clf = VotingClassifier(estimators=[
#     ('rc', RidgeClassifier(tol=1e-2)),
#     ('perc', Perceptron(n_iter=50)),
#     ('pa', PassiveAggressiveClassifier(n_iter=50)),
#     ('knn', KNeighborsClassifier(n_neighbors=len(cat1_le.classes_))),
#     ('rfc', RandomForestClassifier(n_estimators=100)),
#     ('sgd', SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")),
#     ('SVC_with_L1', Pipeline([
#         ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),
#         ('classification', LinearSVC(penalty="l2"))]))
# ])
clf = VotingClassifier(estimators=[
    ('rc', RidgeClassifier(tol=1e-2)),
    ('perc', Perceptron(n_iter=50)),
    ('pa', PassiveAggressiveClassifier(n_iter=50))
])


# In[*]

fit_and_benchmark(clf, x_train, y_train_category_df, x_test, y_test_category_df, cat1_le.classes_)


# In[*]

y_train_category_df.head()


# In[*]

len(cat1_le.classes_)


# In[*]

X_category_df = category_df[['name', 'item_description']]
y_category_df = category_df[['category_name']]

