
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

train_df.drop(['brand_name'], axis=1)

