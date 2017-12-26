
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
from sklearn.externals import joblib
import math
get_ipython().magic('matplotlib inline')


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

print('transform category name to first category as defined by the label encoding space for training set')
train_df['category_name'] = train_df['category_name'].apply(lambda x: convert_catname_cat1(cat1_le, x))
train_df.head()


# In[*]

print('transform category name to first category as defined by the label encoding space for test set')
test_df['category_name'] = test_df['category_name'].apply(lambda x: convert_catname_cat1(cat1_le, x))
test_df.head()


# There are some null values in item description so will need to make fill them.

# In[*]

print('presently number of null values in train and test.')
print(train_df['item_description'].isnull().sum())
print(test_df['item_description'].isnull().sum())


# In[*]

train_df['item_description'] = train_df['item_description'].fillna("")
test_df['item_description'] = test_df['item_description'].fillna("")
print('Num of null values in item description is for training set is {}.'.format(train_df['item_description'].isnull().sum()))
print('Num of null values in item description is for test set is {}.'.format(test_df['item_description'].isnull().sum()))
print('Ideally this number should be 0.')


# In[*]

train_df[train_df.isnull().any(axis=1)].head()


# In[*]

test_df[test_df.isnull().any(axis=1)].head()


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
    print(clf_descr, score, train_time, test_time)
    return clf


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
# clf = VotingClassifier(estimators=[
#     ('rc', RidgeClassifier(tol=1e-2)),
#     ('perc', Perceptron(n_iter=50)),
#     ('pa', PassiveAggressiveClassifier(n_iter=50))
# ])
clf = VotingClassifier(estimators=[
    ('rc', RidgeClassifier(tol=1e-2))
])


# In[*]

clf = fit_and_benchmark(clf, x_train, y_train_category_df, x_test, y_test_category_df, cat1_le.classes_)


# ### fill the category name for the missing values and build the matrix

# In[*]

train_df['total_text'] = train_df['name'] + " " +  train_df['item_description']
train_df.head()


# In[*]

test_df['total_text'] = test_df['name'] + " " +  test_df['item_description']
test_df.head()


# In[*]

train_df_with_no_cat = train_df[train_df['category_name'].isnull()]
train_df_with_no_cat.head()


# In[*]

test_df_with_no_cat = test_df[test_df['category_name'].isnull()]
test_df_with_no_cat.head()


# In[*]

def fill_and_transform_df(df):
    new_df = deepcopy(df)
    for index, row in df.iterrows():
        if pd.isnull(row['category_name']):
            new_df.loc[index]['category_name'] = vectorizer.transform([row['total_text']])
        else:
            new_df.loc[index]['category_name'] = row['category_name']
    return new_df


# In[*]

matrix_train_df = vectorizer.transform(train_df_with_no_cat['total_text'])
pred_train_df = clf.predict(matrix_train_df)
print(pred_train_df.shape, train_df_with_no_cat.shape)


# In[*]

matrix_test_df = vectorizer.transform(test_df_with_no_cat['total_text'])
pred_test_df = clf.predict(matrix_test_df)
print(pred_test_df.shape, test_df_with_no_cat.shape)


# fill the category_names with the predicted values wherever they are not present. This will be used in further predictions using pytorch.

# In[*]

print(train_df.loc[122])
i = 0
for index, row in train_df_with_no_cat.iterrows():
    train_df.loc[train_df.train_id == index, ['category_name']] = pred_train_df[i]
    i += 1
print(train_df.loc[122])


# In[*]

i = 0
for index, row in test_df_with_no_cat.iterrows():
    test_df.loc[test_df.test_id == index, ['category_name']] = pred_test_df[i]
    i += 1


# In[*]

train_df[train_df['category_name'].isnull()]


# In[*]

train_df.head()


# In[*]

fig, ax = plt.subplots(1, 1, figsize=(11, 7), sharex=True)
sns.distplot(np.log(train_df['price'].values+1))


# In[*]

# vectorized error calc
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
# Source: https://www.kaggle.com/jpopham91/rmlse-vectorized


# In[*]

print('lets see if there are some null values')
train_df.isnull().sum()


# In[*]

#PROCESS CATEGORICAL DATA
#print("Handling categorical variables...")
def encode_text(column):
    le = LabelEncoder()
    le.fit(np.hstack([train_df[column], test_df[column]]))
    train_df[column+'_index'] = le.transform(train_df[column])
    test_df[column+'_index'] = le.transform(test_df[column])


# In[*]

train_df.columns.to_series().groupby(train_df.dtypes).groups


# In[*]

train_df.select_dtypes(exclude=['float64', 'int64']).head()


# In[*]

train_df.head()


# In[*]

print('since brand name is not included now so the below code is not really required but keeping it for future consideration')
# encode_text('brand_name')


# In[*]

test_df.head()


# In[*]

class Category:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# In[*]

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
import unicodedata
import re
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    #s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def normalizeLine(sentence):
    return [normalizeString(s) for s in sentence.split('\t')]


# In[*]

def prepareData(lang1,data):
    input_cat = Category(lang1)
    for sentence in data:
        normalize_line = [normalizeString(s) for s in sentence.split('\t')]
        input_cat.addSentence(normalize_line[0])
        
    print("Counted words:")
    print(input_cat.name, input_cat.n_words)
    return input_cat


# In[*]

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    #indexes.append(EOS_token)
    return indexes


# In[*]

def token_fit(column):
    raw_text = np.hstack([(train_df[column]).str.lower(), (test_df[column]).str.lower()])
    cat1 = prepareData(column,raw_text)
    print ("adding train data")
    train_df[column + '_seq'] = [variableFromSentence(cat1,normalizeLine(sentence.lower())[0]) for sentence in train_df[column]]
    print ("adding test data")
    test_df[column + '_seq'] = [variableFromSentence(cat1,normalizeLine(sentence.lower())[0]) for sentence in test_df[column]]


# In[*]

token_fit('name')


# In[*]

token_fit('item_description')


# In[*]

train_df.head()


# In[*]

# save the csvs
train_df.to_csv('data/mercari/train.1.csv')
test_df.to_csv('data/mercari/test.1.csv')
print('transformed train and test data saved.')


# In[*]

# save the classifiers
from sklearn.externals import joblib
joblib.dump(clf, 'data/mercari/clf.pkl')
print('model is saved')


# In[*]

# load the csv and the model
clf = joblib.load('data/mercari/clf.pkl')
train_df = pd.read_csv('data/mercari/train.1.csv')
test_df = pd.read_csv('data/mercari/test.1.csv')


# In[*]

# this is needed because the dtypes of name_seq and item_description_seq is wrong
import ast
train_df['name_seq'] = train_df['name_seq'].apply(ast.literal_eval)
train_df['item_description_seq'] = train_df['item_description_seq'].apply(ast.literal_eval)
test_df['name_seq'] = test_df['name_seq'].apply(ast.literal_eval)
test_df['item_description_seq'] = test_df['item_description_seq'].apply(ast.literal_eval)


# In[*]

#SEQUENCES VARIABLES ANALYSIS
max_name_seq = np.max([np.max(train_df.name_seq.apply(lambda x: len(x))), np.max(test_df.name_seq.apply(lambda x: len(x)))])
max_item_description_seq = np.max([np.max(train_df.item_description_seq.apply(lambda x: len(x))),
                                   np.max(test_df.item_description_seq.apply(lambda x: len(x)))])
print("max name seq "+str(max_name_seq))
print("max item desc seq "+str(max_item_description_seq))


# In[*]

train_df.columns.to_series().groupby(train_df.dtypes).groups


# In[*]

train_df.name_seq.max()


# In[*]

#EMBEDDINGS MAX VALUE
#Base on the histograms, we select the next lengths
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_TEXT = np.max([np.max(train_df.name_seq.max()),
                   np.max(test_df.name_seq.max()),
                   np.max(train_df.item_description_seq.max()),
                   np.max(test_df.item_description_seq.max())])+2
# MAX_GEN_CATEGORY = np.max([train_df.general_cat_index.max(), test_df.general_cat_index.max()])+1
# MAX_SUB_CAT1_CATEGORY = np.max([train_df.subcat_1_index.max(), test_df.subcat_1_index.max()])+1
# MAX_SUB_CAT2_CATEGORY = np.max([train.subcat_2_index.max(), test.subcat_2_index.max()])+1
# MAX_BRAND = np.max([train.brand_name_index.max(), test.brand_name_index.max()])+1
MAX_CONDITION = np.max([train_df.item_condition_id.max(), test_df.item_condition_id.max()])+1
# MAX_CATEGORY_NAME = np.max([train_df.category_name_index.max(), test_df.category_name_index.max()])+1
MAX_CATEGORY_NAME = np.max([train_df.category_name.max(), test_df.category_name.max()])+1


# In[*]

#EXTRACT DEVELOPTMENT TEST
dtrain, dvalid = train_test_split(train_df, random_state=123, train_size=0.99)
print(dtrain.shape)
print(dvalid.shape)


# In[*]

def pad(tensor, length):
    if length > tensor.size(0):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
    else:
        return torch.split(tensor, length, dim=0)[0]


# In[*]

train_df.columns.tolist()


# In[*]

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        name, item_desc, cat_name, item_condition, shipping, target = sample['name'], sample['item_desc'], sample['cat_name'], sample['item_condition'], sample['shipping'], sample['target']
        return {
            'name': pad(torch.from_numpy(np.asarray(name)).long().view(-1),MAX_NAME_SEQ),
            'item_desc': pad(torch.from_numpy(np.asarray(item_desc)).long().view(-1),MAX_ITEM_DESC_SEQ),
            'cat_name':torch.from_numpy(np.asarray(cat_name)),
            'item_condition':torch.from_numpy(np.asarray(item_condition)),
            'shipping':torch.torch.from_numpy(np.asarray(shipping)),
            'target':torch.from_numpy(np.asarray(target))
        }


# In[*]

# Define the Dataset to use in a DataLoader
class MercariDataset(Dataset):
    """Mercari Challenge dataset."""

    def __init__(self, data_pd, transform=None):
        """
        Args:
            data_pd: Data frame with the used columns.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mercari_frame = data_pd
        self.transform = transform

    def __len__(self):
        return len(self.mercari_frame)

    def __getitem__(self, idx):
        name = [self.mercari_frame.name_seq.iloc[idx]]
        item_desc = [self.mercari_frame.item_description_seq.iloc[idx]]
#         brand_name = [self.mercari_frame.brand_name_index.iloc[idx]]
        cat_name = [self.mercari_frame.category_name_index.iloc[idx]]
#         general_category = [self.mercari_frame.general_cat_index.iloc[idx]]
#         subcat1_category = [self.mercari_frame.subcat_1_index.iloc[idx]]
#         subcat2_category = [self.mercari_frame.subcat_2_index.iloc[idx]]
        item_condition = [self.mercari_frame.item_condition_id.iloc[idx]]
        shipping = [self.mercari_frame.shipping.iloc[idx]]
        target = [self.mercari_frame.target.iloc[idx]]
        sample = {'name': name,
                'item_desc': item_desc,
#                'brand_name': brand_name,
               'cat_name': cat_name,   
#                'general_category': general_category,
#                'subcat1_category': subcat1_category,
#                'subcat2_category': subcat2_category,
               'item_condition': item_condition,
               'shipping': shipping,
               'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[*]

mercari_datasets = {
    'train': MercariDataset(dtrain,transform=transforms.Compose([ToTensor()])), 
    'val': MercariDataset(dvalid,transform=transforms.Compose([ToTensor()]))
}
dataset_sizes = {x: len(mercari_datasets[x]) for x in ['train', 'val']}


# In[*]

mercari_dataloaders = {
    x: torch.utils.data.DataLoader(mercari_datasets[x], batch_size=50, shuffle=True) for x in ['train', 'val']
}


# In[*]

mercari_dataloaders


# In[*]

# Some Useful Time functions
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[*]




# In[*]

# Definition of the Pytorch Model
class RegressionNeural(nn.Module):
    def __init__(self, max_sizes):
        super(RegressionNeural, self).__init__()
        # declaring all the embedding for the various items
        self.name_embedding = nn.Embedding(np.asscalar(max_sizes['max_text']), 50)
        self.item_embedding = nn.Embedding(np.asscalar(max_sizes['max_text']), 50)
#         self.brand_embedding = nn.Embedding(np.asscalar(max_sizes['max_brand']), 10)
#         self.gencat_embedding = nn.Embedding(np.asscalar(max_sizes['max_gen_category']), 10)
#         self.subcat1_embedding = nn.Embedding(np.asscalar(max_sizes['max_subcat1_category']), 10)
#         self.subcat2_embedding = nn.Embedding(np.asscalar(max_sizes['max_subcat2_category']), 10)
        self.condition_embedding = nn.Embedding(np.asscalar(max_sizes['max_condition']), 5)
        # I am adding an embedding just based on Category name without separating it into the 3 pieces
        self.catname_embedding = nn.Embedding(np.asscalar(max_sizes['max_cat_name']), 10)
        
        ## I am trying to throw all types of convolutional model on the name and item embedding and haven't
        ## had any luck.  
        #self.conv1_name = nn.Conv1d(max_sizes['max_name_seq'], 1, 3, stride=1)
        #self.conv1_item_desc = nn.Conv1d(max_sizes['max_item_desc_seq'], 1, 5, stride=5) 
        
        self.conv1_name = nn.Conv1d(50, 1, 2, stride=1)
        # I am not using these other convolutions as they didn't seem to improve my result
        self.conv2_name = nn.Conv1d(16, 8, 2, stride=1)
        self.conv3_name = nn.Conv1d(8, 4, 2, stride=1)
        
        self.conv1_item_desc = nn.Conv1d(50, 1, 5, stride=5) 
        # I am not using these other convolutions as they didn't see to improve my result
        self.conv2_item_desc = nn.Conv1d(64, 16, 5, stride=1)
        self.conv3_item_desc = nn.Conv1d(16, 4, 5, stride=1)
        
        #self.conv1 = nn.Conv1d(64, 32, 3, stride=1)
        #self.conv2 = nn.Conv1d(32, 16, 3, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        
        self.input_fc1_count = 50 #1214 #206 #16+10+10+10+10+5+1
        self.fc1 = nn.Linear(self.input_fc1_count, 64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,1)
        
        self.relu = nn.ReLU()  
            
    def forward(self, x, batchsize):
        embed_name = self.name_embedding(x['name'])
        #print ("embed_name size",embed_name.size())
        #embed_name = (self.conv1_name(embed_name))
        
        # I am swapping the Embedding size and the sequence length so that convolution is done across multiple words
        # using all the embeddings.  Without this, the 1-D convolution was doing convolution using all the words but
        # a slice of embeddings.  I don't think that is the correct way to do 1D convolution.  
        embed_name = F.relu(self.conv1_name(embed_name.transpose(1,2)))
        #print ("embed_name after 1st conv",embed_name.size())
        #embed_name = F.relu(self.conv2_name(embed_name))
        #print ("embed_name after 2nd conv",embed_name.size())
        #embed_name = self.conv3_name(embed_name)
        #print ("embed_name after 3rd conv",embed_name.size())
        
        embed_item = self.item_embedding(x['item_desc'])
        #print ("embed_item size",embed_item.size())
        #embed_item = (self.conv1_item_desc(embed_item))
        embed_item = F.relu(self.conv1_item_desc(embed_item.transpose(1,2)))
        #print ("embed_item after 1 conv",embed_item.size())
        #embed_item = F.relu(self.conv2_item_desc(embed_item))
        #print ("embed_item after 2 conv",embed_item.size())
        #embed_item = self.conv3_item_desc(embed_item)
        #print ("embed_item after 3rd conv",embed_item.size())
        
#         embed_brand = self.brand_embedding(x['brand_name'])
#         embed_gencat = self.gencat_embedding(x['general_category'])
#         embed_subcat1 = self.subcat1_embedding(x['subcat1_category'])
#         embed_subcat2 = self.subcat2_embedding(x['subcat2_category'])
        embed_condition = self.condition_embedding(x['item_condition'])
        embed_catname = self.catname_embedding(x['cat_name'])
        
        #out = torch.cat((embed_brand.view(batchsize,-1),embed_gencat.view(batchsize,-1), \
        #                 embed_subcat1.view(batchsize,-1), embed_subcat2.view(batchsize,-1), \
        #                 embed_condition.view(batchsize,-1),embed_name.view(batchsize,-1), \
        #                 embed_item.view(batchsize,-1),x['shipping']),1)
        out = torch.cat((embed_brand.view(batchsize,-1), embed_catname.view(batchsize,-1),                          embed_condition.view(batchsize,-1),embed_name.view(batchsize,-1),                          embed_item.view(batchsize,-1),x['shipping']),1)
        #out = self.dropout(out)
        
        out = (self.fc1(out))
        out = F.relu(self.dropout(out))
        out = (self.fc2(out))
        out = (self.dropout(out))
        out = self.fc3(out)
        return out

max_sizes = {
    'max_text':MAX_TEXT,'max_name_seq':MAX_NAME_SEQ,'max_item_desc_seq':MAX_ITEM_DESC_SEQ, 
#     'max_brand':MAX_BRAND,'max_cat_name':MAX_CATEGORY_NAME,'max_gen_category':MAX_GEN_CATEGORY,
    'max_cat_name':MAX_CATEGORY_NAME,
#     'max_subcat1_category':MAX_SUB_CAT1_CATEGORY,'max_subcat2_category':MAX_SUB_CAT2_CATEGORY,
    'max_condition':MAX_CONDITION
}
max_sizes = {k:int(v) for k, v in max_sizes.items()}

deep_learn_model = RegressionNeural(max_sizes)


# In[*]

# Training model function that uses the dataloader to load the data by Batch
def train_model(model, criterion, optimizer, num_epochs=1, print_every = 100):
    start = time.time()

    best_acc = 0.0
    print_loss_total = 0  # Reset every print_every

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            num_batches = dataset_sizes[phase]/50.
            #running_corrects = 0

            # Iterate over data.
            for i_batch, sample_batched in enumerate(mercari_dataloaders[phase]): 
            # get the inputs
                inputs = {'name':Variable(sample_batched['name']), 'item_desc':Variable(sample_batched['item_desc']),                     'brand_name':Variable(sample_batched['brand_name']),                     'cat_name':Variable(sample_batched['cat_name']),                     'general_category':Variable(sample_batched['general_category']),                     'subcat1_category':Variable(sample_batched['subcat1_category']),                     'subcat2_category':Variable(sample_batched['subcat2_category']),                     'item_condition':Variable(sample_batched['item_condition']),                     'shipping':Variable(sample_batched['shipping'].float())}
                prices = Variable(sample_batched['target'].float())   
                batch_size = len(sample_batched['shipping'])   
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs, batch_size)
                #_, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, prices)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                print_loss_total += loss.data[0]
                #running_corrects += torch.sum(preds == labels.data)
                
                
                if (i_batch+1) % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    #print (i_batch / num_batches, i_batch, num_batches)
                    print('%s (%d %d%%) %.4f' % (timeSince(start, i_batch / num_batches),                                                  i_batch, i_batch / num_batches*100, print_loss_avg))
                
                # I have put this just so that the Kernel will run and allow me to publish
                if (i_batch) > 500:
                    break

            epoch_loss = running_loss / num_batches
            #epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
        print()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model


# In[*]

# Set the optimizer Criterion and train the model
criterion = nn.MSELoss()

optimizer_ft = optim.SGD(deep_learn_model.parameters(), lr=0.001, momentum=0.9)
#optimizer_ft = optim.SGD(deep_learn_model.parameters(), lr=0.005)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
train_model(deep_learn_model,criterion,optimizer_ft)
# I have run the model a lot of times with different combination of deep learning configs and I am not able to
# get a loss below 0.0341. 


# In[*]

# Function to calculate the RMSLE on the validation data
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5


# In[*]

# Validate the model results against validation data
def validate(model, print_every = 20, phase = 'val'):
    start = time.time()
    running_loss = 0
    print_loss_total = 0
    num_batches = dataset_sizes[phase]/50.
    y_pred_full = np.array([])
    y_true_full = np.array([])
    for i_batch, sample_batched in enumerate(mercari_dataloaders[phase]): 
    # get the inputs
        inputs = {'name':Variable(sample_batched['name']), 'item_desc':Variable(sample_batched['item_desc']),             'brand_name':Variable(sample_batched['brand_name']),             'cat_name':Variable(sample_batched['cat_name']),             'general_category':Variable(sample_batched['general_category']),             'subcat1_category':Variable(sample_batched['subcat1_category']),             'subcat2_category':Variable(sample_batched['subcat2_category']),             'item_condition':Variable(sample_batched['item_condition']),             'shipping':Variable(sample_batched['shipping'].float())}
        prices = Variable(sample_batched['target'].float())   
        batch_size = len(sample_batched['shipping'])

        # forward
        outputs = model(inputs,batch_size)
        val_preds = target_scaler.inverse_transform(outputs.data.numpy())
        val_preds = np.exp(val_preds)-1
        val_true =  target_scaler.inverse_transform(prices.data.numpy())
        val_true = np.exp(val_true)-1

        #mean_absolute_error, mean_squared_log_error
        y_true = val_true[:,0]
        y_pred = val_preds[:,0]
        y_true_full = np.append(y_true_full,y_true)
        y_pred_full= np.append(y_pred_full,y_pred)
        
        loss = criterion(outputs, prices)
        #print ("output size", val_preds.shape)
        #print ("ypred_full",len(y_pred_full))

        # statistics
        running_loss += loss.data[0]
        print_loss_total += loss.data[0]
        #print("loss data shape", loss.data.size())
        #print("running loss", running_loss)
        #running_corrects += torch.sum(preds == labels.data)


        if (i_batch+1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            #print (i_batch / num_batches, i_batch, num_batches)
            print('%s (%d %d%%) %.4f' % (timeSince(start, i_batch / num_batches),                                          i_batch, i_batch / num_batches*100, print_loss_avg))

    v_rmsle = rmsle(y_true_full, y_pred_full)
    print(" RMSLE error on dev validate: "+str(v_rmsle))
    print("total loss", running_loss / num_batches ) 
    return y_pred_full, y_true_full


# In[*]

# You can see the RMSE loss on validation data is very poor.  
y_pred_val, y_true_val = validate(deep_learn_model)


# In[*]

axes = plt.gca()
axes.set_ylim([0,100])
plt.scatter(y_pred_val,y_true_val)


# In[*]

get_ipython().system('ls -ltr data')


# In[*]

### Implement and save submission
data_val['Survived'] = voting_hard.predict(data_val[data1_x_bin])

#submit file
submit = data_val[['PassengerId','Survived']]
submit.to_csv("data/mercari/submit.csv", index=False)

print('Validation Data Distribution: \n', data_val['Survived'].value_counts(normalize = True))
submit.sample(10)

