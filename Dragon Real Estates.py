#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real Estates - Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv('housing.csv')


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


housing.hist(bins=50, figsize=(20,15))


# ## Train Test Splitting

# In[10]:


import numpy as np
# for learning purupose
def split_train_test(data, test_ratio):
    np.random.seed(42) 
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[11]:


train_set, test_set = split_train_test(housing, 0.2)


# In[12]:


# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[15]:


strat_train_set['CHAS'].value_counts()


# In[16]:


# 376/28


# In[17]:


# strat_test_set['CHAS'].value_counts()


# In[18]:


# 94/7


# In[19]:


housing = strat_train_set.copy()


# ## Looking for Correlations

# In[20]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[21]:


from pandas.plotting import scatter_matrix
attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(housing[attributes], figsize = (12,8))


# In[22]:


housing.plot(kind='scatter', x='RM', y='MEDV', alpha=0.8)


# # Trying out Attribute Combinations

# In[23]:


housing['TAXRM'] = housing['TAX']/housing['RM']


# In[24]:


housing.head()


# In[25]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[26]:


housing.plot(kind='scatter', x='TAXRM', y='MEDV', alpha=0.8)


# In[27]:


housing =  strat_train_set.drop('MEDV', axis=1)
housing_labels =  strat_train_set["MEDV"].copy()


# ## Missing Attributes

# In[28]:


# to take care of missing attributes
# 1. Get rid off the missing data points
# 2. Get rid off the whole attribute
# 3. Set the value to some value (0, mean or meadian)


# In[29]:


a= housing.dropna(subset=['RM']) #option 1
a.shape
# note that the original housing df will remain unchanged


# In[30]:


housing.drop('RM', axis=1).shape #option 2
# note that there is no RM column and also note that the original housing df will remain unchanged


# In[31]:


median = housing['RM'].median() #compute median for option 3


# In[32]:


housing['RM'].fillna(median) #option 3
# note that the original housing df will remain unchanged


# In[33]:


housing.shape


# In[34]:


housing.describe() # before we started filling missing attributes


# In[35]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
imputer.fit(housing)


# In[36]:


imputer.statistics_


# In[37]:


x = imputer.transform(housing)


# In[38]:


housing_tr = pd.DataFrame(x, columns= housing.columns)


# In[39]:


housing_tr.describe() #now rm count will increase from 500 to 501


# ## ScikitLearn Design

# Primarily, three types of objects
# 1. Estimators- It estimates some parameter based on a dataset. Eg imputer. It has a fit method and transform method.
# 

# ## Feature Scaling

# Primarily, two types of feature scaling methods:
# 1. Min-Max scaling (Normalization)
#     (value - min)/(max - min) ranges from 0 to 1
#     Sklearn proivdes a class MinMaxScaler for this
#     
# 2. Standardization
#     (value - mean)/std
#     Sklearn provides a class called Standard Scaler for this

# ## Creating a Pipeline

# In[40]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    #     ........ add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


# In[41]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[42]:


housing_num_tr.shape


# ## Selecting a desired model for Dragon Real Estates

# In[44]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[45]:


some_data = housing.iloc[:5]


# In[46]:


some_labels = housing_labels.iloc[:5]


# In[47]:


prepared_data = my_pipeline.transform(some_data)


# In[48]:


model.predict(prepared_data)


# In[49]:


list(some_labels)


# ## Evaluating the model

# In[50]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[51]:


rmse


# In[52]:


#due to its high mean sq error we will discard this model and we'll use decision tree regressor
# but decision tree regressor does overfiiting that's why mse is 0


# ## Using Better Evaluation Technique- Cross Valdiation

# In[53]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring= 'neg_mean_squared_error', cv= 10)
rmse_scores = np.sqrt(-scores)


# In[54]:


rmse_scores


# In[55]:


def print_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Std Dev: ', scores.std())


# In[56]:


print_scores(rmse_scores)


# ## Saving the model

# In[57]:


from joblib import dump, load
dump(model, 'Dragon.joblib')


# ## Testing the model on test data

# In[61]:


x_test = strat_test_set.drop('MEDV', axis=1)
y_test = strat_test_set['MEDV'].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(y_test))


# In[59]:


final_rmse


# In[63]:


prepared_data[0]


# ## Using the model

# In[64]:


from joblib import dump, load
import numpy as np
model = load('Dragon.joblib')
features = np.array([[-0.44241248,  34.18716752, -1.12581552, -0.27288841, -1.42038605,
       -99.54568298, -122.7412613 ,  9.56284386, -0.99534776, -0.57387797,
       -0.99428207,  0.43852974, -9.49833679]])
model.predict(features)


# In[ ]:




