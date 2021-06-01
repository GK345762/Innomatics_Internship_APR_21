#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# #### Dataset - adult.csv : 
# 
# https://drive.google.com/file/d/1GtwjrZpP6jiZWXyYShiMHBamUstFgaEF/view?usp=sharing

# # Dataset Description
# 
# An individual’s annual income results from various factors. Intuitively, it is influenced by the individual’s education level, age, gender, occupation, and etc.
# 
# #### Fields
# 
# The dataset contains 16 columns
# 
# Target filed: Income
# 
# -- The income is divide into two classes: 50K
# 
# Number of attributes: 14
# 
# -- These are the demographics and other features to describe a person
# 
# We can explore the possibility in predicting income level based on the individual’s personal information.
# 
# #### Acknowledgements
# 
# This dataset named “adult” is found in the UCI machine learning repository
# http://www.cs.toronto.edu/~delve/data/adult/desc.html
# 
# The detailed description on the dataset can be found in the original UCI documentation
# http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

# #### The Adult dataset
# 
# The information is a replica of the notes for the abalone dataset from the UCI repository.
# 
# #### 1. Title of Database: adult
# #### 2. Sources:
# 
# (a) Original owners of database (name/phone/snail address/email address)
#     US Census Bureau.
#     
# (b) Donor of database (name/phone/snail address/email address)
#     Ronny Kohavi and Barry Becker,
#     Data Mining and Visualization
#     Silicon Graphics.
#     e-mail: ronnyk@sgi.com
#     
# (c) Date received (databases may change over time without name change!)
#     05/19/96
#     
# #### 3. Past Usage:
# 
# (a) Complete reference of article where it was described/used
#     @inproceedings{kohavi-nbtree,
#     author={Ron Kohavi},
#     title={Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid},
#     booktitle={Proceedings of the Second International Conference on Knowledge Discovery and Data Mining},
#     year = 1996,
#     pages={to appear}}
#     
# (b) Indication of what attribute(s) were being predicted
#     Salary greater or less than 50,000.
#     
# (b) Indication of study's results (i.e. Is it a good domain to use?)
#     Hard domain with a nice number of records.
#     The following results obtained using MLC++ with default settings
#     for the algorithms mentioned below.
#     
# Algorithm	                Error
# 1	C4.5	                15.54
# 2	C4.5-auto	            14.46
# 3	C4.5-rules	            14.94
# 4	Voted ID3 (0.6)	        15.64
# 5	Voted ID3 (0.8)	        16.47
# 6	T2	                    16.84
# 7	1R	                    19.54
# 8	NBTree	                14.10
# 9	CN2	                    16.00
# 10	HOODG	                14.82
# 11	FSS Naive Bayes	        14.05
# 12	IDTM (Decision table)	14.46
# 13	Naive-Bayes	            16.12
# 14	Nearest-neighbor (1)	21.42
# 15	Nearest-neighbor (3)	20.35
# 16	OC1	                    15.04
# 17	Pebls	                Crashed. Unknown why (bounds WERE increased)
# 
# ##### 4. Relevant Information Paragraph:
# 
# Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
# 
# #### 5. Number of Instances
# 
#   48842 instances, mix of continuous and discrete (train=32561, test=16281)
#   45222 if instances with unknown values are removed (train=30162, test=15060)
#   Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
#   
# #### 6. Number of Attributes
# 
#   6 continuous, 8 nominal attributes.
#   
# #### 7. Attribute Information:
# 
#   01. age: continuous.
#   02. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
#   03. fnlwgt: continuous.
#   04. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-         4th, 10th, Doctorate, 5th-6th, Preschool.
#   05. education-num: continuous.
#   06. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
#   07. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-      op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
#   08. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
#   09. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
#   10. sex: Female, Male.
#   11. capital-gain: continuous.
#   12. capital-loss: continuous.
#   13. hours-per-week: continuous.
#   14. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan,       Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland,             France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand,           Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
#       class: >50K, <=50K
#       
# #### 8. Missing Attribute Values:
#  
# 7% have missing values.
#  
# #### 9. Class Distribution:
#  
# Probability for the label '>50K' : 23.93% / 24.78% (without unknowns)
# Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
# 
# #### 10. Notes for Delve
# 
#   1. One prototask (income) has been defined, using attributes 1-13 as inputs and income level as a binary target.
#   2. Missing values - These are confined to attributes 2 (workclass), 7 (occupation) and 14 (native-country). The prototask only      uses cases with no missing values.
#   3. The income prototask comes with two priors, differing according to if attribute 4 (education) is considered to be nominal        or ordinal.

# In[2]:


df = pd.read_csv('F:/Innomatics_Internship_APR_21/Task - 11 (EDA + Missing Value + Outlier + ML Model)/adult.csv')


# In[3]:


df.head()


# #### Step - 1 - Introduction -> Give a detailed data description and objective

# #### 1. Introduction 
# 
# A census is the procedure of systematically acquiring and recording information about the members of a given population. The census is a special, wide-range activity, which takes place once a decade in the entire country. The purpose is to gather information about the general population, in order to present a full and reliable picture of the population in the country - its housing conditions and demographic, social and economic characteristics. The information collected includes data on age, gender, country of origin, marital status, housing conditions, marriage, education, employment, etc.

# #### 1.1 Data description
# 
# This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). The prediction task is to determine whether a person makes over $50K a year.

# #### 1.2 Features Description
# 
# ##### 1. Categorical Attributes
# 
# 1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
#               Individual work category
# 2. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th,               10th, Doctorate, 5th-6th, Preschool.
#               Individual's highest education degree
# 3. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
#               Individual marital status
# 4. occupation:Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-               inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
#               Individual's occupation
# 5. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
#               Individual's relation in a family
# 6. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
#               Race of Individual
# 7. sex: Female, Male.
# 8. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan,                   Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal,                     Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua,                     Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
#               Individual's native country
#               
# ##### 2. Continuous Attributes
# 
# 1. age: continuous.
#         Age of an individual
# 2. fnlwgt: final weight, continuous.
#         The weights on the CPS files are controlled to independent estimates of the civilian noninstitutional population of      the US. These are prepared monthly for us by Population Division here at the Census Bureau.
# 3. capital-gain: continuous.
#         capital-loss: continuous.
# 4. hours-per-week: continuous.
#         Individual's working hour per week

# #### 1.3 Objective of this project
# 
# The goal of this machine learning project is to predict whether a person makes over 50K a year or not given their demographic variation. This is a classification problem.

# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.describe(include='object')


# In[7]:


df.info()


# ##### There are 15 columns and 48842 rows. Our datas datatypes are int64 and object. Let's check more.

# In[8]:


head5 = df.head()
head5


# In[9]:


print('Size of Data: ',df.size)


# In[10]:


# exploring data statistically
statisticalSummary = df.describe()
statisticalSummary


# ##  Data Preprocessing¶

# In[11]:


df.isna().sum()


# In[12]:


df['age'].value_counts()


# #### Filtering Data

# In[13]:


greater25 = df.get(df['age'] >= 25)
greater25 


# In[14]:


lesser25 = df.get(df['age'] < 25)
lesser25 


# #### Data Cleaning
# 

# In[15]:


# Check duplicate data exist

check_dup = df.duplicated().any() 
print("Are there any duplicated values in data? ",check_dup)

if check_dup:
  df = df.drop_duplicates()
else:
    print("There are not duplicated values in data.")


# In[16]:


# Firstly we should chechk if there is NaN values.

isMissing=df.isna().any() # There are no missing values. But let's examine a bit more carefully.. 

print(df.head(20)) # we show that there are values , '?'.

print(isMissing) # Actually '?' values are missing values. So...

df_nan= df.replace('?',np.nan) # I replaced '?' values with Nan

isMissingNow=df_nan.isna().any() # Now we can see there are NaN/missing values. 

# Note : Trick point is seeing '?' values.Of course you can drop values '?' without using converting NaN.But i guess this way is so clear.

print(isMissingNow) 

df_nan


# In[17]:


# There are more ways hande these missing values .I practiced two ways; drop nan values and fill with mean.

# First just drop nan values.

df_no_missing_values = df_nan.dropna()

df_no_missing_values.notna().any() # If there is no nan values , return True.


# In[18]:


# Second fill missing values with mean.

# I examined data above and see that there are no nan value in categorical values.

# So I choosed numerical columns have missing values and saved them in an array.

cols_m = [1,3,5,6,7,8,9,13,14] # columns have missing values

col_mods =[] # average or mod values for these columns.

for x in cols_m :

  c = df_nan.columns[x] # get column name 
  mod= df_nan[c].mode()[0] # calculate mod/average/mean of that column.
  col_mods.append(mod) # add mod/average/mean values .
  df_filled = df_nan # copy df_nan
  df_filled[c]=df_filled[c].replace(np.nan,mod) # fill values .

print("NaN data var mı ? \n------------------")

print(df_filled.isna().any())

df_filled

#Note: to do this , also you can just use df.fillna(np.mean(df[column_name]),axis=1) 


# ### Data Visualization

# #### Univariate Analysis

# In[19]:


# "Age" distibution using histogram, pdf and cdf 

import matplotlib.pyplot as plt
import seaborn as sns

dt = df_filled
sns.set()

fig, eks = plt.subplots(1,2, figsize=(15,5),dpi=120)
eks[0].set_title("Histogram w/ PDF")
eks[1].set_title("Histogram w/ CDF")
args_cum = {"cumulative":True}
sns.distplot(dt["age"], bins=20, kde=False, axlabel='Ages', ax=eks[0]) 
sns.distplot(dt["age"], bins=20, kde=False, axlabel='Ages',hist_kws=args_cum, ax=eks[1])


# In[20]:


# Count plot for categorical attibute "education","martial-status" and "occupation".

ser1 = df_filled["education"]
ser2 = df_filled["marital-status"]
ser3 = df_filled["occupation"]
order = ser1.append(ser2.append(ser3)).unique() 

plt.figure(figsize=(15,10))

ax = sns.countplot(x=ser3, palette='Set1', order=order, zorder=3) #plotting process
ax = sns.countplot(x=ser2, palette='Set2', order=order, zorder=2)
ax = sns.countplot(x=ser1, palette='Set3', order=order)
ax = ax.set_xticklabels(ax.get_xticklabels(),  
                      rotation=75, 
                      horizontalalignment='right',
                      fontweight='light',
                      fontsize='x-large')
plt.xlabel("Labels")
plt.title("[Education | Marital-status  | Occupation ] Countplot")
plt.show()


# #### Bivariate Analysis

# In[21]:


plt.figure(figsize=(8,6))
chart = sns.scatterplot(x="age",y='education',data=df_filled,hue='gender') 
plt.show(chart) #plotting..


# #### Multivariate Analysis

# In[22]:


correlation = df_filled.corr() # calculate correlation 
plt.figure(figsize=(15,10)) # plot size 
heatmap = sns.heatmap(correlation,vmin=-1,vmax=1,annot=True) 
heatmap.set_title("Correlations of features' Heatmap" , fontdict={'fontsize':12}, pad=12);


# In[23]:


plt.figure(figsize=(10,5))
sns.pairplot(df_filled)


# #### KNN

# In[24]:


df.head()


# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


one_hot = pd.get_dummies(df["race"])
df = df.drop("race",axis=1)
df = df.join(one_hot)
df.head()


# In[27]:


one_hot = pd.get_dummies(df["workclass"])
df = df.drop("workclass",axis=1)
df = df.join(one_hot)
df.head()


# In[28]:


one_hot = pd.get_dummies(df["gender"])
df = df.drop("gender",axis=1)
df = df.join(one_hot)
df.head()


# In[29]:


df.columns


# In[30]:


df['income'].value_counts()


# In[31]:


df['income'] = df['income'].map({'<=50K':0,'>50K':1})


# In[32]:


df.head()


# In[33]:


y = df['income']


# In[34]:


x = df[['age', 'educational-num','capital-gain', 'capital-loss',
       'hours-per-week','Amer-Indian-Eskimo','Asian-Pac-Islander', 'Black', 
        'Other', 'White', 'Female', 'Male', '?','Federal-gov', 'Local-gov',
        'Never-worked', 'Private', 'Self-emp-inc','Self-emp-not-inc','State-gov'
        ,'Without-pay']]


# In[35]:


y


# In[36]:


x


# In[37]:


x.columns


# In[38]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0,stratify=y) 


# In[39]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[40]:


scores=[]
for k in range(1,30):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    print('Accuracy for k=',k,'is:',round(accuracy_score(y_pred,y_test),2))
    if(k > 1 and accuracy_score(y_pred,y_test) > max(scores)):
      final_model = KNeighborsClassifier(k)
      print("Saving Model")
    scores.append(round(accuracy_score(y_pred,y_test),2))
final_model.fit(x_train,y_train)


# In[41]:


y_pred=final_model.predict(x_test)
print('Accuracy for k=',k,'is:',round(accuracy_score(y_pred,y_test),2))


# ### RandomForest

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


# In[44]:


df.head()


# In[45]:


#Finding the special characters in the data frame 
df.isin(['?']).sum(axis=0)


# In[48]:


#replacing those values to nan
df['native-country'] = df['native-country'].replace('?',np.nan)
df['occupation'] = df['occupation'].replace('?',np.nan)


# In[49]:


#droping the row which contain NaN values
df.dropna(how='any',inplace=True)


# In[50]:


#running a loop of value_counts of each column to find out unique values. 
for c in df.columns:
    print ("---- %s ---" % c)
    print (df[c].value_counts())


# In[52]:


df.drop(['educational-num','age', 'hours-per-week', 'fnlwgt', 'capital-gain','capital-loss', 'native-country'], axis=1, inplace=True)


# #### Data Visualization

# In[56]:


df.groupby('education').income.mean().plot(kind='bar')


# #### RandomForest

# In[59]:


para_forest={
 'n_estimators':[60,80,100],
 'criterion': ["gini", "entropy"],
 'max_depth': [None,5 , 10],
 'max_features': ['auto',10,20,30]    
}


# In[60]:


grid_rf=GridSearchCV(estimator=RandomForestClassifier(),param_grid=para_forest,cv=3,n_jobs=-1).fit(x_train,y_train)


# In[61]:


grid_rf.best_params_


# In[62]:


rf=RandomForestClassifier(criterion = 'gini',
 max_depth = 10,
 max_features = 'auto',
 n_estimators = 80)


# In[63]:


rf.fit(x_train,y_train)


# In[64]:


rf_pred=rf.predict(x_test)
rf_pred


# In[65]:


def matt(a,b):
    print('Confusion Matrix')
    print(metrics.confusion_matrix(a,b))
    print('---------------')
    print("Accuracy:",metrics.accuracy_score(a,b))
    print("Recall:",metrics.recall_score(a,b))
    print("precision:",metrics.precision_score(a,b))
    print("F1_Score:",metrics.f1_score(a,b))


# In[66]:


matt(y_test, rf_pred)


# #### Logistic regression

# In[67]:


para_reg={'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.001,0.01,0.1,1,10,100,1000],
          'fit_intercept': [False,True],
          'verbose': [0,1,2,3],
          'multi_class': ['auto', 'ovr', 'multinomial']
}


# In[68]:


grid_reg=GridSearchCV(estimator=LogisticRegression(),param_grid=para_reg,cv=3,n_jobs=-1).fit(x_train,y_train)


# In[69]:


grid_reg.best_params_


# In[70]:


reg = LogisticRegression(C = 0.1,
 fit_intercept = False,
 multi_class = 'multinomial',
 penalty = 'l2',
 verbose = 0)


# In[71]:


reg.fit(x_train,y_train)


# In[72]:


reg_pred=reg.predict(x_test)


# In[73]:


matt(y_test, reg_pred)


# #### KNeighborsClassifier

# In[74]:


p_3={
 'n_neighbors': [1,3,5],  
 'weights': ['uniform','distance'],
   'leaf_size' : [45,70,90]
 }


# In[75]:


grid_knn=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=p_3,cv=3,n_jobs=-1).fit(x_train,y_train)


# In[76]:


grid_knn.best_params_


# In[77]:


knn=KNeighborsClassifier(n_neighbors = 5,  
 weights = 'uniform',
   leaf_size = 70 )


# In[78]:


knn.fit(x_train,y_train)


# In[79]:


knn_pred=knn.predict(x_test)


# In[80]:


matt(y_test, knn_pred)


# #### Conclusion

# After using the 3 different model we are getting accuracy like LogisticRegression < KNeighbors < Random Forest. Hence, Random forest is best for this datset.

# #### Prediction on unknown data

# Now i will use unknow data set as to train my best model and will try to predict.

# In[83]:


unknown_data = []
for i in range(len(df.columns)):
    unknown_data.append(int(input('Enter value for '+df.columns[i]+' : ')))
    
    
t = rf.predict([unknown_data])


# In[84]:


if t == 0:
    print('Income will be equal to or less than 50k')
else:
    print('Income will be more than 50k')

