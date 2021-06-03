#!/usr/bin/env python
# coding: utf-8

# ### PUBG Data Description

# In a PUBG game, up to 100 players start in each match (matchId). Players can be on teams (groupId) which get ranked at the end of the game (winPlacePerc) based on how many other teams are still alive when they are eliminated. In game, players can pick up different munitions, revive downed-but-not-out (knocked) teammates, drive vehicles, swim, run, shoot, and experience all of the consequences -- such as falling too far or running themselves over and eliminating themselves.
# You are provided with a large number of anonymized PUBG game stats, formatted so that each row contains one player's post-game stats. The data comes from matches of all types: solos, duos, squads, and custom; there is no guarantee of there being 100 players per match, nor at most 4 players per group.

# ##### Data fields
# 
# 01. DBNOs - Number of enemy players knocked.
# 02. assists - Number of enemy players this player damaged that were killed by teammates.
# 03. boosts - Number of boost items used.
# 04. damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.
# 05. headshotKills - Number of enemy players killed with headshots.
# 06. heals - Number of healing items used.
# 07. Id - Player’s Id
# 08. killPlace - Ranking in match of number of enemy players killed.
# 09. killPoints - Kills-based external ranking of players. (Think of this as an Elo ranking where only kills matter.) If there is     a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
# 10. killStreaks - Max number of enemy players killed in a short amount of time.
# 11. kills - Number of enemy players killed.
# 12. longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a           player and driving away may lead to a large longestKill stat.
# 13. matchDuration - Duration of match in seconds.
# 14. matchId - ID to identify matches. There are no matches that are in both the training and testing set.
# 15. matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-     fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
# 16. rankPoints - Elo-like ranking of players. This ranking is inconsistent and is being deprecated in the API’s next version, so     use with caution. Value of -1 takes the place of “None”.
# 17. revives - Number of times this player revived teammates.
# 18. rideDistance - Total distance traveled in vehicles measured in meters.
# 19. roadKills - Number of kills while in a vehicle.
# 20. swimDistance - Total distance traveled by swimming measured in meters.
# 21. teamKills - Number of times this player killed a teammate.
# 22. vehicleDestroys - Number of vehicles destroyed.
# 23. walkDistance - Total distance traveled on foot measured in meters.
# 24. weaponsAcquired - Number of weapons picked up.
# 25. winPoints - Win-based external ranking of players. (Think of this as an Elo ranking where only winning matters.) If there is     a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
# 26. groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a       different groupId each time.
# 27. numGroups - Number of groups we have data for in the match.
# 28. maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips         over placements.
# 29. winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0         corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing       chunks in a match.
# 

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('F:/Innomatics_Internship_APR_21/PUBG/pubg.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


print('Size of Data: ',df.size)


# In[6]:


# exploring data statistically
statisticalSummary = df.describe()
statisticalSummary


# In[7]:


numerical_subset = df.select_dtypes(include=['int64', 'float64'])
numerical_subset


# In[8]:


object_subset = df.select_dtypes(include=['object'])
object_subset


# In[9]:


df['matchType'].value_counts()


# In[10]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit_transform(df['matchType'])


# In[11]:


numerical_subset.hist(bins=20, figsize=(20, 15))
plt.show()


# ### Descriptive Analysis

# After we have explored the overview of the dataset, we are going to dig out any anomaly in the dataset and clean it out.

# In[12]:


df.isna().sum()


# In[13]:


data = df.copy()
data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'
plt.figure(figsize=(15,10))
sns.countplot(data['kills'].astype('str').sort_values())
plt.title("Kill Count",fontsize=15)
plt.show()


# In[14]:


data_types = df[df['matchType'].isin(['duo','duo-fpp','solo','solo-fpp','squad','squad-fpp'])]
plt.figure(figsize=(15,10))
sns.countplot(data_types['matchType'].astype('str').sort_values())
plt.title("Match Types",fontsize=15)
plt.show()


# In[15]:


data_types = df[df['matchType'].isin(['duo','duo-fpp','solo','solo-fpp','squad','squad-fpp'])]
plt.figure(figsize=(15,10))
sns.countplot(data_types['matchType'].astype('str').sort_values())
plt.title("Match Types",fontsize=15)
plt.show()


# In[16]:


kills = df['kills']

print('Mode:', kills.mode()[0])
print('Median:', kills.median())
print('Mean:', kills.mean())
print('Range:', kills.max() - kills.min())
print('S.E. mean:', kills.std() / np.sqrt(kills.count()))
IQR = kills.quantile(0.75)-kills.quantile(0.25)
print('IQR:', IQR)
print('IQR deviation:', IQR / 2)

kills_non_zero = kills[kills > 0]
print('Non-zero decile ratio:', kills_non_zero.quantile(0.9) / kills_non_zero.quantile(0.1))


# In[17]:


plt.figure(figsize=(15,10))
plt.title("Walking Distance Distribution",fontsize=15)
walkdistance = df['walkDistance']
sns.distplot(walkdistance)
plt.show()


# In[18]:


from scipy.stats import norm, kstest
from scipy.stats import kurtosis, skew
from scipy import stats

loc, scale = norm.fit(walkdistance)
# create a normal distribution with loc and scale
n = norm(loc=loc, scale=scale)
kstest(walkdistance, n.cdf)


# ### p-value < 0.05 => not normally distributed

# In[19]:


print('kurtosis of distribution: {}'.format(kurtosis(walkdistance)))
print('skewness of distribution: {}'.format(skew(walkdistance)))

Kurtosis < 3 => platicurtic
Skewness > 1 => positively skewed
# In[20]:


plt.figure(figsize=(15,10))
plt.title("Damage Dealt",fontsize=15)
damageDealt = df['damageDealt']
sns.distplot(damageDealt)
plt.show()


# In[21]:


loc, scale = norm.fit(damageDealt)
# create a normal distribution with loc and scale
n = norm(loc=loc, scale=scale)
kstest(damageDealt, n.cdf)


# ### p-value < 0.05 => not normally distributed

# In[22]:


print('kurtosis of distribution: {}'.format(kurtosis(damageDealt)))
print('skewness of distribution: {}'.format(skew(damageDealt)))

Kurtosis > 3 => leptocurtic
Skewness > 1 => positively skewed
# ### Relations

# In[23]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[24]:


killsPlace = df['killPlace']

plt.figure(figsize=(15,10))
plt.scatter(damageDealt,walkdistance,  c = 'green', edgecolor = 'none', marker = '.')
plt.xlabel('kills place')
plt.ylabel('walk distance')
plt.title('Correlation between kills place and walk distance')
plt.show()


# In[25]:


weaponsAcquired  = df['weaponsAcquired']

plt.figure(figsize=(15,10))
plt.scatter(weaponsAcquired ,kills,  c = 'blue', edgecolor = 'none', marker = '.')
plt.xlabel('weaponsAcquired')
plt.ylabel('kills')
plt.title('Correlation between weapons acquired and kills')
plt.show()


# ### Tests

# In[27]:


killers = df[df['kills']>0]
non_killers = df[df['kills']==0]


# Mann-Whitney U-test (since parameters are not normally distributed) and independent test (as we use different groups during our comparison).
# H0: the mean of variables of killers and non-killers groups are same.
# H1: the mean of variables of killers and non-killers are different.

# In[28]:


stats.mannwhitneyu(killers['walkDistance'].dropna(),non_killers['walkDistance'].dropna())


# p-value < 0.05 => mean of walk distance of killers and non-killers are different

# In[29]:


stats.mannwhitneyu(killers['matchDuration'].dropna(),non_killers['matchDuration'].dropna())


# p-value < 0.05 => mean of match duration of killers and non-killers are different
# 
# 

# ### Cluster Analysis

# In[30]:


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
        
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# In[31]:


columns = df[['kills','walkDistance','swimDistance','weaponsAcquired','assists']]
columns = columns[:10000]
#columns = columns.fillna(0)
columns

from sklearn import preprocessing

x = columns.values #returns a numpy array
scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x)
columns_std = pd.DataFrame(x_scaled)
columns_std = columns_std.fillna(0)
columns_std


# In[32]:


from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
model = model.fit(columns_std)

plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=2)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# In[33]:


model_4 = AgglomerativeClustering(n_clusters=4)
model_4 = model_4.fit_predict(columns_std)
model_4


# In[34]:


from sklearn.decomposition import PCA

pca = PCA(n_components=3)
reduced = pca.fit_transform(columns_std)
columns_pca = pd.DataFrame(reduced)
columns_pca


# In[35]:


columns_pca['label'] = model_4
columns_pca['label'].value_counts()


# In[36]:


u_labels = np.unique(model_4)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15,10))
ax = Axes3D(fig)
#plotting the results:
 
for i in u_labels:
    ax.scatter(columns_pca[columns_pca['label'] == i][0], columns_pca[columns_pca['label'] == i][1], columns_pca[columns_pca['label'] == i][2])
    #plt.scatter(columns_pca[columns_pca['label'] == i][0] , columns_pca[columns_pca['label'] == i][1] , label = i)

plt.legend()
plt.show()


# In[37]:


for i in u_labels:
    print('Group # ', i)
    print(columns[columns_pca['label']==i].describe())
    print('\n')

Group 0: Killers & looters
Group 1: Assistants
Group 2: Swimmers
Group 3: Losers
# ### Regression model

# In[38]:


import xgboost as xgb


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

df.dropna(subset=["winPlacePerc"], inplace=True) # droping rows with missing labels
train10k = df[:100000]
X = train10k.drop(["Id","groupId","matchId","matchType","winPlacePerc"], axis=1)
y = train10k["winPlacePerc"]

col_names = X.columns
transformer = Normalizer().fit(X)
X = transformer.transform(X)


# In[39]:


X = pd.DataFrame(X, columns=col_names)


# In[40]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

D_train = xgb.DMatrix(X_train, label=Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)


# In[41]:


param = {
    'eta': 0.15, 
    'max_depth': 5,  
    'num_class': 2} 

steps = 20  # The number of training iterations
model = xgb.train(param, D_train, steps)


# In[42]:


fig, ax1 = plt.subplots(figsize=(8,15))
xgb.plot_importance(model, ax=ax1)
plt.show()


# In[43]:


from sklearn.metrics import mean_squared_error

preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("MSE = {}".format(mean_squared_error(Y_test, best_preds)))


# In[ ]:





# In[ ]:





# In[ ]:




