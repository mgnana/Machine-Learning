
# coding: utf-8
About this Dataset
Overview
The data has been split into two groups:

training set (train.csv)
test set (test.csv)
The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

Data Dictionary
VariableDefinitionKey survival Survival 0 = No, 1 = Yes pclass Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd sex Sex Age Age in years sibsp # of siblings / spouses aboard the Titanic parch # of parents / children aboard the Titanic ticket Ticket number fare Passenger fare cabin Cabin number embarked Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.
# In[1]:


#data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import os


# In[2]:


#visulaization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[4]:


#machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[5]:


## Importing the data


# In[88]:


train_df = pd.read_csv('C:/Users/marga/Noteboooks/.ipynb_checkpoints/train_titanic_ML.csv', low_memory = False)


# In[89]:


test_df = pd.read_csv('C:/Users/marga/Noteboooks/.ipynb_checkpoints/test_titanic_ML.csv', low_memory = False)


# In[90]:


combine = [train_df, test_df]


# In[91]:


train_df.describe()


# In[92]:


train_df.head()


# # Analyse by pivoting features

# In[93]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[94]:


train_df[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[95]:


train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[96]:


train_df[['Parch', 'Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# # Visualization of data

# In[97]:


g = sns.FacetGrid(train_df, col = 'Survived')
g.map(plt.hist, 'Age', bins = 20)


# In[98]:


grid = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass', size = 2.2, aspect = 1.6)
grid.map(plt.hist, 'Age', bins = 20, alpha = .5)
grid.add_legend();


# In[99]:


grid = sns.FacetGrid(train_df, row = 'Embarked', col = 'Survived', size = 2.2, aspect = 1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha = .5, ci = None)
grid.add_legend()


# # Preparation of data

# In[100]:


train_df.head()


# In[101]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis = 1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis = 1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


# In[102]:


train_df.head()


# In[103]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand = False)
    
pd.crosstab(train_df['Title'],train_df['Sex'])


# We can replace many titles with a more common name or classify them as Rare.

# In[104]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index = False).mean()


# We can convert the categorical titles to ordinal.

# In[105]:


title_mapping = {"Mr":1, "Miss": 2, "Mrs":3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    
train_df.head()


# In[106]:


train_df['Title'].unique()


# In[107]:


train_df.head()


# In[108]:


test_df.head()


# In[110]:


train_df = train_df.drop(['Name','PassengerId'], axis = 1)
test_df = test_df.drop(['Name'], axis = 1)
combine = [train_df, test_df]
print(train_df.shape)
print(test_df.shape)


# In[111]:


test_df.columns


# In[112]:


train_df.columns


# Let us start by converting Sex feature to a new feature called Gender where female=1 and male=0.

# In[113]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male':0}).astype(int)


# In[114]:


train_df.head()


# Let us start by preparing an empty array to contain guessed Age values based on Pclass X Gender combinations.

# Let us prepare age bands and determine correlations with Survived

# In[115]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[116]:


for dataset in combine:
    dataset.loc[dataset['Age']<=16, 'Age'] = 0
    dataset.loc[(dataset['Age']>16) & (dataset['Age'] <=32), 'Age'] = 1
    dataset.loc[(dataset['Age']>32) & (dataset['Age'] <=48), 'Age'] = 2
    dataset.loc[(dataset['Age']>48) & (dataset['Age'] <=64), 'Age'] = 3
    dataset.loc[dataset['Age']>64, 'Age']  = 4
train_df.head()


# In[85]:


train_df['Age'].unique()


# In[117]:


#Dropping AgeBand feature


# In[118]:


train_df = train_df.drop(['AgeBand'], axis = 1)
combine = [train_df, test_df]
train_df.head()


# 1. Create new feature combining existing features
# We can create a new feature for FamilySize which combines Parch and SibSp. 
# This will enable us to drop Parch and SibSp from our datasets.

# In[119]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# We can create another feature called IsAlone

# In[120]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# Let us drop Parch, SibSp and FamilySize features in favor of IsAlone.

# In[122]:


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)
combine = [train_df, test_df]


# In[123]:


train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[125]:


train_df['Embarked'].unique()


# Here we can see that 'Embarked' column has a missing value, lets fill this missing value with mode

# In[132]:


freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


# In[133]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[134]:


train_df['Embarked'].unique()


# Converting categorical feature to numerical

# In[135]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C': 1, 'Q':2}).astype(int)
    
train_df.head()


# In[136]:


train_df['Embarked'].unique()


# In[137]:


train_df.dtypes


# In[142]:


train_df['Fare'].isnull().unique()


# In[143]:


test_df['Fare'].isnull().unique()


# Replacing the null values in Fare to the median of the fare value

# In[144]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# In[145]:


test_df['Fare'].isnull().unique()


# Now, lets create a FareBand

# In[146]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index = False).mean().sort_values(by= 'FareBand', ascending = True)


# Let's convert the FareBand to ordinal values on the FareBand.

# In[148]:


for dataset in combine:
    dataset.loc[dataset['Fare']<= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare']> 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare']> 14.454) & (dataset['Fare'] <= 31.0), 'Fare'] = 2
    dataset.loc[dataset['Fare']> 31.0, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
train_df = train_df.drop(['FareBand'], axis = 1)
combine = [train_df, test_df]

train_df.head()


# In[149]:


test_df.head()


# In[167]:


train_df.Age.isnull().unique()


# In[171]:


freq_Age = train_df.Age.dropna().mode()[0]
freq_Age


# In[172]:


for dataset in combine:
    dataset['Age'] = dataset['Age'].fillna(freq_Age)
    
train_df[['Age', 'Survived']].groupby(['Age'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[173]:


train_df.Age.isnull().unique()


# In[174]:


test_df.Age.isnull().unique()


# # Model, Predict and Solve

# In[175]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# # Logistic Regression

# In[176]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[184]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df


# In[186]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

#Feature Importance


# # Support Vector Machines

# In[187]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# # k-Nearest Neighbors

# In[188]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# # Gaussian Naive Bayes

# In[190]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# # > ## Perceptron

# In[191]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# # Linear SVC

# In[193]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# # Stochastic Gradient Descent

# In[194]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# # Decision Tree

# In[195]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# # Random Forest

# In[196]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


'C:/Users/marga/Noteboooks/.ipynb_checkpoints/titanicModified.csv'


# In[197]:


my_submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
my_submission.to_csv('C:/Users/marga/Noteboooks/.ipynb_checkpoints/submission.csv', index=False)


# # Model Evaluation

# In[198]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# # Here we can see that Random Forest model is better than the other models with a prediction score of 86.87
