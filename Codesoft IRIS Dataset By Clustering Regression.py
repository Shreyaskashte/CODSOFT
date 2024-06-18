#!/usr/bin/env python
# coding: utf-8

# # IRIS FLOWER CLASSIFICATION

# The Iris flower dataset consists of three species: setosa, versicolor,
# and virginica. These species can be distinguished based on their
# measurements. Now, we have the measurements
# of Iris flowers categorized by their respective species. Our
# objective is to train a machine learning model that can learn from
# these measurements and accurately classify the Iris flowers into
# their respective species.

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Importing Dataset IRIS dataset
df= pd.read_csv('IRIS.csv')


# In[3]:


df.head()


# # 

# ## Inspecting Dataframe

# ##### Balance of Dataframe

# In[4]:


df.species.value_counts()


# ##### Check for null values

# In[5]:


df.isnull().sum()


# ##### Summary Statistics

# In[6]:


df.describe()


# Since our dataset is both balanced and free of null values, we are in an excellent position to proceed with model building. The absence of missing values ensures that we won't need to impute data, and a balanced dataset allows us to focus on developing accurate and fair predictive models without worrying about class imbalance issues. Let's move forward to the next steps of our machine learning project.

# # 

# ## Feature Engineering

# ##### Encoding Categorical Variables

# In[7]:


# Change species text column in number column by using label-encoder.
from sklearn.preprocessing import LabelEncoder


# In[8]:


encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])


# Dataset after applying label encoder

# In[9]:


df.head()


# In[10]:


df.species.value_counts()


# # 

# # Data Visualisation

# ### Univariate Analysis

# In[11]:


plt.figure(figsize=[10,5])
sns.countplot(df['species'])


# In[12]:


# Plotting Subplots for Iris Dataset Features

plt.figure(figsize=[20,10])
plt.subplot(2,2,1)
df.petal_width.plot()
plt.xlabel('petal_width')
plt.subplot(2,2,2)
df.petal_length.plot()
plt.xlabel('petal_length')
plt.subplot(2,2,3)
df.sepal_length.plot()
plt.xlabel('sepal_length')
plt.subplot(2,2,4)
df.sepal_width.plot()
plt.xlabel('sepal_width')


# ##### Boxplots

# In[13]:


# Plotting boxplots for Iris Dataset Features

plt.figure(figsize=[20,10])
plt.subplot(2,2,1)
sns.boxplot(df.petal_width)
plt.xlabel('petal_width')
plt.subplot(2,2,2)
sns.boxplot(df.petal_length)
plt.xlabel('petal_length')
plt.subplot(2,2,3)
sns.boxplot(df.sepal_length)
plt.xlabel('sepal_length')
plt.subplot(2,2,4)
sns.boxplot(df.sepal_width)
plt.xlabel('sepal_width')


# #### Multivariate Analysis
# ##### Correlation Heatmap

# In[14]:


df1 = df.corr()


# In[15]:


df1


# In[16]:


plt.figure(figsize=[15,7])
sns.heatmap(df1,annot=True)


# ##### 

# # 

# #   Model Training and Evaluation

# #####  Split the Data

# In[17]:


# Importing Split Dictionary
from sklearn.model_selection import train_test_split


# In[18]:


# Putting feature variable to X
X = df.drop(['species'], axis=1)
X.head()


# In[19]:


# Putting response variable to y
y = df['species']
y.head()


# In[20]:


# Splitting the data into train and test
X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.65, test_size=0.35, random_state=100)


# In[21]:


X_train.shape


# In[22]:


X_train.head()


# In[23]:


y_train.head()


# # 

# # Modelling Dataset with k-Nearest Neighbors (k-NN)

# ##### Fit Models

# In[24]:


from sklearn.neighbors import KNeighborsClassifier


# In[25]:


knn = KNeighborsClassifier(n_neighbors=3)  # You can choose the number of neighbors
knn.fit(X_train, y_train)
# Step 3: Make predictions
y_pred = knn.predict(X_train)


# In[26]:


y_pred


# # 

# ## Model Evaluation

# ##### Performance Metrics

# In[27]:


from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[28]:


accuracy = accuracy_score(y_train, y_pred)
print('Accuracy of Train dataset :', accuracy)


# In[29]:


Train_class_report = classification_report(y_train, y_pred)
print(Train_class_report)


# ##### Confusion Matrix

# In[30]:


train_confusion = confusion_matrix(y_train, y_pred)
train_confusion


# In[31]:


# Define class names (assuming Iris dataset classes)
class_names = ['setosa', 'versicolor', 'virginica']

# Create a DataFrame from the confusion matrix
cm_train = pd.DataFrame(train_confusion, index=class_names, columns=class_names)

# Display the DataFrame
print("Confusion Matrix DataFrame:")
print(cm_train)

# Creating a pivot table
pivot_table = cm_train.reset_index().melt(id_vars='index')
pivot_table.columns = ['Actual', 'Predicted', 'Count']

# Display the pivot table
print("\nPivot Table:")
print(pivot_table)


# In[32]:


# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ##### 

# ##### 

# # Test Dataset

# In[33]:


X_test.shape


# ##### Implementing k-Nearest Neighbors (KNN) Algorithm

# In[34]:


# You can choose the number of neighbors
knn.fit(X_test, y_test)


# ##### Training Test Dataset

# In[35]:


# Step 3: Make predictions
y_pred = knn.predict(X_test)


# ##### 

# ### Model Evaluation

# #####  Performance Metrics

# In[36]:


test_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Test dataset :', test_accuracy)


# In[37]:


test_class_report = classification_report(y_test, y_pred)
print(test_class_report)


# ##### Confusion Matrix

# In[38]:


test_confusion = confusion_matrix(y_test, y_pred)
test_confusion


# In[39]:


# Define class names (assuming Iris dataset classes)
class_names = ['setosa', 'versicolor', 'virginica']

# Create a DataFrame from the confusion matrix
cm_test = pd.DataFrame(test_confusion, index=class_names, columns=class_names)

# Display the DataFrame
print("Confusion Matrix DataFrame:")
print(cm_test)

# Creating a pivot table
pivot_table = cm_test.reset_index().melt(id_vars='index')
pivot_table.columns = ['Actual', 'Predicted', 'Count']

# Display the pivot table
print("\nPivot Table:")
print(pivot_table)


# In[40]:


# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ##### 

# Conclusion
# The K-NN clustering algorithm successfully grouped the Iris dataset into three clusters, which largely corresponded to the actual species of the Iris flowers. This demonstrates the algorithm's ability to find natural groupings in the data based on the feature similarities. The accuracy and confusion matrix provided insights into the clustering performance, while visualizations helped in understanding the distribution and separation of the clusters.

# In[ ]:




