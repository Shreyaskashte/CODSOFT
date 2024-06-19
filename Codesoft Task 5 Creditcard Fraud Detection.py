#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection

# In[1]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Import Credit Card Dataset.
df = pd.read_csv('creditcard.csv')


# In[3]:


df.head()


# # 

# ### Inspecting Dataframe

# In[4]:


print(df.info())


# There is no null values present in dataset.

# In[5]:


df.describe()


# In[6]:


# Class Distribution Analysis in Credit Card Fraud Detection Dataset
df.Class.value_counts()


# In the credit card fraud detection dataset, the class values are highly imbalanced, with a significantly lower number of fraudulent transactions compared to non-fraudulent ones. This imbalance poses a challenge for machine learning models, as they tend to be biased towards the majority class. To effectively detect fraud, we will perform various sampling techniques on the dataset and evaluate the performance of different models. This approach aims to improve the detection accuracy of fraudulent transactions and ensure the robustness of the models in real-world scenarios.
# 
# Sampling Techniques:
# 1)  Random Undersampling
# 2)  Random Oversampling

# # 

# ## Visualisation

# ### Univariate Analysis

# ##### Class

# In[7]:


plt.figure(figsize=(6,4))
sns.countplot(df['Class'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.show()


# ##### Amount

# In[8]:


plt.boxplot(df['Amount'])


# ### Multivatiate Analysis

# In[9]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(df.corr(),annot = True)
plt.show()


# In[10]:


features = df.columns[:-1]  # Exclude the 'Class' column
df[features].hist(figsize=(20, 20), bins=50)
plt.suptitle('Feature Distributions', size=20)
plt.show()


# # 

# ### Preparing Dataset for Logistic Regression

# In[11]:


#Dropping the 'Time' column from your dataframe, it's not relevant for analysis.
df = df.drop(['Time'], axis=1)


# In[12]:


# Train -Test split Dataset.
from sklearn.model_selection import train_test_split


# In[13]:


# Exploratory Data Analysis of Credit Card Transactions (Excluding Class Column)
X = df.drop(['Class'], axis=1)
X.head()


# In[14]:


# Separating Class column for prediction.
y = df['Class']
y.head()


# In[15]:


X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)


# ##### "Feature Scaling of Amount Column Using StandardScaler in Credit Card Fraud Detection"

# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


scaler = StandardScaler()
X_train[['Amount']] = scaler.fit_transform(X_train[['Amount']])
X_train.head()


# ###  Statistical Analysis of Credit Card Fraud Detection using Logistic Regression without any Sampling Technique

# In[18]:


import statsmodels.api as sm


# In[19]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# ###### Drop the columns which have p value > 0.05 or 5%

# In[20]:


X_train = X_train.drop(['V2','V3','V5','V6','V11','V12','V15','V16','V17','V18','V19','V23','V24','V25','V26'],axis=1)


# In[21]:


X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# ### VIF

# VIF stands for Variance Inflation Factor. It's a measure of multicollinearity in a regression analysis, which assesses how much the variance of an estimated regression coefficient increases if your predictors are correlated. High VIF values indicate high multicollinearity.

# In[22]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[23]:


X_train_sm.columns


# In[24]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_sm.columns
vif['VIF'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In this all VIF values are below 5, indicating acceptable levels of multicollinearity.

# # 

# ## Prediction

# Now that we have performed Variance Inflation Factor (VIF) analysis and selected the appropriate features, we will use statsmodels to predict the probabilities of the 'Class' feature.

# ##### Predicted Probability of values on Training Set Using Fitted Model

# In[25]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# ####  Reshaping Predicted Probability Values

# In[26]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# #### Creating Dataframe for merging Class, probabilities and predicted values

# In[27]:


y_train_pred_final = pd.DataFrame({'Class':y_train.values, 'Class_Prob':y_train_pred})
y_train_pred_final.head()


# # 

# #### ROC Curve

# In[28]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[29]:


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Class, y_train_pred_final.Class_Prob, drop_intermediate = False )


# In[30]:


draw_roc(y_train_pred_final.  Class, y_train_pred_final.Class_Prob)


# # 

# ### Evaluating Classification Thresholds for Fraud Detection Using Probability Cutoffs

# In[31]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Class_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[32]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Class, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[33]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy', 'sensi' ,'speci'])
plt.show()


# Choosing approximately Cut-off value from above graph, where accuracy sensitivity and specificity intersect.

# In[34]:


y_train_pred_final['predicted'] = y_train_pred_final.Class_Prob.map(lambda x: 1 if x > 0.1 else 0)

# Let's see the head
y_train_pred_final.head()


# # 

# ### Metrics

# In[35]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# ##### Accuracy

# In[36]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Class, y_train_pred_final.predicted))


# #####  Classification Report

# In[37]:


print(classification_report(y_train_pred_final.Class, y_train_pred_final.predicted)) 


# ##### Confusion Matrix

# In[38]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Class, y_train_pred_final.predicted )
print(confusion)


# In[39]:


# Define class names (assuming Iris dataset classes)
class_names = ['Non-Fraudalent','Fraudalent']

# Create a DataFrame from the confusion matrix
cm_train = pd.DataFrame(confusion, index=class_names, columns=class_names)

# Display the DataFrame
print("Confusion Matrix DataFrame:")
print(cm_train)
# Creating a pivot table
pivot_table = cm_train.reset_index().melt(id_vars='index')
pivot_table.columns = ['Actual', 'Predicted', 'Count']

# Display the pivot table
print("\nPivot Table:")
print(pivot_table)


# # 

# ##### As observed in the above ROC curve, the AUC is 0.98 and the accuracy is 1.0. The classification report also indicates biased values due to the imbalanced nature of the data. Therefore, we will not perform further operations on the test data. Instead, we will implement under-sampling and over-sampling techniques to address the class imbalance and improve the detection of credit card fraud.

# # 

# # Sampling Techniques

# In[40]:


pip install imbalanced-learn


# # 

# # 1. Undersampling

# Random undersampling is a technique used to address class imbalance in datasets. In a class-imbalanced dataset, the number of instances in one class significantly outnumbers the instances in the other class. This can lead to biased model performance, where the model tends to predict the majority class more often. Random undersampling helps balance the class distribution by reducing the number of instances in the majority class.

# ##### Importing and Implementing Random Undersampling

# In[41]:


# Import Undersampling Dictionary
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state = 100)


# In[42]:


# Apply Undersampling to Dataset.
X_resampled,y_resampled = rus.fit_resample(X,y)


# In[43]:


# Imbalance Dataset Class value.
y.value_counts()


# In[44]:


# Undersample Dataset Class value
y_resampled.value_counts()


# In[47]:


plt.figure(figsize=(6,4))
sns.countplot(y_resampled)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.show()


# In[48]:


X_resampled


# In[49]:


X_train_sm.columns


# In[50]:


X_train_sm = X_train_sm.drop(['const'], axis = 1)


# In[51]:


X_resampled = pd.DataFrame(X_resampled, columns=X_train_sm.columns)


# In[52]:


X_resampled


# In[53]:


X_resampled.shape


# ##### Splitting Dataset

# In[54]:


# Splitting Dataset
X_train,X_test,y_train,y_test = train_test_split(X_resampled, y_resampled, train_size=0.8, test_size=0.2, random_state=100)


# In[55]:


X_train.describe()


# ##### Feature Scaling train dataset

# In[56]:


# Feature Scaling Amount column
scaler = StandardScaler()
X_train[['Amount']] = scaler.fit_transform(X_train[['Amount']])
X_train.head()


# In[57]:


# Initializing Logistic Regression Model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[58]:


# Getting the predicted values on the train set
logreg.fit(X_train,y_train)
y_train_pred = logreg.predict(X_train)


# #### Train Model Evaluation

# ##### Train Performance Metrics 

# In[59]:


# Let's check the overall accuracy.
print('Accuracy for undersampled train dataset :',metrics.accuracy_score(y_train, y_train_pred))


# In[60]:


print('Precision Score for undersampled train dataset :',precision_score(y_train, y_train_pred))


# In[61]:


print('Recall Score for undersampled train dataset :',recall_score(y_train, y_train_pred))


# In[62]:


print('F1 Score for undersampled train dataset :',f1_score(y_train, y_train_pred))


# In[63]:


print(classification_report(y_train, y_train_pred)) 


# ##### Confusion Metrics

# In[64]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train, y_train_pred)
print(confusion)


# In[65]:


# Define class names (assuming Iris dataset classes)
class_names = ['Non-Fraudalent','Fraudalent']

# Create a DataFrame from the confusion matrix
cm_train = pd.DataFrame(confusion, index=class_names, columns=class_names)

# Display the DataFrame
print("Confusion Matrix DataFrame:")
print(cm_train)
# Creating a pivot table
pivot_table = cm_train.reset_index().melt(id_vars='index')
pivot_table.columns = ['Actual', 'Predicted', 'Count']

# Display the pivot table
print("\nPivot Table:")
print(pivot_table)


# # 

# ### Test Dataset of Undersampling

# ##### Feature Scaling

# In[66]:


# Performing scaling before running regression.
scaler = StandardScaler()
X_test[['Amount']] = scaler.fit_transform(X_test[['Amount']])
X_test.head()


# #####  Logistic Regression on test set

# In[67]:


# Getting the predicted values on the test set
logreg.fit(X_test,y_test)
y_test_pred = logreg.predict(X_test)


# #### Test Model Evaluation

# ##### Performance Metrics

# In[68]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_test, y_test_pred))


# In[69]:


print('Precision Score for undersampled test dataset :',precision_score(y_test, y_test_pred))


# In[70]:


print('Recall Score for undersampled test dataset :',recall_score(y_test, y_test_pred))


# In[71]:


print('F1 Score for undersampled test dataset :',f1_score(y_test, y_test_pred))


# In[72]:


# Classification rwport for test dataset
print(classification_report(y_test, y_test_pred)) 


# ##### Confusion Metrics

# In[73]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[74]:


# Define class names (assuming Iris dataset classes)
class_names = ['Non-Fraudalent','Fraudalent']

# Create a DataFrame from the confusion matrix
cm_train = pd.DataFrame(confusion, index=class_names, columns=class_names)

# Display the DataFrame
print("Confusion Matrix DataFrame:")
print(cm_train)
# Creating a pivot table
pivot_table = cm_train.reset_index().melt(id_vars='index')
pivot_table.columns = ['Actual', 'Predicted', 'Count']

# Display the pivot table
print("\nPivot Table:")
print(pivot_table)


# #### Plotting ROC Curve of Undersampling

# In[75]:


X_test = sm.add_constant(X_test)
logm2 = sm.GLM(y_test,X_test, family = sm.families.Binomial())
res = logm2.fit()


# In[76]:


# Getting the predicted values on the train set
y_test_pred = res.predict(X_test)
y_test_pred[:10]


# In[77]:


# Reshaping predicting values
y_test_pred = y_test_pred.values.reshape(-1)
y_test_pred[:10]


# In[78]:


# Combining class values and its probabilities in one dataset
y_test_pred_final = pd.DataFrame({'Class':y_test.values, 'Class_Prob':y_test_pred})
y_test_pred_final.head()


# In[79]:


# Creating function for ROC Curve
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[80]:


# Running ROC Curve Metrics
fpr, tpr, thresholds = metrics.roc_curve( y_test_pred_final.Class, y_test_pred_final.Class_Prob, drop_intermediate = False )


# In[81]:


# Draw ROC Curve
draw_roc(y_test_pred_final.Class, y_test_pred_final.Class_Prob)


# # 

# # 2. SMOTE (Synthetic Minority Over-sampling Technique)

# SMOTE (Synthetic Minority Over-sampling Technique) is a popular method used to address the issue of class imbalance in datasets, especially in the context of machine learning classification tasks. The primary goal of SMOTE is to generate synthetic samples for the minority class to balance the class distribution.

# ##### Importing and Implementing Random Oversampling

# In[82]:


# Import RandomOversampler
from imblearn.over_sampling import SMOTE


# In[83]:


# Implementing RandomOversampler
smote = SMOTE(random_state = 100)
X_resampled,y_resampled = smote.fit_resample(X,y)


# In[84]:


# Imbalance Dataset Class value.
y.value_counts()


# In[85]:


# Oversampling Dataset Class Value
y_resampled.value_counts()


# In[86]:


plt.figure(figsize=(6,4))
sns.countplot(y_resampled)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.show()


# In[87]:


X_train_sm.columns
X_resampled = pd.DataFrame(X_resampled, columns=X_train_sm.columns)
X_resampled


# In[88]:


X_resampled.shape


# ##### Splitting dataset

# In[89]:


X_train,X_test,y_train,y_test = train_test_split(X_resampled, y_resampled, train_size=0.8, test_size=0.2, random_state=100)


# ##### Feature Scaling

# In[90]:


scaler = StandardScaler()
X_train[['Amount']] = scaler.fit_transform(X_train[['Amount']])
X_train.head()


# In[91]:


# Getting the predicted values on the train set
logreg.fit(X_train,y_train)
y_train_pred = logreg.predict(X_train)


# #### Metrics Evaluation

# ##### Performance Metrics

# In[92]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train, y_train_pred))


# In[93]:


print('Precision Score for undersampled train dataset :',precision_score(y_train, y_train_pred))


# In[94]:


print('Recall Score for undersampled train dataset :', recall_score(y_train, y_train_pred))


# In[95]:


print('F1 Score for undersampled train dataset :', f1_score(y_train, y_train_pred))


# In[96]:


print(classification_report(y_train, y_train_pred)) 


# #####  Confusion Metrics

# In[97]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train, y_train_pred)
print(confusion)


# In[98]:


# Define class names (assuming Iris dataset classes)
class_names = ['Non-Fraudalent','Fraudalent']

# Create a DataFrame from the confusion matrix
cm_train = pd.DataFrame(confusion, index=class_names, columns=class_names)

# Display the DataFrame
print("Confusion Matrix DataFrame:")
print(cm_train)
# Creating a pivot table
pivot_table = cm_train.reset_index().melt(id_vars='index')
pivot_table.columns = ['Actual', 'Predicted', 'Count']

# Display the pivot table
print("\nPivot Table:")
print(pivot_table)


# ### 

# ### Test Dataset Oversampling

# ##### Feature Scaling

# In[99]:


scaler = StandardScaler()
X_test[['Amount']] = scaler.fit_transform(X_test[['Amount']])
X_test.head()


# In[100]:


# Getting the predicted values on the test set
logreg.fit(X_test,y_test)
y_test_pred = logreg.predict(X_test)


# #### Model Evaluation

# ##### Performance Metrics

# In[101]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_test, y_test_pred))


# In[102]:


print('Precision Score for undersampled test dataset :',precision_score(y_test, y_test_pred))


# In[103]:


print('Recall Score for undersampled test dataset :',recall_score(y_test, y_test_pred))


# In[104]:


print('F1 Score for undersampled test dataset :',f1_score(y_test, y_test_pred))


# In[105]:


# Classification rwport for test dataset
print(classification_report(y_test, y_test_pred)) 


# ##### Confusion Metrics

# In[106]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[107]:


# Define class names (assuming Iris dataset classes)
class_names = ['Non-Fraudalent','Fraudalent']

# Create a DataFrame from the confusion matrix
cm_train = pd.DataFrame(confusion, index=class_names, columns=class_names)

# Display the DataFrame
print("Confusion Matrix DataFrame:")
print(cm_train)
# Creating a pivot table
pivot_table = cm_train.reset_index().melt(id_vars='index')
pivot_table.columns = ['Actual', 'Predicted', 'Count']

# Display the pivot table
print("\nPivot Table:")
print(pivot_table)


# #### Plotting ROC Curve

# In[108]:


X_test = sm.add_constant(X_test)
logm2 = sm.GLM(y_test,X_test, family = sm.families.Binomial())
res = logm2.fit()


# In[109]:


# Getting the predicted values on the test set
y_test_pred = res.predict(X_test)
y_test_pred[:10]


# In[110]:


# Reshaping predicting values
y_test_pred = y_test_pred.values.reshape(-1)
y_test_pred[:10]


# In[111]:


# Creating Dataset of Class and Class probabilities
y_test_pred_final = pd.DataFrame({'Class':y_test.values, 'Class_Prob':y_test_pred})
y_test_pred_final.head()


# In[112]:


# Calculate ROC Curve Metrics
fpr, tpr, thresholds = metrics.roc_curve( y_test_pred_final.Class, y_test_pred_final.Class_Prob, drop_intermediate = False )


# In[113]:


draw_roc(y_test_pred_final.Class, y_test_pred_final.Class_Prob)


# In[ ]:





# In[ ]:




