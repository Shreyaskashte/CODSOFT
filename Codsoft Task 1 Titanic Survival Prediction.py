#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset

# In this project, we will use the Titanic dataset to build a machine learning model that predicts whether a passenger survived the disaster. This dataset contains detailed information about the passengers, including their age, gender, ticket class, fare, and cabin. Our goal is to develop a model that can accurately predict survival based on these features.

# In[ ]:


# Importing libraries & Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the Titanic Dataset

# In[ ]:


# Import Titanic Dataset
df = pd.read_csv('Titanic-Dataset.csv')


# In[ ]:


df.head()


# # 

# # Inspecting Dataframe

# In[ ]:


# Let's check the dimensions of the dataframe
df.shape


# In[ ]:


# let's look at the statistical aspects of the dataframe
df.describe()


# In[ ]:


# Let's see the type of each column
df.info()


# In[ ]:


# Checking of dataset balance.
df.Survived.value_counts()


# # 

# ## Handling Missing Values

# In[ ]:


#print the information of variables to check their null values percentage in data .
(df.isnull().sum()/len(df))*100


# ##### Cabin Column

# The 'Cabin' column has 77% missing values. This high percentage suggests that imputation might not be practical or reliable. Instead, we will drop this column to avoid introducing significant noise or bias into our model.

# In[ ]:


df = df.drop(['Cabin'],axis=1)


# ##### Embarked Column

# The 'Embarked' column has a few missing values. We will fill these with the mode, as it is the most common embarkation point and likely represents the most plausible value.

# In[ ]:


df['Embarked'].isnull().sum()


# In[ ]:


df['Embarked'].mode()


# In[ ]:


df['Embarked'].fillna((df.Embarked.mode()[0]), inplace=True)


# ##### Age Column

# In[ ]:


# Distribution of age
df['Age'].plot(kind='hist', bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# We will fill 'Age' column with mean value becaue it is numerical type column.

# In[ ]:


df['Age'].isnull().sum()


# In[ ]:


mean_value = df.Age.mean()


# In[ ]:


mean_value


# In[ ]:


df['Age'].fillna(mean_value, inplace=True)


# In[ ]:


(df.isnull().sum()/len(df))*100


# # 

# # Visualising Data

# ### Univariate Analysis

# In[ ]:


df['Fare'].plot(kind='hist', bins=20)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()


# ### Bivariate Analysis

# In[ ]:


sns.countplot(df.Survived)
plt.show()


# In[ ]:


sns.countplot(df.Pclass,hue = df.Survived)


# In[ ]:


sns.countplot(df.Sex,hue = df.Survived)
plt.show()


# In[ ]:


sns.countplot(df.Survived,hue = df.Embarked)
plt.show()


# In[ ]:


plt.figure(figsize = (5,6)) 
sns.countplot(df.Parch,hue = df.Survived)
plt.show()


# In[ ]:


df.groupby('Survived')['Age'].plot(kind='kde')
plt.show()


# ##### 

# ## Feature Engineering

# In[ ]:


# Dropping unneccesary columns for model building
df = df.drop(['PassengerId','Ticket','Name'],axis=1)


# In[ ]:


df.info()


# #### Data Type Conversion

# The DataFrame df has been prepared for machine learning by converting Pclass, Embarked, SibSp, and Parch to categorical data types, facilitating efficient handling of their categories and optimizing memory usage. 
# Additionally, Age has been converted to integers for clear and consistent representation of passengers' ages. These conversions align with machine learning algorithm requirements, ensuring a solid foundation for model training and further analysis. Adjustments can be made based on specific modeling needs and objectives.

# In[ ]:


# Convert the data type in proper data type
df['Pclass']= df['Pclass'].astype('category')
df['Embarked']= df['Embarked'].astype('category')
df['Age']= df['Age'].astype('int')


# In[ ]:


# Check the changed data types
df.info()


# #### Creating Dummy Variables

# In[ ]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(df[['Sex','Embarked','Pclass']], drop_first=True)

# Adding the results to the master dataframe
df1 = pd.concat([df, dummy1], axis=1)


# In[ ]:


df1.shape


# In[ ]:


df1.head()


# In[ ]:


# Drop the dummies original columns
df1 = df1.drop([ 'Sex', 'Embarked','Pclass'],axis=1)


# In[ ]:


df1.shape


# In[ ]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(df1.corr(),annot = True)
plt.show()


# # 

# ## Split the Data into Training and Testing Sets

# Performing a train-test split is a fundamental step in the process of building and evaluating machine learning models. 

# In[ ]:


# Import split dictionary
from sklearn.model_selection import train_test_split


# In[ ]:


# Drop the target variable 
X = df1.drop(['Survived'], axis=1)
X.head()


# In[ ]:


y = df1['Survived']
y.head()


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)


# # 

# ## Feature Scaling

# Feature scaling is a crucial step in the preprocessing of data for many machine learning algorithms. It involves transforming the features to be on a similar scale, which can improve the performance and convergence speed of the algorithms.
# We will scale 'Age' and 'Fare' column using StandardScaler.

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()
X_train[['Age','Fare']] = scaler.fit_transform(X_train[['Age','Fare']])
X_train.head()


# # 

# ## Logistic Regression Statsmodel

# statsmodels is a comprehensive Python library used for statistical modeling and hypothesis testing. It provides classes and functions to estimate a wide variety of statistical models, conduct statistical tests, and perform data exploration and visualization. The library is particularly useful for conducting detailed statistical analyses and is widely used in academia, research, and industry for econometric and statistical applications.

# In[ ]:


import statsmodels.api as sm


# #### Running Your First Training Model 

# In[ ]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# ##### Assessing the model with StatsModels
# 

# Dropping the features which have p value greater than 0.05 or 5%.

# In[ ]:


X_train_sm = X_train.drop(['Embarked_Q','Embarked_S'],axis=1)


# In[ ]:


X_train_sm = sm.add_constant(X_train_sm)
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


X_train_sm.columns


# In[ ]:


res.params


# In[ ]:


coefficients = res.params
coefficients_df = pd.DataFrame(coefficients, columns=['Coefficient'])
coefficients_df = coefficients_df.drop(['const'])
coefficients_df


# In[ ]:


plt.figure(figsize=(10, 6))
sns.barplot(x=coefficients_df.index, y=coefficients_df['Coefficient'])
plt.title('Feature Importance in Logistic Regression Model')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45)
plt.show()


# #### 

# # VIF

# VIF stands for Variance Inflation Factor. It's a measure of multicollinearity in a regression analysis, which assesses how much the variance of an estimated regression coefficient increases if your predictors are correlated. High VIF values indicate high multicollinearity.

# In[ ]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


X_train_sm.columns


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_sm.columns
vif['VIF'] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In this all VIF values are below 5, indicating acceptable levels of multicollinearity.

# #### Predicted Values on Training Set Using Fitted Model

# In[ ]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# #### Reshaping Predicted Values 

# In[ ]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# #### Creating a dataframe with the actual churn flag and the predicted probabilities on Train Dataset

# In[ ]:


y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'Survive_Prob':y_train_pred})
y_train_pred_final['PassangerId'] = y_train.index
y_train_pred_final.head()


# ####  Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

# In[ ]:


y_train_pred_final['predicted'] = y_train_pred_final.Survive_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# #####  Accuracy Score

# In[ ]:


from sklearn import metrics
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted))


# # 

# ### An ROC curve demonstrates several things:
# 
# - It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
# - The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
# - The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

# In[ ]:


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


# In[ ]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Survived, y_train_pred_final.Survive_Prob, drop_intermediate = False )


# In[ ]:


draw_roc(y_train_pred_final.Survived, y_train_pred_final.Survive_Prob)


# # 

# ## Improve Model

# #### Creating Columns with Different Probability Cutoffs

# In[ ]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Survive_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[ ]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# ###### Plotting Accuracy, Sensitivity, and Specificity for Various Probabilities

# In[ ]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[ ]:





# #### Final Prediction Based on Probability Cutoff

# In[ ]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Survive_Prob.map( lambda x: 1 if x > 0.4 else 0)

y_train_pred_final.head()


# # 

# ## Evaluation Metrics

# ##### Accuracy

# In[ ]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)


# ##### Classification report

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


report = classification_report(y_train_pred_final.Survived, y_train_pred_final.final_predicted)
print(report)


# #### Confusion Matrix on Training Predictions

# In[ ]:


# Confusion matrix 
from sklearn import metrics
confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )
print(confusion)


# In[ ]:


# Define class names (assuming Iris dataset classes)
class_names = ['not_survived', 'Survived']

# Create a DataFrame from the confusion matrix
cm_train = pd.DataFrame(confusion, index=class_names, columns=class_names)

# Display the DataFrame
print("Confusion Matrix DataFrame:")
print(cm_train)


# In[ ]:


# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# #### Metrics beyond simply accuracy

# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# #### Sensitivity, 
# It also known as True Positive Rate or Recall, measures the proportion of actual positive cases that are correctly identified by a classification model. It answers the question: "Of all the actual positive cases, how many did the model correctly identify?"

# In[ ]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# #### Specificity 
# It measures the proportion of actual negative cases that are correctly identified by a classification model. It answers the question: "Of all the actual negative cases, how many did the model correctly identify?"

# In[ ]:


# Let us calculate specificity
TN / float(TN+FP)


# #### False Positive Rate (FPR)
# It also known as the fall-out, is the proportion of actual negative cases that are incorrectly classified as positive by a classification model.

# In[ ]:


# Calculate false postive rate 
print(FP/ float(TN+FP))


# #### Positive Predictive Value (PPV), 
# It also known as precision, measures the proportion of positive predictions made by a classification model that are actually correct. It answers the question: "Of all the cases predicted as positive by the model, how many are actually positive?"

# In[ ]:


# positive predictive value 
print (TP / float(TP+FP))


# #### Negative Predictive Value (NPV) 
# It measures the proportion of negative predictions made by a classification model that are actually correct. It answers the question: "Of all the cases predicted as negative by the model, how many are actually negative?"

# In[ ]:


# Negative predictive value
print (TN / float(TN+ FN))


# # 

# ### Plotting Precision & Recall Curve

# A precision-recall curve is a graphical representation of the trade-off between precision and recall for different threshold values in a binary classification model.To plot a precision-recall curve, you need the true labels of the data and the predicted probabilities from the model.

# In[ ]:


from sklearn.metrics import precision_score, recall_score


# In[ ]:


precision_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)


# In[ ]:


recall_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)


# In[ ]:


from sklearn.metrics import precision_recall_curve


# In[ ]:


y_train_pred_final.Survived, y_train_pred_final.predicted


# In[ ]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Survived, y_train_pred_final.Survive_Prob)


# In[ ]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# # 

# # 

# # Test Dataset

# ### Scaling Age and Fare Features in Test Dataset

# In[ ]:


X_test[['Age','Fare']] = scaler.transform(X_test[['Age','Fare']])


# ##### Using the Same Features as Train Dataset in test dataset 

# In[ ]:


col = X_train.columns


# In[ ]:


col


# In[ ]:


X_train_sm.columns


# In[ ]:


col = col.drop(['Embarked_Q','Embarked_S'])
col


# In[ ]:


X_test = X_test[col]
X_test.head()


# ### Model Building

# In[ ]:


X_test_sm = sm.add_constant(X_test)


# In[ ]:


y_test_pred = res.predict(X_test_sm)


# In[ ]:


y_test_pred[:10]


# In[ ]:


# Converting y_pred to a dataframe 
y_pred_1 = pd.DataFrame(y_test_pred)


# In[ ]:


# Let's see the head
y_pred_1.head()


# In[ ]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[ ]:


# Putting PassangerId to index
y_test_df['PassangerId'] = y_test_df.index


# In[ ]:


# Reset index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[ ]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[ ]:


y_pred_final.head()


# In[ ]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Survive_Prob'})


# In[ ]:


y_pred_final.head()


# In[ ]:


y_pred_final['final_predicted'] = y_pred_final.Survive_Prob.map(lambda x: 1 if x > 0.5 else 0)


# In[ ]:


y_pred_final


# # 

# ## Evaluation metrics of test datset

# ##### Accuracy

# In[ ]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Survived, y_pred_final.final_predicted)


# ##### Precision

# In[ ]:


precision_score(y_pred_final.Survived, y_pred_final.final_predicted)


# #####  Recall

# In[ ]:


recall_score(y_pred_final.Survived, y_pred_final.final_predicted)


# #####  Classification Report

# In[ ]:


report = classification_report(y_pred_final.Survived, y_pred_final.final_predicted )
print(report)


# ##### 

# ### Confusion Matrix of Test Dataset

# In[ ]:


test_confusion = metrics.confusion_matrix(y_pred_final.Survived, y_pred_final.final_predicted )
test_confusion


# In[ ]:


# Define class names (assuming Iris dataset classes)
class_names = ['not_survived', 'Survived']

# Create a DataFrame from the confusion matrix
cm_test = pd.DataFrame(test_confusion, index=class_names, columns=class_names)

# Display the DataFrame
print("Confusion Matrix DataFrame:")
print(cm_test)


# In[ ]:


# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# The conclusion drawn from the analysis of the Titanic dataset depends on various aspects of the exploratory data analysis (EDA) and model evaluation. Here are some general conclusions that might be drawn from a typical analysis:
# 
# ###  **Understanding the Dataset**
# - **Survival Rate:** Approximately 38% of the passengers survived, while 62% did not survive.
# - **Gender:** Women had a significantly higher survival rate compared to men.
# - **Passenger Class:** Passengers in the first class had a higher survival rate compared to those in the second and third classes.
# - **Age:** Younger passengers had a higher chance of survival. Children had a higher survival rate compared to adults.
# - **Fare:** Passengers who paid higher fares (often first-class passengers) had a higher probability of survival.
# 
# ### Summary
# 
# The analysis of the Titanic dataset reveals that survival was significantly influenced by gender, passenger class, and age. Women, first-class passengers, and younger individuals had higher chances of survival. The logistic regression model provided insights into the predictive power of these features, and precision-recall analysis highlighted the importance of choosing appropriate thresholds for different contexts.
# 
# This comprehensive analysis not only sheds light on historical events but also offers valuable lessons for future maritime safety and emergency response strategies.

# In[ ]:





# In[ ]:




