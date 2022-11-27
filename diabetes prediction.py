#!/usr/bin/env python
# coding: utf-8

# # PROJECT : DIABETES PREDICTION

# # Importing The Dependencies

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# # Data Collection and Analysis

# # PIMA Diabetes Dataset

# In[5]:


# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')


# In[4]:


pd.read_csv


# In[5]:


# printing the first 5 rows of the dataset
diabetes_dataset.head()


# In[6]:


# printing the last 5 tail of the dataset
diabetes_dataset.tail()


# In[7]:


# number of rows and columns in this dataset
diabetes_dataset.shape


# In[8]:


# getting the statistical measures of the data
diabetes_dataset.describe()


# In[6]:


# getting the info og the data
diabetes_dataset.info()


# In[7]:


diabetes_dataset['Outcome'].value_counts()


# 0 --> Non-Diabetic
# 1 --> Diabetic

# In[11]:


diabetes_dataset.groupby('Outcome').mean()


# In[12]:


# seprating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome',axis=1)


# In[13]:


print(X)


# In[14]:


Y = diabetes_dataset['Outcome']


# In[15]:


print(Y)


# # Data Standardization

# In[16]:


scaler = StandardScaler()


# In[17]:


scaler.fit(X)


# In[18]:


standardized_data = scaler.transform(X)


# In[19]:


print(standardized_data)


# In[20]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[21]:


print(X)


# In[22]:


print(Y)


# # Train Test Split 

# In[23]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[24]:


print(X.shape)


# In[25]:


print(X_train)


# In[26]:


print(X_test)


# In[27]:


print(X_train.shape)


# In[28]:


print(X_test.shape)


# In[29]:


print(Y_train)


# In[30]:


print(Y_test)


# # Training the Model 

# In[31]:


classifier = svm.SVC(kernel='linear')


# In[32]:


#training the support vector machine classifier
classifier.fit(X_train, Y_train)


# # Model Evaluation

# # Accuracy Score

# In[33]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[34]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[35]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[36]:


print('Accuracy score of the test data : ', test_data_accuracy)


# # Making a Predictive System

# In[37]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)


# In[38]:


# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[39]:


# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)


# In[40]:


prediction = classifier.predict(std_data)
print(prediction)


# In[41]:


if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')


# In[ ]:





# In[ ]:




