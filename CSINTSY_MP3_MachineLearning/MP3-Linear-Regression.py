#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pandas import DataFrame
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
#Import the xls into a dataframe using pandas
file = r'Concrete_Data.xls'
df = pd.read_excel(file)
df.head() #Prints the first few rows 


# In[3]:


df.describe()


# In[4]:


#Let Y contain the output variable compressive strength
#Let X contain the data regarding the features
Y = df['Concrete compressive strength(MPa, megapascals) ']
X = df.drop('Concrete compressive strength(MPa, megapascals) ', axis = 1)
#Perform train test split (80% train / 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=1)


# In[5]:


'''
    Hyperparameter tuning using Grid search CV 
'''
#Set the possible values for lambda
alphas = [i for i in range(1,1500)]
#Store the values of cross_val
stored = []

#Store the first set of data to test (e.g. 0.1)
regressor = Ridge(alpha = 0)
cross_scores = cross_val_score(regressor,X_train,y_train,cv = 10,scoring = 'neg_mean_squared_error')
stored.append(cross_scores.mean())
best_value = cross_scores.mean()
best_alpha = 0

#Loop for the remaining possible values of lambda 
for i in range(1,len(alphas)): 
    #Create a model with the given lambda 
    regressor = Ridge(alpha = alphas[i])
    
    #Store the value of cross-validated score
    cross_scores = cross_val_score(regressor,X_train,y_train,cv = 10,scoring = 'neg_mean_squared_error')
    stored.append(cross_scores.mean())
    
    #Update the values if a smaller value was found 
    if cross_scores.mean() < best_value :
        best_value = cross_scores.mean()
        best_alpha = alphas[i]
print('Best alpha %f Best Value %f'%(best_alpha,best_value))


# In[5]:


plt.plot(alphas, stored)
plt.xlabel('Alpha Values')
plt.ylabel('R^2 Scores')


# In[6]:


#Create Ridge regression model using chosen alpha
regressor = Ridge(alpha = best_alpha)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)


# In[10]:


plt.title("Predicted Y vs. Actual Y")
plt.xlabel("Actual Y", fontsize = 9)
plt.ylabel("Predicted Y", fontsize = 9)
plt.scatter(y_test, y_pred,  color='black')


# In[19]:


coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df


# In[75]:


#Performance metrics of the trained model 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R^2 score:', regressor.score(X_test,y_test))


# In[76]:


#Cross-validation scores
r2 = cross_val_score(regressor,X_train,y_train,cv = 10,scoring = 'r2')
print('Cross-validated R2',r2.mean())
mae = cross_val_score(regressor,X_train,y_train,cv = 10,scoring = 'neg_mean_absolute_error')
print('Cross-validated MAE:',-mae.mean())
mse = cross_val_score(regressor,X_train,y_train,cv = 10,scoring = 'neg_mean_squared_error')
print('Cross-validated MSE:',-mse.mean())
print('Cross-validated RMSE:',np.sqrt(-mse.mean()))


# In[ ]:




