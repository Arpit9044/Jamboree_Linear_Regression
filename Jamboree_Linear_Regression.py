#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv(r"C:\Users\hp\Downloads\Jamboree_Admission.csv")
df


# In[3]:


df.shape


# - There are 500 samples and 8 features along with one target variable which is continuous in nature.

# In[4]:


print(df.columns.tolist())


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df[df.duplicated()]


# - There are no duplicates present in the data.

# In[8]:


df.isna().sum()


# - There are no null values in the data as well.

# # Univariate Analysis

# In[9]:


sns.histplot(df['GRE Score'])


# In[10]:


sns.histplot(df['TOEFL Score'])


# In[11]:


sns.countplot(df['SOP'])


# In[12]:


sns.countplot(df['LOR '])


# In[13]:


sns.histplot(df['CGPA'])


# In[14]:


sns.countplot(df['Research'])


# In[15]:


sns.countplot(df['University Rating'])


# # Bi-Variate Analysis

# In[16]:


plt.figure(figsize=(25,5))

plt.subplot(1,2,1)
sns.lineplot(df['GRE Score'],df['Chance of Admit '])

plt.subplot(1,2,2)
sns.scatterplot(df['GRE Score'],df['Chance of Admit '])

plt.show()


# We can se a trend here.More the GRE Score ,higher the chances of admission.They are positively correlated.

# In[17]:


plt.figure(figsize=(25,5))

plt.subplot(1,2,1)
sns.lineplot(df['TOEFL Score'],df['Chance of Admit '])

plt.subplot(1,2,2)
sns.scatterplot(df['TOEFL Score'],df['Chance of Admit '])

plt.show()


# We can se a trend here.More the TOEFL Score ,higher the chances of admission.They are positively correlated.

# In[18]:


plt.figure(figsize=(25,8))

plt.subplot(1,2,1)
sns.barplot(df['University Rating'],df['Chance of Admit '])

plt.subplot(1,2,2)
sns.boxplot(df['University Rating'],df['Chance of Admit '])

plt.show()


# - We can see a positive correlation here.Better the university,better is the chance of admission.However,there are some outliers which suggest even if the university ranking is good,it does not mean that admission is bound to happen.

# In[19]:


plt.figure(figsize=(25,8))

plt.subplot(1,2,1)
sns.barplot(df['SOP'],df['Chance of Admit '])

plt.subplot(1,2,2)
sns.boxplot(df['SOP'],df['Chance of Admit '])

plt.show()


# - We can see a positive correlation here.Better the SOP,better is the chance of admission.However,there are few outliers which suggest even if the SOP is good,it does not mean that admission is bound to happen.

# In[20]:


plt.figure(figsize=(25,8))

plt.subplot(1,2,1)
sns.barplot(df['LOR '],df['Chance of Admit '])

plt.subplot(1,2,2)
sns.boxplot(df['LOR '],df['Chance of Admit '])

plt.show()


# - We can see a positive correlation here.Better the LOR,better is the chance of admission.However,there are very few outliers which suggest even if the LOR is good,it does not mean that admission is bound to happen.

# In[21]:


plt.figure(figsize=(25,5))

plt.subplot(1,2,1)
sns.lineplot(df['CGPA'],df['Chance of Admit '])

plt.subplot(1,2,2)
sns.scatterplot(df['CGPA'],df['Chance of Admit '])

plt.show()


# We can se a trend here.More the CGPA ,higher the chances of admission.They are positively correlated.

# In[22]:


plt.figure(figsize=(25,8))

plt.subplot(1,2,1)
sns.barplot(df['Research'],df['Chance of Admit '])

plt.subplot(1,2,2)
sns.boxplot(df['Research'],df['Chance of Admit '])

plt.show()


# - Getting involved in Research works improves the chances of admission.

# In[23]:


plt.figure(figsize=(25,5))

plt.subplot(1,2,1)
sns.lineplot(df['GRE Score'],df['TOEFL Score'])

plt.subplot(1,2,2)
sns.scatterplot(df['GRE Score'],df['TOEFL Score'])

plt.show()


# - If GRE Score is high,so is the TOEFL Score.There is a positive correlation between them.

# In[24]:


df


# # Multivariate Analysis

# In[25]:


sns.scatterplot(data=df,x='GRE Score',y='TOEFL Score',size='Chance of Admit ')
plt.show()


# - If a student is having high scores in both TOEFL and GRE, chances of admission are higher.

# In[26]:


sns.scatterplot(data=df,x='GRE Score',y='TOEFL Score',size='CGPA')
plt.show()


# - Student who have score high CGPA in college end up getting more marks in GRE and TOEFL.

# In[27]:


sns.scatterplot(data=df,x='GRE Score',y='TOEFL Score',hue='Research')
plt.show()


# - STudent scoring high marks in TOEFL and GRE also involve in Research work to increase their probability of getting admitted.

# In[28]:


sns.scatterplot(data=df,x='GRE Score',y='TOEFL Score',hue='University Rating')
plt.show()


# - Students from high rated universities have high GRE and TOEFL Score.

# In[29]:


sns.boxplot(data=df,x='University Rating',y='Chance of Admit ',hue='Research')


# - We can see the effect of Uiversity Rating and Research with Chance of Admit.

# # Outliers Detection and Treatment

# In[30]:


plt.figure(figsize=(25,25))

plt.subplot(4,2,1)
sns.boxplot(y=df['GRE Score'])
plt.ylabel('GRE Score',size=20)

plt.subplot(4,2,2)
sns.boxplot(y=df['TOEFL Score'])
plt.ylabel('TOEFL Score',size=20)

plt.subplot(4,2,3)
sns.boxplot(y=df['University Rating'])
plt.ylabel('University Rating',size=20)

plt.subplot(4,2,4)
sns.boxplot(y=df['SOP'])
plt.ylabel('SOP',size=20)

plt.subplot(4,2,5)
sns.boxplot(y=df['CGPA'])
plt.ylabel('CGPA',size=20)

plt.subplot(4,2,6)
sns.boxplot(y=df['LOR '])
plt.ylabel('LOR ',size=20)

plt.subplot(4,2,7)
sns.boxplot(y=df['Chance of Admit '])
plt.ylabel('Chance of Admit ',size=20)

plt.show()


# - As there are no outliers present in the data, there is not need for outliers treatment.

# In[31]:


#As serial no is just an index,we will remove that column.
df.drop('Serial No.',axis=1,inplace=True)


# In[32]:


df.corr()


# In[33]:


sns.heatmap(df.corr(),annot=True)


# # Model Building

# In[34]:


from sklearn.model_selection import train_test_split
X = df.drop(['Chance of Admit '], axis=1)
y = df['Chance of Admit ']


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=True)


# In[36]:


#Standardization
from sklearn.preprocessing import StandardScaler
X_train_columns=X_train.columns
std=StandardScaler()
X_train_std=std.fit_transform(X_train)


# In[37]:


X_train=pd.DataFrame(X_train_std, columns=X_train_columns)


# In[38]:


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,Ridge,LinearRegression
from sklearn.metrics import mean_squared_error
models = [
 ['Linear Regression :', LinearRegression()],

 ['Lasso Regression :', Lasso(alpha=0.1)], #try with different alpha values
 ['Ridge Regression :', Ridge(alpha=1.0)] #try with different alpha values
 ]
print("Results without removing features with multicollinearity ...")
for name,model in models:
 model.fit(X_train, y_train.values)
 predictions = model.predict(std.transform(X_test))
 print(name, (np.sqrt(mean_squared_error(y_test, predictions))))


# **Linear Regression using Statsmodel library**
# 
# - Adjusted. R-squared reflects the fit of the model. R-squared values range from 0 to 1, where a higher value
# generally indicates a better fit, assuming certain conditions are met.
# - const coefficient is your Y-intercept. It means that if both the Interest_Rate and Unemployment_Rate
# coefficients are zero, then the expected output (i.e., the Y) would be equal to the const coefficient.
# - Interest_Rate coefficient represents the change in the output Y due to a change of one unit in the interest rate
# (everything else held constant)
# - Unemployment_Rate coefficient represents the change in the output Y due to a change of one unit in the
# unemployment rate (everything else held constant)
# - std err reflects the level of accuracy of the coefficients. The lower it is, the higher is the level of accuracy
# - P >|t| is your p-value. A p-value of less than 0.05 is considered to be statistically significant
# Confidence Interval represents the range in which our coefficients are likely to fall (with a likelihood of 95%)

# In[39]:


import statsmodels.api as sm
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train.values, X_train).fit()
print(model.summary())


# As p_values for SOP is less than 0.05, this ,means that the coefficient associated with that feature is not significant.Hence , we will remove the columns and rebuild the model.

# In[40]:


X_train_new=X_train.drop(columns='SOP')


# In[41]:


model1 = sm.OLS(y_train.values, X_train_new).fit()
print(model1.summary())


# In[42]:


X_train_new=X_train_new.drop(columns='University Rating')


# In[43]:


model2 = sm.OLS(y_train.values, X_train_new).fit()
print(model2.summary())


# After removing SOP and University rating we have all the p_values below 0.05.

# **VIF(Variance Inflation Factor)**
# 
# - “ VIF score of an independent variable represents how well the variable is explained by other independent
# variables.
# - So, the closer the R^2 value to 1, the higher the value of VIF and the higher the multicollinearity with the
# particular independent variable.
# 

# In[44]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=pd.DataFrame()
vif['Features']=X_train_new.columns
vif['VIF']=[variance_inflation_factor(X_train_new.values,i) for i in range(X_train_new.shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif


# - Now, we can see that the values of VIFs are less than 5.

# In[45]:


X_test_std= std.transform(X_test)
X_test=pd.DataFrame(X_test_std, columns=X_train_columns) # col name same as train data
X_test = sm.add_constant(X_test)
X_test_del=list(set(X_test.columns).difference(set(X_train_new.columns)))
print(f'Dropping {X_test_del} from test set')
X_test_new=X_test.drop(columns=X_test_del)


# In[49]:


#Prediction from the clean model
pred = model2.predict(X_test_new)
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
print('Mean Absolute Error ', mean_absolute_error(y_test.values,pred) )
print('Root Mean Square Error ', np.sqrt(mean_squared_error(y_test.values,pred) ))


# # Assumptions of Linear Regression

# **Mean of Residuals**

# In[51]:


residuals = y_test.values-pred
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))


# - Mean of residuals is almost zero which means that the assumption is satisfied.

# - **Test for Homoscedasticity**

# In[54]:


p = sns.scatterplot(x=pred,y=residuals)
plt.xlabel('predicted values')
plt.ylabel('Residuals')
plt.ylim(-0.4,0.4)
plt.xlim(0,1)
p = sns.lineplot([0,26],[0,0],color='blue')
p = plt.title("Residuals vs fitted values plot for homoscedasticity check")


# In[55]:


import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(residuals, X_test)
lzip(name, test)


# **Here null hypothesis is - error terms are homoscedastic and since p-values >0.05, we fail to reject the
# null hypothesis**
# 
# - This means that the errors do have constant variance.

# **Normality of residuals**

# In[57]:


sns.distplot(residuals,kde=True)
plt.title('Normality of error terms/residuals')
plt.show()


# In[58]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test.values, pred)
fig.suptitle('y_test vs y_pred', fontsize=20) # Plot heading
plt.xlabel('y_test', fontsize=18) # X-label
plt.ylabel('y_pred', fontsize=16) # Y-label


# From the above graph following interpretations can be made:
# - If the chances of admission is low, our model is not able to perform that well i.e. there is a difference between actual and predicted value.
# - With chances of admission increasing and becoming high, the performance of the model gets better.
# 
# Why model's performance is low when chance of admission is low:
# - Less Training Data: To solve this, increase training data for low chances of admission so that our model can learn the patterns more effectively.
# - Less Complex Data: There are multiple ways to solve this
#  - Derive some features using feature engineering
#  - Build a polynomial regression model
#  - Use more features

# **Recommendations** :
# 
# - Involving in research work is recommended for the Students which in turn improves the chance of admission.
# - High TOEFL and GRE Score will also increase the chances of admission.
# - The company should focus on students who have low chances of admission and try to understand the factors that can lead to their admission.
