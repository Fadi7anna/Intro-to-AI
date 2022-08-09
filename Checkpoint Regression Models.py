#!/usr/bin/env python
# coding: utf-8

# In[96]:


#Importing libraries

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 

from sklearn.metrics import mean_squared_error
from sklearn import metrics


# In[97]:


#Imporing the data set

data =pd.read_csv(r'C:\\Users\\Fadi\\Desktop\\kc_house_data.csv')
data.head()

data.info()
data.describe()


# In[98]:


#FIRST STAGE: Preprocessing
##Checking if there's any NaN values to fill/clean it

data.isnull() 
###creating bool series True for NaN values 

data.isnull().sum()
data.isnull().sum().sum()

##Concl:There's no NaN or empty values in our Dataset.


# In[99]:


#SECOND STAGE: Visualization of Data 

print(data.corr())


# In[100]:


##The Correlations between features on heatmap

import seaborn as sns

def plot_correlation_map(data):
    corr = data.corr()
s , ax = plt.subplots( figsize =( 17 , 14 ) )
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
sns.heatmap(data.corr(), cmap = cmap,square=True,cbar_kws={ 'shrink' : .9 },ax=ax,annot = True,annot_kws = { 'fontsize' : 10 })


#The map represents the correlation between features (The Pearson's Correlation).
#The diagonal is dark red, with the value=1, because those squares coreelates the one feature to itself.
#The rest squares are lighter colors because they lesser than 1 or -1 (value of perfect correlation).
#So, it tells us how strong is the correlation between features.


# In[101]:


#Heatmap between top 6 correlated variables with "price"

corrMatrix=data[["price","bathrooms","sqft_living","view",
                  "grade","sqft_above","sqft_living15",]].corr()

sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between top features');


# In[102]:


#Visualizing some important features to exploratory data analysis

sns.distplot(data["price"], bins=50, hist=True, kde=True, color= "red")

plt.show()


# In[103]:


data['Construction and Reconstrucion Age'] = data['yr_renovated'] - data['yr_built']
plt.scatter(data['Construction and Reconstrucion Age'], data['price'])
plt.ylabel('Price')
plt.xlabel("Construction Age of house");


# In[104]:


plt.scatter(data['grade'], data['price'])
plt.ylabel('Price')
plt.xlabel("grade/quality level of construction and design");


# In[105]:


plt.scatter(data['sqft_living'], data['price'])
plt.ylabel('Price')
plt.xlabel("sqft_living");


# In[106]:


plt.scatter(data['sqft_living15'], data['price'])
plt.ylabel('Price')
plt.xlabel("sqft_living");


# In[107]:


plt.scatter(data['sqft_above'], data['price'])
plt.ylabel('Price')
plt.xlabel("sqft_living");


# In[108]:


###THIRD STAGE: Building Models
#Linear Regression model for 1 input

# extract x and y from our data

x= data["price"].values[:,np.newaxis]
y= data["sqft_living"].values

# splitting data with test size of 35%

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.35, random_state=40)
 
#Where: test_size defines the percentage you want for the test
#in our case 00.35 means 35% for testing and 65 % for training.
#Random_state: random_state number splits the test and training datasets with a random manner.


# In[109]:


model=LinearRegression()   #build linear regression model
model.fit(x_train,y_train)  #fitting the training data


# In[110]:


plt.scatter(x , y, color="r")
plt.title("Linear Regression")
plt.ylabel("sqft_living")
plt.xlabel("price")
plt.plot(x, model.predict(x),color="k")
plt.show()

predicted=model.predict(x_test) #testing our model’s performance


# In[116]:


y_pred = model.predict(x_test)
print(y_pred)


# In[117]:


print("MSE", mean_squared_error(y_test, predicted))
print("R squared", metrics.r2_score(y_test, predicted))


# In[118]:


# Linear regression with more than one feature: sqft_living, grade, sqft_above

#extract x and y from our data

x1=data[["sqft_living", "grade"]]  #we have more than one input
y1=data["price"].values

 #splitting data with test size of 35%
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.35,random_state=40)

model1=LinearRegression() #build linear regression model
model.fit(x1_train,y1_train) #fitting the training data
predicted1=model.predict(x1_test) #testing our model’s performance

print("MSE", mean_squared_error(y1_test,predicted1))
print("R squared", metrics.r2_score(y1_test,predicted1))

##Apparently, from the R suqared value, this model is more accurate than the previous Linear one.
##(i.e. The Multi-linear model is performing better)


# In[119]:


###Applying Polynomial model on features

xp= data[["sqft_living", "grade"]]
yp= data["price"].values

xp_train, xp_test, yp_train, yp_test = train_test_split(xp, yp, test_size=0.35, random_state=40)  #splitting data
lg=LinearRegression()
poly=PolynomialFeatures(degree=2)

xp_train_fit = poly.fit_transform(xp_train) #transforming our input data
lg.fit(xp_train_fit, yp_train)
xp_test_ = poly.fit_transform(xp_test)
predictedp = lg.predict(xp_test_)

print("MSE: ", metrics.mean_squared_error(yp_test, predictedp))
print("R squared: ", metrics.r2_score(yp_test,predictedp))


##Since the R2 value are more than the R2 value of both previous models, the Multi-Linear and the Linear, so it's more accurate and fitting the data


# In[120]:


###Plotting out Polynomial Regression Model

xp= data["sqft_living"].values.reshape(-1,1)
yp= data["price"].values
poly = PolynomialFeatures(degree = 2) 
xp_poly = poly.fit_transform(xp) 
poly.fit(xp_poly, yp) 
lg=LinearRegression()
lg.fit(xp_poly, yp) 

plt.scatter(xp, yp, color="r")
plt.title("Linear regression")
plt.ylabel("Price")
plt.xlabel("sqft_living")
plt.plot(xp, lg.predict(poly.fit_transform(xp)), color="k") 


# In[ ]:




