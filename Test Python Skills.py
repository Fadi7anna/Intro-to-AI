#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Question 1
n=[]
for x in range(2000,3201):
    if (x%7==0) and (x%5!=0):
        n.append(x)
print(n)


# In[6]:


#Question 2
num = int(input("enter a number: "))
fac = 1
for i in range(1, num+1):
    fac = fac * i
print("factorial of ", num, " is ", fac)


# In[44]:


#Question 3
num= int(input("Enter a number: "))
mydict = {i : i*i for i in range(1,num+1)}
print(mydict)
 


# In[50]:


#Question 5
import numpy as np
b=np.array([[0,1],[2,3],[4,5]])
print(b)
mylist=b.tolist()
print(mylist)


# In[52]:


#Question 6
import numpy as np
x=[ 0., 1., 2.]
y=[2., 1., 0.]
cov=np.cov(x,y)
print(cov)


# In[63]:


#Question 7
D=int(input("Enter a number: "))
import math
print(math.sqrt(2 * 50 * D)//30)


# In[108]:


#Question 4
new_str = ''.join([word[i] for i in range(len(word)) if i != 3])
print(new_str)


# In[ ]:




