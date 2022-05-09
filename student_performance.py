# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:34:42 2019

@author: Prakash Agrawal
"""

import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy


#loading of data
data=pandas.read_csv(r"C:\Users\Admin\Desktop\train.csv")

data.info()
data.describe()
data.isna() # No NAN data is present

#slicing
Df=pandas.DataFrame(data, columns=['Medu','Fedu','G3'])
Df
Df1=Df.iloc[0:3]
Df1







cor=data.corr()
plt.figure(figsize=(15,10))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()

#Age, Medu, Fedu, traveltime,failures,goout,G1,G2 #more impacting

plt.figure(figsize=(12,5))
sns.distplot(data.G3[data.sex=='M'])
sns.distplot(data.G3[data.sex=='F'])
plt.legend(['M','F'])
plt.show()

plt.figure(figsize=(12,5))
sns.distplot(data.G1[data.sex=='M'])
sns.distplot(data.G1[data.sex=='F'])
plt.legend(['M','F'])
plt.show()


data.info()  #object(categorical value)


cat=['school','address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
       'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic']   #removed non categorical value

len(cat)#for checking categorical values

for i in cat:
    c=list(data[i].unique())
    plt.figure(figsize=(12,5))
    for j in c:
        sns.distplot(data.G3[data[i]==j])
    plt.title(i)
    plt.legend(c)
    plt.show()
    
#romantic,internet,Fjob,address   (important which is impacting)
#higher,schoolsup,
#Age, Medu, Fedu, traveltime,failures,goout,G1,G2
data2=data[['sex', 'age', 'address','Medu', 'Fedu',
       'Fjob','traveltime','failures', 'schoolsup',
       'higher', 'internet', 'romantic', 'goout' ,'G1', 'G2', 'G3']]

data2.shape

xdata=data2.drop(['G3'],axis=1)
ydata=data2['G3']

xdata.shape
xdata['teacher']=xdata['Fjob']=='teacher'  # here we have taken whose father is teacher or not

plt.figure(figsize=(12,5))   #want if father is teacher then yes otherwise no
sns.distplot(xdata.G1[xdata.teacher==True])
sns.distplot(xdata.G1[xdata.teacher==False])
plt.legend(['T','F'])
plt.show()


xdata.drop(['Fjob'],axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
xdata.sex=le1.fit_transform(xdata.sex) # it is converted male as 1 and female as 1

le2=LabelEncoder()
xdata.address=le2.fit_transform(xdata.address)  #it is converted urban as 1 and rural as 0

le3=LabelEncoder()
xdata.schoolsup=le3.fit_transform(xdata.schoolsup) # school support yes as 1 and no as 0

le4=LabelEncoder()
xdata.higher=le4.fit_transform(xdata.higher)

le5=LabelEncoder()
xdata.internet=le5.fit_transform(xdata.internet)

le6=LabelEncoder()
xdata.romantic=le6.fit_transform(xdata.romantic)


###########################################################

#split the data into train and test section (spliting is used to check whether the data is good or bad)

from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(xdata,ydata,test_size=0.2,random_state=3)


### Decision Tree
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(max_depth=3)
#train the algorithm
model.fit(xtr,ytr)

#check the r2 score of the model(accuracy)
model.score(xtr,ytr)#train data

model.score(xts,yts)#test data

"""
#prediction from xtr
M = 1
age=16
urban=1
Med=4
Fed=2
traveltime=2
failures=0
schoolup=1
higheredu=1
internet=0
romantic=1
goout=2
g1=11
g2=13
G3?
"""

ip=[[1,16,1,4,2,2,0,1,1,0,1,2,1,11,13]]
model.predict(ip)# so by the prediction it will able to score 11.35 as G3

#-----------------------------

xts.iloc[18,:]   # checking for real one student suppose (18th row and all the column) 
yts.iloc[18]   #10

model.predict(numpy.array(xts.iloc[18,:]).reshape(1,15)) #algorithm used for student score  7.95

