import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt


# Loading file
  train=pd.read_csv('/kaggle/input/titanic/train.csv')
  
  test=pd.read_csv('/kaggle/input/titanic/test.csv')

  pd.set_option('display.max_rows', 10)

# Data Exploratry Analysis (DEA)
  train
  train.info()
  
  train.describe(include=[np.object])
  
  train.describe()

# FEATURE SELECTION

# selecting strong corelation between variable
  train.corr()[(train.corr()>0.5) | (train.corr()<-0.5)]['Survived'].dropna()

# using heat map and pairplot to find colinearity between the features and visualization of feature
  temp=train[temp.index].corr().sort_values('SalePrice',ascending=False)
  
  plt.figure(figsize=(12,8))
  
  sns.heatmap(data=temp,vmin=0,vmax=1,annot=True)
  
  sns.pairplot(train,hue='Survived')

# Finding among strong feature finding with very low co-linearity among them 
  for i in range(temp.shape[0]):
  
      for j in range(temp.shape[1]):
      
          if (temp.iloc[i,j]<0.2) and (temp.iloc[i,j]>-0.2):
          
              print(temp.index[i],temp.columns[j],temp.iloc[i,j])
              
# FEATURE ENGINEERING

# Droping missing columns
  train.drop(['Cabin'],axis=1,inplace=True)
  
 #Category Encoding of feature 
  def val_change(val):
  
    if val=='male':
    
        return 1
        
    else:
    
        return 0
        
  #Feature Construction
  
  train['relative']=train['Parch']+train['SibSp']
  
  # Outlier Data
   pd.set_option('display.max_rows', None)
   
   train.describe(percentiles=[x/100 for x in range(101)])
   
  # Again checking the correlation among selected Feature
   sns.heatmap(train.corr(),annot=True,vmin=-1,vmax=1)
   
   train['Sex']=train['Sex'].map(val_change)

  # 3-D visualization of Data for better understanding of the 
  w=train['Survived']
 
  x=train['Pclass']
  
  y=train['Age']
  
  z=train['Sex']
  
  plt.figure(figsize=(15,15))
  
  axes = plt.axes(projection='3d')
  
  axes.scatter3D(x,y,z,c=w)
  
  plt.legend(train['Survived'])
  
  # Modeling the data
  from sklearn.preprocessing import StandardScaler 
  
  scaler=StandardScaler()
  
  x_train=scaler.fit_transform(train[['Pclass','Sex','relative']])
  
  y_train=train['Survived']
  
  #Fit the train data
  from sklearn.linear_model import LogisticRegression
  
  model=LogisticRegression()
  
  model.fit(x_train,y_train)
  
  #Preparing Test data set
  test['Sex']=test["Sex"].map(val_change)
  
  test['relative']=test['Parch']+test['SibSp']
  
  x_test=scaler.fit_transform(test[['Pclass','Sex','relative']])
  
  # Predicting the outcome for test Data set
  y_test=model.predict(x_test)
  
