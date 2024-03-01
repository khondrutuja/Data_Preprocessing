# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 08:29:53 2023

@author: Dell
"""

# one hot encoder
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
#we use ethnic diversity dataset
df=pd.read_csv("C:/DATA/2-dataset/ethnic_diversity.csv")
df.columns
#We have salaries and age as a numerical column, let us make them
#at position 0 and 1 so to make further data processing easy

df=df[['Salaries','age','Position','State','Sex','MaritalDesc','CitizenDesc','EmploymentStatus','Department','Zip','Race']]
#check the dataframe in variable exploere
#we want only nominal data and ordinal data for processing
#hence skip oth and 1st column and apply to onehotencoder
enc_df=pd.DataFrame(enc.fit_transform(df.iloc[:,2:]).toarray())
#############################################################
#Lable encoder=nominal data but not ordinal data
from sklearn.preprocessing import LabelEncoder
#creating instance of lable encoder
labelencoder=LabelEncoder()
#split tha data into input or output variable
X=df.iloc[:,0:9]
y=df['Race']
df.columns
#we have nominal data Sex,MaritalDesc,CitizenDesc,
#We want to convert lable encoder
X['Sex']=labelencoder.fit_transform(X['Sex'])
X['MaritalDesc']=labelencoder.fit_transform(X['MaritalDesc'])
X['CitizenDesc']=labelencoder.fit_transform(X['CitizenDesc'])
#Lable encoder y
y=labelencoder.fit_transform(y)
#this is going to create an array,hence convert
#it back to dataframe
y=pd.DataFrame(y)
df_new=pd.concat([X,y],axis=1)
#if you will see variable explorer,y do not have column name
#Hence rename the column
df_new=df_new.rename(columns={0:'Race'})
#####################################################
















