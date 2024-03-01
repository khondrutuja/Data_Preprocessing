# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:50:29 2023

@author: Dell
"""

import pandas as pd
import numpy as np

df=pd.read_csv("C:/DATA/2-dataset/ethnic_diversity.csv")
df
########################
#EDA
df.shape
df.columns
df.info
df.size
print(df.dtypes)
df.discribe()    
df.index
#Access one columns contentethnic_diversity
df["Employee_Name"]
#Access multiple column
df[["Employee_Name","EmpID"]]
df1=df[2:]
df1
df['Employee_Name'][3]
df.describe()
###############################################3
df.Salaries=df.Salaries.astype(int)
df.dtypes
#now the data type of salaries is int
df.age=df.age.astype(float)
df.dtypes
######################################
#finding the dublicate
df_new=pd.read_csv("C:/DATA/2-dataset/education.csv")
duplicate=df_new.duplicated()

duplicate
sum(duplicate)
#output is 0


df_new1=pd.read_csv("C:/DATA/2-dataset/mtcars_dup.csv")
df_new1
duplicate1=df_new1.duplicated()
duplicate1
sum(duplicate1)
#there are 3 duplicate records
#row 17 is duplicate of row 2 like wise you can 3 duplicate
########################################
#outlier analysys treatement

import pandas as pd
import seaborn as sns
df=pd.read_csv("C:/DATA/2-dataset/ethnic_diversity.csv")
#Now let us find outlier in Salaries
sns.boxplot(df.Salaries)
#there are outlier
#let us check outliers in age column
sns.boxplot(df.age)
# there are no age
#let us calculate IQR
IQR=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
#have observe IQR in variable explorer
# no IQR is in capital letters
#create constant
IQR
#but if we will try as I,Iqr or iqr then it is showing

lower_limit=df.Salaries.quantile(0.25)-1.5*IQR
higher_limit=df.Salaries.quantile(0.75)+1.5*IQR
################################################
#Triming
import numpy as np
outliers_df=np.where(df.Salaries>higher_limit,True,np.where(df.Salaries<lower_limit,True,False))
#you can check outliers_df column in variable explorer
df_trimmed=df.loc[~outliers_df]
df.shape
df_trimmed.shape
#####################################################
#Replacement Tech
#Drowback of trimming is we are lossing the data
df=pd.read_csv("C:/DATA/2-dataset/ethnic_diversity.csv")
df.describe()

#record no.23 has got outliers
#map all the outlier values to upper limit
df.replaced=pd.DataFrame(np.where(df.Salaries>higher_limit,higher_limit,np.where(df.Salaries<lower_limit,lower_limit,df.Salaries)))
df
sns.boxplot(df.replaced[0])
#################################################
#Winsorizer
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Salaries']
                  )
#copy winsorizer and paste in help tab of
#top right window,study the method
df_t=winsor.fit_transform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])
################################################
##6/10/2022
import pandas as pd
import numpy as np
df=pd.read_csv("C:/DATA/2-dataset/Boston.csv")
df
############################
#EDA
df.shape
df.columns
df.info
df.size
print(df.dtypes)

df.index
#Access one columns contentethnic_diversity
df["crim"]
#Access multiple column
df[["zn","indus"]]
df1=df[2:]
df1
df['crim'][3]
df.describe()

df.tax=df.tax.astype(int)
df.dtypes
#now the data type of salaries is int
df.age=df.age.astype(float)
df.dtypes

sns.boxplot(df['tax'])
################################
#finding the dublicate

duplicate=df.duplicated()

duplicate
sum(duplicate)
######################################
#outlier analysys treatement


#there are outlier
#let us check outliers in age column
sns.boxplot(df.age)
# there are no age
#let us calculate IQR
IQR=df.tax.quantile(0.75)-df.tax.quantile(0.25)
#have observe IQR in variable explorer
# no IQR is in capital letter
#create constant
IQR
#but if we will try as I,Iqr or iqr then it is showing

lower_limit=df.tax.quantile(0.25)-1.5*IQR
higher_limit=df.tax.quantile(0.75)+1.5*IQR


################################################






































