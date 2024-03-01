# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:33:43 2023

@author: Dell
"""
####################################
#zero variance and near zero variance
# if there are no variance in  result the ML model
import pandas as pd
df=pd.read_csv("C:/DATA/2-dataset/ethnic_diversity.csv")
df.var()
#Emp ID and Zip is nominal data
# salaries has 4.44 which is not closed to zero
#####################################################
# varieance is zero or not or dataferame has complete varieance
df.var()==0
###############################
# none of them are equal
df.var(axis=0)==0 # for row
df.var(axis=1)==0 # for column
######################################
import pandas as pd
import numpy as np
df=pd.read_csv("C:/DATA/2-dataset/modified ethnic.csv")
#check for null value
df.isna().sum()
#################################################
#cereate an imputer that creates NaN values
#mean and midiean use for numeric data
#mode is use for discrite data
from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
# check the datframe dataframe
df["Salaries"]=pd.DataFrame(mean_imputer.fit_transform(df[['Salaries']]))
#check the dataframe
df['Salaries'].isna().sum()
#0

####################################################

import pandas as pd
data=pd.read_csv("C:/DATA/2-dataset/ethnic_diversity.csv")
data.head(10)
data.info
# it gives size,null values,row,columns and columns data types

data.describe()
data['Salaries_new']=pd.cut(data['Salaries'],bins=[min(data.Salaries),data.Salaries.mean(),max(data.Salaries)],labels=["Low","High"])
data.Salaries_new.value_counts()
data['Salaries_new']=pd.cut(data['Salaries'],bins=[min(data.Salaries),data.Salaries.quantile(0.25),data.Salaries.mean(),data.Salaries.quantile(0.75),max(data.Salaries)],labels=["group1","group2","group3","group4"])
data.Salaries_new.value_counts()
############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("C:/DATA/2-dataset/animal_category.csv")
df.shape
df.drop(["Index"],axis=1,inplace=True)
#check df again
df_new=pd.get_dummies(df)
df_new.shape
#Here we are getting 30 rows and 14 columns
#we are getting two columns for  homly and gender,one colums
#delete second column of gender and second column of homely
df_new.drop(["Gender_Male",'Homly_Yes'],axis=1,inplace=True)
df_new.shape
#Now we are getting 30,12
df_new.rename(columns={'Gender_Female':'Gender','Homly_No':'Homly'},inplace=True)
##########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("C:/DATA/2-dataset/ethnic_diversity.csv")
df.shape
df.columns
df.drop(["Index"],axis=1,inplace=True)
#check df again

df_new=pd.get_dummies(df)
df_new.shape
df_new.drop(["Position",'State'],axis=1,inplace=True)
df_new.shape
df_new.drop({"Employee_Name":'name','EmpID':'ID'})
df_new.shape










