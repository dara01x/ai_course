#lec 1- Dr.Adnan - example 2- page 8-9
import pandas as pn
from sklearn import datasets, linear_model

x,y, _= datasets.make_regression(n_samples=50,n_features=1,coef = True ,n_informative=1,noise = 10, random_state= 3 ) # generate a random dataset of random numbers
    #under score here means a variable that we don't want, python will get rid of it for us
    #because make_regresion function returns 3 thing (input"saved in x" , output "saved in y", coofitioent"we get rid of it")
    #if we don't want coef we can remove them by setting (coef = False) 

df = pn.DataFrame()
df['kitchen area'] = pn.Series(x[:,0]) # sets the first column(feature) to df, df is also an array but with more capabilaties
df['house price'] = pn.Series(y)
print(df.head(4))#print the first 4 rows from df
print(df.describe())

# outout 
#   kitchen area  house price
# 0     -1.101068   -75.647429
# 1     -2.419083  -191.949001
# 2     -0.131914    -2.323307
# 3      1.486148   123.385734
#        kitchen area  house price
# count     50.000000    50.000000
# mean      -0.293927   -18.692906
# std        1.008602    74.648357
# min       -2.419083  -191.949001
# 25%       -0.911315   -70.523494
# 50%       -0.355515   -25.443380
# 75%        0.201662    26.182767
# max        1.976111   152.093363