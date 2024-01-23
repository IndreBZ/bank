import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
#%matplotlib inline 
#import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools

bank_data= pd.read_csv('https://raw.githubusercontent.com/IndreBZ/bank/716b7cd2a11db98dc4175a8ccce927869caca041/bank2.csv')
print("The first 5 rows of the bank data") 
bank_data.head()
## output
The first 5 rows of the bank data
age	job	marital	education	default	balance	housing	loan	contact	day	month	duration	campaign	pdays	previous	poutcome	y
0	58	management	married	tertiary	no	2143	yes	no	unknown	5	may	261	1	-1	0	unknown	no
1	44	technician	single	secondary	no	29	yes	no	unknown	5	may	151	1	-1	0	unknown	no
2	33	entrepreneur	married	secondary	no	2	yes	yes	unknown	5	may	76	1	-1	0	unknown	no
3	47	blue-collar	married	unknown	no	1506	yes	no	unknown	5	may	92	1	-1	0	unknown	no
4	33	unknown	single	unknown	no	1	no	no	unknown	5	may	198	1	-1	0	unknown	no

bank_data.shape
## output
(45211, 17)

bank_data.dtypes
##output
age           int64
job          object
marital      object
education    object
default      object
balance       int64
housing      object
loan         object
contact      object
day           int64
month        object
duration      int64
campaign      int64
pdays         int64
previous      int64
poutcome     object
y            object
dtype: object

##find missing values
print('Find if there are missed values in data set')
missing_data = bank_data.isnull()
missing_data.head(5)
## no missing values
##output
Find if there are missed values in data set

age	job	marital	education	default	balance	housing	loan	contact	day	month	duration	campaign	pdays	previous	poutcome	y
0	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False
1	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False
2	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False
3	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False
4	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False	False


##DATA MANIPULATION TASK
#1. select random subsample of data set
print('random subsample of data set')
sample = bank_data.sample(frac=0.5, replace=False, random_state=1)
print('50% sample of data set')
print(sample.shape)
##output
random subsample of data set
50% sample of data set
(22606, 17)

#2. filter desired rows using simple and more complex conditions;
df = bank_data
print("how many clients aged(30-50) subscribed a term deposit?")
age_df = df[(df["age"] > 30) & (df["age"] < 50) & (df["y"] == "yes")]
print(age_df.shape)
print("how many clients aged(30-50) didn't subscribe a term deposit?")
age_df1 = df[(df["age"] > 30) & (df["age"] < 50) & (df["y"] == "no")]
print(age_df1.shape)
print("group by job and term deposit and calculate sum of balance. Filter jobs where balance > average")
job_result_grouped = df.groupby(["y", "job"]).agg({"balance": "sum"})
balance =job_result_grouped ["balance"].mean()
print("Balance average",balance)
job_result = job_result_grouped[(job_result_grouped["balance"] > balance)]
print(job_result)
##output
how many clients aged(30-50) subscribed a term deposit?
(2759, 17)
how many clients aged(30-50) didn't subscribe a term deposit?
(25228, 17)
group by job and term deposit and calculate sum of balance. Filter jobs where balance > average
Balance average 2566236.75
                  balance
y   job                  
no  admin.        4966497
    blue-collar   9596143
    management   13895227
    retired       3103899
    services      3731449
    technician    7972198
yes management    2785061

#3. drop unnecessary variables, rename some variables
#unnecessary variables could be related with the last contact of the current campaign
df.drop(["contact", "day","month","duration"], axis=1, inplace=True)
df.head()
print('rename some variables')
df.rename(columns={'campaign':'num_of_contact'}, inplace=True)
df.rename(columns={'pdays':'num_of_days'}, inplace=True)
df.head()
##output
	age	job	marital	education	default	balance	housing	loan	num_of_contact	num_of_days	previous	poutcome	y
0	58	management	married	tertiary	no	2143	yes	no	1	-1	0	unknown	no
1	44	technician	single	secondary	no	29	yes	no	1	-1	0	unknown	no
2	33	entrepreneur	married	secondary	no	2	yes	yes	1	-1	0	unknown	no
3	47	blue-collar	married	unknown	no	1506	yes	no	1	-1	0	unknown	no
4	33	unknown	single	unknown	no	1	no	no	1	-1	0	unknown	no

#4. calculate summarizing statistics (for full sample and by categorical variables as well)
print(bank_data.dtypes)
print(bank_data.describe())
print(bank_data.info())
#summarizing statistics only for categorical variables
df.describe()
##output
age                int64
job               object
marital           object
education         object
default           object
balance            int64
housing           object
loan              object
num_of_contact     int64
num_of_days        int64
previous           int64
poutcome          object
y                 object
dtype: object
                age        balance  num_of_contact   num_of_days      previous
count  45211.000000   45211.000000    45211.000000  45211.000000  45211.000000
mean      40.936210    1362.272058        2.763841     40.197828      0.580323
std       10.618762    3044.765829        3.098021    100.128746      2.303441
min       18.000000   -8019.000000        1.000000     -1.000000      0.000000
25%       33.000000      72.000000        1.000000     -1.000000      0.000000
50%       39.000000     448.000000        2.000000     -1.000000      0.000000
75%       48.000000    1428.000000        3.000000     -1.000000      0.000000
max       95.000000  102127.000000       63.000000    871.000000    275.000000
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 45211 entries, 0 to 45210
Data columns (total 13 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   age             45211 non-null  int64 
 1   job             45211 non-null  object
 2   marital         45211 non-null  object
 3   education       45211 non-null  object
 4   default         45211 non-null  object
 5   balance         45211 non-null  int64 
 6   housing         45211 non-null  object
 7   loan            45211 non-null  object
 8   num_of_contact  45211 non-null  int64 
 9   num_of_days     45211 non-null  int64 
 10  previous        45211 non-null  int64 
 11  poutcome        45211 non-null  object
 12  y               45211 non-null  object
dtypes: int64(5), object(8)
memory usage: 4.5+ MB
None
#5. create new variables using simple transformation and custom functions
print('change categorical columns to numeric')
df['y'].replace(['no', 'yes'],[0, 1], inplace=True)
df['job'].replace(["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],[0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)
df['marital'].replace(["married","divorced","single"],[0, 1,2], inplace=True)
df['education'].replace(["unknown","secondary","primary","tertiary"],[0,1,2,3], inplace=True)
df['default'].replace(['no', 'yes'],[0, 1], inplace=True)
df['housing'].replace(['no', 'yes'],[0, 1], inplace=True)
df['loan'].replace(['no', 'yes'],[0, 1], inplace=True)
df['poutcome'].replace(["unknown","other","failure","success"],[0,1,2,3], inplace=True)
print("Change column data type")
#look into updated data types and summarizing statistics for new dataset df
df.dtypes
df.describe()
##output
change categorical columns to numeric
	age	job	marital	education	default	balance	housing	loan	num_of_contact	num_of_days	previous	poutcome	y
count	45211.000000	45211.000000	45211.000000	45211.000000	45211.000000	45211.000000	45211.000000	45211.000000	45211.000000	45211.000000	45211.000000	45211.000000	45211.000000
mean	40.936210	6.018159	0.680963	1.698856	0.018027	1362.272058	0.555838	0.160226	2.763841	40.197828	0.580323	0.357767	0.116985
std	10.618762	3.543218	0.884908	0.938627	0.133049	3044.765829	0.496878	0.366820	3.098021	100.128746	2.303441	0.804435	0.321406
min	18.000000	0.000000	0.000000	0.000000	0.000000	-8019.000000	0.000000	0.000000	1.000000	-1.000000	0.000000	0.000000	0.000000
25%	33.000000	3.000000	0.000000	1.000000	0.000000	72.000000	0.000000	0.000000	1.000000	-1.000000	0.000000	0.000000	0.000000
50%	39.000000	7.000000	0.000000	1.000000	0.000000	448.000000	1.000000	0.000000	2.000000	-1.000000	0.000000	0.000000	0.000000
75%	48.000000	10.000000	2.000000	3.000000	0.000000	1428.000000	1.000000	0.000000	3.000000	-1.000000	0.000000	0.000000	0.000000
max	95.000000	11.000000	2.000000	3.000000	1.000000	102127.000000	1.000000	1.000000	63.000000	871.000000	275.000000	3.000000	1.000000

#6. order data set by several variables.
#df['job'].value_counts()
sorted_balance = df.sort_values(by=['balance'], ascending=True)
sorted_balance
sorted_age = df.sort_values(by=['age'], ascending=False)
sorted_age 
sorted_job_educational = df.sort_values(by=['job','education'], ascending=True)
sorted_job_educational

