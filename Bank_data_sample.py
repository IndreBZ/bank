# %%
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
#%matplotlib inline 

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

# %%
bank_data= pd.read_csv('https://raw.githubusercontent.com/IndreBZ/bank/main/bank_sample.csv')
print("The first 5 rows of the bank data") 
bank_data.head()

#the first model
##The model
df_LR_new = bank_data
print('change categorical columns to numeric')
df_LR_new['y'].replace(['no', 'yes'],[0, 1], inplace=True)
df_LR_new['job'].replace(["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],[0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)
df_LR_new['marital'].replace(["married","divorced","single"],[0, 1,2], inplace=True)
df_LR_new['education'].replace(["unknown","secondary","primary","tertiary"],[0,1,2,3], inplace=True)
df_LR_new['default'].replace(['no', 'yes'],[0, 1], inplace=True)
df_LR_new['housing'].replace(['no', 'yes'],[0, 1], inplace=True)
df_LR_new['loan'].replace(['no', 'yes'],[0, 1], inplace=True)
df_LR_new['contact'].replace(['unknown','telephone','cellular'],[0, 1,2], inplace=True)
df_LR_new['month'].replace(['jan', 'feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],[0, 1,2,3,4,5,6,7,8,9,10,11], inplace=True)
df_LR_new['poutcome'].replace(["unknown","other","failure","success"],[0,1,2,3], inplace=True)
print("Change column data type")
df_LR_new.dtypes

x = np.asarray(df_LR_new[['age','job','marital','education','default','balance','housing','loan','campaign','pdays','previous','poutcome']])
print('x',x[0:5])
y = np.asarray(df_LR_new['y'])
print('y',y[0:5])
df_LR_new.corr()

# %%
x = np.asarray(df_LR_new[['age','job','education','marital','default','housing','balance','loan','contact','duration']])## added duration which was deleted in the df dataset
y = np.asarray(df_LR_new['y'])
x = preprocessing.StandardScaler().fit(x).transform(x)
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=5)
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)
yhat = LR.predict(x_test)
yhat_prob = LR.predict_proba(x_test)
jaccard_score(y_test, yhat,pos_label=0)
#define x(dependent variable) and y(independent variable) of dataset
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))
# %%
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['y=1','y=0'],normalize= False,  title='Confusion matrix')

# %%

df_LR_new.corr()



len(bank_data)

# %%
