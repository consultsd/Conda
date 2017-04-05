
# coding: utf-8

# In[1]:

#Open command prompt (Windows Power Shell) and type the following -
#pip install pandas-profiling
import numpy as np
import pandas as pd
import pandas_profiling 

#To find version of pandas
pd.__version__

#Type the following in Windows Powershell to upgrade to latest version
#conda update pandas

get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 25,10
import matplotlib.pyplot as plt

pd.set_option('display.precision',2)
pd.set_option('display.float_format','{:,}'.format)

#pd.reset_option('all')
#http://chrisalbon.com/ #Link for Pandas operations


# In[2]:

#Importing train and test datasets
data_path = "C:/Users/Sharath P Dandamudi/Desktop/"
train_file = data_path + "TrainingData.csv"
test_file = data_path +  "EvaluationData.csv"

train1 = pd.read_csv(train_file,sep='|',low_memory=False) 
test1 = pd.read_csv(test_file,sep='|',low_memory=False)


# In[3]:

train1.head(10)


# In[4]:

#To avoid rerunning the codes to import datsets if overwritten
train=train1
test=test1


# In[5]:

#Checking the dimensions of train and test datasets
print train.shape,test.shape


# In[34]:

#Combine train and test into a dataset
train['source']= 'train'
test['source'] = 'test'
data=pd.concat([train, test],ignore_index=True)
data.shape


# In[7]:

data.head(10)


# In[8]:

#EDA 
pfr = pandas_profiling.ProfileReport(data)
output_file = data_path + "eda_output.html"
pfr.to_file(output_file)


# In[35]:

print "Frequency distribution of target variable in the train dataset"
train['Status'].value_counts()


# In[36]:

#Class distribution
np.round(100. * train['Status'].value_counts() / len(train['Status']))


# In[22]:

data.columns


# In[40]:

#Reducing categories for categorical variable - User tenure class
data['User_tenure_class New'] = data['User_tenure_class'].apply(lambda x: 
                                                                    '1-3 YRS' if x in ['1-3 YRS'] 
                                                                else '3-10 YRS' if x in ['3-5 YRS','5-10 YRS']
                                                                else '0-12 MON' if x in ['0-6 MONTHS','6-12 MONTHS','Not available']
                                                                else '10+ YRS')
np.around((100. * data['User_tenure_class New'].value_counts() / len(data['User_tenure_class New'])),decimals=2)


# In[42]:

#Reducing categories for categorical variable - User category
data['User_Category New'] = data['User_Category'].apply(lambda x: 
                                                        'EM and Others' if x in ['EM','Others','NaN'] else x)
np.around((100. * data['User_Category New'].value_counts() / len(data['User_Category New'])),decimals=2)


# In[ ]:

#Reducing categories for categorical variable - Weekday
data['Weekday New'] = data['Weekday (based on drop-off date)'].apply(lambda x: 
                                                                    'Weekday' if x not in ['Fri','Sat','Sun'] else 'Weekend')
np.around((100. * data['Weekday New'].value_counts() / len(data['Weekday New'])),decimals=2)


# In[43]:

#Reducing categories for categorical variable - Request type 1
data['Request Type New'] = data['Request type 1'].apply(lambda x: 
                                                        'Others' if x not in ['Formatting & Consistency','Content Creation (mixed)'
                                                                              ,'Visual enhancements'] else x)
np.around((100. * data['Request Type New'].value_counts() / len(data['Request Type New'])),decimals=2)


# In[45]:

#Reducing categories for categorical variable - Office
data['Office New'] = data['Office'].apply(lambda x: 
                                                'High Risk' if x in ['PIT','HOU','LAN','PNW','LLN','BDP','BUC',
                                                                            'AFD','NAI','AMI','GRK','KAR','CAI','BAH',
                                                                            'NBI','SAI','SKB','OSL','HEL','SCC','STH'] 
                                                else 'Low Risk' if x in ['MSR','MIL','ADD','MOC','CAS','QTR','LUA','BRV']
                                                else 'Medium Risk')
np.around((100. * data['Office New'].value_counts() / len(data['Office New'])),decimals=2)


# In[46]:

#Perfoming numeric coding of all categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['User_tenure_class New','User_Category New','Weekday New','Request Type New','Office New']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])
data.head(5)


# In[47]:

#Perfoming one-hot encoding of all categorical variables
data = pd.get_dummies(data, columns=var_to_encode)
data.head(5)


# In[56]:

#Separating train and test data again
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']


# In[57]:

#Dropping the source -(Train/Test) variable
if __name__ == '__main__':
    train.drop('source',axis=1,inplace=True)
    test.drop(['source','Status'],axis=1,inplace=True)


# In[50]:

#Exporting the cleansed datasets for further analysis
train.to_csv(data_path + 'train_modified.csv',index=False)
test.to_csv(data_path + 'test_modified.csv',index=False)


# In[58]:

#Dropping the identifier variable in the training dataset
train_id=train['Res Id']
train.drop(['Res Id'],axis=1,inplace=True)


# In[59]:

#Dropping categorical variables
var = [
'Activity code',
'BookingCreator (Sanitized)',
'CC Code (Sanitized)',
'CONFIRMATION_SENT_TIME',
'CSS Designation',
'CSS/User (Sanitized)',
'CSSTimeZone',
'Category',
'Charge Code (Sanitized)',
'Cluster (Charge code)',
'Complexity1',
'Complexity2',
'Criticality',
'DL_Extn_Confirmed',
'Deadline',
'Edits',
'EstimatedTime',
'First_confirm_Creator (Sanitized)',
'First_confirm_Time',
'Functional class',
'Functional practice',
'MixType',
'NEGOTIATION_SENT_TIME',
'New',
'NonEnglish',
'Office',
'PP',
'Practice',
'Practice type',
'Prop_TZone',
'Prop_dl_date',
'Prop_dl_time',
'Prop_dropoff_date',
'Prop_dropoff_time',
'RFI_SENT_TIME',
'RTR',
'Region (Charge code)',
'Request type 1',
'Request type 2',
'Request_Submitted_Time',
'ReservationCreateddatetime',
'Reservedby (Sanitized)',
'Serviceline',
'Template',
'TierType1',
'TierType2',
'User_Category',
'User_tenure_class',
'Weekday (based on drop-off date)',
'dropoff_date_css',
'dropoff_time_css',
'formatting'
]

train.drop(var,axis=1,inplace=True)
data.columns


# In[60]:

train.head(5)


# In[61]:

#Plotting feature importance using built-in function - XGBoost
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

X_train = train.drop(['Status'], axis=1)
Y_train = train['Status']

#Fitting model on training data
xgb_fimp = XGBClassifier()
xgb_fimp.fit(X_train, Y_train)

#Plotting feature importance
plot_importance(xgb_fimp)
pyplot.show()


# In[81]:

#XG boost model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
seed = 7
test_size = 0.33
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_train, Y_train, test_size=test_size,
random_state=seed)
# fit model on training data
xgb = XGBClassifier()
xgb.fit(X_train_new, y_train_new)
# make predictions for test data
y_pred = xgb.predict(X_test_new)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test_new, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[76]:

#Confusion matrix
matrix_xgb = confusion_matrix(Y_test_new,y_pred)
matrix_xgb


# In[79]:

# Random Forest Classification
import pandas
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

X_train = train.drop(['Status'], axis=1)
Y_train = train['Status']

num_folds = 10
num_instances = len(X_train)
seed = 7
num_trees = 100
max_features = 3
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
model_rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
model_rf.fit(X_train_new, y_train_new)


# In[77]:

#Split into train and test datasets 
#Conventional random sampling - Logistic regression
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
test_size = 0.33
seed = 7

X_train_new, X_test_new, Y_train_new, Y_test_new = cross_validation.train_test_split(X_train, Y_train,
test_size=test_size, random_state=seed)
log_reg = LogisticRegression()
log_reg.fit(X_train_new, Y_train_new)
result = model.score(X_test_new, Y_test_new)
print("Accuracy: %.3f%%") % (result*100.0)


# In[75]:

# K fold cross validation - Logistic regression
num_folds = 10
num_instances = len(X_train)
seed = 7
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
log_reg = LogisticRegression()
results = cross_validation.cross_val_score(log_reg, X_train, Y_train, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


# In[78]:

#Confusion Matrix - Logistic regression (This will be relevant only for binary classification problems)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

test_size = 0.33
seed = 7

X_train_new, X_test_new, Y_train_new, Y_test_new = cross_validation.train_test_split(X_train, Y_train,
test_size=test_size, random_state=seed)
    
log_reg = LogisticRegression()
log_reg.fit(X_train_new, Y_train_new)
predicted = log_reg.predict(X_test_new)
matrix_log_reg= confusion_matrix(Y_test_new, predicted)
matrix_log_reg


# In[114]:

test_X = test.copy()


# In[115]:

test_X.shape


# In[116]:

test_X.drop(var,axis=1,inplace=True)
test_X.drop('Res Id',axis=1,inplace=True)


# In[117]:

test_X.head(10)


# In[118]:

pred_test_xgb = xgb.predict(test_X)


# In[119]:

test_id = np.array(test['Res Id'])


# In[121]:

# Writing the submission file #
out_df = pd.DataFrame({"Res Id":test_id,"Status":pred_test_xgb})
out_df.to_csv("C:/Users/Sharath P Dandamudi/Desktop/sub.csv", sep='|', index=False)


# In[ ]:



