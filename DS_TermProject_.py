import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


# Step 2. Dataset
print("===================================================================")
print("                         Step 2. Dataset")
print("===================================================================")

df=pd.read_csv("adult.data",header=None)

features = ['age','workclass','fnlwgt','education',
            'education num','marital-status','occupation','relationship',
            'race','sex','capital-gain','capital-loss','hours-per-week','native-country','outcome']

# make columns.
df.columns=features
'''
# replace dirty value to NaN
for col in features:
    if df[col].dtype == 'object':
      df[col] = df[col].str.strip()
df.replace({"?":np.nan},inplace=True)

print(df.isna().sum())
'''
selected_features = ['age', 'sex', 'workclass', 'education','education num',
                     'capital-gain','capital-loss','hours-per-week','outcome']
data = df[selected_features]


print("----- data.head(5) -----\n", data.head(5))
print("\n----- data.shape -----\n",data.shape)
print("\n----- data.index -----\n",data.index)
print("\n----- data.columns -----\n",data.columns)



# Step 3. Missing Values
print("===================================================================")
print("                    Step 3. Missing Values")
print("===================================================================")

# in original data
print("\n----- df.isna().sum() -> in original data -----\n")
for col in features:
    if df[col].dtype == 'object':
      df[col] = df[col].str.strip()
df.replace({"?":np.nan},inplace=True)

print(df.isna().sum())

print("\n----- before : data.isna().sum() -> data used in use -----\n")
# replace dirty value to NaN
for col in selected_features:
    if data[col].dtype == 'object':
      data[col] = data[col].astype(str).str.strip()
        # data[col] = data[col].str.strip()

data.replace({"?":np.nan},inplace=True)

# check for missing values
print(data.isnull().sum())

print("\n\tdata['workclass']'s data : ",data['workclass'][27])

# replace missing values with median
data['age'].fillna(data['age'].median(), inplace=True)
data['sex'].fillna(data['sex'].mode()[0], inplace=True)
data['workclass'].fillna(data['workclass'].mode()[0], inplace=True)
data['education'].fillna(data['education'].mode()[0], inplace=True)
data['education num'].fillna(data['education num'].median(), inplace=True)
data['capital-gain'].fillna(data['capital-gain'].median(), inplace=True)
data['capital-loss'].fillna(data['capital-loss'].median(), inplace=True)
data['hours-per-week'].fillna(data['hours-per-week'].median(), inplace=True)

print("\n----- after : data.isna().sum() -----\n")
# check for missing values again
print(data.isnull().sum())

print("\n\tdata['workclass']'s data : ",data['workclass'][27])

print()
print(data.shape)

print()



# Keep real data.
real_data=data.copy()




#===============================================================================
#============================== encoding function =================================
#===============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def label_enc(data):
    # encode the categorical columns using LabelEncoder
    le = LabelEncoder()
    data_encoded = data.apply(lambda col: le.fit_transform(col))
    return data_encoded

def onehot_enc(data):
    # Get the list of columns to encode
    cols_to_encode = data.columns.tolist()

    # Apply one-hot encoding to the selected columns
    for col in cols_to_encode:
        dummies = pd.get_dummies(data[col], drop_first=False)
        data = pd.concat([data, dummies], axis=1)
        data.drop(col, axis=1, inplace=True)

    return data





#===============================================================================
#============================== get y_pred function =================================
#===============================================================================


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Get y_pred value using training datas and X_test data.
def logistic_reg(X_train,y_train,X_test,cutoff):

    # Logistic Regression model
    log_reg = LogisticRegression()

    # fit training set
    log_reg.fit(X_train, y_train)

    # get y_pred using predict X_test.
    y_pred_prob = log_reg.predict_proba(X_test)
    y_pred=(y_pred_prob[:, 1] >= cutoff).astype(int)


    return y_pred


def decision_cls(X_train,y_train,X_test,cutoff):
    #Decision tree classifier
    dcs=DecisionTreeClassifier()
    
    #Train Decision tree classifer
    dcs = dcs.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred_prob = dcs.predict_proba(X_test)
    y_pred=(y_pred_prob[:, 1] >= cutoff).astype(int)
    
    return y_pred
    

def knn_cls(X_train,y_train,X_test,cutoff):
    #K Neighbors Classifier
    knn=KNeighborsClassifier()
    
    #Train knn classifer
    knn = knn.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred_prob = knn.predict_proba(X_test)
    y_pred=(y_pred_prob[:, 1] >= cutoff).astype(int)
    
    return y_pred
    




#===============================================================================
#============================== Modeling (Clustering) =================================
#===============================================================================

# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# import itertools

# features = ['age', 'sex', 'workclass', 'education','education num',
#                      'capital-gain','capital-loss','hours-per-week']

# en_X=encoding_data

# for combination in itertools.combinations(features, 2):
#   combination=list(combination)
#   f1=combination[0]
#   f2=combination[1]

#   # make clustering .
#   kmeans = KMeans(n_clusters=3, random_state=0)

#   # model fitting
#   kmeans.fit(en_X)

#   # get clustering labels_
#   labels = kmeans.labels_

#   print("---------------------- ( ",f1,",",f2," ) ----------------------")

#   # Visualize
#   # We have 8 features without outcome.
#   # So, Make all available cases of features.

#   plt.scatter(en_X[f1], en_X[f2] , c=labels)
#   plt.xlabel(f1)
#   plt.ylabel(f2)
#   plt.show()



from sklearn.model_selection import train_test_split

def standard_scale(data):
    # feature scaling (standard scaler)
    scaler = StandardScaler()

    data= scaler.fit_transform(data)
    
    
    return  data

def minmax_scale(data):
    # feature scaling (minmax scaler)
    scaler = MinMaxScaler()

    data= scaler.fit_transform(data)
    
    
    return  data

def maxabs_scale(data):
    # feature scaling (maxabs scaler)
    scaler = MaxAbsScaler()

    data= scaler.fit_transform(data)
    
    
    return  data

def robust_scale(data):
    # feature scaling (robustscaler)
    scaler = RobustScaler()

    data= scaler.fit_transform(data)
    
    
    return  data



#algorithm func
al_func=[logistic_reg,decision_cls,knn_cls]

#scaler func
sc_func=[standard_scale,minmax_scale,maxabs_scale,robust_scale]

#encoding func
enc_func=[label_enc,onehot_enc]


#evaluation result
result=[]
result_num=0



numerical_data=data[['age','education num','capital-gain','hours-per-week']]
categorical_data=data[['sex','workclass','education']]

features = ['age', 'sex', 'workclass', 'education','education num','capital-gain','capital-loss','hours-per-week']

#k-fold cross validation using same algoritm, for every selected parameters
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

y =LabelEncoder().fit_transform(data['outcome'])



K=[2,3,4,5,6,7,8,9,10]# k for k-fold 
cutoff=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for j in cutoff:
    for i in K:
        for sc_f in sc_func:
            for enc_f in enc_func:
            
           
            
                num_d=sc_f(numerical_data)
                cat_d = enc_f(categorical_data)
                num_d = pd.DataFrame(num_d, columns=numerical_data.columns)
                cat_d = pd.DataFrame(cat_d, columns=cat_d.columns)
           
                
                X = pd.concat([num_d, cat_d], axis=1)
            
            
            
                kf=KFold(n_splits=i)
                for al_f in al_func:
                    y_accuracy=[]
                    y_precision=[]
                    y_recall=[]
                    y_f1=[]
                
                    for train_index,test_index in kf.split(X):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train,y_test=y[train_index],y[test_index]

                        y_pred=al_f(X_train,y_train,X_test,j)
                        
                        accuracy = accuracy_score(y_test,y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        
                        y_accuracy.append(accuracy)
                        y_precision.append(precision)
                        y_recall.append(recall)
                        y_f1.append(f1)
                
                
                    result.append([[i,sc_f,enc_f,al_f,sum(y_accuracy) / len(y_accuracy),sum(y_precision) / len(y_precision),sum(y_recall) / len(y_recall),sum(y_f1) / len(y_f1),j],list(features)])
                    #print("k=",result[result_num][0][0],", used features=",result[result_num][1],", scaler=",result[result_num][0][1].__name__,", encoder=",result[result_num][0][2].__name__,", algorithm=",result[result_num][0][3].__name__,", accuracy=",result[result_num][0][4],", precision=",result[result_num][0][5],", recall=",result[result_num][0][6],", f1=",result[result_num][0][7],", cutoff=",result[result_num][0][8])
                    result_num+=1
 


#k-fold cross validation using same algoritm, but select defferent combination of model parameters
import itertools
for j in cutoff:
    for i in K:
        for sc_f in sc_func:
            for enc_f in enc_func:
            
           
            
                num_d=sc_f(numerical_data)
                cat_d = enc_f(categorical_data)
                num_d = pd.DataFrame(num_d, columns=numerical_data.columns)
                cat_d = pd.DataFrame(cat_d, columns=cat_d.columns)
           
                
                X = pd.concat([num_d, cat_d], axis=1)
                for feature_num in range(6):
                    for combination in itertools.combinations(features, feature_num):
                        combination=list(combination)
                        X=X[combination]
            
                        kf=KFold(n_splits=i)
                        for al_f in al_func:
                            y_accuracy=[]
              
                            for train_index,test_index in kf.split(X):
                                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                                y_train,y_test=y[train_index],y[test_index]

                                y_pred=al_f(X_train,y_train,X_test,j)
                        
                        accuracy = accuracy_score(y_test,y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        
                        y_accuracy.append(accuracy)
                        y_precision.append(precision)
                        y_recall.append(recall)
                        y_f1.append(f1)
                
                
                    result.append([[i,sc_f,enc_f,al_f,sum(y_accuracy) / len(y_accuracy),sum(y_precision) / len(y_precision),sum(y_recall) / len(y_recall),sum(y_f1) / len(y_f1),j],combination])
                    #print("k=",result[result_num][0][0],", used features=",result[result_num][1],", scaler=",result[result_num][0][1].__name__,", encoder=",result[result_num][0][2].__name__,", algorithm=",result[result_num][0][3].__name__,", accuracy=",result[result_num][0][4],", precision=",result[result_num][0][5],", recall=",result[result_num][0][6],", f1=",result[result_num][0][7],", cutoff=",result[result_num][0][8])
                    result_num+=1                

#Top 5 Results with High Accuracy            
result.sort(key=lambda x: x[0][4], reverse=True)
for i in range(5):
    print("k=",result[i][0][0],", used features=",result[i][1],", scaler=",result[i][0][1].__name__,", encoder=",result[i][0][2].__name__,", algorithm=",result[i][0][3].__name__,", accuracy=",result[i][0][4],", precision=",result[i][0][5],", recall=",result[i][0][6],", f1=",result[i][0][7],", cutoff=",result[i][0][8])

#Top 5 Results with High Precision
result.sort(key=lambda x: x[0][5], reverse=True)                    
for i in range(5):
    print("k=",result[i][0][0],", used features=",result[i][1],", scaler=",result[i][0][1].__name__,", encoder=",result[i][0][2].__name__,", algorithm=",result[i][0][3].__name__,", accuracy=",result[i][0][4],", precision=",result[i][0][5],", recall=",result[i][0][6],", f1=",result[i][0][7],", cutoff=",result[i][0][8])

#Top 5 Results with High Recall
result.sort(key=lambda x: x[0][6], reverse=True)                    
for i in range(5):
    print("k=",result[i][0][0],", used features=",result[i][1],", scaler=",result[i][0][1].__name__,", encoder=",result[i][0][2].__name__,", algorithm=",result[i][0][3].__name__,", accuracy=",result[i][0][4],", precision=",result[i][0][5],", recall=",result[i][0][6],", f1=",result[i][0][7],", cutoff=",result[i][0][8])

#Top 5 Results with High f1
result.sort(key=lambda x: x[0][7], reverse=True)                    
for i in range(5):
    print("k=",result[i][0][0],", used features=",result[i][1],", scaler=",result[i][0][1].__name__,", encoder=",result[i][0][2].__name__,", algorithm=",result[i][0][3].__name__,", accuracy=",result[i][0][4],", precision=",result[i][0][5],", recall=",result[i][0][6],", f1=",result[i][0][7],", cutoff=",result[i][0][8])
