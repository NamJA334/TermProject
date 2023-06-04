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

#features of adult data
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
#==============================algorithm(classification,regression) function =================================
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
    y_pred_prob = log_reg.predict_proba(X_test)#probability of y
    
    y_pred=(y_pred_prob[:, 1] >= cutoff).astype(int)#y prediction based on cutoff


    return y_pred


def decision_cls(X_train,y_train,X_test,cutoff):
    #Decision tree classifier
    dcs=DecisionTreeClassifier()
    
    #Train Decision tree classifer
    dcs = dcs.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred_prob = dcs.predict_proba(X_test)#probability of y
    y_pred=(y_pred_prob[:, 1] >= cutoff).astype(int)#y prediction based on cutoff
    
    return y_pred
    

def knn_cls(X_train,y_train,X_test,cutoff):
    #K Neighbors Classifier
    knn=KNeighborsClassifier()
    
    #Train knn classifer
    knn = knn.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred_prob = knn.predict_proba(X_test)#probability of y
    y_pred=(y_pred_prob[:, 1] >= cutoff).astype(int)#y prediction based on cutoff
    
    return y_pred
    





#===============================================================================
#============================== scalinig function =================================
#===============================================================================

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
enc_func=[label_enc]



#result=[k,used features, scaler, encoder, algorithm, accuracy, precision, recall, f1, cutoff]
result=[]
result_num=0



numerical_data=data[['age','education num','capital-gain','capital-loss','hours-per-week']]#numerical data for scaling
categorical_data=data[['sex','workclass','education']]#categorical data for encoding



#k-fold cross validation using same algoritm, for every selected parameters
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

#'outcome'
y =LabelEncoder().fit_transform(data['outcome'])



K=[10]# k for k-fold 
cutoff=[0.5]# cutoff for ROC

#PCA function for numerical data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
def cls_pca(data):
    pca=PCA(n_components=1)
    data_reduced=pca.fit_transform(data)
    data_reduced=pd.DataFrame(data_reduced,columns=['new_X'])
    
    return data_reduced

#corvariance
features = numerical_data.columns

cov_matrix = np.cov(standard_scale(numerical_data[features]).T)


cov_df = pd.DataFrame(cov_matrix, columns=features, index=features)

print(cov_df)

#education num is positively correlated with capital-gain and hours-per-week
new_X=cls_pca(numerical_data[['education num','capital-gain','hours-per-week']])
print(new_X)

#merge data
numerical_data=pd.concat([numerical_data[['age','capital-loss']],new_X],axis=1)
print(numerical_data)


#k-fold cross validation using same algoritm, but select defferent combination of model parameters
import itertools
for j in cutoff:#cutoff
    for i in K:# k for k-fold cross validation
        for sc_f in sc_func:#for scaling
            for enc_f in enc_func:#for encoding
                
                
                num_d=sc_f(numerical_data)#numerical data after scaling
                cat_d = enc_f(categorical_data)#categorical data after encoding
                num_d = pd.DataFrame(num_d, columns=numerical_data.columns)
                cat_d = pd.DataFrame(cat_d, columns=cat_d.columns)
                X = pd.concat([num_d, cat_d], axis=1)#merge cat data and num data

                for feature_num in range(2,X.shape[1]+1):#feature select
                     
                     
                     
                    #for select features
                    for combination in itertools.combinations(X.columns, feature_num):
                        
                        
                        combination=list(combination)#make list
                        
                        
                        X=X[combination]
                        
            
                        kf=KFold(n_splits=i)
                        for al_f in al_func:#for algorithm
                            
                            #List to get the average value
                            y_accuracy=[]
                            y_precision=[]
                            y_recall=[]
                            y_f1=[]
                            y_test_=[]
                            y_pred_=[]
              
                            for train_index,test_index in kf.split(X):#divide data
                                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                                y_train,y_test=y[train_index],y[test_index]

                                y_pred=al_f(X_train,y_train,X_test,j)#running model and get prediction data

                                #get score
                                accuracy = accuracy_score(y_test,y_pred)
                                precision = precision_score(y_test, y_pred)
                                recall = recall_score(y_test, y_pred)
                                f1 = f1_score(y_test, y_pred)

                                #append to list
                                y_accuracy.append(accuracy)
                                y_precision.append(precision)
                                y_recall.append(recall)
                                y_f1.append(f1)
                                y_test_=y_test_+list(y_test)
                                y_pred_=y_pred_+list(y_pred)
                
                            #append to result and print
                            result.append([[i,sc_f,enc_f,al_f,sum(y_accuracy) / len(y_accuracy),sum(y_precision) / len(y_precision),sum(y_recall) / len(y_recall),sum(y_f1) / len(y_f1),j,y_test_,y_pred_],combination])
                            #print("k=",result[result_num][0][0],", used features=",result[result_num][1],", scaler=",result[result_num][0][1].__name__,", encoder=",result[result_num][0][2].__name__,", algorithm=",result[result_num][0][3].__name__,", accuracy=",result[result_num][0][4],", precision=",result[result_num][0][5],", recall=",result[result_num][0][6],", f1=",result[result_num][0][7],", cutoff=",result[result_num][0][8])
                            result_num+=1
                            X = pd.concat([num_d, cat_d], axis=1)#changes to original              



result_good_model=[]
#Top 5 Results with High Accuracy            
result.sort(key=lambda x: x[0][4], reverse=True)
print("Top 5 Results with High Accuracy")
for i in range(5):
    print("k=",result[i][0][0],", used features=",result[i][1],", scaler=",result[i][0][1].__name__,", encoder=",result[i][0][2].__name__,", algorithm=",result[i][0][3].__name__,", accuracy=",result[i][0][4],", precision=",result[i][0][5],", recall=",result[i][0][6],", f1=",result[i][0][7],", cutoff=",result[i][0][8])
    result_good_model.append(result[i])

#Top 5 Results with High Precision
result.sort(key=lambda x: x[0][5], reverse=True)
 
print("Top 5 Results with High Precision")                   
for i in range(5):
    print("k=",result[i][0][0],", used features=",result[i][1],", scaler=",result[i][0][1].__name__,", encoder=",result[i][0][2].__name__,", algorithm=",result[i][0][3].__name__,", accuracy=",result[i][0][4],", precision=",result[i][0][5],", recall=",result[i][0][6],", f1=",result[i][0][7],", cutoff=",result[i][0][8])
    result_good_model.append(result[i])

#Top 5 Results with High Recall
result.sort(key=lambda x: x[0][6], reverse=True)   
print("Top 5 Results with High Recall")                 
for i in range(5):
    print("k=",result[i][0][0],", used features=",result[i][1],", scaler=",result[i][0][1].__name__,", encoder=",result[i][0][2].__name__,", algorithm=",result[i][0][3].__name__,", accuracy=",result[i][0][4],", precision=",result[i][0][5],", recall=",result[i][0][6],", f1=",result[i][0][7],", cutoff=",result[i][0][8])
    result_good_model.append(result[i])

#Top 5 Results with High f1
result.sort(key=lambda x: x[0][7], reverse=True)  
print("Top 5 Results with High f1")                  
for i in range(5):
    print("k=",result[i][0][0],", used features=",result[i][1],", scaler=",result[i][0][1].__name__,", encoder=",result[i][0][2].__name__,", algorithm=",result[i][0][3].__name__,", accuracy=",result[i][0][4],", precision=",result[i][0][5],", recall=",result[i][0][6],", f1=",result[i][0][7],", cutoff=",result[i][0][8])
    result_good_model.append(result[i])




#ROC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

model_accuracy=[]

#print ROC graph
for i in range(len(result_good_model)):
    #tpr(true positive rate), fpr(false positive rate), thresholds for use roc_curve function
    fpr, tpr, thresholds = roc_curve(result_good_model[i][0][9], result_good_model[i][0][10])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve %d'%i)
    plt.legend(loc="lower right")

    plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle='--', color='gray')
    plt.text(0.5, 0.5, 'AUC = %0.2f' % roc_auc, ha='center', va='center', color='gray')

    plt.show()
    
    

    model_accuracy.append([roc_auc,i])
    
#sort by AUC
model_accuracy.sort(key=lambda x: x[0], reverse=True) 


#print best model
print()
print("best model:")
print("k=",result[model_accuracy[0][1]][0][0],", used features=",result[model_accuracy[0][1]][1],", scaler=",result[model_accuracy[0][1]][0][1].__name__,", encoder=",result[model_accuracy[0][1]][0][2].__name__,", algorithm=",result[model_accuracy[0][1]][0][3].__name__,", accuracy=",result[model_accuracy[0][1]][0][4],", precision=",result[model_accuracy[0][1]][0][5],", recall=",result[model_accuracy[0][1]][0][6],", f1=",result[model_accuracy[0][1]][0][7],", cutoff=",result[model_accuracy[0][1]][0][8])
    
     
