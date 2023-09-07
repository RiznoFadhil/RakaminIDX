#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[108]:


import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score,roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)


# # Dataset import and preparation

# In[109]:


csv_file_path = "loan_data_2007_2014.csv"
ftdf = pd.read_csv(csv_file_path, index_col=0)
ftdf.head(1)
ftdf.shape


# In[110]:


columns = list(ftdf.columns)
columns.sort()

ftdf = ftdf[columns]
ftdf.head(1)


# # EDA data numerik

# In[111]:


ftdf.info()


# In[112]:


colnames = ftdf.columns

NA_pct = []

for col in colnames:
    
    NA_pct.append([round((ftdf[col].isna().sum() / ftdf.shape[0]) * 100, 2), ftdf[col].nunique(), ftdf[col].unique()[:5]])
    
NA_pct = pd.DataFrame(NA_pct, index = [colnames], columns = ['Persentase NA', 'Banyak Nilai Unik', "Sampel Nilai Unik"]).reset_index()

NA_pct.rename(columns={"level_0": "Fitur"}, inplace=True)

NA_pct


# # Drop fitur dengan persentase NA > 20% dan fitur member_id dan id

# In[113]:


deleted_col = list(NA_pct[NA_pct['Persentase NA'] > 20]['Fitur']) + ['member_id', 'id']
deleted_col


# In[114]:


ftdf.drop(labels = deleted_col, axis = 1, inplace = True)
ftdf.shape


# # Membuat label dari fitur loan_status

# In[115]:


loan_status = ftdf['loan_status'].value_counts()
loan_status


# # Label akan terdiri dari 2 jenis, good(1) dan bad(0)

# In[116]:


good = ['Current','Fully Paid']
bad = ['Charged Off', 'Late (31-120 days)', 'In Grace Period', 'Does not meet the credit policy. Status:Fully Paid', 'Late (16-30 days)' 
       'Default', 'Does not meet the credit policy. Status:Charged Off']
def add_Label(values):  
    if values in good:
        return 1
    return 0


# In[118]:


new_ftdf = ftdf[ftdf['loan_status'].isin(good + bad)].copy()
new_ftdf['loan_status'] = new_ftdf['loan_status'].apply(add_Label)
new_ftdf.head(1)


# In[119]:


new_ftdf.shape


# In[120]:


new_ftdf['loan_status'].value_counts()


# # Korelasi

# In[121]:


correl = (new_ftdf.select_dtypes(exclude=object)
                .corr()
                .dropna(how="all", axis=0)
                .dropna(how="all", axis=1)
)
correl


# In[122]:


correl['loan_status'].sort_values(ascending = False)


# In[123]:


min_correl, max_correl = 0.1, 0.99

pos_correl = (correl > min_correl) & (correl < max_correl)
neg_correl = (correl > -max_correl) & (correl < -min_correl)

# Nilai mutlak korelasi
filter_correl = correl[pos_correl | neg_correl]


# In[124]:


fiturHC = filter_correl[(filter_correl >= 0.5) & (filter_correl <= 0.8)]
fiturLC = filter_correl[(filter_correl >= 0.2) & (filter_correl < 0.5)]


# In[125]:


selected_fiturHC = fiturHC.columns[fiturHC.notnull().any()].tolist()
selected_fiturHC


# In[126]:


selected_fiturLC = fiturLC.columns[fiturLC.notnull().any()].tolist()
selected_fiturLC


# In[127]:


fiturnumerik = set(selected_fiturHC).union(set(selected_fiturLC))
fiturnumerik = list(fiturnumerik)
fiturnumerik.sort()
fiturnumerik, len(fiturnumerik)


# In[128]:


new_ftdf[list(fiturnumerik)]


# In[129]:


y = new_ftdf['loan_status']


# In[130]:


new_ftdf.drop('loan_status', inplace = True, axis = 1)


# In[131]:


fiturnumerik.remove('loan_status')


# In[132]:


plt.figure(figsize=(24,28))
i = 1
for colname in fiturnumerik:
    plt.subplot(7, 3, i)
    sns.boxplot(x=new_ftdf[list(fiturnumerik)][colname])
    plt.title(colname, fontsize=20)
    plt.xlabel(' ')
    plt.tight_layout()
    i += 1


# In[133]:


new_ftdf[fiturnumerik].isna().sum()


# # Imputasi NA dan Data Cleaning

# In[134]:


# IQR = Q3 - Q1
# Lower Limit = Q1 - 1.5 * IQR
# Upper Limit = Q3 + 1.5 * IQR
for colnames in fiturnumerik:

    IQR = new_ftdf[colnames].describe()[6] - new_ftdf[colnames].describe()[4]
    batas_bawah = new_ftdf[colnames].describe()[4] - (1.5 * IQR)
    batas_atas = new_ftdf[colnames].describe()[6] + (1.5 * IQR)
    
    new_ftdf.loc[new_ftdf[colnames] >= batas_atas, colnames] = batas_atas
    new_ftdf.loc[new_ftdf[colnames] <= batas_bawah, colnames] = batas_bawah
    
    # Fill NA
    new_ftdf[colnames].fillna(value = new_ftdf[colnames].describe()[5], inplace = True)
    


# In[135]:


plt.figure(figsize=(24,28))
i = 1
for colname in fiturnumerik:
    plt.subplot(7, 3, i)
    sns.boxplot(x=new_ftdf[list(fiturnumerik)][colname])
    plt.title(colname, fontsize=20)
    plt.xlabel(' ')
    plt.tight_layout()
    i += 1


# # EDA data kategorik

# In[136]:


fitur_kategorik = list(new_ftdf.select_dtypes(include='O').columns)
fitur_kategorik.sort()
fitur_kategorik, len(fitur_kategorik)


# In[137]:


columns_info = []

for colnames in new_ftdf[fitur_kategorik].columns:
    info = [colnames,
            new_ftdf[colnames].isna().sum(),
            f"{round((new_ftdf[colnames].isna().sum() / new_ftdf[colnames].shape[0]) * 100, 2)}%",
            new_ftdf[colnames].nunique(),
            new_ftdf[colnames].value_counts(),
            new_ftdf[colnames].unique(),
           ]
    
    columns_info.append(info)


# In[138]:


info_kategorik = pd.DataFrame(data = columns_info, 
                                  columns = ['col names', 'null values', '% null values', 'n unqiue', 'val count','unique'])
info_kategorik


# # Peubah kategorik yang akan dipilih adalah initial_list_status, pymnt_plan, term, dan verification_status

# In[139]:


fiturkategorik_pilih = ["initial_list_status", "pymnt_plan", "term", "verification_status"]


# In[140]:


fitur = fiturkategorik_pilih + fiturnumerik
fitur.sort()
fitur, len(fitur)


# In[141]:


for colname in new_ftdf:
    if colname not in fitur:
        new_ftdf.drop(labels=[colname], inplace = True, axis = 1)


# In[143]:


new_ftdf.shape


# In[144]:


new_ftdf.info()


# In[145]:


new_ftdf.isna().sum()


# In[146]:


new_ftdf['loan_status'] = y


# In[147]:


new_ftdf.head(5)


# In[148]:


new_ftdf['loan_status'].value_counts()


# # Hot one decoding untuk fitur kategorik

# In[149]:


y = new_ftdf['loan_status'] 
X = new_ftdf.drop(labels = ['loan_status'], axis = 1)


# In[150]:


X = pd.get_dummies(X, columns = ["initial_list_status", "pymnt_plan", "term", "verification_status"], prefix_sep='__')


# In[152]:


X.head(5)


# # Spilt dataset menjadi train dan test data : 80:20

# In[153]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify = y)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Feature Scaling

# In[154]:


scaler = MinMaxScaler()
scaler.fit(X_train)

X_trainScale = scaler.transform(X_train)
X_testScale = scaler.transform(X_test)


# # SMOTE untuk masalah imbalance data
# 
# 

# In[156]:


sm = SMOTE(random_state=25)

sm.fit(X_trainScale, y_train)

X_smote, y_smote = sm.fit_resample(X_trainScale, y_train)

X_smote.shape, X_trainScale.shape, y_smote.shape, y_train.shape


# # Regresi Logistik

# In[157]:


logistik = LogisticRegression(random_state = 25, max_iter=500, solver="sag", class_weight="balanced", n_jobs=-1) 
logistik.fit(X_smote, y_smote)


# In[158]:


logistik.score(X_smote, y_smote), logistik.score(X_testScale, y_test)


# # Evaluasi Model (Accuracy, Recall, Precision, AUC)

# In[159]:


y_pred_proba = logistik.predict_proba(X_testScale)
y_pred = logistik.predict(X_testScale)


# In[160]:


accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
AUC = roc_auc_score(y_test, y_pred_proba[:, 1])


# In[161]:


print(accuracy)
print(recall)
print(precision)
print(AUC)


# In[163]:


report = classification_report(y_true = y_test, y_pred = logistik.predict(X_testScale))
print(report)


# # Confusion Matrix 

# In[164]:


conf_matrix = confusion_matrix(y_true = y_test, y_pred = logistik.predict(X_testScale ))


# In[165]:


plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="g")
plt.show()

