# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:14:03 2022

@author: alexx
"""

# Classificazione:
# Prova singolo soggetto + singolo sensore
# Prova singolo soggetto + 3 sensori combinati
# Prova tutti soggetti + singolo sensore
# Prova tutti soggetti + 3 soggetti combinati
# LOSO: Un solo soggetto funge da test, gli altri insieme da training
# NB:
# Standardizzazione features (tolgo la media e divido per std): dopo divisione in training/test
# Inizialmente valuto l'accuratezza con tutte le features
# Analisi rilevanza features
#   Analisi features fatta solo sul training
#   Coppie di feature con alta correlazione: plot correlazione
# Provare classificatori diversi: modelli diversi hanno iperparametri diversi


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

np.random.seed(123) # To have reproducible results for different runs

file_path = "tab_tot.mat"
data = loadmat(file_path)
data = data.get('tab_tot')

columns = ["mean_imu", "std_imu", "f1_imu", "p1_imu", "tot_power_imu", "f625_imu", "p625_imu", "p1/tot_imu",
            "mean_ecg", "std_ecg", "f1_ecg", "p1_ecg", "tot_power_ecg", "f625_ecg", "p625_ecg", "p1/tot_ecg",
            "mean_ppg", "std_ppg", "f1_ppg", "p1_ppg", "f2_ppg", "p2_ppg", "tot_power_ppg", "f625_ppg", "p625_ppg", "p1/tot_ppg", "class"]

# from Numpy array to Pandas DataFrame
df = pd.DataFrame(data, columns = columns)

dic_activities = {1: "Walking", 2: "Drinking", 3: "Step", 4: "Sleeping", 5: "Sit-to-stand"}

for k, v in dic_activities.items(): 
    df['class'][df['class']==k] = v
    
# randomly shuffle the rows
df = df.sample(frac=1).reset_index(drop=True)

# print the first 10 samples    
df.head(5)    
    
plt.figure(figsize = (5,3.6))
df['class'].value_counts().plot(kind = 'bar');

# FEATURE EXPLORATION before split
import seaborn as sns
sns.pairplot(df, hue='class', vars = ['mean_imu', 'std_imu'], aspect = 1.2); # you can explore other features by specifying them in the parameter "vars"

plt.figure()
corr=df.corr()
corr.shape
sns.heatmap(corr, cmap='viridis')    
    
X = df.drop('class', axis = 1) # drop the last column of df
n_samples, n_features = X.shape

y = df['class'] # select the last column of df
n_classes = len((y.unique())) # number of classes 
y.unique()

# SPLIT
from sklearn.model_selection import train_test_split

random_state = 42 # numero: basta che sia intero per avere sempre le stesse righe destinate a training set (e test set)
'''the parameter "stratify" ensure that, after the split, each set contains approximately 
the same percentage of samples of each class as the original set'''
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, 
                                                    train_size = 0.7, stratify = y) # stratify per mantenere proporzionalità
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# FEATURE EXPLORATION after split
corr = np.array(corr)
thres = 0.75
sopra_soglia = []
for i in range(0, np.size(corr, axis=1)):
    for j in range(i, np.size(X_train, axis=1)):
        if i != j:
            if np.abs(corr[i,j]) > thres:
                sopra_soglia.append([i,j])

index_to_delete = []
correlat = corr
for cont in range(0, np.size(sopra_soglia, axis=0)):
    coppia = sopra_soglia[cont]
    el1 = coppia[0]
    el2 = coppia[1]
    corr1 = np.sum(correlat[: , el1]) / np.size(correlat, axis=1)
    corr2 = np.sum(correlat[: , el2]) / np.size(correlat, axis=1)
    if corr1 > corr2:
        index_to_delete.append(el1)
    else:
        index_to_delete.append(el2)
        
index_to_delete = np.array(index_to_delete) # index_to_delete = list(dict.fromkeys(index_to_delete))
index_to_delete = np.unique(index_to_delete)            # index_to_delete.sort()
correlat = np.delete(correlat, index_to_delete, 0)
correlat = np.delete(correlat, index_to_delete, 1)

# plt.figure()
# sns.heatmap(correlat, cmap='viridis')

X_train = X_train.drop(X.columns[index_to_delete], axis = 1)
X_test = X_test.drop(X.columns[index_to_delete], axis = 1)

from sklearn.tree import DecisionTreeClassifier            # 1. choose model class

# model = DecisionTreeClassifier(random_state=random_state)  # 2. instantiate model with default hyperprameters
# model.fit(X_train, y_train)                                # 3. fit model to data (model training)
# y_pred_train = model.predict(X_train)                      # 4. predict on training data

from sklearn.metrics import accuracy_score

# accuracy_train = accuracy_score(y_train, y_pred_train)
# print("The accuracy on training set is {0:.2f}%".format(accuracy_train * 100))

# y_pred_test = model.predict(X_test)                  # 4. predict on new data
# accuracy_test = accuracy_score(y_test, y_pred_test)
# print("The accuracy on test set is {0:.2f}%".format(accuracy_test * 100))



# # CLASSIFICATION WITH VALIDATION
# X_train_t, X_val, y_train_t, y_val = train_test_split(X_train, 
#                                                       y_train, 
#                                                       random_state = random_state,
#                                                       train_size = 0.7,
#                                                       stratify = y_train) 
# print("There are {} samples in the train_t dataset".format(X_train_t.shape[0]))
# print("There are {} samples in the val dataset".format(X_val.shape[0]))

# model = DecisionTreeClassifier(random_state = random_state)
# model.fit(X_train_t, y_train_t)
# fitted_max_depth = model.tree_.max_depth # max_depth of the tree fitted on the training set
# parameter_values = np.arange(1,fitted_max_depth+1) 
# parameter_values

# scores = []
# for par in parameter_values: 
#     estimator = DecisionTreeClassifier(max_depth = par,
#                                        random_state = random_state)
#     estimator.fit(X_train_t, y_train_t)
#     y_pred_val = estimator.predict(X_val)
#     score =  accuracy_score(y_val, y_pred_val) * 100 
#     scores.append(score)

# plt.figure(figsize = (6.5, 3))
# plt.plot(parameter_values, scores, '-o')
# plt.xlabel('max_depth')
# plt.ylabel('accuracy (%)')

# plt.title("Validation accuracy varying max_depth of tree")
# plt.show();

# top_par = parameter_values[np.argmax(scores)]
# print("Validation: The best accuracy is obtained with MAX_DEPTH={}".format(top_par))



# CLASSIFICATION WITH K-FOLD CROSS VALIDATION
from sklearn.model_selection import StratifiedKFold, cross_val_score

# For this part it is convenient to have X and y as numpy arrays instead of pandas objects
X_train = np.array(X_train)
y_train = np.array(y_train)
scores = []
fold = 0

# kf = StratifiedKFold(5, shuffle=True, random_state=random_state) # 1° elemento: n° fold

# for train_t, val in kf.split(X_train,y_train):  
#     fold += 1
#     X_train_t = X_train[train_t]
#     y_train_t = y_train[train_t]
#     X_val = X_train[val]
#     y_val = y_train[val]

#     estimator = DecisionTreeClassifier(random_state=random_state)
#     estimator.fit(X_train_t, y_train_t)
#     y_pred_val = estimator.predict(X_val)
#     score =  accuracy_score(y_val, y_pred_val) * 100 
#     print("Score of fold %d: %.2f%%" % (fold, score))
#     scores.append(score)
# print('Average score: %.2f%%' % (np.mean(scores)))

avg_scores = []

kf = StratifiedKFold(5, shuffle=True, random_state=random_state)

parameter_values = np.arange(1,14)
for par in parameter_values:
    scores = []
    for train_t, val in kf.split(X_train,y_train):
        X_train_t = X_train[train_t]
        y_train_t = y_train[train_t]
        X_val = X_train[val]
        y_val = y_train[val]
        # for i in range(0, np.size(X_train_t, axis=1)): # Standardizzazione colonna per colonna
        #     X_train_t[:,i] = (X_train_t[:,i]  - np.mean(X_train_t[:,i])) / np.std(X_train_t[:,i])
        #     X_val[:,i] = (X_val[:,i]  - np.mean(X_val[:,i])) / np.std(X_val[:,i])
        estimator = DecisionTreeClassifier(max_depth = par,
                                            random_state=random_state)  
        estimator.fit(X_train_t, y_train_t)
        y_pred_val = estimator.predict(X_val)
        score =  accuracy_score(y_val, y_pred_val) * 100 
        scores.append(score)
    avg_scores.append(np.mean(scores))

plt.figure(figsize = (6.5, 3))
plt.plot(parameter_values, avg_scores, '-o')
plt.xlabel('max_depth')
plt.ylabel('accuracy (%)')

plt.title("Cross-Validation accuracy varying max_depth of tree")
plt.show();

top_par = parameter_values[np.argmax(avg_scores)]
estimator = DecisionTreeClassifier(max_depth = top_par, random_state=random_state)
estimator.fit(pd.DataFrame(X_train), pd.DataFrame(y_train))

# for i in range(0, np.size(X_test, axis=1)): # Standardizzazione colonna per colonna
#     X_test[:,i]  = (X_test[:,i]  - np.mean(X_test[:,i])) / np.std(X_test[:,i])
    
y_pred = estimator.predict(np.array(X_test))
accuracy_cv = accuracy_score(np.array(y_test), y_pred) * 100
print("The accuracy on test set tuned with cross_validation is {:.1f}% with depth {}".format(accuracy_cv, top_par))


# CONFUSION MATRIX

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

list_activities = sorted(y_test.unique())

df_cm = pd.DataFrame(cm.astype('int'), index = [i for i in list_activities],
                  columns = [i for i in list_activities])
plt.figure(figsize = (6,4))
sns.set(font_scale=1)
sns.heatmap(df_cm, annot=True, annot_kws={"size": 9}, fmt='g', cmap = 'Blues', cbar = False);

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


