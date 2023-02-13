# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 20:19:31 2023

@author: alexx
"""
# Classification - LOSO : Un solo soggetto funge da test, gli altri insieme da training
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import seaborn as sns
# DIFFERENT CLASSIFIERS
from sklearn.tree import DecisionTreeClassifier  # 1. choose model class
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Le figure di ogni soggetto non sono importanti (credo), commentatele per non avere 100 figure

np.random.seed(123) # To have reproducible results for different runs

file_path = "tab_loso.mat"
data = loadmat(file_path)
data = data.get('tab_loso')

columns = ["mean_imu", "std_imu", "f1_imu", "p1_imu", "tot_power_imu", "f625_imu", "p625_imu", "p1/tot_imu",
            "mean_ecg", "std_ecg", "f1_ecg", "p1_ecg", "tot_power_ecg", "f625_ecg", "p625_ecg", "p1/tot_ecg",
            "mean_ppg", "std_ppg", "f1_ppg", "p1_ppg", "f2_ppg", "p2_ppg", "tot_power_ppg", "f625_ppg", "p625_ppg", "p1/tot_ppg", "class", "subject"]

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
corr=df.drop('subject', axis=1).corr()

random_state = 42
algoritmo = 5
subj_scores = []
subject_accuracy = []
for subj in range(1, 16):
    train_set = df[df['subject'] != subj];
    X_train = train_set.drop(['class','subject'], axis=1) # drop the last 2 columns of df
    y_train = train_set['class'] # select the last column of df
    test_set = df[df['subject'] == subj];
    X_test = test_set.drop(['class','subject'], axis=1) # drop the last 2 columns of df
    y_test = test_set['class'] # select the last column of df

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
                
    index_to_delete = np.array(index_to_delete) 
    index_to_delete = np.unique(index_to_delete)
    correlat = np.delete(correlat, index_to_delete, 0)
    correlat = np.delete(correlat, index_to_delete, 1)
                
    # plt.figure()
    # sns.heatmap(correlat, cmap='viridis')
        
    X_train = X_train.drop(X_train.columns[index_to_delete], axis = 1)
    X_test = X_test.drop(X_test.columns[index_to_delete], axis = 1)    

    # CLASSIFICATION WITH K-FOLD CROSS VALIDATION
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    
    # For this part it is convenient to have X and y as numpy arrays instead of pandas objects
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)    
    
    avg_scores = []    
    kf = StratifiedKFold(5, shuffle=True, random_state=random_state)
    if algoritmo == 1:
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
                estimator = DecisionTreeClassifier(max_depth = par, random_state=random_state)  
                estimator.fit(X_train_t, y_train_t)
                y_pred_val = estimator.predict(X_val)
                score =  accuracy_score(y_val, y_pred_val) * 100 
                scores.append(score) # Accuratezza per ogni diverso split in train/test
            avg_scores.append(np.mean(scores)) # Accuratezza media (1 per ogni grado)
        # subj_scores.append([np.argmax(avg_scores), np.amax(avg_scores)]) # Grado del modello per cui si ottiene la più alta accuratezza (1 per ogni soggetto)

        top_par = parameter_values[np.argmax(avg_scores)]
        estimator = DecisionTreeClassifier(max_depth = top_par, random_state=random_state)
        estimator.fit(pd.DataFrame(X_train), pd.DataFrame(y_train))
        
        # for i in range(0, np.size(X_test, axis=1)): # Standardizzazione colonna per colonna
        #     X_test[:,i]  = (X_test[:,i]  - np.mean(X_test[:,i])) / np.std(X_test[:,i])
        y_pred = estimator.predict(np.array(X_test))
        accuracy_cv = accuracy_score(np.array(y_test), y_pred) * 100
        subject_accuracy.append([subj, np.amax(avg_scores), top_par, accuracy_cv])
        # Soggetto - Max Accuratezza fase Validazione - Grado del modello per Max Accuratezza - Max Accuratezza fase Test
        
        plt.figure(figsize = (6.5, 3))
        plt.plot(parameter_values, avg_scores, '-o', color='red')
        plt.stem(top_par, accuracy_cv)
        plt.legend(['Validation Accuracy', 'Test Accuracy'])
        plt.xlabel('max_depth')
        plt.ylabel('accuracy (%)')
        
        plt.title("Accuracy varying max_depth of tree - Decision Tree Classifier - Subject: {}".format(subj))
        plt.show();
    
        print("The accuracy on test set tuned with cross_validation is {:.1f}% with depth {}".format(accuracy_cv, top_par))

    if algoritmo == 2:
        shrinkage_values = np.arange(0, 1.1, 0.1)
        for shr in shrinkage_values:
            scores = []
            for train_t, val in kf.split(X_train,y_train):
                X_train_t = X_train[train_t]
                y_train_t = y_train[train_t]
                X_val = X_train[val]
                y_val = y_train[val]
                # for i in range(0, np.size(X_train_t, axis=1)): # Standardizzazione colonna per colonna
                #     X_train_t[:,i] = (X_train_t[:,i]  - np.mean(X_train_t[:,i])) / np.std(X_train_t[:,i])
                #     X_val[:,i] = (X_val[:,i]  - np.mean(X_val[:,i])) / np.std(X_val[:,i])
                estimator = LinearDiscriminantAnalysis(solver='lsqr', shrinkage = shr)  
                estimator.fit(X_train_t, y_train_t)
                y_pred_val = estimator.predict(X_val)
                score =  accuracy_score(y_val, y_pred_val) * 100 
                scores.append(score)
            avg_scores.append(np.mean(scores))

        top_par = shrinkage_values[np.argmax(avg_scores)]
        estimator = LinearDiscriminantAnalysis(solver='lsqr', shrinkage = shr)
        estimator.fit(pd.DataFrame(X_train), pd.DataFrame(y_train))        
        
        # for i in range(0, np.size(X_test, axis=1)): # Standardizzazione colonna per colonna
        #     X_test[:,i]  = (X_test[:,i]  - np.mean(X_test[:,i])) / np.std(X_test[:,i])
        y_pred = estimator.predict(np.array(X_test))
        accuracy_cv = accuracy_score(np.array(y_test), y_pred) * 100
        subject_accuracy.append([subj, np.amax(avg_scores), top_par, accuracy_cv])
        # Soggetto - Max Accuratezza fase Validazione - Shrinkage parameter per Max Accuratezza - Max Accuratezza fase Test
        
        plt.figure(figsize = (6.5, 3))
        plt.plot(shrinkage_values, avg_scores, '-o', color='red')
        plt.stem(top_par, accuracy_cv)
        plt.legend(['Validation Accuracy', 'Test Accuracy'])
        plt.xlabel('shrinkage value')
        plt.ylabel('accuracy (%)')

        plt.title("Accuracy varying shrinkage value - Linear Discriminant Analysis - Subject: {}".format(subj))
        plt.show();
        
        print("The accuracy on test set tuned with cross_validation is {:.1f}% with shrinkage parameter {}".format(accuracy_cv, top_par))
        
    if algoritmo == 3:
        neighbor_values=np.arange(1,10)
        for n in neighbor_values:
            scores = []
            for train_t, val in kf.split(X_train,y_train):
                X_train_t = X_train[train_t]
                y_train_t = y_train[train_t]
                X_val = X_train[val]
                y_val = y_train[val]
                # for i in range(0, np.size(X_train_t, axis=1)): # Standardizzazione colonna per colonna
                #     X_train_t[:,i] = (X_train_t[:,i]  - np.mean(X_train_t[:,i])) / np.std(X_train_t[:,i])
                #     X_val[:,i] = (X_val[:,i]  - np.mean(X_val[:,i])) / np.std(X_val[:,i])
                estimator = KNeighborsClassifier(n_neighbors=n)
                estimator.fit(X_train_t, y_train_t)
                y_pred_val = estimator.predict(X_val)
                score =  accuracy_score(y_val, y_pred_val) * 100 
                scores.append(score)
            avg_scores.append(np.mean(scores))
        
        top_par = neighbor_values[np.argmax(avg_scores)]
        estimator = KNeighborsClassifier(n_neighbors=3)
        estimator.fit(pd.DataFrame(X_train), pd.DataFrame(y_train))
        
        # for i in range(0, np.size(X_test, axis=1)): # Standardizzazione colonna per colonna
        #     X_test[:,i]  = (X_test[:,i]  - np.mean(X_test[:,i])) / np.std(X_test[:,i])
        y_pred = estimator.predict(np.array(X_test))
        accuracy_cv = accuracy_score(np.array(y_test), y_pred) * 100
        subject_accuracy.append([subj, np.amax(avg_scores), top_par, accuracy_cv])
        # Soggetto - Max Accuratezza fase Validazione - N° neighbors per Max Accuratezza - Max Accuratezza fase Test

        plt.figure(figsize = (6.5, 3))
        plt.plot(neighbor_values, avg_scores, '-o', color='red')
        plt.stem(top_par, accuracy_cv)
        plt.legend(['Validation Accuracy', 'Test Accuracy'])
        plt.xlabel('neighbor values')
        plt.ylabel('accuracy (%)')

        plt.title("Accuracy varying number of neighbors - K-Neighbor Classifier - Subject: {}".format(subj))
        plt.show();
        
        print("The accuracy on test set tuned with cross_validation is {:.1f}% with n° of neighbor: {}".format(accuracy_cv, top_par))

    if algoritmo == 4:
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
                estimator = GaussianNB()
                estimator.fit(X_train_t, y_train_t)
                y_pred_val = estimator.predict(X_val)
                score =  accuracy_score(y_val, y_pred_val) * 100 
                scores.append(score)
            avg_scores.append(np.mean(scores))

        estimator = GaussianNB()
        estimator.fit(pd.DataFrame(X_train), pd.DataFrame(y_train))   
        
        # for i in range(0, np.size(X_test, axis=1)): # Standardizzazione colonna per colonna
        #     X_test[:,i]  = (X_test[:,i]  - np.mean(X_test[:,i])) / np.std(X_test[:,i])
        y_pred = estimator.predict(np.array(X_test))
        accuracy_cv = accuracy_score(np.array(y_test), y_pred) * 100
        subject_accuracy.append([subj, np.amax(avg_scores), 0, accuracy_cv])
        # Soggetto - Max Accuratezza fase Validazione - / - Max Accuratezza fase Test

        plt.figure(figsize = (6.5, 3))
        plt.plot(1, avg_scores[1], '-o', color='red')
        plt.stem(1, accuracy_cv)
        plt.legend(['Validation Accuracy', 'Test Accuracy'])
        plt.ylabel('accuracy (%)')

        plt.title("Accuracy - Gaussian Naive Bayes Classifier - Subject: {}".format(subj))
        plt.show();
        
    if algoritmo == 5:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
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
                estimator = make_pipeline(StandardScaler(par), SVC(gamma='auto')) 
                estimator.fit(X_train_t, y_train_t)
                #Pipeline(steps=[('standardscaler', StandardScaler()),('svc', SVC(gamma='auto'))])
                y_pred_val = estimator.predict(X_val)
                score =  accuracy_score(y_val, y_pred_val) * 100 
                scores.append(score)
            avg_scores.append(np.mean(scores))

        top_par = parameter_values[np.argmax(avg_scores)]
        estimator = make_pipeline(StandardScaler(par), SVC(gamma='auto'))
        estimator.fit(X_train_t, y_train_t)  
        
        # for i in range(0, np.size(X_test, axis=1)): # Standardizzazione colonna per colonna
        #     X_test[:,i]  = (X_test[:,i]  - np.mean(X_test[:,i])) / np.std(X_test[:,i])
        y_pred = estimator.predict(np.array(X_test))
        accuracy_cv = accuracy_score(np.array(y_test), y_pred) * 100
        subject_accuracy.append([subj, np.amax(avg_scores), top_par, accuracy_cv])
        # Soggetto - Max Accuratezza fase Validazione - Grado del modello per Max Accuratezza - Max Accuratezza fase Test
        
        plt.figure(figsize = (6.5, 3))
        plt.plot(1, avg_scores[1], '-o', color='red')
        plt.stem(1, accuracy_cv)
        plt.legend(['Validation Accuracy', 'Test Accuracy'])
        plt.ylabel('accuracy (%)')

        plt.title("Accuracy varying max_depth of tree - Support Vector Classifier - Subject: {}".format(subj))
        plt.show();
        
        print("The accuracy on test set tuned with cross_validation is {:.1f}% with parameter: {}".format(accuracy_cv, top_par))
        
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
    
subject_accuracy = np.array(subject_accuracy)
subj_valid_accuracy = subject_accuracy[:, 1]
subj_test_accuracy = subject_accuracy[:, 3]
plt.figure(figsize = (6.5, 3))
plt.plot(range(1, 16), subj_valid_accuracy, '-o', color='red')
plt.plot(range(1, 16), subj_test_accuracy, '-o', color='blue')
plt.legend(['Validation Accuracy', 'Test Accuracy'])
plt.xlabel('Subject')
plt.ylabel('Accuracy (%)')
plt.title('LOSO Accuracy for each subject')
    
    
    
    
    
    
    
    
    
    
    
    
    
    