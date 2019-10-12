# Importing the libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Importing the dataset

url = "breast-cancer-wisconsin.data"
cols = ["code","clump thick","Uniformity of Cell Size", 
        "Uniformity of Cell Shape","Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin" ,
        "Normal Nucleoli", "Mitoses", "target"]
df = pd.read_csv(url, names=cols)
df.replace('?',sys.float_info.min,inplace=True)
features = cols[1:9]
#print("len(features)",len(features))
X = df.loc[:, features].values
y = df.loc[:,['target']].values
#print(y)
# Splitting the dataset into the Training set and Test set



X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.1, random_state=11)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=11)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#principle component analysis

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
explained_variance=pca.explained_variance_ratio_




# Fitting SVM to the Training set

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)

trained_model=classifier.fit(X_train,y_train)
trained_model.fit(X_train,y_train )




y_pred = classifier.predict(X_test)



from sklearn.metrics import confusion_matrix
cm_SVM = confusion_matrix(y_test, y_pred)
print(cm_SVM)
print("Accuracy score of train SVM")
print(accuracy_score(y_train, trained_model.predict(X_train))*100)

print("Accuracy score of test SVM")
print(accuracy_score(y_test, y_pred)*100)