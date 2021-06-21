import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import librosa
import numpy as np
import pandas as pd

from MFCC import *

def get_lables(path):
    return path.split("\\")[7]

# signal,sr = librosa.load("C:\\Users\\Admin\\Desktop\\APP\\thu_uyen\\TrainingData\\Đơn tấu\\Guitar Classic\\[gac][cla]0526__2.wav",sr=44100)
#
# obj = Mfcc()
# tmp = np.array(obj.mfcc(signal,sr))
# print(tmp.shape)
# print(np.mean(tmp,axis=1).shape)
# df = pd.read_csv("datas_2.csv")
#
# res= []
# for path in df.values:
#     tmp = []
#     signal, sr = librosa.load(path[0],sr=44100)
#     tmp.append(get_lables(path[0]))
#     tmp_2 = np.array(obj.mfcc(signal, sr))
#     tmp_2 = np.mean(tmp_2,axis=1)
#     for i in tmp_2:
#         tmp.append(i)
#     res.append(tmp)
#
# df = pd.DataFrame(data=res,columns=["label","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"])
# df.to_csv("mfcc.csv",index=False)
import pandas as pd
import pickle

df = pd.read_csv("mfcc.csv")
from sklearn.model_selection import train_test_split
X = df.drop(["label"],axis=1).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def KNN(X_train,y_train,X_test,y_test):
  from sklearn.metrics import confusion_matrix
  neigh = KNeighborsClassifier(n_neighbors=3)
  neigh.fit(X_train, y_train)
  y_pred = neigh.predict(X_test)
  print(confusion_matrix(y_test, y_pred))
  print("acc KNN = {} %".format(accuracy_score(y_test, y_pred) * 100))
  pickle.dump(neigh, open("KNNer.bin", 'wb'))
  return accuracy_score(y_test, y_pred)

def random_forest(X_train,y_train,X_test,y_test):
  clf = RandomForestClassifier(n_estimators=300)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print("acc random_forest = {} %".format(accuracy_score(y_test, y_pred) * 100))
  return accuracy_score(y_test, y_pred)

KNN(X_train,y_train,X_test,y_test)
random_forest(X_train,y_train,X_test,y_test)
