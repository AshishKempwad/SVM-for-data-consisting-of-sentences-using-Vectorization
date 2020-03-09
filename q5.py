import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer


class AuthorClassifier:
    def _init_(self):
        self.clf=None
    
    def train(self,path):
        dataset=pd.read_csv(path)
        X=dataset.iloc[:,1]
        Y=dataset.iloc[:,2]
        X_data=X.to_numpy()
        Y_label=Y.to_numpy()
#         from sklearn.feature_extraction.text import CountVectorizer
#         vectorizer = CountVectorizer()
#         X1= vectorizer.fit_transform(X_data)
        vectorize = TfidfVectorizer(stop_words = 'english', lowercase = 'true')
        X1 = vectorize.fit_transform(X_data)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X1,Y_label, test_size=0.3,random_state=109)
        from sklearn import svm
        clf = svm.SVC(kernel='linear',C=1) # Linear Kernel
        clf.fit(X_train,y_train)
        self.clf=clf
        
    def predict(self,path):
        dataset=pd.read_csv(path)
        X=dataset.iloc[:,1]
#         print(X.shape)
        X_data=X.to_numpy()
#         from sklearn.feature_extraction.text import CountVectorizer
#         vectorizer = CountVectorizer()
#         X1= vectorizer.fit_transform(X_data)
        vectorize = TfidfVectorizer(stop_words = 'english', lowercase = 'true')
        X1 = vectorize.fit_transform(X_data)
        y_pred = self.clf.predict(X1)
#         print(y_pred.shape)
        return y_pred
        
        

            

            
