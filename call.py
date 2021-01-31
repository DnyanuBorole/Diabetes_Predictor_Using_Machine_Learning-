import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import sys
import csv

def predict(testing,testdata1):

        df = pd.read_csv(testing)
        # print(df)
        x_train= np.array(df.drop(['Outcome'], 1))
        # print("X=", x_train)

        y_train = np.array(df['Outcome'])
        # print("y=", y_train)


        tf = pd.read_csv(testdata1)
        testdata = np.array(tf)
        testdata = testdata.reshape(len(testdata), -1)
        # print(testdata)


        clf1 = SVC()
        clf1.fit(x_train, y_train)  # build train model
        prediction1=clf1.predict(testdata)
        if prediction1[0] == 1 :
               return "POSITIVE"
        else:
               return "NEGATIVE"


        # clf = KNeighborsClassifier()
        # clf.fit(x_train, y_train)  # build train model
        # prediction1 = clf.predict(testdata)
        # print("prediction value=", prediction1[0])

        # return prediction1[0]





# predict("diabetes.csv","testdiabetes.csv")