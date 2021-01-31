import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def predict(diabdata,testdia):
        df = pd.read_csv(diabdata)
        print(df)
        x_train= np.array(df.drop(['Outcome'], 1))
        # print("X=", x_train)

        y_train = np.array(df['Outcome'])
        # print("y=", y_train)


        tf = pd.read_csv(testdia)
        testdata = np.array(tf)
        testdata = testdata.reshape(len(testdata), -1)
        # print(testdata)


        clf1 = SVC()
        clf1.fit(x_train, y_train)  # build train model
        prediction1=clf1.predict(testdata)
        print("prediction value=",prediction1[0])

        if prediction1[0] == 1:
            print("POSITIVE")
        else:
            print("NEGATIVE")

        clf = KNeighborsClassifier()
        clf.fit(x_train, y_train)  # build train model
        prediction1 = clf.predict(testdata)
        print("prediction value=", prediction1[0])

        # if prediction1[0] == 1:
        #         print("POSITIVE")
        # else:
        #         print("NEGATIVE")





        # clf = LogisticRegression()
        # clf.fit(x_train,y_train)
        # prediction1 = clf.predict(testdata)
        # print("prediction value=", prediction1[0])
        #
        # if prediction1[0] == 1:
        #         print("POSITIVE")
        # else:
        #         print("NEGATIVE")










predict("diabetes.csv","testdiabetes.csv")