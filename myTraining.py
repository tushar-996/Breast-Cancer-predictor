import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
<<<<<<< HEAD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
=======
>>>>>>> 447f8cb7052d1bf3a626d19db087db50a2f0f4c5
import pickle



<<<<<<< HEAD
def data_split(data):

    data = data.iloc[:, 0:32]
    x =  data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']].to_numpy()
    y =  data[['diagnosis']].to_numpy()
    labeled = LabelEncoder()
    y = labeled.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    X_train, X_test, Y_train, Y_test = data_split(df)

    clf = LogisticRegression(random_state=0)
=======
def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    train, test = data_split(df, 0.2)
    X_train = train[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    X_test = test[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()

    Y_train = train[['infectionProb']].to_numpy().reshape(2000,)
    Y_test = test[['infectionProb']].to_numpy().reshape(499, )

    clf = LogisticRegression()
>>>>>>> 447f8cb7052d1bf3a626d19db087db50a2f0f4c5
    clf.fit(X_train, Y_train)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)
    file.close()

 
