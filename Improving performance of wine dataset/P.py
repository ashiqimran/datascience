import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


#read from the csv file and return a Pandas DataFrame.
data = pd.read_csv('wine.csv')

#Randomized data
random_data = data.sample(frac = 1.0).reset_index(drop=True)

#Class Label : Y
Y = data.quality

#Feature Columns: X
X = data.drop(['quality','residual sugar','total sulfur dioxide','citric acid','fixed acidity'],axis=1)

#Spliting data 0.75 for training, 0.25 for testing
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, train_size=0.75, test_size=0.25)
RF = RandomForestClassifier(n_estimators=25, n_jobs=5)
RF.fit(X_train,Y_train)
Y_Predict = RF.predict(X_test)
print("Accuracy of Model 1: {:.2f}".format(RF.score(X_test, Y_test)))

print("\nConfusion matrix:")
print(pd.crosstab(Y_test, Y_Predict, rownames=['True'], colnames=['Predicted'], margins=True))
print("\n")
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

pipeline = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=100, n_jobs=5))

param_grid = { 'randomforestclassifier__max_features': ['auto', 'sqrt',7],
                'randomforestclassifier__max_depth': [None,15,10]}


#10 fold stratified cross validation
kf = StratifiedKFold(n_splits=10)
Y = random_data.quality
X = random_data.drop(['quality','residual sugar','total sulfur dioxide','citric acid','fixed acidity'],axis=1)

accuracy = 0.0
foldNo = 0
for train_index, test_index in kf.split(X,Y):
    foldNo += 1
    X_train, X_test, Y_train, Y_test = X.ix[train_index], X.ix[test_index], Y.ix[train_index], Y.ix[test_index]
    clf = GridSearchCV(pipeline,param_grid, cv = kf)
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    #Calculating Accuracy
    accuracy_scores = clf.score(X_test,Y_test)
    print("Accuracy on Fold: {} = {:.2f}".format(foldNo,accuracy_scores))
    accuracy += accuracy_scores

print("\nAverage Accuracy on Model 2 {:.2f}".format(accuracy/10))
