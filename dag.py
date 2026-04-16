import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
#dagshub.init('https://dagshub.com/ramashishmaurya/Mental-Health-Prediction.mlflow')

import dagshub
dagshub.init(repo_owner='ramashishmaurya', repo_name='Mental-Health-Prediction', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/ramashishmaurya/Mental-Health-Prediction.mlflow')



from sklearn.datasets import load_wine

data = load_wine()
x = data.data
y=data.target

x_train , x_test ,y_train , y_test = train_test_split(x,y ,test_size=0.2,random_state=42)


import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

n_estimators = 10 
max_depth= 5 

# Set experiment
mlflow.set_experiment("wine_datasets")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Log params
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)

    # Log metric (ONLY scalar)
    mlflow.log_metric('accuracy', acc)

    # ✅ Save confusion matrix as image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # ✅ Log model correctly
    mlflow.sklearn.log_model(model, "RandomForest")



#322222222222222222222222222222222222222222222222
    
from sklearn.tree import DecisionTreeClassifier

with mlflow.start_run(run_name='DecisionTree'):
    model =DecisionTreeClassifier(max_depth=7 ,criterion='gini')
    model.fit(x_train ,y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test , y_pred)
    conf = confusion_matrix(y_test ,y_pred)

    mlflow.log_param('max_depth' ,7)
    mlflow.log_param('criterion','gini')

    mlflow.log_metric('accuracy' ,acc)

    mlflow.sklearn.log_model(model , 'DecisionTreeClssifier')



##333333333333333333333333333333333333333333333333333333
    

from sklearn.ensemble import BaggingClassifier



with mlflow.start_run(run_name='baggingclassifier'):
    model = BaggingClassifier()
    model.fit(x_train ,y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test ,y_pred)

    mlflow.log_metric('accuracy' ,acc)

    mlflow.sklearn.log_model(model ,'BaggingClassifier')


##################4444444444444444444444444


from sklearn.ensemble import AdaBoostClassifier

with mlflow.start_run(run_name='Adaboostclassifier'):
    model = AdaBoostClassifier()
    model.fit(x_train ,y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test ,y_pred)

    mlflow.log_metric('accuracy' ,acc)

    mlflow.sklearn.log_model(model ,'Adaboostclassifier')
    
