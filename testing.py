import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import xgboost as xgb

def pred(clf, test):

	if hasattr(clf, 'predict_proba'):
		return clf.predict_proba(test)[:, 1]
	return clf.predict(test)

dataset = pd.read_csv("raw_mod.csv")

training_data, testing_data = train_test_split(dataset, train_size=0.5)

training_data.reset_index(drop=True, inplace=True)
testing_data.reset_index(drop=True, inplace=True)

y_result = np.asarray(testing_data.BOOKED) #will test on these y values
y = np.asarray(training_data.BOOKED)

training_features = training_data.copy()
testing_features = testing_data.copy()

del training_features['BOOKED']
del testing_features['BOOKED']

complete_feature_matrix = pd.concat([training_features, testing_features])

transformed_data = complete_feature_matrix

transformed_training_data = transformed_data[:training_data.shape[0]]
transformed_testing_data = transformed_data[training_data.shape[0]:]

clf = xgb.XGBRegressor(objective="binary:logistic")

param_dist = {'n_estimators': range(500, 516, 1),
			'learning_rate': [0.02, 0.019, 0.0195],
		  'max_depth': [3, 4, 5]
		  }
hyperparam_search = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, n_iter = 10, 
						 scoring='roc_auc', error_score=0, verbose=3, n_jobs=-1)

hyperparam_search.fit(transformed_training_data, y)

best_model = hyperparam_search.best_estimator_
best_model.fit(transformed_training_data, y)
y_pred = pred(best_model, transformed_testing_data)
y_pred = (y_pred - y_pred.min())/(y_pred.max() - y_pred.min())
print "AUC: " + str(metrics.roc_auc_score(y_result, y_pred))

