import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import metrics
#import xgboost as xgb
import matplotlib.pyplot as plt


def pred(clf, test):

	if hasattr(clf, 'predict_proba'):
		return clf.predict_proba(test)[:, 1]
	return clf.predict(test)

selected_features = ['IISNFAGE', 'IICDUAGE', 'CIGARYLU', 'IISLTYFU', 'CDCGMO', 'CIGTRY', 'IICGRAGE', 'IIMTHAGE', 'BMI2', 
'HPDRGTLK', 'SUMYR', 'CIGEVER', 'TRIMEST', 'DCIGMON', 'HPUSEALC', 'SNFEVER', 'COCYLU', 'IISLTAGE', 'IISTMAGE', 'IRSLTAGE', 
'IICGCRV', 'ADPBNUM', 'CDNOCGMO', 'IICGRYFU', 'YOPBNUM', 'ADPSDAYS', 'WHODASC3', 'TRANMLU', 'ECSMLU', 'CIGINCRS', 'ANALYLU',
'PREG2', 'IICD2YFU', 'CIGARMLU', 'RK5ALDLY', 'CDUFLAG', 'WTPOUND2', 'IIALCYFU', 'IRCGRAGE', 'MRJAGLST', 'INHYLU', 'IIINHAGE',
'IIMJAGE', 'IRALCAGE', 'IITRNAGE', 'CIGAVGM', 'IRMJAGE', 'MVIN5YR2', 'IISNFYFU', 'SMIPP_U', 'IRSNFAGE', 'IRLSDAGE',
'IRALCYFU', 'SUMAGE', 'MJAGE', 'HALMLU', 'ANALYFU', 'CHEWYLU', 'SNUFMLU', 'CGRAGLST', 'COCAGLST', 'NMVSOPT2', 'CPNPSYYR',
'IILSDAGE', 'IICHWAGE', 'IRMTHAGE', 'ALCYRTOT', 'IRCRKAGE', 'RKCOCREG', 'CHWAGLST', 'ALCEVER', 'IICGRFM', 'IIALCFY',
'CIGAGLST', 'ALCTRY', 'STIMYLU', 'SNFAGLST', 'CIGFNSMK', 'CPNPSYMN', 'SCHFELT', 'CIG30MEN', 'SEDOTHS2', 'CH30EST', 
'IEMAGE', 'IIHALAGE', 'II2HALFY', 'HPUSEDRG', 'SNUFYLU', 'CHEWMLU', 'IIANLAGE', 'SNUFYFU', 'CIGAVGD', 'ALCREC', 'SUMMON',
'IIANLYFU', 'CIGRNOUT', 'IMPYDAYS', 'ALCAVGM', 'IRCHWAGE', 'MRJYRBFR']

selected_features.append('BOOKED')

print "Started feature processing...."
#feature processing
dataset = pd.read_csv("raw.csv")

dataset = dataset[selected_features]

#change all response variables to binary
dataset['BOOKED'] = dataset['BOOKED'].map({1: 1, 2: 0, 3: 1, 85: 0, 94: 0, 97: 0, 98: 0})

dataset.to_csv("raw_mod.csv", index=False)

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

transformed_training_data = np.asarray(transformed_data[:training_data.shape[0]])
transformed_testing_data = np.asarray(transformed_data[training_data.shape[0]:])

print "Finished feature processing...."


# base_clfs = [xgb.XGBRegressor(n_estimators=590, max_depth=4, learning_rate=0.12, objective="binary:logistic"),
# 			xgb.XGBRegressor(n_estimators=514, max_depth=3, learning_rate=0.01925, objective="binary:logistic"),
# 			xgb.XGBRegressor(n_estimators=350, max_depth=5, learning_rate=0.03, objective="binary:logistic"),
# 			RandomForestClassifier(n_estimators=450, max_depth=6, n_jobs=-1, criterion='gini', max_features='auto')]

base_clfs = [RandomForestClassifier(n_estimators=450, max_depth=6, n_jobs=-1, criterion='gini', max_features='auto'), 
			  LogisticRegression(C=10, solver='sag'), LinearRegression(normalize=True), LassoCV()]

print "Started learning...."


#ExtraTreesClassifier(n_estimators=300, n_jobs=-1, criterion='entropy')

skf = StratifiedKFold(n_splits=4)
folds = list()

for (train, test) in skf.split(transformed_training_data, y):
	folds.append((train, test))

stack_train_data = np.zeros((training_data.shape[0], len(base_clfs)))
stack_test_data = np.zeros((testing_data.shape[0], len(base_clfs)))


for i, clf in enumerate(base_clfs):
	print "Analyzing classifier: ", (i + 1)
	stack_test_data_i = np.zeros((testing_data.shape[0], len(folds)))
	for j, (train, test) in enumerate(folds):
		print "Analyzing fold", j
		X_train, X_test = transformed_training_data[train], transformed_training_data[test]
		y_train, y_test = y[train], y[test]
		clf.fit(X_train, y_train)
		y_pred = pred(clf, X_test)
		stack_train_data[test, i] = y_pred
		stack_test_data_i[:, j] = pred(clf, transformed_testing_data)
	stack_test_data[:, i] = stack_test_data_i.mean(1)

print "Stacking has started...."

meta_clf = LogisticRegression()
meta_clf.fit(stack_train_data, y)
y_pred = meta_clf.predict_proba(stack_test_data)[:, 1]
y_pred_train = meta_clf.predict_proba(stack_train_data)[:, 1]

print "Result normalization has started...."

y_pred = (y_pred - y_pred.min())/(y_pred.max() - y_pred.min())

y_pred_train = (y_pred_train - y_pred_train.min())/(y_pred_train.max() - y_pred_train.min())

print "AUC: " + str(metrics.roc_auc_score(y_result, y_pred))

# meta_clf = LogisticRegression()
# meta_clf.fit(transformed_training_data, y)
# y_pred = meta_clf.predict_proba(transformed_testing_data)[:, 1]
# y_pred = (y_pred - y_pred.min())/(y_pred.max() - y_pred.min())
# print "AUC: " + str(metrics.roc_auc_score(y_result, y_pred))