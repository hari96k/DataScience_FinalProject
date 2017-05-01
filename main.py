import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn import cross_validation
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt

complete_data = pd.read_csv("raw.csv")

#default necessary features
selected_features = ["CATAG6", "IRSEX", "NEWRACE2", "HEALTH2", "EDUCCAT2", "INCOME",
"ALCEVER", "MJEVER", "COCEVER", "BOOKED", "HEREVER", "CIGEVER", "CRKEVER"]
del_features = []
p = .3
#delete feature with legitimate skips of p% of total data
#ASSUMPTION: legitimate skip ID's are max and >= 90.
for col_idx in xrange(len(complete_data)):
	if list(complete_data)[col_idx] not in selected_features:
		legitimate_skip_freq = 0
		legitimate_skip_id = max(complete_data.iloc[:, col_idx]) #candidate
		num_samples = len(complete_data.iloc[:, col_idx])
		if legitimate_skip_id >= 90: #legitimate skip exits
			for row_idx in xrange(len(complete_data.iloc[:, col_idx])):
				if complete_data.iloc[row_idx, col_idx] == legitimate_skip_id:
					legitimate_skip_freq += 1
		if legitimate_skip_freq > int(len(complete_data.iloc[:, col_idx])*p):
			del_features.append(col_idx) #delete ith column

for col_idx in del_features:
	del complete_data[col_idx]


	







#K-fold on complete_data to create training and testing
