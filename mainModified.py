import numpy as np
import pandas as pd

print("Started reading file")
complete_data = pd.read_csv("raw.csv")
print("Initial shape "+ str(complete_data.shape))
print("Finished reading file")
#default necessary features
selected_features = ["CATAG6", "IRSEX", "NEWRACE2", "HEALTH2", "EDUCCAT2", "INCOME",
"ALCEVER", "MJEVER", "COCEVER", "BOOKED", "HEREVER", "CIGEVER", "CRKEVER"]
del_features = []
p = .3
print("Feature selection starting")
#delete feature with legitimate skips of p% of total data
#ASSUMPTION: legitimate skip ID's are max and >= 90.
columnMax=complete_data.max(axis=0)
num_samples = len(complete_data.index)
for col_name in complete_data:
	print(col_name)
	if col_name not in selected_features:
		legitimate_skip_freq = 0
		legitimate_skip_id = complete_data[col_name].max() #candidate
		if legitimate_skip_id >= 90: #legitimate skip exits
			legitimate_skip_freq=len(complete_data[complete_data[col_name]==legitimate_skip_id])
		if legitimate_skip_freq > int(num_samples*p):
			del_features.append(col_name) #delete ith column
complete_data.drop(del_features,axis=1)
print("Feature selection ended")
complete_data.to_csv("subset2.csv")
print("Final shape " + str(complete_data.shape))
print("Done writing file")



	







#K-fold on complete_data to create training and testing
