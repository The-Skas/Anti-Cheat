from data_helpers import *

from hmmlearn.hmm import GaussianHMM

import pandas as pd
import sys
import pdb
from sklearn.externals import joblib


player_id = sys.argv[3] if len(sys.argv) >= 4 else 0

dfplayer = clean_data_to_numbers(sys.argv[1], player_id=player_id)

## Get all Non-Null ViewDiffs and Aimbotdist
## Issue with this is the fucking up the time series...
dfplayer = dfplayer[(dfplayer.TrueViewDiff.notnull()) & (dfplayer.AimbotDist.notnull())]

# Split rounds into multiple dataframes.
rounds = dfplayer.Round.unique()
dictrounds = {elem : pd.DataFrame for elem in rounds}
for key in dictrounds.keys():
	_df = dfplayer[:][dfplayer.Round == key]
	_length = len(_df)
	dictrounds[key] = (_df , _length)

# Train hmm
X = np.reshape(np.array([]),(0,1))

for key in dictrounds.keys():
	# Get at index 0 since its tupled of df
	S = np.column_stack([dictrounds[key][0].TrueViewDiff.values] ) #, dictrounds[key][0].AimbotDist.values]) 
	X = np.vstack([X,S])

lengths = [ dictrounds[key][1] for key in dictrounds.keys()]

model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000).fit(X, lengths)

dfplayer['HMM'] = model.predict(X)

pdb.set_trace()
joblib.dump(model, "models/HMM_Aim_View.pkl")


test_data, test_passenger_id = clean_data_to_numbers('data/test.csv', remove_columns)


# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 10000, max_features='sqrt',min_samples_split=1)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# Take the same decision trees and run it on the test data
result = forest.predict(test_data)

write_model("data/RandomForestModel.csv", result, test_passenger_id)


# Do score here.
print "Done!"

