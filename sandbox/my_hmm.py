from data_helpers import *

from hmmlearn.hmm import GaussianHMM

import pandas as pd
import sys
import pdb
from sklearn.externals import joblib
#plot
import csgo_plot

# Split rounds into multiple dataframes.

def split_rounds(df):
	rounds = df.Round.unique()
	dictrounds = {elem : pd.DataFrame for elem in rounds}
	for key in dictrounds.keys():
		_df = dfplayer[:][dfplayer.Round == key]
		_length = len(_df)
		dictrounds[key] = (_df , _length)
	return dictrounds

def create_markov_model(df, columns, n_components, covariance_type="diag", n_iter=1000):
	# Given a data frame, list of columns for the markov model

	# drop all data which has any of the columns null.
	#remove null values

	dfclean = df.dropna(subset=columns)
	# Get rounds
	dictrounds = split_rounds(dfclean)

	# Get Appropriate Training Dimensions
	X = np.reshape(np.array([]),(0,len(columns)))

	for key in dictrounds.keys():
		# Get at index 0 since its tupled of df
		S = dictrounds[key][0][columns].as_matrix()
		X = np.vstack([X,S])

	lengths = [ dictrounds[key][1] for key in dictrounds.keys()]

	model = GaussianHMM(n_components= n_components, covariance_type= covariance_type , n_iter= n_iter).fit(X, lengths)

	dfclean['HMM'] = model.predict(X)

	joblib.dump(model, "models/"+'_'.join(columns)+".pkl")

	return model, X





player_id = sys.argv[3] if len(sys.argv) >= 4 else 0

dfplayer = clean_data_to_numbers(sys.argv[1], player_id=player_id)

# Chose 16 components due to dividing around a circle.
model_1, X_1 = create_markov_model(dfplayer, ["TrueViewSin","TrueViewCos"], n_components = 16)
# Plot as an Axis Around the model.
pdb.set_trace()
csgo_plot.plot_scatter_hmm(X_1[:,0] ,X_1[:,1] , model_1, X_1)

#Chose 4 components to give some variance.
model_2, X_2 = create_markov_model(dfplayer, ["TrueViewDiff"] , n_components=4)
# Plot as a linegraph over time.

pdb.set_trace()

## Get all Non-Null ViewDiffs and Aimbotdist
## Issue with this is the fucking up the time series...

write_model("data/RandomForestModel.csv", result, test_passenger_id)


# Do score here.
print "Done!"

