from data_helpers import *

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

def evaluate_accuracy_of_removed_columns(columns=[]):
	train_data, train_passenger_id = data_munge('data/train.csv', columns)
	"""
	This function returns the accuracy of the forest given the columns to be removed 
	"""
	# for i,x in enumerate(list_of_columns)

	train_data, train_passenger_id = data_munge('data/train.csv', columns)

	valid_data, train_data = np.array_split(train_data, 2)

	# Create the random forest object which will include all the parameters
	# for the fit
	forest = RandomForestClassifier(n_estimators = 1000, max_features='auto', min_samples_split=1)

	# Fit the training data to the Survived labels and create the decision trees
	forest = forest.fit(train_data[0::,1::],train_data[0::,0])

	return forest.score(valid_data[0::,1::], valid_data[0::, 0])

def feature_selection_decision_tree():
	list_of_columns =  ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Gender', 'AgeIsNull', 'FamilySize', 'Age*Class']

	mutable_list = list(list_of_columns)

	final_removeable_columns = list()
	# Backward Feature Selection
	best_accuracy = evaluate_accuracy_of_removed_columns()


	array_of_best_results = list()

	amount_redundant_loop = 0
	for i, x in enumerate(list_of_columns):
		print(i)
		print(x)
		best_column_i = -1
		for j, y in enumerate(mutable_list):
			temp_removeable_columns = list(final_removeable_columns)

			# We pick the index from mutable list, and not list_of_columns
			# Although in the first iteration they have the same values,
			# After each x+1, mutable list will have a column removed from it.
			
			temp_removeable_columns.append(mutable_list[j])

			temp_accuracy = evaluate_accuracy_of_removed_columns(temp_removeable_columns)

			# if The accuracy improved after removing the given columns
			if(temp_accuracy > best_accuracy):
				best_accuracy = temp_accuracy
				best_column_i = j

				col_acc_list = list()
				col_acc_list.append(best_accuracy)
				col_acc_list.append(temp_removeable_columns)

				array_of_best_results.append(col_acc_list)
		
		# Add the column that provided the most accuracy when removed.
		if(best_column_i != -1):
			final_removeable_columns.append(mutable_list[best_column_i])
			del mutable_list[best_column_i]
		else:
			++amount_redundant_loop


		# Note: If best_column is empty, then dont add anything to final_removable_clumns
		# 
	# TO SORT  l.sort(key=lambda x: x[2])
	array_of_best_results.sort(key=lambda x: x[0])
	print array_of_best_results
	return array_of_best_results


print evaluate_accuracy_of_removed_columns(['Parch', 'FamilySize', 'Age', 'AgeIsNull', 'SibSp', 'Pclass'])

# feature_selection_decision_tree()

# Do score here.
print "Done!"

