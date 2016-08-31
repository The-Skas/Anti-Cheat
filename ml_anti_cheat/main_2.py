import pandas as pd
import numpy as np
import pdb
import math
import pylab as P
import csv as csv
# P.show()
# Some_data_frame.hist() //Shows graph


# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('data/train.csv', header=0)

# Prints out the data types of each column
df.dtypes

# Prints out all columns and how many non-null values they contain.
df.info()

# Prints out statistic analysis
df.describe()

# This prints out
df['Age'][0:10]
df.Age[0:10]

# This gets the mean of age.
df.Age.mean()

# Prints out the the three columns
df[ ['Sex', 'Pclass', 'Age'] ]

# Gets all (Age columns) data with age > 60
df[df['Age'] > 60]

# Gets All (Sex, Pclass, Age columns) data with age > 60
df[df['Age'] > 60][['Survived','Sex', 'Pclass', 'Age']]

# Selects all data with null value as age.
df[df['Age'].isnull()]

# Print chart
#************
# Sort data to cabins alphabetical order.
# This is assuming that the alphabet order of cabins represnt
# the layout
# Create data before Altering df
data = df[df['Cabin'].isnull() == False]
sorted_data = data[['Cabin', 'Survived']].sort('Cabin')

final_data = sorted_data.copy()

for index, row in sorted_data.iterrows():
	final_data.loc[index, 'Index'] = index.astype(int)
	final_data.loc[index, 'Survived'] = row['Survived']
# final_data.plot(kind='scatter', x='Index', y='Survived')
# P.show()
# END

# Creates a new column 'Gender', and maps the upper case character 'F' or 'M'
df['Sex'] = df['Sex'].map( lambda x: x[0].upper() )

# Sets all genders to equate to 1 or 0
df['Gender'] = df['Sex'].map({'F': 0, 'M': 1})



# Maps all non null values of Embarked to numbers.
df['Embarked']=  df[df['Embarked'].isnull() == False].Embarked.map({'C':1,'Q':2,'S':3})
# Gets the median
Embarked_median = df['Embarked'].median()
# Maps all Embarked null values to the median
df['Embarked']=df['Embarked'].fillna(Embarked_median)

median_ages = np.zeros((2,3))

# +1 range for iterations
for i in range(0,2):
	for j in range(df['Pclass'].min(), df['Pclass'].max()+1):
		median_ages[i,j-1] = df[(df['Gender'] == i ) & (df['Pclass'] == j)].Age.dropna().mean()



temp_df = df.copy()
temp_df['AgeFill'] = df['Age']

# Gets the first 10 values, from the filtered dataFrame.
df[ df['Age'].isnull() ][['Gender','Pclass','Age']].head(10)


for i in range(0, 2):
    for j in range(0, 3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'Age'] = median_ages[i,j]

# Convert all floats to a range of 0.5 or 1.0
# The reason being to fit the codempo rules (Refer to data)
df['Age']= df['Age'].map(lambda x: math.ceil(x * 2.0) * 0.5)

# This creates a new column ('AgeIsNull') 
# 
# pd: this is the pandas library
# pd.isnull(arg1): this is a function that converts the dataFrame rows
# 				   into a true/false table.
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']

# This multiplies the Age of the person by the social 
# class. It adds to the fact that higher ages are even
# LESS likely to survive
df['Age*Class'] = df.Age * df.Pclass

# Since skipi doesnt work well with strings
df.dtypes[df.dtypes.map(lambda x: x=='object')]




# Setting up for machine learning yikes! Horrible!
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin'], axis=1)

# Drops all columns that have any null value.. 
# uh? wtf? Super bad.
df = df.dropna()





train_data = df.values

def data_munge(file,additional_columns = [], drop_columns_default = ['Name', 'Sex', 'Ticket', 'Cabin']):
	df = pd.read_csv(file, header=0)
	# Convert gender to number
	df['Gender'] = df['Sex'].map({'female': 0, 'male': 1})

	# Maps all non null values of Embarked to numbers.
	df['Embarked']=  df[df['Embarked'].isnull() == False].Embarked.map({'C':1,'Q':2,'S':3})
	# Gets the median
	Embarked_median = df['Embarked'].median()
	# Overwrites all of column 'Embarked' null values to equal the median 'Embarked'
	# TODO: Create a model to predict 'Embarked'.
	df['Embarked']=df['Embarked'].fillna(Embarked_median)

	# Creates an array of 6 values. 2 Rows, 3 columns.
	median_ages = np.zeros((2,3))

	# For each Male/Female, we will have Three different median ages
	# depending on what their Economic class ('Pclass') is.
	for i in range(0,2):
		for j in range(df['Pclass'].min(), df['Pclass'].max()+1):
			median_ages[i,j-1] = df[(df['Gender'] == i ) & (df['Pclass'] == j)].Age.dropna().median()

	# AgeIsNull
	df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

	#  stores the median age for rows with null 'Age'
	for i in range(0, 2):
	    for j in range(0, 3):
	        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'Age'] = median_ages[i,j]

	# Convert all floats to a range of 0.5 or 1.0
	# The reason being to fit the compo rules (Refer to data)
	df['Age']= df['Age'].map(lambda x: math.ceil(x * 2.0) * 0.5)

	# *** DO MEAN FOR FARE ****
	mean_fare = np.zeros((2,3))
	for i in range(0,2):
		for j in range(df['Pclass'].min(), df['Pclass'].max()+1):
			mean_fare[i,j-1] = df[(df['Gender'] == i ) & (df['Pclass'] == j)].Fare.dropna().mean()


	for i in range(0, 2):
	    for j in range(0, 3):
	        df.loc[(df.Fare.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'Fare'] = mean_fare[i,j]
	# This creates a new column ('AgeIsNull') 
	# 
	# pd: this is the pandas library
	# pd.isnull(arg1): this is a function that converts the dataFrame rows
	# 				   into a true/false table.

	df['FamilySize'] = df['SibSp'] + df['Parch']

	# This multiplies the Age of the person by the social 
	# class. It adds to the fact that higher ages are even
	# LESS likely to survive
	df['Age*Class'] = df.Age * df.Pclass

	# Since skipi doesnt work well with strings
	df.dtypes[df.dtypes.map(lambda x: x=='object')]
	# Setting up for machine learning yikes! Horrible!
	# The values you drop can improve or make worse.
	df = df.drop(drop_columns_default + additional_columns, axis=1)
	# Drops all columns that have any null value.. 
	# uh? wtf? Super bad.
	df = df.dropna()


	# To store Id
	passengerIds = df['PassengerId']

	# Drop Id since output format issues
	df = df.drop(['PassengerId'], axis = 1)


	return df.values, passengerIds

def get_array_id_from_file(file):
	df = pd.read_csv(file, header=0)

	return df['PassengerId']

def write_model(fileName, output, passengersId):
	prediction_file = open(fileName, "wb")
	prediction_file_object = csv.writer(prediction_file)
	prediction_file_object.writerow(["PassengerId", "Survived"])
	
	for i,x in enumerate(passengersId):       # For each row in test.csv
	        prediction_file_object.writerow([x, output[i].astype(int)])    # predict 1

	prediction_file.close()


