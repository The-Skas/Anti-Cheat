import pandas as pd
import numpy as np
import pdb
import math
import pylab as P
import csv as csv
import pdb
from collections import namedtuple

def _debug():
	my_pos = np.array([-1752.0130 , 1980.272 , 10.346030  ])
 	e_pos = np.array([-1364.9800,  2555.752,     5.275375])
 	alpha = 56.60156
 	beta = 0.351562
 	print "Ok"
 	return player_look_intersect(alpha, beta, my_pos, e_pos)


# https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
#
def line_plane_intersect(l_0, l_vec, p_0, normal_vec):
	""" Calculates intersect between a vector and a plane.
	returns 0 if there is no intersect, otherwise:
	returns intersection point [x,y,z]
	"""
	upper_eq = ((p_0 - l_0) * normal_vec).sum()
	lower_eq = (l_vec * normal_vec).sum()
	
	if(lower_eq == 0):
		return 0
	else:
		dist = upper_eq / lower_eq
		return dist * l_vec + l_0

def dir_from_angle(deg_alpha, deg_beta, r=1.0):
	rad_alpha = math.radians(deg_alpha)
	rad_beta  = math.radians(deg_beta)

	x = r * math.cos(rad_beta) * math.sin(rad_alpha)
	y = r * math.cos(rad_beta) * math.cos(rad_alpha)
	z = r * math.sin(rad_beta)

	#Weird, should be [x, y, z], but changing to fit demo data.
	return np.array([y, x, -z])


def orthogonal(vec1, vec2):
	if (vec1 * vec2).sum() == 0:
		return True
	else:
		return False

P_VIEW_Z_OFFSET = 50
Intersect = namedtuple('Intersect', ['point', 'normal', 'localx', 'localy'])
def player_look_intersect(p_view_x, p_view_y, p_pos , e_pos ):
	""" Calculates the point of intersection of the players look direction
	relative to a specific position.
	"""
	p_look_dir = dir_from_angle(p_view_x, p_view_y)
	# Need to set Z to 0.
	l_0 = np.array([p_pos[0], p_pos[1], p_pos[2] + P_VIEW_Z_OFFSET])
	
	normal_vec = np.array(p_look_dir)
	normal_vec[2] = 0

	#Calculate Intersection Point.
	intersection_point = line_plane_intersect(l_0 =l_0, l_vec= p_look_dir , p_0 = e_pos, normal_vec = normal_vec)

	#Calculate 2d coordinate relative: 
	#Where:e_pos is the origin
	#	   u is the relative 'x'-axis,
	#	   v is the relative 'y'-axis.
	v = np.array([0,0,1])
	u = np.cross(normal_vec, v)

	y = ((intersection_point - e_pos) * v).sum()
	x = ((intersection_point - e_pos) * u).sum()

	#Todo: So this is the bottle neck apparently. Creating tuples. 
	#	   Perhaps an alternative would be to re-use the same class by reference.
	return Intersect(intersection_point, normal_vec, x, y)

def player_intersects(df,enemy_name="Eugene", player_id=76561197979652439, start_tick=24056):
	#TODO: Change this its hacky. Well acctually fuck it not worth it.
	dfplayer = df[(df.Steam_ID == player_id) & (df.Tick >= start_tick)].reset_index()
	dfenemy  = df[(df.Name    == enemy_name)& (df.Tick >= start_tick)].reset_index()
						#TODO: Using iterrows is inefficient
	#adding new column, to overwrite
	dfplayer["XAimbot"] = dfplayer["Tick"]
	dfplayer["YAimbot"] = dfplayer["Tick"]

	for i, (player, enemy) in enumerate(zip(dfplayer.iterrows(), dfenemy.iterrows())):
		p_pos = np.array([player[1].PlayerX, player[1].PlayerY, player[1].PlayerZ])
		e_pos = np.array([ enemy[1].PlayerX,  enemy[1].PlayerY,  enemy[1].PlayerZ])

		intersect = player_look_intersect(player[1].ViewX, player[1].ViewY, p_pos, e_pos)
		dfplayer.set_value(i, "XAimbot", intersect.localx)
		dfplayer.set_value(i, "YAimbot", intersect.localy) 

	pdb.set_trace()

	print "stop here."





def clean_data_to_numbers(file,additional_columns = [], drop_columns_default = ['Name', 'Sex', 'Ticket', 'Cabin']):
	df = pd.read_csv(file,delimiter=';', header=0)

	#Get rows where tick is greater then 4570 and filter out bots.
	# dfplayer= df[(df.Tick > 4570) & (df.Steam_ID > 0)]
	dfplayer= df[(df.Steam_ID > 0)]
	dfeugene = df[(df.Name == "Eugene")]
	#Drop Rows.
	dfplayer= dfplayer.drop(["Steam_ID","PlayerX", "PlayerY", "PlayerZ", "Unnamed: 17"], axis=1)

	#Calculates the difference between previous viewAngle-X, and current viewAngleX. 
	#Then get value.
	dfplayer['ViewXDiff'] = ((dfplayer.ViewX - dfplayer.ViewX.shift(1) + 180) % 360 - 180).abs()
	#Categorizes the angles.
	dfplayer['ViewXDiffBin'] = pd.cut(dfplayer.ViewXDiff,3,labels=["low","medium","high"])

	#Same for Y
	dfplayer['ViewYDiff'] = ((dfplayer.ViewY - dfplayer.ViewY.shift(1) + 180) % 360 - 180).abs()
    #Bin angles to three:
	dfplayer['ViewYDiffBin'] =  pd.cut(dfplayer.ViewYDiff,3,labels=["low","medium","high"])

	#Get acctual distance traveled of angle diff.. This needs testing probs.
	dfplayer['ViewDiff'] = ((dfplayer.ViewYDiff)**2  + (dfplayer.ViewXDiff)**2).apply(np.sqrt)
	dfplayer['ViewDiffBin'] =  pd.cut(dfplayer.ViewDiff,2,labels=["low", "high"])
	# dfplayer[dfplayer.ViewDiff > 20].drop(["Name", "ViewX", "ViewY","ViewXDiff", "ViewYDiff", "ViewXDiffBin", "ViewYDiffBin"], axis=1)[:50]
	player_intersects(df)
	pdb.set_trace()

	# Convert gender to number
	df['Gender'] = df['Sex'].map({'female': 0, 'male': 1})

	# Maps all non null values of Embarked to numbers.
	pdb.set_trace()
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
	pdb.set_trace()

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