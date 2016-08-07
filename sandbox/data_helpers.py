import pandas as pd
import numpy as np
import pdb
import math
import pylab as P
import csv as csv
import pdb

# My files
import csgo_math

#plot
from mpl_toolkits.axes_grid.axislines import SubplotZero
import matplotlib.pyplot as plt
from matplotlib import cm
from  matplotlib.animation import FuncAnimation


def plane_plot(x, y):
	fig = plt.figure(1)
	ax = SubplotZero(fig, 111)
	fig.add_subplot(ax)

	for direction in ["xzero", "yzero"]:
		ax.axis[direction].set_axisline_style("-|>")
		ax.axis[direction].set_visible(True)

	for direction in ["left", "right", "bottom", "top"]:
		ax.axis[direction].set_visible(False)
	ax.plot(x.values, y.values)
	plt.show()

def plot_scatter_m4a1s(dfplayer):
	_df = dfplayer[(dfplayer.Tick >=20938)& (dfplayer.Tick <= 21080) & (dfplayer.AimYPunchAccel < 0)][['Tick', 'AimYPunchAngle', 'AimXPunchAngle', 'ViewX', 'ViewY']][0:30]

	#Must create new figure for each plot.
	cm_subsection = np.linspace(0.0, 0.4, 20)
	colors = [ cm.jet(x) for x in cm_subsection]

	#Plot creating a fade away color for each line.. 
	fig= plt.figure()
	fig.suptitle('Hacker M4A1-S')
	plt.xlabel('ViewX')
	plt.ylabel('ViewY')
	for i, x in enumerate(_df.ViewX.values):
		plt.scatter(x=_df.ViewX.iloc[i], y=_df.ViewY.iloc[i], color=colors[i])

	fig = plt.figure()
	fig.suptitle('Recoil M4A1-S')
	plt.xlabel('XPunch')
	plt.ylabel('YPunch')
	plt.scatter(x=-_df.AimXPunchAngle, y= -_df.AimYPunchAngle, color='blue')

	fig = plt.figure()
	fig.suptitle('Hacker M4A1-S vs Recoil M4A1-s')
	plt.scatter(x=_df.ViewX, y=_df.ViewY, color='red')
	plt.scatter(x= _df.ViewX.iloc[0] - _df.AimXPunchAngle, y= _df.ViewY.iloc[0] -_df.AimYPunchAngle, color='blue')

	fig = plt.figure()
	fig.suptitle('Hacker M4A1-S vs Recoil M4A1-s')
	plt.scatter(x=_df.ViewX, y=_df.ViewY, color='red')
	plt.scatter(x= _df.ViewX.iloc[0] - 2.0*_df.AimXPunchAngle, y= _df.ViewY.iloc[0] - 2.0*_df.AimYPunchAngle, color='blue')


	fig = plt.figure()
	fig.suptitle('Hacker M4A1-S vs Recoil M4A1-s vs Shots Angle')
	plt.scatter(x=_df.ViewX, y=_df.ViewY, color='red')
	plt.scatter(x= _df.ViewX.iloc[0] - 2.0*_df.AimXPunchAngle, y= _df.ViewY.iloc[0] - 2.0*_df.AimYPunchAngle, color='blue')
	plt.scatter(x=_df.ViewX + 2.0*_df.AimXPunchAngle , y=_df.ViewY + 2.0*_df.AimYPunchAngle, color='green')
	plt.show()

	# fig = plt.figure()
	# axes = plt.gca()
	# axes.set_xlim([245,251])
	# axes.set_ylim([0,8])
	# fig.suptitle('Hacker Aim while removing recoil')
	# plt.show()

def _debug():
	my_pos = np.array([-1752.0130 , 1980.272 , 10.346030  ])
 	e_pos = np.array([-1364.9800,  2555.752,     5.275375])
 	alpha = 56.60156
 	beta = 0.351562
 	print "Ok"
 	return player_look_intersect(alpha, beta, my_pos, e_pos)


# https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
#


def player_intersects(df,enemy_name="Eugene", player_id=76561197979652439, start_tick=24056, end_tick=100000000):
	#TODO: Change this its hacky. Well acctually fuck it not worth it.
	dfplayer = df[(df.Steam_ID == player_id) & (df.Tick >= start_tick) & (df.Tick <= end_tick) ].reset_index()
	dfenemy  = df[(df.Name    == enemy_name)& (df.Tick >= start_tick)  & (df.Tick <= end_tick)].reset_index()
						#TODO: Using iterrows is inefficient
	#adding new column, to overwrite
	dfplayer["XAimbot"] = dfplayer["Tick"]
	dfplayer["YAimbot"] = dfplayer["Tick"]
	dfplayer["Intersect"] = dfplayer["Tick"].astype(str)

	""" DELETE HERE"""###
	dfplayer["TimeDiff"] = (dfplayer.Time - dfplayer.Time.shift(1))
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
	
	dfplayer['AimYPunchAccel'] = dfplayer.AimYPunchVel - dfplayer.AimYPunchVel.shift(1)
	dfplayer['AimYPunchAccelDiff'] = dfplayer.AimYPunchAccel - dfplayer.AimYPunchAccel.shift(1)

	dfplayer["TrueViewX"]= dfplayer.ViewX + 2.0*dfplayer.AimXPunchAngle
	dfplayer["TrueViewY"]= dfplayer.ViewY + 2.0*dfplayer.AimYPunchAngle
	dfplayer["TrueViewDiff"] = ((dfplayer.TrueViewX)**2  + (dfplayer.TrueViewY)**2).apply(np.sqrt)

	""" TO HERE """ ######
	

	p_get_i = {x: i+1 for i, x in  enumerate(dfplayer.columns)}
	e_get_i = {x: i+1 for i, x in  enumerate(dfenemy.columns)}
	for i, (player, enemy) in enumerate(zip(dfplayer.itertuples(), dfenemy.itertuples())):
		intersect = csgo_math.player_look_intersect(player, enemy, p_get_i, e_get_i)
		dfplayer.set_value(i, "XAimbot", intersect.localx)
		dfplayer.set_value(i, "YAimbot", intersect.localy) 
		dfplayer.set_value(i, "Intersect", "|#|".join(map(str, intersect.point)))

	return dfplayer
	# print '{:f}'.format(t1-t0)
	# pdb.set_trace()


	# print "stop here."

	##Plot Fun###

	##First Draw Player and Enemy Position.

	# fig1 = plt.figure()
	# axes = plt.gca()
	# axes.set_xlim([dfenemy.iloc[2].X - 5, dfenemy.iloc[2].X + 5])
	# axes.set_ylim([dfenemy.iloc[2].Y - 5,dfenemy.iloc[2].Y + 5])
	# i = 0
	# def update(frame):
	# 	i = frame
	# 	fig1.clear()
	# 	axes = plt.gca()
	# 	axes.set_xlim([dfplayer.X.min() - 5, dfenemy.X.max() + 5])
	# 	axes.set_ylim([dfplayer.Y.min() - 5, dfenemy.Y.max() + 5])

	# 	plt.scatter(x=dfplayer.iloc[i].X, y=dfplayer.iloc[i].Y, color="green")
	# 	plt.scatter(x=dfenemy.iloc[i].X, y=dfenemy.iloc[i].Y, color="red")
	# 	intersect = dfplayer.iloc[i].Intersect.split("|#|")
	# 	plt.scatter(x=float(intersect[0]),y=float(intersect[1]), color="blue")

	# def update_plane(frame):
	# 	i = frame
	# 	fig2.clear()
	# 	axes = plt.gca()
	# 	axes.set_xlim([-10,10])
	# 	axes.set_ylim([-10,10])
	# 	plt.scatter(x=0, y=0, color="red")
	# 	plt.scatter(x=float(dfplayer.iloc[i].XAimbot),y=float(dfplayer.iloc[i].YAimbot), color="blue")

	# fig2 = plt.figure()
	# # animation = FuncAnimation(fig1, update, interval=24)
	# animation_2 = FuncAnimation(fig2, update_plane, interval=24)

	# plt.show()
	# pdb.set_trace()



import sys
def clean_data_to_numbers(file,additional_columns = [], drop_columns_default = ['Name', 'Sex', 'Ticket', 'Cabin'], player_id = 0):
	
	
	#Main data frame
	df = pd.read_csv(file,delimiter=';', header=0)
	#Get all hurt data.
	dfhurt = pd.read_csv(sys.argv[2], delimiter=';', header=0)

	#Derive Get all Shots.
	dfshots = df[(df.HasShot == True)][['Tick','Name','Steam_ID', 'Weapon']]
	
	#Useful for merging when wanting to Find all shots that hit.
	#Due to sometimes having None as weapon.
	dfshothits = pd.merge(left=dfshots, right=dfhurt, left_on=["Tick", "Steam_ID"] ,right_on=["Tick", "Attacker"], how='right').drop(["Steam_ID"], axis=1)


	#Get rows where tick is greater then 4570 and filter out bots.
	# dfplayer= df[(df.Tick > 4570) & (df.Steam_ID > 0)]
	dfplayer = None
	if not player_id:
		#TODO: Assuming no bots exceeding 1000
		dfplayer= df[(df.Steam_ID > 1000)]
	else:
		dfplayer = df[(df.Steam_ID == int(player_id))]

	dfeugene = df[(df.Name == "Eugene")]
	#Drop Rows.
	dfplayer= dfplayer.drop(["Steam_ID","X", "Y", "Z"], axis=1)

	dfplayer["TimeDiff"] = (dfplayer.Time - dfplayer.Time.shift(1))
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
	
	dfplayer['AimYPunchAccel'] = dfplayer.AimYPunchVel - dfplayer.AimYPunchVel.shift(1)
	dfplayer['AimYPunchAccelDiff'] = dfplayer.AimYPunchAccel - dfplayer.AimYPunchAccel.shift(1)

	dfplayer["TrueViewX"]= dfplayer.ViewX + 2.0*dfplayer.AimXPunchAngle
	dfplayer["TrueViewY"]= dfplayer.ViewY + 2.0*dfplayer.AimYPunchAngle
	dfplayer["TrueViewDiff"] = ((dfplayer.TrueViewX)**2  + (dfplayer.TrueViewY)**2).apply(np.sqrt)

	# dfplayer[dfplayer.ViewDiff > 20].drop(["Name", "ViewX", "ViewY","ViewXDiff", "ViewYDiff", "ViewXDiffBin", "ViewYDiffBin"], axis=1)[:50]
	
	dfplayer = player_intersects(df, start_tick=0) #, enemy_name = "ENVYUS apEXmousse[D]", player_id=76561197995369711, start_tick=47900, end_tick=48500)
	
	#Remove all data not part of a round
	dfplayer = dfplayer[dfplayer.Round != 0]

	#Create a distance metric from the target
	dfplayer['AimbotDist'] = ((dfplayer.XAimbot)**2 + (dfplayer.YAimbot**2)).apply(np.sqrt)

	return dfplayer

	# pdb.set_trace()
	#Plot:
	# plot_scatter_m4a1s(dfplayer)	

	# Convert gender to number

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