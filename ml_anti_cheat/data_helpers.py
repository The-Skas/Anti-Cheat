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
	dfplayer = df[(df.Steam_ID == int(player_id)) & (df.Tick >= start_tick) & (df.Tick <= end_tick) ].reset_index()

	dfenemy = None
	if enemy_name:
		dfenemy  = df[(df.Name    == enemy_name)& (df.Tick >= start_tick)  & (df.Tick <= end_tick)].reset_index()
						#TODO: Using iterrows is inefficient
	dfplayer = dfplayer[dfplayer.Alive]
	#adding new column, to overwrite
	dfplayer["XAimbot"] = dfplayer["Tick"]
	dfplayer["YAimbot"] = dfplayer["Tick"]
	dfplayer["Intersect"] = dfplayer["Tick"].astype(str)
	
	#Calculate the difference in time.
	dfplayer["TimeDiff"] = (dfplayer.Time - dfplayer.Time.shift(1))
	#Calculates the difference between previous viewAngle-X, and current viewAngleX. 
	#Then get value.
	dfplayer['ViewXDiff'] = ((dfplayer.ViewX - dfplayer.ViewX.shift(1) + 180) % 360 - 180)
	#Categorizes the angles.
	dfplayer['ViewXDiffBin'] = pd.cut(dfplayer.ViewXDiff,3,labels=["low","medium","high"])

	#Same for Y
	dfplayer['ViewYDiff'] = ((dfplayer.ViewY - dfplayer.ViewY.shift(1) + 180) % 360 - 180)
    	#Bin angles to three:
	dfplayer['ViewYDiffBin'] =  pd.cut(dfplayer.ViewYDiff,3,labels=["low","medium","high"])

	#Get acctual distance traveled of angle diff.. This needs testing probs.
	dfplayer['ViewDiff'] = ((dfplayer.ViewYDiff)**2  + (dfplayer.ViewXDiff)**2).apply(np.sqrt)
	dfplayer["ViewDiffSpeed"] =  dfplayer.ViewDiff / dfplayer.TimeDiff
	dfplayer["ViewDiffAccel"] = (dfplayer.ViewDiffSpeed - dfplayer.ViewDiffSpeed.shift(1)).round().abs()
	#dfplayer['ViewDiffBin'] =  pd.cut(dfplayer.ViewDiff,2,labels=["low", "high"])
	
	dfplayer['AimYPunchAccel'] = dfplayer.AimYPunchVel - dfplayer.AimYPunchVel.shift(1)
	dfplayer['AimYPunchAccelDiff'] = dfplayer.AimYPunchAccel - dfplayer.AimYPunchAccel.shift(1)

	dfplayer["TrueViewX"]= dfplayer.ViewX + 2.0*dfplayer.AimXPunchAngle
	dfplayer["TrueViewY"]= dfplayer.ViewY + 2.0*dfplayer.AimYPunchAngle
	
	dfplayer["TrueViewXDiff"] = ((dfplayer.TrueViewX - dfplayer.TrueViewX.shift(1) + 180)  % 360 - 180)
	dfplayer["TrueViewXVel"] = dfplayer.TrueViewXDiff / dfplayer.TimeDiff

	dfplayer["TrueViewYDiff"]=  ((dfplayer.TrueViewY - dfplayer.TrueViewY.shift(1) + 180)  % 360 - 180)
	dfplayer["TrueViewYVel"] = dfplayer.TrueViewYDiff / dfplayer.TimeDiff

	dfplayer["TrueViewDiff"] = ((dfplayer.TrueViewXDiff)**2  + (dfplayer.TrueViewYDiff)**2).apply(np.sqrt)

	# Understand the formula (ang_t - ang_t_1 ) + 180) % 360 - 180 ## gives us the shortest distance
	dfplayer["TrueViewXAngDiff"]= (dfplayer.TrueViewX - dfplayer.TrueViewX.shift(1) + 180) % 360 - 180
	dfplayer["TrueViewYAngDiff"]= (dfplayer.TrueViewY - dfplayer.TrueViewY.shift(1) + 180) % 360 - 180 
	dfplayer["TrueViewDiffSpeed"] = dfplayer.TrueViewDiff / dfplayer.TimeDiff


	#Get Sin value
											###    y / x
	dfplayer["TrueViewRad"]  =  dfplayer.apply(lambda row: math.atan2(row.TrueViewYAngDiff , row.TrueViewXAngDiff ), axis=1) 
	dfplayer["TrueViewSin"]  =  dfplayer.apply(lambda row: math.sin(row.TrueViewRad), axis=1)
	dfplayer["TrueViewCos"]  =  dfplayer.apply(lambda row: math.cos(row.TrueViewRad), axis=1)



	dfplayer["ViewRad"] = dfplayer.apply(lambda row: math.atan2(row.ViewXDiff , row.ViewYDiff) , axis=1)
	dfplayer["ViewRadDiff"] = dfplayer.ViewRad - dfplayer.ViewRad.shift(1)
	dfplayer["ViewRadDiffSpeed"] = dfplayer.ViewRadDiff / dfplayer.TimeDiff
	## Used to measure changes in Mouse mdir

	dfplayer["TrueViewRadDiff"] = dfplayer.TrueViewRad - dfplayer.TrueViewRad.shift(1)
	dfplayer["TrueViewRadDiffSpeed"] = dfplayer.TrueViewRadDiff / dfplayer.TimeDiff 



	

	#Calculate intersections
	p_get_i = {x: i+1 for i, x in  enumerate(dfplayer.columns)}
	e_get_i = {x: i+1 for i, x in  enumerate(dfenemy.columns)}
	for i, (player, enemy) in enumerate(zip(dfplayer.itertuples(), dfenemy.itertuples())):
		intersect = csgo_math.player_look_intersect(player, enemy, p_get_i, e_get_i)
		dfplayer.set_value(i, "XAimbot", intersect.localx)
		dfplayer.set_value(i, "YAimbot", intersect.localy) 
		dfplayer.set_value(i, "Intersect", "|#|".join(map(str, intersect.point)))

	## Drop rounds last:
		#Remove all data not part of a round
	dfplayer['AimbotDist'] = ((dfplayer.XAimbot)**2 + (dfplayer.YAimbot**2)).apply(np.sqrt)
	dfplayer = dfplayer[dfplayer.Round != 0]

	return dfplayer


import sys
def data_munge(file, filehurt, dictargs, additional_columns = [], drop_columns_default = ['Name', 'Sex', 'Ticket', 'Cabin']):
	
	
	#Main data frame
	df = pd.read_csv(file,delimiter=';', header=0)

	#Get all hurt data.
	dfhurt = pd.read_csv(filehurt, delimiter=';', header=0)

	#Derive Get all Shots.
	dfshots = df[(df.HasShot == True)][['Tick','Name','Steam_ID', 'Weapon']]
	
	#Useful for merging when wanting to Find all shots that hit.
	#Due to sometimes having None as weapon.
	dfshothits = pd.merge(left=dfshots, right=dfhurt, left_on=["Tick", "Steam_ID"] ,right_on=["Tick", "Attacker"], how='right').drop(["Steam_ID"], axis=1)


	#Get rows where tick is greater then 4570 and filter out bots.
	# dfplayer= df[(df.Tick > 4570) & (df.Steam_ID > 0)]
	dfplayer = None
	if not dictargs["id"]:
		#TODO: Assuming no bots exceeding 1000
		dfplayer= df[(df.Steam_ID > 1000)]
	else:
		dfplayer = df[(df.Steam_ID == int(dictargs["id"]))]

	dfeugene = df[(df.Name == "Eugene")]
	#Drop Rows.
	dfplayer= dfplayer.drop(["Steam_ID","X", "Y", "Z"], axis=1)

	# dfplayer[dfplayer.ViewDiff > 20].drop(["Name", "ViewX", "ViewY","ViewXDiff", "ViewYDiff", "ViewXDiffBin", "ViewYDiffBin"], axis=1)[:50]
	dfplayer = player_intersects(df, player_id=int(dictargs["id"]), start_tick=int(dictargs["start_tick"]), end_tick=int(dictargs["end_tick"]), enemy_name=None) #, enemy_name = "ENVYUS apEXmousse[D]", player_id=76561197995369711, start_tick=47900, end_tick=48500)
	
	#Create a distance metric from the target
	

	return dfplayer

	# pdb.set_trace()
	#Plot:
	# plot_scatter_m4a1s(dfplayer)	

	# Convert gender to number

def time_warp_data(df):
	pass
	#Given df..
	# We check the difference in time between the data and normalize it.
	# Speed is already normalized... All data that might vary

	## Normalize 

