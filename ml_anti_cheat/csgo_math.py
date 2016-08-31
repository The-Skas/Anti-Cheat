import numpy as np
import math
from collections import namedtuple

"""
This file serves to collect my math methods used for intersections and such...
Probably needs reworking.
"""


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

def player_look_intersect(player, enemy , p_get_i, e_get_i):
	""" Calculates the point of of the player's look direction relative
	to an enemy.

	params: player is a row from a player's dataframe.
			enemy  is a row from an enemie's dataframe.
	"""
	p_pos = np.array([ player[p_get_i["X"]]  ,  player[p_get_i["Y"]],    player[p_get_i["Z"]]])

	p_look_dir = dir_from_angle(player[p_get_i['ViewX']] + 2.0 * player[p_get_i['AimXPunchAngle']], player[p_get_i['ViewY']] + 2.0 * player[p_get_i['AimYPunchAngle']])
	
	l_0 = np.array([ player[p_get_i["X"]]  ,  player[p_get_i["Y"]],    player[p_get_i["Z"]] + player[p_get_i["ViewZOffset"]]])
	

	## The plane is perpindicular to the Z Axis.
	normal_vec = np.array(p_look_dir)
	normal_vec[2] = 0

	#Calculate Intersection Point.
	e_pos = np.array([ enemy[e_get_i['X']],  enemy[e_get_i['Y']],  enemy[e_get_i['Z']] + enemy[e_get_i['ViewZOffset']] ])
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