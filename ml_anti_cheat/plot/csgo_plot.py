from mpl_toolkits.axes_grid.axislines import SubplotZero
import matplotlib.pyplot as plt
from matplotlib import cm
from  matplotlib.animation import FuncAnimation
import numpy as np
import pdb
import os
from matplotlib.lines import Line2D

def plot_plane_diff(xs, ys, title="Plane"):
	xs = xs.astype(float)

	fig = plt.figure()
	fig.suptitle(title)

	hacker_y = ys.copy()
	hacker_x = xs.copy()

	hacker_y[ ys <= 0] = np.nan
	hacker_x[ ys <= 0] = np.nan

	plt.xlabel('Tick')
	plt.scatter(xs[ys>0],  ys[ys> 0] , c="red")
	plt.scatter(xs[ys==0], ys[ys==0], c="blue")
	plt.scatter(xs[ys<0],  ys[ys < 0], c="green")
	plt.axhline(0, color='blue')

	if(os.path.isdir("/Volumes/Skas_HardD/Skas/Project/Plots/")):
		fig.savefig("/Volumes/Skas_HardD/Skas/Project/Plots/Fair_vs_Hacker/"+title+".png")

	plt.show(block= False)




def plot_plane_hmm(x, y, model,  X, title="Hacker",xlabel="Tick", ylabel="Diff"):
	fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
	fig.suptitle(title)
	colours = cm.rainbow(np.linspace(0, 1, model.n_components))
	hidden_states = model.predict(X)
	for i, (ax, colour) in enumerate(zip(axs, colours)):
		mask = hidden_states == i
		ax.set_title("{0}th hidden state".format(i))
		ax.plot(x[mask], y[mask], ".-", c=colour)
		ax.grid(True)

	if(os.path.isdir("/Volumes/Skas_HardD/Skas/Project/Plots/")):
		fig.savefig("/Volumes/Skas_HardD/Skas/Project/Plots/"+title+".png")

	plt.show(block=False)


def plot_scatter_hmm(x,y, model, X, title='Hacker View Direction'):
	#Must create new figure for each plot.
	#Plot creating a fade away color for each line.. 
	fig = plt.figure()
	fig.suptitle(title)
	plt.xlabel('ViewCos')
	plt.ylabel('ViewSin')
	#loop here over color

	colours = cm.rainbow(np.linspace(0, 1, model.n_components))

	# Use fancy indexing to plot data in each state.
	hidden_states = model.predict(X)
	for i, (state, colour) in enumerate(zip(range(model.n_components), colours)):
		mask = hidden_states == i
		plt.scatter(x=x[mask], y=y[mask], color=colour) #color=colors[i])

	if(os.path.isdir("/Volumes/Skas_HardD/Skas/Project/Plots/")):
		fig.savefig("/Volumes/Skas_HardD/Skas/Project/Plots/"+title+".png")


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

