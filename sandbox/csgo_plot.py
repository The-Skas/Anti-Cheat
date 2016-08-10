from mpl_toolkits.axes_grid.axislines import SubplotZero
import matplotlib.pyplot as plt
from matplotlib import cm
from  matplotlib.animation import FuncAnimation
import numpy as np
import pdb

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

	fig.savefig("/Volumes/Skas_HardD/Skas/Project/Plots/"+title+".png")
	plt.show(block=False)


def plot_scatter_hmm(x,y, model, X, title='Hacker View Direction'):

	#Must create new figure for each plot.

	#Plot creating a fade away color for each line.. 
	fig= plt.figure()
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

	fig.savefig("/Volumes/Skas_HardD/Skas/Project/Plots/"+title+".png")




