from mpl_toolkits.axes_grid.axislines import SubplotZero
import matplotlib.pyplot as plt
from matplotlib import cm
from  matplotlib.animation import FuncAnimation
import numpy as np
import pdb
def plot_plane(x, y, model,  X, name="Distance",xlabel="Tick", ylabel="Diff"):
	fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)

	colours = cm.rainbow(np.linspace(0, 1, model.n_components))
	hidden_states = model.predict(X)
	for i, (ax, colour) in enumerate(zip(axs, colours)):
		mask = hidden_states == i
		ax.set_title("{0}th hidden state".format(i))
		ax.plot(x[mask], y[mask], ".-", c=colour)
		ax.grid(True)
	plt.show(block=False)


def plot_scatter_hmm(x,y, model, X):

	#Must create new figure for each plot.

	#Plot creating a fade away color for each line.. 
	fig= plt.figure()
	fig.suptitle('Hacker View Direction')
	plt.xlabel('ViewCos')
	plt.ylabel('ViewSin')
	#loop here over color

	colours = cm.rainbow(np.linspace(0, 1, model.n_components))

	# Use fancy indexing to plot data in each state.
	hidden_states = model.predict(X)
	for i, (state, colour) in enumerate(zip(range(model.n_components), colours)):
		mask = hidden_states == i
		plt.scatter(x=x[mask], y=y[mask], color=colour) #color=colors[i])



