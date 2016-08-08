from mpl_toolkits.axes_grid.axislines import SubplotZero
import matplotlib.pyplot as plt
from matplotlib import cm
from  matplotlib.animation import FuncAnimation
import numpy as np
import pdb
def plot_plane(x, y):
	fig = plt.figure(1)
	ax = SubplotZero(fig, 111)
	fig.add_subplot(ax)

	for direction in ["xzero", "yzero"]:
		ax.axis[direction].set_axisline_style("-|>")
		ax.axis[direction].set_visible(True)

	for direction in ["left", "right", "bottom", "top"]:
		ax.axis[direction].set_visible(False)
	ax.plot(x, y)
	plt.show()


def plot_scatter_hmm(x,y, model, X):

	#Must create new figure for each plot.

	#Plot creating a fade away color for each line.. 
	fig= plt.figure()
	fig.suptitle('Hacker M4A1-S')
	plt.xlabel('ViewX')
	plt.ylabel('ViewY')
	#loop here over color
	pdb.set_trace()

	colours = cm.rainbow(np.linspace(0, 1, model.n_components))

	# Use fancy indexing to plot data in each state.
	hidden_states = model.predict(X)
	for i, (state, colour) in enumerate(zip(range(model.n_components), colours)):
		mask = hidden_states == i
		pdb.set_trace()
		plt.scatter(x=x[mask], y=y[mask], color=colour) #color=colors[i])

	plt.show()


